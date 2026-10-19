#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUGraph.h>
#include <c10/core/InferenceMode.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "strata/runtime_api.h"

namespace py = pybind11;
using at::Tensor;

namespace strata {
struct Graph {
    c10_npu::NPUGraph graph;
    void begin() { graph.capture_begin(); }
    void end() { graph.capture_end(); }
    void replay() { graph.replay(); }
    void reset() { graph.reset(); }
};

struct Request {
    int64_t id, frames, emitted, position, limit;
    std::string pcm;
};

struct Event {
    int64_t id, frames;
    bool done;
    std::string pcm, reason;
};

struct Codec {
    std::shared_ptr<Graph> graph;
    Tensor indices, codes, waves;
};

class Runtime {
public:
    Runtime(std::map<std::string, Tensor> buffers, int64_t slots,
            int64_t max_frames, int64_t eos, int64_t codec_batch)
        : b_(std::move(buffers)), slots_(slots), max_frames_(max_frames),
          eos_(eos), codec_batch_(codec_batch) {
        TORCH_CHECK(slots > 0 && (slots & (slots - 1)) == 0, "power-of-two slots required");
        TORCH_CHECK(max_frames > 1 && codec_batch > 0, "invalid frame/codec bound");
    }

    int64_t size() const { return requests_.size(); }
    std::vector<int64_t> identifiers() const {
        std::vector<int64_t> result;
        for (const auto& r : requests_) result.push_back(r.id);
        return result;
    }
    int64_t batch() const {
        int64_t result = 1;
        while (result < size()) result *= 2;
        return result;
    }
    int64_t required_position() const {
        int64_t result = 0;
        for (const auto& r : requests_) result = std::max(result, r.position);
        return result;
    }
    void set_kv(std::vector<Tensor> kv, int64_t capacity) {
        kv_ = std::move(kv);
        capacity_ = capacity;
        decodes_.clear();
    }
    void set_codec_state(std::vector<Tensor> buffers) { codec_state_ = std::move(buffers); }
    bool has_decode(int64_t batch) const { return decodes_.count(batch) != 0; }
    bool ready() const { return required_position() < capacity_ && has_decode(batch()); }
    void register_decode(int64_t batch, std::shared_ptr<Graph> graph) {
        TORCH_CHECK(batch > 0 && batch <= slots_, "invalid graph batch");
        decodes_[batch] = std::move(graph);
    }
    void register_codec(int64_t batch, int64_t frames, std::shared_ptr<Graph> graph,
                        Tensor indices, Tensor codes, Tensor waves) {
        codecs_[{batch, frames}] = {std::move(graph), indices, codes, waves};
    }

    void admit(int64_t id, Tensor hidden, Tensor token, Tensor trailing, Tensor pad,
               Tensor uniforms, Tensor seen, int64_t position, int64_t rope, int64_t limit) {
        c10::InferenceMode inference;
        TORCH_CHECK(size() < slots_, "request arena is full");
        TORCH_CHECK(limit >= 2 && limit <= max_frames_, "output limit exceeds native arena");
        TORCH_CHECK(position >= 0 && position < capacity_, "prefill exceeds KV capacity");
        TORCH_CHECK(std::none_of(requests_.begin(), requests_.end(),
                    [id](const Request& r) { return r.id == id; }), "duplicate request id");
        auto row = size();
        TORCH_CHECK(trailing.size(1) < b_.at("trailing").size(1), "trailing text exceeds arena");
        b_.at("hidden")[row].copy_(hidden[0]);
        b_.at("tokens")[row].copy_(token[0]);
        b_.at("seen")[row].copy_(seen[0]);
        b_.at("uniforms")[row].fill_(0.5);
        b_.at("uniforms")[row].narrow(0, 0, uniforms.size(0)).copy_(uniforms);
        b_.at("trailing")[row].copy_(pad[0].expand_as(b_.at("trailing")[row]));
        b_.at("trailing")[row].narrow(0, 0, trailing.size(1)).copy_(trailing[0]);
        b_.at("text_lengths")[row].fill_(trailing.size(1));
        b_.at("frames")[row].zero_();
        b_.at("positions")[row].fill_(position);
        b_.at("rope")[row].fill_(rope);
        b_.at("history")[row].zero_();
        b_.at("active")[row].fill_(true);
        for (auto& buffer : codec_state_) buffer[row].zero_();
        requests_.push_back({id, 0, 0, position, limit, {}});
    }

    void cancel(const std::vector<int64_t>& ids) {
        c10::InferenceMode inference;
        for (int64_t i = size() - 1; i >= 0; --i)
            if (std::find(ids.begin(), ids.end(), requests_[i].id) != ids.end()) erase(i);
    }

    std::vector<Event> step() {
        c10::InferenceMode inference;
        if (requests_.empty()) return {};
        TORCH_CHECK(required_position() < capacity_, "KV capacity must grow before replay");
        auto it = decodes_.find(batch());
        TORCH_CHECK(it != decodes_.end(), "decode graph has not been prepared");
        it->second->replay();
        auto tokens = b_.at("tokens").narrow(0, 0, size()).to(at::kCPU).contiguous();
        const auto* ids = tokens.data_ptr<int64_t>();
        std::map<int64_t, std::vector<int64_t>> due;
        std::vector<int64_t> finished;
        for (int64_t i = 0; i < size(); ++i) {
            auto& r = requests_[i];
            ++r.frames;
            ++r.position;
            const bool stopped = ids[i] == eos_;
            const bool limited = r.frames >= r.limit - 1;
            const int64_t target = r.emitted == 0 ? 2 : r.emitted < 8 ? r.emitted * 2 : r.emitted + 8;
            if (r.frames >= target || stopped || limited) due[r.frames - r.emitted].push_back(i);
            if (stopped || limited) finished.push_back(i);
        }
        std::vector<Event> events;
        for (const auto& [length, rows] : due) {
            TORCH_CHECK(length >= 1 && length <= 8, "invalid incremental codec length");
            for (size_t offset = 0; offset < rows.size(); offset += codec_batch_) {
                auto count = std::min<int64_t>(codec_batch_, rows.size() - offset);
                auto codec_it = codecs_.find({count, length});
                TORCH_CHECK(codec_it != codecs_.end(), "codec graph has not been prepared");
                auto& codec = codec_it->second;
                auto indices = at::empty({count}, at::TensorOptions().dtype(at::kLong).device(at::kCPU));
                auto* index_ptr = indices.data_ptr<int64_t>();
                for (int64_t j = 0; j < count; ++j) {
                    auto row = rows[offset + j];
                    index_ptr[j] = row;
                    codec.codes[j].copy_(b_.at("history")[row]
                        .narrow(0, requests_[row].emitted, length).transpose(0, 1));
                }
                codec.indices.copy_(indices);
                codec.graph->replay();
                auto audio = codec.waves.to(at::kFloat).to(at::kCPU).contiguous();
                TORCH_CHECK(audio.size(0) == count && audio.numel() == count * length * 1920,
                            "unexpected codec audio shape");
                const auto* samples = audio.data_ptr<float>();
                auto samples_per_row = audio.numel() / count;
                for (int64_t j = 0; j < count; ++j) {
                    auto& r = requests_[rows[offset + j]];
                    std::string pcm(samples_per_row * 2, '\0');
                    for (int64_t k = 0; k < samples_per_row; ++k) {
                        const float value = samples[j * samples_per_row + k];
                        TORCH_CHECK(std::isfinite(value), "codec emitted nonfinite audio");
                        const auto sample = static_cast<int16_t>(std::clamp(value, -1.0f, 1.0f) * 32767.0f);
                        pcm[2 * k] = static_cast<char>(static_cast<uint16_t>(sample) & 255);
                        pcm[2 * k + 1] = static_cast<char>(static_cast<uint16_t>(sample) >> 8);
                    }
                    r.pcm += pcm;
                    r.emitted = r.frames;
                    events.push_back({r.id, r.frames, false, std::move(pcm), {}});
                }
            }
        }
        for (auto i = finished.rbegin(); i != finished.rend(); ++i) {
            auto& r = requests_[*i];
            events.push_back({r.id, r.frames, true, std::move(r.pcm), ids[*i] == eos_ ? "stop" : "length"});
            erase(*i);
        }
        return events;
    }

    void close() {
        decodes_.clear();
        codecs_.clear();
        requests_.clear();
        kv_.clear();
        codec_state_.clear();
        b_.clear();
    }

private:
    void erase(int64_t row) {
        auto last = size() - 1;
        if (row != last) {
            for (auto& [name, buffer] : b_) buffer[row].copy_(buffer[last]);
            for (auto& buffer : kv_) buffer[row].copy_(buffer[last]);
            for (auto& buffer : codec_state_) buffer[row].copy_(buffer[last]);
            requests_[row] = std::move(requests_[last]);
        }
        // Padded graph rows must never advance beyond the KV/history bounds.
        b_.at("active")[last].fill_(false);
        b_.at("positions")[last].zero_();
        b_.at("frames")[last].zero_();
        requests_.pop_back();
    }
    std::map<std::string, Tensor> b_;
    int64_t slots_, max_frames_, eos_, codec_batch_, capacity_ = 0;
    std::vector<Request> requests_;
    std::vector<Tensor> kv_, codec_state_;
    std::map<int64_t, std::shared_ptr<Graph>> decodes_;
    std::map<std::pair<int64_t, int64_t>, Codec> codecs_;
};
} // namespace strata

PYBIND11_MODULE(_strata_native, m) {
    py::class_<strata::Graph, std::shared_ptr<strata::Graph>>(m, "Graph")
        .def(py::init<>()).def("begin", &strata::Graph::begin).def("end", &strata::Graph::end)
        .def("replay", &strata::Graph::replay).def("reset", &strata::Graph::reset);
    py::class_<strata::Runtime>(m, "Runtime")
        .def(py::init<std::map<std::string, Tensor>, int64_t, int64_t, int64_t, int64_t>())
        .def("size", &strata::Runtime::size).def("identifiers", &strata::Runtime::identifiers)
        .def("batch", &strata::Runtime::batch).def("required_position", &strata::Runtime::required_position)
        .def("set_kv", &strata::Runtime::set_kv).def("set_codec_state", &strata::Runtime::set_codec_state)
        .def("has_decode", &strata::Runtime::has_decode).def("register_decode", &strata::Runtime::register_decode)
        .def("ready", &strata::Runtime::ready)
        .def("register_codec", &strata::Runtime::register_codec).def("admit", &strata::Runtime::admit)
        .def("cancel", &strata::Runtime::cancel).def("close", &strata::Runtime::close)
        .def("address", [](strata::Runtime& runtime) { return reinterpret_cast<uintptr_t>(&runtime); })
        .def("step", [](strata::Runtime& runtime) {
            std::vector<strata::Event> events;
            { py::gil_scoped_release release; events = runtime.step(); }
            py::list result;
            for (auto& e : events)
                result.append(py::make_tuple(e.id, e.frames, e.done, py::bytes(e.pcm), e.reason));
            return result;
        });
}

namespace {
thread_local std::string api_error;
int api_step(void* core, StrataEmit emit, void* context) {
    try {
        for (const auto& event : static_cast<strata::Runtime*>(core)->step()) {
            StrataEventView view{event.id, event.frames, event.done, event.pcm.data(), event.pcm.size(), event.reason.c_str()};
            emit(&view, context);
        }
        return 0;
    } catch (const std::exception& error) {
        api_error = error.what();
        return -1;
    }
}
}
extern "C" const StrataRuntimeAPI* strata_runtime_api() {
    static const StrataRuntimeAPI api{1,
        [](void* core) { return static_cast<strata::Runtime*>(core)->size(); },
        [](void* core) { return static_cast<strata::Runtime*>(core)->ready(); },
        api_step, []() { return api_error.c_str(); }};
    return &api;
}
