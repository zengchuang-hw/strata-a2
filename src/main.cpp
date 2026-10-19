#include <pybind11/embed.h>
#include <chrono>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include "strata/runtime_api.h"

namespace py = pybind11;
namespace {
struct Event {
    int64_t id, frames;
    bool done;
    std::string pcm, reason;
};
void collect(const StrataEventView* view, void* context) {
    static_cast<std::vector<Event>*>(context)->push_back(
        {view->id, view->frames, view->done != 0, std::string(view->pcm, view->pcm_bytes), view->reason});
}
void write_u32(std::ostream& out, uint32_t value, int bytes = 4) {
    for (int i = 0; i < bytes; ++i) out.put(static_cast<char>((value >> (8 * i)) & 255));
}
void write_wave(const std::filesystem::path& path, const std::string& pcm) {
    if (pcm.empty() || pcm.size() % 2 || pcm.size() >= UINT32_MAX - 36)
        throw std::runtime_error("invalid WAV payload");
    std::ofstream out(path, std::ios::binary);
    out.write("RIFF", 4); write_u32(out, 36 + pcm.size()); out.write("WAVEfmt ", 8);
    write_u32(out, 16); write_u32(out, 1, 2); write_u32(out, 1, 2);
    write_u32(out, 24000); write_u32(out, 48000); write_u32(out, 2, 2); write_u32(out, 16, 2);
    out.write("data", 4); write_u32(out, pcm.size()); out.write(pcm.data(), pcm.size());
    if (!out.good()) throw std::runtime_error("WAV output write failed");
}
}

int main(int argc, char** argv) {
    PyConfig config;
    PyConfig_InitPythonConfig(&config);
    auto configured = PyConfig_SetBytesString(&config, &config.program_name, STRATA_PYTHON_EXECUTABLE);
    if (PyStatus_Exception(configured)) {
        PyConfig_Clear(&config);
        std::cerr << "Cannot configure the build's Python environment\n";
        return 1;
    }
    py::scoped_interpreter interpreter(&config);
    py::object model;
    try {
        auto path = py::module_::import("sys").attr("path");
        path.attr("insert")(0, STRATA_BOOTSTRAP_DIR);
        path.attr("insert")(0, STRATA_NATIVE_BUILD);
        auto bootstrap = py::module_::import("cli_bootstrap");
        py::list arguments;
        for (int i = 1; i < argc; ++i) arguments.append(argv[i]);
        auto session = bootstrap.attr("initialize")(arguments).cast<py::dict>();
        model = session["model"];
        auto module_path = py::module_::import("_strata_native").attr("__file__").cast<std::string>();
        void* module = dlopen(module_path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!module) throw std::runtime_error(dlerror());
        auto get_api = reinterpret_cast<const StrataRuntimeAPI* (*)()>(dlsym(module, "strata_runtime_api"));
        if (!get_api) throw std::runtime_error("native runtime API was not exported");
        const auto* api = get_api();
        if (api->version != 1) throw std::runtime_error("incompatible native runtime API");
        auto* core = reinterpret_cast<void*>(model.attr("core").attr("address")().cast<uintptr_t>());
        auto inputs = session["inputs"].cast<py::list>();
        auto slots = session["slots"].cast<int64_t>();
        auto output = std::filesystem::path(session["output"].cast<std::string>());
        int64_t cursor = 0, completed = 0;
        py::list records;
        std::map<int64_t, double> first_audio;
        std::map<int64_t, std::chrono::steady_clock::time_point> admitted;
        auto began = std::chrono::steady_clock::now();
        while (cursor < static_cast<int64_t>(inputs.size()) || api->size(core)) {
            py::list submissions;
            auto free_slots = slots - api->size(core);
            while (free_slots-- > 0 && cursor < static_cast<int64_t>(inputs.size())) {
                auto id = cursor++;
                submissions.append(py::make_tuple(id, inputs[id]));
                admitted[id] = std::chrono::steady_clock::now();
            }
            if (submissions.size()) model.attr("admit_batch")(submissions);
            if (!api->ready(core)) model.attr("prepare_native")();
            std::vector<Event> events;
            int status;
            { py::gil_scoped_release release; status = api->step(core, collect, &events); }
            if (status) throw std::runtime_error(api->last_error());
            for (const auto& event : events) {
                auto elapsed = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - admitted.at(event.id)).count();
                if (!event.done) { first_audio.emplace(event.id, elapsed); continue; }
                if (event.reason != "stop") throw std::runtime_error("request hit output limit");
                write_wave(output / (std::to_string(event.id) + ".wav"), event.pcm);
                py::dict record;
                record["id"] = event.id; record["frames"] = event.frames;
                record["pcm_bytes"] = event.pcm.size(); record["finish_reason"] = event.reason;
                record["first_audio_seconds"] = first_audio.at(event.id);
                record["elapsed_seconds"] = elapsed;
                records.append(record);
                ++completed;
            }
        }
        if (completed != static_cast<int64_t>(inputs.size())) throw std::runtime_error("missing native completion");
        bootstrap.attr("finish")(session, records,
            std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count());
        model.attr("close")();
        dlclose(module);
        return 0;
    } catch (py::error_already_set& error) {
        if (error.matches(PyExc_SystemExit)) {
            auto code = error.value().attr("code");
            return code.is_none() ? 0 : py::isinstance<py::int_>(code) ? code.cast<int>() : 1;
        }
        std::cerr << error.what() << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
    }
    if (model) { try { model.attr("close")(); } catch (...) {} }
    return 1;
}
