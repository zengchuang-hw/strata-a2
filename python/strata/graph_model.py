# Capture construction adapted from Apache-2.0 qwen3-tts-engine
# b4451682abac651f26046f8d27467de3f007e1e7 and official Qwen3-TTS components.
# Qwen attention copyright: The Qwen team, Alibaba Group and HuggingFace Inc., 2026.
"""Python-only loading/capture adapter; frame and codec execution live in C++."""

import logging

import torch
from transformers.cache_utils import DynamicCache

import _strata_native as native
from qwen3_tts_engine.continuous import ContinuousModel, SlotCache, SlotGraph
from qwen3_tts_engine.codec_slots import CodecSlotGraph
from qwen3_tts_engine.sampling import categorical, random_table

LOGGER = logging.getLogger(__name__)


class NativeModel(ContinuousModel):
    def __init__(self, model_path, device, *, max_slots=64, max_frames=2048,
                 text_capacity=256, codec_batch_size=4, prewarm_decode=False):
        self.core = None
        super().__init__(
            model_path, device, max_slots=max_slots, pack_qkv=True,
            rms_norm="native", pack_mlp=True, fused_swiglu=True,
            native_gqa=True, text_only_rope=True, native_rope=True,
            native_rope_scope="predictor", triton_talker_rope=True,
            native_kv_update=True, triton_kv_update=True,
            incremental_codec=True, batch_prefill=True, graph_prefill=True,
            bucket_prefill=True, prewarm_prefill=True, direct_prompt=True,
            triton_snake=True, cache_codec_codebooks=True,
            codec_batch_size=codec_batch_size, triton_cdf=True,
        )
        weight = next(self.talker.parameters())
        dim = self.talker.model.config.hidden_size
        vocab = self.talker.config.vocab_size
        opts = {"device": device}
        self.buffers = {
            "hidden": weight.new_zeros(max_slots, 1, dim),
            "tokens": torch.zeros(max_slots, 1, dtype=torch.long, **opts),
            "active": torch.zeros(max_slots, dtype=torch.bool, **opts),
            "seen": torch.zeros(max_slots, vocab, dtype=torch.bool, **opts),
            "uniforms": torch.full((max_slots, max_frames + 1, 16), 0.5, **opts),
            "trailing": weight.new_zeros(max_slots, text_capacity + 1, dim),
            "text_lengths": torch.zeros(max_slots, dtype=torch.long, **opts),
            "frames": torch.zeros(max_slots, dtype=torch.long, **opts),
            "positions": torch.zeros(max_slots, dtype=torch.long, **opts),
            "rope": torch.zeros(max_slots, dtype=torch.long, **opts),
            "history": torch.zeros(max_slots, max_frames, 16, dtype=torch.long, **opts),
        }
        self.core = native.Runtime(self.buffers, max_slots, max_frames, self.eos, codec_batch_size)
        self.core.set_kv(self.arena.keys + self.arena.values, self.arena.capacity)
        self.core.set_codec_state(self.codec.buffers)
        self.native_decodes = {}
        self.native_codecs = {}
        self.max_frames = max_frames
        self._capture_codecs(codec_batch_size)
        if prewarm_decode:
            batch = 1
            while batch <= max_slots:
                self._capture_decode(batch)
                batch *= 2

    @property
    def states(self):
        return self.core.identifiers() if self.core is not None else []

    @states.setter
    def states(self, unused):
        # The reference constructor assigns an empty list; active state is native.
        if unused:
            raise ValueError("Native request state cannot be assigned from Python")

    @torch.inference_mode()
    def ensure_capacity(self, required):
        previous = self.arena
        super().ensure_capacity(required)
        if self.core is not None and previous is not self.arena:
            self.native_decodes.clear()
            self.core.set_kv(self.arena.keys + self.arena.values, self.arena.capacity)

    @torch.inference_mode()
    def add_state(self, identifier, payload, prompt, hidden, logits, position, rope_delta):
        limit = payload["max_new_tokens"]
        if not 2 <= limit <= self.max_frames:
            raise ValueError(f"max_new_tokens must be 2..{self.max_frames}")
        uniforms = random_table(payload["seed"], limit).to(self.device)
        seen = torch.zeros(1, self.talker.config.vocab_size, device=self.device, dtype=torch.bool)
        token = self.semantic(logits, uniforms[0, 0:1], seen,
                              torch.zeros(1, device=self.device, dtype=torch.long))
        seen.scatter_(1, token, True)
        self.core.admit(identifier, hidden, token, prompt["trailing_text_hidden"],
                        prompt["tts_pad_embed"], uniforms, seen, position, rope_delta, limit)

    @torch.inference_mode()
    def _capture(self, compute, touched):
        saved = [buffer.clone() for buffer in touched]

        def restore():
            for buffer, original in zip(touched, saved):
                buffer.copy_(original)

        stream = torch.npu.Stream(device=self.device)
        stream.wait_stream(torch.npu.current_stream())
        graph = native.Graph()
        try:
            with torch.npu.stream(stream):
                for _ in range(3):
                    compute()
                    restore()
                torch.npu.synchronize()
                graph.begin()
                output = compute()
                graph.end()
                restore()
            torch.npu.current_stream().wait_stream(stream)
            torch.npu.synchronize()
        except BaseException:
            # A capture failure is fatal for this candidate, never silent fallback.
            LOGGER.exception("Native graph capture failed")
            raise
        return graph, output

    @torch.inference_mode()
    def _capture_codecs(self, max_batch):
        for batch in range(1, min(max_batch, self.max_slots) + 1):
            for length in range(1, 9):
                adapter = CodecSlotGraph(self.codec, batch, length)
                graph, output = self._capture(adapter.compute, self.codec.buffers)
                self.native_codecs[batch, length] = (graph, adapter, output)
                self.core.register_codec(batch, length, graph, adapter.indices, adapter.codes, output)
        LOGGER.info("Prepared %d native codec graphs", len(self.native_codecs))

    @torch.inference_mode()
    def _capture_decode(self, batch):
        b = {name: value[:batch] for name, value in self.buffers.items()}
        rows = torch.arange(batch, device=self.device)
        predictor = self.talker.code_predictor
        predictor_positions = [torch.arange(2, device=self.device)] + [
            torch.tensor([i + 1], device=self.device) for i in range(1, 15)]
        mask = torch.zeros(1, 1, 2, 2, device=self.device, dtype=b["hidden"].dtype)
        mask[..., 0, 1] = torch.finfo(mask.dtype).min
        talker = SlotGraph(self.arena, batch)
        talker.positions = b["positions"]
        talker.cache = SlotCache(self.arena, talker.positions)

        def compute():
            draws = b["uniforms"][rows, b["frames"].clamp(max=self.max_frames)]
            first = self.talker.get_input_embeddings()(b["tokens"])
            inputs = torch.cat((b["hidden"], first), dim=1)
            cache = DynamicCache()
            token = None
            codes = []
            for step in range(15):
                output = predictor(
                    inputs_embeds=inputs if step == 0 else None, input_ids=token,
                    generation_steps=step, past_key_values=cache, use_cache=True,
                    attention_mask={"full_attention": mask if step == 0 else None},
                    cache_position=predictor_positions[step],
                    position_ids=predictor_positions[step].unsqueeze(0),
                    output_hidden_states=False, return_dict=True)
                token = categorical(output.logits[:, -1], draws[:, step + 1], triton_cdf=True)
                codes.append(token)
            subcodes = torch.cat(codes, dim=-1)
            embeds = torch.cat([first] + [predictor.get_input_embeddings()[i](subcodes[:, i:i + 1])
                                         for i in range(15)], dim=1).sum(1, keepdim=True)
            text_index = torch.minimum(b["frames"], b["text_lengths"])
            text = b["trailing"][rows, text_index].unsqueeze(1)
            talker.inputs = embeds + text
            talker.rotary = (b["positions"] + b["rope"]).view(1, batch, 1).expand(3, -1, -1)
            hidden = talker.compute()
            logits = self.talker.codec_head(hidden)[:, -1]
            next_frame = b["frames"] + 1
            uniform = b["uniforms"][rows, next_frame.clamp(max=self.max_frames), 0]
            next_token = self.semantic(logits, uniform, b["seen"], next_frame)
            updated_seen = b["seen"].scatter(1, next_token, True)
            all_codes = torch.cat((b["tokens"], subcodes), dim=-1)
            indices = b["frames"].clamp(max=self.max_frames - 1).view(batch, 1, 1).expand(-1, 1, 16)
            b["history"].scatter_(1, indices, all_codes.unsqueeze(1))
            b["hidden"].copy_(hidden)
            b["tokens"].copy_(next_token)
            b["seen"].copy_(updated_seen)
            b["positions"].add_(b["active"])
            b["frames"].add_(b["active"])

        touched = [b[name] for name in ("hidden", "tokens", "seen", "positions", "frames", "history")]
        touched += [value[:batch] for value in self.arena.keys + self.arena.values]
        graph, _ = self._capture(compute, touched)
        # Keep constants, the adapter and every graph-dependent object alive.
        self.native_decodes[batch] = (graph, talker, predictor_positions, mask, rows)
        self.core.register_decode(batch, graph)
        LOGGER.info("Prepared native decode B%d KV%d", batch, self.arena.capacity)

    @torch.inference_mode()
    def prepare_native(self):
        self.ensure_capacity(self.core.required_position())
        batch = self.core.batch()
        if not self.core.has_decode(batch):
            self._capture_decode(batch)

    @torch.inference_mode()
    def tick(self, submissions=(), emit=None, cancelled=()):
        torch.npu.set_device(self.device)
        self.core.cancel(list(cancelled))
        if submissions:
            self.admit_batch(submissions)
        if self.core.size() == 0:
            return []
        if not self.core.ready():
            self.prepare_native()
        results = []
        for identifier, frames, done, pcm, reason in self.core.step():
            if done:
                results.append((identifier, {"pcm": pcm, "sample_rate": 24000,
                                              "codec_frames": frames, "finish_reason": reason}))
            elif emit is not None:
                emit(identifier, pcm, 24000)
        return results

    @torch.inference_mode()
    def close(self):
        torch.npu.synchronize()
        self.core.close()
        self.native_decodes.clear()
        self.native_codecs.clear()
        self.buffers.clear()
        super().close()
