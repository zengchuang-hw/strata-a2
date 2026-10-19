# Graph attention structure adapted from Apache-2.0 Qwen3-TTS / qwen3-tts-engine
# b4451682abac651f26046f8d27467de3f007e1e7. Original Qwen attention copyright:
# The Qwen team, Alibaba Group and the HuggingFace Inc. team, 2026.
"""Paired predictor-only rotary kernel; numerical qualification is mandatory."""

import torch
import triton
import triton.language as tl


@triton.jit
def _rotate_one(X, C, S, Y, row, tile, H: tl.constexpr, D: tl.constexpr,
                LENGTH: tl.constexpr, SHARED: tl.constexpr, ROUND_BF16: tl.constexpr):
    heads = tile * 4 + tl.arange(0, 4)[:, None]
    dims = tl.arange(0, D // 2)[None, :]
    base = (row * H + heads) * D + dims
    mask = heads < H
    first = tl.load(X + base, mask, other=0).to(tl.float32)
    second = tl.load(X + base + D // 2, mask, other=0).to(tl.float32)
    coefficient = (row % LENGTH if SHARED else row) * D + dims
    cos1 = tl.load(C + coefficient).to(tl.float32)
    cos2 = tl.load(C + coefficient + D // 2).to(tl.float32)
    sin1 = tl.load(S + coefficient).to(tl.float32)
    sin2 = tl.load(S + coefficient + D // 2).to(tl.float32)
    left, right = first * cos1, -second * sin1
    left2, right2 = second * cos2, first * sin2
    if ROUND_BF16:
        left = left.to(tl.bfloat16).to(tl.float32)
        right = right.to(tl.bfloat16).to(tl.float32)
        left2 = left2.to(tl.bfloat16).to(tl.float32)
        right2 = right2.to(tl.bfloat16).to(tl.float32)
    tl.store(Y + base, (left + right).to(tl.bfloat16), mask)
    tl.store(Y + base + D // 2, (left2 + right2).to(tl.bfloat16), mask)


@triton.jit
def _paired(Q, K, C, S, QO, KO, HQ: tl.constexpr, HK: tl.constexpr,
            D: tl.constexpr, LENGTH: tl.constexpr, ROWS: tl.constexpr,
            SHARED: tl.constexpr, ROUND_BF16: tl.constexpr):
    qtiles: tl.constexpr = triton.cdiv(HQ, 4)
    tiles: tl.constexpr = qtiles + triton.cdiv(HK, 4)
    for task in range(tl.program_id(0), ROWS * tiles, tl.num_programs(0)):
        row = task // tiles
        tile = task % tiles
        if tile < qtiles:
            _rotate_one(Q, C, S, QO, row, tile, HQ, D, LENGTH, SHARED, ROUND_BF16)
        else:
            _rotate_one(K, C, S, KO, row, tile - qtiles, HK, D, LENGTH, SHARED, ROUND_BF16)


def paired_rotary(query, key, cos, sin, *, round_bf16=False):
    if (query.ndim != 4 or key.ndim != 4 or query.shape[:2] != key.shape[:2]
            or query.shape[-1] != key.shape[-1] or query.shape[-1] not in (64, 128)
            or query.shape[2] not in (8, 16) or key.shape[2] not in (8, 16)):
        raise ValueError("unsupported BSND predictor rotary shape")
    batch, length, heads, width = query.shape
    if cos.shape not in ((1, length, width), (batch, length, width)) or sin.shape != cos.shape:
        raise ValueError("rotary coefficients do not match predictor positions")
    if any(t.dtype != torch.bfloat16 or t.device != query.device or not t.is_contiguous()
           for t in (query, key, cos, sin)) or query.device.type != "npu":
        raise ValueError("contiguous matching BF16 NPU tensors required")
    qout, kout = torch.empty_like(query), torch.empty_like(key)
    tiles = triton.cdiv(heads, 4) + triton.cdiv(key.shape[2], 4)
    cores = torch.npu.get_device_properties(query.device).vector_core_num
    _paired[(min(cores, batch * length * tiles),)](
        query, key, cos, sin, qout, kout, heads, key.shape[2], width, length,
        batch * length, cos.shape[0] == 1, round_bf16, enable_fp_fusion=False)
    return qout, kout


class PredictorAttention(torch.nn.Module):
    def __init__(self, packed):
        super().__init__()
        if packed.multimodal or not packed.native_gqa or not packed.native_rope:
            raise ValueError("paired rotary only replaces native-GQA predictor attention")
        self.packed = packed

    def forward(self, hidden_states, position_embeddings, attention_mask,
                past_key_values=None, cache_position=None, **kwargs):
        from qwen3_tts_engine.npu_attention import attention
        packed, reference = self.packed, self.packed.reference
        shape = hidden_states.shape[:-1]
        heads = (*shape, -1, reference.head_dim)
        query, key, value = torch.nn.functional.linear(hidden_states, packed.weight, packed.bias).split(packed.sizes, -1)
        query = reference.q_norm(query.reshape(heads))
        key = reference.k_norm(key.reshape(heads))
        value = value.reshape(heads).transpose(1, 2)
        cos, sin = position_embeddings
        query, key = paired_rotary(query, key, cos, sin)
        query, key = query.transpose(1, 2), key.transpose(1, 2)
        if past_key_values is not None:
            key, value = past_key_values.update(key, value, reference.layer_idx,
                {"sin": sin, "cos": cos, "cache_position": cache_position})
        output, weights = attention(reference, query, key, value, attention_mask,
            dropout=0.0, scaling=reference.scaling, sliding_window=reference.sliding_window, **kwargs)
        return reference.o_proj(output.reshape(*shape, -1).contiguous()), weights


def install(predictor):
    replacements = []
    for parent in list(predictor.modules()):
        for name, child in list(parent.named_children()):
            if type(child).__name__ == "PackedAttention" and not child.multimodal:
                replacements.append((parent, name, child))
                setattr(parent, name, PredictorAttention(child))
    if not replacements:
        raise ValueError("no predictor attention modules were replaced")
    return replacements
