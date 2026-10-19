# Source provenance

The loading and graph adapters reuse and adapt Apache-2.0 code from
[InferMatrixLab/qwen3-tts-engine](https://github.com/InferMatrixLab/qwen3-tts-engine/tree/b4451682abac651f26046f8d27467de3f007e1e7),
in particular `continuous.py`, `codec_slots.py` and `packed_attention.py`.
Its [Apache-2.0 license](https://github.com/InferMatrixLab/qwen3-tts-engine/blob/b4451682abac651f26046f8d27467de3f007e1e7/LICENSE)
applies to those reused portions. Official model components come from
`qwen-tts==0.1.1`; their original Qwen/Alibaba/HuggingFace copyright and license
notices remain in that dependency. The model is separately identified by the
11 locally hashed runtime assets and the pinned reference manifest.

Strata was studied at `477ac13cbe8e582c20484871ce459d20d82df09d` for runtime and
resource-ownership ideas. No Strata CUDA performance claim is imported as Ascend
evidence, and no MoE placement implementation is applied to this dense model.

The new C++ core, C API and per-iteration operator implementations are research
artifacts. Their actual validation scope is recorded in the campaign plan and
iteration records, separately from the upstream source licenses and reported
measurements.
