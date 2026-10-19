# Strata-A2

A Strata-style **layered C++ inference runtime** for Qwen3-TTS-12Hz-1.7B-CustomVoice
on Ascend 910B3 (A2). The project is generated and optimized with the
[InferRSI](https://github.com/InferMatrixLab/InferRSI) iterative loop:
every optimization iteration is merged as a pull request whose description
records what was optimized, the mechanism, and the full trial-and-error trail
(rejected candidates included).

The performance objective is to beat the frozen
[InferMatrixLab/qwen3-tts-engine](https://github.com/InferMatrixLab/qwen3-tts-engine)
reference (`b4451682abac651f26046f8d27467de3f007e1e7`) on the same physical
device, measured as end-to-end audio throughput per concurrency with a
first-audio-latency guard. Upstream 910B1 numbers are imported evidence, not
local results; only same-device AB/BA measurements count.

## Architecture in one page

The runtime is split across a versioned C ABI so that torch, CANN and the
model adapter never enter the executable's initial loader namespace:

| Layer | Location | Responsibility |
| --- | --- | --- |
| C++ core | `src/runtime.cpp`, `include/strata/runtime_api.h` | request arena, state ownership, NPU graph registry and replay, codec scheduling, slot compaction, PCM conversion |
| Launcher | `src/main.cpp` | embedded-Python bootstrap, then drives the core through the C API |
| Python bootstrap | `python/strata/` | official weight loading, tokenization, prefill, NPU graph capture (decode + codec), adapters |
| Serving | `serve/` | pinned reference HTTP protocol (`/v1/audio/speech`) |
| Measurement | `bench/` | alternating AB/BA HTTP campaign, independent assessment, probes, profiling |
| Tests | `tests/` | compiled CPU ownership contracts (no NPU required) |

Buffer lifetime and state ownership follow the Strata engine's design ideas:
padded graph arenas sized in powers of two, per-slot request state owned by the
core, cancellation by row compaction instead of free-list churn, and graph
captures that own every tensor they touch. Strata was studied at
`477ac13cbe8e582c20484871ce459d20d82df09d`; no Strata CUDA/MoE result is
imported as Ascend evidence.

## Iteration log

| PR | Iteration | Result |
| --- | --- | --- |
| — | frozen contract and design | [docs/task-contract.md](docs/task-contract.md) |

Each merged PR adds its row here and keeps a full record under
[docs/iterations/](docs/iterations/).

## Status

Research artifact. Build, correctness, performance and delivery are judged
separately (PASS / FAIL / INCONCLUSIVE); the current status per gate is kept in
the latest iteration record. Nothing here is a production service.

## License

Apache-2.0. Reused Apache-2.0 portions of the pinned reference engine keep
their attribution in [NOTICE.md](NOTICE.md).
