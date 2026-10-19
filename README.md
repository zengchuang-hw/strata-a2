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
| — | frozen contract and design | [docs/task-contract.md](docs/task-contract.md), [docs/engine-design.md](docs/engine-design.md) |
| [#1](../../pull/1) | 001 bootstrap layered runtime | accepted: build + ownership tests + byte-identical probes ([record](docs/iterations/001-bootstrap.md)) |
| [#3](../../pull/3) | 002 paired predictor rotary kernel | formal verdict FAIL overall — C1–C8 PASS (+5.0…+12.2%, byte-identical), C16–C64 FAIL; kept as parent ([record](docs/iterations/002-predictor-rotary.md)) |
| [#4](../../pull/4) | 003 wide rotary tiles at rows≥16 | **formal cohort PASS at every concurrency** (+7.0…+13.1%, byte-identical) — performance objective met ([record](docs/iterations/003-wide-tiles.md)) |

Each merged PR adds its row here and keeps a full record under
[docs/iterations/](docs/iterations/).

## Build and run

Linux/aarch64 with PyTorch 2.10, torch_npu 2.10.0.post4, CANN 9.1,
`qwen-tts==0.1.1`, transformers 4.57.3, Triton Ascend, C++17 and CMake
(`requirements-a2.txt`). A pinned, clean
[InferMatrixLab/qwen3-tts-engine](https://github.com/InferMatrixLab/qwen3-tts-engine)
checkout at `b4451682abac651f26046f8d27467de3f007e1e7` supplies the model
adapters and graph construction; the checkout is verified at startup and its
Apache-2.0 files keep their attribution in [NOTICE.md](NOTICE.md).

```bash
cmake -S <repo> -B <build> -DPython3_EXECUTABLE=<venv>/bin/python -DCMAKE_BUILD_TYPE=Release
cmake --build <build> -j 2
export OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2
export PYTHONPATH=<reference-checkout>:$PYTHONPATH

# Native launcher: Python loads weights and captures graphs once,
# the C++ main loop replays them and writes complete WAV files.
<build>/strata-run --reference-root <reference-checkout> --model <model-dir> \
  --workload <workload.jsonl> --slots 4 --requests 4 --output <outdir>

# Protocol-compatible HTTP entrypoint on the same native C++ core.
<venv>/bin/python serve/serve.py --reference-root <reference-checkout> \
  --native-build <build> --model <model-dir> --max-batch-size 64 --prewarm-decode --port 8199
```

Compiled ownership tests need no NPU:

```bash
PYTHONPATH=<build>:$PYTHONPATH <venv>/bin/python tests/test_native_state.py -v
```

`bench/probe.py` compares complete generation and exercises cancellation/slot
reuse; `bench/campaign.py` runs alternating reference/native HTTP trials and
`bench/assess.py` audits the cohort evidence against the frozen gates. Probe
and readiness timings include graph capture and are never benchmark results.


## Status

Research artifact. Build, correctness, performance and delivery are judged
separately (PASS / FAIL / INCONCLUSIVE); the current status per gate is kept in
the latest iteration record. Nothing here is a production service.

## License

Apache-2.0. Reused Apache-2.0 portions of the pinned reference engine keep
their attribution in [NOTICE.md](NOTICE.md).
