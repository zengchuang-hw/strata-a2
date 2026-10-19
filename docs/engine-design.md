# Engine design (S3)

Deliverable level: **specialized engine** — an independently buildable and
launchable runtime project with loading, computation, state management and
defined interfaces. It is not a general model framework: the Qwen3-TTS adapter
is the first (and current only) model integration, and the weight loader,
tokenizer and graph construction stay in Python by design.

## Routes considered

| Route | Verdict |
| --- | --- |
| Reuse the reference Python runtime as-is | rejected: leaves the decode loop, state ownership and scheduling inside the reference implementation; no C++ core to iterate on |
| Compile the model to a static graph library (MLC/TensorRT style) | rejected for this contract: Ascend graph capture via torch_npu NPUGraph already provides replay; a compiler route would change the numeric identity of every operator at once and cannot preserve byte-identical audio |
| Mixed-bootstrap specialized runtime (selected) | Python performs official weight loading, tokenization, prefill and NPU graph capture once per shape; the C++ core owns requests, replays graphs every step, and owns output conversion |

The selected route keeps official model loading semantics bit-exact while
moving per-frame control flow out of Python. The boundary is a versioned C ABI
(`include/strata/runtime_api.h`) so torch, CANN and pybind11 never enter the
launcher executable's initial loader namespace; the Python extension is
imported through the normal interpreter bootstrap first, then `dlopen`-ed and
queried for `strata_runtime_api()`.

## Module table

| Module | Source | Responsibility |
| --- | --- | --- |
| `src/runtime.cpp` (`strata` namespace) | generated | request arena, admission/cancel, frame advancement, decode/codec graph registries, codec batching, row compaction, FP32→int16 PCM conversion, EOS/limit finishing |
| `include/strata/runtime_api.h` | generated | versioned C ABI (`StrataRuntimeAPI` v1): `size`, `ready`, `step`, `last_error` |
| `src/main.cpp` (`strata-run`) | generated | embedded-Python bootstrap (pybind11), admission pacing, WAV writing, request records |
| `python/strata/reference.py` | generated | pinned reference checkout guard (clean tree at the frozen revision) |
| `python/strata/graph_model.py` | adapted, Apache-2.0 | `NativeModel`: official weights, buffers, admission, graph capture/replay registration |
| `python/strata/cli_bootstrap.py` | generated | CLI session for the launcher: workload parsing, model construction, result record |
| `serve/serve.py` | adapted | HTTP service on the pinned reference streaming protocol |
| `bench/campaign.py` | generated | alternating AB/BA HTTP trials with environment/source identity fingerprinting |
| `bench/assess.py` | generated | independent audit of cohort evidence and frozen gates |
| `bench/probe.py` | generated | complete-model development probe (stream/final consistency, cancellation, reuse) |
| `tests/test_native_state.py` | generated | compiled CPU ownership contracts (no NPU) |

Reused external components: PyTorch 2.10 (allocator, NPU runtime), torch_npu
2.10.0.post4 + CANN 9.1 (device runtime, NPUGraph, `npu_rotary_mul`),
`qwen-tts==0.1.1` (official model components), transformers 4.57.3
(tokenizer/config), Triton Ascend (device-side kernels), aiohttp (HTTP). All
versions are fingerprinted into every measurement identity.

## Memory plan

Per-slot buffers sized for `max_slots` (power of two) and `max_frames`:

- `hidden` `[slots,1,dim]` BF16, `tokens` `[slots,1]` int64, `seen` `[slots,vocab]` bool
- `uniforms` `[slots,max_frames+1,16]` pre-drawn sampling uniforms (fixed at admission)
- `trailing` `[slots,text_capacity+1,dim]` tail-text embedding ring, `text_lengths`
- `frames`, `positions`, `rope` per-slot int64 scalars, `history` `[slots,max_frames,16]` code ring
- KV arena shared across slots (`arena.keys + arena.values`), grown by the
  reference allocator; the core is re-pointed with `set_kv` and all captured
  decode graphs are invalidated on growth
- codec state buffers owned by the codec adapter, zeroed per admission

The launcher admits up to `slots` requests and the core replays the decode
graph at `batch()` = next power of two ≥ active requests; padded rows are kept
inert by `active=false` plus zeroed `positions`/`frames`.

## State ownership and buffer lifetime

- Active request state lives in the C++ core only. Python's `states` setter
  rejects assignment; `states` reads native identifiers.
- Admission appends into dense rows; cancellation compacts the last row into
  the hole (buffers, KV, codec state, request record) and resets the vacated
  padded row so it can never advance KV/history bounds.
- Every NPU graph capture clones and restores the tensors it touches, runs
  three warmup iterations, then captures on a dedicated stream; capture
  failure is fatal, never a silent fallback.
- Graph-dependent Python objects (adapters, constants) are kept alive for the
  lifetime of the graph; dropping them would free captured memory.

## Failure handling

- `step()` raises when the required decode/codec graph is missing; live rows do
  not advance (covered by `tests/test_native_state.py`).
- Non-finite codec output aborts the request loop instead of writing silence.
- The launcher refuses to report completion for requests that hit the output
  limit (finish reason must be `stop`).
- The HTTP adapter rejects `max_new_tokens` outside the native arena with 400
  instead of truncating silently.

## What is deliberately out of scope

Multi-device execution, quantization routes, MoE expert placement, speculative
decoding, and any model other than the pinned Qwen3-TTS revision.
