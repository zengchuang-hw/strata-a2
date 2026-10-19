# Frozen task contract (S0)

Status: frozen 2026-10-19. Any change to precision, model, workload or gates
starts a new contract and a new baseline; older results are retained as they
were recorded.

## Objective

Build a Strata-style layered C++ inference runtime for
Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice on one physical Ascend 910B3 device, and
improve end-to-end streaming performance against
`InferMatrixLab/qwen3-tts-engine` at `b4451682abac651f26046f8d27467de3f007e1e7`
(the performance reference), on the same physical device. A C++ launcher that
merely forwards the original Python inference loop is insufficient: request
ownership, frame advancement, graph replay and slot lifecycle must execute in
the C++ core.

Imported identities:

- Performance baseline: `InferMatrixLab/qwen3-tts-engine` @
  `b4451682abac651f26046f8d27467de3f007e1e7`. Its published numbers describe
  `e783a64b0fb4f9ade2ffb3980b3e1f74a00c9a6a` on 910B1 and are imported evidence,
  never this campaign's baseline.
- Measurement approach: InferRSI measurement source `bd5e356ac9267362c80f27adba61d76e047583c7`.
- Design reference: Strata `477ac13cbe8e582c20484871ce459d20d82df09d` for graph,
  buffer lifetime and state ownership ideas only.

## Acceptance gates

`build`, `correctness`, `performance`, `delivery` are judged separately.
Overall PASS requires every required gate PASS; missing evidence stays
INCONCLUSIVE and can never become PASS.

1. Inspect devices, model assets and the compiler/runtime without disturbing
   other jobs; use a dedicated source tree and environment.
2. Reproduce the optimized frozen Python baseline on a free physical device.
3. Implement the C++ execution/state core and a build/run entrypoint. Record
   every Python bootstrap, graph-construction and official-model dependency.
4. Validate operators and state, then complete real English/Chinese generation,
   EOS, long/short requests, cancellation, interleaving and slot reuse.
5. Profile outside timed runs; change one mechanism per candidate. Freeze
   source, build flags, dependency identities and binary SHA256 before formal
   comparisons.
6. Identical BF16 weights, prompt corpus, voices, sampling, output limits,
   streaming format and hardware. Measure successful full requests,
   audio-duration throughput, TTFA p50/p95 and mean RTF at
   C1/2/4/8/16/32/64, at least three alternating AB/BA repeat pairs, 128
   requests per cohort. Truncation or failures never count as speedups.
7. Performance gate: throughput improvement greater than 5% at each qualified
   concurrency with no more than 5% TTFA p95 regression, reported per
   concurrency. Thresholds are never relaxed after runs.
8. Prefer byte-identical audio for semantics-preserving changes. If output
   changes, run independently pinned ASR and speaker scoring: WER/CER no more
   than +1 percentage point and speaker cosine no more than -0.02 versus the
   best measured baseline per metric/concurrency, plus request/audio
   integrity. Missing scorers stay INCONCLUSIVE.

## Workload and model identity

- 128-request frozen prompt corpus, mixed English/Chinese, hashed into every
  run identity; the benchmark sends the pinned streaming protocol.
- 11 runtime model assets must match the pinned manifest SHA256/size at every
  arm; the reference revision is
  `0c0e3051f131929182e2c023b9537f8b1c68adfe` for the pinned model manifest.
- Environment identity per arm: package versions, system library versions
  (glibc pinned), CANN install identity, model asset hashes, source and both
  native binary SHA256, fingerprinted before and after every arm.

## Boundary conditions

- No merge into other projects, no package publication, no production service
  activation as part of this campaign.
- The control tooling that records evidence stays independent of torch,
  inference engines and accelerator runtimes.
- Failed, rejected and slower trials are retained, never deleted.
