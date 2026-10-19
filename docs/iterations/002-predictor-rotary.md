# Iteration 002 — paired predictor rotary kernel (C002)

Candidate: C002 / parent: C001 (iteration 001) / contract: `docs/task-contract.md`

## Bottleneck (profile, diagnostic only)

A B4 diagnostic profile of the warmed native frame graph
(`bench/profile_native.py`) attributed a measurable device-time share to the
predictor rotary calls: the code predictor applies rotary to Q and K as
separate `npu_rotary_mul` ops for each of its 15 codebook steps per frame.
The profiler emitted a RECORD-stop warning, so kernel accounting stays
diagnostic and was never promoted to throughput or critical-path evidence.

## Hypothesis

One fused Triton launch applying rotary to predictor Q and K together, keeping
the exact native FP32 intermediate / BF16 store arithmetic, removes redundant
launches and small-op overhead in every decode step and improves end-to-end
throughput without changing output.

## Modification

- `python/strata/predictor_rope.py`: paired Q/K rotary kernel plus a
  `PredictorAttention` adapter replacing only non-multimodal `PackedAttention`
  modules inside the code predictor (`install()`).
- Plumbing: `--predictor-rope {native,paired}` on the launcher, HTTP server,
  campaign and probe CLIs; the default stays `native` so parent behavior is
  always reproducible.

## Trial-and-error trail

1. **BF16-intermediate variant: rejected.** The first kernel rounded
   intermediates to BF16 between multiply and add. The operator probe showed
   nonzero deviation from `npu_rotary_mul`; it is retained in the probe as the
   deliberate counterexample (`round_bf16=True` must NOT be exact), not
   selected.
2. **Selected unrounded kernel: exact.** FP32 multiply/add per element, BF16
   only at store. All 32 unrounded operator cases matched the native operator
   exactly: batches 1/4/32/64 × lengths 1/2 × head dims 64/128 ×
   shared/per-request coefficients.
3. **Complete-model probes: byte-identical.** Plain and cancel/reuse probes
   reproduced the recorded reference PCM hashes exactly.
4. **Exploratory environment-controlled C4 comparison** (128 requests, pinned
   environment, before freezing): throughput 170.68 vs 157.99 frames/s
   (+8.0%), TTFA p50 76.87 vs 83.01 ms, TTFA p95 95.80 vs 104.33 ms, mean RTF
   0.292 vs 0.315; all 128 outputs byte-identical, zero failed requests. A
   single exploratory pair — explicitly not formal acceptance.
5. **Environment discipline from iteration 001's C001 lesson.** The formal
   cohort ran with hard identity re-checks around every arm (source, binaries,
   packages, system libraries, CANN install, 11 model assets); a drift would
   abort the campaign rather than produce a number.

## Formal frozen cohort (`formal-c002-01`)

Three alternating AB/BA pairs, C1/2/4/8/16/32/64, 128 requests per cohort,
identical workload/voices/sampling/hardware, environment identity fingerprinted
before and after every arm. Throughput in frames/s (audio-duration basis).

| C | reference thr (3 reps) | native thr (3 reps) | min-native/max-ref | TTFA p95 max-native/min-ref | verdict |
| --- | --- | --- | --- | --- | --- |
| 1 | 61.77 / 61.33 / 59.58 | 70.76 / 69.31 / 70.63 | 1.1220 | 1.0241 | PASS |
| 2 | 97.26 / 97.43 / 94.59 | 106.77 / 103.35 / 106.25 | 1.0608 | 0.9466 | PASS |
| 4 | 156.77 / 159.60 / 154.88 | 175.18 / 168.51 / 173.61 | 1.0559 | 0.9576 | PASS |
| 8 | 246.93 / 247.59 / 240.55 | 270.23 / 260.06 / 263.46 | 1.0504 | 0.9404 | PASS |
| 16 | 339.15 / 359.88 / 351.44 | 389.75 / 371.59 / 386.65 | 1.0325 | 0.9485 | FAIL (throughput) |
| 32 | 453.65 / 458.88 / 450.62 | 489.08 / 473.72 / 473.74 | 1.0323 | 1.2520 | FAIL (throughput + TTFA) |
| 64 | 581.15 / 583.34 / 568.93 | 608.58 / 610.71 / 622.86 | 1.0433 | 1.1094 | FAIL (throughput + TTFA) |

Independent assessment verdict (`bench/assess.py`): **overall FAIL**
(`formal_coverage: true`, `relative_quality: PASS_EXACT_PCM`). The frozen
thresholds (>5% throughput at every concurrency, ≤5% TTFA p95 regression) were
not relaxed after the runs.

## Result and decision

- Performance: PASS at C1–C8 (throughput +5.0%…+12.2%, TTFA p95 never
  regressed beyond the gate); FAIL at C16–C64 (+3.2…+4.3% throughput below
  threshold; TTFA p95 +10.9% at C64 and +25.2% at C32).
- Quality: byte-identical PCM in every matched pair across all 42 cohorts.
- Decision: the candidate stays in-tree as the parent for iteration 003,
  which targets the high-concurrency regression. Working mechanism hypothesis:
  at batch ≥ 16 the paired kernel's per-row program count (6 tiles/row for
  16 Q-heads + 8 K-heads at head-dim 128) adds device scheduling latency to
  every decode step; TTFA at high concurrency is dominated by the shared
  batched step latency, so the per-step overhead becomes visible exactly where
  the reference's fused `npu_rotary_mul` path does not pay it. Iteration 003
  candidates: widen tiles (8 heads per program) to halve program count, or
  gate the paired kernel to the batch range where it wins.

## Port validation (this tree)

The formal numbers above were measured on the pre-rename lineage. This port
renames identifiers only; the port itself was re-validated in the dedicated
campaign container on the 910B3 device before merge:

- [x] `bench/rope_probe.py`: status PASS — 32/32 unrounded cases exactly equal
      `npu_rotary_mul`; 32/32 `round_bf16=True` counterexample cases non-exact
      (max abs error 0.03125), i.e. the rejected variant stays rejected.
- [x] Complete-model paired probes (`--predictor-rope paired`): plain probe
      ids 0/1/64/65 and cancel/reuse probe ids 0/64/65/2 all finished with
      `stop`, streamed chunks equal final PCM, and every PCM SHA256 equals the
      recorded reference evidence hashes — byte-identical in the paired mode.

## Result (final)

**Iteration 002 accepted as parent with the formal performance verdict
attached: overall FAIL.** C1–C8 PASS the frozen gate with byte-identical
audio; C16–C64 do not. The candidate remains in-tree because iteration 003
optimizes exactly the measured regression (high-concurrency per-step
latency); thresholds stay frozen.
