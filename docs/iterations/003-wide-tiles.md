# Iteration 003 — wide predictor rotary tiles at high batch (C003)

Candidate: C003 / parent: C002 (iteration 002, #3) / contract: `docs/task-contract.md`

## Bottleneck

Iteration 002's frozen cohort failed the throughput gate at C16–C64
(+3.2…+4.3% instead of >5%) with TTFA p95 outliers at C32/C64. The rotary
kernel's per-step device cost was the suspected mechanism.

## Measurement (this iteration's profile work)

`bench/rope_bench.py` measures the paired kernel against the reference native
operator pair at decode shapes, twice:

1. **Eager launches (misleading).** Every paired width showed a ~220µs floor
   independent of batch and tile width, while the native pair cost ~75µs.
   That floor is host-side dispatch cost; the engine replays the kernel inside
   a captured NPU graph, so eager numbers do not describe the engine path.
   Recorded as a lesson: only in-graph replay timing matches this engine.
2. **In-graph replay (decisive, vector cores = 40, per launch):**

| width | B | L | native pair | hpt4 (iter 002) | hpt8 | hpt16 |
| --- | --- | --- | --- | --- | --- | --- |
| 128 | 16 | 1 | 28.7µs | 12.6µs | 11.1µs | **10.5µs** |
| 128 | 16 | 2 | 88.1µs | 18.4µs | 12.5µs | **12.2µs** |
| 128 | 32 | 1 | 27.9µs | 20.2µs | 12.7µs | **11.4µs** |
| 128 | 32 | 2 | 161.7µs | 31.1µs | 18.3µs | **14.3µs** |
| 128 | 64 | 1 | 28.4µs | 34.3µs | 20.2µs | **15.6µs** |
| 128 | 64 | 2 | 307.7µs | 53.7µs | 39.2µs | **31.3µs** |

Per frame (14 steps at L=1 + 1 step at L=2), width 128: hpt4 costs ~195/314/534µs
at B16/32/64 while hpt16 costs ~159/174/250µs — the wide tile halves the
program count (2 tasks/row instead of 6) and saves ~36/140/284µs per frame.
The native pair collapses at L=2 with large batches (308µs at B64).

## Hypothesis

Widening the tile to 16 heads (one Q tile + one K tile per row) at rows ≥ 16
reduces per-step device latency where iteration 002 failed (C16–C64) while
keeping the iteration-002 kernel below 16 rows, so C1–C8's passing graphs stay
byte-identical.

## Modification

- `python/strata/predictor_rope.py`: heads-per-tile becomes a kernel constexpr;
  shipped `paired_rotary` selects 16 when `batch*length >= 16` else 4 (the
  iteration-002 width). `paired_rotary_width` exposes explicit widths for the
  bench/probe.
- `bench/rope_probe.py`: batch 16 added so the rows=16 and rows=32 boundary
  shapes of the guard are operator-verified.
- `bench/rope_bench.py`: new diagnostic microbench (eager + in-graph sections,
  exactness check for every width). Never an engine benchmark.

## Trial-and-error

1. **Eager microbench dead end (recorded, kept).** The first measurement showed
   width irrelevant (~220µs floor everywhere) — the hypothesis looked dead.
   Recognizing the floor as eager-dispatch artifact (the engine only replays
   in-graph) redirected the work to graph-replay timing, which reversed the
   conclusion.
2. **Width selection.** hpt16 wins or ties hpt8 at every measured shape with
   rows ≥ 16; both are exactly equal to the native operator. hpt16 at rows < 16
   was not measured and is not shipped: the guard keeps hpt4 there.
3. **Guard boundary.** rows=16 shapes (B16 L1 and B8 L2 — identical to the
   kernel) are measured for hpt16; rows<16 stays on the narrower tile to keep
   C1–C8 behavior bit-identical rather than assuming the unmeasured case.

## Validation (this tree, before merge)

Completed in the dedicated campaign container on the 910B3 device:

- [x] `bench/rope_probe.py`: status PASS — 40/40 unrounded cases exactly equal
      `npu_rotary_mul` across batches 1/4/16/32/64 × lengths 1/2 × widths
      64/128 × shared/per-request coefficients (both guard paths included);
      BF16 counterexample stays non-exact (max abs error 0.03125).
- [x] Paired complete-model probes (`--predictor-rope paired`): plain probe
      ids 0/1/64/65 and cancel/reuse probe ids 0/64/65/2 all finished with
      `stop` and every PCM SHA256 equals the recorded reference evidence
      hashes — byte-identical.

## Formal cohort

A fresh frozen campaign for this candidate (`formal-c003-01`, three alternating
AB/BA pairs, C1/2/4/8/16/32/64, 128 requests, identity checks around every
arm) is the acceptance evidence. Result recorded here when complete; missing
results stay INCONCLUSIVE.

## Result

Pending formal cohort; this file is updated when it completes.
