# Iteration 001 — bootstrap the layered runtime

Candidate: C001 / parent: none (campaign start) / contract: `docs/task-contract.md`

## Scope

Promote the minimal-complete mixed-bootstrap C++ runtime into the layered
framework layout and freeze it as the optimization parent. Semantics-preserving:
no kernel, scheduling or numeric change is intended in this iteration.

- New layout: `include/strata` + `src` (C++ core, C ABI, `strata-run` launcher),
  `python/strata` (bootstrap adapter), `serve`, `bench`, `tests`.
- Identity rename: extension module `_strata_native`, namespace `strata`,
  executable `strata-run`, C ABI `StrataRuntimeAPI`/`strata_runtime_api()`.
- New `python/strata/reference.py` holds the pinned-checkout guard used by
  every entrypoint (previously duplicated through `serve.py` imports).
- `bench/campaign.py` source fingerprinting now covers the whole source tree
  (the earlier per-directory glob would have missed the renamed layout) and
  excludes `build/`/`.git/` from the hashed set.
- The predictor rotary option is deliberately absent here; it returns with its
  own iteration and its own validation trail.

## Trial-and-error trail from the bootstrap phase (retained failures)

1. **Launcher could not find the virtual environment.** The executable was
   started with the system Python and failed to import the pinned stack. Fix:
   `PyConfig_SetBytesString` pins `program_name` to the build's interpreter
   (`STRATA_PYTHON_EXECUTABLE`), so the embedded interpreter resolves the same
   site-packages the CMake probe used.
2. **CANN initialization failed in the launcher.** Linking torch/torch_npu into
   the executable's initial loader namespace initialized accelerator runtime
   state before Python existed. Fix: the C ABI boundary — the executable links
   only Python/pybind11/dl, imports the extension through the normal Python
   bootstrap, then `dlopen`s the already-loaded module and queries
   `strata_runtime_api()`.
3. **Cold probes are not benchmarks.** Initial Triton compilation produced
   hundreds of kernel variants (several minutes), and first probes include
   graph capture. All cold wall times were recorded as diagnostics only and
   never promoted to speedup claims.
4. **Exploratory HTTP campaign C001 (pre-iteration measurement).** 128 fixed
   requests at C1/C4/C32, one reference/native pair, all 384 candidate outputs
   byte-identical to the baseline PCM, but throughput gains were only
   7.1% / 0.3% / 4.6%. The run also crossed a container glibc update during
   debugger installation, so it is not a qualified environment-controlled
   comparison; source, binaries and evidence are retained remotely under
   `candidates/c001`. Lessons taken into this campaign:
   - host-side micro-optimizations alone were not the bottleneck;
   - the environment (glibc, libstdc++, CANN install, model assets, binaries)
     must be fingerprinted before and after every arm — now enforced by
     `bench/campaign.py` with hard identity re-checks around each trial.
5. **Incompatible optional `kernels` package** in the shared image was removed
   only inside the dedicated campaign container; the shared development
   container was not modified.

## Risks

- The identity rename could miss a symbol and break the C ABI contract.
  Guarded by: clean build (`-Wall -Wextra`), compiled ownership tests, and a
  complete-model probe whose PCM must match the recorded reference hashes.
- Layout move changes import resolution. Guarded by the probe actually running
  through `strata-run`, `serve.py`, `campaign.py` and `probe.py` in the new
  tree.

## Validation (this iteration)

Recorded before merge, on one physical 910B3 device in the dedicated campaign
container (see `docs/task-contract.md` for environment identity):

- [ ] Clean CMake build of `_strata_native` + `strata-run` (build log retained).
- [ ] `tests/test_native_state.py`: compiled CPU ownership contracts pass.
- [ ] 4-request native probe (mixed English/Chinese, indices 0/1/64/65 of the
      frozen workload): EOS in every request, streamed chunks reconstruct the
      final PCM, and every PCM SHA256 equals the recorded reference probe hash.
- [ ] Cancellation/reuse probe: one live request cancelled, replacement
      admitted, all surviving outputs byte-identical to reference.

## Result

Pending the validation runs above; this file is updated before merge.
