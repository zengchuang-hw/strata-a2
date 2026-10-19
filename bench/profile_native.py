"""Diagnostic CPU/NPU trace of a warmed native frame graph; never a benchmark."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "strata"))

from reference import load_reference


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-root", type=Path, required=True)
    p.add_argument("--native-build", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--workload", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--predictor-rope", choices=["native", "paired"], default="native")
    args = p.parse_args()
    if not 1 <= args.steps <= 16:
        p.error("diagnostic steps must be 1..16")
    load_reference(args.reference_root)
    sys.path.insert(0, str(args.native_build.resolve()))
    import torch
    import torch_npu
    from graph_model import NativeModel

    args.output.mkdir(parents=True, exist_ok=False)
    data = args.workload.read_bytes()
    rows = [dict(json.loads(line), instructions="") for line in data.splitlines() if line.strip()]
    model = NativeModel(str(args.model), "npu:0", max_slots=4, predictor_rope=args.predictor_rope)
    try:
        model.tick([(i, rows[i]) for i in (2, 64, 65, 66)])
        for _ in range(7):
            model.tick()
        torch.npu.synchronize()
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            record_shapes=False,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(args.output)),
        ):
            for _ in range(args.steps):
                with torch.profiler.record_function("strata_native_frame"):
                    model.tick()
            torch.npu.synchronize()
        summary = {"status": "DIAGNOSTIC_ONLY", "scope": "profiled frame/codec execution; not service performance",
                   "workload_sha256": hashlib.sha256(data).hexdigest(), "steps": args.steps,
                   "active_requests": len(model.states), "decode_shapes": sorted(model.native_decodes),
                   "reference_revision": "b4451682abac651f26046f8d27467de3f007e1e7"}
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    finally:
        model.close()


if __name__ == "__main__":
    main()
