"""Load/capture for the strata-run launcher; no Python decode loop."""

import argparse
import hashlib
import json
from pathlib import Path

from reference import load_reference


def initialize(arguments):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--prewarm-decode", action="store_true")
    parser.add_argument("--predictor-rope", choices=["native", "paired"], default="native")
    args = parser.parse_args(list(arguments))
    if args.slots < 1 or args.slots & (args.slots - 1) or args.requests < 1:
        parser.error("positive requests and power-of-two slots required")
    load_reference(args.reference_root)
    from graph_model import NativeModel
    inputs = [json.loads(line) for line in args.workload.read_text().splitlines() if line.strip()]
    if len(inputs) < args.requests:
        raise ValueError("workload has fewer inputs than requested")
    inputs = inputs[:args.requests]
    for payload in inputs:
        payload.setdefault("instructions", "")
    args.output.mkdir(parents=True, exist_ok=False)
    model = NativeModel(str(args.model), "npu:0", max_slots=args.slots,
                        max_frames=args.max_new_tokens, prewarm_decode=args.prewarm_decode,
                        predictor_rope=args.predictor_rope)
    return {"model": model, "inputs": inputs, "slots": args.slots,
            "output": str(args.output),
            "workload_sha256": hashlib.sha256(args.workload.read_bytes()).hexdigest()}


def finish(session, records, wall_seconds):
    result = {"status": "INCONCLUSIVE", "scope": "native executable complete generation",
              "quality": "NOT_SCORED", "performance_acceptance": "NOT_EVALUATED",
              "workload_sha256": session["workload_sha256"],
              "wall_seconds": wall_seconds, "records": list(records)}
    (Path(session["output"]) / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
