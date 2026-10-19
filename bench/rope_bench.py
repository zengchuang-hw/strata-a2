"""Diagnostic per-launch rotary timing; never a benchmark of the engine.

Compares the fused paired kernel at several heads-per-tile widths against the
reference native operator pair at decode-relevant shapes, and checks exactness
of every width in the same run. Microbench numbers select a tile-width guard;
they are not end-to-end evidence.
"""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "strata"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="path of the JSON result file to write")
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--widths", type=int, nargs="+", default=[4, 8, 16],
                        help="heads-per-tile variants to measure")
    args = parser.parse_args()
    import torch
    import torch_npu
    import predictor_rope as pr
    from predictor_rope import paired_rotary
    torch.npu.set_device(0)
    torch.manual_seed(20261019)
    cores = torch.npu.get_device_properties(0).vector_core_num
    rows = []
    with torch.inference_mode():
        for width in (64, 128):
            for batch in (1, 16, 32, 64):
                for length in (1, 2):
                    query = torch.randn(batch, length, 16, width, device="npu:0", dtype=torch.bfloat16)
                    key = torch.randn(batch, length, 8, width, device="npu:0", dtype=torch.bfloat16)
                    angle = torch.randn(1, length, width // 2, device="npu:0")
                    cos = torch.cat((angle.cos(), angle.cos()), -1).bfloat16()
                    sin = torch.cat((angle.sin(), angle.sin()), -1).bfloat16()

                    def native():
                        for t in (query, key):
                            torch_npu.npu_rotary_mul(t, cos.unsqueeze(2), sin.unsqueeze(2))

                    reference = [torch_npu.npu_rotary_mul(t, cos.unsqueeze(2), sin.unsqueeze(2))
                                 for t in (query, key)]
                    for hpt in args.widths:
                        result = pr.paired_rotary_width(query, key, cos, sin, heads_per_tile=hpt)
                        exact = all(torch.equal(a, b) for a, b in zip(reference, result))

                        def run(fn):
                            for _ in range(10):
                                fn()
                            torch.npu.synchronize()
                            start = time.perf_counter()
                            for _ in range(args.repeats):
                                fn()
                            torch.npu.synchronize()
                            return (time.perf_counter() - start) / args.repeats * 1e6

                        native_us = run(native)
                        paired_us = run(lambda h=hpt: pr.paired_rotary_width(query, key, cos, sin,
                                                                             heads_per_tile=h))
                        rows.append({"width": width, "batch": batch, "length": length,
                                     "heads_per_tile": hpt, "exact": exact,
                                     "native_pair_us": round(native_us, 2),
                                     "paired_us": round(paired_us, 2)})
        torch.npu.synchronize()
    result = {"status": "DIAGNOSTIC_ONLY",
              "scope": "per-launch rotary microbench + width exactness; not engine performance",
              "vector_core_num": cores, "repeats": args.repeats, "cases": rows}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
