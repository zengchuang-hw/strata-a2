"""Compare paired rotary with the exact selected native operator on real shapes."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "strata"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    import torch_npu
    from predictor_rope import paired_rotary
    torch.npu.set_device(0)
    torch.manual_seed(20261019)
    rows = []
    with torch.inference_mode():
        for width in (64, 128):
            for batch in (1, 4, 16, 32, 64):
                for length in (1, 2):
                    for shared in (True, False):
                        query = torch.randn(batch, length, 16, width, device="npu:0", dtype=torch.bfloat16)
                        key = torch.randn(batch, length, 8, width, device="npu:0", dtype=torch.bfloat16)
                        angle = torch.randn(1 if shared else batch, length, width // 2, device="npu:0")
                        cos = torch.cat((angle.cos(), angle.cos()), -1).bfloat16()
                        sin = torch.cat((angle.sin(), angle.sin()), -1).bfloat16()
                        reference = [torch_npu.npu_rotary_mul(t, cos.unsqueeze(2), sin.unsqueeze(2)) for t in (query, key)]
                        for rounded in (False, True):
                            result = paired_rotary(query, key, cos, sin, round_bf16=rounded)
                            exact = all(torch.equal(a, b) for a, b in zip(reference, result))
                            error = max((a.float() - b.float()).abs().max().item() for a, b in zip(reference, result))
                            rows.append({"batch": batch, "length": length, "width": width,
                                         "shared_coefficients": shared, "round_bf16": rounded,
                                         "exact": exact, "max_abs_error": error})
        torch.npu.synchronize()
    unrounded = [r for r in rows if not r["round_bf16"]]
    result = {"status": "PASS" if all(r["exact"] for r in unrounded) else "FAIL",
              "scope": "paired predictor rotary versus native operator; not full speech quality or performance",
              "cases": rows}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
