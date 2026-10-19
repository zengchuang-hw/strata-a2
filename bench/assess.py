"""Audit complete cohort evidence and compare the frozen performance gates."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import wave


class EvidenceViolation(ValueError):
    """Observed invalid output or lifecycle, rather than absent evidence."""


def load(path):
    return json.loads(path.read_text())


def percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    low = int(index)
    return ordered[low] + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) * (index - low)


def audit_cohort(path, requests, concurrency, workload_sha):
    spec, result = load(path / "runspec.json"), load(path / "results.json")
    if (spec["requests"] != requests or spec["concurrency"] != concurrency
            or spec["workload_sha256"] != workload_sha or spec["sample_rate"] != 24000
            or spec["frame_rate"] != 12.5):
        raise EvidenceViolation("cohort identity or measurement definition differs")
    rows = result["requests"]
    if len(rows) != requests or sorted(r["index"] for r in rows) != list(range(requests)):
        raise ValueError("request coverage is incomplete or duplicated")
    hashes = {}
    for row in rows:
        terminal = row.get("terminal") or {}
        if (row.get("success") is not True or terminal.get("type") != "speech.audio.done"
                or terminal.get("finish_reason") != "stop"):
            raise EvidenceViolation("failed, truncated or unterminated request")
        with wave.open(str(path / f"{row['index']:05d}.wav"), "rb") as audio:
            if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, 24000):
                raise EvidenceViolation("audio format differs")
            pcm = audio.readframes(audio.getnframes())
        digest = hashlib.sha256(pcm).hexdigest()
        if not pcm or digest != row["pcm_sha256"] or len(pcm) != row["pcm_bytes"]:
            raise EvidenceViolation("retained audio does not match benchmark evidence")
        if not math.isclose(len(pcm) / 48000, row["audio_seconds"], rel_tol=1e-9):
            raise EvidenceViolation("audio duration differs")
        times = [row[k] for k in ("ttfa_seconds", "elapsed_seconds", "audio_seconds")]
        if not all(math.isfinite(v) and v > 0 for v in times) or times[0] > times[1]:
            raise EvidenceViolation("invalid request timing")
        chunks = row["chunks"]
        if sum(c["bytes"] for c in chunks) != len(pcm):
            raise EvidenceViolation("streamed audio coverage differs")
        first = next((c for c in chunks if c["bytes"]), None)
        if first is None or not math.isclose(first["seconds"], times[0], rel_tol=1e-9):
            raise EvidenceViolation("first audio timing differs")
        hashes[row["index"]] = digest
    metrics = result["metrics"]
    wall = metrics["wall_seconds"]
    if not math.isfinite(wall) or wall <= 0:
        raise EvidenceViolation("invalid cohort wall time")
    actual = {
        "throughput_frames_s": sum(r["audio_seconds"] for r in rows) * 12.5 / wall,
        "ttfa_p50_ms": percentile([r["ttfa_seconds"] * 1000 for r in rows], .5),
        "ttfa_p95_ms": percentile([r["ttfa_seconds"] * 1000 for r in rows], .95),
        "rtf_mean": statistics.mean(r["elapsed_seconds"] / r["audio_seconds"] for r in rows),
    }
    if any(not math.isclose(metrics[k], value, rel_tol=1e-8) for k, value in actual.items()):
        raise EvidenceViolation("reported aggregate differs from request evidence")
    return actual, hashes


def assess(root):
    spec = load(root / "runspec.json")
    campaign = load(root / "campaign.json")
    repetitions = spec["repetitions"]
    expected = {f"r{r}-{arm}" for r in range(1, repetitions + 1) for arm in ("reference", "native")}
    if campaign["status"] == "FAIL":
        raise EvidenceViolation("campaign execution failed")
    if set(campaign["completed"]) != expected:
        raise ValueError("campaign did not complete all requested arms")
    cohorts = {}
    for name in sorted(expected):
        lifecycle = load(root / name / "lifecycle.json")
        if lifecycle["status"] != "COMPLETED" or lifecycle["server_exit"] != 0:
            raise EvidenceViolation("server lifecycle did not complete cleanly")
        if sorted(c["concurrency"] for c in lifecycle["cohorts"]) != sorted(spec["concurrencies"]):
            raise ValueError("server cohort coverage differs")
        if any(c["exit"] != 0 for c in lifecycle["cohorts"]):
            raise EvidenceViolation("benchmark process failed")
        for concurrency in spec["concurrencies"]:
            cohorts[name, concurrency] = audit_cohort(root / name / f"c{concurrency}",
                spec["requests"], concurrency, spec["workload_sha256"])
    rows = []
    quality_exact = True
    performance_pass = True
    for concurrency in spec["concurrencies"]:
        baseline = [cohorts[f"r{r}-reference", concurrency][0] for r in range(1, repetitions + 1)]
        native = [cohorts[f"r{r}-native", concurrency][0] for r in range(1, repetitions + 1)]
        exact = all(cohorts[f"r{r}-reference", concurrency][1] == cohorts[f"r{r}-native", concurrency][1]
                    for r in range(1, repetitions + 1))
        quality_exact &= exact
        throughput_ratio = min(m["throughput_frames_s"] for m in native) / max(m["throughput_frames_s"] for m in baseline)
        ttfa_ratio = max(m["ttfa_p95_ms"] for m in native) / min(m["ttfa_p95_ms"] for m in baseline)
        passed = throughput_ratio > 1.05 and ttfa_ratio <= 1.05
        performance_pass &= passed
        rows.append({"concurrency": concurrency, "reference": baseline, "native": native,
                     "min_native_vs_max_reference_throughput": throughput_ratio,
                     "max_native_vs_min_reference_ttfa_p95": ttfa_ratio,
                     "performance": "PASS" if passed else "FAIL", "exact_pcm": exact})
    formal = (spec.get("formal_coverage") is True and repetitions >= 3 and spec["requests"] == 128
              and sorted(spec["concurrencies"]) == [1, 2, 4, 8, 16, 32, 64])
    status = "FAIL" if not performance_pass else "PASS" if formal and quality_exact else "INCONCLUSIVE"
    return {"status": status, "formal_coverage": formal, "cohorts": rows,
            "performance": "PASS" if performance_pass and formal else "FAIL" if not performance_pass else "INCONCLUSIVE",
            "relative_quality": "PASS_EXACT_PCM" if quality_exact else "INCONCLUSIVE_REQUIRES_ASR_AND_SPEAKER",
            "scope": "matched local comparison; exact PCM establishes relative equivalence only",
            "subjective_quality": "NOT_EVALUATED"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = assess(args.run)
    except EvidenceViolation as error:
        result = {"status": "FAIL", "error": str(error)}
    except (OSError, ValueError, KeyError, TypeError, wave.Error) as error:
        result = {"status": "INCONCLUSIVE", "error": f"{type(error).__name__}: {error}"}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return {"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 2}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
