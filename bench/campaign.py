"""Alternate complete reference/native HTTP trials and retain all evidence."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "strata"))

from reference import REFERENCE_SHA, load_reference

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def source_identity(root, native_build):
    files = {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*"))
             if p.is_file() and p.suffix in {".py", ".cpp", ".h", ".txt"}
             and "__pycache__" not in p.parts and "evidence" not in p.parts
             and "build" not in p.parts and ".git" not in p.parts}
    binaries = {p.name: sha256(p) for p in native_build.glob("_strata_native*.so")}
    if len(binaries) != 1:
        raise ValueError("exactly one native module is required")
    if not (native_build / "strata-run").is_file():
        raise ValueError("the native executable is missing")
    binaries["strata-run"] = sha256(native_build / "strata-run")
    return {"source": files, "binaries": binaries, "reference_revision": REFERENCE_SHA}


def runtime_identity(model, reference_root):
    packages = {name: importlib.metadata.version(name) for name in (
        "torch", "torch_npu", "qwen-tts", "transformers", "tokenizers", "numpy",
        "aiohttp", "triton", "triton_ascend", "huggingface-hub")}
    system = subprocess.check_output(
        ["dpkg-query", "-W", "-f=${binary:Package}=${Version}\n", "libc6", "libstdc++6"], text=True)
    cann = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"))
    install = cann / "aarch64-linux/ascend_toolkit_install.info"
    if not install.is_file():
        raise ValueError("CANN install identity is unavailable")
    expected = json.loads((reference_root / "evidence/campaign-20261017/results/model-audit.json").read_text())
    assets = {}
    for item in expected["files"]:
        if item["path"] in {"README.md", ".gitattributes"}:
            continue
        path = model / item["path"]
        digest = sha256(path)
        if digest != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise ValueError(f"model asset differs from pinned source: {item['path']}")
        assets[item["path"]] = digest
    if len(assets) != 11:
        raise ValueError("runtime model asset coverage differs")
    variables = {name: os.environ.get(name) for name in (
        "ASCEND_RT_VISIBLE_DEVICES", "ASCEND_HOME_PATH", "LD_LIBRARY_PATH", "LD_PRELOAD",
        "OMP_NUM_THREADS", "PYTHONPATH", "HF_HUB_OFFLINE", "TASK_QUEUE_ENABLE")}
    return {"packages": packages, "system_libraries": system, "python": sys.version,
            "cann_install_sha256": sha256(install), "variables": variables, "model_assets": assets}


def trial(args, arm, target):
    target.mkdir()
    command = json.loads((args.reference_root / "evidence/campaign-20261017/plan/continuous-direct-prompt-command.json").read_text())
    command[0] = sys.executable
    command[command.index("--model") + 1] = str(args.model)
    command[command.index("--port") + 1] = str(args.port)
    if arm == "native":
        command = [sys.executable, str(ROOT / "serve" / "serve.py"),
                   "--reference-root", str(args.reference_root), "--native-build", str(args.native_build),
                   "--model", str(args.model), "--port", str(args.port), "--max-batch-size", "64",
                   "--prewarm-decode", "--predictor-rope", args.predictor_rope]
    record = {"arm": arm, "server_command": command, "server_exit": None,
              "status": "INCONCLUSIVE", "cohorts": []}
    write(target / "lifecycle.json", record)
    with (target / "server.log").open("wb") as log:
        server = subprocess.Popen(command, cwd=args.reference_root, stdout=log, stderr=log,
                                  start_new_session=True)
        record["server_pid"] = server.pid
        try:
            deadline = time.monotonic() + args.readiness_timeout
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"server exited before readiness: {server.returncode}")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=2) as response:
                        health = json.load(response)
                    if health.get("status") == "ready":
                        break
                except (OSError, urllib.error.URLError):
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError("server readiness deadline exceeded")
                time.sleep(1)
            for concurrency in args.concurrencies:
                output = target / f"c{concurrency}"
                benchmark = [sys.executable, "-m", "qwen3_tts_engine.benchmark",
                             "--endpoint", f"http://127.0.0.1:{args.port}/v1/audio/speech",
                             "--model", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "--engine", arm,
                             "--source-revision", REFERENCE_SHA if arm == "reference" else args.candidate_identity,
                             "--workload", str(args.workload), "--output", str(output),
                             "--concurrency", str(concurrency), "--requests", str(args.requests),
                             "--warmup", str(max(2, concurrency)), "--frame-rate", "12.5"]
                with (target / f"benchmark-c{concurrency}.log").open("wb") as bench_log:
                    result = subprocess.run(benchmark, cwd=args.reference_root, stdout=bench_log,
                                            stderr=bench_log, timeout=args.benchmark_timeout)
                record["cohorts"].append({"concurrency": concurrency, "command": benchmark,
                                          "exit": result.returncode})
                write(target / "lifecycle.json", record)
                if result.returncode:
                    raise RuntimeError(f"C{concurrency} benchmark failed")
                print(f"{target.name} C{concurrency} completed", flush=True)
            record["status"] = "COMPLETED"
        except BaseException as error:
            record["status"] = "FAIL"
            record["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGINT)
                try:
                    server.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait()
            record["server_exit"] = server.returncode
            if server.returncode != 0:
                record["status"] = "FAIL"
            write(target / "lifecycle.json", record)
    if record["status"] != "COMPLETED":
        raise RuntimeError("server did not complete and shut down cleanly")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--native-build", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--requests", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--port", type=int, default=8199)
    parser.add_argument("--readiness-timeout", type=float, default=900)
    parser.add_argument("--benchmark-timeout", type=float, default=3600)
    parser.add_argument("--predictor-rope", choices=["native", "paired"], default="native")
    args = parser.parse_args()
    if (args.repetitions < 1 or args.requests < 1 or min(args.concurrencies) < 1
            or max(args.concurrencies) > 64 or len(set(args.concurrencies)) != len(args.concurrencies)
            or min(args.readiness_timeout, args.benchmark_timeout) <= 0):
        parser.error("invalid cohort or time bounds")
    for name in ("reference_root", "native_build", "model", "workload", "output"):
        setattr(args, name, getattr(args, name).resolve())
    load_reference(args.reference_root)
    args.output.mkdir(parents=True, exist_ok=False)
    source = source_identity(ROOT, args.native_build)
    runtime = runtime_identity(args.model, args.reference_root)
    write(args.output / "environment-identity.json", runtime)
    args.candidate_identity = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    write(args.output / "identity.json", source)
    write(args.output / "runspec.json", {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                                        "workload_sha256": sha256(args.workload),
                                        "formal_coverage": args.repetitions >= 3 and args.requests == 128
                                        and sorted(args.concurrencies) == [1, 2, 4, 8, 16, 32, 64]})
    summary = {"status": "INCONCLUSIVE", "quality": "NOT_SCORED", "completed": []}
    try:
        for repetition in range(args.repetitions):
            order = ["reference", "native"] if repetition % 2 == 0 else ["native", "reference"]
            for arm in order:
                if source_identity(ROOT, args.native_build) != source:
                    raise ValueError("candidate source or binary changed during campaign")
                if runtime_identity(args.model, args.reference_root) != runtime:
                    raise ValueError("runtime environment or model changed during campaign")
                load_reference(args.reference_root)
                name = f"r{repetition + 1}-{arm}"
                trial(args, arm, args.output / name)
                if source_identity(ROOT, args.native_build) != source:
                    raise ValueError("candidate source or binary changed during trial")
                if runtime_identity(args.model, args.reference_root) != runtime:
                    raise ValueError("runtime environment or model changed during trial")
                load_reference(args.reference_root)
                summary["completed"].append(name)
                write(args.output / "campaign.json", summary)
    except BaseException as error:
        summary.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        # Measurement completion never substitutes for independent quality gates.
        write(args.output / "campaign.json", summary)


if __name__ == "__main__":
    main()
