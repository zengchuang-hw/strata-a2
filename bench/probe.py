"""Complete-model development probe; timings are diagnostic, not acceptance."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "strata"))

from reference import load_reference


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-root", type=Path, required=True)
    p.add_argument("--native-build", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--engine", choices=["reference", "native"], required=True)
    p.add_argument("--workload", type=Path, required=True)
    p.add_argument("--indices", default="0,1,64,65")
    p.add_argument("--slots", type=int, default=4)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cancel-and-reuse", action="store_true")
    p.add_argument("--predictor-rope", choices=["native", "paired"], default="native")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    load_reference(args.reference_root)
    sys.path.insert(0, str(args.native_build.resolve()))
    import torch
    import torch_npu  # noqa: F401
    from qwen3_tts_engine.continuous import ContinuousModel

    if args.engine == "native":
        from graph_model import NativeModel
        model = NativeModel(str(args.model), "npu:0", max_slots=args.slots,
                            predictor_rope=args.predictor_rope)
    else:
        model = ContinuousModel(
            str(args.model), "npu:0", max_slots=args.slots, pack_qkv=True,
            rms_norm="native", pack_mlp=True, fused_swiglu=True,
            native_gqa=True, text_only_rope=True, native_rope=True,
            native_rope_scope="predictor", triton_talker_rope=True,
            native_kv_update=True, triton_kv_update=True,
            incremental_codec=True, batch_prefill=True, graph_prefill=True,
            bucket_prefill=True, prewarm_prefill=True, direct_prompt=True,
            triton_snake=True, cache_codec_codebooks=True,
            codec_batch_size=4, triton_cdf=True, graph_embeddings=True,
            graph_sampling=True, prewarm_codec=True,
        )
    rows = [json.loads(line) for line in args.workload.read_text().splitlines() if line.strip()]
    inputs = []
    for index in map(int, args.indices.split(",")):
        payload = dict(rows[index])
        payload.setdefault("instructions", "")
        inputs.append((index, payload))
    if len(inputs) > args.slots:
        raise ValueError("probe inputs must fit the slot arena")
    first = {}
    chunks = {}
    records = []
    start = time.perf_counter()

    def emit(identifier, pcm, rate):
        if rate != 24000 or not pcm:
            raise ValueError("invalid audio chunk")
        first.setdefault(identifier, time.perf_counter() - start)
        chunks.setdefault(identifier, bytearray()).extend(pcm)

    try:
        for tick in range(2050):
            submissions = inputs if tick == 0 else []
            cancelled = [inputs[1][0]] if args.cancel_and_reuse and tick == 6 else []
            if args.cancel_and_reuse and tick == 7:
                extra = dict(rows[2], instructions="")
                submissions = [(2, extra)]
            for identifier, result in model.tick(submissions, emit, cancelled):
                pcm = result["pcm"]
                if result["finish_reason"] != "stop" or not pcm:
                    raise ValueError(f"request {identifier} incomplete: {result['finish_reason']}")
                if bytes(chunks[identifier]) != pcm:
                    raise ValueError("streamed audio differs from final audio")
                path = args.output / f"{identifier:05d}.wav"
                with wave.open(str(path), "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(24000)
                    wav.writeframes(pcm)
                records.append({"id": identifier, "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
                                "pcm_bytes": len(pcm), "frames": result["codec_frames"],
                                "first_audio_seconds": first[identifier],
                                "elapsed_seconds": time.perf_counter() - start,
                                "finish_reason": result["finish_reason"]})
            if not model.states:
                break
        else:
            raise RuntimeError("native probe did not finish within the frame bound")
        expected = {item[0] for item in inputs}
        if args.cancel_and_reuse:
            expected.remove(inputs[1][0])
            expected.add(2)
        if {r["id"] for r in records} != expected:
            raise ValueError("missing or unexpected completion")
        torch.npu.synchronize()
        result = {"status": "INCONCLUSIVE", "scope": "development complete-model probe",
                  "engine": args.engine, "records": records,
                  "wall_seconds": time.perf_counter() - start,
                  "cancel_and_reuse": args.cancel_and_reuse,
                  "quality": "NOT_SCORED", "performance_acceptance": "NOT_EVALUATED"}
        (args.output / "probe.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    finally:
        model.close()


if __name__ == "__main__":
    main()
