"""Serve the native runtime using the pinned reference streaming protocol."""

import argparse
from functools import partial
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "strata"))

from reference import load_reference


def validate_native_limit(payload, capacity):
    if isinstance(payload, dict):
        limit = payload.get("max_new_tokens", 2048)
        if type(limit) is int and not 2 <= limit <= capacity:
            raise ValueError(f"max_new_tokens must fit the native arena [2,{capacity}]")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--native-build", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8199)
    parser.add_argument("--max-batch-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--text-capacity", type=int, default=256)
    parser.add_argument("--codec-batch-size", type=int, default=4)
    parser.add_argument("--queue-size", type=int, default=128)
    parser.add_argument("--prefill-per-tick", type=int, default=64)
    parser.add_argument("--prewarm-decode", action="store_true")
    parser.add_argument("--predictor-rope", choices=["native", "paired"], default="native")
    options = parser.parse_args()
    if (options.max_batch_size < 1 or options.max_batch_size & (options.max_batch_size - 1)
            or options.queue_size < 1 or options.prefill_per_tick < 1
            or options.codec_batch_size < 1 or options.codec_batch_size > 8
            or options.max_new_tokens < 2 or options.text_capacity < 1):
        parser.error("invalid native arena or scheduling bounds")
    load_reference(options.reference_root)
    sys.path.insert(0, str(options.native_build.resolve()))
    from aiohttp import web
    from graph_model import NativeModel
    from qwen3_tts_engine.server import SpeechService

    options.continuous = options.streaming = True
    options.prefill_interval = 1
    options.batch_wait_ms = 0
    factory = partial(NativeModel, max_slots=options.max_batch_size,
                      max_frames=options.max_new_tokens, text_capacity=options.text_capacity,
                      codec_batch_size=options.codec_batch_size,
                      prewarm_decode=options.prewarm_decode,
                      predictor_rope=options.predictor_rope)
    service = SpeechService(options, factory)

    async def bounded_speech(incoming):
        try:
            validate_native_limit(await incoming.json(), options.max_new_tokens)
        except (ValueError, TypeError) as error:
            return web.json_response({"error": str(error)}, status=400)
        return await service.speech(incoming)

    app = web.Application(client_max_size=1024 * 1024)
    app.router.add_get("/health", service.health)
    app.router.add_post("/v1/audio/speech", bounded_speech)
    app.on_startup.append(service.start)
    app.on_cleanup.append(service.stop)
    logging.basicConfig(level=logging.INFO)
    web.run_app(app, host=options.host, port=options.port, handler_cancellation=True)


if __name__ == "__main__":
    main()
