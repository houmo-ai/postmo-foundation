import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import websockets


def pretty_event(tag: str, msg: dict[str, Any]) -> None:
    print(f"{tag} {json.dumps(msg, ensure_ascii=False)}")


async def recv_until_target(ws: Any) -> dict[str, Any]:
    while True:
        raw = await ws.recv()
        msg = json.loads(raw)
        pretty_event("[RECV]", msg)
        if msg.get("type") in {"transcript", "error", "done"}:
            return msg


async def send_full_once(ws: Any, audio_bytes: bytes) -> None:
    await ws.send(json.dumps({"type": "reset"}))
    pretty_event("[SEND]", {"type": "reset"})
    reset_ack = json.loads(await ws.recv())
    pretty_event("[RECV]", reset_ack)

    await ws.send(audio_bytes)
    pretty_event(
        "[SEND]",
        {
            "type": "audio_full",
            "size": len(audio_bytes),
        },
    )
    msg = await recv_until_target(ws)
    if msg.get("type") == "error":
        raise RuntimeError(str(msg))


async def run_chunk_mode(
    ws: Any,
    audio_bytes: bytes,
    chunk_size: int,
) -> None:
    await ws.send(json.dumps({"type": "reset"}))
    pretty_event("[SEND]", {"type": "reset"})
    reset_ack = json.loads(await ws.recv())
    pretty_event("[RECV]", reset_ack)

    pos = 0
    while pos < len(audio_bytes):
        chunk = audio_bytes[pos: pos + chunk_size]
        pos += len(chunk)
        await ws.send(chunk)
        pretty_event(
            "[SEND]",
            {
                "type": "audio_chunk",
                "size": len(chunk),
                "pos": pos,
                "total": len(audio_bytes),
            },
        )
        msg = await recv_until_target(ws)
        if msg.get("type") == "error":
            raise RuntimeError(str(msg))


async def main() -> None:
    parser = argparse.ArgumentParser(description="WebSocket ASR demo")
    parser.add_argument("--url", default="ws://127.0.0.1:7803/ws/audio/transcriptions")
    parser.add_argument("--audio", default="../audio/long_audio.mp3")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--prompt", default="")
    parser.add_argument(
        "--mode",
        default="delta",
        choices=["delta", "full", "chunk_only"],
        help="Response mode from server",
    )
    parser.add_argument("--chunk-size", type=int, default=32000)
    parser.add_argument("--max-size", type=int, default=64 * 1024 * 1024)
    args = parser.parse_args()

    audio_bytes = Path(args.audio).read_bytes()

    async with websockets.connect(args.url, max_size=args.max_size) as ws:
        ready = json.loads(await ws.recv())
        pretty_event("[RECV]", ready)

        config_msg = {
            "type": "config",
            "language": args.language,
            "prompt": args.prompt,
            "return_mode": args.mode,
        }
        await ws.send(json.dumps(config_msg, ensure_ascii=False))
        pretty_event("[SEND]", config_msg)
        ack = json.loads(await ws.recv())
        pretty_event("[RECV]", ack)

        if args.mode == "full":
            await send_full_once(ws, audio_bytes)
        else:
            await run_chunk_mode(ws, audio_bytes, args.chunk_size)

        await ws.send(json.dumps({"type": "stop"}))
        pretty_event("[SEND]", {"type": "stop"})
        try:
            done = json.loads(await ws.recv())
            pretty_event("[RECV]", done)
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
