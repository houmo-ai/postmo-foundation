import asyncio
import json
from pathlib import Path
import websockets

WS_URL = "ws://127.0.0.1:7803/ws/audio/transcriptions"
AUDIO_PATH = "../audio/long_audio.mp3"


async def recv_until_transcript_or_error(ws, tag=""):
    """读取服务端消息直到收到 transcript/error/done。"""
    while True:
        raw = await ws.recv()
        print(f"{tag} recv:", raw)
        msg = json.loads(raw)
        if msg.get("type") in {"transcript", "error", "done"}:
            return msg


async def send_full_audio_once(ws, audio_bytes: bytes):
    print("\n=== 模式1：一次性发送整段音频 ===")
    await ws.send(audio_bytes)
    msg = await recv_until_transcript_or_error(ws, tag="[FULL]")
    return msg


async def send_audio_in_chunks(ws, audio_bytes: bytes, chunk_size: int = 32000):
    print("\n=== 模式2：分片实时发送 ===")
    # 可选：清理上一轮上下文
    await ws.send(json.dumps({"type": "reset"}))
    reset_ack = await ws.recv()
    print("[CHUNK] recv:", reset_ack)

    pos = 0
    while pos < len(audio_bytes):
        chunk = audio_bytes[pos:pos + chunk_size]
        pos += len(chunk)

        await ws.send(chunk)
        msg = await recv_until_transcript_or_error(ws, tag="[CHUNK]")
        if msg.get("type") == "error":
            print("[CHUNK] error:", msg)
            return msg

    return {"type": "done_local"}


async def main():
    audio_bytes = Path(AUDIO_PATH).read_bytes()

    async with websockets.connect(WS_URL, max_size=64 * 1024 * 1024) as ws:
        # 1) ready
        ready = await ws.recv()
        print("recv:", ready)

        # 2) config
        await ws.send(json.dumps({
            "type": "config",
            "language": "zh",
            "prompt": "",
            "return_mode": "delta"  # delta or full
        }, ensure_ascii=False))
        ack = await ws.recv()
        print("recv:", ack)

        # 3) 一次性发送
        await send_full_audio_once(ws, audio_bytes)

        # 4) 分片发送（模拟实时）
        await send_audio_in_chunks(ws, audio_bytes, chunk_size=32000)

        # 5) stop，获取最终文本
        await ws.send(json.dumps({"type": "stop"}))
        try:
            done = await ws.recv()
            print("recv:", done)
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
