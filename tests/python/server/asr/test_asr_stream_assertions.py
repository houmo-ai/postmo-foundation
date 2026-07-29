from __future__ import annotations

import asyncio
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))

from asr_stream_assertions import TranscriptionStreamAssertions
from asr_websocket import (
    assert_vad_quality,
    character_error_rate,
    configure_utf8_console,
    effective_vad_trailing_silence_ms,
    limit_pcm16_audio_bytes,
    read_wav_pcm,
    receive_transcripts,
)


def delta(item_id: str, text: str) -> dict[str, object]:
    return {
        "type": "conversation.item.input_audio_transcription.delta",
        "item_id": item_id,
        "content_index": 0,
        "delta": text,
    }


def completed(item_id: str, text: str) -> dict[str, object]:
    return {
        "type": "conversation.item.input_audio_transcription.completed",
        "item_id": item_id,
        "content_index": 0,
        "transcript": text,
    }


class FakeWebSocket:
    def __init__(self, events: list[dict[str, object]]) -> None:
        self.events = list(events)

    async def recv(self) -> str:
        if self.events:
            return json.dumps(self.events.pop(0), ensure_ascii=False)
        raise asyncio.TimeoutError


class TranscriptionStreamAssertionsTest(unittest.TestCase):
    def test_utf8_console_configuration_is_safe(self) -> None:
        configure_utf8_console()

    def test_extends_trailing_silence_to_close_final_vad_turn(self) -> None:
        self.assertEqual(effective_vad_trailing_silence_ms(750, 2000), 2200)
        self.assertEqual(effective_vad_trailing_silence_ms(2500, 2000), 2500)

    def test_vad_quality_ignores_case_whitespace_and_punctuation(self) -> None:
        self.assertEqual(character_error_rate("Hello，世界！", " hello 世界 "), 0.0)
        assert_vad_quality("今天把两个人叫过来。", "今天把两个人叫过来", 0.08, 0.85, 1.15)

    def test_vad_quality_rejects_missing_content(self) -> None:
        with self.assertRaisesRegex(AssertionError, "CER"):
            assert_vad_quality("今天把两个人叫过来", "今天", 0.08, 0.85, 1.15)

    def test_reads_wave_format_extensible_pcm(self) -> None:
        fmt = (
            struct.pack("<HHIIHHH", 0xFFFE, 2, 16_000, 64_000, 4, 16, 22)
            + struct.pack("<HI", 16, 3)
            + struct.pack("<H14s", 1, b"\0" * 14)
        )
        pcm = struct.pack("<hhhh", 1000, 3000, -1000, 1000)
        body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
        body += b"data" + struct.pack("<I", len(pcm)) + pcm
        wav = b"RIFF" + struct.pack("<I", len(body)) + body

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "extensible.wav"
            path.write_bytes(wav)
            samples, sample_rate = read_wav_pcm(path)

        self.assertEqual(sample_rate, 16_000)
        self.assertEqual(samples.size, 2)
        self.assertAlmostEqual(float(samples[0]), 2000.0 / 32768.0, places=6)
        self.assertAlmostEqual(float(samples[1]), 0.0, places=6)

    def test_limits_audio_to_original_bug_payload_size(self) -> None:
        samples = np.arange(60_000, dtype=np.float32)
        limited = limit_pcm16_audio_bytes(samples, 96_000)
        self.assertEqual(limited.size, 48_000)

    def test_rejects_odd_pcm16_byte_limit(self) -> None:
        samples = np.zeros(10, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "must be even"):
            limit_pcm16_audio_bytes(samples, 3)

    def test_accepts_multiple_deltas_that_exactly_match_completed(self) -> None:
        assertions = TranscriptionStreamAssertions()
        assertions.consume(delta("item_1", "今天把两个人"))
        assertions.consume(delta("item_1", "叫过来，主要。"))
        assertions.consume(completed("item_1", "今天把两个人叫过来，主要。"))

    def test_rejects_completed_without_any_delta(self) -> None:
        assertions = TranscriptionStreamAssertions()
        with self.assertRaisesRegex(
            AssertionError,
            "No non-empty conversation.item.input_audio_transcription.delta",
        ):
            assertions.consume(completed("item_1", "今天把两个人叫过来，主要。"))

    def test_rejects_only_empty_deltas(self) -> None:
        assertions = TranscriptionStreamAssertions()
        assertions.consume(delta("item_1", ""))
        with self.assertRaisesRegex(
            AssertionError,
            "No non-empty conversation.item.input_audio_transcription.delta",
        ):
            assertions.consume(completed("item_1", ""))

    def test_rejects_delta_sum_mismatch(self) -> None:
        assertions = TranscriptionStreamAssertions()
        assertions.consume(delta("item_1", "今天把两个人"))
        with self.assertRaisesRegex(AssertionError, "Delta sum != completed transcript"):
            assertions.consume(completed("item_1", "今天把两个人叫过来，主要。"))

    def test_rejects_qwen_internal_markers(self) -> None:
        for leaked_text in (
            "language Chinese<asr_text>今天把两个人叫过来，主要。",
            "今天把两个人叫过来，主要。</asr_text>",
        ):
            with self.subTest(leaked_text=leaked_text):
                assertions = TranscriptionStreamAssertions()
                with self.assertRaisesRegex(AssertionError, "Qwen ASR internal marker"):
                    assertions.consume(delta("item_1", leaked_text))

    def test_tracks_interleaved_items_independently(self) -> None:
        assertions = TranscriptionStreamAssertions()
        assertions.consume(delta("item_1", "第一条"))
        assertions.consume(delta("item_2", "第二条"))
        assertions.consume(completed("item_2", "第二条"))
        assertions.consume(completed("item_1", "第一条"))

    def test_rejects_delta_after_completed(self) -> None:
        assertions = TranscriptionStreamAssertions()
        assertions.consume(delta("item_1", "完成"))
        assertions.consume(completed("item_1", "完成"))
        with self.assertRaisesRegex(AssertionError, "Delta received after completed"):
            assertions.consume(delta("item_1", "额外文本"))

    def test_rejects_stream_that_ends_before_completed(self) -> None:
        assertions = TranscriptionStreamAssertions()
        assertions.consume(delta("item_1", "未完成"))
        with self.assertRaisesRegex(AssertionError, "stream ended before completed"):
            assertions.finish()


class ReceiveTranscriptsStrictModeTest(unittest.IsolatedAsyncioTestCase):
    async def test_strict_mode_accepts_openai_realtime_event_sequence(self) -> None:
        websocket = FakeWebSocket(
            [
                {"type": "input_audio_buffer.committed", "item_id": "item_1"},
                delta("item_1", "今天把两个人"),
                delta("item_1", "叫过来，主要。"),
                completed("item_1", "今天把两个人叫过来，主要。"),
            ]
        )
        sender_done = asyncio.Event()
        sender_done.set()

        transcript = await receive_transcripts(
            websocket,
            sender_done,
            final_timeout=1.0,
            assert_streaming=True,
        )

        self.assertEqual(transcript, "今天把两个人叫过来，主要。")

    async def test_strict_mode_rejects_original_zero_delta_bug(self) -> None:
        websocket = FakeWebSocket(
            [
                {"type": "input_audio_buffer.committed", "item_id": "item_1"},
                completed(
                    "item_1",
                    "language Chinese<asr_text>今天把两个人叫过来，主要。",
                ),
            ]
        )
        sender_done = asyncio.Event()
        sender_done.set()

        with self.assertRaises(AssertionError):
            await receive_transcripts(
                websocket,
                sender_done,
                final_timeout=1.0,
                assert_streaming=True,
            )


if __name__ == "__main__":
    unittest.main()
