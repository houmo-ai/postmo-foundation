"""Assertions for OpenAI-compatible input audio transcription events."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


DELTA_EVENT = "conversation.item.input_audio_transcription.delta"
COMPLETED_EVENT = "conversation.item.input_audio_transcription.completed"
QWEN_INTERNAL_MARKERS = (
    "<asr_text>",
    "</asr_text>",
)


@dataclass
class _Turn:
    deltas: list[str] = field(default_factory=list)
    transcript: str | None = None


class TranscriptionStreamAssertions:
    """Validate append-only transcription deltas for each Realtime item."""

    def __init__(self, *, require_delta: bool = True, reject_internal_markers: bool = True) -> None:
        self.require_delta = require_delta
        self.reject_internal_markers = reject_internal_markers
        self._turns: dict[str, _Turn] = {}

    def consume(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")
        if event_type not in (DELTA_EVENT, COMPLETED_EVENT):
            return

        item_id = event.get("item_id")
        if not isinstance(item_id, str) or not item_id:
            raise AssertionError(f"{event_type} must contain a non-empty item_id: {event!r}")

        turn = self._turns.setdefault(item_id, _Turn())
        if event_type == DELTA_EVENT:
            if turn.transcript is not None:
                raise AssertionError(f"Delta received after completed for item {item_id}")
            delta = event.get("delta")
            if not isinstance(delta, str):
                raise AssertionError(f"Delta must be a string for item {item_id}: {delta!r}")
            self._assert_no_internal_markers(delta, item_id, "delta")
            turn.deltas.append(delta)
            return

        transcript = event.get("transcript")
        if not isinstance(transcript, str):
            raise AssertionError(f"Completed transcript must be a string for item {item_id}: {transcript!r}")
        if turn.transcript is not None:
            raise AssertionError(f"Duplicate completed event for item {item_id}")

        self._assert_no_internal_markers(transcript, item_id, "completed transcript")
        delta_sum = "".join(turn.deltas)
        if self.require_delta and not any(turn.deltas):
            raise AssertionError(
                f"No non-empty {DELTA_EVENT} event received for item {item_id}; completed={transcript!r}"
            )
        if delta_sum != transcript:
            raise AssertionError(
                f"Delta sum != completed transcript for item {item_id}: "
                f"deltas={delta_sum!r} completed={transcript!r}"
            )
        turn.transcript = transcript

    def finish(self) -> None:
        incomplete = [item_id for item_id, turn in self._turns.items() if turn.transcript is None]
        if incomplete:
            raise AssertionError(f"Transcription stream ended before completed for items: {incomplete!r}")

    def _assert_no_internal_markers(self, text: str, item_id: str, source: str) -> None:
        if not self.reject_internal_markers:
            return
        marker = next((value for value in QWEN_INTERNAL_MARKERS if value in text), None)
        if marker is not None:
            raise AssertionError(
                f"Qwen ASR internal marker {marker!r} leaked in {source} for item {item_id}: {text!r}"
            )
