from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi.responses import Response
from pydantic import Field
from vllm.entrypoints.openai.engine.protocol import ErrorResponse

from hliepython.model_server.engines.protocol.chat_completions import (
    OpenAIBaseModel,
)


DEFAULT_PROMPT_WAV_PATH = str(
    Path(__file__).resolve().parents[1] / "files" / "audio" / "alloy.mp3"
)


class AudioSpeechRequest(OpenAIBaseModel):
    input: str | None = Field(default=None, min_length=1)
    model: str
    voice: str | None = Field(default="alloy")
    instructions: str | None = None
    prompt_wav_path: str = Field(default=DEFAULT_PROMPT_WAV_PATH)
    response_format: Literal[
        "mp3",
        "opus",
        "aac",
        "flac",
        "wav",
        "pcm",
        "b64_json",
    ] | None = None
    speed: float | None = Field(default=1.0, ge=0.25, le=4.0)
    text_frontend: bool | None = True


class AudioSpeechBase64Response(OpenAIBaseModel):
    audio: str
    format: str = "wav"
    sample_rate: int
    voice: str
    model: str | None = None


AudioSpeechResponse = Response | AudioSpeechBase64Response | ErrorResponse