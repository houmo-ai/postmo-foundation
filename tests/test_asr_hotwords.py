from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from hliepython.model_server.engines import asr_engine


class _Renderer:
    async def render_cmpl_async(self, prompts):
        return prompts


class _ModelClass:
    captured_params = None
    supports_explicit_language_detection = False

    @classmethod
    def validate_language(cls, language):
        return language

    @classmethod
    def get_generation_prompt(cls, stt_params):
        cls.captured_params = stt_params
        return {
            "prompt_token_ids": [99],
            "multi_modal_data": {"audio": stt_params.audio},
        }


def _make_engine(arch_name: str = "WhisperForConditionalGeneration"):
    engine = asr_engine.ASREngine.__new__(asr_engine.ASREngine)
    engine.model_cls = _ModelClass
    engine.arch_name = arch_name
    engine.task_type = "transcribe"
    engine.max_audio_filesize_mb = 25
    engine.asr_config = SimpleNamespace(
        sample_rate=16000,
        allow_audio_chunking=False,
        max_audio_clip_s=None,
        min_energy_split_window_size=None,
        overlap_chunk_second=0,
    )
    engine.model_config = SimpleNamespace()
    engine.engine = SimpleNamespace(
        model_config=SimpleNamespace(),
        renderer=_Renderer(),
    )
    engine.tokenizer = SimpleNamespace(encode=lambda text: [1, 2, 3])
    return engine


def _make_request(*, prompt="", hotwords=None):
    return SimpleNamespace(
        language=None,
        to_language=None,
        prompt=prompt,
        hotwords=hotwords,
        response_format="json",
    )


def test_preprocess_forwards_prompt_and_hotwords(monkeypatch):
    monkeypatch.setattr(
        asr_engine,
        "load_audio",
        lambda _buffer, sr: (np.zeros(sr, dtype=np.float32), sr),
    )
    monkeypatch.setattr(asr_engine, "get_audio_duration", lambda **_kwargs: 1.0)
    monkeypatch.setattr(
        asr_engine,
        "parse_model_prompt",
        lambda _model_config, prompt: prompt,
    )

    engine = _make_engine()
    asyncio.run(
        engine._preprocess_speech_to_text(
            _make_request(prompt="existing context", hotwords=["HLIE", "后摩"]),
            b"wav-data",
            "test-request",
        )
    )

    params = _ModelClass.captured_params
    assert params.request_prompt == "existing context"
    assert params.hotwords == "HLIE, 后摩"


def test_qwen3_asr_prepends_context_prompt(monkeypatch):
    monkeypatch.setattr(
        asr_engine,
        "load_audio",
        lambda _buffer, sr: (np.zeros(sr, dtype=np.float32), sr),
    )
    monkeypatch.setattr(asr_engine, "get_audio_duration", lambda **_kwargs: 1.0)
    monkeypatch.setattr(
        asr_engine,
        "parse_model_prompt",
        lambda _model_config, prompt: prompt,
    )

    encoded_texts = []
    engine = _make_engine("Qwen3ASRForConditionalGeneration")
    engine.tokenizer = SimpleNamespace(
        encode=lambda text: encoded_texts.append(text) or [1, 2, 3]
    )
    prompts, _ = asyncio.run(
        engine._preprocess_speech_to_text(
            _make_request(prompt="会议主题是芯片", hotwords="HLIE, 后摩智能"),
            b"wav-data",
            "test-request",
        )
    )

    assert encoded_texts == [
        "<|im_start|>system\n会议主题是芯片\n可能出现的关键词：HLIE, 后摩智能<|im_end|>\n"
    ]
    assert prompts[0]["prompt_token_ids"] == [1, 2, 3, 99]


def test_normalize_hotwords_rejects_invalid_value():
    with pytest.raises(ValueError, match="hotwords"):
        asr_engine.normalize_hotwords({"word": "HLIE"})
