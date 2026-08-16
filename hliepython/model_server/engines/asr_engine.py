import io
import os
import time
import asyncio
import math
import re
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from fastapi import Request
from hliepython.utils.logging import async_logger as logger

from typing import Literal, TypeAlias, TypeVar, cast, Callable
from collections.abc import AsyncGenerator, Set

from hliepython.model_server.engines.base_engine import InferenceEngine
from hliepython.model_server.engines.utils import (
    make_vllm_frontend_namespace,
    merge_additional_config,
    move_vllm_engine_args_from_additional_config,
    split_vllm_config,
)
from hliepython.model_server.utils.gguf_metadata import apply_gguf_metadata_defaults_to_vllm_engine_args

from vllm.entrypoints.utils import get_max_tokens
from vllm.multimodal.audio import get_audio_duration, split_audio
from vllm.model_executor.model_loader.utils import get_model_architecture 
from vllm.multimodal.media.audio import load_audio
from vllm.renderers.inputs import DictPrompt, EncoderDecoderDictPrompt
# from vllm.inputs import EncoderDecoderInputs, ProcessorInputs
from vllm.inputs import EncoderDecoderInput
from vllm.multimodal.processing import ProcessorInputs
from vllm.logprobs import FlatLogprobs, Logprob
from vllm import AsyncLLMEngine, SamplingParams
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.outputs import RequestOutput
from vllm.entrypoints.openai.engine.serving import OpenAIServing, SpeechToTextRequest
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse,
    RequestResponseMetadata,
)
from vllm.config.speech_to_text import SpeechToTextParams
from vllm.exceptions import VLLMValidationError
from vllm.entrypoints.speech_to_text.transcription.protocol import (  
    TranscriptionRequest,  
    TranscriptionResponse,  
    TranscriptionResponseVerbose,  
)
from vllm.entrypoints.openai.speech_to_text.protocol import (
    TranscriptionResponseStreamChoice,
    TranscriptionSegment,
    TranscriptionStreamResponse,
    TranslationResponse,
    TranslationResponseStreamChoice,
    TranslationResponseVerbose,
    TranslationSegment,
    TranslationStreamResponse,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ErrorResponse,
    FunctionCall,
    PromptTokenUsageInfo,
    RequestResponseMetadata,
    ToolCall,
    UsageInfo,
)
from vllm.renderers.inputs.preprocess import parse_enc_dec_prompt, parse_model_prompt
from vllm.sampling_params import RequestOutputKind

from vllm_houmo.models.export_meta_from_gguf import parse_gguf

SpeechToTextResponse: TypeAlias = TranscriptionResponse | TranslationResponse
SpeechToTextResponseVerbose: TypeAlias = (
    TranscriptionResponseVerbose | TranslationResponseVerbose
)
SpeechToTextSegment: TypeAlias = TranscriptionSegment | TranslationSegment
T = TypeVar("T", bound=SpeechToTextResponse)
V = TypeVar("V", bound=SpeechToTextResponseVerbose)
S = TypeVar("S", bound=SpeechToTextSegment)


@dataclass(frozen=True)
class RealtimeTranscriptionResult:
    delta: str = ""
    logprobs: list[dict[str, Any]] | None = None
    usage: dict[str, Any] | None = None

ResponseType: TypeAlias = (
    TranscriptionResponse
    | TranslationResponse
    | TranscriptionResponseVerbose
    | TranslationResponseVerbose
)

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# As per https://platform.openai.com/docs/guides/speech-to-text#overview.
# TODO configurable
VLLM_MAX_AUDIO_CLIP_FILESIZE_MB = 25
MAX_AUDIO_CLIP_SECONDS = 20
OVERLAP_CHUNK_SECOND = 1
MIN_ENERGY_WINDOW_SIZE = 1600  # 1600 ~ 100ms for 16000 Hz audio

def asr_inter_chunk_separator(
    language: str | None, no_space_languages: Set[str]
) -> str:
    """Space to insert between ASR text chunks for streaming and non-streaming join.

    Languages in ``no_space_languages`` (e.g. Chinese, Japanese) use an empty
    separator; others use a single ASCII space.
    """
    return "" if language and language.lower() in no_space_languages else " "

class ASREngine(InferenceEngine):
    """ASR 推理引擎（对齐 OpenAI /v1/audio/transcriptions）。"""

    def __init__(self, instance_config: Dict[str, Any]):
        self.vllm_config = instance_config.get(
            "vllm_config",
            instance_config.get("vllm_extra_config", {}),
        )

        self.task_type = "transcribe"
        self.enable_force_include_usage = False
        self.model_sr = 16000
        self.hop_length = 160
        self.max_audio_clip_s = MAX_AUDIO_CLIP_SECONDS
        self.max_audio_filesize_mb = VLLM_MAX_AUDIO_CLIP_FILESIZE_MB
        self.vllm_frontend_args = None
        self.vllm_api_server_args: dict[str, Any] = {}
        self.vllm_unknown_args: dict[str, Any] = {}

        super().__init__(instance_config)

    def _load_model(self):
        if AsyncLLMEngine is None:
            raise RuntimeError("vLLM not available")

        try:
            (
                vllm_engine_args,
                vllm_frontend_args,
                vllm_api_server_args,
                vllm_unknown_args,
            ) = split_vllm_config(self.vllm_config)
            self.vllm_api_server_args = vllm_api_server_args
            self.vllm_unknown_args = vllm_unknown_args
            self.vllm_frontend_args = make_vllm_frontend_namespace(
                vllm_frontend_args,
                vllm_engine_args,
            )
            if vllm_frontend_args:
                logger.info(
                    "vLLM serving/frontend 层参数已接收；ASR 当前仍复用项目内 speech-to-text 流程: "
                    f"{vllm_frontend_args}"
                )
            if vllm_api_server_args:
                logger.info(
                    "vLLM API server 层参数已接收；当前 worker 已是 HTTP server，"
                    f"不会再启动第二个 vLLM API server: {vllm_api_server_args}"
                )
            if vllm_unknown_args:
                logger.warning(f"vLLM 未识别参数不会传入 AsyncEngineArgs: {vllm_unknown_args}")

            additional_config = dict(self.additional_config)
            vit_model_name = additional_config.pop("vit_model_name", None)
            if vit_model_name:
                visual_model_path = (
                    vit_model_name
                    if os.path.isabs(vit_model_name)
                    else os.path.join(self.model_path, vit_model_name)
                )
                additional_config["visual_model_path"] = visual_model_path

            additional_config.setdefault("device_ids", self.device_ids)
            user_additional_config = vllm_engine_args.pop("additional_config", None)
            additional_config = merge_additional_config(
                additional_config,
                user_additional_config,
            )
            vllm_engine_args, additional_config = move_vllm_engine_args_from_additional_config(
                vllm_engine_args,
                additional_config,
            )
            vllm_engine_args = apply_gguf_metadata_defaults_to_vllm_engine_args(
                self.model_path,
                vllm_engine_args,
            )

            parse_gguf(self.model_path)
            
            # 配置 vLLM 引擎参数
            engine_arg_kwargs = {
                **vllm_engine_args,
                "model": self.model_path,
                "additional_config": additional_config,
            }
            logger.info(f"vllm 启动参数: {engine_arg_kwargs}")
            logger.info(f"vllm serving/frontend 参数: {vllm_frontend_args}")

            engine_args = AsyncEngineArgs(**engine_arg_kwargs)
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            self.engine_client = self.engine

            self.model_cls, self.arch_name = get_model_architecture(self.engine.model_config)
            self.asr_config = self.model_cls.get_speech_to_text_config(
                self.engine.model_config, self.task_type
            )
            self.tokenizer = self.engine.get_tokenizer()
            self.model_config = self.engine.model_config
            self.default_sampling_params = self.model_config.get_diff_sampling_param()

            logger.info(f"vllm 实际启动参数：{self.engine.vllm_config}")
            logger.info(f"asr 模型配置参数：{self.asr_config}")
            logger.info(f"模型配置：{self.engine.vllm_config.model_config}")
            logger.info(f"并行配置：{self.engine.vllm_config.parallel_config}")
            logger.info(f"vLLM 引擎加载完成 {self.model_path} 到设备 {self.device_ids}")

        except Exception as e:
            logger.error(f"ASREngine: vLLM 引擎加载失败 {e}")
            raise RuntimeError(f"Failed to load vLLM engine: {e}")


    @staticmethod
    def _post_process_asr_text(text: str | None) -> str:
        if not text:
            return ""
        if "<asr_text>" in text:
            text = text.rsplit("<asr_text>", 1)[1]

        # 一些 Qwen3-ASR 输出会在开头带控制前缀，如 "language English"
        # 这里仅在文本开头位置清理，避免误伤正文。
        text = re.sub(r"^language\s+[A-Za-z_\-]+\s*", "", text)
        return text

    async def _speech_to_text_stream_generator(
        self,
        request: SpeechToTextRequest,
        list_result_generator: list[AsyncGenerator[RequestOutput, None]],
        request_id: str,
        request_metadata: RequestResponseMetadata,
        audio_duration_s: float,
        chunk_object_type: Literal["translation.chunk", "transcription.chunk"],
        response_stream_choice_class: type[TranscriptionResponseStreamChoice]
        | type[TranslationResponseStreamChoice],
        stream_response_class: type[TranscriptionStreamResponse]
        | type[TranslationStreamResponse],
    ) -> AsyncGenerator[str, None]:
        created_time = int(time.time())
        model_name = request.model

        completion_tokens = 0
        num_prompt_tokens = 0

        include_usage = request.stream_include_usage \
            if request.stream_include_usage else False
        include_continuous_usage = (
            request.stream_continuous_usage_stats
            if include_usage and request.stream_continuous_usage_stats
            else False
        )

        try:
            for result_generator in list_result_generator:
                async for res in result_generator:
                    # On first result.
                    if res.prompt_token_ids is not None:
                        num_prompt_tokens = len(res.prompt_token_ids)
                        if audio_tokens := self.model_cls.get_num_audio_tokens(
                            audio_duration_s, self.asr_config, self.model_config
                        ):
                            num_prompt_tokens += audio_tokens

                    # We need to do it here, because if there are exceptions in
                    # the result_generator, it needs to be sent as the FIRST
                    # response (by the try...catch).

                    # Just one output (n=1) supported.
                    assert len(res.outputs) == 1
                    output = res.outputs[0]

                    # TODO: For models that output structured formats (e.g.,
                    # Qwen3-ASR with "language X<asr_text>" prefix), streaming
                    # would need buffering to strip the prefix properly since
                    # deltas may split the tag across chunks.
                    delta_message = DeltaMessage(content=output.text)
                    completion_tokens += len(output.token_ids)

                    if output.finish_reason is None:
                        # Still generating, send delta update.
                        choice_data = response_stream_choice_class(delta=delta_message)
                    else:
                        # Model is finished generating.
                        choice_data = response_stream_choice_class(
                            delta=delta_message,
                            finish_reason=output.finish_reason,
                            stop_reason=output.stop_reason,
                        )

                    chunk = stream_response_class(
                        id=request_id,
                        object=chunk_object_type,
                        created=created_time,
                        choices=[choice_data],
                        model=model_name,
                    )

                    # handle usage stats if requested & if continuous
                    if include_continuous_usage:
                        chunk.usage = UsageInfo(
                            prompt_tokens=num_prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=num_prompt_tokens + completion_tokens,
                        )

                    data = chunk.model_dump_json(exclude_unset=True)
                    yield f"data: {data}\n\n"

            # Once the final token is handled, if stream_options.include_usage
            # is sent, send the usage.
            if include_usage:
                final_usage = UsageInfo(
                    prompt_tokens=num_prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=num_prompt_tokens + completion_tokens,
                )

                final_usage_chunk = stream_response_class(
                    id=request_id,
                    object=chunk_object_type,
                    created=created_time,
                    choices=[],
                    model=model_name,
                    usage=final_usage,
                )
                final_usage_data = final_usage_chunk.model_dump_json(
                    exclude_unset=True, exclude_none=True
                )
                yield f"data: {final_usage_data}\n\n"

            # report to FastAPI middleware aggregate usage across all choices
            request_metadata.final_usage_info = UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=num_prompt_tokens + completion_tokens,
            )

        except Exception as e:
            logger.exception("Error in %s stream generator.", self.task_type)
            data = self.create_streaming_error_response(e)
            yield f"data: {data}\n\n"
        # Send the final done message after all response.n are finished
        yield "data: [DONE]\n\n"

    async def _detect_language(
        self,
        audio_chunk: np.ndarray,
        request_id: str,
    ) -> str:
        """Auto-detect the spoken language from an audio chunk.

        Delegates prompt construction and output parsing to the model class
        via ``get_language_detection_prompt`` and
        ``parse_language_detection_output``.
        """
        prompt = self.model_cls.get_language_detection_prompt(
            audio_chunk,
            self.asr_config,
        )
        allowed_token_ids = self.model_cls.get_language_token_ids(
            self.tokenizer,
        )
        sampling_params = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            allowed_token_ids=allowed_token_ids,
        )

        result_generator = self.engine_client.generate(
            prompt,
            sampling_params,
            request_id,
        )

        final_output: RequestOutput
        async for final_output in result_generator:
            if final_output.finished:
                break

        token_ids = list(final_output.outputs[0].token_ids)
        lang = self.model_cls.parse_language_detection_output(
            token_ids,
            self.tokenizer,
        )

        logger.info("Auto-detected language: '%s'", lang)
        return lang

    async def _preprocess_speech_to_text(
        self,
        request: SpeechToTextRequest,
        audio_data: bytes,
        request_id: str,
    ) -> tuple[list[ProcessorInputs], float]:
        # Validate request
        language = self.model_cls.validate_language(request.language)
        # Skip to_language validation to avoid extra logging for Whisper.
        request.to_language = (
            self.model_cls.validate_language(request.to_language)
            if request.to_language
            else None
        )

        if len(audio_data) / 1024**2 > self.max_audio_filesize_mb:
            raise VLLMValidationError(
                "Maximum file size exceeded",
                parameter="audio_filesize_mb",
                value=len(audio_data) / 1024**2,
            )

        # Decode audio bytes.  For container formats (MP4, M4A, WebM) that
        # soundfile cannot detect from a BytesIO stream, _load_audio_bytes
        # transparently falls back to ffmpeg via an in-memory fd.
        # NOTE resample to model SR here for efficiency. This is also a
        # pre-requisite for chunking, as it assumes Whisper SR.
        try:
            with io.BytesIO(audio_data) as buf:
                y, sr = load_audio(buf, sr=self.asr_config.sample_rate)
        except Exception as exc:
            raise ValueError("Invalid or unsupported audio file.") from exc

        duration = get_audio_duration(y=y, sr=sr)
        do_split_audio = self.asr_config.allow_audio_chunking and (
            self.asr_config.max_audio_clip_s is not None
            and duration > self.asr_config.max_audio_clip_s
        )

        if not do_split_audio:
            chunks = [y]
        else:
            assert self.asr_config.max_audio_clip_s is not None
            assert self.asr_config.min_energy_split_window_size is not None
            chunks = split_audio(
                audio_data=y,
                sample_rate=int(sr),
                max_clip_duration_s=self.asr_config.max_audio_clip_s,
                overlap_duration_s=self.asr_config.overlap_chunk_second,
                min_energy_window_size=self.asr_config.min_energy_split_window_size,
            )
        logger.info(
            f"音频被分段成 {len(chunks)} 段，原始时长 {duration:.2f} 秒，"
            f"分段时长 {[get_audio_duration(y=chunk, sr=sr) for chunk in chunks]} 秒"
        )

        if request.language is None and getattr(
            self.model_cls, "supports_explicit_language_detection", False
        ):
            # Auto-detect language from the first chunk.
            request.language = await self._detect_language(
                chunks[0], f"{request_id}-lang_detect"
            )

        parsed_prompts: list[DictPrompt] = []
        for chunk in chunks:
            # stt_params = request.build_stt_params(
            #     audio=chunk,
            #     stt_config=self.asr_config,
            #     model_config=self.model_config,
            #     task_type=self.task_type,
            # )
            stt_params = SpeechToTextParams(
                audio=chunk,
                stt_config=self.asr_config,
                model_config=self.model_config,
                language=language,
                task_type=self.task_type,
                # request_prompt=self.prompt,
                to_language=request.to_language,
                # hotwords=self.hotwords,
            )

            prompt = self.model_cls.get_generation_prompt(stt_params)

            parsed_prompt: DictPrompt
            if request.response_format == "verbose_json":
                parsed_prompt = parse_enc_dec_prompt(prompt)
                parsed_prompt = self._preprocess_verbose_prompt(parsed_prompt)
            else:
                parsed_prompt = parse_model_prompt(self.engine.model_config, prompt)

            parsed_prompts.append(parsed_prompt)

        engine_prompts = await self.engine.renderer.render_cmpl_async(parsed_prompts)
        return engine_prompts, duration

    def _preprocess_verbose_prompt(self, prompt: EncoderDecoderDictPrompt):
        dec_prompt = prompt["decoder_prompt"]

        if not (isinstance(dec_prompt, dict) and "prompt" in dec_prompt):
            raise VLLMValidationError(
                "Expected decoder_prompt to contain text",
                parameter="decoder_prompt",
                value=type(dec_prompt).__name__,
            )

        dec_prompt["prompt"] = dec_prompt["prompt"].replace(
            "<|notimestamps|>", "<|0.00|>"
        )

        return prompt

    def create_sampling_params(
        self,
        request: SpeechToTextRequest
    ) -> SamplingParams:
        stream = getattr(request, "stream", False)
        n = getattr(request, "n", 1)
        seed = getattr(request, "seed", None)
        max_completion_tokens = getattr(request, "max_completion_tokens", None)

        output_kind = RequestOutputKind.CUMULATIVE \
            if stream else RequestOutputKind.FINAL_ONLY

        # 如果request.xxx 为None，那么就使用SamplingParams.from_optional的默认值
        return SamplingParams.from_optional(
            # 基础采样参数
            n=n,
            # temperature=request.temperature,
            temperature=0,
            # top_p=request.top_p,
            # top_k=request.top_k if request.top_k is not None else -1,
            # min_p=request.min_p if request.min_p is not None else 0.0,
            seed=seed,

            # 惩罚参数
            # presence_penalty=request.presence_penalty,
            # frequency_penalty=request.frequency_penalty,
            # repetition_penalty=request.repetition_penalty,

            # 停止条件
            max_tokens=max_completion_tokens,

            # 输出类型和额外参数
            output_kind=output_kind,
        )

    @staticmethod
    def _get_decoder_prompt_len(engine_prompts: list[ProcessorInputs]) -> int:
        """Get the length of the decoder prompt. Currently we need to offset
        by the decoder prompt length when running beam search because the mm
        encoder is not currently cached and runs on decode calls; because of
        this, we need to make sure the redundant encoder calls won't exceed
        the context :(

        FIXME (Alex) - this will be removed in the very near future once the
        encoder/decoder caching is implemented.
        """
        input_len = 0
        assert len(engine_prompts) > 0
        first_eng_prompt = engine_prompts[0]

        if first_eng_prompt.get("type") == "enc_dec":
            first_eng_prompt = cast(EncoderDecoderInput, first_eng_prompt)
            input_len = len(first_eng_prompt["decoder_prompt"]["prompt_token_ids"])
        return input_len

    async def _create_speech_to_text(
        self,
        audio_data: bytes,
        request: SpeechToTextRequest,
        raw_request: Request,
        response_class: type[ResponseType],
        stream_generator_method: Callable[..., AsyncGenerator[str, None]],
    ) -> T | V | AsyncGenerator[str, None] | ErrorResponse:
        """Base method for speech-to-text operations like transcription and
        translation."""
        if request.stream and request.use_beam_search:
            return self.create_error_response(
                "Streaming is not currently supported with beam search"
            )

        # error_check_ret = await self._check_model(request)
        # if error_check_ret is not None:
        #     return error_check_ret

        # If the engine is dead, raise the engine's DEAD_ERROR.
        # This is required for the streaming case, where we return a
        # success status before we actually start generating text :).
        if self.engine.errored:
            raise self.engine.dead_error

        if request.response_format not in ["text", "json", "verbose_json"]:
            return self.create_error_response(
                "Currently only support response_format: "
                "`text`, `json` or `verbose_json`"
            )

        if (
            request.response_format == "verbose_json"
            and not self.model_cls.supports_segment_timestamp
        ):
            return self.create_error_response(
                f"Currently do not support verbose_json for {request.model}"
            )

        if request.response_format == "verbose_json" and request.stream:
            return self.create_error_response(
                "verbose_json format doesn't support streaming case"
            )
        request_id = f"{self.task_type}-{self._base_request_id(raw_request)}"

        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        # lora_request = self._maybe_get_adapters(request)

        engine_prompts, duration_s = await self._preprocess_speech_to_text(
            request=request,
            audio_data=audio_data,
            request_id=request_id,
        )

        # Schedule the request and get the result generator.
        max_model_len = self.model_config.max_model_len
        list_result_generator: list[AsyncGenerator[RequestOutput, None]] | None = None

        input_len = (
            ASREngine._get_decoder_prompt_len(engine_prompts)
            if request.use_beam_search
            else 0
        )

        # Unlike most decoder-only models, whisper generation length is not
        # constrained by the size of the input audio, which is mapped to a
        # fixed-size log-mel-spectogram. Still, allow for fewer tokens to be
        # generated by respecting the extra completion tokens arg.
        max_tokens = get_max_tokens(
            max_model_len,
            request.max_completion_tokens,
            input_len,
            self.default_sampling_params,
        )

        if request.use_beam_search:
            sampling_params = request.to_beam_search_params(
                max_tokens, self.default_sampling_params
            )
        else:
            sampling_params = request.to_sampling_params(
                max_tokens,
                self.default_sampling_params,
            )

        if request.response_format == "verbose_json":
            sampling_params.logprobs = 1

        list_result_generator = []
        for i, engine_prompt in enumerate(engine_prompts):
            request_id_item = f"{request_id}_{i}"

            # self._log_inputs(
            #     request_id_item,
            #     engine_prompt,
            #     params=sampling_params,
            #     lora_request=lora_request,
            # )

            # trace_headers = (
            #     None
            #     if raw_request is None
            #     else await self._get_trace_headers(raw_request.headers)
            # )

            # if isinstance(sampling_params, BeamSearchParams):
            if 0:
                generator = self.beam_search(
                    prompt=engine_prompt,
                    params=sampling_params,
                    request_id=request_id_item,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                )
            else:
                generator = self.engine.generate(
                    engine_prompt,
                    sampling_params,
                    request_id_item,
                    # lora_request=lora_request,
                    # trace_headers=trace_headers,
                )

            list_result_generator.append(generator)

        if request.stream:
            return stream_generator_method(
                request, list_result_generator, request_id, request_metadata, duration_s
            )

        # Non-streaming response.
        total_segments = []
        text_parts = []
        try:
            assert list_result_generator is not None
            segments_types: dict[str, type[SpeechToTextSegment]] = {
                "transcribe": TranscriptionSegment,
                "translate": TranslationSegment,
            }
            segment_class: type[SpeechToTextSegment] = segments_types[self.task_type]
            text = ""
            chunk_size_in_s = self.asr_config.max_audio_clip_s
            if chunk_size_in_s is None:
                assert len(list_result_generator) == 1, (
                    "`max_audio_clip_s` is set to None, audio cannot be chunked"
                )
            for idx, result_generator in enumerate(list_result_generator):
                start_time = (
                    float(idx * chunk_size_in_s) if chunk_size_in_s is not None else 0.0
                )
                async for op in result_generator:
                    if request.response_format == "verbose_json":
                        assert op.outputs[0].logprobs
                        segments: list[SpeechToTextSegment] = (
                            self._get_verbose_segments(
                                tokens=tuple(op.outputs[0].token_ids),
                                segment_class=segment_class,
                                request=request,
                                start_time=start_time,
                                log_probs=op.outputs[0].logprobs,
                            )
                        )

                        total_segments.extend(segments)
                        text_parts.extend([seg.text for seg in segments])
                    else:
                        raw_text = op.outputs[0].text
                        text_parts.append(self.model_cls.post_process_output(raw_text))

            text = "".join(text_parts)
            if self.task_type == "transcribe":
                final_response: ResponseType
                # add usage in TranscriptionResponse.
                usage = {
                    "type": "duration",
                    # rounded up as per openAI specs
                    "seconds": int(math.ceil(duration_s)),
                }
                if request.response_format != "verbose_json":
                    final_response = cast(
                        T, TranscriptionResponse(text=text, usage=usage)
                    )
                else:
                    final_response = cast(
                        V,
                        TranscriptionResponseVerbose(
                            text=text,
                            language=request.language,
                            duration=str(duration_s),
                            segments=total_segments,
                        ),
                    )
            else:
                # no usage in response for translation task
                if request.response_format != "verbose_json":
                    final_response = cast(T, TranslationResponse(text=text))
                else:
                    final_response = cast(
                        V,
                        TranslationResponseVerbose(
                            text=text,
                            language=request.language,
                            duration=str(duration_s),
                            segments=total_segments,
                        ),
                    )
            return final_response
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")

    def _get_verbose_segments(
        self,
        tokens: tuple,
        log_probs: FlatLogprobs | list[dict[int, Logprob]],
        request: SpeechToTextRequest,
        segment_class: type[SpeechToTextSegment],
        start_time: float = 0,
    ) -> list[SpeechToTextSegment]:
        """
        Convert tokens to verbose segments.

        This method expects the model to produce
        timestamps as tokens (similar to Whisper).
        If the tokens do not include timestamp information,
        the segments may not be generated correctly.

        Note: No_speech_prob field is not supported
        in this implementation and will be None. See docs for details.
        """
        BASE_OFFSET = 0.02
        init_token = self.tokenizer.encode("<|0.00|>", add_special_tokens=False)[0]
        if tokens[-1] == self.tokenizer.eos_token_id:
            tokens = tokens[:-1]

        tokens_with_start = (init_token,) + tokens
        segments: list[SpeechToTextSegment] = []
        last_timestamp_start = 0

        if tokens_with_start[-2] < init_token and tokens_with_start[-1] >= init_token:
            tokens_with_start = tokens_with_start + (tokens_with_start[-1],)
        avg_logprob = 0.0
        for idx in range(1, len(tokens_with_start)):
            # Timestamp tokens (e.g., <|0.00|>) are assumed to be sorted.
            # If the ordering is violated, this slicing may produce incorrect results.
            token = tokens_with_start[idx]
            if token >= init_token and tokens_with_start[idx - 1] >= init_token:
                sliced_timestamp_tokens = tokens_with_start[last_timestamp_start:idx]
                start_timestamp = sliced_timestamp_tokens[0] - init_token
                end_timestamp = sliced_timestamp_tokens[-1] - init_token
                text = self.tokenizer.decode(sliced_timestamp_tokens[1:-1])
                text_bytes = text.encode("utf-8")

                casting_segment = cast(
                    SpeechToTextSegment,
                    segment_class(
                        id=len(segments),
                        seek=start_time,
                        start=start_time + BASE_OFFSET * start_timestamp,
                        end=start_time + BASE_OFFSET * end_timestamp,
                        temperature=request.temperature,
                        text=text,
                        # The compression ratio measures
                        # how compressible the generated text is.
                        # A higher ratio indicates more repetitive content,
                        # which is a strong sign of hallucination in outputs.
                        compression_ratio=len(text_bytes)
                        / len(zlib.compress(text_bytes)),
                        tokens=sliced_timestamp_tokens[1:-1],
                        avg_logprob=avg_logprob / (idx - last_timestamp_start),
                    ),
                )
                segments.append(casting_segment)
                last_timestamp_start = idx
                avg_logprob = 0
            else:
                avg_logprob += log_probs[idx - 1][token].logprob
        return segments

    async def create_audio_transcriptions(
        self,
        request: TranscriptionRequest,
        raw_request: Request | None = None,
    ) -> (
        TranscriptionResponse
        | TranscriptionResponseVerbose
        | AsyncGenerator[str, None]
        | ErrorResponse
    ):
        """Transcription API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/audio/createTranscription
        for the API specification. This API mimics the OpenAI transcription API.
        """
        audio_data = await request.file.read()

        return await self._create_speech_to_text(
            audio_data=audio_data,
            request=request,
            raw_request=raw_request,
            response_class=(
                TranscriptionResponseVerbose
                if request.response_format == "verbose_json"
                else TranscriptionResponse
            ),
            stream_generator_method=self.transcription_stream_generator,
        )

    async def transcription_stream_generator(
        self,
        request: TranscriptionRequest,
        result_generator: list[AsyncGenerator[RequestOutput, None]],
        request_id: str,
        request_metadata: RequestResponseMetadata,
        audio_duration_s: float,
    ) -> AsyncGenerator[str, None]:
        generator = self._speech_to_text_stream_generator(
            request=request,
            list_result_generator=result_generator,
            request_id=request_id,
            request_metadata=request_metadata,
            audio_duration_s=audio_duration_s,
            chunk_object_type="transcription.chunk",
            response_stream_choice_class=TranscriptionResponseStreamChoice,
            stream_response_class=TranscriptionStreamResponse,
        )
        async for chunk in generator:
            yield chunk

    async def websocket_transcribe_bytes(
        self,
        audio_bytes: bytes,
        language: str | None = None,
        prompt: str = "",
    ) -> str:
        """WebSocket 场景下的单段音频转写，兼容需要完整文本的调用方。"""
        text_parts: list[str] = []
        async for delta in self.websocket_transcribe_stream(
            audio_bytes,
            language=language,
            prompt=prompt,
        ):
            text_parts.append(delta)
        return "".join(text_parts)

    async def websocket_transcribe_stream(
        self,
        audio_bytes: bytes,
        language: str | None = None,
        prompt: str = "",
    ) -> AsyncGenerator[str, None]:
        """按 vLLM 累计生成结果产出 OpenAI Realtime 风格文本增量。"""
        async for result in self.websocket_transcribe_stream_with_usage(
            audio_bytes,
            language=language,
            prompt=prompt,
        ):
            if result.delta:
                yield result.delta

    async def websocket_transcribe_stream_with_usage(
        self,
        audio_bytes: bytes,
        language: str | None = None,
        prompt: str = "",
        include_logprobs: bool = False,
    ) -> AsyncGenerator[RealtimeTranscriptionResult, None]:
        """产出文本增量、可选 token logprobs 和最终 usage。"""
        if not audio_bytes:
            return

        # 用轻量对象承载 _preprocess/create_sampling_params 所需字段。
        class _WsRequest:
            def __init__(self, lang: str | None, prmpt: str):
                self.language = lang
                self.to_language = None
                self.prompt = prmpt
                self.response_format = "json"
                # CUMULATIVE 使 vLLM 在生成期间持续返回当前完整候选文本。
                self.stream = True
                self.n = 1
                self.seed = None
                self.temperature = 0.0
                self.top_p = 1.0
                self.min_p = 0.0
                self.presence_penalty = 0.0
                self.frequency_penalty = 0.0
                self.max_completion_tokens = None

        ws_request = _WsRequest(language, prompt)
        ws_request_id = f"ws-asr-pre-{self._base_request_id(None)}"
        prompts, audio_duration_s = await self._preprocess_speech_to_text(
            ws_request,
            audio_bytes,
            request_id=ws_request_id,
        )
        sampling_params = self.create_sampling_params(ws_request)
        if include_logprobs:
            # vLLM 中 0 表示仅返回实际选中 token 的 log probability，
            # 不额外返回 top candidates，符合 Realtime transcription schema。
            sampling_params.logprobs = 0

        text_input_tokens = 0
        output_tokens = 0
        completed_logprobs: list[dict[str, Any]] = []

        for idx, prompt_item in enumerate(prompts):
            request_id = f"ws-asr-{self._base_request_id(None)}-{idx}"
            result_generator = self.engine.generate(prompt_item, sampling_params, request_id)

            previous_text = ""
            previous_visible_logprob_count = 0
            segment_input_tokens = 0
            segment_output_tokens = 0
            is_qwen_asr = "qwen" in str(self.arch_name).lower()
            async for out in result_generator:
                if not out.outputs:
                    continue
                output = out.outputs[0]
                raw_text = output.text or ""
                if out.prompt_token_ids is not None:
                    segment_input_tokens = len(out.prompt_token_ids)
                segment_output_tokens = len(output.token_ids or [])

                # Qwen3-ASR 的控制前缀可能跨多个生成结果拆分。在 <asr_text>
                # 完整出现前不向客户端泄漏前缀；若模型最终没有输出该标签，
                # 则在结束时回退到通用后处理。
                if is_qwen_asr and "<asr_text>" not in raw_text:
                    if output.finish_reason is None:
                        continue
                    current_text = self._post_process_asr_text(raw_text)
                else:
                    current_text = self._post_process_asr_text(raw_text)

                visible_logprobs: list[dict[str, Any]] = []
                if include_logprobs:
                    visible_logprobs = self._get_realtime_transcription_logprobs(
                        raw_text=raw_text,
                        visible_text=current_text,
                        token_ids=output.token_ids or [],
                        token_logprobs=output.logprobs,
                    )
                    if visible_logprobs[:previous_visible_logprob_count] != (
                        completed_logprobs[-previous_visible_logprob_count:]
                        if previous_visible_logprob_count
                        else []
                    ):
                        raise RuntimeError(
                            "ASR 流式累计 logprobs 发生前文改写，无法转换为追加式数据"
                        )

                if current_text.startswith(previous_text):
                    delta = current_text[len(previous_text):]
                else:
                    # OpenAI transcription.delta 是仅追加协议，无法表达撤回或
                    # 替换已经发送的内容，因此拒绝非单调累计输出。
                    raise RuntimeError(
                        "ASR 流式累计输出发生前文改写，无法转换为追加式 delta"
                    )

                previous_text = current_text
                if delta:
                    delta_logprobs = (
                        visible_logprobs[previous_visible_logprob_count:]
                        if include_logprobs
                        else None
                    )
                    if delta_logprobs:
                        completed_logprobs.extend(delta_logprobs)
                    previous_visible_logprob_count = len(visible_logprobs)
                    yield RealtimeTranscriptionResult(
                        delta=delta,
                        logprobs=delta_logprobs,
                    )

            text_input_tokens += segment_input_tokens
            output_tokens += segment_output_tokens

        audio_tokens = self.model_cls.get_num_audio_tokens(
            audio_duration_s,
            self.asr_config,
            self.model_config,
        ) or 0
        input_tokens = text_input_tokens + audio_tokens
        yield RealtimeTranscriptionResult(
            logprobs=completed_logprobs if include_logprobs else None,
            usage={
                "type": "tokens",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "input_token_details": {
                    "audio_tokens": audio_tokens,
                    "text_tokens": text_input_tokens,
                },
            }
        )

    def _get_realtime_transcription_logprobs(
        self,
        raw_text: str,
        visible_text: str,
        token_ids: list[int],
        token_logprobs: FlatLogprobs | list[dict[int, Logprob]] | None,
    ) -> list[dict[str, Any]]:
        """将 vLLM 累计 token logprobs 转换为可见转录文本的标准结构。"""
        if token_logprobs is None:
            raise RuntimeError("ASR 后端未返回请求的 logprobs")

        records: list[dict[str, Any]] = []
        for index, token_id in enumerate(token_ids):
            if index >= len(token_logprobs):
                break
            candidates = token_logprobs[index]
            selected = candidates.get(token_id) if candidates is not None else None
            if selected is None:
                raise RuntimeError(f"ASR 后端缺少 token {token_id} 的 logprob")
            token = selected.decoded_token
            if token is None:
                token = self.tokenizer.decode([token_id])
            records.append({
                "token": token,
                "bytes": list(token.encode("utf-8", errors="replace")),
                "logprob": max(selected.logprob, -9999.0),
            })

        decoded_text = "".join(record["token"] for record in records)
        alignment_text = decoded_text if decoded_text.endswith(visible_text) else raw_text
        if not alignment_text.endswith(visible_text):
            raise RuntimeError("ASR logprobs 无法与转录文本对齐")
        hidden_prefix_len = len(alignment_text) - len(visible_text)

        visible_records: list[dict[str, Any]] = []
        offset = 0
        for record in records:
            token = record["token"]
            token_end = offset + len(token)
            if token_end <= hidden_prefix_len:
                offset = token_end
                continue
            if offset < hidden_prefix_len:
                token = token[hidden_prefix_len - offset:]
            remaining_visible_chars = len(visible_text) - sum(
                len(item["token"]) for item in visible_records
            )
            token = token[:remaining_visible_chars]
            if token:
                visible_records.append({
                    **record,
                    "token": token,
                    "bytes": list(token.encode("utf-8", errors="replace")),
                })
            offset = token_end
            if sum(len(item["token"]) for item in visible_records) >= len(visible_text):
                break
        return visible_records
