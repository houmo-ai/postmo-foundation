import io
import json
import time
import asyncio
import librosa
import math
import re
from http import HTTPStatus
from typing import Any, Dict

import numpy as np
from math import ceil
from fastapi import Request
from fastapi.responses import Response
from loguru import logger

from typing import Final, Literal, TypeAlias, TypeVar, cast
from typing import Callable, Literal, Optional, TypeVar, Union, cast
from collections.abc import AsyncGenerator

from hliepython.model_server.engines.base_engine import InferenceEngine

from collections.abc import Mapping, Sequence
from vllm.multimodal.audio import get_audio_duration, split_audio
from vllm.multimodal.media.audio import load_audio
from vllm.renderers.inputs import DictPrompt, EncoderDecoderDictPrompt
from vllm.logprobs import FlatLogprobs, Logprob
from vllm.inputs.data import PromptType
from vllm.entrypoints.utils import create_error_response
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.outputs import RequestOutput
from vllm.entrypoints.openai.engine.serving import OpenAIServing, SpeechToTextRequest
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse,
    RequestResponseMetadata,
)
from vllm.exceptions import VLLMValidationError
from vllm.entrypoints.openai.speech_to_text.protocol import (
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionResponseStreamChoice,
    TranscriptionResponseVerbose,
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
from vllm.inputs import (
    ExplicitEncoderDecoderPrompt,
    PromptType,
    TextPrompt,
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

ResponseType: TypeAlias = (
    TranscriptionResponse
    | TranslationResponse
    | TranscriptionResponseVerbose
    | TranslationResponseVerbose
)


# As per https://platform.openai.com/docs/guides/speech-to-text#overview.
# TODO configurable
VLLM_MAX_AUDIO_CLIP_FILESIZE_MB = 25
MAX_AUDIO_CLIP_SECONDS = 30
OVERLAP_CHUNK_SECOND = 1
MIN_ENERGY_WINDOW_SIZE = 1600  # 1600 ~ 100ms for 16000 Hz audio

class ASREngine(InferenceEngine):
    """ASR 推理引擎（对齐 OpenAI /v1/audio/transcriptions）。"""

    def __init__(self, instance_config: Dict[str, Any]):
        self.vllm_extra_config = instance_config.get("vllm_extra_config", {})
        super().__init__(instance_config)

        self.max_audio_clip_s = MAX_AUDIO_CLIP_SECONDS

        self.task_type = "transcribe"
        self.model_sr = 16000
        self.hop_length = 160
        
        self.max_audio_filesize_mb = VLLM_MAX_AUDIO_CLIP_FILESIZE_MB

    def _load_model(self):
        if AsyncLLMEngine is None:
            raise RuntimeError("vLLM not available")

        try:
            additional_config = {}
            additional_config["device_ids"] = self.device_ids

            parse_gguf(self.model_path)

            engine_args = AsyncEngineArgs(
                model=self.model_path,
                trust_remote_code=True,
                limit_mm_per_prompt={"audio": 1},
                additional_config=additional_config,
            )

            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            logger.info(f"ASREngine: vLLM 引擎加载完成 {self.model_path} 到设备 {self.device_ids}")

        except Exception as e:
            logger.error(f"ASREngine: vLLM 引擎加载失败 {e}")
            raise RuntimeError(f"Failed to load vLLM engine: {e}")

    @staticmethod
    def create_error_response(
        message: str | Exception,
        err_type: str = "BadRequestError",
        status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
        param: str | None = None,
    ) -> ErrorResponse:
        return create_error_response(message, err_type, status_code, param)

    def create_streaming_error_response(
            self,
            message: str,
            err_type: str = "BadRequestError",
            status_code: HTTPStatus = HTTPStatus.BAD_REQUEST) -> str:
        json_str = json.dumps({
            "error":
            self.create_error_response(message=message,
                                       err_type=err_type,
                                       status_code=status_code).model_dump()
        })
        return json_str

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
        response_stream_choice_class: Union[
            type[TranscriptionResponseStreamChoice],
            type[TranslationResponseStreamChoice]],
        stream_response_class: Union[type[TranscriptionStreamResponse],
                                     type[TranslationStreamResponse]],
    ) -> AsyncGenerator[str, None]:
        created_time = int(time.time())
        model_name = request.model

        completion_tokens = 0
        num_prompt_tokens = 0

        include_usage = request.stream_include_usage \
            if request.stream_include_usage else False
        include_continuous_usage = request.stream_continuous_usage_stats\
            if include_usage and request.stream_continuous_usage_stats\
            else False

        try:
            for result_generator in list_result_generator:
                previous_text = ""
                async for res in result_generator:
                    # On first result.
                    if res.prompt_token_ids is not None:
                        # Do not account the 4-tokens `<|startoftranscript|>..`
                        # Could be negative when language token
                        # is not specified.
                        num_prompt_tokens = max(
                            len(res.prompt_token_ids) - 4, 0)
                        # NOTE(NickLucche) user can't pass encoder
                        # prompts directly at least not to Whisper.
                        # One indicator of the encoder amount of processing
                        # is the log-mel spectogram length.
                        num_prompt_tokens += ceil(
                            audio_duration_s * self.model_sr / self.hop_length)

                    # We need to do it here, because if there are exceptions in
                    # the result_generator, it needs to be sent as the FIRST
                    # response (by the try...catch).

                    # Just one output (n=1) supported.
                    assert len(res.outputs) == 1
                    output = res.outputs[0]

                    processed_text = self._post_process_asr_text(output.text)
                    delta_text = processed_text[len(previous_text):]
                    previous_text = processed_text
                    delta_message = DeltaMessage(content=delta_text)
                    completion_tokens += len(output.token_ids)

                    if output.finish_reason is None:
                        # Still generating, send delta update.
                        choice_data = response_stream_choice_class(
                            delta=delta_message)
                    else:
                        # Model is finished generating.
                        choice_data = response_stream_choice_class(
                            delta=delta_message,
                            finish_reason=output.finish_reason,
                            stop_reason=output.stop_reason)

                    chunk = stream_response_class(id=request_id,
                                                  object=chunk_object_type,
                                                  created=created_time,
                                                  choices=[choice_data],
                                                  model=model_name)

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
                final_usage = UsageInfo(prompt_tokens=num_prompt_tokens,
                                        completion_tokens=completion_tokens,
                                        total_tokens=num_prompt_tokens +
                                        completion_tokens)

                final_usage_chunk = stream_response_class(
                    id=request_id,
                    object=chunk_object_type,
                    created=created_time,
                    choices=[],
                    model=model_name,
                    usage=final_usage)
                final_usage_data = (final_usage_chunk.model_dump_json(
                    exclude_unset=True, exclude_none=True))
                yield f"data: {final_usage_data}\n\n"

            # report to FastAPI middleware aggregate usage across all choices
            request_metadata.final_usage_info = UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=num_prompt_tokens + completion_tokens)

        except Exception as e:
            # TODO: Use a vllm-specific Validation Error
            logger.exception("Error in %s stream generator.", self.task_type)
            data = create_error_response(str(e), "internal_error", HTTPStatus.INTERNAL_SERVER_ERROR)
            yield f"data: {data}\n\n"
        # Send the final done message after all response.n are finished
        yield "data: [DONE]\n\n"

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

    def _find_split_point(self, wav: np.ndarray, start_idx: int,
                          end_idx: int) -> int:
        """Find the best point to split audio by 
        looking for silence or low amplitude.
        Args:
            wav: Audio tensor [1, T]
            start_idx: Start index of search region
            end_idx: End index of search region
        Returns:
            Index of best splitting point
        """
        segment = wav[start_idx:end_idx]

        # Calculate RMS energy in small windows
        min_energy = math.inf
        quietest_idx = 0
        for i in range(0,
                       len(segment) - MIN_ENERGY_WINDOW_SIZE,
                       MIN_ENERGY_WINDOW_SIZE):
            window = segment[i:i + MIN_ENERGY_WINDOW_SIZE]
            energy = (window**2).mean()**0.5
            if energy < min_energy:
                quietest_idx = i + start_idx
                min_energy = energy
        return quietest_idx

    def _split_audio(self, audio_data: np.ndarray,
                     sample_rate: int) -> list[np.ndarray]:
        chunk_size = sample_rate * self.max_audio_clip_s
        overlap_size = sample_rate * OVERLAP_CHUNK_SECOND
        chunks = []
        i = 0
        while i < audio_data.shape[-1]:
            if i + chunk_size >= audio_data.shape[-1]:
                # handle last chunk
                chunks.append(audio_data[..., i:])
                break

            # Find the best split point in the overlap region
            search_start = i + chunk_size - overlap_size
            search_end = min(i + chunk_size, audio_data.shape[-1])
            split_point = self._find_split_point(audio_data, search_start,
                                                 search_end)

            # Extract chunk up to the split point
            chunks.append(audio_data[..., i:split_point])
            i = split_point
        return chunks

    async def _preprocess_speech_to_text(
        self,
        request: SpeechToTextRequest,
        audio_data: bytes,
        request_id: str,
    ) -> tuple[list[PromptType], float]:
        # Validate request
        lang = request.language or "en"

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
                y, sr = load_audio(buf, sr=self.model_sr)
        except Exception as exc:
            raise ValueError("Invalid or unsupported audio file.") from exc

        duration = get_audio_duration(y=y, sr=sr)
        chunks = [y] if duration < self.max_audio_clip_s else self._split_audio(y, int(sr))

        parsed_prompts: list[DictPrompt] = []
        for chunk in chunks:
            
            decoder_text = (
                f"<|prev|>{request.prompt}" if request.prompt else ""
            ) + f"<|startoftranscript|><|{lang}|><|transcribe|><|notimestamps|>"

            prompt = ExplicitEncoderDecoderPrompt(
                encoder_prompt=TextPrompt(
                    prompt="",  # Whisper does not support encoder prompt.
                    multi_modal_data={"audio": (chunk, sr)},
                ),
                decoder_prompt=TextPrompt(prompt=decoder_text),
            )

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

        output_kind = RequestOutputKind.DELTA \
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

        if request.response_format not in ['text', 'json', 'verbose_json']:
            return self.create_error_response(
            "Currently only support response_format `text`, `json` or `verbose_json`")

        request_id = f"{self.task_type}-{self._base_request_id(raw_request)}"

        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        try:
            prompts, duration_s = await self._preprocess_speech_to_text(
                request=request,
                audio_data=audio_data,
                request_id=request_id,
            )

        except ValueError as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(str(e))

        list_result_generator: Optional[list[AsyncGenerator[RequestOutput,
                                                            None]]] = None
        try:  
            sampling_params = self.create_sampling_params(request)
            logger.info(f"sampling_params: {sampling_params}")

            logger.info(f" ---------------------------------------- ")
            logger.info(f"prompts: {prompts}")
            logger.info(f"sampling_params: {sampling_params}")
            logger.info(f"request_id: {request_id}")
            logger.info(f" ---------------------------------------- ")

            list_result_generator = [
                self.engine.generate(
                    prompt,
                    sampling_params,
                    request_id,
                ) for prompt in prompts
            ]
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        if request.stream:
            return stream_generator_method(
                request,
                list_result_generator,
                request_id,
                request_metadata,
                duration_s,
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
            # chunk_size_in_s = self.asr_config.max_audio_clip_s
            chunk_size_in_s = 30
            if chunk_size_in_s is None:
                assert len(list_result_generator) == 1, (
                    "`max_audio_clip_s` is set to None, audio cannot be chunked"
                )
            for idx, result_generator in enumerate(list_result_generator):
                start_time = (
                    float(idx * chunk_size_in_s) if chunk_size_in_s is not None else 0.0
                )
                async for op in result_generator:
                    logger.info(f"op.outputs: {op.outputs}")
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
                        text_parts.append(self.post_process_output(raw_text))
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
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))


    @classmethod
    def post_process_output(cls, text: str) -> str:
        """
        Post-process the raw model output text.

        Some ASR models output structured formats (e.g., language tags,
        special tokens) that need to be stripped before returning to the user.

        Args:
            text: Raw decoded text from the model.

        Returns:
            Cleaned transcription text.
        """
        return text

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

    async def websocket_transcribe_bytes(
        self,
        audio_bytes: bytes,
        language: str | None = None,
        prompt: str = "",
    ) -> str:
        """WebSocket 场景下的单段音频转写。"""
        if not audio_bytes:
            return ""

        # 用轻量对象承载 _preprocess/create_sampling_params 所需字段。
        class _WsRequest:
            def __init__(self, lang: str | None, prmpt: str):
                self.language = lang
                self.to_language = None
                self.prompt = prmpt
                self.response_format = "json"
                self.stream = False
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
        prompts, _ = await self._preprocess_speech_to_text(
            ws_request,
            audio_bytes,
            request_id=ws_request_id,
        )
        sampling_params = self.create_sampling_params(ws_request)

        merged_text = ""
        for idx, prompt_item in enumerate(prompts):
            request_id = f"ws-asr-{self._base_request_id(None)}-{idx}"
            result_generator = self.engine.generate(prompt_item, sampling_params, request_id)

            prev = ""
            chunk_text = ""
            async for out in result_generator:
                if not out.outputs:
                    continue
                processed = self._post_process_asr_text(out.outputs[0].text)
                delta = processed[len(prev):]
                prev = processed
                chunk_text += delta

            merged_text += chunk_text

        return merged_text