import os
import json
import jinja2
import asyncio
import uuid
import time
import hashlib
import inspect
from http import HTTPStatus
from hliepython.utils.logging import async_logger as logger
from collections.abc import AsyncGenerator
from collections.abc import Sequence as GenericSequence
from fastapi import Request
from fastapi.responses import StreamingResponse

from transformers import AutoTokenizer
from typing import Dict, Any

from qwen_vl_utils import process_vision_info

from hliepython.model_server.engines.utils import (
    make_vllm_frontend_namespace,
    merge_additional_config,
    move_vllm_engine_args_from_additional_config,
    split_vllm_config,
    should_include_usage,
)
from hliepython.model_server.engines.base_engine import InferenceEngine
from hliepython.model_server.engines.protocol.chat_completions import (
    ChatCompletionRequest,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseStreamChoice,
    DeltaMessage,
    ChatCompletionStreamResponse,
    UsageInfo,
    ChatCompletionResponseChoice,
    ChatMessage,
    ChatCompletionLogProb,
    ChatCompletionLogProbs,
    ChatCompletionLogProbsContent,
    Logprob
)
from hliepython.model_server.utils.gguf_metadata import (
    apply_gguf_metadata_defaults_to_vllm_engine_args,
    get_gguf_metadata,
)

from vllm.entrypoints.chat_utils import parse_chat_messages
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse,
)

from vllm.outputs import RequestOutput
from vllm.v1.engine.exceptions import EngineDeadError
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind
from vllm.entrypoints.chat_utils import load_chat_template
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest as VllmChatCompletionRequest,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses
from vllm.entrypoints.serve.render.serving import OpenAIServingRender

class TextEngine(InferenceEngine):
    """文本推理引擎"""

    def __init__(self, instance_config: Dict[str, Any]):
        
        self.model_name = instance_config.get("model_name", "")
        self.model_path = instance_config.get("model_path", "")
        self.model_type = instance_config.get("model_type", "llm")
        self.vllm_config = instance_config.get(
            "vllm_config",
            instance_config.get("vllm_extra_config", {}),
        )
        self.additional_config = instance_config.get(
            "additional_config", {}
        ) or {}

        self.response_role = "assistant"
        self.default_enable_think = False
        self.return_tokens_as_token_ids: bool = False
        self.enable_force_include_usage: bool = False
        self.vllm_frontend_args = None
        self.vllm_api_server_args: dict[str, Any] = {}
        self.vllm_unknown_args: dict[str, Any] = {}
        self.openai_serving_models = None
        self.openai_serving_render = None
        self.openai_serving_chat = None
        self.openai_serving_responses = None
        self.use_vllm_openai_serving = True
        self.chat_template = None
        self.chat_template_kwargs: dict[str, Any] = {}

        # 待开发功能
        self.use_harmony = False
        self.reasoning_parser = False

        super().__init__(instance_config)

    async def _abort_vllm_request(self, request_id: str):
        """在取消请求时通知 vLLM 停止底层推理。"""
        try:
            abort_fn = getattr(self.engine, "abort", None)
            if not callable(abort_fn):
                return
            result = abort_fn(request_id)
            if asyncio.iscoroutine(result):
                await result
            logger.info(f"TextEngine: 已通知 vLLM 终止请求 {request_id}")
        except Exception as e:
            logger.warning(f"TextEngine: 终止请求 {request_id} 失败: {e}")

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
            if vllm_api_server_args:
                logger.info(
                    "vLLM API server 层参数已接收；当前 worker 已是 HTTP server，"
                    f"不会再启动第二个 vLLM API server: {vllm_api_server_args}"
                )
            if vllm_unknown_args:
                logger.warning(f"vLLM 未识别参数不会传入 AsyncEngineArgs: {vllm_unknown_args}")

            self.vllm_frontend_args = make_vllm_frontend_namespace(
                vllm_frontend_args,
                vllm_engine_args,
            )
            self.response_role = getattr(self.vllm_frontend_args, "response_role", self.response_role)
            self.return_tokens_as_token_ids = bool(
                getattr(self.vllm_frontend_args, "return_tokens_as_token_ids", self.return_tokens_as_token_ids)
            )
            self.enable_force_include_usage = bool(
                getattr(self.vllm_frontend_args, "enable_force_include_usage", self.enable_force_include_usage)
            )
            self.chat_template = getattr(self.vllm_frontend_args, "chat_template", None)
            self.chat_template_kwargs = getattr(
                self.vllm_frontend_args,
                "default_chat_template_kwargs",
                None,
            ) or {}
            additional_config = dict(self.additional_config)
            vit_model_name = additional_config.pop("vit_model_name", None)

            if vit_model_name:
                visual_model_path = (
                    vit_model_name
                    if os.path.isabs(vit_model_name)
                    else os.path.join(self.model_path, vit_model_name)
                )
                additional_config.update(
                    {
                        "visual_model_path": visual_model_path,
                        "visual_model_w": 448,
                        "visual_model_h": 448,
                    }
                )

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

            # parse_gguf(self.model_path)

            # 配置 vLLM 引擎参数
            engine_arg_kwargs = {
                **vllm_engine_args,
                "model": self.model_path,
                "additional_config": additional_config,
            }
            logger.info(f"vllm 启动参数: {engine_arg_kwargs}")
            logger.info(f"vllm frontend/chat serving 参数: {vllm_frontend_args}")

            engine_args = AsyncEngineArgs(**engine_arg_kwargs)
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            self.engine_client = self.engine

            self._init_vllm_openai_serving(vllm_frontend_args)
            logger.info(f"vllm 实际启动参数：{self.engine.vllm_config}")
            logger.info(f"模型配置：{self.engine.vllm_config.model_config}")
            logger.info(f"并行配置：{self.engine.vllm_config.parallel_config}")
            logger.info(f"vLLM 引擎加载完成 {self.model_path} 到设备 {self.device_ids}")

            self.tokenizer = self.engine.get_tokenizer()

            gguf_metadata = get_gguf_metadata(self.model_path, include_tensors=False)
            logger.debug(f"GGUF Metadata: {gguf_metadata}")

        except Exception as e:
            logger.error(f"TextEngine: vLLM 引擎加载失败 {e}")
            raise RuntimeError(f"Failed to load vLLM engine: {e}")

    def _init_vllm_openai_serving(self, vllm_frontend_args: dict[str, Any]) -> None:
        """初始化 vLLM OpenAI chat serving 层，但复用当前 worker 的 HTTP server。"""
        try:
            frontend_args = self.vllm_frontend_args
            if frontend_args is None:
                return

            served_model_names = getattr(
                frontend_args, "served_model_name", None) or [self.model_name]
            if isinstance(served_model_names, str):
                served_model_names = [served_model_names]

            base_model_paths = [
                BaseModelPath(name=name, model_path=self.model_path)
                for name in served_model_names
            ]
            self.openai_serving_models = OpenAIServingModels(
                engine_client=self.engine,
                base_model_paths=base_model_paths,
                lora_modules=getattr(frontend_args, "lora_modules", None),
            )

            request_logger = None
            if getattr(frontend_args, "enable_log_requests", False):
                request_logger = RequestLogger(max_log_len=getattr(frontend_args, "max_log_len", None))

            resolved_chat_template = load_chat_template(self.chat_template)
            reasoning_parser = getattr(frontend_args, "reasoning_parser", None)
            if reasoning_parser is None:
                structured_outputs_config = getattr(frontend_args, "structured_outputs_config", None)
                reasoning_parser = getattr(structured_outputs_config, "reasoning_parser", "") or ""

            self.openai_serving_render = OpenAIServingRender(
                model_config=self.engine.model_config,
                renderer=self.engine.renderer,
                model_registry=self.openai_serving_models.registry,
                request_logger=request_logger,
                chat_template=resolved_chat_template,
                chat_template_content_format=getattr(frontend_args, "chat_template_content_format", "auto"),
                trust_request_chat_template=bool(getattr(frontend_args, "trust_request_chat_template", False)),
                enable_auto_tools=bool(getattr(frontend_args, "enable_auto_tool_choice", False)),
                exclude_tools_when_tool_choice_none=bool(
                    getattr(frontend_args, "exclude_tools_when_tool_choice_none", False)
                ),
                tool_parser=getattr(frontend_args, "tool_call_parser", None),
                reasoning_parser=reasoning_parser,
                default_chat_template_kwargs=self.chat_template_kwargs,
                log_error_stack=bool(getattr(frontend_args, "log_error_stack", False)),
            )

            chat_kwargs = {
                "engine_client": self.engine,
                "models": self.openai_serving_models,
                "response_role": self.response_role,
                "openai_serving_render": self.openai_serving_render,
                "request_logger": request_logger,
                "chat_template": resolved_chat_template,
                "chat_template_content_format": getattr(frontend_args, "chat_template_content_format", "auto"),
                "trust_request_chat_template": bool(getattr(frontend_args, "trust_request_chat_template", False)),
                "return_tokens_as_token_ids": self.return_tokens_as_token_ids,
                "reasoning_parser": reasoning_parser,
                "enable_auto_tools": bool(getattr(frontend_args, "enable_auto_tool_choice", False)),
                "exclude_tools_when_tool_choice_none": bool(
                    getattr(frontend_args, "exclude_tools_when_tool_choice_none", False)
                ),
                "tool_parser": getattr(frontend_args, "tool_call_parser", None),
                "enable_prompt_tokens_details": bool(
                    getattr(frontend_args, "enable_prompt_tokens_details", False)
                ),
                "enable_force_include_usage": self.enable_force_include_usage,
                "enable_log_outputs": bool(getattr(frontend_args, "enable_log_outputs", False)),
                "enable_log_deltas": bool(getattr(frontend_args, "enable_log_deltas", True)),
                "default_chat_template_kwargs": self.chat_template_kwargs,
            }
            allowed = set(inspect.signature(OpenAIServingChat).parameters)
            self.openai_serving_chat = OpenAIServingChat(
                **{key: value for key, value in chat_kwargs.items() if key in allowed}
            )
            responses_kwargs = {
                "engine_client": self.engine,
                "models": self.openai_serving_models,
                "openai_serving_render": self.openai_serving_render,
                "request_logger": request_logger,
                "chat_template": resolved_chat_template,
                "chat_template_content_format": getattr(frontend_args, "chat_template_content_format", "auto"),
                "return_tokens_as_token_ids": self.return_tokens_as_token_ids,
                "reasoning_parser": reasoning_parser,
                "enable_auto_tools": bool(getattr(frontend_args, "enable_auto_tool_choice", False)),
                "tool_parser": getattr(frontend_args, "tool_call_parser", None),
                "enable_prompt_tokens_details": bool(
                    getattr(frontend_args, "enable_prompt_tokens_details", False)
                ),
                "enable_force_include_usage": self.enable_force_include_usage,
                "enable_log_outputs": bool(getattr(frontend_args, "enable_log_outputs", False)),
                "default_chat_template_kwargs": self.chat_template_kwargs,
            }
            allowed = set(inspect.signature(OpenAIServingResponses).parameters)
            self.openai_serving_responses = OpenAIServingResponses(
                **{key: value for key, value in responses_kwargs.items() if key in allowed}
            )
            logger.info(f"vLLM OpenAI chat serving 层初始化完成: {vllm_frontend_args}")
        except Exception as exc:
            self.use_vllm_openai_serving = False
            logger.warning(f"vLLM OpenAI chat serving 层初始化失败，将回退旧 chat 流程: {exc}")

    def _to_vllm_chat_request(self, request: ChatCompletionRequest) -> VllmChatCompletionRequest:
        payload = request.model_dump(mode="json", exclude_none=True)
        payload.pop("server_config", None)
        payload.setdefault("model", self.model_name)
        if payload.get("return_tokens_as_token_ids") is None:
            payload["return_tokens_as_token_ids"] = self.return_tokens_as_token_ids
        return VllmChatCompletionRequest.model_validate(payload)

    def _to_vllm_responses_request(self, request: ResponsesRequest) -> ResponsesRequest:
        payload = request.model_dump(mode="json", exclude_none=True)
        payload.pop("server_config", None)
        payload.pop("port", None)
        payload.setdefault("model", self.model_name)
        return ResponsesRequest.model_validate(payload)


    def get_chat_request_role(self, request: ChatCompletionRequest) -> str:
        logger.debug(f" ---------------------------------------------- ")
        logger.debug(f"request.messages: {request.messages}")
        logger.debug(f" ---------------------------------------------- ")
        if request.add_generation_prompt:
            return self.response_role
        return request.messages[-1]["role"]

    def _create_sampling_params(
        self,
        request: ChatCompletionRequest
    ) -> SamplingParams:
        output_kind = RequestOutputKind.DELTA \
            if request.stream else RequestOutputKind.FINAL_ONLY

        # 如果request.xxx 为None，那么就使用SamplingParams.from_optional的默认值
        return SamplingParams.from_optional(
            # 基础采样参数
            n=request.n,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k if request.top_k is not None else -1,
            min_p=request.min_p if request.min_p is not None else 0.0,
            seed=request.seed,

            # 惩罚参数
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            repetition_penalty=request.repetition_penalty,

            # 停止条件
            max_tokens=request.max_completion_tokens if request.max_completion_tokens \
                is not None else (request.max_tokens if request.max_tokens is not None else 2048),
            min_tokens=request.min_tokens if request.min_tokens is not None else 0,
            stop=request.stop,
            stop_token_ids=request.stop_token_ids,
            ignore_eos=request.ignore_eos,

            # 输出控制
            logprobs=self._get_logprobs_value(request),
            prompt_logprobs=request.prompt_logprobs,
            detokenize=request.detokenize if hasattr(request, 'detokenize') else True,
            skip_special_tokens=request.skip_special_tokens,
            spaces_between_special_tokens=request.spaces_between_special_tokens,
            include_stop_str_in_output=request.include_stop_str_in_output,

            # 高级参数
            structured_outputs=request.structured_outputs,
            logit_bias=request.logit_bias,
            allowed_token_ids=request.allowed_token_ids,
            bad_words=request.bad_words if hasattr(request, 'bad_words') else None,

            # 输出类型和额外参数
            output_kind=output_kind,
            extra_args=request.vllm_xargs if hasattr(request, 'vllm_xargs') else None,
            skip_clone=True,  # 每次请求创建新实例，可以跳过克隆
        )

    async def _preprocess_chat(
        self,
        request: ChatCompletionRequest,
        request_messages: str
    ):
        conversation = parse_chat_messages(request_messages)

        enable_think = False
        reasoning_effort = request.reasoning_effort
        if reasoning_effort is not None:
            enable_think = reasoning_effort != "none"

        chat_template_kwargs = dict(self.chat_template_kwargs)
        if request.chat_template_kwargs:
            chat_template_kwargs.update(request.chat_template_kwargs)
        chat_template_kwargs.setdefault("enable_thinking", enable_think)

        request_prompt = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,  # 返回字符串而非 token IDs
            add_generation_prompt=True,
            chat_template=request.chat_template or self.chat_template,
            **chat_template_kwargs,
        )

        sampling_params = self._create_sampling_params(request)

        return conversation, request_prompt, sampling_params

    def _get_logprobs_value(self, request: ChatCompletionRequest):
        # vllm 的SamplingParams 中的logprobs=None 和 logprobs=0 有区别
        #  - logprobs=None: 完全不返回任何 log 概率信息
        #  - logprobs=0: 只返回采样 token 的 log 概率，不返回其他候选 token
        if not getattr(request, "logprobs", False):
            return None

        top_logprobs = getattr(request, "top_logprobs", None)
        if top_logprobs is None:
            return 0

        # OpenAI 官方限制是 0~20，防御性校验
        return max(0, min(int(top_logprobs), 20))

    async def create_chat_completions(
        self,
        request: ChatCompletionRequest,
        raw_request: Request | None = None,
    ):
        if self.use_vllm_openai_serving and self.openai_serving_chat is not None:
            try:
                vllm_request = self._to_vllm_chat_request(request)
                result = await self.openai_serving_chat.create_chat_completion(
                    vllm_request,
                    raw_request,
                )
                if isinstance(result, AsyncGenerator):
                    return StreamingResponse(result, media_type="text/event-stream")
                return result
            except Exception as exc:
                logger.exception(f"vLLM OpenAI chat serving 处理失败，将回退旧 chat 流程: {exc}")

        try:
            model_name = self.model_name
            (conversation, request_prompt, sampling_params) \
                = await self._preprocess_chat(request, request.messages)

        except (ValueError, TypeError, RuntimeError, jinja2.TemplateError) as e:
            import traceback
            logger.info("Error in preprocessing prompt inputs")
            return self.create_error_response(f"{e} {e.__cause__} {traceback.format_exc()}")

        # 条件性处理视觉信息
        image_data = None
        try:
            image_data, _ = process_vision_info(conversation)
        except Exception as e:
            logger.info("图像处理失败")
            return self.create_error_response(f"{e} {e.__cause__}")

        engine_input = {
            "prompt": request_prompt
        }
        if image_data is not None:
            engine_input["multi_modal_data"] = {"image": image_data}

        request_id = self._base_request_id(raw_request)

        fingerprint_data = {
            "model_name": model_name,
            "sampling_params": sampling_params.__dict__,
            "prompt": engine_input,
        }

        serializable_data = self.serialize_for_fingerprint(fingerprint_data)
        fingerprint_str = json.dumps(serializable_data, sort_keys=True)
        system_fingerprint = f"fp-{hashlib.sha256(fingerprint_str.encode()).hexdigest()[:10]}"

        # Schedule the request and get the result generator.
        generator: AsyncGenerator[RequestOutput, None]

        logger.info(f"engine_input: {engine_input}")

        try:
            generator = self.engine.generate(
                sampling_params=sampling_params,
                request_id=request_id,
                prompt=engine_input,
            )
        except EngineDeadError:
            logger.critical("vLLM engine is dead")
            # return self.create_error_response("vLLM engine is dead")
            raise
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        if request.stream:
            return StreamingResponse(
                self.chat_completion_stream_generator(
                    request,
                    generator,
                    request_id,
                    system_fingerprint,
                    model_name,
                    conversation,
                ),
                media_type="text/event-stream",
            )

        try:
            return await self.chat_completion_full_generator(
                request,
                generator,
                request_id,
                system_fingerprint,
                model_name,
                conversation,
            )
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

    async def create_responses(
        self,
        request: ResponsesRequest,
        raw_request: Request | None = None,
    ):
        if not self.use_vllm_openai_serving or self.openai_serving_responses is None:
            return self.create_error_response("当前模型不支持 Responses API")

        try:
            vllm_request = self._to_vllm_responses_request(request)
            return await self.openai_serving_responses.create_responses(
                vllm_request,
                raw_request,
            )
        except Exception as exc:
            logger.exception(f"vLLM Responses API 处理失败: {exc}")
            return self.create_error_response(str(exc))


    async def chat_completion_stream_generator(
        self,
        request: ChatCompletionRequest,
        generator,
        request_id: str,
        system_fingerprint: str,
        model_name: str,
        conversation,
    ) -> AsyncGenerator[str, None]:
        created_time = int(time.time())
        chunk_object_type = "chat.completion.chunk"
        first_iteration = True

        # Send response for each token for each request.n (index)
        num_choices = 1 if request.n is None else request.n

        previous_num_tokens = [0] * num_choices
        finish_reason_sent = [False] * num_choices
        num_prompt_tokens = 0

        tool_choice_function_name = None
        history_tool_call_cnt = 0
        all_previous_token_ids = None

        stream_options = request.stream_options
        include_usage, include_continuous_usage = should_include_usage(
            stream_options, self.enable_force_include_usage
        )

        try:
            async for res in generator:
                logger.debug(" ------------------------------------------- ")
                logger.debug(f"本次迭代的结果：{res}")
                logger.debug(" ------------------------------------------- ")
                if res.prompt_token_ids is not None:
                    num_prompt_tokens = len(res.prompt_token_ids)
                    if res.encoder_prompt_token_ids is not None:
                        num_prompt_tokens += len(res.encoder_prompt_token_ids)

                # We need to do it here, because if there are exceptions in
                # the result_generator, it needs to be sent as the FIRST
                # response (by the try...catch).
                if first_iteration:
                    num_cached_tokens = res.num_cached_tokens
                    # Send first response for each request.n (index) with
                    # the role
                    role = self.get_chat_request_role(request)

                    # NOTE num_choices defaults to 1 so this usually executes
                    # once per request
                    for i in range(num_choices):
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=DeltaMessage(
                                role=role,
                                content="",
                            ),
                            logprobs=None,
                            finish_reason=None,
                        )

                        # return prompt_token_ids at the first chunk ever
                        chunk = ChatCompletionStreamResponse(
                            id=request_id,
                            object=chunk_object_type,
                            created=created_time,
                            choices=[choice_data],
                            model=model_name,
                            # prompt_token_ids=(
                            #     res.prompt_token_ids
                            #     if request.return_token_ids
                            #     else None
                            # ),
                            system_fingerprint=system_fingerprint,
                        )


                        # if continuous usage stats are requested, add it
                        if include_continuous_usage:
                            chunk.usage = UsageInfo(
                                prompt_tokens=num_prompt_tokens,
                                completion_tokens=0,
                                total_tokens=num_prompt_tokens,
                            )

                        data = chunk.model_dump_json(exclude_unset=True)
                        yield f"data: {data}\n\n"

                    first_iteration = False

                for output in res.outputs:
                    i = output.index
                    tool_parser = None

                    if request.logprobs and request.top_logprobs is not None:
                        assert output.logprobs is not None, "Did not output logprobs"
                        logprobs = self._create_chat_logprobs(
                            token=output.text,
                            token_ids=output.token_ids,
                            token_logprobs=output.cumulative_logprob,
                            top_logprobs=output.logprobs,
                            num_output_top_logprobs=request.top_logprobs,
                        )
                    else:
                        logprobs = None

                    delta_text = output.text

                    # 作用是在分块预填充（chunked prefill）场景下过滤空响应块，避免向客户端发送无意义的空数据
                    # if (
                    #     not delta_text
                    #     and not output.token_ids
                    #     and not previous_num_tokens[i]
                    # ):
                    #     # Chunked prefill case, don't return empty chunks
                    #     continue

                    delta_message = DeltaMessage(content=delta_text)

                    # set the previous values for the next iteration
                    previous_num_tokens[i] += len(output.token_ids)

                    # if the message delta is None (e.g. because it was a
                    # "control token" for tool calls or the parser otherwise
                    # wasn't ready to send a token, then
                    #   get the next token without streaming a chunk
                    if delta_message is None:
                        # NOTE: If return_token_ids is enabled, we still need to
                        # send a chunk with token_ids even if delta_message is None
                        # to ensure all tokens are included in the response
                        if (
                            output.finish_reason is None
                            and not request.return_token_ids
                        ):
                            continue
                        delta_message = DeltaMessage()

                    if output.finish_reason is None:
                        # Send token-by-token response for each request.n
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=delta_message,
                            logprobs=logprobs,
                            finish_reason=None,
                        )

                    # if the model is finished generating
                    else:
                        # check to make sure we haven't "forgotten" to stream
                        #   any tokens that were generated but previously
                        #   matched by partial json parsing
                        # only happens if we are NOT using structured outputs
                        auto_tools_called = False
                        if tool_parser:
                            auto_tools_called = len(tool_parser.prev_tool_call_arr) > 0
                            index = (
                                len(tool_parser.prev_tool_call_arr) - 1
                                if auto_tools_called
                                else 0
                            )
                        else:
                            index = 0

                        finish_reason_ = (
                            output.finish_reason if output.finish_reason else "stop"
                        )

                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=delta_message,
                            logprobs=logprobs,
                            finish_reason=finish_reason_,
                            stop_reason=output.stop_reason,
                            # token_ids=(
                            #     as_list(output.token_ids)
                            #     if request.return_token_ids
                            #     else None
                            # ),
                        )

                        finish_reason_sent[i] = True

                    # choice_data = maybe_filter_parallel_tool_calls(choice_data, request)
                    chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        object=chunk_object_type,
                        created=created_time,
                        choices=[choice_data],
                        model=model_name,
                        system_fingerprint=system_fingerprint,
                    )

                    # handle usage stats if requested & if continuous
                    if include_continuous_usage:
                        completion_tokens = previous_num_tokens[i]
                        chunk.usage = UsageInfo(
                            prompt_tokens=num_prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=num_prompt_tokens + completion_tokens,
                        )

                    data = chunk.model_dump_json(exclude_unset=True)
                    yield f"data: {data}\n\n"

            # once the final token is handled, if stream_options.include_usage
            # is sent, send the usage
            if include_usage:
                completion_tokens = sum(previous_num_tokens)
                final_usage = UsageInfo(
                    prompt_tokens=num_prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=num_prompt_tokens + completion_tokens,
                )
                # if self.enable_prompt_tokens_details and num_cached_tokens:
                #     final_usage.prompt_tokens_details = PromptTokenUsageInfo(
                #         cached_tokens=num_cached_tokens
                #     )

                final_usage_chunk = ChatCompletionStreamResponse(
                    id=request_id,
                    object=chunk_object_type,
                    created=created_time,
                    choices=[],
                    model=model_name,
                    usage=final_usage,
                    system_fingerprint=system_fingerprint,
                    # timings=build_timings(
                    #     created_time,
                    #     UsageInfo(
                    #         prompt_tokens=num_prompt_tokens,
                    #         completion_tokens=completion_tokens,
                    #         total_tokens=num_prompt_tokens + completion_tokens,
                    #     )
                    # ),
                )
                final_usage_data = final_usage_chunk.model_dump_json(
                    exclude_unset=True, exclude_none=True
                )
                yield f"data: {final_usage_data}\n\n"
        except EngineDeadError:
            logger.critical("vLLM engine is dead during streaming")
            raise
        except Exception as e:
            # TODO: Use a vllm-specific Validation Error
            logger.exception("Error in chat completion stream generator.")
            data = self.create_streaming_error_response(str(e))
            yield f"data: {data}\n\n"

        # Send the final done message after all response.n are finished
        yield "data: [DONE]\n\n"

    async def chat_completion_full_generator(
        self,
        request: ChatCompletionRequest,
        result_generator,
        request_id: str,
        system_fingerprint: str,
        model_name: str,
        conversation,
    ) -> ErrorResponse | ChatCompletionResponse:
        created_time = int(time.time())
        final_res: RequestOutput | None = None

        try:
            async for res in result_generator:
                logger.debug(f" ---------------------------------- ")
                logger.debug(f"res: {res}")
                logger.debug(f" ---------------------------------- ")
                final_res = res
        except EngineDeadError:
            logger.critical(f"Engine dead, abort {request_id}")
            raise
        except asyncio.CancelledError:
            # return self.create_error_response("Client disconnected")
            logger.warning(f"Client disconnected, abort {request_id}")
            await self.engine.abort(request_id)
            raise
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        assert final_res is not None

        choices: list[ChatCompletionResponseChoice] = []
        history_tool_call_cnt = 0

        role = self.get_chat_request_role(request)
        for output in final_res.outputs:
            token_ids = output.token_ids
            out_logprobs = output.logprobs
            tool_call_info = None

            if request.logprobs and request.top_logprobs is not None:
                assert out_logprobs is not None, "Did not output logprobs"
                logprobs = self._create_chat_logprobs(
                    token=output.text,
                    token_ids=output.token_ids,
                    token_logprobs=output.cumulative_logprob,
                    top_logprobs=output.logprobs,
                    num_output_top_logprobs=request.top_logprobs,
                )
            else:
                logprobs = None

            if self.use_harmony:
                # Todo
                pass

            if self.reasoning_parser:
                # Todo
                pass
            else:
                reasoning = None
                content = output.text

            auto_tools_called = False
            # if auto tools are not enabled, and a named tool choice using
            #   outlines is not being used
            if auto_tools_called:
                # Todo
                pass
            else:
                logger.warning(
                    "Error in chat_completion_full_generator - cannot determine"
                    " if tools should be extracted. Returning a standard chat "
                    "completion."
                )
                message = ChatMessage(role=role, reasoning=reasoning, content=content)

            # In OpenAI's API, when a tool is called, the finish_reason is:
            # "tool_calls" for "auto" or "required" tool calls,
            # and "stop" for named tool calls.
            is_finish_reason_tool_calls = False

            choice_data = ChatCompletionResponseChoice(
                index=output.index,
                message=message,
                logprobs=logprobs,
                finish_reason="tool_calls"
                if is_finish_reason_tool_calls
                else output.finish_reason
                if output.finish_reason
                else "stop",
                stop_reason=output.stop_reason,
                # token_ids=(
                #     as_list(output.token_ids) if request.return_token_ids else None
                # ),
            )
            # choice_data = maybe_filter_parallel_tool_calls(choice_data, request)

            choices.append(choice_data)

        assert final_res.prompt_token_ids is not None
        logger.info(f"final res: {final_res}")
        num_prompt_tokens = len(final_res.prompt_token_ids)
        num_generated_tokens = sum(
            len(output.token_ids) for output in final_res.outputs
        )
        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )
        response = ChatCompletionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage,
            system_fingerprint=system_fingerprint,
        )

        return response


    def _create_chat_logprobs(
        self,
        token: GenericSequence[str],
        token_ids: GenericSequence[int],
        token_logprobs: GenericSequence[float],
        top_logprobs: GenericSequence[dict[int, Logprob] | None],
        num_output_top_logprobs: int | None = None,
        return_as_token_id: bool | None = None,
    ) -> ChatCompletionLogProbs:
        """Create OpenAI-style logprobs."""
        logprobs_content: list[ChatCompletionLogProbsContent] = []

        should_return_as_token_id = (
            return_as_token_id
            if return_as_token_id is not None
            else self.return_tokens_as_token_ids
        )
        for i, token_id in enumerate(token_ids):
            step_top_logprobs = top_logprobs[i]
            if step_top_logprobs is None or step_top_logprobs.get(token_id) is None:
                if should_return_as_token_id:
                    token = f"token_id:{token_id}"
                else:
                    # token = tokenizer.decode(token_id)
                    token = f"token_id:{token_id}"

                logprobs_content.append(
                    ChatCompletionLogProbsContent(
                        token=token,
                        bytes=list(token.encode("utf-8", errors="replace")),
                    )
                )
            else:
                step_token = step_top_logprobs[token_id]
                step_decoded = step_token.decoded_token

                logprobs_content.append(
                    ChatCompletionLogProbsContent(
                        token=step_decoded,
                        logprob=max(step_token.logprob, -9999.0),
                        bytes=(
                            None
                            if step_decoded is None
                            else list(step_decoded.encode("utf-8", errors="replace"))
                        ),
                        top_logprobs=self._get_top_logprobs(
                            step_top_logprobs,
                            num_output_top_logprobs,
                            should_return_as_token_id,
                        ),
                    )
                )

        return ChatCompletionLogProbs(content=logprobs_content)

    def _get_top_logprobs(
        self,
        logprobs: dict[int, Logprob],
        top_logprobs: int | None,
        should_return_as_token_id: bool,
    ) -> list[ChatCompletionLogProb]:
        results: list[ChatCompletionLogProb] = []

        # logger.info(f"logprobs: {logprobs}")

        for index, (token_id, logprob_obj) in enumerate(logprobs.items()):
            # 是否需要截断 top_logprobs
            if top_logprobs is not None:
                if top_logprobs == -1:
                    pass  # -1 表示返回全部
                elif index >= top_logprobs:
                    break

            # 解码 token（或直接返回 token_id）
            token = logprob_obj.decoded_token

            results.append(
                ChatCompletionLogProb(
                    token=token,
                    logprob=max(logprob_obj.logprob, -9999.0),
                    bytes=list(token.encode("utf-8", errors="replace")),
                )
            )

        return results
