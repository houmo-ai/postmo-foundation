# Copyright 2024-2026 HLIEPython Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -*- coding: UTF-8 -*-
import time
import json
import torch
import uuid
from PIL import Image
from fastapi import HTTPException
from pydantic import (
    BaseModel, ConfigDict, Field, TypeAdapter,
    ValidationInfo, field_validator, model_validator)
from typing import List, Literal, ClassVar, Optional, Union, Tuple, Dict, Any, Callable, cast
from typing import Union, Iterable
from typing_extensions import Literal, Required, TypedDict, TypeAlias
from typing import Annotated, Any, ClassVar, Literal

from functools import partial
from collections.abc import Iterable

from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import (
    BeamSearchParams,
    RepetitionDetectionParams,
    RequestOutputKind,
    SamplingParams,
    StructuredOutputsParams,
)
from vllm.utils import random_uuid
from vllm.entrypoints.chat_utils import (
    ConversationMessage,
    make_tool_call_id,
)
from vllm.entrypoints.openai.engine.protocol import (
    OpenAIBaseModel
)

from openai.types.chat.chat_completion_content_part_input_audio_param import InputAudio

from hliepython.utils.logging import async_logger as logger

_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1
_LONG_INFO = torch.iinfo(torch.long)

#####################  Request  #####################
class Audio(TypedDict, total=False):
    id: Required[str]


class Function(TypedDict, total=False):
    arguments: Required[str]
    name: Required[str]


class ImageURL(TypedDict, total=False):
    url: Required[str]
    detail: Literal["auto", "low", "high"]


# class InputAudio(TypedDict, total=False):
#     data: Required[str]
#     format: Required[Literal["wav", "mp3"]]


class FileFile(TypedDict, total=False):
    file_data: str
    file_id: str
    filename: str


class FunctionCall(TypedDict, total=False):
    arguments: Required[str]
    name: Required[str]


class ChatCompletionContentPartTextParam(TypedDict, total=False):
    text: Required[str]
    type: Required[Literal["text"]]


class ChatCompletionContentPartImageParam(TypedDict, total=False):
    image_url: Required[ImageURL]
    type: Required[Literal["image_url"]]


class ChatCompletionContentPartInputAudioParam(TypedDict, total=False):
    input_audio: Required[InputAudio]
    type: Required[Literal["input_audio"]]


class File(TypedDict, total=False):
    file: Required[FileFile]
    type: Required[Literal["file"]]


class ChatCompletionContentPartRefusalParam(TypedDict, total=False):
    refusal: Required[str]
    type: Required[Literal["refusal"]]


class ChatCompletionMessageToolCallParam(TypedDict, total=False):
    id: Required[str]
    function: Required[Function]
    type: Required[Literal["function"]]


ChatCompletionContentPartParam: TypeAlias = Union[
    ChatCompletionContentPartTextParam,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartInputAudioParam,
    File,
]


ContentArrayOfContentPart: TypeAlias = Union[
    ChatCompletionContentPartTextParam,
    ChatCompletionContentPartRefusalParam
]


class ChatCompletionMessageDeveloperParam(TypedDict, total=False):
    content: Required[Union[str, Iterable[ChatCompletionContentPartTextParam]]]
    role: Required[Literal["developer"]]
    name: str


class ChatCompletionMessageSystemParam(TypedDict, total=False):
    content: Required[Union[str, Iterable[ChatCompletionContentPartTextParam]]]
    role: Required[Literal["system"]]
    name: str


class ChatCompletionMessageUserParam(TypedDict, total=False):
    content: Required[Union[str, list[ChatCompletionContentPartParam]]]
    role: Required[Literal["user"]]
    name: str


class ChatCompletionMessageAssistantParam(TypedDict, total=False):
    role: Required[Literal["assistant"]]
    audio: Optional[Audio]
    content: Union[str, Iterable[ContentArrayOfContentPart], None]
    function_call: Optional[FunctionCall]
    name: str
    refusal: Optional[str]
    tool_calls: Iterable[ChatCompletionMessageToolCallParam]


class ChatCompletionMessageToolParam(TypedDict, total=False):
    content: Required[Union[str, Iterable[ChatCompletionContentPartTextParam]]]
    role: Required[Literal["tool"]]
    tool_call_id: Required[str]


class ChatCompletionMessageFunctionParam(TypedDict, total=False):
    content: Required[Optional[str]]
    name: Required[str]
    role: Required[Literal["function"]]


ChatCompletionMessageParam: TypeAlias = Union[
    ChatCompletionMessageDeveloperParam,
    ChatCompletionMessageSystemParam,
    ChatCompletionMessageUserParam,
    ChatCompletionMessageAssistantParam,
    ChatCompletionMessageToolParam,
    ChatCompletionMessageFunctionParam,
]


# class ConversationMessage(TypedDict, total=False):
#     role: Required[str]
#     """The role of the message's author."""

#     content: Union[Optional[str], list[dict[str, str]]]
#     """The contents of the message"""


class StreamOptions(OpenAIBaseModel):
    include_usage: Optional[bool] = True


class JsonSchemaResponseFormat(OpenAIBaseModel):
    name: str
    description: Optional[str] = None
    # schema is the field in openai but that causes conflicts with pydantic so
    # instead use json_schema with an alias
    json_schema: Optional[dict[str, Any]] = Field(default=None, alias='schema')
    strict: Optional[bool] = None


class ResponseFormat(OpenAIBaseModel):
    # type must be "json_schema", "json_object", or "text"
    type: Optional[Literal["text", "json_object", "json_schema"]] = None
    json_schema: Optional[JsonSchemaResponseFormat] = None


class FunctionDefinition(OpenAIBaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[dict[str, Any]] = None
    strict: Optional[bool] = False


class ChatCompletionToolsParam(OpenAIBaseModel):
    type: Literal["function"] = "function"
    function: FunctionDefinition


class ChatCompletionNamedFunction(OpenAIBaseModel):
    name: str


class ChatCompletionNamedToolChoiceParam(OpenAIBaseModel):
    function: ChatCompletionNamedFunction
    type: Literal["function"] = "function"

class ChatCompletionRequest(OpenAIBaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/chat/create
    messages: list[ChatCompletionMessageParam]
    model: str | None = None
    frequency_penalty: float | None = 0.0
    logit_bias: dict[str, float] | None = None
    logprobs: bool | None = False
    top_logprobs: int | None = 0
    max_tokens: int | None = Field(
        default=None,
        deprecated="max_tokens is deprecated in favor of "
        "the max_completion_tokens field",
    )
    max_completion_tokens: int | None = None
    n: int | None = 1
    presence_penalty: float | None = 0.0
    response_format: ResponseFormat | None = None
    seed: int | None = Field(None, ge=_LONG_INFO.min, le=_LONG_INFO.max)
    stop: str | list[str] | None = []
    stream: bool | None = False
    stream_options: StreamOptions | None = None
    temperature: float | None = None
    top_p: float | None = None
    tools: list[ChatCompletionToolsParam] | None = None
    tool_choice: (
        Literal["none"]
        | Literal["auto"]
        | Literal["required"]
        | ChatCompletionNamedToolChoiceParam
        | None
    ) = "none"
    # 控制推理强度，可选值为 "none"（默认，不进行推理）、"low"、"medium"（中度推理）和 "high"。
    reasoning_effort: Literal["none", "low", "medium", "high"] | None = None
    thinking_token_budget: int | None = None        # 限制思考阶段的token 数量
    include_reasoning: bool = True                  # 控制是否在响应中包含模型的推理内容
    parallel_tool_calls: bool | None = True

    # NOTE this will be ignored by vLLM
    user: str | None = None

    # --8<-- [start:houmo-chat-completion-params]
    server_config: dict[str, Any] | None = Field(
        default=None,
        description=(
            "这里用于配置服务端的启动参数，例如启动的设备ID、端口、vllm_config等。"
        ),
    )
    # --8<-- [end:houmo-chat-completion-params]

    # --8<-- [start:chat-completion-sampling-params]
    use_beam_search: bool = False
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    length_penalty: float = 1.0
    stop_token_ids: list[int] | None = []
    include_stop_str_in_output: bool = False
    ignore_eos: bool = False
    min_tokens: int = 0
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    truncate_prompt_tokens: Annotated[int, Field(ge=-1, le=_INT64_MAX)] | None = None
    prompt_logprobs: int | None = None
    allowed_token_ids: list[int] | None = None
    bad_words: list[str] = Field(default_factory=list)
    # --8<-- [end:chat-completion-sampling-params]

    # --8<-- [start:chat-completion-extra-params]
    echo: bool = Field(
        default=False,
        description=(
            "If true, the new message will be prepended with the last message "
            "if they belong to the same role."
        ),
    )
    add_generation_prompt: bool = Field(
        default=True,
        description=(
            "If true, the generation prompt will be added to the chat template. "
            "This is a parameter used by chat template in tokenizer config of the "
            "model."
        ),
    )
    continue_final_message: bool = Field(
        default=False,
        description=(
            "If this is set, the chat will be formatted so that the final "
            "message in the chat is open-ended, without any EOS tokens. The "
            "model will continue this message rather than starting a new one. "
            'This allows you to "prefill" part of the model\'s response for it. '
            "Cannot be used at the same time as `add_generation_prompt`."
        ),
    )
    add_special_tokens: bool = Field(
        default=False,
        description=(
            "If true, special tokens (e.g. BOS) will be added to the prompt "
            "on top of what is added by the chat template. "
            "For most models, the chat template takes care of adding the "
            "special tokens so this should be set to false (as is the "
            "default)."
        ),
    )
    documents: list[dict[str, str]] | None = Field(
        default=None,
        description=(
            "A list of dicts representing documents that will be accessible to "
            "the model if it is performing RAG (retrieval-augmented generation)."
            " If the template does not support RAG, this argument will have no "
            "effect. We recommend that each document should be a dict containing "
            '"title" and "text" keys.'
        ),
    )
    chat_template: str | None = Field(
        default=None,
        description=(
            "A Jinja template to use for this conversion. "
            "As of transformers v4.44, default chat template is no longer "
            "allowed, so you must provide a chat template if the tokenizer "
            "does not define one."
        ),
    )
    chat_template_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Additional keyword args to pass to the template renderer. "
            "Will be accessible by the chat template."
        ),
    )
    media_io_kwargs: dict[str, dict[str, Any]] | None = Field(
        default=None,
        description=(
            "Additional kwargs to pass to the media IO connectors, "
            "keyed by modality. Merged with engine-level media_io_kwargs."
        ),
    )
    mm_processor_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=("Additional kwargs to pass to the HF processor."),
    )
    structured_outputs: StructuredOutputsParams | None = Field(
        default=None,
        description="Additional kwargs for structured outputs",
    )
    priority: int = Field(
        default=0,
        ge=_INT64_MIN,
        le=_INT64_MAX,
        description=(
            "The priority of the request (lower means earlier handling; "
            "default: 0). Any priority other than 0 will raise an error "
            "if the served model does not use priority scheduling."
        ),
    )
    request_id: str = Field(
        default_factory=random_uuid,
        description=(
            "The request_id related to this request. If the caller does "
            "not set it, a random_uuid will be generated. This id is used "
            "through out the inference process and return in response."
        ),
    )

    return_tokens_as_token_ids: bool | None = Field(
        default=None,
        description=(
            "If specified with 'logprobs', tokens are represented "
            " as strings of the form 'token_id:{token_id}' so that tokens "
            "that are not JSON-encodable can be identified."
        ),
    )
    return_token_ids: bool | None = Field(
        default=None,
        description=(
            "If specified, the result will include token IDs alongside the "
            "generated text. In streaming mode, prompt_token_ids is included "
            "only in the first chunk, and token_ids contains the delta tokens "
            "for each chunk. This is useful for debugging or when you "
            "need to map generated text back to input tokens."
        ),
    )

    cache_salt: str | None = Field(
        default=None,
        description=(
            "If specified, the prefix cache will be salted with the provided "
            "string to prevent an attacker to guess prompts in multi-user "
            "environments. The salt should be random, protected from "
            "access by 3rd parties, and long enough to be "
            "unpredictable (e.g., 43 characters base64-encoded, corresponding "
            "to 256 bit)."
        ),
    )

    kv_transfer_params: dict[str, Any] | None = Field(
        default=None,
        description="KVTransfer parameters used for disaggregated serving.",
    )

    vllm_xargs: dict[str, str | int | float | list[str | int | float]] | None = Field(
        default=None,
        description=(
            "Additional request parameters with (list of) string or "
            "numeric values, used by custom extensions."
        ),
    )

    repetition_detection: RepetitionDetectionParams | None = Field(
        default=None,
        description="Parameters for detecting repetitive N-gram patterns "
        "in output tokens. If such repetition is detected, generation will "
        "be ended early. LLMs can sometimes generate repetitive, unhelpful "
        "token patterns, stopping only when they hit the maximum output length "
        "(e.g. 'abcdabcdabcd...' or '\\emoji \\emoji \\emoji ...'). This feature "
        "can detect such behavior and terminate early, saving time and tokens.",
    )

    # --8<-- [end:chat-completion-extra-params]

    # Default sampling parameters for chat completion requests
    _DEFAULT_SAMPLING_PARAMS: dict = {
        "repetition_penalty": 1.0,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
    }

    @model_validator(mode="before")
    @classmethod
    def validate_response_format(cls, data):
        response_format = data.get("response_format")
        if response_format is None:
            return data

        rf_type = (
            response_format.get("type")
            if isinstance(response_format, dict)
            else getattr(response_format, "type", None)
        )

        if rf_type == "json_schema":
            json_schema = (
                response_format.get("json_schema")
                if isinstance(response_format, dict)
                else getattr(response_format, "json_schema", None)
            )
            if json_schema is None:
                raise VLLMValidationError(
                    "When response_format type is 'json_schema', the "
                    "'json_schema' field must be provided.",
                    parameter="response_format",
                )

        return data

    @model_validator(mode="before")
    @classmethod
    def validate_stream_options(cls, data):
        if data.get("stream_options") and not data.get("stream"):
            raise VLLMValidationError(
                "Stream options can only be defined when `stream=True`.",
                parameter="stream_options",
            )

        return data

    @model_validator(mode="before")
    @classmethod
    def check_logprobs(cls, data):
        if (prompt_logprobs := data.get("prompt_logprobs")) is not None:
            if data.get("stream") and (prompt_logprobs > 0 or prompt_logprobs == -1):
                raise VLLMValidationError(
                    "`prompt_logprobs` are not available when `stream=True`.",
                    parameter="prompt_logprobs",
                )

            if prompt_logprobs < 0 and prompt_logprobs != -1:
                raise VLLMValidationError(
                    "`prompt_logprobs` must be a positive value or -1.",
                    parameter="prompt_logprobs",
                    value=prompt_logprobs,
                )
        if (top_logprobs := data.get("top_logprobs")) is not None:
            if top_logprobs < 0 and top_logprobs != -1:
                raise VLLMValidationError(
                    "`top_logprobs` must be a positive value or -1.",
                    parameter="top_logprobs",
                    value=top_logprobs,
                )

            if (top_logprobs == -1 or top_logprobs > 0) and not data.get("logprobs"):
                raise VLLMValidationError(
                    "when using `top_logprobs`, `logprobs` must be set to true.",
                    parameter="top_logprobs",
                )

        return data

    @model_validator(mode="before")
    @classmethod
    def check_structured_outputs_count(cls, data):
        if isinstance(data, ValueError):
            raise data

        if data.get("structured_outputs", None) is None:
            return data

        structured_outputs_kwargs = data["structured_outputs"]
        # structured_outputs may arrive as a dict (from JSON/raw kwargs) or
        # as a StructuredOutputsParams dataclass instance.
        is_dataclass = isinstance(structured_outputs_kwargs, StructuredOutputsParams)
        count = sum(
            (
                getattr(structured_outputs_kwargs, k, None)
                if is_dataclass
                else structured_outputs_kwargs.get(k)
            )
            is not None
            for k in ("json", "regex", "choice")
        )
        # you can only use one kind of constraints for structured outputs
        if count > 1:
            raise ValueError(
                "You can only use one kind of constraints for structured "
                "outputs ('json', 'regex' or 'choice')."
            )
        # you can only either use structured outputs or tools, not both
        if count > 1 and data.get("tool_choice", "none") not in (
            "none",
            "auto",
            "required",
        ):
            raise ValueError(
                "You can only either use constraints for structured outputs "
                "or tools, not both."
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_tool_usage(cls, data):
        # if "tool_choice" is not specified but tools are provided,
        # default to "auto" tool_choice
        if "tool_choice" not in data and data.get("tools"):
            data["tool_choice"] = "auto"

        # if "tool_choice" is "none" -- no validation is needed for tools
        if "tool_choice" in data and data["tool_choice"] == "none":
            return data

        # if "tool_choice" is specified -- validation
        if "tool_choice" in data and data["tool_choice"] is not None:
            # ensure that if "tool choice" is specified, tools are present
            if "tools" not in data or data["tools"] is None:
                raise ValueError("When using `tool_choice`, `tools` must be set.")

            # make sure that tool choice is either a named tool
            # OR that it's set to "auto" or "required"
            if data["tool_choice"] not in ["auto", "required"] and not isinstance(
                data["tool_choice"], dict
            ):
                raise ValueError(
                    f"Invalid value for `tool_choice`: {data['tool_choice']}! "
                    'Only named tools, "none", "auto" or "required" '
                    "are supported."
                )

            # if tool_choice is "required" but the "tools" list is empty,
            # override the data to behave like "none" to align with
            # OpenAI’s behavior.
            if (
                data["tool_choice"] == "required"
                and isinstance(data["tools"], list)
                and len(data["tools"]) == 0
            ):
                data["tool_choice"] = "none"
                del data["tools"]
                return data

            # ensure that if "tool_choice" is specified as an object,
            # it matches a valid tool
            correct_usage_message = (
                'Correct usage: `{"type": "function",'
                ' "function": {"name": "my_function"}}`'
            )
            if isinstance(data["tool_choice"], dict):
                valid_tool = False
                function = data["tool_choice"].get("function")
                if not isinstance(function, dict):
                    raise ValueError(
                        f"Invalid value for `function`: `{function}` in "
                        f"`tool_choice`! {correct_usage_message}"
                    )
                if "name" not in function:
                    raise ValueError(
                        f"Expected field `name` in `function` in "
                        f"`tool_choice`! {correct_usage_message}"
                    )
                function_name = function["name"]
                if not isinstance(function_name, str) or len(function_name) == 0:
                    raise ValueError(
                        f"Invalid `name` in `function`: `{function_name}`"
                        f" in `tool_choice`! {correct_usage_message}"
                    )
                for tool in data["tools"]:
                    if tool["function"]["name"] == function_name:
                        valid_tool = True
                        break
                if not valid_tool:
                    raise ValueError(
                        "The tool specified in `tool_choice` does not match any"
                        " of the specified `tools`"
                    )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_generation_prompt(cls, data):
        if data.get("continue_final_message") and data.get("add_generation_prompt"):
            raise ValueError(
                "Cannot set both `continue_final_message` and "
                "`add_generation_prompt` to True."
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_cache_salt_support(cls, data):
        if data.get("cache_salt") is not None and (
            not isinstance(data["cache_salt"], str) or not data["cache_salt"]
        ):
            raise ValueError(
                "Parameter 'cache_salt' must be a non-empty string if provided."
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_system_message_content_type(cls, data):
        """Warn if system messages contain non-text content.

        According to OpenAI API spec, system messages can only be of type
        'text'. We log a warning instead of rejecting to avoid breaking
        users who intentionally send multimodal system messages.
        See: https://platform.openai.com/docs/api-reference/chat/create#chat_create-messages-system_message
        """
        if not isinstance(data, dict):
            return data
        messages = data.get("messages", [])
        for msg in messages:
            # Check if this is a system message
            if isinstance(msg, dict) and msg.get("role") == "system":
                content = msg.get("content")

                # If content is a list (multimodal format)
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict):
                            part_type = part.get("type")
                            # Infer type when 'type' field is not explicit
                            if part_type is None:
                                if "image_url" in part or "image_pil" in part:
                                    part_type = "image_url"
                                elif "image_embeds" in part:
                                    part_type = "image_embeds"
                                elif "audio_url" in part:
                                    part_type = "audio_url"
                                elif "input_audio" in part:
                                    part_type = "input_audio"
                                elif "audio_embeds" in part:
                                    part_type = "audio_embeds"
                                elif "video_url" in part:
                                    part_type = "video_url"

                            # Warn about non-text content in system messages
                            if part_type and part_type != "text":
                                logger.warning(
                                    "System messages should only contain text "
                                    "content according to the OpenAI API spec. "
                                    "Found content type: '%s'.",
                                    part_type,
                                )

        return data

    @model_validator(mode="before")
    @classmethod
    def set_include_reasoning_for_none_effort(cls, data: Any) -> Any:
        if data.get("reasoning_effort") == "none":
            data["include_reasoning"] = False
        return data


class ImageGenerationRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    n: Optional[int] = Field(default=1, ge=1, le=10)
    image: Optional[list] = None
    size: Optional[Literal["1024x1024","256x256","512x512"]] = "1024x1024"
    quality: Optional[Literal["low", "medium", "high", "auto"]] = "auto"
    background: Optional[str] = "transparent"
    response_format: Optional[Literal["url", "b64_json"]]  = "b64_json"
    user: Optional[str] = "default_user"
    # sd
    steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    seed: Optional[int] = None


#####################  Response  #####################

class PromptTokenUsageInfo(OpenAIBaseModel):
    cached_tokens: Optional[int] = None
    audio_tokens: Optional[int] = None


class CompletinTokenUsageInfo(OpenAIBaseModel):
    accepted_prediction_tokens: Optional[int] = None
    audio_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    rejected_prediction_tokens: Optional[int] = None


class UsageInfo(OpenAIBaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: int = 0
    prompt_tokens_details: Optional[PromptTokenUsageInfo] = None
    completion_tokens_details: Optional[CompletinTokenUsageInfo] = None

class TimingsInfo(OpenAIBaseModel):
    # 输入 (prefill 阶段)
    prompt_n: int = 0
    prompt_ms: float = 0.0
    prompt_per_token_ms: float = 0.0
    prompt_per_second: float = 0.0

    # 输出 (decode 阶段)
    predicted_n: int = 0
    predicted_ms: float = 0.0
    predicted_per_token_ms: float = 0.0
    predicted_per_second: float = 0.0


class FunctionCall(OpenAIBaseModel):
    name: str
    arguments: str


class ToolCall(OpenAIBaseModel):
    id: str = Field(default_factory=make_tool_call_id)
    type: Literal["function"] = "function"
    function: FunctionCall


class DeltaFunctionCall(BaseModel):
    name: str | None = None
    arguments: str | None = None


# a tool call delta where everything is optional
class DeltaToolCall(OpenAIBaseModel):
    id: str | None = None
    type: Literal["function"] | None = None
    index: int
    function: DeltaFunctionCall | None = None


class ChatMessage(OpenAIBaseModel):
    role: str
    content: str | None = None
    refusal: str | None = None
    # annotations: OpenAIAnnotation | None = None
    # audio: OpenAIChatCompletionAudio | None = None
    function_call: FunctionCall | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)


class ChatCompletionLogProb(OpenAIBaseModel):
    token: str
    logprob: float = -9999.0
    bytes: list[int] | None = None


class ChatCompletionLogProbsContent(ChatCompletionLogProb):
    token: str
    logprob: float
    bytes: list[int] | None = None
    top_logprobs: list[ChatCompletionLogProb] = Field(default_factory=list)


class ChatCompletionLogProbs(OpenAIBaseModel):
    content: list[ChatCompletionLogProbsContent] | None = None
    # refusal


class ChatCompletionResponseChoice(OpenAIBaseModel):
    index: int
    message: ChatMessage
    logprobs: ChatCompletionLogProbs | None = None
    finish_reason: str | None = "stop"


class ChatCompletionResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{random_uuid()}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionResponseChoice]
    service_tier: Literal["auto", "default", "flex", "scale", "priority"] | None = "default"
    system_fingerprint: str | None = "deprecated"
    usage: UsageInfo
    timings: TimingsInfo | None = None   # 新增llama.cpp 字段


class DeltaMessage(OpenAIBaseModel):
    role: str | None = None
    content: str | None = None
    tool_calls: list[DeltaToolCall] = Field(default_factory=list)
    refusal: str | None = None


class ChatCompletionResponseStreamChoice(OpenAIBaseModel):
    index: int
    delta: DeltaMessage
    logprobs: ChatCompletionLogProbs | None = None
    finish_reason: str | None = None


class ChatCompletionStreamResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{random_uuid()}")
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionResponseStreamChoice]
    service_tier: Literal["auto", "default", "flex", "scale", "priority"] | None = None
    system_fingerprint: str | None = None
    usage: UsageInfo | None = Field(default=None)
    timings: TimingsInfo | None = None   # 新增llama.cpp 字段


class ImageURL(BaseModel):
    url: str


class ImageB64JSON(BaseModel):
    b64_json: str


#####################  Embedding  #####################
EncodingFormat = Literal["float", "base64", "bytes"]

class EmbeddingRequest(OpenAIBaseModel):
    model: Optional[str] = None
    input: Union[List[int], List[List[int]], str, List[str]]
    encoding_format: EncodingFormat = "float"
    dimensions: Optional[int] = None
    user: Optional[str] = None

    # @model_validator(mode="before")
    # @classmethod
    # def validate_encoding_format(cls, values):
    #     val = values.get("encoding_format")
    #     if val not in ("float", "base64"):
    #         raise ValueError("encoding_format must be 'float' or 'base64'")
    #     return values

    @model_validator(mode="before")
    @classmethod
    def validate_input(cls, values):
        val = values.get("input")
        # 字符串类型直接通过
        if isinstance(val, str):
            return values

        # list[int] 类型
        if all(isinstance(i, int) for i in val):
            return values

        # list[list[int]] 类型
        if all(isinstance(i, list) and all(isinstance(ii, int) for ii in i) for i in val):
            return values

        # list[str] 类型
        if all(isinstance(i, str) for i in val):
            return values

        raise ValueError("input must be str, list[int], list[list[int]] or list[str]")


class EmbeddingResponseData(OpenAIBaseModel):
    index: int
    object: str = "embedding"
    embedding: Union[list[float], str]


class EmbeddingResponse(OpenAIBaseModel):
    object: str = "list"
    model: str
    data: list[EmbeddingResponseData]
    usage: UsageInfo


from dataclasses import dataclass

# We use dataclass for now because it is used for
# openai server output, and msgspec is not serializable.
# TODO(sang): Fix it.
@dataclass
class Logprob:
    """Infos for supporting OpenAI compatible logprobs and token ranks.

    Attributes:
        logprob: The logprob of chosen token
        rank: The vocab rank of chosen token (>=1)
        decoded_token: The decoded chosen token index
    """

    logprob: float
    rank: int | None = None
    decoded_token: str | None = None



