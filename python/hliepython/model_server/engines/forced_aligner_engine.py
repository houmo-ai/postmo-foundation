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

import os
from http import HTTPStatus
from typing import Any, Dict

from fastapi import Request
from hliepython.utils.logging import async_logger as logger

from hliepython.model_server.engines.base_engine import InferenceEngine
from hliepython.model_server.engines.utils import (
    make_vllm_frontend_namespace,
    merge_additional_config,
    move_vllm_engine_args_from_additional_config,
    split_vllm_config,
)
from hliepython.model_server.utils.gguf_metadata import (
    apply_gguf_metadata_defaults_to_vllm_engine_args,
)

from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.speech_to_text.transcription.protocol import TranscriptionRequest
from vllm.entrypoints.utils import create_error_response
from vllm_houmo.forced_aligner_router.serving import OpenAIForcedAlignTranscription
from vllm_houmo.models.export_meta_from_gguf import parse_gguf


os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


class ForcedAlignerEngine(InferenceEngine):
    """Qwen3 ForcedAligner 推理引擎。

    该模型是 pooling/token_classify 架构，不是 Whisper 风格生成式 ASR，
    因此不能复用 ASREngine 的 token 解码和 timestamp 逻辑。
    """

    TARGET_ARCHITECTURE = "Qwen3ASRForcedAlignerForTokenClassification"

    def __init__(self, instance_config: Dict[str, Any]):
        self.vllm_config = instance_config.get(
            "vllm_config",
            instance_config.get("vllm_extra_config", {}),
        )
        self.vllm_frontend_args = None
        self.vllm_api_server_args: dict[str, Any] = {}
        self.vllm_unknown_args: dict[str, Any] = {}
        self.engine: AsyncLLMEngine | None = None
        self.serving: OpenAIForcedAlignTranscription | None = None
        super().__init__(instance_config)

    def _load_model(self):
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

            forced_defaults: dict[str, Any] = {
                "load_format": "dummy",
                "hf_overrides": {
                    "architectures": [self.TARGET_ARCHITECTURE],
                },
                "enforce_eager": True,
                "runner": "pooling",
                "max_model_len": 411,
            }
            for key, value in forced_defaults.items():
                vllm_engine_args.setdefault(key, value)

            vllm_engine_args = apply_gguf_metadata_defaults_to_vllm_engine_args(
                self.model_path,
                vllm_engine_args,
            )
            # 部分 vllm-houmo runner 依赖全局 GGUF 元信息缓存。
            parse_gguf(self.model_path)

            engine_arg_kwargs = {
                **vllm_engine_args,
                "model": self.model_path,
                "additional_config": additional_config,
            }
            logger.info(f"ForcedAligner vllm 启动参数: {engine_arg_kwargs}")
            logger.info(f"ForcedAligner vllm serving/frontend 参数: {vllm_frontend_args}")
            engine_args = AsyncEngineArgs(**engine_arg_kwargs)
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            self.model_config = self.engine.model_config
            self.serving = OpenAIForcedAlignTranscription(
                self.engine,
                self.model_config,
            )
            logger.info(f"ForcedAlignerEngine: vLLM 引擎加载完成 {self.model_path}")
            logger.info(f"vllm 实际启动参数：{self.engine.vllm_config}")
        except Exception as e:
            logger.error(f"ForcedAlignerEngine: vLLM 引擎加载失败 {e}")
            raise RuntimeError(f"Failed to load ForcedAligner vLLM engine: {e}") from e

    @staticmethod
    def create_error_response(
        message: str | Exception,
        err_type: str = "BadRequestError",
        status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
        param: str | None = None,
    ) -> ErrorResponse:
        return create_error_response(message, err_type, status_code, param)

    async def create_audio_transcriptions(
        self,
        request: TranscriptionRequest,
        raw_request: Request | None = None,
    ):
        if self.serving is None:
            raise RuntimeError("ForcedAligner serving is not initialized")
        audio_data = await request.file.read()
        return await self.serving.create_transcription(
            audio_data=audio_data,
            request=request,
            raw_request=raw_request,
        )