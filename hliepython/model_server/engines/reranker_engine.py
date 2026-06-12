from typing import Any, Dict

from fastapi import Request
from fastapi.responses import Response

from hliepython.utils.logging import async_logger as logger
from hliepython.model_server.engines.base_engine import InferenceEngine
from hliepython.model_server.engines.utils import (
    make_vllm_frontend_namespace,
    merge_additional_config,
    move_vllm_engine_args_from_additional_config,
    split_vllm_config,
)

try:
    from vllm import AsyncLLMEngine
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.entrypoints.chat_utils import ChatTemplateConfig, load_chat_template
    from vllm.entrypoints.logger import RequestLogger
    from vllm.entrypoints.openai.models.protocol import BaseModelPath
    from vllm.entrypoints.openai.models.serving import OpenAIServingModels
    from vllm.entrypoints.pooling.scoring.protocol import RerankRequest, ScoreRequest
    from vllm.entrypoints.pooling.scoring.serving import ServingScores
except ImportError as exc:
    raise ImportError("vLLM 库未安装，RerankerEngine 无法使用本地推理功能！请安装 vLLM 或检查环境配置。") from exc

from vllm_houmo.models.export_meta_from_gguf import parse_gguf


class RerankerEngine(InferenceEngine):
    """Qwen3 reranker 推理引擎（底层使用 vLLM pooling scoring API）。"""

    def __init__(self, instance_config: Dict[str, Any]):
        self.model_name = instance_config.get("model_name", "")
        self.model_path = instance_config.get("model_path", "")
        self.model_type = instance_config.get("model_type", "reranker")
        self.vllm_config = instance_config.get(
            "vllm_config",
            instance_config.get("vllm_extra_config", {}),
        )
        self.additional_config = instance_config.get("additional_config", {}) or {}
        self.vllm_frontend_args = None
        self.vllm_api_server_args: dict[str, Any] = {}
        self.vllm_unknown_args: dict[str, Any] = {}
        self.openai_serving_models = None
        self.serving_scores = None
        super().__init__(instance_config)

    def _load_model(self):
        """加载 reranker 模型（使用 vLLM classify/score pooling）。"""
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
            additional_config = dict(self.additional_config)
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

            parse_gguf(self.model_path)

            engine_arg_kwargs = {
                **vllm_engine_args,
                "model": self.model_path,
                "additional_config": additional_config,
            }

            logger.info(f"RerankerEngine vllm 启动参数: {engine_arg_kwargs}")
            engine_args = AsyncEngineArgs(**engine_arg_kwargs)
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            self._init_vllm_scoring_serving()
            logger.info(f"RerankerEngine: vLLM 引擎加载完成 {self.model_path} 到设备 {self.device_ids}")

        except Exception as e:
            logger.error(f"RerankerEngine: vLLM 引擎加载失败 {e}")
            raise RuntimeError(f"Failed to load vLLM reranker engine: {e}")

    def _init_vllm_scoring_serving(self) -> None:
        """初始化 vLLM score/rerank serving 层，复用当前 worker HTTP server。"""
        frontend_args = self.vllm_frontend_args
        if frontend_args is None:
            frontend_args = make_vllm_frontend_namespace({}, {})

        served_model_names = getattr(frontend_args, "served_model_name", None) or [self.model_name]
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

        chat_template_config = ChatTemplateConfig(
            chat_template=load_chat_template(getattr(frontend_args, "chat_template", None)),
            chat_template_content_format=getattr(frontend_args, "chat_template_content_format", "auto"),
            trust_request_chat_template=bool(getattr(frontend_args, "trust_request_chat_template", False)),
        )
        supported_tasks = tuple(getattr(self.engine.model_config, "supported_tasks", None) or ("classify",))

        self.serving_scores = ServingScores(
            self.engine,
            self.openai_serving_models,
            supported_tasks=supported_tasks,
            request_logger=request_logger,
            chat_template_config=chat_template_config,
            enable_flash_late_interaction=bool(
                getattr(frontend_args, "enable_flash_late_interaction", True)
            ),
        )

    async def create_score(
        self,
        request: ScoreRequest,
        raw_request: Request | None = None,
    ) -> Response:
        if self.serving_scores is None:
            raise RuntimeError("vLLM score serving 未初始化")
        return await self.serving_scores(request, raw_request)

    async def create_rerank(
        self,
        request: RerankRequest,
        raw_request: Request | None = None,
    ) -> Response:
        if self.serving_scores is None:
            raise RuntimeError("vLLM rerank serving 未初始化")
        return await self.serving_scores(request, raw_request)