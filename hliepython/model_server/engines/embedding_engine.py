import asyncio
import inspect
import time
from http import HTTPStatus
from typing import Any, Dict

from fastapi import Request
from hliepython.utils.logging import async_logger as logger

from hliepython.model_server.engines.base_engine import InferenceEngine
from hliepython.model_server.engines.protocol.chat_completions import ErrorResponse, UsageInfo
from hliepython.model_server.engines.protocol.embeddings import (
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingResponseData,
    encode_embedding_base64,
)
from hliepython.model_server.engines.utils import (
    merge_additional_config,
    move_vllm_engine_args_from_additional_config,
    normalize_vllm_engine_args,
)
from hliepython.model_server.utils.gguf_metadata import apply_gguf_metadata_defaults_to_vllm_engine_args

try:
    from vllm import AsyncLLMEngine, PoolingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.exceptions import EngineDeadError
except ImportError:
    # logger.error("vLLM 库未安装，EmbeddingEngine 无法使用本地推理功能！请安装 vLLM 或检查环境配置。")
    raise ImportError("vLLM 库未安装，EmbeddingEngine 无法使用本地推理功能！请安装 vLLM 或检查环境配置。")

from vllm_houmo.models.export_meta_from_gguf import parse_gguf


class EmbeddingEngine(InferenceEngine):
    """OpenAI embeddings 推理引擎（底层使用 vLLM）。"""

    def __init__(self, instance_config: Dict[str, Any]):
        self.model_name = instance_config.get("model_name", "")
        self.model_path = instance_config.get("model_path", "")
        self.model_type = instance_config.get("model_type", "embedding")
        self.vllm_config = instance_config.get(
            "vllm_config",
            instance_config.get("vllm_extra_config", {}),
        )
        self.additional_config = instance_config.get("additional_config", {}) or {}
        super().__init__(instance_config)

    @staticmethod
    def _set_supported_engine_default(
        engine_args: dict[str, Any],
        key: str,
        value: Any,
    ):
        """仅在当前 vLLM AsyncEngineArgs 支持某参数时设置默认值。"""
        try:
            if key in inspect.signature(AsyncEngineArgs).parameters:
                engine_args.setdefault(key, value)
        except (TypeError, ValueError):
            pass

    def _load_model(self):
        """加载 embedding 模型（使用 vLLM encode/pooling）。"""
        try:
            vllm_engine_args = normalize_vllm_engine_args(self.vllm_config)
            user_additional_config = vllm_engine_args.pop("additional_config", None)
            additional_config = merge_additional_config(
                dict(self.additional_config),
                user_additional_config,
            )
            additional_config.setdefault("device_ids", self.device_ids)
            vllm_engine_args, additional_config = move_vllm_engine_args_from_additional_config(
                vllm_engine_args,
                additional_config,
            )
            vllm_engine_args = apply_gguf_metadata_defaults_to_vllm_engine_args(
                self.model_path,
                vllm_engine_args,
            )

            # vLLM 新版本通过 runner/convert 指定 pooling/embed；部分旧版本
            # 不支持这些参数，因此仅在当前 AsyncEngineArgs 签名支持时设置。
            self._set_supported_engine_default(vllm_engine_args, "runner", "pooling")
            self._set_supported_engine_default(vllm_engine_args, "convert", "embed")

            parse_gguf(self.model_path)

            engine_arg_kwargs = {
                **vllm_engine_args,
                "model": self.model_path,
                "additional_config": additional_config,
            }

            # logger.info(f"EmbeddingEngine vllm 启动参数: {engine_arg_kwargs}")
            engine_args = AsyncEngineArgs(**engine_arg_kwargs)
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            # logger.info(f"EmbeddingEngine: vLLM 引擎加载完成 {self.model_path} 到设备 {self.device_ids}")

        except Exception as e:
            # logger.error(f"EmbeddingEngine: vLLM 引擎加载失败 {e}")
            raise RuntimeError(f"Failed to load vLLM embedding engine: {e}")

    def create_error_response(
        self,
        message: str,
        err_type: str = "BadRequestError",
        status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
    ) -> ErrorResponse:
        return ErrorResponse(
            message=message,
            type=err_type,
            code=status_code.value,
        )

    def _normalize_inputs(self, request_input: Any) -> list[dict[str, Any]]:
        if isinstance(request_input, str):
            return [{"prompt": request_input}]

        if isinstance(request_input, list) and all(isinstance(item, str) for item in request_input):
            return [{"prompt": item} for item in request_input]

        if isinstance(request_input, list) and all(isinstance(item, int) for item in request_input):
            return [{"prompt_token_ids": request_input}]

        if isinstance(request_input, list) and all(isinstance(item, list) for item in request_input):
            return [{"prompt_token_ids": item} for item in request_input]

        raise ValueError("input must be str, list[str], list[int] or list[list[int]]")

    def _create_pooling_params(self, request: EmbeddingRequest) -> PoolingParams:
        pooling_params_config = dict(self.additional_config.get("pooling_params", {}) or {})
        if request.dimensions is not None:
            try:
                if "dimensions" in inspect.signature(PoolingParams).parameters:
                    pooling_params_config.setdefault("dimensions", request.dimensions)
            except (TypeError, ValueError):
                pass
        try:
            if "task" in inspect.signature(PoolingParams).parameters:
                pooling_params_config.setdefault("task", "embed")
        except (TypeError, ValueError):
            pass
        return PoolingParams(**pooling_params_config)

    async def _collect_encode_output(
        self,
        prompt_input: dict[str, Any],
        pooling_params: PoolingParams,
        request_id: str,
    ):
        encode_fn = self.engine.encode
        try:
            result = encode_fn(
                inputs=prompt_input,
                pooling_params=pooling_params,
                request_id=request_id,
            )
        except TypeError:
            result = encode_fn(prompt_input, pooling_params, request_id)

        if inspect.isawaitable(result):
            result = await result

        if hasattr(result, "__aiter__"):
            final_output = None
            async for output in result:
                final_output = output
            return final_output

        if isinstance(result, list):
            return result[-1] if result else None

        return result

    def _embedding_to_float_list(self, embedding_data: Any) -> list[float]:
        if hasattr(embedding_data, "detach"):
            embedding_data = embedding_data.detach().cpu()
        if hasattr(embedding_data, "numpy"):
            embedding_data = embedding_data.numpy()
        if hasattr(embedding_data, "tolist"):
            embedding_data = embedding_data.tolist()

        if (
            isinstance(embedding_data, list)
            and len(embedding_data) == 1
            and isinstance(embedding_data[0], list)
        ):
            embedding_data = embedding_data[0]

        if not isinstance(embedding_data, list):
            raise ValueError("vLLM embedding output is not a vector")

        return [float(value) for value in embedding_data]

    def _extract_embedding_data(self, output: Any) -> Any:
        outputs = getattr(output, "outputs", None)
        if outputs is None:
            raise ValueError("vLLM embedding output has no outputs field")

        if hasattr(outputs, "data"):
            return outputs.data
        if hasattr(outputs, "embedding"):
            return outputs.embedding
        if isinstance(outputs, dict):
            if "data" in outputs:
                return outputs["data"]
            if "embedding" in outputs:
                return outputs["embedding"]

        raise ValueError("vLLM embedding output has no data field")

    def _build_response(
        self,
        request: EmbeddingRequest,
        outputs: list[tuple[int, Any]],
    ) -> EmbeddingResponse:
        data_list: list[EmbeddingResponseData] = []
        total_tokens = 0

        for index, output in outputs:
            embedding_vector = self._embedding_to_float_list(
                self._extract_embedding_data(output)
            )
            if request.dimensions is not None:
                embedding_vector = embedding_vector[: request.dimensions]

            if request.encoding_format == "base64":
                embedding_out = encode_embedding_base64(embedding_vector)
            else:
                embedding_out = embedding_vector

            prompt_token_ids = getattr(output, "prompt_token_ids", None) or []
            total_tokens += len(prompt_token_ids)

            data_list.append(
                EmbeddingResponseData(
                    index=index,
                    object="embedding",
                    embedding=embedding_out,
                )
            )

        data_list.sort(key=lambda item: item.index)

        return EmbeddingResponse(
            model=request.model or self.model_name,
            data=data_list,
            usage=UsageInfo(
                prompt_tokens=total_tokens,
                total_tokens=total_tokens,
            ),
        )

    async def create_embeddings(
        self,
        request: EmbeddingRequest,
        raw_request: Request | None = None,
    ) -> EmbeddingResponse | ErrorResponse:
        request_started_at = self._perf_counter()
        perf_stages: list[dict[str, Any]] = []
        try:
            normalize_t0 = self._perf_counter()
            prompt_inputs = self._normalize_inputs(request.input)
            pooling_params = self._create_pooling_params(request)
            base_request_id = self._base_request_id(raw_request)
            perf_stages.append({
                "stage": "input_prepare",
                "duration_s": self._perf_counter() - normalize_t0,
                "note": "normalize inputs + pooling params",
            })

            encode_t0 = self._perf_counter()
            tasks = [
                self._collect_encode_output(
                    prompt_input=prompt_input,
                    pooling_params=pooling_params,
                    request_id=f"{base_request_id}-{index}",
                )
                for index, prompt_input in enumerate(prompt_inputs)
            ]
            encode_outputs = await asyncio.gather(*tasks)
            perf_stages.append({
                "stage": "model_encode",
                "duration_s": self._perf_counter() - encode_t0,
                "note": "run embedding encode requests",
            })

            outputs = []
            for index, output in enumerate(encode_outputs):
                if output is None:
                    raise ValueError(f"No embedding generated for input index {index}")
                outputs.append((index, output))

            response_build_t0 = self._perf_counter()
            response = self._build_response(request, outputs)
            perf_stages.append({
                "stage": "response_build",
                "duration_s": self._perf_counter() - response_build_t0,
                "note": "assemble embedding response",
            })
            self._log_perf_table(
                title="Embedding inference",
                request_id=base_request_id,
                stage_timings=perf_stages,
                extra_metrics={
                    "input_count": len(prompt_inputs),
                    "prompt_tokens": response.usage.prompt_tokens,
                    "total_tokens": response.usage.total_tokens,
                    "encoding_format": request.encoding_format,
                },
                total_s=self._perf_counter() - request_started_at,
            )

            return response

        except EngineDeadError:
            # logger.critical("vLLM engine is dead")
            raise
        except ValueError as e:
            return self.create_error_response(str(e))
