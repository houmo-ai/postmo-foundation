import os
import io
import json
import uuid
import hashlib
from http import HTTPStatus
from typing import Dict, Any, NoReturn
from fastapi import Request
from abc import ABC, abstractmethod

from hliepython.model_server.engines.protocol.audio_speech import (
    AudioSpeechRequest,
    AudioSpeechResponse,
)
from hliepython.utils.logging import async_logger as logger

from vllm.entrypoints.utils import create_error_response
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse,
)

MASK_64_BITS = (1 << 64) - 1

def random_uuid() -> str:
    return f"{uuid.uuid4().int & MASK_64_BITS:016x}"  # 16 hex chars

class InferenceEngine(ABC):
    """抽象推理引擎接口"""

    def __init__(self, instance_config: Dict[str, Any]):
        self.config = instance_config
        self.model_name = instance_config.get("model_name", "")
        self.additional_config = instance_config.get(
            "additional_config",
            instance_config.get("extra_config", {}),
        ) or {}
        self.device_ids = instance_config.get(
            "device_ids",
            self.additional_config.get("device_ids", []),
        )
        self.model_path = instance_config.get("model_path", "")
        self._setup_environment()
        self._load_model()

    @staticmethod
    def _base_request_id(
        raw_request: Request | None, default: str | None = None
    ) -> str | None:
        """Pulls the request id to use from a header, if provided"""
        if raw_request is not None and (
            (req_id := raw_request.headers.get("X-Request-Id")) is not None
        ):
            return req_id

        return random_uuid() if default is None else default

    @staticmethod
    def serialize_for_fingerprint(obj: Any) -> Any:
        """递归将不可序列化对象转换为可序列化形式"""
        if isinstance(obj, dict):
            return {k: InferenceEngine.serialize_for_fingerprint(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [InferenceEngine.serialize_for_fingerprint(v) for v in obj]
        elif 'PIL' in str(type(obj)):  # 检测 PIL.Image.Image
            buffered = io.BytesIO()
            obj.save(buffered, format="PNG")
            return hashlib.sha256(buffered.getvalue()).hexdigest()
        else:
            return obj

    def _setup_environment(self):
        """设置环境变量，如 GPU 设备"""
        if self.device_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in self.device_ids)
            os.environ["NVIDIA_VISIBLE_DEVICES"] = ",".join(str(d) for d in self.device_ids)

    @abstractmethod
    def _load_model(self):
        """加载模型"""
        pass

    def _unsupported(self, capability: str) -> NoReturn:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support capability: {capability}"
        )

    def create_chat_completions(
        self,
        request: Dict[str, Any],
        raw_request: Request | None = None,
    ) -> Dict[str, Any]:
        """文本聊天完成（子类按需实现）"""
        return self._unsupported("create_chat_completions")

    def create_images_generations(
        self,
        request: Dict[str, Any],
        raw_request: Request | None = None,
    ) -> Dict[str, Any]:
        """图像生成（子类按需实现）"""
        return self._unsupported("create_images_generations")

    def create_audio_speech(
        self,
        request: AudioSpeechRequest,
        raw_request: Request | None = None,
    ) -> AudioSpeechResponse:
        """文本到语音（子类按需实现）"""
        return self._unsupported("create_audio_speech")

    def create_audio_transcriptions(
        self,
        request: Dict[str, Any],
        raw_request: Request | None = None,
    ) -> Dict[str, Any]:
        """语音转写（子类按需实现）"""
        return self._unsupported("create_audio_transcriptions")

    def create_embeddings(
        self,
        request: Dict[str, Any],
        raw_request: Request | None = None,
    ) -> Dict[str, Any]:
        """文本向量化（子类按需实现）"""
        return self._unsupported("create_embeddings")

    def health_check(self) -> bool:
        """健康检查"""
        return True

    def parse_device_ids(self, device_ids: str) -> list[int]:
        if not device_ids:
            return []

        ids = [s.strip() for s in device_ids.split(",") if s.strip() != ""]

        device_ids_list = []
        for s in ids:
            try:
                device_ids_list.append(int(s))
            except ValueError:
                logger.error(f"Invalid device ID: {s} is not an integer")
                raise ValueError(f"Invalid device ID: {s} is not an integer")

        return device_ids_list

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
        message: str | Exception,
        err_type: str = "BadRequestError",
        status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
        param: str | None = None,
    ) -> str:
        json_str = json.dumps(
            self.create_error_response(
                message=message,
                err_type=err_type,
                status_code=status_code,
                param=param,
            ).model_dump()
        )
        return json_str
