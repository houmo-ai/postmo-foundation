from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import importlib.resources

import yaml
from loguru import logger


def _read_config_yaml(config_path: Optional[str] = None) -> Dict[str, Any]:
    if config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    local_path = Path("config.yaml")
    if local_path.exists():
        with local_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    logger.warning("config.yaml not found in current directory, fallback to package resource")
    with importlib.resources.files("hliepython").joinpath("config.yaml").open(
        "r", encoding="utf-8"
    ) as f:
        return yaml.safe_load(f) or {}


def _as_dict(value: Any, field_name: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    return value


def _as_list(value: Any, field_name: str) -> List[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be an array")
    return value


def _as_int(value: Any, default: int, field_name: str) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc


def _normalize_model_root_dir(model_root_dir: Any) -> str:
    return str(Path(str(model_root_dir)).expanduser())


def _resolve_model_path(model_path: Any, model_root_dir: str) -> str:
    normalized_path = str(model_path).strip()
    if not normalized_path:
        raise ValueError("models[].instances[].model_path must not be empty")

    path_obj = Path(normalized_path).expanduser()
    if path_obj.is_absolute():
        return str(path_obj)

    return str(Path(model_root_dir).expanduser() / path_obj)


@dataclass
class ServerConfig:
    main_port: int = 8000
    log_level: str = "info"
    vllm_log_level: str = "info"
    model_root_dir: str = "/houmo/llm/models"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ServerConfig":
        cfg = _as_dict(data, "server")
        return cls(
            main_port=_as_int(cfg.get("main_port"), 8000, "server.main_port"),
            log_level=str(cfg.get("log_level", "info")).lower(),
            vllm_log_level=str(cfg.get("vllm_log_level", "info")).lower(),
            model_root_dir=_normalize_model_root_dir(
                cfg.get("model_root_dir", "/houmo/llm/models")
            ),
        )


@dataclass
class ModelInstanceConfig:
    port: int
    model_path: str
    model_type: str = "text"
    device_ids: List[int] = field(default_factory=list)
    vllm_config: Dict[str, Any] = field(default_factory=dict)
    additional_config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        model_name: str,
        model_root_dir: str,
    ) -> "ModelInstanceConfig":
        inst = _as_dict(data, f"models[].instances[] ({model_name})")
        if "port" not in inst:
            raise ValueError(f"models[].instances[].port is required ({model_name})")
        if "model_path" not in inst:
            raise ValueError(f"models[].instances[].model_path is required ({model_name})")

        additional_config = _as_dict(
            inst.get("additional_config", inst.get("extra_config", {})),
            "models[].instances[].additional_config",
        )
        if "enable_cacheprompt" in additional_config and not isinstance(
            additional_config["enable_cacheprompt"], str
        ):
            raise ValueError("models[].instances[].additional_config.enable_cacheprompt must be a string")
        additional_config.setdefault("enable_cacheprompt", "")

        raw_device_ids = _as_list(
            additional_config.get("device_ids", inst.get("device_ids", [])),
            "models[].instances[].additional_config.device_ids",
        )
        device_ids: List[int] = [
            _as_int(device_id, 0, "models[].instances[].additional_config.device_ids[]")
            for device_id in raw_device_ids
        ]
        additional_config["device_ids"] = device_ids

        return cls(
            port=_as_int(inst.get("port"), 0, "models[].instances[].port"),
            model_path=_resolve_model_path(inst.get("model_path"), model_root_dir),
            model_type=str(inst.get("model_type", "text")),
            device_ids=device_ids,
            vllm_config=_as_dict(
                inst.get("vllm_config", inst.get("vllm_extra_config", {})),
                "models[].instances[].vllm_config",
            ),
            additional_config=additional_config,
        )


@dataclass
class ModelConfig:
    name: str
    model_type: str = "text"
    instances: List[ModelInstanceConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], model_root_dir: str) -> "ModelConfig":
        model = _as_dict(data, "models[]")
        name = str(model.get("name", "")).strip()
        if not name:
            raise ValueError("models[].name is required")

        instances_raw = _as_list(model.get("instances", []), f"models[].instances ({name})")
        instances = [
            ModelInstanceConfig.from_dict(
                inst,
                model_name=name,
                model_root_dir=model_root_dir,
            )
            for inst in instances_raw
        ]

        return cls(
            name=name,
            model_type=str(model.get("model_type", "text")),
            instances=instances,
        )


@dataclass
class AppConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    preload_models_on_startup: bool = True
    models: List[ModelConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        cfg = _as_dict(data, "root")
        server = ServerConfig.from_dict(cfg.get("server", {}))

        models_raw = _as_list(cfg.get("models", []), "models")
        models = [
            ModelConfig.from_dict(model_raw, model_root_dir=server.model_root_dir)
            for model_raw in models_raw
        ]

        return cls(
            server=server,
            preload_models_on_startup=bool(cfg.get("preload_models_on_startup", True)),
            models=models,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_app_config(config_path: Optional[str] = None) -> AppConfig:
    raw_cfg = _read_config_yaml(config_path=config_path)
    return AppConfig.from_dict(raw_cfg)
