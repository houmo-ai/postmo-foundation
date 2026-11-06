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

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import importlib.resources
import os

import yaml
from hliepython.config.utils import (
    as_bool,
    as_dict,
    as_int,
    as_list,
    normalize_model_root_dir,
    resolve_model_file_path,
    resolve_model_path,
)
from hliepython.utils.logging import async_logger as logger


DEFAULT_MAIN_PORT = 7800
DEFAULT_LOG_LEVEL = "info"
DEFAULT_VLLM_LOG_LEVEL = "warning"
DEFAULT_MODEL_ROOT_DIR = "/opt/houmo/models"

ENV_MAIN_PORT = "MAIN_PORT"
ENV_LOG_LEVEL = "LOG_LEVEL"
ENV_VLLM_LOG_LEVEL = "VLLM_LOG_LEVEL"
ENV_MODEL_ROOT_DIR = "MODEL_ROOT_DIR"
SUPPORTED_ENV_KEYS = frozenset(
    {
        ENV_MAIN_PORT,
        ENV_LOG_LEVEL,
        ENV_VLLM_LOG_LEVEL,
        ENV_MODEL_ROOT_DIR,
    }
)

@dataclass
class EnvConfig:
    main_port: int = DEFAULT_MAIN_PORT
    log_level: str = DEFAULT_LOG_LEVEL
    vllm_log_level: str = DEFAULT_VLLM_LOG_LEVEL
    model_root_dir: str = DEFAULT_MODEL_ROOT_DIR
    loaded_env_path: Optional[str] = None
    unsupported_keys: List[str] = field(default_factory=list)

    @staticmethod
    def _parse_dotenv_line(line: str) -> Optional[Tuple[str, str]]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            return None
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].lstrip()
        if "=" not in stripped:
            return None

        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            return None
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        return key, value

    @classmethod
    def load(cls, env_path: Optional[str] = None) -> "EnvConfig":
        dotenv_path = Path(env_path) if env_path else Path(".env")
        dotenv_values: Dict[str, str] = {}
        unsupported_keys: List[str] = []

        if dotenv_path.exists():
            with dotenv_path.open("r", encoding="utf-8") as f:
                for line in f:
                    parsed = cls._parse_dotenv_line(line)
                    if parsed is None:
                        continue

                    key, value = parsed
                    if key not in SUPPORTED_ENV_KEYS:
                        unsupported_keys.append(key)
                        continue

                    dotenv_values[key] = value
                    os.environ.setdefault(key, value)

        if unsupported_keys:
            logger.warning(
                f".env 中存在程序不支持的配置项，已忽略: {sorted(unsupported_keys)}"
            )

        def pick(env_key: str, default: str) -> str:
            value = os.getenv(env_key)
            if value not in (None, ""):
                return value
            value = dotenv_values.get(env_key)
            if value not in (None, ""):
                return value
            return default

        return cls(
            main_port=as_int(pick(ENV_MAIN_PORT, str(DEFAULT_MAIN_PORT)), DEFAULT_MAIN_PORT, ENV_MAIN_PORT),
            log_level=pick(ENV_LOG_LEVEL, DEFAULT_LOG_LEVEL).lower(),
            vllm_log_level=pick(ENV_VLLM_LOG_LEVEL, DEFAULT_VLLM_LOG_LEVEL).lower(),
            model_root_dir=normalize_model_root_dir(pick(ENV_MODEL_ROOT_DIR, DEFAULT_MODEL_ROOT_DIR)),
            loaded_env_path=str(dotenv_path) if dotenv_path.exists() else None,
            unsupported_keys=unsupported_keys,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


ENV_CONFIG = EnvConfig()


def _env_first(env_value: Any, cfg: Dict[str, Any], cfg_key: str, default: Any) -> Any:
    if env_value not in (None, ""):
        return env_value
    return cfg.get(cfg_key, default)


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


def _resolve_nested_model_refs(value: Any, model_root_dir: str) -> Any:
    if isinstance(value, dict):
        resolved: Dict[str, Any] = {}
        for key, item in value.items():
            if key == "model" and isinstance(item, str):
                resolved[key] = resolve_model_path(item, model_root_dir)
            else:
                resolved[key] = _resolve_nested_model_refs(item, model_root_dir)
        return resolved
    if isinstance(value, list):
        return [_resolve_nested_model_refs(item, model_root_dir) for item in value]
    return value


def _resolve_vllm_config_model_refs(
    vllm_config: Dict[str, Any],
    model_root_dir: str,
) -> Dict[str, Any]:
    return _resolve_nested_model_refs(vllm_config, model_root_dir)


@dataclass
class LoggingConfig:
    level: str = "INFO"
    directory: str = "logs"
    sync_enabled: bool = True
    sync_level: str = "INFO"
    sync_filename: str = "sync.log"
    console_enabled: bool = True
    rotation: str = "100 MB"
    retention: str = "14 days"
    compression: Optional[str] = None
    serialize: bool = False
    backtrace: bool = False
    diagnose: bool = False

    @classmethod
    def from_dict(
        cls,
        data: Optional[Dict[str, Any]],
        legacy_level: str = "info",
    ) -> "LoggingConfig":
        cfg = as_dict(data or {}, "server")
        legacy_logging = as_dict(cfg.get("logging", {}), "server.logging")

        def pick(key: str, legacy_key: Optional[str] = None, default: Any = None) -> Any:
            if key in cfg:
                return cfg.get(key)
            lookup_key = legacy_key or key
            if lookup_key in legacy_logging:
                return legacy_logging.get(lookup_key)
            return default

        level = str(_env_first(ENV_CONFIG.log_level, cfg, "log_level", pick("level", default=legacy_level or DEFAULT_LOG_LEVEL))).upper()
        sync_level = str(pick("sync_log_level", default=level)).upper()
        return cls(
            level=level,
            directory=str(pick("log_dir", "directory", "logs")),
            sync_enabled=as_bool(pick("sync_log_enabled", default=True), True),
            sync_level=sync_level,
            sync_filename=str(pick("sync_log_filename", "sync_filename", "sync.log")),
            console_enabled=as_bool(pick("console_log_enabled", "console_enabled", True), True),
            rotation=str(pick("log_rotation", "rotation", "100 MB")),
            retention=str(pick("log_retention", "retention", "14 days")),
            compression=(
                str(pick("log_compression", "compression"))
                if pick("log_compression", "compression") not in (None, "")
                else None
            ),
            serialize=as_bool(pick("log_serialize", "serialize", False), False),
            backtrace=as_bool(pick("log_backtrace", "backtrace", False), False),
            diagnose=as_bool(pick("log_diagnose", "diagnose", False), False),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ServerConfig:
    main_port: int = DEFAULT_MAIN_PORT
    log_level: str = DEFAULT_LOG_LEVEL
    vllm_log_level: str = DEFAULT_VLLM_LOG_LEVEL
    model_root_dir: str = DEFAULT_MODEL_ROOT_DIR
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ServerConfig":
        cfg = as_dict(data, "server")
        logging_config = LoggingConfig.from_dict(
            cfg,
            legacy_level=str(cfg.get("log_level", "info")),
        )
        return cls(
            main_port=as_int(
                _env_first(ENV_CONFIG.main_port, cfg, "main_port", DEFAULT_MAIN_PORT),
                DEFAULT_MAIN_PORT,
                "server.main_port",
            ),
            log_level=logging_config.level.lower(),
            vllm_log_level=str(
                _env_first(ENV_CONFIG.vllm_log_level, cfg, "vllm_log_level", DEFAULT_VLLM_LOG_LEVEL)
            ).lower(),
            model_root_dir=normalize_model_root_dir(
                _env_first(ENV_CONFIG.model_root_dir, cfg, "model_root_dir", DEFAULT_MODEL_ROOT_DIR)
            ),
            logging=logging_config,
        )


@dataclass
class ModelInstanceConfig:
    port: int
    model_path: str
    model_type: str = "llm"
    device_ids: List[int] = field(default_factory=list)
    vllm_config: Dict[str, Any] = field(default_factory=dict)
    additional_config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        model_name: str,
        model_root_dir: str,
        parent_model_path: Optional[str] = None,
        parent_model_type: str = "llm",
    ) -> "ModelInstanceConfig":
        inst = as_dict(data, f"models[].instances[] ({model_name})")
        vllm_config = as_dict(
            inst.get("vllm_config", inst.get("vllm_extra_config", {})),
            "models[].instances[].vllm_config",
        )

        model_path = inst.get("model_path", parent_model_path)
        if model_path in (None, ""):
            raise ValueError(f"models[].model_path is required ({model_name})")

        port = inst.get("port")
        if port is None:
            port = cls._extract_port_from_vllm_config(vllm_config)
        if port is None:
            raise ValueError(
                f"models[].instances[].vllm_config.api-server.port is required ({model_name})"
            )

        additional_config = as_dict(
            inst.get("additional_config", inst.get("extra_config", {})),
            "models[].instances[].additional_config",
        )
        if "enable_cacheprompt" in additional_config and not isinstance(
            additional_config["enable_cacheprompt"], str
        ):
            raise ValueError("models[].instances[].additional_config.enable_cacheprompt must be a string")
        additional_config.setdefault("enable_cacheprompt", "")

        raw_device_ids = as_list(
            additional_config.get("device_ids", inst.get("device_ids", [])),
            "models[].instances[].additional_config.device_ids",
        )
        device_ids: List[int] = [
            as_int(device_id, 0, "models[].instances[].additional_config.device_ids[]")
            for device_id in raw_device_ids
        ]
        additional_config["device_ids"] = device_ids
        additional_config = cls._resolve_additional_config_model_refs(
            additional_config,
            model_root_dir,
        )

        return cls(
            port=as_int(port, 0, "models[].instances[].vllm_config.api-server.port"),
            model_path=resolve_model_path(model_path, model_root_dir),
            model_type=str(inst.get("model_type", parent_model_type or "llm")),
            device_ids=device_ids,
            vllm_config=_resolve_vllm_config_model_refs(vllm_config, model_root_dir),
            additional_config=additional_config,
        )

    @staticmethod
    def _extract_port_from_vllm_config(vllm_config: Dict[str, Any]) -> Optional[Any]:
        for section_key in ("api-server", "api_server", "api_server_args", "server"):
            section = vllm_config.get(section_key)
            if isinstance(section, dict) and section.get("port") is not None:
                return section.get("port")
        return None

    @staticmethod
    def _resolve_additional_config_model_refs(
        additional_config: Dict[str, Any],
        model_root_dir: str,
    ) -> Dict[str, Any]:
        resolved = dict(additional_config)
        vit_model_name = resolved.get("vit_model_name")
        if isinstance(vit_model_name, str) and vit_model_name.strip():
            resolved["vit_model_name"] = resolve_model_file_path(
                vit_model_name,
                model_root_dir,
            )
        return resolved


@dataclass
class ModelConfig:
    name: str
    model_type: str = "llm"
    instances: List[ModelInstanceConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], model_root_dir: str) -> "ModelConfig":
        model = as_dict(data, "models[]")
        model_path = model.get("model_path")
        model_type = str(model.get("model_type", "llm"))
        name = str(model.get("name") or model_path or "").strip()
        if not name:
            raise ValueError("models[].model_path is required")

        instances_raw = as_list(model.get("instances", []), f"models[].instances ({name})")
        instances = [
            ModelInstanceConfig.from_dict(
                inst,
                model_name=name,
                model_root_dir=model_root_dir,
                parent_model_path=model_path,
                parent_model_type=model_type,
            )
            for inst in instances_raw
        ]

        return cls(
            name=name,
            model_type=model_type,
            instances=instances,
        )


@dataclass
class AppConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    preload_models_on_startup: bool = True
    models: List[ModelConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        cfg = as_dict(data, "root")
        server = ServerConfig.from_dict(cfg.get("server", {}))

        models_raw = as_list(cfg.get("models", []), "models")
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


def load_app_config(config_path: Optional[str] = None, env_path: Optional[str] = None) -> AppConfig:
    loaded_env_config = EnvConfig.load(env_path=env_path)
    ENV_CONFIG.__dict__.update(loaded_env_config.__dict__)
    raw_cfg = _read_config_yaml(config_path=config_path)
    return AppConfig.from_dict(raw_cfg)
