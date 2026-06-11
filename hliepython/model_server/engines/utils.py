import json
import inspect
from argparse import Namespace
from typing import Any

from hliepython.model_server.engines.protocol.chat_completions import StreamOptions


_VLLM_JSON_OBJECT_ARGS = {"speculative_config"}
_VLLM_ENGINE_ARGS_IN_ADDITIONAL_CONFIG = {"speculative_config"}
_VLLM_ENGINE_SECTION_KEYS = {"engine", "engine_args", "async_engine_args"}
_VLLM_FRONTEND_SECTION_KEYS = {"serving", "frontend", "frontend_args", "chat_serving"}
_VLLM_API_SERVER_SECTION_KEYS = {"server", "api_server", "api_server_args"}


def _signature_param_names(obj: Any) -> set[str]:
    try:
        return set(inspect.signature(obj).parameters)
    except (TypeError, ValueError):
        return set()


def _load_vllm_frontend_arg_names() -> tuple[set[str], set[str]]:
    try:
        from vllm.entrypoints.openai.cli_args import BaseFrontendArgs, FrontendArgs
    except Exception:
        base_args = {
            "chat_template",
            "chat_template_content_format",
            "trust_request_chat_template",
            "default_chat_template_kwargs",
            "response_role",
            "return_tokens_as_token_ids",
            "enable_auto_tool_choice",
            "exclude_tools_when_tool_choice_none",
            "tool_call_parser",
            "tool_parser_plugin",
            "tool_server",
            "max_log_len",
            "enable_prompt_tokens_details",
            "enable_server_load_tracking",
            "enable_force_include_usage",
            "enable_tokenizer_info_endpoint",
            "enable_log_outputs",
            "enable_log_deltas",
            "log_error_stack",
            "tokens_only",
            "fingerprint_mode",
            "fingerprint_value",
        }
        api_server_args = {
            "host",
            "port",
            "api_key",
            "allowed_origins",
            "allowed_methods",
            "allowed_headers",
            "uvicorn_log_level",
            "root_path",
            "ssl_keyfile",
            "ssl_certfile",
            "ssl_ca_certs",
            "ssl_cert_reqs",
            "disable_fastapi_docs",
        }
        return base_args, base_args | api_server_args

    base_args = _signature_param_names(BaseFrontendArgs)
    return base_args, base_args | _signature_param_names(FrontendArgs)


def _load_vllm_engine_arg_names() -> set[str]:
    try:
        from vllm.engine.arg_utils import AsyncEngineArgs
    except Exception:
        return set()

    return _signature_param_names(AsyncEngineArgs)


_VLLM_FRONTEND_ARG_NAMES, _VLLM_OPENAI_ARG_NAMES = _load_vllm_frontend_arg_names()
_VLLM_API_SERVER_ARG_NAMES = _VLLM_OPENAI_ARG_NAMES - _VLLM_FRONTEND_ARG_NAMES
_VLLM_ENGINE_ARG_NAMES = _load_vllm_engine_arg_names()


def _normalize_config_key(raw_key: str, config_name: str) -> str:
    if not isinstance(raw_key, str):
        raise ValueError(f"{config_name} keys must be strings")

    key = raw_key.strip().lstrip("-").replace("-", "_")
    if not key:
        raise ValueError(f"{config_name} keys must not be empty")
    return key


def _parse_json_object_arg(key: str, value, config_name: str):
    if key not in _VLLM_JSON_OBJECT_ARGS or not isinstance(value, str):
        return value

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{config_name}.{key} must be a JSON object string: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError(f"{config_name}.{key} must be a JSON object")
    return parsed


def normalize_vllm_config(vllm_config: dict | None) -> dict:
    """归一化 vLLM 配置 key 和 JSON 对象参数，但不做层级分流。"""
    if not vllm_config:
        return {}

    if not isinstance(vllm_config, dict):
        raise ValueError("vllm_config must be an object")

    normalized = {}
    for raw_key, value in vllm_config.items():
        key = _normalize_config_key(raw_key, "vllm_config")
        value = _parse_json_object_arg(key, value, "vllm_config")

        if key.startswith("no_"):
            normalized[key[3:]] = False if value is None or bool(value) else True
        else:
            normalized[key] = value

    return normalized


def _put_explicit_layer(
    source: dict,
    target: dict,
    allowed_names: set[str],
    unknown_args: dict,
    layer_name: str,
    accept_all: bool = False,
) -> None:
    normalized = normalize_vllm_config(source)
    for key, value in normalized.items():
        if accept_all or key in allowed_names:
            target[key] = value
        else:
            unknown_args[f"{layer_name}.{key}"] = value


def split_vllm_config(vllm_config: dict | None) -> tuple[dict, dict, dict, dict]:
    """把统一 vllm_config 自动拆成 engine/frontend/api_server/unknown 四层。

    - engine：AsyncEngineArgs 参数，直接用于创建 vLLM 引擎。
    - frontend：vLLM OpenAI chat serving 层参数。
    - api_server：vLLM OpenAI API server HTTP 层参数，当前项目保留但不再启动第二个 HTTP server。
    - unknown：当前 vLLM 版本无法识别的参数，保留到 frontend 命名空间，避免
      误传给 AsyncEngineArgs 导致启动失败。
    """
    engine_args: dict[str, Any] = {}
    frontend_args: dict[str, Any] = {}
    api_server_args: dict[str, Any] = {}
    unknown_args: dict[str, Any] = {}

    if not vllm_config:
        return engine_args, frontend_args, api_server_args, unknown_args

    if not isinstance(vllm_config, dict):
        raise ValueError("vllm_config must be an object")

    flat_config: dict[str, Any] = {}
    for raw_key, value in vllm_config.items():
        key = _normalize_config_key(raw_key, "vllm_config")
        if key in _VLLM_ENGINE_SECTION_KEYS:
            if not isinstance(value, dict):
                raise ValueError(f"vllm_config.{key} must be an object")
            _put_explicit_layer(value, engine_args, _VLLM_ENGINE_ARG_NAMES, unknown_args, key)
        elif key in _VLLM_FRONTEND_SECTION_KEYS:
            if not isinstance(value, dict):
                raise ValueError(f"vllm_config.{key} must be an object")
            _put_explicit_layer(
                value,
                frontend_args,
                _VLLM_FRONTEND_ARG_NAMES,
                unknown_args,
                key,
                accept_all=True,
            )
        elif key in _VLLM_API_SERVER_SECTION_KEYS:
            if not isinstance(value, dict):
                raise ValueError(f"vllm_config.{key} must be an object")
            _put_explicit_layer(
                value,
                api_server_args,
                _VLLM_API_SERVER_ARG_NAMES,
                unknown_args,
                key,
                accept_all=True,
            )
        else:
            flat_config[raw_key] = value

    normalized = normalize_vllm_config(flat_config)

    for key, value in normalized.items():
        if key in _VLLM_ENGINE_ARG_NAMES:
            engine_args[key] = value
        elif key in _VLLM_FRONTEND_ARG_NAMES:
            frontend_args[key] = value
        elif key in _VLLM_API_SERVER_ARG_NAMES:
            api_server_args[key] = value
        else:
            unknown_args[key] = value

    return engine_args, frontend_args, api_server_args, unknown_args


def make_vllm_frontend_namespace(
    frontend_config: dict | None,
    engine_config: dict | None = None,
) -> Namespace:
    """构造接近 vLLM API server args 的 frontend namespace。

    该对象用于在当前项目内承载 vLLM OpenAI API server / chat serving 层参数，
    例如 enable_auto_tool_choice、tool_call_parser、chat_template 等。缺省值尽量
    来自 vLLM BaseFrontendArgs/FrontendArgs 签名，以适配不同 vLLM 版本。
    """
    values: dict[str, Any] = {}

    try:
        from vllm.entrypoints.openai.cli_args import BaseFrontendArgs, FrontendArgs

        for cls in (BaseFrontendArgs, FrontendArgs):
            for name, param in inspect.signature(cls).parameters.items():
                if name in values:
                    continue
                default = param.default
                if default is inspect.Signature.empty:
                    continue
                if default.__class__.__name__ == "_HAS_DEFAULT_FACTORY_CLASS":
                    continue
                values[name] = default
    except Exception:
        pass

    values.update(engine_config or {})
    values.update(frontend_config or {})
    return Namespace(**values)


def normalize_vllm_engine_args(vllm_config: dict | None) -> dict:
    """将配置文件中的 vLLM 参数转换为 AsyncEngineArgs 可接收的 kwargs。

    支持两种写法：
    - Python 风格：max_model_len: 65536
    - CLI 风格：max-model-len: 65536
    - CLI 原样风格：--speculative-config: '{"model":"...","method":"mtp"}'

    对 CLI 的否定布尔参数也做兼容，例如：
    - no-enable-prefix-caching: true -> enable_prefix_caching=False
    - no_enable_prefix_caching: true -> enable_prefix_caching=False

    对 vLLM CLI 中本身接收 JSON 的对象参数也做兼容，例如：
    - speculative-config: '{"model":"...","method":"mtp","num_speculative_tokens":5}'
      -> speculative_config={"model":"...","method":"mtp","num_speculative_tokens":5}

    这里只返回当前 vLLM 版本 AsyncEngineArgs 支持的参数；frontend/api server
    参数会由 split_vllm_config 分流，避免误传给 AsyncEngineArgs 导致启动失败。
    """
    engine_args, _, _, _ = split_vllm_config(vllm_config)
    return engine_args


def merge_additional_config(
    base_config: dict | None,
    user_additional_config: dict | None,
) -> dict:
    """合并服务端默认 additional_config 和 vllm_config.additional_config。"""
    merged = dict(base_config or {})
    if user_additional_config is None:
        return merged

    if not isinstance(user_additional_config, dict):
        raise ValueError("vllm_config.additional_config must be an object")

    merged.update(user_additional_config)
    return merged


def move_vllm_engine_args_from_additional_config(
    vllm_engine_args: dict,
    additional_config: dict | None,
) -> tuple[dict, dict]:
    """兼容把少量 vLLM 引擎参数误写到 additional_config 的场景。

    例如：
    additional_config:
      --speculative-config: '{"model":"...","method":"mtp","num_speculative_tokens":5}'

    会被转换并移动到 AsyncEngineArgs 的 speculative_config 参数，避免继续留在
    additional_config 中被底层忽略。若 vllm_config 已显式配置相同参数，则以
    vllm_config 为准。
    """
    engine_args = dict(vllm_engine_args or {})
    config = dict(additional_config or {})

    for raw_key in list(config.keys()):
        key = _normalize_config_key(raw_key, "additional_config")
        if key not in _VLLM_ENGINE_ARGS_IN_ADDITIONAL_CONFIG:
            continue

        value = config.pop(raw_key)
        if key not in engine_args:
            engine_args[key] = _parse_json_object_arg(key, value, "additional_config")

    return engine_args, config

def should_include_usage(
    stream_options: StreamOptions | None, enable_force_include_usage: bool
) -> tuple[bool, bool]:
    if stream_options:
        include_usage = stream_options.include_usage or enable_force_include_usage
        # include_continuous_usage = include_usage and bool(
        #     stream_options.continuous_usage_stats  # Openai API 没有该参数
        # )

        # 这里修改下实现
        include_continuous_usage = include_usage and enable_force_include_usage
    else:
        include_usage, include_continuous_usage = enable_force_include_usage, False
    return include_usage, include_continuous_usage
