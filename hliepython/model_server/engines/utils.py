import json

from hliepython.model_server.engines.protocol.chat_completions import StreamOptions


_VLLM_JSON_OBJECT_ARGS = {"speculative_config"}
_VLLM_ENGINE_ARGS_IN_ADDITIONAL_CONFIG = {"speculative_config"}


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

    这里不维护 vLLM 参数白名单，配置了什么就透传什么；如果参数不是当前
    vLLM 版本的 AsyncEngineArgs 支持项，底层 vLLM 会按自身逻辑报错。
    """
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
