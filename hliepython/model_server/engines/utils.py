from hliepython.model_server.engines.protocol.chat_completions import StreamOptions


def normalize_vllm_engine_args(vllm_config: dict | None) -> dict:
    """将配置文件中的 vLLM 参数转换为 AsyncEngineArgs 可接收的 kwargs。

    支持两种写法：
    - Python 风格：max_model_len: 65536
    - CLI 风格：max-model-len: 65536

    对 CLI 的否定布尔参数也做兼容，例如：
    - no-enable-prefix-caching: true -> enable_prefix_caching=False
    - no_enable_prefix_caching: true -> enable_prefix_caching=False

    这里不维护 vLLM 参数白名单，配置了什么就透传什么；如果参数不是当前
    vLLM 版本的 AsyncEngineArgs 支持项，底层 vLLM 会按自身逻辑报错。
    """
    if not vllm_config:
        return {}

    if not isinstance(vllm_config, dict):
        raise ValueError("vllm_config must be an object")

    normalized = {}
    for raw_key, value in vllm_config.items():
        if not isinstance(raw_key, str):
            raise ValueError("vllm_config keys must be strings")

        key = raw_key.strip().replace("-", "_")
        if not key:
            raise ValueError("vllm_config keys must not be empty")

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
