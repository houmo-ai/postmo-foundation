from hliepython.model_server.engines.protocol.chat_completions import StreamOptions

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
