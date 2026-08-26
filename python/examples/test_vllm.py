#!/usr/bin/env python3
"""
测试 vLLM 集成
"""
import asyncio
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hliepython.model_server.engines.text_engine import TextEngine

async def test_vllm_with_real_model():
    """使用真实模型测试 vLLM 集成"""
    print("开始测试 vLLM 真实模型集成...")

    # 使用一个小的测试模型
    instance_config = {
        "model_name": "test_model",
        "device_ids": [0],  # 使用 GPU
        "model_path": "microsoft/DialoGPT-small",  # 小型对话模型
        "extra_config": {
            "max_model_len": 512,
            "gpu_memory_utilization": 0.5
        }
    }

    # 创建引擎实例
    engine = TextEngine(instance_config)

    # 测试请求
    test_request = {
        "messages": [
            {"role": "user", "content": "Hello, how are you?"}
        ],
        "stream": False,
        "max_tokens": 50,
        "temperature": 0.7
    }

    try:
        # 调用 chat_completion
        result = engine.chat_completion(test_request)
        print("推理结果:")
        if isinstance(result, dict) and "choices" in result:
            print(f"Response: {result['choices'][0]['message']['content']}")
        else:
            print(result)
        print("✅ 测试成功")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

async def test_streaming():
    """测试流式推理"""
    print("\n开始测试流式推理...")

    instance_config = {
        "model_name": "test_model",
        "device_ids": [0],
        "model_path": "microsoft/DialoGPT-small",
        "extra_config": {
            "max_model_len": 512,
            "gpu_memory_utilization": 0.5
        }
    }

    engine = TextEngine(instance_config)

    test_request = {
        "messages": [
            {"role": "user", "content": "Tell me a short story."}
        ],
        "stream": True,
        "max_tokens": 100,
        "temperature": 0.8
    }

    try:
        result = engine.chat_completion(test_request)
        print(f"流式响应类型: {type(result)}")
        print("✅ 流式测试成功")

    except Exception as e:
        print(f"❌ 流式测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_vllm_with_real_model())
    asyncio.run(test_streaming())