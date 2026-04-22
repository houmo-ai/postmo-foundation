#!/usr/bin/env python3
"""
测试修复后的异步推理
"""
import asyncio
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hliepython.model_server.engines.text_engine import TextEngine

async def test_non_streaming():
    """测试非流式推理"""
    print("测试非流式推理...")

    # 使用配置文件中的参数
    instance_config = {
        "model_name": "test_async",
        "device_ids": [0],
        "model_path": "/home/ywh/workspaces/models/official/Qwen3-0.6B",  # 使用实际模型路径
        "extra_config": {
            "max_model_len": 1024,
            "gpu_memory_utilization": 0.2,
            "tensor_parallel_size": 1
        }
    }

    try:
        print("创建引擎...")
        engine = TextEngine(instance_config)
        print("✅ 引擎创建成功")

        # 测试非流式请求
        test_request = {
            "messages": [{"role": "user", "content": "Hello, how are you?"}],
            "stream": False,
            "max_tokens": 50,
            "temperature": 0.7
        }

        print("执行非流式推理...")
        result = await engine.chat_completion(test_request)
        print(f"推理结果类型: {type(result)}")

        if isinstance(result, dict) and "choices" in result:
            print(f"✅ 推理成功: {result['choices'][0]['message']['content'][:50]}...")
        else:
            print(f"❌ 推理失败: {result}")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

async def test_streaming():
    """测试流式推理"""
    print("\n测试流式推理...")

    instance_config = {
        "model_name": "test_stream",
        "device_ids": [0],
        "model_path": "/home/ywh/workspaces/models/official/Qwen3-0.6B",
        "extra_config": {
            "max_model_len": 1024,
            "gpu_memory_utilization": 0.2,
            "tensor_parallel_size": 1
        }
    }

    try:
        engine = TextEngine(instance_config)

        test_request = {
            "messages": [{"role": "user", "content": "Tell me a short story."}],
            "stream": True,
            "max_tokens": 100,
            "temperature": 0.8
        }

        result = await engine.chat_completion(test_request)
        print(f"流式响应类型: {type(result)}")
        print("✅ 流式测试成功")

    except Exception as e:
        print(f"❌ 流式测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_non_streaming())
    asyncio.run(test_streaming())