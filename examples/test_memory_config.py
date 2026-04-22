#!/usr/bin/env python3
"""
测试配置修改后的vLLM内存使用
"""
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hliepython.model_server.engines.text_engine import TextEngine

def test_memory_config():
    """测试内存配置"""
    print("测试内存配置...")

    # 使用配置文件中的参数
    instance_config = {
        "model_name": "test_memory",
        "device_ids": [0],
        "model_path": "/home/ywh/workspaces/models/official/Qwen3-0.6B",  # 使用实际模型路径
        "extra_config": {
            "max_model_len": 1024,  # 从配置文件中读取
            "gpu_memory_utilization": 0.2,  # 从配置文件中读取
            "tensor_parallel_size": 1
        }
    }

    try:
        print(f"配置参数: max_model_len={instance_config['extra_config']['max_model_len']}, gpu_memory_utilization={instance_config['extra_config']['gpu_memory_utilization']}")
        engine = TextEngine(instance_config)
        print("✅ 引擎创建成功，内存配置正常")

        # 测试简单推理
        test_request = {
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
            "max_tokens": 50,
            "temperature": 0.7
        }

        result = engine.chat_completion(test_request)
        if "choices" in result and result["choices"]:
            print(f"✅ 推理成功: {result['choices'][0]['message']['content'][:50]}...")
        else:
            print(f"❌ 推理失败: {result}")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_memory_config()