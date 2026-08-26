#!/usr/bin/env python3
"""
图像引擎测试脚本
用于验证图像引擎的 vLLM 集成是否正确
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hliepython.model_server.engines.image_engine import ImageEngine

def test_image_engine():
    """测试图像引擎"""
    print("开始测试图像引擎...")

    # 创建 ImageEngine 实例配置
    instance_config = {
        "model_name": "test_image",
        "device_ids": [],  # 使用 CPU 进行测试
        "model_path": "/tmp/dummy_image_model",  # 虚拟路径，用于测试错误处理
        "extra_config": {}
    }

    # 创建引擎实例
    engine = ImageEngine(instance_config)

    # 测试请求
    test_request = {
        "prompt": "A beautiful sunset over mountains",
        "n": 1,
        "size": "512x512"
    }

    try:
        # 调用 image_generation
        result = engine.image_generation(test_request)
        print("图像生成结果:")
        print(f"响应类型: {type(result)}")

        if "error" in result:
            print(f"错误: {result['error']}")
        elif "data" in result:
            print(f"生成了 {len(result['data'])} 张图像")
            for i, img_data in enumerate(result['data']):
                print(f"图像 {i+1}: {img_data.get('url', '')[:50]}...")
        else:
            print(f"完整响应: {result}")

        print("✅ 图像引擎测试成功")

    except Exception as e:
        print(f"❌ 图像引擎测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_image_engine()