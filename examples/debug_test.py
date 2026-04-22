#!/usr/bin/env python3
"""
调试测试脚本
用于验证VSCode调试配置是否正确
"""

def test_debug():
    """简单的调试测试函数"""
    print("开始调试测试...")

    # 设置断点测试
    x = 10
    y = 20
    result = x + y

    print(f"x = {x}, y = {y}")
    print(f"result = {result}")

    # 测试列表和循环
    data = [1, 2, 3, 4, 5]
    for i, item in enumerate(data):
        print(f"索引 {i}: {item}")

    # 测试字典
    config = {
        "model": "test",
        "port": 8000,
        "debug": True
    }

    print("配置信息:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    print("调试测试完成!")
    return result

if __name__ == "__main__":
    test_debug()