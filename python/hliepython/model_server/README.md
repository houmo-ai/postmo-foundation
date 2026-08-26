# 多进程多实例 OpenAI API 服务端

## 概述

这是一个支持多进程、多模型实例的 OpenAI API 兼容服务端。主进程负责请求分发，子进程处理实际的 API 调用。底层推理引擎与服务端解耦，支持文本、图像、TTS 等多种推理类型。

## 架构

- **主进程 (main.py)**: 运行在主端口（默认7800），负责根据模型名或端口号将请求转发到对应的子进程
- **子进程 (worker.py)**: 每个子进程绑定一个模型和端口，处理具体的 API 请求
- **推理适配器 (inference_adapter.py)**: 工厂类，创建不同类型的推理引擎
- **推理引擎 (engines/)**: 具体实现，支持文本、图像、TTS 等推理
- **配置 (config.yaml)**: 定义模型和端口映射

## 支持的接口

当前支持：
- `/v1/chat/completions` - 聊天完成（支持流式和非流式）
- `/v1/images/generations` - 图片生成
- `/v1/audio/speech` - 文本到语音（TTS）
- `POST /models/load` - 动态加载模型
- `POST /models/unload` - 动态卸载模型

## 推理引擎类型

- **TextEngine**: 文本聊天推理
- **ImageEngine**: 图像生成推理
- **TTSEngine**: 文本到语音推理

## 配置示例

config.yaml:
```yaml
main_port: 7800
models:
  - name: "qwen3_4b"
    instances:
      - port: 8001
        model_path: "/path/to/qwen3_4b/instance1"
        model_type: "text"
        device_ids: [1]
      - port: 8002
        model_path: "/path/to/qwen3_4b/instance2"
        model_type: "text"
        device_ids: [0,2]

  - name: "sd_xl_1"
    instances:
      - port: 8010
        model_path: "/path/to/sd_xl"
        model_type: "image"
        device_ids: [3]

  - name: "tts_en"
    instances:
      - port: 8020
        model_path: "/path/to/tts"
        model_type: "tts"
        device_ids: [4]
```

## 如何添加新推理引擎

### 1. 创建引擎类

在 `engines/` 目录下创建新引擎，继承 `InferenceEngine`：

```python
from inference_adapter import InferenceEngine

class NewEngine(InferenceEngine):
    def _load_model(self):
        # 加载模型逻辑
        pass
    
    def chat_completion(self, request):
        # 实现聊天完成
        pass
    
    def image_generation(self, request):
        # 实现图像生成
        pass
    
    def tts(self, request):
        # 实现 TTS
        pass
```

### 2. 注册到工厂

在 `inference_adapter.py` 的 `InferenceEngineFactory.get_engine()` 中添加：

```python
elif model_type == "new_type":
    from engines.new_engine import NewEngine
    return NewEngine(instance_config)
```

### 3. 添加路由

在 `worker.py` 的 `routes` 字典中添加：

```python
routes: Dict[str, str] = {
    "/v1/chat/completions": "chat_completion",
    "/v1/images/generations": "image_generation",
    "/v1/audio/speech": "tts",
    "/v1/new/endpoint": "new_method",  # 新接口
}
```

## 使用方式

### 不同模型单实例
```bash
curl -X POST http://127.0.0.1:7800/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3_4b", "messages":[{"role":"user","content":"Hello"}]}'
```

### 相同模型多实例
```bash
curl -X POST http://127.0.0.1:7800/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3_4b", "port":8001, "messages":[{"role":"user","content":"Hello"}]}'
```

### 图像生成
```bash
curl -X POST http://127.0.0.1:7800/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"model":"sd_xl_1","prompt":"a cute cat","size":"512x512"}'
```

### TTS
```bash
curl -X POST http://127.0.0.1:7800/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"tts_en","text":"Hello world","voice":"en"}'
```

### 动态加载模型
```bash
curl -X POST http://127.0.0.1:7800/models/load \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3_4b"}'
```

### 动态卸载模型
按模型卸载：
```bash
curl -X POST http://127.0.0.1:7800/models/unload \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3_4b"}'
```

按端口卸载：
```bash
curl -X POST http://127.0.0.1:7800/models/unload \
  -H "Content-Type: application/json" \
  -d '{"port":8001}'
```

## 启动服务

```bash
python main.py
```

这将启动主进程和所有配置的子进程，每个子进程会根据 `model_type` 加载对应的推理引擎。

## 扩展说明

- **解耦优势**: 服务端只负责路由和协议转换，推理引擎专注于计算
- **可扩展性**: 新推理类型只需实现引擎接口，无需修改服务端代码
- **资源隔离**: 每个实例独立 GPU 设备和模型路径
- **统一接口**: 所有引擎实现相同的抽象方法，确保一致性
