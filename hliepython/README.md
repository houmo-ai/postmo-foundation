# 多进程多实例 vLLM 本地推理服务端

## 概述

这是一个支持多进程、多模型实例的本地推理服务端，使用 vLLM 作为底层推理引擎。主进程负责请求分发，子进程处理实际的推理任务。

支持多种推理后端：
- **vLLM**: 高性能本地 LLM 推理
- **本地图像生成**: 集成 diffusers 库进行图像生成

## 架构

- **主进程 (main.py)**: 运行在主端口（默认8000），负责根据模型名或端口号将请求转发到对应的子进程
- **子进程 (worker.py)**: 每个子进程绑定一个模型和端口，处理具体的推理请求
- **配置 (config.yaml)**: 定义模型和端口映射

## 支持的接口

当前支持：
- `/v1/chat/completions` - 聊天完成（支持流式和非流式）
- `/v1/images/generations` - 图像生成（vLLM + diffusers）
- `/v1/audio/speech` - 文本到语音

## 推理引擎

### vLLM 集成

项目使用 vLLM 作为唯一的底层推理引擎，提供高性能的本地 LLM 推理。

#### 配置 vLLM

1. 确保安装了 vLLM：
```bash
pip install vllm>=0.4.0
```

2. 在 `config.yaml` 中配置模型路径：
```yaml
models:
  - name: "your-model-name"
    instances:
      - port: 8001
        model_path: "/path/to/your/model"  # HuggingFace 模型路径或本地路径
        device_ids: [0]  # GPU 设备 ID
        extra_config:
          max_model_len: 4096
          gpu_memory_utilization: 0.9
```

#### 图像引擎特性

- **vLLM 集成**: 使用多模态模型生成图像描述
- **本地图像生成**: 集成 diffusers 库进行本地图像生成
- **灵活配置**: 支持不同图像生成模型和参数

#### 图像生成配置示例

```yaml
models:
  - name: "image-gen"
    instances:
      - port: 8003
        model_path: "/path/to/stable-diffusion-model"  # 或多模态模型路径
        device_ids: [0]
        extra_config:
          max_model_len: 512
          gpu_memory_utilization: 0.8
```

## 🐛 VSCode 调试配置

项目已配置 VSCode 调试环境，支持在 `vllm-gpu-ywh` 虚拟环境中进行调试。

### 调试配置说明

1. **Python: hliepython (vllm-gpu-ywh)** - 直接运行主程序
2. **Python: hliepython (模块模式)** - 以模块方式运行
3. **Python: 测试vLLM集成** - 运行测试脚本
4. **Python: 当前文件** - 调试当前打开的Python文件

### 使用步骤

1. 确保已激活 `vllm-gpu-ywh` 虚拟环境
2. 在 VSCode 中按 `F5` 或点击调试面板
3. 选择合适的调试配置
4. 开始调试

### 构建任务

在 VSCode 中可以使用以下任务（Ctrl+Shift+P → Tasks: Run Task）：

- **构建Whl包** - 构建分发包
- **安装依赖** - 安装项目依赖
- **运行测试** - 执行测试
- **清理构建文件** - 清理构建产物

#### vLLM 特性

- **异步推理**: 使用 `AsyncLLMEngine` 提供异步推理能力
- **流式输出**: 支持实时流式文本生成
- **多 GPU 支持**: 支持张量并行和流水线并行

## 如何添加新接口

### 1. 在 worker.py 中创建接口处理器

继承 `InterfaceHandler` 类，重写必要方法：

```python
class NewInterfaceHandler(InterfaceHandler):
    def __init__(self):
        # endpoint: 接口路径
        # supports_stream: 是否支持流式响应
        # upstream_path: 上游 API 路径（可选，默认与 endpoint 相同）
        super().__init__("/v1/new-interface", supports_stream=False)
    
    def _build_upstream_request(self, req_data: Dict[str, Any]) -> Dict[str, Any]:
        # 构造发送给上游 API 的请求体
        return {
            "model": OPENAI_MODEL,
            # 其他参数...
            **req_data
        }
```

### 2. 注册接口

在文件末尾添加注册代码：

```python
register_interface("/v1/new-interface", NewInterfaceHandler())
```

### 3. 特殊处理

如果新接口需要特殊的请求处理逻辑，可以重写其他方法：
- `_handle_non_stream()`: 自定义非流式响应处理
- `_handle_stream()`: 自定义流式响应处理

## 示例：添加 Embeddings 接口

```python
class EmbeddingsHandler(InterfaceHandler):
    def __init__(self):
        super().__init__("/v1/embeddings", supports_stream=False)
    
    def _build_upstream_request(self, req_data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "model": OPENAI_MODEL,
            "input": req_data.get("input"),
            "encoding_format": req_data.get("encoding_format", "float")
        }

register_interface("/v1/embeddings", EmbeddingsHandler())
```

## 配置示例

config.yaml:
```yaml
main_port: 8000
models:
  - name: "qwen3_4b"
    ports: [8001, 8002]  # 多实例
  - name: "llama3_4b"
    ports: [8003]        # 单实例
```

## 使用方式

### 不同模型单实例
```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3_4b", "messages":[{"role":"user","content":"Hello"}]}'
```

### 相同模型多实例
```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3_4b", "port":8001, "messages":[{"role":"user","content":"Hello"}]}'
```

## 启动服务

```bash
python main.py
```

这将启动主进程和所有配置的子进程。
