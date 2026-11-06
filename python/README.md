# HLIEPython

A high-performance, multi-process, multi-instance OpenAI-compatible API server for serving multiple AI models simultaneously.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)

## Overview

HLIEPython is a production-ready inference server that provides OpenAI-compatible APIs for multiple AI model types including text generation, image generation, speech synthesis (TTS), speech recognition (ASR), embeddings, and more. It features intelligent request routing, multi-GPU support, and dynamic model loading/unloading.

### Key Features

- 🚀 **Multi-Process Architecture**: Main process handles routing, worker processes handle inference
- 🔄 **Multiple Model Instances**: Run multiple instances of the same model for load balancing
- 🎯 **Intelligent Routing**: Automatic request distribution based on model name or port
- 🔌 **OpenAI Compatible**: Drop-in replacement for OpenAI API endpoints
- 🖼️ **Multi-Modal Support**: Text, image, audio (TTS/ASR), embeddings, reranking
- 🔥 **GPU Management**: Fine-grained GPU device allocation per instance
- 📊 **Dynamic Loading**: Load and unload models at runtime without restart
- 🌊 **Streaming Support**: Real-time streaming for chat completions and ASR

### Supported Model Types

- **Text Generation**: LLMs powered by vLLM (Qwen, LLaMA, etc.)
- **Image Generation**: Stable Diffusion and compatible models
- **Text-to-Speech (TTS)**: CosyVoice and other TTS engines
- **Speech Recognition (ASR)**: Whisper and other ASR models with WebSocket support
- **Embeddings**: Text embedding models
- **Reranking**: Text reranking models
- **Forced Alignment**: Audio-text alignment models

## Quick Start

### Prerequisites

- Python 3.11 or higher
- CUDA-compatible GPU(s) recommended
- 8GB+ RAM (varies by model size)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd python

# Install dependencies
pip install -e .

# Or install from PyPI (when published)
pip install hliepython
```

### Configuration

Create a configuration file `config.yaml`:

```yaml
main_port: 7800
models:
  - name: "my-text-model"
    instances:
      - port: 8001
        model_path: "/path/to/your/model"
        model_type: "text"
        device_ids: [0]
```

Copy the environment template and customize:

```bash
cp .env.example .env
# Edit .env to set your configuration
```

### Running the Server

```bash
# Start the server
hliepython

# Or run directly with Python
python -m hliepython.main
```

The server will start on the configured `main_port` (default: 7800).

## Usage Examples

### Chat Completion

```bash
curl -X POST http://localhost:7800/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-text-model",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Streaming Chat

```bash
curl -X POST http://localhost:7800/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-text-model",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
  }'
```

### Image Generation

```bash
curl -X POST http://localhost:7800/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-image-model",
    "prompt": "A beautiful sunset",
    "size": "512x512"
  }'
```

### Text-to-Speech

```bash
curl -X POST http://localhost:7800/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-tts-model",
    "input": "Hello world",
    "voice": "default"
  }' --output speech.mp3
```

### Dynamic Model Management

Load a model at runtime:

```bash
curl -X POST http://localhost:7800/models/load \
  -H "Content-Type: application/json" \
  -d '{"model": "my-text-model"}'
```

Unload a model:

```bash
curl -X POST http://localhost:7800/models/unload \
  -H "Content-Type: application/json" \
  -d '{"model": "my-text-model"}'
```

## Architecture

HLIEPython uses a multi-process architecture for high performance and isolation:

- **Main Process** ([main.py](hliepython/main.py)): Runs on the main port, routes requests to appropriate workers
- **Worker Processes** ([worker.py](hliepython/model_server/worker.py)): Each binds to a specific port and model instance
- **Inference Engines** ([engines/](hliepython/model_server/engines/)): Modular engine implementations for different model types
- **Configuration** ([config.yaml](hliepython/config.yaml)): Centralized model and port mapping

See the [architecture documentation](hliepython/model_server/README.md) for more details.

## API Compatibility

HLIEPython implements OpenAI-compatible endpoints:

- `POST /v1/chat/completions` - Chat completion (streaming and non-streaming)
- `POST /v1/images/generations` - Image generation
- `POST /v1/audio/speech` - Text-to-speech synthesis
- `POST /v1/audio/transcriptions` - Speech-to-text transcription
- `POST /v1/embeddings` - Text embeddings
- `GET /v1/models` - List available models
- `POST /models/load` - Dynamic model loading
- `POST /models/unload` - Dynamic model unloading

## Configuration Reference

### Environment Variables

See [.env.example](.env.example) for available options:

- `MAIN_PORT`: Main server port (default: 7800)
- `LOG_LEVEL`: Logging level (debug, info, warning, error)
- `VLLM_LOG_LEVEL`: vLLM-specific logging level
- `MODEL_ROOT_DIR`: Root directory for model files

### Model Configuration

Each model in `config.yaml` supports:

- `name`: Model identifier for API requests
- `instances`: Array of instance configurations
  - `port`: Port number for this instance
  - `model_path`: Path to model files
  - `model_type`: Engine type (text, image, tts, asr, embedding, reranker, forced_aligner)
  - `device_ids`: GPU device IDs to use

## Development

### Project Structure

```
hliepython/
├── main.py                 # Main server entry point
├── config.yaml             # Model configuration
├── config/                 # Configuration management
├── model_server/
│   ├── worker.py          # Worker process implementation
│   ├── inference_adapter.py  # Engine factory
│   ├── slot_tracker.py    # Request slot management
│   └── engines/           # Inference engine implementations
│       ├── base_engine.py
│       ├── text_engine.py
│       ├── image_engine.py
│       ├── asr_engine.py
│       ├── tts_engine.py
│       ├── embedding_engine.py
│       ├── reranker_engine.py
│       └── forced_aligner_engine.py
├── utils/                  # Utility modules
└── examples/               # Usage examples
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_models_listing.py

# Run with coverage
pytest --cov=hliepython tests/
```

### Adding a New Engine

See [hliepython/model_server/README.md](hliepython/model_server/README.md) for detailed instructions on extending the system with new inference engines.

## Docker Support

Build and run with Docker:

```bash
# Build image
docker build -t hliepython -f docker/Dockerfile .

# Run with docker-compose
docker-compose -f docker-compose.hliepython.yaml up
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone the repository
git clone <repository-url>
cd python

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Built with:
- [vLLM](https://github.com/vllm-project/vllm) - High-performance LLM inference
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [Transformers](https://github.com/huggingface/transformers) - Model implementations

## Support

- Documentation: See [docs](hliepython/model_server/README.md)
- Issues: Please file issues on GitHub
- Examples: Check the [examples/](examples/) directory

## Changelog

### v3.1.0
- Multi-modal support for text, image, audio
- Dynamic model loading/unloading
- WebSocket support for real-time ASR
- Improved GPU management
- OpenAI-compatible API endpoints
