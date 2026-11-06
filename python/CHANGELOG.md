# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.1.0] - 2024-11-06

### Added
- Multi-modal support for text, image, and audio processing
- Dynamic model loading and unloading capabilities
- WebSocket support for real-time ASR streaming
- Support for multiple inference engines (text, image, TTS, ASR, embedding, reranker)
- Fine-grained GPU device allocation per model instance
- OpenAI-compatible API endpoints
- Intelligent request routing based on model name or port
- Multi-process architecture for improved performance
- Comprehensive configuration via YAML and environment variables

### Features
- Text generation with vLLM backend
- Image generation support
- Text-to-speech (TTS) synthesis
- Automatic speech recognition (ASR)
- Text embedding generation
- Text reranking
- Forced alignment for audio-text synchronization

### Documentation
- Complete README with quick start guide
- Architecture documentation
- Contributing guidelines
- Apache 2.0 license
- Code of conduct
- API usage examples

### Infrastructure
- Docker support with docker-compose configurations
- Comprehensive test suite
- Example scripts for various use cases

## [Unreleased]

### Planned
- Additional model format support
- Performance optimizations
- Enhanced monitoring and metrics
- Extended API compatibility

---

For older versions, please see the git commit history.
