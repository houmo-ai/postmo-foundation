# Contributing to HLIEPython

Thank you for your interest in contributing to HLIEPython! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for all contributors.

## How to Contribute

### Reporting Bugs

Before creating a bug report, please check existing issues to avoid duplicates. When creating a bug report, include:

- A clear and descriptive title
- Steps to reproduce the issue
- Expected behavior vs actual behavior
- Environment details (OS, Python version, GPU info)
- Relevant logs or error messages
- Minimal code example if applicable

### Suggesting Features

Feature requests are welcome! Please provide:

- Clear description of the feature
- Use cases and motivation
- Possible implementation approach (optional)
- Impact on existing functionality

### Pull Requests

1. **Fork the repository** and create a new branch from `master`
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed
   - Keep commits focused and atomic

3. **Test your changes**
   ```bash
   pytest tests/
   ```

4. **Commit your changes**
   - Write clear, descriptive commit messages
   - Reference related issues (e.g., "Fixes #123")

5. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Open a Pull Request**
   - Provide a clear description of the changes
   - Link to related issues
   - Explain any breaking changes

## Development Setup

### Prerequisites

- Python 3.11+
- CUDA-compatible GPU (for full testing)
- Git

### Setup Instructions

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/hliepython.git
cd hliepython

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Install development dependencies
pip install pytest pytest-cov pytest-asyncio black flake8 mypy
```

## Coding Standards

### Python Style

- Follow PEP 8 style guidelines
- Use type hints where applicable
- Maximum line length: 100 characters
- Use meaningful variable and function names

### Code Formatting

We recommend using `black` for code formatting:

```bash
black hliepython/
```

### Linting

Run linting before submitting:

```bash
flake8 hliepython/ --max-line-length=100
```

### Type Checking

Use `mypy` for type checking:

```bash
mypy hliepython/
```

## Testing Guidelines

### Writing Tests

- Place tests in the `tests/` directory
- Name test files as `test_*.py`
- Use descriptive test function names
- Test both success and failure cases
- Mock external dependencies when appropriate

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_models_listing.py

# Run with coverage report
pytest --cov=hliepython --cov-report=html tests/

# Run with verbose output
pytest -v tests/
```

## Adding New Features

### Adding a New Inference Engine

1. Create a new engine class in `hliepython/model_server/engines/`
2. Inherit from `InferenceEngine` base class
3. Implement required methods
4. Register in `inference_adapter.py`
5. Add tests in `tests/`
6. Update documentation

Example:

```python
from hliepython.model_server.engines.base_engine import InferenceEngine

class NewEngine(InferenceEngine):
    def _load_model(self):
        # Load model logic
        pass
    
    def chat_completion(self, request):
        # Implement endpoint
        pass
```

See [hliepython/model_server/README.md](hliepython/model_server/README.md) for detailed instructions.

### Adding a New API Endpoint

1. Define the request/response schema in `hliepython/model_server/engines/protocol/`
2. Add the route handler in `worker.py`
3. Implement the method in relevant engine(s)
4. Add tests
5. Update API documentation

## Documentation

### Code Documentation

- Add docstrings to all public functions, classes, and modules
- Use Google-style or NumPy-style docstrings
- Include parameter types and return types
- Provide usage examples for complex functions

Example:

```python
def process_request(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """Process a chat completion request.
    
    Args:
        request: The chat completion request object
        
    Returns:
        The chat completion response
        
    Raises:
        ValueError: If request is invalid
    """
    pass
```

### Updating Documentation

- Update README.md for user-facing changes
- Update inline documentation for code changes
- Add examples for new features
- Update configuration documentation

## Project Structure

```
hliepython/
├── main.py                    # Main server entry point
├── config/                    # Configuration management
├── model_server/
│   ├── worker.py             # Worker process
│   ├── inference_adapter.py  # Engine factory
│   ├── engines/              # Inference engines
│   └── protocol/             # API request/response schemas
├── utils/                     # Utilities
├── examples/                  # Usage examples
└── tests/                     # Test suite
```

## Commit Message Guidelines

Follow conventional commits format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:

```
feat(engines): add support for new model type

Implement support for video generation models with
streaming output capabilities.

Closes #123
```

```
fix(worker): handle graceful shutdown on SIGTERM

Ensure all pending requests complete before shutdown.
```

## License

By contributing to HLIEPython, you agree that your contributions will be licensed under the Apache License 2.0.

## Questions?

Feel free to open an issue for questions or discussion. We appreciate your contributions!
