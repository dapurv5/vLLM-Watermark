# vLLM-Watermark

A Python package for implementing various watermarking algorithms for LLM outputs, with support for different backends including vLLM and SGLang.

## Features

- Multiple watermarking algorithms (KGW, Gumbel, Semantic)
- Support for different LLM backends (vLLM, SGLang)
- Easy to extend with new algorithms and backends
- Production-ready implementation
- Type-safe configuration using Pydantic
- Comprehensive testing and documentation

## Installation

### From PyPI

```bash
pip install vllm-watermark
```

### From Source

1. Clone the repository:
```bash
git clone https://github.com/yourusername/vllm-watermark.git
cd vllm-watermark
```

2. Create and activate conda environment:
```bash
conda env create -f environment.yml
conda activate vllm-watermark
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/vllm-watermark.git
cd vllm-watermark
```

2. Create and activate conda environment:
```bash
conda env create -f environment.yml
conda activate vllm-watermark
```

3. Install pre-commit hooks:
```bash
pre-commit install
```

4. Install VSCode extensions:
   - Python
   - Pylance
   - Black Formatter
   - isort
   - Python Test Explorer

The repository includes VSCode settings for:
- Black formatting on save
- isort import sorting
- Mypy type checking
- Pytest integration
- Line wrapping at 88 characters
- Trailing whitespace removal

## Project Structure

```
vllm_watermark/
├── LICENSE
├── README.md
├── environment.yml          # Conda environment file
├── pyproject.toml          # Package configuration
├── .vscode/               # VSCode settings
├── docs/                  # Documentation
├── examples/              # Example usage
├── tests/                 # Test suite
└── vllm_watermark/        # Main package
    ├── __init__.py
    ├── config.py          # Type-safe configuration
    ├── core.py            # Base classes and factory
    ├── algorithms/        # Watermarking algorithms
    │   ├── __init__.py
    │   ├── kgw.py
    │   ├── gumbel.py
    │   └── semantic.py
    ├── utils.py
    └── backends/          # LLM backend adapters
        ├── __init__.py
        ├── vllm_adapter.py
        └── sglang_adapter.py
```

## Development Guidelines

1. **Code Style**: Follow PEP 8 guidelines. We use `black` for formatting and `isort` for import sorting.

2. **Type Hints**: Use type hints for all function arguments and return values.

3. **Testing**: Write tests for all new features. Run tests with:
```bash
pytest
```

4. **Documentation**: Update documentation for any new features or changes.

5. **Pre-commit**: We use pre-commit hooks to ensure code quality. They run automatically on commit.

## Releasing to PyPI

1. Update version in `pyproject.toml`
2. Build the package:
```bash
python -m build
```
3. Upload to PyPI:
```bash
python -m twine upload dist/*
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and ensure they pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.