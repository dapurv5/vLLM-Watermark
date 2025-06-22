<div align="center">
  <img src="resources/vLLM-WM-Logo.png" alt="vLLM-Watermark Logo" width="200"/>
  <h1>vLLM-Watermark</h1>
  <p><strong>A Python package for implementing various watermarking algorithms for LLM outputs</strong></p>

  [![PyPI version](https://badge.fury.io/py/vllm-watermark.svg)](https://badge.fury.io/py/vllm-watermark)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
  [![Documentation](https://img.shields.io/badge/docs-sphinx-blue.svg)](https://vllm-watermark.readthedocs.io/)
</div>

## Overview

vLLM-Watermark is a comprehensive Python package that extends [vLLM](https://github.com/vllm-project/vllm) with state-of-the-art watermarking algorithms for Large Language Models (LLMs). It provides seamless integration with vLLM's high-performance inference engine while adding robust watermarking capabilities for text generation.

### Key Features

- 🔒 **Multiple Watermarking Algorithms**: Support for OpenAI, Maryland, and PF watermarking schemes
- ⚡ **High Performance**: Built on top of vLLM for fast inference
- 🎯 **Flexible Detection**: Multiple detection algorithms with statistical significance testing
- 🔧 **Easy Integration**: Simple API that works with existing vLLM workflows
- 📊 **Comprehensive Metrics**: P-values, confidence scores, and statistical analysis
- 🧪 **Research Ready**: Designed for both production use and academic research

## Supported Algorithms

| Algorithm | Description | Paper |
|-----------|-------------|-------|
| **OpenAI** | Power-law transformation with n-gram hashing | [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226) |
| **Maryland** | Statistical watermarking with hypothesis testing | [A Statistical Approach to Neural Text Generation](https://arxiv.org/abs/1906.02429) |
| **PF** | Prefix-free coding watermarking | [Prefix-Free Code Distribution Matching](https://arxiv.org/abs/2201.12677) |

## Installation

### From PyPI

```bash
pip install vllm-watermark
```

### From Source

```bash
git clone https://github.com/yourusername/vllm-watermark.git
cd vllm-watermark
pip install -e ".[dev]"
```

### Dependencies

- Python 3.8+
- PyTorch 2.0+
- vLLM 0.2.0+
- Transformers 4.30.0+
- NumPy 1.20.0+

## Quick Start

### Basic Usage

```python
from vllm import LLM, SamplingParams
from vllm_watermark.core import WatermarkedLLMs, WatermarkingAlgorithm
from vllm_watermark.watermark_detectors import WatermarkDetectors, DetectionAlgorithm

# Load your model
llm = LLM(model="meta-llama/Llama-3.2-1B")

# Create a watermarked LLM
wm_llm = WatermarkedLLMs.create(
    llm,
    algo=WatermarkingAlgorithm.OPENAI,
    seed=42,
    ngram=2
)

# Generate watermarked text
prompts = ["Write a short poem about AI"]
sampling_params = SamplingParams(temperature=0.8, max_tokens=64)
outputs = wm_llm.generate(prompts, sampling_params)

# Detect watermark
detector = WatermarkDetectors.create(
    algo=DetectionAlgorithm.OPENAI_Z,
    model=llm,
    ngram=2,
    seed=42,
    threshold=0.05
)

for output in outputs:
    text = output.outputs[0].text
    result = detector.detect(text)
    print(f"Watermarked: {result['is_watermarked']}")
    print(f"P-value: {result['pvalue']:.6f}")
```

### Advanced Configuration

```python
# Maryland watermarking with custom parameters
wm_llm = WatermarkedLLMs.create(
    llm,
    algo=WatermarkingAlgorithm.MARYLAND,
    gamma=0.5,  # watermark strength
    delta=2.0,  # threshold parameter
    seed=123
)

# PF watermarking
wm_llm = WatermarkedLLMs.create(
    llm,
    algo=WatermarkingAlgorithm.PF,
    payload=0x12345678,  # custom payload
    seed=456
)
```

## Examples

Check out the `examples/` directory for complete working examples:

- [`example_openai.py`](examples/example_openai.py) - OpenAI watermarking with detection
- [`example_maryland.py`](examples/example_maryland.py) - Maryland watermarking with statistical analysis

## Documentation

For detailed documentation, including API reference, mathematical foundations, and advanced usage patterns, visit:

📖 **[Documentation](https://vllm-watermark.readthedocs.io/)**

## Mathematical Foundation

vLLM-Watermark implements several watermarking algorithms based on rigorous mathematical foundations:

### OpenAI Watermarking

The OpenAI algorithm uses a power-law transformation:

$$P_w(t) = \frac{P(t)^\gamma}{\sum_{t'} P(t')^\gamma}$$

where $P(t)$ is the original token probability and $\gamma$ controls the watermark strength.

### Maryland Watermarking

The Maryland approach uses hypothesis testing with:

$$H_0: \text{Text is not watermarked}$$
$$H_1: \text{Text is watermarked}$$

Detection is based on statistical significance testing of n-gram distributions.

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/yourusername/vllm-watermark.git
cd vllm-watermark
pip install -e ".[dev]"
pre-commit install
```

### Running Tests

```bash
pytest tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use vLLM-Watermark in your research, please cite:

```bibtex
@software{vllm_watermark,
  title={vLLM-Watermark: A Python package for implementing various watermarking algorithms for LLM outputs},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/vllm-watermark}
}
```

## Acknowledgments

- Built on top of [vLLM](https://github.com/vllm-project/vllm) for high-performance inference
- Implements watermarking algorithms from leading research papers
- Inspired by the need for robust AI-generated content identification

## Support

- 📚 [Documentation](https://vllm-watermark.readthedocs.io/)
- 🐛 [Issue Tracker](https://github.com/yourusername/vllm-watermark/issues)
- 💬 [Discussions](https://github.com/yourusername/vllm-watermark/discussions)

---

<div align="center">
  <p>Made with ❤️ for the AI community</p>
</div>