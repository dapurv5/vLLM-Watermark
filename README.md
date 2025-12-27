<div align="center">
  <img src="resources/vLLM-WM-Logo.png" alt="vLLM-Watermark Logo" width="200"/>
  <h1>vLLM-Watermark</h1>
  <p><strong>Tiny. Hackable. Lightning-fast watermarking for researchers built on vLLM</strong></p>

  <!-- [![PyPI version](https://badge.fury.io/py/vllm-watermark.svg)](https://badge.fury.io/py/vllm-watermark) -->
  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18068257.svg)](https://doi.org/10.5281/zenodo.18068257)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
  [![Documentation](https://img.shields.io/badge/docs-sphinx-blue.svg)](https://vermaapurv.com/vLLM-Watermark/index.html)
  [![Read More](https://img.shields.io/badge/read%20more-blog-purple?logo=readme&logoColor=white)](https://dapurv5.github.io/2025-10-04-vllm-watermark/)
</div>

## Supported Algorithms

| Algorithm | Description | Paper |
|-----------|-------------|-------|
| **Maryland** | Statistical watermarking with hypothesis testing | [A Watermark for Large Language Models](https://arxiv.org/pdf/2301.10226) |
| **OpenAI** | Power-law transformation with n-gram hashing | [Gumbel Watermarking](https://scottaaronson.blog/?p=6823) |
| **PF** | Prefix-free coding watermarking | [Permute-and-Flip Watermarking](https://arxiv.org/abs/2402.05864) |

## Installation

### From Source

```bash
git clone https://github.com/dapurv5/vLLM-Watermark.git
cd vLLM-Watermark
pip install -e ".[dev]"
```

### Dependencies (SLURM)
```bash
# Create conda environment
conda create -n ml_dev311 python=3.11
conda activate ml_dev311

# Install uv for fast package management
conda install -c conda-forge uv

# Install dependencies from requirements
uv pip install -r requirements-slurm.txt

# Install vllm-watermark package
uv pip install -e .
```



## Prerequisites (SLURM)
```
module load CUDA/12.6.0
module load GCC/13.3.0
```

The following also works
```
module load CUDA/12.8.0
module load GCC/14.2.0
```

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
    seed=42,  # Can also pass delta and gamma params here for MARYLAND watermark
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


## Citation

If you use vLLM-Watermark in your research, please cite:

```bibtex
@software{vllm_watermark,
  title     = {vLLM-Watermark: A tiny, hackable research framework for
               LLM watermarking experiments},
  author    = {Verma, Apurv},
  year      = {2025},
  url       = {https://github.com/dapurv5/vLLM-Watermark},
  doi       = {10.5281/zenodo.18068257},
  publisher = {Zenodo},
  version   = {v0.1.0}
}
```
---
