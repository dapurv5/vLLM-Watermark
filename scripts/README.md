# vLLM-Watermark Scripts

This directory contains scripts for watermarking text using various algorithms supported by vLLM-Watermark.

## Quick Start

For most users, start with the **Generation** scripts to create watermarked text, then use the **Pairs** scripts if you need both watermarked and unwatermarked text for evaluation.

## Script Categories

### 🎯 [Generation](./generation/)
Scripts for generating watermarked text from input prompts:
- `generate_watermarked.py` - Generate watermarked text with detection evaluation

### 🔄 [Pairs](./pairs/)
Scripts for generating both watermarked and unwatermarked text pairs:
- `generate_wm_and_unwm.py` - Generate paired outputs for comparison
- `run_hf.sh` - Example with HuggingFace datasets
- `run_jsonl.sh` - Example with local JSONL files

### 🛠️ [Utilities](./utilities/)
Helper scripts for project maintenance:
- `publish-docs.sh` - Publish documentation to GitHub Pages

## Common Parameters

Most scripts share these common parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--watermarking_algorithm` | Algorithm to use (OPENAI, MARYLAND, PF) | OPENAI |
| `--model_name` | Model to use for generation | meta-llama/Llama-3.2-1B |
| `--seed` | Random seed for reproducibility | 42 |
| `--ngram` | N-gram size for watermarking | 2 |
| `--max_tokens` | Maximum tokens to generate | 64-128 |
| `--temperature` | Sampling temperature | 0.8 |
| `--top_p` | Top-p sampling parameter | 0.95 |
| `--detection_threshold` | Detection threshold | 0.05 |

## Supported Algorithms

- **OPENAI**: Power-law transformation with n-gram hashing
- **MARYLAND**: Statistical watermarking with hypothesis testing
- **PF**: Prefix-free coding watermarking

## Getting Help

Each subdirectory contains detailed documentation for its scripts. Start with the README in the category that matches your use case.
