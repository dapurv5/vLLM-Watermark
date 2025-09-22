# Watermarking Benchmarking Suite

Comprehensive benchmarking tool for evaluating watermarking algorithms performance on the C4 dataset. Uses isolated processes to ensure complete GPU memory cleanup between algorithms.

## Quick Start

```bash
# Install required dependencies
pip install tabulate

# From project root directory
./scripts/benchmark/run_benchmark.sh
```

## Supported Algorithms

| Algorithm | Parameters | Description |
|-----------|------------|-------------|
| `OPENAI` | `ngram`, `seed`, `payload` | Power-law transformation with n-gram hashing |
| `MARYLAND` | `ngram`, `seed`, `gamma`, `delta` | Statistical watermarking with hypothesis testing |
| `MARYLAND_L` | `ngram`, `seed`, `gamma`, `delta` | Maryland algorithm with logit processing |
| `PF` | `ngram`, `seed`, `payload` | Prefix-free coding watermarking |

## Usage

### Shell Script (Recommended)

```bash
# Default: All algorithms, 5000 samples, meta-llama/Llama-3.2-1B
./scripts/benchmark/run_benchmark.sh

# Custom model
./scripts/benchmark/run_benchmark.sh meta-llama/Llama-3.2-3B

# Specific algorithms
./scripts/benchmark/run_benchmark.sh meta-llama/Llama-3.2-1B "OPENAI MARYLAND"

# Custom sample count
./scripts/benchmark/run_benchmark.sh meta-llama/Llama-3.2-1B "OPENAI PF" 1000

# Custom dataset
./scripts/benchmark/run_benchmark.sh meta-llama/Llama-3.2-1B "MARYLAND" 500 resources/test_data.jsonl
```

### Python Script (Advanced)

```bash
python scripts/benchmark/benchmark_watermarks.py \
    --model_name meta-llama/Llama-3.2-1B \
    --algorithms OPENAI MARYLAND PF \
    --num_samples 5000 \
    --data_path resources/datasets/c4/processed_c4.jsonl
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `meta-llama/Llama-3.2-1B` | HuggingFace model name |
| `algorithms` | `["OPENAI", "MARYLAND", "MARYLAND_L", "PF"]` | List of algorithms to benchmark |
| `num_samples` | `5000` | Number of samples from C4 dataset |
| `max_tokens` | `512` | Maximum tokens to generate (fixed) |
| `seed` | `42` | Random seed (fixed) |
| `temperature` | `0.7` | Sampling temperature |
| `top_p` | `0.9` | Top-p sampling parameter |
| `detection_threshold` | `0.01` | Detection threshold |
| `gpu_memory_utilization` | `0.6` | GPU memory utilization |

## Output Metrics

The benchmark provides comprehensive metrics for each algorithm:

### Detection Performance
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1 Score**: 2 × (Precision × Recall) / (Precision + Recall)
- **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
- **FPR**: False Positive Rate = FP / (FP + TN)
- **FNR**: False Negative Rate = FN / (TP + FN)

### Performance Metrics
- **Input Tokens/Second**: Throughput for input processing
- **Output Tokens/Second**: Throughput for text generation
- **Generation Time**: Time spent generating watermarked + unwatermarked text
- **Detection Time**: Time spent running detection
- **Total Time**: Generation + Detection time

### Configuration Details
Each result includes the algorithm's configuration parameters for reproducibility.

## Output Files

Results are saved to `output/benchmark/` directory:


## Dataset Requirements

### Primary Dataset (Recommended)
```
resources/datasets/c4/processed_c4.jsonl
```
- Large-scale web crawl dataset
- Each line: `{"prompt": "text content", ...}`

### Test Dataset (Fallback)
```
resources/test_data.jsonl
```
- Smaller dataset for testing
- Same format as C4


## Examples

### Basic Benchmark
```bash
# Quick test with all algorithms
./scripts/benchmark/run_benchmark.sh
```

### Production Benchmark
```bash
# Full benchmark with larger model
./scripts/benchmark/run_benchmark.sh meta-llama/Llama-3.2-3B "OPENAI MARYLAND PF" 10000
```

### Algorithm Comparison
```bash
# Compare specific algorithms
./scripts/benchmark/run_benchmark.sh meta-llama/Llama-3.2-1B "OPENAI MARYLAND"
```

### Development Testing
```bash
# Small test run
./scripts/benchmark/run_benchmark.sh meta-llama/Llama-3.2-1B "OPENAI" 50 resources/test_data.jsonl
```
