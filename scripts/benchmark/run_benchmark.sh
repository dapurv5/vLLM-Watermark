#!/usr/bin/env bash
set -euo pipefail

# vLLM-Watermark Benchmarking Script
# This script runs comprehensive benchmarks on watermarking algorithms

echo "🌊 vLLM-Watermark Benchmarking Suite"
echo "===================================="

# Check if we're running from the correct directory
if [ ! -f "scripts/benchmark/benchmark_watermarks.py" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    echo "   Current directory: $(pwd)"
    echo "   Expected: /home/av787/vLLM-Watermark/"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p output/benchmark

# Default parameters
DEFAULT_MODEL="meta-llama/Llama-3.2-1B"
DEFAULT_ALGORITHMS="OPENAI MARYLAND MARYLAND_L PF"
DEFAULT_NUM_SAMPLES=5000
DEFAULT_DATA_PATH="resources/datasets/c4/processed_c4.jsonl"

# Parse command line arguments
MODEL_NAME="${1:-$DEFAULT_MODEL}"
ALGORITHMS="${2:-$DEFAULT_ALGORITHMS}"
NUM_SAMPLES="${3:-$DEFAULT_NUM_SAMPLES}"
DATA_PATH="${4:-$DEFAULT_DATA_PATH}"

echo ""
echo "📋 Benchmark Configuration:"
echo "   🤖 Model: $MODEL_NAME"
echo "   🔧 Algorithms: $ALGORITHMS"
echo "   📊 Samples: $NUM_SAMPLES"
echo "   📁 Dataset: $DATA_PATH"
echo "   📝 Max tokens: 512 (fixed)"
echo "   🎲 Seed: 42 (fixed)"
echo ""

# Check if the dataset exists
if [ ! -f "$DATA_PATH" ]; then
    echo "⚠️  Warning: Dataset not found at $DATA_PATH"
    echo ""
    echo "🔍 Looking for alternative datasets..."

    # Check for test data
    if [ -f "resources/test_data.jsonl" ]; then
        echo "✅ Found test dataset, using: resources/test_data.jsonl"
        DATA_PATH="resources/test_data.jsonl"
        NUM_SAMPLES=100  # Limit samples for test data
        echo "   📊 Reduced samples to: $NUM_SAMPLES"
    else
        echo "❌ Error: No suitable dataset found"
        echo ""
        echo "💡 Please ensure one of the following datasets is available:"
        echo "   - resources/datasets/c4/processed_c4.jsonl (recommended)"
        echo "   - resources/test_data.jsonl (for testing)"
        echo ""
        echo "📖 See the README for dataset preparation instructions"
        exit 1
    fi
fi

# Ask for confirmation before running
echo "🚀 Ready to run benchmark. This may take several minutes..."
read -p "   Continue? (y/N): " confirm

if [[ ! $confirm =~ ^[Yy]$ ]]; then
    echo "❌ Benchmark cancelled"
    exit 0
fi

echo ""
echo "⏱️  Starting benchmark at $(date)"
echo "🔄 This may take 10-30 minutes depending on your hardware..."
echo ""

# Run the benchmark
python scripts/benchmark/benchmark_watermarks.py \
    --model_name "$MODEL_NAME" \
    --algorithms "$ALGORITHMS" \
    --num_samples "$NUM_SAMPLES" \
    --data_path "$DATA_PATH" \
    --max_tokens 512 \
    --temperature 0.7 \
    --top_p 0.9 \
    --detection_threshold 0.01 \
    --gpu_memory_utilization 0.4 \
    --output_dir "output/benchmark"

echo ""
echo "✅ Benchmark completed at $(date)"
echo "💾 Results saved in: output/benchmark/"
echo ""
echo "📊 To view results:"
echo "   - Check the console output above for the summary table"
echo "   - Open the CSV file in output/benchmark/ with Excel or Google Sheets"
echo ""
echo "🔧 To run with different parameters:"
echo "   $0 <model_name> \"<algorithms>\" <num_samples> <data_path>"
echo ""
echo "📖 Examples:"
echo "   $0 meta-llama/Llama-3.2-3B \"OPENAI MARYLAND\" 1000"
echo "   $0 meta-llama/Llama-3.2-1B \"PF\" 500 resources/test_data.jsonl"
echo ""
echo "🎉 Benchmarking complete!"
