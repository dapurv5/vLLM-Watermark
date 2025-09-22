#!/usr/bin/env bash
set -euo pipefail

# Quick test script for generate_watermarked.py
# Runs with minimal parameters for fast testing

echo "🚀 Quick Watermark Generation Test"
echo "=================================="

# Check if we're in the right directory
if [ ! -f "scripts/generation/generate_watermarked.py" ]; then
    echo "❌ Error: Please run from project root directory"
    exit 1
fi

# Default parameters for quick testing
INPUT_FILE="${1:-resources/test_data.jsonl}"
INPUT_KEY="${2:-prompt}"
OUTPUT_FILE="${3:-output/quick_test.jsonl}"
OUTPUT_KEY="${4:-watermarked_text}"
ALGORITHM="${5:-OPENAI}"
MODEL="${6:-meta-llama/Llama-3.2-1B}"

echo "📁 Input: $INPUT_FILE"
echo "🔑 Input key: $INPUT_KEY"
echo "💾 Output: $OUTPUT_FILE"
echo "🏷️  Output key: $OUTPUT_KEY"
echo "🔧 Algorithm: $ALGORITHM"
echo "🤖 Model: $MODEL"
echo ""

# Create output directory
mkdir -p "$(dirname "$OUTPUT_FILE")"

# Run with minimal settings for speed
python scripts/generation/generate_watermarked.py \
    "$INPUT_FILE" \
    "$INPUT_KEY" \
    "$OUTPUT_FILE" \
    "$OUTPUT_KEY" \
    --watermarking_algorithm "$ALGORITHM" \
    --model_name "$MODEL" \
    --max_tokens 32 \
    --max_examples 5 \
    --gpu_memory_utilization 0.6

echo ""
echo "✅ Quick test completed!"
echo "💾 Results saved to: $OUTPUT_FILE"
echo ""
echo "Usage: $0 [input_file] [input_key] [output_file] [output_key] [algorithm] [model]"
