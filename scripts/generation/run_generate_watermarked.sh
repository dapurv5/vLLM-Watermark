#!/usr/bin/env bash
set -euo pipefail

# vLLM-Watermark Generation Example Script
# This script demonstrates how to use generate_watermarked.py

echo "🌊 vLLM-Watermark Generation Example"
echo "===================================="

# Check if we're running from the correct directory
if [ ! -f "scripts/generation/generate_watermarked.py" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    echo "   Current directory: $(pwd)"
    echo "   Expected: /home/av787/vLLM-Watermark/"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p output

# Ask if user wants to run the full parameter example
read -p "📋 Do you want to run the full parameter example? (y/N): " run_full

if [[ $run_full =~ ^[Yy]$ ]]; then
    echo ""
    echo "📝 Running full parameter example..."
    echo "Input: resources/datasets/c4/processed_c4.jsonl"
    echo "Output: output/watermarked_c4.jsonl"
    echo ""

    # Check if the C4 dataset exists, if not, create a note
    if [ ! -f "resources/datasets/c4/processed_c4.jsonl" ]; then
        echo "⚠️  Note: C4 dataset not found, using test data instead"
        INPUT_FILE="resources/test_data.jsonl"
        OUTPUT_FILE="output/watermarked_test_full.jsonl"
    else
        INPUT_FILE="resources/datasets/c4/processed_c4.jsonl"
        OUTPUT_FILE="output/watermarked_c4.jsonl"
    fi

    python scripts/generation/generate_watermarked.py \
        "$INPUT_FILE" \
        prompt \
        "$OUTPUT_FILE" \
        watermarked_response \
        --watermarking_algorithm OPENAI \
        --model_name meta-llama/Llama-3.2-1B \
        --seed 12345 \
        --ngram 3 \
        --max_tokens 128 \
        --temperature 0.7 \
        --top_p 0.9 \
        --detection_threshold 0.01 \
        --max_examples 10000 \
        --gpu_memory_utilization 0.6

    echo ""
    echo "✅ Full parameter example completed!"
    echo "💾 Output saved to: $OUTPUT_FILE"
fi

echo ""
echo "🎉 Generation examples completed!"
echo ""
echo "📚 For more information, see: scripts/generation/README.md"
echo "🔧 Available algorithms: OPENAI, MARYLAND, PF"
echo "🤖 Available models: Any model supported by vLLM"
