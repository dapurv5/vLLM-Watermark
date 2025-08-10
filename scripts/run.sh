#!/usr/bin/env bash
set -euo pipefail

python scripts/generate_wm_and_unwm.py \
  --input_path "truthfulqa/truthful_qa" \
  --hf_name "generation" \
  --hf_split "validation" \
  --input_key "question" \
  --output_path "output/truthfulqa_pairs.jsonl" \
  --watermarking_algorithm "MARYLAND" \
  --model_name "meta-llama/Llama-3.2-1B" \
  --seed 42 \
  --ngram 4 \
  --detection_threshold 0.05 \
  --temperature 0.8 \
  --max_tokens 128 \
  --top_p 0.95 \
  --num_wm_generations_per_prompt 4 \
  --num_unwm_generations_per_prompt 2 \
  --dataset_start_row 0 \
  --dataset_end_row 800
