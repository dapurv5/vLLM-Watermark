# Generate Watermarked

This script processes JSONL files by applying watermarking to specified text fields and evaluating detection performance.

## Usage

```bash
python scripts/generate_watermarked.py \
    resources/test_data.jsonl \
    prompt \
    output/watermarked_test.jsonl \
    watermarked_text \
    --watermarking_algorithm OPENAI \
    --model_name meta-llama/Llama-3.2-1B
```

## Parameters

### Required Parameters (Positional)
1. `input_path`: Path to input JSONL file
2. `input_key`: Key in JSON to read text from
3. `output_path`: Path to output JSONL file
4. `output_key`: Key to add watermarked text to

### Optional Parameters (Named)
- `--watermarking_algorithm`: Watermarking algorithm (default: OPENAI)
  - Options: OPENAI, MARYLAND, PF
- `--model_name`: Model name (default: meta-llama/Llama-3.2-1B)
- `--seed`: Random seed (default: 42)
- `--ngram`: N-gram size for watermarking (default: 2)
- `--max_tokens`: Maximum tokens to generate (default: 64)
- `--temperature`: Sampling temperature (default: 0.8)
- `--top_p`: Top-p sampling parameter (default: 0.95)
- `--detection_threshold`: Detection threshold (default: 0.05)
- `--max_examples`: Maximum number of examples to process (default: None - process all)
- `--gpu_memory_utilization`: GPU memory utilization fraction (default: 0.8)

## Example with Full Parameters

```bash
python scripts/generate_watermarked.py \
    resources/datasets/c4/processed_c4.jsonl \
    prompt \
    output/watermarked_c4.jsonl \
    watermarked_response \
    --watermarking_algorithm OPENAI \
    --model_name meta-llama/Llama-3.2-1B \
    --seed 12345 \
    --ngram 3 \
    --max_tokens 128 \
    --temperature 0.7 \
    --top_p 0.9 \
    --detection_threshold 0.01 \
    --max_examples 100 \
    --gpu_memory_utilization 0.6
```

## Output

The script will:
1. Load the input JSONL file
2. Apply watermarking to each text field
3. Save the results with the additional watermarked field
4. Evaluate detection performance
5. Print comprehensive metrics including:
   - Average time per example
   - Tokens per second
   - Detection accuracy
   - F1 score, precision, and recall

## Input File Format

The input JSONL file should contain one JSON object per line:

```json
{"prompt": "Write a story about AI", "id": 1, "category": "creative"}
{"prompt": "Explain quantum computing", "id": 2, "category": "technical"}
```

## Output File Format

The output JSONL file will preserve all original fields and add the watermarked content:

```json
{"prompt": "Write a story about AI", "id": 1, "category": "creative", "watermarked_text": "Generated watermarked story..."}
{"prompt": "Explain quantum computing", "id": 2, "category": "technical", "watermarked_text": "Generated watermarked explanation..."}
```
