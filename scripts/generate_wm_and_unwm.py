#!/usr/bin/env python3
"""
Generate Watermarked and Unwatermarked Text Pairs

This script processes a JSONL dataset and, for each prompt, generates:
- num_wm_generations_per_prompt watermarked completions
- num_unwm_generations_per_prompt unwatermarked completions

It preserves the same output fields as the user's earlier script, so the
resulting JSONL is compatible with existing consumers that expect fields like:

- For multi-generation (beamed) mode (any num_*_generations_per_prompt > 1):
  - "watermarked_texts": list[str]
  - "unwatermarked_texts": list[str]
  - "watermarked_texts.is_watermarked": list[bool]
  - "unwatermarked_texts.is_watermarked": list[bool]
  - "watermarked_texts.score": list[float]
  - "unwatermarked_texts.score": list[float]
  - "watermarked_texts.pvalue": list[float]
  - "unwatermarked_texts.pvalue": list[float]
  - plus a randomly selected single text from each list with fields:
    - "watermarked_text", "unwatermarked_text"
    - "watermarked_text.is_watermarked", "unwatermarked_text.is_watermarked"
    - "watermarked_text.score", "unwatermarked_text.score"
    - "watermarked_text.pvalue", "unwatermarked_text.pvalue"

- For single-generation (greedy) mode (both nums == 1):
  - "watermarked_text", "unwatermarked_text"
  - "watermarked_text.is_watermarked", "unwatermarked_text.is_watermarked"
  - "watermarked_score", "unwatermarked_score"
  - "watermarked_pvalue", "unwatermarked_pvalue"

CLI mostly mirrors scripts/generate_watermarked.py and adds the two counts.
"""

import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Tuple

import fire
from tqdm import tqdm

# Environment adjustments consistent with the other script
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

# Make project importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vllm import LLM, SamplingParams

from vllm_watermark.core import (
    DetectionAlgorithm,
    WatermarkedLLMs,
    WatermarkingAlgorithm,
)
from vllm_watermark.watermark_detectors import WatermarkDetectors


class WatermarkPairsProcessor:
    """Generate watermarked/unwatermarked pairs per prompt."""

    def load_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        data: List[Dict[str, Any]] = []
        print(f"Loading data from {file_path}...")
        with open(file_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Loading JSONL"):
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    def save_jsonl(self, data: List[Dict[str, Any]], file_path: str) -> None:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        print(f"Saving data to {file_path}...")
        with open(file_path, "w", encoding="utf-8") as f:
            for item in tqdm(data, desc="Saving JSONL"):
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def get_algorithm_enum(self, algo_str: str) -> WatermarkingAlgorithm:
        algo_map = {
            "OPENAI": WatermarkingAlgorithm.OPENAI,
            "MARYLAND": WatermarkingAlgorithm.MARYLAND,
            "PF": WatermarkingAlgorithm.PF,
        }
        if algo_str.upper() not in algo_map:
            raise ValueError(f"Unsupported watermarking algorithm: {algo_str}")
        return algo_map[algo_str.upper()]

    def get_detection_algorithm(
        self, watermark_algo: WatermarkingAlgorithm
    ) -> DetectionAlgorithm:
        detection_map = {
            WatermarkingAlgorithm.OPENAI: DetectionAlgorithm.OPENAI_Z,
            WatermarkingAlgorithm.MARYLAND: DetectionAlgorithm.MARYLAND_Z,
            WatermarkingAlgorithm.PF: DetectionAlgorithm.PF,
        }
        return detection_map[watermark_algo]

    def load_and_validate_data(
        self, input_path: str, input_key: str, max_examples: int | None = None
    ) -> List[Dict[str, Any]]:
        try:
            data = self.load_jsonl(input_path)
            print(f"✅ Loaded {len(data)} examples")
        except Exception as e:
            print(f"❌ Error loading input file: {e}")
            sys.exit(1)

        if max_examples is not None:
            data = data[:max_examples]
            print(f"📊 Processing first {len(data)} examples")

        if not data or input_key not in data[0]:
            print(f"❌ Error: Key '{input_key}' not found in input data")
            print(f"   Available keys: {list(data[0].keys()) if data else 'No data'}")
            sys.exit(1)

        return data

    def initialize_models(
        self,
        model_name: str,
        watermarking_algorithm: str,
        seed: int,
        ngram: int,
        detection_threshold: float,
        gpu_memory_utilization: float = 0.8,
    ) -> Tuple[Any, Any, Any]:
        print("🚀 Initializing model and watermarking...")
        try:
            print(f"   Loading model: {model_name}")
            print(f"   GPU memory utilization: {gpu_memory_utilization}")

            llm = LLM(
                model=model_name,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=2048,
            )

            watermark_algo = self.get_algorithm_enum(watermarking_algorithm)
            print(f"   Watermarking algorithm: {watermarking_algorithm}")

            wm_llm = WatermarkedLLMs.create(
                llm, algo=watermark_algo, seed=seed, ngram=ngram
            )

            detection_algo = self.get_detection_algorithm(watermark_algo)
            detector = WatermarkDetectors.create(
                algo=detection_algo,
                model=llm,
                ngram=ngram,
                seed=seed,
                payload=0,
                threshold=detection_threshold,
            )

            print("✅ Model and watermarking initialized successfully")
            return llm, wm_llm, detector

        except Exception as e:
            error_msg = str(e)
            print(f"❌ Error initializing model: {error_msg}")
            if "Free memory" in error_msg and "GPU memory utilization" in error_msg:
                print("\n💡 GPU Memory Troubleshooting:")
                print("   This error indicates insufficient GPU memory.")
                print("   Try the following solutions:")
                print(
                    "   1. Use a smaller model (e.g., meta-llama/Llama-3.2-1B-Instruct)"
                )
                print("   2. Reduce --gpu_memory_utilization (e.g., 0.6 or 0.4)")
                print("   3. Close other GPU processes")
                print("   4. Use CPU-only mode if available")
                try:
                    print(
                        f"   5. Current memory needed: ~{error_msg.split('(')[1].split(')')[0]} GiB"
                    )
                except Exception:
                    pass
            sys.exit(1)

    def extract_prompts(self, data: List[Dict[str, Any]], input_key: str) -> List[str]:
        prompts: List[str] = []
        skipped_count = 0
        print("📝 Extracting prompts...")
        for item in tqdm(data, desc="Extracting prompts"):
            if input_key in item and item[input_key]:
                prompts.append(item[input_key])
            else:
                skipped_count += 1
        if skipped_count > 0:
            print(
                f"⚠️  Skipped {skipped_count} items with empty or missing '{input_key}' field"
            )
        if not prompts:
            print("❌ No valid prompts found to process")
            sys.exit(1)
        print(f"✅ Extracted {len(prompts)} valid prompts")
        return prompts

    def _collect_texts(self, outputs: List[Any]) -> List[List[str]]:
        """Convert vLLM outputs to list[list[str]] per prompt."""
        texts_per_prompt: List[List[str]] = []
        for out in outputs:
            seq_texts = [seq.text for seq in out.outputs]
            texts_per_prompt.append(seq_texts)
        return texts_per_prompt

    def _batch_detect(self, texts_per_prompt: List[List[str]], detector: Any):
        """Detect watermark metrics per text. Returns parallel structures."""
        flags: List[List[bool]] = []
        scores: List[List[float]] = []
        pvalues: List[List[float]] = []
        for texts in texts_per_prompt:
            f_row: List[bool] = []
            s_row: List[float] = []
            p_row: List[float] = []
            for text in texts:
                try:
                    det = detector.detect(text)
                    f_row.append(bool(det["is_watermarked"]))
                    s_row.append(float(det.get("score", 0.0)))
                    p_row.append(float(det.get("pvalue", 1.0)))
                except Exception:
                    f_row.append(False)
                    s_row.append(0.0)
                    p_row.append(1.0)
            flags.append(f_row)
            scores.append(s_row)
            pvalues.append(p_row)
        return flags, scores, pvalues

    def _print_batch_metrics(
        self, wm_flags: List[List[bool]], unwm_flags: List[List[bool]]
    ):
        wm_flat = [flag for row in wm_flags for flag in row]
        unwm_flat = [flag for row in unwm_flags for flag in row]
        if wm_flat:
            wm_correct = sum(1 for x in wm_flat if x)
            fnr = (len(wm_flat) - wm_correct) / max(1, len(wm_flat))
        else:
            fnr = 0.0
        if unwm_flat:
            unwm_correct = sum(1 for x in unwm_flat if not x)
            fpr = (len(unwm_flat) - unwm_correct) / max(1, len(unwm_flat))
        else:
            fpr = 0.0
        print(f"Batch FNR: {fnr:.4f}, Batch FPR: {fpr:.4f}")

    def process_dataset(
        self,
        input_path: str,
        input_key: str,
        output_path: str,
        output_key: str,  # accepted for CLI compatibility; not required by this script
        watermarking_algorithm: str = "OPENAI",
        model_name: str = "meta-llama/Llama-3.2-1B",
        seed: int = 42,
        ngram: int = 2,
        max_tokens: int = 64,
        temperature: float = 0.8,
        top_p: float = 0.95,
        detection_threshold: float = 0.05,
        max_examples: int | None = None,
        gpu_memory_utilization: float = 0.8,
        num_wm_generations_per_prompt: int = 1,
        num_unwm_generations_per_prompt: int = 1,
    ):
        print("🌊 WM + UNWM GENERATION")
        print("=" * 60)
        print(f"📁 Input file: {input_path}")
        print(f"🔑 Input key: {input_key}")
        print(f"💾 Output file: {output_path}")
        print(f"🏷️  Output key (ignored): {output_key}")
        print(f"🔧 Algorithm: {watermarking_algorithm}")
        print(f"🤖 Model: {model_name}")
        print(f"🎲 Seed: {seed}")
        print(f"📊 N-gram: {ngram}")
        print(f"📝 Max tokens: {max_tokens}")
        print(f"🌡️  Temperature: {temperature}")
        print(f"🎯 Top-p: {top_p}")
        print(f"🚨 Detection threshold: {detection_threshold}")
        print(f"🎮 GPU memory utilization: {gpu_memory_utilization}")
        print(f"🔁 num_wm_generations_per_prompt: {num_wm_generations_per_prompt}")
        print(f"🔁 num_unwm_generations_per_prompt: {num_unwm_generations_per_prompt}")
        if max_examples:
            print(f"📊 Max examples: {max_examples}")
        print("=" * 60)

        data = self.load_and_validate_data(input_path, input_key, max_examples)

        llm, wm_llm, detector = self.initialize_models(
            model_name,
            watermarking_algorithm,
            seed,
            ngram,
            detection_threshold,
            gpu_memory_utilization,
        )

        prompts = self.extract_prompts(data, input_key)

        # Build sampling params for each generator
        sampling_wm = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
            n=max(1, int(num_wm_generations_per_prompt)),
        )
        sampling_unwm = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
            n=max(1, int(num_unwm_generations_per_prompt)),
        )

        # Generate watermarked
        print("🎯 Generating watermarked text...")
        gen_wm_start = time.time()
        wm_outputs = wm_llm.generate(prompts, sampling_wm)
        gen_wm_time = time.time() - gen_wm_start

        # Generate unwatermarked
        print("🎯 Generating unwatermarked text...")
        gen_unwm_start = time.time()
        unwm_outputs = llm.generate(prompts, sampling_params=sampling_unwm)
        gen_unwm_time = time.time() - gen_unwm_start

        wm_texts = self._collect_texts(wm_outputs)
        unwm_texts = self._collect_texts(unwm_outputs)

        # Detect
        print("🔍 Detecting watermarks...")
        wm_flags, wm_scores, wm_pvalues = self._batch_detect(wm_texts, detector)
        unwm_flags, unwm_scores, unwm_pvalues = self._batch_detect(unwm_texts, detector)

        # Print quick batch metrics
        self._print_batch_metrics(wm_flags, unwm_flags)

        # Build result rows merging with original items
        beamed = (sampling_wm.n > 1) or (sampling_unwm.n > 1)
        results: List[Dict[str, Any]] = []
        for i, item in enumerate(data):
            row: Dict[str, Any] = {}
            if beamed:
                row.update(
                    {
                        "watermarked_texts": wm_texts[i],
                        "unwatermarked_texts": unwm_texts[i],
                        "watermarked_texts.is_watermarked": wm_flags[i],
                        "unwatermarked_texts.is_watermarked": unwm_flags[i],
                        "watermarked_texts.score": wm_scores[i],
                        "unwatermarked_texts.score": unwm_scores[i],
                        "watermarked_texts.pvalue": wm_pvalues[i],
                        "unwatermarked_texts.pvalue": unwm_pvalues[i],
                    }
                )

                # Randomly select one from each set (emulate prior script behavior)
                rand_idx_wm = (
                    random.randint(0, len(wm_texts[i]) - 1) if wm_texts[i] else 0
                )
                rand_idx_unwm = (
                    random.randint(0, len(unwm_texts[i]) - 1) if unwm_texts[i] else 0
                )
                if wm_texts[i]:
                    row["watermarked_text"] = wm_texts[i][rand_idx_wm]
                    row["watermarked_text.is_watermarked"] = wm_flags[i][rand_idx_wm]
                    row["watermarked_text.score"] = wm_scores[i][rand_idx_wm]
                    row["watermarked_text.pvalue"] = wm_pvalues[i][rand_idx_wm]
                else:
                    row["watermarked_text"] = ""
                    row["watermarked_text.is_watermarked"] = False
                    row["watermarked_text.score"] = 0.0
                    row["watermarked_text.pvalue"] = 1.0

                if unwm_texts[i]:
                    row["unwatermarked_text"] = unwm_texts[i][rand_idx_unwm]
                    row["unwatermarked_text.is_watermarked"] = unwm_flags[i][
                        rand_idx_unwm
                    ]
                    row["unwatermarked_text.score"] = unwm_scores[i][rand_idx_unwm]
                    row["unwatermarked_text.pvalue"] = unwm_pvalues[i][rand_idx_unwm]
                else:
                    row["unwatermarked_text"] = ""
                    row["unwatermarked_text.is_watermarked"] = False
                    row["unwatermarked_text.score"] = 0.0
                    row["unwatermarked_text.pvalue"] = 1.0

            else:
                # Greedy-like single outputs
                wm_text = wm_texts[i][0] if wm_texts[i] else ""
                unwm_text = unwm_texts[i][0] if unwm_texts[i] else ""
                row.update(
                    {
                        "watermarked_text": wm_text,
                        "unwatermarked_text": unwm_text,
                        "watermarked_text.is_watermarked": (
                            bool(wm_flags[i][0]) if wm_flags[i] else False
                        ),
                        "unwatermarked_text.is_watermarked": (
                            bool(unwm_flags[i][0]) if unwm_flags[i] else False
                        ),
                        "watermarked_score": (
                            float(wm_scores[i][0]) if wm_scores[i] else 0.0
                        ),
                        "unwatermarked_score": (
                            float(unwm_scores[i][0]) if unwm_scores[i] else 0.0
                        ),
                        "watermarked_pvalue": (
                            float(wm_pvalues[i][0]) if wm_pvalues[i] else 1.0
                        ),
                        "unwatermarked_pvalue": (
                            float(unwm_pvalues[i][0]) if unwm_pvalues[i] else 1.0
                        ),
                    }
                )

            # Also duplicate to output_key for compatibility if caller expects it
            if row.get("watermarked_text") is not None:
                row[output_key] = row["watermarked_text"]

            # Merge with original item
            results.append(item | row)

        # Save
        try:
            self.save_jsonl(results, output_path)
            print(f"✅ Saved {len(results)} examples to {output_path}")
        except Exception as e:
            print(f"❌ Error saving output file: {e}")
            sys.exit(1)

        # Timings
        print("=" * 60)
        print("⏱️  TIMINGS")
        print("=" * 60)
        print(f"Watermarked generation time: {gen_wm_time:.2f}s")
        print(f"Unwatermarked generation time: {gen_unwm_time:.2f}s")
        print("=" * 60)


def main():
    processor = WatermarkPairsProcessor()
    fire.Fire(processor.process_dataset)


if __name__ == "__main__":
    main()
