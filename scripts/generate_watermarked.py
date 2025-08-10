#!/usr/bin/env python3
"""
Generate Watermarked Dataset Script

This script processes JSONL files by applying watermarking to specified text fields
and evaluating detection performance.

Usage:
    python scripts/generate_watermarked.py \
        resources/datasets/c4/processed_c4.jsonl \
        prompt \
        output/watermarked_c4.jsonl \
        watermarked_text \
        --watermarking_algorithm OPENAI \
        --model_name meta-llama/Llama-3.2-1B
"""

import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import fire
from tqdm import tqdm

# Set required environment variables
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vllm import LLM, SamplingParams

from vllm_watermark.core import (
    DetectionAlgorithm,
    WatermarkedLLMs,
    WatermarkingAlgorithm,
)
from vllm_watermark.watermark_detectors import WatermarkDetectors


class WatermarkDatasetProcessor:
    """Main class for processing datasets with watermarking."""

    def __init__(self):
        """Initialize the processor."""
        pass

    def load_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """Load data from a JSONL file."""
        data = []
        print(f"Loading data from {file_path}...")
        with open(file_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Loading JSONL"):
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    def save_jsonl(self, data: List[Dict[str, Any]], file_path: str) -> None:
        """Save data to a JSONL file."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        print(f"Saving data to {file_path}...")
        with open(file_path, "w", encoding="utf-8") as f:
            for item in tqdm(data, desc="Saving JSONL"):
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def get_algorithm_enum(self, algo_str: str) -> WatermarkingAlgorithm:
        """Convert string to WatermarkingAlgorithm enum."""
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
        """Get corresponding detection algorithm."""
        detection_map = {
            WatermarkingAlgorithm.OPENAI: DetectionAlgorithm.OPENAI_Z,
            WatermarkingAlgorithm.MARYLAND: DetectionAlgorithm.MARYLAND_Z,
            WatermarkingAlgorithm.PF: DetectionAlgorithm.PF,
        }
        return detection_map[watermark_algo]

    def count_tokens(self, text: str, tokenizer) -> int:
        """Count tokens in text using the model's tokenizer."""
        try:
            tokens = tokenizer.encode(text)
            return len(tokens)
        except Exception:
            # Fallback to word count approximation
            return len(text.split())

    def load_and_validate_data(
        self, input_path: str, input_key: str, max_examples: int | None = None
    ) -> List[Dict[str, Any]]:
        """Load and validate input data."""
        try:
            data = self.load_jsonl(input_path)
            print(f"✅ Loaded {len(data)} examples")
        except Exception as e:
            print(f"❌ Error loading input file: {e}")
            sys.exit(1)

        # Limit examples if specified
        if max_examples is not None:
            data = data[:max_examples]
            print(f"📊 Processing first {len(data)} examples")

        # Validate input key exists
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
        """Initialize LLM, watermarked LLM, and detector."""
        print("🚀 Initializing model and watermarking...")
        try:
            # Try to create LLM with lower GPU memory utilization to avoid OOM
            print(f"   Loading model: {model_name}")
            print(f"   GPU memory utilization: {gpu_memory_utilization}")

            llm = LLM(
                model=model_name,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=2048,  # Reduce max length to save memory
            )

            watermark_algo = self.get_algorithm_enum(watermarking_algorithm)
            print(f"   Watermarking algorithm: {watermarking_algorithm}")

            # Create watermarked LLM
            wm_llm = WatermarkedLLMs.create(
                llm, algo=watermark_algo, seed=seed, ngram=ngram
            )

            # Create detector
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
                print(
                    f"   5. Current memory needed: ~{error_msg.split('(')[1].split(')')[0]} GiB"
                )

            sys.exit(1)

    def create_base_llm(
        self, model_name: str, gpu_memory_utilization: float = 0.8
    ) -> Any:
        """Create and return the base vLLM LLM without watermark wrapping."""
        print("🚀 Initializing base model...")
        try:
            print(f"   Loading model: {model_name}")
            print(f"   GPU memory utilization: {gpu_memory_utilization}")
            llm = LLM(
                model=model_name,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=2048,
            )
            print("✅ Base model initialized")
            return llm
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Error initializing base model: {error_msg}")
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

    def wrap_llm_with_watermark(
        self,
        llm: Any,
        watermarking_algorithm: str,
        seed: int,
        ngram: int,
        detection_threshold: float,
    ) -> Tuple[Any, Any]:
        """Wrap an existing base LLM with the watermarking logic and create a detector."""
        print("🔧 Wrapping base model with watermarking...")
        watermark_algo = self.get_algorithm_enum(watermarking_algorithm)
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
        print("✅ Watermark wrapper and detector ready")
        return wm_llm, detector

    def extract_prompts(self, data: List[Dict[str, Any]], input_key: str) -> List[str]:
        """Extract prompts from data for batch processing."""
        prompts = []
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

    def generate_watermarked_text(
        self, wm_llm: Any, prompts: List[str], sampling_params: SamplingParams
    ) -> List[Any]:
        """Generate watermarked text for all prompts."""
        print(f"🎯 Generating watermarked text for {len(prompts)} prompts...")
        try:
            with tqdm(total=len(prompts), desc="Generating watermarked") as pbar:
                # Pass a callback so core can update this bar per prompt during serial generation
                outputs = wm_llm.generate(
                    prompts,
                    sampling_params,
                    progress_callback=pbar.update,
                )
            return outputs
        except Exception as e:
            print(f"❌ Error during watermarked generation: {e}")
            sys.exit(1)

    def generate_unwatermarked_text(
        self, llm: Any, prompts: List[str], sampling_params: SamplingParams
    ) -> List[Any]:
        """Generate unwatermarked text for all prompts using base LLM."""
        print(f"🎯 Generating UN-watermarked text for {len(prompts)} prompts...")
        try:
            with tqdm(total=len(prompts), desc="Generating unwatermarked") as pbar:
                outputs = llm.generate(prompts, sampling_params)
                pbar.update(len(prompts))
            return outputs
        except Exception as e:
            print(f"❌ Error during unwatermarked generation: {e}")
            sys.exit(1)

    def process_outputs_dual(
        self,
        data: List[Dict[str, Any]],
        unwm_outputs: List[Any],
        wm_outputs: List[Any],
        input_key: str,
        watermarked_output_key: str,
        unwatermarked_output_key: str,
        tokenizer: Any,
    ) -> Tuple[List[Dict[str, Any]], int, int, int]:
        """Process both un/watermarked outputs and create result data.

        Returns: (processed_data, total_tokens_overall, total_tokens_unwm, total_tokens_wm)
        """
        processed_data = []
        total_tokens_overall = 0
        total_tokens_unwm = 0
        total_tokens_wm = 0
        unwm_idx = 0
        wm_idx = 0

        print("🔄 Processing outputs (unwatermarked + watermarked)...")
        for item in tqdm(data, desc="Processing outputs"):
            new_item = item.copy()

            if input_key in item and item[input_key]:
                # Unwatermarked
                if unwm_idx < len(unwm_outputs):
                    unwm_output = unwm_outputs[unwm_idx]
                    unwm_text = unwm_output.outputs[0].text
                    new_item[unwatermarked_output_key] = unwm_text
                    tokens = self.count_tokens(unwm_text, tokenizer)
                    total_tokens_unwm += tokens
                    total_tokens_overall += tokens
                    unwm_idx += 1
                else:
                    new_item[unwatermarked_output_key] = ""

                # Watermarked
                if wm_idx < len(wm_outputs):
                    wm_output = wm_outputs[wm_idx]
                    wm_text = wm_output.outputs[0].text
                    new_item[watermarked_output_key] = wm_text
                    tokens = self.count_tokens(wm_text, tokenizer)
                    total_tokens_wm += tokens
                    total_tokens_overall += tokens
                    wm_idx += 1
                else:
                    new_item[watermarked_output_key] = ""
            else:
                new_item[unwatermarked_output_key] = ""
                new_item[watermarked_output_key] = ""

            processed_data.append(new_item)

        print(
            f"✅ Processed {len(processed_data)} items, total tokens (both): {total_tokens_overall}"
        )
        return processed_data, total_tokens_overall, total_tokens_unwm, total_tokens_wm

    def evaluate_detection_for_texts(
        self, texts: List[str], is_watermarked: bool, detector: Any
    ) -> Tuple[List[bool], List[bool]]:
        """Run detection on a list of texts and return predictions and true labels."""
        predictions: List[bool] = []
        true_labels: List[bool] = []
        desc = "watermarked" if is_watermarked else "unwatermarked"
        for text in tqdm(texts, desc=f"Testing {desc}"):
            try:
                detection_result = detector.detect(text)
                predictions.append(detection_result["is_watermarked"])
                true_labels.append(is_watermarked)
            except Exception as e:
                print(f"⚠️  Detection failed for {desc} text: {e}")
                predictions.append(False)
                true_labels.append(is_watermarked)
        return predictions, true_labels

    def evaluate_detection_performance_dual(
        self,
        processed_data: List[Dict[str, Any]],
        watermarked_key: str,
        unwatermarked_key: str,
        detector: Any,
    ) -> Tuple[List[bool], List[bool], int, int]:
        """Evaluate detection on both watermarked and unwatermarked texts from processed data."""
        print("🔍 Evaluating detection performance (both sets)...")

        predictions: List[bool] = []
        true_labels: List[bool] = []

        watermarked_texts = [
            item[watermarked_key]
            for item in processed_data
            if watermarked_key in item and item[watermarked_key]
        ]
        unwatermarked_texts = [
            item[unwatermarked_key]
            for item in processed_data
            if unwatermarked_key in item and item[unwatermarked_key]
        ]

        print(f"   Testing {len(watermarked_texts)} watermarked texts...")
        for text in tqdm(watermarked_texts, desc="Testing watermarked"):
            try:
                detection_result = detector.detect(text)
                predictions.append(detection_result["is_watermarked"])
                true_labels.append(True)
            except Exception as e:
                print(f"⚠️  Detection failed for watermarked text: {e}")
                predictions.append(False)
                true_labels.append(True)

        print(f"   Testing {len(unwatermarked_texts)} unwatermarked texts...")
        for text in tqdm(unwatermarked_texts, desc="Testing unwatermarked"):
            try:
                detection_result = detector.detect(text)
                predictions.append(detection_result["is_watermarked"])
                true_labels.append(False)
            except Exception as e:
                print(f"⚠️  Detection failed for unwatermarked text: {e}")
                predictions.append(False)
                true_labels.append(False)

        return (
            predictions,
            true_labels,
            len(watermarked_texts),
            len(unwatermarked_texts),
        )

    def calculate_metrics(
        self,
        predictions: List[bool],
        true_labels: List[bool],
        generation_time: float,
        detection_time: float,
        total_tokens: int,
        num_prompts: int,
    ) -> Dict[str, float | int]:
        """Calculate performance metrics including FPR/FNR."""
        total_time = generation_time + detection_time
        avg_time_per_example = generation_time / num_prompts if num_prompts > 0 else 0
        tokens_per_second = total_tokens / generation_time if generation_time > 0 else 0

        tp = fp = tn = fn = 0
        if len(predictions) == len(true_labels) and len(predictions) > 0:
            for pred, true in zip(predictions, true_labels):
                if pred and true:
                    tp += 1
                elif pred and not true:
                    fp += 1
                elif not pred and not true:
                    tn += 1
                elif not pred and true:
                    fn += 1

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (
            float(2 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )
        accuracy = (
            float((tp + tn) / (tp + tn + fp + fn)) if (tp + tn + fp + fn) > 0 else 0.0
        )
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        fnr = float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0

        return {
            "total_time": total_time,
            "generation_time": generation_time,
            "detection_time": detection_time,
            "avg_time_per_example": avg_time_per_example,
            "tokens_per_second": tokens_per_second,
            "total_tokens": total_tokens,
            "accuracy": accuracy,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "fpr": fpr,
            "fnr": fnr,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }

    def print_metrics(
        self,
        metrics: Dict[str, float],
        num_prompts: int,
        num_watermarked: int,
        num_unwatermarked: int,
        output_path: str,
        watermarking_algorithm: str,
        model_name: str,
    ):
        """Print comprehensive performance metrics."""
        print("\n" + "=" * 60)
        print("📊 PERFORMANCE METRICS")
        print("=" * 60)
        print(f"📝 Total examples processed: {num_prompts}")
        print(f"⏱️  Total generation time: {metrics['generation_time']:.2f} seconds")
        print(f"🔍 Total detection time: {metrics['detection_time']:.2f} seconds")
        print(f"⏰ Total time: {metrics['total_time']:.2f} seconds")
        print(
            f"📈 Average time per example: {metrics['avg_time_per_example']:.4f} seconds"
        )
        print(f"🎯 Total tokens generated: {int(metrics['total_tokens'])}")
        print(f"🚀 Tokens per second: {metrics['tokens_per_second']:.2f}")

        print("\n" + "=" * 60)
        print("🔍 DETECTION METRICS")
        print("=" * 60)
        print(f"🎯 Detection accuracy: {metrics['accuracy']:.4f}")
        print(f"📊 Detection F1 score: {metrics['f1_score']:.4f}")
        print(f"✨ Detection precision: {metrics['precision']:.4f}")
        print(f"🔄 Detection recall: {metrics['recall']:.4f}")
        print(f"🚫 False Positive Rate (FPR): {metrics['fpr']:.4f}")
        print(f"⚠️  False Negative Rate (FNR): {metrics['fnr']:.4f}")
        print(
            f"   TP: {int(metrics['tp'])}  FP: {int(metrics['fp'])}  TN: {int(metrics['tn'])}  FN: {int(metrics['fn'])}"
        )
        print(f"✅ Watermarked samples tested: {num_watermarked}")
        print(f"🧪 Unwatermarked samples tested: {num_unwatermarked}")

        print("\n" + "=" * 60)
        print("📋 SUMMARY")
        print("=" * 60)
        print(f"✅ Successfully processed {num_prompts} examples")
        print(f"💾 Output saved to: {output_path}")
        print(f"🔧 Watermarking algorithm: {watermarking_algorithm}")
        print(f"🤖 Model: {model_name}")
        print("=" * 60)

    def process_dataset(
        self,
        input_path: str,
        input_key: str,
        output_path: str,
        output_key: str,
        unwatermarked_output_key: str = "unwatermarked_text",
        watermarking_algorithm: str = "OPENAI",
        model_name: str = "meta-llama/Llama-3.2-1B",
        seed: int = 42,
        ngram: int = 2,
        max_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.95,
        detection_threshold: float = 0.05,
        max_examples: int | None = None,
        gpu_memory_utilization: float = 0.8,
    ):
        """
        Main processing function that orchestrates the entire workflow.

        Args:
            input_path: Path to input JSONL file
            input_key: Key in JSON to read text from
            output_path: Path to output JSONL file
            output_key: Key to add watermarked text to
            watermarking_algorithm: Watermarking algorithm (OPENAI, MARYLAND, PF)
            model_name: Model name to use
            seed: Random seed
            ngram: N-gram size for watermarking
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            detection_threshold: Detection threshold
            max_examples: Maximum number of examples to process
            gpu_memory_utilization: GPU memory utilization fraction
        """
        print("🌊 WATERMARKED DATASET GENERATION")
        print("=" * 60)
        print(f"📁 Input file: {input_path}")
        print(f"🔑 Input key: {input_key}")
        print(f"💾 Output file: {output_path}")
        print(f"🏷️  Watermarked output key: {output_key}")
        print(f"🏷️  Unwatermarked output key: {unwatermarked_output_key}")
        print(f"🔧 Algorithm: {watermarking_algorithm}")
        print(f"🤖 Model: {model_name}")
        print(f"🎲 Seed: {seed}")
        print(f"📊 N-gram: {ngram}")
        print(f"📝 Max tokens: {max_tokens}")
        print(f"🌡️  Temperature: {temperature}")
        print(f"🎯 Top-p: {top_p}")
        print(f"🚨 Detection threshold: {detection_threshold}")
        print(f"🎮 GPU memory utilization: {gpu_memory_utilization}")
        if max_examples:
            print(f"📊 Max examples: {max_examples}")
        print("=" * 60)

        # Load and validate data
        data = self.load_and_validate_data(input_path, input_key, max_examples)

        # Create base LLM ONLY (do NOT wrap yet)
        llm = self.create_base_llm(
            model_name=model_name,
            gpu_memory_utilization=gpu_memory_utilization,
        )

        # Get tokenizer for token counting
        tokenizer = llm.get_tokenizer()

        # Sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature, top_p=top_p, max_tokens=max_tokens
        )

        # Extract prompts
        prompts = self.extract_prompts(data, input_key)

        # Generate UNWATERMARKED text first (as requested)
        unwm_start_time = time.time()
        unwm_outputs = self.generate_unwatermarked_text(llm, prompts, sampling_params)
        unwm_generation_time = time.time() - unwm_start_time

        # Then wrap the base LLM and generate WATERMARKED text
        wm_llm, detector = self.wrap_llm_with_watermark(
            llm=llm,
            watermarking_algorithm=watermarking_algorithm,
            seed=seed,
            ngram=ngram,
            detection_threshold=detection_threshold,
        )
        # Generate watermarked text (core will enforce serialization for OpenAI-style)
        wm_start_time = time.time()
        wm_outputs = self.generate_watermarked_text(wm_llm, prompts, sampling_params)
        wm_generation_time = time.time() - wm_start_time

        # Process outputs
        processed_data, total_tokens, total_tokens_unwm, total_tokens_wm = (
            self.process_outputs_dual(
                data,
                unwm_outputs,
                wm_outputs,
                input_key,
                output_key,
                unwatermarked_output_key,
                tokenizer,
            )
        )

        # Save processed data
        try:
            self.save_jsonl(processed_data, output_path)
            print(f"✅ Saved {len(processed_data)} examples to {output_path}")
        except Exception as e:
            print(f"❌ Error saving output file: {e}")
            sys.exit(1)

        # Prepare text sets for separate evaluation
        watermarked_texts = [
            item[output_key]
            for item in processed_data
            if output_key in item and item[output_key]
        ]
        unwatermarked_texts = [
            item[unwatermarked_output_key]
            for item in processed_data
            if unwatermarked_output_key in item and item[unwatermarked_output_key]
        ]

        # Evaluate detection performance separately
        wm_detection_start = time.time()
        preds_wm, labels_wm = self.evaluate_detection_for_texts(
            watermarked_texts, True, detector
        )
        wm_detection_time = time.time() - wm_detection_start

        unwm_detection_start = time.time()
        preds_unwm, labels_unwm = self.evaluate_detection_for_texts(
            unwatermarked_texts, False, detector
        )
        unwm_detection_time = time.time() - unwm_detection_start

        # Calculate metrics separately
        metrics_wm = self.calculate_metrics(
            preds_wm,
            labels_wm,
            wm_generation_time,
            wm_detection_time,
            total_tokens_wm,
            len(watermarked_texts),
        )
        metrics_unwm = self.calculate_metrics(
            preds_unwm,
            labels_unwm,
            unwm_generation_time,
            unwm_detection_time,
            total_tokens_unwm,
            len(unwatermarked_texts),
        )

        # Print metrics separately
        print("\n" + "=" * 60)
        print("📊 PERFORMANCE METRICS (SEPARATE)")
        print("=" * 60)
        print("UNWATERMARKED:")
        print(
            f"  ⏱️  Gen time: {metrics_unwm['generation_time']:.2f}s  🔍 Det time: {metrics_unwm['detection_time']:.2f}s"
        )
        print(
            f"  🎯 Tokens: {int(metrics_unwm['total_tokens'])}  🚀 TPS: {metrics_unwm['tokens_per_second']:.2f}"
        )
        print("WATERMARKED:")
        print(
            f"  ⏱️  Gen time: {metrics_wm['generation_time']:.2f}s  🔍 Det time: {metrics_wm['detection_time']:.2f}s"
        )
        print(
            f"  🎯 Tokens: {int(metrics_wm['total_tokens'])}  🚀 TPS: {metrics_wm['tokens_per_second']:.2f}"
        )

        print("\n" + "=" * 60)
        print("🔍 DETECTION METRICS (SEPARATE)")
        print("=" * 60)
        print("UNWATERMARKED (should be False):")
        print(
            f"  Acc: {metrics_unwm['accuracy']:.4f}  F1: {metrics_unwm['f1_score']:.4f}  Pre: {metrics_unwm['precision']:.4f}  Rec: {metrics_unwm['recall']:.4f}  FPR: {metrics_unwm['fpr']:.4f}  FNR: {metrics_unwm['fnr']:.4f}"
        )
        print(
            f"  TP: {int(metrics_unwm['tp'])}  FP: {int(metrics_unwm['fp'])}  TN: {int(metrics_unwm['tn'])}  FN: {int(metrics_unwm['fn'])}  N: {len(unwatermarked_texts)}"
        )
        print("WATERMARKED (should be True):")
        print(
            f"  Acc: {metrics_wm['accuracy']:.4f}  F1: {metrics_wm['f1_score']:.4f}  Pre: {metrics_wm['precision']:.4f}  Rec: {metrics_wm['recall']:.4f}  FPR: {metrics_wm['fpr']:.4f}  FNR: {metrics_wm['fnr']:.4f}"
        )
        print(
            f"  TP: {int(metrics_wm['tp'])}  FP: {int(metrics_wm['fp'])}  TN: {int(metrics_wm['tn'])}  FN: {int(metrics_wm['fn'])}  N: {len(watermarked_texts)}"
        )

        print("\n" + "=" * 60)
        print("📋 SUMMARY")
        print("=" * 60)
        print(f"✅ Successfully processed {len(prompts)} examples")
        print(f"💾 Output saved to: {output_path}")
        print(f"🔧 Watermarking algorithm: {watermarking_algorithm}")
        print(f"🤖 Model: {model_name}")
        print("=" * 60)


def main():
    """Entry point for Fire CLI."""
    processor = WatermarkDatasetProcessor()
    fire.Fire(processor.process_dataset)


if __name__ == "__main__":
    main()
