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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
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
            # Add progress bar for generation
            with tqdm(total=len(prompts), desc="Generating text") as pbar:
                outputs = wm_llm.generate(prompts, sampling_params)
                pbar.update(len(prompts))
            return outputs
        except Exception as e:
            print(f"❌ Error during generation: {e}")
            sys.exit(1)

    def process_outputs(
        self,
        data: List[Dict[str, Any]],
        outputs: List[Any],
        input_key: str,
        output_key: str,
        tokenizer: Any,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Process outputs and create result data."""
        processed_data = []
        total_tokens = 0
        output_idx = 0

        print("🔄 Processing outputs...")
        for item in tqdm(data, desc="Processing outputs"):
            new_item = item.copy()

            if input_key in item and item[input_key]:
                if output_idx < len(outputs):
                    output = outputs[output_idx]
                    generated_text = output.outputs[0].text
                    new_item[output_key] = generated_text

                    # Count tokens
                    tokens = self.count_tokens(generated_text, tokenizer)
                    total_tokens += tokens

                    output_idx += 1
                else:
                    new_item[output_key] = ""
            else:
                new_item[output_key] = ""

            processed_data.append(new_item)

        print(
            f"✅ Processed {len(processed_data)} items, generated {total_tokens} tokens"
        )
        return processed_data, total_tokens

    def evaluate_detection_performance(
        self, processed_data: List[Dict[str, Any]], output_key: str, detector: Any
    ) -> Tuple[List[bool], List[bool], int]:
        """Evaluate detection performance on watermarked and non-watermarked text."""
        print("🔍 Evaluating detection performance...")

        predictions = []
        true_labels = []
        watermarked_count = 0

        # Test watermarked texts
        watermarked_texts = [
            item[output_key]
            for item in processed_data
            if output_key in item and item[output_key]
        ]
        watermarked_count = len(watermarked_texts)

        print(f"   Testing {watermarked_count} watermarked texts...")
        for text in tqdm(watermarked_texts, desc="Testing watermarked"):
            try:
                detection_result = detector.detect(text)
                predictions.append(detection_result["is_watermarked"])
                true_labels.append(True)  # All generated texts should be watermarked
            except Exception as e:
                print(f"⚠️  Detection failed for watermarked text: {e}")
                predictions.append(False)
                true_labels.append(True)

        # Test with non-watermarked text for false positive rate
        non_watermarked_samples = [
            "This is a test sentence that was not generated with watermarking.",
            "Another example of non-watermarked text for testing purposes.",
            "Random text without any watermarking applied to it.",
            "Plain text that should not be detected as watermarked.",
            "Simple sentence for baseline comparison testing.",
        ]

        print(f"   Testing {len(non_watermarked_samples)} non-watermarked texts...")
        for text in tqdm(non_watermarked_samples, desc="Testing non-watermarked"):
            try:
                detection_result = detector.detect(text)
                predictions.append(detection_result["is_watermarked"])
                true_labels.append(False)  # These should not be detected as watermarked
            except Exception as e:
                print(f"⚠️  Detection failed for non-watermarked text: {e}")
                predictions.append(False)
                true_labels.append(False)

        return predictions, true_labels, watermarked_count

    def calculate_metrics(
        self,
        predictions: List[bool],
        true_labels: List[bool],
        generation_time: float,
        detection_time: float,
        total_tokens: int,
        num_prompts: int,
    ) -> Dict[str, float | int]:
        """Calculate and return all performance metrics."""
        total_time = generation_time + detection_time
        avg_time_per_example = generation_time / num_prompts if num_prompts > 0 else 0
        tokens_per_second = total_tokens / generation_time if generation_time > 0 else 0

        # Detection metrics
        if len(predictions) > 0 and len(set(true_labels)) > 1:  # Need both classes
            try:
                accuracy = float(accuracy_score(true_labels, predictions))
                f1 = float(f1_score(true_labels, predictions))
                precision = float(
                    precision_score(true_labels, predictions, zero_division=0)
                )
                recall = float(recall_score(true_labels, predictions, zero_division=0))
            except Exception as e:
                print(f"⚠️  Error calculating metrics: {e}")
                accuracy = f1 = precision = recall = 0.0
        else:
            accuracy = f1 = precision = recall = 0.0

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
        }

    def print_metrics(
        self,
        metrics: Dict[str, float],
        num_prompts: int,
        num_watermarked: int,
        num_non_watermarked: int,
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
        print(f"✅ Watermarked samples tested: {num_watermarked}")
        print(f"❌ Non-watermarked samples tested: {num_non_watermarked}")

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
        print(f"🏷️  Output key: {output_key}")
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

        # Initialize models
        llm, wm_llm, detector = self.initialize_models(
            model_name,
            watermarking_algorithm,
            seed,
            ngram,
            detection_threshold,
            gpu_memory_utilization,
        )

        # Get tokenizer for token counting
        tokenizer = llm.get_tokenizer()

        # Sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature, top_p=top_p, max_tokens=max_tokens
        )

        # Extract prompts
        prompts = self.extract_prompts(data, input_key)

        # Generate watermarked text
        start_time = time.time()
        outputs = self.generate_watermarked_text(wm_llm, prompts, sampling_params)
        generation_time = time.time() - start_time

        # Process outputs
        processed_data, total_tokens = self.process_outputs(
            data, outputs, input_key, output_key, tokenizer
        )

        # Save processed data
        try:
            self.save_jsonl(processed_data, output_path)
            print(f"✅ Saved {len(processed_data)} examples to {output_path}")
        except Exception as e:
            print(f"❌ Error saving output file: {e}")
            sys.exit(1)

        # Evaluate detection performance
        detection_start_time = time.time()
        predictions, true_labels, num_watermarked = self.evaluate_detection_performance(
            processed_data, output_key, detector
        )
        detection_time = time.time() - detection_start_time

        # Calculate metrics
        metrics = self.calculate_metrics(
            predictions,
            true_labels,
            generation_time,
            detection_time,
            total_tokens,
            len(prompts),
        )

        # Count samples for reporting
        num_non_watermarked = 5  # Fixed number of non-watermarked test samples

        # Print metrics
        self.print_metrics(
            metrics,
            len(prompts),
            num_watermarked,
            num_non_watermarked,
            output_path,
            watermarking_algorithm,
            model_name,
        )


def main():
    """Entry point for Fire CLI."""
    processor = WatermarkDatasetProcessor()
    fire.Fire(processor.process_dataset)


if __name__ == "__main__":
    main()
