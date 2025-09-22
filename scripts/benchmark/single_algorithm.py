#!/usr/bin/env python3
"""
Single algorithm runner for subprocess execution.
This module handles running a single watermarking algorithm in isolation.
"""

import json
import os
import sys
import time
from dataclasses import dataclass
from typing import List

# Set required environment variables
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from config import ALGORITHM_CONFIGS
from utils import calculate_metrics, generate_texts, initialize_model, run_detection
from vllm import SamplingParams

from vllm_watermark.core import WatermarkedLLMs
from vllm_watermark.watermark_detectors import WatermarkDetectors


@dataclass
class BenchmarkResult:
    """Results from benchmarking a single algorithm."""

    algorithm_name: str
    config_string: str
    precision: float
    recall: float
    f1: float
    fpr: float
    fnr: float
    accuracy: float
    input_tokens_per_second: float
    output_tokens_per_second: float
    generation_time: float
    detection_time: float
    total_time: float
    total_input_tokens: int
    total_output_tokens: int
    num_samples: int
    tp: int
    fp: int
    tn: int
    fn: int


def run_single_algorithm_benchmark(
    algorithm_name: str,
    model_name: str,
    prompts: List[str],
    sampling_params: SamplingParams,
    detection_threshold: float,
    gpu_memory_utilization: float,
    output_dir: str = "/tmp",
) -> BenchmarkResult:
    """Run benchmark for a single algorithm with progress reporting."""

    # Initialize progress tracking
    progress_file = os.path.join(output_dir, f"{algorithm_name}_progress.json")
    start_time = time.time()

    def update_progress(stage: str, details: dict = None):
        """Update progress file."""
        try:
            progress_data = {
                "algorithm": algorithm_name,
                "stage": stage,
                "timestamp": time.time(),
                "elapsed_time": time.time() - start_time,
                "details": details or {},
            }
            with open(progress_file, "w") as f:
                json.dump(progress_data, f)
        except Exception:
            pass

    try:
        update_progress("starting")
        print(f"🔬 Benchmarking {algorithm_name} algorithm...")

        # Get algorithm config
        if algorithm_name not in ALGORITHM_CONFIGS:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")

        algorithm_config = ALGORITHM_CONFIGS[algorithm_name]
        print(f"   Configuration: {algorithm_config.get_config_string()}")

        # Initialize model
        update_progress("initializing_model")
        llm = initialize_model(model_name, gpu_memory_utilization)
        tokenizer = llm.get_tokenizer()

        # Step 1: Generate unwatermarked text (baseline) - MUST be done first before wrapping
        update_progress("generating_unwatermarked", {"total_prompts": len(prompts)})
        unwm_outputs, unwm_generation_time = generate_texts(
            llm, prompts, sampling_params, is_watermarked=False
        )

        # Step 2: Create watermarked LLM (this modifies the original LLM)
        update_progress("creating_watermark")
        print("   🌊 Creating watermarked LLM...")
        wm_llm = WatermarkedLLMs.create(
            llm, algo=algorithm_config.algorithm, **algorithm_config.params
        )

        # Step 3: Generate watermarked text
        update_progress("generating_watermarked", {"total_prompts": len(prompts)})
        wm_outputs, wm_generation_time = generate_texts(
            wm_llm, prompts, sampling_params, is_watermarked=True
        )

        # Step 4: Create detector
        update_progress("creating_detector")
        print("   🔍 Creating detector...")
        detector = WatermarkDetectors.create(
            algo=algorithm_config.detection_algorithm,
            model=llm,  # Use original llm reference for detector
            threshold=detection_threshold,
            **algorithm_config.params,
        )

        # Step 5: Extract generated texts
        update_progress("extracting_texts")
        unwm_texts = [output.outputs[0].text for output in unwm_outputs]
        wm_texts = [output.outputs[0].text for output in wm_outputs]

        # Count tokens
        total_input_tokens = sum(len(tokenizer.encode(prompt)) for prompt in prompts)
        total_unwm_tokens = sum(len(tokenizer.encode(text)) for text in unwm_texts)
        total_wm_tokens = sum(len(tokenizer.encode(text)) for text in wm_texts)
        total_output_tokens = total_unwm_tokens + total_wm_tokens

        # Step 6: Run detection
        update_progress("detecting_unwatermarked", {"total_texts": len(unwm_texts)})
        unwm_detection_start = time.time()
        unwm_predictions = run_detection(unwm_texts, detector, is_watermarked=False)
        unwm_detection_time = time.time() - unwm_detection_start

        update_progress("detecting_watermarked", {"total_texts": len(wm_texts)})
        wm_detection_start = time.time()
        wm_predictions = run_detection(wm_texts, detector, is_watermarked=True)
        wm_detection_time = time.time() - wm_detection_start

        # Step 7: Calculate metrics
        update_progress("calculating_metrics")
        total_detection_time = unwm_detection_time + wm_detection_time
        total_generation_time = unwm_generation_time + wm_generation_time

        metrics = calculate_metrics(unwm_predictions, wm_predictions)

        # Calculate throughput
        input_tokens_per_second = (
            total_input_tokens / total_generation_time
            if total_generation_time > 0
            else 0.0
        )
        output_tokens_per_second = (
            total_output_tokens / total_generation_time
            if total_generation_time > 0
            else 0.0
        )

        update_progress("completed")

        return BenchmarkResult(
            algorithm_name=algorithm_config.name,
            config_string=algorithm_config.get_config_string(),
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1=metrics["f1"],
            fpr=metrics["fpr"],
            fnr=metrics["fnr"],
            accuracy=metrics["accuracy"],
            input_tokens_per_second=input_tokens_per_second,
            output_tokens_per_second=output_tokens_per_second,
            generation_time=total_generation_time,
            detection_time=total_detection_time,
            total_time=total_generation_time + total_detection_time,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            num_samples=len(prompts),
            tp=metrics["tp"],
            fp=metrics["fp"],
            tn=metrics["tn"],
            fn=metrics["fn"],
        )

    except Exception as e:
        update_progress("failed", {"error": str(e)})
        raise


def main():
    """Entry point for single algorithm execution."""
    if len(sys.argv) != 3:
        print("Usage: python single_algorithm.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Extract output directory from input file path
    output_dir = os.path.dirname(input_file)

    # Load parameters
    with open(input_file, "r") as f:
        data = json.load(f)

    prompts = data["prompts"]
    sampling_params_dict = data["sampling_params"]
    algorithm_name = data["algorithm_name"]
    model_name = data["model_name"]
    detection_threshold = data["detection_threshold"]
    gpu_memory_utilization = data["gpu_memory_utilization"]

    # Force vLLM V0 for MARYLAND_L (logit processors unsupported in V1)
    if algorithm_name == "MARYLAND_L":
        os.environ["VLLM_USE_V1"] = "0"

    # Recreate sampling params
    sampling_params = SamplingParams(**sampling_params_dict)

    # Run benchmark with progress tracking
    result = run_single_algorithm_benchmark(
        algorithm_name=algorithm_name,
        model_name=model_name,
        prompts=prompts,
        sampling_params=sampling_params,
        detection_threshold=detection_threshold,
        gpu_memory_utilization=gpu_memory_utilization,
        output_dir=output_dir,  # Pass output directory for progress tracking
    )

    # Save results
    result_dict = {
        "algorithm_name": result.algorithm_name,
        "config_string": result.config_string,
        "precision": result.precision,
        "recall": result.recall,
        "f1": result.f1,
        "fpr": result.fpr,
        "fnr": result.fnr,
        "accuracy": result.accuracy,
        "input_tokens_per_second": result.input_tokens_per_second,
        "output_tokens_per_second": result.output_tokens_per_second,
        "generation_time": result.generation_time,
        "detection_time": result.detection_time,
        "total_time": result.total_time,
        "total_input_tokens": result.total_input_tokens,
        "total_output_tokens": result.total_output_tokens,
        "num_samples": result.num_samples,
        "tp": result.tp,
        "fp": result.fp,
        "tn": result.tn,
        "fn": result.fn,
    }

    with open(output_file, "w") as f:
        json.dump(result_dict, f)


if __name__ == "__main__":
    main()
