"""
Utility functions for watermarking benchmarking.
"""

import json
import os
import time
from typing import Any, Dict, List

from tqdm import tqdm
from vllm import LLM, SamplingParams

from vllm_watermark.core import DetectionAlgorithm, WatermarkingAlgorithm


def load_jsonl(file_path: str, max_samples: int = None) -> List[Dict[str, Any]]:
    """Load data from a JSONL file."""
    print(f"📁 Loading dataset from: {file_path}")

    if not os.path.exists(file_path):
        print(f"❌ Error: Dataset file not found at {file_path}")
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, desc="Loading dataset")):
            if max_samples and i >= max_samples:
                break

            line = line.strip()
            if line:
                data.append(json.loads(line))

    print(f"✅ Loaded {len(data)} samples from dataset")
    return data


def initialize_model(model_name: str, gpu_memory_utilization: float = 0.4) -> LLM:
    """Initialize the base LLM model."""
    print(f"🚀 Initializing model: {model_name}")
    print(f"   GPU memory utilization: {gpu_memory_utilization}")

    try:
        llm = LLM(
            model=model_name,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=2048,
        )
        print("✅ Model initialized successfully")
        return llm
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error initializing model: {error_msg}")

        if "Free memory" in error_msg and "GPU memory utilization" in error_msg:
            print("\n💡 GPU Memory Troubleshooting:")
            print("   This error indicates insufficient GPU memory.")
            print("   Try reducing GPU memory utilization:")
            print("   --gpu_memory_utilization 0.3")
            print("   or use fewer samples: --num_samples 1000")

        raise


def get_algorithm_detection_mapping():
    """Get mapping from watermarking to detection algorithms."""
    return {
        WatermarkingAlgorithm.OPENAI: DetectionAlgorithm.OPENAI_Z,
        WatermarkingAlgorithm.OPENAI_DR: DetectionAlgorithm.OPENAI_Z,
        WatermarkingAlgorithm.MARYLAND: DetectionAlgorithm.MARYLAND_Z,
        WatermarkingAlgorithm.MARYLAND_L: DetectionAlgorithm.MARYLAND_Z,
        WatermarkingAlgorithm.PF: DetectionAlgorithm.PF,
    }


def count_tokens(text: str, tokenizer) -> int:
    """Count tokens in text using the model's tokenizer."""
    try:
        tokens = tokenizer.encode(text)
        return len(tokens)
    except Exception:
        # Fallback to word count approximation
        return len(text.split())


def extract_prompts(data: List[Dict[str, Any]], input_key: str) -> List[str]:
    """Extract prompts from data."""
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
        raise ValueError(f"No valid prompts found with key '{input_key}'")

    print(f"✅ Extracted {len(prompts)} valid prompts")
    return prompts


def generate_texts(
    llm: LLM,
    prompts: List[str],
    sampling_params: SamplingParams,
    is_watermarked: bool = False,
):
    """Generate texts using LLM."""
    text_type = "watermarked" if is_watermarked else "unwatermarked"
    print(f"🎯 Generating {text_type} text for {len(prompts)} prompts...")

    try:
        start_time = time.time()
        with tqdm(total=len(prompts), desc=f"Generating {text_type}") as pbar:
            if hasattr(llm, "generate"):
                # Handle both watermarked and regular LLM
                if is_watermarked and hasattr(llm, "progress_callback"):
                    outputs = llm.generate(
                        prompts, sampling_params, progress_callback=pbar.update
                    )
                else:
                    outputs = llm.generate(prompts, sampling_params)
                    pbar.update(len(prompts))
            else:
                raise AttributeError("LLM object doesn't have generate method")

        generation_time = time.time() - start_time
        return outputs, generation_time

    except Exception as e:
        print(f"❌ Error during {text_type} generation: {e}")
        raise


def run_detection(texts: List[str], detector, is_watermarked: bool) -> List[bool]:
    """Run detection on a list of texts."""
    predictions = []
    text_type = "watermarked" if is_watermarked else "unwatermarked"

    print(f"🔍 Testing detection on {text_type} text...")
    for text in tqdm(texts, desc=f"Detecting {text_type}"):
        try:
            detection_result = detector.detect(text)
            predictions.append(detection_result["is_watermarked"])
        except Exception as e:
            print(f"⚠️  Detection failed: {e}")
            predictions.append(False)  # Default to not watermarked

    return predictions


def calculate_metrics(
    unwm_predictions: List[bool], wm_predictions: List[bool]
) -> Dict[str, float]:
    """Calculate all performance metrics from predictions."""
    # Calculate metrics from separate sets for clarity
    # Unwatermarked texts: should be detected as NOT watermarked (negatives)
    unwm_correct = sum(1 for pred in unwm_predictions if not pred)  # True Negatives
    unwm_incorrect = sum(1 for pred in unwm_predictions if pred)  # False Positives

    # Watermarked texts: should be detected as watermarked (positives)
    wm_correct = sum(1 for pred in wm_predictions if pred)  # True Positives
    wm_incorrect = sum(1 for pred in wm_predictions if not pred)  # False Negatives

    # Standard confusion matrix notation
    tp = wm_correct  # Correctly detected watermarks
    fp = unwm_incorrect  # Incorrectly flagged as watermarked
    tn = unwm_correct  # Correctly identified as not watermarked
    fn = wm_incorrect  # Missed watermarks

    # Calculate standard ML metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    # Calculate FPR and FNR from their respective sets (more intuitive)
    fpr = (
        unwm_incorrect / len(unwm_predictions) if len(unwm_predictions) > 0 else 0.0
    )  # False alarms on clean text
    fnr = (
        wm_incorrect / len(wm_predictions) if len(wm_predictions) > 0 else 0.0
    )  # Missed watermarks

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "fpr": fpr,
        "fnr": fnr,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }
