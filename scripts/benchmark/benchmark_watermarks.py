#!/usr/bin/env python3
"""
Simplified Watermarking Benchmarking Script

This script coordinates benchmarking of multiple watermarking algorithms
using isolated processes for complete memory cleanup.
"""

import csv
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Union

import fire
from tabulate import tabulate

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from config import ALGORITHM_CONFIGS
from single_algorithm import BenchmarkResult
from utils import extract_prompts, load_jsonl


class WatermarkBenchmarker:
    """Main benchmarking class for watermarking algorithms."""

    def run_single_algorithm(
        self,
        algorithm_name: str,
        model_name: str,
        prompts: List[str],
        sampling_params_dict: Dict[str, Any],
        detection_threshold: float,
        gpu_memory_utilization: float,
        output_dir: str,
    ) -> BenchmarkResult:
        """Run a single algorithm in an isolated process for complete memory cleanup."""

        # Save prompts to input file for subprocess
        input_file = os.path.join(output_dir, f"{algorithm_name}_input.json")
        os.makedirs(output_dir, exist_ok=True)

        with open(input_file, "w") as f:
            json.dump(
                {
                    "prompts": prompts,
                    "sampling_params": sampling_params_dict,
                    "algorithm_name": algorithm_name,
                    "model_name": model_name,
                    "detection_threshold": detection_threshold,
                    "gpu_memory_utilization": gpu_memory_utilization,
                },
                f,
            )

        # Results file
        results_file = os.path.join(output_dir, f"{algorithm_name}_results.json")

        # Run subprocess
        single_algorithm_script = os.path.join(
            os.path.dirname(__file__), "single_algorithm.py"
        )
        cmd = [sys.executable, single_algorithm_script, input_file, results_file]

        print(f"🔄 Running {algorithm_name} in isolated process...")

        # Create progress file for monitoring
        progress_file = os.path.join(output_dir, f"{algorithm_name}_progress.json")

        try:
            # Run subprocess with real-time output streaming
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=os.environ,  # Ensure environment variables are propagated
            )

            # Monitor progress in real-time
            last_output_time = time.time()
            timeout_seconds = 600  # 5 minutes timeout per stage

            print(f"   📡 Monitoring progress (PID: {process.pid})...")

            for line in process.stdout:
                line = line.strip()
                if line:
                    print(f"   📋 {line}")
                    last_output_time = time.time()

                # Check if process is still alive and hasn't been silent too long
                if process.poll() is None:  # Process still running
                    if time.time() - last_output_time > timeout_seconds:
                        print(
                            f"   ⚠️  No output for {timeout_seconds}s, process may be stuck"
                        )
                        print(f"   🔍 Check progress file: {progress_file}")

                # Check for progress file updates
                if os.path.exists(progress_file):
                    try:
                        with open(progress_file, "r") as f:
                            progress_data = json.load(f)
                        stage = progress_data.get("stage", "unknown")
                        elapsed = progress_data.get("elapsed_time", 0)
                        print(f"   🔄 Stage: {stage} (elapsed: {elapsed:.1f}s)")
                    except Exception:
                        pass

            # Wait for completion and check return code
            return_code = process.wait()
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, cmd)

            print(f"✅ {algorithm_name} process completed successfully")

        except subprocess.CalledProcessError as e:
            print(f"❌ {algorithm_name} process failed with return code {e.returncode}")
            # Try to get any remaining output
            if hasattr(e, "stdout") and e.stdout:
                print(f"   Output: {e.stdout}")
            raise e
        except Exception as e:
            print(f"❌ {algorithm_name} process failed: {e}")
            if "process" in locals() and process.poll() is None:
                print("   🛑 Terminating stuck process...")
                process.terminate()
                process.wait(timeout=10)
                if process.poll() is None:
                    process.kill()  # Force kill if terminate didn't work
            raise e
        finally:
            # Clean up progress file
            if os.path.exists(progress_file):
                try:
                    os.remove(progress_file)
                except Exception:
                    pass

        # Load results
        if not os.path.exists(results_file):
            raise RuntimeError(f"Results file not found: {results_file}")

        with open(results_file, "r") as f:
            result_data = json.load(f)

        # Keep the files for debugging/inspection (don't delete)
        print(f"   📁 Input saved: {input_file}")
        print(f"   📁 Results saved: {results_file}")

        # Convert back to BenchmarkResult
        return BenchmarkResult(**result_data)

    def run_benchmark(
        self,
        model_name: str = "meta-llama/Llama-3.2-1B",
        algorithms: Union[str, List[str]] = None,
        num_samples: int = 5000,
        data_path: str = "resources/datasets/c4/processed_c4.jsonl",
        input_key: str = "prompt",
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        detection_threshold: float = 0.01,
        gpu_memory_utilization: float = 0.4,
        output_dir: str = "output/benchmark",
    ) -> List[BenchmarkResult]:
        """Run comprehensive benchmark on specified algorithms."""

        if algorithms is None:
            algorithms = ["OPENAI", "MARYLAND", "MARYLAND_L", "PF"]
        elif isinstance(algorithms, str):
            # Parse space-separated string into list
            algorithms = algorithms.split()

        print("🌊 vLLM-Watermark Benchmarking Suite")
        print("=" * 60)
        print(f"🤖 Model: {model_name}")
        print(f"🔧 Algorithms: {', '.join(algorithms)}")
        print(f"📊 Samples: {num_samples}")
        print(f"📝 Max tokens: {max_tokens}")
        print("🎲 Seed: 42")
        print(f"🌡️  Temperature: {temperature}")
        print(f"🎯 Top-p: {top_p}")
        print(f"🚨 Detection threshold: {detection_threshold}")
        print("=" * 60)

        # Validate algorithms
        invalid_algorithms = [
            algo for algo in algorithms if algo not in ALGORITHM_CONFIGS
        ]
        if invalid_algorithms:
            print(f"❌ Error: Invalid algorithms: {invalid_algorithms}")
            print(f"   Available algorithms: {list(ALGORITHM_CONFIGS.keys())}")
            sys.exit(1)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Load dataset
        data = load_jsonl(data_path, num_samples)

        # Extract prompts once
        prompts = extract_prompts(data, input_key)

        # Create sampling parameters dictionary for subprocess communication
        sampling_params_dict = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }

        # Run benchmarks using isolated processes for complete memory cleanup
        results = []

        for i, algorithm_name in enumerate(algorithms):
            try:
                print(f"\n{'='*60}")
                print(f"🔄 Algorithm {i+1}/{len(algorithms)}: {algorithm_name}")
                print(f"{'='*60}")

                result = self.run_single_algorithm(
                    algorithm_name=algorithm_name,
                    model_name=model_name,
                    prompts=prompts,
                    sampling_params_dict=sampling_params_dict,
                    detection_threshold=detection_threshold,
                    gpu_memory_utilization=gpu_memory_utilization,
                    output_dir=output_dir,
                )
                results.append(result)
                print(f"✅ {algorithm_name} benchmark completed")

            except Exception as e:
                print(f"❌ Error benchmarking {algorithm_name}: {e}")
                continue

        # Display results table
        self.display_results_table(results)

        # Save results to CSV
        csv_path = os.path.join(output_dir, f"benchmark_results_{int(time.time())}.csv")
        self.save_results_to_csv(results, csv_path, model_name, num_samples)

        print(f"\n💾 Results saved to: {csv_path}")
        print("🎉 Benchmarking completed!")

        return results

    def display_results_table(self, results: List[BenchmarkResult]):
        """Display results in a nice table format."""
        if not results:
            print("❌ No results to display")
            return

        print("\n" + "=" * 120)
        print("📊 BENCHMARK RESULTS")
        print("=" * 120)

        # Prepare table data
        headers = [
            "Algorithm",
            "Configuration",
            "Precision",
            "Recall",
            "F1-Score",
            "Accuracy",
            "FPR",
            "FNR",
            "Generation TPS",
            "Total Time (s)",
        ]

        table_data = []
        for result in results:
            # Calculate combined generation TPS (input + output tokens)
            total_gen_tokens = result.total_input_tokens + result.total_output_tokens
            combined_tps = (
                total_gen_tokens / result.generation_time
                if result.generation_time > 0
                else 0.0
            )

            row = [
                result.algorithm_name,
                result.config_string,
                f"{result.precision:.3f}",
                f"{result.recall:.3f}",
                f"{result.f1:.3f}",
                f"{result.accuracy:.3f}",
                f"{result.fpr:.3f}",
                f"{result.fnr:.3f}",
                f"{combined_tps:.1f}",
                f"{result.total_time:.1f}",
            ]
            table_data.append(row)

        print(tabulate(table_data, headers=headers, tablefmt="grid"))

        # Additional details
        print("\n" + "=" * 80)
        print("📋 DETAILED METRICS & EXPLANATION")
        print("=" * 80)
        print("Metrics Explanation:")
        print("• Precision: TP/(TP+FP) - Of detected watermarks, how many were real?")
        print("• Recall: TP/(TP+FN) - Of real watermarks, how many were detected?")
        print("• F1-Score: Harmonic mean of Precision and Recall")
        print("• Accuracy: (TP+TN)/(TP+TN+FP+FN) - Overall correctness")
        print("• FPR: False alarms on clean text (lower is better)")
        print("• FNR: Missed watermarks (lower is better)")
        print()

        for result in results:
            print(f"🔬 {result.algorithm_name}:")
            print(f"   Configuration: {result.config_string}")
            print(
                f"   Confusion Matrix - TP: {result.tp}, FP: {result.fp}, TN: {result.tn}, FN: {result.fn}"
            )
            print(
                f"   Performance - Gen: {result.generation_time:.1f}s, Det: {result.detection_time:.1f}s"
            )
            print(
                f"   Tokens - Input: {result.total_input_tokens:,}, Output: {result.total_output_tokens:,}"
            )
            print(f"   Samples: {result.num_samples:,} processed")
            print()

    def save_results_to_csv(
        self,
        results: List[BenchmarkResult],
        csv_path: str,
        model_name: str,
        num_samples: int,
    ):
        """Save results to CSV file for easy import to Google Sheets."""
        fieldnames = [
            "algorithm",
            "configuration",
            "precision",
            "recall",
            "f1_score",
            "accuracy",
            "false_positive_rate",
            "false_negative_rate",
            "generation_tokens_per_second",
            "generation_time_seconds",
            "detection_time_seconds",
            "total_time_seconds",
            "total_input_tokens",
            "total_output_tokens",
            "num_samples_processed",
            "true_positives",
            "false_positives",
            "true_negatives",
            "false_negatives",
            "model_name",
            "max_tokens",
            "temperature",
            "top_p",
            "seed",
        ]

        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for result in results:
                # Calculate combined generation TPS for CSV
                total_gen_tokens = (
                    result.total_input_tokens + result.total_output_tokens
                )
                combined_tps = (
                    total_gen_tokens / result.generation_time
                    if result.generation_time > 0
                    else 0.0
                )

                writer.writerow(
                    {
                        "algorithm": result.algorithm_name,
                        "configuration": result.config_string,
                        "precision": result.precision,
                        "recall": result.recall,
                        "f1_score": result.f1,
                        "accuracy": result.accuracy,
                        "false_positive_rate": result.fpr,
                        "false_negative_rate": result.fnr,
                        "generation_tokens_per_second": combined_tps,
                        "generation_time_seconds": result.generation_time,
                        "detection_time_seconds": result.detection_time,
                        "total_time_seconds": result.total_time,
                        "total_input_tokens": result.total_input_tokens,
                        "total_output_tokens": result.total_output_tokens,
                        "num_samples_processed": result.num_samples,
                        "true_positives": result.tp,
                        "false_positives": result.fp,
                        "true_negatives": result.tn,
                        "false_negatives": result.fn,
                        "model_name": model_name,
                        "max_tokens": 512,  # Fixed as per requirements
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "seed": 42,  # Fixed as per requirements
                    }
                )


def main():
    """Entry point for Fire CLI."""
    benchmarker = WatermarkBenchmarker()
    fire.Fire(benchmarker.run_benchmark)


if __name__ == "__main__":
    main()
