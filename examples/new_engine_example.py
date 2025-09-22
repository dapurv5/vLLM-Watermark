"""
Example demonstrating the new watermark engine architecture.

This example shows how to use the new clean engine approach that avoids
monkey patching and provides better stability across vLLM versions.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vllm import SamplingParams

from vllm_watermark.core import (
    DetectionAlgorithm,
    WatermarkedLLMs,
    WatermarkingAlgorithm,
)
from vllm_watermark.watermark_detectors import WatermarkDetectors


def main():
    # Example 1: Using the new engine (recommended)
    print("=== New Engine Example ===")

    # Create watermarked LLM with the new engine
    watermarked_llm = WatermarkedLLMs.create(
        model="meta-llama/Llama-3.2-1B",  # Model name as string
        algo=WatermarkingAlgorithm.OPENAI,
        use_new_engine=True,  # Use the new clean engine (default)
        debug=True,
        ngram=2,
        seed=42,
        # vLLM parameters
        enforce_eager=True,
        max_model_len=1024,
    )

    # Create OpenAI detector with matching parameters
    detector = WatermarkDetectors.create(
        algo=DetectionAlgorithm.OPENAI_Z,
        model=watermarked_llm,  # Factory handles tokenizer extraction and vocab size inference
        ngram=2,
        seed=42,
        payload=0,  # Match the generator's payload
        threshold=0.05,
    )

    # Example prompt
    prompts = [
        "Cluster comprises IBM's Opteron-based eServer 325 server and systems management"
        + "software and storage devices that can run Linux and Windows operating systems"
    ]

    # Sampling parameters
    sampling_params = SamplingParams(temperature=1.0, top_p=0.95, max_tokens=64)

    outputs = watermarked_llm.generate(prompts, sampling_params)

    print("=== OPENAI WATERMARK EXAMPLE ===")
    # Print the outputs and detection results
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}")
        print(f"Generated text: {generated_text!r}\n")

        # Test detection on watermarked text
        detection_result = detector.detect(generated_text)
        print("OpenAI Detector Results:")
        print(f"  Is watermarked: {detection_result['is_watermarked']}")
        print(f"  Detection score: {detection_result['score']:.4f}")
        print(f"  P-value: {detection_result['pvalue']:.6f}")

        print("-" * 50)


if __name__ == "__main__":
    main()
