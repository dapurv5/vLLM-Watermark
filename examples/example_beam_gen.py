import os
import sys

# export VLLM_USE_V1=1  # (V0 Sampler preferred)
# export VLLM_ENABLE_V1_MULTIPROCESSING=0
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vllm import LLM, SamplingParams

from vllm_watermark.core import (
    DetectionAlgorithm,
    WatermarkedLLMs,
    WatermarkingAlgorithm,
)
from vllm_watermark.watermark_detectors import WatermarkDetectors

# Load the vLLM model
llm = LLM(model="meta-llama/Llama-3.2-1B")

# Create an OpenAI-DR watermarked LLM (this wraps and patches the LLM)
wm_llm = WatermarkedLLMs.create(
    llm,
    algo=WatermarkingAlgorithm.OPENAI_DR,
    seed=42,
    ngram=2,
    payload=0,
)

# Reuse the OpenAI detector with matching parameters
detector = WatermarkDetectors.create(
    algo=DetectionAlgorithm.OPENAI_Z,
    model=llm,
    ngram=2,
    seed=42,
    payload=0,
    threshold=0.05,
)

# Example prompt
prompts = ["Write a short poem about artificial intelligence"]

# Sampling parameters
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=64,
    n=4,
)

# Generate outputs using the watermarked LLM
outputs = wm_llm.generate(prompts, sampling_params)

print("=== OPENAI DOUBLE RANDOMIZATION WATERMARK EXAMPLE ===")
# Print the outputs and detection results for all returned generations
for output in outputs:
    print(f"Prompt: {output.prompt!r}")
    for i, candidate in enumerate(output.outputs, start=1):
        generated_text = candidate.text
        print(f"\nGeneration {i}:")
        print(f"  Text: {generated_text!r}")

        # Test detection on watermarked text
        detection_result = detector.detect(generated_text)
        print("  OpenAI Detector Results:")
        print(f"    Is watermarked: {detection_result['is_watermarked']}")
        print(f"    Detection score: {detection_result['score']:.4f}")
        print(f"    P-value: {detection_result['pvalue']:.6f}")

    print("-" * 50)

# Test with non-watermarked text for comparison
print("\n=== COMPARISON WITH NON-WATERMARKED TEXT ===")
non_watermarked_text = (
    "This is a test sentence that was not generated with watermarking. "
    "It should not be detected as watermarked."
)
print(f"Non-watermarked text: {non_watermarked_text!r}\n")

non_wm_result = detector.detect(non_watermarked_text)
print("OpenAI Detector Results:")
print(f"  Is watermarked: {non_wm_result['is_watermarked']}")
print(f"  Detection score: {non_wm_result['score']:.4f}")
print(f"  P-value: {non_wm_result['pvalue']:.6f}")

print("\n=== EXPLANATION ===")
print(
    "OpenAI watermarking with double randomization samples from r^(1/p) instead of taking argmax."
)
print(
    "Lower p-values (< threshold) indicate higher confidence that text is watermarked."
)
print(
    "Ensure generator and detector share parameters (seed, ngram, payload) for accurate detection."
)
