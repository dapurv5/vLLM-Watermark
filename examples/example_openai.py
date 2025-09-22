import os
import sys

# export VLLM_USE_V1=1  # (V0 Sampler does not give correct results)
# export VLLM_ENABLE_V1_MULTIPROCESSING=0
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vllm import SamplingParams

from vllm_watermark.core import (
    DetectionAlgorithm,
    WatermarkedLLMs,
    WatermarkingAlgorithm,
)
from vllm_watermark.watermark_detectors import WatermarkDetectors

# Create a watermarked LLM using the new clean implementation
wm_llm = WatermarkedLLMs.create(
    model="meta-llama/Llama-3.2-1B",  # Pass model name directly
    algo=WatermarkingAlgorithm.OPENAI,
    seed=42,
    ngram=2,
    debug=False,  # Set to True for detailed logging
    # vLLM parameters
    enforce_eager=True,
    max_model_len=1024,
)

# Create OpenAI detector with matching parameters
detector = WatermarkDetectors.create(
    algo=DetectionAlgorithm.OPENAI_Z,
    model=wm_llm,  # Use the watermarked LLM for tokenizer extraction
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

# Generate outputs using the watermarked LLM
outputs = wm_llm.generate(prompts, sampling_params)

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

# Test with non-watermarked text for comparison
print("\n=== COMPARISON WITH NON-WATERMARKED TEXT ===")
non_watermarked_text = "This is a test sentence that was not generated with watermarking. It should not be detected as watermarked."
print(f"Non-watermarked text: {non_watermarked_text!r}\n")

non_wm_result = detector.detect(non_watermarked_text)
print("OpenAI Detector Results:")
print(f"  Is watermarked: {non_wm_result['is_watermarked']}")
print(f"  Detection score: {non_wm_result['score']:.4f}")
print(f"  P-value: {non_wm_result['pvalue']:.6f}")

print("\n=== EXPLANATION ===")
print("OpenAI watermarking uses power-law transformation of random values.")
print(
    "Lower p-values (< threshold) indicate higher confidence that text is watermarked."
)
print("The detector uses z-score approximation for fast p-value calculation.")
print(
    "Make sure generator and detector use the same parameters (seed, ngram, payload)."
)
