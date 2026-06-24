import os
import sys

os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vllm import LLM, SamplingParams

from vllm_watermark.core import (
    DetectionAlgorithm,
    WatermarkedLLMs,
    WatermarkingAlgorithm,
)
from vllm_watermark.watermark_detectors import WatermarkDetectors

# Load the vLLM model
llm = LLM(
    model="meta-llama/Llama-3.2-1B",
    enforce_eager=True,
    max_model_len=1024,
)

# Create a watermarked LLM with the black-box algorithm
# n_candidates controls detection power: expected max PRF score ≈ m/(m+1).
# m=16 → 0.94 (weak), m=128 → 0.992 (strong), m=256 → 0.996 (very strong).
# Trade-off: m times more expensive to generate, but zero distortion.
wm_llm = WatermarkedLLMs.create(
    model=llm,
    algo=WatermarkingAlgorithm.BLACKBOX,
    hash_key=15485863,
    ngram=4,
    n_candidates=128,
)

# Create the matching detector
detector = WatermarkDetectors.create(
    algo=DetectionAlgorithm.BLACKBOX,
    model=llm,
    ngram=4,
    hash_key=15485863,
    threshold=0.02,
)

# Example prompts
prompts = [
    "Cluster comprises IBM's Opteron-based eServer 325 server and systems management"
    + " software and storage devices that can run Linux and Windows operating systems",
    "The research team published a groundbreaking paper on machine learning techniques"
    + " for natural language processing and text generation",
]

# Sampling parameters — temperature > 0 is required for diverse candidates
sampling_params = SamplingParams(temperature=1.0, top_p=0.95, max_tokens=128)

# Generate outputs using the watermarked LLM
outputs = wm_llm.generate(prompts, sampling_params)

print("=== BLACK-BOX WATERMARK EXAMPLE ===")
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}")
    print(f"Generated text: {generated_text!r}\n")

    detection_result = detector.detect(generated_text)
    print("Black-Box Detector Results:")
    print(f"  Is watermarked: {detection_result['is_watermarked']}")
    print(f"  Score: {detection_result['score']:.6f}")
    print(f"  P-value: {detection_result['pvalue']:.6f}")
    print("-" * 50)

# Test with non-watermarked text
print("\n=== COMPARISON WITH NON-WATERMARKED TEXT ===")
non_watermarked_text = (
    "This is a test sentence that was not generated with watermarking. "
    "It should not be detected as watermarked by the black-box detector."
)
print(f"Non-watermarked text: {non_watermarked_text!r}\n")

non_wm_result = detector.detect(non_watermarked_text)
print("Black-Box Detector Results:")
print(f"  Is watermarked: {non_wm_result['is_watermarked']}")
print(f"  Score: {non_wm_result['score']:.6f}")
print(f"  P-value: {non_wm_result['pvalue']:.6f}")

print("\n=== EXPLANATION ===")
print("The black-box watermark uses best-of-m rejection sampling.")
print("It does NOT modify logits or sampling — it generates m candidate")
print("sequences and selects the one with the highest keyed PRF score.")
print("The Gumbel-Max trick ensures the output distribution is identical")
print("to the unwatermarked model (distortion-free).")
print("Detection hashes n-grams with the secret key and computes the")
print("Irwin-Hall CDF of the sum of uniform PRF values.")
print("Trade-off: m times more expensive to generate, but zero distortion.")
print("Detection power: expected max PRF score ≈ m/(m+1).")
print("  m=16  → score~0.94 (weak),  m=128 → score~0.992 (strong)")
print("  m=256 → score~0.996 (very strong, paper recommended)")
