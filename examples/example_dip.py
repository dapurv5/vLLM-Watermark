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

# Create a watermarked LLM with DIP algorithm
wm_llm = WatermarkedLLMs.create(
    model=llm,
    algo=WatermarkingAlgorithm.DIP,
    seed=42,
    ngram=2,
    alpha=0.45,
    gamma=0.5,
    hash_key=15485863,
)

# Create DIP detector with matching parameters
detector = WatermarkDetectors.create(
    algo=DetectionAlgorithm.DIP,
    model=llm,
    ngram=2,
    seed=42,
    gamma=0.5,
    hash_key=15485863,
    threshold=0.05,
)

# Example prompts
prompts = [
    "Cluster comprises IBM's Opteron-based eServer 325 server and systems management"
    + " software and storage devices that can run Linux and Windows operating systems",
    "The research team published a groundbreaking paper on machine learning techniques"
    + " for natural language processing and text generation",
]

# Sampling parameters
sampling_params = SamplingParams(temperature=1.0, top_p=0.95, max_tokens=128)

# Generate outputs using the watermarked LLM
outputs = wm_llm.generate(prompts, sampling_params)

print("=== DIP (DiPmark) WATERMARK EXAMPLE ===")
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}")
    print(f"Generated text: {generated_text!r}\n")

    detection_result = detector.detect(generated_text)
    print("DIP Detector Results:")
    print(f"  Is watermarked: {detection_result['is_watermarked']}")
    print(f"  Score (green fraction): {detection_result['score']:.4f}")
    print(f"  P-value: {detection_result['pvalue']:.6f}")
    print("-" * 50)

# Test with non-watermarked text
print("\n=== COMPARISON WITH NON-WATERMARKED TEXT ===")
non_watermarked_text = (
    "This is a test sentence that was not generated with watermarking. "
    "It should not be detected as watermarked by the DIP detector."
)
print(f"Non-watermarked text: {non_watermarked_text!r}\n")

non_wm_result = detector.detect(non_watermarked_text)
print("DIP Detector Results:")
print(f"  Is watermarked: {non_wm_result['is_watermarked']}")
print(f"  Score (green fraction): {non_wm_result['score']:.4f}")
print(f"  P-value: {non_wm_result['pvalue']:.6f}")

print("\n=== EXPLANATION ===")
print("DIP (DiPmark) uses permutation-based probability redistribution.")
print("For each context, a random vocabulary permutation is generated.")
print("Probability mass is redistributed via quantile splitting (alpha=0.45).")
print("Unlike Maryland's flat delta bias, DIP's boost adapts to token probability.")
print("Detection checks each token's quantile position in the permutation.")
print("Context history tracking prevents over-watermarking repeated n-grams.")
