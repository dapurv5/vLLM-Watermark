import os
import sys

# export VLLM_ENABLE_V1_MULTIPROCESSING=0
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vllm import LLM, SamplingParams

from vllm_watermark.core import WatermarkedLLMs, WatermarkingAlgorithm
from vllm_watermark.watermark_detectors import MarylandDetectorZ, OpenaiDetectorZ

# Load the vLLM model
llm = LLM(model="meta-llama/Llama-3.2-1B")

# Create a OpenAI watermarked LLM (this wraps and patches the LLM)
wm_llm = WatermarkedLLMs.create(
    llm, algo=WatermarkingAlgorithm.OPENAI, seed=42, ngram=2
)

# Create multiple detectors for comparison
tokenizer = llm.get_tokenizer()
detectors = {
    "Maryland": MarylandDetectorZ(
        tokenizer=tokenizer,
        ngram=2,
        seed=42,
        seeding="hash",
        salt_key=35317,
        gamma=0.5,
        threshold=0.05,
    ),
    "OpenAI": OpenaiDetectorZ(
        tokenizer=tokenizer,
        ngram=2,
        seed=42,
        seeding="hash",
        salt_key=35317,
        threshold=0.05,
    ),
}

# Example prompt
prompts = ["Write a short poem about Microsoft"]

# Sampling parameters
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=64)

# Generate outputs using the watermarked LLM
outputs = wm_llm.generate(prompts, sampling_params)

print("=== WATERMARKED TEXT DETECTION ===")
# Print the outputs and detection results
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}")
    print(f"Generated text: {generated_text!r}\n")

    # Test with all detectors
    for detector_name, detector in detectors.items():
        detection_result = detector.detect(generated_text)
        print(f"{detector_name} Detector:")
        print(f"  Is watermarked: {detection_result['is_watermarked']}")
        print(f"  Detection score: {detection_result['score']:.4f}")
        print(f"  P-value: {detection_result['pvalue']:.6f}")

    print("-" * 50)

# Test with non-watermarked text for comparison
print("\n=== NON-WATERMARKED TEXT DETECTION ===")
non_watermarked_text = "This is a test sentence that was not generated with watermarking. It should not be detected as watermarked by any of the detectors."
print(f"Non-watermarked text: {non_watermarked_text!r}\n")

for detector_name, detector in detectors.items():
    non_wm_result = detector.detect(non_watermarked_text)
    print(f"{detector_name} Detector:")
    print(f"  Is watermarked: {non_wm_result['is_watermarked']}")
    print(f"  Detection score: {non_wm_result['score']:.4f}")
    print(f"  P-value: {non_wm_result['pvalue']:.6f}")

print("\n=== EXPLANATION ===")
print(
    "Different detectors may show varying effectiveness depending on the watermarking algorithm used."
)
print(
    "Lower p-values (< threshold) indicate higher confidence that the text is watermarked."
)
print("The detectors use different statistical methods for p-value calculation.")
