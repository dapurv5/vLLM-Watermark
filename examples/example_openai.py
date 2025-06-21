import os
import sys

# export VLLM_ENABLE_V1_MULTIPROCESSING=0
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from loguru import logger
from vllm import LLM, SamplingParams

from vllm_watermark.core import WatermarkedLLMs, WatermarkingAlgorithm
from vllm_watermark.watermark_detectors import MarylandDetectorZ, OpenaiDetectorZ


def infer_vocab_size(model, tokenizer):
    """Infer vocab size from model to handle Llama tokenizer issues."""
    # Try to get vocab size from model config
    if hasattr(model.llm_engine, "model_executor"):
        try:
            if hasattr(model.llm_engine.model_executor, "driver_worker"):
                worker = model.llm_engine.model_executor.driver_worker
                if hasattr(worker, "model_runner") and hasattr(
                    worker.model_runner, "model"
                ):
                    model_config = worker.model_runner.model.config
                    if hasattr(model_config, "vocab_size"):
                        logger.info(
                            f"Found vocab size from model config: {model_config.vocab_size}"
                        )
                        return model_config.vocab_size
        except Exception as e:
            pass

    # Fallback to tokenizer methods
    try:
        logger.info(f"Found vocab size from tokenizer: {len(tokenizer.get_vocab())}")
        vocab_size = len(tokenizer.get_vocab())
    except Exception as e:
        logger.info(f"Found vocab size from tokenizer: {tokenizer.vocab_size}")
        vocab_size = tokenizer.vocab_size

    return vocab_size


# Load the vLLM model
llm = LLM(model="meta-llama/Llama-3.2-1B")

# Create a OpenAI watermarked LLM (this wraps and patches the LLM)
wm_llm = WatermarkedLLMs.create(
    llm, algo=WatermarkingAlgorithm.OPENAI, seed=42, ngram=2
)

# Infer vocab size for Llama models
tokenizer = llm.get_tokenizer()
vocab_size = infer_vocab_size(llm, tokenizer)
print(f"Inferred vocab size: {vocab_size}")

# Create multiple detectors for comparison
detectors = {
    "Maryland": MarylandDetectorZ(
        tokenizer=tokenizer,
        ngram=2,
        seed=42,
        seeding="hash",
        salt_key=35317,
        gamma=0.5,
        threshold=0.05,
        vocab_size=vocab_size,
    ),
    "OpenAI": OpenaiDetectorZ(
        tokenizer=tokenizer,
        ngram=2,
        seed=42,
        seeding="hash",
        salt_key=35317,
        payload=0,  # Match the generator's payload
        threshold=0.05,
        vocab_size=vocab_size,
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
    "The OpenAI detector should be most effective for detecting OpenAI-watermarked text."
)
print(
    "Lower p-values (< threshold) indicate higher confidence that the text is watermarked."
)
print("The detectors use different statistical methods for p-value calculation.")
print(
    "Make sure the generator and detector use the same parameters (seed, ngram, payload, etc.)"
)
