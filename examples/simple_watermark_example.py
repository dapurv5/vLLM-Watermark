"""
Test the simple watermark implementation.
"""

import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# IMPORTANT: For watermarking to work, disable multiprocessing
# This allows the sampler replacement to work correctly
os.environ["VLLM_USE_V1"] = "1"  # Use V1 for better performance
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"  # Required for watermarking

from vllm import LLM, SamplingParams

from vllm_watermark.core import DetectionAlgorithm, WatermarkingAlgorithm
from vllm_watermark.simple_watermark import create_watermarked_llm
from vllm_watermark.watermark_detectors import WatermarkDetectors
from vllm_watermark.watermark_generators import WatermarkGenerators


def test_openai_watermark():
    """Test OpenAI watermark with the simple implementation."""
    print("=== Testing Simple OpenAI Watermark ===")

    # Model name
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    # Create watermark generator
    generator = WatermarkGenerators.create(
        algo=WatermarkingAlgorithm.OPENAI,
        model=model_name,  # Pass model name for tokenizer extraction
        ngram=4,
        seed=42,
        payload=0,
    )

    print(f"Created generator: {type(generator).__name__}")

    # Create base LLM first
    llm = LLM(
        model=model_name,
        enforce_eager=True,
        max_model_len=1024,
    )

    print("Created base LLM")

    # Create watermarked LLM
    watermarked_llm = create_watermarked_llm(
        llm=llm,
        watermark_generator=generator,
        debug=False,  # Enable debug to see sampler replacement structure
    )

    print("Created watermarked LLM")

    # Create detector
    detector = WatermarkDetectors.create(
        algo=DetectionAlgorithm.OPENAI_Z,
        model=watermarked_llm,
        ngram=4,
        seed=42,
        payload=0,
        threshold=0.05,
    )

    print("Created detector")

    # Test prompt
    prompt1 = "Cluster comprises IBM's Opteron-based eServer 325 server and systems management software and storage devices that can run Linux and Windows operating systems. IBM was founded in "
    prompt2 = "Write a short story about the famous plays of Shakespeare"

    # Generate watermarked text
    # NOTE: Add small frequency penalty to force vLLM V1 to provide prompt_token_ids
    # This is a workaround for the vLLM V1 bug where prompt_token_ids is None when no_penalties=True
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=0.95,
        max_tokens=20,  # Test with longer generation
        frequency_penalty=0.001,  # Tiny penalty to force prompt token access
    )

    print(f"Generating with fixed batching...")
    outputs = watermarked_llm.generate([prompt2, prompt1], sampling_params)
    results = []
    if outputs:
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}")
            print(f"Generated text: {generated_text!r}\n")

            # Test detection
            detection_result = detector.detect(generated_text)
            print("\nDetection Results:")
            print(f"  Is watermarked: {detection_result['is_watermarked']}")
            print(f"  Detection score: {detection_result['score']:.4f}")
            print(f"  P-value: {detection_result['pvalue']:.6f}")

            print("-" * 50)
            # A properly watermarked text should have a low p-value (< 0.05)
            if detection_result["pvalue"] < 0.05:
                print("✅ SUCCESS: Watermark detected (low p-value)!")
            else:
                print("❌ FAILURE: Watermark not detected (high p-value)")

            results.append(detection_result["pvalue"] < 0.05)
    else:
        print("❌ No output generated")
        return False
    return all(results)


if __name__ == "__main__":
    success = test_openai_watermark()
    sys.exit(0 if success else 1)
