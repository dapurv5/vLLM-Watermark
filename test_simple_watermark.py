"""
Test the simple watermark implementation.
"""

import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from vllm import SamplingParams

from vllm_watermark.core import DetectionAlgorithm, WatermarkingAlgorithm
from vllm_watermark.simple_watermark import create_watermarked_llm
from vllm_watermark.watermark_detectors import WatermarkDetectors
from vllm_watermark.watermark_generators import WatermarkGenerators


def test_openai_watermark():
    """Test OpenAI watermark with the simple implementation."""
    print("=== Testing Simple OpenAI Watermark ===")
    
    # Model name
    model_name = "meta-llama/Llama-3.2-1B"
    
    # Create watermark generator
    generator = WatermarkGenerators.create(
        algo=WatermarkingAlgorithm.OPENAI,
        model=model_name,  # Pass model name for tokenizer extraction
        ngram=2,
        seed=42,
        payload=0
    )
    
    print(f"Created generator: {type(generator).__name__}")
    
    # Create watermarked LLM
    watermarked_llm = create_watermarked_llm(
        model=model_name,
        watermark_generator=generator,
        debug=True,  # Enable debug logging
        enforce_eager=True,
        max_model_len=1024
    )
    
    print("Created watermarked LLM")
    
    # Create detector
    detector = WatermarkDetectors.create(
        algo=DetectionAlgorithm.OPENAI_Z,
        model=watermarked_llm,
        ngram=2,
        seed=42,
        payload=0,
        threshold=0.05
    )
    
    print("Created detector")
    
    # Test prompt
    prompt = "Cluster comprises IBM's Opteron-based eServer 325 server and systems management software and storage devices that can run Linux and Windows operating systems"
    
    # Generate watermarked text
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=0.95,
        max_tokens=64
    )
    
    print(f"Generating with prompt: {prompt[:50]}...")
    outputs = watermarked_llm.generate([prompt], sampling_params)
    
    if outputs:
        generated_text = outputs[0].outputs[0].text
        print(f"Generated text: {generated_text}")
        
        # Test detection
        detection_result = detector.detect(generated_text)
        print("\nDetection Results:")
        print(f"  Is watermarked: {detection_result['is_watermarked']}")
        print(f"  Detection score: {detection_result['score']:.4f}")
        print(f"  P-value: {detection_result['pvalue']:.6f}")
        
        # A properly watermarked text should have a low p-value (< 0.05)
        if detection_result['pvalue'] < 0.05:
            print("✅ SUCCESS: Watermark detected (low p-value)!")
        else:
            print("❌ FAILURE: Watermark not detected (high p-value)")
            
        return detection_result['pvalue'] < 0.05
    else:
        print("❌ No output generated")
        return False


if __name__ == "__main__":
    success = test_openai_watermark()
    sys.exit(0 if success else 1)
