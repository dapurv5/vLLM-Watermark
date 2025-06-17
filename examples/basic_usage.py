"""
Basic example demonstrating how to use the vLLM-Watermark package.
"""

import torch

from vllm_watermark import WatermarkFactory


def main():
    # Create a watermarking algorithm instance
    watermark = WatermarkFactory.create(
        "kgw",  # Algorithm name
        gamma=0.5,  # Algorithm-specific parameter
        delta=2.0,  # Algorithm-specific parameter
    )

    # Example logits (batch_size=1, vocab_size=5)
    logits = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])

    # Apply watermarking
    watermarked_logits = watermark.apply(logits)

    # Example text to check for watermark
    text = "This is an example text that might contain a watermark."

    # Detect watermark
    result = watermark.detect(text)
    print(f"Detection result: {result}")

if __name__ == "__main__":
    main()