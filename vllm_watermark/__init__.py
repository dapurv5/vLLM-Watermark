"""
vLLM-Watermark: A Python package for implementing various watermarking algorithms for LLM outputs.
"""

__version__ = "0.1.0"

from vllm_watermark.core import BaseWatermark, WatermarkFactory

__all__ = ["BaseWatermark", "WatermarkFactory"]
