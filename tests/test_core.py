"""
Tests for core functionality.
"""

import pytest
import torch

from vllm_watermark.core import BaseWatermark, WatermarkFactory


class DummyWatermark(BaseWatermark):
    """Dummy watermarking algorithm for testing."""

    def apply(self, logits: torch.Tensor, **kwargs) -> torch.Tensor:
        return logits

    def detect(self, text: str, **kwargs) -> dict:
        return {"is_watermarked": True}


def test_watermark_factory():
    # Register the dummy algorithm
    WatermarkFactory.register("dummy")(DummyWatermark)

    # Test creation
    watermark = WatermarkFactory.create("dummy", test_param=1)
    assert isinstance(watermark, DummyWatermark)
    assert watermark.config["test_param"] == 1

    # Test listing algorithms
    algorithms = WatermarkFactory.list_algorithms()
    assert "dummy" in algorithms

    # Test unknown algorithm
    with pytest.raises(ValueError):
        WatermarkFactory.create("unknown")


def test_base_watermark():
    # Test that BaseWatermark cannot be instantiated
    with pytest.raises(TypeError):
        BaseWatermark()