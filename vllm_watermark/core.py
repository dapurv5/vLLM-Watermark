"""
Core functionality for the vLLM-Watermark package.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, Union

import torch

from vllm_watermark.config import WatermarkConfig


class BaseWatermark(ABC):
    """Abstract base class for all watermarking algorithms."""

    def __init__(self, config: WatermarkConfig):
        """Initialize the watermarking algorithm.

        Args:
            config: Configuration for the watermarking algorithm
        """
        self.config = config
        if config.seed is not None:
            torch.manual_seed(config.seed)
        self.device = torch.device(config.device)

    @abstractmethod
    def apply(self, logits: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply the watermarking algorithm to the logits.

        Args:
            logits: Input logits tensor
            **kwargs: Additional parameters for the watermarking algorithm

        Returns:
            Modified logits tensor
        """
        pass

    @abstractmethod
    def detect(self, text: str, **kwargs) -> Dict[str, Any]:
        """Detect watermark in the given text.

        Args:
            text: Input text to check for watermark
            **kwargs: Additional parameters for detection

        Returns:
            Dictionary containing detection results
        """
        pass


class WatermarkFactory:
    """Factory class for creating watermarking algorithm instances."""

    _algorithms: Dict[str, Type[BaseWatermark]] = {}
    _configs: Dict[str, Type[WatermarkConfig]] = {}

    @classmethod
    def register(cls, name: str, config_class: Type[WatermarkConfig]) -> callable:
        """Register a watermarking algorithm.

        Args:
            name: Name of the algorithm
            config_class: Configuration class for the algorithm

        Returns:
            Decorator function
        """
        def decorator(algorithm_class: Type[BaseWatermark]) -> Type[BaseWatermark]:
            cls._algorithms[name] = algorithm_class
            cls._configs[name] = config_class
            return algorithm_class
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseWatermark:
        """Create a watermarking algorithm instance.

        Args:
            name: Name of the algorithm
            **kwargs: Parameters for the algorithm configuration

        Returns:
            Instance of the watermarking algorithm

        Raises:
            ValueError: If the algorithm is not registered
        """
        if name not in cls._algorithms:
            raise ValueError(f"Unknown watermarking algorithm: {name}")

        config_class = cls._configs[name]
        config = config_class(**kwargs)
        return cls._algorithms[name](config)

    @classmethod
    def list_algorithms(cls) -> list[str]:
        """List all registered algorithms.

        Returns:
            List of algorithm names
        """
        return list(cls._algorithms.keys())

    @classmethod
    def get_config_class(cls, name: str) -> Type[WatermarkConfig]:
        """Get the configuration class for an algorithm.

        Args:
            name: Name of the algorithm

        Returns:
            Configuration class for the algorithm

        Raises:
            ValueError: If the algorithm is not registered
        """
        if name not in cls._configs:
            raise ValueError(f"Unknown watermarking algorithm: {name}")
        return cls._configs[name]