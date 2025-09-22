"""
Core watermarking functionality for vLLM.

This module provides the main interfaces for creating watermarked LLMs
using a clean, production-ready implementation without monkey patching.
"""

from enum import Enum
from typing import Union

from loguru import logger


class WatermarkingAlgorithm(Enum):
    """Supported watermarking algorithms."""

    OPENAI = "openai"
    OPENAI_DR = "openai_dr"
    MARYLAND = "maryland"
    MARYLAND_L = "maryland_l"
    PF = "pf"


class DetectionAlgorithm(Enum):
    """Supported detection algorithms."""

    OPENAI = "openai"
    OPENAI_Z = "openai_z"
    OPENAI_DR = "openai_dr"
    MARYLAND = "maryland"
    MARYLAND_Z = "maryland_z"
    PF_SAMPLING = "pf_sampling"


class WatermarkUtils:
    """Utility functions for watermarking."""

    @staticmethod
    def get_tokenizer(model_or_tokenizer):
        """
        Extract tokenizer from a model or return the tokenizer if already a tokenizer.

        Args:
            model_or_tokenizer: Either a vLLM LLM instance, model string, or tokenizer

        Returns:
            The tokenizer object
        """
        if hasattr(model_or_tokenizer, "get_tokenizer"):
            # It's a vLLM LLM instance
            return model_or_tokenizer.get_tokenizer()
        elif hasattr(model_or_tokenizer, "tokenizer"):
            # It's a vLLM LLM instance (different attribute name)
            return model_or_tokenizer.tokenizer
        elif hasattr(model_or_tokenizer, "encode") and hasattr(
            model_or_tokenizer, "decode"
        ):
            # It's already a tokenizer
            return model_or_tokenizer
        elif isinstance(model_or_tokenizer, str):
            # It's a model name string - load tokenizer
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(model_or_tokenizer)
        else:
            raise ValueError(
                f"Cannot extract tokenizer from {type(model_or_tokenizer)}. "
                "Expected vLLM LLM instance, tokenizer, or model name string."
            )

    @staticmethod
    def infer_vocab_size(model, tokenizer=None):
        """
        Infer vocabulary size from model or tokenizer.

        Args:
            model: Model instance, model name, or tokenizer
            tokenizer: Optional tokenizer (if not provided, extracted from model)

        Returns:
            int: Vocabulary size
        """
        if tokenizer is None:
            tokenizer = WatermarkUtils.get_tokenizer(model)

        # Try different ways to get vocab size
        if hasattr(tokenizer, "vocab_size"):
            return tokenizer.vocab_size
        elif hasattr(tokenizer, "get_vocab"):
            return len(tokenizer.get_vocab())
        elif hasattr(tokenizer, "__len__"):
            return len(tokenizer)
        else:
            # Fallback: try to get vocab size from model if it's a vLLM instance
            if hasattr(model, "llm_engine") and hasattr(
                model.llm_engine, "model_config"
            ):
                model_config = model.llm_engine.model_config
                if hasattr(model_config, "vocab_size"):
                    return model_config.vocab_size

            raise ValueError(
                f"Cannot infer vocabulary size from model type {type(model)} "
                f"and tokenizer type {type(tokenizer)}"
            )


class WatermarkedLLMs:
    """Factory for creating watermarked LLMs using the new clean implementation."""

    @staticmethod
    def create(
        model: Union[str, "LLM"],
        algo: WatermarkingAlgorithm = WatermarkingAlgorithm.OPENAI,
        debug: bool = False,
        **kwargs,
    ):
        """
        Create a watermarked LLM using the clean implementation.

        Args:
            model: Model name string or vLLM LLM instance
            algo: Watermarking algorithm to use
            debug: Enable debug logging
            **kwargs: Additional arguments for watermark generator and vLLM

        Returns:
            Watermarked LLM instance
        """
        from vllm_watermark.simple_watermark import create_watermarked_llm
        from vllm_watermark.watermark_generators import WatermarkGenerators

        logger.info(f"Creating watermarked LLM with algorithm: {algo.value}")

        # Extract model name for tokenizer if needed
        model_name = (
            model
            if isinstance(model, str)
            else getattr(model, "model_config", {}).get("model", model)
        )

        # Separate watermark generator kwargs from vLLM kwargs
        generator_kwargs = {}
        vllm_kwargs = {}

        for key, value in kwargs.items():
            if key in ["ngram", "seed", "payload", "tokenizer"]:
                generator_kwargs[key] = value
            else:
                vllm_kwargs[key] = value

        # Create watermark generator
        generator = WatermarkGenerators.create(
            algo=algo,
            model=model_name,
            **generator_kwargs,
        )

        # If model is a string, pass it to create_watermarked_llm to create the LLM
        # If model is already an LLM instance, we need to create a new one (can't modify existing)
        if isinstance(model, str):
            watermarked_llm = create_watermarked_llm(
                model=model,
                watermark_generator=generator,
                debug=debug,
                **vllm_kwargs,
            )
        else:
            # If passed an existing LLM, we need to create a new one with same config
            # Extract model name from the existing LLM
            if hasattr(model, "model_config") and hasattr(model.model_config, "model"):
                model_name = model.model_config.model
            else:
                raise ValueError("Cannot extract model name from existing LLM instance")

            watermarked_llm = create_watermarked_llm(
                model=model_name,
                watermark_generator=generator,
                debug=debug,
                **vllm_kwargs,
            )

        logger.info("Successfully created watermarked LLM")
        return watermarked_llm
