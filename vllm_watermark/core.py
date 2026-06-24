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
    PF = "pf"
    UNIGRAM = "unigram"
    SYNTHID = "synthid"
    DIP = "dip"
    SWEET = "sweet"
    BLACKBOX = "blackbox"


class DetectionAlgorithm(Enum):
    """Supported detection algorithms."""

    OPENAI = "openai"
    OPENAI_Z = "openai_z"
    OPENAI_DR = "openai_dr"
    OPENAI_PL = "openai_pl"
    MARYLAND = "maryland"
    MARYLAND_Z = "maryland_z"
    PF = "pf"
    UNIGRAM = "unigram"
    UNIGRAM_Z = "unigram_z"
    SYNTHID = "synthid"
    DIP = "dip"
    SWEET = "sweet"
    BLACKBOX = "blackbox"


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
        model,
        algo: WatermarkingAlgorithm = WatermarkingAlgorithm.OPENAI,
        debug: bool = False,
        **kwargs,
    ):
        """
        Create a watermarked LLM using the clean implementation.

        Args:
            model: vLLM LLM instance to add watermarking to
            algo: Watermarking algorithm to use
            debug: Enable debug logging
            **kwargs: Additional arguments for watermark generator

        Returns:
            Watermarked LLM instance
        """
        from vllm_watermark.simple_watermark import create_watermarked_llm
        from vllm_watermark.watermark_generators import WatermarkGenerators

        logger.info(f"Creating watermarked LLM with algorithm: {algo.value}")

        # Extract model name for tokenizer from LLM instance
        if hasattr(model, "model_config") and hasattr(model.model_config, "model"):
            model_name = model.model_config.model
        else:
            # Fallback to the model object itself for tokenizer creation
            model_name = model

        # Separate watermark generator kwargs from vLLM kwargs
        generator_kwargs = {}
        vllm_kwargs = {}

        for key, value in kwargs.items():
            if key in ["ngram", "seed", "payload", "tokenizer", "gamma", "delta", "hash_key", "z_threshold", "keys", "sampling_table_size", "sampling_table_seed", "context_history_size", "alpha", "entropy_threshold", "ignore_history", "n_candidates", "ctx_len"]:
                generator_kwargs[key] = value
            else:
                vllm_kwargs[key] = value

        # Only support LLM objects
        from vllm import LLM

        if not isinstance(model, LLM):
            raise ValueError(
                f"Expected vLLM LLM instance, got {type(model)}. "
                "Please create an LLM object first and pass it to the factory."
            )

        if vllm_kwargs:
            logger.warning(
                f"LLM instance provided, ignoring vllm_kwargs: {vllm_kwargs}"
            )

        # Handle different algorithm types
        if algo == WatermarkingAlgorithm.BLACKBOX:
            from vllm_watermark.blackbox_watermark import BlackBoxWatermarkedLLM

            watermarked_llm = BlackBoxWatermarkedLLM(
                llm=model,
                key=generator_kwargs.get("hash_key", 15485863),
                n_candidates=generator_kwargs.get("n_candidates", 16),
                ctx_len=generator_kwargs.get("ctx_len", generator_kwargs.get("ngram", 4)),
                debug=debug,
            )
        else:
            # Use sampler replacement approach for other algorithms
            generator = WatermarkGenerators.create(
                algo=algo,
                model=model_name,
                **generator_kwargs,
            )
            watermarked_llm = create_watermarked_llm(
                llm=model,
                watermark_generator=generator,
                debug=debug,
            )

        logger.info("Successfully created watermarked LLM")
        return watermarked_llm
