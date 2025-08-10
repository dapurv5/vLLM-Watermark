"""Factory for creating watermark generators with automatic configuration."""

from typing import Optional, Union

import torch
from loguru import logger

# Import shared enums and utilities
from vllm_watermark.core import WatermarkingAlgorithm, WatermarkUtils


class WatermarkGenerators:
    """Factory for creating watermark generators with automatic configuration."""

    @staticmethod
    def create(
        algo: Union[WatermarkingAlgorithm, str],
        model,
        tokenizer=None,
        vocab_size: Optional[int] = None,
        ngram: int = 1,
        seed: int = 0,
        seeding: str = "hash",
        salt_key: int = 35317,
        # Algorithm-specific parameters
        payload: int = 0,  # For OpenAI
        gamma: float = 0.5,  # For Maryland
        delta: float = 1.0,  # For Maryland
        **kwargs,
    ):
        """Create a watermark generator with automatic configuration.

        Args:
            algo: Watermarking algorithm to use
            model: Model object (required for generators)
            tokenizer: Tokenizer object (extracted from model if not provided)
            vocab_size: Vocabulary size (auto-inferred if not provided)
            ngram: N-gram size for seeding
            seed: Random seed
            seeding: Seeding method ("hash", "simple", etc.)
            salt_key: Salt key for hashing
            payload: Payload for OpenAI generators
            gamma: Gamma parameter for Maryland generators
            delta: Delta parameter for Maryland generators
            **kwargs: Additional algorithm-specific parameters

        Returns:
            Configured watermark generator

        Examples:
            # Simple usage
            generator = WatermarkGenerators.create(
                algo=WatermarkingAlgorithm.OPENAI,
                model=llm,
                ngram=2,
                seed=42
            )

            # With custom parameters
            generator = WatermarkGenerators.create(
                algo="maryland",
                model=llm,
                ngram=2,
                seed=42,
                gamma=0.7
            )
        """
        # Convert string to enum if needed
        if isinstance(algo, str):
            algo = WatermarkingAlgorithm(algo.lower())

        # Determine tokenizer and vocab_size using shared utilities
        if tokenizer is None:
            tokenizer = WatermarkUtils.get_tokenizer(model)
        if vocab_size is None:
            vocab_size = WatermarkUtils.infer_vocab_size(model, tokenizer)

        logger.info(f"Creating {algo.value} generator with vocab_size={vocab_size}")

        # Create the appropriate generator
        if algo == WatermarkingAlgorithm.OPENAI:
            from .openai_generator import OpenaiGenerator

            return OpenaiGenerator(
                model=model,
                tokenizer=tokenizer,
                ngram=ngram,
                seed=seed,
                seeding=seeding,
                salt_key=salt_key,
                payload=payload,
                **kwargs,
            )
        elif algo == WatermarkingAlgorithm.OPENAI_DR:
            from .openai_generator import OpenaiGeneratorDoubleRandomization

            return OpenaiGeneratorDoubleRandomization(
                model=model,
                tokenizer=tokenizer,
                ngram=ngram,
                seed=seed,
                seeding=seeding,
                salt_key=salt_key,
                payload=payload,
                **kwargs,
            )
        elif algo == WatermarkingAlgorithm.MARYLAND:
            from .maryland_generator import MarylandGenerator

            return MarylandGenerator(
                model=model,
                tokenizer=tokenizer,
                ngram=ngram,
                seed=seed,
                seeding=seeding,
                salt_key=salt_key,
                gamma=gamma,
                delta=delta,
                **kwargs,
            )
        elif algo == WatermarkingAlgorithm.MARYLAND_L:
            # For MARYLAND_L, we return the logit processor directly
            # instead of a generator, since this will be used differently
            from vllm_watermark.logit_processors import MarylandLogitProcessor

            vocab_size = WatermarkUtils.infer_vocab_size(model, tokenizer)

            return MarylandLogitProcessor(
                vocab_size=vocab_size,
                gamma=gamma,
                delta=delta,
                ngram=ngram,
                seed=seed,
                salt_key=salt_key,
                payload=payload,
                seeding=seeding,
                device="cuda" if torch.cuda.is_available() else "cpu",
                **kwargs,
            )
        elif algo == WatermarkingAlgorithm.PF:
            from .pf_generator import PFGenerator

            return PFGenerator(
                model=model,
                tokenizer=tokenizer,
                ngram=ngram,
                seed=seed,
                seeding=seeding,
                salt_key=salt_key,
                **kwargs,
            )
        else:
            raise ValueError(f"Unsupported watermarking algorithm: {algo}")
