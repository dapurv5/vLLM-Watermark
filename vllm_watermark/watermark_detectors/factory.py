"""Factory for creating watermark detectors with automatic configuration."""

from typing import Optional, Union

from loguru import logger

# Import shared enums and utilities
from vllm_watermark.core import DetectionAlgorithm, WatermarkUtils


class WatermarkDetectors:
    """Factory for creating watermark detectors with automatic configuration."""

    @staticmethod
    def create(
        algo: Union[DetectionAlgorithm, str],
        model=None,
        tokenizer=None,
        vocab_size: Optional[int] = None,
        ngram: int = 1,
        seed: int = 0,
        seeding: str = "hash",
        salt_key: int = 35317,
        threshold: float = 0.05,
        # Algorithm-specific parameters
        payload: int = 0,  # For OpenAI
        gamma: float = 0.5,  # For Maryland / Unigram / DIP / SWEET
        delta: float = 1.0,  # For Maryland / Unigram
        hash_key: int = 15485863,  # For Unigram / DIP / SWEET
        ignore_history: bool = False,  # For DIP
        alpha: float = 0.45,  # For DIP (generator-only, consumed here to prevent leakage)
        entropy_threshold: float = 3.0,  # For SWEET (generator-only, consumed here)
        **kwargs,
    ):
        """Create a watermark detector with automatic configuration.

        Args:
            algo: Detection algorithm to use
            model: Model object (vLLM or HuggingFace) - will extract tokenizer and vocab_size
            tokenizer: Tokenizer object (alternative to model)
            vocab_size: Vocabulary size (auto-inferred if not provided)
            ngram: N-gram size for seeding
            seed: Random seed
            seeding: Seeding method ("hash", "simple", etc.)
            salt_key: Salt key for hashing
            threshold: Detection threshold for p-values
            payload: Payload for OpenAI detectors
            gamma: Gamma parameter for Maryland detectors
            delta: Delta parameter for Maryland detectors
            **kwargs: Additional algorithm-specific parameters

        Returns:
            Configured watermark detector

        Examples:
            # Simple usage with model
            detector = WatermarkDetectors.create(
                algo=DetectionAlgorithm.OPENAI_Z,
                model=llm,
                ngram=2,
                seed=42
            )

            # Usage with tokenizer only
            detector = WatermarkDetectors.create(
                algo="openai_z",
                tokenizer=tokenizer,
                vocab_size=128256,
                ngram=2,
                seed=42
            )
        """
        # Convert string to enum if needed
        if isinstance(algo, str):
            algo = DetectionAlgorithm(algo.lower())

        # Determine tokenizer and vocab_size using shared utilities
        if model is not None:
            if tokenizer is None:
                tokenizer = WatermarkUtils.get_tokenizer(model)
            if vocab_size is None:
                vocab_size = WatermarkUtils.infer_vocab_size(model, tokenizer)
        elif tokenizer is not None:
            if vocab_size is None:
                # Try to infer from tokenizer only
                try:
                    vocab_size = len(tokenizer.get_vocab())
                except:
                    vocab_size = tokenizer.vocab_size
        else:
            raise ValueError("Must provide either 'model' or 'tokenizer' parameter")

        logger.info(f"Creating {algo.value} detector with vocab_size={vocab_size}")

        # Create the appropriate detector
        if algo == DetectionAlgorithm.OPENAI:
            from .openai_detectors import OpenaiDetector

            return OpenaiDetector(
                tokenizer=tokenizer,
                vocab_size=vocab_size,
                ngram=ngram,
                seed=seed,
                seeding=seeding,
                salt_key=salt_key,
                payload=payload,
                threshold=threshold,
                **kwargs,
            )
        elif algo == DetectionAlgorithm.OPENAI_Z:
            from .openai_detectors import OpenaiDetectorZ

            return OpenaiDetectorZ(
                tokenizer=tokenizer,
                vocab_size=vocab_size,
                ngram=ngram,
                seed=seed,
                seeding=seeding,
                salt_key=salt_key,
                payload=payload,
                threshold=threshold,
                **kwargs,
            )
        elif algo == DetectionAlgorithm.MARYLAND:
            from .maryland_detectors import MarylandDetector

            return MarylandDetector(
                tokenizer=tokenizer,
                vocab_size=vocab_size,
                ngram=ngram,
                seed=seed,
                seeding=seeding,
                salt_key=salt_key,
                gamma=gamma,
                delta=delta,
                threshold=threshold,
                **kwargs,
            )
        elif algo == DetectionAlgorithm.MARYLAND_Z:
            from .maryland_detectors import MarylandDetectorZ

            return MarylandDetectorZ(
                tokenizer=tokenizer,
                vocab_size=vocab_size,
                ngram=ngram,
                seed=seed,
                seeding=seeding,
                salt_key=salt_key,
                gamma=gamma,
                delta=delta,
                threshold=threshold,
                **kwargs,
            )
        elif algo == DetectionAlgorithm.PF:
            from .pf_detector import PFDetector

            return PFDetector(
                tokenizer=tokenizer,
                vocab_size=vocab_size,
                ngram=ngram,
                seed=seed,
                seeding=seeding,
                salt_key=salt_key,
                threshold=threshold,
                **kwargs,
            )
        elif algo in (DetectionAlgorithm.UNIGRAM, DetectionAlgorithm.UNIGRAM_Z):
            from .unigram_detector import UnigramDetector

            return UnigramDetector(
                tokenizer=tokenizer,
                vocab_size=vocab_size,
                ngram=ngram,
                seed=seed,
                seeding=seeding,
                salt_key=salt_key,
                gamma=gamma,
                delta=delta,
                hash_key=hash_key,
                threshold=threshold,
                **kwargs,
            )
        elif algo == DetectionAlgorithm.SYNTHID:
            from .synthid_detector import SynthIDDetector

            return SynthIDDetector(
                tokenizer=tokenizer,
                vocab_size=vocab_size,
                ngram=ngram,
                threshold=threshold,
                **kwargs,
            )
        elif algo == DetectionAlgorithm.DIP:
            from .dip_detector import DIPDetector

            return DIPDetector(
                tokenizer=tokenizer,
                vocab_size=vocab_size,
                ngram=ngram,
                seed=seed,
                seeding=seeding,
                salt_key=salt_key,
                gamma=gamma,
                hash_key=hash_key,
                ignore_history=ignore_history,
                threshold=threshold,
                **kwargs,
            )
        elif algo == DetectionAlgorithm.SWEET:
            from .sweet_detector import SWEETDetector

            return SWEETDetector(
                tokenizer=tokenizer,
                vocab_size=vocab_size,
                ngram=ngram,
                seed=seed,
                seeding=seeding,
                salt_key=salt_key,
                gamma=gamma,
                hash_key=hash_key,
                threshold=threshold,
                **kwargs,
            )
        elif algo == DetectionAlgorithm.BLACKBOX:
            from .blackbox_detector import BlackBoxDetector

            ctx_len = kwargs.pop("ctx_len", ngram)
            return BlackBoxDetector(
                tokenizer=tokenizer,
                key=hash_key,
                ctx_len=ctx_len,
                threshold=threshold,
                **kwargs,
            )
        else:
            raise ValueError(f"Unsupported detection algorithm: {algo}")
