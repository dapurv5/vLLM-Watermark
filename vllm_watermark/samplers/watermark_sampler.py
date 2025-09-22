"""
Watermark-enabled sampler that cleanly inherits from vLLM's base sampler.

This approach avoids monkey patching by creating a proper subclass of vLLM's Sampler
that integrates watermarking into the sampling process.
"""

import os
from typing import Optional

import torch
from loguru import logger

from vllm_watermark.watermark_generators.base import WmGenerator

# Import the correct sampler based on vLLM version
env_use_v1 = os.environ.get("VLLM_USE_V1")
if env_use_v1 is not None and env_use_v1.strip() == "0":
    # Force V0 imports when V1 is explicitly disabled
    from vllm.model_executor.layers.sampler import Sampler as BaseSampler
    from vllm.model_executor.layers.sampler import SamplerOutput
    from vllm.model_executor.sampling_metadata import SamplingMetadata

    logger.info("Using V0 sampler for watermarking")
else:
    # Prefer V1 if available; otherwise fall back to V0
    try:
        from vllm.model_executor.layers.sampler import SamplerOutput
        from vllm.v1.sample.metadata import SamplingMetadata
        from vllm.v1.sample.sampler import Sampler as BaseSampler

        logger.info("Using V1 sampler for watermarking")
    except ImportError:
        try:
            from vllm.model_executor.layers.sampler import Sampler as BaseSampler
            from vllm.model_executor.layers.sampler import SamplerOutput
            from vllm.model_executor.sampling_metadata import SamplingMetadata

            logger.info("Using V0 sampler for watermarking (V1 not available)")
        except ImportError:
            raise ImportError("Could not import any sampler class from vLLM")


class WatermarkSampler(BaseSampler):
    """
    A watermark-enabled sampler that inherits from vLLM's base sampler.

    This sampler intercepts the sampling process to apply watermarking algorithms
    while preserving all the functionality of the base sampler.
    """

    def __init__(self, watermark_generator: WmGenerator, model, debug: bool = False):
        super().__init__()
        self.watermark_generator = watermark_generator
        self.model = model
        self.debug = debug

        if self.debug:
            logger.debug(
                f"Initialized WatermarkSampler with generator: {type(watermark_generator).__name__}"
            )

    def forward(
        self, logits: torch.Tensor, sampling_metadata: SamplingMetadata
    ) -> Optional[SamplerOutput]:
        """
        Forward pass that applies watermarking before calling the base sampler.

        Args:
            logits: Model logits tensor [batch_size, vocab_size]
            sampling_metadata: vLLM sampling metadata

        Returns:
            SamplerOutput with watermarked token selections
        """
        if self.debug:
            logger.debug(
                f"WatermarkSampler.forward called with logits shape: {logits.shape}"
            )

        # Apply watermarking to logits
        watermarked_logits = self._apply_watermarking(logits, sampling_metadata)

        # Call the parent sampler with watermarked logits
        return super().forward(watermarked_logits, sampling_metadata)

    def _apply_watermarking(
        self, logits: torch.Tensor, sampling_metadata: SamplingMetadata
    ) -> torch.Tensor:
        """
        Apply watermarking to the logits based on the sampling metadata.

        Args:
            logits: Original logits from the model
            sampling_metadata: Sampling parameters and context

        Returns:
            Modified logits with watermarking applied
        """
        # Check if we should apply watermarking based on sampling parameters
        should_watermark, temperature, top_p = self._should_apply_watermarking(
            sampling_metadata
        )

        if not should_watermark:
            if self.debug:
                logger.debug("Skipping watermarking (conditions not met)")
            return logits

        # Build n-gram contexts for watermarking
        ngram_contexts = self._extract_ngram_contexts(sampling_metadata)

        if not ngram_contexts:
            if self.debug:
                logger.debug("No n-gram contexts available, skipping watermarking")
            return logits

        # Apply watermarking through the generator
        try:
            if hasattr(self.watermark_generator, "logits_processor"):
                # Use logits processor approach if available
                ngram_tokens = torch.tensor(
                    ngram_contexts, dtype=torch.long, device=logits.device
                )
                watermarked_logits = self.watermark_generator.logits_processor(
                    logits.to(torch.float32), ngram_tokens
                )
                if self.debug:
                    logger.debug("Applied watermark via logits processor")
                return watermarked_logits
            else:
                # Use sampling approach - modify logits to force selection
                return self._apply_sampling_watermark(
                    logits, ngram_contexts, temperature, top_p
                )

        except Exception as e:
            logger.error(f"Error applying watermarking: {e}")
            # Return original logits if watermarking fails
            return logits

    def _apply_sampling_watermark(
        self,
        logits: torch.Tensor,
        ngram_contexts: list,
        temperature: float,
        top_p: float,
    ) -> torch.Tensor:
        """
        Apply watermarking by sampling tokens and modifying logits to force selection.

        This method is used when the watermark generator doesn't provide a logits processor.
        """
        try:
            # Convert n-gram contexts to tensor
            ngram_tokens = torch.tensor(
                ngram_contexts, dtype=torch.long, device=logits.device
            )

            # Sample watermarked tokens using the generator
            sampled_tokens = self.watermark_generator.sample_next(
                logits.to(torch.float32),
                ngram_tokens,
                temperature,
                top_p,
            )

            if sampled_tokens is not None and len(sampled_tokens) == logits.shape[0]:
                # Create new logits with forced selection
                modified_logits = torch.full_like(logits, -float("inf"))
                batch_indices = torch.arange(logits.shape[0], device=logits.device)
                modified_logits[batch_indices, sampled_tokens] = 0.0

                if self.debug:
                    logger.debug(
                        f"Applied watermark via forced token selection: {sampled_tokens.tolist()}"
                    )

                return modified_logits
            else:
                if self.debug:
                    logger.debug("Watermark sampling failed, using original logits")
                return logits

        except Exception as e:
            logger.error(f"Error in sampling watermark: {e}")
            return logits

    def _should_apply_watermarking(
        self, sampling_metadata: SamplingMetadata
    ) -> tuple[bool, float, float]:
        """
        Determine if watermarking should be applied based on sampling parameters.

        Returns:
            Tuple of (should_apply, temperature, top_p)
        """
        # Extract sampling parameters (this logic may need adjustment based on vLLM version)
        try:
            # Try to get temperature and top_p from sampling metadata
            # The exact API may vary between V0 and V1
            if (
                hasattr(sampling_metadata, "seq_groups")
                and sampling_metadata.seq_groups
            ):
                seq_group = sampling_metadata.seq_groups[0]
                sampling_params = seq_group.sampling_params

                temperature = getattr(sampling_params, "temperature", 1.0)
                top_p = getattr(sampling_params, "top_p", 1.0)

                # Only apply watermarking for non-greedy sampling
                should_apply = temperature > 0.01

                return should_apply, temperature, top_p
            else:
                # Fallback for different metadata structures
                return False, 1.0, 1.0

        except Exception as e:
            logger.debug(f"Error extracting sampling parameters: {e}")
            return False, 1.0, 1.0

    def _extract_ngram_contexts(self, sampling_metadata: SamplingMetadata) -> list:
        """
        Extract n-gram contexts from the sampling metadata for watermarking.

        Returns:
            List of n-gram token sequences for each sequence in the batch
        """
        ngram_contexts = []

        try:
            if (
                hasattr(sampling_metadata, "seq_groups")
                and sampling_metadata.seq_groups
            ):
                for seq_group in sampling_metadata.seq_groups:
                    for seq_id in seq_group.seq_ids:
                        # Get the sequence data
                        if (
                            hasattr(seq_group, "seq_data")
                            and seq_id in seq_group.seq_data
                        ):
                            seq_data = seq_group.seq_data[seq_id]

                            # Combine prompt and output tokens
                            prompt_tokens = getattr(seq_data, "prompt_token_ids", [])
                            output_tokens = getattr(seq_data, "output_token_ids", [])
                            all_tokens = prompt_tokens + output_tokens

                            # Extract the last n-gram for watermarking
                            ngram_size = self.watermark_generator.ngram
                            if len(all_tokens) >= ngram_size:
                                ngram_context = all_tokens[-ngram_size:]
                                ngram_contexts.append(ngram_context)
                            else:
                                # Pad with zeros if we don't have enough tokens
                                padded_context = [0] * (
                                    ngram_size - len(all_tokens)
                                ) + all_tokens
                                ngram_contexts.append(padded_context)
                        else:
                            # Fallback: use zero-padded context
                            ngram_contexts.append([0] * self.watermark_generator.ngram)

            if self.debug and ngram_contexts:
                logger.debug(f"Extracted {len(ngram_contexts)} n-gram contexts")

        except Exception as e:
            logger.debug(f"Error extracting n-gram contexts: {e}")

        return ngram_contexts
