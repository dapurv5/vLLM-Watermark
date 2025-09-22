"""
Simple, clean watermarked LLM implementation.

This approach creates a custom sampler and injects it directly into vLLM
without any monkey patching.
"""

import os
from typing import Optional

import torch
from loguru import logger
from vllm import LLM, SamplingParams

from vllm_watermark.watermark_generators.base import WmGenerator

# Import the correct sampler based on vLLM version
env_use_v1 = os.environ.get("VLLM_USE_V1")
if env_use_v1 is not None and env_use_v1.strip() == "0":
    from vllm.model_executor.layers.sampler import Sampler as BaseSampler
    from vllm.model_executor.layers.sampler import SamplerOutput
    from vllm.model_executor.sampling_metadata import SamplingMetadata

    VLLM_VERSION = "V0"
else:
    try:
        from vllm.model_executor.layers.sampler import SamplerOutput
        from vllm.v1.sample.metadata import SamplingMetadata
        from vllm.v1.sample.sampler import Sampler as BaseSampler

        VLLM_VERSION = "V1"
    except ImportError:
        from vllm.model_executor.layers.sampler import Sampler as BaseSampler
        from vllm.model_executor.layers.sampler import SamplerOutput
        from vllm.model_executor.sampling_metadata import SamplingMetadata

        VLLM_VERSION = "V0"

logger.info(f"Using vLLM {VLLM_VERSION} for watermarking")


class WatermarkSampler(BaseSampler):
    """A simple watermark sampler that applies watermarking during sampling."""

    def __init__(self, watermark_generator: WmGenerator, debug: bool = False):
        super().__init__()
        self.watermark_generator = watermark_generator
        self.debug = debug

        if self.debug:
            logger.info(
                f"Initialized WatermarkSampler with {type(watermark_generator).__name__}"
            )

    def forward(
        self, logits: torch.Tensor, sampling_metadata: SamplingMetadata
    ) -> Optional[SamplerOutput]:
        """Apply watermarking to logits before calling the base sampler."""

        if self.debug:
            logger.debug(f"WatermarkSampler.forward: logits shape {logits.shape}")

        # Extract information needed for watermarking
        watermarked_logits = self._apply_watermarking(logits, sampling_metadata)

        # Call the parent sampler with watermarked logits
        return super().forward(watermarked_logits, sampling_metadata)

    def _apply_watermarking(
        self, logits: torch.Tensor, sampling_metadata: SamplingMetadata
    ) -> torch.Tensor:
        """Apply watermarking by directly sampling tokens and forcing selection."""

        try:
            # Extract sampling parameters
            temperature, top_p = self._extract_sampling_params(sampling_metadata)

            # Only apply watermarking for non-greedy sampling
            if temperature <= 0.01:
                if self.debug:
                    logger.debug("Skipping watermarking for greedy sampling")
                return logits

            # Extract n-gram contexts
            ngram_contexts = self._extract_ngram_contexts(sampling_metadata)

            if not ngram_contexts:
                if self.debug:
                    logger.debug("No n-gram contexts found, skipping watermarking")
                return logits

            # Apply watermarking by direct token sampling
            batch_size = logits.shape[0]
            if len(ngram_contexts) != batch_size:
                if self.debug:
                    logger.debug(
                        f"Mismatch: {len(ngram_contexts)} contexts vs {batch_size} batch size"
                    )
                return logits

            # Convert to tensor
            ngram_tokens = torch.tensor(
                ngram_contexts, dtype=torch.long, device=logits.device
            )

            # Use the watermark generator to sample tokens
            sampled_tokens = self.watermark_generator.sample_next(
                logits.to(torch.float32), ngram_tokens, temperature, top_p
            )

            if sampled_tokens is not None:
                # Force selection of watermarked tokens by zeroing out all other logits
                watermarked_logits = torch.full_like(logits, -float("inf"))
                batch_indices = torch.arange(batch_size, device=logits.device)
                watermarked_logits[batch_indices, sampled_tokens] = 0.0

                if self.debug:
                    logger.debug(
                        f"Applied watermarking: selected tokens {sampled_tokens.tolist()}"
                    )

                return watermarked_logits
            else:
                if self.debug:
                    logger.debug("Watermark sampling returned None")
                return logits

        except Exception as e:
            logger.error(f"Error in watermarking: {e}")
            return logits

    def _extract_sampling_params(
        self, sampling_metadata: SamplingMetadata
    ) -> tuple[float, float]:
        """Extract temperature and top_p from sampling metadata."""
        try:
            if (
                hasattr(sampling_metadata, "seq_groups")
                and sampling_metadata.seq_groups
            ):
                seq_group = sampling_metadata.seq_groups[0]
                sampling_params = seq_group.sampling_params
                temperature = getattr(sampling_params, "temperature", 1.0)
                top_p = getattr(sampling_params, "top_p", 1.0)
                return temperature, top_p
            else:
                return 1.0, 1.0
        except Exception as e:
            if self.debug:
                logger.debug(f"Error extracting sampling params: {e}")
            return 1.0, 1.0

    def _extract_ngram_contexts(self, sampling_metadata: SamplingMetadata) -> list:
        """Extract n-gram contexts from sampling metadata."""
        ngram_contexts = []

        try:
            if self.debug:
                logger.debug(f"Sampling metadata type: {type(sampling_metadata)}")
                logger.debug(
                    f"Sampling metadata attributes: {[attr for attr in dir(sampling_metadata) if not attr.startswith('_')]}"
                )

            # Check if this is V1 SamplingMetadata with direct token access
            if hasattr(sampling_metadata, "prompt_token_ids") and hasattr(
                sampling_metadata, "output_token_ids"
            ):
                # V1 approach: tokens are directly on sampling_metadata
                prompt_tokens = getattr(sampling_metadata, "prompt_token_ids", [])
                output_tokens = getattr(sampling_metadata, "output_token_ids", [])

                if self.debug:
                    logger.debug(
                        f"V1 metadata - Prompt tokens: {len(prompt_tokens) if prompt_tokens is not None else 'None'}"
                    )
                    logger.debug(
                        f"V1 metadata - Output tokens: {len(output_tokens) if output_tokens is not None else 'None'}"
                    )

                # Handle the case where these might be tensors or nested structures
                if prompt_tokens is not None:
                    if hasattr(prompt_tokens, "tolist"):
                        prompt_tokens = prompt_tokens.tolist()
                    # Flatten if it's a nested list (batch dimension)
                    if (
                        isinstance(prompt_tokens, list)
                        and len(prompt_tokens) > 0
                        and isinstance(prompt_tokens[0], list)
                    ):
                        prompt_tokens = prompt_tokens[0]  # Take first batch element
                elif prompt_tokens is None:
                    prompt_tokens = []

                if output_tokens is not None:
                    if hasattr(output_tokens, "tolist"):
                        output_tokens = output_tokens.tolist()
                    # Flatten if it's a nested list (batch dimension)
                    if (
                        isinstance(output_tokens, list)
                        and len(output_tokens) > 0
                        and isinstance(output_tokens[0], list)
                    ):
                        output_tokens = output_tokens[0]  # Take first batch element
                elif output_tokens is None:
                    output_tokens = []

                # Ensure both are flat lists of integers
                if not isinstance(prompt_tokens, list):
                    prompt_tokens = []
                if not isinstance(output_tokens, list):
                    output_tokens = []

                # For V1, we might have multiple sequences batched together
                # For now, assume single sequence and create one context
                all_tokens = prompt_tokens + output_tokens

                if self.debug:
                    logger.debug(f"Combined tokens length: {len(all_tokens)}")
                    logger.debug(
                        f"Last 10 tokens: {all_tokens[-10:] if len(all_tokens) > 10 else all_tokens}"
                    )

                # Extract last n-gram
                ngram_size = self.watermark_generator.ngram
                if len(all_tokens) >= ngram_size:
                    ngram_context = all_tokens[-ngram_size:]
                else:
                    # Pad with zeros if not enough tokens
                    ngram_context = [0] * (ngram_size - len(all_tokens)) + all_tokens

                ngram_contexts.append(ngram_context)

            elif (
                hasattr(sampling_metadata, "seq_groups")
                and sampling_metadata.seq_groups
            ):
                # V0 approach: use seq_groups structure
                if self.debug:
                    logger.debug(
                        f"V0 metadata - Found {len(sampling_metadata.seq_groups)} sequence groups"
                    )

                for i, seq_group in enumerate(sampling_metadata.seq_groups):
                    if self.debug:
                        logger.debug(
                            f"Seq group {i} attributes: {[attr for attr in dir(seq_group) if not attr.startswith('_')]}"
                        )
                        logger.debug(
                            f"Seq group {i} seq_ids: {getattr(seq_group, 'seq_ids', 'NOT_FOUND')}"
                        )

                    for seq_id in seq_group.seq_ids:
                        seq_data = None

                        # Try different ways to access sequence data
                        if (
                            hasattr(seq_group, "seq_data")
                            and seq_id in seq_group.seq_data
                        ):
                            seq_data = seq_group.seq_data[seq_id]
                        elif hasattr(seq_group, "seqs") and seq_id in seq_group.seqs:
                            seq_data = seq_group.seqs[seq_id]

                        if seq_data is not None:
                            if self.debug:
                                logger.debug(
                                    f"Seq data attributes: {[attr for attr in dir(seq_data) if not attr.startswith('_')]}"
                                )

                            # Try different attribute names for tokens
                            prompt_tokens = []
                            output_tokens = []

                            # Common attribute names in vLLM
                            for attr_name in [
                                "prompt_token_ids",
                                "prompt_tokens",
                                "tokens",
                            ]:
                                if hasattr(seq_data, attr_name):
                                    prompt_tokens = getattr(seq_data, attr_name, [])
                                    break

                            for attr_name in [
                                "output_token_ids",
                                "output_tokens",
                                "generated_tokens",
                            ]:
                                if hasattr(seq_data, attr_name):
                                    output_tokens = getattr(seq_data, attr_name, [])
                                    break

                            # Alternative: get data from sequence object
                            if hasattr(seq_data, "data"):
                                data = seq_data.data
                                prompt_tokens = getattr(
                                    data, "prompt_token_ids", prompt_tokens
                                )
                                output_tokens = getattr(
                                    data, "output_token_ids", output_tokens
                                )

                            all_tokens = prompt_tokens + output_tokens

                            if self.debug:
                                logger.debug(
                                    f"Prompt tokens length: {len(prompt_tokens)}"
                                )
                                logger.debug(
                                    f"Output tokens length: {len(output_tokens)}"
                                )
                                logger.debug(
                                    f"All tokens: {all_tokens[-10:] if len(all_tokens) > 10 else all_tokens}"
                                )

                            # Extract last n-gram
                            ngram_size = self.watermark_generator.ngram
                            if len(all_tokens) >= ngram_size:
                                ngram_context = all_tokens[-ngram_size:]
                            else:
                                # Pad with zeros if not enough tokens
                                ngram_context = [0] * (
                                    ngram_size - len(all_tokens)
                                ) + all_tokens

                            ngram_contexts.append(ngram_context)
                        else:
                            if self.debug:
                                logger.debug(
                                    f"Could not find seq_data for seq_id: {seq_id}"
                                )
                            # Fallback: zero-padded context
                            ngram_contexts.append([0] * self.watermark_generator.ngram)
            else:
                if self.debug:
                    logger.debug(
                        "No compatible metadata structure found for n-gram extraction"
                    )

            if self.debug:
                logger.debug(
                    f"Extracted {len(ngram_contexts)} n-gram contexts: {ngram_contexts}"
                )

        except Exception as e:
            if self.debug:
                logger.debug(f"Error extracting n-gram contexts: {e}")
                import traceback

                logger.debug(f"Traceback: {traceback.format_exc()}")

        return ngram_contexts


class WatermarkedLLM:
    """Simple watermarked LLM that replaces the sampler directly."""

    def __init__(
        self,
        model: str,
        watermark_generator: WmGenerator,
        debug: bool = False,
        **llm_kwargs,
    ):
        """
        Create a watermarked LLM.

        Args:
            model: Model name or path
            watermark_generator: Watermark generator to use
            debug: Enable debug logging
            **llm_kwargs: Arguments for vLLM LLM
        """
        self.watermark_generator = watermark_generator
        self.debug = debug

        # Force eager execution for simpler debugging
        llm_kwargs.setdefault("enforce_eager", True)

        # Create the base LLM
        self.llm = LLM(model=model, **llm_kwargs)

        # Replace the sampler with our watermark sampler
        self._replace_sampler()

        if self.debug:
            logger.info(f"Created WatermarkedLLM with model: {model}")

    def _replace_sampler(self):
        """Replace vLLM's sampler with our watermark sampler."""
        try:
            # Create our watermark sampler
            watermark_sampler = WatermarkSampler(
                watermark_generator=self.watermark_generator, debug=self.debug
            )

            # Find and replace samplers in the engine
            if hasattr(self.llm, "llm_engine"):
                engine = self.llm.llm_engine

                # Navigate to model executor and find samplers
                samplers_replaced = 0

                if hasattr(engine, "model_executor"):
                    # V0 structure
                    model_executor = engine.model_executor
                    samplers_replaced += self._replace_samplers_in_executor(
                        model_executor, watermark_sampler
                    )

                elif hasattr(engine, "engine_core"):
                    # V1 structure
                    engine_core = engine.engine_core
                    if hasattr(engine_core, "model_executor"):
                        model_executor = engine_core.model_executor
                        samplers_replaced += self._replace_samplers_in_executor(
                            model_executor, watermark_sampler
                        )

                if samplers_replaced > 0:
                    logger.info(
                        f"Successfully replaced {samplers_replaced} sampler(s) with WatermarkSampler"
                    )
                else:
                    logger.warning("No samplers found to replace")

        except Exception as e:
            logger.error(f"Failed to replace sampler: {e}")
            raise

    def _replace_samplers_in_executor(self, model_executor, watermark_sampler) -> int:
        """Replace samplers in a model executor. Returns number of samplers replaced."""
        count = 0

        # Check driver worker
        if hasattr(model_executor, "driver_worker"):
            if hasattr(model_executor.driver_worker, "model_runner"):
                runner = model_executor.driver_worker.model_runner
                if hasattr(runner, "sampler"):
                    old_sampler = runner.sampler
                    runner.sampler = watermark_sampler
                    count += 1
                    if self.debug:
                        logger.debug(
                            f"Replaced sampler in driver_worker: {type(old_sampler).__name__} -> WatermarkSampler"
                        )

            # V1 might have nested worker structure
            if hasattr(model_executor.driver_worker, "worker") and hasattr(
                model_executor.driver_worker.worker, "model_runner"
            ):
                runner = model_executor.driver_worker.worker.model_runner
                if hasattr(runner, "sampler"):
                    old_sampler = runner.sampler
                    runner.sampler = watermark_sampler
                    count += 1
                    if self.debug:
                        logger.debug(
                            f"Replaced sampler in driver_worker.worker: {type(old_sampler).__name__} -> WatermarkSampler"
                        )

        # Check if model_executor itself has a model_runner (some configurations)
        if hasattr(model_executor, "model_runner") and hasattr(
            model_executor.model_runner, "sampler"
        ):
            old_sampler = model_executor.model_runner.sampler
            model_executor.model_runner.sampler = watermark_sampler
            count += 1
            if self.debug:
                logger.debug(
                    f"Replaced sampler in model_executor: {type(old_sampler).__name__} -> WatermarkSampler"
                )

        # Check workers list (distributed case)
        if hasattr(model_executor, "workers"):
            for i, worker in enumerate(model_executor.workers):
                if hasattr(worker, "model_runner") and hasattr(
                    worker.model_runner, "sampler"
                ):
                    old_sampler = worker.model_runner.sampler
                    worker.model_runner.sampler = watermark_sampler
                    count += 1
                    if self.debug:
                        logger.debug(
                            f"Replaced sampler in worker[{i}]: {type(old_sampler).__name__} -> WatermarkSampler"
                        )

        return count

    def generate(self, prompts, sampling_params=None, **kwargs):
        """Generate text with watermarking applied."""
        if self.debug:
            num_prompts = len(prompts) if isinstance(prompts, list) else 1
            logger.debug(f"Generating for {num_prompts} prompt(s)")

        return self.llm.generate(prompts, sampling_params=sampling_params, **kwargs)

    def get_tokenizer(self):
        """Get the tokenizer."""
        return self.llm.get_tokenizer()

    def __getattr__(self, name):
        """Delegate unknown attributes to the base LLM."""
        return getattr(self.llm, name)


def create_watermarked_llm(
    model: str, watermark_generator: WmGenerator, debug: bool = False, **llm_kwargs
) -> WatermarkedLLM:
    """Create a watermarked LLM."""
    return WatermarkedLLM(
        model=model, watermark_generator=watermark_generator, debug=debug, **llm_kwargs
    )
