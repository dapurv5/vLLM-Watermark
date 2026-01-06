"""
Simple, clean watermarked LLM implementation.

This approach creates a custom sampler and injects it directly into vLLM
without any monkey patching.
"""

import os
from typing import Optional

import torch
from loguru import logger
from vllm import LLM

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
        from vllm.v1.outputs import SamplerOutput
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
        watermarked_logits, sampled_tokens = self._apply_watermarking(
            logits, sampling_metadata
        )

        # If watermarking was applied, return the sampled tokens directly
        # (the watermark generator already applied temperature/top_p sampling)
        if sampled_tokens is not None:
            # Create SamplerOutput directly with our watermarked tokens
            # Reshape to match expected format: [batch_size, 1]
            if sampled_tokens.dim() == 1:
                sampled_tokens = sampled_tokens.unsqueeze(-1)

            if VLLM_VERSION == "V1":
                return SamplerOutput(
                    sampled_token_ids=sampled_tokens, logprobs_tensors=None
                )
            else:
                # For V0, construct the more complex SamplerOutput
                return self._create_v0_sampler_output(sampled_tokens, sampling_metadata)
        else:
            # No watermarking applied, use parent sampler
            return super().forward(watermarked_logits, sampling_metadata)

    def _apply_watermarking(
        self, logits: torch.Tensor, sampling_metadata: SamplingMetadata
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply watermarking by directly sampling tokens and forcing selection.

        Returns:
            tuple: (watermarked_logits, sampled_tokens)
                - watermarked_logits: Modified logits for fallback to parent sampler
                - sampled_tokens: Directly sampled tokens if watermarking applied, None otherwise
        """

        try:
            # Extract sampling parameters
            temperature, top_p = self._extract_sampling_params(sampling_metadata)

            # Only apply watermarking for non-greedy sampling
            if temperature <= 0.01:
                if self.debug:
                    logger.debug("Skipping watermarking for greedy sampling")
                return logits, None

            # Extract n-gram contexts
            ngram_contexts = self._extract_ngram_contexts(sampling_metadata)

            if not ngram_contexts:
                if self.debug:
                    logger.debug("No n-gram contexts found, skipping watermarking")
                return logits, None

            if self.debug:
                logger.debug(
                    f"About to apply watermarking with {len(ngram_contexts)} contexts"
                )

            # Apply watermarking by direct token sampling
            batch_size = logits.shape[0]
            if len(ngram_contexts) != batch_size:
                if self.debug:
                    logger.debug(
                        f"Mismatch: {len(ngram_contexts)} contexts vs {batch_size} batch size"
                    )
                return logits, None

            # Convert to tensor
            ngram_tokens = torch.tensor(
                ngram_contexts, dtype=torch.long, device=logits.device
            )

            # Use the watermark generator to sample tokens
            sampled_tokens = self.watermark_generator.sample_next(
                logits.to(torch.float32), ngram_tokens, temperature, top_p
            )

            # Commenting this out since we are constructing SamplerOutput directly
            # with our watermarked tokens

            # if sampled_tokens is not None:
            #     # Force selection of watermarked tokens by zeroing out all other logits
            #     watermarked_logits = torch.full_like(logits, -float("inf"))
            #     batch_indices = torch.arange(batch_size, device=logits.device)
            #     # Flatten sampled_tokens if it's nested (e.g., [[6660], [388]] -> [6660, 388])
            #     if sampled_tokens.dim() > 1:
            #         sampled_tokens = sampled_tokens.squeeze(-1)
            #     watermarked_logits[batch_indices, sampled_tokens] = 0.0

            #     if self.debug:
            #         logger.debug(
            #             f"Applied watermarking: selected tokens {sampled_tokens.tolist()}"
            #         )

            #     return watermarked_logits, sampled_tokens
            # else:
            #     if self.debug:
            #         logger.debug("Watermark sampling returned None")
            #     return logits, None
            return logits, sampled_tokens

        except Exception as e:
            logger.error(f"Error in watermarking: {e}")
            return logits, None

    def _create_v0_sampler_output(
        self, sampled_tokens: torch.Tensor, sampling_metadata: SamplingMetadata
    ) -> SamplerOutput:
        """Create a V0 SamplerOutput with the watermarked tokens."""
        # Import V0-specific classes
        from vllm.sequence import CompletionSequenceGroupOutput, Logprob, SequenceOutput

        outputs = []
        batch_size = sampled_tokens.shape[0]

        # Create outputs for each sequence group
        for i in range(batch_size):
            token_id = sampled_tokens[i, 0].item()  # Extract scalar token ID

            # Create minimal logprobs (we don't have the actual logprobs from watermark generator)
            logprobs = {token_id: Logprob(logprob=0.0, rank=None, decoded_token=None)}

            # Create SequenceOutput
            seq_output = SequenceOutput(
                parent_seq_id=i,  # Use index as sequence ID
                output_token=token_id,
                logprobs=logprobs,
            )

            # Create CompletionSequenceGroupOutput
            group_output = CompletionSequenceGroupOutput(
                samples=[seq_output], prompt_logprobs=None
            )

            outputs.append(group_output)

        # Create SamplerOutput with the constructed outputs and token tensor
        return SamplerOutput(outputs=outputs, sampled_token_ids=sampled_tokens)

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
                        f"V1 metadata - Prompt tokens type: {type(prompt_tokens)}"
                    )
                    logger.debug(
                        f"V1 metadata - Prompt tokens content: {prompt_tokens}"
                    )
                    logger.debug(
                        f"V1 metadata - Output tokens type: {type(output_tokens)}"
                    )
                    logger.debug(
                        f"V1 metadata - Output tokens content: {output_tokens}"
                    )

                    # Debug the full sampling metadata structure
                    logger.debug(f"Full sampling metadata: {vars(sampling_metadata)}")

                # Handle batched tokens properly - process each sequence in the batch
                # Convert to lists if they're tensors
                if prompt_tokens is not None and hasattr(prompt_tokens, "tolist"):
                    prompt_tokens = prompt_tokens.tolist()
                    # Get padding token from watermark generator's tokenizer
                    pad_token_id = None
                    if hasattr(self.watermark_generator, "tokenizer"):
                        tokenizer = self.watermark_generator.tokenizer
                        if hasattr(tokenizer, "pad_token_id"):
                            pad_token_id = tokenizer.pad_token_id
                    if pad_token_id is not None:
                        if isinstance(prompt_tokens[0], list):
                            # Batched case: remove padding from each sequence
                            prompt_tokens = [
                                [token for token in seq if token != pad_token_id]
                                for seq in prompt_tokens
                            ]
                        else:
                            # Single sequence case: remove padding
                            prompt_tokens = [
                                token
                                for token in prompt_tokens
                                if token != pad_token_id
                            ]
                if output_tokens is not None and hasattr(output_tokens, "tolist"):
                    output_tokens = output_tokens.tolist()

                # Determine if we have batched data
                is_batched_prompts = (
                    isinstance(prompt_tokens, list)
                    and len(prompt_tokens) > 0
                    and isinstance(prompt_tokens[0], list)
                )
                is_batched_outputs = (
                    isinstance(output_tokens, list)
                    and len(output_tokens) > 0
                    and isinstance(output_tokens[0], list)
                )

                # Handle batched vs single sequence
                if is_batched_prompts or is_batched_outputs:
                    # We have batched data - process each sequence separately
                    batch_prompt_tokens = (
                        prompt_tokens
                        if is_batched_prompts
                        else [prompt_tokens] if prompt_tokens else [[]]
                    )
                    batch_output_tokens = (
                        output_tokens
                        if is_batched_outputs
                        else [output_tokens] if output_tokens else [[]]
                    )

                    # Ensure both batches have the same length
                    max_batch_size = max(
                        len(batch_prompt_tokens), len(batch_output_tokens)
                    )

                    # Pad shorter batch with empty lists
                    while len(batch_prompt_tokens) < max_batch_size:
                        batch_prompt_tokens.append([])
                    while len(batch_output_tokens) < max_batch_size:
                        batch_output_tokens.append([])

                    if self.debug:
                        logger.debug(f"Processing {max_batch_size} sequences in batch")

                    # Process each sequence in the batch
                    for i in range(max_batch_size):
                        seq_prompt_tokens = (
                            batch_prompt_tokens[i] if batch_prompt_tokens[i] else []
                        )
                        seq_output_tokens = (
                            batch_output_tokens[i] if batch_output_tokens[i] else []
                        )

                        # Combine tokens for this sequence
                        all_tokens = seq_prompt_tokens + seq_output_tokens

                        if self.debug:
                            logger.debug(
                                f"Sequence {i}: Combined tokens length: {len(all_tokens)}"
                            )
                            logger.debug(
                                f"Sequence {i}: Prompt tokens: {seq_prompt_tokens}"
                            )
                            logger.debug(
                                f"Sequence {i}: Output tokens: {seq_output_tokens}"
                            )
                            if len(all_tokens) > 10:
                                logger.debug(
                                    f"Sequence {i}: Last 10 tokens: {all_tokens[-10:]}"
                                )
                            else:
                                logger.debug(f"Sequence {i}: All tokens: {all_tokens}")

                        # Extract last n-gram for this sequence - ONLY use generated tokens for watermarking
                        ngram_size = self.watermark_generator.ngram

                        # Only use output tokens for n-gram context (not prompt tokens)
                        if len(seq_output_tokens) >= ngram_size:
                            ngram_context = seq_output_tokens[-ngram_size:]
                        elif len(seq_output_tokens) > 0:
                            # Pad with zeros if not enough output tokens yet
                            ngram_context = [0] * (
                                ngram_size - len(seq_output_tokens)
                            ) + seq_output_tokens
                        else:
                            # No output tokens yet - skip this sequence
                            continue

                        ngram_contexts.append(ngram_context)

                else:
                    # Single sequence case
                    prompt_tokens = prompt_tokens if prompt_tokens else []
                    output_tokens = output_tokens if output_tokens else []

                    if self.debug:
                        logger.debug(
                            f"Single sequence: Prompt tokens length: {len(prompt_tokens)}"
                        )
                        logger.debug(
                            f"Single sequence: Output tokens length: {len(output_tokens)}"
                        )

                    # Extract last n-gram - ONLY use generated tokens for watermarking
                    ngram_size = self.watermark_generator.ngram
                    if len(output_tokens) >= ngram_size:
                        ngram_context = output_tokens[-ngram_size:]
                    elif len(output_tokens) > 0:
                        # Pad with zeros if not enough output tokens yet
                        ngram_context = [0] * (
                            ngram_size - len(output_tokens)
                        ) + output_tokens
                    else:
                        # No output tokens yet - skip watermarking
                        return ngram_contexts

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
        llm,
        watermark_generator: WmGenerator,
        debug: bool = False,
    ):
        """
        Create a watermarked LLM.

        Args:
            llm: vLLM LLM instance to add watermarking to
            watermark_generator: Watermark generator to use
            debug: Enable debug logging
        """
        from vllm import LLM

        if not isinstance(llm, LLM):
            raise ValueError(f"Expected vLLM LLM instance, got {type(llm)}")

        self.llm = llm
        self.watermark_generator = watermark_generator
        self.debug = debug

        # Replace the sampler with our watermark sampler
        self._replace_sampler()

        if self.debug:
            model_name = getattr(llm, "model_name", "unknown")
            logger.info(f"Created WatermarkedLLM with model: {model_name}")

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
                logger.info(f"Engine: {engine}")
                # Navigate to model executor and find samplers
                samplers_replaced = 0

                if hasattr(engine, "model_executor"):
                    # V0 structure
                    model_executor = engine.model_executor
                    logger.info(f"Model executor: {model_executor}")
                    samplers_replaced += self._replace_samplers_in_executor(
                        model_executor, watermark_sampler
                    )

                elif hasattr(engine, "engine_core"):
                    # V1 structure
                    logger.info(f"Engine core: {engine.engine_core}")
                    engine_core = engine.engine_core

                    # Debug: Log engine_core structure
                    if self.debug:
                        logger.debug(f"Engine core type: {type(engine_core)}")
                        logger.debug(f"Engine core attributes: {dir(engine_core)}")

                    # Handle V1 multiprocessing case (SyncMPClient)
                    if hasattr(engine_core, "engine_core"):
                        # Multiprocessing case - get the actual engine core
                        actual_engine_core = engine_core.engine_core
                        logger.info(f"Actual engine core (MP): {actual_engine_core}")
                        if hasattr(actual_engine_core, "model_executor"):
                            model_executor = actual_engine_core.model_executor
                            logger.info(f"Model executor (MP): {model_executor}")
                            samplers_replaced += self._replace_samplers_in_executor(
                                model_executor, watermark_sampler
                            )
                    elif hasattr(engine_core, "model_executor"):
                        # Single process case
                        model_executor = engine_core.model_executor
                        logger.info(f"Model executor: {model_executor}")
                        samplers_replaced += self._replace_samplers_in_executor(
                            model_executor, watermark_sampler
                        )

                if samplers_replaced > 0:
                    logger.info(
                        f"Successfully replaced {samplers_replaced} sampler(s) with WatermarkSampler"
                    )
                else:
                    logger.warning("No samplers found to replace")

                    # Check if this is a V1 multiprocessing issue
                    if hasattr(engine, "engine_core") and "SyncMPClient" in str(
                        type(engine.engine_core)
                    ):
                        logger.warning(
                            "Detected vLLM V1 with multiprocessing enabled. "
                            "For watermarking to work, please set: VLLM_ENABLE_V1_MULTIPROCESSING=0"
                        )

        except Exception as e:
            logger.error(f"Failed to replace sampler: {e}")
            raise

    def _replace_samplers_in_executor(self, model_executor, watermark_sampler) -> int:
        """Replace samplers in a model executor. Returns number of samplers replaced.
    
        Works across different vLLM executor layouts by:
          1) targeting common paths (fast path), and
          2) falling back to a safe bounded object-graph walk.
        """
        logger = globals().get("logger", None)
    
        def _dbg(msg: str):
            if self.debug and logger is not None:
                logger.debug(msg)
    
        # --- Helpers -------------------------------------------------------------
    
        def _try_set_sampler(obj) -> int:
            """Try to set sampler-like attributes on `obj`. Returns 1 if replaced, else 0."""
            if obj is None:
                return 0
    
            # Common sampler attribute names seen across versions / refactors.
            sampler_attr_candidates = ("sampler", "_sampler", "sampler_layer")
    
            for attr in sampler_attr_candidates:
                if hasattr(obj, attr):
                    try:
                        old = getattr(obj, attr)
                        # Skip if it's already the watermark sampler
                        if old is watermark_sampler:
                            return 0
                        setattr(obj, attr, watermark_sampler)
                        _dbg(
                            f"Replaced {type(obj).__name__}.{attr}: "
                            f"{type(old).__name__} -> {type(watermark_sampler).__name__}"
                        )
                        return 1
                    except Exception as e:
                        _dbg(
                            f"Failed replacing {type(obj).__name__}.{attr}: {e!r}"
                        )
            return 0
    
        def _maybe_replace_on_runner(container) -> int:
            """If `container` has a runner-like attribute, try to replace sampler on that runner."""
            if container is None:
                return 0
    
            # runner attribute names across vLLM versions/configs
            runner_attr_candidates = ("model_runner", "_model_runner", "runner")
    
            for rattr in runner_attr_candidates:
                if hasattr(container, rattr):
                    runner = getattr(container, rattr, None)
                    if runner is None:
                        continue
                    # Prefer replacing on runner itself.
                    replaced = _try_set_sampler(runner)
                    if replaced:
                        return 1
    
                    # Some layouts tuck sampler one level deeper.
                    # e.g., runner.model_runner.sampler (rare, but cheap to try)
                    if hasattr(runner, "model_runner"):
                        replaced = _try_set_sampler(getattr(runner, "model_runner", None))
                        if replaced:
                            return 1
            return 0
    
        def _walk_object_graph(root, max_depth: int = 6):
            """Yield objects reachable from `root` (bounded), avoiding cycles."""
            seen = set()
    
            def push(obj, depth):
                if obj is None:
                    return
                oid = id(obj)
                if oid in seen:
                    return
                seen.add(oid)
    
                yield (obj, depth)
    
                if depth >= max_depth:
                    return
    
                # Expand common containers
                if isinstance(obj, dict):
                    for v in obj.values():
                        yield from push(v, depth + 1)
                    return
                if isinstance(obj, (list, tuple, set)):
                    for v in obj:
                        yield from push(v, depth + 1)
                    return
    
                # Expand object attributes
                try:
                    d = getattr(obj, "__dict__", None)
                except Exception:
                    d = None
    
                if isinstance(d, dict):
                    for v in d.values():
                        # Skip obviously huge tensors/modules if you want; here we keep it generic.
                        yield from push(v, depth + 1)
    
            yield from push(root, 0)
    
        # --- Debug introspection -------------------------------------------------
    
        if self.debug:
            _dbg(f"Model executor type: {type(model_executor)}")
            # Avoid dumping gigantic dir() logs unless you really want it.
            # _dbg(f"Model executor attributes: {dir(model_executor)}")
    
        # --- Fast paths (cheap checks) ------------------------------------------
    
        count = 0
        replaced_runner_ids = set()
    
        def _count_replace(obj) -> None:
            nonlocal count
            if obj is None:
                return
            oid = id(obj)
            if oid in replaced_runner_ids:
                return
            # Try direct sampler replacement on obj
            if _try_set_sampler(obj):
                replaced_runner_ids.add(oid)
                count += 1
    
        def _count_replace_container(container) -> None:
            """Try replace via runner inside container."""
            nonlocal count
            if container is None:
                return
            if _maybe_replace_on_runner(container):
                # Mark the runner itself if possible to avoid double counting
                for rattr in ("model_runner", "_model_runner", "runner"):
                    if hasattr(container, rattr):
                        runner = getattr(container, rattr, None)
                        if runner is not None:
                            replaced_runner_ids.add(id(runner))
                            break
                count += 1
    
        # 1) model_executor.model_runner(.sampler)
        _count_replace_container(model_executor)
    
        # 2) driver/worker patterns (present in some versions, absent in others)
        for attr in ("driver_worker", "_driver_worker", "driver"):
            if hasattr(model_executor, attr):
                dw = getattr(model_executor, attr, None)
                _count_replace_container(dw)
    
                # Some v1-ish nestings: driver_worker.worker.model_runner
                for nested in ("worker", "_worker"):
                    if hasattr(dw, nested):
                        _count_replace_container(getattr(dw, nested, None))
    
        # 3) workers collections (distributed cases)
        for attr in ("workers", "worker", "worker_pool", "tp_workers", "pp_workers"):
            if hasattr(model_executor, attr):
                ws = getattr(model_executor, attr, None)
                if isinstance(ws, dict):
                    iterable = ws.values()
                elif isinstance(ws, (list, tuple, set)):
                    iterable = ws
                else:
                    iterable = [ws]
                for w in iterable:
                    _count_replace_container(w)
    
        # If fast paths worked, we're done.
        if count > 0:
            return count
    
        # --- Fallback: bounded graph walk ---------------------------------------
        # Find any object that *either*:
        #   - is a runner-like container with model_runner, or
        #   - directly has a sampler-like attr.
        #
        # This is what fixes “sometimes it can not find the sampler to replace”.
        for obj, depth in _walk_object_graph(model_executor, max_depth=6):
            # First: direct sampler attrs on obj
            if id(obj) not in replaced_runner_ids and _try_set_sampler(obj):
                replaced_runner_ids.add(id(obj))
                count += 1
                continue
    
            # Second: sampler on runner inside obj
            # (common case: worker.model_runner.sampler)
            if _maybe_replace_on_runner(obj):
                # Record runner id if found
                for rattr in ("model_runner", "_model_runner", "runner"):
                    if hasattr(obj, rattr):
                        runner = getattr(obj, rattr, None)
                        if runner is not None:
                            replaced_runner_ids.add(id(runner))
                            break
                count += 1
    
        return count

    def generate(self, prompts, sampling_params=None, **kwargs):
        """Generate text with watermarking applied."""
        if self.debug:
            num_prompts = len(prompts) if isinstance(prompts, list) else 1
            logger.debug(f"Generating for {num_prompts} prompt(s)")

        # Filter out unsupported arguments before passing to vLLM
        supported_args = {}
        unsupported_args = {}

        # List of known supported arguments for vLLM LLM.generate()
        vllm_supported_args = {"use_tqdm", "lora_request", "prompt_adapter_request"}

        for key, value in kwargs.items():
            if key in vllm_supported_args:
                supported_args[key] = value
            else:
                unsupported_args[key] = value

        if unsupported_args and self.debug:
            logger.debug(
                f"Filtering out unsupported arguments: {list(unsupported_args.keys())}"
            )

        return self.llm.generate(
            prompts, sampling_params=sampling_params, **supported_args
        )

    def get_tokenizer(self):
        """Get the tokenizer."""
        return self.llm.get_tokenizer()

    def __getattr__(self, name):
        """Delegate unknown attributes to the base LLM."""
        return getattr(self.llm, name)


def create_watermarked_llm(
    llm, watermark_generator: WmGenerator, debug: bool = False
) -> WatermarkedLLM:
    """
    Create a watermarked LLM using a clean sampler override approach.

    Args:
        llm: vLLM LLM instance to add watermarking to
        watermark_generator: Watermark generator instance
        debug: Enable debug logging

    Returns:
        WatermarkedLLM instance with watermarking capabilities

    Raises:
        ValueError: If llm or watermark_generator is invalid
        RuntimeError: If watermark injection fails
    """
    from vllm import LLM

    # Input validation
    if not isinstance(llm, LLM):
        raise ValueError(f"Expected vLLM LLM instance, got {type(llm)}")

    if watermark_generator is None:
        raise ValueError("Watermark generator cannot be None")

    if not hasattr(watermark_generator, "sample_next"):
        raise ValueError("Watermark generator must have a 'sample_next' method")

    try:
        return WatermarkedLLM(
            llm=llm,
            watermark_generator=watermark_generator,
            debug=debug,
        )
    except Exception as e:
        logger.error(f"Failed to create watermarked LLM: {e}")
        raise RuntimeError(f"Watermark injection failed: {e}") from e
