import torch
from loguru import logger

from vllm_watermark.watermark_generators.base import WmGenerator

# Import the correct sampler based on vLLM version
try:
    # Try V1 sampler first
    from vllm.v1.sample.metadata import SamplingMetadata
    from vllm.v1.sample.sampler import Sampler as V1Sampler

    base_sampler_class = V1Sampler
    logger.info("Using V1 sampler")
except ImportError:
    try:
        # Fall back to V0 sampler
        from vllm.model_executor.layers.sampler import Sampler
        from vllm.model_executor.sampling_metadata import SamplingMetadata

        base_sampler_class = Sampler
        logger.info("Using V0 sampler")

        # This currently doesn't work because the sampler output is not iterable
        # You shouldn't see this error if export is set properly
        """
        [rank0]:   File "/home/av787/anaconda3/envs/ml_dev311/lib/python3.11/site-packages/vllm/engine/llm_engine.py", line 1409, in step
        [rank0]:     self._advance_to_next_step(
        [rank0]:   File "/home/av787/anaconda3/envs/ml_dev311/lib/python3.11/site-packages/vllm/engine/llm_engine.py", line 1175, in _advance_to_next_step
        [rank0]:     zip(seq_group_metadata_list, output, scheduled_seq_groups):
        [rank0]:     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        [rank0]: TypeError: 'SamplerOutput' object is not iterable
        """
    except ImportError:
        raise ImportError("Could not import any sampler class from vLLM")


# Override the sampler to apply Watermarking
class CustomSampler(base_sampler_class):
    def __init__(self, model, watermark_generator: WmGenerator, debug=False):
        super().__init__()
        self.llm = model
        self.watermark_generator = watermark_generator
        self.debug = debug

    def __call__(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ):
        if self.debug:
            logger.debug("WatermarkedSampler called")

        # For V1, completely override the sampling process
        return self.forward(
            logits, sampling_metadata, self.watermark_generator, self.debug
        )

    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        watermark_generator: WmGenerator,
        debug: bool = False,
    ):
        if debug:
            logger.debug("WatermarkedSampler.forward called")

        # Import the SamplerOutput class based on vLLM version
        is_v1_outputs = False
        try:
            from vllm.v1.outputs import SamplerOutput

            is_v1_outputs = True
            if debug:
                logger.debug("Using V1 SamplerOutput")
        except ImportError:
            from vllm.model_executor.layers.sampler import SamplerOutput

            if debug:
                logger.debug("Using V0 SamplerOutput")

        # Check if we should apply watermarking
        should_watermark, temperature, top_p = self._should_apply_watermarking(
            sampling_metadata
        )

        if should_watermark:
            if debug:
                logger.debug(
                    f"Applying watermarking with temp={temperature}, top_p={top_p}"
                )

            # Build n-gram contexts
            ngram_list = self._build_ngram_contexts(
                logits.shape[0], sampling_metadata, watermark_generator, debug
            )

            if ngram_list:
                # Apply Gumbel watermarking to get sampled tokens directly
                ngram_tokens = torch.tensor(
                    ngram_list, dtype=torch.long, device=logits.device
                )
                sampled_tokens = watermark_generator.sample_next(
                    logits, ngram_tokens, temperature, top_p
                )
                if debug:
                    logger.debug(
                        f"Watermarked sampling produced tokens: {sampled_tokens}"
                    )
            else:
                logger.warning(
                    "No n-gram context available, falling back to original sampler"
                )
                return super().forward(logits, sampling_metadata)
        else:
            if debug:
                logger.debug("Watermarking not applicable, using original sampler")
            return super().forward(logits, sampling_metadata)

        # Create SamplerOutput with our watermarked tokens
        # Shape expectations differ between versions:
        # - V1 expects a 1D tensor of shape [batch_size]
        # - V0 expects a 2D tensor of shape [batch_size, 1]
        if is_v1_outputs:
            sampled_token_ids = sampled_tokens  # [batch_size]
        else:
            sampled_token_ids = sampled_tokens.unsqueeze(-1)  # [batch_size, 1]

        return SamplerOutput(
            sampled_token_ids=sampled_token_ids,
            logprobs_tensors=None,  # We could compute logprobs if needed
        )

    def _should_apply_watermarking(self, sampling_metadata):
        """Check if watermarking should be applied and extract parameters."""
        has_temperature = False
        temperature = 1.0
        top_p = 1.0

        # Handle different vLLM versions (V0 vs V1)
        if hasattr(sampling_metadata, "seq_groups"):
            # V0 structure
            for seq_group in sampling_metadata.seq_groups:
                if hasattr(seq_group, "sampling_params"):
                    sampling_params = seq_group.sampling_params
                    if (
                        hasattr(sampling_params, "temperature")
                        and sampling_params.temperature is not None
                        and sampling_params.temperature >= 1e-5
                    ):
                        has_temperature = True
                        temperature = sampling_params.temperature
                        top_p = getattr(sampling_params, "top_p", 1.0)
                        break
        else:
            # V1 structure
            if (
                hasattr(sampling_metadata, "all_greedy")
                and not sampling_metadata.all_greedy
                and hasattr(sampling_metadata, "temperature")
                and sampling_metadata.temperature is not None
            ):
                temp_tensor = sampling_metadata.temperature
                if torch.any(temp_tensor >= 1e-5):
                    has_temperature = True
                    temperature = temp_tensor[0].item()
                    if (
                        hasattr(sampling_metadata, "top_p")
                        and sampling_metadata.top_p is not None
                    ):
                        top_p = sampling_metadata.top_p[0].item()

        return has_temperature, temperature, top_p

    def _build_ngram_contexts(
        self,
        batch_size,
        sampling_metadata,
        watermark_generator: WmGenerator,
        debug: bool = False,
    ):
        """Build n-gram contexts for watermarking."""
        ngram_list = []

        if hasattr(sampling_metadata, "seq_groups"):
            # V0: Build from seq_groups
            if debug:
                logger.debug("Using V0 sampling metadata structure")

            for seq_group in sampling_metadata.seq_groups:
                for seq_id in seq_group.seq_ids:
                    # Try to get token history
                    token_history = []
                    if hasattr(seq_group, "seq_data") and seq_id in seq_group.seq_data:
                        seq_data = seq_group.seq_data[seq_id]
                        token_history = seq_data.get_token_ids()
                    elif hasattr(seq_group, "seqs"):
                        seq = seq_group.seqs.get(seq_id)
                        if seq:
                            token_history = seq.get_token_ids()

                    # Build n-gram
                    ngram = [watermark_generator.pad_id] * max(
                        0, watermark_generator.ngram - len(token_history)
                    ) + token_history[-watermark_generator.ngram :]
                    ngram_list.append(ngram)

                    if debug:
                        logger.debug(f"Built V0 n-gram: {ngram}")
        else:
            # V1: Access token history from the worker's input batch
            if debug:
                logger.debug("V1 detected: Attempting to access real token history")

            # Try to access the input batch from the model runner
            # The sampler is part of the model runner which has access to input_batch
            input_batch = None

            # Try to find the input batch through various possible paths
            try:
                # Method 1: Check if we can access it through the engine's model executor
                if hasattr(self.llm, "llm_engine"):
                    engine = self.llm.llm_engine
                    if hasattr(engine, "engine_core") and hasattr(
                        engine.engine_core, "engine_core"
                    ):
                        engine_core = engine.engine_core.engine_core
                        if hasattr(engine_core, "model_executor"):
                            model_executor = engine_core.model_executor
                            if hasattr(model_executor, "driver_worker") and hasattr(
                                model_executor.driver_worker, "worker"
                            ):
                                worker = model_executor.driver_worker.worker
                                if hasattr(worker, "model_runner") and hasattr(
                                    worker.model_runner, "input_batch"
                                ):
                                    input_batch = worker.model_runner.input_batch
                                    if debug:
                                        logger.debug(
                                            "Found input_batch via engine_core path"
                                        )
            except Exception as e:
                logger.error(f"Could not access input_batch via engine_core: {e}")

            # Method 2: Try to get it from the current frame context (advanced reflection)
            if input_batch is None:
                try:
                    import inspect

                    # Look up the call stack to find the model runner
                    for frame_info in inspect.stack():
                        frame_locals = frame_info.frame.f_locals
                        # Look for 'self' that might be a model runner with input_batch
                        if "self" in frame_locals:
                            obj = frame_locals["self"]
                            if hasattr(obj, "input_batch"):
                                input_batch = obj.input_batch
                                if debug:
                                    logger.debug(
                                        f"Found input_batch via stack inspection: {type(obj).__name__}"
                                    )
                                break
                except Exception as e:
                    logger.error(
                        f"Could not access input_batch via stack inspection: {e}"
                    )

            if input_batch is not None:
                if debug:
                    logger.debug(
                        f"Successfully found input_batch with {input_batch.num_reqs} requests"
                    )

                # Build n-grams from real token history
                for i in range(min(batch_size, input_batch.num_reqs)):
                    try:
                        # Get the cached request state for this request
                        if hasattr(input_batch, "req_output_token_ids") and i < len(
                            input_batch.req_output_token_ids
                        ):
                            output_token_ids = input_batch.req_output_token_ids[i] or []
                            prompt_token_ids = []

                            # Extract prompt tokens
                            if hasattr(input_batch, "token_ids_cpu") and hasattr(
                                input_batch, "num_prompt_tokens"
                            ):
                                num_prompt_tokens = input_batch.num_prompt_tokens[i]
                                prompt_token_ids = input_batch.token_ids_cpu[
                                    i, :num_prompt_tokens
                                ].tolist()

                            # Build complete token history
                            all_token_ids = prompt_token_ids + output_token_ids

                            if debug:
                                logger.debug(
                                    f"Request {i}: {len(all_token_ids)} total tokens"
                                )

                            # Build n-gram context from the last n tokens
                            if len(all_token_ids) >= watermark_generator.ngram:
                                ngram = all_token_ids[-watermark_generator.ngram :]
                            else:
                                # Pad with pad_id if we don't have enough history
                                padding_needed = watermark_generator.ngram - len(
                                    all_token_ids
                                )
                                ngram = [
                                    watermark_generator.pad_id
                                ] * padding_needed + all_token_ids

                            ngram_list.append(ngram)
                            if debug:
                                logger.debug(f"Built n-gram: {ngram}")
                        else:
                            logger.warning(
                                f"Could not get token IDs for request {i}, using dummy context"
                            )
                            ngram = [
                                watermark_generator.pad_id
                            ] * watermark_generator.ngram
                            ngram_list.append(ngram)
                    except Exception as e:
                        logger.error(f"Error building n-gram for request {i}: {e}")
                        # Fallback to dummy context
                        ngram = [watermark_generator.pad_id] * watermark_generator.ngram
                        ngram_list.append(ngram)
            else:
                logger.warning(
                    "Could not access input_batch, falling back to dummy contexts"
                )
                # Fallback: Use dummy contexts
                for i in range(batch_size):
                    ngram = [watermark_generator.pad_id] * watermark_generator.ngram
                    ngram_list.append(ngram)

        return ngram_list
