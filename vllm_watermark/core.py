import torch
from loguru import logger
from vllm.entrypoints.llm import LLM

from vllm_watermark.watermark_generators.base import BaseGenerator
from vllm_watermark.watermark_generators.gumbel_generator import GumbelGenerator


# Factory for creating watermarked LLMs
class WatermarkedLLMs:
    @staticmethod
    def create(model, algo: str = "gumbel", debug: bool = False, **kwargs) -> LLM:
        if algo == "gumbel":
            generator = GumbelGenerator(model, model.get_tokenizer(), **kwargs)
            return WatermarkedLLM(model, generator, debug=debug)
        raise ValueError(f"Unknown watermarking algorithm: {algo}")


class WatermarkedLLM:
    def __init__(self, llm: LLM, generator: BaseGenerator, debug: bool = False):
        self.llm = llm
        self.generator = generator
        self.debug = debug
        self._patch_vllm_sampling()

    def _find_and_patch_samplers(self):
        """Find all samplers in the vLLM infrastructure and patch them."""
        patched_count = 0
        possible_paths = []

        # Check if we're using V1 or V0 engine
        if hasattr(self.llm.llm_engine, "engine_core"):
            # V1 Engine structure: llm_engine.engine_core.model_executor
            logger.info("Detected V1 engine structure")

            # Check if we're using multiprocessing mode (SyncMPClient)
            engine_core = self.llm.llm_engine.engine_core
            if engine_core.__class__.__name__ == "SyncMPClient":
                raise RuntimeError(
                    "Cannot patch vLLM sampler when using V1 multiprocessing mode. "
                    "The engine runs in a separate process and the sampler is not accessible. "
                    "\n\nTo fix this, you have two options:"
                    "\n1. Disable V1 multiprocessing by setting: VLLM_ENABLE_V1_MULTIPROCESSING=0"
                    "\n2. Disable V1 entirely by setting: VLLM_USE_V1=0"
                    "\n\nExample:"
                    "\n  import os"
                    "\n  os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'"
                    "\n  # Then create your LLM instance"
                    "\n  llm = LLM(model='microsoft/DialoGPT-medium')"
                    "\n  wm_llm = WatermarkedLLMs.create(llm, algo='gumbel', seed=42, ngram=2)"
                )

            # V1 InprocClient case - we can access the engine_core directly
            if hasattr(engine_core, "engine_core"):
                actual_engine_core = engine_core.engine_core
                model_executor = actual_engine_core.model_executor

                # V1 uses driver_worker which wraps the actual worker
                if hasattr(model_executor, "driver_worker"):
                    driver_worker = model_executor.driver_worker
                    # The worker wrapper contains the actual worker
                    if hasattr(driver_worker, "worker") and hasattr(
                        driver_worker.worker, "model_runner"
                    ):
                        model_runner = driver_worker.worker.model_runner
                        if hasattr(model_runner, "sampler"):
                            possible_paths.append(
                                (
                                    "engine_core.engine_core.model_executor.driver_worker.worker.model_runner",
                                    model_runner,
                                )
                            )
                    # Some V1 setups may have model_runner directly on driver_worker
                    elif hasattr(driver_worker, "model_runner"):
                        model_runner = driver_worker.model_runner
                        if hasattr(model_runner, "sampler"):
                            possible_paths.append(
                                (
                                    "engine_core.engine_core.model_executor.driver_worker.model_runner",
                                    model_runner,
                                )
                            )

                # Check if model_executor has workers list (distributed case)
                if hasattr(model_executor, "workers"):
                    for i, worker in enumerate(model_executor.workers):
                        if hasattr(worker, "worker") and hasattr(
                            worker.worker, "model_runner"
                        ):
                            model_runner = worker.worker.model_runner
                            if hasattr(model_runner, "sampler"):
                                possible_paths.append(
                                    (
                                        f"engine_core.engine_core.model_executor.workers[{i}].worker.model_runner",
                                        model_runner,
                                    )
                                )
                        elif hasattr(worker, "model_runner"):
                            model_runner = worker.model_runner
                            if hasattr(model_runner, "sampler"):
                                possible_paths.append(
                                    (
                                        f"engine_core.engine_core.model_executor.workers[{i}].model_runner",
                                        model_runner,
                                    )
                                )

        else:
            # V0 Engine structure: llm_engine.model_executor
            logger.info("Detected V0 engine structure")
            if hasattr(self.llm.llm_engine, "model_executor"):
                model_executor = self.llm.llm_engine.model_executor

                # Check if model_executor is itself a worker with model_runner
                if hasattr(model_executor, "model_runner") and hasattr(
                    model_executor.model_runner, "sampler"
                ):
                    possible_paths.append(
                        ("model_executor.model_runner", model_executor.model_runner)
                    )

                # Check if model_executor has driver_worker
                if hasattr(model_executor, "driver_worker") and hasattr(
                    model_executor.driver_worker, "model_runner"
                ):
                    possible_paths.append(
                        (
                            "model_executor.driver_worker.model_runner",
                            model_executor.driver_worker.model_runner,
                        )
                    )

                # Check if model_executor has workers list
                if hasattr(model_executor, "workers"):
                    for i, worker in enumerate(model_executor.workers):
                        if hasattr(worker, "model_runner") and hasattr(
                            worker.model_runner, "sampler"
                        ):
                            possible_paths.append(
                                (
                                    f"model_executor.workers[{i}].model_runner",
                                    worker.model_runner,
                                )
                            )

        if self.debug:
            logger.debug(f"Found {len(possible_paths)} potential sampler paths:")
            for path_name, model_runner in possible_paths:
                logger.debug(
                    f"  - {path_name}: sampler = {type(model_runner.sampler).__name__}"
                )

        # Patch all found samplers
        for path_name, model_runner in possible_paths:
            if hasattr(model_runner, "sampler"):
                if self.debug:
                    logger.debug(f"Patching sampler at {path_name}")
                original_sampler = model_runner.sampler
                watermarked_sampler = self._create_watermarked_sampler()
                model_runner.sampler = watermarked_sampler
                logger.info(
                    f"Successfully patched sampler: {type(original_sampler).__name__} -> {type(watermarked_sampler).__name__}"
                )
                patched_count += 1
            else:
                logger.warning(f"No sampler found at {path_name}")

        return patched_count

    def _create_watermarked_sampler(self):
        """Create a watermarked sampler instance."""
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
            except ImportError:
                raise ImportError("Could not import any sampler class from vLLM")

        # Capture the WatermarkedLLM instance
        watermark_llm = self

        class WatermarkedSampler(base_sampler_class):
            def __call__(
                self,
                logits: torch.Tensor,
                sampling_metadata: SamplingMetadata,
            ):
                if watermark_llm.debug:
                    logger.debug("WatermarkedSampler called")

                # For V1, completely override the sampling process
                return self.forward(logits, sampling_metadata)

            def forward(
                self,
                logits: torch.Tensor,
                sampling_metadata: SamplingMetadata,
            ):
                if watermark_llm.debug:
                    logger.debug("WatermarkedSampler.forward called")

                # Import the SamplerOutput class based on vLLM version
                try:
                    from vllm.v1.outputs import SamplerOutput

                    if watermark_llm.debug:
                        logger.debug("Using V1 SamplerOutput")
                except ImportError:
                    from vllm.model_executor.layers.sampler import SamplerOutput

                    if watermark_llm.debug:
                        logger.debug("Using V0 SamplerOutput")

                # Check if we should apply watermarking
                should_watermark, temperature, top_p = self._should_apply_watermarking(
                    sampling_metadata
                )

                if should_watermark:
                    if watermark_llm.debug:
                        logger.debug(
                            f"Applying Gumbel watermarking with temp={temperature}, top_p={top_p}"
                        )

                    # Build n-gram contexts
                    ngram_list = self._build_ngram_contexts(
                        logits.shape[0], sampling_metadata
                    )

                    if ngram_list:
                        # Apply Gumbel watermarking to get sampled tokens directly
                        ngram_tokens = torch.tensor(
                            ngram_list, dtype=torch.long, device=logits.device
                        )
                        sampled_tokens = watermark_llm.generator.sample_next(
                            logits, ngram_tokens, temperature, top_p
                        )
                        if watermark_llm.debug:
                            logger.debug(
                                f"Watermarked sampling produced tokens: {sampled_tokens}"
                            )
                    else:
                        logger.warning(
                            "No n-gram context available, falling back to original sampler"
                        )
                        return super().forward(logits, sampling_metadata)
                else:
                    if watermark_llm.debug:
                        logger.debug(
                            "Watermarking not applicable, using original sampler"
                        )
                    return super().forward(logits, sampling_metadata)

                # Create SamplerOutput with our watermarked tokens
                # For V1, the expected shape is [batch_size, 1]
                sampled_token_ids = sampled_tokens.unsqueeze(
                    -1
                )  # Shape: [batch_size, 1]

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

            def _build_ngram_contexts(self, batch_size, sampling_metadata):
                """Build n-gram contexts for watermarking."""
                ngram_list = []

                if hasattr(sampling_metadata, "seq_groups"):
                    # V0: Build from seq_groups
                    if watermark_llm.debug:
                        logger.debug("Using V0 sampling metadata structure")

                    for seq_group in sampling_metadata.seq_groups:
                        for seq_id in seq_group.seq_ids:
                            # Try to get token history
                            token_history = []
                            if (
                                hasattr(seq_group, "seq_data")
                                and seq_id in seq_group.seq_data
                            ):
                                seq_data = seq_group.seq_data[seq_id]
                                token_history = seq_data.get_token_ids()
                            elif hasattr(seq_group, "seqs"):
                                seq = seq_group.seqs.get(seq_id)
                                if seq:
                                    token_history = seq.get_token_ids()

                            # Build n-gram
                            ngram = [watermark_llm.generator.pad_id] * max(
                                0, watermark_llm.generator.ngram - len(token_history)
                            ) + token_history[-watermark_llm.generator.ngram :]
                            ngram_list.append(ngram)

                            if watermark_llm.debug:
                                logger.debug(f"Built V0 n-gram: {ngram}")
                else:
                    # V1: Access token history from the worker's input batch
                    if watermark_llm.debug:
                        logger.debug(
                            "V1 detected: Attempting to access real token history"
                        )

                    # Try to access the input batch from the model runner
                    # The sampler is part of the model runner which has access to input_batch
                    input_batch = None

                    # Try to find the input batch through various possible paths
                    try:
                        # Method 1: Check if we can access it through the engine's model executor
                        if hasattr(watermark_llm.llm, "llm_engine"):
                            engine = watermark_llm.llm.llm_engine
                            if hasattr(engine, "engine_core") and hasattr(
                                engine.engine_core, "engine_core"
                            ):
                                engine_core = engine.engine_core.engine_core
                                if hasattr(engine_core, "model_executor"):
                                    model_executor = engine_core.model_executor
                                    if hasattr(
                                        model_executor, "driver_worker"
                                    ) and hasattr(
                                        model_executor.driver_worker, "worker"
                                    ):
                                        worker = model_executor.driver_worker.worker
                                        if hasattr(worker, "model_runner") and hasattr(
                                            worker.model_runner, "input_batch"
                                        ):
                                            input_batch = (
                                                worker.model_runner.input_batch
                                            )
                                            if watermark_llm.debug:
                                                logger.debug(
                                                    f"Found input_batch via engine_core path"
                                                )
                    except Exception as e:
                        logger.error(
                            f"Could not access input_batch via engine_core: {e}"
                        )

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
                                        if watermark_llm.debug:
                                            logger.debug(
                                                f"Found input_batch via stack inspection: {type(obj).__name__}"
                                            )
                                        break
                        except Exception as e:
                            logger.error(
                                f"Could not access input_batch via stack inspection: {e}"
                            )

                    if input_batch is not None:
                        if watermark_llm.debug:
                            logger.debug(
                                f"Successfully found input_batch with {input_batch.num_reqs} requests"
                            )

                        # Build n-grams from real token history
                        for i in range(min(batch_size, input_batch.num_reqs)):
                            try:
                                req_id = input_batch.req_ids[i]
                                # Get the cached request state for this request
                                if hasattr(
                                    input_batch, "req_output_token_ids"
                                ) and i < len(input_batch.req_output_token_ids):
                                    output_token_ids = (
                                        input_batch.req_output_token_ids[i] or []
                                    )
                                    prompt_token_ids = []

                                    # Extract prompt tokens
                                    if hasattr(
                                        input_batch, "token_ids_cpu"
                                    ) and hasattr(input_batch, "num_prompt_tokens"):
                                        num_prompt_tokens = (
                                            input_batch.num_prompt_tokens[i]
                                        )
                                        prompt_token_ids = input_batch.token_ids_cpu[
                                            i, :num_prompt_tokens
                                        ].tolist()

                                    # Build complete token history
                                    all_token_ids = prompt_token_ids + output_token_ids

                                    if watermark_llm.debug:
                                        logger.debug(
                                            f"Request {i}: {len(all_token_ids)} total tokens"
                                        )

                                    # Build n-gram context from the last n tokens
                                    if (
                                        len(all_token_ids)
                                        >= watermark_llm.generator.ngram
                                    ):
                                        ngram = all_token_ids[
                                            -watermark_llm.generator.ngram :
                                        ]
                                    else:
                                        # Pad with pad_id if we don't have enough history
                                        padding_needed = (
                                            watermark_llm.generator.ngram
                                            - len(all_token_ids)
                                        )
                                        ngram = [
                                            watermark_llm.generator.pad_id
                                        ] * padding_needed + all_token_ids

                                    ngram_list.append(ngram)
                                    if watermark_llm.debug:
                                        logger.debug(f"Built n-gram: {ngram}")
                                else:
                                    logger.warning(
                                        f"Could not get token IDs for request {i}, using dummy context"
                                    )
                                    ngram = [
                                        watermark_llm.generator.pad_id
                                    ] * watermark_llm.generator.ngram
                                    ngram_list.append(ngram)
                            except Exception as e:
                                logger.error(
                                    f"Error building n-gram for request {i}: {e}"
                                )
                                # Fallback to dummy context
                                ngram = [
                                    watermark_llm.generator.pad_id
                                ] * watermark_llm.generator.ngram
                                ngram_list.append(ngram)
                    else:
                        logger.warning(
                            "Could not access input_batch, falling back to dummy contexts"
                        )
                        # Fallback: Use dummy contexts
                        for i in range(batch_size):
                            ngram = [
                                watermark_llm.generator.pad_id
                            ] * watermark_llm.generator.ngram
                            ngram_list.append(ngram)

                return ngram_list

        return WatermarkedSampler()

    def _patch_vllm_sampling(self):
        """Patch vLLM's sampling mechanism to include watermarking."""
        logger.info("Initializing watermarked sampling...")

        patched_count = self._find_and_patch_samplers()

        if patched_count > 0:
            logger.info(f"Successfully patched {patched_count} sampler(s)")
        else:
            logger.warning("Warning: No samplers were found and patched")
            if self.debug:
                logger.debug("Available attributes on llm_engine:")
                logger.debug(
                    [
                        attr
                        for attr in dir(self.llm.llm_engine)
                        if not attr.startswith("_")
                    ]
                )

    def generate(self, *args, **kwargs):
        if self.debug:
            logger.debug("WatermarkedLLM.generate called")
        return self.llm.generate(*args, **kwargs)
