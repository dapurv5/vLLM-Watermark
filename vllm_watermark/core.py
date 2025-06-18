import torch
from vllm.entrypoints.llm import LLM

from vllm_watermark.watermark_generators.base import BaseGenerator
from vllm_watermark.watermark_generators.gumbel_generator import GumbelGenerator


# Factory for creating watermarked LLMs
class WatermarkedLLMs:
    @staticmethod
    def create(model, algo: str = "gumbel", **kwargs) -> LLM:
        if algo == "gumbel":
            generator = GumbelGenerator(model, model.get_tokenizer(), **kwargs)
            return WatermarkedLLM(model, generator)
        raise ValueError(f"Unknown watermarking algorithm: {algo}")


class WatermarkedLLM:
    def __init__(self, llm: LLM, generator: BaseGenerator):
        self.llm = llm
        self.generator = generator
        self._patch_vllm_sampling()

    def _find_and_patch_samplers(self):
        """Find all samplers in the vLLM infrastructure and patch them."""
        patched_count = 0
        possible_paths = []

        # Check if we're using V1 or V0 engine
        if hasattr(self.llm.llm_engine, "engine_core"):
            # V1 Engine structure: llm_engine.engine_core.model_executor
            print("Detected V1 engine structure")

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
            print("Detected V0 engine structure")
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

        print(f"Found {len(possible_paths)} potential sampler paths:")
        for path_name, model_runner in possible_paths:
            print(f"  - {path_name}: sampler = {type(model_runner.sampler).__name__}")

        # Patch all found samplers
        for path_name, model_runner in possible_paths:
            if hasattr(model_runner, "sampler"):
                print(f"Patching sampler at {path_name}")
                original_sampler = model_runner.sampler
                watermarked_sampler = self._create_watermarked_sampler()
                model_runner.sampler = watermarked_sampler
                print(
                    f"Successfully patched sampler at {path_name}: {type(original_sampler).__name__} -> {type(watermarked_sampler).__name__}"
                )
                patched_count += 1
            else:
                print(f"Warning: No sampler found at {path_name}")

        return patched_count

    def _create_watermarked_sampler(self):
        """Create a watermarked sampler instance."""
        # Import the correct sampler based on vLLM version
        try:
            # Try V1 sampler first
            from vllm.v1.sample.metadata import SamplingMetadata
            from vllm.v1.sample.sampler import Sampler as V1Sampler

            base_sampler_class = V1Sampler
            print("Using V1 sampler")
        except ImportError:
            try:
                # Fall back to V0 sampler
                from vllm.model_executor.layers.sampler import Sampler
                from vllm.model_executor.sampling_metadata import SamplingMetadata

                base_sampler_class = Sampler
                print("Using V0 sampler")
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
                print("=== WatermarkedSampler called ===")
                print(f"Logits shape: {logits.shape}")

                # For V1, completely override the sampling process
                return self.forward(logits, sampling_metadata)

            def forward(
                self,
                logits: torch.Tensor,
                sampling_metadata: SamplingMetadata,
            ):
                print("=== WatermarkedSampler.forward called ===")
                print(f"Logits shape: {logits.shape}")

                # Import the SamplerOutput class based on vLLM version
                try:
                    from vllm.v1.outputs import SamplerOutput

                    print("Using V1 SamplerOutput")
                except ImportError:
                    from vllm.model_executor.layers.sampler import SamplerOutput

                    print("Using V0 SamplerOutput")

                # Check if we should apply watermarking
                should_watermark, temperature, top_p = self._should_apply_watermarking(
                    sampling_metadata
                )

                if should_watermark:
                    print(
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
                        print(f"Watermarked sampling produced tokens: {sampled_tokens}")
                    else:
                        print(
                            "No n-gram context available, falling back to original sampler"
                        )
                        return super().forward(logits, sampling_metadata)
                else:
                    print("Watermarking not applicable, using original sampler")
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
                else:
                    # V1: Use dummy contexts (limitation)
                    print("V1 detected: Using dummy n-gram context")
                    for i in range(batch_size):
                        ngram = [
                            watermark_llm.generator.pad_id
                        ] * watermark_llm.generator.ngram
                        ngram_list.append(ngram)

                return ngram_list

        return WatermarkedSampler()

    def _patch_vllm_sampling(self):
        """Patch vLLM's sampling mechanism to include watermarking."""
        print("Attempting to patch vLLM sampling...")

        patched_count = self._find_and_patch_samplers()

        if patched_count > 0:
            print(f"Successfully patched {patched_count} sampler(s)")
        else:
            print("Warning: No samplers were found and patched")
            print("Available attributes on llm_engine:")
            print(
                [attr for attr in dir(self.llm.llm_engine) if not attr.startswith("_")]
            )

    def generate(self, *args, **kwargs):
        print("WatermarkedLLM.generate called with kwargs:", kwargs)
        return self.llm.generate(*args, **kwargs)
