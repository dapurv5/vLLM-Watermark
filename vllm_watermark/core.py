from enum import Enum
from typing import Union

from loguru import logger
from vllm.entrypoints.llm import LLM
from vllm.model_executor.layers.sampler import Sampler
from vllm.v1.sample.sampler import Sampler as V1Sampler  # type: ignore


class WatermarkingAlgorithm(Enum):
    """Supported watermarking algorithms for generation."""

    OPENAI = "openai"
    MARYLAND = "maryland"
    MARYLAND_L = "maryland_l"  # Maryland with logit processor (no sampler patching)
    PF = "pf"


class DetectionAlgorithm(Enum):
    """Supported detection algorithms (may have multiple variants per watermarking algorithm)."""

    OPENAI = "openai"
    OPENAI_Z = "openai_z"
    MARYLAND = "maryland"
    MARYLAND_Z = "maryland_z"
    PF = "pf"


class WatermarkUtils:
    """Shared utility functions for watermarking components."""

    @staticmethod
    def infer_vocab_size(model, tokenizer):
        """Infer vocab size from model to handle Llama tokenizer issues."""
        # Try to get vocab size from vLLM model config first
        if hasattr(model, "llm_engine"):
            try:
                if hasattr(model.llm_engine, "model_executor"):
                    if hasattr(model.llm_engine.model_executor, "driver_worker"):
                        worker = model.llm_engine.model_executor.driver_worker
                        if hasattr(worker, "model_runner") and hasattr(
                            worker.model_runner, "model"
                        ):
                            model_config = worker.model_runner.model.config
                            if hasattr(model_config, "vocab_size"):
                                logger.info(
                                    f"Found vocab size from model config: {model_config.vocab_size}"
                                )
                                return model_config.vocab_size
            except Exception as e:
                logger.debug(f"Could not get vocab size from model config: {e}")

        # Try HuggingFace model approach
        if hasattr(model, "config") and hasattr(model.config, "vocab_size"):
            logger.info(
                f"Found vocab size from HF model config: {model.config.vocab_size}"
            )
            return model.config.vocab_size

        # Fallback to tokenizer methods
        try:
            vocab_size = len(tokenizer.get_vocab())
            logger.info(f"Found vocab size from tokenizer len(): {vocab_size}")
            return vocab_size
        except Exception as e:
            logger.debug(f"Could not get vocab size from tokenizer.get_vocab(): {e}")

        try:
            vocab_size = tokenizer.vocab_size
            logger.info(f"Found vocab size from tokenizer.vocab_size: {vocab_size}")
            return vocab_size
        except Exception as e:
            logger.debug(f"Could not get vocab size from tokenizer.vocab_size: {e}")

        # Final fallback
        logger.warning("Could not determine vocab size, using default 50257")
        return 50257

    @staticmethod
    def get_tokenizer(model_or_tokenizer):
        """Extract tokenizer from model or return tokenizer directly."""
        if hasattr(model_or_tokenizer, "get_tokenizer"):
            # vLLM model
            return model_or_tokenizer.get_tokenizer()
        elif hasattr(model_or_tokenizer, "tokenizer"):
            # HuggingFace model with tokenizer attribute
            return model_or_tokenizer.tokenizer
        elif hasattr(model_or_tokenizer, "encode"):
            # Already a tokenizer
            return model_or_tokenizer
        else:
            raise ValueError(
                "Could not extract tokenizer from the provided object. "
                "Please provide either a model with get_tokenizer() method or a tokenizer directly."
            )


class WatermarkedLLM:
    def __init__(
        self,
        llm: LLM,
        sampler: Union[Sampler, V1Sampler] = None,
        logit_processor=None,
        debug: bool = False,
    ):
        self.llm = llm
        self.sampler = sampler
        self.logit_processor = logit_processor
        self.debug = debug

        # Validate that exactly one of sampler or logit_processor is provided
        if sampler is not None and logit_processor is not None:
            raise ValueError(
                "Cannot provide both sampler and logit_processor. Choose one approach."
            )
        if sampler is None and logit_processor is None:
            raise ValueError("Must provide either sampler or logit_processor.")

        if self.sampler is not None:
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
            engine_core = self.llm.llm_engine.engine_core  # type: ignore
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
                    model_executor.model_runner, "sampler"  # type: ignore
                ):
                    possible_paths.append(
                        ("model_executor.model_runner", model_executor.model_runner)  # type: ignore
                    )

                # Check if model_executor has driver_worker
                if hasattr(model_executor, "driver_worker") and hasattr(
                    model_executor.driver_worker, "model_runner"  # type: ignore
                ):
                    possible_paths.append(
                        (
                            "model_executor.driver_worker.model_runner",
                            model_executor.driver_worker.model_runner,  # type: ignore
                        )
                    )

                # Check if model_executor has workers list
                if hasattr(model_executor, "workers"):
                    for i, worker in enumerate(model_executor.workers):  # type: ignore
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
                watermarked_sampler = self.sampler
                model_runner.sampler = watermarked_sampler
                logger.info(
                    f"Successfully patched sampler: {type(original_sampler).__name__} -> {type(watermarked_sampler).__name__}"
                )
                patched_count += 1
            else:
                logger.warning(f"No sampler found at {path_name}")
        return patched_count

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

    def _is_v1_engine(self):
        """Check if we're using vLLM V1 engine."""
        return hasattr(self.llm.llm_engine, "engine_core")

    def generate(self, prompts, sampling_params=None, **kwargs):
        if self.debug:
            logger.debug("WatermarkedLLM.generate called")

        if self.logit_processor is not None:
            # Check if we're using V1 engine, which doesn't support logits processors
            if self._is_v1_engine():
                raise RuntimeError(
                    "vLLM V1 engine does not support logits processors. "
                    "To use MARYLAND_L watermarking, you have two options:\n"
                    "1. Disable V1 entirely by setting: VLLM_USE_V1=0\n"
                    "2. Use MARYLAND instead of MARYLAND_L (uses sampler patching)\n\n"
                    "Example:\n"
                    "  import os\n"
                    "  os.environ['VLLM_USE_V1'] = '0'\n"
                    "  # Then recreate your LLM instance\n"
                    "  llm = LLM(model='meta-llama/Llama-3.2-1B')\n"
                    "  wm_llm = WatermarkedLLMs.create(llm, algo=WatermarkingAlgorithm.MARYLAND_L, ...)"
                )

            # Use logit processor approach (V0 engine)
            # Import here to avoid circular imports
            from vllm import SamplingParams

            # If no sampling params provided, create default ones
            if sampling_params is None:
                sampling_params = SamplingParams()

            # Clone sampling params to avoid modifying the original
            # Use the built-in clone method to preserve all parameters correctly
            modified_sampling_params = sampling_params.clone()

            # Add our logit processor to the existing ones
            if modified_sampling_params.logits_processors is None:
                modified_sampling_params.logits_processors = []

            modified_sampling_params.logits_processors.append(self.logit_processor)

            # Generate with the modified sampling parameters
            return self.llm.generate(
                prompts, sampling_params=modified_sampling_params, **kwargs
            )
        else:
            # Use sampler patching approach (traditional)
            return self.llm.generate(prompts, sampling_params=sampling_params, **kwargs)


# Factory for creating watermarked LLMs
class WatermarkedLLMs:
    @staticmethod
    def create(
        model,
        algo: WatermarkingAlgorithm = WatermarkingAlgorithm.OPENAI,
        debug: bool = False,
        **kwargs,
    ) -> WatermarkedLLM:
        from vllm_watermark.watermark_generators import WatermarkGenerators

        if algo == WatermarkingAlgorithm.MARYLAND_L:
            # MARYLAND_L uses logit processors instead of sampler patching
            logit_processor = WatermarkGenerators.create(
                algo=algo, model=model, **kwargs
            )
            return WatermarkedLLM(model, logit_processor=logit_processor, debug=debug)

        elif algo == WatermarkingAlgorithm.OPENAI:
            # OPENAI uses sampler patching
            from vllm_watermark.samplers.custom_sampler import CustomSampler

            generator = WatermarkGenerators.create(algo=algo, model=model, **kwargs)
            sampler = CustomSampler(model, generator, debug=debug)  # type: ignore
            return WatermarkedLLM(model, sampler=sampler, debug=debug)

        elif algo == WatermarkingAlgorithm.MARYLAND:
            # MARYLAND uses sampler patching
            from vllm_watermark.samplers.custom_sampler import CustomSampler

            generator = WatermarkGenerators.create(algo=algo, model=model, **kwargs)
            sampler = CustomSampler(model, generator, debug=debug)  # type: ignore
            return WatermarkedLLM(model, sampler=sampler, debug=debug)

        elif algo == WatermarkingAlgorithm.PF:
            # PF uses sampler patching
            from vllm_watermark.samplers.custom_sampler import CustomSampler

            generator = WatermarkGenerators.create(algo=algo, model=model, **kwargs)
            sampler = CustomSampler(model, generator, debug=debug)  # type: ignore
            return WatermarkedLLM(model, sampler=sampler, debug=debug)

        else:
            raise ValueError(f"Unsupported watermarking algorithm: {algo}")
