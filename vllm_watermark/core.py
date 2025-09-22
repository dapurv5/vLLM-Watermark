import logging
import os
import sys
from contextlib import contextmanager
from enum import Enum
from typing import Optional, Union

from loguru import logger
from vllm.entrypoints.llm import LLM
from vllm.model_executor.layers.sampler import Sampler
from vllm.v1.sample.sampler import Sampler as V1Sampler  # type: ignore


class WatermarkingAlgorithm(Enum):
    """Supported watermarking algorithms for generation."""

    OPENAI = "openai"
    OPENAI_DR = "openai_dr"
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
    """Legacy wrapper for backward compatibility. Prefer using WatermarkedLLMs.create() instead."""

    def __init__(
        self,
        llm: LLM,
        sampler: Optional[Union[Sampler, V1Sampler]] = None,
        logit_processor=None,
        debug: bool = False,
        algo: Optional[WatermarkingAlgorithm] = None,
        suppress_serial_logs: bool = True,
        use_new_engine: bool = True,  # New parameter to control which engine to use
    ):
        self.llm = llm
        self.sampler = sampler
        self.logit_processor = logit_processor
        self.debug = debug
        self.algo = algo
        self.suppress_serial_logs = suppress_serial_logs
        self.use_new_engine = use_new_engine

        # Enforce serial generation for OpenAI-style watermarks to avoid batching issues
        # Note: The new engine handles this more elegantly
        self.force_serial_generation = (
            algo in {WatermarkingAlgorithm.OPENAI, WatermarkingAlgorithm.OPENAI_DR}
            if algo is not None and not use_new_engine
            else False
        )

        # Validate that exactly one of sampler or logit_processor is provided
        if sampler is not None and logit_processor is not None:
            raise ValueError(
                "Cannot provide both sampler and logit_processor. Choose one approach."
            )
        if sampler is None and logit_processor is None:
            raise ValueError("Must provide either sampler or logit_processor.")

        # Use legacy monkey patching only if new engine is disabled
        if self.sampler is not None and not use_new_engine:
            self._patch_vllm_sampling()
        elif self.sampler is not None and use_new_engine:
            logger.warning(
                "Sampler-based watermarking with new engine is handled automatically. "
                "The provided sampler parameter will be ignored."
            )

    @contextmanager
    def _quiet_serial_logs(self, preserve_stdio: bool = False):
        """Temporarily suppress library progress/logs during serial loop.

        When redirecting, we capture original stdio so the caller can briefly
        restore it around their own progress updates.
        """
        if not self.suppress_serial_logs:
            yield
            return
        logger_names = [
            "vllm",
            "vllm.engine",
            "vllm.model_executor",
            "vllm.worker",
            "transformers",
            "accelerate",
            "torch.distributed",
        ]
        previous_levels: dict[str, int] = {}
        prev_env: dict[str, str | None] = {}
        old_stdout, old_stderr = sys.stdout, sys.stderr
        # Save originals for temporary restoration during progress updates
        self._original_stdout = old_stdout
        self._original_stderr = old_stderr
        try:
            # Reduce log levels to minimize per-call chatter
            for name in logger_names:
                lg = logging.getLogger(name)
                previous_levels[name] = lg.level
                lg.setLevel(logging.ERROR)

            # Disable tqdm/progress globally via env vars read at bar creation time
            for key, value in {
                "TQDM_DISABLE": "1",
                "DISABLE_TQDM": "1",
            }.items():
                prev_env[key] = os.environ.get(key)
                os.environ[key] = value

            if preserve_stdio:
                # Keep stdout/stderr so outer progress can render
                yield
            else:
                # Redirect stdout/stderr to suppress any progress bar writes from libraries
                with open(os.devnull, "w") as devnull:
                    sys.stdout = devnull
                    sys.stderr = devnull
                    yield
        finally:
            # Restore stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr

            # Restore env vars
            for key, val in prev_env.items():
                if val is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = val

            # Restore logger levels
            for name, level in previous_levels.items():
                logging.getLogger(name).setLevel(level)

            # Clean up
            try:
                del self._original_stdout
                del self._original_stderr
            except Exception:
                pass

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

        # Never forward progress_callback to vLLM; only used by serial wrapper
        progress_callback = kwargs.pop("progress_callback", None)

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
            # For OpenAI-style watermarks, avoid batched generation due to alignment issues
            if (
                self.force_serial_generation
                and isinstance(prompts, list)
                and len(prompts) > 1
            ):
                aggregated_outputs = []
                # Always suppress library output; we'll temporarily restore stdio for the callback
                with self._quiet_serial_logs(preserve_stdio=False):
                    for single_prompt in prompts:
                        single_result = self.llm.generate(
                            [single_prompt],
                            sampling_params=sampling_params,
                            **kwargs,
                        )
                        # vLLM returns a list for multiple inputs; extract the single item
                        if isinstance(single_result, list) and len(single_result) > 0:
                            aggregated_outputs.append(single_result[0])
                        else:
                            aggregated_outputs.append(single_result)
                        if callable(progress_callback):
                            try:
                                # Temporarily restore stdio so the outer tqdm can render
                                old_out, old_err = sys.stdout, sys.stderr
                                sys.stdout, sys.stderr = (
                                    getattr(self, "_original_stdout", old_out),
                                    getattr(self, "_original_stderr", old_err),
                                )
                                try:
                                    progress_callback(1)
                                finally:
                                    sys.stdout, sys.stderr = old_out, old_err
                            except Exception:
                                pass
                return aggregated_outputs

            return self.llm.generate(prompts, sampling_params=sampling_params, **kwargs)


# Factory for creating watermarked LLMs
class WatermarkedLLMs:
    @staticmethod
    def create(
        model,
        algo: WatermarkingAlgorithm = WatermarkingAlgorithm.OPENAI,
        debug: bool = False,
        use_new_engine: bool = True,
        **kwargs,
    ):
        """
        Create a watermarked LLM instance.

        Args:
            model: Model name/path or vLLM LLM instance
            algo: Watermarking algorithm to use
            debug: Enable debug logging
            use_new_engine: Use new clean engine (recommended) vs legacy monkey patching
            **kwargs: Additional arguments for watermark generator or vLLM

        Returns:
            Watermarked LLM instance
        """
        from vllm_watermark.watermark_generators import WatermarkGenerators

        # Split kwargs into generator args and vLLM args
        generator_kwargs = {}
        vllm_kwargs = {}

        # Known generator parameters
        generator_params = {
            "ngram",
            "seed",
            "seeding",
            "salt_key",
            "payload",
            "gamma",
            "delta",
            "vocab_size",
            "tokenizer",
        }

        for key, value in kwargs.items():
            if key in generator_params:
                generator_kwargs[key] = value
            else:
                vllm_kwargs[key] = value

        if use_new_engine:
            # Use the new clean engine approach
            from vllm_watermark.engine import create_watermarked_llm

            # Handle both string model names and LLM instances
            if isinstance(model, str):
                model_name = model
            else:
                # If model is already an LLM instance, we need to create a new one
                # with the same configuration for the new engine
                logger.warning(
                    "Using new engine with existing LLM instance. "
                    "Extracting model name and recreating LLM for clean integration."
                )
                # Try to extract model name from existing LLM
                if hasattr(model, "llm_engine") and hasattr(
                    model.llm_engine, "model_config"
                ):
                    model_name = model.llm_engine.model_config.model
                else:
                    raise ValueError(
                        "Cannot extract model name from LLM instance. "
                        "Please provide model name as string when using new engine."
                    )

            # Create watermark generator
            # For new engine, we always create the generator first
            generator = WatermarkGenerators.create(
                algo=algo,
                model=model_name,  # Pass model name for tokenizer inference
                **generator_kwargs,
            )

            # Create watermarked LLM with new engine
            return create_watermarked_llm(
                model=model_name,
                watermark_generator=generator,
                debug=debug,
                **vllm_kwargs,
            )

        else:
            # Use legacy approach with monkey patching
            logger.warning(
                "Using legacy engine with monkey patching. "
                "Consider migrating to use_new_engine=True for better stability."
            )

            # Ensure model is an LLM instance for legacy approach
            if isinstance(model, str):
                from vllm import LLM

                model = LLM(model=model, **vllm_kwargs)

            # Legacy logic for different algorithms
            try:
                import os

                use_v1_env = os.environ.get("VLLM_USE_V1")
                is_v1_disabled = use_v1_env is not None and use_v1_env.strip() == "0"
            except Exception:
                is_v1_disabled = False

            if algo == WatermarkingAlgorithm.MARYLAND and is_v1_disabled:
                logger.info(
                    "VLLM_USE_V1=0 detected; using MARYLAND_L (logit processor) instead of sampler patching."
                )
                algo = WatermarkingAlgorithm.MARYLAND_L

            if algo == WatermarkingAlgorithm.MARYLAND_L:
                # MARYLAND_L uses logit processors instead of sampler patching
                logit_processor = WatermarkGenerators.create(
                    algo=algo, model=model, **generator_kwargs
                )
                return WatermarkedLLM(
                    model,
                    logit_processor=logit_processor,
                    debug=debug,
                    algo=algo,
                    use_new_engine=False,
                )

            elif algo in {
                WatermarkingAlgorithm.OPENAI,
                WatermarkingAlgorithm.OPENAI_DR,
                WatermarkingAlgorithm.MARYLAND,
                WatermarkingAlgorithm.PF,
            }:
                # These algorithms use sampler patching in legacy mode
                from vllm_watermark.samplers.custom_sampler import CustomSampler

                generator = WatermarkGenerators.create(
                    algo=algo, model=model, **generator_kwargs
                )
                sampler = CustomSampler(model, generator, debug=debug)  # type: ignore
                return WatermarkedLLM(
                    model, sampler=sampler, debug=debug, algo=algo, use_new_engine=False
                )

            else:
                raise ValueError(f"Unsupported watermarking algorithm: {algo}")
