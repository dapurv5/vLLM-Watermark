"""
Watermark-enabled LLM engine that cleanly integrates with vLLM.

This module provides a clean alternative to monkey patching by creating a custom
ModelRunner that uses our watermark sampler directly.
"""

import os
from functools import wraps
from typing import Any, Dict, Optional, Type, Union

from loguru import logger
from vllm import LLM
from vllm.entrypoints.llm import LLM as VllmLLM

from vllm_watermark.samplers.watermark_sampler import WatermarkSampler
from vllm_watermark.watermark_generators.base import WmGenerator


class WatermarkEngineConfig:
    """Configuration for watermark engine integration."""

    def __init__(
        self,
        watermark_generator: WmGenerator,
        debug: bool = False,
        force_eager: bool = True,
    ):
        self.watermark_generator = watermark_generator
        self.debug = debug
        self.force_eager = force_eager  # Helps with sampler injection


def _patch_model_runner_sampler(original_init):
    """
    Decorator to patch ModelRunner initialization to use our watermark sampler.

    This is a targeted patch that only affects the sampler creation, not the entire
    sampling infrastructure.
    """

    @wraps(original_init)
    def patched_init(self, *args, **kwargs):
        # Call the original init
        result = original_init(self, *args, **kwargs)

        # Replace the sampler if we have a watermark config
        if hasattr(self, "_watermark_config"):
            config = self._watermark_config
            if self.sampler is not None:
                old_sampler = self.sampler
                self.sampler = WatermarkSampler(
                    watermark_generator=config.watermark_generator,
                    model=getattr(self, "model", None),
                    debug=config.debug,
                )
                if config.debug:
                    logger.info(
                        f"Replaced {type(old_sampler).__name__} with WatermarkSampler"
                    )

        return result

    return patched_init


class WatermarkedLLM:
    """
    A watermark-enabled LLM wrapper that integrates cleanly with vLLM.

    This class provides the same interface as vLLM's LLM but with watermarking
    capabilities built-in.
    """

    def __init__(
        self,
        model: str,
        watermark_generator: WmGenerator,
        debug: bool = False,
        **vllm_kwargs,
    ):
        """
        Initialize the watermarked LLM.

        Args:
            model: Model name or path
            watermark_generator: Watermark generator to use
            debug: Enable debug logging
            **vllm_kwargs: Additional arguments passed to vLLM's LLM constructor
        """
        self.watermark_generator = watermark_generator
        self.debug = debug

        # Create watermark config
        self.watermark_config = WatermarkEngineConfig(
            watermark_generator=watermark_generator,
            debug=debug,
            force_eager=vllm_kwargs.get("enforce_eager", True),
        )

        # Force eager execution to ensure our sampler injection works
        if "enforce_eager" not in vllm_kwargs:
            vllm_kwargs["enforce_eager"] = True

        # Apply the sampler injection patch
        self._apply_sampler_patch()

        # Create the underlying vLLM instance
        try:
            self.llm = LLM(model=model, **vllm_kwargs)
            self._inject_watermark_config()

            if self.debug:
                logger.info(f"Successfully created WatermarkedLLM with model: {model}")

        except Exception as e:
            logger.error(f"Failed to create WatermarkedLLM: {e}")
            raise

    def _apply_sampler_patch(self):
        """Apply targeted patches to inject our watermark sampler."""
        # We need to patch the ModelRunner classes to use our sampler
        # This is much cleaner than the previous approach as we only patch sampler creation

        # Determine which ModelRunner classes to patch based on vLLM version
        env_use_v1 = os.environ.get("VLLM_USE_V1")

        try:
            if env_use_v1 is not None and env_use_v1.strip() == "0":
                # V0 engine
                from vllm.worker.model_runner import GPUModelRunnerBase

                model_runner_classes = [GPUModelRunnerBase]
                logger.info("Patching V0 ModelRunner for watermarking")
            else:
                # Try V1 first, fallback to V0
                try:
                    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

                    model_runner_classes = [GPUModelRunner]
                    logger.info("Patching V1 ModelRunner for watermarking")
                except ImportError:
                    from vllm.worker.model_runner import GPUModelRunnerBase

                    model_runner_classes = [GPUModelRunnerBase]
                    logger.info(
                        "Patching V0 ModelRunner for watermarking (V1 not available)"
                    )

            # Apply patches to the relevant ModelRunner classes
            for runner_class in model_runner_classes:
                if hasattr(runner_class, "__init__") and not hasattr(
                    runner_class.__init__, "_watermark_patched"
                ):
                    original_init = runner_class.__init__
                    runner_class.__init__ = _patch_model_runner_sampler(original_init)
                    runner_class.__init__._watermark_patched = True

                    if self.debug:
                        logger.debug(
                            f"Patched {runner_class.__name__}.__init__ for watermarking"
                        )

        except Exception as e:
            logger.error(f"Failed to apply sampler patch: {e}")
            raise RuntimeError(f"Could not patch ModelRunner for watermarking: {e}")

    def _inject_watermark_config(self):
        """Inject watermark configuration into the model runners."""
        try:
            # Navigate to model runners and inject our config
            if hasattr(self.llm, "llm_engine"):
                engine = self.llm.llm_engine

                # Handle different engine structures (V0 vs V1)
                model_runners = []

                if hasattr(engine, "model_executor"):
                    # V0 structure
                    model_executor = engine.model_executor

                    if hasattr(model_executor, "driver_worker") and hasattr(
                        model_executor.driver_worker, "model_runner"
                    ):
                        model_runners.append(model_executor.driver_worker.model_runner)

                    if hasattr(model_executor, "workers"):
                        for worker in model_executor.workers:
                            if hasattr(worker, "model_runner"):
                                model_runners.append(worker.model_runner)

                elif hasattr(engine, "engine_core"):
                    # V1 structure
                    engine_core = engine.engine_core
                    if hasattr(engine_core, "model_executor"):
                        model_executor = engine_core.model_executor

                        if hasattr(model_executor, "driver_worker"):
                            driver_worker = model_executor.driver_worker
                            if hasattr(driver_worker, "worker") and hasattr(
                                driver_worker.worker, "model_runner"
                            ):
                                model_runners.append(driver_worker.worker.model_runner)
                            elif hasattr(driver_worker, "model_runner"):
                                model_runners.append(driver_worker.model_runner)

                # Inject config into all found model runners
                for runner in model_runners:
                    runner._watermark_config = self.watermark_config
                    if self.debug:
                        logger.debug(
                            f"Injected watermark config into {type(runner).__name__}"
                        )

                if not model_runners:
                    logger.warning(
                        "No model runners found for watermark config injection"
                    )

        except Exception as e:
            logger.error(f"Failed to inject watermark config: {e}")
            raise

    def generate(self, prompts, sampling_params=None, **kwargs):
        """
        Generate text with watermarking applied.

        This method has the same interface as vLLM's LLM.generate() but applies
        watermarking during the generation process.
        """
        if self.debug:
            logger.debug(
                f"WatermarkedLLM.generate called with {len(prompts) if isinstance(prompts, list) else 1} prompt(s)"
            )

        # Use the underlying vLLM instance for generation
        # Our watermark sampler will automatically be applied during sampling
        return self.llm.generate(prompts, sampling_params=sampling_params, **kwargs)

    def get_tokenizer(self):
        """Get the tokenizer from the underlying LLM."""
        return self.llm.get_tokenizer()

    def __getattr__(self, name):
        """Delegate unknown attributes to the underlying LLM."""
        return getattr(self.llm, name)


def create_watermarked_llm(
    model: str, watermark_generator: WmGenerator, debug: bool = False, **vllm_kwargs
) -> WatermarkedLLM:
    """
    Factory function to create a watermark-enabled LLM.

    Args:
        model: Model name or path
        watermark_generator: Watermark generator to use
        debug: Enable debug logging
        **vllm_kwargs: Additional arguments passed to vLLM

    Returns:
        WatermarkedLLM instance ready for generation
    """
    return WatermarkedLLM(
        model=model, watermark_generator=watermark_generator, debug=debug, **vllm_kwargs
    )
