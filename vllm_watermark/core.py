from enum import Enum
from typing import Union

from loguru import logger
from vllm.entrypoints.llm import LLM
from vllm.model_executor.layers.sampler import Sampler
from vllm.v1.sample.sampler import Sampler as V1Sampler


class WatermarkingAlgorithm(Enum):
    OPENAI = "openai"
    PF = "pf"
    MARYLAND = "maryland"


class WatermarkedLLM:
    def __init__(
        self,
        llm: LLM,
        sampler: Union[Sampler, V1Sampler] = None,
        debug: bool = False,
    ):
        self.llm = llm
        self.sampler = sampler
        self.debug = debug
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

    def generate(self, *args, **kwargs):
        if self.debug:
            logger.debug("WatermarkedLLM.generate called")
        return self.llm.generate(*args, **kwargs)


# Factory for creating watermarked LLMs
class WatermarkedLLMs:
    @staticmethod
    def create(
        model,
        algo: WatermarkingAlgorithm = WatermarkingAlgorithm.OPENAI,
        debug: bool = False,
        **kwargs,
    ) -> WatermarkedLLM:
        if algo == WatermarkingAlgorithm.OPENAI:
            from vllm_watermark.samplers.custom_sampler import CustomSampler
            from vllm_watermark.watermark_generators.openai_generator import (
                OpenaiGenerator,
            )

            generator = OpenaiGenerator(model, model.get_tokenizer(), **kwargs)
            sampler = CustomSampler(model, generator, debug=debug)
            return WatermarkedLLM(model, sampler, debug=debug)
        elif algo == WatermarkingAlgorithm.MARYLAND:
            from vllm_watermark.samplers.custom_sampler import CustomSampler
            from vllm_watermark.watermark_generators.maryland_generator import (
                MarylandGenerator,
            )

            generator = MarylandGenerator(model, model.get_tokenizer(), **kwargs)
            sampler = CustomSampler(model, generator, debug=debug)
            return WatermarkedLLM(model, sampler, debug=debug)
        raise ValueError(f"Unknown watermarking algorithm: {algo}")
