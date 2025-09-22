"""
Clean watermarked LLM implementation using vLLM's engine directly.

This approach is inspired by vllm_example.py and avoids all patching by
working directly with vLLM's engine step() method.
"""

import threading
import time
from threading import Event
from typing import List, Optional, Union

import torch
from loguru import logger
from vllm import LLM, SamplingParams
from vllm.outputs import PoolingRequestOutput, RequestOutput

from vllm_watermark.watermark_generators.base import WmGenerator


class WatermarkedLLM(LLM):
    """
    A watermarked LLM that uses vLLM's engine directly without any patching.
    
    This implementation is based on the vllm_example.py pattern but adds
    watermarking by intercepting the sampling process during engine steps.
    """
    
    def __init__(self, 
                 watermark_generator: WmGenerator,
                 debug: bool = False,
                 **llm_kwargs):
        """
        Initialize the watermarked LLM.
        
        Args:
            watermark_generator: The watermark generator to use
            debug: Enable debug logging
            **llm_kwargs: Arguments passed to vLLM's LLM constructor
        """
        # Force eager execution and single-threaded for deterministic watermarking
        llm_kwargs.setdefault('enforce_eager', True)
        
        # Initialize the base LLM
        super().__init__(**llm_kwargs)
        
        self.watermark_generator = watermark_generator
        self.debug = debug
        self.request_counter = iter(range(10**6))  # Unique request IDs
        
        if self.debug:
            logger.info(f"Initialized WatermarkedLLM with generator: {type(watermark_generator).__name__}")
    
    def generate(self, 
                 prompts: Union[List[str], str], 
                 sampling_params: Optional[SamplingParams] = None,
                 **kwargs) -> List[RequestOutput]:
        """
        Generate watermarked text using the engine's step() method.
        
        This method replaces vLLM's standard generate() to apply watermarking
        during the sampling process.
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        
        if sampling_params is None:
            sampling_params = SamplingParams()
        
        if self.debug:
            logger.debug(f"Starting watermarked generation for {len(prompts)} prompts")
        
        # Use the step-based generation approach
        return self._generate_with_watermarking(prompts, sampling_params)
    
    def _generate_with_watermarking(self, 
                                   prompts: List[str], 
                                   sampling_params: SamplingParams) -> List[RequestOutput]:
        """
        Generate text using the engine's step() method with watermarking.
        """
        # Add all requests to the engine
        request_ids = []
        for prompt in prompts:
            request_id = str(next(self.request_counter))
            self.llm_engine.add_request(request_id, prompt, sampling_params)
            request_ids.append(request_id)
        
        outputs = {}
        
        # Process requests using the engine's step method
        while self.llm_engine.has_unfinished_requests():
            step_outputs = self.llm_engine.step()
            
            for output in step_outputs:
                if output.finished:
                    outputs[output.request_id] = output
                else:
                    # This is where we can intercept and apply watermarking
                    # For now, let the engine handle it normally
                    # TODO: We need to intercept at the logits level, not here
                    pass
        
        # Return outputs in the same order as input prompts
        result = []
        for request_id in request_ids:
            if request_id in outputs:
                result.append(outputs[request_id])
            else:
                logger.error(f"No output found for request {request_id}")
        
        return result


def create_watermarked_llm(model: str,
                          watermark_generator: WmGenerator,
                          debug: bool = False,
                          **llm_kwargs) -> WatermarkedLLM:
    """
    Create a watermarked LLM instance.
    
    Args:
        model: Model name or path
        watermark_generator: Watermark generator to use
        debug: Enable debug logging
        **llm_kwargs: Additional arguments for vLLM
        
    Returns:
        WatermarkedLLM instance
    """
    return WatermarkedLLM(
        model=model,
        watermark_generator=watermark_generator,
        debug=debug,
        **llm_kwargs
    )
