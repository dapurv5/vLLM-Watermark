import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from vllm import LLM, SamplingParams

from vllm_watermark.watermark_generators.gumbel_generator import WatermarkGenerator

# Load the vLLM model
llm = LLM(model="meta-llama/Llama-3.2-1B")

# Create a Gumbel watermark generator (this wraps and patches the LLM)
wm_llm = WatermarkGenerator.create(llm, algo="gumbel", seed=42)

# Example prompt
prompts = ["Write a short poem about Microsoft"]

# Sampling parameters
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=32)

# Generate outputs using the watermarked LLM
outputs = wm_llm.generate(prompts, sampling_params)

# Print the outputs
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
