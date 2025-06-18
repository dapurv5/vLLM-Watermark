import os
import sys

# export VLLM_ENABLE_V1_MULTIPROCESSING=0
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vllm import LLM, SamplingParams

from vllm_watermark.core import WatermarkedLLMs

# Load the vLLM model
llm = LLM(model="meta-llama/Llama-3.2-1B")

# Create a Gumbel watermarked LLM (this wraps and patches the LLM)
wm_llm = WatermarkedLLMs.create(llm, algo="gumbel", seed=42, ngram=2)

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
