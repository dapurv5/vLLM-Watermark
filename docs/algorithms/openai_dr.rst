Randomized Gumbel Watermarking
===============================

A variant of the Gumbel watermark that introduces controlled randomness to enhance output diversity while maintaining detectability.

.. raw:: html

   <div style="margin: 20px 0; line-height: 1.8;">
       <p style="margin: 5px 0; font-size: 14px;">
           <strong>Example code:</strong>
           <a href="https://github.com/dapurv5/vLLM-Watermark/blob/main/examples/example_openai_double_randomization.py" target="_blank" style="display: inline-block; background-color: #6c757d; color: white; padding: 5px 12px; border-radius: 3px; font-size: 12px; text-decoration: none; margin-left: 8px;">📓 View On GitHub</a>
       </p>
       <p style="margin: 5px 0; font-size: 14px;">
           <strong>Paper:</strong> Verma & Phan (2025) - <a href="https://arxiv.org/pdf/2506.04462" target="_blank">Watermarking Degrades Alignment in Language Models</a>
       </p>
   </div>

Overview
--------

Standard Gumbel watermarking uses deterministic token selection (argmax), which limits response diversity across multiple generations with the same context. This variant addresses that limitation by introducing a second stage of randomness.

Key Difference
--------------

The standard Gumbel method computes:

.. math::

   x_t^* = \text{arg max}_{v \in V} \left[ u_v^{1/p_v} \right]

The randomized variant instead uses:

.. math::

   x_t^* \sim \text{Multinomial}\left( u_v^{1/p_v} \right)

After computing the transformed scores :math:`u_v^{1/p_v}`, instead of selecting the maximum deterministically, this method samples from a multinomial distribution where the scores serve as unnormalized probabilities. This introduces controlled randomness while preserving the watermark signal.

Properties
----------

- **Distortion:** Introduces slight distortion (no longer strictly distortion-free)
- **Diversity:** Enhanced output diversity compared to standard Gumbel
- **Detectability:** Maintains robust watermark detection (`Verma & Phan, 2025 <https://arxiv.org/pdf/2506.04462>`_)
- **Use case:** Designed for applications requiring alignment recovery via best-of-N sampling

Paper Reference
---------------

Verma, A., Phan, N., & Trivedi, S. (2025). Watermarking Degrades Alignment in Language Models: Analysis and Mitigation. *arXiv preprint arXiv:2506.04462*. https://arxiv.org/pdf/2506.04462

Example Code
------------

.. code-block:: python

   import os
   os.environ["VLLM_USE_V1"] = "1"
   os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

   from vllm import LLM, SamplingParams
   from vllm_watermark.core import WatermarkedLLMs, WatermarkingAlgorithm

   # Load the vLLM model
   llm = LLM(
       model="meta-llama/Llama-3.2-1B",
       enforce_eager=True,
       max_model_len=1024,
   )

   # Create a Randomized Gumbel watermarked LLM
   wm_llm = WatermarkedLLMs.create(
       model=llm,
       algo=WatermarkingAlgorithm.OPENAI_DR,
       seed=42,
       ngram=2,
   )

   # Generate watermarked text with enhanced diversity
   prompts = ["Write about machine learning applications"]
   sampling_params = SamplingParams(temperature=1.0, top_p=0.95, max_tokens=64)
   outputs = wm_llm.generate(prompts, sampling_params)

   for output in outputs:
       print(output.outputs[0].text)
