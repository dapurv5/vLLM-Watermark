Permute-and-Flip (PF) Watermarking
====================================

The Permute-and-Flip (PF) watermarking scheme uses prefix-free coding principles to embed watermarks while maintaining text quality.

.. raw:: html

   <div style="margin: 20px 0; line-height: 1.8;">
       <p style="margin: 5px 0; font-size: 14px;">
           <strong>Example code:</strong>
           <a href="https://github.com/dapurv5/vLLM-Watermark/blob/main/examples/simple_watermark_example.py" target="_blank" style="display: inline-block; background-color: #6c757d; color: white; padding: 5px 12px; border-radius: 3px; font-size: 12px; text-decoration: none; margin-left: 8px;">📓 View On GitHub</a>
       </p>
       <p style="margin: 5px 0; font-size: 14px;">
           <strong>Author:</strong> Lean et al.
       </p>
   </div>

Theory
------

PF watermarking operates by creating a permutation of the vocabulary and selectively "flipping" tokens based on cryptographic keys. The approach leverages prefix-free coding to ensure that the watermark can be detected without knowledge of the exact generation process.

The algorithm maintains a mapping between original tokens and their permuted counterparts, with the permutation determined by a cryptographic hash of the context. During generation, tokens are selectively replaced according to this mapping with probability :math:`\alpha`.

The sampling distribution is modified as:

.. math::

   P_{wm}(t) = (1-\alpha) \cdot P_{orig}(t) + \alpha \cdot P_{perm}(t)

where :math:`P_{orig}(t)` is the original token probability, :math:`P_{perm}(t)` is the probability under the permutation, and :math:`\alpha \in [0,1]` controls the watermarking strength.

Detection analyzes the frequency of tokens that appear in positions consistent with the permutation pattern, using statistical tests to determine if the observed pattern deviates significantly from random.

.. note::
   PF watermarking is designed to be robust against various text transformations while maintaining high text quality.

Paper reference
---------------

Lean, M., et al. (2024). Permute-and-Flip: An optimally robust and watermarkable decoder for LLMs. *arXiv preprint arXiv:2402.05864*. https://arxiv.org/abs/2402.05864

Example code
------------

.. code-block:: python

   import os
   import sys

   os.environ["VLLM_USE_V1"] = "1"
   os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

   from vllm import LLM, SamplingParams
   from vllm_watermark.core import (
       DetectionAlgorithm,
       WatermarkedLLMs,
       WatermarkingAlgorithm,
   )
   from vllm_watermark.watermark_detectors import WatermarkDetectors

   # Load the vLLM model
   llm = LLM(model="meta-llama/Llama-3.2-1B")

   # Create a PF watermarked LLM
   wm_llm = WatermarkedLLMs.create(
       llm,
       algo=WatermarkingAlgorithm.PF,
       seed=42,
       ngram=2,
       alpha=0.1,  # Permutation probability
   )

   # Create PF detector with matching parameters
   detector = WatermarkDetectors.create(
       algo=DetectionAlgorithm.PF,
       model=llm,
       ngram=2,
       seed=42,
       alpha=0.1,
       threshold=0.05,
   )

   # Generate watermarked text
   prompts = ["Explain quantum computing in simple terms"]
   sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=64)
   outputs = wm_llm.generate(prompts, sampling_params)

   # Detect watermark
   for output in outputs:
       generated_text = output.outputs[0].text
       detection_result = detector.detect(generated_text)

       print(f"Generated: {generated_text}")
       print(f"Watermarked: {detection_result['is_watermarked']}")
       print(f"P-value: {detection_result['pvalue']:.6f}")

Notes
-----

- Uses prefix-free coding principles for robust watermarking
- Parameter :math:`\alpha` controls the permutation probability
- Designed to be robust against text transformations
- Maintains high text quality with carefully tuned permutation strategies
- Detection is based on statistical analysis of token permutation patterns
