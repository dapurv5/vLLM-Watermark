KGW (Maryland) Watermarking
===========================

The KGW watermark by Kirchenbauer et al. (2023) partitions the vocabulary into "green" and "red" tokens using a pseudorandom function that maps the previous tokens to determine token partitioning.

.. raw:: html

   <div style="margin: 20px 0; line-height: 1.8;">
       <p style="margin: 5px 0; font-size: 14px;">
           <strong>Example code:</strong>
           <a href="https://github.com/dapurv5/vLLM-Watermark/blob/main/examples/example_maryland.py" target="_blank" style="display: inline-block; background-color: #6c757d; color: white; padding: 5px 12px; border-radius: 3px; font-size: 12px; text-decoration: none; margin-left: 8px;">📓 View On GitHub</a>
       </p>
       <p style="margin: 5px 0; font-size: 14px;">
           <strong>Author:</strong> Kirchenbauer et al.
       </p>
   </div>

Theory
------

The KGW watermark partitions the vocabulary into "green" and "red" tokens using a pseudorandom function (PRF) that maps the previous :math:`h` tokens to a random seed value, determining the partitioning of tokens.

At each generation step :math:`t`, the logit scores for tokens in the green set :math:`G_t` are increased by a fixed bias :math:`\delta`, promoting their selection:

.. math::

   \text{logit}_{wm}(x_t) = \text{logit}(x_t) + \delta \cdot \mathbf{1}_{x_t \in G_t}

where :math:`\mathbf{1}` is the indicator function.

Detection is performed by re-computing the green token sets and counting how many generated tokens :math:`|s|` belong to these sets. Under the null hypothesis (unwatermarked text), :math:`|s|` approximately follows a binomial distribution.

The presence of a watermark is tested by computing the z-score:

.. math::

   z = \frac{|s| - \gamma T}{\sqrt{\gamma(1-\gamma)T}}

where :math:`T` is the total token count and :math:`\gamma` is the expected fraction of green tokens. A large z-score indicates the presence of the watermark.

.. note::
   The watermark can be detected without model access, making it practical for deployment scenarios.

Paper reference
---------------

Kirchenbauer, J., Geiping, J., Wen, Y., Katz, J., Miers, I., & Goldstein, T. (2023). A Watermark for Large Language Models. *arXiv preprint arXiv:2301.10226*. https://arxiv.org/pdf/2301.10226

Example code
------------

.. code-block:: python

   import os
   import sys

   # IMPORTANT: For watermarking to work, set these environment variables
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

   # Create a KGW/Maryland watermarked LLM
   wm_llm = WatermarkedLLMs.create(
       llm,
       algo=WatermarkingAlgorithm.MARYLAND,
       seed=42,
       ngram=2,
       gamma=0.5,  # Expected fraction of green tokens
       delta=2.0   # Logit bias for green tokens
   )

   # Create KGW detector with matching parameters
   detector = WatermarkDetectors.create(
       algo=DetectionAlgorithm.MARYLAND_Z,
       model=llm,
       ngram=2,
       seed=42,
       gamma=0.5,
       delta=2.0,
       threshold=0.05,
   )

   # Generate watermarked text
   prompts = ["Write a short story about a robot learning to paint"]
   sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=64)
   outputs = wm_llm.generate(prompts, sampling_params)

   # Detect watermark
   for output in outputs:
       generated_text = output.outputs[0].text
       detection_result = detector.detect(generated_text)

       print(f"Generated: {generated_text}")
       print(f"Watermarked: {detection_result['is_watermarked']}")
       print(f"P-value: {detection_result['pvalue']:.6f}")
       print(f"Z-score: {detection_result['score']:.4f}")

Notes
-----

- Uses pseudorandom function to partition vocabulary into green/red sets
- Parameter :math:`\delta` controls the strength of the watermark bias
- Parameter :math:`\gamma` sets the expected fraction of green tokens (typically 0.5)
- Detection is model-agnostic and based on statistical hypothesis testing
- Higher :math:`\delta` values create stronger watermarks but may affect text quality
