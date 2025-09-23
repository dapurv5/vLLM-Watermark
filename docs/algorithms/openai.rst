OpenAI/Gumbel Watermarking
===========================

The Gumbel watermarking scheme by Aaronson (2023) leverages the Gumbel-Max trick for deterministic token selection while preserving the original sampling distribution.

.. raw:: html

   <div style="margin: 20px 0; line-height: 1.8;">
       <p style="margin: 5px 0; font-size: 14px;">
           <strong>Example code:</strong>
           <a href="https://github.com/dapurv5/vLLM-Watermark/blob/main/examples/example_openai.py" target="_blank" style="display: inline-block; background-color: #6c757d; color: white; padding: 5px 12px; border-radius: 3px; font-size: 12px; text-decoration: none; margin-left: 8px;">📓 View On GitHub</a>
       </p>
       <p style="margin: 5px 0; font-size: 14px;">
           <strong>Author:</strong> Scott Aaronson
       </p>
   </div>

Theory
------

The Gumbel watermark uses the Gumbel-Max trick for deterministic token selection by hashing the preceding :math:`h` tokens with a key :math:`k` to generate scores :math:`r_t` for each token in the vocabulary at timestep :math:`t`.

Token selection uses:

.. math::

   \text{arg max}_{x_t \in V} \left[ \log P(x_t|x_{<t}) - \log(-\log(r_{x_t})) \right]

The detection score is computed as:

.. math::

   \text{Score}(x) = \sum_{t=1}^{n} \log\left(\frac{1}{1 - r_{x_t}}\right)

which follows a gamma distribution :math:`\Gamma(n, 1)` under the watermark hypothesis.

For random :math:`r \sim \text{Uniform}([0, 1])^{|V|}`, the transformation :math:`-\log(-\log(r))` follows a Gumbel(0,1) distribution. This enables distortion-free sampling when :math:`h` is large, since adding Gumbel noise to log probabilities and taking argmax corresponds exactly to sampling from the original softmax distribution.

.. note::
   This preserves the distribution while ensuring consistent output for fixed seeds, although at the cost of response diversity.

Paper reference
---------------

Aaronson, S. (2023). Gumbel Watermarking. *Scott Aaronson's Blog*. https://scottaaronson.blog/?p=6823

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
   llm = LLM(
       model="meta-llama/Llama-3.2-1B",
       enforce_eager=True,
       max_model_len=1024,
   )

   # Create a Gumbel watermarked LLM
   wm_llm = WatermarkedLLMs.create(
       model=llm,
       algo=WatermarkingAlgorithm.OPENAI,  # Uses Gumbel watermarking
       seed=42,
       ngram=2,
   )

   # Create Gumbel detector with matching parameters
   detector = WatermarkDetectors.create(
       algo=DetectionAlgorithm.OPENAI_Z,
       model=llm,
       ngram=2,
       seed=42,
       payload=0,
       threshold=0.05,
   )

   # Generate watermarked text
   prompts = ["Write about machine learning applications"]
   sampling_params = SamplingParams(temperature=1.0, top_p=0.95, max_tokens=64)
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

- Uses deterministic token selection via Gumbel-Max trick
- Preserves original sampling distribution when hash length is large
- Detection based on gamma distribution of accumulated scores
- Trade-off between consistency and response diversity
