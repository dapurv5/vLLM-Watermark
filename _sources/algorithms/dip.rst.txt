DiPmark (DIP) Watermarking
===========================

Permutation-based probability redistribution that uses cumulative-probability quantile splitting instead of a flat logit bias.

.. raw:: html

   <div style="margin: 20px 0; line-height: 1.8;">
       <p style="margin: 5px 0; font-size: 14px;">
           <strong>Example code:</strong>
           <a href="https://github.com/dapurv5/vLLM-Watermark/blob/main/examples/example_dip.py" target="_blank" style="display: inline-block; background-color: #6c757d; color: white; padding: 5px 12px; border-radius: 3px; font-size: 12px; text-decoration: none; margin-left: 8px;">📓 View On GitHub</a>
       </p>
       <p style="margin: 5px 0; font-size: 14px;">
           <strong>Author:</strong> Wu et al.
       </p>
   </div>

Theory
------

Unlike green-list methods (KGW, Unigram) that add a flat :math:`\delta` to logits, DiPmark generates a random vocabulary permutation :math:`\pi` for each context and redistributes probability mass using **cumulative-probability quantile splitting**.

For a given context, the algorithm:

1. Generates a deterministic permutation :math:`\pi` of the vocabulary by hashing the context with a secret key
2. Sorts token probabilities according to :math:`\pi`
3. Computes cumulative probabilities in the permuted order
4. Splits the distribution at quantile boundaries :math:`\alpha` and :math:`1-\alpha`, boosting tokens in the upper portion

The parameter :math:`\alpha \in [0, 0.5]` controls the watermarking strength. Unlike KGW's flat delta, the probability boost adapts to the token's position in the cumulative distribution, making the watermark more stealthy.

**Detection** checks each token's quantile position in the context-seeded permutation. Tokens landing in the boosted portion (quantile :math:`\geq \gamma`) are counted as "green", and a z-score is computed against the binomial null.

.. note::
   DiPmark tracks context history to avoid over-watermarking repeated n-grams. Contexts seen before are skipped during both generation and detection.

Paper reference
---------------

Wu, Y., Hu, Z., Zhang, H., & Huang, H. (2023). DiPmark: A Stealthy, Efficient and Resilient Watermark for Large Language Models. *arXiv preprint arXiv:2310.07710*. https://arxiv.org/abs/2310.07710

Example code
------------

.. code-block:: python

   import os
   os.environ["VLLM_USE_V1"] = "1"
   os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

   from vllm import LLM, SamplingParams
   from vllm_watermark.core import (
       DetectionAlgorithm,
       WatermarkedLLMs,
       WatermarkingAlgorithm,
   )
   from vllm_watermark.watermark_detectors import WatermarkDetectors

   llm = LLM(model="meta-llama/Llama-3.2-1B", enforce_eager=True, max_model_len=1024)

   wm_llm = WatermarkedLLMs.create(
       model=llm,
       algo=WatermarkingAlgorithm.DIP,
       seed=42,
       ngram=2,
       alpha=0.45,
       gamma=0.5,
       hash_key=15485863,
   )

   detector = WatermarkDetectors.create(
       algo=DetectionAlgorithm.DIP,
       model=llm,
       ngram=2,
       seed=42,
       gamma=0.5,
       hash_key=15485863,
       threshold=0.05,
   )

   prompts = ["Write about machine learning applications"]
   sampling_params = SamplingParams(temperature=1.0, top_p=0.95, max_tokens=128)
   outputs = wm_llm.generate(prompts, sampling_params)

   for output in outputs:
       generated_text = output.outputs[0].text
       result = detector.detect(generated_text)
       print(f"Watermarked: {result['is_watermarked']}, P-value: {result['pvalue']:.6f}")
