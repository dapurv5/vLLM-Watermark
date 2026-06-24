Unigram Watermarking
====================

Context-independent green/red list watermarking where the green list is fixed at initialization and does not depend on preceding tokens.

.. raw:: html

   <div style="margin: 20px 0; line-height: 1.8;">
       <p style="margin: 5px 0; font-size: 14px;">
           <strong>Example code:</strong>
           <a href="https://github.com/dapurv5/vLLM-Watermark/blob/main/examples/example_unigram.py" target="_blank" style="display: inline-block; background-color: #6c757d; color: white; padding: 5px 12px; border-radius: 3px; font-size: 12px; text-decoration: none; margin-left: 8px;">📓 View On GitHub</a>
       </p>
       <p style="margin: 5px 0; font-size: 14px;">
           <strong>Author:</strong> Zhao et al.
       </p>
   </div>

Theory
------

Unlike KGW/Maryland, which recomputes a new green/red partition at every token position based on the preceding :math:`h` tokens, Unigram uses a **single fixed partition** determined at initialization by hashing a secret key.

The vocabulary :math:`V` is split into a green set :math:`G` (fraction :math:`\gamma`) and red set :math:`R` (fraction :math:`1-\gamma`) using a SHA-256 hash of the secret ``hash_key``. During generation, green tokens receive a logit bias :math:`\delta`:

.. math::

   \text{logit}_{wm}(v) = \text{logit}(v) + \delta \cdot \mathbf{1}_{v \in G}

Detection counts how many generated tokens fall in :math:`G` and computes a z-score:

.. math::

   z = \frac{|s|_G - \gamma T}{\sqrt{\gamma(1-\gamma)T}}

where :math:`|s|_G` is the count of green tokens and :math:`T` is the total token count.

.. note::
   Because the green list is context-independent, Unigram watermarks are more robust to text edits (insertions, deletions, reordering) than context-dependent methods like KGW. The trade-off is that a fixed partition can introduce more detectable artifacts in the token distribution.

Paper reference
---------------

Zhao, X., Ananth, P., Li, L., & Wang, Y.-X. (2024). Provable Robust Watermarking for AI-Generated Text. *ICLR 2024*. https://arxiv.org/abs/2306.17439

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
       algo=WatermarkingAlgorithm.UNIGRAM,
       seed=42,
       ngram=1,
       gamma=0.5,
       delta=2.0,
       hash_key=15485863,
   )

   detector = WatermarkDetectors.create(
       algo=DetectionAlgorithm.UNIGRAM_Z,
       model=llm,
       ngram=1,
       seed=42,
       gamma=0.5,
       delta=2.0,
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
