Black-Box Watermarking
======================

Distortion-free watermarking via best-of-:math:`m` rejection sampling. The model's sampling distribution is completely unmodified — the watermark is embedded by selecting the highest-scoring candidate from :math:`m` independent generations.

.. raw:: html

   <div style="margin: 20px 0; line-height: 1.8;">
       <p style="margin: 5px 0; font-size: 14px;">
           <strong>Example code:</strong>
           <a href="https://github.com/dapurv5/vLLM-Watermark/blob/main/examples/example_blackbox.py" target="_blank" style="display: inline-block; background-color: #6c757d; color: white; padding: 5px 12px; border-radius: 3px; font-size: 12px; text-decoration: none; margin-left: 8px;">📓 View On GitHub</a>
       </p>
       <p style="margin: 5px 0; font-size: 14px;">
           <strong>Author:</strong> Bahri &amp; Wieting
       </p>
   </div>

Theory
------

The black-box watermark generates :math:`m` candidate sequences from the unmodified model and selects the one with the highest score under a keyed pseudorandom function (PRF).

For each candidate sequence, the score is computed by:

1. Extracting all n-grams from the sequence
2. Hashing each n-gram with the secret key to produce a uniform PRF value :math:`u_i \in [0, 1]`
3. Computing the Irwin-Hall CDF of the sum :math:`\sum_i u_i`

The candidate with the highest score is returned. Under no watermark, scores are uniform on :math:`[0, 1]`. With :math:`m` candidates, the expected maximum score is :math:`\approx m/(m+1)`.

**Detection** recomputes the same PRF score on the text. The p-value is :math:`1 - \text{score}`.

The key property is **zero distortion**: since all :math:`m` candidates are drawn from the original model distribution, the selected output is also a valid sample from the original distribution. The trade-off is computational cost — generation is :math:`m` times more expensive.

.. note::
   Detection power depends on ``n_candidates``: :math:`m=16` gives weak detection (score :math:`\approx 0.94`), :math:`m=128` gives strong detection (score :math:`\approx 0.992`), :math:`m=256` gives very strong detection (score :math:`\approx 0.996`).

Paper reference
---------------

Bahri, D. & Wieting, J. (2026). A Watermark for Black-Box Language Models. *TMLR* (arXiv:2410.02099). https://arxiv.org/abs/2410.02099

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

   # n_candidates controls detection power vs. generation cost
   wm_llm = WatermarkedLLMs.create(
       model=llm,
       algo=WatermarkingAlgorithm.BLACKBOX,
       hash_key=15485863,
       ngram=4,
       n_candidates=128,
   )

   detector = WatermarkDetectors.create(
       algo=DetectionAlgorithm.BLACKBOX,
       model=llm,
       ngram=4,
       hash_key=15485863,
       threshold=0.02,
   )

   prompts = ["Write about machine learning applications"]
   sampling_params = SamplingParams(temperature=1.0, top_p=0.95, max_tokens=128)
   outputs = wm_llm.generate(prompts, sampling_params)

   for output in outputs:
       generated_text = output.outputs[0].text
       result = detector.detect(generated_text)
       print(f"Watermarked: {result['is_watermarked']}, P-value: {result['pvalue']:.6f}")
