SWEET Watermarking
==================

Entropy-selective watermarking that only applies green-list bias at high-entropy positions, preserving text quality where the model is confident.

.. raw:: html

   <div style="margin: 20px 0; line-height: 1.8;">
       <p style="margin: 5px 0; font-size: 14px;">
           <strong>Example code:</strong>
           <a href="https://github.com/dapurv5/vLLM-Watermark/blob/main/examples/example_sweet.py" target="_blank" style="display: inline-block; background-color: #6c757d; color: white; padding: 5px 12px; border-radius: 3px; font-size: 12px; text-decoration: none; margin-left: 8px;">📓 View On GitHub</a>
       </p>
       <p style="margin: 5px 0; font-size: 14px;">
           <strong>Author:</strong> Lee et al.
       </p>
   </div>

Theory
------

SWEET extends the KGW green-list approach with an **entropy gate**: the logit bias is only applied at positions where the model's output entropy exceeds a threshold :math:`\tau`.

At each generation step :math:`t`, the entropy of the output distribution is computed:

.. math::

   H_t = -\sum_{v \in V} p_v \log p_v

If :math:`H_t > \tau`, the standard green-list bias is applied:

.. math::

   \text{logit}_{wm}(v) = \text{logit}(v) + \delta \cdot \mathbf{1}_{v \in G_t}

If :math:`H_t \leq \tau`, the logits are left unchanged.

The intuition is that at low-entropy positions (where the model is confident about the next token), applying a bias is both unnecessary (the "correct" token will be chosen anyway) and harmful (it could push the model away from the obvious choice). At high-entropy positions, many tokens are plausible, so biasing toward green tokens has minimal impact on quality.

**Detection** counts green tokens across all positions using the standard z-score test. The entropy threshold is not needed for detection — the generator's selective biasing still produces a detectable excess of green tokens overall.

.. note::
   SWEET trades slightly lower detection power for better text quality. The entropy threshold :math:`\tau` controls this trade-off: higher thresholds apply the watermark at fewer positions.

Paper reference
---------------

Lee, T., et al. (2023). Who Wrote this Code? Watermarking for Code Generation. *arXiv preprint arXiv:2305.15060*. https://arxiv.org/abs/2305.15060

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
       algo=WatermarkingAlgorithm.SWEET,
       seed=42,
       ngram=2,
       gamma=0.5,
       delta=2.0,
       hash_key=15485863,
       entropy_threshold=3.0,
   )

   detector = WatermarkDetectors.create(
       algo=DetectionAlgorithm.SWEET,
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
