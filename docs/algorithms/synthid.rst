SynthID Watermarking
====================

Google DeepMind's multi-layer tournament watermarking that uses non-distortionary probability updates to embed a watermark signal across multiple depth layers.

.. raw:: html

   <div style="margin: 20px 0; line-height: 1.8;">
       <p style="margin: 5px 0; font-size: 14px;">
           <strong>Example code:</strong>
           <a href="https://github.com/dapurv5/vLLM-Watermark/blob/main/examples/example_synthid.py" target="_blank" style="display: inline-block; background-color: #6c757d; color: white; padding: 5px 12px; border-radius: 3px; font-size: 12px; text-decoration: none; margin-left: 8px;">📓 View On GitHub</a>
       </p>
       <p style="margin: 5px 0; font-size: 14px;">
           <strong>Author:</strong> Dathathri et al. (Google DeepMind)
       </p>
   </div>

Theory
------

SynthID assigns binary g-values :math:`g_{v,d} \in \{0, 1\}` to each vocabulary token :math:`v` at each of :math:`D` depth layers, using a Linear Congruential Generator (LCG) hash of the context tokens and a pre-computed binary sampling table.

At each depth layer :math:`d`, the probabilities are updated via a **non-distortionary tournament**:

.. math::

   p_v' = p_v \cdot (1 + g_{v,d} - m_d)

where :math:`m_d = \sum_v p_v \cdot g_{v,d}` is the total probability mass on :math:`g=1` tokens. This preserves the total probability mass (:math:`\sum_v p_v' = 1`) while biasing sampling toward :math:`g=1` tokens.

After :math:`D` layers of tournament updates, tokens with :math:`g=1` across more layers have amplified probability, creating a detectable signal.

**Detection** recomputes the g-values for each token in the text and computes the mean g-value across all positions and layers. Under no watermark, the expected mean is 0.5. A z-test determines statistical significance.

.. note::
   SynthID is designed to be **non-distortionary**: the tournament preserves probability mass at each step, meaning the expected output distribution is closer to the original model than additive-bias methods like KGW.

Paper reference
---------------

Dathathri, S., et al. (2024). Scalable watermarking for identifying large language model outputs. *Nature*. https://www.nature.com/articles/s41586-024-08025-4

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
       algo=WatermarkingAlgorithm.SYNTHID,
       ngram=4,
       seed=42,
   )

   detector = WatermarkDetectors.create(
       algo=DetectionAlgorithm.SYNTHID,
       model=llm,
       ngram=4,
       threshold=0.05,
   )

   prompts = ["Write about machine learning applications"]
   sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=128)
   outputs = wm_llm.generate(prompts, sampling_params)

   for output in outputs:
       generated_text = output.outputs[0].text
       result = detector.detect(generated_text)
       print(f"Watermarked: {result['is_watermarked']}, P-value: {result['pvalue']:.6f}")
