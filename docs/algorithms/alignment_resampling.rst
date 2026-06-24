Alignment Resampling
====================

Best-of-N resampling that recovers alignment quality degraded by watermarking. Wraps any watermarked LLM and uses a reward model to select the highest-quality candidate from :math:`N` watermarked outputs.

.. raw:: html

   <div style="margin: 20px 0; line-height: 1.8;">
       <p style="margin: 5px 0; font-size: 14px;">
           <strong>Example code:</strong>
           <a href="https://github.com/dapurv5/vLLM-Watermark/blob/main/examples/example_alignment_resampling.py" target="_blank" style="display: inline-block; background-color: #6c757d; color: white; padding: 5px 12px; border-radius: 3px; font-size: 12px; text-decoration: none; margin-left: 8px;">📓 View On GitHub</a>
       </p>
       <p style="margin: 5px 0; font-size: 14px;">
           <strong>Paper:</strong> Verma & Phan (2025) - <a href="https://arxiv.org/pdf/2506.04462" target="_blank">Watermarking Degrades Alignment in Language Models</a>
       </p>
   </div>

Overview
--------

Watermarking modifies the sampling distribution, which can degrade the alignment quality of instruction-tuned models. Alignment Resampling mitigates this by generating :math:`N` watermarked candidates per prompt and selecting the one with the highest reward model score.

Because all :math:`N` candidates carry the watermark, detection still works on the selected output. The reward model only selects for quality — it does not remove the watermark signal.

How It Works
------------

1. Generate :math:`N` watermarked candidates using any watermarking algorithm
2. Score each candidate with a reward model
3. Return the highest-scoring candidate

The alignment improvement scales as :math:`\sqrt{\log N}` — diminishing returns mean :math:`N=4` captures most of the benefit.

**Key property:** This is algorithm-agnostic. It works with any watermarking method (OPENAI, MARYLAND, DIP, SWEET, etc.) since it operates at the output level, not the logit level.

.. note::
   The default reward model is ``OpenAssistant/reward-model-deberta-v3-base`` (86M parameters). You can substitute any reward model or custom scoring function.

Paper reference
---------------

Verma, A., Phan, N., & Trivedi, S. (2025). Watermarking Degrades Alignment in Language Models: Analysis and Mitigation. *arXiv preprint arXiv:2506.04462*. https://arxiv.org/pdf/2506.04462

Example code
------------

.. code-block:: python

   import os
   os.environ["VLLM_USE_V1"] = "1"
   os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

   from vllm import LLM, SamplingParams
   from vllm_watermark.alignment_resampling import AlignmentResampledLLM, load_reward_scorer
   from vllm_watermark.core import (
       DetectionAlgorithm,
       WatermarkedLLMs,
       WatermarkingAlgorithm,
   )
   from vllm_watermark.watermark_detectors import WatermarkDetectors

   llm = LLM(model="meta-llama/Llama-3.2-1B", enforce_eager=True, max_model_len=1024)

   # Step 1: Create any watermarked LLM
   wm_llm = WatermarkedLLMs.create(
       model=llm,
       algo=WatermarkingAlgorithm.MARYLAND,
       seed=42, ngram=2, gamma=0.5, delta=1.0,
   )

   # Step 2: Load reward scorer and wrap with alignment resampling
   scorer = load_reward_scorer()
   aligned_llm = AlignmentResampledLLM(wm_llm, scorer, n_samples=4)

   # Step 3: Detect as usual (same params as the watermark)
   detector = WatermarkDetectors.create(
       algo=DetectionAlgorithm.MARYLAND_Z,
       model=llm,
       ngram=2, seed=42, gamma=0.5, delta=1.0, threshold=0.05,
   )

   prompts = ["Write about machine learning applications"]
   sampling_params = SamplingParams(temperature=1.0, top_p=0.95, max_tokens=128)
   outputs = aligned_llm.generate(prompts, sampling_params)

   for output in outputs:
       generated_text = output.outputs[0].text
       result = detector.detect(generated_text)
       print(f"Watermarked: {result['is_watermarked']}, P-value: {result['pvalue']:.6f}")
