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

The Classical Gumbel-Max Formulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In its classical form, the Gumbel-Max trick samples tokens according to:

.. math::

   x_t^* = \text{arg max}_{v \in V} \left[ \log p_v + G_v \right]

where :math:`p_v` is the probability of token :math:`v` from the language model, and :math:`G_v \sim \text{Gumbel}(0, 1)` are i.i.d. Gumbel-distributed random variables.

For watermarking, we replace the random Gumbel noise with **deterministic** pseudo-random values generated from a hash of the context. Using uniform random variables :math:`u_v \sim \text{Uniform}([0, 1])`, we can generate Gumbel noise via the transformation :math:`G_v = -\log(-\log(u_v))`. This gives us:

.. math::

   x_t^* = \text{arg max}_{v \in V} \left[ \log p_v - \log(-\log(u_v)) \right]

Implementation via an Equivalent Form
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The expression above can be simplified. Note that it is mathematically equivalent to:

.. math::

   x_t^* = \text{arg max}_{v \in V} \left[ u_v^{1/p_v} \right]

This is what the implementation uses. We prove the equivalence below.

Mathematical Proof of Equivalence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Starting with the Gumbel-Max formulation:

.. math::

   \text{arg max}_v \left[ \log(p_v) - \log(-\log(u_v)) \right]

We can manipulate this step by step:

.. math::

   &= \text{arg max}_v \left[ \log(p_v) - \log(-\log(u_v)) \right] \\
   &= \text{arg max}_v \left[ \log \left( \frac{p_v}{-\log(u_v)} \right) \right] \quad \text{(log subtraction)} \\
   &= \text{arg max}_v \left[ \exp \left( \log \left( \frac{p_v}{-\log(u_v)} \right) \right) \right] \quad \text{(exp preserves argmax)} \\
   &= \text{arg max}_v \left[ p_v \cdot \frac{1}{-\log(u_v)} \right] \\
   &= \text{arg max}_v \left[ p_v \cdot \exp(-\log(-\log(u_v))) \right] \\
   &= \text{arg min}_v \left[ \frac{-\log(u_v)}{p_v} \right] \quad \text{(minimizing denominator)} \\
   &= \text{arg max}_v \left[ \frac{\log(u_v)}{p_v} \right] \quad \text{(flip sign, flip min/max)} \\
   &= \text{arg max}_v \left[ \log(u_v^{1/p_v}) \right] \\
   &= \text{arg max}_v \left[ u_v^{1/p_v} \right] \quad \text{(log is monotonic)}

Therefore, computing :math:`\text{arg max}_v [u_v^{1/p_v}]` is equivalent to the Gumbel-Max trick. This simplification has several advantages:

1. **Computational Efficiency:** Avoids nested logarithms
2. **Numerical Stability:** The form :math:`u^{1/p}` is more stable than :math:`-\log(-\log(u))`
3. **Implementation Simplicity:** Fewer operations and clearer code

Implementation Details
~~~~~~~~~~~~~~~~~~~~~~

The algorithm proceeds as follows:

1. **Generate deterministic random values:** Hash the previous :math:`h` tokens to obtain a seed, then generate :math:`|V|` uniform random values :math:`\{u_v\}`

2. **Apply nucleus (top-p) sampling:** Filter to keep only the top-p probability mass

3. **Compute the transformed scores:** For each token :math:`v` in the filtered vocabulary, compute :math:`u_v^{1/p_v}`

4. **Token selection:** Choose :math:`\text{arg max}_v [u_v^{1/p_v}]`

This preserves the original sampling distribution while ensuring deterministic outputs.

Interpretation
~~~~~~~~~~~~~~

The transformation :math:`u_v^{1/p_v}` balances probability and randomness:

- For high-probability tokens (:math:`p_v` large), :math:`1/p_v` is small, so :math:`u_v^{1/p_v}` remains close to :math:`u_v`
- For low-probability tokens (:math:`p_v` small), :math:`1/p_v` is large, so :math:`u_v^{1/p_v}` can be significantly boosted if :math:`u_v` is large

High-probability tokens tend to win without needing large :math:`u_v` values, but occasionally a lower-probability token with a favorable :math:`u_v` can prevail. This reproduces sampling from the original distribution.

The Gumbel-Max trick adds controlled randomness that, when combined with the true probabilities, produces samples from the correct distribution. The implementation achieves this without explicitly computing Gumbel random variables.

Detection
~~~~~~~~~

The detection score is computed as:

.. math::

   \text{Score}(x) = \sum_{t=1}^{n} \log\left(\frac{1}{1 - u_{x_t}}\right)

which follows a gamma distribution :math:`\Gamma(n, 1)` under the watermark hypothesis, enabling statistical detection.

.. note::
   This approach preserves the distribution while ensuring consistent output for fixed seeds, though it does reduce response diversity across different generations with the same context.

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

See Also
--------

For applications requiring enhanced output diversity, see :doc:`openai_dr` (Randomized Gumbel), which introduces controlled randomness at the cost of slight distortion.
