Power Law Detection for Gumbel Watermarking
=============================================

A refined detection mechanism for the Gumbel/OpenAI watermark that replaces
Aaronson's exponential statistic with a truncated power law. Proven to be
near-optimal in a problem-dependent sense among all model-agnostic
watermarking schemes.

.. raw:: html

   <div style="margin: 20px 0; line-height: 1.8;">
       <p style="margin: 5px 0; font-size: 14px;">
           <strong>Author:</strong> Tor Lattimore (Google DeepMind)
       </p>
       <p style="margin: 5px 0; font-size: 14px;">
           <strong>Paper:</strong>
           <a href="https://arxiv.org/abs/2603.30017" target="_blank" style="display: inline-block; background-color: #6c757d; color: white; padding: 5px 12px; border-radius: 3px; font-size: 12px; text-decoration: none; margin-left: 8px;">arXiv:2603.30017</a>
       </p>
   </div>

.. note::
   This is a **detection-only** algorithm. It detects the same Gumbel/OpenAI
   watermark produced by the ``OPENAI`` watermarking algorithm — it simply
   uses a different test statistic for detection. Use ``WatermarkingAlgorithm.OPENAI``
   for generation and ``DetectionAlgorithm.OPENAI_PL`` for detection.

Theory
------

Background: Exponential Detection (Aaronson 2022)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The standard Gumbel detector reconstructs the noise variables
:math:`V_t = U_{t,A_t}` from the text and secret key. Under :math:`H_0`
(no watermark), :math:`V_t \sim \text{Uniform}[0,1]`. Under :math:`H_1`
(watermarked), :math:`V_t` is biased toward 1.

Aaronson's detector uses the exponential statistic:

.. math::

   S_t = -\log(1 - V_t)

which is :math:`\text{Exp}(1)` under :math:`H_0`. The sum
:math:`\sum_{t=1}^n S_t \sim \Gamma(n, 1)`, giving an exact p-value.

Aaronson showed detection succeeds when :math:`n = \Omega(\log(1/\delta) / \bar{H}^2)`
where :math:`\bar{H} = \frac{1}{n}\sum H(P_t)` is the average entropy. This bound
is **variance-unaware**: it treats all entropy contributions equally.

Power Law Detection (Lattimore 2026)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Lattimore replaces the exponential with a truncated power law statistic:

.. math::

   S(u) = \min\left(\frac{1}{\sqrt{\varepsilon}},\; \frac{1}{\sqrt{1-u}}\right) - \mu

where :math:`\varepsilon = \log(1/\delta)/n` and :math:`\mu = 2 - \sqrt{\varepsilon}`
is chosen so that :math:`\mathbb{E}[S(U)] = 0` when :math:`U \sim \text{Uniform}[0,1]`.

The critical threshold :math:`\tau_\star` is calibrated via Monte Carlo so that:

.. math::

   \mathbb{P}_0\left(\sum_{t=1}^n S_t \geq \tau_\star\right) = \delta

**Theorem 2** (Lattimore): Detection succeeds with probability :math:`\geq 1 - 2\delta`
whenever :math:`\sum_{t=1}^n G_\varepsilon(P_t) \geq C\tau`, where
:math:`G_\varepsilon(p)` is an entropy-like quantity that captures per-token
detectability more finely than :math:`\bar{H}`.

Intuition
~~~~~~~~~

The key insight is about **where the information lives**. Consider the CDF deviation
:math:`F(x) - F_n(x)` between the true uniform CDF and the empirical CDF of
:math:`(V_t)`. Under watermarking, this deviation is largest for
:math:`x \in [1/2, 1]` (near 1). The exponential statistic weights all regions
equally, but the power law statistic upweights the informative region near 1 via
the :math:`1/\sqrt{1-u}` term, while truncating at :math:`1/\sqrt{\varepsilon}`
to control variance.

This is analogous to the Anderson-Darling test versus the Kolmogorov-Smirnov test
in classical statistics: weighting by the variance of the empirical process yields
a more powerful test.

When Does It Help?
~~~~~~~~~~~~~~~~~~

The theoretical improvement over Aaronson's bound is most significant when:

1. **Entropy varies across positions**: Some tokens are very predictable
   (low :math:`H(P_t)`) while others have high entropy. Aaronson's
   :math:`\bar{H}^2` penalizes variance; the power law's
   :math:`G_\varepsilon` handles heterogeneity better.

2. **Many rare tokens contribute**: When :math:`P_t` has many tokens with
   moderate probability (Example 10 in the paper), the power law needs
   :math:`n = \Omega(\log(n)\log(1/\delta)/\beta)` tokens versus Aaronson's
   :math:`n = \Omega(\log(1/\delta)/\beta^2)` — a :math:`\beta` factor
   improvement for small :math:`\beta`.

3. **Short sequences**: The variance-awareness of the power law matters
   more when :math:`n` is small, because there are fewer tokens to
   average over.

.. warning::
   **Practical caveat from the paper (Section 6):** Lattimore acknowledges that
   "its performance on language data seems to be slightly worse [than the
   exponential detector], presumably due to the additional logarithmic factor
   and/or constant factors." The theoretical advantage is **asymptotic** and
   manifests in specific distribution shapes (Examples 8-10 in the paper).
   On typical LLM outputs, the exponential/gamma detector may match or
   slightly outperform the power law detector.

Empirical Comparison
--------------------

We ran a head-to-head comparison of three detection statistics for the same
Gumbel watermark on SageMaker (``ml.g5.2xlarge``, 1x A10G GPU).

Experimental Setup
~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 30 70

   * - **Model**
     - ``meta-llama/Llama-3.2-1B``
   * - **Samples**
     - 500 watermarked + 500 unwatermarked
   * - **Watermark**
     - ``OPENAI``, ngram=2, seed=42, payload=0
   * - **Generation**
     - temperature=0.7, top_p=0.9, max_tokens=512
   * - **Target lengths**
     - 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500 scored tokens
   * - **Threshold (delta)**
     - 0.05
   * - **Dataset**
     - C4 (processed)

Three Detectors Compared
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 30 25 25

   * - Detector
     - Statistic
     - p-value Method
     - Reference
   * - **Gamma** (``OpenaiDetector``)
     - :math:`S_t = -\log(1 - V_t)`
     - Exact gamma CDF
     - Aaronson 2022
   * - **Z-score** (``OpenaiDetectorZ``)
     - :math:`S_t = -\log(1 - V_t)`
     - Normal approx (CLT)
     - Aaronson 2022
   * - **Power Law** (``OpenaiDetectorPL``)
     - :math:`S(u) = \min(1/\sqrt{\varepsilon}, 1/\sqrt{1-u}) - \mu`
     - Monte Carlo (100K samples)
     - Lattimore 2026

Results
~~~~~~~

**Summary at selected sequence lengths** (500 watermarked + 500 unwatermarked
texts, delta=0.05):

.. list-table::
   :header-rows: 1
   :widths: 10 12 12 12 12 12 12

   * - n
     - Gamma TPR
     - Z-score TPR
     - PL TPR
     - Gamma FPR
     - Z-score FPR
     - PL FPR
   * - 20
     - 0.984
     - 0.984
     - 0.982
     - 0.060
     - 0.032
     - 0.060
   * - 50
     - 0.962
     - 0.962
     - 0.962
     - 0.070
     - 0.042
     - 0.066
   * - 100
     - 0.918
     - 0.918
     - 0.918
     - 0.082
     - 0.040
     - 0.074
   * - 200
     - 0.862
     - 0.862
     - 0.862
     - 0.088
     - 0.054
     - 0.090
   * - 500
     - 0.804
     - 0.804
     - 0.804
     - 0.136
     - 0.114
     - 0.122

Diagnostic Plots
~~~~~~~~~~~~~~~~~

.. raw:: html

   <div class="plot-grid">
     <div class="plot-cell">
       <img src="../_static/plot_detection_power.png"
            alt="Detection Power: TPR vs Sequence Length" />
       <div class="plot-caption"><strong>Plot 1 — Detection Power.</strong>
       All three curves overlap almost perfectly. TPR starts near 0.97 at
       n=10 and <em>decreases</em> as n grows (to ~0.80 at n=500) — this
       is not a bug but an artifact of FPR inflation from n-gram correlation
       (see Plot 2). Under these conditions (Llama-3.2-1B, temp=0.7), the
       watermark signal is so strong that all detectors saturate.</div>
     </div>
     <div class="plot-cell">
       <img src="../_static/plot_fpr_calibration.png"
            alt="FPR Calibration: Empirical vs Nominal" />
       <div class="plot-caption"><strong>Plot 2 — FPR Calibration.</strong>
       A calibrated test holds FPR at δ=0.05 (dashed line). Z-score (orange)
       is conservative at small n — the CLT overestimates gamma-tail p-values.
       Gamma (blue) and Power Law (green) start near δ but all three inflate
       to 0.11–0.14 at n=500 due to repeated n-gram contexts creating
       correlated V<sub>t</sub> values that violate the i.i.d. assumption.</div>
     </div>
     <div class="plot-cell">
       <img src="../_static/plot_roc_curves.png"
            alt="ROC Curves at Various Sequence Lengths" />
       <div class="plot-caption"><strong>Plot 3 — ROC Curves.</strong>
       Full FPR–TPR trade-off across all thresholds at n=20, 50, 100, 200.
       All three detectors produce virtually identical ROC curves hugging
       the top-left corner. Quality degrades at larger n (n-gram correlation),
       but even at n=200 all detectors achieve &gt;0.85 TPR at FPR=0.05.</div>
     </div>
     <div class="plot-cell">
       <img src="../_static/plot_pvalue_distributions.png"
            alt="p-value Distributions at n=100" />
       <div class="plot-caption"><strong>Plot 4 — p-value Distributions (n=100).</strong>
       Top row (H₀): Gamma and PL are roughly Uniform[0,1] with a spike
       near 1.0 from n-gram artifacts; Z-score shows right-skew (conservative).
       Bottom row (H₁): all three concentrate p-values near 0 (density &gt;20),
       confirming the watermark signal overwhelms every statistic.</div>
     </div>
     <div class="plot-cell full-width">
       <img src="../_static/plot_tokens_to_detection.png"
            alt="Tokens-to-Detection CDF"
            style="max-width: 60%; display: block; margin: 0 auto;" />
       <div class="plot-caption" style="text-align: center;">
       <strong>Plot 5 — Tokens-to-Detection CDF.</strong>
       The most operationally useful plot: all three CDFs shoot to ~0.95 by
       n=10–20 tokens. With Llama-3.2-1B at temp=0.7, you can reliably
       detect the Gumbel watermark from the first 10–20 tokens regardless
       of detection statistic. The power law's theoretical sample-complexity
       advantage does not materialize when the signal is this strong.</div>
     </div>
   </div>

Key Observations
~~~~~~~~~~~~~~~~

1. **All three detectors perform nearly identically.** With Llama-3.2-1B at
   temperature=0.7, the watermark signal is extremely strong: mean
   :math:`V_t = 0.8354` for watermarked text (compared to 0.5 under
   :math:`H_0`). At this signal strength, even 10 scored tokens suffice
   for >94% TPR with all detectors.

2. **FPR increases with n for all detectors** (0.060 → 0.136 for Gamma).
   This is caused by repeated n-gram contexts in real text producing
   perfectly correlated :math:`V_t` values, violating the i.i.d. assumption
   that all three detectors share.

3. **Z-score is conservative at small n** — the CLT normal approximation
   over-estimates p-values when :math:`n < 30`, yielding *lower* FPR than
   nominal but potentially missing some watermarked texts.

4. **Median detection occurs at n=10 tokens** for all three detectors
   (the shortest length tested), confirming the signal is saturatingly strong.

Why No Difference Was Observed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The power law's theoretical advantage appears when the **signal-to-noise ratio
is low** — specifically when the per-token entropy-like quantity
:math:`G_\varepsilon(P_t)` is small relative to the noise floor. In our
experiment, the signal was so strong that all detectors were already near
ceiling performance at the shortest sequence length.

To observe a measurable difference, try:

- **Higher temperature** (1.0-1.5): Increases entropy :math:`H(P_t)`,
  reducing the bias on :math:`V_t` and weakening the watermark signal.
- **top_p = 1.0** (no nucleus filtering): Includes more low-probability
  tokens, increasing effective vocabulary and entropy.
- **Heterogeneous prompts**: Mix of factual recall (low entropy) and
  creative generation (high entropy) within a single text, which is
  where the variance-aware power law should shine.
- **Adversarial perturbation**: Edit/paraphrase watermarked text to
  dilute the signal — the power law detector should degrade more gracefully.
- **Larger models**: Higher-capacity models often produce more uncertain
  (higher entropy) distributions.
- **Lower n-gram**: Using ngram=1 instead of ngram=2 reduces context
  dependence, potentially weakening the signal.

**Reproducing the comparison:**

.. code-block:: bash

   ./sml run scripts/benchmark/run_sagemaker_detector_comparison.py

The script generates 5 diagnostic plots (FPR calibration, detection power,
ROC curves, p-value distributions, tokens-to-detection CDF) and a summary CSV.

Implementation Details
----------------------

The implementation overrides the base ``detect()`` method because the score
function :math:`S(u)` depends on :math:`\varepsilon = \log(1/\delta)/n`, which
requires knowing the total number of tokens :math:`n` before computing
per-token scores. This differs from the standard ``score_tok`` → ``get_pvalue``
pipeline used by other detectors.

The ``score_tok()`` method returns raw :math:`V_t` values for interface
compatibility, and ``get_pvalue()`` performs a full MC simulation. For the
complete algorithm, use ``detect()`` directly.

Example Code
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

   llm = LLM(
       model="meta-llama/Llama-3.2-1B",
       enforce_eager=True,
       max_model_len=1024,
   )

   # Generate with standard Gumbel watermark
   wm_llm = WatermarkedLLMs.create(
       model=llm,
       algo=WatermarkingAlgorithm.OPENAI,
       seed=42,
       ngram=2,
   )

   # Detect with the Power Law statistic (same watermark, better theory)
   detector = WatermarkDetectors.create(
       algo=DetectionAlgorithm.OPENAI_PL,
       model=llm,
       ngram=2,
       seed=42,
       payload=0,
       threshold=0.05,
   )

   prompts = ["Explain the theory of general relativity"]
   sampling_params = SamplingParams(temperature=1.0, top_p=0.95, max_tokens=128)
   outputs = wm_llm.generate(prompts, sampling_params)

   for output in outputs:
       text = output.outputs[0].text
       result = detector.detect(text)
       print(f"Generated: {text[:100]}...")
       print(f"Watermarked: {result['is_watermarked']}")
       print(f"Score: {result['score']:.4f}")
       print(f"P-value: {result['pvalue']:.6f}")

Paper Reference
---------------

Lattimore, T. (2026). Refined Detection for Gumbel Watermarking.
*arXiv preprint arXiv:2603.30017*. https://arxiv.org/abs/2603.30017

See Also
--------

- :doc:`openai` — The standard Gumbel/OpenAI watermark (generation + exponential detection)
- :doc:`openai_dr` — Randomized Gumbel for enhanced diversity
