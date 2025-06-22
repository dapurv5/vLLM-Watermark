Mathematical Foundations
========================

This document provides the mathematical foundations and theoretical background for the watermarking algorithms implemented in vLLM-Watermark.

Overview
--------

Watermarking algorithms for language models work by subtly modifying the token selection process during text generation. The goal is to embed a detectable signal (watermark) while maintaining the quality and coherence of the generated text.

Mathematical Notation
---------------------

Throughout this document, we use the following notation:

- :math:`P(t)` - Original probability of token :math:`t`
- :math:`P_w(t)` - Watermarked probability of token :math:`t`
- :math:`\gamma` - Watermark strength parameter
- :math:`\delta` - Detection threshold parameter
- :math:`H(\cdot)` - Hash function
- :math:`\text{seed}` - Random seed for reproducibility
- :math:`n` - N-gram size for context hashing

OpenAI Watermarking Algorithm
-----------------------------

The OpenAI watermarking algorithm, introduced in "A Watermark for Large Language Models" (Kirchenbauer et al., 2023), uses a power-law transformation of token probabilities.

Mathematical Formulation
~~~~~~~~~~~~~~~~~~~~~~~~

The algorithm works as follows:

1. **Context Hashing**: For each token position, compute a hash of the previous :math:`n` tokens:

   .. math::

      h_i = H(\text{seed}, t_{i-n+1}, t_{i-n+2}, \ldots, t_i)

2. **Green List Selection**: Use the hash to determine a "green list" of tokens:

   .. math::

      \text{green}_i = \{t : h_i \bmod 2 = 0\}

3. **Probability Transformation**: Apply a power-law transformation to green list tokens:

   .. math::

      P_w(t) = \begin{cases}
         \frac{P(t)^\gamma}{\sum_{t' \in \text{green}_i} P(t')^\gamma} & \text{if } t \in \text{green}_i \\
         \frac{P(t)^\gamma}{\sum_{t' \notin \text{green}_i} P(t')^\gamma} & \text{if } t \notin \text{green}_i
      \end{cases}

4. **Normalization**: Ensure probabilities sum to 1:

   .. math::

      P_w(t) = \frac{P_w(t)}{\sum_{t'} P_w(t')}

Detection Algorithm
~~~~~~~~~~~~~~~~~~~

The detection algorithm computes a test statistic based on the proportion of green list tokens:

.. math::

   S = \frac{1}{L} \sum_{i=1}^{L} \mathbb{I}[t_i \in \text{green}_i]

where :math:`L` is the text length and :math:`\mathbb{I}[\cdot]` is the indicator function.

Under the null hypothesis (no watermark), :math:`S` follows a binomial distribution:

.. math::

   S \sim \text{Binomial}(L, 0.5)

The p-value is computed as:

.. math::

   p\text{-value} = P(S \geq s_{\text{obs}} | H_0)

where :math:`s_{\text{obs}}` is the observed test statistic.

Z-Score Approximation
~~~~~~~~~~~~~~~~~~~~~

For computational efficiency, we use a z-score approximation:

.. math::

   Z = \frac{S - 0.5}{\sqrt{0.25/L}}

Under :math:`H_0`, :math:`Z \sim \mathcal{N}(0, 1)`, and the p-value is:

.. math::

   p\text{-value} = 1 - \Phi(Z)

where :math:`\Phi` is the standard normal cumulative distribution function.

Maryland Watermarking Algorithm
-------------------------------

The Maryland algorithm, based on "A Statistical Approach to Neural Text Generation" (Aaronson & Kirchner, 2023), uses statistical hypothesis testing for watermarking.

Mathematical Formulation
~~~~~~~~~~~~~~~~~~~~~~~~

1. **Token Scoring**: For each token, compute a score based on the model's logits:

   .. math::

      s_i = \log P(t_i) - \log P_{\text{uniform}}(t_i)

2. **Cumulative Score**: Compute the cumulative score:

   .. math::

      C_i = \sum_{j=1}^{i} s_j

3. **Watermarking Decision**: Apply watermarking based on a threshold:

   .. math::

      P_w(t) = \begin{cases}
         P(t) \cdot \exp(\gamma) & \text{if } C_{i-1} < \delta \\
         P(t) & \text{otherwise}
      \end{cases}

Detection Algorithm
~~~~~~~~~~~~~~~~~~~

The detection algorithm computes a test statistic:

.. math::

   T = \frac{C_L}{\sqrt{L \cdot \text{Var}(s)}}

where :math:`\text{Var}(s)` is the variance of token scores.

Under :math:`H_0`, :math:`T \sim \mathcal{N}(0, 1)`, and the p-value is:

.. math::

   p\text{-value} = 2 \cdot (1 - \Phi(|T|))

PF Watermarking Algorithm
-------------------------

The Prefix-Free (PF) watermarking algorithm uses prefix-free coding to embed payloads in generated text.

Mathematical Formulation
~~~~~~~~~~~~~~~~~~~~~~~~

1. **Payload Encoding**: Encode the payload :math:`m` as a binary string:

   .. math::

      b = \text{encode}(m)

2. **Prefix-Free Construction**: Construct a prefix-free code :math:`C`:

   .. math::

      C = \{c_1, c_2, \ldots, c_k\}

   where no code word is a prefix of another.

3. **Token Selection**: For each position, select tokens based on the code:

   .. math::

      P_w(t) = \begin{cases}
         P(t) \cdot \exp(\gamma) & \text{if } t \in C_i \\
         P(t) & \text{otherwise}
      \end{cases}

Detection Algorithm
~~~~~~~~~~~~~~~~~~~

The detection algorithm extracts the embedded payload:

.. math::

   \hat{m} = \text{decode}(\hat{b})

where :math:`\hat{b}` is the extracted binary string.

The detection confidence is based on the Hamming distance:

.. math::

   \text{confidence} = 1 - \frac{d_H(b, \hat{b})}{|b|}

where :math:`d_H` is the Hamming distance.

Statistical Properties
----------------------

False Positive Rate
~~~~~~~~~~~~~~~~~~~

The false positive rate (FPR) is the probability of incorrectly detecting a watermark in non-watermarked text:

.. math::

   \text{FPR} = P(\text{detect} | H_0) = \alpha

where :math:`\alpha` is the significance level (typically 0.05).

False Negative Rate
~~~~~~~~~~~~~~~~~~~

The false negative rate (FNR) is the probability of failing to detect a watermark in watermarked text:

.. math::

   \text{FNR} = P(\text{not detect} | H_1) = \beta

The power of the test is :math:`1 - \beta`.

Detection Power
~~~~~~~~~~~~~~~

The detection power depends on the watermark strength :math:`\gamma` and text length :math:`L`:

.. math::

   \text{Power} = P(\text{detect} | H_1) = 1 - \beta

For the OpenAI algorithm, the power increases with :math:`\gamma` and :math:`L`.

Parameter Selection
-------------------

Watermark Strength :math:`\gamma`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The watermark strength :math:`\gamma` controls the trade-off between detectability and text quality:

- **Small :math:`\gamma`** (0.1-0.3): Subtle watermarking, high text quality, lower detection power
- **Medium :math:`\gamma`** (0.3-0.7): Balanced approach, good detection power
- **Large :math:`\gamma`** (0.7-1.0): Strong watermarking, potentially lower text quality, high detection power

Optimal Selection
~~~~~~~~~~~~~~~~~

The optimal :math:`\gamma` depends on your specific use case:

.. math::

   \gamma^* = \arg\max_{\gamma} \text{Power}(\gamma) \text{ s.t. } \text{Quality}(\gamma) \geq \text{threshold}

Detection Threshold :math:`\delta`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The detection threshold controls the false positive rate:

.. math::

   \alpha = P(S \geq \delta | H_0)

Typical values:
- **Conservative**: :math:`\alpha = 0.01` (1% FPR)
- **Standard**: :math:`\alpha = 0.05` (5% FPR)
- **Liberal**: :math:`\alpha = 0.10` (10% FPR)

N-gram Size :math:`n`
~~~~~~~~~~~~~~~~~~~~~

The n-gram size affects the context used for watermarking:

- **Small :math:`n`** (1-2): Local context, faster computation
- **Large :math:`n`** (3-5): More context, potentially better detection

Computational Complexity
------------------------

Time Complexity
~~~~~~~~~~~~~~~

- **Generation**: :math:`O(L \cdot V)` where :math:`L` is text length and :math:`V` is vocabulary size
- **Detection**: :math:`O(L)` for most algorithms
- **Hash computation**: :math:`O(n)` per token

Space Complexity
~~~~~~~~~~~~~~~~

- **Generation**: :math:`O(V)` for probability storage
- **Detection**: :math:`O(1)` for most algorithms
- **Hash storage**: :math:`O(n)` for context window

Security Considerations
-----------------------

Adversarial Attacks
~~~~~~~~~~~~~~~~~~~

Watermarking algorithms may be vulnerable to various attacks:

1. **Paraphrasing**: Rewriting text to remove watermarks
2. **Token Substitution**: Replacing tokens with synonyms
3. **Text Truncation**: Removing parts of the text
4. **Model Fine-tuning**: Training to avoid watermarking

Robustness Measures
~~~~~~~~~~~~~~~~~~~

To improve robustness:

1. **Multiple Algorithms**: Use different watermarking schemes
2. **Parameter Randomization**: Vary parameters across generations
3. **Ensemble Detection**: Combine multiple detection methods
4. **Adversarial Training**: Train against known attacks

Theoretical Bounds
------------------

Information-Theoretic Limits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The maximum information that can be embedded is bounded by:

.. math::

   I \leq H(P) - H(P_w)

where :math:`H(\cdot)` is the entropy.

Detection Bounds
~~~~~~~~~~~~~~~~

For a given false positive rate :math:`\alpha`, the minimum detectable watermark strength is:

.. math::

   \gamma_{\min} = \sqrt{\frac{2 \log(1/\alpha)}{L}}

where :math:`L` is the text length.

References
----------

1. Kirchenbauer, J., et al. (2023). "A Watermark for Large Language Models." arXiv:2301.10226.
2. Aaronson, S., & Kirchner, H. (2023). "A Statistical Approach to Neural Text Generation." arXiv:2301.03109.
3. Kuditipudi, R., et al. (2023). "Robust Distortion-Free Watermarks for Language Models." arXiv:2307.15593.

.. note::
   This mathematical foundation provides the theoretical basis for the watermarking algorithms.
   For practical implementation details, see the :doc:`api` documentation.