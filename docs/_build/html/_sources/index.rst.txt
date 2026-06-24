.. image:: ../resources/vLLM-WM-Logo.png
   :width: 120px
   :align: center

vLLM-Watermark
============================

Tiny. Hackable. Lightning-fast watermarking for researchers built on vLLM

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   installation

.. toctree::
   :maxdepth: 1
   :caption: Performance evaluation

   benchmark

.. toctree::
   :maxdepth: 1
   :caption: Watermarking algorithms

   algorithms/openai
   algorithms/openai_pl
   algorithms/openai_dr
   algorithms/maryland
   algorithms/pf
   algorithms/unigram
   algorithms/synthid
   algorithms/dip
   algorithms/sweet
   algorithms/blackbox
   algorithms/alignment_resampling

Supported Algorithms
--------------------

.. list-table::
   :header-rows: 1
   :class: algorithm-comparison

   * - Algorithm
     - Description
     - Paper
   * - **Gumbel/OpenAI**
     - Gumbel-Max trick for deterministic sampling
     - `Aaronson (2023) <https://scottaaronson.blog/?p=6823>`_
   * - **Power Law Detection**
     - Near-optimal detection for Gumbel watermarks
     - `Lattimore (2026) <https://arxiv.org/abs/2603.30017>`_
   * - **Randomized Gumbel**
     - Gumbel with double randomization for diversity
     - `Verma & Phan (2025) <https://arxiv.org/pdf/2506.04462>`_
   * - **KGW/Maryland**
     - Context-dependent green-red list with logit bias
     - `Kirchenbauer et al. (2023) <https://arxiv.org/pdf/2301.10226>`_
   * - **PF (Permute-and-Flip)**
     - Prefix-free coding with token permutations
     - `Lean et al. (2024) <https://arxiv.org/abs/2402.05864>`_
   * - **Unigram**
     - Context-independent fixed green-red list
     - `Zhao et al. (2024) <https://arxiv.org/abs/2306.17439>`_
   * - **SynthID**
     - Multi-layer tournament watermarking (non-distortionary)
     - `Dathathri et al. (2024) <https://www.nature.com/articles/s41586-024-08025-4>`_
   * - **DIP (DiPmark)**
     - Permutation-based probability redistribution
     - `Wu et al. (2023) <https://arxiv.org/abs/2310.07710>`_
   * - **SWEET**
     - Entropy-selective green-list biasing
     - `Lee et al. (2023) <https://arxiv.org/abs/2305.15060>`_
   * - **Black-Box**
     - Best-of-m rejection sampling (zero distortion)
     - `Bahri & Wieting (2026) <https://arxiv.org/abs/2410.02099>`_
   * - **Alignment Resampling**
     - Best-of-N with reward model (wraps any watermark)
     - `Verma & Phan (2025) <https://arxiv.org/pdf/2506.04462>`_

.. note::
   Each algorithm has different trade-offs between detectability, robustness, and text quality. See individual algorithm pages for detailed theory and examples.

Quick start
-----------

1. Install the package (see :doc:`installation`)
2. Choose an algorithm from :doc:`algorithms/index`
3. Run the example code to try it locally

For detailed API information, refer to the docstrings in the repository code.

Citation
--------

If you use vLLM-Watermark in your research, please cite:

.. raw:: html

   <div class="citation-box">

.. code-block:: bibtex

   @software{vllm_watermark,
     title  = {vLLM-Watermark: A tiny, hackable research framework for
               LLM watermarking experiments},
     author = {Apurv Verma},
     year   = {2025},
     url    = {https://github.com/dapurv5/vLLM-Watermark}
   }

.. raw:: html

   </div>
