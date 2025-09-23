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
   algorithms/maryland
   algorithms/pf

Supported Algorithms
--------------------

The following table summarizes the three main watermarking algorithms implemented in this package:

.. list-table::
   :header-rows: 1
   :class: algorithm-comparison

   * - Algorithm
     - Description
     - Paper
   * - **Gumbel/OpenAI**
     - Uses Gumbel-Max trick for deterministic sampling
     - `Aaronson (2023) <https://scottaaronson.blog/?p=6823>`_
   * - **KGW/Maryland**
     - Green-red token partitioning with logit bias
     - `Kirchenbauer et al. (2023) <https://arxiv.org/pdf/2301.10226>`_
   * - **PF (Permute-and-Flip)**
     - Prefix-free coding with token permutations
     - `Lean et al. (2024) <https://arxiv.org/abs/2402.05864>`_

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