.. image:: ../resources/vLLM-WM-Logo.png
   :width: 120px
   :align: center

Welcome to vLLM-Watermark's documentation!
==========================================

vLLM-Watermark is a Python package for implementing various watermarking algorithms for LLM outputs, with support for different backends including vLLM and SGLang.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   api
   examples
   mathematical_foundations
   contributing

Installation
------------

You can install vLLM-Watermark using pip:

.. code-block:: bash

   pip install vllm-watermark

For development installation:

.. code-block:: bash

   git clone https://github.com/dapurv5/vLLM-Watermark.git
   cd vllm-watermark
   pip install -e ".[dev]"

Quick Start
-----------

Here's a quick example of how to use vLLM-Watermark:

.. code-block:: python

   from vllm_watermark import WatermarkGenerator, WatermarkDetector

   # Create a watermarking algorithm instance
   llm = LLM(model="meta-llama/Llama-3.2-1B") # vLLM model
   wm_llm = WatermarkGenerator.create(llm, "kgw", gamma=0.5, delta=2.0)

   # Apply watermarking while generating
   watermarked_logits = wm_llm.generate(prompt)

   kgw_detector = WatermarkDetector.create("kgw", gamma=0.5, delta=2.0)
   # Detect watermark in text
   result = kgw_detector.detect(watermarked_logits)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`