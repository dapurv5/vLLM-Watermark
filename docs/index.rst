Welcome to vLLM-Watermark's documentation!
=====================================

vLLM-Watermark is a Python package for implementing various watermarking algorithms for LLM outputs, with support for different backends including vLLM and SGLang.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   api
   contributing

Installation
-----------

You can install vLLM-Watermark using pip:

.. code-block:: bash

   pip install vllm-watermark

For development installation:

.. code-block:: bash

   git clone https://github.com/yourusername/vllm-watermark.git
   cd vllm-watermark
   pip install -e ".[dev]"

Quick Start
----------

Here's a quick example of how to use vLLM-Watermark:

.. code-block:: python

   from vllm_watermark import WatermarkFactory

   # Create a watermarking algorithm instance
   watermark = WatermarkFactory.create("kgw", gamma=0.5, delta=2.0)

   # Apply watermarking to logits
   watermarked_logits = watermark.apply(logits)

   # Detect watermark in text
   result = watermark.detect(text)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`