Installation Guide
==================

This guide covers how to install vLLM-Watermark on different platforms and environments.

System Requirements
-------------------

- **Python**: 3.11 or higher
- **CUDA**: 12.2 or higher (for GPU acceleration)
- **Memory**: At least 8GB RAM (16GB+ recommended)
- **Storage**: 2GB+ free space for models and dependencies

.. note::
   vLLM-Watermark is designed to work with CUDA-enabled GPUs for optimal performance.
   CPU-only operation is possible but significantly slower.

Installation Methods
--------------------

From PyPI (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~

The easiest way to install vLLM-Watermark is using pip:

.. code-block:: bash

   pip install vllm-watermark

For development dependencies:

.. code-block:: bash

   pip install vllm-watermark[dev]

From Source
~~~~~~~~~~~

To install from the latest development version:

.. code-block:: bash

   git clone https://github.com/dapurv5/vLLM-Watermark.git
   cd vllm-watermark
   pip install -e ".[dev]"

Using Conda
~~~~~~~~~~~

If you prefer using conda:

.. code-block:: bash

   conda create -n vllm-watermark python=3.10
   conda activate vllm-watermark
   pip install vllm-watermark

Docker Installation
~~~~~~~~~~~~~~~~~~~

For containerized deployment:

.. code-block:: dockerfile

   FROM python:3.10-slim

   # Install system dependencies
   RUN apt-get update && apt-get install -y \
       build-essential \
       git \
       && rm -rf /var/lib/apt/lists/*

   # Install vLLM-Watermark
   RUN pip install vllm-watermark

   # Set working directory
   WORKDIR /app

   # Copy your application code
   COPY . .

   CMD ["python", "your_app.py"]

Verification
------------

After installation, verify that everything is working:

.. code-block:: python

   import vllm_watermark
   print(f"vLLM-Watermark version: {vllm_watermark.__version__}")

   # Test basic functionality
   from vllm_watermark.core import WatermarkingAlgorithm
   print("Available algorithms:", [algo.value for algo in WatermarkingAlgorithm])

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**CUDA Version Mismatch**
   If you encounter CUDA version conflicts:

   .. code-block:: bash

      # Check your CUDA version
      nvidia-smi

      # Install compatible PyTorch version
      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

**Memory Issues**
   For out-of-memory errors:

   .. code-block:: python

      import os
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use specific GPU
      os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'  # Disable multiprocessing

**Import Errors**
   If you get import errors for vLLM:

   .. code-block:: bash

      # Reinstall vLLM with compatible version
      pip uninstall vllm
      pip install vllm>=0.2.0

**Permission Issues**
   For permission errors on Linux/macOS:

   .. code-block:: bash

      # Use user installation
      pip install --user vllm-watermark

      # Or use virtual environment
      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate
      pip install vllm-watermark

Performance Optimization
------------------------

GPU Memory Management
~~~~~~~~~~~~~~~~~~~~~

To optimize GPU memory usage:

.. code-block:: python

   import os

   # Limit GPU memory usage
   os.environ['CUDA_MEM_FRACTION'] = '0.8'

   # Use mixed precision
   os.environ['VLLM_USE_FP16'] = '1'

Multi-GPU Setup
~~~~~~~~~~~~~~~

For multi-GPU environments:

.. code-block:: python

   import os

   # Use all available GPUs
   os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

   # Or specify specific GPUs
   os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

Development Setup
-----------------

For contributors and developers:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/dapurv5/vLLM-Watermark.git
   cd vllm-watermark

   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install in development mode
   pip install -e ".[dev]"

   # Install pre-commit hooks
   pre-commit install

   # Run tests
   pytest

   # Build documentation
   cd docs
   make html

Next Steps
----------

After successful installation:

1. Read the :doc:`usage` guide for basic usage
2. Check out the :doc:`examples` for working examples
3. Explore the :doc:`api` for detailed API reference
4. Join our community for support and discussions