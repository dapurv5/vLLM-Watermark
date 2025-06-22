API Reference
=============

This section provides detailed API documentation for all classes, functions, and modules in vLLM-Watermark.

Core Module
-----------

.. automodule:: vllm_watermark.core
   :members:
   :undoc-members:
   :show-inheritance:

Watermark Generators
--------------------

.. automodule:: vllm_watermark.watermark_generators
   :members:
   :undoc-members:
   :show-inheritance:

Watermark Detectors
-------------------

.. automodule:: vllm_watermark.watermark_detectors
   :members:
   :undoc-members:
   :show-inheritance:

Samplers
--------

.. automodule:: vllm_watermark.samplers
   :members:
   :undoc-members:
   :show-inheritance:

Logit Processors
----------------

.. automodule:: vllm_watermark.logit_processors
   :members:
   :undoc-members:
   :show-inheritance:

Factory Classes
---------------

Watermark Generator Factory
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: vllm_watermark.watermark_generators.factory
   :members:
   :undoc-members:
   :show-inheritance:

Watermark Detector Factory
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: vllm_watermark.watermark_detectors.factory
   :members:
   :undoc-members:
   :show-inheritance:

Base Classes
------------

Base Watermark Generator
~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: vllm_watermark.watermark_generators.base
   :members:
   :undoc-members:
   :show-inheritance:

Base Watermark Detector
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: vllm_watermark.watermark_detectors.base
   :members:
   :undoc-members:
   :show-inheritance:

Enumerations
------------

.. automodule:: vllm_watermark.core
   :members: WatermarkingAlgorithm, DetectionAlgorithm
   :undoc-members:

Data Structures
---------------

Detection Results
~~~~~~~~~~~~~~~~~

The detection methods return dictionaries with the following structure:

.. code-block:: python

   {
       'is_watermarked': bool,           # Whether watermark was detected
       'pvalue': float,                  # Statistical p-value
       'score': float,                   # Detection score
       'confidence': float,              # Detection confidence (0-1)
       'test_statistic': float,          # Test statistic value
       'threshold': float,               # Detection threshold used
       'metadata': dict                  # Additional algorithm-specific data
   }

Generation Results
~~~~~~~~~~~~~~~~~~

The generation methods return vLLM-compatible output objects with additional watermarking metadata:

.. code-block:: python

   {
       'text': str,                      # Generated text
       'tokens': List[int],              # Token IDs
       'logprobs': List[float],          # Token log probabilities
       'watermark_info': dict            # Watermarking metadata
   }

Error Handling
--------------

Common Exceptions
~~~~~~~~~~~~~~~~~

.. code-block:: python

   class WatermarkError(Exception):
       """Base exception for watermarking errors."""
       pass

   class WatermarkGenerationError(WatermarkError):
       """Raised when watermark generation fails."""
       pass

   class WatermarkDetectionError(WatermarkError):
       """Raised when watermark detection fails."""
       pass

   class UnsupportedAlgorithmError(WatermarkError):
       """Raised when an unsupported algorithm is requested."""
       pass

   class ParameterError(WatermarkError):
       """Raised when invalid parameters are provided."""
       pass

Usage Examples
--------------

Basic Watermarking
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from vllm import LLM, SamplingParams
   from vllm_watermark.core import WatermarkedLLMs, WatermarkingAlgorithm

   # Create watermarked LLM
   llm = LLM(model="meta-llama/Llama-3.2-1B")
   wm_llm = WatermarkedLLMs.create(
       llm,
       algo=WatermarkingAlgorithm.OPENAI,
       seed=42,
       ngram=2,
       gamma=0.5
   )

   # Generate watermarked text
   prompts = ["Write a short story"]
   sampling_params = SamplingParams(temperature=0.8, max_tokens=100)
   outputs = wm_llm.generate(prompts, sampling_params)

Advanced Detection
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from vllm_watermark.watermark_detectors import WatermarkDetectors, DetectionAlgorithm

   # Create detector with custom parameters
   detector = WatermarkDetectors.create(
       algo=DetectionAlgorithm.OPENAI_Z,
       model=llm,
       ngram=2,
       seed=42,
       threshold=0.01,  # Conservative threshold
       payload=0        # For algorithms that support payloads
   )

   # Detect watermark with error handling
   try:
       result = detector.detect(text)
       if result['is_watermarked']:
           print(f"Watermark detected! P-value: {result['pvalue']:.6f}")
       else:
           print("No watermark detected")
   except WatermarkDetectionError as e:
       print(f"Detection failed: {e}")

Batch Processing
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Batch watermarking
   texts = ["Text 1", "Text 2", "Text 3"]
   results = []

   for text in texts:
       try:
           result = detector.detect(text)
           results.append(result)
       except WatermarkDetectionError:
           results.append({'is_watermarked': False, 'error': 'Detection failed'})

   # Analyze results
   watermarked_count = sum(1 for r in results if r.get('is_watermarked', False))
   print(f"Detected {watermarked_count}/{len(texts)} watermarked texts")

Performance Optimization
------------------------

Memory Management
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import os
   import gc

   # Optimize GPU memory
   os.environ['CUDA_MEM_FRACTION'] = '0.8'
   os.environ['VLLM_USE_FP16'] = '1'

   # Reuse detector instances
   detector = WatermarkDetectors.create(...)

   for text in texts:
       result = detector.detect(text)
       # Process result...

   # Clean up
   del detector
   gc.collect()

Caching
~~~~~~~

.. code-block:: python

   from functools import lru_cache

   @lru_cache(maxsize=1000)
   def cached_detect(text_hash, detector_params):
       """Cache detection results for repeated texts."""
       detector = WatermarkDetectors.create(**detector_params)
       return detector.detect(text_hash)

Configuration
-------------

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # vLLM configuration
   os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'  # Disable multiprocessing
   os.environ['VLLM_USE_V1'] = '0'                     # Use V0 engine
   os.environ['CUDA_VISIBLE_DEVICES'] = '0'            # Use specific GPU

   # Watermarking configuration
   os.environ['VLLM_WATERMARK_DEBUG'] = '1'            # Enable debug mode
   os.environ['VLLM_WATERMARK_LOG_LEVEL'] = 'INFO'     # Set log level

Logging
~~~~~~~

.. code-block:: python

   import logging

   # Configure logging
   logging.basicConfig(
       level=logging.INFO,
       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
   )

   # Get logger for watermarking
   logger = logging.getLogger('vllm_watermark')
   logger.setLevel(logging.DEBUG)

Best Practices
--------------

1. **Parameter Consistency**: Always use the same parameters for generation and detection
2. **Error Handling**: Implement robust error handling for production use
3. **Memory Management**: Monitor GPU memory usage and clean up resources
4. **Caching**: Cache detector instances and results when appropriate
5. **Logging**: Use appropriate log levels for debugging and monitoring
6. **Testing**: Test with your specific models and use cases

.. note::
   For more detailed examples and use cases, see the :doc:`examples` section.
   For mathematical foundations, see :doc:`mathematical_foundations`.