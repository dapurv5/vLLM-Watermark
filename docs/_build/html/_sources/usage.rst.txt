Usage Guide
===========

This guide covers how to use vLLM-Watermark for text generation and watermark detection.

Quick Start
-----------

Basic Example
~~~~~~~~~~~~~

Here's a minimal example to get you started:

.. code-block:: python

   from vllm import LLM, SamplingParams
   from vllm_watermark.core import WatermarkedLLMs, WatermarkingAlgorithm
   from vllm_watermark.watermark_detectors import WatermarkDetectors, DetectionAlgorithm

   # Load a model
   llm = LLM(model="meta-llama/Llama-3.2-1B")

   # Create watermarked LLM
   wm_llm = WatermarkedLLMs.create(
       llm,
       algo=WatermarkingAlgorithm.OPENAI,
       seed=42,
       ngram=2
   )

   # Generate watermarked text
   prompts = ["Write a short story about a robot"]
   sampling_params = SamplingParams(temperature=0.8, max_tokens=100)
   outputs = wm_llm.generate(prompts, sampling_params)

   # Detect watermark
   detector = WatermarkDetectors.create(
       algo=DetectionAlgorithm.OPENAI_Z,
       model=llm,
       ngram=2,
       seed=42,
       threshold=0.05
   )

   for output in outputs:
       text = output.outputs[0].text
       result = detector.detect(text)
       print(f"Text: {text}")
       print(f"Watermarked: {result['is_watermarked']}")
       print(f"P-value: {result['pvalue']:.6f}")

Watermarking Algorithms
-----------------------

OpenAI Watermarking
~~~~~~~~~~~~~~~~~~~

The OpenAI algorithm uses a power-law transformation of token probabilities:

.. math::

   P_w(t) = \frac{P(t)^\gamma}{\sum_{t'} P(t')^\gamma}

where :math:`P(t)` is the original token probability and :math:`\gamma` controls the watermark strength.

.. code-block:: python

   from vllm_watermark.core import WatermarkingAlgorithm

   # Create OpenAI watermarked LLM
   wm_llm = WatermarkedLLMs.create(
       llm,
       algo=WatermarkingAlgorithm.OPENAI,
       seed=42,        # Random seed for reproducibility
       ngram=2,        # N-gram size for hashing
       gamma=0.5       # Watermark strength (0.1-1.0)
   )

   # Generate with OpenAI watermarking
   outputs = wm_llm.generate(prompts, sampling_params)

Maryland Watermarking
~~~~~~~~~~~~~~~~~~~~~

The Maryland algorithm uses statistical hypothesis testing:

.. math::

   H_0: \text{Text is not watermarked}
   H_1: \text{Text is watermarked}

.. code-block:: python

   # Create Maryland watermarked LLM
   wm_llm = WatermarkedLLMs.create(
       llm,
       algo=WatermarkingAlgorithm.MARYLAND,
       seed=123,
       gamma=0.5,      # Watermark strength
       delta=2.0       # Detection threshold
   )

   # Generate with Maryland watermarking
   outputs = wm_llm.generate(prompts, sampling_params)

PF Watermarking
~~~~~~~~~~~~~~~

Prefix-free coding watermarking for robust detection:

.. code-block:: python

   # Create PF watermarked LLM
   wm_llm = WatermarkedLLMs.create(
       llm,
       algo=WatermarkingAlgorithm.PF,
       seed=456,
       payload=0x12345678  # Custom payload
   )

   # Generate with PF watermarking
   outputs = wm_llm.generate(prompts, sampling_params)

Detection Methods
-----------------

OpenAI Detection
~~~~~~~~~~~~~~~~

Multiple detection variants are available:

.. code-block:: python

   from vllm_watermark.watermark_detectors import DetectionAlgorithm

   # Standard OpenAI detection
   detector = WatermarkDetectors.create(
       algo=DetectionAlgorithm.OPENAI,
       model=llm,
       ngram=2,
       seed=42,
       threshold=0.05
   )

   # Z-score approximation (faster)
   detector_z = WatermarkDetectors.create(
       algo=DetectionAlgorithm.OPENAI_Z,
       model=llm,
       ngram=2,
       seed=42,
       threshold=0.05
   )

   # Detect watermark
   result = detector.detect(text)
   print(f"Is watermarked: {result['is_watermarked']}")
   print(f"P-value: {result['pvalue']:.6f}")
   print(f"Score: {result['score']:.4f}")

Maryland Detection
~~~~~~~~~~~~~~~~~~

Statistical detection with hypothesis testing:

.. code-block:: python

   # Maryland detection
   detector = WatermarkDetectors.create(
       algo=DetectionAlgorithm.MARYLAND,
       model=llm,
       gamma=0.5,
       delta=2.0,
       threshold=0.05
   )

   result = detector.detect(text)
   print(f"Test statistic: {result['test_statistic']:.4f}")
   print(f"P-value: {result['pvalue']:.6f}")
   print(f"Confidence interval: {result['confidence_interval']}")

PF Detection
~~~~~~~~~~~~

Prefix-free detection with payload extraction:

.. code-block:: python

   # PF detection
   detector = WatermarkDetectors.create(
       algo=DetectionAlgorithm.PF,
       model=llm,
       payload=0x12345678
   )

   result = detector.detect(text)
   print(f"Extracted payload: {result['payload']}")
   print(f"Payload match: {result['payload_match']}")

Advanced Usage
--------------

Custom Sampling Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~

Combine watermarking with advanced sampling:

.. code-block:: python

   from vllm import SamplingParams

   # Advanced sampling with watermarking
   sampling_params = SamplingParams(
       temperature=0.8,
       top_p=0.95,
       top_k=50,
       max_tokens=200,
       stop=["\n\n", "END"],
       presence_penalty=0.1,
       frequency_penalty=0.1
   )

   outputs = wm_llm.generate(prompts, sampling_params)

Batch Processing
~~~~~~~~~~~~~~~~

Process multiple prompts efficiently:

.. code-block:: python

   # Batch generation
   prompts = [
       "Write a poem about AI",
       "Explain quantum computing",
       "Tell a story about space travel"
   ]

   outputs = wm_llm.generate(prompts, sampling_params)

   # Batch detection
   texts = [output.outputs[0].text for output in outputs]
   results = [detector.detect(text) for text in texts]

   for i, (text, result) in enumerate(zip(texts, results)):
       print(f"Text {i+1}: {'Watermarked' if result['is_watermarked'] else 'Not watermarked'}")

Parameter Tuning
~~~~~~~~~~~~~~~~

Optimize watermark strength and detection:

.. code-block:: python

   import numpy as np

   # Test different gamma values
   gamma_values = [0.1, 0.3, 0.5, 0.7, 0.9]
   results = []

   for gamma in gamma_values:
       wm_llm = WatermarkedLLMs.create(
           llm,
           algo=WatermarkingAlgorithm.OPENAI,
           seed=42,
           ngram=2,
           gamma=gamma
       )

       outputs = wm_llm.generate(prompts, sampling_params)
       text = outputs[0].outputs[0].text

       detector = WatermarkDetectors.create(
           algo=DetectionAlgorithm.OPENAI_Z,
           model=llm,
           ngram=2,
           seed=42,
           threshold=0.05
       )

       result = detector.detect(text)
       results.append({
           'gamma': gamma,
           'pvalue': result['pvalue'],
           'score': result['score']
       })

   # Find optimal gamma
   best_result = min(results, key=lambda x: x['pvalue'])
   print(f"Optimal gamma: {best_result['gamma']}")

Error Handling
--------------

Robust error handling for production use:

.. code-block:: python

   import logging
   from vllm_watermark.core import WatermarkedLLMs

   logging.basicConfig(level=logging.INFO)

   try:
       # Create watermarked LLM with error handling
       wm_llm = WatermarkedLLMs.create(
           llm,
           algo=WatermarkingAlgorithm.OPENAI,
           seed=42,
           ngram=2,
           debug=True  # Enable debug mode
       )

       outputs = wm_llm.generate(prompts, sampling_params)

   except Exception as e:
       logging.error(f"Watermarking failed: {e}")
       # Fallback to non-watermarked generation
       outputs = llm.generate(prompts, sampling_params)

Performance Optimization
------------------------

GPU Memory Management
~~~~~~~~~~~~~~~~~~~~~

Optimize for large models:

.. code-block:: python

   import os

   # Memory optimization
   os.environ['CUDA_MEM_FRACTION'] = '0.8'
   os.environ['VLLM_USE_FP16'] = '1'

   # Use smaller batch sizes for large models
   sampling_params = SamplingParams(
       temperature=0.8,
       max_tokens=100,
       max_tokens_per_batch=50  # Limit batch size
   )

Caching and Reuse
~~~~~~~~~~~~~~~~~

Reuse watermarked LLM instances:

.. code-block:: python

   # Create once, reuse many times
   wm_llm = WatermarkedLLMs.create(
       llm,
       algo=WatermarkingAlgorithm.OPENAI,
       seed=42,
       ngram=2
   )

   # Reuse for multiple generations
   for i in range(10):
       outputs = wm_llm.generate([f"Prompt {i}"], sampling_params)
       # Process outputs...

Best Practices
--------------

1. **Consistent Parameters**: Use the same seed and parameters for generation and detection
2. **Threshold Selection**: Choose appropriate detection thresholds (0.01-0.1)
3. **Model Compatibility**: Ensure your model is compatible with vLLM
4. **Memory Management**: Monitor GPU memory usage for large models
5. **Error Handling**: Implement robust error handling for production use
6. **Testing**: Test watermarking and detection on your specific use case

.. note::
   Always test watermarking and detection with your specific model and use case.
   Different models may require different parameter tuning for optimal results.

Next Steps
----------

- Explore the :doc:`api` for detailed API reference
- Check out :doc:`examples` for more complex examples
- Learn about the :doc:`mathematical_foundations` of watermarking algorithms