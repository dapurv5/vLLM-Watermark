Performance Benchmarking
========================

Comprehensive benchmarking tool for evaluating watermarking algorithms performance on the C4 dataset. Uses isolated processes to ensure complete GPU memory cleanup between algorithms.

Quick Start
-----------

.. code-block:: bash

   # Install required dependencies
   pip install tabulate

   # From project root directory
   ./scripts/benchmark/run_benchmark.sh

Supported Algorithms
--------------------

.. list-table::
   :header-rows: 1

   * - Algorithm
     - Parameters
     - Description
   * - **OPENAI**
     - ngram, seed, payload
     - Power-law transformation with n-gram hashing
   * - **MARYLAND**
     - ngram, seed, gamma, delta
     - Statistical watermarking with hypothesis testing
   * - **MARYLAND_L**
     - ngram, seed, gamma, delta
     - Maryland algorithm with logit processing
   * - **PF**
     - ngram, seed, payload
     - Prefix-free coding watermarking

Usage Examples
--------------

**Shell Script (Recommended):**

.. code-block:: bash

   # Default: All algorithms, 5000 samples, meta-llama/Llama-3.2-1B
   ./scripts/benchmark/run_benchmark.sh

   # Custom model
   ./scripts/benchmark/run_benchmark.sh meta-llama/Llama-3.2-3B

   # Specific algorithms
   ./scripts/benchmark/run_benchmark.sh meta-llama/Llama-3.2-1B "OPENAI MARYLAND"

   # Custom sample count
   ./scripts/benchmark/run_benchmark.sh meta-llama/Llama-3.2-1B "OPENAI PF" 1000

**Python Script (Advanced):**

.. code-block:: bash

   python scripts/benchmark/benchmark_watermarks.py \
       --model_name meta-llama/Llama-3.2-1B \
       --algorithms OPENAI MARYLAND PF \
       --num_samples 5000 \
       --data_path resources/datasets/c4/processed_c4.jsonl

Output Metrics
--------------

The benchmark provides comprehensive metrics for each algorithm:

**Detection Performance:**
- Precision: TP / (TP + FP)
- Recall: TP / (TP + FN)
- F1 Score: 2 × (Precision × Recall) / (Precision + Recall)
- Accuracy: (TP + TN) / (TP + TN + FP + FN)
- FPR: False Positive Rate = FP / (FP + TN)
- FNR: False Negative Rate = FN / (TP + FN)

**Performance Metrics:**
- Input Tokens/Second: Throughput for input processing
- Output Tokens/Second: Throughput for text generation
- Generation Time: Time spent generating watermarked + unwatermarked text
- Detection Time: Time spent running detection
- Total Time: Generation + Detection time

.. note::
   Results are saved to the ``output/benchmark/`` directory with detailed configuration parameters for reproducibility.
