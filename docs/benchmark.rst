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

* **Precision:** TP / (TP + FP)
* **Recall:** TP / (TP + FN)
* **F1 Score:** 2 × (Precision × Recall) / (Precision + Recall)
* **Accuracy:** (TP + TN) / (TP + TN + FP + FN)
* **FPR:** False Positive Rate = FP / (FP + TN)
* **FNR:** False Negative Rate = FN / (TP + FN)

**Performance Metrics:**

* **Input Tokens/Second:** Throughput for input processing
* **Output Tokens/Second:** Throughput for text generation
* **Generation Time:** Time spent generating watermarked + unwatermarked text
* **Detection Time:** Time spent running detection
* **Total Time:** Generation + Detection time

Sample Results
--------------

LLaMA-3.2-1B Performance on C4 Dataset (500 samples)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Detection Performance Comparison**

.. list-table::
   :header-rows: 1
   :widths: 15 15 15 15 15 15 15

   * - Algorithm
     - Configuration
     - Precision
     - Recall
     - F1 Score
     - Accuracy
     - FPR
   * - **OPENAI**
     - ngram=2, seed=42, payload=0
     - 0.941
     - 0.996
     - 0.968
     - 0.967
     - 0.062
   * - **MARYLAND**
     - ngram=2, seed=42, γ=0.5, δ=1.0
     - 0.902
     - 0.882
     - 0.892
     - 0.893
     - 0.096
   * - **MARYLAND_L**
     - ngram=2, seed=42, γ=0.5, δ=1.0
     - 0.899
     - 0.962
     - 0.929
     - 0.927
     - 0.108

**Performance Metrics Comparison**

.. list-table::
   :header-rows: 1
   :widths: 15 20 20 20 20

   * - Algorithm
     - Generation Rate (tokens/s)
     - Generation Time (s)
     - Detection Time (s)
     - Total Time (s)
   * - **OPENAI**
     - 7,487
     - 54.9
     - 56.9
     - 111.8
   * - **MARYLAND**
     - 5,968
     - 66.1
     - 104.4
     - 170.5
   * - **MARYLAND_L**
     - 3,093
     - 139.1
     - 200.7
     - 339.8

**Key Observations:**

- **OPENAI** achieves the highest precision (0.941) and recall (0.996), making it excellent for applications requiring minimal false positives and false negatives
- **MARYLAND_L** provides a good balance with high recall (0.962) but at the cost of slower generation speed (3,093 tokens/s)
- **MARYLAND** offers moderate performance across all metrics with faster processing than MARYLAND_L
- Generation throughput varies significantly: OPENAI (7,487 tokens/s) > MARYLAND (5,968 tokens/s) > MARYLAND_L (3,093 tokens/s)

.. note::
   Results are saved to the ``output/benchmark/`` directory with detailed configuration parameters for reproducibility.
