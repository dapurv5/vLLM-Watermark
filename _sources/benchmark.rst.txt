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
   * - **PF**
     - ngram, seed, payload
     - Prefix-free coding watermarking
   * - **UNIGRAM**
     - ngram, seed, gamma, delta, hash_key
     - Unigram-based statistical watermarking
   * - **SYNTHID**
     - ngram, seed
     - Google DeepMind's tournament-based watermarking
   * - **DIP**
     - ngram, seed, alpha, gamma, hash_key
     - Distributional-preserving watermarking
   * - **SWEET**
     - ngram, seed, gamma, delta, hash_key, entropy_threshold
     - Entropy-aware statistical watermarking
   * - **BLACKBOX**
     - ngram, hash_key, n_candidates
     - Black-box watermark detection via candidate search

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
   :widths: 12 22 10 10 10 10 10

   * - Algorithm
     - Configuration
     - Precision
     - Recall
     - F1 Score
     - Accuracy
     - FPR
   * - **OPENAI**
     - ngram=2, seed=42, payload=0
     - 0.925
     - 0.992
     - 0.958
     - 0.958
     - 0.008
   * - **MARYLAND**
     - ngram=2, seed=42, γ=0.5, δ=1.0
     - 0.922
     - 0.964
     - 0.942
     - 0.942
     - 0.016
   * - **PF**
     - ngram=2, seed=42, payload=0
     - 0.902
     - 0.998
     - 0.948
     - 0.945
     - 0.108
   * - **UNIGRAM**
     - ngram=1, seed=42, γ=0.5, δ=2.0
     - 0.882
     - 0.998
     - 0.936
     - 0.932
     - 0.134
   * - **SYNTHID**
     - ngram=4, seed=42
     - 0.929
     - 0.996
     - 0.961
     - 0.960
     - 0.076
   * - **DIP**
     - ngram=2, seed=42, α=0.45, γ=0.5
     - 1.000
     - 0.896
     - 0.945
     - 0.948
     - 0.000
   * - **SWEET**
     - ngram=2, seed=42, γ=0.5, δ=2.0
     - 0.914
     - 0.952
     - 0.932
     - 0.931
     - 0.090
   * - **BLACKBOX**
     - ngram=4, n_candidates=128
     - 1.000
     - 0.430
     - 0.601
     - 0.715
     - 0.000

**Key Observations:**

- **SYNTHID** achieves the best F1 score (0.961) with strong precision (0.929) and near-perfect recall (0.996)
- **OPENAI** is close behind (F1=0.958) and has the lowest FPR (0.008) among high-recall algorithms
- **DIP** and **BLACKBOX** achieve perfect precision (1.000, zero false positives), but BLACKBOX trades this for lower recall (0.430)
- **PF** and **UNIGRAM** have near-perfect recall (0.998) but higher false positive rates (0.108, 0.134)
- **MARYLAND** and **SWEET** offer balanced precision-recall trade-offs with moderate FPR

.. note::
   Results are saved to the ``output/benchmark/`` directory with detailed configuration parameters for reproducibility. BLACKBOX was run with 100 samples due to its computationally expensive candidate search.
