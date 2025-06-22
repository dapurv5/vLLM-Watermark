Examples
========

This section provides comprehensive examples demonstrating various use cases and advanced patterns for vLLM-Watermark.

Basic Examples
--------------

OpenAI Watermarking
~~~~~~~~~~~~~~~~~~~

Complete example with OpenAI watermarking and detection:

.. literalinclude:: ../examples/example_openai.py
   :language: python
   :caption: OpenAI Watermarking Example

Maryland Watermarking
~~~~~~~~~~~~~~~~~~~~~

Example using Maryland watermarking algorithm:

.. literalinclude:: ../examples/example_maryland.py
   :language: python
   :caption: Maryland Watermarking Example

Advanced Examples
-----------------

Multi-Algorithm Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~

Compare different watermarking algorithms:

.. code-block:: python

   from vllm import LLM, SamplingParams
   from vllm_watermark.core import WatermarkedLLMs, WatermarkingAlgorithm
   from vllm_watermark.watermark_detectors import WatermarkDetectors, DetectionAlgorithm

   llm = LLM(model="meta-llama/Llama-3.2-1B")
   prompt = "Write a short story about a detective"

   # Test different algorithms
   algorithms = [
       (WatermarkingAlgorithm.OPENAI, DetectionAlgorithm.OPENAI_Z),
       (WatermarkingAlgorithm.MARYLAND, DetectionAlgorithm.MARYLAND),
       (WatermarkingAlgorithm.PF, DetectionAlgorithm.PF)
   ]

   results = {}
   for gen_algo, det_algo in algorithms:
       print(f"\n=== Testing {gen_algo.value} ===")

       # Generate watermarked text
       wm_llm = WatermarkedLLMs.create(
           llm, algo=gen_algo, seed=42, ngram=2
       )
       outputs = wm_llm.generate([prompt], SamplingParams(max_tokens=100))
       text = outputs[0].outputs[0].text

       # Detect watermark
       detector = WatermarkDetectors.create(
           algo=det_algo, model=llm, seed=42, threshold=0.05
       )
       result = detector.detect(text)

       results[gen_algo.value] = {
           'text': text,
           'is_watermarked': result['is_watermarked'],
           'pvalue': result['pvalue'],
           'score': result.get('score', 0)
       }

       print(f"Text: {text[:100]}...")
       print(f"Watermarked: {result['is_watermarked']}")
       print(f"P-value: {result['pvalue']:.6f}")

Parameter Optimization
~~~~~~~~~~~~~~~~~~~~~~

Find optimal parameters for your use case:

.. code-block:: python

   import numpy as np
   from vllm import LLM, SamplingParams
   from vllm_watermark.core import WatermarkedLLMs, WatermarkingAlgorithm
   from vllm_watermark.watermark_detectors import WatermarkDetectors, DetectionAlgorithm

   llm = LLM(model="meta-llama/Llama-3.2-1B")
   prompt = "Explain quantum computing in simple terms"

   # Test different gamma values
   gamma_values = np.linspace(0.1, 1.0, 10)
   results = []

   for gamma in gamma_values:
       # Generate with current gamma
       wm_llm = WatermarkedLLMs.create(
           llm,
           algo=WatermarkingAlgorithm.OPENAI,
           seed=42,
           ngram=2,
           gamma=gamma
       )

       outputs = wm_llm.generate([prompt], SamplingParams(max_tokens=100))
       text = outputs[0].outputs[0].text

       # Detect watermark
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
           'score': result['score'],
           'text_length': len(text)
       })

   # Find optimal gamma
   best_result = min(results, key=lambda x: x['pvalue'])
   print(f"Optimal gamma: {best_result['gamma']:.2f}")
   print(f"Best p-value: {best_result['pvalue']:.6f}")

Batch Processing
~~~~~~~~~~~~~~~~

Process multiple texts efficiently:

.. code-block:: python

   from vllm import LLM, SamplingParams
   from vllm_watermark.core import WatermarkedLLMs, WatermarkingAlgorithm
   from vllm_watermark.watermark_detectors import WatermarkDetectors, DetectionAlgorithm
   from concurrent.futures import ThreadPoolExecutor
   import time

   llm = LLM(model="meta-llama/Llama-3.2-1B")

   # Create watermarked LLM once
   wm_llm = WatermarkedLLMs.create(
       llm, algo=WatermarkingAlgorithm.OPENAI, seed=42, ngram=2
   )

   # Create detector once
   detector = WatermarkDetectors.create(
       algo=DetectionAlgorithm.OPENAI_Z,
       model=llm,
       ngram=2,
       seed=42,
       threshold=0.05
   )

   # Batch of prompts
   prompts = [
       "Write a poem about AI",
       "Explain machine learning",
       "Tell a story about space",
       "Describe quantum physics",
       "Write about climate change"
   ]

   # Generate all texts
   sampling_params = SamplingParams(temperature=0.8, max_tokens=100)
   outputs = wm_llm.generate(prompts, sampling_params)
   texts = [output.outputs[0].text for output in outputs]

   # Detect watermarks in parallel
   def detect_watermark(text):
       try:
           result = detector.detect(text)
           return {
               'text': text,
               'is_watermarked': result['is_watermarked'],
               'pvalue': result['pvalue'],
               'score': result['score']
           }
       except Exception as e:
           return {
               'text': text,
               'error': str(e),
               'is_watermarked': False
           }

   start_time = time.time()
   with ThreadPoolExecutor(max_workers=4) as executor:
       results = list(executor.map(detect_watermark, texts))
   end_time = time.time()

   print(f"Processed {len(texts)} texts in {end_time - start_time:.2f} seconds")

   for i, result in enumerate(results):
       print(f"Text {i+1}: {'Watermarked' if result['is_watermarked'] else 'Not watermarked'}")
       if 'pvalue' in result:
           print(f"  P-value: {result['pvalue']:.6f}")

Production Use Cases
--------------------

API Server
~~~~~~~~~~

Example of integrating watermarking into a FastAPI server:

.. code-block:: python

   from fastapi import FastAPI, HTTPException
   from pydantic import BaseModel
   from vllm import LLM, SamplingParams
   from vllm_watermark.core import WatermarkedLLMs, WatermarkingAlgorithm
   from vllm_watermark.watermark_detectors import WatermarkDetectors, DetectionAlgorithm
   import logging

   app = FastAPI(title="vLLM-Watermark API")
   logger = logging.getLogger(__name__)

   # Global model instances
   llm = None
   wm_llm = None
   detector = None

   class GenerateRequest(BaseModel):
       prompt: str
       max_tokens: int = 100
       temperature: float = 0.8
       watermark: bool = True
       seed: int = 42

   class DetectRequest(BaseModel):
       text: str
       threshold: float = 0.05

   class GenerateResponse(BaseModel):
       text: str
       is_watermarked: bool
       watermark_info: dict

   class DetectResponse(BaseModel):
       is_watermarked: bool
       pvalue: float
       score: float
       confidence: float

   @app.on_event("startup")
   async def startup_event():
       global llm, wm_llm, detector

       try:
           llm = LLM(model="meta-llama/Llama-3.2-1B")
           wm_llm = WatermarkedLLMs.create(
               llm,
               algo=WatermarkingAlgorithm.OPENAI,
               seed=42,
               ngram=2
           )
           detector = WatermarkDetectors.create(
               algo=DetectionAlgorithm.OPENAI_Z,
               model=llm,
               ngram=2,
               seed=42,
               threshold=0.05
           )
           logger.info("Models loaded successfully")
       except Exception as e:
           logger.error(f"Failed to load models: {e}")
           raise

   @app.post("/generate", response_model=GenerateResponse)
   async def generate_text(request: GenerateRequest):
       try:
           sampling_params = SamplingParams(
               temperature=request.temperature,
               max_tokens=request.max_tokens
           )

           if request.watermark:
               outputs = wm_llm.generate([request.prompt], sampling_params)
               text = outputs[0].outputs[0].text
               is_watermarked = True
           else:
               outputs = llm.generate([request.prompt], sampling_params)
               text = outputs[0].outputs[0].text
               is_watermarked = False

           return GenerateResponse(
               text=text,
               is_watermarked=is_watermarked,
               watermark_info={"seed": request.seed}
           )
       except Exception as e:
           logger.error(f"Generation failed: {e}")
           raise HTTPException(status_code=500, detail=str(e))

   @app.post("/detect", response_model=DetectResponse)
   async def detect_watermark(request: DetectRequest):
       try:
           result = detector.detect(request.text)
           return DetectResponse(
               is_watermarked=result['is_watermarked'],
               pvalue=result['pvalue'],
               score=result['score'],
               confidence=result.get('confidence', 0.0)
           )
       except Exception as e:
           logger.error(f"Detection failed: {e}")
           raise HTTPException(status_code=500, detail=str(e))

   @app.get("/health")
   async def health_check():
       return {"status": "healthy", "models_loaded": llm is not None}

   if __name__ == "__main__":
       import uvicorn
       uvicorn.run(app, host="0.0.0.0", port=8000)

Content Moderation
~~~~~~~~~~~~~~~~~~

Use watermarking for content moderation and attribution:

.. code-block:: python

   from vllm import LLM, SamplingParams
   from vllm_watermark.core import WatermarkedLLMs, WatermarkingAlgorithm
   from vllm_watermark.watermark_detectors import WatermarkDetectors, DetectionAlgorithm
   import json
   from datetime import datetime

   class ContentModerator:
       def __init__(self, model_name="meta-llama/Llama-3.2-1B"):
           self.llm = LLM(model=model_name)
           self.wm_llm = WatermarkedLLMs.create(
               self.llm,
               algo=WatermarkingAlgorithm.OPENAI,
               seed=42,
               ngram=2
           )
           self.detector = WatermarkDetectors.create(
               algo=DetectionAlgorithm.OPENAI_Z,
               model=self.llm,
               ngram=2,
               seed=42,
               threshold=0.01  # Conservative threshold
           )
           self.generation_log = []

       def generate_content(self, prompt, user_id, content_type="general"):
           """Generate watermarked content with logging."""
           sampling_params = SamplingParams(temperature=0.8, max_tokens=200)
           outputs = self.wm_llm.generate([prompt], sampling_params)
           text = outputs[0].outputs[0].text

           # Log generation
           log_entry = {
               'timestamp': datetime.now().isoformat(),
               'user_id': user_id,
               'content_type': content_type,
               'prompt': prompt,
               'generated_text': text,
               'watermark_seed': 42,
               'model': self.llm.model_name
           }
           self.generation_log.append(log_entry)

           return text

       def verify_content(self, text, user_id=None):
           """Verify if content was generated by this system."""
           try:
               result = self.detector.detect(text)

               verification = {
                   'is_watermarked': result['is_watermarked'],
                   'pvalue': result['pvalue'],
                   'confidence': result.get('confidence', 0.0),
                   'timestamp': datetime.now().isoformat()
               }

               # Check generation log for attribution
               if user_id:
                   for entry in reversed(self.generation_log):
                       if entry['user_id'] == user_id and entry['generated_text'] == text:
                           verification['attribution'] = {
                               'user_id': user_id,
                               'generation_time': entry['timestamp'],
                               'content_type': entry['content_type']
                           }
                           break

               return verification
           except Exception as e:
               return {
                   'error': str(e),
                   'is_watermarked': False
               }

       def export_log(self, filename="generation_log.json"):
           """Export generation log for audit purposes."""
           with open(filename, 'w') as f:
               json.dump(self.generation_log, f, indent=2)

   # Usage example
   moderator = ContentModerator()

   # Generate content
   text = moderator.generate_content(
       "Write a news article about AI safety",
       user_id="user123",
       content_type="news"
   )

   # Verify content
   verification = moderator.verify_content(text, user_id="user123")
   print(f"Content verification: {verification}")

   # Export log
   moderator.export_log()

Research Applications
---------------------

A/B Testing Framework
~~~~~~~~~~~~~~~~~~~~~

Compare watermarking algorithms in research settings:

.. code-block:: python

   import numpy as np
   import pandas as pd
   from vllm import LLM, SamplingParams
   from vllm_watermark.core import WatermarkedLLMs, WatermarkingAlgorithm
   from vllm_watermark.watermark_detectors import WatermarkDetectors, DetectionAlgorithm
   from scipy import stats
   import matplotlib.pyplot as plt

   class WatermarkABTest:
       def __init__(self, model_name="meta-llama/Llama-3.2-1B"):
           self.llm = LLM(model=model_name)
           self.results = []

       def run_experiment(self, prompts, algorithms, n_runs=10):
           """Run A/B test comparing different algorithms."""

           for algo_name, (gen_algo, det_algo) in algorithms.items():
               print(f"Testing {algo_name}...")

               # Create watermarked LLM
               wm_llm = WatermarkedLLMs.create(
                   self.llm, algo=gen_algo, seed=42, ngram=2
               )

               # Create detector
               detector = WatermarkDetectors.create(
                   algo=det_algo, model=self.llm, seed=42, threshold=0.05
               )

               for run in range(n_runs):
                   for prompt in prompts:
                       # Generate text
                       outputs = wm_llm.generate(
                           [prompt],
                           SamplingParams(temperature=0.8, max_tokens=100)
                       )
                       text = outputs[0].outputs[0].text

                       # Detect watermark
                       result = detector.detect(text)

                       # Store results
                       self.results.append({
                           'algorithm': algo_name,
                           'run': run,
                           'prompt': prompt,
                           'text': text,
                           'is_watermarked': result['is_watermarked'],
                           'pvalue': result['pvalue'],
                           'score': result['score'],
                           'text_length': len(text)
                       })

       def analyze_results(self):
           """Analyze experimental results."""
           df = pd.DataFrame(self.results)

           # Detection rate by algorithm
           detection_rates = df.groupby('algorithm')['is_watermarked'].mean()
           print("Detection Rates:")
           print(detection_rates)

           # P-value distributions
           pvalue_stats = df.groupby('algorithm')['pvalue'].agg(['mean', 'std', 'min', 'max'])
           print("\nP-value Statistics:")
           print(pvalue_stats)

           # Statistical significance tests
           algorithms = df['algorithm'].unique()
           for i, algo1 in enumerate(algorithms):
               for algo2 in algorithms[i+1:]:
                   pvals1 = df[df['algorithm'] == algo1]['pvalue']
                   pvals2 = df[df['algorithm'] == algo2]['pvalue']

                   # Mann-Whitney U test
                   stat, pval = stats.mannwhitneyu(pvals1, pvals2, alternative='two-sided')
                   print(f"\n{algo1} vs {algo2}: U={stat:.2f}, p={pval:.4f}")

           return df

       def plot_results(self, df):
           """Create visualization of results."""
           fig, axes = plt.subplots(2, 2, figsize=(12, 10))

           # Detection rate
           detection_rates = df.groupby('algorithm')['is_watermarked'].mean()
           detection_rates.plot(kind='bar', ax=axes[0,0], title='Detection Rate')
           axes[0,0].set_ylabel('Detection Rate')

           # P-value distribution
           for algo in df['algorithm'].unique():
               pvals = df[df['algorithm'] == algo]['pvalue']
               axes[0,1].hist(pvals, alpha=0.7, label=algo, bins=20)
           axes[0,1].set_title('P-value Distribution')
           axes[0,1].set_xlabel('P-value')
           axes[0,1].legend()

           # Score distribution
           for algo in df['algorithm'].unique():
               scores = df[df['algorithm'] == algo]['score']
               axes[1,0].hist(scores, alpha=0.7, label=algo, bins=20)
           axes[1,0].set_title('Score Distribution')
           axes[1,0].set_xlabel('Score')
           axes[1,0].legend()

           # Text length vs p-value
           for algo in df['algorithm'].unique():
               subset = df[df['algorithm'] == algo]
               axes[1,1].scatter(subset['text_length'], subset['pvalue'],
                               alpha=0.6, label=algo)
           axes[1,1].set_title('Text Length vs P-value')
           axes[1,1].set_xlabel('Text Length')
           axes[1,1].set_ylabel('P-value')
           axes[1,1].legend()

           plt.tight_layout()
           plt.savefig('watermark_ab_test_results.png', dpi=300, bbox_inches='tight')
           plt.show()

   # Run A/B test
   ab_test = WatermarkABTest()

   algorithms = {
       'OpenAI': (WatermarkingAlgorithm.OPENAI, DetectionAlgorithm.OPENAI_Z),
       'Maryland': (WatermarkingAlgorithm.MARYLAND, DetectionAlgorithm.MARYLAND),
       'PF': (WatermarkingAlgorithm.PF, DetectionAlgorithm.PF)
   }

   prompts = [
       "Write a short story about a robot",
       "Explain machine learning",
       "Describe quantum physics",
       "Write a poem about nature"
   ]

   ab_test.run_experiment(prompts, algorithms, n_runs=5)
   results_df = ab_test.analyze_results()
   ab_test.plot_results(results_df)

Error Handling Examples
-----------------------

Robust Error Handling
~~~~~~~~~~~~~~~~~~~~~

Example with comprehensive error handling:

.. code-block:: python

   import logging
   from vllm import LLM, SamplingParams
   from vllm_watermark.core import WatermarkedLLMs, WatermarkingAlgorithm
   from vllm_watermark.watermark_detectors import WatermarkDetectors, DetectionAlgorithm

   logging.basicConfig(level=logging.INFO)
   logger = logging.getLogger(__name__)

   class RobustWatermarker:
       def __init__(self, model_name="meta-llama/Llama-3.2-1B"):
           self.model_name = model_name
           self.llm = None
           self.wm_llm = None
           self.detector = None
           self.initialize_models()

       def initialize_models(self):
           """Initialize models with error handling."""
           try:
               logger.info(f"Loading model: {self.model_name}")
               self.llm = LLM(model=self.model_name)

               logger.info("Creating watermarked LLM")
               self.wm_llm = WatermarkedLLMs.create(
                   self.llm,
                   algo=WatermarkingAlgorithm.OPENAI,
                   seed=42,
                   ngram=2,
                   debug=True
               )

               logger.info("Creating detector")
               self.detector = WatermarkDetectors.create(
                   algo=DetectionAlgorithm.OPENAI_Z,
                   model=self.llm,
                   ngram=2,
                   seed=42,
                   threshold=0.05
               )

               logger.info("Models initialized successfully")
           except Exception as e:
               logger.error(f"Failed to initialize models: {e}")
               raise

       def generate_safe(self, prompt, max_retries=3):
           """Generate text with retry logic."""
           for attempt in range(max_retries):
               try:
                   sampling_params = SamplingParams(
                       temperature=0.8,
                       max_tokens=100
                   )

                   outputs = self.wm_llm.generate([prompt], sampling_params)
                   text = outputs[0].outputs[0].text

                   logger.info(f"Generated text successfully (attempt {attempt + 1})")
                   return text

               except Exception as e:
                   logger.warning(f"Generation attempt {attempt + 1} failed: {e}")
                   if attempt == max_retries - 1:
                       logger.error("All generation attempts failed")
                       raise
                   continue

       def detect_safe(self, text, max_retries=3):
           """Detect watermark with retry logic."""
           for attempt in range(max_retries):
               try:
                   result = self.detector.detect(text)
                   logger.info(f"Detection successful (attempt {attempt + 1})")
                   return result

               except Exception as e:
                   logger.warning(f"Detection attempt {attempt + 1} failed: {e}")
                   if attempt == max_retries - 1:
                       logger.error("All detection attempts failed")
                       return {
                           'is_watermarked': False,
                           'error': str(e),
                           'pvalue': 1.0,
                           'score': 0.0
                       }
                   continue

       def process_batch_safe(self, prompts):
           """Process a batch of prompts with error handling."""
           results = []

           for i, prompt in enumerate(prompts):
               try:
                   logger.info(f"Processing prompt {i+1}/{len(prompts)}")

                   # Generate text
                   text = self.generate_safe(prompt)

                   # Detect watermark
                   detection = self.detect_safe(text)

                   results.append({
                       'prompt': prompt,
                       'text': text,
                       'detection': detection,
                       'success': True
                   })

               except Exception as e:
                   logger.error(f"Failed to process prompt {i+1}: {e}")
                   results.append({
                       'prompt': prompt,
                       'text': None,
                       'detection': None,
                       'success': False,
                       'error': str(e)
                   })

           return results

   # Usage
   try:
       watermarker = RobustWatermarker()

       prompts = [
           "Write a short story",
           "Explain AI",
           "Tell a joke"
       ]

       results = watermarker.process_batch_safe(prompts)

       for result in results:
           if result['success']:
               print(f"Success: {result['text'][:50]}...")
               print(f"Watermarked: {result['detection']['is_watermarked']}")
           else:
               print(f"Failed: {result['error']}")

   except Exception as e:
       logger.error(f"Application failed: {e}")

.. note::
   These examples demonstrate various use cases and best practices.
   Adapt them to your specific requirements and always test thoroughly in your environment.