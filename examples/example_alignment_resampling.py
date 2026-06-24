import os
import sys

os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vllm import LLM, SamplingParams

from vllm_watermark.alignment_resampling import AlignmentResampledLLM
from vllm_watermark.core import (
    DetectionAlgorithm,
    WatermarkedLLMs,
    WatermarkingAlgorithm,
)
from vllm_watermark.watermark_detectors import WatermarkDetectors

# Load the vLLM model
llm = LLM(
    model="meta-llama/Llama-3.2-1B",
    enforce_eager=True,
    max_model_len=1024,
)

# Step 1: Create a watermarked LLM (any algorithm works)
wm_llm = WatermarkedLLMs.create(
    model=llm,
    algo=WatermarkingAlgorithm.MARYLAND,
    seed=42,
    ngram=2,
    gamma=0.5,
    delta=1.0,
)

# Step 2: Load a reward scorer
# Default: OpenAssistant/reward-model-deberta-v3-base (86M params, single forward pass)
from vllm_watermark.alignment_resampling import load_reward_scorer

scorer = load_reward_scorer()

# Alternatives:
# - Larger reward model: load_reward_scorer("RLHFlow/ArmoRM-Llama3-8B-v0.1")
# - Custom: any callable(prompt: str, texts: list[str]) -> list[float]

# Step 3: Wrap with alignment resampling (Best-of-N)
aligned_llm = AlignmentResampledLLM(
    watermarked_llm=wm_llm,
    scorer=scorer,
    n_samples=4,
)

# Create detector (same params as the watermark — detection is unchanged)
detector = WatermarkDetectors.create(
    algo=DetectionAlgorithm.MARYLAND_Z,
    model=llm,
    ngram=2,
    seed=42,
    gamma=0.5,
    delta=1.0,
    threshold=0.05,
)

# Example prompts
prompts = [
    "Cluster comprises IBM's Opteron-based eServer 325 server and systems management"
    + " software and storage devices that can run Linux and Windows operating systems",
    "The research team published a groundbreaking paper on machine learning techniques"
    + " for natural language processing and text generation",
]

# Sampling parameters — temperature > 0 gives diverse candidates
sampling_params = SamplingParams(temperature=1.0, top_p=0.95, max_tokens=128)

# Generate with alignment resampling
outputs = aligned_llm.generate(prompts, sampling_params)

print("=== ALIGNMENT RESAMPLING (Best-of-N) EXAMPLE ===")
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}")
    print(f"Generated text: {generated_text!r}\n")

    detection_result = detector.detect(generated_text)
    print("Maryland Detector Results (watermark preserved after BoN selection):")
    print(f"  Is watermarked: {detection_result['is_watermarked']}")
    print(f"  Score (green fraction): {detection_result['score']:.4f}")
    print(f"  P-value: {detection_result['pvalue']:.6f}")
    print("-" * 50)

print("\n=== EXPLANATION ===")
print("Alignment Resampling generates N watermarked candidates per prompt,")
print("scores each with a reward model, and returns the best one.")
print("All candidates carry the watermark, so detection still works.")
print("This fixes alignment degradation caused by watermarking:")
print("  - Watermarking can bias toward green-list tokens, hurting quality")
print("  - BoN selection picks the candidate with the best reward score")
print("  - Theoretical guarantee: alignment improves as sqrt(log(N))")
print("  - Works with ANY watermark algorithm (Maryland, OpenAI, DIP, etc.)")
