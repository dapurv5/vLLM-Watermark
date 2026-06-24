"""Alignment Resampling (Best-of-N) for watermarked LLMs.

Reference: AlignMark — alignment-preserving watermark resampling.

Wraps any watermarked LLM and selects the highest-reward candidate
from N watermarked outputs per prompt. The watermark remains detectable
because all candidates are watermarked; the reward model just picks
the one with the best alignment quality.

Usage:
    wm_llm = WatermarkedLLMs.create(model=llm, algo=..., ...)
    scorer = load_reward_scorer()  # 86M DeBERTa reward model
    aligned_llm = AlignmentResampledLLM(wm_llm, scorer, n_samples=4)
    outputs = aligned_llm.generate(prompts, sampling_params)
"""

from loguru import logger


class AlignmentResampledLLM:
    """Watermarked LLM with Best-of-N alignment resampling.

    Generates N watermarked candidates per prompt using the underlying
    watermarked LLM, scores each with a reward model, and returns the
    highest-scoring one.
    """

    def __init__(self, watermarked_llm, scorer, n_samples: int = 4):
        """
        Args:
            watermarked_llm: Any watermarked LLM (from WatermarkedLLMs.create())
            scorer: Callable (prompt: str, texts: list[str]) -> list[float]
                    Returns a reward score for each text (higher is better).
            n_samples: Number of watermarked candidates to generate per prompt.
        """
        self.watermarked_llm = watermarked_llm
        self.scorer = scorer
        self.n_samples = n_samples

    def generate(self, prompts, sampling_params=None, **kwargs):
        from vllm import SamplingParams

        if sampling_params is None:
            sampling_params = SamplingParams()

        if isinstance(prompts, str):
            prompts = [prompts]

        n = self.n_samples

        expanded_prompts = []
        for prompt in prompts:
            expanded_prompts.extend([prompt] * n)

        all_outputs = self.watermarked_llm.generate(
            expanded_prompts, sampling_params, **kwargs
        )

        selected_outputs = []
        for prompt_idx, prompt in enumerate(prompts):
            start = prompt_idx * n
            end = start + n
            candidates = all_outputs[start:end]

            texts = [c.outputs[0].text for c in candidates]
            scores = self.scorer(prompt, texts)

            best_idx = max(range(len(scores)), key=lambda i: scores[i])
            selected_outputs.append(candidates[best_idx])

            logger.debug(
                f"Prompt {prompt_idx}: scores={[f'{s:.3f}' for s in scores]}, "
                f"selected idx={best_idx}"
            )

        return selected_outputs

    def get_tokenizer(self):
        return self.watermarked_llm.get_tokenizer()

    def __getattr__(self, name):
        return getattr(self.watermarked_llm, name)


def load_reward_scorer(
    model_name: str = "OpenAssistant/reward-model-deberta-v3-base",
    device="auto",
):
    """Load a classifier-style reward model and return a scorer callable.

    Default: OpenAssistant/reward-model-deberta-v3-base (86M params).
    Takes (question, answer) as input, returns a scalar reward score
    in a single forward pass — no generation or parsing needed.

    Args:
        model_name: HuggingFace model name for a reward model.
        device: Device for the reward model ("auto", "cuda", "cpu").

    Returns:
        Callable (prompt: str, texts: list[str]) -> list[float]
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        device_map=device,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )

    def scorer(prompt: str, texts: list[str]) -> list[float]:
        scores = []
        for text in texts:
            inputs = tokenizer(
                prompt, text, return_tensors="pt", truncation=True, max_length=512,
            ).to(model.device)
            with torch.no_grad():
                output = model(**inputs)
                scores.append(output.logits[0].float().item())
        return scores

    return scorer
