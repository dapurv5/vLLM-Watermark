import torch
from vllm.sampling_params import SamplingType


class GumbelSampler:
    def __init__(self, gumbel_generator):
        self.gumbel_generator = gumbel_generator

    def sample(
        self,
        logits: torch.Tensor,  # (bsz, vocab_size)
        ngram_tokens: torch.LongTensor,  # (bsz, ngram)
        temperature: float = 0.8,
        top_p: float = 0.95,
        sampling_type: str = "random",
    ) -> torch.LongTensor:
        if sampling_type in (SamplingType.RANDOM, SamplingType.RANDOM_SEED):
            return self.gumbel_generator.sample_next(
                logits, ngram_tokens, temperature, top_p
            )
        else:
            # fallback to greedy
            return torch.argmax(logits, dim=-1)
