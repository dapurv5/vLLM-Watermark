"""SWEET watermark generator.

Implements entropy-selective watermarking from:
Lee et al., "Who Wrote this Code? Watermarking for Code Generation" (2023)

Only applies green-list logit bias at high-entropy positions where the model
is uncertain. Low-entropy positions (confident predictions) are left
un-watermarked, preserving text quality where bias would be pointless.
"""

from typing import cast

import torch

from .base import WmGenerator


class SWEETGenerator(WmGenerator):
    """Generate watermarked text with entropy-selective green-list biasing."""

    def __init__(
        self,
        *args,
        gamma: float = 0.5,
        delta: float = 2.0,
        hash_key: int = 15485863,
        entropy_threshold: float = 3.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.delta = delta
        self.hash_key = hash_key
        self.entropy_threshold = entropy_threshold

    def _seed_rng_sweet(self, ngram_tokens: torch.Tensor) -> None:
        """Seed RNG using multiplicative hash of context tokens (matching MarkLLM)."""
        time_result = 1
        for i in range(ngram_tokens.shape[-1]):
            time_result *= ngram_tokens[i].item()

        vocab_size = getattr(self, '_vocab_size_cached', None)
        if vocab_size is None:
            from ..core import WatermarkUtils
            vocab_size = WatermarkUtils.infer_vocab_size(self.model, self.tokenizer)
            self._vocab_size_cached = vocab_size

        prev_token = time_result % vocab_size
        self.rng.manual_seed(self.hash_key * prev_token)

    def _get_greenlist(self, ngram_tokens: torch.Tensor, vocab_size: int) -> torch.LongTensor:
        """Generate green list for the given context."""
        self._seed_rng_sweet(ngram_tokens)
        greenlist_size = int(vocab_size * self.gamma)
        perm = torch.randperm(vocab_size, generator=self.rng, device=self.device)
        return perm[:greenlist_size]

    def sample_next(
        self,
        logits: torch.FloatTensor,
        ngram_tokens: torch.LongTensor,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> torch.LongTensor:
        batch_size, vocab_size = logits.shape
        biased_logits = logits.clone()

        probs_for_entropy = torch.softmax(logits, dim=-1)
        entropy = -(probs_for_entropy * torch.log(probs_for_entropy + 1e-10)).sum(dim=-1)

        for i in range(batch_size):
            if entropy[i].item() <= self.entropy_threshold:
                continue

            greenlist = self._get_greenlist(ngram_tokens[i], vocab_size)
            biased_logits[i, greenlist] += self.delta

        if temperature > 0:
            probs = torch.softmax(biased_logits / temperature, dim=-1)
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > top_p
            probs_sort[mask] = 0.0
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
            next_token = torch.multinomial(
                probs_sort, num_samples=1, generator=self.rng
            )
            next_token = torch.gather(probs_idx, -1, next_token)
        else:
            next_token = torch.argmax(biased_logits, dim=-1)

        next_token = next_token.reshape(-1).to(dtype=torch.long)
        return cast(torch.LongTensor, next_token)
