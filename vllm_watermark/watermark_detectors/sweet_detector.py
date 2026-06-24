"""SWEET watermark detector.

Standard-mode detection: counts green tokens across ALL positions using
the same green-list construction as the generator. This works because
the generator biased green tokens at high-entropy positions, creating
a detectable excess of green tokens overall.

Entropy-weighted detection (requiring model access) is deferred to future work.
"""

import numpy as np
import torch
from scipy import special

from .base import WmDetector


class SWEETDetector(WmDetector):
    """Detect SWEET entropy-selective watermarks using green-list counting."""

    def __init__(
        self,
        tokenizer,
        ngram: int = 1,
        seed: int = 0,
        seeding: str = "hash",
        salt_key: int = 35317,
        gamma: float = 0.5,
        hash_key: int = 15485863,
        **kwargs,
    ):
        kwargs.pop('entropy_threshold', None)
        super().__init__(tokenizer, ngram, seed, seeding, salt_key, **kwargs)
        self.gamma = gamma
        self.hash_key = hash_key

    def _seed_rng_sweet(self, ngram_tokens: list[int]) -> None:
        """Seed RNG using multiplicative hash (matching generator)."""
        time_result = 1
        for tok in ngram_tokens:
            time_result *= tok

        vocab_size = int(self.vocab_size) if self.vocab_size else 128256
        prev_token = time_result % vocab_size
        self.rng.manual_seed(self.hash_key * prev_token)

    def score_tok(self, ngram_tokens: list[int], token_id: int):
        """Score a token by checking green-list membership."""
        vocab_size = int(self.vocab_size) if self.vocab_size else 128256
        self._seed_rng_sweet(ngram_tokens)
        greenlist_size = int(vocab_size * self.gamma)
        perm = torch.randperm(vocab_size, generator=self.rng, device=self.device)
        greenlist = perm[:greenlist_size]

        is_green = (greenlist == token_id).any().float().item()
        return torch.tensor([is_green])

    def get_pvalue(self, score: float, ntoks: int, eps: float = 1e-200):
        """Compute p-value using z-score approximation."""
        if ntoks < 1:
            return 1.0
        z = (score - self.gamma * ntoks) / np.sqrt(
            ntoks * self.gamma * (1 - self.gamma)
        )
        pvalue = 0.5 * special.erfc(z / np.sqrt(2))
        return max(float(pvalue), eps)
