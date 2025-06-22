"""Maryland-style watermark detectors."""

import numpy as np
import torch
from scipy import special

from .base import WmDetector


class MarylandDetector(WmDetector):
    """Maryland detector using binomial distribution for p-value calculation."""

    def __init__(
        self,
        tokenizer,
        ngram: int = 1,
        seed: int = 0,
        seeding: str = "hash",
        salt_key: int = 35317,
        gamma: float = 0.5,
        delta: float = 1.0,
        **kwargs,
    ):
        super().__init__(tokenizer, ngram, seed, seeding, salt_key, **kwargs)
        self.gamma = gamma
        self.delta = delta

    def score_tok(self, ngram_tokens, token_id):
        """score_t = 1 if token_id in greenlist else 0.

        The last line shifts the scores by token_id.
        ex: scores[0] = 1 if token_id in greenlist else 0
            scores[1] = 1 if token_id in (greenlist shifted of 1) else 0
            ...
        The score for each payload will be given by scores[payload]
        """
        seed = self.get_seed_rng(ngram_tokens)
        self.rng.manual_seed(seed)
        scores = torch.zeros(self.vocab_size)
        vocab_permutation = torch.randperm(
            self.vocab_size, generator=self.rng, device=self.device
        )
        greenlist = vocab_permutation[
            : int(self.gamma * self.vocab_size)
        ]  # gamma * n toks in the greenlist
        scores[greenlist] = 1
        return scores.roll(-token_id)

    def get_pvalue(self, score: int, ntoks: int, eps: float = 1e-200):
        """from cdf of a binomial distribution"""
        pvalue = special.betainc(score, 1 + ntoks - score, self.gamma)
        return max(pvalue, eps)


class MarylandDetectorZ(WmDetector):
    """Maryland detector using z-score approximation for p-value calculation."""

    def __init__(
        self,
        tokenizer,
        ngram: int = 1,
        seed: int = 0,
        seeding: str = "hash",
        salt_key: int = 35317,
        gamma: float = 0.5,
        delta: float = 1.0,
        **kwargs,
    ):
        super().__init__(tokenizer, ngram, seed, seeding, salt_key, **kwargs)
        self.gamma = gamma
        self.delta = delta

    def score_tok(self, ngram_tokens, token_id):
        """same as MarylandDetector but using zscore"""
        seed = self.get_seed_rng(ngram_tokens)
        self.rng.manual_seed(seed)
        scores = torch.zeros(self.vocab_size)
        vocab_permutation = torch.randperm(
            self.vocab_size, generator=self.rng, device=self.device
        )
        greenlist = vocab_permutation[: int(self.gamma * self.vocab_size)]  # gamma * n
        scores[greenlist] = 1
        return scores.roll(-token_id)

    def get_pvalue(self, score: int, ntoks: int, eps: float = 1e-200):
        """from cdf of a normal distribution"""
        zscore = (score - self.gamma * ntoks) / np.sqrt(
            self.gamma * (1 - self.gamma) * ntoks
        )
        pvalue = 0.5 * special.erfc(zscore / np.sqrt(2))
        return max(pvalue, eps)
