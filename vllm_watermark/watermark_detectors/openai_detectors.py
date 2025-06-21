"""OpenAI-style watermark detectors."""

import numpy as np
import torch
from scipy import special

from .base import WmDetector


class OpenaiDetector(WmDetector):
    """Original OpenAI watermark detector using gamma distribution."""

    def __init__(
        self,
        tokenizer,
        ngram: int = 1,
        seed: int = 0,
        seeding: str = "hash",
        salt_key: int = 35317,
        payload: int = 0,
        **kwargs,
    ):
        super().__init__(tokenizer, ngram, seed, seeding, salt_key, **kwargs)
        self.payload = payload

    def score_tok(self, ngram_tokens, token_id):
        """
        score_t = -log(1 - rt[token_id]])
        The last line shifts the scores by token_id.
        ex: scores[0] = r_t[token_id]
            scores[1] = (r_t shifted of 1)[token_id]
            ...
        The score for each payload will be given by scores[payload]
        """
        seed = self.get_seed_rng(ngram_tokens)
        self.rng.manual_seed(seed)
        rs = torch.rand(self.vocab_size, generator=self.rng, device=self.device)  # n
        rs = rs.roll(-self.payload)  # Apply payload shift like in generator
        scores = -(1 - rs).log().roll(-token_id)
        return scores.cpu()

    def get_pvalue(self, score: float, ntoks: int, eps: float = 1e-200):
        """from cdf of a gamma distribution"""
        pvalue = special.gammaincc(ntoks, score)
        return max(pvalue, eps)


class OpenaiDetectorZ(WmDetector):
    """OpenAI detector with z-score approximation for p-value calculation."""

    def __init__(
        self,
        tokenizer,
        ngram: int = 1,
        seed: int = 0,
        seeding: str = "hash",
        salt_key: int = 35317,
        payload: int = 0,
        **kwargs,
    ):
        super().__init__(tokenizer, ngram, seed, seeding, salt_key, **kwargs)
        self.payload = payload

    def score_tok(self, ngram_tokens, token_id):
        """same as OpenaiDetector but using zscore"""
        seed = self.get_seed_rng(ngram_tokens)
        self.rng.manual_seed(seed)
        rs = torch.rand(self.vocab_size, generator=self.rng, device=self.device)  # n
        rs = rs.roll(-self.payload)  # Apply payload shift like in generator
        scores = -(1 - rs).log().roll(-token_id)
        return scores.cpu()

    def get_pvalue(self, score: float, ntoks: int, eps: float = 1e-200):
        """from cdf of a normal distribution"""
        mu0 = 1
        sigma0 = np.pi / np.sqrt(6)
        zscore = (score / ntoks - mu0) / (sigma0 / np.sqrt(ntoks))
        pvalue = 0.5 * special.erfc(zscore / np.sqrt(2))
        return max(pvalue, eps)
