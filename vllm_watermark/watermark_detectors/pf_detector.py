"""Prefix-Free (PF) watermark detector."""

import torch
from scipy import special

from .base import WmDetector


class PFDetector(WmDetector):
    """Prefix-free watermark detector."""

    def __init__(
        self,
        tokenizer,
        ngram: int = 1,
        seed: int = 0,
        seeding: str = "hash",
        salt_key: int = 35317,
        **kwargs,
    ):
        super().__init__(tokenizer, ngram, seed, seeding, salt_key, **kwargs)

    def score_tok(self, ngram_tokens, token_id):
        """score_t = -log(1 - rt[token_id]]).

        The last line shifts the scores by token_id.
        ex: scores[0] = r_t[token_id]
            scores[1] = (r_t shifted of 1)[token_id]
            ...
        The score for each payload will be given by scores[payload]
        """
        seed = self.get_seed_rng(ngram_tokens)
        self.rng.manual_seed(seed)
        rs = torch.rand(self.vocab_size, generator=self.rng, device=self.device)  # n
        # Ensure rs is non-zero or assign a very small value to zero-valued elements
        rs[rs == 0] = 1e-4
        scores = -rs.log().roll(-token_id)
        return scores.cpu()

    def get_pvalue(self, score: float, ntoks: int, eps: float = 1e-200):
        """from cdf of a gamma distribution"""
        pvalue = special.gammaincc(ntoks, score)
        return max(pvalue, eps)
