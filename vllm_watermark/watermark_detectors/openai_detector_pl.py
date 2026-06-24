"""Power Law detector for Gumbel watermarking.

Implements Algorithm 3 from Lattimore (2026) 'Refined Detection for Gumbel
Watermarking'. Replaces Aaronson's exponential statistic with a truncated
power law that is near-optimal in a problem-dependent sense.

Reference: https://arxiv.org/abs/2603.30017
"""

from functools import lru_cache
from typing import List

import numpy as np
import torch

from .base import WmDetector


class OpenaiDetectorPL(WmDetector):
    """Power Law detector for Gumbel/OpenAI watermarking (Lattimore 2026).

    Uses the statistic S(u) = min(1/sqrt(eps), 1/sqrt(1-u)) - mu
    where eps = log(1/delta)/n and mu = 2 - sqrt(eps).
    The critical threshold tau_star is calibrated via Monte Carlo so that
    the false positive rate equals exactly delta.
    """

    def __init__(
        self,
        tokenizer,
        ngram: int = 1,
        seed: int = 0,
        seeding: str = "hash",
        salt_key: int = 35317,
        payload: int = 0,
        n_mc_samples: int = 200_000,
        mc_seed: int = 2026,
        **kwargs,
    ):
        super().__init__(tokenizer, ngram, seed, seeding, salt_key, **kwargs)
        self.payload = payload
        self.n_mc_samples = n_mc_samples
        self.mc_rng = np.random.default_rng(mc_seed)

    @staticmethod
    def power_law_score(u: float, epsilon: float) -> float:
        """S(u) = min(1/sqrt(eps), 1/sqrt(1-u)) - mu (Eq. 2)."""
        mu = 2.0 - np.sqrt(epsilon)
        return min(1.0 / np.sqrt(epsilon), 1.0 / np.sqrt(1.0 - u)) - mu

    @staticmethod
    def power_law_score_vectorized(
        u: np.ndarray, epsilon: float
    ) -> np.ndarray:
        """Vectorized S(u) for Monte Carlo simulation."""
        mu = 2.0 - np.sqrt(epsilon)
        return np.minimum(1.0 / np.sqrt(epsilon), 1.0 / np.sqrt(1.0 - u)) - mu

    def _mc_null_sums(self, n: int, epsilon: float) -> np.ndarray:
        """Sample sum(S_t) under H_0 (V_t iid Uniform[0,1])."""
        U = self.mc_rng.uniform(0, 1, size=(self.n_mc_samples, n))
        scores = self.power_law_score_vectorized(U, epsilon)
        return scores.sum(axis=1)

    def _get_vt(self, ngram_tokens: List[int], token_id: int) -> float:
        """Reconstruct V_t = U_{t, A_t} from the secret key."""
        seed = self.get_seed_rng(ngram_tokens)
        self.rng.manual_seed(seed)
        vocab_size = int(self.vocab_size) if self.vocab_size is not None else 0
        rs = torch.rand((vocab_size,), generator=self.rng, device=self.device)
        rs = rs.roll(-self.payload)
        return rs[token_id].item()

    def detect(self, text: str) -> dict:
        """Detect watermark using the power law statistic (Algorithm 3)."""
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)

        start_pos = self.ngram + 1
        if len(token_ids) <= start_pos:
            return {"is_watermarked": False, "score": 0.0, "pvalue": 1.0}

        vt_values = []
        for cur_pos in range(start_pos, len(token_ids)):
            ngram_tokens = token_ids[cur_pos - self.ngram : cur_pos]
            vt = self._get_vt(ngram_tokens, token_ids[cur_pos])
            vt_values.append(vt)

        n = len(vt_values)
        if n == 0:
            return {"is_watermarked": False, "score": 0.0, "pvalue": 1.0}

        delta = self.threshold
        epsilon = np.log(1.0 / delta) / n

        vt_arr = np.array(vt_values)
        st_values = self.power_law_score_vectorized(vt_arr, epsilon)
        total_score = float(st_values.sum())

        null_sums = self._mc_null_sums(n, epsilon)
        pvalue = float(np.mean(null_sums >= total_score))
        pvalue = max(pvalue, 1.0 / self.n_mc_samples)

        return {
            "is_watermarked": pvalue < delta,
            "score": total_score,
            "pvalue": pvalue,
        }

    def score_tok(self, ngram_tokens: List[int], token_id: int):
        """Return raw V_t value (uniform) as score increment.

        Note: the full power law transformation requires knowing n (total
        tokens), so this returns the raw uniform value. Use detect() for
        the complete algorithm.
        """
        seed = self.get_seed_rng(ngram_tokens)
        self.rng.manual_seed(seed)
        vocab_size = int(self.vocab_size) if self.vocab_size is not None else 0
        rs = torch.rand((vocab_size,), generator=self.rng, device=self.device)
        rs = rs.roll(-self.payload)
        vt = rs[token_id]
        return vt.unsqueeze(0).cpu()

    def get_pvalue(self, score: float, ntoks: int, eps: float = 1e-200):
        """Compute p-value via Monte Carlo for the power law statistic.

        Here 'score' is treated as the sum of power-law-transformed values
        S(V_t), not raw V_t values. Use detect() for the standard pipeline.
        """
        if ntoks <= 0:
            return 1.0
        delta = self.threshold
        epsilon = np.log(1.0 / delta) / ntoks
        null_sums = self._mc_null_sums(ntoks, epsilon)
        pvalue = float(np.mean(null_sums >= score))
        return max(pvalue, eps)
