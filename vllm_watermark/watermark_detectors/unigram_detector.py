"""Unigram watermark detector.

Detects watermarks by counting green list tokens and computing a z-score.
Uses the same fixed green list mask as UnigramGenerator.
"""

import hashlib

import numpy as np
from scipy import special

from .base import WmDetector


class UnigramDetector(WmDetector):
    """Unigram detector using z-score on green token proportion."""

    def __init__(
        self,
        tokenizer,
        ngram: int = 1,
        seed: int = 0,
        seeding: str = "hash",
        salt_key: int = 35317,
        gamma: float = 0.5,
        delta: float = 2.0,
        hash_key: int = 15485863,
        **kwargs,
    ):
        super().__init__(tokenizer, ngram, seed, seeding, salt_key, **kwargs)
        self.gamma = gamma
        self.delta = delta
        self.hash_key = hash_key

        vocab_size = int(self.vocab_size) if self.vocab_size else 0
        green_count = int(gamma * vocab_size)
        mask = np.array(
            [True] * green_count + [False] * (vocab_size - green_count)
        )
        rng = np.random.default_rng(self._hash_fn(hash_key))
        rng.shuffle(mask)
        self.green_mask = mask

    @staticmethod
    def _hash_fn(x: int) -> int:
        x = np.int64(x)
        return int.from_bytes(hashlib.sha256(x).digest()[:4], "little")

    def detect(self, text: str):
        """Detect watermark by counting green list tokens."""
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) == 0:
            return {"is_watermarked": False, "score": 0.0, "pvalue": 1.0}

        green_count = sum(
            1 for tid in token_ids
            if tid < len(self.green_mask) and self.green_mask[tid]
        )
        T = len(token_ids)

        z_score = (green_count - self.gamma * T) / np.sqrt(
            T * self.gamma * (1 - self.gamma)
        )
        pvalue = 0.5 * special.erfc(z_score / np.sqrt(2))
        pvalue = max(pvalue, 1e-200)

        return {
            "is_watermarked": pvalue < self.threshold,
            "score": z_score,
            "pvalue": pvalue,
        }

    def score_tok(self, ngram_tokens, token_id):
        """Score a single token (1 if green, 0 if red)."""
        import torch

        vocab_size = int(self.vocab_size) if self.vocab_size else 0
        scores = torch.zeros((vocab_size,), device=self.device)
        green_indices = np.where(self.green_mask)[0]
        scores[green_indices] = 1
        return scores.roll(-token_id)

    def get_pvalue(self, score: float, ntoks: int, eps: float = 1e-200):
        """Z-score based p-value."""
        z_score = (score - self.gamma * ntoks) / np.sqrt(
            self.gamma * (1 - self.gamma) * ntoks
        )
        pvalue = 0.5 * special.erfc(z_score / np.sqrt(2))
        return max(pvalue, eps)
