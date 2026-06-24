"""DiPmark watermark detector.

Detects permutation-based watermarks by checking each token's quantile
position in the context-seeded random permutation. Tokens in the boosted
portion of the permutation (quantile >= gamma) are counted as "green".
Z-score is computed against the binomial null hypothesis.
"""

import hashlib
from math import sqrt

import numpy as np
import torch
from scipy import special

from .base import WmDetector


class DIPDetector(WmDetector):
    """Detect DiPmark permutation-based watermarks."""

    def __init__(
        self,
        tokenizer,
        ngram: int = 1,
        seed: int = 0,
        seeding: str = "hash",
        salt_key: int = 35317,
        gamma: float = 0.5,
        hash_key: int = 15485863,
        ignore_history: bool = False,
        **kwargs,
    ):
        kwargs.pop('alpha', None)
        super().__init__(tokenizer, ngram, seed, seeding, salt_key, **kwargs)
        self.gamma = gamma
        self.hash_key = str(hash_key).encode("utf-8")
        self.ignore_history = ignore_history
        self.context_history: set[bytes] = set()

    def _extract_context_code(self, ngram_tokens: list[int]) -> bytes:
        return torch.tensor(ngram_tokens, dtype=torch.long).numpy().tobytes()

    def _dip_seed(self, context_code: bytes) -> int:
        m = hashlib.sha256()
        m.update(context_code)
        m.update(self.hash_key)
        return int.from_bytes(m.digest(), "big") % (2**32 - 1)

    def score_tok(self, ngram_tokens: list[int], token_id: int):
        """Score a token by its quantile position in the context-seeded permutation."""
        ctx_code = self._extract_context_code(ngram_tokens)

        is_repeated = ctx_code in self.context_history
        if not self.ignore_history:
            self.context_history.add(ctx_code)

        if is_repeated and not self.ignore_history:
            return torch.tensor([0.0])

        seed = self._dip_seed(ctx_code)
        rng = torch.Generator(device=self.device)
        rng.manual_seed(seed)

        vocab_size = int(self.vocab_size) if self.vocab_size else 128256
        shuffle = torch.randperm(vocab_size, generator=rng, device=self.device)

        position = (shuffle == token_id).nonzero(as_tuple=True)[0]
        if len(position) == 0:
            return torch.tensor([0.0])

        quantile = (position.item() + 1) / vocab_size
        is_green = 1.0 if quantile >= self.gamma else 0.0
        return torch.tensor([is_green])

    def get_pvalue(self, score: float, ntoks: int, eps: float = 1e-200):
        """Compute p-value using z-score approximation of the binomial test."""
        if ntoks < 1:
            return 1.0
        expected_green_fraction = 1.0 - self.gamma
        expected = expected_green_fraction * ntoks
        std = sqrt(ntoks * expected_green_fraction * self.gamma)
        if std == 0:
            return 1.0
        z = (score - expected) / std
        pvalue = 0.5 * special.erfc(z / np.sqrt(2))
        return max(float(pvalue), eps)

    def detect(self, text: str):
        """Detect watermark, resetting context history per call."""
        self.context_history.clear()
        result = super().detect(text)
        self.context_history.clear()
        return result
