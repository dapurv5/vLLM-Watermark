"""Unigram watermark generator.

Implements context-independent green/red list watermarking.
Unlike Maryland (KGW), the green list is fixed at initialization
and does not depend on preceding tokens.

Reference: Zhao et al., "Provable Robust Watermarking for AI-Generated Text" (ICLR 2024)
"""

import hashlib
from typing import cast

import numpy as np
import torch

from .base import WmGenerator


class UnigramGenerator(WmGenerator):
    """Generate watermarked text using Unigram green list bias."""

    def __init__(
        self,
        *args,
        gamma: float = 0.5,
        delta: float = 2.0,
        hash_key: int = 15485863,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.delta = delta
        self.hash_key = hash_key

        from ..core import WatermarkUtils

        vocab_size = WatermarkUtils.infer_vocab_size(self.model, self.tokenizer)
        self.unigram_vocab_size = vocab_size

        green_count = int(gamma * vocab_size)
        mask = np.array(
            [True] * green_count + [False] * (vocab_size - green_count)
        )
        rng = np.random.default_rng(self._hash_fn(hash_key))
        rng.shuffle(mask)

        self.green_mask = torch.tensor(mask, dtype=torch.bool, device=self.device)

    @staticmethod
    def _hash_fn(x: int) -> int:
        x = np.int64(x)
        return int.from_bytes(hashlib.sha256(x).digest()[:4], "little")

    def sample_next(
        self,
        logits: torch.FloatTensor,
        ngram_tokens: torch.LongTensor,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> torch.LongTensor:
        batch_size, vocab_size = logits.shape
        modified_logits = logits.clone()

        green_mask = self.green_mask.to(logits.device)
        if vocab_size > len(green_mask):
            padding = torch.zeros(vocab_size - len(green_mask), dtype=torch.bool, device=logits.device)
            green_mask = torch.cat([green_mask, padding])
        elif vocab_size < len(green_mask):
            green_mask = green_mask[:vocab_size]
        for i in range(batch_size):
            modified_logits[i, green_mask] += self.delta

        if temperature > 0:
            probs = torch.softmax(modified_logits / temperature, dim=-1)
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
            next_token = torch.argmax(modified_logits, dim=-1)

        next_token = next_token.reshape(-1).to(dtype=torch.long)
        return cast(torch.LongTensor, next_token)
