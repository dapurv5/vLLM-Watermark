"""DiPmark watermark generator.

Implements permutation-based probability redistribution from:
Wu et al., "DiPmark: A Stealthy, Efficient and Resilient Watermark
for Large Language Models" (2023)

Unlike green-list methods (Maryland, Unigram) that add a flat delta to logits,
DIP generates a random vocabulary permutation per context and redistributes
probability mass using cumulative-probability quantile splitting.
"""

import hashlib
from typing import cast

import torch
import torch.nn.functional as F

from .base import WmGenerator


class DIPGenerator(WmGenerator):
    """Generate watermarked text using DiPmark permutation-based reweighting."""

    def __init__(
        self,
        *args,
        alpha: float = 0.45,
        gamma: float = 0.5,
        hash_key: int = 15485863,
        ignore_history: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.hash_key = str(hash_key).encode("utf-8")
        self.ignore_history = ignore_history
        self.context_history: set[bytes] = set()

    def _extract_context_code(self, ngram_tokens: torch.Tensor) -> bytes:
        return ngram_tokens.detach().cpu().numpy().tobytes()

    def _dip_seed(self, context_code: bytes) -> int:
        m = hashlib.sha256()
        m.update(context_code)
        m.update(self.hash_key)
        return int.from_bytes(m.digest(), "big") % (2**32 - 1)

    def _reweight_logits(
        self,
        shuffle: torch.LongTensor,
        p_logits: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Redistribute probability mass via cumulative-probability quantile splitting."""
        unshuffle = torch.argsort(shuffle, dim=-1)

        s_p_logits = torch.gather(p_logits, -1, shuffle)
        s_log_cumsum = torch.logcumsumexp(s_p_logits, dim=-1)
        s_log_cumsum = s_log_cumsum - s_log_cumsum[..., -1:]
        s_cumsum = torch.exp(s_log_cumsum)
        s_p = F.softmax(s_p_logits, dim=-1)

        def _boundary_portions(threshold):
            boundary = torch.argmax(
                (s_cumsum > threshold).to(torch.int), dim=-1, keepdim=True
            )
            p_boundary = torch.gather(s_p, -1, boundary)
            portion = (torch.gather(s_cumsum, -1, boundary) - threshold) / p_boundary
            portion = torch.clamp(portion, 0, 1)
            all_portion = (s_cumsum > threshold).type_as(p_logits)
            all_portion.scatter_(-1, boundary, portion)
            return all_portion

        portion_1 = _boundary_portions(self.alpha)
        portion_2 = _boundary_portions(1.0 - self.alpha)

        s_shift = torch.log(portion_2 / 2 + portion_1 / 2 + 1e-30)
        shift_logits = torch.gather(s_shift, -1, unshuffle)

        return p_logits + shift_logits

    def sample_next(
        self,
        logits: torch.FloatTensor,
        ngram_tokens: torch.LongTensor,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> torch.LongTensor:
        batch_size, vocab_size = logits.shape
        reweighted = logits.clone()

        for i in range(batch_size):
            ctx_code = self._extract_context_code(ngram_tokens[i])
            is_repeated = ctx_code in self.context_history

            if not self.ignore_history:
                self.context_history.add(ctx_code)

            if is_repeated and not self.ignore_history:
                continue

            seed = self._dip_seed(ctx_code)
            rng = torch.Generator(device=logits.device)
            rng.manual_seed(seed)
            shuffle = torch.randperm(vocab_size, generator=rng, device=logits.device)

            reweighted[i:i+1] = self._reweight_logits(
                shuffle.unsqueeze(0), reweighted[i:i+1]
            )

        if temperature > 0:
            probs = torch.softmax(reweighted / temperature, dim=-1)
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
            next_token = torch.argmax(reweighted, dim=-1)

        next_token = next_token.reshape(-1).to(dtype=torch.long)
        return cast(torch.LongTensor, next_token)
