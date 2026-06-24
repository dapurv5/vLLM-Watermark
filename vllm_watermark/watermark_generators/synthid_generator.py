"""Google SynthID watermark generator.

Implements multi-layer tournament watermarking from:
Dathathri et al., "Scalable watermarking for identifying large language model outputs" (Nature 2024)

Uses Linear Congruential Generator (LCG) hashing and a pre-computed binary
sampling table to assign g-values at multiple depth layers. Logits are updated
via a non-distortionary tournament that preserves probability mass.
"""

from typing import cast

import numpy as np
import torch

from .base import WmGenerator


class SynthIDGenerator(WmGenerator):
    """Generate watermarked text using Google SynthID tournament watermarking."""

    def __init__(
        self,
        *args,
        keys: list[int] | None = None,
        sampling_table_size: int = 65536,
        sampling_table_seed: int = 0,
        context_history_size: int = 1024,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if keys is None:
            keys = [
                654, 400, 836, 123, 340, 443, 597, 160, 57, 29,
                590, 639, 13, 715, 468, 990, 966, 226, 324, 585,
                118, 504, 421, 521, 129, 669, 732, 225, 90, 960,
            ]

        self.keys_tensor = torch.tensor(keys, dtype=torch.long, device=self.device)
        self.depth = len(keys)
        self.sampling_table_size = sampling_table_size
        self.context_history_size = context_history_size

        rng = np.random.Generator(np.random.PCG64(sampling_table_seed))
        table_np = rng.integers(0, 2, size=sampling_table_size)
        self.sampling_table = torch.tensor(
            table_np, dtype=torch.long, device=self.device,
        )

        self.context_history: set[int] = set()

    def _accumulate_hash(
        self,
        current_hash: torch.LongTensor,
        data: torch.LongTensor,
    ) -> torch.LongTensor:
        """LCG hash with musl/newlib parameters."""
        multiplier = 6364136223846793005
        increment = 1
        for i in range(data.shape[-1]):
            current_hash = current_hash + data[..., i]
            current_hash = current_hash * multiplier
            current_hash = current_hash + increment
        return current_hash

    def _compute_g_values(
        self,
        context: torch.LongTensor,
        vocab_size: int,
    ) -> torch.FloatTensor:
        """Compute g-values for all vocab tokens given context.

        Args:
            context: (bsz, ngram) context tokens
            vocab_size: size of vocabulary

        Returns:
            g_values: (bsz, vocab_size, depth) binary g-values
        """
        bsz = context.shape[0]

        context_hash = torch.ones(bsz, dtype=torch.long, device=self.device)
        context_hash = self._accumulate_hash(context_hash, context)

        all_tokens = torch.arange(vocab_size, dtype=torch.long, device=self.device)
        all_tokens = all_tokens.unsqueeze(0).expand(bsz, -1)

        token_hash = context_hash.unsqueeze(1).expand(-1, vocab_size)
        token_hash = token_hash + all_tokens
        token_hash = token_hash * 6364136223846793005
        token_hash = token_hash + 1

        g_values = torch.zeros(bsz, vocab_size, self.depth, device=self.device)

        for d in range(self.depth):
            key_val = self.keys_tensor[d]
            depth_hash = token_hash + key_val
            depth_hash = depth_hash * 6364136223846793005
            depth_hash = depth_hash + 1

            table_idx = depth_hash.abs() % self.sampling_table_size
            g_values[:, :, d] = self.sampling_table[table_idx].float()

        return g_values

    def _update_scores_non_distortionary(
        self,
        logits: torch.FloatTensor,
        g_values: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Apply non-distortionary tournament update.

        For each depth layer, multiply probabilities by (1 + g_i - g_mass_i)
        where g_mass_i is the probability mass on g=1 tokens.
        """
        probs = torch.softmax(logits, dim=-1)

        for d in range(self.depth):
            g_d = g_values[:, :, d]
            g_mass = (g_d * probs).sum(dim=-1, keepdim=True)
            probs = probs * (1.0 + g_d - g_mass)
            probs = probs.clamp(min=0)

        log_probs = torch.log(probs + 1e-30)
        return log_probs

    def _is_repeated_context(self, context: torch.LongTensor) -> list[bool]:
        """Check if context has been seen before. Updates history."""
        results = []
        for i in range(context.shape[0]):
            ctx_hash = hash(tuple(context[i].cpu().tolist()))
            if ctx_hash in self.context_history:
                results.append(True)
            else:
                self.context_history.add(ctx_hash)
                if len(self.context_history) > self.context_history_size:
                    oldest = next(iter(self.context_history))
                    self.context_history.discard(oldest)
                results.append(False)
        return results

    def sample_next(
        self,
        logits: torch.FloatTensor,
        ngram_tokens: torch.LongTensor,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> torch.LongTensor:
        batch_size, vocab_size = logits.shape

        is_repeated = self._is_repeated_context(ngram_tokens)
        g_values = self._compute_g_values(ngram_tokens, vocab_size)
        updated_logits = self._update_scores_non_distortionary(logits, g_values)

        for i in range(batch_size):
            if is_repeated[i]:
                updated_logits[i] = logits[i]

        if temperature > 0:
            probs = torch.softmax(updated_logits / temperature, dim=-1)
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
            next_token = torch.argmax(updated_logits, dim=-1)

        next_token = next_token.reshape(-1).to(dtype=torch.long)
        return cast(torch.LongTensor, next_token)
