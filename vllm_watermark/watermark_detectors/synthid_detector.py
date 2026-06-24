"""Google SynthID watermark detector.

Uses the same LCG hashing and sampling table as the generator to recompute
g-values from tokenized text, then scores via mean g-value aggregation.
Watermarked text has mean g-value > 0.5 because the tournament biases
sampling toward g=1 tokens.
"""

import numpy as np
import torch


class SynthIDDetector:
    """Detect SynthID tournament watermark via mean g-value scoring."""

    def __init__(
        self,
        tokenizer,
        vocab_size: int | None = None,
        keys: list[int] | None = None,
        ngram: int = 4,
        sampling_table_size: int = 65536,
        sampling_table_seed: int = 0,
        threshold: float = 0.52,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.ngram = ngram
        self.threshold = threshold
        self.sampling_table_size = sampling_table_size

        if keys is None:
            keys = [
                654, 400, 836, 123, 340, 443, 597, 160, 57, 29,
                590, 639, 13, 715, 468, 990, 966, 226, 324, 585,
                118, 504, 421, 521, 129, 669, 732, 225, 90, 960,
            ]

        self.keys = keys
        self.depth = len(keys)

        rng = np.random.Generator(np.random.PCG64(sampling_table_seed))
        self.sampling_table = rng.integers(0, 2, size=sampling_table_size)

    def _accumulate_hash(self, current_hash: np.ndarray, data: np.ndarray) -> np.ndarray:
        """LCG hash matching the generator's torch implementation."""
        multiplier = np.int64(6364136223846793005)
        increment = np.int64(1)
        current_hash = current_hash.astype(np.int64)
        for i in range(data.shape[-1]):
            current_hash = current_hash + data[..., i].astype(np.int64)
            current_hash = current_hash * multiplier
            current_hash = current_hash + increment
        return current_hash

    def _compute_g_values_for_tokens(
        self,
        context_tokens: np.ndarray,
        target_tokens: np.ndarray,
    ) -> np.ndarray:
        """Compute g-values for actual tokens given their context.

        Args:
            context_tokens: (num_positions, ngram) context windows
            target_tokens: (num_positions,) the token at each position

        Returns:
            g_values: (num_positions, depth) binary g-values
        """
        num_pos = context_tokens.shape[0]

        context_hash = np.ones(num_pos, dtype=np.int64)
        context_hash = self._accumulate_hash(context_hash, context_tokens)

        token_hash = context_hash + target_tokens.astype(np.int64)
        token_hash = token_hash * np.int64(6364136223846793005)
        token_hash = token_hash + np.int64(1)

        g_values = np.zeros((num_pos, self.depth))

        for d in range(self.depth):
            depth_hash = token_hash + np.int64(self.keys[d])
            depth_hash = depth_hash * np.int64(6364136223846793005)
            depth_hash = depth_hash + np.int64(1)

            table_idx = np.abs(depth_hash) % self.sampling_table_size
            g_values[:, d] = self.sampling_table[table_idx]

        return g_values

    def detect(self, text: str) -> dict:
        """Detect SynthID watermark in text.

        Returns dict with is_watermarked, score (mean g-value), and pvalue.
        """
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)

        if len(token_ids) <= self.ngram:
            return {"is_watermarked": False, "score": 0.5, "pvalue": 1.0}

        token_arr = np.array(token_ids, dtype=np.int64)

        contexts = []
        targets = []
        for i in range(self.ngram, len(token_arr)):
            contexts.append(token_arr[i - self.ngram : i])
            targets.append(token_arr[i])

        context_tokens = np.stack(contexts)
        target_tokens = np.array(targets, dtype=np.int64)

        g_values = self._compute_g_values_for_tokens(context_tokens, target_tokens)

        mean_g = np.mean(g_values)

        num_positions = g_values.shape[0]
        total_values = num_positions * self.depth
        std_err = np.sqrt(0.25 / total_values) if total_values > 0 else 1.0
        z_score = (mean_g - 0.5) / std_err

        from scipy import special
        pvalue = float(0.5 * special.erfc(z_score / np.sqrt(2)))

        return {
            "is_watermarked": pvalue < self.threshold,
            "score": float(mean_g),
            "pvalue": pvalue,
        }
