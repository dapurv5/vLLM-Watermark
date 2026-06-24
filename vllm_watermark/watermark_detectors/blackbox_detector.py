"""Black-box watermark detector.

Reference: "A Watermark for Black-Box Language Models"
           Bahri & Wieting, TMLR 2026 (arXiv:2410.02099)

Detection scores a sequence by hashing its n-grams with the secret key,
mapping each hash to a uniform PRF value, and computing the Irwin-Hall
CDF of their sum. Under no watermark, the score is uniform on [0,1].
Watermarked text has a score biased toward 1.
"""

import hashlib

import numpy as np
import scipy.stats
from loguru import logger


TokenIds = tuple[int, ...]


def _int_hash(tokens: TokenIds, key: int) -> int:
    m = hashlib.sha256()
    code = str(key) + str(tokens)
    m.update(bytes(code, "utf-8"))
    return int(m.hexdigest(), 16)


def _calc_prn(seeds: list) -> float:
    if not seeds:
        seeds = [None]
    vals = [np.random.default_rng(seed).uniform() for seed in seeds]
    return float(scipy.stats.irwinhall.cdf(np.sum(vals), len(vals)))


def _score_seq(token_ids: TokenIds, key: int, ctx_len: int) -> float:
    """Score a single sequence (no prefix needed for detection)."""
    seeds: list[int] = []
    seen: set[int] = set()
    for i in range(len(token_ids)):
        ctx = token_ids[max(0, i + 1 - ctx_len) : i + 1]
        seed = _int_hash(ctx, key)
        if seed not in seen:
            seeds.append(seed)
            seen.add(seed)
    return _calc_prn(seeds)


class BlackBoxDetector:
    """Detector for the black-box rejection-sampling watermark.

    Scores text by hashing n-grams with the secret key and computing
    the Irwin-Hall CDF of the sum of uniform PRF values. The p-value
    is 1 - score.
    """

    def __init__(
        self,
        tokenizer,
        key: int = 15485863,
        ctx_len: int = 4,
        threshold: float = 0.02,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.key = key
        self.ctx_len = ctx_len
        self.threshold = threshold

    def detect(self, text: str) -> dict:
        token_ids = tuple(self.tokenizer.encode(text, add_special_tokens=False))
        if len(token_ids) < 2:
            return {"is_watermarked": False, "score": 0.0, "pvalue": 1.0}

        score = _score_seq(token_ids, self.key, self.ctx_len)
        pvalue = 1.0 - score
        return {
            "is_watermarked": pvalue < self.threshold,
            "score": score,
            "pvalue": pvalue,
        }
