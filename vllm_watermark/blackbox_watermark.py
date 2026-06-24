"""Black-box watermark using best-of-m rejection sampling.

Reference: "A Watermark for Black-Box Language Models"
           Bahri & Wieting, TMLR 2026 (arXiv:2410.02099)

Unlike all other algorithms in this package, this does NOT modify logits or
sampling. Instead it generates m candidate sequences from the LLM, scores
each with a keyed pseudorandom function over n-grams, and selects the
highest-scoring one via the Gumbel-Max trick (distortion-free).
"""

import collections
import hashlib
from typing import Optional

import numpy as np
import scipy.stats
from loguru import logger


TokenIds = tuple[int, ...]


def _int_hash(tokens: TokenIds, key: int) -> int:
    """SHA-256 hash of (key, tokens) -> integer seed."""
    m = hashlib.sha256()
    code = str(key) + str(tokens)
    m.update(bytes(code, "utf-8"))
    return int(m.hexdigest(), 16)


def _calc_prn(seeds: list) -> float:
    """Compute PRF value from seeds using Irwin-Hall CDF."""
    if not seeds:
        seeds = [None]
    vals = [np.random.default_rng(seed).uniform() for seed in seeds]
    return float(scipy.stats.irwinhall.cdf(np.sum(vals), len(vals)))


def score_seqs(
    seqs: list[TokenIds],
    key: int,
    ctx_len: int,
    prefix: TokenIds,
) -> dict[TokenIds, float]:
    """Score sequences using keyed PRF over n-grams.

    For each sequence, extracts n-grams, hashes them with the secret key,
    deduplicates seeds across all sequences (shared n-grams are randomly
    assigned to one sequence), and computes the Irwin-Hall CDF of the
    sum of uniform PRF values.
    """
    seeds_with_id: list[tuple[int, int]] = []
    for seq_id, tokens in enumerate(seqs):
        tokens_with_prefix = prefix + tokens
        for i in range(len(prefix), len(tokens_with_prefix)):
            ctx = tokens_with_prefix[max(0, i + 1 - ctx_len) : i + 1]
            seed = _int_hash(ctx, key)
            seeds_with_id.append((seed, seq_id))

    np.random.shuffle(seeds_with_id)

    used_seeds: set[int] = set()
    deduped_seeds: list[list[int]] = [[] for _ in seqs]
    for seed, seq_id in seeds_with_id:
        if seed not in used_seeds:
            deduped_seeds[seq_id].append(seed)
            used_seeds.add(seed)

    seq_to_prn: dict[TokenIds, float] = {}
    for seq_id, seeds in enumerate(deduped_seeds):
        seq_to_prn[seqs[seq_id]] = _calc_prn(seeds)
    return seq_to_prn


class BlackBoxWatermarkedLLM:
    """Watermarked LLM using best-of-m rejection sampling.

    For each prompt, generates m candidate sequences using the underlying
    LLM, scores each candidate with a keyed PRF, and selects the
    highest-scoring one. The Gumbel-Max trick (raising scores to the
    power m/count) ensures the output distribution is identical to the
    unwatermarked model (distortion-free property).
    """

    def __init__(
        self,
        llm,
        key: int = 15485863,
        n_candidates: int = 16,
        ctx_len: int = 4,
        debug: bool = False,
    ):
        self.llm = llm
        self.key = key
        self.n_candidates = n_candidates
        self.ctx_len = ctx_len
        self.debug = debug

        if self.debug:
            logger.info(
                f"Created BlackBoxWatermarkedLLM with "
                f"n_candidates={n_candidates}, ctx_len={ctx_len}"
            )

    def generate(self, prompts, sampling_params=None, **kwargs):
        """Generate watermarked text via best-of-m rejection sampling.

        Duplicates each prompt m times, generates all candidates in one
        vLLM batch call, then selects the highest-scoring candidate per
        prompt using the keyed PRF.
        """
        from vllm import SamplingParams

        if sampling_params is None:
            sampling_params = SamplingParams()

        if isinstance(prompts, str):
            prompts = [prompts]

        m = self.n_candidates

        expanded_prompts = []
        for prompt in prompts:
            expanded_prompts.extend([prompt] * m)

        if self.debug:
            logger.info(
                f"Generating {m} candidates for {len(prompts)} prompt(s) "
                f"({len(expanded_prompts)} total requests)"
            )

        all_outputs = self.llm.generate(expanded_prompts, sampling_params, **kwargs)

        selected_outputs = []
        tokenizer = self.llm.get_tokenizer()

        for prompt_idx, prompt in enumerate(prompts):
            start = prompt_idx * m
            end = start + m
            candidates = all_outputs[start:end]

            candidate_token_ids = []
            for output in candidates:
                token_ids = tuple(output.outputs[0].token_ids)
                candidate_token_ids.append(token_ids)

            prompt_token_ids = tuple(
                tokenizer.encode(prompt, add_special_tokens=False)
            )

            seq_counter = collections.Counter(candidate_token_ids)
            unique_seqs = list(seq_counter.keys())
            seq_to_prn = score_seqs(
                unique_seqs, self.key, self.ctx_len, prompt_token_ids
            )

            scores = [
                seq_to_prn[seq] ** (float(m) / seq_counter[seq])
                for seq in unique_seqs
            ]
            best_idx = int(np.argmax(scores))
            best_seq = unique_seqs[best_idx]

            if self.debug:
                logger.info(
                    f"Prompt {prompt_idx}: {len(unique_seqs)}/{m} unique candidates, "
                    f"best score={scores[best_idx]:.4f}"
                )

            for output in candidates:
                if tuple(output.outputs[0].token_ids) == best_seq:
                    selected_outputs.append(output)
                    break

        return selected_outputs

    def get_tokenizer(self):
        return self.llm.get_tokenizer()

    def __getattr__(self, name):
        return getattr(self.llm, name)
