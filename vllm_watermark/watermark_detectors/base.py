# Reference: https://github.com/facebookresearch/three_bricks

from typing import Any, List

import numpy as np
import numpy.typing as npt
import torch


class WmDetector:
    """Abstract base class for watermark detectors."""

    def __init__(
        self,
        tokenizer,
        ngram: int = 1,
        seed: int = 0,
        seeding: str = "hash",
        salt_key: int = 35317,
        vocab_size: int | None = None,
        threshold: float = 0.05,
    ):
        # model config
        self.tokenizer = tokenizer

        # Set vocab size
        self.vocab_size = vocab_size

        # watermark config
        self.ngram = ngram
        self.salt_key = salt_key
        self.seed = seed
        self.threshold = threshold
        self.hashtable = torch.randperm(1000003)
        self.seeding = seeding
        self.rng = torch.Generator()
        self.rng.manual_seed(self.seed)
        # Move RNG to GPU if CUDA is available
        # This assumes that the code is being run with CUDA_VISIBLE_DEVICES set to a single GPU
        if torch.cuda.is_available():
            device = torch.device("cuda")
            self.device = device
            self.rng = torch.Generator(device=device)
        else:
            self.device = torch.device("cpu")
            self.rng = torch.Generator()
        self.rng.manual_seed(self.seed)

    def hashint(self, integer_tensor: torch.Tensor) -> torch.Tensor:
        """Adapted from https://github.com/jwkirchenbauer/lm-watermarking"""
        integer_tensor = integer_tensor.to(dtype=torch.long)
        return self.hashtable[integer_tensor.cpu() % len(self.hashtable)]

    def get_seed_rng(self, input_ids: List[int]) -> int:
        """Seed RNG with hash of input_ids.

        Adapted from https://github.com/jwkirchenbauer/lm-watermarking
        """
        if self.seeding == "hash":
            seed: int = int(self.seed)
            for i in input_ids:
                seed = (seed * self.salt_key + i) % (2**64 - 1)
        elif self.seeding == "additive":
            tmp = self.salt_key * torch.sum(torch.tensor(input_ids))
            tmp_hash = self.hashint(tmp)
            seed = int(tmp_hash.item())  # Convert tensor to int
        elif self.seeding == "skip":
            seed = self.salt_key * input_ids[0]
            seed = int(self.hashint(torch.tensor(seed)).item())  # Convert tensor to int
        elif self.seeding == "min":
            tmp_hash = self.hashint(self.salt_key * torch.tensor(input_ids))
            seed = int(torch.min(tmp_hash).item())  # Convert tensor to int
        return seed

    def aggregate_scores(
        self, scores: List[List[np.ndarray]], aggregation: str = "mean"
    ) -> List[np.ndarray]:
        """Aggregate scores along a text."""
        scores_arr: npt.NDArray[Any] = np.asarray(scores, dtype=object)
        if aggregation == "sum":
            return [np.sum(ss, axis=0) for ss in scores_arr]
        elif aggregation == "mean":
            vocab_size: int = int(self.vocab_size or 0)
            return [
                (
                    np.mean(ss, axis=0)
                    if ss.shape[0] != 0
                    else np.ones(shape=(vocab_size,))
                )
                for ss in scores_arr
            ]
        elif aggregation == "max":
            return [np.max(ss, axis=0) for ss in scores_arr]
        else:
            raise ValueError(f"Aggregation {aggregation} not supported.")

    def get_scores_by_t(
        self,
        texts: List[str],
        scoring_method: str = "none",
        ntoks_max: int | None = None,
        payload_max: int = 0,
    ) -> List[List[np.ndarray]]:
        """Get score increment for each token in list of texts.

        Args:
            texts: list of texts
            scoring_method:
                'none': score all ngrams
                'v1': only score tokens for which wm window is unique
                'v2': only score unique {wm window+tok} is unique
            ntoks_max: maximum number of tokens
            payload_max: maximum number of messages

        Note that this won't return a list of empty np.array if len(tokens_id[ii]) > self.ngram + 1

        Output:
            score_lists: list of [np array of score increments for every token and payload] for each text
        """
        bsz = len(texts)
        tokens_id = [self.tokenizer.encode(x, add_special_tokens=False) for x in texts]
        if ntoks_max is not None:
            tokens_id = [x[:ntoks_max] for x in tokens_id]
        score_lists = []
        for ii in range(bsz):
            total_len = len(tokens_id[ii])
            start_pos = self.ngram + 1
            rts = []
            seen_ntuples = set()
            for cur_pos in range(start_pos, total_len):
                ngram_tokens = tokens_id[ii][cur_pos - self.ngram : cur_pos]  # h
                if scoring_method == "v1":
                    tup_for_unique = tuple(ngram_tokens)
                    if tup_for_unique in seen_ntuples:
                        continue
                    seen_ntuples.add(tup_for_unique)
                elif scoring_method == "v2":
                    tup_for_unique = tuple(
                        ngram_tokens + tokens_id[ii][cur_pos : cur_pos + 1]
                    )
                    if tup_for_unique in seen_ntuples:
                        continue
                    seen_ntuples.add(tup_for_unique)
                rt = self.score_tok(ngram_tokens, tokens_id[ii][cur_pos])
                rt = rt.detach().cpu().numpy()[: payload_max + 1]
                rts.append(rt)
            score_lists.append(rts)
        return score_lists

    def get_pvalues(
        self, scores: List[List[np.ndarray]], eps: float = 1e-200
    ) -> np.ndarray:
        """Get p-value for each text.

        Args:
            score_lists: list of [list of score increments for each token] for each text

        Output:
            pvalues: np array of p-values for each text and payload
        """
        pvalues: List[List[float]] = []
        scores_arr: npt.NDArray[Any] = np.asarray(
            scores, dtype=object
        )  # bsz x ntoks x payload_max
        for ss in scores_arr:
            ntoks = ss.shape[0]
            scores_by_payload = (
                np.sum(ss, axis=0) if ntoks != 0 else np.zeros(shape=ss.shape[-1])
            )  # payload_max
            pvalues_by_payload = [
                self.get_pvalue(score, ntoks, eps=eps) for score in scores_by_payload
            ]
            pvalues.append(pvalues_by_payload)
        return np.asarray(pvalues)  # bsz x payload_max

    def get_pvalues_by_t(self, scores: List[float]) -> List[float]:
        """Get p-value for each text."""
        pvalues = []
        cum_score = 0
        cum_toks = 0
        for ss in scores:
            cum_score += ss
            cum_toks += 1
            pvalue = self.get_pvalue(cum_score, cum_toks)
            pvalues.append(pvalue)
        return pvalues

    def detect(self, text: str):
        """Detect if a text is watermarked."""
        scores_no_aggreg = self.get_scores_by_t([text])
        pvalues = self.get_pvalues(scores_no_aggreg)
        scores = self.aggregate_scores(scores_no_aggreg)
        # Assuming we're interested in the first payload (index 0)
        if len(pvalues) == 0 or len(pvalues[0]) == 0:
            print(f"No pvalues found for text: {text} because it's too short")
            return {
                "is_watermarked": False,
                "score": 0.0,
                "pvalue": 1.0,
            }
        pvalue = pvalues[0][0]
        score = scores[0][0]
        is_watermarked = pvalue < self.threshold
        return {
            "is_watermarked": is_watermarked,
            "score": score,
            "pvalue": pvalue,
        }

    def score_tok(self, ngram_tokens: List[int], token_id: int):
        """for each token in the text, compute the score increment"""
        raise NotImplementedError

    def get_pvalue(self, score: float, ntoks: int, eps: float = 1e-200):
        """compute the p-value for a couple of score and number of tokens"""
        raise NotImplementedError
