"""Prefix-Free (PF) watermark generator."""

import torch

from .base import WmGenerator


class PFGenerator(WmGenerator):
    """Generate text using Prefix-Free watermarking method."""

    def __init__(self, *args, nowm: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.nowm = nowm

    def sample_next(
        self,
        logits: torch.FloatTensor,  # (bsz, vocab_size): logits for last token
        ngram_tokens: torch.LongTensor,  # (bsz, ngram): tokens to consider when seeding
        temperature: float = 0.8,  # temperature for sampling
        top_p: float = 0.95,  # top p for sampling
    ) -> torch.LongTensor:
        """
        From ngram tokens, select the next token based on the following:
        - hash the ngram tokens and get a seed
        - use the seed to generate V random number r between [0,1]
        - select argmax with u_i + exp(r)
        payload (the message) is encoded by shifting the secret vector r by `payload`.
        """
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > top_p
            probs_sort[mask] = 0.0
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
            log_probs = probs_sort.log()

            for ii in range(ngram_tokens.shape[0]):  # batch of texts
                # seed with hash of ngram tokens
                seed = self.get_seed_rng(ngram_tokens[ii])
                self.rng.manual_seed(seed)
                # generate rs randomly between [0,1]
                rs = torch.rand(
                    self.tokenizer.vocab_size,
                    generator=self.rng,
                    device=probs_sort.device,
                )
                rs = rs.roll(-self.payload)
                rs = rs[probs_idx[ii]]
                # add watermark
                log_probs[ii] = log_probs[ii] - rs.log()
            next_token = torch.argmax(log_probs, dim=-1, keepdim=True)
            next_token = torch.gather(probs_idx, -1, next_token)
        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)
        return next_token
