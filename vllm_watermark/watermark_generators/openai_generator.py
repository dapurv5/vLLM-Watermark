"""OpenAI-style watermark generator."""

from typing import cast

import torch

from .base import WmGenerator


class OpenaiGenerator(WmGenerator):
    """Generate text using OpenAI's watermarking method."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        - select argmax ( r^(1/p) )
        payload (the message) is encoded by shifting the secret vector r by `payload`.
        """
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > top_p
            probs_sort[mask] = 0.0
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
            for ii in range(ngram_tokens.shape[0]):  # batch of texts
                # seed with hash of ngram tokens
                seed = self.get_seed_rng(ngram_tokens[ii])
                self.rng.manual_seed(seed)
                # generate rs randomly between [0,1]
                vocab_size = logits.shape[-1]
                rs = torch.rand(vocab_size, generator=self.rng, device=self.device)
                rs = rs.roll(-self.payload)
                rs = rs[probs_idx[ii].to(self.device)]
                # compute r^(1/p)
                probs_sort[ii] = torch.pow(rs, 1 / probs_sort[ii].to(self.device))
            # select argmax ( r^(1/p) )
            next_token = torch.argmax(probs_sort, dim=-1, keepdim=True)
            next_token = torch.gather(probs_idx, -1, next_token)
        else:
            next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
        # Ensure return dtype is LongTensor for type checkers
        next_token = next_token.to(dtype=torch.long)
        return cast(torch.LongTensor, next_token)


class OpenaiGeneratorDoubleRandomization(WmGenerator):
    """OpenAI watermarking with multinomial sampling over transformed scores."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sample_next(
        self,
        logits: torch.FloatTensor,
        ngram_tokens: torch.LongTensor,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> torch.LongTensor:
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > top_p
            probs_sort[mask] = 0.0
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))

            batch_size = ngram_tokens.shape[0]
            next_token = torch.empty(
                batch_size, dtype=torch.long, device=probs_sort.device
            )

            vocab_size = logits.shape[-1]
            for ii in range(batch_size):
                seed = self.get_seed_rng(ngram_tokens[ii])
                self.rng.manual_seed(seed)

                rs = torch.rand(
                    vocab_size, generator=self.rng, device=probs_sort.device
                )
                rs = rs.roll(-self.payload)
                rs = rs[probs_idx[ii]]

                scores = torch.pow(rs, 1 / probs_sort[ii])
                sampled_idx_sorted = torch.multinomial(
                    scores, num_samples=1, generator=self.rng
                )
                sampled_token = torch.gather(
                    probs_idx[ii : ii + 1], -1, sampled_idx_sorted.view(1, 1)
                ).squeeze(0)
                next_token[ii] = sampled_token.squeeze(-1)
        else:
            next_token = torch.argmax(logits, dim=-1)

        next_token = next_token.to(dtype=torch.long)
        return cast(torch.LongTensor, next_token)
