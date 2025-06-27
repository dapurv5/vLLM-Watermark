"""Maryland-style watermark generator."""

import torch

from ..logit_processors import MarylandLogitProcessor
from .base import WmGenerator


class MarylandGenerator(WmGenerator):
    """Generate text using Maryland's watermarking method."""

    def __init__(self, *args, gamma: float = 0.5, delta: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.delta = delta

        # Initialize the Maryland logit processor
        # We need to get vocab size from the model/tokenizer
        from ..core import WatermarkUtils

        vocab_size = WatermarkUtils.infer_vocab_size(self.model, self.tokenizer)

        self.maryland_processor = MarylandLogitProcessor(
            vocab_size=vocab_size,
            gamma=gamma,
            delta=delta,
            ngram=self.ngram,
            seed=self.seed,
            salt_key=self.salt_key,
            payload=self.payload,
            seeding=self.seeding,
            device=str(self.device),
        )

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
        - use the seed to partition the vocabulary into greenlist (gamma*V words) and blacklist
        - add delta to greenlist words' logits
        payload (the message) is encoded by shifting the secret vector r by `payload`.

        Shape information:
        - logits: (bsz, vocab_size) where bsz is batch size and vocab_size is vocabulary size
        - ngram_tokens: (bsz, ngram) where ngram is the context window size
        - probs: (bsz, vocab_size) probabilities after softmax
        - probs_sort: (bsz, vocab_size) sorted probabilities in descending order
        - probs_idx: (bsz, vocab_size) indices of sorted probabilities
        - probs_sum: (bsz, vocab_size) cumulative sum of sorted probabilities
        - mask: (bsz, vocab_size) boolean mask for top-p filtering
        - next_token: (bsz,) final sampled token indices
        """
        logits = self.logits_processor(logits, ngram_tokens)  # (bsz, vocab_size)
        if temperature > 0:
            # Convert logits to probabilities with temperature scaling
            probs = torch.softmax(logits / temperature, dim=-1)  # (bsz, vocab_size)
            # Sort probabilities in descending order
            probs_sort, probs_idx = torch.sort(
                probs, dim=-1, descending=True
            )  # (bsz, vocab_size)
            # Compute cumulative probabilities
            probs_sum = torch.cumsum(probs_sort, dim=-1)  # (bsz, vocab_size)
            # Create mask for top-p (nucleus) sampling
            mask = probs_sum - probs_sort > top_p  # (bsz, vocab_size)
            probs_sort[mask] = 0.0  # Zero out probabilities beyond top-p
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))  # Renormalize
            # Sample next token using multinomial sampling
            next_token = torch.multinomial(
                probs_sort, num_samples=1, generator=self.rng
            )  # (bsz, 1)
            # Map back to original vocabulary indices
            next_token = torch.gather(probs_idx, -1, next_token)  # (bsz, 1)
        else:
            # If temperature is 0, just take argmax
            next_token = torch.argmax(logits, dim=-1)  # (bsz,)
        next_token = next_token.reshape(-1)  # (bsz,)
        return next_token

    def logits_processor(self, logits, ngram_tokens):
        """Process logits to add bias to words in greenlist."""
        return self.maryland_processor.process_batch_logits(logits, ngram_tokens)
