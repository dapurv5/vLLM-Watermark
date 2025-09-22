"""Maryland-style logit processor for vLLM compatibility."""

from typing import List

import torch


class MarylandLogitProcessor:
    """
    Maryland watermark logit processor compatible with vLLM.

    This processor implements the Maryland watermarking method by:
    - Hashing ngram tokens to get a seed
    - Using the seed to partition the vocabulary into greenlist and blacklist
    - Adding delta bias to greenlist words' logits
    - Shifting the bias by payload amount
    """

    def __init__(
        self,
        vocab_size: int,
        gamma: float = 0.5,
        delta: float = 1.0,
        ngram: int = 1,
        seed: int = 0,
        salt_key: int = 35317,
        payload: int = 0,
        seeding: str = "hash",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize Maryland logit processor.

        Args:
            vocab_size: Size of the vocabulary
            gamma: Fraction of vocabulary to include in greenlist (default: 0.5)
            delta: Bias amount to add to greenlist tokens (default: 1.0)
            ngram: N-gram size for seeding (default: 1)
            seed: Random seed (default: 0)
            salt_key: Salt key for hashing (default: 35317)
            payload: Payload for shifting bias (default: 0)
            seeding: Seeding method ("hash", "additive", "skip", "min") (default: "hash")
            device: Device to run on (default: "cuda" if available else "cpu")
        """
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.delta = delta
        self.ngram = ngram
        self.seed = seed
        self.salt_key = salt_key
        self.payload = payload
        self.seeding = seeding
        self.device = torch.device(device)

        # Initialize RNG and hashtable
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(self.seed)
        self.hashtable = torch.randperm(1000003, device=self.device)

    def hashint(self, integer_tensor: torch.Tensor) -> torch.Tensor:
        """Hash integer tensor using hashtable."""
        integer_tensor = integer_tensor.to(dtype=torch.long, device=self.device)
        return self.hashtable[integer_tensor % len(self.hashtable)]

    def get_seed_rng(self, input_ids: torch.Tensor) -> int:
        """Get seed from input_ids based on seeding method."""
        # Ensure dtype/device and convert to list to ensure consistency with detector
        input_ids = input_ids.to(dtype=torch.long, device=self.device)
        input_ids_list = input_ids.cpu().tolist()

        if self.seeding == "hash":
            seed = self.seed
            for i in input_ids_list:
                seed = (seed * self.salt_key + i) % (2**64 - 1)
        elif self.seeding == "additive":
            seed_val = self.salt_key * int(sum(input_ids_list))
            seed = int(self.hashint(torch.tensor(seed_val, device=self.device)).item())
        elif self.seeding == "skip":
            seed_val = self.salt_key * int(input_ids_list[0])
            seed = int(self.hashint(torch.tensor(seed_val, device=self.device)).item())
        elif self.seeding == "min":
            seed_tensor = self.hashint(self.salt_key * input_ids)
            seed = int(torch.min(seed_tensor).item())
        else:
            raise ValueError(f"Unknown seeding method: {self.seeding}")
        return seed

    def __call__(self, token_ids: List[int], logits: torch.Tensor) -> torch.Tensor:
        """
        Apply Maryland watermarking to logits.

        This is the vLLM-compatible interface that takes:
        - token_ids: List of previously generated token IDs
        - logits: Tensor of logits for next token (vocab_size,)

        Returns:
        - Modified logits tensor with watermark bias applied
        """
        # Convert token_ids to tensor and get ngram context
        # Ensure token_ids is a list (vLLM may pass it as tuple)
        token_ids = list(token_ids)

        if len(token_ids) >= self.ngram:
            ngram_tokens = torch.tensor(
                token_ids[-self.ngram :], device=self.device, dtype=torch.long
            )
        else:
            # Pad with zeros if we don't have enough tokens
            padding_needed = self.ngram - len(token_ids)
            padded_tokens = [0] * padding_needed + token_ids
            ngram_tokens = torch.tensor(
                padded_tokens, device=self.device, dtype=torch.long
            )

        # Clone logits to avoid modifying original
        modified_logits = logits.clone()

        # Get seed from ngram tokens
        seed = self.get_seed_rng(ngram_tokens)
        self.rng.manual_seed(seed)

        # Generate vocabulary permutation
        vocab_permutation = torch.randperm(
            self.vocab_size, generator=self.rng, device=self.device
        )

        # Create greenlist
        greenlist = vocab_permutation[: int(self.gamma * self.vocab_size)]

        # Get actual logits size to handle vocab size mismatches
        actual_vocab_size = modified_logits.size(-1)

        # Create bias tensor with the actual vocab size
        bias = torch.zeros(actual_vocab_size, device=self.device)

        # Only apply bias to valid indices (handle vocab size mismatch)
        valid_greenlist = greenlist[greenlist < actual_vocab_size]
        if len(valid_greenlist) > 0:
            bias[valid_greenlist] = self.delta

        # Apply payload shift (handle wraparound for smaller vocab)
        if self.payload < actual_vocab_size:
            bias = bias.roll(-self.payload)

        # Add bias to logits
        modified_logits += bias

        return modified_logits

    def process_batch_logits(
        self, logits: torch.Tensor, ngram_tokens: torch.LongTensor
    ) -> torch.Tensor:
        """
        Process a batch of logits with different ngram contexts.

        This method is for compatibility with the MarylandGenerator class.

        Args:
            logits: (batch_size, vocab_size) logits tensor
            ngram_tokens: (batch_size, ngram) tensor of ngram tokens

        Returns:
            Modified logits tensor with watermark bias applied
        """
        batch_size, vocab_size = logits.shape
        modified_logits = logits.clone()

        for i in range(batch_size):
            # Get seed for this sequence
            seed = self.get_seed_rng(ngram_tokens[i])
            self.rng.manual_seed(seed)

            # Generate vocabulary permutation
            vocab_permutation = torch.randperm(
                vocab_size, generator=self.rng, device=self.device
            )

            # Create greenlist
            greenlist = vocab_permutation[: int(self.gamma * vocab_size)]

            # Create bias tensor
            bias = torch.zeros(vocab_size, device=self.device)
            bias[greenlist] = self.delta

            # Apply payload shift
            bias = bias.roll(-self.payload)

            # Add bias to this sequence's logits
            modified_logits[i] += bias

        return modified_logits
