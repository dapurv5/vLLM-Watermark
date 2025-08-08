# Reference: https://github.com/facebookresearch/three_bricks
# Reference: https://github.com/XuandongZhao/pf-decoding

from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import List

import torch


class WmGenerator(ABC):
    """Abstract base class for watermark generators."""

    def __init__(
        self,
        model,
        tokenizer,
        ngram: int = 1,
        seed: int = 0,
        seeding: str = "hash",
        salt_key: int = 35317,
        payload: int = 0,
    ):
        # model config
        self.tokenizer = tokenizer
        self.model = model
        self.max_seq_len = 1024

        # Handle pad_token_id - use eos_token_id as fallback if pad_token_id is None
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = tokenizer.eos_token_id
            if self.pad_id is None:
                # Final fallback - use 0 (typically corresponds to some special token)
                self.pad_id = 0

        self.eos_id = tokenizer.eos_token_id

        # watermark config
        self.ngram = ngram
        self.salt_key = salt_key
        self.seed = seed
        self.hashtable = torch.randperm(1000003)
        self.seeding = seeding
        self.rng = torch.Generator()
        self.rng.manual_seed(self.seed)
        self.payload = payload

        # Use CUDA if available (same as detector)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.rng = torch.Generator(device=self.device)
        else:
            self.device = torch.device("cpu")
            self.rng = torch.Generator()
        self.rng.manual_seed(self.seed)

        # Move hashtable to GPU if available
        self.hashtable = torch.randperm(1000003).to(self.device)

    def hashint(self, integer_tensor: torch.Tensor) -> torch.Tensor:
        """Optimized hashint using GPU."""
        # Ensure the tensor is on the same device as the hashtable
        integer_tensor = integer_tensor.to(self.hashtable.device)
        return self.hashtable[integer_tensor % len(self.hashtable)]

    def get_seed_rng(self, input_ids: torch.Tensor) -> int:
        """
        Seed RNG with hash of input_ids.
        Adapted from https://github.com/jwkirchenbauer/lm-watermarking
        """
        # Ensure correct dtype/device handling
        input_ids = input_ids.to(dtype=torch.long, device=self.hashtable.device)

        if self.seeding == "hash":
            seed = self.seed
            for i in input_ids:
                seed = (seed * self.salt_key + int(i.item())) % (2**64 - 1)
        elif self.seeding == "additive":
            summed = int(torch.sum(input_ids).item())
            seed_tensor = torch.tensor(
                self.salt_key * summed, device=self.hashtable.device, dtype=torch.long
            )
            seed = int(self.hashint(seed_tensor).item())
        elif self.seeding == "skip":
            first_id = int(input_ids[0].item())
            seed_tensor = torch.tensor(
                self.salt_key * first_id, device=self.hashtable.device, dtype=torch.long
            )
            seed = int(self.hashint(seed_tensor).item())
        elif self.seeding == "min":
            seed_tensor = self.hashint(self.salt_key * input_ids)
            seed = int(torch.min(seed_tensor).item())
        else:
            raise ValueError(f"Unknown seeding method: {self.seeding}")
        return seed

    @torch.inference_mode()
    def generate(
        self,
        prompts: List[str],
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> List[str]:
        """
        Generate text from prompts.
        Adapted from https://github.com/facebookresearch/llama/
        """

        bsz = len(prompts)
        prompt_tokens = [
            self.tokenizer.encode(x, add_special_tokens=False) for x in prompts
        ]
        # Truncate prompts that are too long
        prompt_tokens = [t[: self.max_seq_len] for t in prompt_tokens]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])
        total_len = min(self.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.pad_id, device=self.device).long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : min(len(t), total_len)] = torch.tensor(
                t[:total_len], device=self.device
            ).long()
        input_text_mask = tokens != self.pad_id

        start_pos = min_prompt_size
        prev_pos = 0
        outputs = None

        autocast_ctx = (
            torch.autocast(device_type="cuda")
            if torch.cuda.is_available()
            else nullcontext()
        )
        with autocast_ctx:  # Enable AMP for faster computation
            for cur_pos in range(start_pos, total_len):
                outputs = self.model.forward(
                    tokens[:, prev_pos:cur_pos],
                    use_cache=True,
                    past_key_values=(
                        None if outputs is None else outputs.past_key_values
                    ),
                )
                ngram_tokens = tokens[:, cur_pos - self.ngram : cur_pos]
                next_toks = self.sample_next(
                    outputs.logits[:, -1, :], ngram_tokens, temperature, top_p
                )
                tokens[:, cur_pos] = torch.where(
                    input_text_mask[:, cur_pos], tokens[:, cur_pos], next_toks
                )
                prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens):
            # Keep tensor on GPU while finding length and EOS
            curr_len = len(prompt_tokens[i]) + max_gen_len
            t = t[:curr_len]

            # Find EOS token while still on GPU
            eos_indices = (t == self.eos_id).nonzero()
            if len(eos_indices) > 0:
                t = t[: eos_indices[0]]

            # Only move to CPU when absolutely necessary for tokenizer
            decoded.append(self.tokenizer.decode(t.cpu().tolist()))
        torch.cuda.empty_cache()
        return decoded

    @abstractmethod
    def sample_next(
        self,
        logits: torch.FloatTensor,  # (bsz, vocab_size): logits for last token
        ngram_tokens: torch.Tensor,  # (bsz, ngram): tokens to consider when seeding
        temperature: float = 0.8,  # temperature for sampling
        top_p: float = 0.95,  # top p for sampling
    ) -> torch.LongTensor:
        """Sample next token using the watermarking algorithm."""
        pass
