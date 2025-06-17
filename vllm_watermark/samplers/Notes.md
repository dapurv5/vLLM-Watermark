### Specifications


- Here is a naive pure torch implementation of OpenAI/Gumbel watermark
```python
from typing import List

import torch


class WmGenerator:
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
        self.pad_id = tokenizer.pad_token_id
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
        self.device = self.model.device
        # Move RNG to GPU if CUDA is available
        if torch.cuda.is_available():
            self.rng = torch.Generator(device=self.device)
        else:
            self.rng = torch.Generator()
        self.rng.manual_seed(self.seed)

        # Move hashtable to GPU if available
        self.hashtable = torch.randperm(1000003).to(self.device)

    def hashint(self, integer_tensor: torch.LongTensor) -> torch.LongTensor:
        """Optimized hashint using GPU."""
        return self.hashtable[integer_tensor % len(self.hashtable)]

    def get_seed_rng(self, input_ids: torch.LongTensor) -> int:
        """
        Seed RNG with hash of input_ids.
        Adapted from https://github.com/jwkirchenbauer/lm-watermarking
        """
        if self.seeding == "hash":
            seed = self.seed
            for i in input_ids:
                seed = (seed * self.salt_key + i.item()) % (2**64 - 1)
        elif self.seeding == "additive":
            seed = self.salt_key * torch.sum(input_ids).item()
            seed = self.hashint(seed)
        elif self.seeding == "skip":
            seed = self.salt_key * input_ids[0].item()
            seed = self.hashint(seed)
        elif self.seeding == "min":
            seed = self.hashint(self.salt_key * input_ids)
            seed = torch.min(seed).item()
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

        with torch.amp.autocast("cuda"):  # Enable AMP for faster computation
            for cur_pos in range(start_pos, total_len):
                outputs = self.model.forward(
                    tokens[:, prev_pos:cur_pos],
                    use_cache=True,
                    past_key_values=outputs.past_key_values if prev_pos > 0 else None,
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

    def sample_next(
        self,
        logits: torch.FloatTensor,  # (bsz, vocab_size): logits for last token
        ngram_tokens: torch.LongTensor,  # (bsz, ngram): tokens to consider when seeding
        temperature: float = 0.8,  # temperature for sampling
        top_p: float = 0.95,  # top p for sampling
    ) -> torch.LongTensor:
        """Vanilla sampling with temperature and top p."""
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > top_p
            probs_sort[mask] = 0.0
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
            next_token = torch.multinomial(
                probs_sort, num_samples=1, generator=self.rng
            )  # one hot of next token, ordered by original probs
            next_token = torch.gather(
                probs_idx, -1, next_token
            )  # one hot of next token, ordered by vocab
        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)
        return next_token


class OpenaiGenerator(WmGenerator):
    """Generate text using LLaMA and Aaronson's watermarking method."""

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
                rs = rs[probs_idx[ii]]
                # compute r^(1/p)
                probs_sort[ii] = torch.pow(rs, 1 / probs_sort[ii])
            # select argmax ( r^(1/p) )
            next_token = torch.argmax(probs_sort, dim=-1, keepdim=True)
            next_token = torch.gather(probs_idx, -1, next_token)
        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)
        return next_token
```


- Incorporate this logic in vLLM by overriding core vLLM classes especially around sampling.
- Write an example file to run this on the `meta-llama/Llama-3.2-1B` model. This is the entry point we will use for debugging and developing our code for this watermark.
- Remember there are 2 types of watermark sampling based and logit based. OpenAI watermark is a sampling based watermark and cannot be implemented by overriding a LogitsProcessor (common for KGW Red-Green watermarks)
- Create appropriate classes to have the following experience for the user
```python
   from vllm_watermark import WatermarkGenerator, WatermarkDetector

   # Create a watermarking algorithm instance
   llm = LLM(model="meta-llama/Llama-3.2-1B") # vLLM model
   wm_llm = WatermarkGenerator.create(llm, "gumbel", seed=42.0)

   # Apply watermarking while generating
   watermarked_logits = wm_llm.generate(prompt)
```
- Let's not worry about the detector for now.
- The implementation should use vLLM primitives and not have a loop over tokens like in the naive reference implementation and deeply integrate with vLLM.