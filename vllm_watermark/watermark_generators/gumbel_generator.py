import torch

from vllm_watermark.watermark_generators.base import BaseGenerator


# GumbelGenerator: Implements the Gumbel watermarking sampling algorithm
class GumbelGenerator(BaseGenerator):
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
        self.tokenizer = tokenizer
        self.model = model
        self.max_seq_len = 1024
        self.pad_id = tokenizer.pad_token_id
        self.eos_id = tokenizer.eos_token_id
        self.ngram = ngram
        self.salt_key = salt_key
        self.seed = seed
        self.hashtable = torch.randperm(1000003)
        self.seeding = seeding
        self.rng = torch.Generator()
        self.rng.manual_seed(self.seed)
        self.payload = payload
        self.device = self.model.device if hasattr(self.model, "device") else "cpu"
        if torch.cuda.is_available():
            self.rng = torch.Generator(device=self.device)
        else:
            self.rng = torch.Generator()
        self.rng.manual_seed(self.seed)
        self.hashtable = torch.randperm(1000003).to(self.device)

    def hashint(self, integer_tensor: torch.LongTensor) -> torch.LongTensor:
        return self.hashtable[integer_tensor % len(self.hashtable)]

    def get_seed_rng(self, input_ids: torch.LongTensor) -> int:
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

    def sample_next(
        self,
        logits: torch.FloatTensor,  # (bsz, vocab_size)
        ngram_tokens: torch.LongTensor,  # (bsz, ngram)
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
            for ii in range(ngram_tokens.shape[0]):
                seed = self.get_seed_rng(ngram_tokens[ii])
                self.rng.manual_seed(seed)
                vocab_size = logits.shape[-1]
                rs = torch.rand(vocab_size, generator=self.rng, device=self.device)
                rs = rs.roll(-self.payload)
                rs = rs[probs_idx[ii].to(self.device)]
                probs_sort[ii] = torch.pow(rs, 1 / probs_sort[ii].to(rs.device))
            next_token = torch.argmax(probs_sort, dim=-1, keepdim=True)
            next_token = torch.gather(probs_idx, -1, next_token)
        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)
        return next_token
