from typing import Any, List, Optional, Tuple

import torch

from vllm_watermark.samplers.gumbel_sampler import GumbelSampler


# GumbelGenerator: Implements the Gumbel watermarking sampling algorithm
class GumbelGenerator:
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


class WatermarkedLLM:
    def __init__(self, llm: Any, generator: GumbelGenerator):
        self.llm = llm
        self.generator = generator
        self._patch_vllm_sampling()

    def _patch_vllm_sampling(self):
        import vllm.model_executor.layers.sampler as vllm_sampler

        gumbel_sampler = GumbelSampler(self.generator)

        def patched_sample_with_torch(
            probs,
            logprobs,
            sampling_metadata,
            sampling_tensors,
            include_gpu_probs_tensor,
            modify_greedy_probs,
        ):
            from vllm.sampling_params import SamplingType

            categorized_seq_group_ids = {t: [] for t in SamplingType}
            categorized_sample_indices = sampling_metadata.categorized_sample_indices
            for i, seq_group in enumerate(sampling_metadata.seq_groups):
                sampling_params = seq_group.sampling_params
                sampling_type = sampling_params.sampling_type
                categorized_seq_group_ids[sampling_type].append(i)
            sample_results_dict = {}
            sample_metadata = {}
            multinomial_samples = {}
            greedy_samples = None
            if include_gpu_probs_tensor:
                sampled_token_ids_tensor = torch.full(
                    (logprobs.shape[0], 1), -1, dtype=torch.long, device=logprobs.device
                )
            else:
                sampled_token_ids_tensor = None
            for sampling_type in SamplingType:
                sample_indices = categorized_sample_indices[sampling_type]
                num_tokens = len(sample_indices)
                if num_tokens == 0:
                    continue
                seq_group_id = categorized_seq_group_ids[sampling_type]
                seq_groups = [sampling_metadata.seq_groups[i] for i in seq_group_id]
                sample_metadata[sampling_type] = (seq_group_id, seq_groups)
                long_sample_indices = sample_indices.long()
                if sampling_type == SamplingType.GREEDY:
                    greedy_samples = torch.argmax(logprobs[long_sample_indices], dim=-1)
                    if sampled_token_ids_tensor is not None:
                        sampled_token_ids_tensor[long_sample_indices] = (
                            greedy_samples.unsqueeze(-1)
                        )
                    if modify_greedy_probs:
                        vllm_sampler._modify_greedy_probs_inplace(
                            logprobs, probs, long_sample_indices, greedy_samples
                        )
                elif sampling_type in (SamplingType.RANDOM, SamplingType.RANDOM_SEED):
                    # Build real n-gram context for each sampled token
                    ngram_list = []
                    # For each sample index, find the corresponding sequence group and seq_id
                    for group_idx, seq_group in zip(seq_group_id, seq_groups):
                        # Each sample_indices entry corresponds to a token to be sampled for a seq_id in seq_group.seq_ids
                        for seq_id in seq_group.seq_ids:
                            seq_data = seq_group.seq_data[seq_id]
                            token_history = seq_data.get_token_ids()
                            # Get the last n tokens (pad with pad_id if not enough tokens)
                            ngram = [self.generator.pad_id] * max(
                                0, self.generator.ngram - len(token_history)
                            ) + token_history[-self.generator.ngram :]
                            ngram_list.append(ngram)
                    ngram_tokens = torch.tensor(
                        ngram_list, dtype=torch.long, device=probs.device
                    )
                    temperature = 0.8
                    top_p = 0.95
                    sampled = gumbel_sampler.sample(
                        logprobs[long_sample_indices],
                        ngram_tokens,
                        temperature,
                        top_p,
                        sampling_type=sampling_type,
                    )
                    multinomial_samples[sampling_type] = sampled.unsqueeze(-1)
                    if sampled_token_ids_tensor is not None:
                        sampled_token_ids_tensor[long_sample_indices] = (
                            sampled.unsqueeze(-1)
                        )
                else:
                    raise ValueError(f"Unsupported sampling type: {sampling_type}")
            from vllm.model_executor.layers.sampler import (
                SampleResultArgsType,
                get_pythonized_sample_results,
            )

            maybe_deferred_args = SampleResultArgsType(
                sampling_metadata=sampling_metadata,
                sample_metadata=sample_metadata,
                multinomial_samples=multinomial_samples,
                greedy_samples=greedy_samples,
                sample_results_dict=sample_results_dict,
            )
            if not sampling_metadata.skip_sampler_cpu_output:
                return (
                    get_pythonized_sample_results(maybe_deferred_args),
                    sampled_token_ids_tensor,
                )
            else:
                return (maybe_deferred_args, sampled_token_ids_tensor)

        vllm_sampler._sample_with_torch = patched_sample_with_torch

    def generate(self, *args, **kwargs):
        return self.llm.generate(*args, **kwargs)


# Minimal WatermarkGenerator factory for user experience
class WatermarkGenerator:
    @staticmethod
    def create(model, algo: str = "gumbel", **kwargs):
        if algo == "gumbel":
            generator = GumbelGenerator(model, model.get_tokenizer(), **kwargs)
            return WatermarkedLLM(model, generator)
        raise ValueError(f"Unknown watermarking algorithm: {algo}")
