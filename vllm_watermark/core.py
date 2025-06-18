import torch
from vllm.entrypoints.llm import LLM

from vllm_watermark.watermark_generators.base import BaseGenerator
from vllm_watermark.watermark_generators.gumbel_generator import GumbelGenerator


# Factory for creating watermarked LLMs
class WatermarkedLLMs:
    @staticmethod
    def create(model, algo: str = "gumbel", **kwargs) -> LLM:
        if algo == "gumbel":
            generator = GumbelGenerator(model, model.get_tokenizer(), **kwargs)
            return WatermarkedLLM(model, generator)
        raise ValueError(f"Unknown watermarking algorithm: {algo}")


class WatermarkedLLM:
    def __init__(self, llm: LLM, generator: BaseGenerator):
        self.llm = llm
        self.generator = generator
        self._patch_vllm_sampling()

    def _patch_vllm_sampling(self):
        import vllm.model_executor.layers.sampler as vllm_sampler

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
                    if sampling_type in (SamplingType.RANDOM, SamplingType.RANDOM_SEED):
                        sampled = self.generator.sample_next(
                            logprobs[long_sample_indices],
                            ngram_tokens,
                            temperature,
                            top_p,
                        )
                    else:
                        sampled = torch.argmax(logprobs[long_sample_indices], dim=-1)
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
