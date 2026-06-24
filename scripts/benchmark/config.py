"""
Configuration for watermarking algorithms.
"""

from dataclasses import dataclass
from typing import Any, Dict

from vllm_watermark.core import DetectionAlgorithm, WatermarkingAlgorithm


@dataclass
class AlgorithmConfig:
    """Configuration for a watermarking algorithm."""

    name: str
    algorithm: WatermarkingAlgorithm
    detection_algorithm: DetectionAlgorithm
    params: Dict[str, Any]

    def get_config_string(self) -> str:
        """Get a human-readable configuration string."""
        config_parts = [f"ngram={self.params.get('ngram', 2)}"]
        config_parts.append(f"seed={self.params.get('seed', 42)}")

        if "gamma" in self.params:
            config_parts.append(f"gamma={self.params['gamma']}")
        if "delta" in self.params:
            config_parts.append(f"delta={self.params['delta']}")
        if "payload" in self.params:
            config_parts.append(f"payload={self.params['payload']}")

        return ", ".join(config_parts)


# Default algorithm configurations
ALGORITHM_CONFIGS = {
    "OPENAI": AlgorithmConfig(
        name="OPENAI",
        algorithm=WatermarkingAlgorithm.OPENAI,
        detection_algorithm=DetectionAlgorithm.OPENAI_Z,
        params={
            "ngram": 2,
            "seed": 42,
            "payload": 0,
        },
    ),
    "MARYLAND": AlgorithmConfig(
        name="MARYLAND",
        algorithm=WatermarkingAlgorithm.MARYLAND,
        detection_algorithm=DetectionAlgorithm.MARYLAND_Z,
        params={
            "ngram": 2,
            "seed": 42,
            "gamma": 0.5,
            "delta": 1.0,
        },
    ),
    "PF": AlgorithmConfig(
        name="PF",
        algorithm=WatermarkingAlgorithm.PF,
        detection_algorithm=DetectionAlgorithm.PF,
        params={
            "ngram": 2,
            "seed": 42,
            "payload": 0,
        },
    ),
    "UNIGRAM": AlgorithmConfig(
        name="UNIGRAM",
        algorithm=WatermarkingAlgorithm.UNIGRAM,
        detection_algorithm=DetectionAlgorithm.UNIGRAM_Z,
        params={
            "ngram": 1,
            "seed": 42,
            "gamma": 0.5,
            "delta": 2.0,
            "hash_key": 15485863,
        },
    ),
    "SYNTHID": AlgorithmConfig(
        name="SYNTHID",
        algorithm=WatermarkingAlgorithm.SYNTHID,
        detection_algorithm=DetectionAlgorithm.SYNTHID,
        params={
            "ngram": 4,
            "seed": 42,
        },
    ),
    "DIP": AlgorithmConfig(
        name="DIP",
        algorithm=WatermarkingAlgorithm.DIP,
        detection_algorithm=DetectionAlgorithm.DIP,
        params={
            "ngram": 2,
            "seed": 42,
            "alpha": 0.45,
            "gamma": 0.5,
            "hash_key": 15485863,
        },
    ),
    "SWEET": AlgorithmConfig(
        name="SWEET",
        algorithm=WatermarkingAlgorithm.SWEET,
        detection_algorithm=DetectionAlgorithm.SWEET,
        params={
            "ngram": 2,
            "seed": 42,
            "gamma": 0.5,
            "delta": 2.0,
            "hash_key": 15485863,
            "entropy_threshold": 3.0,
        },
    ),
    "BLACKBOX": AlgorithmConfig(
        name="BLACKBOX",
        algorithm=WatermarkingAlgorithm.BLACKBOX,
        detection_algorithm=DetectionAlgorithm.BLACKBOX,
        params={
            "ngram": 4,
            "hash_key": 15485863,
            "n_candidates": 128,
        },
    ),
}
