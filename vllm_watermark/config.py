"""
Configuration management for vLLM-Watermark.
"""

from typing import Dict, Optional, Union

from pydantic import BaseModel, Field


class WatermarkConfig(BaseModel):
    """Base configuration for watermarking algorithms."""

    # Common parameters
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    device: str = Field(default="cuda", description="Device to run the watermarking on")

    # Algorithm-specific parameters
    algorithm_params: Dict[str, Union[float, int, str, bool]] = Field(
        default_factory=dict,
        description="Algorithm-specific parameters"
    )


class KGWConfig(WatermarkConfig):
    """Configuration for KGW watermarking algorithm."""

    gamma: float = Field(default=0.5, description="Gamma parameter for KGW")
    delta: float = Field(default=2.0, description="Delta parameter for KGW")
    context_size: int = Field(default=5, description="Context window size")


class GumbelConfig(WatermarkConfig):
    """Configuration for Gumbel watermarking algorithm."""

    temperature: float = Field(default=1.0, description="Temperature for Gumbel sampling")
    top_k: int = Field(default=50, description="Top-k sampling parameter")


class SemanticConfig(WatermarkConfig):
    """Configuration for Semantic watermarking algorithm."""

    threshold: float = Field(default=0.7, description="Semantic similarity threshold")
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Model to use for semantic embeddings"
    )