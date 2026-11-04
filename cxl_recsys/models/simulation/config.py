"""CPU-only configuration for the random-weight DeepSeek-style Indexer."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class IndexerConfig:
    n_heads: int = 64
    head_dim: int = 128
    q_lora_rank: int = 128
    rope_dim: int = 64
    rope_theta: float = 10000.0
    rope_factor: float = 40.0
    original_seq_len: int = 4096
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    seed: int = 0
    query_tile: int = 32

    def __post_init__(self):
        for name in (
            "n_heads",
            "head_dim",
            "q_lora_rank",
            "rope_dim",
            "original_seq_len",
            "query_tile",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"indexer_{name} must be a positive integer")
        if self.head_dim & (self.head_dim - 1):
            raise ValueError("indexer_head_dim must be a power of two for Hadamard")
        if self.rope_dim % 2 or self.rope_dim > self.head_dim:
            raise ValueError("indexer_rope_dim must be even and <= indexer_head_dim")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("indexer_seed must be an integer in [0, 2**63)")
        for name in ("rope_theta", "rope_factor", "beta_fast", "beta_slow"):
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"indexer_{name} must be finite and positive")
        if self.rope_theta <= 1 or self.rope_factor < 1:
            raise ValueError("Indexer requires rope_theta > 1 and rope_factor >= 1")
        if self.beta_fast <= self.beta_slow:
            raise ValueError("indexer_beta_fast must exceed indexer_beta_slow")
