"""Test backbone operators with a random-weight DeepSeek-style Indexer.

The attention/MLP/MoE simulation is not a trained model. Indexer Q/K projection,
normalization, RoPE/Hadamard and weighted ReLU Top-K are real BF16/FP32 operations
with fixed random weights, not random index sampling or the reference FP8 path.
"""

from dataclasses import dataclass

import torch

from cxl_recsys.models.simulation.config import IndexerConfig
from cxl_recsys.models.simulation.indexer import DSAIndexer

REFERENCE_COMMIT = "87e509a2e5a100d221c97df52c6e8be7835f0057"


@dataclass
class Projection:
    q_nope: torch.Tensor
    q_pe: torch.Tensor
    kv: torch.Tensor
    qr: torch.Tensor


class DSAStub:
    def __init__(
        self,
        dim: int,
        topk: int,
        layers=2,
        max_history=4096,
        indexer: IndexerConfig | None = None,
    ):
        if dim < 2:
            raise ValueError("DSA test requires both latent KV and positional key")
        # dim=576 gives the reference's 512 latent + 64 positional components.
        # Smaller configurations retain the roles, not the real model shapes.
        self.rope_dim = max(1, dim // 9)
        self.latent_dim = dim - self.rope_dim
        self.dim = dim
        self.topk = topk
        self.indexer = DSAIndexer(
            dim, layers, topk, max_history, indexer or IndexerConfig()
        )

    def embed(self, tokens: list[int]) -> torch.Tensor:
        token = torch.tensor(tokens, device="cuda", dtype=torch.int64)[:, None]
        column = torch.arange(self.dim, device="cuda")[None, :]
        return (((token * 17 + column * 7) % 251).float() / 128 - 1).bfloat16()

    def norm(self, hidden: torch.Tensor) -> torch.Tensor:
        return (hidden.float() / (1 + hidden.float().abs())).bfloat16()

    def project(self, hidden: torch.Tensor, positions: torch.Tensor, layer: int):
        qr = self.indexer.query_latent(hidden, layer)
        attention_query = self.indexer.attention_query(qr, layer)
        q_nope = attention_query[:, : self.latent_dim].clone()
        q_pe = attention_query[:, self.latent_dim :].clone()
        latent = self.norm(hidden[:, : self.latent_dim] + (layer + 1) / 32)
        positional = self.norm(
            hidden[:, self.latent_dim :].float()
            + ((positions[:, None] * 13) % 97).float() / 128
        )
        return Projection(q_nope, q_pe, torch.cat((latent, positional), dim=-1), qr)

    def absorb_query(self, q_nope):
        """Absorb the KV expansion projection into Q for unified Extend (identity stub)."""
        return q_nope.clone()

    def select_batch(self, positions, hidden, qr, layer, sequences):
        """Score actual Indexer Q/K and return the layer's packed causal Top-K."""
        return self.indexer.select_batch(positions, hidden, qr, layer, sequences)

    def consume(self, q_nope, q_pe, packed, indices):
        """Consume packed KV for the whole layer, regardless of cached length."""
        key = packed[:, : self.latent_dim]
        value = key
        positional = packed[:, self.latent_dim :]
        output = torch.empty(
            (len(q_nope), self.dim), device=packed.device, dtype=packed.dtype
        )
        tile = max(1, min(256, (4 * 1024 * 1024) // (indices.shape[1] * self.dim)))
        for start in range(0, len(indices), tile):
            stop = min(start + tile, len(indices))
            selected = indices[start:stop]
            valid = (selected >= 0)[:, :, None]
            # torch.mean scales the reduction by a reciprocal. Match that
            # rounding, including BF16 half-way cases, across query batch sizes.
            inverse_counts = valid.sum(dim=1).float().reciprocal()
            selected = selected.clamp_min(0)

            def mean(component):
                return (
                    component[selected].float().masked_fill(~valid, 0).sum(dim=1)
                    * inverse_counts
                )

            latent_output = mean(value) + q_nope[start:stop].float() / 8
            positional_output = mean(positional) + q_pe[start:stop].float() / 8
            latent_output += mean(key) / 16
            output[start:stop] = torch.cat(
                (latent_output, positional_output), dim=1
            ).bfloat16()
        return output

    def block_output(self, attention, residual):
        residual = (residual + attention).bfloat16()
        normalized = self.norm(residual)
        # Placeholder for the block's MLP/MoE; output feeds the next block.
        hidden = (normalized.float() * 0.5).bfloat16()
        return hidden, residual

    def final_norm(self, hidden, residual):
        """Finish the backbone for every current input position; no LM head."""
        return self.norm(hidden + residual)
