"""Random-weight DSA Indexer with persistent BF16 history keys on the GPU.

Follows DeepSeek-V3.2-Exp Indexer at 87e509a2: low-rank Q, shared normalized K,
non-interleaved RoPE, normalized Hadamard, weighted ReLU logits and causal Top-K.
BF16 GEMMs/FP32 aggregation replace the reference's FP8 kernels on A100. This
is a numerical/flow prototype, not a trained Indexer or an FP8 reproduction.
Only precompiled PyTorch operators are used; there is no runtime JIT.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from cxl_recsys.models.simulation.config import IndexerConfig


@dataclass
class IndexerSequence:
    """One packed sequence's persistent key buffer and logical boundaries."""

    history_keys: torch.Tensor
    cached_length: int
    history_length: int
    query_start: int
    query_end: int


@dataclass
class IndexerWeights:
    query_a: torch.Tensor
    query_b: torch.Tensor
    key: torch.Tensor
    head_weights: torch.Tensor
    attention_query: torch.Tensor


class DSAIndexer:
    def __init__(
        self,
        hidden_dim: int,
        layers: int,
        topk: int,
        max_history: int,
        config: IndexerConfig,
        device="cuda",
    ):
        if min(hidden_dim, layers, topk, max_history) <= 0:
            raise ValueError("Invalid Indexer dimensions")
        self.config = config
        self.topk = topk
        self.device = torch.device(device)
        generator = torch.Generator(device="cpu").manual_seed(config.seed)

        def weight(output_dim, input_dim, dtype=torch.bfloat16):
            # Initialization is isolated from the process-global RNG, performed
            # once at startup, and shared by all users of this model instance.
            value = torch.randn(output_dim, input_dim, generator=generator)
            return (value / math.sqrt(input_dim)).to(device=self.device, dtype=dtype)

        self.layers = [
            IndexerWeights(
                query_a=weight(config.q_lora_rank, hidden_dim),
                query_b=weight(config.n_heads * config.head_dim, config.q_lora_rank),
                key=weight(config.head_dim, hidden_dim),
                head_weights=weight(config.n_heads, hidden_dim, torch.float32),
                attention_query=weight(hidden_dim, config.q_lora_rank),
            )
            for _ in range(layers)
        ]
        self.freqs = self._rope_frequencies(max_history)
        # A small, exact +/-1 Hadamard matrix lets a precompiled GEMM replace a
        # JIT transform dependency. Normalize AFTER the transform, as upstream.
        hadamard = torch.ones(1, 1)
        while len(hadamard) < config.head_dim:
            hadamard = torch.cat(
                (
                    torch.cat((hadamard, hadamard), 1),
                    torch.cat((hadamard, -hadamard), 1),
                ),
                0,
            )
        self.hadamard = hadamard.to(device=self.device, dtype=torch.bfloat16)

    def _rope_frequencies(self, max_history):
        config = self.config
        dim = config.rope_dim
        freqs = config.rope_theta ** (torch.arange(0, dim, 2).float() / dim)
        inverse = 1 / freqs
        # Keep the frequency profile fixed for the lifetime of the model. Never
        # change old keys' positional transform as individual users grow.
        if max_history > config.original_seq_len:

            def correction(rotations):
                return (
                    dim
                    * math.log(config.original_seq_len / (rotations * 2 * math.pi))
                    / (2 * math.log(config.rope_theta))
                )

            low = max(math.floor(correction(config.beta_fast)), 0)
            high = min(math.ceil(correction(config.beta_slow)), dim - 1)
            if low == high:
                high += 0.001
            ramp = ((torch.arange(dim // 2) - low) / (high - low)).clamp(0, 1)
            smooth = 1 - ramp
            inverse = inverse / config.rope_factor * (1 - smooth) + inverse * smooth
        return inverse.to(self.device)

    def query_latent(self, hidden, layer):
        """MLA's shared normalized low-rank query; weights are fixed per layer."""
        projected = functional.linear(hidden, self.layers[layer].query_a).float()
        return (
            projected * torch.rsqrt(projected.square().mean(-1, keepdim=True) + 1e-6)
        ).bfloat16()

    def attention_query(self, query_latent, layer):
        """Project the SAME latent into the surrounding test attention's width."""
        return functional.linear(query_latent, self.layers[layer].attention_query)

    def _rotate(self, value, positions):
        """Non-interleaved RoPE followed by normalized Hadamard rotation."""
        half = self.config.rope_dim // 2
        angles = positions.float()[:, None] * self.freqs[None, :]
        if value.ndim == 3:
            angles = angles[:, None, :]
        cosine, sine = angles.cos(), angles.sin()
        first, second = value[..., :half].float(), value[..., half : 2 * half].float()
        rotated = torch.cat(
            (
                first * cosine - second * sine,
                first * sine + second * cosine,
                value[..., 2 * half :].float(),
            ),
            dim=-1,
        ).bfloat16()
        # FP32 output of this small transform avoids BF16 rounding before the
        # normalization scale, while storing projected Q/K in BF16 afterwards.
        transformed = torch.mm(
            rotated.reshape(-1, self.config.head_dim),
            self.hadamard,
            out_dtype=torch.float32,
        ).view_as(rotated)
        return (transformed * self.config.head_dim**-0.5).bfloat16()

    def project(self, hidden, query_latent, positions, layer):
        """Generate Indexer Q/K and signed head weights for ALL current tokens."""
        config, weights = self.config, self.layers[layer]
        query = functional.linear(query_latent, weights.query_b).view(
            len(hidden), config.n_heads, config.head_dim
        )
        key = functional.linear(hidden, weights.key)
        key = functional.layer_norm(
            key.float(), (config.head_dim,), eps=1e-6
        ).bfloat16()
        query = self._rotate(query, positions)
        key = self._rotate(key, positions)
        head_weights = functional.linear(hidden.float(), weights.head_weights)
        head_weights *= (config.n_heads * config.head_dim) ** -0.5
        return query, key, head_weights

    def _scores(self, query, keys, head_weights):
        """Batched MQA logits; never allocate all queries' per-head scores."""
        batch, queries, heads, dim = query.shape
        flat_query = query.transpose(1, 2).reshape(batch, heads * queries, dim)
        logits = torch.bmm(flat_query, keys.transpose(1, 2), out_dtype=torch.float32)
        logits = logits.view(batch, heads, queries, keys.shape[1])
        logits.relu_()
        logits *= head_weights.transpose(1, 2).unsqueeze(-1)
        return logits.sum(dim=1)

    def _topk(self, scores, positions, key_positions):
        """Choose Top-K with deterministic logical-ID tie breaking.

        The first Top-K finds the cutoff score. Equal cutoff values use the
        smallest logical positions, so adding masked future positions cannot
        change a prefix's selection. The second Top-K compacts integer IDs;
        the returned set is in logical order, which HiSparse does not constrain.
        """
        valid = key_positions <= positions.unsqueeze(-1)
        scores.masked_fill_(~valid, float("-inf"))
        width = min(self.topk, scores.shape[-1])
        cutoff = scores.topk(width, dim=-1, sorted=False).values.amin(-1, keepdim=True)
        above = scores > cutoff
        equal = (scores == cutoff) & valid
        remaining = width - above.sum(-1, keepdim=True)
        selected = (above | (equal & (equal.cumsum(-1) <= remaining))) & valid
        sentinel = torch.iinfo(torch.int64).max
        logical = torch.where(selected, key_positions, sentinel)
        indices = logical.topk(width, dim=-1, largest=False, sorted=True).values
        indices.masked_fill_(indices == sentinel, -1)
        return functional.pad(indices, (0, self.topk - width), value=-1)

    def select_batch(self, positions, hidden, query_latent, layer, sequences):
        """Append only history K, then score a padded/ragged batch on the GPU.

        Per-sequence loops copy/describe buffers only. Projection and score
        computation are batched. Candidate K is temporary and never copied into
        persistent history_keys; each query masks its own future positions.
        """
        query, current_keys, head_weights = self.project(
            hidden, query_latent, positions, layer
        )
        config = self.config
        batch = len(sequences)
        max_queries = max(item.query_end - item.query_start for item in sequences)
        max_length = max(
            item.cached_length + item.query_end - item.query_start for item in sequences
        )
        keys = torch.zeros(
            (batch, max_length, config.head_dim),
            dtype=torch.bfloat16,
            device=self.device,
        )
        queries = query.new_zeros((batch, max_queries, config.n_heads, config.head_dim))
        weights = head_weights.new_zeros((batch, max_queries, config.n_heads))
        query_positions = positions.new_full((batch, max_queries), -1)
        for row, item in enumerate(sequences):
            begin, end = item.query_start, item.query_end
            count = end - begin
            cached, history = item.cached_length, item.history_length
            # Both cached prefixes and new K stay in HBM. Full MLA KV is not
            # read from CXL to compute Indexer scores.
            keys[row, :cached].copy_(item.history_keys[:cached])
            keys[row, cached : cached + count].copy_(current_keys[begin:end])
            item.history_keys[cached:history].copy_(
                current_keys[begin : begin + history - cached]
            )
            queries[row, :count].copy_(query[begin:end])
            weights[row, :count].copy_(head_weights[begin:end])
            query_positions[row, :count].copy_(positions[begin:end])
        key_positions = torch.arange(max_length, device=self.device, dtype=torch.int64)
        output = positions.new_empty((batch, max_queries, self.topk))
        # Tiling queries bounds the B*H*Q*N workspace. Keep the layer's final
        # [total_queries, topk] matrix on GPU for a single HiSparse invocation.
        for begin in range(0, max_queries, config.query_tile):
            end = min(begin + config.query_tile, max_queries)
            scores = self._scores(queries[:, begin:end], keys, weights[:, begin:end])
            output[:, begin:end] = self._topk(
                scores, query_positions[:, begin:end], key_positions
            )
        return torch.cat(
            [
                output[row, : item.query_end - item.query_start]
                for row, item in enumerate(sequences)
            ]
        )
