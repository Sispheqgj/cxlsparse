"""Metadata for the SSA authors' FLA mean-pooling NSA checkpoint."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

NSA_REVISION = "bd67af59b90afa34b25f61d2922e612d10dba3bd"
CHECKPOINT_REVISION = "3dc3694d2f873820122b094276ae626a44f00a36"


@dataclass(frozen=True)
class NSAConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    block_size: int
    block_counts: int
    window_size: int
    tie_word_embeddings: bool

    @classmethod
    def load(cls, model_path):
        path = Path(model_path)
        raw = json.loads((path / "config.json").read_text())
        if raw.get("architectures") != ["LlamaNSAForCausalLM"]:
            raise ValueError("NSA requires a LlamaNSAForCausalLM checkpoint")
        if raw.get("torch_dtype") != "bfloat16" or raw.get("hidden_act") != "silu":
            raise ValueError("NSA requires BF16 weights and the SiLU MLP")
        if raw.get("attention_bias", False) or raw.get("mlp_bias", False):
            raise ValueError("NSA bias projections are not supported")
        if raw.get("rope_scaling") is not None:
            raise ValueError("NSA currently requires unscaled, fixed RoPE")
        config = cls(**{name: raw[name] for name in cls.__dataclass_fields__})
        for name in cls.__dataclass_fields__:
            value = getattr(config, name)
            if name in ("rms_norm_eps", "rope_theta"):
                if (
                    not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value <= 0
                ):
                    raise ValueError(f"Invalid NSA {name}")
            elif name == "tie_word_embeddings":
                if type(value) is not bool:
                    raise ValueError("NSA tie_word_embeddings must be boolean")
            elif type(value) is not int or value <= 0:
                raise ValueError(f"Invalid NSA {name}")
        if config.head_dim != 64:
            raise ValueError("NSA currently requires head_dim=64")
        group = config.num_attention_heads // config.num_key_value_heads
        if config.num_attention_heads % (16 * config.num_key_value_heads) or group & (
            group - 1
        ):
            raise ValueError("FLA NSA requires a GQA group size divisible by 16")
        if (
            config.block_size < 16
            or config.block_size & (config.block_size - 1)
            or config.block_counts & (config.block_counts - 1)
            or config.block_counts > config.block_size
            or config.selected_tokens > 16384
            or config.window_size > 4096
        ):
            raise ValueError("Unsupported NSA block/selection/window geometry")
        if not (path / "model.safetensors").is_file():
            raise ValueError(f"Missing NSA model.safetensors in {path}")
        return config

    @property
    def record_dim(self):
        return 2 * self.head_dim

    @property
    def selected_tokens(self):
        return self.block_size * self.block_counts

    def required_pages(self, users, history, page_size):
        return (
            users
            * self.num_hidden_layers
            * self.num_key_value_heads
            * ((history + page_size - 1) // page_size)
        )


def prepare_nsa(args):
    """Check metadata and derive pool geometry before mapping backing memory."""
    if not args.model_path:
        raise ValueError("nsa_1b requires --model-path (local pinned checkpoint)")
    config = NSAConfig.load(args.model_path)
    args.dim = config.record_dim
    args.layers = config.num_hidden_layers
    args.topk = config.selected_tokens
    args.hot = max(args.hot, args.topk)
    if args.command == "worker":
        if (
            min(
                args.max_users,
                args.max_history,
                args.page_size,
                args.nsa_query_tile,
                args.nsa_graph_cache_size,
            )
            <= 0
        ):
            raise ValueError("NSA capacities and query tile must be positive")
        required = config.required_pages(
            args.max_users, args.max_history, args.page_size
        )
        if args.pages < required:
            raise ValueError(f"NSA backing pool needs at least {required} pages")
    return config
