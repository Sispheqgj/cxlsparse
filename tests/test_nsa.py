"""NSA backbone numerics, packed prefix reuse, output mapping and preflight."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer, decoders, models, pre_tokenizers

from cxl_recsys.cli import parse_args
from cxl_recsys.models.base import CacheContext
from cxl_recsys.models.nsa.config import NSAConfig, prepare_nsa
from cxl_recsys.worker.inference import execute_batch


@pytest.fixture
def checkpoint(tmp_path):
    from safetensors.torch import save_file

    cfg = NSAConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=32,
        num_key_value_heads=2,
        head_dim=64,
        vocab_size=256,
        rms_norm_eps=1e-5,
        rope_theta=500000.0,
        max_position_embeddings=2048,
        block_size=16,
        block_counts=16,
        window_size=128,
        tie_word_embeddings=True,
    )
    raw = {
        **asdict(cfg),
        "architectures": ["LlamaNSAForCausalLM"],
        "torch_dtype": "bfloat16",
        "hidden_act": "silu",
    }
    (tmp_path / "config.json").write_text(json.dumps(raw))
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    tokenizer = Tokenizer(models.BPE({c: i for i, c in enumerate(alphabet)}, []))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    rng = torch.Generator().manual_seed(12)

    def normal(*shape):
        return (torch.randn(shape, generator=rng) * 0.05).bfloat16()

    weights = {
        "model.embed_tokens.weight": normal(256, 32),
        "model.norm.weight": torch.ones(32).bfloat16(),
    }
    for layer in range(2):
        p = f"model.layers.{layer}."
        for name in ("input_layernorm.weight", "post_attention_layernorm.weight"):
            weights[p + name] = torch.ones(32).bfloat16()
        for name, shape in {
            "self_attn.q_proj.weight": (2048, 32),
            "self_attn.k_proj.weight": (128, 32),
            "self_attn.v_proj.weight": (128, 32),
            "self_attn.o_proj.weight": (32, 2048),
            "self_attn.g_proj.weight": (96, 32),
            "mlp.gate_proj.weight": (64, 32),
            "mlp.up_proj.weight": (64, 32),
            "mlp.down_proj.weight": (32, 64),
        }.items():
            weights[p + name] = normal(*shape)
    save_file(weights, tmp_path / "model.safetensors")
    return tmp_path, cfg, weights, tokenizer


def test_config_preflight_and_build_geometry(checkpoint):
    path, cfg, _, _ = checkpoint
    args = parse_args(
        [
            "worker",
            "--model",
            "nsa_1b",
            "--model-path",
            str(path),
            "--max-users",
            "2",
            "--max-history",
            "1024",
            "--pages",
            "128",
        ]
    )
    assert prepare_nsa(args) == cfg
    assert (args.layers, args.dim, args.topk, args.hot) == (2, 128, 256, 256)
    args.pages = 127
    with pytest.raises(ValueError, match="at least 128"):
        prepare_nsa(args)
    args.model_path = None
    with pytest.raises(ValueError, match="model-path"):
        prepare_nsa(args)
    build = parse_args(
        ["build-kernels", "--model", "nsa_1b", "--model-path", str(path)]
    )
    assert prepare_nsa(build) == cfg


@pytest.mark.parametrize(
    "field,value",
    [
        ("architectures", ["LlamaForCausalLM"]),
        ("rope_scaling", {"factor": 4}),
        ("head_dim", 128),
        ("num_attention_heads", 8),
        ("block_counts", 32),
        ("block_size", 15),
        ("window_size", 0),
        ("rms_norm_eps", 0),
        ("tie_word_embeddings", 1),
    ],
)
def test_unsupported_checkpoint_is_rejected(checkpoint, field, value):
    path, _, _, _ = checkpoint
    target = path / "config.json"
    raw = json.loads(target.read_text())
    raw[field] = value
    target.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        NSAConfig.load(path)


def make_model(path, cfg, backend="dram", **options):
    from cxl_recsys.mempool import MemoryPool
    from cxl_recsys.models.nsa import NSAModel

    pool = MemoryPool(cfg.required_pages(2, 1024, 64), cfg.record_dim)
    if backend == "cxl_registered":
        # Registered anonymous memory exercises the direct CUDA transport.
        # It does not represent physical CXL/devdax validation.
        pool.backend = "cxl"
    return NSAModel(
        pool,
        max_users=2,
        max_history=1024,
        layers=cfg.num_hidden_layers,
        model_path=path,
        **options,
    )


def dense_oracle(ids, cfg, weights):
    """Independent resident full-matrix oracle where all causal blocks fit."""
    assert len(ids) <= cfg.selected_tokens
    w = {k: v.cuda() for k, v in weights.items()}
    hidden = F.embedding(
        torch.tensor(ids, device="cuda"), w["model.embed_tokens.weight"]
    )
    length, heads, group, dim = len(ids), 2, 16, 64
    pos = torch.arange(length, device="cuda")
    phase = pos.float()[:, None] / (
        cfg.rope_theta ** (torch.arange(0, dim, 2, device="cuda").float() / dim)
    )
    phase = torch.cat((phase, phase), -1)[:, None]

    def norm(x, weight):
        return (
            x.float()
            * (x.float().square().mean(-1, keepdim=True) + cfg.rms_norm_eps).rsqrt()
        ).bfloat16() * weight

    def attention(q, k, v, mask):
        score = (
            torch.einsum(
                "qhd,khd->hqk", q.float(), k.repeat_interleave(group, 1).float()
            )
            / dim**0.5
        )
        probability = score.masked_fill(~mask, -torch.inf).softmax(-1).nan_to_num()
        return torch.einsum(
            "hqk,khd->qhd", probability, v.repeat_interleave(group, 1).float()
        ).bfloat16()

    for layer in range(cfg.num_hidden_layers):
        p = f"model.layers.{layer}."
        x = norm(hidden, w[p + "input_layernorm.weight"])
        q = F.linear(x, w[p + "self_attn.q_proj.weight"]).view(length, 32, dim)
        k = F.linear(x, w[p + "self_attn.k_proj.weight"]).view(length, heads, dim)
        v = F.linear(x, w[p + "self_attn.v_proj.weight"]).view_as(k)
        q, k = [
            a * phase.cos().bfloat16()
            + torch.cat((-a[..., dim // 2 :], a[..., : dim // 2]), -1)
            * phase.sin().bfloat16()
            for a in (q, k)
        ]
        gates = (
            F.linear(x, w[p + "self_attn.g_proj.weight"]).view(length, 32, 3).sigmoid()
        )
        complete = length // 16
        ck = (
            k[: complete * 16]
            .reshape(complete, 16, heads, dim)
            .float()
            .mean(1)
            .bfloat16()
        )
        cv = (
            v[: complete * 16]
            .reshape(complete, 16, heads, dim)
            .float()
            .mean(1)
            .bfloat16()
        )
        cmp = (
            attention(
                q,
                ck,
                cv,
                (torch.arange(complete, device="cuda") + 1) * 16 <= pos[:, None] + 1,
            )
            if complete
            else torch.zeros_like(q)
        )
        causal = pos[None] <= pos[:, None]
        slc = attention(q, k, v, causal)
        window = attention(q, k, v, causal & (pos[None] > pos[:, None] - 128))
        mixed = torch.addcmul(slc * gates[..., 1, None], cmp, gates[..., 0, None])
        mixed = torch.addcmul(mixed, window, gates[..., 2, None])
        hidden = hidden + F.linear(mixed.flatten(1), w[p + "self_attn.o_proj.weight"])
        x = norm(hidden, w[p + "post_attention_layernorm.weight"])
        hidden = hidden + F.linear(
            F.silu(F.linear(x, w[p + "mlp.gate_proj.weight"]))
            * F.linear(x, w[p + "mlp.up_proj.weight"]),
            w[p + "mlp.down_proj.weight"],
        )
    return norm(hidden, w["model.norm.weight"])


@pytest.mark.gpu
@pytest.mark.parametrize("length", [7, 17, 143])
def test_backbone_matches_independent_resident_oracle(checkpoint, length):
    path, cfg, weights, _ = checkpoint
    model = make_model(path, cfg)
    ids = [i % 256 for i in range(length)]
    try:
        actual = model.forward(ids, range(length), cache=CacheContext("u", length - 3))
        expected = dense_oracle(ids, cfg, weights)
        relative = (
            actual.float() - expected.float()
        ).square().mean().sqrt() / expected.float().square().mean().sqrt()
        assert relative < 0.008
        assert model.snapshot()["u"]["kv_length"] == length - 3
    finally:
        model.close()


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["dram", "cxl_registered"])
@pytest.mark.parametrize("cuda_graphs", [False, True])
def test_packed_append_cold_reuse_and_candidate_output(
    checkpoint, backend, cuda_graphs
):
    path, cfg, _, tokenizer = checkpoint
    model = make_model(path, cfg, backend, cuda_graphs=cuda_graphs)
    requests = []
    try:
        for visit, histories in enumerate(((15, 31), (17, 33), (129, 319), (129, 319))):
            requests = []
            for user, length in enumerate(histories):
                suffix = tokenizer.encode(
                    f"Candidates visit {visit}: (A) lamp (B) bottle (C) bag\nAnswer:\n",
                    add_special_tokens=False,
                ).ids
                requests.append(
                    dict(
                        request_id=f"r{visit}-{user}",
                        user_id=str(user),
                        history_token_ids=[(j + user * 7) % 256 for j in range(length)],
                        candidate_token_ids=suffix,
                        candidate_item_ids=[f"u{user}-v{visit}-{i}" for i in range(3)],
                    )
                )
            model.cache.invalidate()
            actual = execute_batch(model, list(reversed(requests)))
            fresh = make_model(path, cfg)
            try:
                expected = execute_batch(fresh, requests)
            finally:
                fresh.close()
            for request, cached, reference in zip(requests, reversed(actual), expected):
                assert cached["recommended_item_id"] == reference["recommended_item_id"]
                assert cached["recommended_item_id"] in request["candidate_item_ids"]
                assert cached["flow"] == "nsa_mean_pool" and cached["model"] == "nsa_1b"
            for user, length in enumerate(histories):
                state = model.users[str(user)]
                assert len(state["tokens"]) == length
                for summary in state["summaries"]:
                    assert len(summary["compressed"]) == length // 16
                    assert len(summary["tail"]) == length % 16
                    assert len(summary["window"]) == min(127, length)
        assert model.pool.stats()["read_bytes"] > 0
        if cuda_graphs:
            assert model.graphs is not None
            assert model.graphs.stats()["captures"] > 0
            assert model.graphs.stats()["replays"] > model.graphs.stats()["captures"]
        if backend == "cxl_registered":
            assert model.pool.stats()["read_kernel_launches"] > 0
        before = model.snapshot()
        bad = requests[0]
        bad["history_token_ids"][0] += 1
        with pytest.raises(ValueError, match="append-only"):
            execute_batch(model, [bad])
        assert model.snapshot() == before
        with pytest.raises(RuntimeError, match="failed"):
            execute_batch(model, [requests[1]])
    finally:
        model.close()


@pytest.mark.gpu
def test_graph_and_eager_hidden_states_match_after_prefix_growth(checkpoint):
    path, cfg, _, _ = checkpoint
    eager = make_model(path, cfg, cache_backend="reference")
    graphed = make_model(path, cfg, cuda_graphs=True, cache_backend="device")
    try:
        for history, candidate in ((17, 31), (17, 72), (129, 31), (129, 93)):
            ids = [i % 256 for i in range(history)] + [candidate] * 7
            context = CacheContext("u", history)
            actual = graphed.forward(ids, range(len(ids)), cache=context)
            expected = eager.forward(ids, range(len(ids)), cache=context)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert (
                eager.pool.stats()["write_bytes"] == graphed.pool.stats()["write_bytes"]
            )
    finally:
        eager.close()
        graphed.close()


def test_real_nsa_gr_prompt_and_single_token_answer_mapping():
    from cxl_recsys.scheduler.gr import generate_gr, gr_request

    path = Path("artifacts/models/NSA-1B/tokenizer.json")
    if not path.exists():
        pytest.skip("NSA tokenizer not downloaded")
    args = parse_args(
        [
            "scheduler",
            "--gr-tokenizer",
            str(path),
            "--gr-prompt-format",
            "nsa",
            "--gr-user-lengths",
            "512",
            "--gr-item-lengths",
            "128",
            "--gr-candidate-count",
            "3",
            "--users",
            "2",
            "--num-requests",
            "4",
            "--gr-sampling",
            "sequential",
        ]
    )
    tokenizer = Tokenizer.from_file(str(path))
    rows = list(generate_gr(args))
    for row in rows:
        request = gr_request(row, 0, args.max_history)
        text = tokenizer.decode(
            request["candidate_token_ids"], skip_special_tokens=False
        )
        assert text.endswith("\nAnswer:\n")
        for option in "ABC":
            encoded = tokenizer.encode(text + option, add_special_tokens=False).ids
            assert encoded[:-1] == request["candidate_token_ids"]
            assert (
                encoded[-1] == tokenizer.encode(option, add_special_tokens=False).ids[0]
            )
    assert (
        rows[0]["input_ids"][: rows[0]["stable_prefix_tokens"]]
        == rows[2]["input_ids"][: rows[2]["stable_prefix_tokens"]]
    )
