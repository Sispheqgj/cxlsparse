"""Independent numerical, causal, lifetime and batch tests for the DSA Indexer."""

import math
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as functional

from cxl_recsys.cli import parse_args
from cxl_recsys.models.simulation.config import IndexerConfig
from cxl_recsys.models.simulation.indexer import DSAIndexer, IndexerSequence


def small_config(**overrides):
    return replace(
        IndexerConfig(n_heads=4, head_dim=8, q_lora_rank=8, rope_dim=4, query_tile=3),
        **overrides,
    )


def indexer(topk=4, **overrides):
    return DSAIndexer(8, 2, topk, 64, small_config(**overrides))


@pytest.mark.parametrize(
    "overrides",
    [
        {"n_heads": 0},
        {"head_dim": 7},
        {"q_lora_rank": -1},
        {"rope_dim": 3},
        {"rope_dim": 16},
        {"rope_theta": 1},
        {"rope_factor": 0.5},
        {"beta_fast": 1},
        {"beta_slow": float("nan")},
        {"query_tile": 0},
        {"seed": -1},
        {"seed": 2**63},
        {"original_seq_len": 0},
    ],
)
def test_invalid_indexer_configuration(overrides):
    with pytest.raises(ValueError):
        small_config(**overrides)


def test_indexer_config_cli_overrides_yaml(tmp_path):
    path = tmp_path / "indexer.yml"
    path.write_text(
        "worker:\n  indexer_n_heads: 8\n  indexer_head_dim: 64\n  indexer_rope_dim: 32\n  indexer_seed: 17\n"
    )
    args = parse_args(["worker", "--config", str(path), "--indexer-n-heads", "4"])
    assert (args.indexer_n_heads, args.indexer_head_dim, args.indexer_rope_dim) == (
        4,
        64,
        32,
    )
    assert args.indexer_seed == 17
    assert args.indexer_query_tile == 32


@pytest.mark.gpu
def test_weights_initialized_once_reproducible_and_layer_specific():
    state = torch.random.get_rng_state().clone()
    left, right, changed = indexer(), indexer(), indexer(seed=1)
    assert torch.equal(torch.random.get_rng_state(), state)
    for name in ("query_a", "query_b", "key", "head_weights", "attention_query"):
        assert torch.equal(
            getattr(left.layers[0], name), getattr(right.layers[0], name)
        )
        assert not torch.equal(
            getattr(left.layers[0], name), getattr(left.layers[1], name)
        )
        assert not torch.equal(
            getattr(left.layers[0], name), getattr(changed.layers[0], name)
        )
        assert not getattr(left.layers[0], name).requires_grad


@pytest.mark.gpu
def test_projection_matches_explicit_rope_hadamard_and_normalization():
    operator = indexer()
    hidden = torch.arange(40, device="cuda").reshape(5, 8).bfloat16() / 32
    positions = torch.tensor([0, 1, 7, 13, 33], device="cuda")
    weights = operator.layers[0]
    projected = functional.linear(hidden, weights.query_a).float()
    expected_qr = (
        projected / (projected.square().mean(-1, keepdim=True) + 1e-6).sqrt()
    ).bfloat16()
    qr = operator.query_latent(hidden, 0)
    torch.testing.assert_close(qr, expected_qr)
    query, keys, gates = operator.project(hidden, qr, positions, 0)
    raw_query = functional.linear(qr, weights.query_b).view(5, 4, 8)
    raw_key = functional.layer_norm(
        functional.linear(hidden, weights.key).float(), (8,), eps=1e-6
    ).bfloat16()

    def reference_rotation(values):
        # Use complex arithmetic, independently of the implementation's sin/cos
        # pair expressions. Only the first rope_dim dimensions rotate.
        paired = (
            values[..., :4]
            .float()
            .reshape(*values.shape[:-1], 2, 2)
            .transpose(-1, -2)
            .contiguous()
        )
        complex_values = torch.view_as_complex(paired)
        angles = positions[:, None].float() / torch.tensor([1, 100], device="cuda")
        frequencies = torch.polar(torch.ones_like(angles), angles)
        if values.ndim == 3:
            frequencies = frequencies[:, None]
        rotated = torch.view_as_real(complex_values * frequencies)
        rotated = torch.cat(
            (rotated[..., 0], rotated[..., 1], values[..., 4:].float()), -1
        ).bfloat16()
        result = rotated.float() @ operator.hadamard.float()
        return (result / math.sqrt(8)).bfloat16()

    torch.testing.assert_close(query, reference_rotation(raw_query), rtol=0, atol=0)
    torch.testing.assert_close(keys, reference_rotation(raw_key), rtol=0, atol=0)
    expected_gates = functional.linear(
        hidden.float(), weights.head_weights
    ) / math.sqrt(4 * 8)
    torch.testing.assert_close(gates, expected_gates)


@pytest.mark.gpu
def test_yarn_frequencies_follow_reference_and_remain_fixed():
    config = small_config(head_dim=128, rope_dim=64)
    short = DSAIndexer(8, 1, 4, 128, config)
    long = DSAIndexer(8, 1, 4, 8192, config)
    frequencies = 1 / config.rope_theta ** (torch.arange(0, 64, 2).float() / 64)

    def correction(rotations):
        return 64 * math.log(4096 / (rotations * 2 * math.pi)) / (2 * math.log(10000))

    low, high = max(math.floor(correction(32)), 0), min(math.ceil(correction(1)), 63)
    smooth = 1 - ((torch.arange(32) - low) / (high - low)).clamp(0, 1)
    expected = frequencies / 40 * (1 - smooth) + frequencies * smooth
    torch.testing.assert_close(short.freqs.cpu(), frequencies)
    torch.testing.assert_close(long.freqs.cpu(), expected)


@pytest.mark.gpu
def test_scores_relu_before_signed_head_weight_reduction():
    operator = indexer()
    query = torch.tensor(
        [[[[1.0, -1.0], [-1.0, 2.0]], [[2.0, 0.0], [0.0, -1.0]]]],
        device="cuda",
        dtype=torch.bfloat16,
    )
    keys = torch.tensor(
        [[[2.0, 1.0], [-1.0, 3.0], [0.0, -2.0]]], device="cuda", dtype=torch.bfloat16
    )
    gates = torch.tensor([[[1.0, -2.0], [-1.0, 0.5]]], device="cuda")
    actual = operator._scores(query, keys, gates)
    expected = torch.empty((1, 2, 3), device="cuda")
    for t in range(2):
        for j in range(3):
            expected[0, t, j] = sum(
                gates[0, t, h]
                * torch.dot(query[0, t, h].float(), keys[0, j].float()).clamp_min(0)
                for h in range(2)
            )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert bool((actual < 0).any())  # Gates are NOT softmaxed or rectified.


@pytest.mark.gpu
@pytest.mark.parametrize("topk", [1, 4, 8, 128])
def test_topk_is_unique_causal_padded_and_has_stable_ties(topk):
    operator = indexer(topk=topk)
    scores = torch.tensor(
        [
            [
                [9.0, 9.0, 8.0, 7.0, 7.0],
                [2.0, 3.0, 4.0, 5.0, 6.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ],
        device="cuda",
    )
    positions = torch.tensor([[4, 1, 2]], device="cuda")
    actual = operator._topk(scores.clone(), positions, torch.arange(5, device="cuda"))
    expected = []
    for row, position in zip(scores[0].tolist(), positions[0].tolist()):
        chosen = sorted(range(position + 1), key=lambda j: (-row[j], j))[:topk]
        expected.append(sorted(chosen) + [-1] * (topk - len(chosen)))
    assert actual[0].tolist() == expected
    # Masked future positions cannot perturb the chosen prefix, even on ties.
    extended = functional.pad(scores, (0, 13), value=1000)
    assert torch.equal(
        actual, operator._topk(extended, positions, torch.arange(18, device="cuda"))
    )


@pytest.mark.gpu
def test_ragged_batch_scoring_matches_reference_and_preserves_history(monkeypatch):
    operator = indexer(topk=4)
    hidden = torch.arange(56, device="cuda").reshape(7, 8).bfloat16() / 64
    positions = torch.tensor([2, 3, 4, 0, 1, 2, 3], device="cuda")
    qr = operator.query_latent(hidden, 0)
    query, new_keys, gates = operator.project(hidden, qr, positions, 0)
    first = torch.full((64, 8), -17, device="cuda", dtype=torch.bfloat16)
    second = torch.full_like(first, -23)
    first[:2] = torch.arange(16, device="cuda").reshape(2, 8)
    old = first[:2].clone()
    sequences = [
        IndexerSequence(first, 2, 4, 0, 3),
        IndexerSequence(second, 0, 2, 3, 7),
    ]
    expected_rows = []
    for start, end, keys in (
        (0, 3, torch.cat((old, new_keys[:3]))),
        (3, 7, new_keys[3:]),
    ):
        logits = torch.einsum("qhd,nd->qhn", query[start:end].float(), keys.float())
        scores = (logits.relu() * gates[start:end, :, None]).sum(1)
        for offset, values in enumerate(scores.tolist()):
            position = positions[start + offset].item()
            chosen = sorted(range(int(position) + 1), key=lambda j: (-values[j], j))[:4]
            expected_rows.append(sorted(chosen) + [-1] * (4 - len(chosen)))
    seen = []
    score = operator._scores

    def bounded_scores(query, keys, weights):
        seen.append(query.shape)
        assert query.shape[0] == 2
        assert query.shape[1] <= operator.config.query_tile
        return score(query, keys, weights)

    monkeypatch.setattr(operator, "_scores", bounded_scores)
    actual = operator.select_batch(positions, hidden, qr, 0, sequences)
    assert actual.tolist() == expected_rows
    assert len(seen) == 2
    assert torch.equal(first[:2], old)
    assert torch.equal(first[2:4], new_keys[:2])
    assert torch.equal(second[:2], new_keys[3:5])
    assert bool((first[4:] == -17).all())
    assert bool((second[2:] == -23).all())


@pytest.mark.gpu
def test_model_persists_only_history_keys_and_does_not_resample_or_jit(monkeypatch):
    from cxl_recsys.common.request import placeholder
    from cxl_recsys.mempool import MemoryPool
    from cxl_recsys.models.simulation.model import TestModel
    from cxl_recsys.worker.inference import execute_request

    model = TestModel(
        MemoryPool(8, 64), max_users=1, max_history=128, indexer=small_config()
    )
    captured = []
    project = model.ops.indexer.project

    def capture(hidden, qr, positions, layer):
        result = project(hidden, qr, positions, layer)
        captured.append((layer, result[1].clone()))
        return result

    def forbidden(*args, **kwargs):
        raise AssertionError("Forward must not generate random values or compile")

    monkeypatch.setattr(model.ops.indexer, "project", capture)
    for name in ("rand", "randn", "randint", "randperm", "compile"):
        monkeypatch.setattr(torch, name, forbidden)
    try:
        execute_request(model, placeholder(history_length=3))
        before = []
        for layer in range(2):
            buffer = model.users["user-0"]["index_keys"][layer]
            assert torch.equal(buffer[:3], captured[layer][1][:3])
            with torch.inference_mode():
                buffer[3:].fill_(-19)
            before.append(buffer[:3].clone())
        result = execute_request(model, placeholder(history_length=5))
        assert result["indexer"]["mode"] == "dsa_random_weights"
        assert result["indexer"]["history_key_bytes"] == 5 * 2 * 8 * 2
        for layer in range(2):
            buffer = model.users["user-0"]["index_keys"][layer]
            assert torch.equal(buffer[:3], before[layer])
            assert torch.equal(buffer[3:5], captured[2 + layer][1][:2])
            assert bool((buffer[5:] == -19).all())
            before[layer] = buffer.clone()
        execute_request(model, placeholder(history_length=5))
        for layer in range(2):
            assert torch.equal(
                model.users["user-0"]["index_keys"][layer], before[layer]
            )
    finally:
        model.close()
