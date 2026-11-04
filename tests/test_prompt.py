"""Input assembly must preserve supplied tokens without inserting a template."""

import copy

import pytest

from cxl_recsys.common.request import placeholder
from cxl_recsys.worker.prompt import build_prompt


def test_prompt_contains_exactly_the_supplied_tokens():
    request = placeholder(history_length=3)
    original = copy.deepcopy(request)
    prompt = build_prompt(request["history_token_ids"], request["candidates"])

    assert prompt.input_ids == [1, 2, 3, 101, 201, 102, 202]
    assert prompt.positions == list(range(7))
    assert prompt.last_input_position == 6
    assert prompt.history_length == 3
    assert prompt.candidate_tokens == 4
    assert request == original
    for removed in (
        "roles",
        "labels",
        "control_tokens",
        "response_position",
        "output_token_ids",
    ):
        assert not hasattr(prompt, removed)


@pytest.mark.parametrize("count", [1, 2, 27, 64, 128])
def test_candidate_count_does_not_add_label_or_separator_tokens(count):
    request = placeholder(history_length=3, candidates=count)
    expected = request["history_token_ids"] + [
        token for candidate in request["candidates"] for token in candidate["token_ids"]
    ]
    prompt = build_prompt(request["history_token_ids"], request["candidates"])

    assert prompt.input_ids == expected
    assert len(prompt.input_ids) == 3 + 2 * count
    assert prompt.positions == list(range(len(expected)))
    assert prompt.candidate_tokens == 2 * count


def test_arbitrary_supplied_tokens_are_preserved_without_mutation_or_aliasing():
    history = [0, 42, 42]
    candidates = [
        {"item_id": "already-formatted-a", "token_ids": [7]},
        {"item_id": "already-formatted-b", "token_ids": [42, 0, 2**31 - 1]},
    ]
    original = copy.deepcopy((history, candidates))
    prompt = build_prompt(history, candidates)

    assert prompt.input_ids == [0, 42, 42, 7, 42, 0, 2**31 - 1]
    assert prompt.positions == list(range(7))
    assert prompt.candidate_tokens == 4
    assert (history, candidates) == original
    prompt.input_ids[0] = 100
    prompt.input_ids[3] = 200
    assert (history, candidates) == original


def test_candidate_order_is_preserved_without_relabeling():
    request = placeholder(history_length=1)
    prompt = build_prompt(
        request["history_token_ids"], list(reversed(request["candidates"]))
    )
    assert prompt.input_ids == [1, 102, 202, 101, 201]
    assert prompt.last_input_position == 4


@pytest.mark.parametrize("count", [0, 129])
def test_prompt_rejects_unsupported_candidate_count(count):
    request = placeholder(candidates=count)
    with pytest.raises(ValueError):
        build_prompt(request["history_token_ids"], request["candidates"])


def test_flat_candidate_suffix_is_preserved_without_splitting_or_insertions():
    history = [0, 42, 42]
    suffix = [7, 0, 2**31 - 1] * 171
    original = copy.deepcopy((history, suffix))
    prompt = build_prompt(history, candidate_token_ids=suffix)
    assert prompt.input_ids == history + suffix
    assert prompt.positions == list(range(516))
    assert prompt.history_length == 3
    assert prompt.candidate_tokens == 513
    assert prompt.last_input_position == 515
    prompt.input_ids[0] = 5
    prompt.input_ids[3] = 6
    assert (history, suffix) == original


@pytest.mark.parametrize("suffix", [[], [0] * 32769])
def test_prompt_rejects_empty_or_oversized_flat_suffix(suffix):
    with pytest.raises(ValueError, match="Candidate suffix"):
        build_prompt([1], candidate_token_ids=suffix)


def test_prompt_rejects_mixing_legacy_and_flat_candidates():
    with pytest.raises(ValueError, match="cannot be mixed"):
        build_prompt(
            [1], [{"item_id": "item", "token_ids": [2]}], candidate_token_ids=[3]
        )
