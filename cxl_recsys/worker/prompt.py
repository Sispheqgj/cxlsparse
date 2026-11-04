"""Flatten request tokens without adding instructions, labels or delimiters.

The supplied tokens already encode the task. Worker preserves their values and
order and only supplies absolute positions and the reusable history boundary.
"""

from dataclasses import dataclass

from ..common.request import MAX_CANDIDATE_TOKENS, MAX_CANDIDATES


@dataclass
class PreparedInput:
    input_ids: list[int]
    positions: list[int]
    history_length: int
    candidate_tokens: int

    @property
    def last_input_position(self) -> int:
        return self.positions[-1]


def build_prompt(
    history, candidates=None, *, candidate_token_ids=None
) -> PreparedInput:
    """History + supplied candidate tokens in their original order, unchanged.

    No injected tokens or candidate branches. Any instruction/template tokens
    must already be present in the supplied input; Worker does not infer them.
    Return the complete logical sequence. The model validates and skips any
    cached prefix; the Worker never needs to inspect the model's page tables.
    """
    token_ids = list(history)
    if candidate_token_ids is not None:
        if candidates is not None:
            raise ValueError("candidates cannot be mixed with candidate_token_ids")
        if not isinstance(candidate_token_ids, list) or not (
            1 <= len(candidate_token_ids) <= MAX_CANDIDATE_TOKENS
        ):
            raise ValueError(
                f"Candidate suffix must have 1..{MAX_CANDIDATE_TOKENS} tokens"
            )
        token_ids.extend(candidate_token_ids)
    else:
        if candidates is None or not 1 <= len(candidates) <= MAX_CANDIDATES:
            raise ValueError(f"Candidate count must be in 1..{MAX_CANDIDATES}")
        for candidate in candidates:
            token_ids.extend(candidate["token_ids"])
    return PreparedInput(
        input_ids=token_ids,
        positions=list(range(len(token_ids))),
        history_length=len(history),
        candidate_tokens=len(token_ids) - len(history),
    )
