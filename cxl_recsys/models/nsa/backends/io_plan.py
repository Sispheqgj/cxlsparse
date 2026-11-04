"""CPU planning and independent references for selected-branch NSA IO.

A plan belongs to exactly one layer and one user/history. Physical block keys
are (KV head, absolute block index); combining different users' or layers'
backing stores in a plan is unsupported. Logical selections retain duplicates
and per-query masks even when their physical transfers are deduplicated.
"""

from dataclasses import dataclass
from typing import Iterable

import torch

_INTEGERS = (torch.int32, torch.int64)
_INT32_MAX = torch.iinfo(torch.int32).max


@dataclass(frozen=True)
class BlockStage:
    """Compact physical block IDs, retaining every valid logical occurrence."""

    ids: torch.Tensor
    counts: torch.Tensor


@dataclass(frozen=True)
class SparseBlockPlan:
    """Whole-layer resident stage followed by independently consumable chunks."""

    queries: int
    heads: int
    history_blocks: int
    block_size: int
    block_keys: tuple[tuple[int, int], ...]
    valid_tokens: torch.Tensor
    resident_blocks: int
    chunk_ranges: tuple[tuple[int, int], ...]
    stages: tuple[BlockStage, ...]
    starts: torch.Tensor
    row_positions: torch.Tensor
    all_stage: BlockStage
    selected_token_occurrences: int

    def stats(self, element_size: int = 2) -> dict[str, int | float]:
        """JSON-ready accounting; BF16 by default, actual DMA counted by runtime.

        Payload is the minimum permitted prefix in each required physical block.
        Staging includes zero padding; copying a whole staging chunk would
        transfer that padding too and must be reported as such by the runtime.
        Metadata bytes count all materialized CPU tensors, including all_stage.
        """
        if not isinstance(element_size, int) or isinstance(element_size, bool):
            raise ValueError("element_size must be a positive integer")
        if element_size <= 0:
            raise ValueError("element_size must be a positive integer")
        unique = len(self.block_keys)
        tokens = int(self.valid_tokens.sum())
        resident_tokens = int(self.valid_tokens[: self.resident_blocks].sum())
        missing_tokens = tokens - resident_tokens
        occurrences = int(self.all_stage.counts.sum())
        missing_blocks = unique - self.resident_blocks
        row_bytes = 128 * element_size
        tensors = [self.valid_tokens, self.starts, self.row_positions]
        for stage in (*self.stages, self.all_stage):
            tensors.extend((stage.ids, stage.counts))
        return {
            "queries": self.queries,
            "heads": self.heads,
            "selected_block_occurrences": occurrences,
            "selected_token_occurrences": self.selected_token_occurrences,
            "unique_blocks": unique,
            "resident_blocks": self.resident_blocks,
            "missing_blocks": missing_blocks,
            "chunks": len(self.chunk_ranges),
            "unique_tokens": tokens,
            "resident_tokens": resident_tokens,
            "missing_tokens": missing_tokens,
            "payload_bytes": tokens * row_bytes,
            "resident_payload_bytes": resident_tokens * row_bytes,
            "missing_payload_bytes": missing_tokens * row_bytes,
            "staging_bytes": unique * self.block_size * row_bytes,
            "staging_padding_bytes": (unique * self.block_size - tokens) * row_bytes,
            "missing_staging_bytes": missing_blocks * self.block_size * row_bytes,
            "missing_padding_bytes": (missing_blocks * self.block_size - missing_tokens)
            * row_bytes,
            "metadata_bytes": sum(t.numel() * t.element_size() for t in tensors),
            "block_reuse_factor": occurrences / unique if unique else 0.0,
            "token_reuse_factor": (
                self.selected_token_occurrences / tokens if tokens else 0.0
            ),
            "resident_unique_fraction": self.resident_blocks / unique
            if unique
            else 0.0,
        }


def _cpu_integer(tensor: torch.Tensor, name: str, ndim: int) -> None:
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.device.type != "cpu"
        or tensor.dtype not in _INTEGERS
        or tensor.ndim != ndim
    ):
        raise ValueError(f"{name} must be a CPU int32/int64 tensor of rank {ndim}")


def _positive_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _selection_inputs(
    selections: torch.Tensor,
    positions: torch.Tensor,
    heads: int,
    history_blocks: int,
    block_size: int,
) -> None:
    _positive_integer(block_size, "block_size")
    _cpu_integer(selections, "selections", 3)
    _cpu_integer(positions, "positions", 1)
    if heads <= 0 or selections.shape[1] != heads:
        raise ValueError(
            "selections KV-head dimension must match nonempty backing heads"
        )
    if positions.shape[0] != selections.shape[0]:
        raise ValueError("positions must contain one absolute position per query")
    if bool(((selections < -1) | (selections >= history_blocks)).any()):
        raise ValueError("selection must be -1 or a block inside the backing history")
    if bool(((positions < -1) | (positions > _INT32_MAX)).any()):
        raise ValueError("positions must lie in [-1, int32 maximum]")
    if history_blocks * block_size > _INT32_MAX:
        raise ValueError("history token capacity must fit int32 addressing")


def _stage(rows: Iterable[Iterable[int]]) -> BlockStage:
    lists = [list(row) for row in rows]
    width = max(1, max((len(row) for row in lists), default=0))
    counts = torch.tensor([len(row) for row in lists], dtype=torch.int32)
    # One conversion avoids hundreds of tiny tensor allocations and slice
    # assignments per stage. The zero-row shape still has a positive width.
    ids = torch.tensor(
        [row + [-1] * (width - len(row)) for row in lists], dtype=torch.int32
    ).reshape(len(lists), width)
    return BlockStage(ids=ids, counts=counts)


def plan_blocks(
    selections: torch.Tensor,
    positions: torch.Tensor,
    resident: torch.Tensor,
    block_size: int,
    chunk_blocks: int,
) -> SparseBlockPlan:
    """Plan a single user's single-layer selected union without dense KV IO.

    selections is [queries, KV-heads, selected-block-slots], positions contains
    absolute causal positions, and resident is [KV-heads, history-blocks]. The
    sentinel -1 and blocks starting after a query's position do not participate.
    Other invalid IDs are rejected even when they would be causally masked.

    Physical IDs follow first-use order, stably partitioned resident then
    missing. Stage zero spans every query/head's resident work. Every later
    stage depends on exactly one missing chunk. A shared tail block is copied
    through the longest required prefix; consumers must still apply their own
    row_positions against starts, rather than using valid_tokens as their mask.
    """
    _positive_integer(chunk_blocks, "chunk_blocks")
    if (
        not isinstance(resident, torch.Tensor)
        or resident.device.type != "cpu"
        or resident.dtype != torch.bool
        or resident.ndim != 2
    ):
        raise ValueError("resident must be a CPU bool tensor [heads, history_blocks]")
    heads, history_blocks = resident.shape
    _selection_inputs(selections, positions, heads, history_blocks, block_size)
    queries = selections.shape[0]
    logical_rows: list[list[tuple[int, int]]] = []
    lengths: dict[tuple[int, int], int] = {}
    occurrence_tokens = 0
    for selection, position in zip(
        selections.tolist(), positions.tolist(), strict=True
    ):
        for head, blocks in enumerate(selection):
            row: list[tuple[int, int]] = []
            for block in blocks:
                if block == -1 or block * block_size > position:
                    continue
                key = (head, block)
                tokens = min(block_size, position + 1 - block * block_size)
                lengths[key] = max(lengths.get(key, 0), tokens)
                occurrence_tokens += tokens
                row.append(key)
            logical_rows.append(row)
    residency = resident.tolist()
    resident_keys = [key for key in lengths if residency[key[0]][key[1]]]
    missing_keys = [key for key in lengths if not residency[key[0]][key[1]]]
    keys = tuple(resident_keys + missing_keys)
    compact = {key: index for index, key in enumerate(keys)}
    rows = [[compact[key] for key in row] for row in logical_rows]
    resident_blocks = len(resident_keys)
    chunks = tuple(
        (start, min(start + chunk_blocks, len(keys)))
        for start in range(resident_blocks, len(keys), chunk_blocks)
    )
    stages = tuple(
        _stage([index for index in row if start <= index < stop] for row in rows)
        for start, stop in ((0, resident_blocks), *chunks)
    )
    return SparseBlockPlan(
        queries=queries,
        heads=heads,
        history_blocks=history_blocks,
        block_size=block_size,
        block_keys=keys,
        valid_tokens=torch.tensor([lengths[key] for key in keys], dtype=torch.int32),
        resident_blocks=resident_blocks,
        chunk_ranges=chunks,
        stages=stages,
        starts=torch.tensor(
            [block * block_size for _, block in keys], dtype=torch.int32
        ),
        row_positions=positions.to(torch.int32).repeat_interleave(heads),
        all_stage=_stage(rows),
        selected_token_occurrences=occurrence_tokens,
    )


def _backing(backing: torch.Tensor, heads: int) -> None:
    if (
        not isinstance(backing, torch.Tensor)
        or backing.device.type != "cpu"
        or backing.ndim != 3
        or backing.shape[1:] != (heads, 128)
        or not backing.is_floating_point()
    ):
        raise ValueError(
            "backing must be a floating CPU tensor [history_tokens,heads,128]"
        )


def pack_blocks(
    backing: torch.Tensor,
    plan: SparseBlockPlan,
    pin_memory: bool = False,
) -> torch.Tensor:
    """Read only selected physical prefixes into a zero-padded CPU staging buffer.

    The first 64 elements of a token are K and the last 64 are V. This operation
    neither materializes nor transfers the dense history. Pinned allocation
    requires a working accelerator runtime; failures propagate to the caller.
    """
    _backing(backing, plan.heads)
    if (
        backing.shape[0] + plan.block_size - 1
    ) // plan.block_size != plan.history_blocks:
        raise ValueError("backing history length does not match plan history_blocks")
    for key, valid in zip(plan.block_keys, plan.valid_tokens.tolist(), strict=True):
        if key[1] * plan.block_size + valid > backing.shape[0]:
            raise ValueError("selected causal prefix extends past backing history")
    packed = torch.zeros(
        (len(plan.block_keys), plan.block_size, 128),
        dtype=backing.dtype,
        device="cpu",
        pin_memory=pin_memory,
    )
    for compact, ((head, block), valid) in enumerate(
        zip(plan.block_keys, plan.valid_tokens.tolist(), strict=True)
    ):
        start = block * plan.block_size
        packed[compact, :valid].copy_(backing[start : start + valid, head])
    return packed


def _query(query: torch.Tensor, queries: int, heads: int) -> None:
    if (
        not isinstance(query, torch.Tensor)
        or query.device.type != "cpu"
        or query.ndim != 4
        or query.shape[:2] != (queries, heads)
        or query.shape[-1] != 64
        or query.shape[2] < 1
        or not query.is_floating_point()
    ):
        raise ValueError(
            "query must be a floating CPU tensor [queries,heads,groups,64]"
        )


def selected_attention_oracle(
    query: torch.Tensor,
    backing: torch.Tensor,
    selections: torch.Tensor,
    positions: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Independent FP64 dense-softmax reference over each row's selected tokens.

    This deliberately does not use the compact plan or staged implementation.
    It preserves repeated block occurrences, absolute causality and empty rows.
    It uses FP64 arithmetic, without native BF16 Q/probability rounding, so GPU
    parity tests need an appropriate reduced-precision tolerance.
    """
    _cpu_integer(selections, "selections", 3)
    queries, heads, _ = selections.shape
    _backing(backing, heads)
    _query(query, queries, heads)
    _positive_integer(block_size, "block_size")
    history_blocks = (len(backing) + block_size - 1) // block_size
    _selection_inputs(selections, positions, heads, history_blocks, block_size)
    output = torch.zeros(query.shape, dtype=torch.float64)
    for query_index in range(queries):
        for head in range(heads):
            tokens = []
            for block in selections[query_index, head].tolist():
                if block < 0:
                    continue
                start = block * block_size
                stop = min(start + block_size, int(positions[query_index]) + 1)
                if stop > len(backing):
                    raise ValueError(
                        "selected causal prefix extends past backing history"
                    )
                tokens.extend(range(start, stop))
            if tokens:
                kv = backing[tokens, head].double()
                q = query[query_index, head].double()
                scores = (q @ kv[:, :64].T) * (64**-0.5)
                output[query_index, head] = scores.softmax(dim=-1) @ kv[:, 64:]
    return output


def staged_attention_oracle(
    query: torch.Tensor,
    packed: torch.Tensor,
    plan: SparseBlockPlan,
    stages: Iterable[BlockStage] | None = None,
) -> torch.Tensor:
    """FP64 online-softmax continuation reference, including reordered stages.

    Optional stages support testing alternate block orders. Every logical block
    occurrence must still appear exactly once across the supplied stages.
    """
    _query(query, plan.queries, plan.heads)
    if (
        packed.device.type != "cpu"
        or packed.shape != (len(plan.block_keys), plan.block_size, 128)
        or not packed.is_floating_point()
    ):
        raise ValueError("packed must be floating CPU [unique_blocks,block_size,128]")
    stages = tuple(plan.stages if stages is None else stages)
    rows = plan.queries * plan.heads
    groups = query.shape[2]
    q = query.double().reshape(rows, groups, 64)
    maximum = torch.full((rows, groups), -torch.inf, dtype=torch.float64)
    denominator = torch.zeros((rows, groups), dtype=torch.float64)
    accumulator = torch.zeros((rows, groups, 64), dtype=torch.float64)
    observed: list[list[int]] = [[] for _ in range(rows)]
    for stage in stages:
        if stage.ids.shape[0] != rows or stage.counts.shape != (rows,):
            raise ValueError("stage row count must match plan")
        for row in range(rows):
            count = int(stage.counts[row])
            if count < 0 or count > stage.ids.shape[1]:
                raise ValueError("invalid stage count")
            for block in stage.ids[row, :count].tolist():
                if block < 0 or block >= len(plan.block_keys):
                    raise ValueError("invalid compact block ID")
                observed[row].append(block)
                if plan.block_keys[block][0] != row % plan.heads:
                    raise ValueError("stage block belongs to another KV head")
                valid = min(
                    plan.block_size,
                    int(plan.row_positions[row]) + 1 - int(plan.starts[block]),
                )
                if valid <= 0 or valid > int(plan.valid_tokens[block]):
                    raise ValueError("stage block violates per-query causal access")
                kv = packed[block, :valid].double()
                scores = (q[row] @ kv[:, :64].T) * (64**-0.5)
                updated = torch.maximum(maximum[row], scores.max(dim=-1).values)
                correction = (maximum[row] - updated).exp()
                probability = (scores - updated[:, None]).exp()
                accumulator[row] = (
                    accumulator[row] * correction[:, None] + probability @ kv[:, 64:]
                )
                denominator[row] = denominator[row] * correction + probability.sum(
                    dim=-1
                )
                maximum[row] = updated
    for row in range(rows):
        expected = plan.all_stage.ids[row, : int(plan.all_stage.counts[row])].tolist()
        if sorted(observed[row]) != sorted(expected):
            raise ValueError("stages must preserve each row's selection multiplicity")
    return (
        accumulator / denominator.clamp_min(torch.finfo(torch.float64).tiny)[..., None]
    ).reshape(query.shape)
