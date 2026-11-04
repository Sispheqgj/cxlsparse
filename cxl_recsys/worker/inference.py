"""Worker token forward and optional sparse-model candidate selection."""

import logging

from ..common.timing import StageTimer, synchronize_model
from ..models.base import CacheContext
from .prompt import build_prompt

logger = logging.getLogger(__name__)


def execute_request(model, request):
    """Normalize a single Request to the same packed Worker execution path."""
    return execute_batch(model, [request])[0]


def execute_batch(model, requests, *, timing: StageTimer | None = None):
    """Pack tokens and submit one model call for all independent users.

    Metadata loops do not execute inference. GPU projection, selection and
    consumption operate on the packed batch, with explicit sequence boundaries.
    A singleton also uses this path, with seq_indptr=[0, input_length].
    """
    if not requests:
        raise ValueError("Cannot execute an empty request batch")
    prompts = [
        build_prompt(
            request["history_token_ids"],
            request.get("candidates"),
            candidate_token_ids=request.get("candidate_token_ids"),
        )
        for request in requests
    ]
    input_ids, positions, seq_indptr, caches = [], [], [0], []
    for request, prepared in zip(requests, prompts):
        input_ids.extend(prepared.input_ids)
        positions.extend(prepared.positions)
        seq_indptr.append(len(input_ids))
        caches.append(CacheContext(request["user_id"], prepared.history_length))
    logger.debug(
        "batch_input_prepared batch_size=%d total_tokens=%d seq_indptr=%s",
        len(requests),
        len(input_ids),
        seq_indptr,
    )
    model_timing = None
    if timing is not None:
        timing.mark("preprocess_ms", synchronize=False)
        model_timing = StageTimer(timing.synchronize)
    # Do not require profiling support from CPU test doubles in normal mode.
    timing_args = {"timing": model_timing} if model_timing is not None else {}
    hidden_states = model.forward(
        input_ids, positions, seq_indptr=seq_indptr, caches=caches, **timing_args
    )
    # Final normalization can still be queued after forward returns. Always wait
    # here so it belongs to inference, never to the next task or postprocessing.
    synchronize_model(model)
    if timing is not None and model_timing is not None:
        timing.mark("model_ms", synchronize=False)
        timing.model = model_timing.report()
        timing.model["call_overhead_ms"] = (
            timing.values["model_ms"] - timing.model["total_ms"]
        )
    stats = model.batch_forward_stats()
    if len(stats) != len(requests):
        raise RuntimeError("Model returned an invalid batch diagnostics count")
    results = [
        _result(model, request, prepared, diagnostics)
        for request, prepared, diagnostics in zip(requests, prompts, stats)
    ]
    select = getattr(model, "select_candidate_options", None)
    members = [
        i for i, request in enumerate(requests) if "candidate_item_ids" in request
    ]
    if select is not None and members:
        selected = select(
            hidden_states,
            [stats[i]["batch"]["query_end"] for i in members],
            [requests[i]["candidate_token_ids"] for i in members],
            [len(requests[i]["candidate_item_ids"]) for i in members],
        )
        if len(selected) != len(members):
            raise RuntimeError("Model returned an invalid candidate selection count")
        for i, (option, index) in zip(members, selected):
            results[i].update(
                postprocessing="candidate_selection",
                selected_option=option,
                recommended_item_id=requests[i]["candidate_item_ids"][index],
            )
    del hidden_states
    return results


def _result(model, request, prepared, stats):
    """Serialize diagnostics; candidate selection adds its plain CPU result."""
    result = {key: value for key, value in stats.items() if key != "input_tokens"}
    result.update(
        {
            "request_id": request["request_id"],
            "user_id": request["user_id"],
            "postprocessing": "not_implemented",
            "prompt": {
                "history_tokens": prepared.history_length,
                "candidate_tokens": prepared.candidate_tokens,
                "input_tokens": stats["input_tokens"],
                "last_input_position": prepared.last_input_position,
            },
            "cache": {
                **stats["cache"],
                "candidate_kv_tokens": prepared.candidate_tokens * model.layers,
            },
        }
    )
    if "candidate_item_ids" in request:
        result["candidate_item_ids"] = list(request["candidate_item_ids"])
    for field in ("timestamp", "visit_index"):
        if field in request:
            result[field] = request[field]
    return result
