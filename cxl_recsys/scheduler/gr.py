"""Adapt cxldsagr GR requests without changing their token stream or KV boundary."""

from ..common.request import MAX_CANDIDATE_TOKENS, MAX_CANDIDATES, validate


def gr_request(record, index, max_history):
    if not isinstance(record, dict):
        raise ValueError("GR request must be an object")

    def integer(name, minimum=0):
        value = record[name]
        if type(value) is not int or value < minimum:
            raise ValueError(f"GR {name} must be an integer >= {minimum}")
        return value

    ids = record["input_ids"]
    if not isinstance(ids, list):
        raise ValueError("GR input_ids must be a list")
    prefix = integer("stable_prefix_tokens", 1)
    instruction = integer("instruction_tokens")
    user_tokens = integer("user_tokens", 1)
    item_tokens = integer("item_tokens", 1)
    suffix = integer("candidate_suffix_tokens", 1)
    total = integer("total_input_tokens", 1)
    if (
        total != len(ids)
        or total != user_tokens + item_tokens
        or prefix != instruction + user_tokens
        or total != prefix + suffix
        or item_tokens != instruction + suffix
    ):
        raise ValueError("GR token lengths disagree with input_ids or stable prefix")
    for name, expected in (
        ("history_token_span", [instruction, prefix]),
        ("candidate_token_span", [prefix, total]),
    ):
        span = record[name]
        if (
            not isinstance(span, list)
            or any(type(position) is not int for position in span)
            or span != expected
        ):
            raise ValueError(f"GR {name} must be {expected}")
    mask = record["attention_mask"]
    if (
        not isinstance(mask, list)
        or len(mask) != total
        or any(type(value) is not int or value != 1 for value in mask)
    ):
        raise ValueError("GR attention_mask must be all ones, without padding")
    if prefix > max_history:
        raise ValueError(
            f"GR stable prefix ({prefix} = instruction {instruction} + user "
            f"{user_tokens}) exceeds max_history={max_history}"
        )
    task_id = integer("task_id") if "task_id" in record else index
    uid = integer("user_id")
    items = record["candidate_item_ids"]
    if not isinstance(items, list) or any(
        type(item) is not int or item < 0 for item in items
    ):
        raise ValueError("GR candidate_item_ids must contain nonnegative integers")
    request = {
        "request_id": f"gr-{task_id}",
        "user_id": str(uid),
        "history_token_ids": ids[:prefix],
        "candidate_token_ids": ids[prefix:],
        "candidate_item_ids": [str(item) for item in items],
    }
    for name in ("timestamp", "visit_index"):
        if name in record:
            request[name] = record[name]
    return validate(request, max_history)


def generate_gr(args):
    # Import only when generating: replay of exported GR JSONL needs no resources.
    from GR.input_generator import TextConfig, create_input_generator
    from GR.scheduling import ScheduleConfig

    if not 1 <= args.num_requests <= 10000:
        raise ValueError("GR num_requests must be in 1..10000")
    if not 1 <= args.gr_candidate_count <= MAX_CANDIDATES:
        raise ValueError(f"GR candidate_count must be in 1..{MAX_CANDIDATES}")
    user_lengths = tuple(args.gr_user_lengths)
    item_lengths = tuple(args.gr_item_lengths)
    text = TextConfig(
        user_lengths=user_lengths,
        user_probabilities=tuple(
            args.gr_user_probabilities
            if args.gr_user_probabilities is not None
            else [1 / len(user_lengths)] * len(user_lengths)
        ),
        item_lengths=item_lengths,
        item_probabilities=tuple(
            args.gr_item_probabilities
            if args.gr_item_probabilities is not None
            else [1 / len(item_lengths)] * len(item_lengths)
        ),
        candidate_count=args.gr_candidate_count,
        max_input_tokens=args.max_history + MAX_CANDIDATE_TOKENS,
        history_cache_users=args.gr_history_cache_users,
        prompt_format=args.gr_prompt_format,
    )
    generator = create_input_generator(
        heat_source=args.gr_heat_source,
        heat_path=args.gr_heat_path,
        industrial_heat_field=args.gr_industrial_heat_field,
        curve_dataset=args.gr_curve_dataset,
        curve_field=args.gr_curve_field,
        num_users=args.users,
        text_material=args.gr_text_material,
        text_dataset=args.gr_text_dataset,
        text_catalog_path=args.gr_text_catalog_path,
        data_root=args.gr_data_root,
        tokenizer=args.gr_tokenizer,
        text_config=text,
        schedule_config=ScheduleConfig(
            seed=args.seed,
            qps=args.gr_qps,
            arrival=args.gr_arrival,
            sampling=args.gr_sampling,
        ),
    )
    instruction = len(generator.prefix_ids)
    if max(user_lengths) + instruction > args.max_history:
        raise ValueError(
            "GR user length + instruction tokens exceeds --max-history: "
            f"{max(user_lengths)} + {instruction} > {args.max_history}"
        )
    if min(item_lengths) <= instruction:
        raise ValueError("GR item budget must leave room after the instruction")
    if max(item_lengths) - instruction > MAX_CANDIDATE_TOKENS:
        raise ValueError("GR candidate suffix exceeds the request token limit")
    return generator.iter_generate(args.num_requests)
