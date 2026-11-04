"""Test-only IO contracts and comparisons; inference never verifies KV payload."""

import torch


class LengthOnlySlots:
    """Detect accidental per-token work in O(1) metadata validation."""

    def __init__(self, count):
        self.count = count

    def __len__(self):
        return self.count

    def __iter__(self):
        raise AssertionError("Write validation must not iterate over slots")

    def __getitem__(self, index):
        raise AssertionError("Write validation must not inspect slot values")


def assert_write_slots(pool, slots):
    """Check the producer's trusted-write contract only in tests."""
    assert len(set(slots)) == len(slots)
    assert all(
        type(slot) is int and slot >= 0 and slot // pool.page_size in pool.allocated
        for slot in slots
    )


def history_kv(model, layer=0, user_id="user-0"):
    slots = model.users[user_id]["slots"][layer]
    values = torch.empty((len(slots), model.dim), dtype=torch.bfloat16)
    model.pool.read_many(slots, values)
    return values


def assert_history_equal(left, right):
    for layer in range(left.layers):
        assert torch.equal(history_kv(left, layer), history_kv(right, layer))
        length = len(left.users["user-0"]["tokens"])
        assert torch.equal(
            left.users["user-0"]["index_keys"][layer][:length],
            right.users["user-0"]["index_keys"][layer][:length],
        )
