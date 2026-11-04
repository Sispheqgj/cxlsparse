"""Small-window consumer scheduling, independent of TMA IO ordering.

Reserve an occurrence when its HBM-to-shared copy is issued. Two shared slots
alternate as before; the window adds candidates, not payload buffers. When no
candidate is ready, compute the current tile before retrying or blocking.
"""


def make_ready_consumer(consume, *, window):
    from tilelang import language as T

    @T.macro
    def consume_row(
        row,
        KV,
        IDs,
        Counts,
        Starts,
        Positions,
        Flags,
        q,
        tiles,
        score,
        acc,
        maximum,
        previous,
        denominator,
        factor,
        block_sum,
        score_shared,
        probability,
    ):
        remaining = T.alloc_local((1,), "uint32")
        current = T.alloc_local((1,), "int32")
        upcoming = T.alloc_local((1,), "int32")
        remaining[0] = T.Cast("uint32", (T.uint64(1) << Counts[row]) - 1)
        if Counts[row] > 0:
            current[0] = T.call_extern(
                "int32",
                f"nsa_pick_ready<{window}>",
                T.address_of(IDs[row, 0]),
                T.address_of(Flags[0]),
                remaining[0],
            )
            if current[0] < 0:
                current[0] = T.call_extern("int32", "__ffs", remaining[0]) - 1
                T.call_extern(
                    "void", "nsa_wait_ready", T.address_of(Flags[IDs[row, current[0]]])
                )
            remaining[0] &= ~(T.uint32(1) << current[0])
            T.async_copy(KV[IDs[row, current[0]], :, :], tiles[0, :, :])
            T.ptx_wait_group(0)
            T.sync_threads()

        for step in T.serial(Counts[row]):
            slot = step % 2
            next_slot = (step + 1) % 2
            upcoming[0] = -1
            if step + 1 < Counts[row]:
                upcoming[0] = T.call_extern(
                    "int32",
                    f"nsa_pick_ready<{window}>",
                    T.address_of(IDs[row, 0]),
                    T.address_of(Flags[0]),
                    remaining[0],
                )
                if upcoming[0] >= 0:
                    remaining[0] &= ~(T.uint32(1) << upcoming[0])
                    T.async_copy(
                        KV[IDs[row, upcoming[0]], :, :], tiles[next_slot, :, :]
                    )

            consume(
                row,
                IDs[row, current[0]],
                slot,
                Starts,
                Positions,
                q,
                tiles,
                score,
                acc,
                maximum,
                previous,
                denominator,
                factor,
                block_sum,
                score_shared,
                probability,
            )

            if step + 1 < Counts[row]:
                if upcoming[0] < 0:
                    # Completion may have advanced during MMA. Recheck the
                    # same bounded window before waiting for its first entry.
                    upcoming[0] = T.call_extern(
                        "int32",
                        f"nsa_pick_ready<{window}>",
                        T.address_of(IDs[row, 0]),
                        T.address_of(Flags[0]),
                        remaining[0],
                    )
                    if upcoming[0] < 0:
                        upcoming[0] = T.call_extern("int32", "__ffs", remaining[0]) - 1
                        T.call_extern(
                            "void",
                            "nsa_wait_ready",
                            T.address_of(Flags[IDs[row, upcoming[0]]]),
                        )
                    remaining[0] &= ~(T.uint32(1) << upcoming[0])
                    T.async_copy(
                        KV[IDs[row, upcoming[0]], :, :], tiles[next_slot, :, :]
                    )
                T.ptx_wait_group(0)
                T.sync_threads()
                current[0] = upcoming[0]

    return consume_row
