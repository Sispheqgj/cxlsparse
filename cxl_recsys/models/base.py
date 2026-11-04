"""Model boundary used by Worker; importing it does not initialize CUDA."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from cxl_recsys.models.simulation.config import IndexerConfig

if TYPE_CHECKING:
    from torch import Tensor

    from ..common.timing import StageTimer


@dataclass(frozen=True)
class CacheContext:
    """Generic cache identity and reusable prefix boundary, not a Request.

    Tokens before cacheable_prefix_length may survive the forward. Tokens after
    it are temporary. An existing prefix must match before any state is changed.
    """

    cache_key: str
    cacheable_prefix_length: int


class BaseModel(ABC):
    name: str

    def __init__(
        self,
        pool,
        max_users=16,
        max_history=4096,
        layers=2,
        hot=32,
        topk=8,
        dim=64,
        *,
        indexer: IndexerConfig | None = None,
    ):
        self.ensure_available()
        if max_users <= 0 or layers <= 0 or max_history <= 0:
            raise ValueError("Invalid model limits")
        if pool.dim != dim:
            raise ValueError("Model KV dimension must match the backing pool")
        self.pool = pool
        self.max_users = max_users
        self.max_history = max_history
        self.layers = layers
        self.hot = hot
        self.topk = topk
        self.dim = dim
        self.indexer_config = indexer or IndexerConfig()
        self._last_forward_stats = {}
        self._last_batch_stats = []

    @classmethod
    def ensure_available(cls):
        """Preflight before the CLI allocates/maps any backing memory."""

    @abstractmethod
    def forward(
        self,
        input_ids: Sequence[int],
        positions: Sequence[int],
        *,
        cache: CacheContext | None = None,
        seq_indptr: Sequence[int] | None = None,
        caches: Sequence[CacheContext] | None = None,
        timing: "StageTimer | None" = None,
    ) -> "Tensor":
        """Return final-normalized hidden states for all newly computed tokens.

        Worker flattens supplied tokens unchanged and owns future output processing.
        This interface receives no Request, candidate objects or output vocabulary.
        seq_indptr partitions complete logical sequences in input_ids/positions,
        with one CacheContext per sequence. Positions restart at zero for each
        sequence. Cached prefixes are reused; only new tokens are computed.
        Passing cache alone normalizes a flat input to a batch of one.
        All batch sizes use the same packed computation, never per-user forwards.
        The returned tensor is [sum(new_input_tokens), hidden_dim], in batch order.
        No LM head, logits, sampling or generated tokens belong to this forward.
        Calls are serialized and update KV in place. Any failure invalidates the
        whole run; discard this model instance instead of retrying the forward.
        An optional timing collector records setup, layers and finalization; it
        never changes token/cache semantics or the hidden-state return contract.
        """

    def synchronize(self) -> None:
        """Wait for outstanding model work; CUDA adapters must override this hook.

        Worker calls this even without breakdown to measure completed execution,
        rather than asynchronous GPU launch latency. CPU adapters need no wait.
        """

    def forward_stats(self) -> dict:
        """Small JSON diagnostics, separate from the forward's Tensor return.

        Describes a successful batch of one; empty after a multi-sequence call
        or a failed call. This is a view of batch diagnostics, not a separate path.
        Never retains hidden states or any other GPU tensor.
        """
        return self._last_forward_stats.copy()

    def batch_forward_stats(self) -> list[dict]:
        """Per-sequence diagnostics in batch order, without retaining tensors."""
        return [stats.copy() for stats in self._last_batch_stats]

    @abstractmethod
    def snapshot(self) -> dict:
        """Return user_id -> cached KV state for Scheduler placement metadata."""

    def close(self):
        """Release the backing pool after Worker has drained accepted requests."""
        self.pool.close()
