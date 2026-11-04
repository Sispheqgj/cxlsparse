# FLA NSA serving kernels

Source: <https://github.com/fla-org/native-sparse-attention>, revision
`bd67af59b90afa34b25f61d2922e612d10dba3bd`, MIT (see LICENSE and source.json).
Hatch installs this directory as `nsa_runtime`; no upstream training dependencies
or remote model Python are imported. Unchanged upstream files remain in
`../nsa_reference` for independent verification.

`kernels.py` derives the compression forward and online top-k kernels directly
from that revision. QK, online softmax/LSE and AV computation are retained.
Explicit adaptations:

- Read absolute query positions independently of Q storage rows, for full
  prefill and candidate-only suffix queries.
- Pass the complete compressed block count separately from Q length; load
  contiguous K/V summaries without reconstructing full historical KV.
- Set merge width to `max(block_size, 2 * block_counts)`; the original 16/16
  geometry drops eight older candidates at an online merge boundary.
- Prioritize the current block on saturated-score ties, then smaller block IDs.
  Invalid future entries rank below valid zero-probability entries. `utils.py`
  extends the upstream bitonic comparator for deterministic ties.
- Use one warp / three stages, with no autotuning during graph capture. This is
  the H100 numerical-validation configuration, not an all-device optimum claim.

The wrapper `cxl_recsys/models/nsa/fla.py` launches one compression and one top-k
kernel per user/layer across all new Q rows. It does not materialize a Q-by-history
score matrix or use PyTorch softmax/argsort. `nsa_kernel_backend=torch` retains the
earlier correctness path. Its FP32 scoring can round differently from FLA BF16
QK; full-backbone checks retain the existing 0.025 relative-RMSE threshold.

Selected attention still consumes the HiSparse compact indexed buffer through
the project kernel. Sliding-window attention still uses PyTorch SDPA. This is
not a call to upstream's complete `parallel_nsa` function.
