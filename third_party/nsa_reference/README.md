# Pinned FLA NSA validation reference

Source: <https://github.com/fla-org/native-sparse-attention>, commit
`bd67af59b90afa34b25f61d2922e612d10dba3bd`. `source.json` records the original
file paths and SHA256 hashes. `parallel.py`, `naive.py`, `utils.py` and `LICENSE`
are unchanged snapshots. This directory is validation-only. Serving imports the
FLA-derived `../nsa_runtime` kernels for compression/top-k and the project's
`nsa_attention.py` for compact selected KV, not the FLA training stack or remote
model code. The indexed BF16 attention kernel uses one warp and
three pipeline stages for the checkpoint's GQA=16/head_dim=64 geometry; matching
the H100 arithmetic avoids amplifying tiny rounding differences through top-k.

`forward_only.py` contains the three original forward kernels and their wrappers,
with imports limited to PyTorch/Triton and the local sorting helper. Use only
uniform resident sequences (`offsets=None`). Backward kernels and optional FLA /
FlashAttention dependencies are omitted. Kernels JIT-compile during validation;
this is separate from the offline HiSparse build used by Worker.

There is one algorithmic wrapper correction: `parallel_nsa_topk` uses
`BS = block_size; BC = max(BS, 2 * S)` instead of `BC = BS = block_size`.
The upstream online merge retains `BC/2` candidates from each side. For this
checkpoint's `block_size=block_counts=16`, `BC=16` retains only eight earlier
candidates once it crosses a 16-block chunk. A 513-token random-input test exposed
a 23% relative attention error against global top-16 selection. Using `BC=32`
retains all 16 candidates while leaving compression/selection block size at 16.
Serving uses the corrected merge width; the optional PyTorch comparison path
computes global top-k directly. Saturated-score ties prioritize the current block;
other exact ties prefer lower block IDs. This is an explicit correction, not a
claim of bitwise identity to the unmodified released kernel.

The model is the SSA authors' `zen-E/NSA-1B` baseline at
`3dc3694d2f873820122b094276ae626a44f00a36`, not DeepSeek's original 27B checkpoint.
It uses non-overlapping mean pooling, unlike the learnable overlapping compression
in the original NSA paper. Its three sigmoid gates, GQA block selection and local
window are retained. Model source is available at
<https://huggingface.co/zen-E/NSA-1B> and <https://github.com/zhenyi4/ssa>.
