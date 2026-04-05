"""
Profile MoE dispatch: per-token sort (Higgs) vs global batch sort (mlx-lm).

Tests gather_qmm at DeepSeek-V2-Lite scale:
  hidden=2048, intermediate=1408, 64 experts, top_k=6, 4-bit quantized
"""

import time
import mlx.core as mx
import mlx.nn
import numpy as np

# DeepSeek-V2-Lite MoE dimensions
HIDDEN = 2048
INTERMEDIATE = 1408
NUM_EXPERTS = 64
TOP_K = 6
GROUP_SIZE = 64
BITS = 4

WARMUP = 3
ITERS = 10


def make_weights(out_dim, in_dim):
    """Create quantized expert weights [num_experts, out_dim, in_dim]."""
    w_full = mx.random.normal((NUM_EXPERTS, out_dim, in_dim))
    w, scales, biases = mx.quantize(w_full, group_size=GROUP_SIZE, bits=BITS)
    mx.eval(w, scales, biases)
    return w, scales, biases


def make_inputs(B, L):
    """Create input tensor and random top-k expert indices."""
    x = mx.random.normal((B, L, HIDDEN))
    indices = mx.random.randint(0, NUM_EXPERTS, (B, L, TOP_K)).astype(mx.uint32)
    mx.eval(x, indices)
    return x, indices


# -- Approach 1: Higgs (per-token sort or no sort) --

def forward_gather_higgs(x, indices, gate_w, gate_s, gate_b,
                         up_w, up_s, up_b, down_w, down_s, down_b,
                         do_sort=False):
    """Higgs approach: expand x, call gather_qmm with per-token indices."""
    B, L, D = x.shape

    if do_sort:
        # Per-token sort (Qwen3Next path)
        indices = mx.sort(indices, axis=-1)

    x_exp = x.reshape(B, L, 1, 1, D)

    gate_out = mx.gather_qmm(
        x_exp, gate_w, gate_s, gate_b,
        rhs_indices=indices, transpose=True,
        group_size=GROUP_SIZE, bits=BITS, sorted_indices=do_sort,
    )
    up_out = mx.gather_qmm(
        x_exp, up_w, up_s, up_b,
        rhs_indices=indices, transpose=True,
        group_size=GROUP_SIZE, bits=BITS, sorted_indices=do_sort,
    )
    activated = mlx.nn.silu(gate_out) * up_out

    down_out = mx.gather_qmm(
        activated, down_w, down_s, down_b,
        rhs_indices=indices, transpose=True,
        group_size=GROUP_SIZE, bits=BITS, sorted_indices=do_sort,
    )
    return down_out.squeeze(-2)


# -- Approach 2: mlx-lm (global batch sort) --

def _gather_sort(x, indices):
    """mlx-lm's global sort: flatten all tokens, sort by expert index."""
    *_, M = indices.shape  # top_k
    indices_flat = indices.flatten()
    order = mx.argsort(indices_flat)
    inv_order = mx.argsort(order)
    x_sorted = x.reshape(-1, 1, x.shape[-1])[order // M]
    return x_sorted, indices_flat[order], inv_order


def _scatter_unsort(x, inv_order, shape):
    x = x[inv_order]
    x = mx.unflatten(x, 0, shape)
    return x


def forward_gather_mlxlm(x, indices, gate_w, gate_s, gate_b,
                          up_w, up_s, up_b, down_w, down_s, down_b):
    """mlx-lm approach: global sort, then gather_qmm with sorted_indices=True.

    Matches SwitchGLU.__call__ from mlx_lm/models/switch_layers.py exactly.
    """
    orig_shape = indices.shape  # [B, L, top_k]

    # Step 1: expand x (same as SwitchGLU)
    x_exp = x.reshape(x.shape[0], x.shape[1], 1, 1, x.shape[2])

    # Step 2: global sort — flatten all tokens, reorder by expert index
    # x_sorted: [B*L*top_k, 1, D], idx_sorted: [B*L*top_k], inv_order: [B*L*top_k]
    x_sorted, idx_sorted, inv_order = _gather_sort(x_exp, indices)

    gate_out = mx.gather_qmm(
        x_sorted, gate_w, gate_s, gate_b,
        rhs_indices=idx_sorted, transpose=True,
        group_size=GROUP_SIZE, bits=BITS, sorted_indices=True,
    )
    up_out = mx.gather_qmm(
        x_sorted, up_w, up_s, up_b,
        rhs_indices=idx_sorted, transpose=True,
        group_size=GROUP_SIZE, bits=BITS, sorted_indices=True,
    )
    activated = mlx.nn.silu(gate_out) * up_out

    down_out = mx.gather_qmm(
        activated, down_w, down_s, down_b,
        rhs_indices=idx_sorted, transpose=True,
        group_size=GROUP_SIZE, bits=BITS, sorted_indices=True,
    )

    # Step 3: unsort back to [B, L, top_k, 1, D], then squeeze
    out = _scatter_unsort(down_out, inv_order, orig_shape)  # [B, L, top_k, 1, D]
    return out.squeeze(-2)  # [B, L, top_k, D]


def bench(fn, label, *args, **kwargs):
    # Warmup
    for _ in range(WARMUP):
        r = fn(*args, **kwargs)
        mx.eval(r)

    # Timed
    times = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        r = fn(*args, **kwargs)
        mx.eval(r)
        times.append(time.perf_counter() - t0)

    med = sorted(times)[ITERS // 2]
    return med


def main():
    print(f"DeepSeek-V2-Lite MoE dispatch profiling")
    print(f"  {NUM_EXPERTS} experts, top_k={TOP_K}, {BITS}-bit, "
          f"hidden={HIDDEN}, intermediate={INTERMEDIATE}")
    print(f"  {WARMUP} warmup, {ITERS} iters, reporting median\n")

    # Create weights once
    gate_w, gate_s, gate_b = make_weights(INTERMEDIATE, HIDDEN)
    up_w, up_s, up_b = make_weights(INTERMEDIATE, HIDDEN)
    down_w, down_s, down_b = make_weights(HIDDEN, INTERMEDIATE)
    w_args = (gate_w, gate_s, gate_b, up_w, up_s, up_b, down_w, down_s, down_b)

    seq_lens = [1, 32, 128, 512, 1024, 2048]

    print(f"{'SeqLen':>8} | {'Higgs(nosort)':>14} | {'Higgs(ptsort)':>14} | {'mlx-lm(global)':>14} | {'Speedup':>10}")
    print("-" * 75)

    for L in seq_lens:
        B = 1
        x, indices = make_inputs(B, L)

        t_nosort = bench(forward_gather_higgs, f"nosort-{L}",
                         x, indices, *w_args, do_sort=False)
        t_ptsort = bench(forward_gather_higgs, f"ptsort-{L}",
                         x, indices, *w_args, do_sort=True)
        t_global = bench(forward_gather_mlxlm, f"global-{L}",
                         x, indices, *w_args)

        fastest_higgs = min(t_nosort, t_ptsort)
        speedup = fastest_higgs / t_global if t_global > 0 else float('inf')

        print(f"{L:>8} | {t_nosort*1000:>11.2f} ms | {t_ptsort*1000:>11.2f} ms | "
              f"{t_global*1000:>11.2f} ms | {speedup:>8.2f}x")

    # Also time just the sort overhead
    print(f"\n--- Sort overhead alone ---")
    for L in [128, 512, 2048]:
        x, indices = make_inputs(1, L)
        x_exp = x.reshape(1, L, 1, 1, HIDDEN)

        # Time global sort
        for _ in range(WARMUP):
            r = _gather_sort(x_exp, indices)
            mx.eval(*r)
        times = []
        for _ in range(ITERS):
            t0 = time.perf_counter()
            r = _gather_sort(x_exp, indices)
            mx.eval(*r)
            times.append(time.perf_counter() - t0)
        med = sorted(times)[ITERS // 2]
        print(f"  L={L:>5}: global sort = {med*1000:.2f} ms")


if __name__ == "__main__":
    main()
