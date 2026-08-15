#!/usr/bin/env python3
"""Turn imatrix.json routing statistics into a per-tensor quantization
recipe for mlx_lm.convert (see convert_with_recipe.py and README.md).

Usage:
    make_recipe.py --imatrix imatrix.json --target-avg-bits 3.75 \\
        --expert-bits-low 3 --expert-bits-high 4 --other-bits 6 \\
        --group-size 64 --out recipe.json

ds4 rationale: routed MoE experts individually see a small slice of tokens
and tolerate aggressive quantization; a handful of "hot" experts per layer
(the ones the router picks most often) carry disproportionate salience, so
we protect them with a higher bit width. Everything else (attention,
shared experts, embeddings, lm_head) stays at a conservative `other-bits`.

Output shape (see README.md for the full contract, and
crates/higgs-models/src/quant_config.rs in the main higgs crate for the
loader that consumes it):

    {
      "default": {"group_size": G, "bits": B_other},
      "rules": [
        {"path": "model.layers.1.mlp.experts.0.gate_proj", "group_size": G, "bits": 3},
        ...
      ]
    }

IMPORTANT caveat on granularity -- read before relying on the per-expert
split from an actual mlx_lm.convert run: mlx_lm's DeepSeek-V2 implementation
stores all routed experts for one layer/projection as a single fused
`switch_mlp.{gate,up,down}_proj` tensor (see `SwitchGLU` in
`mlx_lm.models.deepseek_v2`), and `mlx.nn.quantize`'s `class_predicate` is
called once per *leaf module* -- so as of the installed mlx_lm (checked
against mlx 0.32 / the cached mlx-community/DeepSeek-Coder-V2-Lite-Instruct-
4bit-mlx checkpoint), there is no way to apply two different bit widths to
different experts within the same fused tensor at conversion time. This
tool still emits the recipe at true per-expert granularity because: (a) it
matches the exact tensor-path convention higgs's own per-tensor loader
expects (`model.layers.N.mlp.experts.M.{gate,up,down}_proj`, see the
`deepseek_expert_projection_quantization` fixture paths in
`crates/higgs-models/src/deepseek_v2.rs`), which is the more useful
long-term artifact, and (b) it documents the intended salience-guided split
even where today's converter can't fully realize it. convert_with_recipe.py
collapses per-expert rules to a per-layer majority-vote decision when
talking to mlx_lm.convert, and prints how many layers had to be collapsed;
see its docstring for the tradeoff.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def expert_param_count(hidden_size: int, moe_intermediate_size: int) -> int:
    """Parameters in one routed expert's gate_proj + up_proj + down_proj.

    gate_proj, up_proj: hidden_size x moe_intermediate_size each.
    down_proj: moe_intermediate_size x hidden_size.
    Total = 3 * hidden_size * moe_intermediate_size (bias-free, matching
    mlx_lm's SwitchGLU/DeepseekV2MLP Linear layers).
    """
    return 3 * hidden_size * moe_intermediate_size


def attention_param_count(cfg: dict) -> int:
    """Approximate MLA attention parameter count for one layer.

    Covers q_proj (or q_a_proj/q_b_proj when q_lora_rank is set),
    kv_a_proj_with_mqa, kv_b_proj, and o_proj. Ignores the small
    q_a_layernorm/kv_a_layernorm norm vectors (O(rank), negligible next to
    the O(hidden_size * rank) projections) and attention bias terms
    (attention_bias is false in every known DeepSeek-V2 config).
    """
    hidden_size = cfg["hidden_size"]
    n_heads = cfg["num_attention_heads"]
    qk_nope = cfg["qk_nope_head_dim"]
    qk_rope = cfg["qk_rope_head_dim"]
    v_head_dim = cfg["v_head_dim"]
    kv_lora_rank = cfg["kv_lora_rank"]
    q_lora_rank = cfg.get("q_lora_rank")

    q_head_dim = qk_nope + qk_rope
    if q_lora_rank:
        q_params = hidden_size * q_lora_rank + q_lora_rank * (n_heads * q_head_dim)
    else:
        q_params = hidden_size * (n_heads * q_head_dim)

    kv_a_params = hidden_size * (kv_lora_rank + qk_rope)
    kv_b_params = kv_lora_rank * (n_heads * (qk_nope + v_head_dim))
    o_params = (n_heads * v_head_dim) * hidden_size

    return q_params + kv_a_params + kv_b_params + o_params


def dense_mlp_param_count(hidden_size: int, intermediate_size: int) -> int:
    """Parameters in a dense (non-MoE) layer's gate/up/down MLP."""
    return 3 * hidden_size * intermediate_size


def shared_expert_param_count(cfg: dict) -> int:
    n_shared = cfg.get("n_shared_experts") or 0
    if not n_shared:
        return 0
    hidden_size = cfg["hidden_size"]
    shared_intermediate = cfg["moe_intermediate_size"] * n_shared
    return 3 * hidden_size * shared_intermediate


def build_model_param_summary(cfg: dict) -> dict:
    """Bucket total model parameters into 'other' (attention, embeddings,
    lm_head, shared experts, dense mlp layers) and 'routed_experts'
    (per-layer, per-expert projection weights), from config.json dims.

    All figures approximate from architecture dims rather than reading
    safetensors directly, per the ds4 P2 spec -- close enough to solve the
    low/high expert split without needing the actual checkpoint on disk.
    """
    hidden_size = cfg["hidden_size"]
    vocab_size = cfg["vocab_size"]
    n_layers = cfg["num_hidden_layers"]
    first_k_dense = cfg.get("first_k_dense_replace", 0)
    n_routed = cfg["n_routed_experts"]
    moe_intermediate = cfg["moe_intermediate_size"]
    intermediate_size = cfg["intermediate_size"]
    tie_embeddings = cfg.get("tie_word_embeddings", False)

    other_params = 0
    other_params += vocab_size * hidden_size  # embed_tokens
    if not tie_embeddings:
        other_params += vocab_size * hidden_size  # lm_head

    n_moe_layers = 0
    for layer_idx in range(n_layers):
        other_params += attention_param_count(cfg)
        if layer_idx < first_k_dense:
            other_params += dense_mlp_param_count(hidden_size, intermediate_size)
        else:
            n_moe_layers += 1
            other_params += shared_expert_param_count(cfg)

    per_expert_params = expert_param_count(hidden_size, moe_intermediate)
    total_routed_params = n_moe_layers * n_routed * per_expert_params

    return {
        "other_params": other_params,
        "total_routed_params": total_routed_params,
        "per_expert_params": per_expert_params,
        "n_moe_layers": n_moe_layers,
        "n_routed_experts": n_routed,
        "total_params": other_params + total_routed_params,
    }


def solve_high_bit_fraction(
    summary: dict,
    target_avg_bits: float,
    other_bits: int,
    expert_bits_low: int,
    expert_bits_high: int,
) -> float:
    """Solve for the fraction of routed-expert parameters that should get
    `expert_bits_high` so the whole-model parameter-weighted average bit
    width is <= target_avg_bits, holding `other_bits` fixed for everything
    outside the routed experts.

    total_params * target >= other_params * other_bits
        + total_routed_params * (f * high + (1 - f) * low)

    Solving for f:
        f <= (target*total_params - other_params*other_bits) / total_routed_params - low
             ---------------------------------------------------------------------------
                                          (high - low)
    """
    total_params = summary["total_params"]
    other_params = summary["other_params"]
    total_routed_params = summary["total_routed_params"]

    if total_routed_params == 0:
        return 0.0
    if expert_bits_high == expert_bits_low:
        return 0.0

    budget_for_routed = target_avg_bits * total_params - other_params * other_bits
    avg_routed_bits_needed = budget_for_routed / total_routed_params
    f = (avg_routed_bits_needed - expert_bits_low) / (
        expert_bits_high - expert_bits_low
    )
    return max(0.0, min(1.0, f))


def load_config(model_config_path: Path | None, imatrix: dict) -> dict:
    if model_config_path is not None:
        return json.loads(model_config_path.read_text())
    if "config" in imatrix:
        return imatrix["config"]
    raise SystemExit(
        "imatrix.json has no embedded 'config' and --model-config was not "
        "given; pass --model-config <path to config.json> explicitly."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imatrix", required=True, type=Path)
    parser.add_argument("--target-avg-bits", required=True, type=float)
    parser.add_argument("--expert-bits-low", required=True, type=int)
    parser.add_argument("--expert-bits-high", required=True, type=int)
    parser.add_argument("--other-bits", required=True, type=int)
    parser.add_argument("--group-size", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--model-config",
        type=Path,
        default=None,
        help="path to the model's config.json (dims used for the parameter "
        "count estimate); defaults to an embedded 'config' key in imatrix.json",
    )
    args = parser.parse_args()

    imatrix = json.loads(args.imatrix.read_text())
    cfg = load_config(args.model_config, imatrix)
    summary = build_model_param_summary(cfg)

    f_high = solve_high_bit_fraction(
        summary,
        args.target_avg_bits,
        args.other_bits,
        args.expert_bits_low,
        args.expert_bits_high,
    )
    n_routed = summary["n_routed_experts"]
    n_high_per_layer = round(f_high * n_routed)

    rules = []
    projections = ("gate_proj", "up_proj", "down_proj")
    for layer_stats in imatrix["layers"]:
        layer_idx = layer_stats["layer"]
        freq = layer_stats["expert_topk_freq"]
        ranked_experts = sorted(range(n_routed), key=lambda e: freq[e], reverse=True)
        high_experts = set(ranked_experts[:n_high_per_layer])
        for expert_idx in range(n_routed):
            bits = (
                args.expert_bits_high
                if expert_idx in high_experts
                else args.expert_bits_low
            )
            for projection in projections:
                rules.append(
                    {
                        "path": f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{projection}",
                        "group_size": args.group_size,
                        "bits": bits,
                    }
                )

    recipe = {
        "default": {"group_size": args.group_size, "bits": args.other_bits},
        "rules": rules,
    }
    args.out.write_text(json.dumps(recipe, indent=2))

    achieved_avg = (
        summary["other_params"] * args.other_bits
        + summary["total_routed_params"]
        * (
            f_high * args.expert_bits_high
            + (1 - f_high) * args.expert_bits_low
        )
    ) / summary["total_params"]
    print(
        f"[make_recipe] {summary['n_moe_layers']} MoE layers x {n_routed} experts, "
        f"{n_high_per_layer} high-bit experts/layer "
        f"({n_high_per_layer / n_routed:.1%})"
    )
    print(
        f"[make_recipe] estimated avg bits/weight: {achieved_avg:.3f} "
        f"(target {args.target_avg_bits}); {len(rules)} rules written to {args.out}"
    )


if __name__ == "__main__":
    main()
