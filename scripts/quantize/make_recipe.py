#!/usr/bin/env python3
"""Turn imatrix.json routing statistics into a per-tensor quantization
recipe for mlx_lm.convert (see convert_with_recipe.py and README.md).

Usage (layer granularity, the default -- matches what mlx_lm.convert can
actually realize for this architecture, see "Granularity" below):
    make_recipe.py --imatrix imatrix.json --granularity layer \\
        --target-effective-bpw 4.45 \\
        --expert-bits-low 3 --expert-bits-high 4 --other-bits 6 \\
        --group-size 64 --out recipe.json

Usage (expert granularity, kept for future per-expert-capable conversion
stacks; see "Granularity" below for why this mode does NOT reflect real
mlx_lm.convert output bytes today):
    make_recipe.py --imatrix imatrix.json --granularity expert \\
        --target-avg-bits 3.75 \\
        --expert-bits-low 3 --expert-bits-high 4 --other-bits 6 \\
        --group-size 64 --out recipe.json

ds4 rationale: routed MoE experts individually see a small slice of tokens
and tolerate aggressive quantization; layers whose MoE block input carries
less signal (lower `input_sq_mean` -- our per-layer salience proxy, see
collect_imatrix.py) tolerate more aggressive quantization than layers whose
input activations are large. Everything else (attention, shared experts,
embeddings, lm_head) stays at a conservative `other-bits`.

Output shape (see README.md for the full contract, and
crates/higgs-models/src/quant_config.rs in the main higgs crate for the
loader that consumes it):

    {
      "default": {"group_size": G, "bits": B_other},
      "rules": [
        {"path": "model.layers.1.mlp.switch_mlp.gate_proj", "group_size": G, "bits": 3},
        ...
      ]
    }

Granularity: mlx_lm's DeepSeek-V2 implementation stores all routed experts
for one layer/projection as a single fused `switch_mlp.{gate,up,down}_proj`
tensor (see `SwitchGLU` in `mlx_lm.models.deepseek_v2`), and
`mlx.nn.quantize`'s `class_predicate` is called once per *leaf module* --
so as of the installed mlx_lm (checked against mlx 0.32 and the cached
mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx checkpoint), there
is no way to apply two different bit widths to different experts within
the same fused tensor at conversion time.

`--granularity layer` (the default) solves the budget at the granularity
mlx_lm.convert can actually deliver: each MoE layer's entire fused
switch_mlp tensor gets one bit width, chosen from a real
parameter-and-group-overhead-weighted byte budget (see
`solve_layer_bit_assignment`), and recipe rules target the fused
`switch_mlp.{proj}` paths directly -- convert_with_recipe.py applies these
with no collapsing needed. This is what actually determines the output
checkpoint's size.

`--granularity expert` reproduces the older per-expert-path recipe (still
useful as the intended-precision artifact and to match higgs's own
per-tensor loader path convention, e.g. for a future unfused conversion
path), but convert_with_recipe.py has to collapse those per-expert rules
to one per-layer majority-vote decision before mlx_lm.convert can use them
-- which means the *actual bytes written* by a real conversion run using
an expert-granularity recipe do NOT match the per-expert numbers this mode
prints, only the post-collapse layer decisions do. Prefer `layer` mode
whenever you need the solved budget to hold in practice.
"""

from __future__ import annotations

import argparse
import json
import sys
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

    moe_layer_indices = []
    for layer_idx in range(n_layers):
        other_params += attention_param_count(cfg)
        if layer_idx < first_k_dense:
            other_params += dense_mlp_param_count(hidden_size, intermediate_size)
        else:
            moe_layer_indices.append(layer_idx)
            other_params += shared_expert_param_count(cfg)

    per_expert_params = expert_param_count(hidden_size, moe_intermediate)
    n_moe_layers = len(moe_layer_indices)
    total_routed_params = n_moe_layers * n_routed * per_expert_params

    return {
        "other_params": other_params,
        "total_routed_params": total_routed_params,
        "per_expert_params": per_expert_params,
        "per_layer_routed_params": n_routed * per_expert_params,
        "n_moe_layers": n_moe_layers,
        "moe_layer_indices": moe_layer_indices,
        "n_routed_experts": n_routed,
        "total_params": other_params + total_routed_params,
    }


def effective_bpw(bits: int, group_size: int) -> float:
    """Effective bits-per-weight for one mlx affine-quantized tensor.

    Each group of `group_size` weights shares one fp16 scale and one fp16
    zero-point/bias, so the true storage cost per weight is
    `bits + (16 + 16) / group_size`. This is what determines actual bytes
    on disk, unlike the raw `bits` figure the --target-avg-bits (expert
    granularity) solve uses -- see the module docstring's "Granularity"
    section for why that distinction matters here.
    """
    return bits + (2 * 16) / group_size


def project_effective_bpw(
    summary: dict,
    other_bits: int,
    group_size: int,
    layer_bits: dict[int, int],
) -> float:
    """Parameter-weighted effective bpw across the whole model given a
    per-MoE-layer bit-width assignment (`layer_bits`: layer_idx -> bits).
    """
    other_eff = effective_bpw(other_bits, group_size)
    total_bits = summary["other_params"] * other_eff
    per_layer_params = summary["per_layer_routed_params"]
    for layer_idx in summary["moe_layer_indices"]:
        bits = layer_bits[layer_idx]
        total_bits += per_layer_params * effective_bpw(bits, group_size)
    return total_bits / summary["total_params"]


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


def solve_layer_bit_assignment(
    imatrix_layers: list[dict],
    summary: dict,
    target_effective_bpw: float,
    other_bits: int,
    expert_bits_low: int,
    expert_bits_high: int,
    group_size: int,
) -> dict[int, int]:
    """Assign each MoE layer's fused switch_mlp tensors either
    `expert_bits_low` or `expert_bits_high`, greedily converting the
    least-salient layers to `expert_bits_low` (ranked ascending by
    `input_sq_mean`, our per-layer salience proxy -- see
    collect_imatrix.py's docstring) until the projected whole-model
    parameter-weighted effective bpw is <= target_effective_bpw.

    Every layer starts at expert_bits_high; layers are flipped to
    expert_bits_low one at a time, lowest salience first, stopping as soon
    as the budget is met. If flipping every MoE layer to expert_bits_low
    still doesn't reach the target, all layers end up low and the caller
    is warned (the model's non-routed weight alone can't be reconciled
    with the target from bit-width choice on experts).
    """
    layer_bits = {idx: expert_bits_high for idx in summary["moe_layer_indices"]}
    if project_effective_bpw(summary, other_bits, group_size, layer_bits) <= target_effective_bpw:
        return layer_bits  # already under budget with everything at high bits

    ranked = sorted(imatrix_layers, key=lambda entry: entry["input_sq_mean"])
    for entry in ranked:
        layer_bits[entry["layer"]] = expert_bits_low
        if (
            project_effective_bpw(summary, other_bits, group_size, layer_bits)
            <= target_effective_bpw
        ):
            return layer_bits

    return layer_bits  # exhausted every layer; still over budget, see caller


def load_config(model_config_path: Path | None, imatrix: dict) -> dict:
    if model_config_path is not None:
        return json.loads(model_config_path.read_text())
    if "config" in imatrix:
        return imatrix["config"]
    raise SystemExit(
        "imatrix.json has no embedded 'config' and --model-config was not "
        "given; pass --model-config <path to config.json> explicitly."
    )


def make_expert_granularity_recipe(imatrix: dict, summary: dict, args) -> list[dict]:
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

    achieved_avg = (
        summary["other_params"] * args.other_bits
        + summary["total_routed_params"]
        * (f_high * args.expert_bits_high + (1 - f_high) * args.expert_bits_low)
    ) / summary["total_params"]
    print(
        f"[make_recipe] granularity=expert: {summary['n_moe_layers']} MoE layers x "
        f"{n_routed} experts, {n_high_per_layer} high-bit experts/layer "
        f"({n_high_per_layer / n_routed:.1%})"
    )
    print(
        f"[make_recipe] estimated avg bits/weight (raw, no group overhead): "
        f"{achieved_avg:.3f} (target {args.target_avg_bits}) -- NOTE: "
        "convert_with_recipe.py must collapse these per-expert rules to one "
        "per-layer decision (majority vote) before a real mlx_lm.convert run "
        "can use them, since mlx_lm stores routed experts as one fused "
        "tensor per layer/projection; the actual output bytes come from "
        "that post-collapse decision, not this raw estimate. Prefer "
        "--granularity layer to solve directly against real output bytes."
    )
    return rules


def make_layer_granularity_recipe(imatrix: dict, summary: dict, args) -> list[dict]:
    layer_bits = solve_layer_bit_assignment(
        imatrix["layers"],
        summary,
        args.target_effective_bpw,
        args.other_bits,
        args.expert_bits_low,
        args.expert_bits_high,
        args.group_size,
    )

    rules = []
    projections = ("gate_proj", "up_proj", "down_proj")
    for layer_idx, bits in sorted(layer_bits.items()):
        for projection in projections:
            rules.append(
                {
                    "path": f"model.layers.{layer_idx}.mlp.switch_mlp.{projection}",
                    "group_size": args.group_size,
                    "bits": bits,
                }
            )

    n_low = sum(1 for bits in layer_bits.values() if bits == args.expert_bits_low)
    n_high = len(layer_bits) - n_low
    projected_bpw = project_effective_bpw(summary, args.other_bits, args.group_size, layer_bits)
    projected_bytes = (
        summary["other_params"] * effective_bpw(args.other_bits, args.group_size)
        + sum(
            summary["per_layer_routed_params"] * effective_bpw(bits, args.group_size)
            for bits in layer_bits.values()
        )
    ) / 8

    print(
        f"[make_recipe] granularity=layer: {n_low} layers at "
        f"{args.expert_bits_low}-bit, {n_high} layers at {args.expert_bits_high}-bit "
        f"(of {summary['n_moe_layers']} MoE layers)"
    )
    print(
        f"[make_recipe] projected effective bpw: {projected_bpw:.4f} "
        f"(target <= {args.target_effective_bpw}); projected total bytes: "
        f"{projected_bytes:,.0f} ({projected_bytes / 1e9:.3f} GB)"
    )
    if projected_bpw > args.target_effective_bpw:
        print(
            "[make_recipe] WARNING: every MoE layer is already at "
            f"{args.expert_bits_low}-bit and the projected effective bpw "
            f"({projected_bpw:.4f}) still exceeds the target "
            f"({args.target_effective_bpw}); lower --other-bits or "
            "--expert-bits-low, or raise --target-effective-bpw, to reconcile.",
            file=sys.stderr,
        )
    return rules


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imatrix", required=True, type=Path)
    parser.add_argument(
        "--granularity",
        choices=("layer", "expert"),
        default="layer",
        help="'layer' (default) solves against the real fused-tensor byte "
        "budget mlx_lm.convert can deliver; 'expert' emits the older "
        "per-expert-path recipe (needs convert_with_recipe.py's majority-"
        "vote collapse to apply, and its raw estimate does not match real "
        "output bytes). See the module docstring's Granularity section.",
    )
    parser.add_argument(
        "--target-avg-bits",
        type=float,
        default=None,
        help="required for --granularity expert: raw parameter-weighted "
        "average bits target (no group overhead)",
    )
    parser.add_argument(
        "--target-effective-bpw",
        type=float,
        default=4.45,
        help="required for --granularity layer: parameter-weighted average "
        "*effective* bits-per-weight target, including per-group fp16 "
        "scale+bias overhead (bits + 32/group_size). Default 4.45 sits "
        "safely under uniform 4-bit group_size=64's 4.5 effective bpw.",
    )
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

    if args.granularity == "expert" and args.target_avg_bits is None:
        raise SystemExit("--granularity expert requires --target-avg-bits")

    imatrix = json.loads(args.imatrix.read_text())
    cfg = load_config(args.model_config, imatrix)
    summary = build_model_param_summary(cfg)

    if args.granularity == "layer":
        rules = make_layer_granularity_recipe(imatrix, summary, args)
    else:
        rules = make_expert_granularity_recipe(imatrix, summary, args)

    recipe = {
        "default": {"group_size": args.group_size, "bits": args.other_bits},
        "rules": rules,
    }
    args.out.write_text(json.dumps(recipe, indent=2))
    print(f"[make_recipe] {len(rules)} rules written to {args.out}")


if __name__ == "__main__":
    main()
