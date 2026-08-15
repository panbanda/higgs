#!/usr/bin/env python3
"""Apply a make_recipe.py recipe.json during an mlx_lm quantized conversion.

Usage:
    convert_with_recipe.py --hf-path <source repo or dir> \\
        --mlx-path <output dir> --recipe recipe.json

Wraps `mlx_lm.convert(..., quantize=True, quant_predicate=<recipe-driven
function>)`. `q_group_size`/`q_bits` are set from the recipe's "default"
bucket, so any tensor the recipe doesn't mention falls back to that.

Granularity caveat (see make_recipe.py's docstring for the full
explanation): recipe.json expresses routed-expert overrides at true
per-expert granularity (`model.layers.N.mlp.experts.M.{proj}`), but
mlx_lm's DeepSeek-V2 model fuses all experts for one layer/projection into
a single `switch_mlp.{proj}` tensor before `mlx.nn.quantize` ever sees it,
and `quant_predicate` is called once per leaf module -- there is no hook
for sub-tensor (per-expert) bit widths at conversion time. This script
detects those fused paths and collapses the per-expert rules for that
layer/projection into one decision by majority vote (ties broken toward
the higher bit width, to fail safe on quality over the size budget). Pass
`--dry-run` to see the collapse decisions and the resulting per-bucket
tensor counts without downloading or converting anything.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

_SWITCH_MLP_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.switch_mlp\.(gate_proj|up_proj|down_proj)$"
)


def build_quant_predicate(recipe: dict, n_routed_experts: int | None):
    """Build the `quant_predicate(path, module, config)` mlx_lm expects.

    Returns (predicate, summary) where `summary` is a dict this script
    mutates as the predicate is invoked, so the caller can print totals
    afterward -- mlx_lm.convert doesn't return per-tensor decisions itself.
    """
    rules_by_path = {rule["path"]: rule for rule in recipe["rules"]}
    default = recipe["default"]
    summary = {
        "bits_counts": Counter(),
        "collapsed_layers": set(),
        "tensor_paths": [],
    }

    def resolve_fused_expert_path(layer_idx: int, projection: str):
        expert_rules = []
        expert_idx = 0
        while True:
            path = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{projection}"
            rule = rules_by_path.get(path)
            if rule is None:
                if n_routed_experts is not None and expert_idx < n_routed_experts:
                    expert_idx += 1
                    continue
                break
            expert_rules.append(rule)
            expert_idx += 1
            if n_routed_experts is not None and expert_idx >= n_routed_experts:
                break
        if not expert_rules:
            return None
        bits_counts = Counter(rule["bits"] for rule in expert_rules)
        top_count = max(bits_counts.values())
        # tie-break toward the higher bit width (fail safe on quality)
        majority_bits = max(
            bits for bits, count in bits_counts.items() if count == top_count
        )
        summary["collapsed_layers"].add((layer_idx, projection))
        return {"group_size": expert_rules[0]["group_size"], "bits": majority_bits}

    def predicate(path, module, config=None):
        # The installed mlx_lm (mlx_lm/utils.py quantize_model.wrapped_predicate)
        # calls quant_predicate(path, module) -- two args, no config -- even
        # though mlx_lm.convert's public type hint advertises a 3-arg
        # Callable[[str, nn.Module, dict], ...]. Accept both call shapes;
        # config is unused here (n_routed_experts is captured via closure
        # from the --hf-path config.json read in main()).
        if not hasattr(module, "to_quantized"):
            return False

        if path in rules_by_path:
            rule = rules_by_path[path]
            decision = {"group_size": rule["group_size"], "bits": rule["bits"]}
        else:
            match = _SWITCH_MLP_RE.match(path)
            decision = (
                resolve_fused_expert_path(int(match.group(1)), match.group(2))
                if match
                else None
            )
            if decision is None:
                decision = {
                    "group_size": default["group_size"],
                    "bits": default["bits"],
                }

        summary["bits_counts"][decision["bits"]] += 1
        summary["tensor_paths"].append((path, decision["bits"]))
        return decision

    return predicate, summary


def print_summary(summary: dict) -> None:
    total = sum(summary["bits_counts"].values())
    print("[convert_with_recipe] tensors per bits bucket:", file=sys.stderr)
    weighted_bits = 0.0
    for bits, count in sorted(summary["bits_counts"].items()):
        print(f"  {bits}-bit: {count} tensors", file=sys.stderr)
        weighted_bits += bits * count
    if total:
        print(
            f"[convert_with_recipe] unweighted avg bits across "
            f"{total} quantized tensors: {weighted_bits / total:.3f} "
            "(not parameter-weighted -- see make_recipe.py's output for "
            "the parameter-weighted estimate)",
            file=sys.stderr,
        )
    if summary["collapsed_layers"]:
        print(
            f"[convert_with_recipe] collapsed {len(summary['collapsed_layers'])} "
            "fused switch_mlp (layer, projection) tensors from per-expert "
            "rules via majority vote (see this script's docstring)",
            file=sys.stderr,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-path", required=True)
    parser.add_argument("--mlx-path", required=True)
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="load the model and apply the predicate to print the bucket "
        "summary, without calling mlx_lm.convert. --hf-path must be an "
        "UNQUANTIZED (e.g. bf16) source checkpoint for this to report "
        "anything -- an already-quantized model's Linear layers have "
        "already been replaced and no longer expose to_quantized(), so "
        "the predicate never fires against one",
    )
    args = parser.parse_args()

    recipe = json.loads(args.recipe.read_text())
    default = recipe["default"]

    n_routed_experts = None
    config_path = Path(args.hf_path) / "config.json"
    if config_path.exists():
        n_routed_experts = json.loads(config_path.read_text()).get("n_routed_experts")

    predicate, summary = build_quant_predicate(recipe, n_routed_experts)

    if args.dry_run:
        import mlx.nn as nn
        import mlx_lm

        print(f"[convert_with_recipe] dry run: loading {args.hf_path}", file=sys.stderr)
        model, _ = mlx_lm.load(args.hf_path)
        model_config = json.loads(config_path.read_text()) if config_path.exists() else {}

        # mlx.nn.Module doesn't expose a public "flattened leaf paths" walk
        # outside nn.quantize itself, so drive nn.quantize directly with a
        # predicate that records decisions but never mutates the module
        # (returns False after recording) -- this reuses mlx_lm's own
        # traversal instead of reimplementing it.
        def recording_predicate(path, module):
            if not hasattr(module, "to_quantized"):
                return False
            predicate(path, module, model_config)
            return False

        nn.quantize(model, class_predicate=recording_predicate)
        print_summary(summary)
        return

    import mlx_lm

    mlx_lm.convert(
        hf_path=args.hf_path,
        mlx_path=args.mlx_path,
        quantize=True,
        q_group_size=default["group_size"],
        q_bits=default["bits"],
        quant_predicate=predicate,
    )
    print_summary(summary)


if __name__ == "__main__":
    main()
