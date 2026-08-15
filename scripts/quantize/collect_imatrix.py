#!/usr/bin/env python3
"""Collect expert-routing and activation statistics from a DeepSeek-V2 MoE
checkpoint by running calibration text through prefill-only forward passes.

Usage:
    collect_imatrix.py --model-dir <bf16 or quantized model dir> \\
        --texts calibration_texts.txt --max-tokens-per-text 512 \\
        --out imatrix.json

The output feeds make_recipe.py, which turns these statistics into a
per-tensor quantization recipe for mlx_lm.convert (see README.md in this
directory for the full ds4 P2 calibration workflow).

Architecture notes (see README.md "Module structure" section for the full
writeup): mlx_lm's `mlx_lm.models.deepseek_v2` represents each MoE block as
`DeepseekV2MoE`, which owns a `MoEGate` (`self.gate`) and a fused
`SwitchGLU` (`self.switch_mlp`) covering all routed experts as one stacked
tensor per projection. `MoEGate.__call__` already does the top-k
partitioning and returns `(inds, scores)` — `inds` are the selected expert
indices per token and `scores` are the corresponding (post-softmax,
pre-normalization option, routed_scaling_factor-scaled) routing weights.
That is exactly the signal we need for routing statistics, so we monkeypatch
`MoEGate.__call__` and `DeepseekV2MoE.__call__` at the *class* level (mlx.nn
modules are plain Python objects whose `__call__` is looked up on the type,
so assigning an instance attribute named `__call__` is silently ignored --
we patch the shared class method once and use a `_imatrix_layer_idx`
instance attribute, set while walking the model, to know which layer a given
call belongs to).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx_lm


class LayerStats:
    """Accumulates routing and activation statistics for one MoE layer."""

    def __init__(self, n_routed_experts: int) -> None:
        self.n_routed_experts = n_routed_experts
        self.expert_select_count = [0] * n_routed_experts
        self.expert_weight_sum = [0.0] * n_routed_experts
        self.n_tokens_routed = 0
        self.input_sq_sum = 0.0
        self.input_sq_count = 0

    def record_routing(self, inds: mx.array, scores: mx.array) -> None:
        # inds/scores: (..., top_k); flatten to token-major python lists.
        # Calibration batches are small (single sequence, <=max-tokens-per-text
        # tokens), so a python-side reduction is simple and fast enough --
        # avoids leaning on mx scatter-add ops for this one-off tool.
        flat_inds = inds.reshape(-1, inds.shape[-1]).tolist()
        flat_scores = scores.reshape(-1, scores.shape[-1]).tolist()
        for token_inds, token_scores in zip(flat_inds, flat_scores):
            self.n_tokens_routed += 1
            for expert_idx, weight in zip(token_inds, token_scores):
                self.expert_select_count[expert_idx] += 1
                self.expert_weight_sum[expert_idx] += weight

    def record_input(self, x: mx.array) -> None:
        squared = (x.astype(mx.float32) ** 2).sum()
        self.input_sq_sum += float(squared)
        self.input_sq_count += x.size

    def to_dict(self, layer_idx: int) -> dict:
        topk_freq = [
            (count / self.n_tokens_routed) if self.n_tokens_routed else 0.0
            for count in self.expert_select_count
        ]
        mean_weight = [
            (self.expert_weight_sum[i] / self.expert_select_count[i])
            if self.expert_select_count[i]
            else 0.0
            for i in range(self.n_routed_experts)
        ]
        input_sq_mean = (
            self.input_sq_sum / self.input_sq_count if self.input_sq_count else 0.0
        )
        return {
            "layer": layer_idx,
            "expert_topk_freq": topk_freq,
            "expert_mean_weight": mean_weight,
            "input_sq_mean": input_sq_mean,
        }


def load_texts(path: Path) -> list[str]:
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        if not isinstance(data, list) or not all(isinstance(t, str) for t in data):
            raise ValueError(f"{path} must contain a JSON list of strings")
        return data
    # One prompt per physical line -- multi-line texts (e.g. code samples)
    # must encode their newlines as a literal "\n" escape within that one
    # line (see calibration_texts.txt), which we unescape here.
    lines = [line.strip().replace("\\n", "\n") for line in path.read_text().splitlines()]
    return [line for line in lines if line]


def find_moe_layers(model) -> list[tuple[int, object]]:
    """Return [(layer_idx, mlp_module), ...] for every MoE layer.

    Raises a clear error if the model doesn't look like the DeepSeek-V2
    architecture this tool was built for (dense mlp.{gate,up,down}_proj for
    some layers, MoE mlp.{gate,switch_mlp} for others).
    """
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise SystemExit(
            "collect_imatrix.py expects a DeepSeek-V2-style model with a "
            "`model.model.layers` list; the loaded model doesn't have that "
            f"attribute (got {type(model).__name__})."
        )
    layers = model.model.layers
    moe_layers = []
    for idx, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            raise SystemExit(
                f"layer {idx} has no `.mlp` submodule; this doesn't look "
                "like a DeepSeek-V2 decoder layer."
            )
        if hasattr(mlp, "gate") and hasattr(mlp, "switch_mlp"):
            moe_layers.append((idx, mlp))
        elif hasattr(mlp, "gate_proj") and hasattr(mlp, "down_proj"):
            continue  # dense mlp layer (e.g. first_k_dense_replace), skip
        else:
            raise SystemExit(
                f"layer {idx} `.mlp` is neither a recognized MoE block "
                "(`.gate` + `.switch_mlp`) nor a dense block (`.gate_proj` + "
                f"`.down_proj`); got attributes: {sorted(vars(mlp).keys())}"
            )
    if not moe_layers:
        raise SystemExit(
            "no MoE layers found (no layer had both `.mlp.gate` and "
            "`.mlp.switch_mlp`) -- this model doesn't appear to be a "
            "DeepSeek-V2 MoE checkpoint."
        )
    return moe_layers


def install_hooks(moe_layers: list[tuple[int, object]], n_routed_experts: int):
    stats: dict[int, LayerStats] = {
        idx: LayerStats(n_routed_experts) for idx, _ in moe_layers
    }

    gate_classes = set()
    moe_classes = set()
    for idx, mlp in moe_layers:
        mlp._imatrix_layer_idx = idx
        mlp.gate._imatrix_layer_idx = idx
        gate_classes.add(type(mlp.gate))
        moe_classes.add(type(mlp))

    if len(gate_classes) != 1 or len(moe_classes) != 1:
        raise SystemExit(
            "expected all MoE layers to share one gate class and one MoE "
            f"block class, found gate classes {gate_classes} and MoE block "
            f"classes {moe_classes}; refusing to patch a heterogeneous model."
        )

    gate_cls = gate_classes.pop()
    moe_cls = moe_classes.pop()
    orig_gate_call = gate_cls.__call__
    orig_moe_call = moe_cls.__call__

    def patched_gate_call(self, x):
        inds, scores = orig_gate_call(self, x)
        stats[self._imatrix_layer_idx].record_routing(inds, scores)
        return inds, scores

    def patched_moe_call(self, x):
        stats[self._imatrix_layer_idx].record_input(x)
        return orig_moe_call(self, x)

    gate_cls.__call__ = patched_gate_call
    moe_cls.__call__ = patched_moe_call

    def restore():
        gate_cls.__call__ = orig_gate_call
        moe_cls.__call__ = orig_moe_call

    return stats, restore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--texts", required=True, type=Path)
    parser.add_argument("--max-tokens-per-text", type=int, default=512)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    print(f"[collect_imatrix] loading model from {args.model_dir}", file=sys.stderr)
    model, tokenizer = mlx_lm.load(args.model_dir)

    moe_layers = find_moe_layers(model)
    n_routed_experts = model.args.n_routed_experts
    print(
        f"[collect_imatrix] found {len(moe_layers)} MoE layers "
        f"({n_routed_experts} routed experts/layer, "
        f"top-{model.args.num_experts_per_tok})",
        file=sys.stderr,
    )

    stats, restore = install_hooks(moe_layers, n_routed_experts)

    texts = load_texts(args.texts)
    if not texts:
        raise SystemExit(f"no calibration texts found in {args.texts}")
    print(f"[collect_imatrix] running {len(texts)} calibration texts", file=sys.stderr)

    n_tokens_total = 0
    try:
        for i, text in enumerate(texts):
            token_ids = tokenizer.encode(text)[: args.max_tokens_per_text]
            if not token_ids:
                continue
            inputs = mx.array([token_ids])
            model(inputs)  # prefill only; no generation/sampling needed.
            # The hooks above call .tolist()/float() on their captured
            # arrays, which forces mlx's lazy graph to evaluate immediately,
            # so no explicit mx.eval() is needed here.
            n_tokens_total += len(token_ids)
            print(
                f"[collect_imatrix] text {i + 1}/{len(texts)}: "
                f"{len(token_ids)} tokens",
                file=sys.stderr,
            )
    finally:
        restore()

    config_path = Path(args.model_dir) / "config.json"
    model_config = (
        json.loads(config_path.read_text()) if config_path.exists() else None
    )

    output = {
        "model": str(args.model_dir),
        "n_tokens": n_tokens_total,
        "layers": [
            stats[idx].to_dict(idx) for idx, _ in moe_layers
        ],
        # Embedded so make_recipe.py can compute parameter-weighted bit
        # budgets without needing the model directory again.
        "config": model_config,
    }
    args.out.write_text(json.dumps(output, indent=2))
    print(
        f"[collect_imatrix] wrote {args.out} "
        f"({len(output['layers'])} layers, {n_tokens_total} tokens)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
