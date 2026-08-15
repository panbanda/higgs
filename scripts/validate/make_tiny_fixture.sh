#!/usr/bin/env bash
# Create a local, random, quantized checkpoint for quality-gate plumbing.
# Prerequisites: pip install mlx-lm torch transformers tokenizers
# This is intentionally not run in CI by default.
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
output_dir="${1:-$root_dir/.tmp/quality-tiny-model}"
hf_dir="$output_dir/hf"
mlx_dir="$output_dir/mlx-4bit"

mkdir -p "$output_dir"
python3 "$root_dir/scripts/validate/tiny_model.py" --out "$hf_dir"
mlx_lm.convert --hf-path "$hf_dir" --mlx-path "$mlx_dir" -q
printf 'Tiny quantized model: %s\n' "$mlx_dir"
