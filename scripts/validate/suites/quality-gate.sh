#!/usr/bin/env bash
# Quality regression suite contract: baseline records, candidate checks.
# This branch predates PR0's validate runner; it consumes its documented env.
set -euo pipefail

: "${BASELINE_BIN_DIR:?BASELINE_BIN_DIR is required}"
: "${CANDIDATE_BIN_DIR:?CANDIDATE_BIN_DIR is required}"
: "${MODEL_DIR:?MODEL_DIR is required}"
: "${OUT_DIR:?OUT_DIR is required}"
: "${RUNS:?RUNS is required}"

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
raw_dir="$OUT_DIR/raw"
fixture="$raw_dir/quality-fixture.json"
result="$raw_dir/quality.json"
prompts="$root_dir/benchmarks/quality/prompts.json"

mkdir -p "$raw_dir"
"$BASELINE_BIN_DIR/quality_gate" record \
  --model-dir "$MODEL_DIR" \
  --prompts "$prompts" \
  --out "$fixture"

set +e
"$CANDIDATE_BIN_DIR/quality_gate" check \
  --model-dir "$MODEL_DIR" \
  --fixture "$fixture" >"$result"
status=$?
set -e

if [ "$status" -ne 0 ]; then
  printf 'quality gate failed; machine-readable result: %s\n' "$result" >&2
  exit "$status"
fi
