#!/usr/bin/env bash
set -euo pipefail

: "${BASELINE_BIN_DIR:?BASELINE_BIN_DIR is required}"
: "${CANDIDATE_BIN_DIR:?CANDIDATE_BIN_DIR is required}"
: "${MODEL_DIR:?MODEL_DIR is required}"
: "${OUT_DIR:?OUT_DIR is required}"
: "${RUNS:?RUNS is required}"

mkdir -p "$OUT_DIR"

"$BASELINE_BIN_DIR/bench_frontier" --model-dir "$MODEL_DIR" --runs "$RUNS" --format json > "$OUT_DIR/frontier-baseline.json"
"$CANDIDATE_BIN_DIR/bench_frontier" --model-dir "$MODEL_DIR" --runs "$RUNS" --format json > "$OUT_DIR/frontier-candidate.json"
