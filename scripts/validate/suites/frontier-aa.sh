#!/usr/bin/env bash
# Context-frontier A/A suite contract: mirrors quality-gate.sh's env contract
# and the run.sh raw/ output convention consumed by report.py.
set -euo pipefail

: "${BASELINE_BIN_DIR:?BASELINE_BIN_DIR is required}"
: "${CANDIDATE_BIN_DIR:?CANDIDATE_BIN_DIR is required}"
: "${MODEL_DIR:?MODEL_DIR is required}"
: "${OUT_DIR:?OUT_DIR is required}"
: "${RUNS:?RUNS is required}"

raw_dir="$OUT_DIR/raw"
mkdir -p "$raw_dir"

if [[ ! -x "$BASELINE_BIN_DIR/bench_frontier" ]]; then
    echo "baseline build predates bench_frontier; rerun against a newer --baseline" >&2
    exit 1
fi

"$BASELINE_BIN_DIR/bench_frontier" --model-dir "$MODEL_DIR" --runs "$RUNS" --format json > "$raw_dir/frontier-baseline.json"
"$CANDIDATE_BIN_DIR/bench_frontier" --model-dir "$MODEL_DIR" --runs "$RUNS" --format json > "$raw_dir/frontier-candidate.json"
