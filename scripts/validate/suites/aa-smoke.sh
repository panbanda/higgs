#!/usr/bin/env bash
# Decode throughput smoke suite. Variables are provided by ../run.sh.
set -euo pipefail

run_side() {
    local side="$1" bin_dir="$2" port="$3" pid
    "$bin_dir/higgs" serve --model "$MODEL_DIR" --port "$port" >"$OUT_DIR/raw/$side-server.log" 2>&1 &
    pid=$!
    "$bin_dir/bench_decode" --host 127.0.0.1 --port "$port" --model "$MODEL_KEY" \
        --manifest "$REPO_ROOT/benchmarks/models.toml" --warmup 1 --trials "$RUNS" --format json \
        >"$OUT_DIR/raw/$side.json"
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
}

run_side baseline "$BASELINE_BIN_DIR" 18899
run_side candidate "$CANDIDATE_BIN_DIR" 18900
