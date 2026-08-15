#!/usr/bin/env bash
# Decode throughput smoke suite. Variables are provided by ../run.sh.
set -euo pipefail

run_side() {
    # Separate declarations: macOS bash 3.2 + set -u rejects a `local` line
    # whose later assignments expand variables assigned earlier in that line.
    local side="$1" bin_dir="$2" port="$3" pid=""
    local log_file="$OUT_DIR/raw/$side-server.log"
    local deadline ready=0 health_status

    cleanup_server() {
        if [[ -n "$pid" ]]; then
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    }
    trap cleanup_server RETURN

    "$bin_dir/higgs" serve --model "$MODEL_DIR" --port "$port" >"$log_file" 2>&1 &
    pid=$!

    deadline=$((SECONDS + 300))
    while ((SECONDS < deadline)); do
        if curl -sf "http://127.0.0.1:$port/health" >/dev/null; then
            ready=1
            break
        fi
        health_status="$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port/health" || true)"
        if [[ "$health_status" == "404" ]] && curl -sf "http://127.0.0.1:$port/" >/dev/null; then
            ready=1
            break
        fi
        sleep 2
    done
    if (( ! ready )); then
        echo "server did not become ready within 300 seconds: $side" >&2
        tail -n 30 "$log_file" >&2 || true
        return 1
    fi

    "$bin_dir/bench_decode" --host 127.0.0.1 --port "$port" --model "$MODEL_KEY" \
        --manifest "$REPO_ROOT/benchmarks/models.toml" --warmup 1 --trials "$RUNS" --format json \
        >"$OUT_DIR/raw/$side.json"
}

run_side baseline "$BASELINE_BIN_DIR" 18899
run_side candidate "$CANDIDATE_BIN_DIR" 18900
