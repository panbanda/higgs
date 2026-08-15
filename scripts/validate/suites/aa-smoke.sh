#!/usr/bin/env bash
# Decode throughput smoke suite. Variables are provided by ../run.sh.
set -euo pipefail

SERVER_PIDS=""

cleanup() {
    local pid
    for pid in $SERVER_PIDS; do
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT

run_side() {
    # Separate declarations: macOS bash 3.2 + set -u rejects a `local` line
    # whose later assignments expand variables assigned earlier in that line.
    local side="$1" bin_dir="$2" port="$3" pid="" pids
    local log_file="$OUT_DIR/raw/$side-server.log"
    local deadline ready=0 health_status

    pids="$(lsof -ti :$port || true)"
    if [[ -n "$pids" ]]; then
        kill $pids 2>/dev/null || true
        sleep 2
        if [[ -n "$(lsof -ti :$port || true)" ]]; then
            echo "port $port is still bound after killing existing server" >&2
            exit 1
        fi
    fi

    "$bin_dir/higgs" serve --model "$MODEL_DIR" --port "$port" >"$log_file" 2>&1 &
    pid=$!
    SERVER_PIDS="${SERVER_PIDS:+$SERVER_PIDS }$pid"

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
        --prompt "Write a detailed essay about the history of computing, covering mechanical calculators, vacuum tubes, transistors, integrated circuits, microprocessors, and modern GPUs. Be thorough and do not stop early." \
        --max-tokens 512 --temperature 0.0 \
        >"$OUT_DIR/raw/$side.json"
}

run_side baseline "$BASELINE_BIN_DIR" 18899
run_side candidate "$CANDIDATE_BIN_DIR" 18900
