#!/usr/bin/env bash
# Run a repeatable baseline/candidate validation suite on macOS.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
CACHE_ROOT="$HOME/.cache/higgs-validate"
BUILD_CACHE="$CACHE_ROOT/builds"
WORKTREE_CACHE="$CACHE_ROOT/worktrees"
RUNS="${RUNS:-5}"
ACCEPTANCE_THRESHOLD="${ACCEPTANCE_THRESHOLD:--5}"

usage() {
    echo "usage: scripts/validate/run.sh [--baseline <ref>] <pr-id>" >&2
    echo "       scripts/validate/run.sh --self-test" >&2
}

self_test() {
    local out
    out="$(mktemp -d "${TMPDIR:-/tmp}/higgs-validate.XXXXXX")"
    trap 'rm -rf "$out"' RETURN
    mkdir -p "$out/raw"
    python3 - "$out/raw/baseline.json" "$out/raw/candidate.json" "$out/raw/metadata.json" <<'PY'
import json
import sys
for path, values in zip(sys.argv[1:3], ([100.0, 101.0], [102.0, 103.0])):
    with open(path, "w") as handle:
        json.dump({"results": {"trials": [{"decode_tokps": value, "tokens_after_first": 64} for value in values]}}, handle)
with open(sys.argv[3], "w") as handle:
    json.dump({"machine": "/Users/alice@buildbox.local 192.168.1.5", "token": "ghp_abcdefghijklmnopqrstuvwxyz123456"}, handle)
PY
    python3 "$SCRIPT_DIR/report.py" --out-dir "$out" --pr-id self-test --threshold -5
    test -s "$out/report.md"
    test -s "$out/runs.csv"
    if grep -Eq 'alice|buildbox|192\.168\.1\.5|ghp_' "$out/report.md" "$out/runs.csv"; then
        echo "self-test failed: PII remained in rendered output" >&2
        return 1
    fi
    echo "self-test passed: report and runs.csv rendered without planted PII"
}

detect_model() {
    local tier manifest
    local ram_bytes ram_gb
    ram_bytes="$(sysctl -n hw.memsize)"
    ram_gb=$((ram_bytes / 1024 / 1024 / 1024))
    if ((ram_gb < 32)); then tier="small"; elif ((ram_gb <= 64)); then tier="medium"; else tier=""; fi
    manifest="$REPO_ROOT/benchmarks/models.toml"
    python3 - "$manifest" "$tier" <<'PY'
import sys
import tomllib

models = tomllib.load(open(sys.argv[1], "rb"))["models"]
tier = sys.argv[2]
selected = next((model for model in models if not tier or tier in model.get("tags", [])), None)
if selected is None:
    raise SystemExit(f"no model matches tier {tier!r}")
print(selected["key"])
print(selected["path"])
PY
}

download_model() {
    local model_path="$1" tool output snapshot_dir
    # Prefer hf: huggingface-cli is deprecated and newer installs ship a stub
    # that exits with an error telling you to use hf.
    if command -v hf >/dev/null 2>&1; then tool="hf"; elif command -v huggingface-cli >/dev/null 2>&1; then tool="huggingface-cli"; else
        echo "missing model download tool: install hf (huggingface_hub)" >&2
        exit 1
    fi
    output="$("$tool" download "$model_path")"
    snapshot_dir="${output##*$'\n'}"
    [[ -n "$snapshot_dir" ]] || { echo "model download did not print a snapshot path" >&2; exit 1; }
    printf '%s\n' "$snapshot_dir"
}

build_ref() {
    local sha="$1"
    local source_dir="$REPO_ROOT"
    local build_dir="$BUILD_CACHE/$sha"
    local shared_target_dir="$CACHE_ROOT/target"
    local metallib_path
    if [[ "$sha" != "$CANDIDATE_SHA" ]]; then
        source_dir="$WORKTREE_CACHE/$sha"
        if [[ ! -e "$source_dir/.git" ]]; then
            mkdir -p "$WORKTREE_CACHE"
            git -C "$REPO_ROOT" worktree add --detach "$source_dir" "$sha"
        fi
    fi
    if [[ ! -x "$build_dir/release/higgs" ]] || [[ ! -x "$build_dir/release/bench_decode" ]]; then
        (cd "$source_dir" && CARGO_TARGET_DIR="$shared_target_dir" cargo build --release -p higgs -p higgs-bench)
        mkdir -p "$build_dir/release"
        cp "$shared_target_dir/release/higgs" "$build_dir/release/higgs"
        cp "$shared_target_dir/release/bench_decode" "$build_dir/release/bench_decode"
    fi
    if [[ -f "$shared_target_dir/release/mlx.metallib" ]]; then
        cp "$shared_target_dir/release/mlx.metallib" "$build_dir/release/mlx.metallib"
    fi
    metallib_path="$(find "$shared_target_dir/release/build" -path "*/mlx-sys-*/out/build/lib/mlx.metallib" -type f -exec ls -t {} + 2>/dev/null | head -n 1 || true)"
    if [[ -z "$metallib_path" ]]; then
        echo "missing mlx.metallib in shared target directory: $shared_target_dir" >&2
        exit 1
    fi
    cp "$metallib_path" "$build_dir/release/mlx.metallib"
    printf '%s\n' "$build_dir/release"
}

if [[ "${1:-}" == "--self-test" ]]; then
    [[ $# -eq 1 ]] || { usage; exit 2; }
    self_test
    exit 0
fi

BASELINE_REF="main"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --baseline) BASELINE_REF="${2:-}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        --*) usage; exit 2 ;;
        *) PR_ID="$1"; shift ;;
    esac
done
[[ -n "${PR_ID:-}" ]] || { usage; exit 2; }
[[ "$PR_ID" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "invalid pr-id: $PR_ID" >&2; exit 2; }

CANDIDATE_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
BASELINE_SHA="$(git -C "$REPO_ROOT" rev-parse "$BASELINE_REF")"
MODEL_OUTPUT="$(detect_model)"
MODEL_KEY="${MODEL_OUTPUT%%$'\n'*}"
MODEL_PATH="${MODEL_OUTPUT#*$'\n'}"
CHIP="$(sysctl -n machdep.cpu.brand_string)"
RAM_GB=$(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))
MACOS_VERSION="$(sw_vers -productVersion)"
MODEL_DIR="$(download_model "$MODEL_PATH")"
BASELINE_BIN_DIR="$(build_ref "$BASELINE_SHA")"
CANDIDATE_BIN_DIR="$(build_ref "$CANDIDATE_SHA")"
OUT_DIR="$REPO_ROOT/validation/$PR_ID"
mkdir -p "$OUT_DIR/raw"
CHIP="$CHIP" RAM_GB="$RAM_GB" MACOS_VERSION="$MACOS_VERSION" BASELINE_SHA="$BASELINE_SHA" CANDIDATE_SHA="$CANDIDATE_SHA" MODEL_KEY="$MODEL_KEY" python3 - "$OUT_DIR/raw/metadata.json" <<'PY'
import json
import os
import sys
json.dump({
    "chip": os.environ["CHIP"],
    "ram_gb": int(os.environ["RAM_GB"]),
    "macos_version": os.environ["MACOS_VERSION"],
    "baseline_sha": os.environ["BASELINE_SHA"],
    "candidate_sha": os.environ["CANDIDATE_SHA"],
    "model": os.environ["MODEL_KEY"],
}, open(sys.argv[1], "w"))
PY
export BASELINE_BIN_DIR CANDIDATE_BIN_DIR MODEL_DIR OUT_DIR RUNS MODEL_KEY REPO_ROOT
# shellcheck source=/dev/null
source "$SCRIPT_DIR/suites/$PR_ID.sh"
python3 "$SCRIPT_DIR/report.py" --out-dir "$OUT_DIR" --pr-id "$PR_ID" --threshold "$ACCEPTANCE_THRESHOLD"
echo "wrote $OUT_DIR/report.md and $OUT_DIR/runs.csv"
