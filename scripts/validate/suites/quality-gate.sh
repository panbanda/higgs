#!/usr/bin/env bash
# Quality regression suite contract: baseline records, candidate checks.
# This branch predates PR0's validate runner; it consumes its documented env.
set -euo pipefail

: "${BASELINE_BIN_DIR:?BASELINE_BIN_DIR is required}"
: "${CANDIDATE_BIN_DIR:?CANDIDATE_BIN_DIR is required}"
: "${MODEL_DIR:?MODEL_DIR is required}"
: "${OUT_DIR:?OUT_DIR is required}"
: "${RUNS:?RUNS is required}"
: "${MODEL_KEY:?MODEL_KEY is required}"

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
raw_dir="$OUT_DIR/raw"
fixture="$raw_dir/quality-fixture.json"
result="$raw_dir/quality.json"
prompts="$root_dir/benchmarks/quality/prompts.json"
committed_fixture="$root_dir/benchmarks/quality/baselines/$MODEL_KEY.json"

mkdir -p "$raw_dir"
if [[ -f "$committed_fixture" ]]; then
  fixture="$committed_fixture"
else
  if [[ ! -x "$BASELINE_BIN_DIR/quality_gate" ]]; then
    echo "baseline build predates quality_gate; commit a baseline fixture instead" >&2
    exit 1
  fi
  "$BASELINE_BIN_DIR/quality_gate" record \
    --model-dir "$MODEL_DIR" \
    --prompts "$prompts" \
    --out "$fixture"
fi

set +e
"$CANDIDATE_BIN_DIR/quality_gate" check \
  --model-dir "$MODEL_DIR" \
  --fixture "$fixture" >"$result"
status=$?
set -e

# Exit code 1 is quality_gate's contract for "ran fine, verdict failed" (see
# Ok(false) in quality_gate.rs) — result is valid JSON with passed=false, so
# let run.sh continue to report.py, which renders the FAIL verdict. Any other
# nonzero status is a real error (bad args, crash) with no reliable JSON on
# stdout, so that one still aborts the suite.
if [ "$status" -eq 1 ]; then
  printf 'quality gate check FAILED (regression detected); see %s for details\n' "$result" >&2
elif [ "$status" -ne 0 ]; then
  printf 'quality gate check errored (exit %s); machine-readable result: %s\n' "$status" "$result" >&2
  exit "$status"
fi
