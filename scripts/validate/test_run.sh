#!/usr/bin/env bash
# Regression checks for validation runner shell-script contracts.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$SCRIPT_DIR/run.sh"
SMOKE_SUITE="$SCRIPT_DIR/suites/aa-smoke.sh"

grep -Fq 'CACHE_ROOT="$HOME/.cache/higgs-validate"' "$RUNNER"
! grep -Fq 'CACHE_ROOT="${HF_HOME:-$DEFAULT_CACHE}"' "$RUNNER"
grep -Fq 'grep -Eq' "$RUNNER"
! grep -Fq 'rg -q' "$RUNNER"

grep -Fq 'trap cleanup_server RETURN' "$SMOKE_SUITE"
grep -Fq 'http://127.0.0.1:$port/health' "$SMOKE_SUITE"
grep -Fq 'sleep 2' "$SMOKE_SUITE"
grep -Fq '300' "$SMOKE_SUITE"

echo "validation runner regression checks passed"
