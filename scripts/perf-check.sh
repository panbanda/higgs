#!/usr/bin/env bash
# Run `higgs bench` on a model and fail when decode speed regressed against a baseline.
#
#   scripts/perf-check.sh --model <dir|hf-id> --baseline <file> [--threshold <percent>]
#   scripts/perf-check.sh --model <dir|hf-id> --baseline <file> --record
#
# --record writes the run to the baseline file instead of comparing. A baseline is only
# meaningful on the same kind of machine it was recorded on; the script warns when the CPU
# differs. Environment: HIGGS_BIN (default target/release/higgs), PERF_THRESHOLD_PERCENT
# (default 10).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${HIGGS_BIN:-$ROOT/target/release/higgs}"
THRESHOLD="${PERF_THRESHOLD_PERCENT:-10}"
MODEL=""
BASELINE=""
OUTPUT=""
RECORD=0
PORT_ARGS=()

usage() {
  sed -n '2,11p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="${2:?--model needs a value}"; shift 2 ;;
    --baseline) BASELINE="${2:?--baseline needs a value}"; shift 2 ;;
    --threshold) THRESHOLD="${2:?--threshold needs a value}"; shift 2 ;;
    --output) OUTPUT="${2:?--output needs a value}"; shift 2 ;;
    --port) PORT_ARGS=(--port "${2:?--port needs a value}"); shift 2 ;;
    --record) RECORD=1; shift ;;
    -h | --help) usage ;;
    *) echo "unknown argument: $1" >&2; usage ;;
  esac
done

[[ -n "$MODEL" && -n "$BASELINE" ]] || usage
[[ -x "$BIN" ]] || { echo "higgs binary not found at $BIN (set HIGGS_BIN)" >&2; exit 2; }
if [[ "$RECORD" == 0 && ! -f "$BASELINE" ]]; then
  echo "baseline $BASELINE does not exist; record one with --record" >&2
  exit 2
fi

if [[ -z "$OUTPUT" ]]; then
  OUTPUT="$(mktemp -t higgs-perf.XXXXXX)"
  trap 'rm -f "$OUTPUT"' EXIT
fi

"$BIN" bench "$MODEL" ${PORT_ARGS[@]+"${PORT_ARGS[@]}"} \
  --scenarios decode,prefill --trials 5 --warmup 1 --max-tokens 128 \
  --prefill-tokens 2048 --json "$OUTPUT"

if [[ "$RECORD" == 1 ]]; then
  mkdir -p "$(dirname "$BASELINE")"
  cp "$OUTPUT" "$BASELINE"
  echo "recorded baseline $BASELINE"
  exit 0
fi

python3 -I - "$BASELINE" "$OUTPUT" "$THRESHOLD" <<'PY'
import json
import sys

baseline_path, current_path, threshold_text = sys.argv[1:4]
threshold = float(threshold_text)
with open(baseline_path) as handle:
    baseline = json.load(handle)
with open(current_path) as handle:
    current = json.load(handle)


def decode_rate(document):
    return document["results"]["decode"]["decode_tokps"]["median"]


def prefill_rate(document):
    sizes = document["results"].get("prefill", {}).get("sizes", [])
    return sizes[-1]["prefill_tokps_median"] if sizes else None


baseline_cpu = baseline["metadata"]["host"]["cpu"]
current_cpu = current["metadata"]["host"]["cpu"]
if baseline_cpu != current_cpu:
    print(
        f"warning: baseline was recorded on '{baseline_cpu}', this machine is '{current_cpu}'",
        file=sys.stderr,
    )

before, after = decode_rate(baseline), decode_rate(current)
change = (after - before) / before * 100.0
print(f"decode tok/s: baseline {before:.1f}, now {after:.1f} ({change:+.1f}%)")
prefill_before, prefill_after = prefill_rate(baseline), prefill_rate(current)
if prefill_before and prefill_after:
    prefill_change = (prefill_after - prefill_before) / prefill_before * 100.0
    print(
        f"prefill tok/s (informational): baseline {prefill_before:.0f}, "
        f"now {prefill_after:.0f} ({prefill_change:+.1f}%)"
    )
if change < -threshold:
    print(f"FAIL: decode speed dropped more than {threshold:g}%", file=sys.stderr)
    sys.exit(1)
print(f"OK: within {threshold:g}% of the baseline")
PY
