#!/usr/bin/env python3
"""Render a scrubbed validation report from bench_decode JSON output."""

import argparse
import csv
import json
import statistics
from pathlib import Path

from scrub import scrub


def values_for(path):
    payload = json.loads(path.read_text())
    trials = payload["results"]["trials"]
    for index, trial in enumerate(trials, 1):
        if trial["tokens_after_first"] < 64:
            raise ValueError(
                f"{path} trial {index}: workload finished too early to measure decode throughput "
                f"(tokens_after_first={trial['tokens_after_first']}, minimum=64)"
            )
    return [float(trial["decode_tokps"]) for trial in trials]


def stats(values):
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--pr-id", required=True)
    parser.add_argument("--threshold", type=float, default=-5.0, help="minimum acceptable median delta percent")
    args = parser.parse_args()
    raw = args.out_dir / "raw"
    try:
        baseline = values_for(raw / "baseline.json")
        candidate = values_for(raw / "candidate.json")
    except ValueError as error:
        raise SystemExit(error)
    base_stats, candidate_stats = stats(baseline), stats(candidate)
    delta = (candidate_stats["median"] / base_stats["median"] - 1.0) * 100.0
    verdict = "PASS" if delta >= args.threshold else "FAIL"

    rows = ["side,run,decode_tokps"]
    rows += [f"baseline,{index},{value:.6f}" for index, value in enumerate(baseline, 1)]
    rows += [f"candidate,{index},{value:.6f}" for index, value in enumerate(candidate, 1)]
    (args.out_dir / "runs.csv").write_text(scrub("\n".join(rows) + "\n"))

    metadata = ""
    metadata_path = raw / "metadata.json"
    if metadata_path.exists():
        metadata = "\n".join(f"- {key}: {value}" for key, value in json.loads(metadata_path.read_text()).items()) + "\n\n"
    report = f"# Validation report: {args.pr_id}\n\n{metadata}| Side | Median decode tok/s | Min | Max | Stdev |\n| --- | ---: | ---: | ---: | ---: |\n"
    report += f"| Baseline | {base_stats['median']:.2f} | {base_stats['min']:.2f} | {base_stats['max']:.2f} | {base_stats['stdev']:.2f} |\n"
    report += f"| Candidate | {candidate_stats['median']:.2f} | {candidate_stats['min']:.2f} | {candidate_stats['max']:.2f} | {candidate_stats['stdev']:.2f} |\n\n"
    report += f"Verdict: **{verdict}** — median delta {delta:+.2f}% (acceptance threshold {args.threshold:+.2f}%).\n"
    (args.out_dir / "report.md").write_text(scrub(report))


if __name__ == "__main__":
    main()
