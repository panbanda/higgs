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


def render_quality_section(quality_path):
    payload = json.loads(quality_path.read_text())
    verdict = "PASS" if payload["passed"] else "FAIL"
    section = f"## Quality gate\n\nVerdict: **{verdict}**\n\n"
    section += "| Prompt | Token exact | Max abs logprob delta |\n| ---: | :---: | ---: |\n"
    for prompt in payload["prompts"]:
        section += (
            f"| {prompt['prompt_index']} | {prompt['token_exact']} | "
            f"{prompt['max_abs_logprob_delta']:.8f} |\n"
        )
    return section


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--pr-id", required=True)
    parser.add_argument("--threshold", type=float, default=-5.0, help="minimum acceptable median delta percent")
    args = parser.parse_args()
    raw = args.out_dir / "raw"
    baseline_path = raw / "baseline.json"
    candidate_path = raw / "candidate.json"
    quality_path = raw / "quality.json"
    has_decode = baseline_path.exists() and candidate_path.exists()
    has_quality = quality_path.exists()
    if not has_decode and not has_quality:
        raise SystemExit(
            f"no results to report: expected decode results at {baseline_path} and {candidate_path}, "
            f"or a quality gate result at {quality_path}"
        )

    rows = ["side,run,decode_tokps"]
    metadata = ""
    metadata_path = raw / "metadata.json"
    if metadata_path.exists():
        metadata = "\n".join(f"- {key}: {value}" for key, value in json.loads(metadata_path.read_text()).items()) + "\n\n"
    report = f"# Validation report: {args.pr_id}\n\n{metadata}"

    if has_decode:
        try:
            baseline = values_for(baseline_path)
            candidate = values_for(candidate_path)
        except ValueError as error:
            raise SystemExit(error)
        base_stats, candidate_stats = stats(baseline), stats(candidate)
        delta = (candidate_stats["median"] / base_stats["median"] - 1.0) * 100.0
        verdict = "PASS" if delta >= args.threshold else "FAIL"

        rows += [f"baseline,{index},{value:.6f}" for index, value in enumerate(baseline, 1)]
        rows += [f"candidate,{index},{value:.6f}" for index, value in enumerate(candidate, 1)]

        report += "| Side | Median decode tok/s | Min | Max | Stdev |\n| --- | ---: | ---: | ---: | ---: |\n"
        report += f"| Baseline | {base_stats['median']:.2f} | {base_stats['min']:.2f} | {base_stats['max']:.2f} | {base_stats['stdev']:.2f} |\n"
        report += f"| Candidate | {candidate_stats['median']:.2f} | {candidate_stats['min']:.2f} | {candidate_stats['max']:.2f} | {candidate_stats['stdev']:.2f} |\n\n"
        report += f"Verdict: **{verdict}** — median delta {delta:+.2f}% (acceptance threshold {args.threshold:+.2f}%).\n"

    if has_quality:
        if has_decode:
            report += "\n"
        report += render_quality_section(quality_path)

    (args.out_dir / "runs.csv").write_text(scrub("\n".join(rows) + "\n"))
    (args.out_dir / "report.md").write_text(scrub(report))


if __name__ == "__main__":
    main()
