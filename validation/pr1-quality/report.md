# Validation report: pr1-quality

- chip: Apple M4 Max
- ram_gb: 128
- macos_version: 26.5.1
- branch: claude/ds4-p1-quality-harness

## Pre-registered acceptance criteria
1. Gate results deterministic across 3 consecutive check runs (identical machine-readable summaries).
2. A deliberate logit perturbation trips the gate (non-zero exit, passed=false).

## Results

Model: mlx-community/Qwen3-1.7B-4bit (dense), 12 prompts x 64 greedy tokens.
- record -> check self-consistency: passed=true, all prompts token_exact.
- 3 consecutive check runs: byte-identical JSON summaries (sha256 equal). Criterion 1 PASS.
- check --perturb-logits 0.5: exit code 1, passed=false. Criterion 2 PASS.

Model: mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx (MoE), 12 prompts x 64 greedy tokens.
- record -> check self-consistency: passed=true, all prompts token_exact.

## Committed baselines (accuracy anchor for later PRs)
- benchmarks/quality/baselines/qwen3-1.7B-4bit.json
- benchmarks/quality/baselines/deepseek-v2-lite-4bit.json
Recorded from this branch (pre-engine-changes numerics identical to main for these paths).

Verdict: PASS
