# Validation report: p2-calibration (ds4 P2 calibration half) — QUALITY CLAIM NEGATIVE

- chip: Apple M4 Max
- ram_gb: 128
- macos_version: 26.5.1
- branch: claude/ds4-p2-calibration
- models: deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct (bf16 reference);
  mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx (uniform baseline);
  locally converted asymmetric checkpoint (this branch's tooling)

## Pre-registered success metrics (recorded before any results)
1. QUALITY: asymmetric >= uniform-4bit against the bf16 reference at equal-or-smaller bytes
   (median over 12 prompts of per-prompt max |delta logprob|, teacher-forced, 64 greedy tokens;
   token-exact count not lower by more than 1).
2. LOADING: the converted asymmetric checkpoint loads and generates in higgs.
3. Tooling smoke-tested; CI green.
Decision rule (pre-registered): if metric 1 fails, tooling merges ONLY with this committed negative
report; the quality claim is recorded as a negative result.

## Method
Calibration: 24 committed texts through the bf16 model (6,121 tokens), collecting per-layer expert
routing frequencies + input second moments (imatrix). Recipe (layer granularity — mlx stores routed
experts as one fused tensor per layer/projection, so per-expert bits are unexpressible at convert
time; documented): 7/26 lowest-salience MoE layers' experts at 3-bit, 19 at 4-bit, everything else
6-bit, gs64; solved against an effective-bpw budget including per-group scale/bias overhead.
Conversion via mlx_lm.convert with a recipe-driven quant predicate. Scoring: higgs quality_gate
teacher-forced against a bf16-recorded fixture (identical protocol both variants).

## Results

### Bytes precondition — MET
Asymmetric 8,682,671,902 bytes vs uniform 8,840,088,702 bytes (1.8% smaller; 4.42 vs ~4.5 eff. bpw).

### Metric 1 QUALITY — FAIL
| Variant | median max|dlogprob| | mean | token-exact |
| --- | ---: | ---: | ---: |
| uniform 4-bit | 3.41 | 3.04 | 0/12 |
| asymmetric (this recipe) | 4.18 | 3.30 | 0/12 |
Per-prompt deltas: uniform [2.89, 1.81, 1.28, 4.91, 3.33, 0.80, 3.59, 3.49, 3.54, 4.22, 4.15, 2.43];
asymmetric [4.13, 0.30, 1.78, 5.43, 4.18, 0.82, 5.04, 4.36, 4.18, 4.65, 4.42, 0.36].
Interpretation: at the ~4.5 effective-bpw budget on DeepSeek-V2-Lite, funding 6-bit attention/shared
weights by dropping 7 expert layers to 3-bit hurts more than it helps — the uniform allocation wins.
The asymmetric variant wins on 3/12 prompts (sometimes by a lot) but loses the aggregate. The ds4
recipe's regime is much lower bit budgets (2-3 bit experts vs 6-8 bit rest); at this budget the claim
does not transfer. Recorded as a negative result for this budget/model; the tooling enables retesting
other budgets cheaply.

### Metric 2 LOADING — PASS, after two real loader fixes
Scoring the converted checkpoint surfaced two genuine loader bugs in the merged #260 code, fixed on
this branch with tests: (a) complete predicate maps (mlx_lm emits an entry for EVERY tensor) tripped
the loud-failure guard on entries that merely restate the scalar default — default-equal overrides
are now no-ops; (b) routed-expert overrides in real converted checkpoints live at the fused
model.layers.N.mlp.switch_mlp.{proj} path, which the DeepSeek-V2 loader did not recognize (only a
per-expert-indexed convention) — real mixed DeepSeek checkpoints could not load at all. After the
fixes the asymmetric checkpoint loads and teacher-forces cleanly (quality_gate exit 0).

### Metric 3 — PASS
Collector/recipe smoke-tested on the cached 4-bit model (26 MoE layers, routing freqs sum to top-k);
end-to-end conversion verified; scripts are stdlib+mlx only.

Verdict: tooling + loader fixes merge with this negative quality result recorded; the asymmetric-
quality claim itself is CLOSED NEGATIVE at the 4.5-bpw budget (results table updated accordingly).
