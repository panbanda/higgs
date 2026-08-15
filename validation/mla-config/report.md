# Validation report: mla-config (MLA latent cache config surface)

- chip: Apple M4 Max
- ram_gb: 128
- macos_version: 26.5.1
- branch: claude/ds4-mla-config-surface
- model: mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx

## Pre-registered success metrics
1. All cargo gates green; doctor coverage (valid / turboquant conflict / non-deepseek warn).
2. Hardware sanity: config mla_latent_cache=true (env unset) activates MLA in the serving engine,
   equivalently to the env knob; config absent stays dense.
3. Reviewer (codex sol, medium) MERGE OK.

## Results

### Metric 1 — PASS
cargo test: higgs 484+2+99, higgs-models 459, higgs-engine 299, all green. clippy -Dwarnings clean for
higgs-models/higgs-engine; the single higgs failure (metrics.rs unchecked_time_subtraction lint rename)
reproduces identically on unmodified main with the local toolchain and is not introduced by this change
(CI's pinned toolchain is authoritative). fmt clean. Doctor: turboquant+mla conflict errors via
KvCacheConfig::validate (single source of truth); non-deepseek_v2 model_type warns (runtime no-op).

### Metric 2 — PASS (instrument note)
Three server runs, ~7100-token greedy request each. RSS proved insensitive (Metal buffer allocations do
not show; all runs ~8.5 GB), so a binary-decisive instrument was used instead: the merged disk prefix
store persists dense prefixes as .hkv files but explicitly rejects MLA-mode prefixes with a debug log.
| Case | .hkv files written | MLA-reject logs |
| --- | ---: | ---: |
| config absent (dense) | 1 | 0 |
| config mla_latent_cache=true, env unset | 0 | 1 |
| env HIGGS_MLA_LATENT_CACHE=true, config absent | 0 | 1 |
The config-knob and env-knob runs also produced byte-identical greedy outputs. This proves the config
field reaches the engine's cache construction and behaves identically to the already-validated env path.

### Precedence (documented, unit-tested)
Explicitly-set env var wins in either direction; otherwise the config field decides; otherwise off.
Default remains OFF; the default-flip still requires a longer soak and is out of scope.

Verdict: PASS (pending reviewer verdict)
