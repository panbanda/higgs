# Validation report: model adapter registry

- chip: Apple M4 Max, 128 GB, macOS 26.5.1
- branch: claude/model-adapters (off main f599e95)

## Why
Model support was keyed on exact `model_type` strings in two disconnected places
(`registry.rs::is_supported` and `model_loader.rs::load_model`). mlx-community/Qwen3.8-27B-4bit only
works today by accident: it declares `model_type: "qwen3_5"` (wrapper arch
`Qwen3_5ForConditionalGeneration`, nested `text_config.model_type: "qwen3_5_text"`). Any genuinely new
version string would hard-fail with no path forward short of editing two files.

## Verification (all run on hardware against real cached checkpoints)

### Behavioral equivalence vs main — PASS
Qwen3.8-27B-4bit, identical greedy request (temp 0, max_tokens 600):

| Build | content | reasoning chars | usage |
| --- | --- | ---: | --- |
| main f599e95 | '391' | 137 | prompt 67 / completion 52 |
| adapter branch | '391' | 137 | prompt 67 / completion 52 |

Byte-identical. Codex also verified Qwen3-1.7B-4bit (transformer path) and
DeepSeek-Coder-V2-Lite-4bit (deepseek_v2 path) generate normally.

### Version tolerance — PASS
Synthetic future checkpoint (/tmp fixture: `model_type: qwen3_9` top-level AND nested, weights
symlinked from Qwen3.8) loads through the `qwen3.5-dense` adapter and generates the correct answer
('391'), with an explicit log line:
  WARN higgs_models::adapter: loading an untested model version through a structurally compatible
  adapter model_type=qwen3_9 adapter="qwen3.5-dense"
Acceptance is gated on a structural check of the fields the loader actually reads; missing/ill-typed
fields error by name rather than loading garbage.

### Unknown family — PASS (still a hard error, now actionable)
`model_type: "mamba"` =>
  UnsupportedModel("mamba; supported families/version ranges: Qwen (Qwen 3.0 (Bonsai 1-bit)),
  Qwen (Qwen 3.5+ MoE), Qwen (Qwen 3.5+ dense), Qwen (Qwen 3 Next), ...")

### Regression caught during review (recorded honestly)
The first implementation preferred the nested `text_config.model_type` unconditionally, making
Qwen3.8's effective type `qwen3_5_text` — which matched no adapter, so a model that works on main
failed to load entirely. Fixed by treating nested + top-level types as CANDIDATES (exact match on any
candidate wins) and normalizing `_text` aliases family-wide, with regression tests using the real
Qwen3.8 config shape. The equivalence table above was re-measured after the fix.

### Gates
clippy -Dwarnings (higgs-models, higgs-engine) clean; fmt clean; tests: higgs-models 470+18,
higgs-engine 298, higgs 488+2+99 — all green.
