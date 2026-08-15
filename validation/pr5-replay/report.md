# Validation report: pr5-replay (exact tool-call replay) — NEGATIVE RESULT

- chip: Apple M4 Max
- ram_gb: 128
- macos_version: 26.5.1
- branch: claude/ds4-p5-toolcall-replay
- model: mlx-community/Qwen3-1.7B-4bit

## Pre-registered acceptance criteria
1. Re-render prefix breaks ~0 on a scripted multi-turn tool session.
2. Turn-2+ TTFT improvement demonstrated.
3. Final outputs token-identical.

## Mechanism verification (works as designed)
- End-to-end engagement confirmed: tool_replay_hits=3, misses=0 on a metrics-enabled server run.
- Byte-identical re-render of the model's original tool-call serialization: unit-tested, including a
  decoy-region guard (literal <tool_call> in user content is never touched; semantic identity of the
  region is verified before splicing).

## System-level results: no measurable benefit
Scripted 2-turn tool sessions (greedy), long shared system prompt, baseline (main) vs candidate:

| Scenario | Baseline median turn-2 TTFT | Candidate | Delta |
| --- | ---: | ---: | ---: |
| Small tool call (weather), thinking on, 8 sessions | 57.5 ms | 57.5 ms | ~0 |
| Large payload (save_document ~300-400 words), thinking on, 6 sessions | 121.0 ms | 102.6 ms | within run noise (spreads overlap) |

Debug-level prefix-cache traces show WHY, three structural reasons in higgs's serving path:
1. Thinking models: turn-1 generation includes reasoning tokens that are cached but stripped from the
   re-rendered turn-2 prompt, so the token divergence occurs BEFORE the tool call. Replaying the
   tool-call bytes cannot extend the prefix past a divergence that precedes it. (Traces: identical
   prefix_len=1024 hits on both sides in the large-payload scenario.)
2. The paged prefix cache reuses 64-token blocks; a sub-block extension from replay is invisible.
   (Traces: both sides hit prefix_len=2176 with the cached sequence ending at ~2234 < the 2240
   boundary.)
3. Typical tool calls (~30-60 tokens) span at most one block.
The remaining beneficiary — multi-block tool payloads with thinking disabled — could not be
demonstrated: with thinking off, Qwen3-1.7B stopped producing tool calls for document-sized payloads
(0/12 sessions).

Criterion 1: not achieved in the thinking-on configuration users actually run (divergence precedes
the call). Criterion 2: FAIL — no demonstrated improvement. Criterion 3: 6/8 sessions byte-identical;
the 2 diffs stem from replay intentionally restoring the model's original serialization (different
prompt bytes than the normalized baseline), a semantic-neutral but not byte-neutral change — noted as
a caveat rather than a pass.

## Verdict: FAIL — do not merge
The implementation is correct and guarded, but in higgs's current architecture the optimization has
no measurable effect. Preconditions for revisiting: (a) reasoning-aware prefix caching (cache keyed on
the re-render form), (b) sub-block prefix reuse, or (c) agentic workloads with multi-block tool
payloads on non-thinking models. Negative result to be recorded in docs/ds4-analysis-2026-08.md.
