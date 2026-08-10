# Benchmarking

This document collects the benchmark methodology and the benchmark-driven defaults referenced from the README.

## Environment

- Benchmarks in the README were run on M4 Max 128GB.
- Temperature was set to `0`.
- Warmup passes were excluded from the reported numbers.

## MLX Tuning Harness

Use the Rust benchmark crate for checked-in benchmark workflows. Python
benchmark scripts are treated as local scratch files and are ignored by git.

```bash
cargo run --release -p higgs-bench --bin bench_decode -- \
  --model qwen3-1.7B-4bit --port 8899 \
  --max-tokens 200 --warmup 1 --trials 5 \
  --temperature 0
```

`bench_decode` evaluates:

- TTFT through the streaming API
- decode throughput from server-reported token usage
- reproducible JSON/Markdown output with redacted host metadata and local
  filesystem paths by default

## MTP Draft-Depth Sweep

Use the focused MTP sweep to compare baseline greedy decode with MTP disabled
against draft depths 1, 2, and 3:

```bash
cargo run --release -p higgs-bench --bin bench_speculative -- \
  --model-path trevon/Qwen3.6-27B-mtp \
  --trials baseline,mtp_default,1,2,3,prompt_lookup \
  --max-tokens 96 --repeats 1
```

The sweep sets `temperature=0`, starts a fresh Higgs server per trial, and
reports completion tokens per second plus filtered MTP/prompt-lookup telemetry
for each setting. Use `--model <key>` to target `benchmarks/models.toml`, or
`--model-path <repo-or-local-path>` for ad-hoc local runs. Local paths are not
printed in the benchmark metadata; pass `--model-name <public-id>` when using a
snapshot path whose request model name cannot be derived automatically.

Set `RUST_LOG=info` when you want the persisted JSON to include Higgs' internal
MTP decode telemetry (`cycles`, drafted tokens, accepted drafts, acceptance
rate, and decode-only tok/s).

### Qwen3.6 MTP Notes

The Qwen3Next MTP path mirrors llama.cpp's merged `draft-mtp` design in the
places that matter for speed and correctness:

- the verifier processes `[confirmed + drafts]` in one backbone batch
- the MTP cache is primed from prompt/first-token backbone hidden states
- accepted draft tokens are advanced into the MTP cache in one sequence pass
- Qwen3Next GDN convolution state is linearized before multi-token verifier
  windows, so batched verifier logits match sequential greedy decode

`HIGGS_MTP_PRIME_PREFILL=0` disables prompt/first-token MTP cache priming for
experiments. `HIGGS_MTP_MIRROR_VERIFY=1` enables full verifier-window MTP cache
mirroring; on the Qwen3.6 27B MTP 8-bit benchmark below it was slightly slower
than the default accepted-prefix replay path, so it remains opt-in.

`HIGGS_MTP_ADAPTIVE_DRAFT=1` enables a lightweight controller that grows the
draft window after high verifier acceptance and backs off after rejections.
`HIGGS_MTP_PROMPT_LOOKUP=1` enables a hybrid path that verifies repeated
prompt/history spans inside the MTP loop and mirrors accepted verifier rows into
the MTP cache, so it can fall back to normal MTP-head cycles on models without
useful repeated spans.

Measured on M4 Max 128GB, `temperature=0`, 96 completion tokens, prompt:
`Write a concise technical explanation of speculative decoding for local LLM inference...`

| Runtime | Mode | Request tok/s | Decode-only tok/s | Speedup vs runtime baseline |
| --- | ---: | ---: | ---: | ---: |
| Higgs | baseline MTP off | 14.14 | n/a | 1.00x |
| Higgs | MTP default | 22.75 | n/a | 1.61x request |
| Higgs | MTP adaptive | 22.11 | n/a | 1.56x request |
| Higgs | MTP hybrid prompt lookup + adaptive | 18.45 | n/a | 1.30x request |
| Higgs | MTP draft depth 2 | 22.79 | n/a | 1.61x request |
| llama.cpp `b9410-031ddb2e0` | baseline | 14.62 | 15.64 | 1.00x |
| llama.cpp `b9410-031ddb2e0` | `draft-mtp`, depth 2 | 21.63 | 24.16 | 1.48x request / 1.55x decode |

The Higgs numbers are from `bench_speculative`, which starts a fresh server per
mode and reports end-to-end request tok/s. The llama.cpp rows use the
OpenAI-compatible server with Qwen thinking disabled via
`chat_template_kwargs.enable_thinking=false`, prompt cache disabled, and
`draft-mtp` depth 2. On this run, Higgs MTP draft depth 2 was `1.05x` faster
than llama.cpp `draft-mtp` depth 2 at the request level.

## DFlash Speculative Decoding

DFlash pairs the target model with a small tap-fed drafter (hidden states tapped
from a handful of target layers feed a lightweight head that proposes a block of
tokens, which the target then verifies in one batched pass). Higgs is, as far as
we know, the first Rust+MLX DFlash engine; on its strong workloads it matches the
acceptance of the community Python-MLX ports.

### Characterization is by output entropy, not task label

The single best predictor of DFlash acceptance is the **target's output entropy**,
not the task name. The `dflash_entropy_sweep` harness (`#[ignore]`,
`crates/higgs-engine/src/simple.rs`) measures, per prompt and gate-OFF (raw
drafter capability): mean top-50 Shannon entropy of the target's greedy
distribution (`H_bits`), top-1 probability, and the per-round accepted-token
count (`accept_mean` + p10/p50/p90 + `accept_frac = accept_mean / block_size`).
`byte_exact` flags whether the DFlash stream is prefix-consistent with the plain
greedy AR stream.

Reproduce (set the target + drafter dirs; `block_size` defaults to 16,
`MAX_TOKENS=160`, `temperature=0`, thinking OFF):

```bash
HIGGS_DFLASH_TARGET_DIR=<target-mlx-dir> \
HIGGS_DFLASH_DRAFTER_DIR=<dflash-drafter-dir> \
cargo test --release -p higgs-engine dflash_entropy_sweep -- --ignored --nocapture
```

`accept_len`, entropy, and `byte_exact` are **thermal-independent** and carry the
characterization. Absolute tok/s on a laptop is thermally confounded (this
harness reloads the model several times and runs 16K-token prefills, so its
per-row tok/s swings widely and is not used for headline numbers); derive speedup
analytically from `accept_len` and use only paired AR/DFlash back-to-back runs on
a non-throttling machine for tok/s.

### Measured: accept_len vs entropy (M4 Max 128GB, `temperature=0`, 160 tokens)

**Qwen3.6-35B-A3B-4bit (MoE) + `modal-labs/Qwen3.6-35B-A3B-DFlash`** — block 16:

| task | H_bits | top1_prob | accept_mean | p10 | p50 | p90 | accept_frac | byte_exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| count 1..200 | 0.006 | 0.997 | 14.82 | 16 | 16 | 16 | 0.926 | true |
| multiplication tables | 0.014 | 0.997 | 14.33 | 9 | 16 | 16 | 0.896 | true |
| fixed-schema JSON | 0.019 | 0.994 | 14.00 | 10 | 16 | 16 | 0.875 | true |
| capitals list | 0.032 | 0.992 | 7.17 | 3 | 8 | 9 | 0.448 | false |
| struct getters | 0.041 | 0.989 | 7.90 | 2 | 6 | 16 | 0.494 | true |
| iterative sort | 0.106 | 0.973 | 8.47 | 2 | 6 | 16 | 0.530 | true |
| CSV table | 0.142 | 0.964 | 6.31 | 1 | 5 | 12 | 0.394 | true |
| EN-FR translation | 0.154 | 0.971 | 2.79 | 1 | 3 | 4 | 0.174 | true |
| unit conversion | 0.205 | 0.945 | 5.23 | 2 | 4 | 10 | 0.327 | false |
| GSM8K word problem | 0.209 | 0.945 | 6.68 | 1 | 5 | 15 | 0.417 | false |
| photosynthesis | 0.695 | 0.842 | 2.37 | 1 | 2 | 4 | 0.148 | false |
| short story | 1.082 | 0.762 | 1.90 | 1 | 2 | 3 | 0.119 | false |

**Qwen3.5-9B-MLX-4bit (dense) + `modal-labs/Qwen3.5-9B-DFlash`** — block 16:

| task | H_bits | top1_prob | accept_mean | p10 | p50 | p90 | accept_frac | byte_exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| count 1..200 | 0.006 | 0.998 | 14.73 | 16 | 16 | 16 | 0.920 | true |
| multiplication tables | 0.019 | 0.997 | 14.91 | 14 | 16 | 16 | 0.932 | true |
| fixed-schema JSON | 0.035 | 0.992 | 15.36 | 13 | 16 | 16 | 0.960 | true |
| capitals list | 0.064 | 0.984 | 8.58 | 2 | 8 | 16 | 0.536 | true |
| unit conversion | 0.143 | 0.963 | 6.33 | 2 | 5 | 11 | 0.396 | true |
| EN-FR translation | 0.242 | 0.953 | 3.00 | 2 | 3 | 4 | 0.188 | true |
| struct getters | 0.258 | 0.940 | 6.54 | 1 | 5 | 16 | 0.409 | false |
| CSV table | 0.288 | 0.922 | 5.73 | 1 | 4 | 12 | 0.358 | true |
| iterative sort | 0.366 | 0.906 | 6.07 | 2 | 5 | 16 | 0.380 | false |
| GSM8K word problem | 0.440 | 0.889 | 4.77 | 1 | 3 | 10 | 0.298 | false |
| photosynthesis | 0.811 | 0.829 | 3.12 | 1 | 2 | 5 | 0.195 | false |
| short story | 1.494 | 0.687 | 2.19 | 1 | 2 | 3 | 0.137 | false |

Three regimes, consistent across both models:

- **Deterministic** (`H < 0.04` bits, top1 ≈ 0.99): `accept_mean` saturates the
  block at 14–15.4, `accept_frac` 0.88–0.96, byte-exact. Counting, tables,
  fixed-schema JSON.
- **Structured / constrained** (`H` ≈ 0.04–0.4 bits): `accept_mean` 5–9. Code
  skeletons, CSV/unit tables, capitals, worked arithmetic.
- **Prose / open generation** (`H > 0.7` bits): `accept_mean` ≈ 1.9–3.1,
  `accept_frac` ≈ 0.12–0.20. Exposition, translation, story.

The two models share the curve shape; **their `accept_len`-vs-entropy curves
nearly coincide** rather than the MoE sitting above the dense model. At the
deterministic end the dense 9B is marginally higher (JSON 15.36 vs 14.00). MoE's
practical edge is therefore **economic, not higher acceptance**: with ~3B active
parameters per token its verify pass is far cheaper to amortize, so the same
`accept_len` converts into more wall-clock speedup than on a dense target.

### Analytic speedup

Each speculative round commits `accept_len` tokens for ~one batched target verify
pass; plain AR commits one token per target pass. Decode speedup is therefore
bounded above by `accept_len` and realized as roughly

```
speedup ≈ accept_len / (1 + c_draft·block + c_verify)
```

where `c_draft` is drafter cost and `c_verify` the batched-verify overhead,
both relative to one target decode step. On the MoE the denominator is near 1
(cheap active-param verify + small DFlash head), so deterministic workloads
approach the acceptance ceiling, while prose floors near parity — which is why
the production gate disables speculation once measured acceptance collapses.
Prior paired AR/DFlash spot-checks on a non-throttling run landed near **2×**
plain-AR (≈1.3× over MTP) for the 35B on low-entropy code, and **~2.5×** for the
9B on low-entropy math; this harness's per-row tok/s is thermally confounded and
is not quoted as a headline.

### Context depth does not erode acceptance — entropy does

Low-entropy acceptance holds flat as prompt depth grows to 16K tokens, validating
the drafter sliding-window port (eviction active above 4096, absolute RoPE
offset). High-entropy prose stays floored at every depth. Depth is not the
variable; entropy is.

| task | model | 512 | 4096 | 16384 |
| --- | --- | ---: | ---: | ---: |
| multiplication (low H) | MoE 35B | 14.08 | 12.92 | 14.91 |
| multiplication (low H) | dense 9B | 13.67 | 13.67 | 14.00 |
| story (high H) | MoE 35B | 2.01 | 2.18 | 2.09 |
| story (high H) | dense 9B | 2.24 | 2.11 | 2.25 |

### Block size trades fraction for raw accepted tokens

On a fixed mid-entropy task (iterative sort), a larger block commits more raw
tokens per round (`accept_mean` rises) but at a falling `accept_frac` — the
drafter runs past its predictable runway. Block 16 maximizes `accept_mean` here;
smaller blocks raise `accept_frac` (less wasted draft) at lower absolute throughput.

| block | MoE accept_mean / frac | dense accept_mean / frac |
| ---: | ---: | ---: |
| 4 | 3.81 / 0.952 | 3.53 / 0.883 |
| 8 | 7.23 / 0.903 | 5.12 / 0.641 |
| 16 | 8.47 / 0.530 | 6.07 / 0.380 |

## Iterations

The harness compares five iterations:

1. baseline
2. latency profile
3. balanced profile
4. throughput profile
5. throughput profile plus safe TurboQuant KV settings

## Benchmark-Driven Defaults

Higgs uses benchmark results to make `auto` a model-aware default rather than a static preset.

Current examples that informed the default:

- `mlx-community/Qwen3-1.7B-4bit`: `balanced` won with `91.8` composite, `339 ms` weighted TTFT, `345.7 tok/s` decode, and `20.36x` prefix-cache speedup.
- `mlx-community/Qwen3.6-35B-A3B-4bit`: `throughput` won with `95.7` composite, `842 ms` weighted TTFT, `119.2 tok/s` decode, and `56.19x` prefix-cache speedup.

That is why `auto` resolves to `balanced` for small and medium models, and `throughput` for large and huge models.

## Rust bench crate (`higgs-bench`)

The `crates/higgs-bench/` crate hosts native-Rust end-to-end bench
binaries. Each binary drives a running higgs server (or, for MLX-direct
benches, the engine in-process) and produces output that satisfies the
contract below.

### Bench output contract

Every bench binary in `higgs-bench` emits a JSON object with three
top-level keys: `metadata`, `params`, `results`.

```json
{
  "metadata": {
    "bench_name": "bench_decode",
    "bench_version": "1.0.0",
    "higgs_version": "1.0.0",
    "git_commit": "abcdef1234...",
    "git_commit_short": "abcdef1",
    "git_dirty": false,
    "started_at": "2026-04-28T00:00:00Z",
    "duration_ms": 12345,
    "host": { "hostname": "...", "os": "...", "cpu": "...", "ram_gb": 128.0, "gpu": "Apple Silicon (MLX)" },
    "mlx_version": null,
    "model": { "key": "qwen3-1.7B-4bit", "path": "...", "quantization": "4bit", "approx_size_gb": 1.2 },
    "args": ["bench_decode", "--port", "8899", "--model", "qwen3-1.7B-4bit"]
  },
  "params":  { /* bench-specific */ },
  "results": { /* bench-specific */ }
}
```

`metadata.git_commit` and `git_dirty` are captured at compile time via
the `built` crate; you must rebuild the bench binary to refresh them. Benchmark
metadata redacts hostnames by default and reduces local absolute paths in
`args`, model refs, and persisted artifact messages to public model IDs,
basenames, or relative `target/bench-results/...` paths.

Every binary supports two output formats:

- `--format json` (default) — single JSON object, machine-parseable.
- `--format markdown` — pasteable into PR descriptions. Includes a
  "How to reproduce" code fence with the exact command (re-quoted from
  `args`), a results table, and an environment table.

Every run also persists the JSON to
`target/bench-results/<bench_name>/<git_commit_short>__<model_key>__<timestamp>.json`,
regardless of `--format`. This directory is gitignored.

### Model manifest

`benchmarks/models.toml` is the source of truth for which models the
benches can target. Bench binaries take `--model <key>` and look up the
entry by key.

```toml
[[models]]
key = "qwen3-1.7B-4bit"
label = "Qwen3-1.7B-4bit (Dense)"
path = "mlx-community/Qwen3-1.7B-4bit"
quantization = "4bit"
approx_size_gb = 1.2
context = 32768
tags = ["small", "dense"]
```

Adding a model is one entry. Tags should mark size (`small`, `medium`,
`large`) and architecture (`dense`, `moe`); benches that filter by tag
will pick the model up automatically.

### `bench_decode`

Drives a running higgs server over the OpenAI streaming chat-completions
API and reports per-trial decode tok/s + TTFT.

```bash
./target/release/higgs serve --model mlx-community/Qwen3-1.7B-4bit --port 8899 &
cargo run --release -p higgs-bench --bin bench_decode -- \
  --port 8899 --model qwen3-1.7B-4bit \
  --max-tokens 200 --warmup 1 --trials 5 \
  --temperature 0.7
```

`results` is `{ trials: [...], ttft_ms_{mean,median,p95,stdev},
decode_tokps_{mean,median,p95,stdev} }`.

`ttft_ms` measures request start → first non-empty streamed content token.
`decode_tokps` measures tokens/sec **after** that first token boundary; it
uses the server-reported `completion_tokens` from the terminal `usage` chunk
(higgs honors `stream_options.include_usage: true`) and only falls back to
SSE chunk count for backends that don't emit usage. The bench also sends
`reasoning: { effort: "none" }` so decode timing reflects time-to-generate,
not time-to-visible-answer for thinking-mode models.

### `bench_speculative`

Starts a fresh Higgs server per speculative mode and compares baseline greedy
decode with MTP draft depths, adaptive MTP, hybrid prompt-lookup+MTP, and
architecture-neutral prompt lookup.

```bash
cargo run --release -p higgs-bench --bin bench_speculative -- \
  --model qwen3.6-27B-mtp-8bit \
  --trials baseline,mtp_default,mtp_adaptive,mtp_hybrid,1,2,3,prompt_lookup,prompt_lookup_unchecked \
  --max-tokens 96 --repeats 3 --format markdown
```

`results.trials[*].speedup_vs_baseline` is computed from mean completion
tokens/sec when a `baseline` trial appears before the speculative trial. Server
logs are captured under `target/bench-results/bench_speculative/logs/`, and the
JSON result stores only filtered speculative telemetry lines.

The `mtp_hybrid` trial intentionally combines MTP heads, adaptive draft depth,
and MTP-local prompt lookup. Use `prompt_lookup` for architecture-neutral prompt
lookup without MTP heads.

### `bench_summarize`

Walks `target/bench-results/`, picks the latest result per
`(bench_name, model_key)` pair, and emits a Markdown table grouped by
model. This is what the README quotes for headline numbers.

```bash
cargo run --release -p higgs-bench --bin bench_summarize
```

### Adding a new bench

1. Create `crates/higgs-bench/src/bin/<name>.rs`.
2. Capture metadata at startup with
   `let mut metadata = higgs_bench::RunMetadata::capture("<name>");`.
3. Look up the model with `higgs_bench::models::find_by_key(...)` and
   set `metadata.model`.
4. Define `Params` and `Results` structs (must implement `Serialize`).
5. Build `BenchOutput { schema_version: higgs_bench::BENCH_SCHEMA_VERSION, metadata, params, results }` and call
   `higgs_bench::persist_result(&output)` plus
   `higgs_bench::format_json` / `format_markdown` based on
   `--format`.
6. Document the new binary in this file with a one-line description and
   a sample command.

## Caveats

- Benchmark numbers depend on hardware class, prompt mix, quantization, and model family.
- README comparison tables should be read as directional comparisons rather than universal guarantees.
- If you change serving defaults or performance-sensitive behavior, rerun the harness and update any user-facing claims that depend on those results.
