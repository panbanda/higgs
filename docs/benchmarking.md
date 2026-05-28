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

Measured on M4 Max 128GB, `temperature=0`, 96 completion tokens, prompt:
`Write a concise technical explanation of speculative decoding for local LLM inference...`

| Runtime | Mode | Request tok/s | Decode-only tok/s | Speedup vs runtime baseline |
| --- | ---: | ---: | ---: | ---: |
| Higgs | baseline MTP off | 14.32 | n/a | 1.00x |
| Higgs | MTP draft depth 2 | 22.89 | 28.0 | 1.60x request / 1.96x vs request baseline |
| llama.cpp `b1-d374e71` | baseline | n/a | 15.9 | 1.00x |
| llama.cpp `b1-d374e71` | MTP draft depth 1 | n/a | 25.0 | 1.57x |
| llama.cpp `b1-d374e71` | MTP draft depth 2 | n/a | 24.3 | 1.53x |

The Higgs request-level number includes HTTP and prompt processing; llama.cpp's
CLI line reports generation only. The closest decode-only comparison from this
run is Higgs MTP depth 2 at `28.0 tok/s` versus llama.cpp's best measured MTP
setting at `25.0 tok/s`.

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
decode with MTP draft depths and architecture-neutral prompt lookup.

```bash
cargo run --release -p higgs-bench --bin bench_speculative -- \
  --model qwen3.6-27B-mtp-8bit \
  --trials baseline,mtp_default,1,2,3,prompt_lookup,prompt_lookup_unchecked \
  --max-tokens 96 --repeats 3 --format markdown
```

`results.trials[*].speedup_vs_baseline` is computed from mean completion
tokens/sec when a `baseline` trial appears before the speculative trial. Server
logs are captured under `target/bench-results/bench_speculative/logs/`, and the
JSON result stores only filtered speculative telemetry lines.

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
5. Build `BenchOutput { metadata, params, results }` and call
   `higgs_bench::persist_result(&output)` plus
   `higgs_bench::format_json` / `format_markdown` based on
   `--format`.
6. Document the new binary in this file with a one-line description and
   a sample command.

## Caveats

- Benchmark numbers depend on hardware class, prompt mix, quantization, and model family.
- README comparison tables should be read as directional comparisons rather than universal guarantees.
- If you change serving defaults or performance-sensitive behavior, rerun the harness and update any user-facing claims that depend on those results.
