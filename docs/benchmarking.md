# Benchmarking

This document collects the benchmark methodology and the benchmark-driven defaults referenced from the README.

## Environment

- Benchmarks in the README were run on M4 Max 128GB.
- Temperature was set to `0`.
- Warmup passes were excluded from the reported numbers.

## MLX Tuning Harness

Use the benchmark harness below to compare five serving iterations on the same local model:

```bash
python3 benchmarks/bench_mlx_tuning.py ~/.cache/lm-studio/models/mlx-community/Qwen3.6-35B-A3B-4bit
```

The harness evaluates:

- TTFT across short, medium, and long prompts
- decode throughput
- short QA accuracy
- long-context retrieval accuracy
- structured-output correctness
- prefix-cache speedup on multi-turn conversations

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
the `built` crate; you must rebuild the bench binary to refresh them.

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

### `bench_summarize`

Walks `target/bench-results/`, picks the latest result per
`(bench_name, model_key)` pair, and emits a Markdown table grouped by
model. This is what the README quotes for headline numbers.

```bash
cargo run --release -p higgs-bench --bin bench_summarize
```

### `bench_ttft_quick`

Sends one non-streaming `max_tokens=1` request at four prompt sizes
(short, medium, long, very_long) with a unique prefix per call to defeat
the prefix cache, and reports the median TTFT per size. Smallest of the
HTTP benches.

```bash
cargo run --release -p higgs-bench --bin bench_ttft_quick -- \
  --port 9999 --model deepseek-v2-lite-4bit --warmup 1 --iters 3
```

### `bench_prefix_cache`

For each model: spawns a higgs server, sends a long shared system prompt
followed by three different short user prompts, and reports the speedup
between the first (cache miss) and subsequent (cache hit) TTFTs. Pass
`--no-spawn` to use a server you already started.

```bash
cargo run --release -p higgs-bench --bin bench_prefix_cache -- \
  --models qwen3.5-35B-a3b-3bit,qwen3.5-27B-4bit
```

### `bench_prefix_cache_turns`

Multi-turn variant: builds a five-turn conversation incrementally and
reports TTFT + speedup-vs-miss at each turn, then again on two new
questions appended to the full history. Tells you whether prefix-cache
speedup degrades as context grows.

```bash
cargo run --release -p higgs-bench --bin bench_prefix_cache_turns -- \
  --models qwen3.5-35B-a3b-3bit
```

### `bench_tq_configs`

Sweeps five TurboQuant KV-cache configurations against one model
(baseline, default 3-bit, no-norm-correction, asymmetric K=4/V=3, and
layer-adaptive). For each config: spawns a higgs server with the right
flags, measures prefill TTFT + decode tok/s at three context sizes, and
generates 10 quality prompts. Quality is reported as Jaccard similarity
(per-prompt average and minimum) versus the baseline config's outputs.

```bash
cargo run --release -p higgs-bench --bin bench_tq_configs -- \
  --model qwen3.5-35B-a3b-3bit
```

### `bench_all`

Sweeps TTFT, prefill tok/s, and decode tok/s across every model in the
manifest matching `--tag` (default `all`), running three prompts of
varying length per model. Spawns and tears down one higgs server per
model.

```bash
cargo run --release -p higgs-bench --bin bench_all -- --tag all
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
