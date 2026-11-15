# Higgs vs oMLX

This document compares Higgs and [oMLX](https://github.com/jundot/omlx). The same client-side harness measured both servers. The document shows where Higgs is ahead, where the two are level, and where Higgs is behind. It also tells you how to reproduce each number.

Status: second pass, on a machine with little background load. Each run started when the 1-minute load average was below 3 and no compile was running. The load average rose to between 3 and 4.3 during some runs, because of desktop applications. Treat differences under about 5% as noise. The first pass ran while builds were running on the same machine, and this pass replaces it. Higgs ran from binaries built at commit `3c73cac`. Columns that say "main" show the first-pass numbers for the `main` commit that the branch started from (`5b49946`).

Some oMLX results changed a lot between two runs of the same model. For these models the harness ran twice: Llama-3.2-1B, Thinklite-27B and Qwen3.6-35B on oMLX, and Llama-3.2-1B and Thinklite-27B on Higgs. Where the two oMLX runs differ by more than 5%, the tables show both values as "low to high". The Higgs runs agreed within 2% on every pair that ran twice.

## Setup

| | |
|---|---|
| Machine | Apple M4 Max, 128 GB, macOS 26.5.1 (Darwin 25.5) |
| Higgs | Binaries built at commit `3c73cac` of the `perf/beat-omlx` branch, which is now merged into `main`. Later `main` commits changed CI, lint and security checks, not decode or prefill. MLX 0.32.2 via mlx-rs `adf439b` |
| oMLX | 0.7.4.dev1 at commit `bab025f`, MLX 0.32.4, mlx-lm `0d4d3d0`, installed from source into a clean venv, default settings, isolated with `--base-path` and `--no-hf-cache` |
| Sampling | temperature 0, thinking disabled on both (`reasoning.effort = none` for Higgs, `chat_template_kwargs.enable_thinking = false` for oMLX) |
| Harness | `bench_compare` (see [benchmarking.md](benchmarking.md#bench_compare)): streaming requests, server-reported token counts, medians of 3 trials after 1 warm-up (2 measured rounds for each concurrency level), one server running at a time |

oMLX's optional custom kernel extensions are not built by a source install. Its `/api/status` reported them unavailable, so its numbers here are without them. oMLX speculative decoding (MTP, DFlash) is off by default and stays off. None of the checkpoints used here ship MTP weights.

Models: `mlx-community/Llama-3.2-1B-Instruct-4bit`, `mlx-community/Qwen3-1.7B-4bit`, `mlx-community/Phi-3-mini-4k-instruct-4bit`, `Sparset/Thinklite-27B` 4-bit (Qwen3.5 dense, 27B), `mlx-community/Qwen3.6-35B-A3B-4bit` (Qwen3.5-family MoE, 35B).

## Summary

| Area | Result |
|---|---|
| Follow-up turn latency (prefix cache hit) | Higgs ahead: 1.6x to 5.3x lower time to first token (first pass: 1.2x to 3.8x) |
| Concurrent throughput, 2 to 4 clients | Higgs ahead on every model tested. The lead is 7% to 29% against the higher oMLX value in the tables, and up to 2.2x where an oMLX run was low |
| Concurrent throughput, 8 clients, small models | Level: within 2% on Llama-1B and Qwen3-1.7B (first pass: up to 6% behind) |
| Concurrent throughput, 8 clients, large models | Higgs ahead: 3% to 32% on Thinklite-27B, 1.7x to 2.0x on Qwen3.6-35B |
| Single-stream decode, dense models | Level (within 2%); both run at the memory-bandwidth limit |
| Single-stream decode, Qwen3.5-family MoE | oMLX ahead: 183 tok/s in both runs against 133 for Higgs |
| Cold prefill throughput | Level on dense models (Higgs 3% to 5% ahead on Qwen3-1.7B, 7% behind on Thinklite-27B at 5,950 tokens). oMLX is 14% to 20% ahead on the MoE model in 5 of 6 samples |
| Memory (peak RSS) | Weak comparison, because resident set size hides Metal buffers. Higgs is lower on the 1B and 1.7B models and higher on Thinklite-27B and Qwen3.6-35B. The MLX allocator cache in Higgs grew to 43 GB because the default limit is far above what the test used; see [Memory](#memory) |
| Feature surface | oMLX broader (audio, rerank, grammar, clustering, admin UI); see [Features](#features) |

## Results

### Time to first token on a follow-up turn

The conversation has about 3,000 tokens of context. Turn 2 repeats the history and adds a short message. This is the usual shape of chat and agent loops.

| Model | Higgs turn 2 | oMLX turn 2 | Higgs cold turn 1 | oMLX cold turn 1 |
|---|---|---|---|---|
| Llama-3.2-1B-4bit | 32 ms | 120 ms | 519 ms | 480 ms |
| Qwen3-1.7B-4bit | 55 ms | 288 ms | 784 ms | 741 ms |
| Phi-3-mini-4bit | 132 ms | not run | 2.37 s | not run |
| Thinklite-27B-4bit | 338 to 350 ms | 1.66 to 1.78 s | 12.3 to 12.6 s | 15.4 to 15.5 s |
| Qwen3.6-35B-A3B-4bit | 95 ms | 156 ms | 1.79 s | 2.86 to 4.26 s |

In the first pass, oMLX needed 457 ms for turn 2 on Thinklite-27B. It now needs 1.7 s on both runs. Turn 2 on Qwen3-1.7B went from 219 ms to 288 ms. This pass did not investigate the cause.

Higgs keeps prefix KV state in memory in 32-token blocks and reuses everything up to the last full block. oMLX reconstructs from its SSD block cache (256-token blocks, RAM tier off by default). On the 1B and 1.7B models, oMLX reported 2,816 cached tokens for turn 2 and Higgs reported 2,944. On the larger models, both reported about 3,000.

Prompts that share a prefix and then diverge also reuse the shared part. An example is the same system prompt and tools with a different user message, or an edited earlier turn. Dense models truncate the cached KV blocks to the shared length. The Qwen3.5/3.6 hybrid models cannot do this directly. The state of their recurrent layers is valid only at the position where Higgs captured it.

Higgs therefore checkpoints that state during prefill. It saves one checkpoint every 512 to 1,024 tokens. The default limit is 8 checkpoints per entry and 2 GiB in total. One checkpoint is 147 MiB on the 27B model. Higgs resumes from the nearest checkpoint.

On Thinklite-27B with a 3,850-token shared context, a new question went from 20.1 s to first token (cold) to 3.7 s (3,072 tokens reused). An exact repeat took 0.1 s. Resuming from a checkpoint is bit-identical to a cold prefill on real weights.

These measurements come from the first pass. This pass did not repeat them.

### Concurrent throughput

The tables show aggregate tokens per second across N simultaneous streaming requests of 256 tokens each (distinct prompts). The mean time to first token is in parentheses. Higgs runs with its default settings (batched decode). The "Higgs main" column is the first-pass run of the `main` commit.

| Model | Clients | Higgs main (first pass) | Higgs | oMLX |
|---|---|---|---|---|
| Llama-3.2-1B-4bit | 1 | 370 | 455 (27 ms) | 445 (32 ms) |
| | 2 | 603 | 786 (47 ms) | 501 to 720 (67 to 70 ms) |
| | 4 | 868 | 1144 (89 ms) | 938 (95 to 101 ms) |
| | 8 | 1002 | 1354 (161 ms) | 1366 (196 ms) |
| Qwen3-1.7B-4bit | 1 | 257 | 306 (40 ms) | 304 (54 ms) |
| | 2 | 383 | 546 (68 ms) | 512 (93 ms) |
| | 4 | 539 | 804 (114 ms) | 702 (172 ms) |
| | 8 | 629 | 945 (240 ms) | 965 (325 ms) |
| Phi-3-mini-4bit | 1 | not run | 157 (0.10 s) | not run |
| | 2 | not run | 282 (0.20 s) | not run |
| | 4 | not run | 399 (0.32 s) | not run |
| | 8 | 164 (serialized) | 453 (0.63 s) | not run |
| Thinklite-27B-4bit | 1 | 27 | 27 (0.61 s) | 24 (0.56 s) |
| | 2 | 27 (serialized) | 50 (1.1 s) | 42 (1.9 to 2.1 s) |
| | 4 | 27 (serialized) | 69 (2.2 s) | 31 to 62 (2.6 to 4.2 s) |
| | 8 | not run | 73 (4.0 s) | 56 to 71 (6.3 to 8.4 s) |
| Qwen3.6-35B-A3B-4bit | 1 | 109 | 122 (0.15 s) | 97 to 120 (0.15 s) |
| | 2 | 110 (serialized) | 193 (0.28 s) | 149 (0.55 to 0.64 s) |
| | 4 | 110 (serialized) | 262 (0.51 s) | 207 (1.0 to 1.2 s) |
| | 8 | not run | 323 (0.83 s) | 160 to 185 (1.7 to 2.1 s) |

How the results differ from the first pass:

- The 8-client lead flipped on the small models. In the first pass, oMLX was ahead by up to 6% on Qwen3-1.7B. Both servers are now within 2%.
- Higgs stays ahead at 2 and 4 clients. The lead on Qwen3-1.7B at 2 clients is 7%, the smallest lead in the table.
- One oMLX sample was low on Llama-1B at 2 clients (501 tok/s in the first run, 720 in the second). Other oMLX results varied between runs: Thinklite-27B at 4 and 8 clients, and Qwen3.6-35B at 1 and 8 clients. This pass does not explain the variation. The two Higgs models that ran twice agreed within 2%.
- Higgs on Qwen3.6-35B at 8 clients is 1.7x to 2.0x faster than oMLX. The first pass had oMLX at 195 tok/s. This pass measured 160 to 185.

Before this work, the large Qwen3.5-family models and Phi-3 could not batch. Concurrent requests waited in line. At 4 clients on Thinklite-27B, the mean time to first token was 14.9 s on `main`.

What changed in Higgs:

- The engine keeps one left-padded batched KV cache across decode steps. It no longer re-stacks every request's cache per layer per token. It rotates each row at its own position with one dynamic RoPE call. It builds step N+1 while the GPU is still computing step N.
- Short prompts that arrive together are prefilled in one padded forward pass.
- Batched decode now covers Phi-3, Starcoder2, Qwen3 MoE, Gemma 2 and Gemma 3. It also covers Gemma 4 (all but the MoE variant) and the Qwen3.5/3.6 hybrid models (dense and MoE). It also covers Llama, Mistral, Qwen2 and Qwen3 dense.
- Hybrid models batch decode only. Each prompt is prefilled on its own, then joins the batch.

### Single-stream decode

The table shows tokens per second after the first token, for one request of 256 tokens.

| Model | Higgs | oMLX | Notes |
|---|---|---|---|
| Llama-3.2-1B-4bit | 480 | 471 to 474 | level |
| Qwen3-1.7B-4bit | 328 | 322 | level |
| Phi-3-mini-4bit | 175 | not run | no oMLX result |
| Thinklite-27B-4bit | 28.9 | 29.4 | level |
| Qwen3.6-35B-A3B-4bit | 133 | 183 | oMLX ahead by 37% in both runs |

On dense models, the two servers land within 2% of each other. A 4-bit model on this hardware decodes at the speed that the memory system reads the weights, not at the speed of the serving stack. Speculative decoding is the only way past this limit. It needs MTP heads or a draft model, and no checkpoint here ships one.

On the Qwen3.5-family MoE model, stock mlx-lm decodes at 135 tok/s. Higgs was at 117 and is now at 133 after it stopped sorting expert lookups for single-token decode. oMLX decoded at 183 tok/s in both runs of this pass. It ships model-specific fused Metal kernels for the one-token MoE block of this architecture (router top-k, gate/up with SwiGLU, down with combine). In the first pass, oMLX also showed a slow mode of 86 to 130 tok/s on some requests. The slow mode did not appear in the decode scenario of this pass. Closing this gap needs equivalent fused kernels in Higgs. This work is open.

The next table lists experiments on the Higgs side, measured on Qwen3.6-35B-A3B-4bit in the first pass. This pass did not repeat them. None was kept, because each gain was within noise.

| Experiment | Decode tok/s | Finding |
|---|---|---|
| Baseline | 134 | 7.6 ms per token |
| Fused router kernel (top-k and score softmax in one launch) | 136 | routing is not where the time goes |
| Gathered 4-bit matrix-vector kernel replacing the three `gather_qmm` calls | 133 | same speed as MLX's kernel, launch for launch |
| Gate, up and SwiGLU fused into one launch (five launches to one) | 136 | fewer launches alone does not explain oMLX's lead |

Ablations show where the 7.6 ms goes: 4.0 ms outside the MoE block (GDN and attention layers, norms, head), 2.6 ms in the routed experts (three gathered products per layer), 0.7 ms in the shared expert, and 0.2 ms to combine them. oMLX's whole step is about 5.4 ms. Halving the routed-expert cost gives about 6.3 ms. Higgs also needs savings of about 0.9 ms elsewhere, for example outside the MoE block, to match oMLX. The custom kernels matched the speed of MLX and did not beat it. The next step is to profile the GPU side directly (thread layout and occupancy of the gathered products) instead of more fusion.

### Cold prefill

The table shows prompt tokens per second for an uncached prompt (prompt length divided by time to first token). Where oMLX ran twice and the two values differ by more than 5%, the table shows both.

| Model | Prompt | Higgs | oMLX |
|---|---|---|---|
| Llama-3.2-1B-4bit | 446 / 1,529 / 5,874 | 5,705 / 5,897 / 5,268 | 5,518 / 5,925 / 5,318 |
| Qwen3-1.7B-4bit | 433 / 1,516 / 5,860 | 3,615 / 3,831 / 3,429 | 3,492 / 3,725 / 3,280 |
| Phi-3-mini-4bit | 518 / 1,813 | 1,560 / 1,577 | not run |
| Thinklite-27B-4bit | 437 / 1,541 / 5,954 | 223 / 243 / 242 | 98 to 170 / 178 to 204 / 259 |
| Qwen3.6-35B-A3B-4bit | 442 / 1,539 / 5,954 | 1,305 / 1,607 / 1,646 | 1,498 / 1,927 / 1,057 to 1,966 |

Prefill is compute-bound, and both stacks use the same MLX kernels. The dense models are therefore close. oMLX is slower on the shortest Thinklite-27B prompt (98 to 170 tok/s against 223) and 7% faster on the longest. On the MoE model, oMLX is faster at 442 and 1,539 tokens (by 14% to 20%). At 5,954 tokens, the two oMLX runs gave 1,057 tok/s (36% behind Higgs) and 1,966 tok/s (19% ahead), so that result is not clear.

### Memory

The table shows the peak resident set size of the server process over the whole benchmark run, which includes the 8-client scenario.

| Model | Higgs | oMLX |
|---|---|---|
| Llama-3.2-1B-4bit | 1.2 GB | 2.3 to 2.5 GB |
| Qwen3-1.7B-4bit | 1.4 GB | 5.7 GB |
| Phi-3-mini-4bit | 2.3 GB | not run |
| Thinklite-27B-4bit | 18.6 to 18.7 GB | 16.2 GB |
| Qwen3.6-35B-A3B-4bit | 22.3 GB | 20.1 GB |

Treat this comparison as weak. Resident set size does not include most Metal buffers, and macOS can reclaim pages during a run. In one probe, the resident set size of Higgs fell from 19.3 GB to 6.7 GB after the last test.

The first-pass Higgs value of 14.9 GB on Thinklite-27B is a different measurement. That run used the serialized engine, tested 1, 2 and 4 clients only, and used different prefill sizes. The first pass did not record a value for Qwen3.6-35B.

A separate probe of Higgs on Thinklite-27B found a larger problem. The MLX allocator cache grows far beyond the weights. By default, Higgs sets only the wired limit and leaves the MLX cache limit unchanged. The MLX default is the smaller of 1.5 times the recommended working set and 0.95 times physical memory. On this machine that limit is larger than the whole benchmark uses, so it does not bound the cache in practice. The cache held 2.2 GB after the model loaded and 42.9 GB after the full benchmark. The total memory footprint of the process reached 65 GB, while the weights use 14.4 GB. Higgs does not clear this cache after prefill by default. A fix is in progress. This document does not yet report the result of the fix.

### Reliability

On `main`, loading the 27B and 35B checkpoints sometimes aborted with a Metal GPU watchdog timeout (`kIOGPUCommandBufferCallbackErrorTimeout`). It happened most often on the first load after a rebuild or from a cold disk. In the first pass it failed on 3 of about 4 such loads. Higgs now reads weights into memory before any GPU evaluation and materializes fused projections one at a time. After the fix, the first load succeeded on both models in every attempt so far. This pass added 3 more loads (Thinklite-27B twice, Qwen3.6-35B once) and all succeeded. The sample is still small.

### Limits of this pass

- The load average was not zero. Desktop applications kept it between 3 and 4.3 during some runs. Small differences may come from this.
- The oMLX results varied between two runs of the same model (see the tables). Each oMLX model ran once or twice, so the tables cannot show how large this variation is.
- oMLX on Phi-3-mini is not run. The first attempt failed, because the model has a 4,096-token context and oMLX rejected the 8,192-token prefill size. A second attempt with smaller prefill sizes did not start, because the machine was not quiet before this pass ended. The Phi-3-mini numbers for Higgs are from the first attempt. Its prefill rows leave out the 8,192-token size (7,141 tokens), which is longer than the model context. Higgs accepted that prompt without an error. This pass did not investigate that behavior.
- The Higgs numbers for the Phi-3-mini prefill rows cover prompts of 518 and 1,813 tokens.

## Features

Higgs features added on this branch to close gaps with oMLX: the OpenAI Responses API (`/v1/responses`, stored responses, `previous_response_id`), on-demand model loading with LRU and idle-TTL eviction, pinning, a memory guard and load/unload/status endpoints, cached-token and timing fields in `usage`, `/v1/tokenize` and `/v1/detokenize`, regex and `guided_json` constrained output, a configurable thinking budget, tool-call parsers for the Qwen, Mistral, Llama, Gemma, GLM and MiniMax formats with forced `tool_choice` where the format allows it, Anthropic `tool_use` output on `/v1/messages`, an MCP client (stdio and Streamable HTTP) with web search/fetch endpoints and an opt-in server-side tool loop, batching as the automatic default, and batched decode for eight more architectures.

| Capability | Higgs | oMLX |
|---|---|---|
| OpenAI chat, completions, embeddings; Anthropic messages | yes | yes |
| OpenAI Responses API | yes | yes |
| Tokenize / detokenize | yes | yes |
| Usage timing and cached-token fields | yes | yes |
| JSON schema and regex constrained output | yes | yes |
| Grammar (EBNF) constrained output | no | yes (optional extra) |
| Tool calling | Qwen, Mistral, Llama, Gemma, GLM, MiniMax formats | many model formats |
| Forced `tool_choice` (`required`, named) | Qwen, Mistral, Llama formats | yes |
| On-demand load, LRU/TTL eviction, pinning, memory guard | yes | yes |
| Continuous batching | on by default | on by default |
| In-memory prefix cache | yes | optional (off by default) |
| Disk prefix cache | opt-in | on by default |
| KV cache quantization | TurboQuant | TurboQuant, affine 4/8-bit |
| Speculative decoding | MTP, prompt lookup | MTP, DFlash, draft-model prefill |
| Vision | LLaVA-Qwen2 | broad (via mlx-vlm) |
| Audio (speech to text, text to speech, realtime) | no | yes |
| Dedicated embedding models, rerank | no | yes |
| MCP client, web search/fetch | yes | yes |
| Remote provider proxying, regex routes, auto-router | yes | no |
| OpenAI/Anthropic format translation for remote providers | yes | no |
| Metrics endpoint, terminal dashboard | yes | status endpoint, admin stats |
| Desktop app | Tauri app with request tracing | menu bar app, web admin UI |
| Model download manager, quantization tooling | no | yes |
| Multi-machine clustering | no | experimental |

Still missing in Higgs relative to oMLX: grammar (EBNF) constrained output, audio, dedicated embedding and rerank models, broad vision support, affine KV quantization, draft-model speculative decoding, a model download manager and quantization tooling, an HTTP admin UI, and clustering.

## Reproducing

```bash
# Higgs
cargo build --release -p higgs -p higgs-bench
./target/release/higgs serve --model <model dir> --port 8897 &
./target/release/bench_compare --base-url http://127.0.0.1:8897 \
  --model <model name> --label higgs --pid <server pid> \
  --extra-body '{"reasoning":{"effort":"none"}}' \
  --trials 3 --max-tokens 256

# oMLX (models as subdirectories of one directory)
omlx serve --model-dir <models dir> --no-hf-cache --base-path <scratch dir> --port 8897 --headless &
./target/release/bench_compare --base-url http://127.0.0.1:8897 \
  --model <model name> --label omlx --pid <server pid> \
  --extra-body '{"chat_template_kwargs":{"enable_thinking":false}}' \
  --trials 3 --max-tokens 256
```

Run one server at a time, and let the first request after a rebuild finish before measuring: the first model load after a new binary is several times slower than later ones.

Follow these steps before you measure:

1. Wait until the load average is low and no compile is running.
2. Send 1 request and let it finish. The harness also sends 1 warm-up request for each scenario.
3. For Phi-3-mini, add `--prefill-tokens 512 --prefill-tokens 2048` to the `bench_compare` command. This model has a 4,096-token context.
4. Run the model again when its oMLX numbers differ much from an earlier run.

Batched decode is checked against one-at-a-time decode on real weights by an ignored test:

```bash
HIGGS_MODEL_PATH=<model dir> cargo test -p higgs-engine --lib -- --ignored batched_decode_matches --nocapture
```

Dense models agree to within half-precision rounding (0.4% to 2.4% of the largest logit, the same spread the model's ordinary forward pass shows between batch sizes). MoE models show a larger spread (12% to 16%) because expert routing amplifies rounding differences between batch sizes; generated text remains coherent but can diverge from the single-request output sooner.
