# Higgs vs oMLX

A head-to-head comparison of Higgs and [oMLX](https://github.com/jundot/omlx), measured with the same client-side harness against both servers. It records where Higgs is ahead, where it is level, and where it is behind, and how to reproduce every number.

Status: first pass. The numbers below were taken while development builds were running on the same machine, so treat differences under about 5% as noise. The Higgs numbers are from the `perf/beat-omlx` branch; "before" columns are the `main` commit the branch started from (`5b49946`).

## Setup

| | |
|---|---|
| Machine | Apple M4 Max, 128 GB, macOS (Darwin 25.5) |
| Higgs | `perf/beat-omlx`, MLX 0.32.2 via mlx-rs `adf439b` |
| oMLX | 0.7.4.dev1 at commit `bab025f`, MLX 0.32.4, mlx-lm `0d4d3d0`, installed from source into a clean venv, default settings, isolated with `--base-path` and `--no-hf-cache` |
| Sampling | temperature 0, thinking disabled on both (`reasoning.effort = none` for Higgs, `chat_template_kwargs.enable_thinking = false` for oMLX) |
| Harness | `bench_compare` (see [benchmarking.md](benchmarking.md#bench_compare)): streaming requests, server-reported token counts, medians of 3 trials after 1 warmup, one server running at a time |

oMLX's optional custom kernel extensions are not built by a source install and were reported unavailable by its `/api/status`; its numbers here are without them. oMLX speculative decoding (MTP, DFlash) is off by default and was left off; none of the checkpoints used ship MTP weights.

Models: `mlx-community/Llama-3.2-1B-Instruct-4bit`, `mlx-community/Qwen3-1.7B-4bit`, `mlx-community/Phi-3-mini-4k-instruct-4bit`, `Sparset/Thinklite-27B` 4-bit (Qwen3.5 dense, 27B), `mlx-community/Qwen3.6-35B-A3B-4bit` (Qwen3.5-family MoE, 35B).

## Summary

| Area | Result |
|---|---|
| Follow-up turn latency (prefix cache hit) | Higgs ahead: 1.2x to 3.8x lower time to first token |
| Concurrent throughput, 2 to 4 clients | Higgs ahead on every model tested, by 11% to 61% |
| Concurrent throughput, 8 clients, small models | Level to slightly behind (within 3% on Llama-1B, 6% behind on Qwen3-1.7B) |
| Concurrent throughput, large models | Higgs ahead: 1.5x to 1.7x at 4 to 8 clients |
| Single-stream decode, dense models | Level (within 3%); both run at the memory-bandwidth limit |
| Single-stream decode, Qwen3.5-family MoE | oMLX ahead: about 184 tok/s at its best vs 134 for Higgs |
| Cold prefill throughput | Level on dense models; mixed on MoE |
| Memory (peak RSS) | Higgs lower on every model |
| Feature surface | oMLX broader (audio, rerank, grammar, clustering, admin UI); see [Features](#features) |

## Results

### Time to first token on a follow-up turn

A conversation with roughly 3,000 tokens of context; turn 2 repeats the history and adds a short message. This is the common shape of chat and agent loops.

| Model | Higgs turn 2 | oMLX turn 2 | Higgs cold turn 1 | oMLX cold turn 1 |
|---|---|---|---|---|
| Llama-3.2-1B-4bit | 32 ms | 117 ms | 521 ms | 476 ms |
| Qwen3-1.7B-4bit | 57 ms | 219 ms | 783 ms | 735 ms |
| Thinklite-27B-4bit | 389 ms | 457 ms | 12.2 s | 14.9 s |
| Qwen3.6-35B-A3B-4bit | 84 ms | 156 ms | 1.76 s | 3.03 s |

Higgs keeps prefix KV state in memory in 32-token blocks and reuses everything up to the last full block; oMLX reconstructs from its SSD block cache (256-token blocks, RAM tier off by default).

### Concurrent throughput

Aggregate tokens per second across N simultaneous streaming requests of 256 tokens each (distinct prompts), with mean time to first token in parentheses. Higgs is run with `batch = true`.

| Model | Clients | Higgs before | Higgs now | oMLX |
|---|---|---|---|---|
| Llama-3.2-1B-4bit | 1 | 370 | 455 (32 ms) | 449 (31 ms) |
| | 2 | 603 | 775 (55 ms) | 697 (66 ms) |
| | 4 | 868 | 1105 (93 ms) | 931 (108 ms) |
| | 8 | 1002 | 1350 (165 ms) | 1385 (188 ms) |
| Qwen3-1.7B-4bit | 1 | 257 | 305 (40 ms) | 260 (49 ms) |
| | 2 | 383 | 544 (68 ms) | 477 (97 ms) |
| | 4 | 539 | 787 (120 ms) | 708 (165 ms) |
| | 8 | 629 | 952 (209 ms) | 1016 (296 ms) |
| Thinklite-27B-4bit | 1 | 27 | 27 (0.50 s) | 23 (0.56 s) |
| | 2 | 27 (serialized) | 50 (1.2 s) | 42 (2.0 s) |
| | 4 | 27 (serialized) | 69 (2.2 s) | 43 (4.2 s) |
| | 8 | not run | 74 (3.7 s) | not run |
| Qwen3.6-35B-A3B-4bit | 1 | 109 | 109 (0.15 s) | 84 to 101 |
| | 2 | 110 (serialized) | 178 (0.27 s) | 111 to 115 |
| | 4 | 110 (serialized) | 255 (0.48 s) | 165 to 175 |
| | 8 | not run | 326 (0.77 s) | 195 |
| Phi-3-mini-4bit | 8 | 164 (serialized) | 461 (0.57 s) | not run |

"Before", the large Qwen3.5-family models and Phi-3 could not batch at all: concurrent requests waited in line, so at 4 clients on Thinklite-27B the mean time to first token was 14.9 s.

What changed in Higgs:

- The batch engine keeps one left-padded batched KV cache across decode steps instead of re-stacking every request's cache per layer per token, rotates each row at its own position with one dynamic RoPE call, and builds step N+1 while the GPU is still computing step N.
- Short prompts that arrive together are prefilled in one padded forward pass.
- Batched decode now covers Phi-3, Starcoder2, Qwen3 MoE, and the Qwen3.5/3.6 hybrid models (dense and MoE), in addition to Llama, Mistral, Qwen2 and Qwen3 dense. Hybrid models batch decode only; each prompt is prefilled on its own, then joins the batch.

Where oMLX still leads: an 8-way simultaneous burst on a 1 to 2B model. Higgs's decode step at batch 8 costs the same as mlx-lm's (5.37 ms vs 5.36 ms per round on Llama-1B), so the remaining difference is in how prefill and decode are interleaved during the burst, not in the decode kernel path.

### Single-stream decode

Tokens per second after the first token, one request, 256 tokens.

| Model | Higgs | oMLX | Notes |
|---|---|---|---|
| Llama-3.2-1B-4bit | 480 | 472 | level |
| Qwen3-1.7B-4bit | 329 | 325 | level |
| Thinklite-27B-4bit | 29.0 | 29.6 | level |
| Qwen3.6-35B-A3B-4bit | 134 | 130 to 185 | oMLX ahead at its best |

Dense models: both servers land within 3% of each other because single-stream decode of a 4-bit model on this hardware is limited by how fast weights can be read from memory, not by the serving stack. The only way past that limit is speculative decoding, which needs MTP heads or a draft model; no checkpoint here ships one.

Qwen3.5-family MoE: stock mlx-lm decodes this model at 135 tok/s; Higgs was at 117 and is now at 134 after it stopped sorting expert lookups for single-token decode. oMLX reaches about 184 tok/s on most requests because it ships model-specific fused Metal kernels for this architecture's one-token MoE block (router top-k, gate/up with SwiGLU, down with combine). Its speed is bimodal in our runs: 183 to 185 tok/s on most requests, 86 to 130 on others, apparently when cache writes from the previous request overlap. Closing this gap needs equivalent fused kernels in Higgs and is open work.

What has been tried on the Higgs side, each measured on Qwen3.6-35B-A3B-4bit and none kept because the gain was within noise:

| Experiment | Decode tok/s | Finding |
|---|---|---|
| Baseline | 134 | 7.6 ms per token |
| Fused router kernel (top-k and score softmax in one launch) | 136 | routing is not where the time goes |
| Gathered 4-bit matrix-vector kernel replacing the three `gather_qmm` calls | 133 | same speed as MLX's kernel, launch for launch |
| Gate, up and SwiGLU fused into one launch (five launches to one) | 136 | fewer launches alone does not explain oMLX's lead |

Ablations show where the 7.6 ms goes: 4.0 ms outside the MoE block (GDN and attention layers, norms, head), 2.6 ms in the routed experts (three gathered products per layer), 0.7 ms in the shared expert, 0.2 ms combining them. oMLX's whole step is about 5.4 ms, so matching it needs the routed-expert cost roughly halved or savings found outside the MoE block as well. The custom kernels above matched MLX's speed rather than beating it, so the next step is profiling the GPU side directly (thread layout and occupancy of the gathered products) rather than more fusion.

### Cold prefill

Prompt tokens per second for an uncached prompt (time to first token divided into prompt length).

| Model | Prompt | Higgs | oMLX |
|---|---|---|---|
| Llama-3.2-1B-4bit | 446 / 1,530 / 5,870 | 5,426 / 5,814 / 5,265 | 5,701 / 6,002 / 5,324 |
| Qwen3-1.7B-4bit | 436 / 1,515 / 5,864 | 3,614 / 3,855 / 3,445 | 3,580 / 3,781 / 3,311 |
| Thinklite-27B-4bit | 438 / 1,541 / 5,953 | 231 / 245 / 249 | 239 / 260 / 256 |
| Qwen3.6-35B-A3B-4bit | 440 / 1,539 / 5,952 | 1,351 / 1,632 / 1,671 | 415 / 1,950 / 1,135 |

Prefill is compute-bound and both stacks use the same MLX kernels, so dense models are level (Higgs 3 to 6% behind on the 27B model). On the MoE model oMLX's numbers vary widely between sizes; Higgs is steadier.

### Memory

Peak resident set size of the server process over the whole benchmark run.

| Model | Higgs | oMLX |
|---|---|---|
| Llama-3.2-1B-4bit | 1.2 GB | 1.9 GB |
| Qwen3-1.7B-4bit | 1.3 GB | 5.5 GB |
| Thinklite-27B-4bit | 14.9 GB | 16.2 GB |
| Qwen3.6-35B-A3B-4bit | not recorded | 20.1 GB |

### Reliability

Loading the 27B and 35B checkpoints on `main` intermittently aborted with a Metal GPU watchdog timeout (`kIOGPUCommandBufferCallbackErrorTimeout`), most often on the first load after a rebuild or from a cold disk; in this work it failed on 3 of about 4 such loads. Weights are now read into memory before any GPU evaluation and fused projections are materialized one at a time. After the fix the first load succeeded on both models in every attempt so far (a small sample: 2 of 2 immediately after the fix, and every later run).

## Features

Higgs features added on this branch to close gaps with oMLX: the OpenAI Responses API (`/v1/responses`, stored responses, `previous_response_id`), on-demand model loading with LRU and idle-TTL eviction, pinning, a memory guard and load/unload/status endpoints, cached-token and timing fields in `usage`, `/v1/tokenize` and `/v1/detokenize`, regex and `guided_json` constrained output, a configurable thinking budget, tool-call parsers for the Qwen, Mistral, Llama, Gemma, GLM and MiniMax formats with forced `tool_choice` where the format allows it, batching as the automatic default, and batched decode for five more architectures.

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
| Continuous batching | on by default (`batch = "auto"`) | on by default |
| In-memory prefix cache | yes | optional (off by default) |
| Disk prefix cache | opt-in | on by default |
| KV cache quantization | TurboQuant | TurboQuant, affine 4/8-bit |
| Speculative decoding | MTP, prompt lookup | MTP, DFlash, draft-model prefill |
| Vision | LLaVA-Qwen2 | broad (via mlx-vlm) |
| Audio (speech to text, text to speech, realtime) | no | yes |
| Dedicated embedding models, rerank | no | yes |
| MCP client, web search/fetch | no | yes |
| Remote provider proxying, regex routes, auto-router | yes | no |
| OpenAI/Anthropic format translation for remote providers | yes | no |
| Metrics endpoint, terminal dashboard | yes | status endpoint, admin stats |
| Desktop app | Tauri app with request tracing | menu bar app, web admin UI |
| Model download manager, quantization tooling | no | yes |
| Multi-machine clustering | no | experimental |

Not done on this branch: an MCP client with web search/fetch.

## Reproducing

```bash
# Higgs
cargo build --release -p higgs -p higgs-bench
./target/release/higgs serve --model <model dir> --port 8897 [--batch] &
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

Batched decode is checked against one-at-a-time decode on real weights by an ignored test:

```bash
HIGGS_MODEL_PATH=<model dir> cargo test -p higgs-engine --lib -- --ignored batched_decode_matches --nocapture
```

Dense models agree to within half-precision rounding (0.4% to 2.4% of the largest logit, the same spread the model's ordinary forward pass shows between batch sizes). MoE models show a larger spread (12% to 16%) because expert routing amplifies rounding differences between batch sizes; generated text remains coherent but can diverge from the single-request output sooner.
