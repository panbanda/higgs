#!/usr/bin/env python3
"""Perplexity & quality benchmark: Higgs TurboQuant vs baseline.

Measures:
  1. Ground-truth PPL via mlx-lm (no server, direct model)
  2. KLD between baseline and TQ via Higgs server logprobs
  3. Decode speed at various context lengths (up to 32K)
  4. Sparse V skip rate estimation

Usage:
    python3 benchmarks/bench_ppl_tq.py <model_path> [--ctx 2048] [--stride 512] [--chunks 50]

Requires: mlx-lm, datasets
"""

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
import urllib.request

import mlx.core as mx
import numpy as np

HIGGS_BIN = os.environ.get("HIGGS_BIN", "./target/release/higgs")


# ---------------------------------------------------------------------------
# Part 1: Ground-truth perplexity via mlx-lm
# ---------------------------------------------------------------------------

def compute_ppl_mlx(model_path, ctx_len=2048, stride=512, max_chunks=50):
    """Compute perplexity using mlx-lm direct inference (no server)."""
    from datasets import load_dataset
    from mlx_lm import load, stream_generate

    print(f"\n{'='*60}")
    print("PART 1: Ground-truth PPL via mlx-lm")
    print(f"  Model: {model_path}")
    print(f"  Context: {ctx_len}, Stride: {stride}, Max chunks: {max_chunks}")
    print(f"{'='*60}")

    # Load dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([x["text"] for x in ds if x["text"].strip()])

    # Load model + tokenizer
    print("Loading model...")
    model, tokenizer = load(model_path)
    print(f"  Model loaded. Peak memory: {mx.get_peak_memory() / 1e9:.1f} GB")

    # Tokenize full text
    tokens = tokenizer.encode(text)
    print(f"  Total tokens: {len(tokens)}")

    # Sliding window PPL
    nlls = []
    n_tokens = 0
    chunk_count = 0

    for start in range(0, len(tokens) - ctx_len, stride):
        if chunk_count >= max_chunks:
            break

        chunk = tokens[start : start + ctx_len]
        input_ids = mx.array(chunk[:-1])[None, :]  # (1, T-1)
        targets = mx.array(chunk[1:])               # (T-1,)

        # Forward pass to get logits
        logits = model(input_ids)
        logits = logits.squeeze(0)  # (T-1, vocab)

        # Only score tokens in the non-overlapping region (after stride)
        if start > 0:
            score_start = ctx_len - stride
        else:
            score_start = 0

        score_logits = logits[score_start:]
        score_targets = targets[score_start:]

        # Log softmax → gather target logprobs
        log_probs = mx.softmax(score_logits, axis=-1)
        log_probs = mx.log(log_probs + 1e-10)

        target_logprobs = mx.take_along_axis(
            log_probs, score_targets[:, None], axis=1
        ).squeeze(-1)

        nll = -target_logprobs.sum().item()
        count = score_targets.shape[0]
        nlls.append(nll)
        n_tokens += count
        chunk_count += 1

        ppl_so_far = math.exp(sum(nlls) / n_tokens)
        mem = mx.get_peak_memory() / 1e9

        if chunk_count % 10 == 0 or chunk_count <= 3:
            print(f"  Chunk {chunk_count}: PPL={ppl_so_far:.3f} "
                  f"(tokens={n_tokens}, mem={mem:.1f}GB)")

        mx.eval(target_logprobs)  # force eval to free graph

    final_ppl = math.exp(sum(nlls) / n_tokens)
    print(f"\n  FINAL PPL: {final_ppl:.4f} ({n_tokens} tokens, {chunk_count} chunks)")
    print(f"  Peak memory: {mx.get_peak_memory() / 1e9:.1f} GB")

    # Free model
    del model
    mx.clear_cache()

    return final_ppl, n_tokens


# ---------------------------------------------------------------------------
# Part 2: Server-based KLD and decode benchmarks
# ---------------------------------------------------------------------------

def api(port, endpoint, body, timeout=300):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/{endpoint}",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read())
    elapsed = time.perf_counter() - t0
    return result, elapsed


def start_server(model_path, port, kv_mode=None, kv_bits=3):
    cmd = [HIGGS_BIN, "serve", "--model", model_path, "--port", str(port)]
    if kv_mode:
        cmd += ["--kv-cache", kv_mode, "--kv-bits", str(kv_bits), "--kv-seed", "0"]
    env = {**os.environ, "HIGGS_ENABLE_THINKING": "0", "HIGGS_NO_CONFIG": "1"}
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for _ in range(90):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2)
            return proc
        except Exception:
            time.sleep(1)
    proc.kill()
    raise RuntimeError(f"Server on port {port} failed to start")


def stop_server(proc):
    if proc and proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def get_model_id(port):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/models")
    with urllib.request.urlopen(req, timeout=5) as resp:
        data = json.loads(resp.read())
    return data["data"][0]["id"]


WORD = "the quick brown fox jumps over the lazy dog "  # ~10 tokens


def chat(port, model, messages, max_tokens=256, temperature=0, logprobs=False,
         top_logprobs=None):
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if logprobs:
        body["logprobs"] = True
        if top_logprobs is not None:
            body["top_logprobs"] = top_logprobs
    result, elapsed = api(port, "chat/completions", body)
    usage = result.get("usage", {})
    choice = result.get("choices", [{}])[0]
    content = choice.get("message", {}).get("content", "")
    lp = choice.get("logprobs", None)
    return content, usage, elapsed, lp



def test_quality_sequential(model_path, port, bits):
    """Quality comparison using one server at a time (safe for 32GB RAM)."""
    print(f"\n{'='*60}")
    print("PART 2: Output quality — baseline vs TQ (sequential)")
    print(f"{'='*60}")

    prompts = [
        "Explain how a hash table works step by step.",
        "What causes the seasons on Earth?",
        "Write a Python function to find the longest common subsequence.",
        "Describe the process of photosynthesis in detail.",
        "What is the difference between TCP and UDP?",
        "Explain the theory of general relativity in simple terms.",
        "How does a neural network learn through backpropagation?",
        "What are the main causes of climate change?",
        "Describe the water cycle and its importance.",
        "How does public key cryptography work?",
    ]

    # Step 1: Collect baseline outputs
    print("\n  Collecting baseline outputs...")
    proc = start_server(model_path, port)
    model_id = get_model_id(port)
    base_outputs = []
    for i, prompt in enumerate(prompts):
        content, usage, elapsed, lp = chat(
            port, model_id, [{"role": "user", "content": prompt}],
            max_tokens=64, logprobs=True, top_logprobs=5,
        )
        base_outputs.append(content)
        print(f"    [{i+1}/{len(prompts)}] {len(content)} chars")
    stop_server(proc)
    print("  Baseline server stopped.")
    time.sleep(3)

    # Step 2: Collect TQ outputs
    print(f"\n  Collecting TQ-{bits}bit outputs...")
    proc = start_server(model_path, port, kv_mode="turboquant", kv_bits=bits)
    model_id = get_model_id(port)
    tq_outputs = []
    for i, prompt in enumerate(prompts):
        content, usage, elapsed, lp = chat(
            port, model_id, [{"role": "user", "content": prompt}],
            max_tokens=64, logprobs=True, top_logprobs=5,
        )
        tq_outputs.append(content)
        print(f"    [{i+1}/{len(prompts)}] {len(content)} chars")
    stop_server(proc)
    print("  TQ server stopped.")
    time.sleep(3)

    # Step 3: Compare
    results = []
    print(f"\n  {'Prompt':<52} | {'Jaccard':>8} | {'Verdict':>10}")
    print("  " + "-" * 75)
    for prompt, c_base, c_tq in zip(prompts, base_outputs, tq_outputs):
        w_base = set(c_base.lower().split())
        w_tq = set(c_tq.lower().split())
        jaccard = (
            len(w_base & w_tq) / len(w_base | w_tq) if (w_base or w_tq) else 0
        )
        verdict = "MATCH" if jaccard > 0.5 else "DIVERGED"
        print(f"  {prompt[:50]:<52} | {jaccard:>7.2f} | {verdict:>10}")
        results.append({"prompt": prompt[:50], "jaccard": jaccard})

    avg = sum(r["jaccard"] for r in results) / len(results) if results else 0
    print(f"\n  Average Jaccard: {avg:.3f}")
    return results, avg


def test_decode_context_sweep(port, model, label, context_sizes=None):
    """Decode speed at various context lengths.

    Uses two non-streaming requests per context to separate TTFT from decode:
      1. max_tokens=1 → elapsed ≈ TTFT (prefill + 1 token)
      2. max_tokens=128 → elapsed = prefill + N tokens
      decode_tps ≈ (ctoks - 1) / (elapsed_128 - elapsed_1)
    """
    if context_sizes is None:
        context_sizes = [100, 1000, 4000, 8000, 16000, 24000, 32000]

    print(f"\n{'='*60}")
    print(f"PART 3: Decode speed vs context length — {label}")
    print(f"{'='*60}")
    print(f"  {'Context':>10} | {'Decode tok/s':>14} | {'TTFT ms':>10} | {'Gen toks':>10}")
    print("  " + "-" * 55)

    results = []
    for ctx_tokens in context_sizes:
        repeat = max(1, ctx_tokens // 10)
        prompt = WORD * repeat

        try:
            # Request 1: TTFT only (max_tokens=1, unique prefix)
            pfx1 = f"[a{time.time():.6f}] Summarize: "
            _, u1, t1, _ = chat(port, model,
                [{"role": "user", "content": pfx1 + prompt}],
                max_tokens=1)
            ptoks = u1.get("prompt_tokens", 0)
            ttft_s = t1

            # Request 2: Full generation (max_tokens=128, different unique prefix)
            pfx2 = f"[b{time.time():.6f}] Summarize: "
            _, u2, t2, _ = chat(port, model,
                [{"role": "user", "content": pfx2 + prompt}],
                max_tokens=128)
            ctoks = u2.get("completion_tokens", 0)

            if ctoks > 1:
                # decode_time ≈ full_time - ttft
                decode_time = max(t2 - ttft_s, 0.01)
                tps = (ctoks - 1) / decode_time
                print(f"  {ptoks:>10} | {tps:>11.1f} t/s | "
                      f"{ttft_s*1000:>7.0f} ms | {ctoks:>10}")
                results.append({"ctx": ptoks, "tps": tps,
                                "ttft": ttft_s * 1000})
            else:
                print(f"  {ptoks:>10} | {'NO OUTPUT':>14} | "
                      f"{ttft_s*1000:>7.0f} ms | {ctoks:>10}")
                results.append({"ctx": ptoks, "tps": 0, "ttft": ttft_s * 1000})

        except Exception as e:
            print(f"  {ctx_tokens:>10} | {'FAILED':>14} | {str(e)[:40]:>40}")
            results.append({"ctx": ctx_tokens, "tps": 0, "error": str(e)[:50]})
            if "memory" in str(e).lower() or "timeout" in str(e).lower():
                print("  *** Stopping sweep — memory or timeout error ***")
                break

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TurboQuant PPL & quality benchmark")
    parser.add_argument("model_path")
    parser.add_argument("--ctx", type=int, default=2048, help="Context length for PPL")
    parser.add_argument("--stride", type=int, default=512, help="Stride for PPL sliding window")
    parser.add_argument("--chunks", type=int, default=50, help="Max chunks for PPL eval")
    parser.add_argument("--bits", type=int, default=3, help="TQ bit width")
    parser.add_argument("--port", type=int, default=8097, help="Base port")
    parser.add_argument("--skip-ppl", action="store_true", help="Skip mlx-lm PPL (memory intensive)")
    parser.add_argument("--skip-server", action="store_true", help="Skip server-based tests")
    args = parser.parse_args()

    print("=" * 60)
    print("HIGGS TURBOQUANT — PPL & QUALITY BENCHMARK")
    print(f"Model: {args.model_path}")
    print(f"TQ bits: {args.bits}")
    print(f"PPL context: {args.ctx}, stride: {args.stride}, chunks: {args.chunks}")
    print("=" * 60)

    # Part 1: Ground-truth PPL
    ppl = None
    if not args.skip_ppl:
        try:
            ppl, n_tokens = compute_ppl_mlx(
                args.model_path,
                ctx_len=args.ctx,
                stride=args.stride,
                max_chunks=args.chunks,
            )
        except Exception as e:
            print(f"\n  PPL computation failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n[Skipping mlx-lm PPL — --skip-ppl flag]")

    if args.skip_server:
        print("\n[Skipping server tests — --skip-server flag]")
        return

    # Part 2 & 3: Server-based tests (sequential — one server at a time)
    port = args.port

    # --- Baseline server ---
    print(f"\n{'='*60}")
    print("Starting BASELINE server (no TQ)...")
    proc_base = None
    try:
        proc_base = start_server(args.model_path, port)
        model_base = get_model_id(port)
        print(f"  Ready: {model_base}")

        # Decode sweep — baseline
        sweep_base = test_decode_context_sweep(
            port, model_base, "baseline",
            context_sizes=[100, 1000, 4000, 8000, 16000],
        )
    finally:
        stop_server(proc_base)
        print("Baseline server stopped.")
        time.sleep(2)  # let port free

    # --- TQ server ---
    print(f"\n{'='*60}")
    print(f"Starting TURBOQUANT server ({args.bits}-bit)...")
    proc_tq = None
    try:
        proc_tq = start_server(args.model_path, port, kv_mode="turboquant", kv_bits=args.bits)
        model_tq = get_model_id(port)
        print(f"  Ready: {model_tq}")

        # Decode sweep — TQ (push further since KV is compressed)
        sweep_tq = test_decode_context_sweep(
            port, model_tq, f"turboquant-{args.bits}bit",
            context_sizes=[100, 1000, 4000, 8000, 16000, 24000, 32000],
        )
    finally:
        stop_server(proc_tq)
        print("TQ server stopped.")
        time.sleep(2)

    # --- Quality comparison: single-server sequential (safe for 32GB) ---
    kld_results, avg_jaccard = test_quality_sequential(
        args.model_path, port, args.bits
    )

    # --- Final summary ---
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")

    if ppl is not None:
        print(f"\n  Ground-truth PPL (mlx-lm): {ppl:.4f}")

    if sweep_base and sweep_tq:
        print(f"\n  Decode speed comparison:")
        print(f"    {'Context':>10} | {'Baseline':>12} | {'TQ {}-bit':>12} | {'Speedup':>8}".format(args.bits))
        print("    " + "-" * 50)
        base_map = {r["ctx"]: r["tps"] for r in sweep_base}
        for r in sweep_tq:
            ctx = r["ctx"]
            tps_tq = r["tps"]
            tps_base = base_map.get(ctx, 0)
            if tps_base > 0 and tps_tq > 0:
                speedup = tps_tq / tps_base
                print(f"    {ctx:>10} | {tps_base:>9.1f} t/s | {tps_tq:>9.1f} t/s | {speedup:>7.2f}x")
            elif tps_tq > 0:
                print(f"    {ctx:>10} | {'N/A':>12} | {tps_tq:>9.1f} t/s | {'':>8}")
            else:
                err = r.get("error", "failed")
                print(f"    {ctx:>10} | {'':>12} | {err:>12} | {'':>8}")

    if avg_jaccard > 0:
        print(f"\n  Output quality (Jaccard similarity): {avg_jaccard:.3f}")
        print(f"    > 0.5 = good, > 0.7 = very similar, < 0.3 = diverged")

    print()


if __name__ == "__main__":
    main()
