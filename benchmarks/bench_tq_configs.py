#!/usr/bin/env python3
"""Benchmark prefill, decode, and quality across TurboQuant configurations.

Runs each config sequentially (one server at a time) to stay safe on 32GB.

Usage:
    python3 benchmarks/bench_tq_configs.py <model_path>
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

HIGGS_BIN = os.environ.get("HIGGS_BIN", "./target/release/higgs")
WORD = "the quick brown fox jumps over the lazy dog "  # ~10 tokens
PORT = 8097


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


def start_server(model_path, port, extra_args=None):
    cmd = [HIGGS_BIN, "serve", "--model", model_path, "--port", str(port)]
    if extra_args:
        cmd += extra_args
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


def chat(port, model, messages, max_tokens=256, temperature=0):
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    result, elapsed = api(port, "chat/completions", body)
    usage = result.get("usage", {})
    choice = result.get("choices", [{}])[0]
    content = choice.get("message", {}).get("content", "")
    return content, usage, elapsed


# ---- Benchmarks per config ----

def measure_ttft(port, model, prompt):
    """TTFT ≈ time for max_tokens=1 request."""
    pfx = f"[ttft{time.time():.6f}] "
    _, usage, elapsed = chat(port, model,
        [{"role": "user", "content": pfx + prompt}], max_tokens=1)
    return elapsed, usage.get("prompt_tokens", 0)


def measure_decode(port, model, prompt, gen_tokens=128):
    """Decode speed = (gen_tokens - 1) / (full_time - ttft)."""
    # TTFT
    ttft_s, ptoks = measure_ttft(port, model, prompt)

    # Full generation
    pfx2 = f"[dec{time.time():.6f}] Summarize: "
    _, usage2, elapsed2 = chat(port, model,
        [{"role": "user", "content": pfx2 + prompt}], max_tokens=gen_tokens)
    ctoks = usage2.get("completion_tokens", 0)

    if ctoks > 1:
        decode_time = max(elapsed2 - ttft_s, 0.01)
        tps = (ctoks - 1) / decode_time
    else:
        tps = 0

    return {"ptoks": ptoks, "ctoks": ctoks, "ttft_ms": ttft_s * 1000,
            "tps": tps, "total_s": elapsed2}


def run_quality_prompts(port, model):
    """Generate 10 prompts, return list of outputs."""
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
    outputs = []
    for prompt in prompts:
        content, _, _ = chat(port, model,
            [{"role": "user", "content": prompt}], max_tokens=64)
        outputs.append(content)
    return prompts, outputs


def bench_config(model_path, label, extra_args=None):
    """Run full benchmark suite for one configuration."""
    print(f"\n{'='*60}")
    print(f"CONFIG: {label}")
    if extra_args:
        print(f"  Args: {' '.join(extra_args)}")
    print(f"{'='*60}")

    proc = start_server(model_path, PORT, extra_args)
    model_id = get_model_id(PORT)
    print(f"  Server ready: {model_id}")

    results = {"label": label, "args": extra_args or []}

    # Warmup
    chat(PORT, model_id, [{"role": "user", "content": "hi"}], max_tokens=2)

    # Prefill + Decode at various context sizes
    context_sizes = [100, 1000, 4000]
    print(f"\n  {'Context':>8} | {'TTFT ms':>10} | {'Decode t/s':>12} | {'Gen toks':>10}")
    print(f"  {'-'*50}")

    sweep = []
    for ctx in context_sizes:
        prompt = WORD * max(1, ctx // 10)
        try:
            r = measure_decode(PORT, model_id, prompt, gen_tokens=64)
            print(f"  {r['ptoks']:>8} | {r['ttft_ms']:>7.0f} ms | {r['tps']:>9.1f} t/s | {r['ctoks']:>10}")
            sweep.append(r)
        except Exception as e:
            print(f"  {ctx:>8} | {'FAILED':>40} | {str(e)[:30]}")
            sweep.append({"ptoks": ctx, "tps": 0, "error": str(e)[:50]})

    results["sweep"] = sweep

    # Quality: 10 prompts
    print(f"\n  Generating quality prompts...")
    prompts, outputs = run_quality_prompts(PORT, model_id)
    results["outputs"] = outputs
    print(f"  Got {len(outputs)} outputs, avg {sum(len(o) for o in outputs)//len(outputs)} chars")

    stop_server(proc)
    print(f"  Server stopped.")
    time.sleep(3)

    return results


def jaccard(a, b):
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    return len(wa & wb) / len(wa | wb) if (wa or wb) else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    args = parser.parse_args()

    print("=" * 60)
    print("TURBOQUANT MULTI-CONFIG BENCHMARK")
    print(f"Model: {args.model_path}")
    print("=" * 60)

    # Define configurations
    configs = [
        ("baseline (no TQ)", None),
        ("TQ default (bits=3, norm ON)", [
            "--kv-cache", "turboquant", "--kv-bits", "3"]),
        ("TQ no-norm-correction", [
            "--kv-cache", "turboquant", "--kv-bits", "3",
            "--kv-no-norm-correction"]),
        ("TQ asymmetric (K=4, V=3)", [
            "--kv-cache", "turboquant", "--kv-bits", "3",
            "--kv-key-bits", "4", "--kv-value-bits", "3"]),
        ("TQ layer-adaptive (8 dense)", [
            "--kv-cache", "turboquant", "--kv-bits", "3",
            "--kv-adaptive-dense-layers", "8"]),
    ]

    all_results = []
    for label, extra_args in configs:
        try:
            r = bench_config(args.model_path, label, extra_args)
            all_results.append(r)
        except Exception as e:
            print(f"\n  CONFIG FAILED: {e}")
            all_results.append({"label": label, "error": str(e)})

    # ---- Summary ----
    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    # Decode speed comparison
    print(f"\n  Decode speed (tok/s):")
    header = f"  {'Context':>8}"
    for r in all_results:
        name = r["label"][:18]
        header += f" | {name:>18}"
    print(header)
    print(f"  {'-'*(10 + 21 * len(all_results))}")

    if all_results and "sweep" in all_results[0]:
        for i in range(len(all_results[0].get("sweep", []))):
            row = f"  {all_results[0]['sweep'][i].get('ptoks', '?'):>8}"
            for r in all_results:
                sweep = r.get("sweep", [])
                if i < len(sweep):
                    tps = sweep[i].get("tps", 0)
                    if tps > 0:
                        row += f" | {tps:>15.1f} t/s"
                    else:
                        row += f" | {'FAIL':>18}"
                else:
                    row += f" | {'N/A':>18}"
            print(row)

    # TTFT comparison
    print(f"\n  TTFT (ms):")
    header = f"  {'Context':>8}"
    for r in all_results:
        name = r["label"][:18]
        header += f" | {name:>18}"
    print(header)
    print(f"  {'-'*(10 + 21 * len(all_results))}")

    if all_results and "sweep" in all_results[0]:
        for i in range(len(all_results[0].get("sweep", []))):
            row = f"  {all_results[0]['sweep'][i].get('ptoks', '?'):>8}"
            for r in all_results:
                sweep = r.get("sweep", [])
                if i < len(sweep):
                    ttft = sweep[i].get("ttft_ms", 0)
                    if ttft > 0:
                        row += f" | {ttft:>15.0f} ms"
                    else:
                        row += f" | {'FAIL':>18}"
                else:
                    row += f" | {'N/A':>18}"
            print(row)

    # Quality comparison (all configs vs baseline)
    baseline_outputs = all_results[0].get("outputs", []) if all_results else []
    if baseline_outputs:
        print(f"\n  Quality (Jaccard vs baseline):")
        for r in all_results[1:]:
            outputs = r.get("outputs", [])
            if outputs and len(outputs) == len(baseline_outputs):
                scores = [jaccard(a, b) for a, b in zip(baseline_outputs, outputs)]
                avg = sum(scores) / len(scores)
                print(f"    {r['label']:<35} avg={avg:.3f}  min={min(scores):.3f}")
            else:
                print(f"    {r['label']:<35} N/A")

    print()


if __name__ == "__main__":
    main()
