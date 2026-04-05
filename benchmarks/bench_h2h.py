#!/usr/bin/env python3
"""Head-to-head benchmark: Higgs vs oMLX on Apple Silicon.

Compares TTFT, decode tok/s, and multi-turn behavior on the same models
using the same OpenAI-compatible streaming API.

Usage:
    python3 bench_h2h.py                    # all models
    python3 bench_h2h.py --models 35B       # just 35B
    python3 bench_h2h.py --skip-multiturn   # single-turn only
    python3 bench_h2h.py --turns 10         # more turns
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HIGGS_BIN = "./target/release/higgs"
OMLX_CLI = "/Applications/oMLX.app/Contents/MacOS/omlx-cli"
LMS = os.path.expanduser("~/.cache/lm-studio/models")

HIGGS_PORT = 8899
OMLX_PORT = 8000  # avoid 8080 in case oMLX GUI is running

MAX_TOKENS = 100
COOLDOWN = 5  # seconds between server swaps
WARMUP_TOKENS = 10

MODELS = {
    "35B": {
        "path": f"{LMS}/NexVeridian/Qwen3.5-35B-A3B-3bit",
        "label": "Qwen3.5-35B-A3B-3bit (MoE)",
        "higgs_name": "Qwen3.5-35B-A3B-3bit",
    },
    "27B": {
        "path": f"{LMS}/mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit",
        "label": "Qwen3.5-27B-4bit (Dense)",
        "higgs_name": "Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit",
    },
    "DSV2": {
        "path": f"{LMS}/mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx",
        "label": "DeepSeek-V2-Lite-4bit (MoE)",
        "higgs_name": "DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx",
    },
}

PROMPTS = {
    "short": "What is 2+2? Answer in one word.",
    "medium": (
        "Write a detailed technical explanation of how transformer architectures work, "
        "covering attention mechanisms, positional encoding, layer normalization, "
        "feed-forward networks, and the differences between encoder and decoder architectures. "
        "Include discussion of multi-head attention, scaled dot-product attention, and how "
        "these components work together."
    ),
    "long": (
        "Write an extremely comprehensive and detailed technical guide covering the following "
        "topics in depth:\n\n"
        "1. COMPILER DESIGN: Explain lexical analysis, parsing (LL, LR, LALR), abstract syntax "
        "trees, semantic analysis, intermediate representations (SSA form, three-address code), "
        "optimization passes (constant folding, dead code elimination, loop unrolling, register "
        "allocation via graph coloring), and code generation for modern CPU architectures.\n\n"
        "2. OPERATING SYSTEMS: Cover process scheduling algorithms (CFS, MLFQ, lottery scheduling), "
        "virtual memory management (page tables, TLB, huge pages, NUMA), file systems (ext4, btrfs, "
        "ZFS internals), I/O scheduling, interrupt handling, system calls.\n\n"
        "3. DISTRIBUTED SYSTEMS: Explain consensus protocols (Paxos, Raft, PBFT), distributed hash "
        "tables, vector clocks, CRDTs, the CAP theorem, leader election algorithms, distributed "
        "transactions (2PC, 3PC, saga pattern).\n\n"
        "4. CRYPTOGRAPHY: Cover symmetric encryption (AES internals, modes of operation), asymmetric "
        "encryption (RSA, elliptic curves, key exchange), hash functions (SHA-256 internals), "
        "digital signatures, zero-knowledge proofs.\n\n"
        "5. DATABASE INTERNALS: Explain B-tree and LSM-tree storage engines, write-ahead logging, "
        "MVCC, query optimization, join algorithms, buffer pool management.\n\n"
        "Be thorough and technical throughout."
    ),
}

SYSTEM_PROMPT = (
    "You are a highly skilled software architect with deep expertise in distributed systems, "
    "database design, and cloud-native applications. You provide thorough, well-reasoned "
    "technical advice with step-by-step reasoning and concrete examples."
)

TURN_QUESTIONS = [
    "Explain the CAP theorem and its practical implications for system design.",
    "How would you design a rate limiter for a distributed API gateway?",
    "Compare event sourcing with traditional CRUD. When would you pick each?",
    "What are the key differences between Raft and Paxos consensus protocols?",
    "Design a notification system that handles 1M users with real-time delivery.",
    "How does MVCC work in PostgreSQL? Walk me through a concurrent update scenario.",
    "What strategies would you use to migrate a monolith to microservices safely?",
]

# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------

server_proc = None


def kill_by_port(port):
    """Kill all processes listening on the given port."""
    try:
        out = subprocess.check_output(
            ["lsof", "-ti", f":{port}"], text=True
        ).strip()
        if out:
            pids = out.splitlines()
            for pid in pids:
                try:
                    os.kill(int(pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError, ValueError):
                    pass
    except subprocess.CalledProcessError:
        pass  # no process on port


def kill_server():
    global server_proc
    if server_proc:
        try:
            os.killpg(os.getpgid(server_proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(server_proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        server_proc = None
    # Kill any stray servers on our ports (catches leaked processes)
    for port in (HIGGS_PORT, OMLX_PORT):
        kill_by_port(port)
    time.sleep(COOLDOWN)


def wait_for_server(port, api_key=None, timeout=180):
    """Wait until /v1/models responds. Returns first model name."""
    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = urllib.request.Request(url)
            if api_key:
                req.add_header("Authorization", f"Bearer {api_key}")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                models = data.get("data", [])
                if models:
                    return models[0]["id"]
        except (urllib.error.URLError, ConnectionRefusedError, OSError, KeyError, IndexError):
            pass
        time.sleep(1)
    return None


def start_higgs(model_path, expected_name=None):
    """Start Higgs server, return model name or None."""
    global server_proc
    # Kill anything still on the port before starting
    kill_by_port(HIGGS_PORT)
    time.sleep(1)
    env = {**os.environ, "HIGGS_ENABLE_THINKING": "0"}
    server_proc = subprocess.Popen(
        [HIGGS_BIN, "serve", "--model", model_path, "--port", str(HIGGS_PORT)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
        preexec_fn=os.setsid,
    )
    model_name = wait_for_server(HIGGS_PORT)
    if model_name and expected_name and expected_name not in model_name:
        log(f"  WARNING: Expected model '{expected_name}' but server reports '{model_name}'")
        kill_server()
        return None
    return model_name


def start_omlx(model_parent_dir, expected_name=None):
    """Start oMLX CLI server, return model name or None.

    oMLX discovers models one level deep from --model-dir, so pass the
    immediate parent of the model directory (e.g. NexVeridian/ or mlx-community/).
    """
    global server_proc
    # Kill anything still on the port before starting
    kill_by_port(OMLX_PORT)
    time.sleep(1)
    server_proc = subprocess.Popen(
        [
            OMLX_CLI, "serve",
            "--model-dir", model_parent_dir,
            "--port", str(OMLX_PORT),
            "--no-cache",
            "--max-num-seqs", "1",
            "--log-level", "warning",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )
    model_name = wait_for_server(OMLX_PORT, api_key="omlx")
    if model_name and expected_name and expected_name not in model_name:
        log(f"  WARNING: Expected model '{expected_name}' but oMLX reports '{model_name}'")
        kill_server()
        return None
    return model_name


def get_rss_mb():
    """Get RSS of server process tree in MB (includes child processes)."""
    if not server_proc:
        return 0
    try:
        # Get RSS of entire process group (parent + children)
        pgid = os.getpgid(server_proc.pid)
        out = subprocess.check_output(
            ["ps", "-o", "rss=", "-g", str(pgid)], text=True
        ).strip()
        total_kb = sum(int(line.strip()) for line in out.splitlines() if line.strip())
        return total_kb / 1024  # KB -> MB
    except (subprocess.CalledProcessError, ValueError, ProcessLookupError):
        return 0


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


def stream_chat(port, messages, max_tokens=MAX_TOKENS, model_name="test",
                api_key=None, temperature=0.0):
    """Send streaming chat completion, measure TTFT + decode."""
    payload = json.dumps({
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }).encode()

    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")

    t0 = time.perf_counter()
    first_token_time = None
    tokens = []
    prompt_tokens = 0
    completion_tokens = 0

    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            buf = b""
            while True:
                chunk = resp.read(1)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        obj = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    choices = obj.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content and first_token_time is None:
                            first_token_time = time.perf_counter()
                        if content:
                            tokens.append(content)
                    usage = obj.get("usage")
                    if usage:
                        prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                        completion_tokens = usage.get("completion_tokens", completion_tokens)
    except Exception as e:
        return {"error": str(e)}

    end = time.perf_counter()
    if first_token_time is None:
        first_token_time = end

    ttft_s = first_token_time - t0
    decode_s = end - first_token_time
    n_completion = completion_tokens or len(tokens)
    decode_tps = n_completion / decode_s if decode_s > 0.001 else 0

    # Estimate prompt tokens if server didn't report (oMLX SSE doesn't)
    if not prompt_tokens:
        prompt_chars = sum(len(m.get("content", "")) for m in messages)
        prompt_tokens = max(1, int(prompt_chars / 3.5))  # rough estimate

    return {
        "ttft_ms": ttft_s * 1000,
        "decode_tps": decode_tps,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": n_completion,
        "total_ms": (end - t0) * 1000,
        "output": "".join(tokens),
        "rss_mb": get_rss_mb(),
    }


def fmt_result(r):
    if "error" in r:
        return f"ERROR: {r['error']}"
    return (
        f"TTFT={r['ttft_ms']:>7.0f}ms  "
        f"decode={r['decode_tps']:>5.1f}tok/s  "
        f"prompt={r['prompt_tokens']:>4d}tok  "
        f"gen={r['completion_tokens']:>3d}tok  "
        f"total={r['total_ms']/1000:>5.1f}s  "
        f"RSS={r['rss_mb']:>5.0f}MB"
    )


# ---------------------------------------------------------------------------
# Test phases
# ---------------------------------------------------------------------------


def run_single_turn(port, model_name, api_key=None):
    """Run short/medium/long single-turn prompts."""
    results = {}
    for label, prompt in PROMPTS.items():
        messages = [{"role": "user", "content": prompt}]
        r = stream_chat(port, messages, model_name=model_name, api_key=api_key)
        results[label] = r
        log(f"    {label:8s}: {fmt_result(r)}")
    return results


def run_multi_turn(port, model_name, num_turns, api_key=None):
    """Run multi-turn conversation, measuring TTFT degradation."""
    results = []
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for i in range(min(num_turns, len(TURN_QUESTIONS))):
        messages.append({"role": "user", "content": TURN_QUESTIONS[i]})
        r = stream_chat(port, messages, max_tokens=80, model_name=model_name,
                        api_key=api_key)
        results.append(r)

        if "error" in r:
            log(f"    turn {i+1}: {fmt_result(r)}")
            break

        # Add assistant response to conversation history
        messages.append({"role": "assistant", "content": r["output"]})

        ctx_tokens = r["prompt_tokens"]
        log(f"    turn {i+1}: {fmt_result(r)}  ctx~{ctx_tokens}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

output_lines = []


def log(msg):
    print(msg, flush=True)
    output_lines.append(msg)


def run_for_backend(backend, model_key, model_info, num_turns, skip_multiturn):
    """Run all benchmarks for one backend + model combo."""
    model_path = model_info["path"]

    log(f"\n  --- {backend} ---")

    expected = model_info.get("higgs_name", os.path.basename(model_path))

    if backend == "higgs":
        port = HIGGS_PORT
        api_key = None
        log(f"  Starting Higgs on :{port} ...")
        model_name = start_higgs(model_path, expected_name=expected)
    else:
        port = OMLX_PORT
        api_key = "omlx"
        log(f"  Starting oMLX on :{port} (--no-cache) ...")
        model_name = start_omlx(os.path.dirname(model_path), expected_name=expected)

    if not model_name:
        log("  FAILED to start server")
        kill_server()
        return None

    log(f"  Server ready: model={model_name}  RSS={get_rss_mb():.0f}MB")

    # Use the right model name for requests
    if backend == "omlx":
        # oMLX discovers models by directory basename
        model_name = os.path.basename(model_path)

    # Warmup
    log("  Warmup...")
    warmup = stream_chat(
        port,
        [{"role": "user", "content": "Say hi."}],
        max_tokens=WARMUP_TOKENS,
        model_name=model_name,
        api_key=api_key,
    )
    if "error" in warmup:
        log(f"  Warmup failed: {warmup['error']}")
        kill_server()
        return None
    log(f"  Warmup done. RSS={get_rss_mb():.0f}MB")

    all_results = {"backend": backend, "model": model_key}

    # Phase 1: Single-turn
    log(f"\n  [Single-turn]")
    all_results["single_turn"] = run_single_turn(port, model_name, api_key)

    # Phase 2: Multi-turn
    if not skip_multiturn:
        log(f"\n  [Multi-turn, {num_turns} turns]")
        all_results["multi_turn"] = run_multi_turn(
            port, model_name, num_turns, api_key
        )

    kill_server()
    return all_results


def print_comparison(higgs_results, omlx_results, model_label):
    """Print side-by-side comparison table."""
    log(f"\n{'='*80}")
    log(f"COMPARISON: {model_label}")
    log(f"{'='*80}")

    # Single-turn
    log(f"\n  {'':12s} {'TTFT (ms)':>20s}   {'Decode (tok/s)':>20s}")
    log(f"  {'Prompt':12s} {'Higgs':>9s} {'oMLX':>9s}   {'Higgs':>9s} {'oMLX':>9s}  {'TTFT':>7s} {'Decode':>7s}")
    log(f"  {'-'*74}")

    h_st = higgs_results.get("single_turn", {})
    o_st = omlx_results.get("single_turn", {})

    for label in ("short", "medium", "long"):
        h = h_st.get(label, {})
        o = o_st.get(label, {})

        if "error" in h or "error" in o:
            log(f"  {label:12s}  (error in one or both backends)")
            continue

        h_ttft = h.get("ttft_ms", 0)
        o_ttft = o.get("ttft_ms", 0)
        h_dec = h.get("decode_tps", 0)
        o_dec = o.get("decode_tps", 0)

        ttft_ratio = f"{o_ttft/h_ttft:.2f}x" if h_ttft > 0 else "—"
        dec_ratio = f"{h_dec/o_dec:.2f}x" if o_dec > 0 else "—"

        log(
            f"  {label:12s} {h_ttft:>8.0f}  {o_ttft:>8.0f}   "
            f"{h_dec:>8.1f}  {o_dec:>8.1f}  "
            f"{ttft_ratio:>7s} {dec_ratio:>7s}"
        )

    # Multi-turn
    h_mt = higgs_results.get("multi_turn", [])
    o_mt = omlx_results.get("multi_turn", [])

    if h_mt and o_mt:
        log(f"\n  Multi-turn TTFT progression:")
        log(f"  {'Turn':>5s}  {'Higgs TTFT':>11s}  {'oMLX TTFT':>11s}  {'Higgs dec':>10s}  {'oMLX dec':>10s}")
        log(f"  {'-'*55}")

        for i in range(min(len(h_mt), len(o_mt))):
            h = h_mt[i]
            o = o_mt[i]
            if "error" in h or "error" in o:
                log(f"  {i+1:>5d}  (error)")
                break
            log(
                f"  {i+1:>5d}  {h['ttft_ms']:>9.0f}ms  {o['ttft_ms']:>9.0f}ms  "
                f"{h['decode_tps']:>8.1f}/s  {o['decode_tps']:>8.1f}/s"
            )


def main():
    parser = argparse.ArgumentParser(description="Higgs vs oMLX head-to-head")
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                        help="Model keys to test (35B, 27B, DSV2)")
    parser.add_argument("--turns", type=int, default=5,
                        help="Number of multi-turn conversation turns")
    parser.add_argument("--skip-multiturn", action="store_true",
                        help="Skip multi-turn tests")
    parser.add_argument("--higgs-only", action="store_true")
    parser.add_argument("--omlx-only", action="store_true")
    args = parser.parse_args()

    # Validate
    if not os.path.isfile(HIGGS_BIN) and not args.omlx_only:
        log(f"Higgs binary not found: {HIGGS_BIN}")
        sys.exit(1)
    if not os.path.isfile(OMLX_CLI) and not args.higgs_only:
        log(f"oMLX CLI not found: {OMLX_CLI}")
        sys.exit(1)

    selected = {k: v for k, v in MODELS.items() if k in args.models}
    available = {k: v for k, v in selected.items() if os.path.isdir(v["path"])}

    if not available:
        log("No models found on disk!")
        log(f"Checked: {[v['path'] for v in selected.values()]}")
        sys.exit(1)

    log("=" * 80)
    log(f"HEAD-TO-HEAD: Higgs vs oMLX — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Max tokens: {MAX_TOKENS}  Turns: {args.turns}  Cooldown: {COOLDOWN}s")
    log(f"Models: {', '.join(available.keys())}")
    log(f"oMLX: --no-cache (paged SSD cache disabled for fair comparison)")
    log("=" * 80)

    all_comparisons = []

    for model_key, model_info in available.items():
        log(f"\n{'#'*80}")
        log(f"# MODEL: {model_info['label']}")
        log(f"# Path:  {model_info['path']}")
        log(f"{'#'*80}")

        higgs_r = None
        omlx_r = None

        # Run Higgs first (it's our project, gives oMLX more cooldown)
        if not args.omlx_only:
            try:
                higgs_r = run_for_backend(
                    "higgs", model_key, model_info, args.turns, args.skip_multiturn
                )
            except Exception as e:
                log(f"  Higgs error: {e}")
                kill_server()

        # Run oMLX
        if not args.higgs_only:
            try:
                omlx_r = run_for_backend(
                    "omlx", model_key, model_info, args.turns, args.skip_multiturn
                )
            except Exception as e:
                log(f"  oMLX error: {e}")
                kill_server()

        if higgs_r and omlx_r:
            all_comparisons.append((model_key, model_info["label"], higgs_r, omlx_r))
            print_comparison(higgs_r, omlx_r, model_info["label"])

    # Final summary
    if all_comparisons:
        log(f"\n{'='*80}")
        log("FINAL SUMMARY")
        log(f"{'='*80}")
        log(f"\n  {'Model':<40s} {'TTFT Higgs':>10s} {'TTFT oMLX':>10s} {'Dec Higgs':>10s} {'Dec oMLX':>10s}")
        log(f"  {'-'*84}")
        for model_key, label, h, o in all_comparisons:
            h_med = h.get("single_turn", {}).get("medium", {})
            o_med = o.get("single_turn", {}).get("medium", {})
            if "error" not in h_med and "error" not in o_med:
                log(
                    f"  {label:<40s} "
                    f"{h_med.get('ttft_ms',0):>8.0f}ms "
                    f"{o_med.get('ttft_ms',0):>8.0f}ms "
                    f"{h_med.get('decode_tps',0):>8.1f}/s "
                    f"{o_med.get('decode_tps',0):>8.1f}/s"
                )

    kill_server()

    # Save results
    ts = time.strftime("%Y%m%d_%H%M%S")
    outfile = f"bench_h2h_{ts}.txt"
    with open(outfile, "w") as f:
        f.write("\n".join(output_lines) + "\n")
    log(f"\nResults saved to {outfile}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("\nInterrupted")
        kill_server()
        sys.exit(1)
