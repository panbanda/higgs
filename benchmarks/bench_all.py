#!/usr/bin/env python3
"""Benchmark all local MLX models with higgs. Measures TTFT, prefill, decode tok/s."""

import json, os, signal, subprocess, sys, time, urllib.request

HIGGS = "./target/release/higgs"
PORT = 8899
BASE = f"http://localhost:{PORT}"
MAX_TOKENS = 200

LMS = os.path.expanduser("~/.cache/lm-studio/models")
MODELS = [
    (f"{LMS}/mlx-community/Qwen3-0.6B-4bit",),
    (f"{LMS}/mlx-community/Qwen3-4B-Instruct-2507-4bit",),
    (f"{LMS}/mlx-community/Qwen3-8B-4bit",),
    (f"{LMS}/mlx-community/Qwen3-14B-4bit",),
    (f"{LMS}/mlx-community/Qwen3.5-0.8B-8bit",),
    (f"{LMS}/mlx-community/Qwen3.5-2B-MLX-8bit",),
    (f"{LMS}/mlx-community/Qwen3.5-4B-MLX-4bit",),
    (f"{LMS}/mlx-community/Qwen3.5-9B-MLX-4bit",),
    (f"{LMS}/NexVeridian/Qwen3.5-35B-A3B-3bit",),
    (f"{LMS}/mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx",),
]

PROMPTS = {
    "short": "Explain what a neural network is in one paragraph.",
    "medium": (
        "Write a detailed technical explanation of how transformer architectures work, "
        "covering attention mechanisms, positional encoding, layer normalization, "
        "feed-forward networks, and the differences between encoder and decoder architectures. "
        "Include discussion of multi-head attention, scaled dot-product attention, and how "
        "these components work together. Also explain the training process including "
        "backpropagation through the attention mechanism. Discuss the key innovations that "
        "made transformers superior to RNNs and LSTMs for sequence modeling tasks. Cover the "
        "evolution from the original Attention Is All You Need paper through modern variants "
        "like GPT, BERT, and their derivatives. Explain how context windows work and the "
        "computational complexity of self-attention."
    ),
    "long": (
        "Write an extremely comprehensive and detailed technical guide covering the following "
        "topics in depth. For each topic, provide multiple paragraphs with specific technical "
        "details, examples, and explanations:\n\n"
        "1. COMPILER DESIGN: Explain lexical analysis, parsing (LL, LR, LALR), abstract syntax "
        "trees, semantic analysis, intermediate representations (SSA form, three-address code), "
        "optimization passes (constant folding, dead code elimination, loop unrolling, register "
        "allocation via graph coloring), and code generation for modern CPU architectures.\n\n"
        "2. OPERATING SYSTEMS: Cover process scheduling algorithms (CFS, MLFQ, lottery scheduling), "
        "virtual memory management (page tables, TLB, huge pages, NUMA), file systems (ext4, btrfs, "
        "ZFS internals), I/O scheduling, interrupt handling, system calls, and the differences "
        "between monolithic and microkernel designs.\n\n"
        "3. DISTRIBUTED SYSTEMS: Explain consensus protocols (Paxos, Raft, PBFT), distributed hash "
        "tables, vector clocks, CRDTs, the CAP theorem and its practical implications, leader "
        "election algorithms, distributed transactions (2PC, 3PC, saga pattern), and how systems "
        "like Spanner, CockroachDB, and TiKV achieve global consistency.\n\n"
        "4. CRYPTOGRAPHY: Cover symmetric encryption (AES internals, modes of operation), asymmetric "
        "encryption (RSA, elliptic curves, key exchange), hash functions (SHA-256 internals), "
        "digital signatures, zero-knowledge proofs, homomorphic encryption, and post-quantum "
        "cryptography approaches.\n\n"
        "5. DATABASE INTERNALS: Explain B-tree and LSM-tree storage engines, write-ahead logging, "
        "MVCC, query optimization (cost-based vs rule-based), join algorithms (nested loop, hash, "
        "sort-merge), buffer pool management, and how modern databases handle concurrent "
        "transactions with different isolation levels.\n\n"
        "6. NETWORKING: Cover TCP congestion control (Reno, CUBIC, BBR), QUIC protocol design, "
        "BGP routing, DNS resolution chain, HTTP/3 internals, TLS certificate chains, NAT "
        "traversal techniques, and software-defined networking concepts.\n\n"
        "7. MACHINE LEARNING SYSTEMS: Explain backpropagation mathematics, gradient descent "
        "variants (SGD, Adam, LAMB), mixed-precision training, data parallelism vs model "
        "parallelism vs pipeline parallelism, attention mechanism computation, KV-cache "
        "optimization, quantization methods (GPTQ, AWQ, GGML), and inference optimization.\n\n"
        "Be thorough and technical throughout."
    ),
}

server_proc = None

def log(msg):
    print(msg, flush=True)

def start_server(model_path):
    global server_proc
    env = os.environ.copy()
    env["HIGGS_ENABLE_THINKING"] = "0"
    server_proc = subprocess.Popen(
        [HIGGS, "serve", "--model", model_path, "--port", str(PORT)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env=env, preexec_fn=os.setsid,
    )
    # Wait for server
    for i in range(120):
        try:
            urllib.request.urlopen(f"{BASE}/v1/models", timeout=2)
            return True
        except:
            time.sleep(1)
            if server_proc.poll() is not None:
                return False
    return False

def kill_server():
    global server_proc
    if server_proc:
        try:
            os.killpg(os.getpgid(server_proc.pid), signal.SIGTERM)
        except:
            pass
        try:
            server_proc.wait(timeout=5)
        except:
            try:
                os.killpg(os.getpgid(server_proc.pid), signal.SIGKILL)
            except:
                pass
        server_proc = None
        time.sleep(2)

def bench(model_name, prompt, label):
    body = json.dumps({
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "stream": True,
    }).encode()

    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )

    start = time.perf_counter()
    first_token_time = None
    prompt_tokens = 0
    completion_tokens = 0

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
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
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except:
                        continue
                    choices = obj.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content and first_token_time is None:
                            first_token_time = time.perf_counter()
                    usage = obj.get("usage")
                    if usage:
                        prompt_tokens = usage.get("prompt_tokens", 0)
                        completion_tokens = usage.get("completion_tokens", 0)
    except Exception as e:
        log(f"  {label:8s}: ERROR - {e}")
        return

    end = time.perf_counter()
    ttft_ms = (first_token_time - start) * 1000 if first_token_time else -1
    decode_s = end - first_token_time if first_token_time else 0.001
    decode_tps = completion_tokens / decode_s if decode_s > 0 else 0
    prefill_tps = prompt_tokens / (ttft_ms / 1000) if ttft_ms > 0 and prompt_tokens > 0 else 0

    log(
        f"  {label:8s}: prompt={prompt_tokens:>5d}tok  "
        f"TTFT={ttft_ms:>7.0f}ms  prefill={prefill_tps:>8.1f}tok/s  "
        f"decode={decode_tps:>6.1f}tok/s  gen={completion_tokens:>3d}tok  "
        f"total={end - start:>5.1f}s"
    )

if __name__ == "__main__":
    results_file = f"bench_results_{time.strftime('%Y%m%d_%H%M%S')}.txt"

    # Tee output to file
    class Tee:
        def __init__(self, f): self.file = f; self.stdout = sys.stdout
        def write(self, s): self.stdout.write(s); self.file.write(s)
        def flush(self): self.stdout.flush(); self.file.flush()

    rf = open(results_file, "w")
    sys.stdout = Tee(rf)

    log("=" * 80)
    log(f"HIGGS BENCHMARK - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Binary: {HIGGS}   Max tokens: {MAX_TOKENS}   Thinking: disabled")
    log("=" * 80)
    log("")

    try:
        for (model_path,) in MODELS:
            model_name = os.path.basename(model_path)
            if not os.path.isdir(model_path):
                log(f"SKIP {model_name} (not found)")
                log("")
                continue

            kill_server()
            log(f"--- {model_name} ---")

            if not start_server(model_path):
                log(f"  FAILED to start server")
                log("")
                kill_server()
                continue

            log(f"  Server ready (PID {server_proc.pid})")

            for label, prompt in PROMPTS.items():
                bench(model_name, prompt, label)

            kill_server()
            log("")
    except KeyboardInterrupt:
        log("\nInterrupted by user")
    finally:
        kill_server()
        log("=" * 80)
        log(f"DONE - results in {results_file}")
        log("=" * 80)
        rf.close()
