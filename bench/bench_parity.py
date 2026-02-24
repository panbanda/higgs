#!/usr/bin/env python3
"""
Parity benchmark: higgs (Rust) vs mlx_lm (Python).

For each supported architecture, generates 200 tokens at temperature=0 with
both engines and compares: tok/s, peak RSS, and output text agreement.

Usage:
    mise exec python@3.12 -- python3 bench/bench_parity.py
"""

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
HIGGS_BIN = str(Path(__file__).parent.parent / "target" / "release" / "higgs")
HIGGS_PORT = 9001
MAX_TOKENS = 200
PROMPT = "Explain the concept of gravitational waves in simple terms."
CODE_PROMPT = "def merge_sort(arr):\n    "

MODELS = [
    ("llama",      "mlx-community/Llama-3.2-1B-Instruct-4bit",         "chat"),
    ("mistral",    "mlx-community/Mistral-7B-Instruct-v0.3-4bit",      "chat"),
    ("qwen2",      "mlx-community/Qwen3-1.7B-4bit",                    "chat"),
    ("qwen3_moe",  "mlx-community/Qwen3-30B-A3B-8bit",                 "chat"),
    ("gemma2",     "mlx-community/gemma-2-2b-it-4bit",                  "chat"),
    ("phi3",       "mlx-community/Phi-3-mini-4k-instruct-4bit",         "chat"),
    ("starcoder2", "mlx-community/starcoder2-3b-4bit",                  "completion"),
    ("deepseek_v2","mlx-community/DeepSeek-V2-Lite-Chat-4bit-mlx",      "chat"),
]


def model_path(repo: str) -> str:
    name = "models--" + repo.replace("/", "--")
    snaps = HF_CACHE / name / "snapshots"
    if not snaps.exists():
        raise FileNotFoundError(f"Model not found in cache: {repo}")
    return str(sorted(snaps.iterdir())[-1])


def rss_mb(pid: int) -> int:
    try:
        out = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(pid)], stderr=subprocess.DEVNULL
        )
        return round(int(out.strip()) / 1024)
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Python mlx_lm measurement
# ---------------------------------------------------------------------------
def measure_python(arch: str, repo: str, mode: str) -> dict:
    """Run mlx_lm.generate in a subprocess, parse its output."""
    path = model_path(repo)

    if mode == "chat":
        prompt_expr = f'tokenizer.apply_chat_template([{{"role": "user", "content": {json.dumps(PROMPT)}}}], tokenize=False, add_generation_prompt=True)'
    else:
        prompt_expr = json.dumps(CODE_PROMPT)

    script = f"""
import time, os, mlx_lm
from mlx_lm.sample_utils import make_sampler
model, tokenizer = mlx_lm.load("{path}")
prompt = {prompt_expr}
sampler = make_sampler(temp=0.0)
# warmup
_ = mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=20, sampler=sampler)
# timed run using stream_generate for accurate token count
text = ""
gen_tokens = 0
t0 = time.perf_counter()
for resp in mlx_lm.stream_generate(model, tokenizer, prompt=prompt, max_tokens={MAX_TOKENS}, sampler=sampler):
    text += resp.text
    gen_tokens = resp.generation_tokens
elapsed = time.perf_counter() - t0
tok_s = round(gen_tokens / elapsed, 1) if elapsed > 0 else 0
import subprocess
rss_out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], stderr=subprocess.DEVNULL)
rss_mb = round(int(rss_out.strip()) / 1024)
import json as _json
print("__RESULT__" + _json.dumps({{"text": text, "elapsed": elapsed, "rss_mb": rss_mb, "tokens": gen_tokens, "tok_s": tok_s}}))
"""

    try:
        result = subprocess.run(
            ["mise", "exec", "python@3.12", "--", "python3", "-c", script],
            capture_output=True, text=True, timeout=300,
        )
        for line in result.stdout.splitlines():
            if line.startswith("__RESULT__"):
                data = json.loads(line[len("__RESULT__"):])
                return {
                    "tok_s": data["tok_s"],
                    "rss_mb": data["rss_mb"],
                    "text": data["text"],
                    "elapsed": round(data["elapsed"], 2),
                    "gen_tokens": data["tokens"],
                }
        # If no result line, print stderr for debugging
        return {"error": result.stderr[:500] if result.stderr else "no output"}
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Rust higgs measurement
# ---------------------------------------------------------------------------
def measure_rust(arch: str, repo: str, mode: str) -> dict:
    """Start higgs server, send request, measure."""
    path = model_path(repo)
    proc = subprocess.Popen(
        [HIGGS_BIN, "--model", path, "--port", str(HIGGS_PORT)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    try:
        # Wait for server ready
        deadline = time.time() + 180
        url = f"http://localhost:{HIGGS_PORT}/v1/models"
        ready = False
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=2) as r:
                    if r.status == 200:
                        models_data = json.loads(r.read())
                        model_id = models_data["data"][0]["id"]
                        ready = True
                        break
            except Exception:
                pass
            time.sleep(1.0)

        if not ready:
            return {"error": "server failed to start"}

        # Warmup
        _post(HIGGS_PORT, mode, model_id, "Hi", 20)

        # Actual run
        t0 = time.perf_counter()
        resp = _post(HIGGS_PORT, mode, model_id,
                     PROMPT if mode == "chat" else CODE_PROMPT, MAX_TOKENS)
        elapsed = time.perf_counter() - t0

        if mode == "chat":
            text = resp["choices"][0]["message"]["content"]
        else:
            text = resp["choices"][0]["text"]

        gen_tokens = resp.get("usage", {}).get("completion_tokens", MAX_TOKENS)
        tok_s = round(gen_tokens / elapsed, 1) if elapsed > 0 else 0
        mem = rss_mb(proc.pid)

        return {
            "tok_s": tok_s,
            "rss_mb": mem,
            "text": text,
            "elapsed": round(elapsed, 2),
            "gen_tokens": gen_tokens,
        }

    except Exception as e:
        return {"error": str(e)}
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
        time.sleep(2)


def _post(port: int, mode: str, model_id: str, prompt: str, max_tokens: int) -> dict:
    if mode == "chat":
        body = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        path = "/v1/chat/completions"
    else:
        body = {
            "model": model_id,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        path = "/v1/completions"

    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"http://localhost:{port}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


# ---------------------------------------------------------------------------
# Text comparison
# ---------------------------------------------------------------------------
def text_similarity(a: str, b: str) -> float:
    """Simple word-level Jaccard similarity."""
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa and not wb:
        return 1.0
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def first_n_words(text: str, n: int = 10) -> str:
    words = text.split()
    return " ".join(words[:n]) + ("..." if len(words) > n else "")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("PARITY BENCHMARK: higgs (Rust) vs mlx_lm (Python)")
    print(f"Max tokens: {MAX_TOKENS}, temperature: 0")
    print("=" * 80)

    results = []

    for arch, repo, mode in MODELS:
        print(f"\n--- {arch}: {repo} ({mode}) ---")

        # Check model is cached
        try:
            model_path(repo)
        except FileNotFoundError:
            print(f"  SKIP: not in HF cache")
            results.append({"arch": arch, "repo": repo, "error": "not cached"})
            continue

        # Python
        print(f"  [python] running...", end=" ", flush=True)
        py = measure_python(arch, repo, mode)
        if "error" in py:
            print(f"ERROR: {py['error']}")
        else:
            print(f"{py['tok_s']} tok/s, {py['rss_mb']} MB")

        # Rust
        print(f"  [rust]   running...", end=" ", flush=True)
        rs = measure_rust(arch, repo, mode)
        if "error" in rs:
            print(f"ERROR: {rs['error']}")
        else:
            print(f"{rs['tok_s']} tok/s, {rs['rss_mb']} MB")

        # Compare
        if "error" not in py and "error" not in rs:
            sim = text_similarity(py["text"], rs["text"])
            prefix_match = py["text"][:50] == rs["text"][:50]
            print(f"  [parity] word overlap: {sim:.1%}, first-50-char match: {prefix_match}")
            print(f"    python: {first_n_words(py['text'], 15)}")
            print(f"    rust:   {first_n_words(rs['text'], 15)}")

        results.append({
            "arch": arch,
            "repo": repo,
            "mode": mode,
            "python": py,
            "rust": rs,
        })

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    header = f"| {'Arch':<12} | {'Python tok/s':>13} | {'Rust tok/s':>11} | {'Ratio':>6} | {'Py RSS':>7} | {'Rs RSS':>7} | {'Parity':>8} |"
    sep = "|" + "-" * 14 + "|" + "-" * 15 + "|" + "-" * 13 + "|" + "-" * 8 + "|" + "-" * 9 + "|" + "-" * 9 + "|" + "-" * 10 + "|"
    print(header)
    print(sep)

    for r in results:
        if "error" in r:
            print(f"| {r['arch']:<12} | {'SKIP':>13} | {'SKIP':>11} | {'':>6} | {'':>7} | {'':>7} | {'':>8} |")
            continue

        py = r.get("python", {})
        rs = r.get("rust", {})

        py_tok = py.get("tok_s", "ERR")
        rs_tok = rs.get("tok_s", "ERR")
        py_rss = py.get("rss_mb", "ERR")
        rs_rss = rs.get("rss_mb", "ERR")

        if isinstance(py_tok, (int, float)) and isinstance(rs_tok, (int, float)) and py_tok > 0:
            ratio = f"{rs_tok / py_tok:.2f}x"
        else:
            ratio = "n/a"

        if "error" not in py and "error" not in rs:
            sim = text_similarity(py["text"], rs["text"])
            parity = f"{sim:.0%}"
        elif "error" in py:
            parity = f"py err"
        elif "error" in rs:
            parity = f"rs err"
        else:
            parity = "n/a"

        print(f"| {r['arch']:<12} | {str(py_tok):>13} | {str(rs_tok):>11} | {ratio:>6} | {str(py_rss):>7} | {str(rs_rss):>7} | {parity:>8} |")

    # Save results
    out = Path(__file__).parent / "parity_results.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nRaw results saved to {out}")

    # Markdown report
    md_lines = [
        "## Parity Benchmark: higgs (Rust) vs mlx_lm (Python)\n",
        f"Max tokens: {MAX_TOKENS}, temperature: 0, single request.\n",
        "| Architecture | Model | Python tok/s | Rust tok/s | Ratio | Python RSS (MB) | Rust RSS (MB) | Word Overlap |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        if "error" in r:
            continue
        py = r.get("python", {})
        rs = r.get("rust", {})
        py_tok = py.get("tok_s", "ERR")
        rs_tok = rs.get("tok_s", "ERR")
        py_rss = py.get("rss_mb", "ERR")
        rs_rss = rs.get("rss_mb", "ERR")
        if isinstance(py_tok, (int, float)) and isinstance(rs_tok, (int, float)) and py_tok > 0:
            ratio = f"{rs_tok / py_tok:.2f}x"
        else:
            ratio = "n/a"
        if "error" not in py and "error" not in rs:
            sim = f"{text_similarity(py['text'], rs['text']):.0%}"
        else:
            sim = "n/a"
        md_lines.append(
            f"| {r['arch']} | {r['repo'].split('/')[-1]} | {py_tok} | {rs_tok} | {ratio} | {py_rss} | {rs_rss} | {sim} |"
        )

    md_out = Path(__file__).parent / "parity_results.md"
    md_out.write_text("\n".join(md_lines) + "\n")
    print(f"Markdown report saved to {md_out}")


if __name__ == "__main__":
    main()
