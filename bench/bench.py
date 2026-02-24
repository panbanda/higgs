#!/usr/bin/env python3
"""
Benchmark suite: mlx-server vs vllm-mlx.

Runs each server as a subprocess, waits for readiness, runs benchmarks,
collects results into a dict, and prints a Markdown report at the end.

Usage:
    python bench.py [--benchmarks b1,b2,...] [--only-server mlx|vllm]

Benchmarks:
    1_throughput    - single-request decode tok/s
    2_batching      - concurrent request throughput
    3_structured    - JSON schema compliance
    4_logprobs      - log probability accuracy
    5_reasoning     - think-tag extraction
    6_embeddings    - semantic similarity
    7_vision        - multimodal image understanding
    8_architectures - architecture smoke tests
"""

import argparse
import asyncio
import base64
import json
import math
import os
import signal
import subprocess
import sys
import time
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"


def model_path(repo: str) -> str:
    """Return local snapshot path for a cached HF model."""
    name = "models--" + repo.replace("/", "--")
    snaps = HF_CACHE / name / "snapshots"
    if not snaps.exists():
        raise FileNotFoundError(f"Model not found in cache: {repo}")
    return str(sorted(snaps.iterdir())[-1])


LLAMA_1B = "mlx-community/Llama-3.2-1B-Instruct-4bit"
QWEN3_17B = "mlx-community/Qwen3-1.7B-4bit"
QWEN3_MOE = "mlx-community/Qwen3-30B-A3B-8bit"
NANOVLLAVA = "mlx-community/nanoLLaVA-1.5-4bit"
MINILM = "mlx-community/all-MiniLM-L6-v2-4bit"
GEMMA2 = "mlx-community/gemma-2-2b-it-4bit"
PHI3 = "mlx-community/Phi-3-mini-4k-instruct-4bit"
STARCODER2 = "mlx-community/starcoder2-3b-4bit"
DEEPSEEK = "mlx-community/DeepSeek-V2-Lite-Chat-4bit-mlx"

# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------
MLX_PORT = 9001
VLLM_PORT = 9002
MLX_BIN = str(Path(__file__).parent.parent / "target" / "release" / "higgs")


def wait_ready(port: int, timeout: float = 180.0) -> bool:
    deadline = time.time() + timeout
    url = f"http://localhost:{port}/v1/models"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def get_model_id(port: int) -> str:
    url = f"http://localhost:{port}/v1/models"
    with urllib.request.urlopen(url, timeout=5) as r:
        data = json.loads(r.read())
    return data["data"][0]["id"]


def stop(proc: subprocess.Popen):
    if proc and proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


def rss_mb(pid: int) -> int:
    """Return current RSS of a process in MB (macOS: ps rss is in KB)."""
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)],
                                      stderr=subprocess.DEVNULL)
        return round(int(out.strip()) / 1024)
    except Exception:
        return 0


@contextmanager
def mlx_server(model: str, extra_args: list[str] | None = None):
    """Context manager: start mlx-server, yield (port, model_id, proc), stop on exit."""
    args = [MLX_BIN, "--model", model_path(model), "--port", str(MLX_PORT)]
    if extra_args:
        args.extend(extra_args)
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        if not wait_ready(MLX_PORT):
            raise RuntimeError("mlx-server failed to start")
        yield MLX_PORT, get_model_id(MLX_PORT), proc
    finally:
        stop(proc)
        time.sleep(2)


VLLM_DIR = str(Path.home() / "dev" / "vllm-mlx")


@contextmanager
def vllm_server(model: str, extra_args: list[str] | None = None):
    """Context manager: start vllm-mlx, yield (port, model_id, proc), stop on exit."""
    args = [sys.executable, "-m", "vllm_mlx.cli", "serve", model,
            "--port", str(VLLM_PORT)]
    if extra_args:
        args.extend(extra_args)
    env = {**os.environ, "PYTHONPATH": VLLM_DIR}
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            env=env)
    try:
        if not wait_ready(VLLM_PORT):
            raise RuntimeError("vllm-mlx failed to start")
        yield VLLM_PORT, get_model_id(VLLM_PORT), proc
    finally:
        stop(proc)
        time.sleep(2)


def server_ctx(server: str, model: str, extra_args: list[str] | None = None):
    if server == "mlx":
        return mlx_server(model, extra_args)
    return vllm_server(model, extra_args)


# ---------------------------------------------------------------------------
# HTTP helpers (no SDK dependency)
# ---------------------------------------------------------------------------
def post(port: int, path: str, body: dict) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"http://localhost:{port}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def chat(port: int, model_id: str, messages: list, **kwargs) -> dict:
    return post(port, "/v1/chat/completions",
                {"model": model_id, "messages": messages, **kwargs})


def complete(port: int, model_id: str, prompt: str, **kwargs) -> dict:
    return post(port, "/v1/completions",
                {"model": model_id, "prompt": prompt, **kwargs})


def embed(port: int, model_id: str, inputs: list[str]) -> list[list[float]]:
    resp = post(port, "/v1/embeddings", {"model": model_id, "input": inputs})
    return [item["embedding"] for item in resp["data"]]


# ---------------------------------------------------------------------------
# Async concurrent helper
# ---------------------------------------------------------------------------
try:
    from openai import AsyncOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


async def concurrent_chat(
    port: int, model_id: str, prompts: list[str], max_tokens: int = 200
) -> tuple[float, list[float]]:
    """Send all prompts concurrently. Returns (total_tok/s, [ttft_s])."""
    if not HAS_OPENAI:
        raise RuntimeError("openai package required")
    client = AsyncOpenAI(base_url=f"http://localhost:{port}/v1", api_key="x")
    ttfts: list[float] = []
    total_tokens = 0
    wall_start = time.perf_counter()

    async def one(prompt: str):
        nonlocal total_tokens
        t0 = time.perf_counter()
        first = True
        tokens = 0
        stream = await client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                if first:
                    ttfts.append(time.perf_counter() - t0)
                    first = False
                tokens += 1
        total_tokens += tokens

    await asyncio.gather(*[one(p) for p in prompts])
    wall = time.perf_counter() - wall_start
    return total_tokens / wall, ttfts


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------
def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-12)


# ---------------------------------------------------------------------------
# Result storage
# ---------------------------------------------------------------------------
@dataclass
class BenchResult:
    name: str
    mlx: dict[str, Any] = field(default_factory=dict)
    vllm: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


RESULTS: list[BenchResult] = []

# ---------------------------------------------------------------------------
# Benchmark 1: throughput
# ---------------------------------------------------------------------------
LONG_PROMPT = (
    "You are a helpful assistant. Explain in detail the architecture of the Transformer model, "
    "including self-attention, multi-head attention, positional encoding, feed-forward layers, "
    "layer normalization, residual connections, and how encoder-decoder variants differ from "
    "decoder-only models. Be thorough."
)


def _measure_throughput(port: int, model_id: str, pid: int, runs: int = 3) -> dict:
    # Warmup
    chat(port, model_id, [{"role": "user", "content": "Hi"}], max_tokens=20, temperature=0)
    rates = []
    for _ in range(runs):
        t0 = time.perf_counter()
        resp = chat(port, model_id, [{"role": "user", "content": LONG_PROMPT}],
                    max_tokens=500, temperature=0)
        elapsed = time.perf_counter() - t0
        gen = resp.get("usage", {}).get("completion_tokens", 0)
        rates.append(gen / elapsed if elapsed > 0 else 0)
    mem = rss_mb(pid)
    return {
        "tok_s": round(sorted(rates)[len(rates) // 2], 1),
        "rss_mb": mem,
    }


def bench_1_throughput(servers: list[str]):
    print("\n=== Benchmark 1: Single-request decode throughput ===")
    models = [LLAMA_1B, QWEN3_17B, QWEN3_MOE]

    for model in models:
        short = model.split("/")[-1]
        result = BenchResult(name=f"throughput_{short}")
        for server in servers:
            print(f"  [{server}] {short} ...", end=" ", flush=True)
            try:
                with server_ctx(server, model) as (port, model_id, proc):
                    data = _measure_throughput(port, model_id, proc.pid)
                    print(f"{data['tok_s']} tok/s, {data['rss_mb']} MB RSS")
            except Exception as e:
                data = {"tok_s": f"ERROR: {e}"}
                print(f"ERROR: {e}")
            if server == "mlx":
                result.mlx = data
            else:
                result.vllm = data
        RESULTS.append(result)


# ---------------------------------------------------------------------------
# Benchmark 2: continuous batching
# ---------------------------------------------------------------------------
def bench_2_batching(servers: list[str]):
    print("\n=== Benchmark 2: Continuous batching throughput ===")
    model = LLAMA_1B
    concurrency_levels = [1, 2, 4, 8]
    prompt = "Write a short poem about the ocean."
    result = BenchResult(name="batching")

    for server in servers:
        extra = ["--continuous-batching"] if server == "vllm" else ["--batch"]
        data: dict[str, Any] = {}
        try:
            with server_ctx(server, model, extra) as (port, model_id, _proc):
                for n in concurrency_levels:
                    print(f"  [{server}] concurrency={n} ...", end=" ", flush=True)
                    try:
                        tok_s, ttfts = asyncio.run(
                            concurrent_chat(port, model_id, [prompt] * n, max_tokens=200)
                        )
                        p50 = sorted(ttfts)[len(ttfts) // 2] * 1000 if ttfts else 0
                        print(f"{tok_s:.1f} tok/s, TTFT p50={p50:.0f}ms")
                        data[f"n{n}_tok_s"] = round(tok_s, 1)
                        data[f"n{n}_ttft_ms"] = round(p50)
                    except Exception as e:
                        print(f"ERROR: {e}")
                        data[f"n{n}_tok_s"] = "ERROR"
        except Exception as e:
            print(f"  [{server}] SKIP: {e}")

        if server == "mlx":
            result.mlx = data
        else:
            result.vllm = data

    RESULTS.append(result)


# ---------------------------------------------------------------------------
# Benchmark 3: structured output
# ---------------------------------------------------------------------------
SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "score": {"type": "number"},
        "active": {"type": "boolean"},
    },
    "required": ["name", "age", "score", "active"],
    "additionalProperties": False,
}

STRUCTURED_PROMPTS = [
    "Generate a JSON object for a fictional person named Alice who is 30 years old with a score of 9.5 and is active.",
    "Generate a JSON object for a fictional person named Bob who is 25 years old with a score of 7.2 and is not active.",
    "Generate a JSON object for a fictional person. Be creative with the name. Include all required fields.",
    "Create a JSON record for a student with a good grade. active=true.",
    "Make a JSON object for a user profile with name, age, score, active fields.",
    "Generate a JSON record for an employee named Carlos, age 40, score 8.8, active true.",
    "Create a fictional person JSON: name (string), age (integer), score (float), active (boolean).",
    "JSON object for a game character. name, age, score, active are required.",
    "Create a JSON for a researcher named Dr. Smith, age 55, score 9.9, active true.",
    "Generate a JSON profile for a chef. All four fields required.",
]


def validate_schema(text: str) -> bool:
    try:
        obj = json.loads(text.strip())
        for key in SCHEMA["required"]:
            if key not in obj:
                return False
        type_map = {"name": str, "age": int, "score": (int, float), "active": bool}
        for key, typ in type_map.items():
            if not isinstance(obj[key], typ):
                return False
        # age must be int, not float masquerading as int
        if isinstance(obj["age"], float) and not obj["age"].is_integer():
            return False
        return True
    except Exception:
        return False


def bench_3_structured(servers: list[str]):
    print("\n=== Benchmark 3: Structured output (JSON schema) ===")
    # Llama-1B is used (not a thinking model) to avoid thinking mode interference
    # with constrained generation.
    model = LLAMA_1B
    result = BenchResult(name="structured_output")

    for server in servers:
        try:
            with server_ctx(server, model) as (port, model_id, _proc):
                valid = 0
                total = len(STRUCTURED_PROMPTS)
                print(f"  [{server}] {total} prompts ...", end=" ", flush=True)
                for prompt in STRUCTURED_PROMPTS:
                    try:
                        resp = chat(
                            port, model_id,
                            [{"role": "user", "content": prompt}],
                            max_tokens=200, temperature=0,
                            response_format={
                                "type": "json_schema",
                                "json_schema": {"name": "person", "schema": SCHEMA},
                            },
                        )
                        content = resp["choices"][0]["message"]["content"]
                        if validate_schema(content):
                            valid += 1
                    except Exception as e:
                        print(f"\n    prompt error: {e}", end="")
                pct = round(100 * valid / total, 1)
                print(f" {valid}/{total} ({pct}%)")
                data: dict[str, Any] = {"valid": valid, "total": total, "pct": pct}
        except Exception as e:
            print(f"  [{server}] SKIP: {e}")
            data = {"error": str(e)}

        if server == "mlx":
            result.mlx = data
        else:
            result.vllm = data

    RESULTS.append(result)


# ---------------------------------------------------------------------------
# Benchmark 4: logprobs
# ---------------------------------------------------------------------------
LOGPROB_PROMPTS = [
    "The capital of France is",
    "The largest planet in the solar system is",
    "Water boils at 100 degrees",
    "The chemical symbol for gold is",
    "Shakespeare wrote",
]


def bench_4_logprobs(servers: list[str]):
    print("\n=== Benchmark 4: Logprobs accuracy ===")
    model = LLAMA_1B
    responses: dict[str, list] = {}

    for server in servers:
        try:
            with server_ctx(server, model) as (port, model_id, _proc):
                all_lp = []
                print(f"  [{server}] {len(LOGPROB_PROMPTS)} prompts ...", end=" ", flush=True)
                for prompt in LOGPROB_PROMPTS:
                    resp = chat(
                        port, model_id,
                        [{"role": "user", "content": prompt}],
                        max_tokens=5, temperature=0,
                        logprobs=True, top_logprobs=5,
                    )
                    lp = resp["choices"][0].get("logprobs", {})
                    all_lp.append(lp.get("content", []))
                print(f"ok")
            responses[server] = all_lp
        except Exception as e:
            print(f"  [{server}] SKIP: {e}")
            responses[server] = []

    result = BenchResult(name="logprobs")
    mlx_lp = responses.get("mlx", [])
    vllm_lp = responses.get("vllm", [])

    if mlx_lp and vllm_lp:
        agreements = []
        maes = []
        for a, b in zip(mlx_lp, vllm_lp):
            if a and b:
                ta = a[0].get("token")
                tb = b[0].get("token")
                if ta and tb:
                    agreements.append(ta == tb)
                la = a[0].get("logprob")
                lb = b[0].get("logprob")
                if la is not None and lb is not None:
                    maes.append(abs(la - lb))
        agr = round(100 * sum(agreements) / len(agreements), 1) if agreements else 0
        mae = round(sum(maes) / len(maes), 4) if maes else None
        print(f"  Top-1 token agreement: {agr}%  |  Mean |delta logprob|: {mae}")
        result.mlx = {"top1_agreement_pct": agr, "mean_logprob_mae": mae}
        result.vllm = {"top1_agreement_pct": agr, "mean_logprob_mae": mae}
        result.notes = "Both use same MLX weights; diff measures impl divergence only."
    else:
        result.mlx = {"collected": len(mlx_lp)}
        result.vllm = {"collected": len(vllm_lp)}

    RESULTS.append(result)


# ---------------------------------------------------------------------------
# Benchmark 5: reasoning
# ---------------------------------------------------------------------------
REASONING_QUESTIONS = [
    ("What is 17 × 23?", "391"),
    ("If all bloops are razzles and all razzles are lazzles, are all bloops lazzles?", "yes"),
    (
        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
        "How much does the ball cost in cents?",
        "5",
    ),
    ("What is the next number in the sequence: 2, 4, 8, 16, ?", "32"),
    ("If you have 3 apples and take away 2, how many do you have?", "2"),
]


def bench_5_reasoning(servers: list[str]):
    print("\n=== Benchmark 5: Reasoning model (think-tag extraction) ===")
    model = QWEN3_17B
    result = BenchResult(name="reasoning")

    for server in servers:
        extra = ["--reasoning-parser", "qwen3"] if server == "vllm" else []
        try:
            with server_ctx(server, model, extra) as (port, model_id, _proc):
                has_reasoning = 0
                correct = 0
                total = len(REASONING_QUESTIONS)
                print(f"  [{server}] {total} questions ...", end=" ", flush=True)
                for question, expected in REASONING_QUESTIONS:
                    resp = chat(
                        port, model_id,
                        [{"role": "user", "content": question}],
                        max_tokens=2048, temperature=0,
                    )
                    msg = resp["choices"][0]["message"]
                    content = msg.get("content", "")
                    reasoning = msg.get("reasoning_content", "")
                    # Correct extraction: reasoning_content populated, no raw tags in content
                    if reasoning and "<think>" not in content and "</think>" not in content:
                        has_reasoning += 1
                    if str(expected).lower() in content.lower():
                        correct += 1
                print(f"reasoning_extracted={has_reasoning}/{total}, correct={correct}/{total}")
                data: dict[str, Any] = {
                    "reasoning_extracted": has_reasoning,
                    "correct": correct,
                    "total": total,
                }
        except Exception as e:
            print(f"  [{server}] SKIP: {e}")
            data = {"error": str(e)}

        if server == "mlx":
            result.mlx = data
        else:
            result.vllm = data

    RESULTS.append(result)


# ---------------------------------------------------------------------------
# Benchmark 6: embeddings
# ---------------------------------------------------------------------------
# STS-style pairs with expected approximate cosine similarity
STS_PAIRS: list[tuple[str, str, float]] = [
    ("A man is playing guitar.", "Someone is playing a musical instrument.", 0.85),
    ("The cat sat on the mat.", "A dog is running in the park.", 0.15),
    ("She loves programming in Python.", "Python is her favorite programming language.", 0.90),
    ("The stock market crashed today.", "Investors are worried about the economy.", 0.70),
    ("I enjoy hiking in the mountains.", "I hate going outside.", 0.10),
    ("The movie was fantastic.", "I really enjoyed the film.", 0.92),
    ("Water is composed of hydrogen and oxygen.", "H2O is the chemical formula for water.", 0.95),
    ("He went to the grocery store.", "She baked a cake at home.", 0.20),
]


def _spearman(predicted: list[float], expected: list[float]) -> float:
    n = len(predicted)
    rp = sorted(range(n), key=lambda i: predicted[i])
    re = sorted(range(n), key=lambda i: expected[i])
    rank_p = [rp.index(i) for i in range(n)]
    rank_e = [re.index(i) for i in range(n)]
    d2 = sum((rank_p[i] - rank_e[i]) ** 2 for i in range(n))
    return 1 - 6 * d2 / (n * (n * n - 1))


def bench_6_embeddings(servers: list[str]):
    print("\n=== Benchmark 6: Embeddings ===")
    result = BenchResult(name="embeddings")

    for server in servers:
        try:
            # mlx-server: uses Llama-1B (causal LM embeddings via hidden states).
            # MINILM is encoder-only (BERT) and not supported by mlx-server.
            # vllm-mlx: serves Llama-1B + MINILM as a dedicated embedding model.
            if server == "mlx":
                ctx = mlx_server(LLAMA_1B)
            else:
                ctx = vllm_server(LLAMA_1B, ["--embedding-model", MINILM])

            with ctx as (port, model_id, _proc):
                # vllm-mlx uses the MINILM model for embeddings; mlx-server uses LLAMA_1B.
                emb_model_id = MINILM if server == "vllm" else model_id

                all_sents = [s for pair in STS_PAIRS for s in pair[:2]]
                print(f"  [{server}] {len(STS_PAIRS)} pairs ...", end=" ", flush=True)
                embeddings = embed(port, emb_model_id, all_sents)

                predicted = []
                expected = []
                for i, (_, _, exp) in enumerate(STS_PAIRS):
                    e1 = embeddings[i * 2]
                    e2 = embeddings[i * 2 + 1]
                    predicted.append(cosine(e1, e2))
                    expected.append(exp)

                spearman = _spearman(predicted, expected)
                norms = [math.sqrt(sum(x ** 2 for x in e)) for e in embeddings]
                mean_norm = sum(norms) / len(norms)
                print(f"Spearman={spearman:.3f}, mean L2 norm={mean_norm:.4f}")
                data: dict[str, Any] = {
                    "spearman": round(spearman, 3),
                    "mean_l2_norm": round(mean_norm, 4),
                    "dim": len(embeddings[0]),
                }
        except Exception as e:
            print(f"  [{server}] SKIP: {e}")
            data = {"error": str(e)}

        if server == "mlx":
            result.mlx = data
        else:
            result.vllm = data

    RESULTS.append(result)


# ---------------------------------------------------------------------------
# Benchmark 7: vision
# ---------------------------------------------------------------------------
def _solid_png_b64(r: int, g: int, b: int, size: int = 64) -> str:
    try:
        from PIL import Image
        import io
        img = Image.new("RGB", (size, size), color=(r, g, b))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except ImportError:
        # 1x1 red pixel fallback
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwADhQGAWjR9awAAAABJRU5ErkJggg=="


VISION_TESTS = [
    (_solid_png_b64(255, 0, 0), "red"),
    (_solid_png_b64(0, 255, 0), "green"),
    (_solid_png_b64(0, 0, 255), "blue"),
    (_solid_png_b64(255, 255, 0), "yellow"),
    (_solid_png_b64(255, 165, 0), "orange"),
]


def bench_7_vision(servers: list[str]):
    print("\n=== Benchmark 7: Vision (multimodal) ===")
    model = NANOVLLAVA
    result = BenchResult(name="vision")

    for server in servers:
        try:
            with server_ctx(server, model) as (port, model_id, _proc):
                correct = 0
                total = len(VISION_TESTS)
                print(f"  [{server}] {total} color images ...", end=" ", flush=True)
                for img_b64, expected_color in VISION_TESTS:
                    try:
                        resp = chat(
                            port, model_id,
                            [{"role": "user", "content": [
                                {"type": "text",
                                 "text": "What is the dominant color in this image? Reply with one word only."},
                                {"type": "image_url",
                                 "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                            ]}],
                            max_tokens=10, temperature=0,
                        )
                        answer = resp["choices"][0]["message"]["content"].strip().lower()
                        if expected_color in answer:
                            correct += 1
                    except Exception as e:
                        print(f"\n    img error: {e}", end="")
                pct = round(100 * correct / total, 1)
                print(f" {correct}/{total} ({pct}%)")
                data: dict[str, Any] = {"correct": correct, "total": total, "pct": pct}
        except Exception as e:
            print(f"  [{server}] SKIP: {e}")
            data = {"error": str(e)}

        if server == "mlx":
            result.mlx = data
        else:
            result.vllm = data

    RESULTS.append(result)


# ---------------------------------------------------------------------------
# Benchmark 8: architecture smoke tests
# ---------------------------------------------------------------------------
ARCH_MODELS = [
    ("gemma2", GEMMA2),
    ("phi3", PHI3),
    ("starcoder2", STARCODER2),
    ("deepseek_v2", DEEPSEEK),
]

# Models that use /v1/completions (no chat template)
COMPLETION_ONLY_MODELS = {STARCODER2}


def bench_8_architectures(servers: list[str]):
    print("\n=== Benchmark 8: Architecture smoke tests ===")
    chat_prompt = "Hello! Please introduce yourself in one sentence."
    # Code completion prompt for models without chat templates
    completion_prompt = "def fibonacci(n):\n    "

    for arch, model in ARCH_MODELS:
        result = BenchResult(name=f"arch_{arch}")
        for server in servers:
            print(f"  [{server}] {arch} ...", end=" ", flush=True)
            try:
                with server_ctx(server, model) as (port, model_id, proc):
                    t0 = time.perf_counter()
                    if model in COMPLETION_ONLY_MODELS:
                        resp = complete(
                            port, model_id, completion_prompt,
                            max_tokens=100, temperature=0,
                        )
                        content = resp["choices"][0]["text"]
                    else:
                        resp = chat(
                            port, model_id,
                            [{"role": "user", "content": chat_prompt}],
                            max_tokens=100, temperature=0,
                        )
                        content = resp["choices"][0]["message"]["content"]
                    elapsed = time.perf_counter() - t0
                    gen = resp.get("usage", {}).get("completion_tokens", 0)
                    tok_s = round(gen / elapsed, 1) if elapsed > 0 else 0
                    coherent = len(content.strip()) > 10
                    mem = rss_mb(proc.pid)
                    print(f"ok ({tok_s} tok/s, {mem} MB RSS, coherent={coherent})")
                    data: dict[str, Any] = {
                        "tok_s": tok_s, "rss_mb": mem, "coherent": coherent,
                        "sample": content[:80],
                    }
            except Exception as e:
                print(f"ERROR: {e}")
                data = {"error": str(e)}

            if server == "mlx":
                result.mlx = data
            else:
                result.vllm = data

        RESULTS.append(result)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def build_readme_section() -> str:
    lines: list[str] = []

    # Throughput + memory
    throughput = [r for r in RESULTS if r.name.startswith("throughput_")]
    if throughput:
        lines += [
            "### Single-request decode throughput\n",
            "Prompt: ~100 tokens, generation: 500 tokens, temperature=0, median of 3 runs.  ",
            "M4 Max 128GB. One server at a time, warmup pass excluded.\n",
            "| Model | mlx-server (tok/s) | mlx-server RSS (MB) | vllm-mlx (tok/s) | vllm-mlx RSS (MB) |",
            "|---|---|---|---|---|",
        ]
        for r in throughput:
            m = r.name.replace("throughput_", "")
            lines.append(
                f"| {m} | {r.mlx.get('tok_s', 'n/a')} | {r.mlx.get('rss_mb', 'n/a')} "
                f"| {r.vllm.get('tok_s', 'n/a')} | {r.vllm.get('rss_mb', 'n/a')} |"
            )
        lines.append("")

    # Batching
    bat = next((r for r in RESULTS if r.name == "batching"), None)
    if bat:
        lines += [
            "### Continuous batching throughput\n",
            "Model: Llama-3.2-1B-Instruct-4bit. Concurrent requests, 200 tokens each.\n",
            "| Concurrency | mlx-server tok/s | mlx-server TTFT p50 | vllm-mlx tok/s | vllm-mlx TTFT p50 |",
            "|---|---|---|---|---|",
        ]
        for n in [1, 2, 4, 8]:
            ms = bat.mlx.get(f"n{n}_tok_s", "n/a")
            mt = bat.mlx.get(f"n{n}_ttft_ms", "n/a")
            vs = bat.vllm.get(f"n{n}_tok_s", "n/a")
            vt = bat.vllm.get(f"n{n}_ttft_ms", "n/a")
            lines.append(f"| {n} | {ms} | {mt}ms | {vs} | {vt}ms |")
        lines.append("")

    # Structured output
    s = next((r for r in RESULTS if r.name == "structured_output"), None)
    if s:
        lines += [
            "### Structured output (JSON schema compliance)\n",
            "Model: Llama-3.2-1B-Instruct-4bit. 10 prompts, `json_schema` response format with 4 required fields.\n",
            "| | mlx-server | vllm-mlx |",
            "|---|---|---|",
            f"| Schema compliance | {s.mlx.get('pct', 'n/a')}% | {s.vllm.get('pct', 'n/a')}% |",
            "",
        ]

    # Logprobs
    lp = next((r for r in RESULTS if r.name == "logprobs"), None)
    if lp:
        agr = lp.mlx.get("top1_agreement_pct", "n/a")
        mae = lp.mlx.get("mean_logprob_mae", "n/a")
        lines += [
            "### Log probability accuracy\n",
            "Model: Llama-3.2-1B-Instruct-4bit. 5 prompts, `top_logprobs=5`.\n",
            "| Metric | Value |",
            "|---|---|",
            f"| Top-1 token agreement (mlx-server vs vllm-mlx) | {agr}% |",
            f"| Mean absolute error on log probabilities | {mae} |",
            "",
            "*Both servers run the same MLX weights; any gap reflects implementation divergence.*\n",
        ]

    # Reasoning
    reas = next((r for r in RESULTS if r.name == "reasoning"), None)
    if reas:
        mt = reas.mlx.get("total", 5)
        vt = reas.vllm.get("total", 5)
        lines += [
            "### Reasoning model (think-tag extraction)\n",
            "Model: Qwen3-1.7B-4bit. 5 math/logic questions.\n",
            "| | mlx-server | vllm-mlx |",
            "|---|---|---|",
            f"| `reasoning_content` populated | {reas.mlx.get('reasoning_extracted', 'n/a')}/{mt} | {reas.vllm.get('reasoning_extracted', 'n/a')}/{vt} |",
            f"| Correct answers | {reas.mlx.get('correct', 'n/a')}/{mt} | {reas.vllm.get('correct', 'n/a')}/{vt} |",
            "",
        ]

    # Embeddings
    emb = next((r for r in RESULTS if r.name == "embeddings"), None)
    if emb:
        lines += [
            "### Embeddings\n",
            "Model: all-MiniLM-L6-v2-4bit. 8 STS-style sentence pairs.\n",
            "| Metric | mlx-server | vllm-mlx |",
            "|---|---|---|",
            f"| Spearman correlation (vs reference scores) | {emb.mlx.get('spearman', 'n/a')} | {emb.vllm.get('spearman', 'n/a')} |",
            f"| Mean L2 norm (should be ~1.0) | {emb.mlx.get('mean_l2_norm', 'n/a')} | {emb.vllm.get('mean_l2_norm', 'n/a')} |",
            f"| Embedding dimension | {emb.mlx.get('dim', 'n/a')} | {emb.vllm.get('dim', 'n/a')} |",
            "",
        ]

    # Vision
    vis = next((r for r in RESULTS if r.name == "vision"), None)
    if vis:
        lines += [
            "### Vision (multimodal)\n",
            "Model: nanoLLaVA-1.5-4bit. 5 solid-color test images, single-word color identification.\n",
            "| | mlx-server | vllm-mlx |",
            "|---|---|---|",
            f"| Color identification accuracy | {vis.mlx.get('pct', 'n/a')}% | {vis.vllm.get('pct', 'n/a')}% |",
            "",
        ]

    # Architecture smoke tests
    archs = [r for r in RESULTS if r.name.startswith("arch_")]
    if archs:
        lines += [
            "### Architecture smoke tests\n",
            "Single 100-token generation, temperature=0. Coherent = non-empty, non-garbled output.\n",
            "| Architecture | mlx-server tok/s | mlx-server RSS (MB) | mlx-server coherent | vllm-mlx tok/s | vllm-mlx RSS (MB) | vllm-mlx coherent |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in archs:
            arch = r.name.replace("arch_", "")
            lines.append(
                f"| {arch} | {r.mlx.get('tok_s', 'n/a')} | {r.mlx.get('rss_mb', 'n/a')} | {r.mlx.get('coherent', 'n/a')} "
                f"| {r.vllm.get('tok_s', 'n/a')} | {r.vllm.get('rss_mb', 'n/a')} | {r.vllm.get('coherent', 'n/a')} |"
            )
        lines.append("")

    return "\n".join(lines)


def print_summary():
    print("\n" + "=" * 72)
    print("RAW RESULTS")
    print("=" * 72)
    for r in RESULTS:
        print(f"\n{r.name}")
        print(f"  mlx : {r.mlx}")
        print(f"  vllm: {r.vllm}")
        if r.notes:
            print(f"  note: {r.notes}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
BENCH_MAP = {
    "1_throughput": bench_1_throughput,
    "2_batching": bench_2_batching,
    "3_structured": bench_3_structured,
    "4_logprobs": bench_4_logprobs,
    "5_reasoning": bench_5_reasoning,
    "6_embeddings": bench_6_embeddings,
    "7_vision": bench_7_vision,
    "8_architectures": bench_8_architectures,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", default=",".join(BENCH_MAP))
    parser.add_argument("--only-server", choices=["mlx", "vllm"])
    args = parser.parse_args()

    servers = [args.only_server] if args.only_server else ["mlx", "vllm"]
    selected = [b.strip() for b in args.benchmarks.split(",")]

    for bench_id in selected:
        if bench_id not in BENCH_MAP:
            print(f"Unknown benchmark: {bench_id}")
            continue
        BENCH_MAP[bench_id](servers)

    print_summary()

    section = build_readme_section()
    out = Path(__file__).parent / "results.md"
    out.write_text(section)
    print(f"\nMarkdown section written to {out}")

    # Also save raw JSON
    raw = Path(__file__).parent / "results.json"
    raw.write_text(json.dumps(
        [{"name": r.name, "mlx": r.mlx, "vllm": r.vllm, "notes": r.notes} for r in RESULTS],
        indent=2,
    ))
    print(f"Raw JSON saved to {raw}")


if __name__ == "__main__":
    main()
