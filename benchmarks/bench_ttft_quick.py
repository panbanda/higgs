"""Quick TTFT benchmark at different prompt lengths."""
import time, json, urllib.request

BASE = "http://localhost:9999/v1"
MODEL = "DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx"

# Generate prompts of increasing length
WORD = "the quick brown fox jumps over the lazy dog "  # ~10 tokens
WARMUP = 1
ITERS = 3

def ttft(prompt, max_tokens=1):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        f"{BASE}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        d = json.loads(resp.read())
    elapsed = time.perf_counter() - t0
    usage = d.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", "?")
    return elapsed, prompt_tokens

def bench_ttft(label, prompt):
    # Warmup with a unique prompt to avoid prefix cache
    for i in range(WARMUP):
        ttft(f"[warmup {i}] {prompt}")

    times = []
    ptoks = None
    for i in range(ITERS):
        # Unique prefix each iter to defeat prefix cache
        t, pt = ttft(f"[run {i} {time.time()}] {prompt}")
        times.append(t)
        ptoks = pt

    med = sorted(times)[len(times) // 2]
    return med, ptoks

print(f"TTFT benchmark — {MODEL}")
print(f"  {WARMUP} warmup, {ITERS} iters, reporting median\n")

prompt_sizes = [
    ("short", WORD * 3),
    ("medium", WORD * 50),
    ("long", WORD * 200),
    ("very_long", WORD * 500),
]

print(f"{'Label':>12} | {'Tokens':>8} | {'TTFT':>10}")
print("-" * 40)

for label, prompt in prompt_sizes:
    med, ptoks = bench_ttft(label, prompt)
    print(f"{label:>12} | {ptoks:>8} | {med*1000:>7.0f} ms")
