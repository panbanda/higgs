#!/usr/bin/env python3
"""Benchmark paged prefix cache: TTFT on cache miss vs cache hit."""

import json
import subprocess
import sys
import time
import signal
import os
import urllib.request
import urllib.error

SERVER = "http://127.0.0.1:8080"
HIGGS = "./target/release/higgs"

# Large system prompt (~1500 words, ~2000 tokens) to exercise prefix caching
SYSTEM_PROMPT = """You are a highly skilled software architect with deep expertise in distributed systems, \
database design, and cloud-native applications. You provide thorough, well-reasoned technical advice. \
When answering questions, you consider trade-offs, scalability implications, and maintainability. \
You draw on experience with microservices, event-driven architectures, and modern DevOps practices. \
You always explain your reasoning step by step and provide concrete examples when possible. \
Your responses are structured with clear headings and bullet points for readability.

You are familiar with AWS, GCP, Azure, Kubernetes, Docker, Terraform, and various CI/CD pipelines. \
You understand the CAP theorem, ACID properties, eventual consistency, and distributed consensus algorithms. \
You can discuss the pros and cons of SQL vs NoSQL databases, message queues vs event streams, \
synchronous vs asynchronous communication patterns, and monolithic vs microservice architectures. \
You are also knowledgeable about security best practices, observability (logging, metrics, tracing), \
and performance optimization techniques. When discussing code, you prefer clean, idiomatic solutions \
that follow established patterns in the relevant ecosystem.

Here is your detailed knowledge base that you must reference when answering:

SECTION 1 - DATABASE DESIGN PRINCIPLES:
When designing databases, always consider the access patterns first. For OLTP workloads, normalize to 3NF \
and use appropriate indexes. For OLAP workloads, consider star or snowflake schemas. Partitioning strategies \
should be based on query patterns: range partitioning for time-series data, hash partitioning for uniform \
distribution, and list partitioning for categorical data. Always plan for data growth and implement proper \
archival strategies. Connection pooling is essential - use PgBouncer for PostgreSQL, ProxySQL for MySQL. \
Read replicas should be used for read-heavy workloads, but be aware of replication lag implications. \
For multi-region deployments, consider CockroachDB, YugabyteDB, or Spanner for global consistency.

SECTION 2 - MICROSERVICES PATTERNS:
Service boundaries should align with business domains (Domain-Driven Design). Use the Strangler Fig pattern \
for migrating from monoliths. Implement Circuit Breakers (Hystrix/Resilience4j) for fault tolerance. \
Use the Saga pattern for distributed transactions - prefer choreography over orchestration for loose coupling. \
API Gateway pattern (Kong, Envoy) handles cross-cutting concerns. Service mesh (Istio, Linkerd) provides \
observability and traffic management. Event sourcing with CQRS is appropriate when audit trails are required \
or when read and write models differ significantly. Use the Outbox pattern for reliable event publishing. \
Idempotency keys are essential for safe retries. Health checks should include both liveness and readiness probes.

SECTION 3 - CACHING STRATEGIES:
Implement caching at multiple levels: CDN for static assets, application-level for computed results, \
database query cache for frequent queries. Use Cache-Aside (Lazy Loading) as the default pattern. \
Write-Through caching ensures consistency but adds latency. Write-Behind (Write-Back) improves write \
performance but risks data loss. Cache invalidation strategies: TTL-based for simplicity, event-driven \
for consistency. Redis Cluster for horizontal scaling, Redis Sentinel for high availability. \
Implement cache warming for predictable traffic patterns. Use bloom filters to prevent cache penetration. \
Consider local caches (Caffeine, Guava) before distributed caches to reduce network overhead.

SECTION 4 - SECURITY BEST PRACTICES:
Implement defense in depth: network segmentation, WAF, rate limiting, input validation, output encoding. \
Use OAuth 2.0 with PKCE for public clients, client credentials for service-to-service. JWT tokens should \
be short-lived (15 minutes) with refresh token rotation. Store secrets in Vault or AWS Secrets Manager, \
never in code or environment variables. Implement mutual TLS for service-to-service communication. \
Use prepared statements to prevent SQL injection. Apply the principle of least privilege for IAM roles. \
Enable audit logging for all sensitive operations. Implement CORS properly - never use wildcard in production. \
Regular dependency scanning with Snyk or Dependabot. Container image scanning with Trivy or Aqua.

SECTION 5 - OBSERVABILITY:
The three pillars: metrics (Prometheus/Datadog), logs (ELK/Loki), traces (Jaeger/Zipkin). \
Use structured logging (JSON) with correlation IDs across services. Implement RED metrics for services \
(Rate, Errors, Duration) and USE metrics for resources (Utilization, Saturation, Errors). \
Set up alerts based on SLOs, not arbitrary thresholds. Use percentiles (p50, p95, p99) instead of averages \
for latency metrics. Implement distributed tracing with OpenTelemetry for cross-service visibility. \
Create dashboards that answer specific questions rather than showing all available metrics. \
Log aggregation should include proper retention policies and cost management. Error tracking with \
Sentry or Bugsnag for application-level issues. Synthetic monitoring for critical user journeys.

SECTION 6 - PERFORMANCE OPTIMIZATION:
Profile before optimizing - use flame graphs for CPU, heap dumps for memory. Database: EXPLAIN ANALYZE \
all slow queries, add covering indexes, optimize JOIN order. Application: minimize allocations, use \
connection pooling, batch operations where possible. Network: enable HTTP/2, use gRPC for internal \
services, implement request coalescing. Frontend: lazy loading, code splitting, image optimization. \
Load testing with k6 or Locust before production releases. Capacity planning based on projected growth \
with 2x headroom. Horizontal scaling is preferred over vertical - design for statelessness. \
Use async processing for non-critical paths. Implement backpressure mechanisms to prevent cascading failures.

You are pragmatic and focus on delivering value rather than over-engineering solutions. Keep answers concise \
but thorough. Always mention relevant trade-offs."""

USER_PROMPTS = [
    "Explain the CAP theorem in one paragraph.",
    "What is eventual consistency? Keep it brief.",
    "Compare Redis and Memcached in three sentences.",
]


def wait_for_server(timeout=120):
    """Wait until the server responds to /v1/models. Returns model name."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = urllib.request.Request(f"{SERVER}/v1/models")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read())
                    return data["data"][0]["id"]
        except (urllib.error.URLError, ConnectionRefusedError, OSError, KeyError, IndexError):
            pass
        time.sleep(1)
    return None


def stream_chat(messages, max_tokens=50, model_name="test"):
    """Send a streaming chat completion and measure TTFT + collect output."""
    payload = json.dumps({
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }).encode()

    req = urllib.request.Request(
        f"{SERVER}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    t0 = time.perf_counter()
    first_token_time = None
    tokens = []

    with urllib.request.urlopen(req, timeout=300) as resp:
        buffer = b""
        while True:
            chunk = resp.read(1)
            if not chunk:
                break
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                line = line.strip()
                if not line or line == b"data: [DONE]":
                    continue
                if line.startswith(b"data: "):
                    try:
                        data = json.loads(line[6:])
                        choices = data.get("choices", [])
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content and first_token_time is None:
                            first_token_time = time.perf_counter()
                        if content:
                            tokens.append(content)
                    except (json.JSONDecodeError, KeyError, IndexError):
                        pass

    total_time = time.perf_counter() - t0
    ttft = (first_token_time - t0) if first_token_time else total_time

    return {
        "ttft_ms": ttft * 1000,
        "total_ms": total_time * 1000,
        "output": "".join(tokens),
        "num_tokens": len(tokens),
    }


def bench_model(model_dir, label):
    """Benchmark a single model: cache miss, cache hit, different prompt."""
    print(f"\n{'='*70}")
    print(f"Model: {label}")
    print(f"Path:  {model_dir}")
    print(f"{'='*70}")

    # Kill any existing server
    subprocess.run(["pkill", "-f", "higgs serve"], capture_output=True)
    time.sleep(1)

    # Start server
    env = {**os.environ, "HIGGS_ENABLE_THINKING": "0"}
    proc = subprocess.Popen(
        [HIGGS, "serve", "--model", model_dir, "--port", "8080"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        print("Waiting for server...", end="", flush=True)
        model_name = wait_for_server()
        if not model_name:
            print(" TIMEOUT")
            return None
        print(f" ready (model={model_name})")

        results = []

        # Warmup: one throwaway request to stabilize GPU
        print("Warmup...", end="", flush=True)
        warmup_msgs = [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Say hello."},
        ]
        stream_chat(warmup_msgs, max_tokens=10, model_name=model_name)
        print(" done")

        for i, user_msg in enumerate(USER_PROMPTS):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]
            tag = "MISS" if i == 0 else "HIT"
            print(f"\n  [{tag}] Request {i+1}: \"{user_msg}\"")

            r = stream_chat(messages, max_tokens=80, model_name=model_name)
            results.append(r)

            print(f"    TTFT:   {r['ttft_ms']:8.1f} ms")
            print(f"    Total:  {r['total_ms']:8.1f} ms  ({r['num_tokens']} tokens)")
            print(f"    Output: {r['output'][:120]}...")

        # Summary
        miss_ttft = results[0]["ttft_ms"]
        hit_ttfts = [r["ttft_ms"] for r in results[1:]]
        avg_hit = sum(hit_ttfts) / len(hit_ttfts)
        speedup = miss_ttft / avg_hit if avg_hit > 0 else 0

        print(f"\n  --- Summary ---")
        print(f"  Cache MISS TTFT:  {miss_ttft:8.1f} ms")
        print(f"  Cache HIT  TTFT:  {avg_hit:8.1f} ms (avg of {len(hit_ttfts)} hits)")
        print(f"  Speedup:          {speedup:8.2f}x")

        return {
            "model": label,
            "miss_ttft_ms": miss_ttft,
            "avg_hit_ttft_ms": avg_hit,
            "speedup": speedup,
            "results": results,
        }

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        time.sleep(2)


def main():
    models = [
        (
            os.path.expanduser("~/.cache/lm-studio/models/NexVeridian/Qwen3.5-35B-A3B-3bit"),
            "Qwen3.5-35B-A3B-3bit (MoE)",
        ),
        (
            os.path.expanduser("~/.cache/lm-studio/models/mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit"),
            "Qwen3.5-27B-4bit (Dense)",
        ),
        (
            os.path.expanduser("~/.cache/lm-studio/models/mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx"),
            "DeepSeek-V2-Lite-4bit (MoE)",
        ),
    ]

    # Filter to models that exist
    models = [(d, l) for d, l in models if os.path.isdir(d)]
    if not models:
        print("No models found!")
        sys.exit(1)

    print(f"Benchmarking {len(models)} models")
    print(f"System prompt: ~{len(SYSTEM_PROMPT.split())} words")

    all_results = []
    for model_dir, label in models:
        result = bench_model(model_dir, label)
        if result:
            all_results.append(result)

    # Final table
    if all_results:
        print(f"\n{'='*70}")
        print("FINAL RESULTS")
        print(f"{'='*70}")
        print(f"{'Model':<35} {'Miss TTFT':>10} {'Hit TTFT':>10} {'Speedup':>8}")
        print("-" * 70)
        for r in all_results:
            print(
                f"{r['model']:<35} {r['miss_ttft_ms']:>8.0f}ms {r['avg_hit_ttft_ms']:>8.0f}ms {r['speedup']:>7.2f}x"
            )


if __name__ == "__main__":
    main()
