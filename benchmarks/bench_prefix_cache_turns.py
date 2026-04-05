#!/usr/bin/env python3
"""Benchmark: does prefix cache speedup degrade with conversation turns?"""

import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error

SERVER = "http://127.0.0.1:8080"
HIGGS = "./target/release/higgs"

# Same large system prompt from main benchmark
SYSTEM_PROMPT = """You are a highly skilled software architect with deep expertise in distributed systems, \
database design, and cloud-native applications. You provide thorough, well-reasoned technical advice. \
When answering questions, you consider trade-offs, scalability implications, and maintainability. \
You draw on experience with microservices, event-driven architectures, and modern DevOps practices. \
You always explain your reasoning step by step and provide concrete examples when possible.

Here is your detailed knowledge base:

SECTION 1 - DATABASE DESIGN PRINCIPLES:
When designing databases, always consider the access patterns first. For OLTP workloads, normalize to 3NF \
and use appropriate indexes. For OLAP workloads, consider star or snowflake schemas. Partitioning strategies \
should be based on query patterns: range partitioning for time-series data, hash partitioning for uniform \
distribution. Connection pooling is essential. Read replicas for read-heavy workloads.

SECTION 2 - MICROSERVICES PATTERNS:
Service boundaries should align with business domains (Domain-Driven Design). Use the Strangler Fig pattern \
for migrating from monoliths. Implement Circuit Breakers for fault tolerance. Use the Saga pattern for \
distributed transactions. API Gateway handles cross-cutting concerns. Service mesh provides observability. \
Event sourcing with CQRS when audit trails are required. Outbox pattern for reliable event publishing.

SECTION 3 - CACHING STRATEGIES:
Implement caching at multiple levels: CDN for static assets, application-level for computed results. \
Use Cache-Aside as the default pattern. Write-Through ensures consistency but adds latency. \
Redis Cluster for horizontal scaling, Redis Sentinel for high availability. Bloom filters prevent \
cache penetration. Consider local caches before distributed caches to reduce network overhead.

SECTION 4 - SECURITY BEST PRACTICES:
Implement defense in depth: network segmentation, WAF, rate limiting, input validation. \
Use OAuth 2.0 with PKCE for public clients. JWT tokens should be short-lived with refresh rotation. \
Store secrets in Vault, never in code. Mutual TLS for service-to-service. Prepared statements for SQL. \
Principle of least privilege for IAM. Audit logging for sensitive operations.

SECTION 5 - OBSERVABILITY:
Three pillars: metrics, logs, traces. Structured logging with correlation IDs. RED metrics for services. \
Alerts based on SLOs not thresholds. Percentiles not averages for latency. OpenTelemetry for tracing. \
Error tracking with Sentry. Synthetic monitoring for critical journeys.

Keep answers concise. Always mention trade-offs."""

# Simulated multi-turn conversation
TURNS = [
    ("What database would you recommend for a time-series IoT platform?",
     "For a time-series IoT platform, I'd recommend TimescaleDB as the primary choice. It extends PostgreSQL with automatic partitioning by time, native compression achieving 90%+ reduction, and continuous aggregates for real-time rollups. Key advantages: you keep full SQL compatibility, can JOIN with relational data, and benefit from PostgreSQL's ecosystem. For extreme scale (millions of writes/sec), consider InfluxDB or QuestDB which sacrifice SQL flexibility for raw ingestion speed. The trade-off is clear: TimescaleDB gives you versatility, while purpose-built TSDBs give you peak throughput."),
    ("How should I handle the ingestion pipeline?",
     "For IoT ingestion, implement a buffered pipeline: devices publish to MQTT (Mosquitto/EMQX), a bridge forwards to Kafka for durability and backpressure, then Kafka consumers batch-insert into TimescaleDB. Key design decisions: use Kafka partitioning by device_id for ordering guarantees, set batch sizes of 1000-5000 rows for optimal insert throughput, and implement dead-letter queues for malformed data. Add schema validation at the MQTT bridge level to reject bad payloads early. For backpressure, Kafka naturally handles this - consumers process at their own pace while producers buffer in topics."),
    ("What about real-time alerting on the data?",
     "Layer your alerting: use TimescaleDB continuous aggregates for threshold-based alerts (e.g., avg temperature > X over 5min windows), and Kafka Streams or Flink for complex event processing (e.g., detecting anomaly patterns across multiple sensors). For the alert pipeline: Kafka topic for raw events, stream processor evaluates rules, alert events go to a separate topic, then a notification service dispatches via PagerDuty/Slack/email. Implement alert deduplication and suppression to avoid alert fatigue. Store alert history in PostgreSQL for audit trails. Use Grafana with TimescaleDB datasource for visualization dashboards."),
    ("How do I scale this to millions of devices?",
     "Scaling to millions of devices requires horizontal scaling at every layer. MQTT: use EMQX cluster with shared subscriptions, each node handles ~500K concurrent connections. Kafka: partition by device_id hash, scale consumers with consumer groups. TimescaleDB: use distributed hypertables across multiple nodes, partition by both time and device_id. Add a device registry service for metadata. Implement connection pooling with PgBouncer. For cost optimization, tier your storage: hot data (recent 7 days) on SSD-backed TimescaleDB, warm data (30 days) on cheaper storage, cold data archived to S3/Parquet for analytics. Use read replicas for dashboard queries to isolate from write path."),
    ("What monitoring should I set up for this infrastructure?",
     "Implement observability across all layers using the RED/USE framework. Infrastructure: Prometheus with node_exporter for CPU/memory/disk, kube-state-metrics for Kubernetes. MQTT: monitor connected clients, message rate, subscription count. Kafka: consumer lag (critical - use Burrow), broker throughput, partition skew. TimescaleDB: query latency p95/p99, connection pool utilization, chunk compression ratio, replication lag. Application: request rate, error rate, latency histograms per endpoint. Create four dashboards: system health overview, ingestion pipeline throughput, database performance, and alert system health. Set SLOs: 99.9% ingestion success rate, p99 query latency < 500ms, alert delivery within 60 seconds."),
]

NEW_QUESTIONS = [
    "Now, what if I need to add machine learning predictions on the incoming data?",
    "Should I use GraphQL or REST for the device management API?",
]


def wait_for_server(timeout=120):
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
    return ttft * 1000, "".join(tokens), len(tokens)


def count_message_tokens(messages):
    """Rough token estimate: ~1.3 tokens per word."""
    total_words = sum(len(m["content"].split()) for m in messages)
    return int(total_words * 1.3)


def bench_model(model_dir, label):
    print(f"\n{'='*70}")
    print(f"Model: {label}")
    print(f"{'='*70}")

    subprocess.run(["pkill", "-f", "higgs serve"], capture_output=True)
    time.sleep(2)

    env = {**os.environ, "HIGGS_ENABLE_THINKING": "0"}
    proc = subprocess.Popen(
        [HIGGS, "serve", "--model", model_dir, "--port", "8080"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )

    try:
        print("Waiting for server...", end="", flush=True)
        model_name = wait_for_server()
        if not model_name:
            print(" TIMEOUT")
            return
        print(f" ready")

        # Warmup
        stream_chat([{"role": "system", "content": "Be brief."},
                     {"role": "user", "content": "Hi."}],
                    max_tokens=5, model_name=model_name)

        # Build conversation history incrementally
        # First: cache miss with system prompt + first question
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        print(f"\n  {'Turn':<6} {'Ctx tokens':>10} {'TTFT (ms)':>10} {'vs Miss':>8}  Question")
        print(f"  {'-'*64}")

        miss_ttft = None

        for turn_idx, (question, fake_answer) in enumerate(TURNS):
            messages.append({"role": "user", "content": question})
            est_tokens = count_message_tokens(messages)

            ttft, output, ntok = stream_chat(messages, max_tokens=60, model_name=model_name)

            if turn_idx == 0:
                miss_ttft = ttft
                tag = "MISS"
                ratio = "1.00x"
            else:
                tag = "HIT"
                ratio = f"{miss_ttft / ttft:.1f}x" if ttft > 0 else "inf"

            print(f"  {turn_idx+1:<6} {est_tokens:>10} {ttft:>10.0f} {ratio:>8}  {question[:50]}...")

            # Add the fake assistant response to history for next turn
            messages.append({"role": "assistant", "content": fake_answer})

        # Now send 2 new questions with the FULL history — cache should hit on prefix
        print(f"\n  --- New questions with full history ({len(TURNS)} prior turns) ---")
        for q in NEW_QUESTIONS:
            messages_copy = list(messages)  # copy to not pollute
            messages_copy.append({"role": "user", "content": q})
            est_tokens = count_message_tokens(messages_copy)

            ttft, output, ntok = stream_chat(messages_copy, max_tokens=60, model_name=model_name)
            ratio = f"{miss_ttft / ttft:.1f}x" if ttft > 0 else "inf"
            print(f"  {'new':<6} {est_tokens:>10} {ttft:>10.0f} {ratio:>8}  {q[:50]}...")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        time.sleep(2)


def main():
    models = [
        (os.path.expanduser("~/.cache/lm-studio/models/NexVeridian/Qwen3.5-35B-A3B-3bit"),
         "Qwen3.5-35B-A3B-3bit (MoE)"),
        (os.path.expanduser("~/.cache/lm-studio/models/mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx"),
         "DeepSeek-V2-Lite-4bit (MoE)"),
    ]
    models = [(d, l) for d, l in models if os.path.isdir(d)]

    print("Prefix cache degradation test: TTFT across conversation turns")
    print(f"System prompt: ~{len(SYSTEM_PROMPT.split())} words")
    print(f"Simulated turns: {len(TURNS)} + {len(NEW_QUESTIONS)} new questions")

    for model_dir, label in models:
        bench_model(model_dir, label)


if __name__ == "__main__":
    main()
