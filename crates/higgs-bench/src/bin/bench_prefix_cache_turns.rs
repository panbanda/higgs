#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::indexing_slicing,
    clippy::shadow_unrelated,
    clippy::shadow_reuse,
    clippy::shadow_same,
    clippy::too_many_lines
)]
//! `bench_prefix_cache_turns` — does prefix-cache speedup degrade as the
//! conversation grows? Port of `benchmarks/bench_prefix_cache_turns.py`.

use std::process::ExitCode;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::Parser;
use higgs_bench::{
    BenchOutput, ModelInfo, OutputFormat, RunMetadata, default_manifest_path, format_json,
    format_markdown, http, models, persist_result, process, server,
};
use serde::Serialize;

const SYSTEM_PROMPT: &str = include_str!("../../assets/prefix_cache_turns_system_prompt.txt");

const TURNS: &[(&str, &str)] = &[
    (
        "What database would you recommend for a time-series IoT platform?",
        "For a time-series IoT platform, I'd recommend TimescaleDB as the primary choice. It extends PostgreSQL with automatic partitioning by time, native compression achieving 90%+ reduction, and continuous aggregates for real-time rollups. Key advantages: you keep full SQL compatibility, can JOIN with relational data, and benefit from PostgreSQL's ecosystem. For extreme scale (millions of writes/sec), consider InfluxDB or QuestDB which sacrifice SQL flexibility for raw ingestion speed. The trade-off is clear: TimescaleDB gives you versatility, while purpose-built TSDBs give you peak throughput.",
    ),
    (
        "How should I handle the ingestion pipeline?",
        "For IoT ingestion, implement a buffered pipeline: devices publish to MQTT (Mosquitto/EMQX), a bridge forwards to Kafka for durability and backpressure, then Kafka consumers batch-insert into TimescaleDB. Key design decisions: use Kafka partitioning by device_id for ordering guarantees, set batch sizes of 1000-5000 rows for optimal insert throughput, and implement dead-letter queues for malformed data. Add schema validation at the MQTT bridge level to reject bad payloads early. For backpressure, Kafka naturally handles this - consumers process at their own pace while producers buffer in topics.",
    ),
    (
        "What about real-time alerting on the data?",
        "Layer your alerting: use TimescaleDB continuous aggregates for threshold-based alerts (e.g., avg temperature > X over 5min windows), and Kafka Streams or Flink for complex event processing (e.g., detecting anomaly patterns across multiple sensors). For the alert pipeline: Kafka topic for raw events, stream processor evaluates rules, alert events go to a separate topic, then a notification service dispatches via PagerDuty/Slack/email. Implement alert deduplication and suppression to avoid alert fatigue. Store alert history in PostgreSQL for audit trails. Use Grafana with TimescaleDB datasource for visualization dashboards.",
    ),
    (
        "How do I scale this to millions of devices?",
        "Scaling to millions of devices requires horizontal scaling at every layer. MQTT: use EMQX cluster with shared subscriptions, each node handles ~500K concurrent connections. Kafka: partition by device_id hash, scale consumers with consumer groups. TimescaleDB: use distributed hypertables across multiple nodes, partition by both time and device_id. Add a device registry service for metadata. Implement connection pooling with PgBouncer. For cost optimization, tier your storage: hot data (recent 7 days) on SSD-backed TimescaleDB, warm data (30 days) on cheaper storage, cold data archived to S3/Parquet for analytics. Use read replicas for dashboard queries to isolate from write path.",
    ),
    (
        "What monitoring should I set up for this infrastructure?",
        "Implement observability across all layers using the RED/USE framework. Infrastructure: Prometheus with node_exporter for CPU/memory/disk, kube-state-metrics for Kubernetes. MQTT: monitor connected clients, message rate, subscription count. Kafka: consumer lag (critical - use Burrow), broker throughput, partition skew. TimescaleDB: query latency p95/p99, connection pool utilization, chunk compression ratio, replication lag. Application: request rate, error rate, latency histograms per endpoint. Create four dashboards: system health overview, ingestion pipeline throughput, database performance, and alert system health. Set SLOs: 99.9% ingestion success rate, p99 query latency < 500ms, alert delivery within 60 seconds.",
    ),
];

const NEW_QUESTIONS: [&str; 2] = [
    "Now, what if I need to add machine learning predictions on the incoming data?",
    "Should I use GraphQL or REST for the device management API?",
];

#[derive(Debug, Parser)]
#[command(
    name = "bench_prefix_cache_turns",
    about = "Measure prefix-cache TTFT degradation across conversation turns",
    version
)]
struct Args {
    #[arg(long)]
    models: Option<String>,

    #[arg(long)]
    tag: Option<String>,

    #[arg(long)]
    manifest: Option<std::path::PathBuf>,

    #[arg(long, default_value_t = 8080)]
    port: u16,

    #[arg(long)]
    no_spawn: bool,

    #[arg(long, default_value_t = 60)]
    max_tokens: u32,

    #[arg(long, default_value_t = 120)]
    server_timeout_s: u64,

    #[arg(long, value_enum, default_value_t = OutputFormat::Json)]
    format: OutputFormat,
}

#[derive(Debug, Serialize)]
struct Params {
    port: u16,
    max_tokens: u32,
    no_spawn: bool,
    num_turns: usize,
    num_new_questions: usize,
    model_keys: Vec<String>,
}

#[derive(Debug, Serialize, Clone)]
struct TurnRecord {
    turn: String,
    est_context_tokens: u32,
    ttft_ms: f64,
    speedup_vs_miss: f64,
    question: String,
}

#[derive(Debug, Serialize, Clone)]
struct ModelSummary {
    model: String,
    miss_ttft_ms: f64,
    turns: Vec<TurnRecord>,
    new_questions: Vec<TurnRecord>,
}

#[derive(Debug, Serialize)]
struct Results {
    per_model: Vec<ModelSummary>,
}

fn main() -> ExitCode {
    let args = Args::parse();
    let runtime = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("error: failed to start tokio runtime: {e}");
            return ExitCode::from(2);
        }
    };
    match runtime.block_on(run(args)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::from(1)
        }
    }
}

async fn run(args: Args) -> Result<()> {
    let mut metadata = RunMetadata::capture("bench_prefix_cache_turns");
    let started = Instant::now();

    let manifest_path = args.manifest.clone().unwrap_or_else(default_manifest_path);
    let manifest = models::load_manifest(&manifest_path)?;
    let selected = select_models(&manifest, &args)?;
    if args.no_spawn && selected.len() != 1 {
        anyhow::bail!("--no-spawn requires exactly one model");
    }
    if let Some(first) = selected.first() {
        metadata.model = Some(ModelInfo {
            key: first.key.clone(),
            path: first.path.clone(),
            quantization: first.quantization.clone(),
            approx_size_gb: first.approx_size_gb,
        });
    }

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(600))
        .build()?;
    let base_url = format!("http://127.0.0.1:{}", args.port);

    let mut per_model = Vec::new();
    for model in &selected {
        let summary = bench_one_model(&client, &base_url, model, &args)
            .await
            .with_context(|| format!("benchmarking model '{}'", model.key))?;
        per_model.push(summary);
    }

    metadata.duration_ms = started.elapsed().as_millis() as u64;
    let params = Params {
        port: args.port,
        max_tokens: args.max_tokens,
        no_spawn: args.no_spawn,
        num_turns: TURNS.len(),
        num_new_questions: NEW_QUESTIONS.len(),
        model_keys: selected.iter().map(|m| m.key.clone()).collect(),
    };

    // Persist one BenchOutput per model so `bench_summarize` (which
    // groups by `metadata.model.key`) attributes each summary correctly.
    // The aggregate Results { per_model } is still rendered to stdout
    // for the human-facing report.
    let by_key: std::collections::HashMap<String, &models::Model> =
        selected.iter().map(|m| (m.key.clone(), m)).collect();
    for summary in &per_model {
        let Some(model) = by_key.get(&summary.model) else {
            continue;
        };
        let mut model_meta = metadata.clone();
        model_meta.model = Some(ModelInfo {
            key: model.key.clone(),
            path: model.path.clone(),
            quantization: model.quantization.clone(),
            approx_size_gb: model.approx_size_gb,
        });
        let single = BenchOutput {
            metadata: model_meta,
            params: &params,
            results: summary,
        };
        match persist_result(&single) {
            Ok(path) => eprintln!("[persisted] {}", path.display()),
            Err(e) => eprintln!("warning: persist {}: {e:#}", model.key),
        }
    }

    let results = Results { per_model };
    let output = BenchOutput {
        metadata,
        params,
        results,
    };
    let rendered = match args.format {
        OutputFormat::Json => format_json(&output)?,
        OutputFormat::Markdown => format_markdown(&output)?,
    };
    println!("{rendered}");
    Ok(())
}

#[allow(clippy::option_if_let_else)]
fn select_models(manifest: &models::Manifest, args: &Args) -> Result<Vec<models::Model>> {
    if let Some(list) = &args.models {
        list.split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|key| {
                manifest
                    .find_by_key(key)
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("model key '{key}' not in manifest"))
            })
            .collect()
    } else if let Some(tag) = &args.tag {
        Ok(manifest.find_by_tag(tag).into_iter().cloned().collect())
    } else {
        Ok(manifest.find_by_tag("large").into_iter().cloned().collect())
    }
}

async fn bench_one_model(
    client: &reqwest::Client,
    base_url: &str,
    model: &models::Model,
    args: &Args,
) -> Result<ModelSummary> {
    eprintln!("=== model: {} ===", model.label);

    let child = if args.no_spawn {
        None
    } else {
        Some(
            process::start_higgs_server(
                &model.path,
                args.port,
                &[],
                Duration::from_secs(args.server_timeout_s),
            )
            .await?,
        )
    };

    let summary = run_turns(client, base_url, model, args).await;

    if let Some(c) = child {
        if let Err(e) = process::stop_server(c).await {
            eprintln!("warning: stop_server: {e:#}");
        }
    }

    summary
}

async fn run_turns(
    client: &reqwest::Client,
    base_url: &str,
    model: &models::Model,
    args: &Args,
) -> Result<ModelSummary> {
    if args.no_spawn {
        server::wait_until_ready(base_url, Duration::from_secs(30)).await?;
    }
    let model_id = http::first_model_id(client, base_url).await?;

    // Warmup
    let warmup = serde_json::json!([
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hi."},
    ]);
    let _ = http::stream_chat(client, base_url, &model_id, &warmup, 5, 0.0).await?;

    let mut messages: Vec<serde_json::Value> =
        vec![serde_json::json!({"role": "system", "content": SYSTEM_PROMPT})];
    let mut turn_records = Vec::new();
    let mut miss_ttft = 0.0_f64;

    for (idx, (question, fake_answer)) in TURNS.iter().enumerate() {
        messages.push(serde_json::json!({"role": "user", "content": question}));
        let est_tokens = estimate_tokens(&messages);
        let value = serde_json::Value::Array(messages.clone());
        let r =
            http::stream_chat(client, base_url, &model_id, &value, args.max_tokens, 0.0).await?;
        if idx == 0 {
            miss_ttft = r.ttft_ms;
        }
        let speedup = if r.ttft_ms > 0.0 {
            miss_ttft / r.ttft_ms
        } else {
            0.0
        };
        let tag = if idx == 0 { "MISS" } else { "HIT" };
        eprintln!(
            "  [turn {}] {tag} ctx={est_tokens} ttft={:.0}ms speedup={:.2}x",
            idx + 1,
            r.ttft_ms,
            speedup
        );
        turn_records.push(TurnRecord {
            turn: format!("{}", idx + 1),
            est_context_tokens: est_tokens,
            ttft_ms: r.ttft_ms,
            speedup_vs_miss: speedup,
            question: (*question).to_owned(),
        });
        messages.push(serde_json::json!({"role": "assistant", "content": fake_answer}));
    }

    let mut new_records = Vec::new();
    for q in NEW_QUESTIONS {
        let mut copy = messages.clone();
        copy.push(serde_json::json!({"role": "user", "content": q}));
        let est_tokens = estimate_tokens(&copy);
        let value = serde_json::Value::Array(copy);
        let r =
            http::stream_chat(client, base_url, &model_id, &value, args.max_tokens, 0.0).await?;
        let speedup = if r.ttft_ms > 0.0 {
            miss_ttft / r.ttft_ms
        } else {
            0.0
        };
        eprintln!(
            "  [new] ctx={est_tokens} ttft={:.0}ms speedup={:.2}x",
            r.ttft_ms, speedup
        );
        new_records.push(TurnRecord {
            turn: "new".to_owned(),
            est_context_tokens: est_tokens,
            ttft_ms: r.ttft_ms,
            speedup_vs_miss: speedup,
            question: q.to_owned(),
        });
    }

    Ok(ModelSummary {
        model: model.label.clone(),
        miss_ttft_ms: miss_ttft,
        turns: turn_records,
        new_questions: new_records,
    })
}

fn estimate_tokens(messages: &[serde_json::Value]) -> u32 {
    let total_words: usize = messages
        .iter()
        .filter_map(|m| m.get("content").and_then(|c| c.as_str()))
        .map(|s| s.split_whitespace().count())
        .sum();
    ((total_words as f64) * 1.3) as u32
}
