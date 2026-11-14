//! `higgs bench` and `higgs selftest`: start a private server for one model, drive it over HTTP
//! with the client code in `higgs-bench`, and report.
//!
//! The server is a child process running this same executable's `serve`, so the measurements
//! cover exactly what a user's server does (HTTP layer, router, engine) and the process's peak
//! resident memory is the model's alone.

#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::error::Error;
use std::path::Path;
use std::time::Duration;

use higgs_bench::client::{Endpoint, EndpointSettings};
use higgs_bench::local_server::{LocalServer, ServerCommand, free_port};
use higgs_bench::rss::RssSampler;
use higgs_bench::runner::{Plan, Results, render_table, run_scenarios};
use higgs_bench::selftest::{
    CheckResult, ModelFacts, has_failures, render_result, render_summary, run_selftest,
};
use higgs_bench::{
    BENCH_SCHEMA_VERSION, BenchOutput, HostInfo, ModelInfo, RunMetadata, compare, format_json,
    public_model_ref,
};
use serde::Serialize;

use crate::config::{
    BenchArgs, Cli, HiggsConfig, LoadMode, ModelConfig, ModelTargetArgs, Secret, SelftestArgs,
};
use crate::engine_loader::served_name;

/// The environment variable that gives the private server the key its client will send.
pub const PRIVATE_API_KEY_VARIABLE: &str = "HIGGS_PRIVATE_SERVER_KEY";

const REQUEST_TIMEOUT: Duration = Duration::from_secs(900);

type CommandResult<T> = Result<T, Box<dyn Error>>;

/// A private server and a client for the one model it serves.
struct Session {
    // Dropping the server stops it, so it stays alive for as long as the session does.
    server: LocalServer,
    endpoint: Endpoint,
    port: u16,
}

/// The server's command line: the file the user chose, or the model given directly. The server
/// keeps only the target model (see [`isolate_for_measurement`]).
fn server_arguments(cli: &Cli, target: &ModelTargetArgs, port: u16) -> Vec<String> {
    let mut arguments = Vec::new();
    if let Some(path) = &cli.config {
        arguments.extend(["--config".to_owned(), path.display().to_string()]);
    }
    if let Some(profile) = &cli.profile {
        arguments.extend(["--profile".to_owned(), profile.clone()]);
    }
    arguments.extend([
        "serve".to_owned(),
        "--host".to_owned(),
        "127.0.0.1".to_owned(),
        "--port".to_owned(),
        port.to_string(),
        "--private-model".to_owned(),
        target.model.clone(),
    ]);
    if cli.config.is_none() && cli.profile.is_none() {
        arguments.extend(["--model".to_owned(), target.model.clone()]);
    }
    arguments
}

/// Whether `model` is the one `selector` names: its served name, an alias or its path.
fn is_selected(model: &ModelConfig, selector: &str) -> bool {
    model.name.as_deref() == Some(selector)
        || model.aliases.iter().any(|alias| alias == selector)
        || model.path == selector
        || served_name(model) == selector
}

/// Restricts a loaded configuration to the one model `selector` names, for a server that exists
/// only to be measured.
///
/// The model loads at startup and never expires, and nothing else in the file is started:
/// other models, routes, providers and tool servers. The rate limit is lifted, the metrics log
/// (which belongs to the user's own server) is not written, and the API key becomes `api_key`,
/// the key the client sends.
pub fn isolate_for_measurement(
    config: &mut HiggsConfig,
    selector: &str,
    api_key: String,
) -> Result<(), String> {
    let Some(position) = config
        .models
        .iter()
        .position(|model| is_selected(model, selector))
    else {
        let available: Vec<String> = config.models.iter().map(served_name).collect();
        return Err(format!(
            "no configured model matches '{selector}'; configured models: {}",
            available.join(", ")
        ));
    };
    let mut model = config.models.swap_remove(position);
    model.load = LoadMode::Startup;
    model.idle_ttl = None;
    model.pinned = true;
    config.models = vec![model];
    config.providers.clear();
    config.routes.clear();
    config.routing = crate::config::RoutingConfig::default();
    config.mcp.servers.clear();
    config.observability.metrics_log.enabled = false;
    config.server.rate_limit = 0;
    config.server.api_key = Some(Secret::new(api_key));
    Ok(())
}

/// The id to request: the only model the private server lists.
fn request_model(listed: &[String]) -> CommandResult<String> {
    match listed {
        [only] => Ok(only.clone()),
        _ => Err(format!("expected the server to list one model, found {listed:?}").into()),
    }
}

async fn start_session(
    cli: &Cli,
    target: &ModelTargetArgs,
    extra_body: serde_json::Map<String, serde_json::Value>,
) -> CommandResult<Session> {
    let port = match target.port {
        Some(port) => port,
        None => free_port()?,
    };
    let api_key = uuid::Uuid::new_v4().to_string();
    let command = ServerCommand {
        executable: std::env::current_exe()?,
        arguments: server_arguments(cli, target, port),
        environment: vec![(PRIVATE_API_KEY_VARIABLE.to_owned(), api_key.clone())],
    };
    eprintln!(
        "[server] loading {} on port {port} (up to {}s)",
        target.model, target.load_timeout
    );
    let mut server = LocalServer::spawn(&command, port)?;
    server
        .wait_ready(Duration::from_secs(target.load_timeout))
        .await?;

    let endpoint = Endpoint::new(EndpointSettings {
        base_url: server.base_url().to_owned(),
        model: String::new(),
        api_key: Some(api_key),
        temperature: 0.0,
        extra_body,
        timeout: REQUEST_TIMEOUT,
    })?;
    let listed = endpoint.list_model_ids().await?;
    let model = request_model(&listed)?;
    eprintln!("[server] ready, serving {model}");
    Ok(Session {
        server,
        endpoint: endpoint.with_model(model),
        port,
    })
}

fn plan_from(args: &BenchArgs) -> Plan {
    let quick = Plan::quick();
    Plan {
        scenarios: if args.scenarios.is_empty() {
            quick.scenarios
        } else {
            args.scenarios.clone()
        },
        trials: args.trials.unwrap_or(quick.trials),
        warmup: args.warmup.unwrap_or(quick.warmup),
        max_tokens: args.max_tokens.unwrap_or(quick.max_tokens),
        prefill_tokens: if args.prefill_tokens.is_empty() {
            quick.prefill_tokens
        } else {
            args.prefill_tokens.clone()
        },
        prefix_cache_tokens: args
            .prefix_cache_tokens
            .unwrap_or(quick.prefix_cache_tokens),
        concurrency: if args.concurrency.is_empty() {
            quick.concurrency
        } else {
            args.concurrency.clone()
        },
        concurrency_rounds: quick.concurrency_rounds,
    }
}

#[derive(Serialize)]
struct BenchParams<'a> {
    model: &'a str,
    port: u16,
    plan: &'a Plan,
}

fn header(model: &str) -> String {
    let host = HostInfo::capture();
    format!(
        "higgs bench: {model} (higgs {}, {}, {:.0} GB)",
        env!("CARGO_PKG_VERSION"),
        host.cpu,
        host.ram_gb
    )
}

fn write_json(
    path: &Path,
    model: &str,
    port: u16,
    plan: &Plan,
    results: Results,
    started: std::time::Instant,
) -> CommandResult<()> {
    let mut metadata = RunMetadata::capture("higgs_bench");
    metadata.higgs_version = Some(env!("CARGO_PKG_VERSION").to_owned());
    metadata.duration_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
    let key = compare::sanitize_model_key(&public_model_ref(model, ""));
    metadata.model = Some(ModelInfo {
        key: key.clone(),
        path: key,
        quantization: "n/a".to_owned(),
        approx_size_gb: 0.0,
    });
    let output = BenchOutput {
        schema_version: BENCH_SCHEMA_VERSION,
        metadata,
        params: BenchParams { model, port, plan },
        results,
    };
    std::fs::write(path, format!("{}\n", format_json(&output)?))?;
    eprintln!("[json] wrote {}", path.display());
    Ok(())
}

/// Runs `higgs bench`.
pub async fn run_bench(cli: &Cli, args: &BenchArgs) -> CommandResult<()> {
    let plan = plan_from(args);
    plan.validate()?;
    let started = std::time::Instant::now();
    // Without reasoning, decode timing measures generating tokens rather than reaching an answer.
    let extra_body = serde_json::Map::from_iter([(
        "reasoning".to_owned(),
        serde_json::json!({ "effort": "none" }),
    )]);
    let session = start_session(cli, &args.target, extra_body).await?;
    let sampler = RssSampler::start(session.server.pid())?;
    let results = run_scenarios(&session.endpoint, &plan, Some(sampler), &|line| {
        eprintln!("{line}");
    })
    .await?;

    println!("{}", header(session.endpoint.model()));
    print!("{}", render_table(&results));
    if let Some(path) = &args.json {
        write_json(
            path,
            session.endpoint.model(),
            session.port,
            &plan,
            results,
            started,
        )?;
    }
    Ok(())
}

/// Runs `higgs selftest`; `true` when no check failed.
pub async fn run_selftest_command(cli: &Cli, args: &SelftestArgs) -> CommandResult<bool> {
    let session = start_session(cli, &args.target, serde_json::Map::new()).await?;
    let facts = ModelFacts::discover(&session.endpoint, session.endpoint.model()).await;
    println!("higgs selftest: {}", session.endpoint.model());
    let results: Vec<CheckResult> = run_selftest(&session.endpoint, &facts, &|result| {
        println!("{}", render_result(result));
    })
    .await;
    println!("{}", render_summary(&results));
    Ok(!has_failures(&results))
}

#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::*;
    use crate::config::Commands;
    use higgs_bench::runner::Scenario;

    fn parse(arguments: &[&str]) -> Cli {
        Cli::parse_from(arguments)
    }

    fn target_of(cli: &Cli) -> &ModelTargetArgs {
        match &cli.command {
            Commands::Bench(args) => &args.target,
            Commands::Selftest(args) => &args.target,
            Commands::Serve(_)
            | Commands::Start(_)
            | Commands::Stop(_)
            | Commands::Attach
            | Commands::Ui
            | Commands::Init
            | Commands::Shellenv
            | Commands::Exec { .. }
            | Commands::Config { .. }
            | Commands::Doctor(_) => panic!("not a bench command"),
        }
    }

    #[test]
    fn a_direct_model_starts_a_simple_mode_server_on_loopback() {
        let cli = parse(&["higgs", "bench", "/models/m", "--port", "8871"]);
        let arguments = server_arguments(&cli, target_of(&cli), 8871);
        assert_eq!(
            arguments,
            [
                "serve",
                "--host",
                "127.0.0.1",
                "--port",
                "8871",
                "--private-model",
                "/models/m",
                "--model",
                "/models/m"
            ]
        );
    }

    #[test]
    fn a_config_file_is_forwarded_instead_of_the_model_flag() {
        let cli = parse(&["higgs", "-c", "/tmp/c.toml", "selftest", "thinklite"]);
        let arguments = server_arguments(&cli, target_of(&cli), 9);
        assert_eq!(
            arguments,
            [
                "--config",
                "/tmp/c.toml",
                "serve",
                "--host",
                "127.0.0.1",
                "--port",
                "9",
                "--private-model",
                "thinklite"
            ]
        );
    }

    #[test]
    fn a_profile_is_forwarded() {
        let cli = parse(&["higgs", "--profile", "work", "bench", "m"]);
        let arguments = server_arguments(&cli, target_of(&cli), 9);
        assert_eq!(arguments.first().map(String::as_str), Some("--profile"));
        assert_eq!(arguments.get(1).map(String::as_str), Some("work"));
        assert!(!arguments.contains(&"--model".to_owned()));
    }

    #[test]
    fn the_private_server_must_list_exactly_one_model() {
        assert_eq!(request_model(&["served".to_owned()]).unwrap(), "served");
        assert!(request_model(&[]).is_err());
        assert!(request_model(&["a".to_owned(), "b".to_owned()]).is_err());
    }

    fn two_model_config() -> HiggsConfig {
        let mut first = ModelConfig::with_path("/models/first");
        first.name = Some("first".to_owned());
        first.aliases = vec!["default".to_owned()];
        first.load = LoadMode::OnDemand;
        first.idle_ttl = Some(Duration::from_secs(60));
        let mut second = ModelConfig::with_path("/models/second");
        second.name = Some("second".to_owned());
        let mut config = HiggsConfig {
            models: vec![first, second],
            ..HiggsConfig::default()
        };
        config.server.rate_limit = 5;
        config.server.api_key = Some(Secret::new("configured".to_owned()));
        config.routes.push(crate::config::RouteConfig {
            name: None,
            description: None,
            pattern: Some("x".to_owned()),
            provider: "remote".to_owned(),
            model: None,
        });
        config
    }

    #[test]
    fn isolation_keeps_only_the_selected_model_and_loads_it_at_startup() {
        let mut config = two_model_config();
        isolate_for_measurement(&mut config, "first", "private".to_owned()).unwrap();
        let names: Vec<String> = config.models.iter().map(served_name).collect();
        assert_eq!(names, ["first"]);
        let model = &config.models[0];
        assert_eq!(model.load, LoadMode::Startup);
        assert_eq!(model.idle_ttl, None);
        assert!(config.routes.is_empty());
        assert!(config.providers.is_empty());
    }

    #[test]
    fn isolation_resolves_an_alias_and_a_path_to_the_same_model() {
        for selector in ["default", "/models/first", "first"] {
            let mut config = two_model_config();
            isolate_for_measurement(&mut config, selector, "private".to_owned()).unwrap();
            assert_eq!(served_name(&config.models[0]), "first", "{selector}");
        }
    }

    #[test]
    fn isolation_replaces_the_key_and_lifts_the_rate_limit() {
        let mut config = two_model_config();
        isolate_for_measurement(&mut config, "second", "private".to_owned()).unwrap();
        let key = config.server.api_key.as_ref().map(Secret::expose);
        assert_eq!(key, Some("private"));
        assert_eq!(config.server.rate_limit, 0);
        assert!(!config.observability.metrics_log.enabled);
    }

    #[test]
    fn isolation_names_the_configured_models_when_nothing_matches() {
        let mut config = two_model_config();
        let message = isolate_for_measurement(&mut config, "third", "k".to_owned()).unwrap_err();
        assert!(message.contains("'third'"), "{message}");
        assert!(message.contains("first, second"), "{message}");
    }

    #[test]
    fn unset_bench_flags_fall_back_to_the_quick_plan() {
        let cli = parse(&["higgs", "bench", "m"]);
        let Commands::Bench(args) = &cli.command else {
            panic!("expected bench");
        };
        let plan = plan_from(args);
        let quick = Plan::quick();
        assert_eq!(plan.trials, quick.trials);
        assert_eq!(plan.scenarios, quick.scenarios);
        assert_eq!(plan.prefill_tokens, quick.prefill_tokens);
        assert_eq!(plan.concurrency, quick.concurrency);
        assert!(plan.validate().is_ok());
    }

    #[test]
    fn bench_flags_override_the_quick_plan() {
        let cli = parse(&[
            "higgs",
            "bench",
            "m",
            "--scenarios",
            "decode,prefix_cache",
            "--trials",
            "5",
            "--max-tokens",
            "64",
            "--prefill-tokens",
            "100,200",
            "--prefix-cache-tokens",
            "1000",
            "--concurrency",
            "2",
            "--json",
            "out.json",
        ]);
        let Commands::Bench(args) = &cli.command else {
            panic!("expected bench");
        };
        let plan = plan_from(args);
        assert_eq!(plan.scenarios, [Scenario::Decode, Scenario::PrefixCache]);
        assert_eq!(plan.trials, 5);
        assert_eq!(plan.max_tokens, 64);
        assert_eq!(plan.prefill_tokens, [100, 200]);
        assert_eq!(plan.prefix_cache_tokens, 1000);
        assert_eq!(plan.concurrency, [2]);
        assert_eq!(args.json.as_deref(), Some(Path::new("out.json")));
    }
}
