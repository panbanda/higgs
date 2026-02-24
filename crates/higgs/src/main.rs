use std::collections::HashMap;
use std::io::IsTerminal;
use std::net::SocketAddr;
use std::sync::Arc;

use clap::Parser;

use higgs::{
    build_router,
    config::{self, Cli, Commands, ConfigAction, HiggsConfig, ServeArgs},
    model_download, model_resolver,
    router::Router,
    state::{AppState, Engine},
};

#[tokio::main]
#[allow(clippy::print_stderr)]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Serve(ref args) => cmd_serve(&cli, args).await,
        Commands::Start(ref _args) => {
            let config_path = resolve_config_path(&cli)?;
            higgs::daemon::detach(&config_path, cli.verbose);
            Ok(())
        }
        Commands::Stop => {
            higgs::daemon::cmd_stop();
            Ok(())
        }
        Commands::Attach => {
            let config = load_config_for_command(&cli)?;
            higgs::daemon::run_attached(&config);
            Ok(())
        }
        Commands::Init => {
            higgs::daemon::cmd_init();
            Ok(())
        }
        Commands::Shellenv => {
            let config = load_config_for_command(&cli).unwrap_or_default();
            higgs::daemon::cmd_shellenv(&config);
            Ok(())
        }
        Commands::Config { ref action } => {
            cmd_config(&cli, action);
            Ok(())
        }
        Commands::Doctor(ref args) => {
            init_tracing(cli.verbose);
            let config = if let Some(ref path) = cli.config {
                config::load_config_file(path, Some(args))?
            } else {
                let default = config::default_config_path();
                if default.exists() {
                    config::load_config_file(&default, Some(args))?
                } else if !args.models.is_empty() {
                    config::build_simple_config(args)?
                } else {
                    return Err("no config to validate; use --config or 'higgs init'".into());
                }
            };
            let result = higgs::doctor::run_doctor(&config).await;
            if result.failures > 0 {
                std::process::exit(1);
            }
            Ok(())
        }
    }
}

/// Resolve a config file path from CLI args or the default location.
#[allow(clippy::print_stderr)]
fn resolve_config_path(cli: &Cli) -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
    if let Some(ref path) = cli.config {
        return Ok(path.clone());
    }
    let default = config::default_config_path();
    if default.exists() {
        Ok(default)
    } else {
        Err(format!(
            "no config file specified or found at {}\nhint: use 'higgs init' to create one",
            default.display()
        )
        .into())
    }
}

/// Load config from CLI path or default location.
fn load_config_for_command(cli: &Cli) -> Result<HiggsConfig, Box<dyn std::error::Error>> {
    let path = resolve_config_path(cli)?;
    config::load_config_file(&path, None).map_err(Into::into)
}

async fn cmd_serve(cli: &Cli, args: &ServeArgs) -> Result<(), Box<dyn std::error::Error>> {
    init_tracing(cli.verbose);

    let simple_mode = config::is_simple_mode(cli, args);

    // Load config: simple mode (--model) or config file mode (--config)
    let higgs_config = if simple_mode {
        config::build_simple_config(args)?
    } else if let Some(ref path) = cli.config {
        config::load_config_file(path, Some(args))?
    } else if args.models.is_empty() {
        let default_path = config::default_config_path();
        if default_path.exists() {
            tracing::info!(path = %default_path.display(), "Auto-discovered config file");
            config::load_config_file(&default_path, Some(args))?
        } else {
            return Err("no --model or --config provided, and no config file found at ~/.config/higgs/config.toml\n\
                hint: use 'higgs serve --model <model>' or 'higgs init' to create a config".into());
        }
    } else {
        config::build_simple_config(args)?
    };

    // Load all local models and build router
    let engines = load_engines(&higgs_config)?;
    let router = Router::from_config(&higgs_config, engines)?;

    // Validate timeout
    let timeout_secs = higgs_config.server.timeout;
    if !timeout_secs.is_finite() || timeout_secs <= 0.0 {
        return Err("timeout must be a positive, finite number".into());
    }

    let api_key = higgs_config.server.api_key.clone();
    let rate_limit = higgs_config.server.rate_limit;
    let bind_addr = format!("{}:{}", higgs_config.server.host, higgs_config.server.port);

    // Create metrics (config mode only)
    let metrics = if simple_mode {
        None
    } else {
        let m = higgs::daemon::create_metrics(&higgs_config);
        higgs::daemon::spawn_eviction_task(&m);
        Some(m)
    };

    // Create shared state
    let http_client = reqwest::Client::new();
    let shared_state = Arc::new(AppState {
        router,
        config: higgs_config,
        http_client,
        metrics,
    });

    // Build router with middleware
    let app = build_router(shared_state, timeout_secs, api_key, rate_limit);

    // Write PID file for daemon management
    higgs::daemon::write_pid_file();

    // Start server
    tracing::info!(addr = %bind_addr, "Starting server");
    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(higgs::daemon::await_shutdown_signal())
    .await?;

    higgs::daemon::remove_pid_file();
    Ok(())
}

fn load_engines(
    config: &HiggsConfig,
) -> Result<HashMap<String, Arc<Engine>>, Box<dyn std::error::Error>> {
    let mut engines = HashMap::new();

    for model_cfg in &config.models {
        let model_path = &model_cfg.path;
        tracing::info!(model = %model_path, "Resolving model path");
        let resolved = match model_resolver::resolve(model_path) {
            Ok(path) => path,
            Err(resolve_err) if model_resolver::is_hf_model_id(model_path) => {
                tracing::debug!(error = %resolve_err, "model not in cache; attempting download");
                let is_interactive = std::io::stdin().is_terminal();
                model_download::offer_download(
                    model_path,
                    is_interactive,
                    &mut std::io::stderr().lock(),
                    std::io::stdin().lock(),
                    || {
                        let status = std::process::Command::new("huggingface-cli")
                            .args(["download", model_path])
                            .status()
                            .map_err(|e| {
                                format!(
                                    "failed to run huggingface-cli: {e}\nInstall with: brew install huggingface-cli"
                                )
                            })?;
                        if status.success() {
                            Ok(())
                        } else {
                            Err(format!(
                                "huggingface-cli download failed for '{model_path}'"
                            ))
                        }
                    },
                )?;
                model_resolver::resolve(model_path)?
            }
            Err(err) => return Err(err.into()),
        };

        tracing::info!(model = %model_path, resolved = %resolved.display(), "Loading model");
        let engine = if model_cfg.batch {
            Engine::load_batch(&resolved)?
        } else {
            Engine::load_simple(&resolved)?
        };
        let name = engine.model_name().to_owned();
        tracing::info!(model_name = %name, "Model loaded");

        if engines.insert(name.clone(), Arc::new(engine)).is_some() {
            return Err(format!(
                "model name collision: two model paths resolve to the same name '{name}'"
            )
            .into());
        }
    }

    Ok(engines)
}

fn cmd_config(_cli: &Cli, action: &ConfigAction) {
    let config_path = config::default_config_path();
    match action {
        ConfigAction::Get { key } => {
            higgs::cli_config::config_get(&config_path, key);
        }
        ConfigAction::Set { key, value } => {
            higgs::cli_config::config_set(&config_path, key, value);
        }
        ConfigAction::Path => {
            #[allow(clippy::print_stdout)]
            {
                println!("{}", config_path.display());
            }
        }
    }
}

fn init_tracing(verbose: bool) {
    let default_filter = if verbose { "higgs=debug" } else { "info" };
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                default_filter
                    .parse()
                    .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"))
            }),
        )
        .init();
}
