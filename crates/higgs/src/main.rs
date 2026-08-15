use std::collections::HashMap;
use std::fs;
use std::io::IsTerminal;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
#[cfg(target_os = "macos")]
use std::time::SystemTime;

use clap::Parser;
use higgs_engine::mlx_tuning::resolve_runtime_tuning;

use higgs::{
    build_router,
    config::{
        self, Cli, Commands, ConfigAction, HiggsConfig, MetricsLogConfig, ServeArgs, StartArgs,
        StopArgs,
    },
    model_download, model_resolver,
    router::Router,
    state::{AppState, Engine},
};

#[tokio::main]
#[allow(clippy::print_stderr)]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    if let Some(ref name) = cli.profile {
        config::validate_profile_name(name)?;
    }
    let profile = cli.profile.as_deref();

    match cli.command {
        Commands::Serve(ref args) => cmd_serve(&cli, args).await,
        Commands::Start(ref args) => {
            reject_legacy_start_flags(args)?;
            let config_path = resolve_config_path(&cli)?;
            higgs::daemon::detach(&config_path, cli.verbose, profile);
            Ok(())
        }
        Commands::Stop(StopArgs { force }) => {
            let exit_code = higgs::daemon::cmd_stop(profile, force);
            if exit_code != 0 {
                std::process::exit(exit_code);
            }
            Ok(())
        }
        Commands::Attach => {
            let config = load_config_for_command(&cli)?;
            higgs::daemon::run_attached(&config, profile);
            Ok(())
        }
        Commands::Init => {
            higgs::daemon::cmd_init(profile);
            Ok(())
        }
        Commands::Shellenv => {
            let config = load_config_for_command(&cli)?;
            higgs::daemon::cmd_shellenv(&config)?;
            Ok(())
        }
        Commands::Exec { ref command } => {
            let config = load_config_for_command(&cli)?;
            higgs::daemon::cmd_exec(&config, command);
        }
        Commands::Config { ref action } => {
            cmd_config(&cli, action);
            Ok(())
        }
        Commands::Doctor(ref args) => {
            init_tracing(cli.verbose);
            let (config, config_path) = if let Some(ref path) = cli.config {
                (
                    config::load_config_file(path, Some(args))?,
                    Some(path.clone()),
                )
            } else if cli.profile.is_some() {
                let path = resolve_config_path(&cli)?;
                let config = config::load_config_file(&path, Some(args))?;
                (config, Some(path))
            } else {
                let default = config::default_config_path();
                if default.exists() {
                    let config = config::load_config_file(&default, Some(args))?;
                    (config, Some(default))
                } else if !args.models.is_empty() {
                    (config::build_simple_config(args)?, None)
                } else {
                    return Err("no config to validate; use --config or 'higgs init'".into());
                }
            };
            let result = higgs::doctor::run_doctor(&config, config_path.as_deref()).await;
            if result.failures > 0 {
                std::process::exit(1);
            }
            Ok(())
        }
    }
}

fn reject_legacy_start_flags(args: &StartArgs) -> Result<(), Box<dyn std::error::Error>> {
    if args.uses_serve_flags() {
        return Err(
            "higgs start is config/profile-only\nhint: use 'higgs serve' for ad hoc --model/--port/--batch flags"
                .into(),
        );
    }
    Ok(())
}

/// Resolve a config file path from CLI args, profile, or the default location.
#[allow(clippy::print_stderr)]
fn resolve_config_path(cli: &Cli) -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
    if let Some(ref path) = cli.config {
        return Ok(path.clone());
    }
    if let Some(ref name) = cli.profile {
        let path = config::profile_config_path(name);
        if path.exists() {
            return Ok(path);
        }
        return Err(format!(
            "profile config not found at {}\nhint: use 'higgs init --profile {name}' to create one",
            path.display()
        )
        .into());
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

    let profile = cli.profile.as_deref();
    let simple_mode = config::is_simple_mode(cli, args);

    // Load config: simple mode (--model) or config file mode (--config)
    let mut higgs_config = if simple_mode {
        config::build_simple_config(args)?
    } else if let Some(ref path) = cli.config {
        config::load_config_file(path, Some(args))?
    } else if cli.profile.is_some() {
        let path = resolve_config_path(cli)?;
        config::load_config_file(&path, Some(args))?
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

    // Rewrite metrics path for profile isolation if still at default
    if let Some(name) = profile {
        let default_path = config::default_metrics_log_path_for_profile(name);
        let generic_default = MetricsLogConfig::default().path;
        if higgs_config.logging.metrics.path == generic_default {
            higgs_config.logging.metrics.path = default_path;
        }
    }

    ensure_local_runtime_ready(&higgs_config)?;
    if higgs_config.local.raise_wired_limit && higgs_config.models.len() > 1 {
        tracing::warn!(
            model_count = higgs_config.models.len(),
            "MLX wired-limit escalation is enabled with multiple resident local models; unified-memory pressure may spike"
        );
    }

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
    let max_body_size = higgs_config.server.max_body_size;
    let cors_origins = higgs_config.server.cors_origins.clone();
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
    let app = build_router(
        shared_state,
        timeout_secs,
        api_key,
        rate_limit,
        max_body_size,
        cors_origins,
    );

    // Start server
    tracing::info!(addr = %bind_addr, "Starting server");
    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;

    // Write PID file after bind succeeds so it's never stale on bind errors
    higgs::daemon::write_pid_file(profile);

    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(higgs::daemon::await_shutdown_signal())
    .await?;

    higgs::daemon::remove_pid_file(profile);
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
                    || download_via_hf_cli(model_path),
                )?;
                model_resolver::resolve(model_path)?
            }
            Err(err) => return Err(err.into()),
        };

        tracing::info!(model = %model_path, resolved = %resolved.display(), "Loading model");
        if model_cfg.batch && !config::resolved_model_supports_batch(&resolved)? {
            return Err(format!(
                "batch=true is only supported for transformer models (llama, mistral, qwen2, qwen3); {model_path} is not supported"
            )
            .into());
        }
        let kv_cache_config = model_cfg.kv_cache_config();
        let engine = if model_cfg.batch {
            Engine::load_batch(&resolved, kv_cache_config, config.local.raise_wired_limit)?
        } else {
            let tuning =
                resolve_runtime_tuning(&resolved, model_cfg.requested_mlx_profile(&config.local));
            Engine::load_simple(
                &resolved,
                kv_cache_config,
                tuning,
                config.local.raise_wired_limit,
                model_cfg.kv_disk_dir.as_deref().map(std::path::Path::new),
                model_cfg.kv_disk_space_mb.saturating_mul(1024 * 1024),
            )?
        };
        let name = model_cfg
            .name
            .clone()
            .unwrap_or_else(|| engine.model_name().to_owned());
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

fn ensure_local_runtime_ready(config: &HiggsConfig) -> Result<(), Box<dyn std::error::Error>> {
    if config.models.is_empty() {
        return Ok(());
    }
    #[cfg(target_os = "macos")]
    {
        let exe = std::env::current_exe()?;
        let metallib = exe.with_file_name("mlx.metallib");
        if !metallib.exists() {
            try_restore_metallib(&exe, &metallib)?;
        }
        if !metallib.exists() {
            return Err(format!(
                "mlx.metallib not found next to executable at {}\nhint: rebuild Higgs, reinstall the Apple Silicon Homebrew package, or use a release artifact that bundles mlx.metallib",
                metallib.display()
            )
            .into());
        }
    }
    Ok(())
}

fn download_via_hf_cli(model_path: &str) -> Result<(), String> {
    const CMD: &str = "hf";

    let status = std::process::Command::new(CMD)
        .args(["download", model_path])
        .status()
        .map_err(|e| format!("failed to run {CMD}: {e}\nInstall with: brew install {CMD}"))?;

    if status.success() {
        Ok(())
    } else {
        Err(format!("{CMD} download failed for '{model_path}'"))
    }
}

#[cfg(target_os = "macos")]
fn try_restore_metallib(exe: &Path, destination: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let Some(profile_dir) = derive_profile_dir(exe) else {
        return Ok(());
    };
    let Some(source) = newest_metallib_candidate(&profile_dir) else {
        return Ok(());
    };
    fs::copy(&source, destination)?;
    tracing::info!(
        source = %source.display(),
        destination = %destination.display(),
        "restored mlx.metallib next to executable from Cargo build output"
    );
    Ok(())
}

#[cfg(target_os = "macos")]
fn derive_profile_dir(exe: &Path) -> Option<PathBuf> {
    let parent = exe.parent()?;
    if parent.file_name().is_some_and(|name| name == "deps") {
        parent.parent().map(Path::to_path_buf)
    } else {
        Some(parent.to_path_buf())
    }
}

#[cfg(target_os = "macos")]
fn newest_metallib_candidate(profile_dir: &Path) -> Option<PathBuf> {
    let build_dir = profile_dir.join("build");
    let entries = fs::read_dir(build_dir).ok()?;
    let candidates = entries
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name();
            let is_mlx_sys = name
                .to_str()
                .is_some_and(|value| value.starts_with("mlx-sys-"));
            if !is_mlx_sys {
                return None;
            }
            let candidate = entry.path().join("out/build/lib/mlx.metallib");
            candidate.exists().then(|| {
                (
                    fs::metadata(&candidate)
                        .and_then(|meta| meta.modified())
                        .ok(),
                    candidate,
                )
            })
        })
        .collect();
    select_latest_metallib_candidate(candidates)
}

#[cfg(target_os = "macos")]
fn select_latest_metallib_candidate(
    candidates: Vec<(Option<SystemTime>, PathBuf)>,
) -> Option<PathBuf> {
    candidates
        .into_iter()
        .max_by(|(left_time, left_path), (right_time, right_path)| {
            left_time
                .cmp(right_time)
                .then_with(|| left_path.cmp(right_path))
        })
        .map(|(_, path)| path)
}

fn cmd_config(cli: &Cli, action: &ConfigAction) {
    let config_path = cli.config.clone().unwrap_or_else(|| {
        cli.profile
            .as_ref()
            .map_or_else(config::default_config_path, |name| {
                config::profile_config_path(name)
            })
    });
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

#[cfg(all(test, target_os = "macos"))]
#[allow(clippy::panic, clippy::unwrap_used, clippy::tests_outside_test_module)]
mod tests {
    use super::*;

    #[test]
    fn select_latest_metallib_candidate_prefers_newest_timestamp() {
        let older = SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(10);
        let newer = SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(20);
        let first = PathBuf::from("/tmp/mlx-sys-a/out/build/lib/mlx.metallib");
        let second = PathBuf::from("/tmp/mlx-sys-b/out/build/lib/mlx.metallib");

        let selected = select_latest_metallib_candidate(vec![
            (Some(older), first),
            (Some(newer), second.clone()),
        ]);

        assert_eq!(selected, Some(second));
    }

    #[test]
    fn select_latest_metallib_candidate_breaks_none_ties_by_path() {
        let first = PathBuf::from("/tmp/mlx-sys-a/out/build/lib/mlx.metallib");
        let second = PathBuf::from("/tmp/mlx-sys-b/out/build/lib/mlx.metallib");

        let selected =
            select_latest_metallib_candidate(vec![(None, first), (None, second.clone())]);

        assert_eq!(selected, Some(second));
    }
}
