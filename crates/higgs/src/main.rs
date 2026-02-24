use std::collections::HashMap;
use std::io::IsTerminal;
use std::net::SocketAddr;
use std::sync::Arc;

use higgs::{
    build_router,
    config::ServerConfig,
    model_download, model_resolver,
    state::{AppState, Engine},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let config = ServerConfig::load()?;

    let mut engines = HashMap::new();
    for model_path in &config.models {
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
                            .map_err(|e| format!("failed to run huggingface-cli: {e}\nInstall with: brew install huggingface-cli"))?;
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
        let engine = if config.batch {
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

    let timeout_secs = config.timeout;
    if !timeout_secs.is_finite() || timeout_secs <= 0.0 {
        return Err("timeout must be a positive, finite number".into());
    }
    let api_key = config.api_key.clone();
    let rate_limit = config.rate_limit;
    let bind_addr = format!("{}:{}", config.host, config.port);

    let shared_state = Arc::new(AppState { engines, config });

    let app = build_router(shared_state, timeout_secs, api_key, rate_limit);

    tracing::info!(addr = %bind_addr, "Starting server");
    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown_signal())
    .await?;

    Ok(())
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    tracing::info!("Shutdown signal received, draining connections");
}
