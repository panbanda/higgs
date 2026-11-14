//! Command-line interface: flags, subcommands and the `serve` arguments.

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};

use higgs_bench::runner::Scenario;

use super::types::{LoadMode, MemoryGuard, MlxProfile};

#[derive(Parser, Debug)]
#[command(
    name = "higgs",
    author,
    version,
    about = "Unified AI gateway: serve local MLX models and proxy to remote providers"
)]
pub struct Cli {
    /// Path to config file (default: ~/.config/higgs/config.toml when auto-discovered).
    #[arg(
        short,
        long,
        global = true,
        value_name = "FILE",
        conflicts_with = "profile"
    )]
    pub config: Option<PathBuf>,

    /// Named config file (resolves to ~/.config/higgs/config.<NAME>.toml).
    #[arg(long, global = true, value_name = "NAME")]
    pub profile: Option<String>,

    /// Enable debug logging.
    #[arg(short, long, global = true)]
    pub verbose: bool,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Start the server in the foreground.
    Serve(ServeArgs),
    /// Start the server as a background daemon from config or profile.
    Start(StartArgs),
    /// Stop a running daemon.
    Stop(StopArgs),
    /// Open the daemon metrics dashboard.
    Attach,
    /// Open the Higgs desktop app (macOS).
    Ui,
    /// Create a default config file at ~/.config/higgs/config.toml.
    Init,
    /// Print shell environment variables (for eval).
    Shellenv(ClientArgs),
    /// Set shell environment and exec a command.
    Exec {
        #[command(flatten)]
        client: ClientArgs,
        /// Command and arguments to execute.
        #[arg(trailing_var_arg = true, required = true)]
        command: Vec<String>,
    },
    /// Read or modify configuration values.
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },
    /// Validate config, check model paths, and probe providers.
    Doctor(ServeArgs),
    /// Measure decode, prefill, prefix-cache and concurrency speed of one model.
    Bench(BenchArgs),
    /// Check one model for correct output: chat, streaming, prefix cache, JSON schema, tool
    /// calls and concurrency. Exits non-zero when a check fails.
    Selftest(SelftestArgs),
}

/// How `bench` and `selftest` reach the model: they start a private server for it.
#[derive(Args, Debug)]
pub struct ModelTargetArgs {
    /// Model directory or Hugging Face ID. With --config or --profile, the name or alias of a
    /// model in that file, which is then served as configured.
    #[arg(value_name = "MODEL")]
    pub model: String,

    /// Port for the private server (default: any free port).
    #[arg(long)]
    pub port: Option<u16>,

    /// Seconds to wait for the model to load and the server to start.
    #[arg(long, value_name = "SECONDS", default_value_t = 900)]
    pub load_timeout: u64,
}

#[derive(Args, Debug)]
pub struct BenchArgs {
    #[command(flatten)]
    pub target: ModelTargetArgs,

    /// Scenarios to run, comma-separated: decode, prefill, `prefix_cache`, concurrency (default:
    /// all).
    #[arg(long, value_enum, value_delimiter = ',')]
    pub scenarios: Vec<Scenario>,

    /// Measured trials per scenario (default: 3).
    #[arg(long)]
    pub trials: Option<u32>,

    /// Unmeasured warm-up runs before each scenario (default: 1).
    #[arg(long)]
    pub warmup: Option<u32>,

    /// Tokens generated per request in the decode and concurrency scenarios (default: 128).
    #[arg(long)]
    pub max_tokens: Option<u32>,

    /// Prompt sizes in tokens for the prefill scenario, comma-separated (default: 512,2048).
    #[arg(long, value_delimiter = ',')]
    pub prefill_tokens: Vec<usize>,

    /// Size in tokens of the first message of the prefix-cache scenario (default: 2048).
    #[arg(long)]
    pub prefix_cache_tokens: Option<usize>,

    /// Simultaneous request counts for the concurrency scenario, comma-separated (default:
    /// 1,2,4,8).
    #[arg(long, value_delimiter = ',')]
    pub concurrency: Vec<usize>,

    /// Also write the results as JSON to this file.
    #[arg(long, value_name = "PATH")]
    pub json: Option<PathBuf>,
}

#[derive(Args, Debug)]
pub struct SelftestArgs {
    #[command(flatten)]
    pub target: ModelTargetArgs,
}

/// How `shellenv` and `exec` find the running server.
///
/// Without these flags the server address comes from the config file when one exists,
/// otherwise from the default `higgs serve` address.
#[derive(Args, Debug, Default)]
pub struct ClientArgs {
    /// Base URL of the running server, for example `http://127.0.0.1:8000`.
    #[arg(long, value_name = "URL", conflicts_with = "port")]
    pub url: Option<String>,

    /// Port of the running server on 127.0.0.1.
    #[arg(long, value_name = "PORT")]
    pub port: Option<u16>,
}

#[derive(Subcommand, Debug)]
pub enum ConfigAction {
    /// Get a configuration value (dot-separated key).
    Get { key: String },
    /// Set a configuration value (dot-separated key).
    Set { key: String, value: String },
    /// Print the resolved config file path.
    Path,
    /// Rewrite a config file that uses deprecated keys in the current schema.
    Migrate {
        /// Rewrite the file in place (keeping a `.bak` copy) instead of printing the result.
        #[arg(long)]
        write: bool,
    },
    /// Print every resolved setting for the server, or for one model, and where each value
    /// comes from.
    Explain {
        /// Model name (or path or alias) to explain; omit for the server-wide settings.
        #[arg(value_name = "MODEL")]
        target: Option<String>,
        /// `serve` flags to apply, so the output shows where a flag overrides the file.
        #[command(flatten)]
        flags: Box<ServeArgs>,
    },
    /// Parse and validate the config file without probing models or the network.
    Check,
}

#[derive(Args, Debug, Default)]
pub struct StartArgs {
    /// Legacy simple-mode model flag. Use `higgs serve --model ...` instead.
    #[arg(long = "model", action = clap::ArgAction::Append, hide = true)]
    pub models: Vec<String>,
    /// Legacy serve flag. Use `higgs serve --host ...` instead.
    #[arg(long, hide = true)]
    pub host: Option<String>,
    /// Legacy serve flag. Use `higgs serve --port ...` instead.
    #[arg(long, hide = true)]
    pub port: Option<u16>,
    /// Legacy serve flag. Use `higgs serve --max-tokens ...` instead.
    #[arg(long, hide = true)]
    pub max_tokens: Option<u32>,
    /// Legacy serve flag. Use `higgs serve --api-key ...` instead.
    #[arg(long, hide = true)]
    pub api_key: Option<String>,
    /// Legacy serve flag. Use `higgs serve --rate-limit ...` instead.
    #[arg(long, hide = true)]
    pub rate_limit: Option<u32>,
    /// Legacy serve flag. Use `higgs serve --timeout ...` instead.
    #[arg(long, hide = true)]
    pub timeout: Option<f64>,
    /// Legacy serve flag. Use `higgs serve --mlx-profile ...` instead.
    #[arg(long, hide = true)]
    pub mlx_profile: Option<MlxProfile>,
    /// Legacy serve flag. Use `higgs serve --batch` instead.
    #[arg(long, hide = true)]
    pub batch: bool,
    /// Legacy serve flag. Use `higgs serve --kv-cache ...` instead.
    #[arg(long, hide = true)]
    pub kv_cache: Option<String>,
    /// Legacy serve flag. Use `higgs serve --kv-bits ...` instead.
    #[arg(long, hide = true)]
    pub kv_bits: Option<u8>,
    /// Legacy serve flag. Use `higgs serve --kv-key-bits ...` instead.
    #[arg(long, hide = true)]
    pub kv_key_bits: Option<u8>,
    /// Legacy serve flag. Use `higgs serve --kv-value-bits ...` instead.
    #[arg(long, hide = true)]
    pub kv_value_bits: Option<u8>,
    /// Legacy serve flag. Use `higgs serve --kv-no-norm-correction` instead.
    #[arg(long, hide = true)]
    pub kv_no_norm_correction: bool,
    /// Legacy serve flag. Use `higgs serve --kv-adaptive-dense-layers ...` instead.
    #[arg(long, hide = true)]
    pub kv_adaptive_dense_layers: Option<u8>,
    /// Legacy serve flag. Use `higgs serve --kv-seed ...` instead.
    #[arg(long, hide = true)]
    pub kv_seed: Option<u64>,
}

impl StartArgs {
    pub const fn uses_serve_flags(&self) -> bool {
        !self.models.is_empty()
            || self.host.is_some()
            || self.port.is_some()
            || self.max_tokens.is_some()
            || self.api_key.is_some()
            || self.rate_limit.is_some()
            || self.timeout.is_some()
            || self.mlx_profile.is_some()
            || self.batch
            || self.kv_cache.is_some()
            || self.kv_bits.is_some()
            || self.kv_key_bits.is_some()
            || self.kv_value_bits.is_some()
            || self.kv_no_norm_correction
            || self.kv_adaptive_dense_layers.is_some()
            || self.kv_seed.is_some()
    }
}

#[derive(Args, Debug, Default)]
pub struct StopArgs {
    /// Send `SIGKILL` if the daemon does not exit after a graceful timeout.
    #[arg(long)]
    pub force: bool,
}

#[derive(Parser, Debug, Default)]
pub struct ServeArgs {
    /// Path to a model directory or `HuggingFace` model ID. May be repeated.
    #[arg(long = "model", action = clap::ArgAction::Append)]
    pub models: Vec<String>,

    /// Serve only this configured model, without daemon bookkeeping. Used by `bench` and
    /// `selftest` for their private server.
    #[arg(long, hide = true, value_name = "MODEL")]
    pub private_model: Option<String>,

    /// Host to bind the server to.
    #[arg(long)]
    pub host: Option<String>,

    /// Port to bind the server to.
    #[arg(long)]
    pub port: Option<u16>,

    /// Default maximum tokens for generation.
    #[arg(long)]
    pub max_tokens: Option<u32>,

    /// API key for authentication (if unset, no auth required).
    #[arg(long)]
    pub api_key: Option<String>,

    /// Rate limit (requests per minute per client, 0 turns it off).
    #[arg(long)]
    pub rate_limit: Option<u32>,

    /// Default request timeout in seconds.
    #[arg(long)]
    pub timeout: Option<f64>,

    /// MLX tuning profile for local models.
    #[arg(long, value_name = "PROFILE")]
    pub mlx_profile: Option<MlxProfile>,

    /// Deprecated, has no effect: models decode concurrent requests together whenever they can.
    #[arg(long)]
    pub batch: bool,

    /// Serve one request at a time per model.
    #[arg(long, conflicts_with = "batch")]
    pub no_batch: bool,

    /// KV cache mode for simple mode models.
    #[arg(long, value_name = "MODE", value_parser = ["off", "turboquant"])]
    pub kv_cache: Option<String>,

    /// Bit width for `TurboQuant` KV caches.
    #[arg(long)]
    pub kv_bits: Option<u8>,

    /// Override key bit width (default: kv-bits - 1).
    #[arg(long)]
    pub kv_key_bits: Option<u8>,

    /// Override value bit width (default: kv-bits).
    #[arg(long)]
    pub kv_value_bits: Option<u8>,

    /// Disable post-quantization norm correction.
    #[arg(long)]
    pub kv_no_norm_correction: bool,

    /// Number of final layers that stay dense (0 = all TQ).
    #[arg(long)]
    pub kv_adaptive_dense_layers: Option<u8>,

    /// Seed used to generate `TurboQuant` rotation/QJL matrices.
    #[arg(long)]
    pub kv_seed: Option<u64>,

    /// When to load models: at startup, or on the first request that uses them.
    #[arg(long, value_enum)]
    pub load: Option<LoadMode>,

    /// Unload a model after this many idle seconds (0 never unloads).
    #[arg(long, value_name = "SECONDS")]
    pub idle_ttl: Option<u64>,

    /// Derive a memory ceiling from total memory by reserving a share for the system.
    #[arg(long, value_enum, value_name = "TIER")]
    pub memory_guard: Option<MemoryGuard>,

    /// Keep at most this many models loaded, evicting the least recently used.
    #[arg(long, value_name = "N")]
    pub max_loaded_models: Option<usize>,

    /// JSON file of MCP servers in the common `{"mcpServers": {...}}` format.
    #[arg(long, value_name = "PATH")]
    pub mcp_config: Option<PathBuf>,
}
