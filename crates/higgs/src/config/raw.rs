//! The on-disk schema: what the TOML file says, before defaults and inheritance are applied.
//!
//! Deprecated spellings never reach these structs; `legacy` rewrites them first. Unknown keys
//! never reach them either; `schema` rejects them first.

use std::collections::BTreeMap;

use higgs_models::turboquant::KvCacheMode;
use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer};

use super::types::{ApiFormat, BatchMode, LoadMode, MemoryGuard, MlxProfile, MtpMode};
use super::units::{ByteSize, HumanDuration};

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct RawConfig {
    pub include: Vec<String>,
    pub server: RawServer,
    pub memory: RawMemory,
    pub defaults: RawModelSettings,
    pub models: Vec<RawModel>,
    pub providers: BTreeMap<String, RawProvider>,
    pub routes: Vec<RawRoute>,
    pub routing: RawRouting,
    pub observability: RawObservability,
    pub experimental: RawExperimental,
    pub mcp: RawMcp,
    pub web: RawWeb,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct RawServer {
    pub host: Option<String>,
    pub port: Option<u16>,
    pub api_key: Option<String>,
    pub request_timeout: Option<HumanDuration>,
    pub max_body_size: Option<ByteSize>,
    pub max_tokens: Option<u32>,
    pub rate_limit: Option<u32>,
    pub cors_origins: Option<Vec<String>>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct RawMemory {
    pub guard: Option<MemoryGuard>,
    pub limit: Option<ByteSize>,
    pub max_loaded_models: Option<usize>,
    pub raise_wired_limit: Option<bool>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct RawKvCache {
    pub mode: Option<KvCacheMode>,
    pub bits: Option<u8>,
    pub key_bits: Option<u8>,
    pub value_bits: Option<u8>,
    pub norm_correction: Option<bool>,
    pub adaptive_dense_layers: Option<u8>,
    pub seed: Option<u64>,
    pub mla_latent: Option<bool>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct RawPrefixCache {
    pub disk_dir: Option<String>,
    pub disk_budget: Option<ByteSize>,
    pub max_entries: Option<u32>,
    pub checkpoint_interval_tokens: Option<u32>,
    pub max_checkpoints_per_entry: Option<u32>,
    pub checkpoint_budget: Option<ByteSize>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct RawPrefill {
    pub yield_tokens: Option<u32>,
}

/// The settings a model entry and `[defaults]` have in common.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct RawModelSettings {
    pub load: Option<LoadMode>,
    pub idle_ttl: Option<HumanDuration>,
    pub batch: Option<BatchMode>,
    pub profile: Option<MlxProfile>,
    pub thinking_budget: Option<u32>,
    pub pinned: Option<bool>,
    pub kv_cache: RawKvCache,
    pub prefix_cache: RawPrefixCache,
    pub prefill: RawPrefill,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RawModel {
    pub path: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub aliases: Vec<String>,
    #[serde(flatten)]
    pub settings: RawModelSettings,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RawProvider {
    pub url: String,
    #[serde(default)]
    pub format: Option<ApiFormat>,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub strip_auth: Option<bool>,
    #[serde(default)]
    pub stub_count_tokens: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RawRoute {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default, rename = "match")]
    pub pattern: Option<String>,
    pub provider: String,
    #[serde(default)]
    pub model: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct RawAutoRouter {
    pub enabled: Option<bool>,
    pub model: Option<String>,
    pub timeout: Option<HumanDuration>,
    pub force: Option<bool>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct RawRouting {
    pub default: Option<String>,
    pub auto: RawAutoRouter,
}

/// How long metrics stay in memory: a duration, `false` to turn the window off, or `true` for
/// the default window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RawRetention {
    Disabled,
    Default,
    Window(HumanDuration),
}

impl<'de> Deserialize<'de> for RawRetention {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct RetentionVisitor;

        impl Visitor<'_> for RetentionVisitor {
            type Value = RawRetention;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a duration such as \"60m\", or true/false")
            }

            fn visit_bool<E: de::Error>(self, value: bool) -> Result<RawRetention, E> {
                Ok(if value {
                    RawRetention::Default
                } else {
                    RawRetention::Disabled
                })
            }

            fn visit_str<E: de::Error>(self, value: &str) -> Result<RawRetention, E> {
                let duration: HumanDuration = value.parse().map_err(E::custom)?;
                Ok(if duration.is_zero() {
                    RawRetention::Disabled
                } else {
                    RawRetention::Window(duration)
                })
            }

            fn visit_u64<E: de::Error>(self, value: u64) -> Result<RawRetention, E> {
                if value == 0 {
                    Ok(RawRetention::Disabled)
                } else {
                    Err(E::custom(format!(
                        "a bare number {value} has no unit; write a string such as \"{value}m\""
                    )))
                }
            }
        }

        deserializer.deserialize_any(RetentionVisitor)
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct RawMetricsLog {
    pub enabled: Option<bool>,
    pub path: Option<String>,
    pub max_size: Option<ByteSize>,
    pub max_files: Option<u32>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct RawObservability {
    pub retention: Option<RawRetention>,
    pub metrics_log: RawMetricsLog,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct RawChunkedPrefill {
    pub threshold: Option<u32>,
    pub chunk: Option<u32>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct RawExperimental {
    pub mtp: Option<MtpMode>,
    pub mtp_draft_tokens: Option<usize>,
    pub mtp_adaptive_draft: Option<bool>,
    pub mtp_prompt_lookup: Option<bool>,
    pub prompt_lookup: Option<bool>,
    pub clear_cache_after_prefill: Option<bool>,
    pub chunked_prefill: RawChunkedPrefill,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct RawMcpServer {
    pub enabled: Option<bool>,
    #[serde(alias = "type")]
    pub transport: Option<String>,
    pub command: Option<String>,
    pub args: Vec<String>,
    pub env: BTreeMap<String, String>,
    pub cwd: Option<String>,
    pub url: Option<String>,
    pub headers: BTreeMap<String, String>,
    pub timeout: Option<HumanDuration>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct RawMcp {
    pub max_result_size: Option<ByteSize>,
    pub servers: BTreeMap<String, RawMcpServer>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct RawWeb {
    pub enabled: Option<bool>,
    pub search_provider: Option<String>,
    pub searxng_url: Option<String>,
    pub allow_private_addresses: Option<bool>,
    pub expose_as_tools: Option<bool>,
    pub timeout: Option<HumanDuration>,
}
