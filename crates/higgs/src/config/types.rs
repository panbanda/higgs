//! Small value types shared by the file schema, the resolved configuration and the CLI.

use std::fmt;

use clap::ValueEnum;
use higgs_engine::mlx_tuning::RequestedMlxProfile;
use serde::{Deserialize, Deserializer, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum, Default)]
#[serde(rename_all = "lowercase")]
pub enum MlxProfile {
    /// Use model-aware defaults (`auto` is `balanced` for small/medium, `throughput` for large/huge).
    #[default]
    Auto,
    /// Alias for legacy env `ttft`; bias for lower TTFT and decoding startup.
    Latency,
    /// Balanced tuning target for mixed latency/throughput behavior.
    Balanced,
    /// Throughput-first tuning target.
    Throughput,
}

impl MlxProfile {
    /// Canonical profile string used for rendering and logs.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Latency => "latency",
            Self::Balanced => "balanced",
            Self::Throughput => "throughput",
        }
    }

    /// Convert the CLI/TOML profile into the engine's requested profile.
    pub const fn to_requested(self) -> RequestedMlxProfile {
        match self {
            Self::Auto => RequestedMlxProfile::Auto,
            Self::Latency => RequestedMlxProfile::Latency,
            Self::Balanced => RequestedMlxProfile::Balanced,
            Self::Throughput => RequestedMlxProfile::Throughput,
        }
    }
}

/// When a local model is loaded into memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum, Default)]
#[serde(rename_all = "snake_case")]
pub enum LoadMode {
    /// Load when the server starts and keep resident until unloaded.
    #[default]
    Startup,
    /// Load on the first request that routes to the model.
    OnDemand,
}

impl LoadMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Startup => "startup",
            Self::OnDemand => "on_demand",
        }
    }
}

/// Memory ceiling tier: how much unified memory is left to the rest of the system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum, Default)]
#[serde(rename_all = "lowercase")]
pub enum MemoryGuard {
    /// No derived ceiling.
    #[default]
    Off,
    /// Reserve 20% of unified memory.
    Safe,
    /// Reserve 8% of unified memory.
    Balanced,
    /// Reserve 2% of unified memory.
    Aggressive,
}

impl MemoryGuard {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Safe => "safe",
            Self::Balanced => "balanced",
            Self::Aggressive => "aggressive",
        }
    }

    pub const fn reserved_percent(self) -> u64 {
        match self {
            Self::Off => 0,
            Self::Safe => 20,
            Self::Balanced => 8,
            Self::Aggressive => 2,
        }
    }
}

/// The per-model `batch` setting: `true`, `false`, or `"auto"`.
///
/// Every model is served by one engine that decodes concurrent requests together
/// whenever the model can, so only `false` changes anything: it limits the model to
/// one request at a time. `true` and `"auto"` are accepted for old configuration files.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BatchMode {
    /// No limit on concurrent requests.
    #[default]
    Auto,
    /// Same as [`Self::Auto`]; `higgs doctor` says it has no effect.
    On,
    /// One request at a time.
    Off,
}

impl BatchMode {
    /// The mode `higgs serve --batch` / `--no-batch` selects.
    pub const fn from_flags(batch: bool, no_batch: bool) -> Self {
        match (batch, no_batch) {
            (true, _) => Self::On,
            (false, true) => Self::Off,
            (false, false) => Self::Auto,
        }
    }
}

impl fmt::Display for BatchMode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Auto => "auto",
            Self::On => "true",
            Self::Off => "false",
        })
    }
}

impl Serialize for BatchMode {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Auto => serializer.serialize_str("auto"),
            Self::On => serializer.serialize_bool(true),
            Self::Off => serializer.serialize_bool(false),
        }
    }
}

impl<'de> Deserialize<'de> for BatchMode {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct BatchModeVisitor;

        impl serde::de::Visitor<'_> for BatchModeVisitor {
            type Value = BatchMode;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("true, false, or \"auto\"")
            }

            fn visit_bool<E: serde::de::Error>(self, value: bool) -> Result<BatchMode, E> {
                Ok(if value { BatchMode::On } else { BatchMode::Off })
            }

            fn visit_str<E: serde::de::Error>(self, value: &str) -> Result<BatchMode, E> {
                match value.trim().to_ascii_lowercase().as_str() {
                    "auto" => Ok(BatchMode::Auto),
                    "true" => Ok(BatchMode::On),
                    "false" => Ok(BatchMode::Off),
                    _ => Err(E::invalid_value(serde::de::Unexpected::Str(value), &self)),
                }
            }
        }

        deserializer.deserialize_any(BatchModeVisitor)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ApiFormat {
    #[serde(rename = "openai")]
    OpenAi,
    Anthropic,
}

impl ApiFormat {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OpenAi => "openai",
            Self::Anthropic => "anthropic",
        }
    }
}

/// Whether the multi-token-prediction head speeds up decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MtpMode {
    /// The MLX profile decides.
    #[default]
    Auto,
    On,
    Off,
}

impl MtpMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::On => "on",
            Self::Off => "off",
        }
    }

    /// `Some(true)` / `Some(false)` when forced, `None` when the profile decides.
    pub const fn forced(self) -> Option<bool> {
        match self {
            Self::Auto => None,
            Self::On => Some(true),
            Self::Off => Some(false),
        }
    }
}

impl<'de> Deserialize<'de> for MtpMode {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct MtpVisitor;

        impl serde::de::Visitor<'_> for MtpVisitor {
            type Value = MtpMode;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("\"auto\", \"on\" or \"off\"")
            }

            fn visit_bool<E: serde::de::Error>(self, value: bool) -> Result<MtpMode, E> {
                Ok(if value { MtpMode::On } else { MtpMode::Off })
            }

            fn visit_str<E: serde::de::Error>(self, value: &str) -> Result<MtpMode, E> {
                match value.trim().to_ascii_lowercase().as_str() {
                    "auto" => Ok(MtpMode::Auto),
                    "on" | "true" | "1" => Ok(MtpMode::On),
                    "off" | "false" | "0" => Ok(MtpMode::Off),
                    _ => Err(E::invalid_value(serde::de::Unexpected::Str(value), &self)),
                }
            }
        }

        deserializer.deserialize_any(MtpVisitor)
    }
}

/// A credential. It never prints its value: `Debug` and `Display` show a mask.
#[derive(Clone, PartialEq, Eq)]
pub struct Secret(String);

impl Secret {
    pub const MASK: &'static str = "********";

    pub const fn new(value: String) -> Self {
        Self(value)
    }

    /// The credential itself, for the one place that sends it.
    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for Secret {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("Secret(<redacted>)")
    }
}

impl fmt::Display for Secret {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(Self::MASK)
    }
}

#[allow(clippy::panic, clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn secrets_never_print_their_value() {
        let secret = Secret::new("sk-very-secret".to_owned());
        assert!(!format!("{secret:?}").contains("sk-very"));
        assert!(!format!("{secret}").contains("sk-very"));
        assert_eq!(secret.expose(), "sk-very-secret");
    }

    #[test]
    fn mtp_modes_accept_words_and_booleans() {
        let parse = |text: &str| serde_json::from_str::<MtpMode>(text).unwrap();
        assert_eq!(parse("\"auto\""), MtpMode::Auto);
        assert_eq!(parse("\"on\""), MtpMode::On);
        assert_eq!(parse("\"off\""), MtpMode::Off);
        assert_eq!(parse("true"), MtpMode::On);
        assert_eq!(parse("false"), MtpMode::Off);
        assert!(serde_json::from_str::<MtpMode>("\"maybe\"").is_err());
    }

    #[test]
    fn batch_mode_round_trips() {
        for mode in [BatchMode::Auto, BatchMode::On, BatchMode::Off] {
            let text = serde_json::to_string(&mode).unwrap();
            assert_eq!(serde_json::from_str::<BatchMode>(&text).unwrap(), mode);
        }
    }
}
