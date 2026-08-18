//! Model-family detection and loading adapters.
//!
//! Detection preserves both nested text-backbone and top-level wrapper model
//! types as resolution candidates. Resolution scores every adapter across all
//! candidates: special structural matches beat exact matches, exact matches on
//! any candidate beat tolerant matches on any candidate, and equal scores are
//! resolved by stable registry order.

use std::{
    borrow::Cow,
    fmt,
    path::{Path, PathBuf},
    sync::OnceLock,
};

use serde_json::Value;

use crate::{AnyModel, error::ModelError};

/// Maximum accepted `config.json` size (10 MiB).
pub const MAX_CONFIG_SIZE: u64 = 10 * 1024 * 1024;

/// A broad model family, independent of checkpoint naming revisions.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ModelFamily {
    /// Alibaba's Qwen language-model family.
    Qwen,
    /// Google's Gemma language-model family.
    Gemma,
    /// Meta's Llama language-model family.
    Llama,
    /// Mistral AI's language-model family.
    Mistral,
    /// Microsoft's Phi language-model family.
    Phi,
    /// The `StarCoder` language-model family.
    Starcoder,
    /// The `DeepSeek` language-model family.
    DeepSeek,
    /// The `LLaVA` vision-language family with a supported text backbone.
    Llava,
    /// A detected family for which Higgs has no built-in family variant.
    Other(String),
}

impl fmt::Display for ModelFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Qwen => f.write_str("Qwen"),
            Self::Gemma => f.write_str("Gemma"),
            Self::Llama => f.write_str("Llama"),
            Self::Mistral => f.write_str("Mistral"),
            Self::Phi => f.write_str("Phi"),
            Self::Starcoder => f.write_str("Starcoder"),
            Self::DeepSeek => f.write_str("DeepSeek"),
            Self::Llava => f.write_str("Llava"),
            Self::Other(name) => f.write_str(name),
        }
    }
}

/// Numeric family version parsed from a checkpoint's effective model type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ModelVersion {
    /// Major version parsed from `model_type`.
    pub major: u32,
    /// Minor version parsed from `model_type`, or zero when only a major version is present.
    pub minor: u32,
}

impl fmt::Display for ModelVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)
    }
}

/// A config parsed once for adapter resolution and loading.
#[derive(Debug, Clone)]
pub struct DetectedModel {
    /// Effective model type from the config consumed by the selected text loader.
    pub model_type: String,
    /// Top-level wrapper model type when `text_config` was selected.
    pub wrapper_model_type: Option<String>,
    /// Top-level architecture names declared by the checkpoint.
    pub architectures: Vec<String>,
    /// Broad family classified from the effective `model_type`.
    pub family: ModelFamily,
    /// Numeric version parsed from the effective `model_type`, when present.
    pub version: Option<ModelVersion>,
    /// Original, unmodified top-level configuration.
    pub raw: Value,
    /// Checkpoint directory containing the detected `config.json` and weights.
    pub dir: PathBuf,
}

impl DetectedModel {
    /// Candidate model types in diagnostic order: effective/nested first,
    /// followed by the top-level wrapper type when present.
    fn model_type_candidates(&self) -> impl Iterator<Item = &str> {
        std::iter::once(self.model_type.as_str()).chain(self.wrapper_model_type.as_deref())
    }

    /// The config object consumed by the effective text loader.
    #[must_use]
    pub fn resolved_config(&self) -> &Value {
        if self.wrapper_model_type.is_some() {
            self.raw.get("text_config").unwrap_or(&self.raw)
        } else {
            &self.raw
        }
    }
}

/// Statically knowable capabilities of an adapter's current implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[allow(clippy::struct_excessive_bools)]
pub struct Capabilities {
    /// Whether the adapter implements vision or other image inputs.
    pub vision: bool,
    /// Whether the adapter can load and use multi-token-prediction layers.
    pub mtp: bool,
    /// Whether the adapter implements mixture-of-experts layers.
    pub moe: bool,
    /// Whether the adapter can store the compressed MLA latent state in the KV cache.
    ///
    /// When false, requesting MLA latent caching is a no-op and diagnostics warn
    /// that the resolved adapter does not implement it.
    pub mla_latent_cache: bool,
    /// Whether the adapter supports the true batched-decode engine used by `batch=true`.
    ///
    /// This is distinct from serving concurrent requests through the normal,
    /// non-batched engine.
    pub batch_engine: bool,
}

/// Human-facing adapter metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterInfo {
    /// Stable identifier used in logs, diagnostics, and configuration inspection.
    pub id: &'static str,
    /// Broad model family implemented by the adapter.
    pub family: ModelFamily,
    /// Human-readable version range the adapter accepts.
    pub version_range: String,
    /// Features implemented by this adapter's current loader.
    pub capabilities: Capabilities,
    /// Short human-readable description of the implementation and its limits.
    pub notes: &'static str,
}

/// Uniform model-family routing and loading interface.
pub trait ModelAdapter: Send + Sync {
    /// Return the adapter's stable identifier.
    fn id(&self) -> &'static str;
    /// Return the broad model family implemented by this adapter.
    fn family(&self) -> ModelFamily;
    /// Return metadata describing the adapter's supported range and capabilities.
    fn describe(&self) -> AdapterInfo;
    /// Score this adapter for a detected checkpoint.
    ///
    /// Higher scores are preferred; equal scores preserve registry order.
    /// Returning `None` means the adapter cannot handle the checkpoint.
    fn match_score(&self, model: &DetectedModel) -> Option<u32>;
    /// Construct a model from the already-detected configuration and checkpoint directory.
    ///
    /// Implementations must consume `DetectedModel::raw` or `resolved_config()`
    /// rather than reopening `config.json`, and return a descriptive error when
    /// the config or weights are incompatible.
    fn load(&self, model: &DetectedModel) -> Result<AnyModel, ModelError>;
}

#[derive(Clone, Copy)]
enum LoadKind {
    Transformer,
    Bonsai,
    Qwen3Next,
    Qwen35Dense,
    Qwen35Moe,
    Qwen3Moe,
    Gemma2,
    Gemma3,
    Gemma4,
    Phi3,
    Starcoder2,
    LlavaQwen2,
    DeepSeekV2,
}

struct BuiltinAdapter {
    id: &'static str,
    family: fn() -> ModelFamily,
    version_range: &'static str,
    capabilities: Capabilities,
    notes: &'static str,
    kind: LoadKind,
}

#[allow(clippy::fn_params_excessive_bools)]
const fn caps(
    vision: bool,
    mtp: bool,
    moe: bool,
    mla_latent_cache: bool,
    batch_engine: bool,
) -> Capabilities {
    Capabilities {
        vision,
        mtp,
        moe,
        mla_latent_cache,
        batch_engine,
    }
}

const fn qwen() -> ModelFamily {
    ModelFamily::Qwen
}
const fn gemma() -> ModelFamily {
    ModelFamily::Gemma
}
const fn phi() -> ModelFamily {
    ModelFamily::Phi
}
const fn starcoder() -> ModelFamily {
    ModelFamily::Starcoder
}
const fn deepseek() -> ModelFamily {
    ModelFamily::DeepSeek
}
const fn llava() -> ModelFamily {
    ModelFamily::Llava
}

static TRANSFORMER_DENSE: BuiltinAdapter = BuiltinAdapter {
    id: "transformer-dense",
    family: qwen,
    version_range: "Qwen 2-3; Llama; Mistral",
    capabilities: caps(false, false, false, false, true),
    notes: "Dense transformer engine for Qwen2/Qwen3, Llama, and Mistral",
    kind: LoadKind::Transformer,
};
static BONSAI_Q1: BuiltinAdapter = BuiltinAdapter {
    id: "bonsai-q1-packed",
    family: qwen,
    version_range: "Qwen 3.0 (Bonsai 1-bit)",
    capabilities: caps(false, false, false, false, false),
    notes: "Packed 1.25-bpw Bonsai-Q1 checkpoints",
    kind: LoadKind::Bonsai,
};
static QWEN3_NEXT: BuiltinAdapter = BuiltinAdapter {
    id: "qwen3-next",
    family: qwen,
    version_range: "Qwen 3 Next",
    capabilities: caps(false, true, true, false, false),
    notes: "Hybrid attention/GDN Qwen3-Next",
    kind: LoadKind::Qwen3Next,
};
static QWEN35_DENSE: BuiltinAdapter = BuiltinAdapter {
    id: "qwen3.5-dense",
    family: qwen,
    version_range: "Qwen 3.5+ dense",
    capabilities: caps(false, true, false, false, false),
    notes: "Qwen3.5 text backbone, including structurally compatible newer revisions",
    kind: LoadKind::Qwen35Dense,
};
static QWEN35_MOE: BuiltinAdapter = BuiltinAdapter {
    id: "qwen3.5-moe",
    family: qwen,
    version_range: "Qwen 3.5+ MoE",
    capabilities: caps(false, true, true, false, false),
    notes: "Qwen3.5 MoE text backbone, including structurally compatible newer revisions",
    kind: LoadKind::Qwen35Moe,
};
static QWEN3_MOE: BuiltinAdapter = BuiltinAdapter {
    id: "qwen3-moe",
    family: qwen,
    version_range: "Qwen 3.0 MoE",
    capabilities: caps(false, false, true, false, false),
    notes: "Qwen3 sparse MoE",
    kind: LoadKind::Qwen3Moe,
};
static GEMMA2: BuiltinAdapter = BuiltinAdapter {
    id: "gemma2",
    family: gemma,
    version_range: "Gemma 2.x",
    capabilities: caps(false, false, false, false, false),
    notes: "Gemma 2 text model",
    kind: LoadKind::Gemma2,
};
static GEMMA3: BuiltinAdapter = BuiltinAdapter {
    id: "gemma3-text",
    family: gemma,
    version_range: "Gemma 3+ text",
    capabilities: caps(false, false, false, false, false),
    notes: "Gemma 3 text backbone",
    kind: LoadKind::Gemma3,
};
static GEMMA4: BuiltinAdapter = BuiltinAdapter {
    id: "gemma4-text",
    family: gemma,
    version_range: "Gemma 4+ text/unified",
    capabilities: caps(false, false, true, false, false),
    notes: "Gemma 4 text backbone with optional MoE",
    kind: LoadKind::Gemma4,
};
static PHI3: BuiltinAdapter = BuiltinAdapter {
    id: "phi3",
    family: phi,
    version_range: "Phi 3.x",
    capabilities: caps(false, false, false, false, false),
    notes: "Phi-3",
    kind: LoadKind::Phi3,
};
static STARCODER2: BuiltinAdapter = BuiltinAdapter {
    id: "starcoder2",
    family: starcoder,
    version_range: "Starcoder 2.x",
    capabilities: caps(false, false, false, false, false),
    notes: "Starcoder2",
    kind: LoadKind::Starcoder2,
};
static LLAVA_QWEN2: BuiltinAdapter = BuiltinAdapter {
    id: "llava-qwen2",
    family: llava,
    version_range: "LLaVA-Qwen2",
    capabilities: caps(true, false, false, false, false),
    notes: "LLaVA vision-language model with Qwen2 text",
    kind: LoadKind::LlavaQwen2,
};
static DEEPSEEK_V2: BuiltinAdapter = BuiltinAdapter {
    id: "deepseek-v2",
    family: deepseek,
    version_range: "DeepSeek 2.x",
    capabilities: caps(false, false, true, true, false),
    notes: "DeepSeek-V2 MLA/MoE",
    kind: LoadKind::DeepSeekV2,
};

static BUILTINS: [&BuiltinAdapter; 13] = [
    &BONSAI_Q1,
    &QWEN35_MOE,
    &QWEN35_DENSE,
    &QWEN3_NEXT,
    &QWEN3_MOE,
    &GEMMA4,
    &GEMMA3,
    &GEMMA2,
    &PHI3,
    &STARCODER2,
    &LLAVA_QWEN2,
    &DEEPSEEK_V2,
    &TRANSFORMER_DENSE,
];

impl BuiltinAdapter {
    fn is_exact(&self, model_type: &str) -> bool {
        let text_alias = strip_text_alias(model_type);
        match self.kind {
            LoadKind::Transformer => {
                matches!(text_alias.as_ref(), "qwen2" | "qwen3")
                    || matches!(model_type, "llama" | "mistral")
            }
            LoadKind::Bonsai => text_alias == "qwen3",
            LoadKind::Qwen3Next => text_alias == "qwen3_next",
            LoadKind::Qwen35Dense => text_alias == "qwen3_5",
            LoadKind::Qwen35Moe => text_alias == "qwen3_5_moe",
            LoadKind::Qwen3Moe => text_alias == "qwen3_moe",
            LoadKind::Gemma2 => model_type == "gemma2",
            LoadKind::Gemma3 => text_alias == "gemma3",
            LoadKind::Gemma4 => {
                matches!(text_alias.as_ref(), "gemma4" | "gemma4_unified")
            }
            LoadKind::Phi3 => model_type == "phi3",
            LoadKind::Starcoder2 => model_type == "starcoder2",
            LoadKind::LlavaQwen2 => model_type == "llava-qwen2",
            LoadKind::DeepSeekV2 => model_type == "deepseek_v2",
        }
    }

    fn tolerant_match(&self, model_type: &str) -> bool {
        match self.kind {
            LoadKind::Qwen35Dense => {
                qwen_revision(model_type).is_some_and(|(minor, moe)| minor >= 5 && !moe)
            }
            LoadKind::Qwen35Moe => {
                qwen_revision(model_type).is_some_and(|(minor, moe)| minor >= 5 && moe)
            }
            LoadKind::Gemma3 => gemma_revision(model_type).is_some_and(|major| major >= 3),
            LoadKind::Gemma4 => gemma_revision(model_type).is_some_and(|major| major >= 4),
            LoadKind::Transformer
            | LoadKind::Bonsai
            | LoadKind::Qwen3Next
            | LoadKind::Qwen3Moe
            | LoadKind::Gemma2
            | LoadKind::Phi3
            | LoadKind::Starcoder2
            | LoadKind::LlavaQwen2
            | LoadKind::DeepSeekV2 => false,
        }
    }

    fn has_exact_candidate(&self, model: &DetectedModel) -> bool {
        model
            .model_type_candidates()
            .any(|model_type| self.is_exact(model_type))
    }

    fn has_exact_resolved_config(&self, model: &DetectedModel) -> bool {
        self.is_exact(&model.model_type)
    }

    fn validate_tolerant(&self, model: &DetectedModel) -> Result<(), ModelError> {
        if self.has_exact_resolved_config(model) {
            return Ok(());
        }
        let config = model.resolved_config();
        let common_integer_fields = [
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "vocab_size",
        ];
        let qwen_integer_fields = [
            "intermediate_size",
            "head_dim",
            "max_position_embeddings",
            "linear_num_value_heads",
            "linear_num_key_heads",
            "linear_key_head_dim",
            "linear_value_head_dim",
            "linear_conv_kernel_dim",
        ];
        let moe_integer_fields = [
            "num_experts",
            "num_experts_per_tok",
            "shared_expert_intermediate_size",
            "moe_intermediate_size",
        ];
        let required_integers = common_integer_fields
            .iter()
            .chain(
                matches!(self.kind, LoadKind::Qwen35Dense | LoadKind::Qwen35Moe)
                    .then_some(qwen_integer_fields.iter())
                    .into_iter()
                    .flatten(),
            )
            .chain(
                matches!(self.kind, LoadKind::Qwen35Moe)
                    .then_some(moe_integer_fields.iter())
                    .into_iter()
                    .flatten(),
            );
        let mut invalid = required_integers
            .into_iter()
            .filter(|field| {
                config
                    .get(*field)
                    .and_then(Value::as_i64)
                    .is_none_or(|value| value <= 0)
            })
            .copied()
            .collect::<Vec<_>>();
        if matches!(self.kind, LoadKind::Qwen35Dense | LoadKind::Qwen35Moe)
            && config
                .get("rms_norm_eps")
                .and_then(Value::as_f64)
                .is_none_or(|value| !value.is_finite() || value <= 0.0)
        {
            invalid.push("rms_norm_eps");
        }
        if invalid.is_empty() {
            Ok(())
        } else {
            Err(ModelError::UnsupportedModel(format!(
                "untested model type '{}' is not structurally compatible with adapter '{}': missing or invalid field(s): {}",
                model.model_type,
                self.id,
                invalid.join(", ")
            )))
        }
    }
}

impl ModelAdapter for BuiltinAdapter {
    fn id(&self) -> &'static str {
        self.id
    }
    fn family(&self) -> ModelFamily {
        (self.family)()
    }
    fn describe(&self) -> AdapterInfo {
        AdapterInfo {
            id: self.id,
            family: self.family(),
            version_range: self.version_range.to_owned(),
            capabilities: self.capabilities,
            notes: self.notes,
        }
    }
    fn match_score(&self, model: &DetectedModel) -> Option<u32> {
        if matches!(self.kind, LoadKind::Bonsai) {
            let group_size = u64::try_from(crate::bonsai_q1::GROUP_SIZE).ok()?;
            let quantization = model.raw.get("quantization")?;
            return (self.has_exact_candidate(model)
                && quantization.get("bits").and_then(Value::as_u64) == Some(1)
                && quantization.get("group_size").and_then(Value::as_u64) == Some(group_size))
            .then_some(2_000);
        }
        if self.has_exact_candidate(model) {
            Some(1_000)
        } else if model
            .model_type_candidates()
            .any(|model_type| self.tolerant_match(model_type))
        {
            Some(if matches!(self.kind, LoadKind::Gemma4) {
                120
            } else {
                100
            })
        } else {
            None
        }
    }
    fn load(&self, model: &DetectedModel) -> Result<AnyModel, ModelError> {
        self.validate_tolerant(model)?;
        if !self.has_exact_candidate(model) {
            tracing::warn!(model_type = %model.model_type, adapter = self.id, "loading an untested model version through a structurally compatible adapter");
        }
        let dir = &model.dir;
        match self.kind {
            LoadKind::Transformer => {
                crate::transformer::model_args_from_value(model.resolved_config())
                    .and_then(|args| crate::transformer::load_model_with_args(dir, args))
                    .map(AnyModel::Transformer)
            }
            LoadKind::Bonsai => crate::bonsai_q1::load_bonsai_q1_with_config(dir, &model.raw)
                .map(AnyModel::BonsaiQ1),
            LoadKind::Qwen3Next => {
                crate::qwen3_next::load_qwen3_next_args_from_value(model.resolved_config().clone())
                    .and_then(|args| crate::qwen3_next::load_qwen3_next_model_with_args(dir, args))
                    .map(AnyModel::Qwen3Next)
            }
            LoadKind::Qwen35Dense => qwen35_args(model)
                .and_then(|args| crate::qwen3_next::load_qwen3_5_model_with_args(dir, args))
                .map(AnyModel::Qwen3Next),
            LoadKind::Qwen35Moe => qwen35_args(model)
                .and_then(|args| crate::qwen3_next::load_qwen3_5_moe_model_with_args(dir, args))
                .map(AnyModel::Qwen3Next),
            LoadKind::Qwen3Moe => serde_json::from_value(model.resolved_config().clone())
                .map_err(ModelError::Json)
                .and_then(|args| crate::qwen3_moe::load_qwen3_moe_model_with_args(dir, args))
                .map(AnyModel::Qwen3Moe),
            LoadKind::Gemma2 => serde_json::from_value(model.resolved_config().clone())
                .map_err(ModelError::Json)
                .and_then(|args| crate::gemma2::load_gemma2_model_with_args(dir, args))
                .map(AnyModel::Gemma2),
            LoadKind::Gemma3 => crate::gemma3::gemma3_model_args_from_value(model.raw.clone())
                .and_then(|args| crate::gemma3::load_gemma3_model_with_args(dir, args))
                .map(AnyModel::Gemma3),
            LoadKind::Gemma4 => crate::gemma4::gemma4_model_args_from_value(model.raw.clone())
                .and_then(|args| crate::gemma4::load_gemma4_model_with_args(dir, args))
                .map(AnyModel::Gemma4),
            LoadKind::Phi3 => serde_json::from_value(model.resolved_config().clone())
                .map_err(ModelError::Json)
                .and_then(|args| crate::phi3::load_phi3_model_with_args(dir, args))
                .map(AnyModel::Phi3),
            LoadKind::Starcoder2 => serde_json::from_value(model.resolved_config().clone())
                .map_err(ModelError::Json)
                .and_then(|args| crate::starcoder2::load_starcoder2_model_with_args(dir, args))
                .map(AnyModel::Starcoder2),
            LoadKind::LlavaQwen2 => {
                crate::llava_qwen2::load_llava_qwen2_model_from_value(dir, &model.raw)
                    .map(AnyModel::LlavaQwen2)
            }
            LoadKind::DeepSeekV2 => serde_json::from_value(model.resolved_config().clone())
                .map_err(ModelError::Json)
                .and_then(|args| crate::deepseek_v2::load_deepseek_v2_model_with_args(dir, args))
                .map(AnyModel::DeepSeekV2),
        }
    }
}

fn qwen35_args(model: &DetectedModel) -> Result<crate::qwen3_next::Qwen3NextModelArgs, ModelError> {
    if model.raw.get("text_config").is_some() {
        crate::qwen3_next::load_qwen3_5_text_config_args_from_value(&model.raw)
    } else {
        let wrapped = serde_json::json!({ "text_config": model.raw.clone() });
        crate::qwen3_next::load_qwen3_5_text_config_args_from_value(&wrapped)
    }
}

fn as_model_adapter(adapter: &'static BuiltinAdapter) -> &'static dyn ModelAdapter {
    adapter
}

/// The process-wide ordered adapter registry.
#[must_use]
pub fn registry() -> &'static [&'static dyn ModelAdapter] {
    static REGISTRY: OnceLock<Vec<&'static dyn ModelAdapter>> = OnceLock::new();
    REGISTRY
        .get_or_init(|| BUILTINS.iter().copied().map(as_model_adapter).collect())
        .as_slice()
}

/// Parse and classify a checkpoint's `config.json` once.
pub fn detect(dir: &Path) -> Result<DetectedModel, ModelError> {
    let path = dir.join("config.json");
    let file = std::fs::File::open(path)?;
    let size = file.metadata()?.len();
    if size > MAX_CONFIG_SIZE {
        return Err(ModelError::UnsupportedModel(format!(
            "config.json too large ({size} bytes, max {MAX_CONFIG_SIZE})"
        )));
    }
    let raw: Value = serde_json::from_reader(file)?;
    let top_model_type = raw
        .get("model_type")
        .and_then(Value::as_str)
        .ok_or_else(|| ModelError::UnsupportedModel("missing model_type in config.json".into()))?;
    let architectures = raw
        .get("architectures")
        .and_then(Value::as_array)
        .map_or_else(Vec::new, |values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(ToOwned::to_owned)
                .collect()
        });
    let nested_type = raw
        .get("text_config")
        .and_then(|value| value.get("model_type"))
        .and_then(Value::as_str);
    let (model_type, wrapper_model_type) = nested_type.map_or_else(
        || (top_model_type.to_owned(), None),
        |nested| (nested.to_owned(), Some(top_model_type.to_owned())),
    );
    let (family, version) = classify(&model_type);
    Ok(DetectedModel {
        model_type,
        wrapper_model_type,
        architectures,
        family,
        version,
        raw,
        dir: dir.to_path_buf(),
    })
}

/// Resolve the most-specific adapter across all detected model-type candidates.
///
/// Exact matches on either the nested or wrapper candidate take precedence over
/// every version-tolerant match. Tolerant selections retain structural checks.
pub fn resolve(model: &DetectedModel) -> Result<&'static dyn ModelAdapter, ModelError> {
    let selected = BUILTINS
        .iter()
        .enumerate()
        .filter_map(|(index, adapter)| {
            adapter
                .match_score(model)
                .map(|score| (score, std::cmp::Reverse(index), *adapter))
        })
        .max_by_key(|(score, index, _)| (*score, *index))
        .map(|(_, _, adapter)| adapter);
    let Some(adapter) = selected else {
        return Err(unsupported_error(&model.model_type));
    };
    // Keep validation in resolve so callers that only inspect a checkpoint
    // still receive structural errors before attempting to load weights.
    adapter.validate_tolerant(model)?;
    Ok(adapter)
}

/// Metadata for every registered adapter, in deterministic resolution order.
#[must_use]
pub fn supported() -> Vec<AdapterInfo> {
    registry()
        .iter()
        .map(|adapter| adapter.describe())
        .collect()
}

/// Whether resolution selected a structurally gated, not-exactly-known version.
#[must_use]
pub fn is_untested_version(adapter: &dyn ModelAdapter, model: &DetectedModel) -> bool {
    !model
        .model_type_candidates()
        .any(|model_type| is_exact_supported_by(adapter.id(), model_type))
}

/// Exact-string compatibility used by the legacy registry facade.
#[must_use]
pub fn is_exact_supported(model_type: &str) -> bool {
    BUILTINS.iter().any(|adapter| adapter.is_exact(model_type))
}

fn is_exact_supported_by(adapter_id: &str, model_type: &str) -> bool {
    BUILTINS
        .iter()
        .any(|adapter| adapter.id == adapter_id && adapter.is_exact(model_type))
}

fn unsupported_error(model_type: &str) -> ModelError {
    let ranges = supported()
        .into_iter()
        .map(|info| format!("{} ({})", info.family, info.version_range))
        .collect::<Vec<_>>()
        .join(", ");
    ModelError::UnsupportedModel(format!(
        "{model_type}; supported families/version ranges: {ranges}"
    ))
}

fn qwen_revision(model_type: &str) -> Option<(u32, bool)> {
    let normalized = strip_text_alias(model_type);
    let rest = normalized.strip_prefix("qwen3_")?;
    let (minor_text, suffix) = rest
        .split_once('_')
        .map_or((rest, None), |(minor, suffix)| (minor, Some(suffix)));
    let minor = minor_text.parse().ok()?;
    match suffix {
        None => Some((minor, false)),
        Some("moe") => Some((minor, true)),
        Some(_) => None,
    }
}

fn gemma_revision(model_type: &str) -> Option<u32> {
    let normalized = strip_text_alias(model_type);
    let rest = normalized.strip_prefix("gemma")?;
    let (major, suffix) = rest
        .split_once('_')
        .map_or((rest, None), |(major, suffix)| (major, Some(suffix)));
    if suffix.is_some_and(|value| !matches!(value, "text" | "unified")) {
        return None;
    }
    major.parse().ok()
}

fn strip_text_alias(model_type: &str) -> Cow<'_, str> {
    model_type.strip_suffix("_text_moe").map_or_else(
        || {
            model_type
                .strip_suffix("_text")
                .map_or(Cow::Borrowed(model_type), Cow::Borrowed)
        },
        |prefix| Cow::Owned(format!("{prefix}_moe")),
    )
}

fn classify(model_type: &str) -> (ModelFamily, Option<ModelVersion>) {
    if model_type.starts_with("llava") {
        return (ModelFamily::Llava, number_after(model_type, "llava-qwen"));
    }
    if model_type.starts_with("qwen") {
        return (ModelFamily::Qwen, number_after(model_type, "qwen"));
    }
    if model_type.starts_with("gemma") {
        return (ModelFamily::Gemma, number_after(model_type, "gemma"));
    }
    if model_type.starts_with("llama") {
        return (ModelFamily::Llama, number_after(model_type, "llama"));
    }
    if model_type.starts_with("mistral") {
        return (ModelFamily::Mistral, number_after(model_type, "mistral"));
    }
    if model_type.starts_with("phi") {
        return (ModelFamily::Phi, number_after(model_type, "phi"));
    }
    if model_type.starts_with("starcoder") {
        return (
            ModelFamily::Starcoder,
            number_after(model_type, "starcoder"),
        );
    }
    if model_type.starts_with("deepseek") {
        return (
            ModelFamily::DeepSeek,
            number_after(model_type, "deepseek_v"),
        );
    }
    (
        ModelFamily::Other(
            model_type
                .split(['_', '-'])
                .next()
                .unwrap_or(model_type)
                .to_owned(),
        ),
        None,
    )
}

fn number_after(model_type: &str, prefix: &str) -> Option<ModelVersion> {
    let rest = model_type.strip_prefix(prefix)?;
    let mut parts = rest.trim_start_matches(['_', '-']).split(['_', '-']);
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next().and_then(|part| part.parse().ok()).unwrap_or(0);
    Some(ModelVersion { major, minor })
}
