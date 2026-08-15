use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Deserializer};

/// The storage format for one checkpoint tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorQuant {
    Quantized { group_size: i32, bits: i32 },
    Unquantized,
}

impl TensorQuant {
    pub const fn group_size(self) -> i32 {
        match self {
            Self::Quantized { group_size, .. } => group_size,
            Self::Unquantized => 0,
        }
    }

    pub const fn bits(self) -> i32 {
        match self {
            Self::Quantized { bits, .. } => bits,
            Self::Unquantized => 0,
        }
    }
}

/// MLX `config.json` quantization settings.
///
/// Scalar `group_size` and `bits` entries specify the fallback. Any other
/// key is a checkpoint tensor path and either overrides that fallback or is
/// `false` when the tensor is stored densely. An object entry (explicit
/// `{"group_size": ..., "bits": ...}`) whose values exactly equal the
/// scalar fallback is treated as a no-op, identically to `true` — `mlx_lm`'s
/// converter can emit a *complete* per-tensor map where every path is
/// listed explicitly, even when most of them just restate the default it
/// was quantized with, and an architecture that never threads a given path
/// through per-tensor resolution shouldn't have to "support" an override
/// that resolves to the same thing either way.
///
/// Routed `MoE` expert weights are stored as one stacked tensor per projection
/// and dispatched through a fused gather kernel. Consequently, all experts in
/// one layer/projection must use the same storage format. Call
/// [`Self::resolve_uniform`] while building such a group; it rejects a
/// per-expert mixed setting rather than loading an incompatible checkpoint.
#[derive(Debug, Clone)]
pub struct QuantizationSettings {
    /// Kept public for source compatibility while callers migrate to the
    /// explicit accessor methods.
    pub group_size: i32,
    /// Kept public for source compatibility while callers migrate to the
    /// explicit accessor methods.
    pub bits: i32,
    tensors: HashMap<String, TensorQuant>,
}

impl QuantizationSettings {
    pub fn new(group_size: i32, bits: i32) -> Self {
        Self {
            group_size,
            bits,
            tensors: HashMap::new(),
        }
    }

    pub const fn default_group_size(&self) -> i32 {
        self.group_size
    }

    pub const fn default_bits(&self) -> i32 {
        self.bits
    }

    pub fn resolve(&self, path: &str) -> TensorQuant {
        self.tensors
            .get(path)
            .copied()
            .unwrap_or(TensorQuant::Quantized {
                group_size: self.group_size,
                bits: self.bits,
            })
    }

    /// Explicit per-tensor overrides from the checkpoint config.
    pub fn overridden_paths(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    /// Whether `path` carries an explicit, non-default-equal override.
    pub fn is_overridden(&self, path: &str) -> bool {
        self.tensors.contains_key(path)
    }

    /// Resolve a fused tensor group, rejecting mixed storage formats.
    pub fn resolve_uniform<'a, I>(&self, paths: I) -> Result<TensorQuant, String>
    where
        I: IntoIterator<Item = &'a str>,
    {
        let mut path_iter = paths.into_iter();
        let Some(first_path) = path_iter.next() else {
            return Err("cannot resolve an empty quantized tensor group".to_owned());
        };
        let first = self.resolve(first_path);
        for path in path_iter {
            let actual = self.resolve(path);
            if actual != first {
                return Err(format!(
                    "fused tensor group requires uniform quantization; {path} resolves to {actual:?}, but {first_path} resolves to {first:?}"
                ));
            }
        }
        Ok(first)
    }

    /// Reject a per-tensor override that the caller's architecture never
    /// read while constructing the model.
    ///
    /// Architectures that only partially support per-tensor overrides (e.g.
    /// `deepseek_v2`, `qwen3_next`) must collect every tensor path they
    /// actually resolved during construction and pass it here. Any
    /// override key outside that set would otherwise be silently ignored —
    /// the weight loads into a default-quantized (or default-dense) module
    /// instead of the format the checkpoint declares, producing wrong
    /// numerics instead of a load-time error.
    pub fn assert_all_overrides_consumed(&self, consumed: &HashSet<String>) -> Result<(), String> {
        for path in self.overridden_paths() {
            if !consumed.contains(path) {
                return Err(format!(
                    "per-tensor quantization override for {path} is not supported by this architecture"
                ));
            }
        }
        Ok(())
    }
}

impl<'de> Deserialize<'de> for QuantizationSettings {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let entries = HashMap::<String, serde_json::Value>::deserialize(deserializer)?;
        let group_size = entries
            .get("group_size")
            .and_then(serde_json::Value::as_i64)
            .and_then(|value| i32::try_from(value).ok())
            .ok_or_else(|| serde::de::Error::custom("quantization.group_size must be an i32"))?;
        let bits = entries
            .get("bits")
            .and_then(serde_json::Value::as_i64)
            .and_then(|value| i32::try_from(value).ok())
            .ok_or_else(|| serde::de::Error::custom("quantization.bits must be an i32"))?;

        let mut tensors = HashMap::new();
        for (path, value) in entries {
            // `mode` is MLX metadata (typically "affine"), not a tensor
            // path. Retain support for Qwen3.5 configs that include it.
            if path == "group_size" || path == "bits" || path == "mode" {
                continue;
            }
            let setting = match value {
                serde_json::Value::Bool(false) => TensorQuant::Unquantized,
                // `true` means "quantize with the scalar defaults" in
                // real mlx-community complete-predicate-map configs
                // (every tensor path explicitly listed, e.g.
                // mlx-community/Qwen3-30B-A3B-3bit). That's identical to
                // the key being absent, so skip inserting it: `resolve`
                // already falls back to the scalar default for untracked
                // paths, and leaving it out of `tensors` means it never
                // appears in `overridden_paths`/`assert_all_overrides_consumed`
                // — an architecture doesn't need to explicitly thread a
                // path whose override is "do the default thing".
                serde_json::Value::Bool(true) => continue,
                serde_json::Value::Object(map) => {
                    let tensor_group_size = map
                        .get("group_size")
                        .and_then(serde_json::Value::as_i64)
                        .and_then(|integer| i32::try_from(integer).ok())
                        .ok_or_else(|| {
                            serde::de::Error::custom(format!(
                                "quantization.{path}.group_size must be an i32"
                            ))
                        })?;
                    let tensor_bits = map
                        .get("bits")
                        .and_then(serde_json::Value::as_i64)
                        .and_then(|integer| i32::try_from(integer).ok())
                        .ok_or_else(|| {
                            serde::de::Error::custom(format!(
                                "quantization.{path}.bits must be an i32"
                            ))
                        })?;
                    if tensor_group_size == group_size && tensor_bits == bits {
                        // An object entry that merely restates the scalar
                        // default is semantically identical to `true` (see
                        // above): mlx_lm.convert's quant_predicate can emit
                        // a *complete* per-tensor map where most entries
                        // just echo the defaults it was called with, and an
                        // architecture that never threads that particular
                        // path shouldn't have to "support" a no-op.
                        continue;
                    }
                    TensorQuant::Quantized {
                        group_size: tensor_group_size,
                        bits: tensor_bits,
                    }
                }
                serde_json::Value::Null
                | serde_json::Value::Number(_)
                | serde_json::Value::String(_)
                | serde_json::Value::Array(_) => {
                    return Err(serde::de::Error::custom(format!(
                        "quantization.{path} must be true, false, or an object with group_size and bits"
                    )));
                }
            };
            tensors.insert(path, setting);
        }
        Ok(Self {
            group_size,
            bits,
            tensors,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::{QuantizationSettings, TensorQuant};

    #[test]
    fn parses_mixed_mlx_quantization_map() -> Result<(), serde_json::Error> {
        let config: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/mixed_quant_config.json"))?;
        let quantization = config.get("quantization").cloned().ok_or_else(|| {
            serde_json::Error::io(std::io::Error::other("fixture lacks quantization"))
        })?;
        let settings: QuantizationSettings = serde_json::from_value(quantization)?;

        assert_eq!(settings.default_group_size(), 128);
        assert_eq!(settings.default_bits(), 4);
        assert_eq!(
            settings.resolve("model.layers.0.mlp.experts.0.gate_proj"),
            TensorQuant::Quantized {
                group_size: 64,
                bits: 2
            }
        );
        assert_eq!(settings.resolve("lm_head"), TensorQuant::Unquantized);
        assert_eq!(
            settings.resolve("model.layers.1.self_attn.q_proj"),
            TensorQuant::Quantized {
                group_size: 128,
                bits: 4
            }
        );
        Ok(())
    }

    #[test]
    fn rejects_unknown_tensor_quantization_shapes() {
        let result = serde_json::from_str::<QuantizationSettings>(
            r#"{"group_size": 64, "bits": 4, "lm_head": 42}"#,
        );
        assert!(result.is_err_and(|err| err.to_string().contains("lm_head")));
    }

    #[test]
    fn accepts_true_as_default_quantization() -> Result<(), serde_json::Error> {
        // Real mlx-community configs (e.g. Qwen3-30B-A3B-3bit) use a complete
        // predicate map where every tensor path is listed explicitly: `true`
        // means "quantize with the scalar defaults", `false` means dense.
        let settings: QuantizationSettings = serde_json::from_str(
            r#"{
                "group_size": 64,
                "bits": 4,
                "model.layers.0.self_attn.q_proj": true,
                "model.layers.0.self_attn.k_proj": true,
                "model.layers.0.self_attn.v_proj": true,
                "model.layers.0.self_attn.o_proj": true,
                "model.layers.1.self_attn.q_proj": true,
                "lm_head": false
            }"#,
        )?;

        assert_eq!(
            settings.resolve("model.layers.0.self_attn.q_proj"),
            TensorQuant::Quantized {
                group_size: 64,
                bits: 4
            }
        );
        assert_eq!(settings.resolve("lm_head"), TensorQuant::Unquantized);
        Ok(())
    }

    #[test]
    fn true_entries_are_excluded_from_overridden_paths() -> Result<(), serde_json::Error> {
        let settings: QuantizationSettings = serde_json::from_str(
            r#"{"group_size": 64, "bits": 4, "model.layers.0.self_attn.q_proj": true, "lm_head": false}"#,
        )?;

        assert_eq!(
            settings.overridden_paths().collect::<Vec<_>>(),
            ["lm_head"],
            "a `true` entry means default behavior and must not be treated as an override"
        );
        Ok(())
    }

    #[test]
    fn true_entries_do_not_trip_assert_all_overrides_consumed() -> Result<(), serde_json::Error> {
        // An architecture that never threads `self_attn.q_proj` per-tensor
        // must still accept a config where that path is explicitly `true`
        // (default behavior), since `true` carries no architecture-specific
        // meaning to honor.
        let settings: QuantizationSettings = serde_json::from_str(
            r#"{"group_size": 64, "bits": 4, "model.layers.0.self_attn.q_proj": true}"#,
        )?;

        assert!(
            settings
                .assert_all_overrides_consumed(&HashSet::new())
                .is_ok()
        );
        Ok(())
    }

    #[test]
    fn rejects_mixed_quantization_in_a_fused_expert_group() -> Result<(), serde_json::Error> {
        let settings: QuantizationSettings = serde_json::from_str(
            r#"{
                "group_size": 64,
                "bits": 4,
                "model.layers.0.mlp.experts.0.gate_proj": {"group_size": 64, "bits": 4},
                "model.layers.0.mlp.experts.1.gate_proj": false
            }"#,
        )?;

        let err = settings
            .resolve_uniform([
                "model.layers.0.mlp.experts.0.gate_proj",
                "model.layers.0.mlp.experts.1.gate_proj",
            ])
            .err()
            .ok_or_else(|| serde_json::Error::io(std::io::Error::other("expected an error")))?;
        assert!(err.contains("model.layers.0.mlp.experts.1.gate_proj"));
        Ok(())
    }

    #[test]
    fn exposes_only_explicit_tensor_overrides() -> Result<(), serde_json::Error> {
        let settings: QuantizationSettings =
            serde_json::from_str(r#"{"group_size": 64, "bits": 4, "model.embed_tokens": false}"#)?;

        assert_eq!(
            settings.overridden_paths().collect::<Vec<_>>(),
            ["model.embed_tokens"]
        );
        Ok(())
    }

    #[test]
    fn assert_all_overrides_consumed_rejects_unhonored_key() -> Result<(), serde_json::Error> {
        let settings: QuantizationSettings = serde_json::from_str(
            r#"{"group_size": 64, "bits": 4, "model.layers.0.self_attn.kv_b_proj": false}"#,
        )?;

        let consumed: HashSet<String> = ["model.embed_tokens".to_owned(), "lm_head".to_owned()]
            .into_iter()
            .collect();
        let err = settings
            .assert_all_overrides_consumed(&consumed)
            .err()
            .ok_or_else(|| serde_json::Error::io(std::io::Error::other("expected an error")))?;
        assert!(err.contains("model.layers.0.self_attn.kv_b_proj"));
        Ok(())
    }

    #[test]
    fn assert_all_overrides_consumed_accepts_honored_key() -> Result<(), serde_json::Error> {
        let settings: QuantizationSettings =
            serde_json::from_str(r#"{"group_size": 64, "bits": 4, "lm_head": false}"#)?;

        let consumed: HashSet<String> = std::iter::once("lm_head".to_owned()).collect();
        assert!(settings.assert_all_overrides_consumed(&consumed).is_ok());
        Ok(())
    }

    #[test]
    fn default_equal_object_entries_are_excluded_from_overridden_paths()
    -> Result<(), serde_json::Error> {
        // mlx_lm.convert can emit a *complete* per-tensor map (every path
        // listed explicitly) where most entries just restate the scalar
        // default as an object rather than `true`. Those must be treated
        // as no-ops exactly like `true`, alongside a genuinely differing
        // override on a supported path.
        let settings: QuantizationSettings = serde_json::from_str(
            r#"{
                "group_size": 64,
                "bits": 6,
                "model.layers.0.self_attn.o_proj": {"group_size": 64, "bits": 6},
                "model.embed_tokens": {"group_size": 64, "bits": 6},
                "model.layers.1.mlp.experts.0.gate_proj": {"group_size": 64, "bits": 4}
            }"#,
        )?;

        assert_eq!(
            settings.overridden_paths().collect::<Vec<_>>(),
            ["model.layers.1.mlp.experts.0.gate_proj"],
            "object entries equal to the scalar default must not count as overrides"
        );
        assert_eq!(
            settings.resolve("model.layers.0.self_attn.o_proj"),
            TensorQuant::Quantized {
                group_size: 64,
                bits: 6
            },
            "resolve() must still return the same value whether or not the \
             entry was recorded as an override"
        );
        Ok(())
    }

    #[test]
    fn default_equal_entry_for_unsupported_path_does_not_trip_assert_all_overrides_consumed()
    -> Result<(), serde_json::Error> {
        // (c) An architecture that never threads self_attn.o_proj through
        // per-tensor resolution must still accept a config where that path
        // is explicitly set to exactly the scalar default -- it's a no-op,
        // not an unsupported override.
        let settings: QuantizationSettings = serde_json::from_str(
            r#"{"group_size": 64, "bits": 6, "model.layers.0.self_attn.o_proj": {"group_size": 64, "bits": 6}}"#,
        )?;

        assert!(
            settings
                .assert_all_overrides_consumed(&HashSet::new())
                .is_ok()
        );
        Ok(())
    }

    #[test]
    fn differing_entry_for_unsupported_path_still_rejected() -> Result<(), serde_json::Error> {
        // (b) A genuinely differing override on a path the architecture
        // never threads through per-tensor resolution must still be
        // rejected loudly, naming the tensor -- only default-equal entries
        // are treated as no-ops.
        let settings: QuantizationSettings = serde_json::from_str(
            r#"{"group_size": 64, "bits": 6, "model.layers.0.self_attn.o_proj": {"group_size": 64, "bits": 4}}"#,
        )?;

        let err = settings
            .assert_all_overrides_consumed(&HashSet::new())
            .err()
            .ok_or_else(|| serde_json::Error::io(std::io::Error::other("expected an error")))?;
        assert!(err.contains("model.layers.0.self_attn.o_proj"));
        Ok(())
    }
}
