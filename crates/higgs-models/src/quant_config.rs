use std::collections::HashMap;

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
/// `false` when the tensor is stored densely.
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
                    TensorQuant::Quantized {
                        group_size: tensor_group_size,
                        bits: tensor_bits,
                    }
                }
                serde_json::Value::Null
                | serde_json::Value::Bool(_)
                | serde_json::Value::Number(_)
                | serde_json::Value::String(_)
                | serde_json::Value::Array(_) => {
                    return Err(serde::de::Error::custom(format!(
                        "quantization.{path} must be false or an object with group_size and bits"
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
            r#"{"group_size": 64, "bits": 4, "lm_head": true}"#,
        );
        assert!(result.is_err_and(|err| err.to_string().contains("lm_head")));
    }
}
