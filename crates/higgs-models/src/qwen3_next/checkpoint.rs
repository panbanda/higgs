//! Checkpoint inspection: MTP weight layout detection and parameter key remapping.

use super::config::{Qwen3NextModelArgs, load_model_args};
use crate::error::ModelError;
use mlx_rs::Array;
use std::{collections::HashMap, path::Path};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MtpWeightLayout {
    None,
    Quantized,
    Dense,
    /// MTP layer is MoE-structured (`mlp.gate` / `shared_expert` / experts),
    /// e.g. Qwen3.6-A3B sidecars — loaded via [`MoeMtpHead`].
    MoeQuantized,
}

fn is_mtp_key(key: &str) -> bool {
    key.starts_with("mtp.") || key.contains(".mtp.")
}

pub(super) fn mtp_weight_layout_from_keys<'a>(
    keys: impl IntoIterator<Item = &'a str>,
) -> MtpWeightLayout {
    let mut has_mtp = false;
    let mut has_unprefixed_mtp = false;
    let mut has_quantized_aux = false;
    let mut has_moe_mlp = false;

    for key in keys {
        if !is_mtp_key(key) {
            continue;
        }
        has_mtp = true;
        has_unprefixed_mtp |= key.starts_with("mtp.");
        has_quantized_aux |= key.ends_with(".scales") || key.ends_with(".biases");
        has_moe_mlp |= key.contains(".mlp.gate.")
            || key.contains(".mlp.shared_expert")
            || key.contains(".mlp.switch_mlp")
            || key.contains(".mlp.experts");
    }

    if has_mtp && has_moe_mlp {
        MtpWeightLayout::MoeQuantized
    } else if has_quantized_aux {
        MtpWeightLayout::Quantized
    } else if has_unprefixed_mtp {
        MtpWeightLayout::Dense
    } else if has_mtp {
        MtpWeightLayout::Quantized
    } else {
        MtpWeightLayout::None
    }
}

pub(super) fn checkpoint_mtp_weight_layout(
    model_path: &Path,
) -> Result<MtpWeightLayout, ModelError> {
    fn safetensors_file_mtp_weight_layout(file_path: &Path) -> Result<MtpWeightLayout, ModelError> {
        let bytes = std::fs::read(file_path)?;
        let metadata = safetensors::SafeTensors::deserialize(&bytes)
            .map_err(|e| ModelError::Io(std::io::Error::other(e.to_string())))?;
        // Auxiliary sidecars may ship truly unprefixed keys (`fc.weight`,
        // `layers.0....`) — normalize so they classify as MTP keys.
        let normalized: Vec<String> = metadata
            .names()
            .into_iter()
            .map(|name| normalize_sidecar_mtp_key(file_path, name.to_owned()))
            .collect();
        Ok(mtp_weight_layout_from_keys(
            normalized.iter().map(String::as_str),
        ))
    }

    let index_path = model_path.join("model.safetensors.index.json");
    if index_path.exists() {
        let file = std::fs::File::open(index_path)?;
        let index: serde_json::Value = serde_json::from_reader(file)?;
        let Some(weight_map) = index
            .get("weight_map")
            .and_then(serde_json::Value::as_object)
        else {
            return Err(ModelError::UnsupportedModel(
                "model.safetensors.index.json missing weight_map".into(),
            ));
        };
        let index_layout = mtp_weight_layout_from_keys(weight_map.keys().map(String::as_str));
        if index_layout != MtpWeightLayout::None {
            return Ok(index_layout);
        }

        let auxiliary_files: Vec<_> = crate::AUXILIARY_SAFETENSORS_FILES
            .iter()
            .map(|file_name| model_path.join(file_name))
            .filter(|file_path| file_path.exists())
            .collect();
        if auxiliary_files.len() > 1 {
            return Err(ModelError::UnsupportedModel(
                "ambiguous MTP sidecars: both mtp.safetensors and model-mtp.safetensors are present; remove one".to_owned(),
            ));
        }

        for file_path in auxiliary_files {
            let layout = safetensors_file_mtp_weight_layout(&file_path)?;
            if layout != MtpWeightLayout::None {
                return Ok(layout);
            }
        }

        return Ok(MtpWeightLayout::None);
    }

    for file_path in crate::collect_safetensors_files(model_path)? {
        let layout = safetensors_file_mtp_weight_layout(&file_path)?;
        if layout != MtpWeightLayout::None {
            return Ok(layout);
        }
    }

    Ok(MtpWeightLayout::None)
}

#[cfg(test)]
pub(super) fn checkpoint_has_mtp_weights(model_path: &Path) -> Result<bool, ModelError> {
    Ok(checkpoint_mtp_weight_layout(model_path)? != MtpWeightLayout::None)
}

/// Whether loading this checkpoint attaches an MTP head: the config asks for
/// one and the weights are present. Mirrors the decision `load` makes, without
/// loading the model.
#[must_use]
pub fn checkpoint_has_mtp_head<P: AsRef<Path>>(model_dir: P) -> bool {
    let model_path = model_dir.as_ref();
    load_model_args(model_path).is_ok_and(|args| {
        args.mtp_num_hidden_layers > 0
            && checkpoint_mtp_weight_layout(model_path)
                .is_ok_and(|layout| layout != MtpWeightLayout::None)
    })
}

pub(super) fn maybe_disable_mtp_without_checkpoint_weights(
    args: &mut Qwen3NextModelArgs,
    model_path: &Path,
) -> Result<(), ModelError> {
    if args.mtp_num_hidden_layers <= 0 {
        return Ok(());
    }

    match checkpoint_mtp_weight_layout(model_path)? {
        MtpWeightLayout::Quantized => {
            args.use_dense_mtp = false;
            return Ok(());
        }
        MtpWeightLayout::Dense => {
            args.use_dense_mtp = true;
            return Ok(());
        }
        MtpWeightLayout::MoeQuantized => {
            tracing::info!(
                "Checkpoint ships an MoE-structured MTP head (Qwen3.6-A3B style); \
                 loading via MoeMtpHead"
            );
            args.use_dense_mtp = false;
            args.use_moe_mtp = true;
            return Ok(());
        }
        MtpWeightLayout::None => {}
    }

    tracing::warn!(
        mtp_num_hidden_layers = args.mtp_num_hidden_layers,
        "Config enables MTP but checkpoint has no MTP weights; disabling MTP for this load"
    );
    args.mtp_num_hidden_layers = 0;
    Ok(())
}

pub(super) fn qwen35_checkpoint_param_key(key: &str) -> Option<&str> {
    if key.starts_with("mtp.") {
        Some(key)
    } else {
        key.strip_prefix("language_model.")
    }
}

pub(super) fn dense_mtp_param_key(stripped: &str) -> Option<String> {
    stripped
        .strip_prefix("mtp.")
        .map(|rest| format!("dense_mtp.{rest}"))
}

pub(super) fn moe_mtp_param_key(stripped: &str) -> Option<String> {
    stripped
        .strip_prefix("mtp.")
        .map(|rest| format!("moe_mtp.{rest}"))
}

/// Normalize a tensor key loaded from an auxiliary MTP sidecar file.
///
/// Some sidecars (e.g. mlx-community MTP drafters) ship truly unprefixed keys
/// (`fc.weight`, `layers.0....`); prefix them with `mtp.` so they map onto the
/// model's MTP head params the same way prefixed sidecars do. Keys from
/// non-auxiliary files are returned unchanged.
pub(super) fn normalize_sidecar_mtp_key(file_path: &Path, key: String) -> String {
    let is_aux = file_path
        .file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| crate::AUXILIARY_SAFETENSORS_FILES.contains(&n));
    // Only prefix truly un-namespaced sidecar keys (`fc.weight`,
    // `layers.0....`). Already-namespaced keys (`mtp.*`, `language_model.mtp.*`)
    // are left intact so `qwen35_checkpoint_param_key` can still strip/remap
    // them — prefixing those would produce unmatchable `mtp.language_model.mtp.*`.
    if is_aux && !is_mtp_key(&key) {
        format!("mtp.{key}")
    } else {
        key
    }
}

pub(super) fn qwen35_target_param_key(
    params: &HashMap<std::rc::Rc<str>, &mut Array>,
    stripped: &str,
) -> Option<(String, bool)> {
    if params.contains_key(stripped) {
        Some((stripped.to_owned(), false))
    } else if let Some(moe_key) =
        moe_mtp_param_key(stripped).filter(|key| params.contains_key(key.as_str()))
    {
        // MoE MTP head: plain remap, no dense rmsnorm adjustment.
        Some((moe_key, false))
    } else {
        dense_mtp_param_key(stripped)
            .filter(|dense_key| params.contains_key(dense_key.as_str()))
            .map(|dense_key| (dense_key, true))
    }
}

pub(super) fn dense_mtp_rmsnorm_weight_key(stripped: &str) -> bool {
    stripped.starts_with("mtp.")
        && stripped.ends_with(".weight")
        && (stripped.contains(".input_layernorm.")
            || stripped.contains(".post_attention_layernorm.")
            || stripped.contains(".q_norm.")
            || stripped.contains(".k_norm.")
            || stripped == "mtp.norm.weight"
            || stripped == "mtp.pre_fc_norm_hidden.weight"
            || stripped == "mtp.pre_fc_norm_embedding.weight")
}

pub(super) fn qwen35_loaded_value(
    stripped: &str,
    value: Array,
    dense_mtp_target: bool,
) -> Result<Array, crate::error::ModelError> {
    if dense_mtp_target && dense_mtp_rmsnorm_weight_key(stripped) {
        let one = Array::from_f32(1.0)
            .as_dtype(value.dtype())
            .map_err(crate::error::ModelError::Mlx)?;
        value.add(&one).map_err(crate::error::ModelError::Mlx)
    } else {
        Ok(value)
    }
}
