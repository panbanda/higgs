//! Weight loading from safetensors, including GDN projection fusion and permutation.

use super::{
    checkpoint::{
        normalize_sidecar_mtp_key, qwen35_checkpoint_param_key, qwen35_loaded_value,
        qwen35_target_param_key,
    },
    config::QuantizationConfig,
};
use crate::error::ModelError;
use mlx_rs::{Array, error::Exception, ops};
use std::path::Path;

pub(super) fn placeholder_param_names<'a, I, K>(params: I) -> Vec<String>
where
    I: IntoIterator<Item = (K, &'a Array)>,
    K: AsRef<str>,
{
    params
        .into_iter()
        .filter(|(_, value)| value.shape() == [1])
        .map(|(name, _)| name.as_ref().to_owned())
        .collect()
}

pub(super) fn ensure_all_model_params_loaded<'a, I, K>(params: I) -> Result<(), ModelError>
where
    I: IntoIterator<Item = (K, &'a Array)>,
    K: AsRef<str>,
{
    let placeholders = placeholder_param_names(params);
    if placeholders.is_empty() {
        return Ok(());
    }

    let examples = placeholders
        .iter()
        .take(10)
        .cloned()
        .collect::<Vec<_>>()
        .join(", ");
    Err(ModelError::MissingWeight(format!(
        "{} model params were not loaded from the checkpoint; examples: {examples}",
        placeholders.len()
    )))
}

/// GDN dimension info extracted from model args before move.
pub(super) struct GdnDims {
    pub(super) num_k_heads: i32,
    pub(super) num_v_heads: i32,
    pub(super) head_k_dim: i32,
    pub(super) head_v_dim: i32,
}

impl GdnDims {
    /// Validate GQA ratio: `num_v_heads` must be divisible by `num_k_heads`.
    pub(super) fn validate(&self) -> Result<(), Exception> {
        if self.num_k_heads == 0 || self.num_v_heads % self.num_k_heads != 0 {
            return Err(Exception::custom(format!(
                "GQA ratio invalid: num_v_heads={} not divisible by num_k_heads={}",
                self.num_v_heads, self.num_k_heads
            )));
        }
        Ok(())
    }
}

/// Build row permutation to convert flat [`q_all|k_all|v_all|z_all`] layout
/// to per-head-grouped [`q_h0|k_h0|v_h0|z_h0|q_h1`|...] for `in_proj_qkvz`.
fn build_qkvz_permutation(d: &GdnDims) -> Result<Vec<i32>, Exception> {
    let nk = d.num_k_heads;
    if nk == 0 || d.num_v_heads % nk != 0 {
        return Err(Exception::custom(format!(
            "GQA ratio invalid: num_v_heads={} not divisible by num_k_heads={nk}",
            d.num_v_heads
        )));
    }
    let dk = d.head_k_dim;
    let v_per_k = d.num_v_heads / nk;
    let dv = d.head_v_dim;
    let key_dim = nk * dk;
    let qkv_rows = key_dim * 2 + d.num_v_heads * dv; // offset for z

    let mut perm = Vec::new();
    for h in 0..nk {
        // q: rows h*dk .. (h+1)*dk from qkv (offset 0)
        for i in 0..dk {
            perm.push(h * dk + i);
        }
        // k: rows key_dim + h*dk .. from qkv
        for i in 0..dk {
            perm.push(key_dim + h * dk + i);
        }
        // v: rows 2*key_dim + h*(v_per_k*dv) .. from qkv
        for i in 0..(v_per_k * dv) {
            perm.push(2 * key_dim + h * v_per_k * dv + i);
        }
        // z: rows h*(v_per_k*dv) .. from z (offset by qkv_rows)
        for i in 0..(v_per_k * dv) {
            perm.push(qkv_rows + h * v_per_k * dv + i);
        }
    }
    Ok(perm)
}

/// Build row permutation for flat [`b_all|a_all`] → per-head-grouped [`b_h0|a_h0|b_h1|a_h1`|...].
fn build_ba_permutation(d: &GdnDims) -> Vec<i32> {
    let nk = d.num_k_heads;
    let v_per_k = d.num_v_heads / nk;
    let nv = d.num_v_heads;

    let mut perm = Vec::new();
    for h in 0..nk {
        // b: rows h*v_per_k .. (h+1)*v_per_k from b
        for i in 0..v_per_k {
            perm.push(h * v_per_k + i);
        }
        // a: rows h*v_per_k .. (h+1)*v_per_k from a (offset by nv)
        for i in 0..v_per_k {
            perm.push(nv + h * v_per_k + i);
        }
    }
    perm
}

/// Concatenate two arrays along dim 0 and permute rows.
fn concat_and_permute(a: &Array, b: &Array, perm: &[i32]) -> Result<Array, Exception> {
    let cat = ops::concatenate(&[a, b], 0)?;
    let perm_arr = Array::from_slice(
        perm,
        &[i32::try_from(perm.len()).map_err(|_| Exception::custom("perm len overflow"))?],
    );
    cat.take_axis(&perm_arr, 0)
}

/// Return true when `a` and `b` can be concatenated along axis 0: the rank
/// matches and every non-axis-0 dimension is identical. Quantized weights pack
/// different bit-widths into different inner shapes, so this guards the BA
/// fusion path from silently producing a malformed `in_proj_ba` matrix.
pub(super) fn can_concatenate0_shapes(a_shape: &[i32], b_shape: &[i32]) -> bool {
    a_shape.len() == b_shape.len()
        && a_shape
            .iter()
            .zip(b_shape.iter())
            .enumerate()
            .all(|(axis, (lhs, rhs))| axis == 0 || lhs == rhs)
}

fn can_concatenate0(a: &Array, b: &Array) -> bool {
    let a_shape = a.shape();
    let b_shape = b.shape();
    can_concatenate0_shapes(a_shape, b_shape)
}

/// Load `Qwen3Next` weights, remapping the `mtp.*` sidecar onto whichever MTP
/// head is active (`mtp` / `dense_mtp` / `moe_mtp`).
///
/// Backbone keys match params directly (via `qwen35_target_param_key`'s
/// direct-match branch), so this is behaviour-compatible with the plain loader
/// for the common `Quantized` layout. The only added behaviour is the
/// `mtp.*` → `dense_mtp.*` / `moe_mtp.*` remap, which the plain loader lacks —
/// without it a dense/MoE draft head selected by
/// `maybe_disable_mtp_without_checkpoint_weights` is silently left uninitialized.
#[allow(clippy::shadow_reuse)]
pub(super) fn load_qwen3_next_weights<M: mlx_rs::module::ModuleParametersExt>(
    model: &mut M,
    model_path: &Path,
    quantization: Option<&QuantizationConfig>,
) -> Result<(), crate::error::ModelError> {
    let safetensors_files = crate::collect_safetensors_files(model_path)?;
    let mut params = model.parameters_mut().flatten();

    for file_path in &safetensors_files {
        let loaded = crate::load_safetensors_shard(file_path)?;
        crate::validate_quantized_tensor_widths(&loaded, quantization)?;

        for (key, value) in loaded {
            let key = normalize_sidecar_mtp_key(file_path, key);
            if let Some((target_key, dense_mtp_target)) = qwen35_target_param_key(&params, &key) {
                if let Some(param) = params.get_mut(target_key.as_str()) {
                    **param = qwen35_loaded_value(&key, value, dense_mtp_target)?;
                    continue;
                }
            }
            tracing::warn!(key = %key, "Weight key not found in model parameters");
        }
    }

    model
        .eval()
        .map_err(|e| crate::error::ModelError::Io(std::io::Error::other(e.to_string())))?;

    Ok(())
}

/// Load Qwen3.5-MoE weights with GDN projection fusion.
///
/// Direct weight loader: strip `language_model.` prefix, no rearrangement.
/// Used when `use_separate_gdn_projections = true`.
#[allow(clippy::shadow_reuse)]
pub(super) fn load_qwen3_5_moe_weights_direct<M: mlx_rs::module::ModuleParametersExt>(
    model: &mut M,
    model_path: &Path,
    quantization: Option<&QuantizationConfig>,
) -> Result<(), crate::error::ModelError> {
    let safetensors_files = crate::collect_safetensors_files(model_path)?;
    let mut params = model.parameters_mut().flatten();
    let mut matched = 0usize;
    let mut unmatched = Vec::new();

    for file_path in &safetensors_files {
        let loaded = crate::load_safetensors_shard(file_path)?;
        crate::validate_quantized_tensor_widths(&loaded, quantization)?;

        for (key, value) in loaded {
            let key = normalize_sidecar_mtp_key(file_path, key);
            let Some(stripped) = qwen35_checkpoint_param_key(&key) else {
                unmatched.push(key);
                continue;
            };
            if let Some((target_key, dense_mtp_target)) = qwen35_target_param_key(&params, stripped)
            {
                if let Some(param) = params.get_mut(target_key.as_str()) {
                    **param = qwen35_loaded_value(stripped, value, dense_mtp_target)?;
                } else {
                    unmatched.push(key);
                    continue;
                }
                matched += 1;
            } else {
                unmatched.push(key);
            }
        }
    }

    tracing::info!(
        matched,
        unmatched_count = unmatched.len(),
        "Direct weight loading stats"
    );
    if !unmatched.is_empty() {
        for k in unmatched.iter().take(10) {
            tracing::debug!(key = %k, "Unmatched weight key (no matching model param)");
        }
        if unmatched.len() > 10 {
            tracing::debug!("... and {} more unmatched keys", unmatched.len() - 10);
        }
    }
    let param_count = params.len();
    // This loader is only used with separate GDN projections (see
    // `load_qwen3_5_model_with_gdn_fallback`). In that mode the fused
    // `in_proj_qkvz` / `in_proj_ba` QLinears are still constructed — as unused
    // placeholders, since the forward path dispatches on
    // `use_separate_projections` — so they must be exempt from the
    // completeness check. Flagging them would reject every mixed-bit
    // checkpoint that *requires* separate projections (e.g. OptiQ quants).
    ensure_all_model_params_loaded(
        params
            .iter()
            .filter(|(name, _)| !(name.contains(".in_proj_qkvz.") || name.contains(".in_proj_ba.")))
            .map(|(name, value)| (std::rc::Rc::<str>::clone(name), &**value)),
    )?;
    tracing::info!(param_count, matched, "Total model parameters loaded");

    model
        .eval()
        .map_err(|e| crate::error::ModelError::Io(std::io::Error::other(e.to_string())))?;

    Ok(())
}

/// Rearranges flat (qkv,z,b,a) projections to per-head-grouped (qkvz,ba)
/// so the model uses the fused 2-dispatch forward path instead of 4 separate.
#[allow(clippy::too_many_lines, clippy::shadow_reuse)]
pub(super) fn load_qwen3_5_moe_weights_fused<M: mlx_rs::module::ModuleParametersExt>(
    model: &mut M,
    model_path: &Path,
    gdn_dims: &GdnDims,
    quantization: Option<&QuantizationConfig>,
) -> Result<(), crate::error::ModelError> {
    use std::collections::HashMap;

    let safetensors_files = crate::collect_safetensors_files(model_path)?;
    let mut params = model.parameters_mut().flatten();

    let qkvz_perm = build_qkvz_permutation(gdn_dims)
        .map_err(|e| crate::error::ModelError::ShapeMismatch(e.to_string()))?;
    let ba_perm = build_ba_permutation(gdn_dims);

    // GDN split keys: collect (part_a, part_b) for each combined target
    // Key format: "model.layers.N.linear_attn.in_proj_qkvz.{weight|scales|biases}"
    let mut gdn_parts: HashMap<String, (Option<Array>, Option<Array>)> = HashMap::new();

    let gdn_remap: &[(&str, &str, &str)] = &[
        ("in_proj_qkv", "in_proj_z", "in_proj_qkvz"),
        ("in_proj_b", "in_proj_a", "in_proj_ba"),
    ];

    for file_path in &safetensors_files {
        let loaded = crate::load_safetensors_shard(file_path)?;
        crate::validate_quantized_tensor_widths(&loaded, quantization)?;

        for (key, value) in loaded {
            let key = normalize_sidecar_mtp_key(file_path, key);
            let Some(stripped) = qwen35_checkpoint_param_key(&key) else {
                continue;
            };

            let mut handled = false;
            for &(part_a_name, part_b_name, combined_name) in gdn_remap {
                for (is_b, split_name) in [(false, part_a_name), (true, part_b_name)] {
                    let needle = format!(".{split_name}.");
                    if let Some(pos) = stripped.find(&needle) {
                        let pfx = &stripped[..pos];
                        let sfx = &stripped[pos + needle.len()..];
                        let map_key = format!("{pfx}.{combined_name}.{sfx}");
                        let entry = gdn_parts.entry(map_key).or_insert((None, None));
                        if is_b {
                            entry.1 = Some(value.clone());
                        } else {
                            entry.0 = Some(value.clone());
                        }
                        handled = true;
                        break;
                    }
                }
                if handled {
                    break;
                }
            }

            if !handled {
                if let Some((target_key, dense_mtp_target)) =
                    qwen35_target_param_key(&params, stripped)
                {
                    if let Some(param) = params.get_mut(target_key.as_str()) {
                        **param = qwen35_loaded_value(stripped, value, dense_mtp_target)?;
                    }
                }
            }
        }
    }

    // Fuse GDN pairs: concat + row permutation
    let mut fused_count = 0usize;
    for (combined_key, (part_a, part_b)) in &gdn_parts {
        let (Some(a), Some(b)) = (part_a, part_b) else {
            return Err(crate::error::ModelError::Io(std::io::Error::other(
                format!("Incomplete GDN projection pair for key: {combined_key}"),
            )));
        };
        if combined_key.contains("in_proj_ba") && !can_concatenate0(a, b) {
            return Err(crate::error::ModelError::ShapeMismatch(format!(
                "Mixed-bit BA fusion requires separate GDN projections for key {combined_key}: {:?} vs {:?}",
                a.shape(),
                b.shape()
            )));
        }
        let Some(param) = params.get_mut(combined_key.as_str()) else {
            return Err(crate::error::ModelError::Io(std::io::Error::other(
                format!("Fused target key not found in model params: {combined_key}"),
            )));
        };
        let perm = if combined_key.contains("in_proj_qkvz") {
            &qkvz_perm
        } else {
            &ba_perm
        };
        match concat_and_permute(a, b, perm) {
            Ok(fused) => {
                // Materialize each fused weight on its own. Leaving all of
                // them lazy makes the final `model.eval()` read and permute
                // every projection in the checkpoint inside one Metal command
                // buffer, which intermittently trips the GPU watchdog
                // (kIOGPUCommandBufferCallbackErrorTimeout) on large models.
                fused.eval().map_err(|e| {
                    crate::error::ModelError::Io(std::io::Error::other(format!(
                        "GDN fusion eval failed for key {combined_key}: {e}"
                    )))
                })?;
                **param = fused;
                fused_count += 1;
            }
            Err(e) => {
                return Err(crate::error::ModelError::Io(std::io::Error::other(
                    format!("GDN fusion failed for key {combined_key}: {e}"),
                )));
            }
        }
    }

    tracing::info!(
        fused_count,
        total_pairs = gdn_parts.len(),
        "Fused GDN projections (4→2 dispatches per layer)"
    );
    ensure_all_model_params_loaded(
        params
            .iter()
            .map(|(name, value)| (std::rc::Rc::<str>::clone(name), &**value)),
    )?;

    model
        .eval()
        .map_err(|e| crate::error::ModelError::Io(std::io::Error::other(e.to_string())))?;

    Ok(())
}
