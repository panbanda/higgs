//! Forward pass with custom `RoPE` positions for sparse prefill.

use super::{attention::Qwen3NextAttention, cache::LayerCache, model::Qwen3NextCausalLM};
use crate::cache::KeyValueCache;
use mlx_rs::{Array, error::Exception, module::Module, nn};

/// Forward pass for a single attention layer with custom `RoPE` positions.
///
/// This is a standalone function to avoid borrow checker issues.
fn forward_attention_sparse(
    attn: &mut Qwen3NextAttention,
    x: &Array,
    positions: &Array,
    cache: &mut crate::cache::SteppingKeyValueCache,
) -> Result<Array, Exception> {
    let shape = x.shape();
    let b = *shape
        .first()
        .ok_or_else(|| Exception::custom("Input must have >= 2 dims"))?;
    let l = *shape
        .get(1)
        .ok_or_else(|| Exception::custom("Input must have >= 2 dims"))?;

    // Q is projected to 2 * num_heads * head_dim (doubled for gating)
    let q_proj_output = attn.q_proj.forward(x)?;
    let q_reshaped = q_proj_output.reshape(&[b, l, attn.num_attention_heads, -1])?;
    let q_halves = q_reshaped.split_equal(2, Some(-1))?;
    let queries_pre = q_halves
        .first()
        .ok_or_else(|| Exception::custom("split produced empty result"))?;
    let gate = q_halves
        .get(1)
        .ok_or_else(|| Exception::custom("split produced empty result"))?
        .reshape(&[b, l, -1])?;

    let keys_raw = attn.k_proj.forward(x)?;
    let values_raw = attn.v_proj.forward(x)?;

    // Per-head RmsNorm then transpose to [B, H, L, D]
    let mut queries = attn
        .q_norm
        .forward(queries_pre)?
        .transpose_axes(&[0, 2, 1, 3])?;
    let mut keys = attn
        .k_norm
        .forward(&keys_raw.reshape(&[b, l, attn.num_key_value_heads, -1])?)?
        .transpose_axes(&[0, 2, 1, 3])?;
    let values = values_raw
        .reshape(&[b, l, attn.num_key_value_heads, -1])?
        .transpose_axes(&[0, 2, 1, 3])?;

    // Apply RoPE at CUSTOM positions using rope_dynamic
    tracing::debug!(
        "forward_attention_sparse: queries.shape={:?}, keys.shape={:?}, positions.shape={:?}",
        queries.shape(),
        keys.shape(),
        positions.shape()
    );
    let (queries_with_rope, keys_with_rope) =
        match attn.apply_rope_at_positions(&queries, &keys, positions) {
            Ok(result) => {
                tracing::debug!("rope_dynamic succeeded");
                result
            }
            Err(e) => {
                tracing::error!("rope_dynamic failed: {:?}", e);
                return Err(e);
            }
        };
    queries = queries_with_rope;
    keys = keys_with_rope;

    // Update cache with custom-positioned keys/values
    let (cached_keys, cached_values) = cache.update_and_fetch(keys, values)?;
    let final_keys = cached_keys;
    let final_values = cached_values;

    // Compute attention
    let output = crate::utils::scaled_dot_product_attention(
        queries,
        final_keys,
        final_values,
        attn.scale,
        None, // No mask needed for sparse prefill
    )?
    .transpose_axes(&[0, 2, 1, 3])?
    .reshape(&[b, l, -1])?;

    let gated = output.multiply(nn::sigmoid(&gate)?)?;
    attn.o_proj.forward(&gated)
}

impl Qwen3NextCausalLM {
    /// Forward pass with custom `RoPE` positions for sparse prefill.
    ///
    /// This method applies `RoPE` at arbitrary (non-contiguous) positions using
    /// `rope_dynamic`, enabling sparse prefill where only selected tokens are processed.
    ///
    /// # Arguments
    /// * `inputs` - Selected tokens [B, N] where N = number of selected tokens
    /// * `positions` - Original positions for each selected token [N]
    /// * `kv_cache` - KV cache to update
    ///
    /// # Returns
    /// Hidden states [B, N, D] with `RoPE` applied at custom positions
    pub fn forward_hidden_sparse(
        &mut self,
        inputs: &Array,
        positions: &Array,
        kv_cache: &mut Vec<Option<LayerCache>>,
    ) -> Result<Array, Exception> {
        let mut h = self.model.embed_tokens.forward(inputs)?;

        if kv_cache.is_empty() {
            *kv_cache = self.make_cache();
        }

        if kv_cache.len() != self.model.layers.len() {
            return Err(Exception::custom(format!(
                "cache length ({}) must match num layers ({})",
                kv_cache.len(),
                self.model.layers.len()
            )));
        }

        // Process each layer with custom `RoPE` positions
        for (layer, layer_cache) in self.model.layers.iter_mut().zip(kv_cache.iter_mut()) {
            let cache = layer_cache
                .as_mut()
                .ok_or_else(|| Exception::custom("Layer cache is None"))?;

            let normed = layer.input_layernorm.forward(&h)?;
            let r = if layer.is_linear {
                // Linear attention (GatedDeltaNet) - standard forward
                let attn = layer
                    .linear_attn
                    .as_mut()
                    .ok_or_else(|| Exception::custom("linear_attn missing"))?;
                let LayerCache::Arrays(ssm_cache) = cache else {
                    return Err(Exception::custom("Expected ArraysCache"));
                };
                attn.forward(&normed, None, ssm_cache)?
            } else {
                // Full attention - use custom `RoPE` positions
                let attn = layer
                    .self_attn
                    .as_mut()
                    .ok_or_else(|| Exception::custom("self_attn missing"))?;
                let LayerCache::KV(layer_kv) = cache else {
                    return Err(Exception::custom("Expected KVCache"));
                };

                // Apply custom RoPE at specified positions
                forward_attention_sparse(attn, &normed, positions, layer_kv)?
            };

            let h2 = h.add(r)?;
            let normed_post = layer.post_attention_layernorm.forward(&h2)?;
            let mlp_out = layer.mlp.forward(&normed_post)?;
            h = h2.add(mlp_out)?;
        }

        self.model.norm.forward(&h)
    }
}
