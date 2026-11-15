//! The public causal language model and its forward, prefill, and MTP decode paths.

use super::{
    cache::{ArraysCache, LayerCache},
    config::Qwen3NextModelArgs,
    decoder::DecoderLayer,
    linear::{QEmbedding, QLinear},
    mlp::consumed_quantization_paths,
    mtp::{DenseMtpHead, MoeMtpHead, MtpHead},
};
use crate::{
    batched_cache::HybridLayer,
    cache::{KeyValueCache, SteppingKeyValueCache},
    utils::{AttentionMask, create_causal_mask},
};
use mlx_rs::{
    Array,
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::Module,
    nn,
    ops::{self, indexing::IndexOp},
};

#[derive(Debug, Clone, ModuleParameters)]
pub(super) struct Qwen3NextInner {
    #[param]
    pub(super) embed_tokens: QEmbedding,
    #[param]
    pub(super) layers: Vec<DecoderLayer>,
    #[param]
    pub(super) norm: nn::RmsNorm,
}

impl Qwen3NextInner {
    fn new(
        args: &Qwen3NextModelArgs,
        ql: i32,
        qb: i32,
        embed_ql: i32,
        embed_qb: i32,
    ) -> Result<Self, Exception> {
        let layers = (0..args.num_hidden_layers)
            .map(|i| DecoderLayer::new(args, i, ql, qb))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            embed_tokens: QEmbedding::new(embed_ql, embed_qb)?,
            layers,
            norm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
        })
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub struct Qwen3NextCausalLM {
    pub args: Qwen3NextModelArgs,
    #[param]
    pub(super) model: Qwen3NextInner,
    #[param]
    pub(super) lm_head: Option<QLinear>,
    #[param]
    mtp: Option<MtpHead>,
    #[param]
    dense_mtp: Option<DenseMtpHead>,
    #[param]
    moe_mtp: Option<MoeMtpHead>,
}

impl Qwen3NextCausalLM {
    pub fn new(args: Qwen3NextModelArgs) -> Result<Self, Exception> {
        if args.full_attention_interval <= 0 {
            return Err(Exception::custom("full_attention_interval must be > 0"));
        }
        if args.linear_num_key_heads <= 0 || args.linear_num_value_heads <= 0 {
            return Err(Exception::custom("linear_num_*_heads must be > 0"));
        }
        if args.linear_conv_kernel_dim <= 0 {
            return Err(Exception::custom("linear_conv_kernel_dim must be > 0"));
        }

        if let Some(settings) = args.quantization.as_ref() {
            settings
                .assert_all_overrides_consumed(&consumed_quantization_paths(&args))
                .map_err(Exception::custom)?;
        }

        let ql = args.quantization.as_ref().map_or(64, |q| q.group_size);
        let qb = args.quantization.as_ref().map_or(4, |q| q.bits);

        let embed_quant = args
            .quantization
            .as_ref()
            .map(|settings| settings.resolve("model.embed_tokens"));
        let embed_ql = embed_quant.map_or(ql, crate::quant_config::TensorQuant::group_size);
        let embed_qb = embed_quant.map_or(qb, crate::quant_config::TensorQuant::bits);
        let model = Qwen3NextInner::new(&args, ql, qb, embed_ql, embed_qb)?;
        let lm_head_quant = args
            .quantization
            .as_ref()
            .map(|settings| settings.resolve("lm_head"));
        let lm_head_ql = lm_head_quant.map_or(ql, crate::quant_config::TensorQuant::group_size);
        let lm_head_qb = lm_head_quant.map_or(qb, crate::quant_config::TensorQuant::bits);
        let lm_head = if args.tie_word_embeddings {
            None
        } else {
            Some(QLinear::new(lm_head_ql, lm_head_qb)?)
        };
        let mtp = (args.mtp_num_hidden_layers > 0 && !args.use_dense_mtp && !args.use_moe_mtp)
            .then(|| MtpHead::new(&args, ql, qb))
            .transpose()?;
        let dense_mtp = (args.mtp_num_hidden_layers > 0 && args.use_dense_mtp)
            .then(|| DenseMtpHead::new(&args))
            .transpose()?;
        let moe_mtp = (args.mtp_num_hidden_layers > 0 && args.use_moe_mtp)
            .then(|| MoeMtpHead::new(&args, ql, qb))
            .transpose()?;

        Ok(Self {
            args,
            model,
            lm_head,
            mtp,
            dense_mtp,
            moe_mtp,
        })
    }

    /// Create the per-layer cache vector.
    pub fn make_cache(&self) -> Vec<Option<LayerCache>> {
        self.model
            .layers
            .iter()
            .map(|layer| {
                if layer.is_linear {
                    Some(LayerCache::Arrays(ArraysCache::new()))
                } else {
                    Some(LayerCache::KV(SteppingKeyValueCache::new()))
                }
            })
            .collect()
    }

    /// Create a hybrid cache with `TurboQuant` on the full-attention KV layers.
    ///
    /// Linear-attention (SSM/GDN) layers get a plain `ArraysCache`; full-attention
    /// layers get a `SteppingKeyValueCache` with `TurboQuant` storage. This matches
    /// the selective compression strategy used by other `TurboQuant` implementations.
    pub fn make_cache_turbo(
        &self,
        kv_cache_config: crate::turboquant::KvCacheConfig,
    ) -> Result<Vec<Option<LayerCache>>, mlx_rs::error::Exception> {
        let n_layers = self.model.layers.len();
        let dense_tail = usize::from(kv_cache_config.adaptive_dense_layers);
        self.model
            .layers
            .iter()
            .enumerate()
            .map(|(i, layer)| {
                if layer.is_linear {
                    Ok(Some(LayerCache::Arrays(ArraysCache::new())))
                } else if dense_tail > 0 && i >= n_layers.saturating_sub(dense_tail) {
                    // Layer-adaptive: final layers stay dense for quality
                    Ok(Some(LayerCache::KV(SteppingKeyValueCache::new())))
                } else {
                    Ok(Some(LayerCache::KV(SteppingKeyValueCache::new_turbo(
                        kv_cache_config,
                        self.args.num_key_value_heads,
                        self.args.head_dim,
                    )?)))
                }
            })
            .collect()
    }

    /// Forward pass returning raw hidden states (before final `RMSNorm`).
    ///
    /// Used internally by `forward_hidden` (which adds norm) and
    /// `forward_with_hidden` (which needs raw states for MTP).
    #[allow(non_snake_case, clippy::too_many_lines)]
    fn forward_raw_hidden(
        &mut self,
        inputs: &Array,
        _mask: Option<&Array>,
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

        // Create attention mask for full-attention layers
        let shape = h.shape();
        let T = *shape
            .get(1)
            .ok_or_else(|| Exception::custom("Hidden state must have >= 2 dims"))?;

        let fa_mask: Option<AttentionMask> = if T > 1 {
            let kv_offset = kv_cache
                .iter()
                .find_map(|lc| match lc.as_ref()? {
                    LayerCache::KV(kv) => Some(kv.offset()),
                    LayerCache::Arrays(_) => None,
                })
                .unwrap_or(0);

            if kv_offset > 0 {
                Some(AttentionMask::Array(create_causal_mask(
                    T,
                    Some(kv_offset),
                )?))
            } else {
                Some(AttentionMask::Causal)
            }
        } else {
            None
        };

        // HIGGS_PROFILE=1: instrument per-layer timing with eval barriers.
        // Samples layers 0-3 (3 GDN + 1 FA), extrapolates to all 64 layers.
        let profiling = std::env::var("HIGGS_PROFILE").is_ok_and(|v| v == "1") && T == 1;
        let mut prof_gdn_attn_ns: u128 = 0;
        let mut prof_gdn_mlp_ns: u128 = 0;
        let mut prof_fa_attn_ns: u128 = 0;
        let mut prof_fa_mlp_ns: u128 = 0;
        let mut prof_gdn_samples: u32 = 0;
        let mut prof_fa_samples: u32 = 0;

        for (layer_idx, (layer, layer_cache)) in self
            .model
            .layers
            .iter_mut()
            .zip(kv_cache.iter_mut())
            .enumerate()
        {
            let cache = layer_cache
                .as_mut()
                .ok_or_else(|| Exception::custom("Layer cache is None"))?;
            let mask = if layer.is_linear {
                None
            } else {
                fa_mask.as_ref()
            };

            let sample_this = profiling && layer_idx < 4;
            let t0 = if sample_this {
                mlx_rs::transforms::eval([&h])?;
                Some(std::time::Instant::now())
            } else {
                None
            };

            let normed = layer.input_layernorm.forward(&h)?;
            let r = if layer.is_linear {
                let attn = layer
                    .linear_attn
                    .as_mut()
                    .ok_or_else(|| Exception::custom("linear_attn missing"))?;
                let LayerCache::Arrays(ssm_cache) = cache else {
                    return Err(Exception::custom("Expected ArraysCache"));
                };
                attn.forward(&normed, mask, ssm_cache)?
            } else {
                let attn = layer
                    .self_attn
                    .as_mut()
                    .ok_or_else(|| Exception::custom("self_attn missing"))?;
                let LayerCache::KV(layer_kv) = cache else {
                    return Err(Exception::custom("Expected KVCache"));
                };
                attn.forward(&normed, mask, layer_kv)?
            };

            if let Some(start) = t0 {
                let h2 = h.add(r)?;
                let normed_post = layer.post_attention_layernorm.forward(&h2)?;
                mlx_rs::transforms::eval([&h2])?;
                let attn_ns = start.elapsed().as_nanos();
                let t1 = std::time::Instant::now();
                let mlp_out = layer.mlp.forward(&normed_post)?;
                h = h2.add(mlp_out)?;
                mlx_rs::transforms::eval([&h])?;
                let mlp_ns = t1.elapsed().as_nanos();

                if layer.is_linear {
                    prof_gdn_attn_ns += attn_ns;
                    prof_gdn_mlp_ns += mlp_ns;
                    prof_gdn_samples += 1;
                } else {
                    prof_fa_attn_ns += attn_ns;
                    prof_fa_mlp_ns += mlp_ns;
                    prof_fa_samples += 1;
                }
            } else {
                let h2 = h.add(r)?;
                let normed_post = layer.post_attention_layernorm.forward(&h2)?;
                let mlp_out = layer.mlp.forward(&normed_post)?;
                h = h2.add(mlp_out)?;
            }

            // Eval every 8 layers during long prefill chunks to bound lazy
            // graph size. Short speculative verifier windows are intentionally
            // left fused; otherwise MTP pays several eval barriers per cycle.
            if should_eval_between_prefill_layers(T, layer_idx) {
                mlx_rs::transforms::eval([&h])?;
            }
        }

        if profiling && prof_gdn_samples > 0 && prof_fa_samples > 0 {
            #[allow(clippy::as_conversions, clippy::cast_precision_loss)]
            {
                let gdn_attn_avg = prof_gdn_attn_ns as f64 / f64::from(prof_gdn_samples);
                let gdn_mlp_avg = prof_gdn_mlp_ns as f64 / f64::from(prof_gdn_samples);
                let fa_attn_avg = prof_fa_attn_ns as f64 / f64::from(prof_fa_samples);
                let fa_mlp_avg = prof_fa_mlp_ns as f64 / f64::from(prof_fa_samples);
                let est_total =
                    (gdn_attn_avg + gdn_mlp_avg).mul_add(48.0, (fa_attn_avg + fa_mlp_avg) * 16.0);
                tracing::info!(
                    gdn_attn_ms = format!("{:.2}", gdn_attn_avg / 1e6),
                    gdn_mlp_ms = format!("{:.2}", gdn_mlp_avg / 1e6),
                    fa_attn_ms = format!("{:.2}", fa_attn_avg / 1e6),
                    fa_mlp_ms = format!("{:.2}", fa_mlp_avg / 1e6),
                    est_total_ms = format!("{:.1}", est_total / 1e6),
                    "PROFILE: per-layer avg (×48 GDN + ×16 FA)"
                );
            }
        }

        Ok(h)
    }

    /// Batched decode: one token for each of the `B` sequences in `cache`.
    /// `inputs` is `[B, 1]`; `mask` is the batch's attention padding mask.
    /// Returns `[B, 1, vocab]` logits.
    pub fn forward_batched(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut [HybridLayer],
    ) -> Result<Array, Exception> {
        if cache.len() != self.model.layers.len() {
            return Err(Exception::custom(format!(
                "cache length ({}) must match num layers ({})",
                cache.len(),
                self.model.layers.len()
            )));
        }
        if !matches!(inputs.shape(), [_, 1]) {
            return Err(Exception::custom("batched decode inputs must be [B, 1]"));
        }
        let attention_mask = mask.map(|padding_mask| AttentionMask::Array(padding_mask.clone()));

        let mut h = self.model.embed_tokens.forward(inputs)?;
        for (layer, layer_cache) in self.model.layers.iter_mut().zip(cache.iter_mut()) {
            let normed = layer.input_layernorm.forward(&h)?;
            let r = match layer_cache {
                HybridLayer::Recurrent(recurrent) => layer
                    .linear_attn
                    .as_mut()
                    .ok_or_else(|| Exception::custom("linear_attn missing"))?
                    .forward(&normed, None, recurrent.state_mut())?,
                HybridLayer::Attention(attention) => layer
                    .self_attn
                    .as_mut()
                    .ok_or_else(|| Exception::custom("self_attn missing"))?
                    .forward(&normed, attention_mask.as_ref(), attention)?,
            };
            let h2 = h.add(r)?;
            let normed_post = layer.post_attention_layernorm.forward(&h2)?;
            let mlp_out = layer.mlp.forward(&normed_post)?;
            h = h2.add(mlp_out)?;
        }

        let normed_hidden = self.model.norm.forward(&h)?;
        match self.lm_head.as_ref() {
            Some(head) => head.forward(&normed_hidden),
            None => self.model.embed_tokens.as_linear(&normed_hidden),
        }
    }

    /// Forward pass returning hidden states (after final `RMSNorm`, before LM head).
    #[allow(non_snake_case)]
    pub fn forward_hidden(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        kv_cache: &mut Vec<Option<LayerCache>>,
    ) -> Result<Array, Exception> {
        let h = self.forward_raw_hidden(inputs, mask, kv_cache)?;
        self.model.norm.forward(&h)
    }

    /// Forward pass producing logits for the **last position only**.
    ///
    /// During inference only the last token's logits are sampled, so we
    /// slice hidden states before the `lm_head` projection. This avoids a
    /// full `quantized_matmul(vocab, hidden)` on T-1 discarded positions.
    /// Returns shape `[B, 1, vocab]`.
    #[allow(non_snake_case)]
    pub fn forward(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        kv_cache: &mut Vec<Option<LayerCache>>,
    ) -> Result<Array, Exception> {
        let h = self.forward_hidden(inputs, mask, kv_cache)?;
        let h_last = h.index((.., -1.., ..)); // [B, 1, hidden]

        match self.lm_head.as_ref() {
            Some(head) => head.forward(&h_last),
            None => self.model.embed_tokens.as_linear(&h_last),
        }
    }

    /// Forward pass producing logits for **only the last token**.
    ///
    /// During prefill we only need the last token's logits for sampling.
    /// Computing the full `[B, L, vocab]` LM head is wasteful for large vocab.
    /// This method computes hidden states for all tokens (needed for KV cache),
    /// then applies the LM head only to the last token.
    #[allow(non_snake_case)]
    pub fn forward_last_token(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        kv_cache: &mut Vec<Option<LayerCache>>,
    ) -> Result<Array, Exception> {
        let h = self.forward_hidden(inputs, mask, kv_cache)?;
        let last_slice = h.index((.., -1, ..));
        // Reshape to [B, 1, D] so the LM head produces [B, 1, vocab]
        let shape = last_slice.shape();
        let batch = *shape
            .first()
            .ok_or_else(|| Exception::custom("forward_last_token: empty shape"))?;
        let last_dim = *shape
            .last()
            .ok_or_else(|| Exception::custom("forward_last_token: empty shape"))?;
        let last_h = last_slice.reshape(&[batch, 1, last_dim])?;
        match self.lm_head.as_ref() {
            Some(head) => head.forward(&last_h),
            None => self.model.embed_tokens.as_linear(&last_h),
        }
    }

    /// Chunked prefill: process the prompt in `chunk_size`-token segments
    /// through all layers. Produces identical logits to `forward()` but with
    /// smaller per-dispatch working sets and lower peak memory.
    ///
    /// Only the **last chunk's** logits are returned (shape `[B, chunk_len, vocab]`).
    /// For full-sequence hidden states, use `forward_hidden` directly.
    #[allow(non_snake_case)]
    pub fn forward_chunked(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        kv_cache: &mut Vec<Option<LayerCache>>,
        chunk_size: i32,
    ) -> Result<Array, Exception> {
        let shape = inputs.shape();
        let T = *shape
            .get(1)
            .ok_or_else(|| Exception::custom("Input must have >= 2 dims"))?;

        // If chunk_size covers the whole sequence, just do a normal forward.
        if chunk_size >= T {
            return self.forward(inputs, mask, kv_cache);
        }

        if kv_cache.is_empty() {
            *kv_cache = self.make_cache();
        }

        // Process all chunks except the last through forward_hidden (discard logits).
        // Cache states must be eval'd between chunks so the next chunk reads
        // materialized values (MLX is lazy).
        let mut offset = 0i32;
        while offset + chunk_size < T {
            let chunk = inputs.index((.., offset..offset + chunk_size));
            let h = self.forward_hidden(&chunk, None, kv_cache)?;
            // Eval hidden output + ALL cache states between chunks.
            // Both KV and SSM/conv must be materialized:
            // - SSM/conv: consumed by GDN FFI kernel (requires concrete arrays)
            // - KV: slice_update creates lazy nodes; without eval, nested
            //   updates accumulate and OOM on long sequences
            let mut targets: Vec<&Array> = vec![&h];
            for lc in kv_cache.iter().flatten() {
                match lc {
                    LayerCache::KV(kv) => targets.extend(kv.eval_targets()),
                    LayerCache::Arrays(ac) => {
                        if let Some(ref s) = ac.ssm_state {
                            targets.push(s);
                        }
                        if let Some(ref c) = ac.conv_state {
                            targets.push(c);
                        }
                    }
                }
            }
            mlx_rs::transforms::eval(targets)?;
            offset += chunk_size;
            crate::progress::report_prefill_progress(offset, T);
        }

        // Last chunk: use forward_last_token which efficiently projects only
        // the last position through the LM head.
        let last_chunk = inputs.index((.., offset..));
        let logits = self.forward_last_token(&last_chunk, None, kv_cache)?;
        // The loop only reports up to the final chunk boundary; emit the
        // terminal 100% mark once the last chunk is processed so clients don't
        // stall just short of `total`.
        crate::progress::report_prefill_progress(T, T);
        Ok(logits)
    }

    // -----------------------------------------------------------------------
    // MTP (Multi-Token Prediction) speculative decode
    // -----------------------------------------------------------------------

    /// Whether this model has an MTP head loaded.
    pub const fn has_mtp(&self) -> bool {
        self.mtp.is_some() || self.dense_mtp.is_some() || self.moe_mtp.is_some()
    }

    /// Create a fresh KV cache for the MTP head (one entry per MTP layer).
    /// Returns `None` if the model has no MTP head.
    pub fn make_mtp_cache(&self) -> Option<Vec<SteppingKeyValueCache>> {
        let layer_count = self
            .mtp
            .as_ref()
            .map(|mtp| mtp.layers.len())
            .or_else(|| self.dense_mtp.as_ref().map(|mtp| mtp.layers.len()))
            .or_else(|| self.moe_mtp.as_ref().map(|mtp| mtp.layers.len()))?;
        Some(
            (0..layer_count)
                .map(|_| SteppingKeyValueCache::new())
                .collect(),
        )
    }

    /// Look up the embedding for a token id. Shape: `[1, 1, hidden_size]`.
    pub fn embed_token(&self, token_id: u32) -> Result<Array, Exception> {
        let token_id_i32 =
            i32::try_from(token_id).map_err(|_| Exception::custom("token_id exceeds i32 range"))?;
        let ids = Array::from_slice(&[token_id_i32], &[1, 1]);
        self.model.embed_tokens.forward(&ids)
    }

    fn embed_tokens_from_ids(&self, token_ids: &[u32]) -> Result<Array, Exception> {
        let token_i32s: Vec<i32> = token_ids
            .iter()
            .map(|&token_id| {
                i32::try_from(token_id).map_err(|_| Exception::custom("token_id exceeds i32 range"))
            })
            .collect::<Result<_, _>>()?;
        let len = i32::try_from(token_i32s.len())
            .map_err(|_| Exception::custom("token id batch exceeds i32 range"))?;
        let ids_array = Array::from_slice(&token_i32s, &[1, len]);
        self.model.embed_tokens.forward(&ids_array)
    }

    fn mtp_attention_mask(
        seq_len: i32,
        mtp_cache: &[SteppingKeyValueCache],
    ) -> Result<Option<AttentionMask>, Exception> {
        if seq_len <= 1 {
            return Ok(None);
        }
        let offset = mtp_cache.first().map_or(0, SteppingKeyValueCache::offset);
        if offset > 0 {
            Ok(Some(AttentionMask::Array(create_causal_mask(
                seq_len,
                Some(offset),
            )?)))
        } else {
            Ok(Some(AttentionMask::Causal))
        }
    }

    /// Run the MTP head to produce draft logits for position t+2.
    ///
    /// - `hidden` — backbone hidden state at position t, shape `[B, 1, D]`.
    /// - `next_token_id` — the confirmed next token (t+1).
    /// - `mtp_cache` — per-layer KV cache for the MTP attention.
    ///
    /// Returns the normalized hidden state for the confirmed token position.
    fn mtp_step_hidden(
        &mut self,
        hidden: &Array,
        next_token_id: u32,
        mtp_cache: &mut [SteppingKeyValueCache],
    ) -> Result<Array, Exception> {
        if !self.has_mtp() {
            return Err(Exception::custom("MTP head not loaded"));
        }

        // Compute embedding before mutable borrow of mtp.
        let next_embed = self.embed_token(next_token_id)?;

        // Scope the mutable borrow: run MTP forward, defer lm_head projection.
        if let Some(mtp) = self.mtp.as_mut() {
            let h_norm = mtp.pre_fc_norm_hidden.forward(hidden)?;
            let e_norm = mtp.pre_fc_norm_embedding.forward(&next_embed)?;
            let concat = ops::concatenate(&[&e_norm, &h_norm], -1)?;
            let mut x = mtp.fc.forward(&concat)?;

            for (layer, kv) in mtp.layers.iter_mut().zip(mtp_cache.iter_mut()) {
                let normed = layer.input_layernorm.forward(&x)?;
                let attn_out = layer.self_attn.forward(&normed, None, kv)?;
                let h2 = x.add(attn_out)?;
                let normed_post = layer.post_attention_layernorm.forward(&h2)?;
                let mlp_out = layer.mlp.forward(&normed_post)?;
                x = h2.add(mlp_out)?;
            }

            return mtp.norm.forward(&x);
        }

        // MoE MTP head (Qwen3.6-A3B style): same loop, MoE MLP.
        if let Some(mtp) = self.moe_mtp.as_mut() {
            let h_norm = mtp.pre_fc_norm_hidden.forward(hidden)?;
            let e_norm = mtp.pre_fc_norm_embedding.forward(&next_embed)?;
            let concat = ops::concatenate(&[&e_norm, &h_norm], -1)?;
            let mut x = mtp.fc.forward(&concat)?;

            for (layer, kv) in mtp.layers.iter_mut().zip(mtp_cache.iter_mut()) {
                let normed = layer.input_layernorm.forward(&x)?;
                let attn_out = layer.self_attn.forward(&normed, None, kv)?;
                let h2 = x.add(attn_out)?;
                let normed_post = layer.post_attention_layernorm.forward(&h2)?;
                let mlp_out = layer.mlp.forward(&normed_post)?;
                x = h2.add(mlp_out)?;
            }

            return mtp.norm.forward(&x);
        }

        let mtp = self
            .dense_mtp
            .as_mut()
            .ok_or_else(|| Exception::custom("MTP head not loaded"))?;

        let h_norm = mtp.pre_fc_norm_hidden.forward(hidden)?;
        let e_norm = mtp.pre_fc_norm_embedding.forward(&next_embed)?;
        let concat = ops::concatenate(&[&e_norm, &h_norm], -1)?;
        let mut x = mtp.fc.forward(&concat)?;

        for (layer, kv) in mtp.layers.iter_mut().zip(mtp_cache.iter_mut()) {
            let normed = layer.input_layernorm.forward(&x)?;
            let attn_out = layer.self_attn.forward(&normed, None, kv)?;
            let h2 = x.add(attn_out)?;
            let normed_post = layer.post_attention_layernorm.forward(&h2)?;
            let mlp_out = layer.mlp.forward(&normed_post)?;
            x = h2.add(mlp_out)?;
        }

        mtp.norm.forward(&x)
    }

    /// Run the MTP head to produce draft logits for position t+2.
    ///
    /// - `hidden` — backbone hidden state at position t, shape `[B, 1, D]`.
    /// - `next_token_id` — the confirmed next token (t+1).
    /// - `mtp_cache` — per-layer KV cache for the MTP attention.
    ///
    /// Returns draft logits of shape `[B, 1, vocab]`.
    pub fn mtp_draft(
        &mut self,
        hidden: &Array,
        next_token_id: u32,
        mtp_cache: &mut [SteppingKeyValueCache],
    ) -> Result<Array, Exception> {
        let (_, logits) = self.mtp_draft_with_hidden(hidden, next_token_id, mtp_cache)?;
        Ok(logits)
    }

    /// Run the MTP head and return both its hidden state and draft logits.
    ///
    /// The hidden state is useful for chained speculative drafting. Final
    /// committed MTP cache state is still replayed with backbone hidden states
    /// after verification.
    pub fn mtp_draft_with_hidden(
        &mut self,
        hidden: &Array,
        next_token_id: u32,
        mtp_cache: &mut [SteppingKeyValueCache],
    ) -> Result<(Array, Array), Exception> {
        let normed = self.mtp_step_hidden(hidden, next_token_id, mtp_cache)?;

        // Now lm_head/embed_tokens can be borrowed immutably.
        let logits = match self.lm_head.as_ref() {
            Some(head) => head.forward(&normed),
            None => self.model.embed_tokens.as_linear(&normed),
        }?;
        Ok((normed, logits))
    }

    /// Advance the MTP cache for a newly accepted token without computing logits.
    pub fn mtp_advance(
        &mut self,
        hidden: &Array,
        next_token_id: u32,
        mtp_cache: &mut [SteppingKeyValueCache],
    ) -> Result<(), Exception> {
        let _ = self.mtp_step_hidden(hidden, next_token_id, mtp_cache)?;
        Ok(())
    }

    /// Advance the MTP cache for multiple accepted tokens in one sequence pass.
    pub fn mtp_advance_many(
        &mut self,
        hidden: &Array,
        next_token_ids: &[u32],
        mtp_cache: &mut [SteppingKeyValueCache],
    ) -> Result<(), Exception> {
        if next_token_ids.is_empty() {
            return Ok(());
        }
        if !self.has_mtp() {
            return Err(Exception::custom("MTP head not loaded"));
        }

        let seq_len = i32::try_from(next_token_ids.len())
            .map_err(|_| Exception::custom("MTP advance token batch exceeds i32 range"))?;
        let expected_layers = self
            .mtp
            .as_ref()
            .map(|mtp| mtp.layers.len())
            .or_else(|| self.dense_mtp.as_ref().map(|mtp| mtp.layers.len()))
            .or_else(|| self.moe_mtp.as_ref().map(|mtp| mtp.layers.len()))
            .ok_or_else(|| Exception::custom("MTP head not loaded"))?;
        Self::validate_mtp_advance_many_inputs(hidden, mtp_cache, expected_layers, seq_len)?;

        let next_embed = self.embed_tokens_from_ids(next_token_ids)?;
        let mask = Self::mtp_attention_mask(seq_len, mtp_cache)?;
        let mask_ref = mask.as_ref();

        if let Some(mtp) = self.mtp.as_mut() {
            let h_norm = mtp.pre_fc_norm_hidden.forward(hidden)?;
            let e_norm = mtp.pre_fc_norm_embedding.forward(&next_embed)?;
            let concat = ops::concatenate(&[&e_norm, &h_norm], -1)?;
            let mut x = mtp.fc.forward(&concat)?;

            for (layer, kv) in mtp.layers.iter_mut().zip(mtp_cache.iter_mut()) {
                let normed = layer.input_layernorm.forward(&x)?;
                let attn_out = layer.self_attn.forward(&normed, mask_ref, kv)?;
                let h2 = x.add(attn_out)?;
                let normed_post = layer.post_attention_layernorm.forward(&h2)?;
                let mlp_out = layer.mlp.forward(&normed_post)?;
                x = h2.add(mlp_out)?;
            }

            let _ = mtp.norm.forward(&x)?;
            return Ok(());
        }

        // MoE MTP head (Qwen3.6-A3B style): same loop, MoE MLP.
        if let Some(mtp) = self.moe_mtp.as_mut() {
            let h_norm = mtp.pre_fc_norm_hidden.forward(hidden)?;
            let e_norm = mtp.pre_fc_norm_embedding.forward(&next_embed)?;
            let concat = ops::concatenate(&[&e_norm, &h_norm], -1)?;
            let mut x = mtp.fc.forward(&concat)?;

            for (layer, kv) in mtp.layers.iter_mut().zip(mtp_cache.iter_mut()) {
                let normed = layer.input_layernorm.forward(&x)?;
                let attn_out = layer.self_attn.forward(&normed, mask_ref, kv)?;
                let h2 = x.add(attn_out)?;
                let normed_post = layer.post_attention_layernorm.forward(&h2)?;
                let mlp_out = layer.mlp.forward(&normed_post)?;
                x = h2.add(mlp_out)?;
            }

            let _ = mtp.norm.forward(&x)?;
            return Ok(());
        }

        let mtp = self
            .dense_mtp
            .as_mut()
            .ok_or_else(|| Exception::custom("MTP head not loaded"))?;

        let h_norm = mtp.pre_fc_norm_hidden.forward(hidden)?;
        let e_norm = mtp.pre_fc_norm_embedding.forward(&next_embed)?;
        let concat = ops::concatenate(&[&e_norm, &h_norm], -1)?;
        let mut x = mtp.fc.forward(&concat)?;

        for (layer, kv) in mtp.layers.iter_mut().zip(mtp_cache.iter_mut()) {
            let normed = layer.input_layernorm.forward(&x)?;
            let attn_out = layer.self_attn.forward(&normed, mask_ref, kv)?;
            let h2 = x.add(attn_out)?;
            let normed_post = layer.post_attention_layernorm.forward(&h2)?;
            let mlp_out = layer.mlp.forward(&normed_post)?;
            x = h2.add(mlp_out)?;
        }

        let _ = mtp.norm.forward(&x)?;
        Ok(())
    }

    fn validate_mtp_advance_many_inputs(
        hidden: &Array,
        mtp_cache: &[SteppingKeyValueCache],
        expected_layers: usize,
        seq_len: i32,
    ) -> Result<(), Exception> {
        Self::validate_mtp_advance_many_shape(
            hidden.shape(),
            mtp_cache.len(),
            expected_layers,
            seq_len,
        )
    }

    pub(super) fn validate_mtp_advance_many_shape(
        hidden_shape: &[i32],
        cache_layers: usize,
        expected_layers: usize,
        seq_len: i32,
    ) -> Result<(), Exception> {
        if cache_layers != expected_layers {
            return Err(Exception::custom(format!(
                "mtp_cache length ({cache_layers}) must match MTP layer count ({expected_layers})"
            )));
        }

        let hidden_seq_len = *hidden_shape
            .get(1)
            .ok_or_else(|| Exception::custom("hidden must be [B, T, D]"))?;
        if hidden_seq_len != seq_len {
            return Err(Exception::custom(format!(
                "hidden sequence length ({hidden_seq_len}) must match next_token_ids length ({seq_len})"
            )));
        }

        Ok(())
    }

    /// Forward pass returning BOTH raw hidden states and logits for all positions.
    ///
    /// Used by MTP speculative decode: the verify pass needs **raw** (pre-norm)
    /// hidden states for the next MTP draft, and logits for acceptance check.
    /// Returns `(raw_hidden, logits)` where both have shape `[B, T, ...]`.
    /// The raw hidden states have NOT been through the final `RMSNorm` — the MTP
    /// head applies its own `pre_fc_norm_hidden` instead.
    #[allow(non_snake_case)]
    pub fn forward_with_hidden(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        kv_cache: &mut Vec<Option<LayerCache>>,
    ) -> Result<(Array, Array), Exception> {
        let h_raw = self.forward_raw_hidden(inputs, mask, kv_cache)?;
        let h_normed = self.model.norm.forward(&h_raw)?;
        let logits = match self.lm_head.as_ref() {
            Some(head) => head.forward(&h_normed)?,
            None => self.model.embed_tokens.as_linear(&h_normed)?,
        };
        Ok((h_raw, logits))
    }
}

impl Qwen3NextCausalLM {
    /// Compute logits from hidden states.
    ///
    /// This is used after sparse forward pass to get final logits.
    pub fn compute_logits(&self, hidden: &Array) -> Result<Array, Exception> {
        self.lm_head.as_ref().map_or_else(
            || self.model.embed_tokens.as_linear(hidden),
            |head| head.forward(hidden),
        )
    }
}

const PREFILL_LAYER_EVAL_INTERVAL: usize = 8;
const PREFILL_LAYER_EVAL_MIN_SEQ_LEN: i32 = 17;

const fn should_eval_between_prefill_layers(seq_len: i32, layer_idx: usize) -> bool {
    seq_len >= PREFILL_LAYER_EVAL_MIN_SEQ_LEN
        && (layer_idx + 1).is_multiple_of(PREFILL_LAYER_EVAL_INTERVAL)
}

#[cfg(test)]
mod prefill_eval_tests {
    use super::should_eval_between_prefill_layers;

    #[test]
    fn skips_layer_eval_barriers_for_short_speculative_windows() {
        assert!(!should_eval_between_prefill_layers(3, 7));
        assert!(!should_eval_between_prefill_layers(8, 7));
    }

    #[test]
    fn keeps_layer_eval_barriers_for_long_prefill_chunks() {
        assert!(should_eval_between_prefill_layers(128, 7));
        assert!(!should_eval_between_prefill_layers(128, 6));
    }
}
