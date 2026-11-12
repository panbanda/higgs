//! KV cache that holds several sequences in one set of batched arrays.
//!
//! Batched decode attends over `[B, H, L, D]` keys/values every step. Keeping
//! those arrays batched between steps means a step only appends one token per
//! row; the alternative (re-stacking per-request caches each step) copies the
//! whole KV state of every request per layer per token.
//!
//! Rows are **left-padded** to a common length, so every row's next token lands
//! at the same index and one `slice_update` appends for the whole batch.
//! Padding slots are masked out of attention. Each row is rotated at its true
//! position for `RoPE`, so cached keys stay valid when a row moves between
//! batches.

use mlx_rs::{
    Array,
    error::Exception,
    ops::{self, indexing::IndexOp},
};

use crate::{
    AnyCache,
    cache::{KeyValueCache, KvCacheView, SteppingKeyValueCache},
    qwen3_next::{ArraysCache, LayerCache, chronological_conv_history},
};

/// Per-layer caches for one sequence, as used by non-batched decode.
pub type KvRow = Vec<Option<SteppingKeyValueCache>>;

/// One layer of a [`BatchedKvCache`].
///
/// Keys and values are shaped `[B, H, capacity, D]`, with each row's left
/// padding tracked alongside. It implements [`KeyValueCache`], so a model's
/// ordinary forward pass runs batched when handed these instead of
/// per-sequence caches.
#[derive(Debug)]
pub struct BatchedLayerCache {
    storage: SteppingKeyValueCache,
    padding: Vec<i32>,
}

impl KeyValueCache for BatchedLayerCache {
    fn offset(&self) -> i32 {
        self.storage.offset()
    }

    fn rope_positions(&self) -> Option<Array> {
        let padded_len = self.storage.offset();
        let positions: Vec<i32> = self
            .padding
            .iter()
            .map(|&padding| padded_len - padding)
            .collect();
        let batch = i32::try_from(positions.len()).ok()?;
        Some(Array::from_slice(&positions, &[batch]))
    }

    fn eval_targets(&self) -> Vec<&Array> {
        self.storage.eval_targets()
    }

    fn max_size(&self) -> Option<i32> {
        None
    }

    fn update_and_view(&mut self, keys: Array, values: Array) -> Result<KvCacheView, Exception> {
        self.storage.update_and_view(keys, values)
    }
}

#[derive(Debug)]
pub struct BatchedKvCache {
    layers: Vec<Option<BatchedLayerCache>>,
    /// Left padding of each row: slots before the row's first real token.
    padding: Vec<i32>,
}

impl BatchedKvCache {
    /// Merge per-sequence caches into one batch. Every row must be a dense
    /// (non-quantized, non-latent) cache with the same number of layers.
    // Callers hand over ownership of their per-sequence caches on purpose.
    #[allow(clippy::needless_pass_by_value)]
    pub fn from_rows(rows: Vec<KvRow>) -> Result<Self, Exception> {
        let lengths: Vec<i32> = rows
            .iter()
            .map(|row| {
                row.first()
                    .and_then(Option::as_ref)
                    .map_or(0, KeyValueCache::offset)
            })
            .collect();
        let padded_len = lengths.iter().copied().max().unwrap_or(0);
        let padding: Vec<i32> = lengths.iter().map(|&length| padded_len - length).collect();
        let num_layers = rows.first().map_or(0, Vec::len);
        if rows.iter().any(|row| row.len() != num_layers) {
            return Err(Exception::custom(
                "batched cache rows must have the same layer count",
            ));
        }

        let mut layers = Vec::with_capacity(num_layers);
        for layer_index in 0..num_layers {
            let mut caches = Vec::with_capacity(rows.len());
            for row in &rows {
                caches.push(
                    row.get(layer_index)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| Exception::custom("batched cache row has an empty layer"))?,
                );
            }
            layers.push(Some(merge_kv_layer(&caches, &lengths, &padding)?));
        }
        Ok(Self { layers, padding })
    }

    /// An empty batch about to receive a left-padded prefill of `steps`
    /// tokens per row, where `lengths` are the rows' true token counts.
    pub fn for_prefill(num_layers: usize, lengths: &[i32], steps: i32) -> Self {
        let padding: Vec<i32> = lengths.iter().map(|&length| steps - length).collect();
        Self {
            layers: (0..num_layers)
                .map(|_| {
                    Some(BatchedLayerCache {
                        storage: SteppingKeyValueCache::new(),
                        padding: padding.clone(),
                    })
                })
                .collect(),
            padding,
        }
    }

    /// Split back into per-sequence caches, dropping each row's padding.
    pub fn into_rows(self) -> Result<Vec<KvRow>, Exception> {
        let padded_len = self.padded_len();
        let mut rows: Vec<KvRow> = self
            .padding
            .iter()
            .map(|_| Vec::with_capacity(self.layers.len()))
            .collect();
        for slot in &self.layers {
            let layer = slot
                .as_ref()
                .ok_or_else(|| Exception::custom("batched cache has an empty layer"))?;
            for (row, split) in
                rows.iter_mut()
                    .zip(split_kv_layer(layer, &self.padding, padded_len)?)
            {
                row.push(Some(split));
            }
        }
        Ok(rows)
    }

    pub const fn batch_size(&self) -> usize {
        self.padding.len()
    }

    pub const fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// True token count of each row.
    pub fn lengths(&self) -> Vec<i32> {
        let padded_len = self.padded_len();
        self.padding
            .iter()
            .map(|&row_padding| padded_len - row_padding)
            .collect()
    }

    /// Tokens held per row including padding (every layer agrees).
    fn padded_len(&self) -> i32 {
        self.layers
            .first()
            .and_then(Option::as_ref)
            .map_or(0, KeyValueCache::offset)
    }

    /// The per-layer caches in the shape a model's forward pass takes.
    pub const fn layers_mut(&mut self) -> &mut Vec<Option<BatchedLayerCache>> {
        &mut self.layers
    }

    /// Boolean `[B, 1, 1, L + new_tokens]` mask (true = attend) covering the
    /// cache contents plus `new_tokens` about to be appended. `None` when no
    /// row has padding, so equal-length batches skip masking entirely.
    pub fn attention_mask(&self, new_tokens: i32) -> Result<Option<Array>, Exception> {
        padding_attention_mask(&self.padding, self.padded_len(), new_tokens)
    }

    /// Boolean `[B, 1, T, T]` mask (true = attend) for prefilling prompts
    /// left-padded to `steps` tokens: causal, and blind to each row's padding.
    /// A padding position still attends to itself, because a fully masked
    /// attention row turns into NaN that would leak through the residual
    /// stream into the cached keys and values.
    pub fn prefill_mask(lengths: &[i32], steps: i32) -> Result<Array, Exception> {
        let batch =
            i32::try_from(lengths.len()).map_err(|_| Exception::custom("batch size overflow"))?;
        let row_padding: Vec<i32> = lengths.iter().map(|&length| steps - length).collect();
        let padding_array = Array::from_slice(&row_padding, &[batch, 1, 1]);
        let positions = ops::arange::<_, i32>(None, steps, None)?;
        let queries = positions.reshape(&[1, steps, 1])?;
        let keys = positions.reshape(&[1, 1, steps])?;
        // A query at `q` sees keys in `[min(padding, q), q]`.
        let first_visible = ops::minimum(&padding_array, &queries)?;
        let mask = keys.ge(first_visible)?.logical_and(keys.le(&queries)?)?;
        mask.reshape(&[batch, 1, steps, steps])
    }

    /// Arrays that must be evaluated to materialize the cache state.
    pub fn eval_targets(&self) -> Vec<&Array> {
        self.layers
            .iter()
            .flatten()
            .flat_map(|layer| layer.storage.eval_targets())
            .collect()
    }
}

/// One layer of a [`HybridBatchedCache`].
#[derive(Debug)]
pub enum HybridLayer {
    /// A full-attention layer: left-padded batched keys and values.
    Attention(BatchedLayerCache),
    /// A recurrent layer: convolution and state arrays stacked on the batch axis.
    Recurrent(RecurrentBatch),
}

/// Recurrent (`GatedDeltaNet`) state of every row of a batch.
///
/// The stacked [`ArraysCache`] carries a single `conv_pos` and `offset` for
/// the whole batch, so rows are normalized to one convolution layout when
/// they join (see [`HybridBatchedCache::from_rows`]).
#[derive(Debug)]
pub struct RecurrentBatch {
    state: ArraysCache,
    /// Tokens each row had processed when the batch was formed.
    row_offsets: Vec<i32>,
    /// `state.offset` when the batch was formed; later growth is the number of
    /// steps taken, which every row has taken together.
    formed_offset: i32,
}

impl RecurrentBatch {
    pub const fn state_mut(&mut self) -> &mut ArraysCache {
        &mut self.state
    }

    fn merge(caches: &[&ArraysCache]) -> Result<Self, Exception> {
        let mut conv_rows = Vec::with_capacity(caches.len());
        let mut ssm_rows = Vec::with_capacity(caches.len());
        let mut conv_history_len = None;
        for cache in caches {
            let ssm_state = cache
                .ssm_state
                .as_ref()
                .ok_or_else(|| Exception::custom("batched cache row has no recurrent state"))?;
            ssm_rows.push(ssm_state.clone());
            match cache.conv_state.as_ref() {
                Some(conv_state) => {
                    conv_history_len = conv_state.shape().get(1).copied();
                    conv_rows.push(chronological_conv_history(
                        conv_state,
                        cache.conv_pos,
                        cache.offset,
                    )?);
                }
                None => {
                    return Err(Exception::custom(
                        "batched cache row has no convolution state",
                    ));
                }
            }
        }
        // After normalization every row's newest entry is the last slot, and
        // slots a row never filled are zero. Reading all `history_len` slots
        // is therefore equivalent to each row's own `min(offset, history_len)`.
        let history_len = conv_history_len.unwrap_or(0);
        Ok(Self {
            state: ArraysCache {
                conv_state: Some(stack_rows(&conv_rows)?),
                ssm_state: Some(stack_rows(&ssm_rows)?),
                conv_pos: history_len - 1,
                offset: history_len,
            },
            row_offsets: caches.iter().map(|cache| cache.offset).collect(),
            formed_offset: history_len,
        })
    }

    fn split(&self) -> Result<Vec<ArraysCache>, Exception> {
        let steps = self.state.offset - self.formed_offset;
        let conv_state =
            self.state.conv_state.as_ref().ok_or_else(|| {
                Exception::custom("batched recurrent layer has no convolution state")
            })?;
        let ssm_state = self
            .state
            .ssm_state
            .as_ref()
            .ok_or_else(|| Exception::custom("batched recurrent layer has no state"))?;
        self.row_offsets
            .iter()
            .zip(0_i32..)
            .map(|(&row_offset, row_index)| {
                let row_range = row_index..row_index + 1;
                Ok(ArraysCache {
                    conv_state: Some(conv_state.index((row_range.clone(), .., ..))),
                    ssm_state: Some(ssm_state.index((row_range, .., .., ..))),
                    conv_pos: self.state.conv_pos,
                    offset: row_offset + steps,
                })
            })
            .collect()
    }
}

/// Several hybrid (attention plus recurrent) sequences decoding together.
///
/// Attention layers are batched exactly like [`BatchedKvCache`]. Recurrent
/// layers stack each row's state on the batch axis; they need no padding
/// because a recurrence has no positions to align.
#[derive(Debug)]
pub struct HybridBatchedCache {
    layers: Vec<HybridLayer>,
    /// Left padding of each row in the attention layers.
    padding: Vec<i32>,
}

impl HybridBatchedCache {
    /// Merge per-sequence hybrid caches into one batch. Attention layers must
    /// be dense, and every recurrent layer must hold state (the row must have
    /// been prefilled).
    // Callers hand over ownership of their per-sequence caches on purpose.
    #[allow(clippy::needless_pass_by_value)]
    pub fn from_rows(rows: Vec<Vec<Option<LayerCache>>>) -> Result<Self, Exception> {
        let num_layers = rows.first().map_or(0, Vec::len);
        if rows.iter().any(|row| row.len() != num_layers) {
            return Err(Exception::custom(
                "batched cache rows must have the same layer count",
            ));
        }
        let first_attention = rows.first().and_then(|row| {
            row.iter()
                .position(|layer| matches!(layer, Some(LayerCache::KV(_))))
        });
        let lengths: Vec<i32> = match first_attention {
            Some(index) => rows
                .iter()
                .map(|row| match row.get(index) {
                    Some(Some(LayerCache::KV(cache))) => Ok(cache.offset()),
                    _ => Err(Exception::custom(
                        "batched cache rows must agree on which layers are attention layers",
                    )),
                })
                .collect::<Result<_, _>>()?,
            None => vec![0; rows.len()],
        };
        let padded_len = lengths.iter().copied().max().unwrap_or(0);
        let padding: Vec<i32> = lengths.iter().map(|&length| padded_len - length).collect();

        let mut layers = Vec::with_capacity(num_layers);
        for layer_index in 0..num_layers {
            let mut attention = Vec::new();
            let mut recurrent = Vec::new();
            for row in &rows {
                match row.get(layer_index) {
                    Some(Some(LayerCache::KV(cache))) => attention.push(cache),
                    Some(Some(LayerCache::Arrays(cache))) => recurrent.push(cache),
                    _ => return Err(Exception::custom("batched cache row has an empty layer")),
                }
            }
            layers.push(match (attention.is_empty(), recurrent.is_empty()) {
                (false, true) => {
                    HybridLayer::Attention(merge_kv_layer(&attention, &lengths, &padding)?)
                }
                (true, false) => HybridLayer::Recurrent(RecurrentBatch::merge(&recurrent)?),
                _ => {
                    return Err(Exception::custom(
                        "batched cache rows must agree on each layer's kind",
                    ));
                }
            });
        }
        Ok(Self { layers, padding })
    }

    /// Split back into per-sequence caches. Attention rows drop their
    /// padding; recurrent rows continue on the unbatched path from the
    /// batch's convolution layout.
    pub fn into_rows(self) -> Result<Vec<Vec<Option<LayerCache>>>, Exception> {
        let padded_len = self.padded_len();
        let mut rows: Vec<Vec<Option<LayerCache>>> = self
            .padding
            .iter()
            .map(|_| Vec::with_capacity(self.layers.len()))
            .collect();
        for layer in &self.layers {
            match layer {
                HybridLayer::Attention(attention) => {
                    for (row, split) in
                        rows.iter_mut()
                            .zip(split_kv_layer(attention, &self.padding, padded_len)?)
                    {
                        row.push(Some(LayerCache::KV(split)));
                    }
                }
                HybridLayer::Recurrent(recurrent) => {
                    for (row, split) in rows.iter_mut().zip(recurrent.split()?) {
                        row.push(Some(LayerCache::Arrays(split)));
                    }
                }
            }
        }
        Ok(rows)
    }

    pub const fn batch_size(&self) -> usize {
        self.padding.len()
    }

    /// Tokens held per attention row including padding.
    fn padded_len(&self) -> i32 {
        self.layers
            .iter()
            .find_map(|layer| match layer {
                HybridLayer::Attention(attention) => Some(attention.offset()),
                HybridLayer::Recurrent(_) => None,
            })
            .unwrap_or(0)
    }

    pub fn layers_mut(&mut self) -> &mut [HybridLayer] {
        &mut self.layers
    }

    /// See [`BatchedKvCache::attention_mask`]; applies to the attention layers.
    pub fn attention_mask(&self, new_tokens: i32) -> Result<Option<Array>, Exception> {
        padding_attention_mask(&self.padding, self.padded_len(), new_tokens)
    }

    pub fn eval_targets(&self) -> Vec<&Array> {
        let mut targets = Vec::new();
        for layer in &self.layers {
            match layer {
                HybridLayer::Attention(attention) => targets.extend(attention.eval_targets()),
                HybridLayer::Recurrent(recurrent) => {
                    targets.extend(&recurrent.state.conv_state);
                    targets.extend(&recurrent.state.ssm_state);
                }
            }
        }
        targets
    }
}

/// A batched cache of whichever kind the loaded architecture uses.
#[derive(Debug)]
pub enum AnyBatchedCache {
    Kv(BatchedKvCache),
    Hybrid(HybridBatchedCache),
}

impl AnyBatchedCache {
    /// Merge per-sequence caches. All rows must be the same kind.
    pub fn from_rows(rows: Vec<AnyCache>) -> Result<Self, Exception> {
        let mut kv_rows = Vec::new();
        let mut hybrid_rows = Vec::new();
        for any_row in rows {
            match any_row {
                AnyCache::KV(row) => kv_rows.push(row),
                AnyCache::Hybrid(row) => hybrid_rows.push(row),
            }
        }
        match (kv_rows.is_empty(), hybrid_rows.is_empty()) {
            (false, true) => BatchedKvCache::from_rows(kv_rows).map(Self::Kv),
            (true, false) => HybridBatchedCache::from_rows(hybrid_rows).map(Self::Hybrid),
            _ => Err(Exception::custom(
                "batched cache rows must all use the same cache kind",
            )),
        }
    }

    /// Split back into per-sequence caches, in row order.
    pub fn into_rows(self) -> Result<Vec<AnyCache>, Exception> {
        match self {
            Self::Kv(cache) => Ok(cache.into_rows()?.into_iter().map(AnyCache::KV).collect()),
            Self::Hybrid(cache) => Ok(cache
                .into_rows()?
                .into_iter()
                .map(AnyCache::Hybrid)
                .collect()),
        }
    }

    pub const fn batch_size(&self) -> usize {
        match self {
            Self::Kv(cache) => cache.batch_size(),
            Self::Hybrid(cache) => cache.batch_size(),
        }
    }

    pub fn eval_targets(&self) -> Vec<&Array> {
        match self {
            Self::Kv(cache) => cache.eval_targets(),
            Self::Hybrid(cache) => cache.eval_targets(),
        }
    }
}

/// Boolean `[B, 1, 1, padded_len + new_tokens]` mask (true = attend) that
/// hides each row's left padding. `None` when no row has padding.
fn padding_attention_mask(
    padding: &[i32],
    padded_len: i32,
    new_tokens: i32,
) -> Result<Option<Array>, Exception> {
    if padding.iter().all(|&row_padding| row_padding == 0) {
        return Ok(None);
    }
    let batch =
        i32::try_from(padding.len()).map_err(|_| Exception::custom("batch size overflow"))?;
    let total = padded_len + new_tokens;
    let padding_array = Array::from_slice(padding, &[batch, 1]);
    let positions = ops::arange::<_, i32>(None, total, None)?.reshape(&[1, total])?;
    let mask = positions.ge(padding_array)?;
    mask.reshape(&[batch, 1, 1, total]).map(Some)
}

/// Merge one layer's per-row caches into a left-padded batched layer.
/// `lengths` are the rows' true token counts and `padding` their left padding.
fn merge_kv_layer(
    caches: &[&SteppingKeyValueCache],
    lengths: &[i32],
    padding: &[i32],
) -> Result<BatchedLayerCache, Exception> {
    let mut row_keys = Vec::with_capacity(caches.len());
    let mut row_values = Vec::with_capacity(caches.len());
    for ((cache, &length), &row_padding) in caches.iter().zip(lengths).zip(padding) {
        let (keys, values) = dense_prefix(cache, length)?;
        row_keys.push(left_pad(&keys, row_padding)?);
        row_values.push(left_pad(&values, row_padding)?);
    }
    Ok(BatchedLayerCache {
        storage: SteppingKeyValueCache::from_arrays(
            stack_rows(&row_keys)?,
            stack_rows(&row_values)?,
        )?,
        padding: padding.to_vec(),
    })
}

/// Split a batched layer back into one unpadded cache per row.
fn split_kv_layer(
    layer: &BatchedLayerCache,
    padding: &[i32],
    padded_len: i32,
) -> Result<Vec<SteppingKeyValueCache>, Exception> {
    let keys = layer
        .storage
        .keys()
        .ok_or_else(|| Exception::custom("batched cache layer has no keys"))?;
    let values = layer
        .storage
        .values()
        .ok_or_else(|| Exception::custom("batched cache layer has no values"))?;
    padding
        .iter()
        .zip(0_i32..)
        .map(|(&row_padding, row_index)| {
            let valid = row_padding..padded_len;
            let row_range = row_index..row_index + 1;
            SteppingKeyValueCache::from_arrays(
                keys.index((row_range.clone(), .., valid.clone(), ..)),
                values.index((row_range, .., valid, ..)),
            )
        })
        .collect()
}

/// Check `[B, T]` prefill inputs against per-row `lengths` and return `T`.
pub(crate) fn validate_prefill_shape(inputs: &Array, lengths: &[i32]) -> Result<i32, Exception> {
    let &[batch, steps] = inputs.shape() else {
        return Err(Exception::custom("prefill inputs must be [B, T]"));
    };
    if usize::try_from(batch).ok() != Some(lengths.len())
        || lengths.iter().copied().max() != Some(steps)
        || lengths.iter().any(|&length| length < 1)
    {
        return Err(Exception::custom(
            "prefill lengths must be positive, one per row, with the longest equal to T",
        ));
    }
    Ok(steps)
}

/// The first `length` cached positions of a dense cache as `[1, H, length, D]`.
fn dense_prefix(cache: &SteppingKeyValueCache, length: i32) -> Result<(Array, Array), Exception> {
    if cache.is_quantized() || cache.is_mla() {
        return Err(Exception::custom("batched decode requires dense KV caches"));
    }
    let keys = cache
        .keys()
        .ok_or_else(|| Exception::custom("batched cache row has no keys"))?;
    let values = cache
        .values()
        .ok_or_else(|| Exception::custom("batched cache row has no values"))?;
    Ok((
        keys.index((.., .., 0..length, ..)),
        values.index((.., .., 0..length, ..)),
    ))
}

fn left_pad(array: &Array, padding: i32) -> Result<Array, Exception> {
    if padding == 0 {
        return Ok(array.clone());
    }
    let shape = array.shape();
    let [batch, heads, _, dim] = <[i32; 4]>::try_from(shape)
        .map_err(|_| Exception::custom("KV cache arrays must be 4D [B, H, T, D]"))?;
    let zeros = ops::zeros_dtype(&[batch, heads, padding, dim], array.dtype())?;
    ops::concatenate(&[&zeros, array], 2)
}

fn stack_rows(rows: &[Array]) -> Result<Array, Exception> {
    match rows {
        [single] => Ok(single.clone()),
        _ => ops::concatenate(&rows.iter().collect::<Vec<_>>(), 0),
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::float_cmp,
    clippy::panic
)]
mod tests {
    use super::*;

    const HEADS: i32 = 2;
    const DIM: i32 = 4;

    /// A one-layer row whose keys/values are filled with `fill` over `length` tokens.
    fn row(length: i32, fill: f32) -> KvRow {
        let keys = ops::full::<f32>(&[1, HEADS, length, DIM], Array::from_f32(fill)).unwrap();
        let values = ops::full::<f32>(&[1, HEADS, length, DIM], Array::from_f32(-fill)).unwrap();
        vec![Some(
            SteppingKeyValueCache::from_arrays(keys, values).unwrap(),
        )]
    }

    #[test]
    fn equal_length_rows_need_no_mask() {
        let batch = BatchedKvCache::from_rows(vec![row(3, 1.0), row(3, 2.0)]).unwrap();
        assert_eq!(batch.lengths(), vec![3, 3]);
        assert!(batch.attention_mask(1).unwrap().is_none());
    }

    #[test]
    fn mask_hides_left_padding_only() {
        let batch = BatchedKvCache::from_rows(vec![row(2, 1.0), row(4, 2.0)]).unwrap();
        let mask = batch.attention_mask(1).unwrap().unwrap();
        assert_eq!(mask.shape(), &[2, 1, 1, 5]);
        let flat = mask.reshape(&[10]).unwrap();
        assert_eq!(
            flat.as_slice::<bool>(),
            &[false, false, true, true, true, true, true, true, true, true]
        );
    }

    #[test]
    fn prefill_mask_is_causal_and_skips_padding() {
        let mask = BatchedKvCache::prefill_mask(&[3, 1], 3).unwrap();
        assert_eq!(mask.shape(), &[2, 1, 3, 3]);
        let flat = mask.reshape(&[18]).unwrap();
        #[rustfmt::skip]
        let expected = [
            // row 0: no padding, plain causal mask
            true, false, false,
            true, true, false,
            true, true, true,
            // row 1: two padding slots; they only see themselves
            true, false, false,
            false, true, false,
            false, false, true,
        ];
        assert_eq!(flat.as_slice::<bool>(), &expected);
    }

    #[test]
    fn rows_round_trip_through_a_batch() {
        let batch = BatchedKvCache::from_rows(vec![row(2, 1.0), row(4, 2.0)]).unwrap();
        let rows = batch.into_rows().unwrap();
        let lengths: Vec<i32> = rows
            .iter()
            .map(|r| r[0].as_ref().unwrap().offset())
            .collect();
        assert_eq!(lengths, vec![2, 4]);
        for (r, fill) in rows.iter().zip([1.0_f32, 2.0]) {
            let keys = r[0].as_ref().unwrap().keys().unwrap().contiguous().unwrap();
            assert!(keys.as_slice::<f32>().iter().all(|&x| x == fill));
        }
    }

    #[test]
    fn appended_tokens_survive_the_round_trip() {
        let mut batch = BatchedKvCache::from_rows(vec![row(1, 1.0), row(3, 2.0)]).unwrap();
        let new_keys = ops::full::<f32>(&[2, HEADS, 1, DIM], Array::from_f32(9.0)).unwrap();
        let new_values = new_keys.clone();
        let layer = batch.layers_mut()[0].as_mut().unwrap();
        let positions = layer.rope_positions().unwrap();
        assert_eq!(positions.as_slice::<i32>(), &[1, 3]);
        layer.update_and_fetch(new_keys, new_values).unwrap();
        assert_eq!(batch.lengths(), vec![2, 4]);

        let rows = batch.into_rows().unwrap();
        let short = rows[0][0]
            .as_ref()
            .unwrap()
            .keys()
            .unwrap()
            .contiguous()
            .unwrap();
        assert_eq!(short.shape(), &[1, HEADS, 2, DIM]);
        let first_head: Vec<f32> = short.as_slice::<f32>()[..8].to_vec();
        assert_eq!(first_head, vec![1.0, 1.0, 1.0, 1.0, 9.0, 9.0, 9.0, 9.0]);
    }

    #[test]
    fn single_row_batch_reuses_the_row() {
        let batch = BatchedKvCache::from_rows(vec![row(5, 3.0)]).unwrap();
        assert_eq!(batch.batch_size(), 1);
        assert!(batch.attention_mask(1).unwrap().is_none());
    }

    #[test]
    fn mismatched_layer_counts_are_rejected() {
        let mut two_layers = row(2, 1.0);
        two_layers.extend(row(2, 1.0));
        assert!(BatchedKvCache::from_rows(vec![row(2, 1.0), two_layers]).is_err());
    }

    const HISTORY: i32 = 3;
    const CONV_DIM: i32 = 2;
    const CONV_WIDTH: usize = 2;

    /// A recurrent row whose ring slots hold `slots` (one value per slot,
    /// replicated across channels), newest entry at `conv_pos`.
    fn recurrent_row(slots: [f32; 3], conv_pos: i32, offset: i32) -> Vec<Option<LayerCache>> {
        let conv_state = Array::from_slice(
            &slots
                .iter()
                .flat_map(|&value| [value; CONV_WIDTH])
                .collect::<Vec<_>>(),
            &[1, HISTORY, CONV_DIM],
        );
        vec![Some(LayerCache::Arrays(ArraysCache {
            conv_state: Some(conv_state),
            ssm_state: Some(
                Array::full::<f32>(
                    &[1, 1, 1, 1],
                    Array::from_f32(f32::from(i16::try_from(offset).unwrap())),
                )
                .unwrap(),
            ),
            conv_pos,
            offset,
        }))]
    }

    fn slot_values(conv_state: &Array) -> Vec<f32> {
        let flat = conv_state.contiguous().unwrap();
        flat.as_slice::<f32>()
            .iter()
            .step_by(CONV_WIDTH)
            .copied()
            .collect()
    }

    fn recurrent_state_mut(batch: &mut HybridBatchedCache) -> &mut ArraysCache {
        let Some(HybridLayer::Recurrent(recurrent)) = batch.layers_mut().first_mut() else {
            panic!("expected a recurrent layer");
        };
        recurrent.state_mut()
    }

    #[test]
    fn recurrent_rows_are_normalized_to_one_ring_layout() {
        // Chronological (oldest first) contents the rows must end up with:
        // row 0 has wrapped (offset 5), row 1 has one token, row 2 two.
        let rows = vec![
            recurrent_row([30.0, 10.0, 20.0], 1, 5),
            recurrent_row([7.0, 0.0, 0.0], 0, 1),
            recurrent_row([1.0, 2.0, 0.0], 1, 2),
        ];
        let mut batch = HybridBatchedCache::from_rows(rows).unwrap();
        let state = recurrent_state_mut(&mut batch);
        assert_eq!(state.conv_pos, HISTORY - 1);
        assert!(state.offset >= HISTORY);
        let stacked = slot_values(state.conv_state.as_ref().unwrap());
        assert_eq!(
            stacked,
            vec![20.0, 30.0, 10.0, 0.0, 0.0, 7.0, 0.0, 1.0, 2.0]
        );
    }

    #[test]
    fn recurrent_rows_keep_their_own_offsets_after_steps() {
        let rows = vec![
            recurrent_row([30.0, 10.0, 20.0], 1, 5),
            recurrent_row([7.0, 0.0, 0.0], 0, 1),
        ];
        let mut batch = HybridBatchedCache::from_rows(rows).unwrap();
        recurrent_state_mut(&mut batch).offset += 2;
        let mut split = batch.into_rows().unwrap();
        let offsets: Vec<i32> = split
            .iter_mut()
            .map(|row| match row.first() {
                Some(Some(LayerCache::Arrays(cache))) => cache.offset,
                _ => panic!("expected a recurrent layer"),
            })
            .collect();
        assert_eq!(offsets, vec![7, 3]);
        for (row, expected_ssm) in split.iter().zip([5.0_f32, 1.0]) {
            let Some(Some(LayerCache::Arrays(cache))) = row.first() else {
                panic!("expected a recurrent layer");
            };
            let ssm = cache.ssm_state.as_ref().unwrap().contiguous().unwrap();
            assert_eq!(ssm.as_slice::<f32>(), &[expected_ssm]);
            assert_eq!(cache.conv_pos, HISTORY - 1);
        }
    }

    #[test]
    fn hybrid_batch_pads_attention_layers_and_stacks_recurrent_ones() {
        let attention_row = |length: i32, fill: f32| {
            let keys = ops::full::<f32>(&[1, HEADS, length, DIM], Array::from_f32(fill)).unwrap();
            LayerCache::KV(SteppingKeyValueCache::from_arrays(keys.clone(), keys).unwrap())
        };
        let hybrid_row = |length: i32, fill: f32, slots: [f32; 3]| {
            let mut layers = recurrent_row(slots, 2, length);
            layers.push(Some(attention_row(length, fill)));
            layers
        };
        let mut batch = HybridBatchedCache::from_rows(vec![
            hybrid_row(2, 1.0, [1.0, 2.0, 3.0]),
            hybrid_row(4, 2.0, [4.0, 5.0, 6.0]),
        ])
        .unwrap();
        assert_eq!(batch.batch_size(), 2);
        let mask = batch.attention_mask(1).unwrap().unwrap();
        assert_eq!(mask.shape(), &[2, 1, 1, 5]);
        let positions = match batch.layers_mut().get(1) {
            Some(HybridLayer::Attention(layer)) => layer.rope_positions().unwrap(),
            _ => panic!("expected an attention layer"),
        };
        assert_eq!(positions.as_slice::<i32>(), &[2, 4]);

        let rows = batch.into_rows().unwrap();
        let lengths: Vec<i32> = rows
            .iter()
            .map(|row| match row.get(1) {
                Some(Some(LayerCache::KV(cache))) => cache.offset(),
                _ => panic!("expected an attention layer"),
            })
            .collect();
        assert_eq!(lengths, vec![2, 4]);
    }

    #[test]
    fn rows_without_recurrent_state_are_rejected() {
        let mut row = recurrent_row([1.0, 2.0, 3.0], 2, 3);
        if let Some(Some(LayerCache::Arrays(cache))) = row.first_mut() {
            cache.ssm_state = None;
        }
        assert!(HybridBatchedCache::from_rows(vec![row]).is_err());
    }
}
