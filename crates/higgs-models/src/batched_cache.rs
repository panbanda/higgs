//! KV cache that holds several sequences in one set of batched arrays.
//!
//! Batched decode attends over `[B, H, L, D]` keys/values every step. Keeping
//! those arrays batched between steps means a step only appends one token per
//! row; the alternative (re-stacking per-request caches each step) copies the
//! whole KV state of every request per layer per token.
//!
//! Rows are **left-padded** to a common length, so every row's next token lands
//! at the same index and one `slice_update` appends for the whole batch.
//! Padding slots are zeros and are masked out of attention. Each row keeps its
//! true position (`lengths`) for `RoPE`, so cached keys stay valid when a row
//! moves between batches.

use mlx_rs::{
    Array,
    error::Exception,
    ops::{self, indexing::IndexOp},
};

use crate::cache::{KeyValueCache, SteppingKeyValueCache};

/// Per-layer caches for one sequence, as used by non-batched decode.
pub type KvRow = Vec<Option<SteppingKeyValueCache>>;

#[derive(Debug)]
pub struct BatchedKvCache {
    /// One cache per layer with keys/values shaped `[B, H, capacity, D]`.
    layers: Vec<SteppingKeyValueCache>,
    /// True token count of each row; the rest of the row is left padding.
    lengths: Vec<i32>,
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
        let num_layers = rows.first().map_or(0, Vec::len);
        if rows.iter().any(|row| row.len() != num_layers) {
            return Err(Exception::custom(
                "batched cache rows must have the same layer count",
            ));
        }

        let mut layers = Vec::with_capacity(num_layers);
        for layer_index in 0..num_layers {
            let mut row_keys = Vec::with_capacity(rows.len());
            let mut row_values = Vec::with_capacity(rows.len());
            for (row, &length) in rows.iter().zip(&lengths) {
                let cache = row
                    .get(layer_index)
                    .and_then(Option::as_ref)
                    .ok_or_else(|| Exception::custom("batched cache row has an empty layer"))?;
                let (keys, values) = dense_prefix(cache, length)?;
                row_keys.push(left_pad(&keys, padded_len - length)?);
                row_values.push(left_pad(&values, padded_len - length)?);
            }
            layers.push(SteppingKeyValueCache::from_arrays(
                stack_rows(&row_keys)?,
                stack_rows(&row_values)?,
            )?);
        }
        Ok(Self { layers, lengths })
    }

    /// An empty batch about to receive a left-padded prefill: `lengths` are
    /// the true token counts the rows will hold once the padded prompt
    /// (as long as the longest row) has been written to every layer.
    pub fn for_prefill(num_layers: usize, lengths: Vec<i32>) -> Self {
        Self {
            layers: (0..num_layers)
                .map(|_| SteppingKeyValueCache::new())
                .collect(),
            lengths,
        }
    }

    /// Split back into per-sequence caches, dropping each row's padding.
    pub fn into_rows(self) -> Result<Vec<KvRow>, Exception> {
        let padded_len = self.padded_len();
        let mut rows: Vec<KvRow> = self
            .lengths
            .iter()
            .map(|_| Vec::with_capacity(self.layers.len()))
            .collect();
        for layer in &self.layers {
            let keys = layer
                .keys()
                .ok_or_else(|| Exception::custom("batched cache layer has no keys"))?;
            let values = layer
                .values()
                .ok_or_else(|| Exception::custom("batched cache layer has no values"))?;
            for ((row, &length), row_index) in rows.iter_mut().zip(&self.lengths).zip(0_i32..) {
                let valid = (padded_len - length)..padded_len;
                let row_range = row_index..row_index + 1;
                row.push(Some(SteppingKeyValueCache::from_arrays(
                    keys.index((row_range.clone(), .., valid.clone(), ..)),
                    values.index((row_range, .., valid, ..)),
                )?));
            }
        }
        Ok(rows)
    }

    pub const fn batch_size(&self) -> usize {
        self.lengths.len()
    }

    pub const fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// True token count of each row, which is also the `RoPE` position of the
    /// next token appended to that row.
    pub fn lengths(&self) -> &[i32] {
        &self.lengths
    }

    fn padded_len(&self) -> i32 {
        self.lengths.iter().copied().max().unwrap_or(0)
    }

    pub fn layer_mut(
        &mut self,
        layer_index: usize,
    ) -> Result<&mut SteppingKeyValueCache, Exception> {
        self.layers
            .get_mut(layer_index)
            .ok_or_else(|| Exception::custom("batched cache layer index out of range"))
    }

    /// Boolean `[B, 1, 1, L + new_tokens]` mask (true = attend) covering the
    /// cache contents plus `new_tokens` about to be appended. `None` when no
    /// row has padding, so equal-length batches skip masking entirely.
    pub fn attention_mask(&self, new_tokens: i32) -> Result<Option<Array>, Exception> {
        let padded_len = self.padded_len();
        if self.lengths.iter().all(|&length| length == padded_len) {
            return Ok(None);
        }
        let batch = i32::try_from(self.lengths.len())
            .map_err(|_| Exception::custom("batch size overflow"))?;
        let row_padding: Vec<i32> = self
            .lengths
            .iter()
            .map(|&length| padded_len - length)
            .collect();
        let total = padded_len + new_tokens;
        let padding_array = Array::from_slice(&row_padding, &[batch, 1]);
        let positions = ops::arange::<_, i32>(None, total, None)?.reshape(&[1, total])?;
        let mask = positions.ge(padding_array)?;
        mask.reshape(&[batch, 1, 1, total]).map(Some)
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

    /// Record that every row grew by `new_tokens` (call once per forward pass,
    /// after all layers were updated).
    pub fn advance(&mut self, new_tokens: i32) {
        for length in &mut self.lengths {
            *length += new_tokens;
        }
    }

    /// Arrays that must be evaluated to materialize the cache state.
    pub fn eval_targets(&self) -> Vec<&Array> {
        self.layers
            .iter()
            .flat_map(SteppingKeyValueCache::eval_targets)
            .collect()
    }
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
#[allow(clippy::unwrap_used, clippy::indexing_slicing, clippy::float_cmp)]
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
        assert_eq!(batch.lengths(), &[3, 3]);
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
        batch
            .layer_mut(0)
            .unwrap()
            .update_and_fetch(new_keys, new_values)
            .unwrap();
        batch.advance(1);
        assert_eq!(batch.lengths(), &[2, 4]);

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
}
