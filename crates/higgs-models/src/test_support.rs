use mlx_rs::{Array, error::Exception, ops};

/// Affine-quantize `weight` into the `(weight, scales, biases)` triple the quantized layers store.
pub fn quantize_affine(
    weight: &Array,
    group_size: i32,
    bits: i32,
) -> Result<(Array, Array, Array), Exception> {
    let (quantized, scales, maybe_biases) = ops::quantize(
        weight,
        ops::QuantizeOptions {
            group_size: Some(group_size),
            bits: Some(bits),
            ..Default::default()
        },
    )?;
    let biases =
        maybe_biases.ok_or_else(|| Exception::custom("affine quantization returned no biases"))?;
    Ok((quantized, scales, biases))
}

/// Inverse of [`quantize_affine`].
pub fn dequantize_affine(
    weight: &Array,
    scales: &Array,
    biases: &Array,
    group_size: i32,
    bits: i32,
) -> Result<Array, Exception> {
    Ok(ops::dequantize(
        weight,
        scales,
        biases,
        ops::DequantizeOptions {
            group_size: Some(group_size),
            bits: Some(bits),
            ..Default::default()
        },
    )?)
}

/// A model's ordinary forward pass over any cache type, so one check can
/// run it per sequence and over a batch.
pub trait ForwardOverCache {
    fn num_layers(&self) -> usize;

    fn forward_over<C: crate::cache::KeyValueCache>(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut Vec<Option<C>>,
    ) -> Result<Array, Exception>;
}

fn last_position(logits: &Array) -> Array {
    use mlx_rs::ops::indexing::IndexOp;
    logits.index((.., -1, ..))
}

fn row_of(logits: &Array, row: usize) -> Result<Array, Exception> {
    use mlx_rs::ops::indexing::IndexOp;
    let index = i32::try_from(row).map_err(|_| Exception::custom("row index"))?;
    Ok(logits.index(index..index + 1))
}

fn relative_difference(expected: &Array, actual: &Array) -> Result<f32, Exception> {
    let largest = ops::max(&expected.abs()?, None)?.item_cast::<f32>();
    let difference = ops::max(&expected.subtract(actual)?.abs()?, None)?.item_cast::<f32>();
    Ok(difference / largest.max(f32::MIN_POSITIVE))
}

fn token_column(tokens: &[u32]) -> Result<Array, Exception> {
    let rows = i32::try_from(tokens.len()).map_err(|_| Exception::custom("too many rows"))?;
    Ok(Array::from_slice(tokens, &[rows, 1]))
}

fn prompt_array(prompt: &[u32]) -> Result<Array, Exception> {
    let length = i32::try_from(prompt.len()).map_err(|_| Exception::custom("long prompt"))?;
    Ok(Array::from_slice(prompt, &[1, length]))
}

fn fresh_row(layers: usize) -> crate::batched_cache::KvRow {
    (0..layers)
        .map(|_| Some(crate::cache::SteppingKeyValueCache::new()))
        .collect()
}

/// Step every row of `batch` through `decode_tokens` together and return the
/// worst difference from the per-row `expected` logits of each step.
fn batched_steps_difference<M: ForwardOverCache>(
    model: &mut M,
    batch: &mut crate::batched_cache::BatchedKvCache,
    decode_tokens: &[u32],
    expected: &[Vec<Array>],
) -> Result<f32, Exception> {
    let mut worst = 0.0_f32;
    for (step, &token) in decode_tokens.iter().enumerate() {
        let mask = batch.attention_mask(1)?;
        let stepped = model.forward_over(
            &token_column(&vec![token; expected.len()])?,
            mask.as_ref(),
            batch.layers_mut(),
        )?;
        let last = last_position(&stepped);
        for (row, row_expected) in expected.iter().enumerate() {
            let reference = row_expected
                .get(step)
                .ok_or_else(|| Exception::custom("missing reference step"))?;
            worst = worst.max(relative_difference(reference, &row_of(&last, row)?)?);
        }
    }
    Ok(worst)
}

/// Worst relative logit difference between running `prompts` through a model
/// one at a time and as a batch: rows prefilled separately and merged, then
/// stepped together, and the same rows prefilled together left-padded.
/// Prompts of different lengths exercise padding; ones longer than a sliding
/// window exercise the window next to padding.
pub fn worst_batched_difference<M: ForwardOverCache>(
    model: &mut M,
    prompts: &[&[u32]],
    decode_tokens: &[u32],
) -> Result<f32, Exception> {
    use crate::batched_cache::BatchedKvCache;

    let layers = model.num_layers();

    let mut expected_prefill = Vec::new();
    let mut expected_steps: Vec<Vec<Array>> = Vec::new();
    let mut merged_rows = Vec::new();
    for prompt in prompts {
        let mut alone = fresh_row(layers);
        let prefilled = model.forward_over(&prompt_array(prompt)?, None, &mut alone)?;
        expected_prefill.push(last_position(&prefilled));
        let mut resumed = fresh_row(layers);
        model.forward_over(&prompt_array(prompt)?, None, &mut resumed)?;
        merged_rows.push(resumed);
        let mut steps = Vec::new();
        for &token in decode_tokens {
            let stepped = model.forward_over(&token_column(&[token])?, None, &mut alone)?;
            steps.push(last_position(&stepped));
        }
        expected_steps.push(steps);
    }

    let mut merged = BatchedKvCache::from_rows(merged_rows)?;
    let merged_worst =
        batched_steps_difference(model, &mut merged, decode_tokens, &expected_steps)?;

    let longest = prompts.iter().map(|prompt| prompt.len()).max().unwrap_or(0);
    let mut padded = Vec::new();
    let mut lengths = Vec::new();
    for prompt in prompts {
        padded.extend(std::iter::repeat_n(0_u32, longest - prompt.len()));
        padded.extend_from_slice(prompt);
        lengths.push(i32::try_from(prompt.len()).map_err(|_| Exception::custom("long prompt"))?);
    }
    let steps = i32::try_from(longest).map_err(|_| Exception::custom("long prompt"))?;
    let rows = i32::try_from(prompts.len()).map_err(|_| Exception::custom("rows"))?;
    let mut prefilled = BatchedKvCache::for_prefill(layers, &lengths, steps);
    let prefill_mask = BatchedKvCache::prefill_mask(&lengths, steps)?;
    let prefill_logits = model.forward_over(
        &Array::from_slice(&padded, &[rows, steps]),
        Some(&prefill_mask),
        prefilled.layers_mut(),
    )?;
    let prefill_last = last_position(&prefill_logits);
    let mut prefill_worst = 0.0_f32;
    for (row, expected) in expected_prefill.iter().enumerate() {
        prefill_worst =
            prefill_worst.max(relative_difference(expected, &row_of(&prefill_last, row)?)?);
    }
    let padded_worst =
        batched_steps_difference(model, &mut prefilled, decode_tokens, &expected_steps)?;
    Ok(merged_worst.max(prefill_worst).max(padded_worst))
}
