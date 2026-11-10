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
