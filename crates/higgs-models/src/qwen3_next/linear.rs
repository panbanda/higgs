//! Quantized linear and embedding layers.

use super::{flags::decode_gemv_enabled, qgemv::qgemv_4bit};
use mlx_rs::{Array, error::Exception, macros::ModuleParameters, module::Param, ops};

type QuantizedParams = (Param<Array>, Param<Array>, Param<Array>);

pub fn init_quantized_params() -> QuantizedParams {
    fn placeholder() -> Param<Array> {
        Param::new(Array::from_slice(&[0.0_f32], &[1]))
    }

    (placeholder(), placeholder(), placeholder())
}

pub fn quantized_forward(
    x: &Array,
    weight: &Array,
    scales: &Array,
    biases: &Array,
    group_size: i32,
    bits: i32,
) -> Result<Array, Exception> {
    ops::quantized_matmul(
        x,
        weight,
        scales,
        biases,
        ops::QuantizedMatmulOptions {
            group_size: Some(group_size),
            bits: Some(bits),
            transpose: true,
            ..Default::default()
        },
    )
    .map_err(Exception::from)
}

/// Quantized linear layer stored as raw weight/scales/biases arrays.
/// Forward uses `quantized_matmul` directly.
#[derive(Debug, Clone, ModuleParameters)]
pub struct QLinear {
    #[param]
    pub(crate) weight: Param<Array>,
    #[param]
    pub(crate) scales: Param<Array>,
    #[param]
    pub(crate) biases: Param<Array>,
    pub(crate) group_size: i32,
    pub(crate) bits: i32,
}

impl QLinear {
    #[allow(clippy::unnecessary_wraps)]
    pub(crate) fn new(group_size: i32, bits: i32) -> Result<Self, Exception> {
        let (weight, scales, biases) = init_quantized_params();
        Ok(Self {
            weight,
            scales,
            biases,
            group_size,
            bits,
        })
    }

    pub(crate) fn forward(&self, x: &Array) -> Result<Array, Exception> {
        if self.bits == 0 {
            dense_linear_no_bias_forward(&self.weight, x)
        } else {
            quantized_forward(
                x,
                &self.weight,
                &self.scales,
                &self.biases,
                self.group_size,
                self.bits,
            )
        }
    }

    /// Decode-only fast path for 4-bit single-token inference.
    ///
    /// Keeps the optimization opt-in so we can wire it into selected hot paths
    /// without changing the default behavior of every quantized linear.
    pub(crate) fn forward_decode_fast(&self, x: &Array) -> Result<Array, Exception> {
        if decode_gemv_enabled()
            && self.bits == 4
            && matches!(x.shape(), [1, 1, _])
            && self.weight.shape().len() == 2
        {
            qgemv_4bit(x, &self.weight, &self.scales, &self.biases, self.group_size)
        } else {
            self.forward(x)
        }
    }
}

/// Dense linear layer with a single weight tensor and no bias.
#[derive(Debug, Clone, ModuleParameters)]
pub(super) struct DenseLinearNoBias {
    #[param]
    weight: Param<Array>,
}

pub(super) fn dense_linear_no_bias_forward(weight: &Array, x: &Array) -> Result<Array, Exception> {
    let shape = x.shape().to_vec();
    let in_features = *shape
        .last()
        .ok_or_else(|| Exception::custom("empty input"))?;
    let batch: i32 = shape.iter().take(shape.len() - 1).product();
    let x2d = x.reshape(&[batch, in_features])?;
    let w = weight.as_dtype(x.dtype())?;
    let out2d = x2d.matmul(&w.transpose()?)?;
    let out_features = *out2d.shape().last().unwrap_or(&0);
    let mut out_shape = shape;
    if let Some(last) = out_shape.last_mut() {
        *last = out_features;
    }
    out2d.reshape(&out_shape)
}

impl DenseLinearNoBias {
    pub(super) fn new() -> Self {
        Self {
            weight: Param::new(Array::from_slice(&[0.0_f32], &[1])),
        }
    }

    pub(super) fn forward(&self, x: &Array) -> Result<Array, Exception> {
        dense_linear_no_bias_forward(&self.weight, x)
    }
}

/// Quantized embedding stored as raw weight/scales/biases arrays.
#[derive(Debug, Clone, ModuleParameters)]
pub struct QEmbedding {
    #[param]
    weight: Param<Array>,
    #[param]
    scales: Param<Array>,
    #[param]
    biases: Param<Array>,
    group_size: i32,
    bits: i32,
}

impl QEmbedding {
    #[allow(clippy::unnecessary_wraps)]
    pub(crate) fn new(group_size: i32, bits: i32) -> Result<Self, Exception> {
        let (weight, scales, biases) = init_quantized_params();
        Ok(Self {
            weight,
            scales,
            biases,
            group_size,
            bits,
        })
    }

    pub(crate) fn forward(&self, indices: &Array) -> Result<Array, Exception> {
        let shape = indices.shape().to_vec();
        let flat = indices.flatten(None, None)?;
        let w = (*self.weight).take_axis(&flat, 0)?;
        let out = if self.bits == 0 {
            w
        } else {
            let s = (*self.scales).take_axis(&flat, 0)?;
            let b = (*self.biases).take_axis(&flat, 0)?;
            ops::dequantize(
                &w,
                &s,
                &b,
                ops::DequantizeOptions {
                    group_size: Some(self.group_size),
                    bits: Some(self.bits),
                    ..Default::default()
                },
            )?
        };
        let mut ret_shape: Vec<i32> = shape;
        ret_shape.push(-1);
        out.reshape(&ret_shape)
    }

    pub(crate) fn as_linear(&self, x: &Array) -> Result<Array, Exception> {
        if self.bits == 0 {
            dense_linear_no_bias_forward(&self.weight, x)
        } else if self.bits == 4 && matches!(x.shape(), [1, 1, _]) && self.weight.shape().len() == 2
        {
            qgemv_4bit(x, &self.weight, &self.scales, &self.biases, self.group_size)
        } else {
            quantized_forward(
                x,
                &self.weight,
                &self.scales,
                &self.biases,
                self.group_size,
                self.bits,
            )
        }
    }
}
