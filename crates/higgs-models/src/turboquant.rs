#![allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::indexing_slicing,
    clippy::too_many_lines
)]

use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char, c_void};
use std::sync::OnceLock;

use mlx_rs::{Array, Dtype, Stream, argmin_axis, error::Exception, ops};
use rand::Rng;
use serde::{Deserialize, Serialize};

const ONE_BIT_CENTROIDS: [f32; 2] = [-0.797_884_6, 0.797_884_6];
const TWO_BIT_CENTROIDS: [f32; 4] = [-1.5104, -0.4528, 0.4528, 1.5104];
const THREE_BIT_CENTROIDS: [f32; 8] = [
    -2.1520, -1.3440, -0.7560, -0.2451, 0.2451, 0.7560, 1.3440, 2.1520,
];
const FOUR_BIT_CENTROIDS: [f32; 16] = [
    -2.7326, -2.0690, -1.6180, -1.2562, -0.9424, -0.6568, -0.3881, -0.1284, 0.1284, 0.3881, 0.6568,
    0.9424, 1.2562, 1.6180, 2.0690, 2.7326,
];
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum KvCacheMode {
    #[default]
    Off,
    Turboquant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvCacheConfig {
    #[serde(default)]
    pub mode: KvCacheMode,
    #[serde(default = "default_bits")]
    pub bits: u8,
    /// Override key bit width (default: bits - 1).
    #[serde(default)]
    pub key_bits_override: Option<u8>,
    /// Override value bit width (default: bits).
    #[serde(default)]
    pub value_bits_override: Option<u8>,
    /// Enable post-quantization norm correction (default: true).
    #[serde(default = "default_norm_correction")]
    pub norm_correction: bool,
    /// Number of final layers that stay dense (0 = all TQ).
    #[serde(default)]
    pub adaptive_dense_layers: u8,
    #[serde(default)]
    pub seed: u64,
}

const fn default_bits() -> u8 {
    3
}

const fn default_norm_correction() -> bool {
    true
}

impl Default for KvCacheConfig {
    fn default() -> Self {
        Self {
            mode: KvCacheMode::Off,
            bits: default_bits(),
            key_bits_override: None,
            value_bits_override: None,
            norm_correction: true,
            adaptive_dense_layers: 0,
            seed: 0,
        }
    }
}

impl KvCacheConfig {
    pub const fn is_turboquant(self) -> bool {
        matches!(self.mode, KvCacheMode::Turboquant)
    }

    pub fn validate(self) -> Result<(), Exception> {
        if self.is_turboquant() {
            let kb = self.key_bits();
            let vb = self.value_bits();
            if !(1..=8).contains(&kb) {
                return Err(Exception::custom(format!(
                    "TurboQuant key bits must be in [1, 8], got {kb}"
                )));
            }
            if !(2..=8).contains(&vb) {
                return Err(Exception::custom(format!(
                    "TurboQuant value bits must be in [2, 8], got {vb}"
                )));
            }
        }
        Ok(())
    }

    /// Effective key bit width. Uses override if set, otherwise `bits - 1`.
    pub const fn key_bits(self) -> u8 {
        match self.key_bits_override {
            Some(kb) => kb,
            None => self.bits - 1,
        }
    }

    /// Effective value bit width. Uses override if set, otherwise `bits`.
    pub const fn value_bits(self) -> u8 {
        match self.value_bits_override {
            Some(vb) => vb,
            None => self.bits,
        }
    }
}

#[derive(Debug, Clone)]
pub struct QuantizedValue {
    pub norm: f32,
    pub codes: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct QuantizedKey {
    pub norm: f32,
    pub gamma: f32,
    pub codes: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct TurboQuantContext {
    pub config: KvCacheConfig,
    pub head_dim: i32,
    pub num_kv_heads: i32,
    pub key_centroids: Vec<f32>,
    pub value_centroids: Vec<f32>,
    pub key_code_bytes: i32,
    pub value_code_bytes: i32,
    pub key_code_words: i32,
    pub value_code_words: i32,
}

impl TurboQuantContext {
    pub fn new(config: KvCacheConfig, head_dim: i32, num_kv_heads: i32) -> Result<Self, Exception> {
        config.validate()?;
        if head_dim <= 0 || num_kv_heads <= 0 {
            return Err(Exception::custom(
                "TurboQuant head_dim and num_kv_heads must be > 0",
            ));
        }
        if head_dim < 2 || (head_dim & (head_dim - 1)) != 0 {
            return Err(Exception::custom(format!(
                "TurboQuant FWHT requires power-of-2 head_dim, got {head_dim}"
            )));
        }
        let dim = usize::try_from(head_dim).map_err(|_| Exception::custom("head_dim overflow"))?;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let key_centroids = scaled_centroids(config.key_bits(), scale)?;
        let value_centroids = scaled_centroids(config.value_bits(), scale)?;

        let key_code_bytes = packed_bytes(dim, config.key_bits())?;
        let value_code_bytes = packed_bytes(dim, config.value_bits())?;
        let key_code_words = packed_words(dim, config.key_bits())?;
        let value_code_words = packed_words(dim, config.value_bits())?;

        Ok(Self {
            config,
            head_dim,
            num_kv_heads,
            key_centroids,
            value_centroids,
            key_code_bytes,
            value_code_bytes,
            key_code_words,
            value_code_words,
        })
    }

    pub fn quantize_value(&self, values: &[f32]) -> Result<QuantizedValue, Exception> {
        let dim =
            usize::try_from(self.head_dim).map_err(|_| Exception::custom("head_dim overflow"))?;
        if values.len() != dim {
            return Err(Exception::custom("TurboQuant value length mismatch"));
        }

        let norm = l2_norm(values);
        if norm <= f32::EPSILON {
            return Ok(QuantizedValue {
                norm: 0.0,
                codes: vec![0; usize::try_from(self.value_code_bytes).unwrap_or(0)],
            });
        }

        let mut normalized: Vec<f32> = values.iter().map(|value| *value / norm).collect();
        fwht_normalized(&mut normalized);
        let indices = quantize_rotated(&normalized, &self.value_centroids);

        // Norm correction: corrected_norm = norm / ||reconstruction||
        // Eliminates systematic norm shrinkage from discrete quantization.
        let final_norm = if self.config.norm_correction {
            let reconstruction_l2 = reconstruction_norm(&indices, &self.value_centroids);
            if reconstruction_l2 > f32::EPSILON {
                norm / reconstruction_l2
            } else {
                norm
            }
        } else {
            norm
        };

        Ok(QuantizedValue {
            norm: final_norm,
            codes: pack_indices(&indices, self.config.value_bits()),
        })
    }

    pub fn quantize_key(&self, keys: &[f32]) -> Result<QuantizedKey, Exception> {
        let dim =
            usize::try_from(self.head_dim).map_err(|_| Exception::custom("head_dim overflow"))?;
        if keys.len() != dim {
            return Err(Exception::custom("TurboQuant key length mismatch"));
        }

        let norm = l2_norm(keys);
        let key_code_len = usize::try_from(self.key_code_bytes).unwrap_or(0);
        if norm <= f32::EPSILON {
            return Ok(QuantizedKey {
                norm: 0.0,
                gamma: 0.0,
                codes: vec![0; key_code_len],
            });
        }

        let mut normalized: Vec<f32> = keys.iter().map(|value| *value / norm).collect();
        fwht_normalized(&mut normalized);
        let mse_indices = quantize_rotated(&normalized, &self.key_centroids);

        // Norm correction for keys
        let final_norm = if self.config.norm_correction {
            let reconstruction_l2 = reconstruction_norm(&mse_indices, &self.key_centroids);
            if reconstruction_l2 > f32::EPSILON {
                norm / reconstruction_l2
            } else {
                norm
            }
        } else {
            norm
        };

        Ok(QuantizedKey {
            norm: final_norm,
            gamma: 0.0,
            codes: pack_indices(&mse_indices, self.config.key_bits()),
        })
    }

    pub fn dequantize_value(&self, value: &QuantizedValue) -> Result<Vec<f32>, Exception> {
        let dim =
            usize::try_from(self.head_dim).map_err(|_| Exception::custom("head_dim overflow"))?;
        let mut rotated = dequantize_rotated(
            &value.codes,
            self.config.value_bits(),
            &self.value_centroids,
            dim,
            value.norm,
        );
        fwht_normalized(&mut rotated);
        Ok(rotated)
    }

    pub fn dequantize_key(&self, key: &QuantizedKey) -> Result<Vec<f32>, Exception> {
        let dim =
            usize::try_from(self.head_dim).map_err(|_| Exception::custom("head_dim overflow"))?;
        let mut rotated = dequantize_rotated(
            &key.codes,
            self.config.key_bits(),
            &self.key_centroids,
            dim,
            key.norm,
        );
        fwht_normalized(&mut rotated);
        Ok(rotated)
    }

    pub fn rotate_queries(&self, queries: &Array) -> Result<Array, Exception> {
        queries
            .as_dtype(Dtype::Float32)?
            .hadamard_transform(None)
    }

    pub fn key_centroids_array(&self) -> Result<Array, Exception> {
        Ok(Array::from_slice(
            &self.key_centroids,
            &[i32::try_from(self.key_centroids.len())
                .map_err(|_| Exception::custom("key centroid len overflow"))?],
        ))
    }

    pub fn value_centroids_array(&self) -> Result<Array, Exception> {
        Ok(Array::from_slice(
            &self.value_centroids,
            &[i32::try_from(self.value_centroids.len())
                .map_err(|_| Exception::custom("value centroid len overflow"))?],
        ))
    }

    /// Batch-quantize values using GPU ops.
    ///
    /// Input: `[H, T, D]` f32 tensor.
    /// Returns `(norms: [H, T], packed_codes: Vec<u8>)` where packed_codes is
    /// laid out as `[H * T * value_code_bytes]` in row-major order.
    pub fn quantize_values_batch(&self, values: &Array) -> Result<BatchQuantizedValues, Exception> {
        let shape = values.shape();
        let h = shape[0];
        let t = shape[1];

        // norms: [H, T]
        let norms = ops::sqrt(&ops::sum_axis(&ops::square(values)?, -1, None)?)?;

        // normalized: [H, T, D] — safe div with epsilon
        let eps = Array::from_f32(f32::EPSILON);
        let safe_norms = ops::maximum(&norms, &eps)?;
        let normalized = ops::divide(values, &safe_norms.expand_dims(-1)?)?;

        // rotated: [H, T, D] = hadamard(normalized) — orthonormal FWHT replaces D×D matmul
        let rotated = normalized.hadamard_transform(None)?;

        // distances: [H, T, D, C] = |rotated[..., None] - centroids|
        let centroids = self.value_centroids_array()?;
        let rotated_expanded = rotated.expand_dims(-1)?; // [H, T, D, 1]
        let distances = ops::abs(&ops::subtract(&rotated_expanded, &centroids)?)?;

        // indices: [H, T, D] as u32
        let indices = argmin_axis!(&distances, -1)?;

        // Norm correction on CPU batch path
        let norms = if self.config.norm_correction {
            let indices_i32 = indices.as_dtype(Dtype::Int32)?;
            let gathered = centroids.take(&indices_i32)?;
            let recon_l2 = ops::sqrt(&ops::sum_axis(&ops::square(&gathered)?, -1, None)?)?;
            let safe_recon = ops::maximum(&recon_l2, &eps)?;
            ops::divide(&norms, &safe_recon)?
        } else {
            norms
        };

        // Pack on CPU: eval indices, read as flat u32, pack into bytes
        indices.eval()?;
        norms.eval()?;

        let indices_flat = indices.as_slice::<u32>();
        let norms_flat = norms.as_slice::<f32>();
        let ht = usize::try_from(h).unwrap() * usize::try_from(t).unwrap();
        let dim = usize::try_from(self.head_dim).unwrap();
        let code_bytes = usize::try_from(self.value_code_bytes).unwrap();
        let bits = self.config.value_bits();

        let mut packed = vec![0_u8; ht * code_bytes];
        for row in 0..ht {
            let row_start = row * dim;
            let out_start = row * code_bytes;
            pack_u32_indices(
                &indices_flat[row_start..row_start + dim],
                bits,
                &mut packed[out_start..out_start + code_bytes],
            );
        }

        Ok(BatchQuantizedValues {
            norms: norms_flat.to_vec(),
            packed_codes: packed,
            code_bytes,
        })
    }

    /// Batch-quantize keys using GPU ops.
    ///
    /// Input: `[H, T, D]` f32 tensor.
    /// Returns `BatchQuantizedKeys` with norms, gammas, and packed MSE codes.
    pub fn quantize_keys_batch(&self, keys: &Array) -> Result<BatchQuantizedKeys, Exception> {
        let shape = keys.shape();
        let h = shape[0];
        let t = shape[1];
        let key_bits = self.config.key_bits();

        // norms: [H, T]
        let norms = ops::sqrt(&ops::sum_axis(&ops::square(keys)?, -1, None)?)?;

        // normalized: [H, T, D]
        let eps = Array::from_f32(f32::EPSILON);
        let safe_norms = ops::maximum(&norms, &eps)?;
        let normalized = ops::divide(keys, &safe_norms.expand_dims(-1)?)?;

        // rotated: [H, T, D] = hadamard(normalized)
        let rotated = normalized.hadamard_transform(None)?;

        // MSE quantize: find nearest centroid in rotated space
        let key_centroids = self.key_centroids_array()?;
        let rotated_expanded = rotated.expand_dims(-1)?; // [H, T, D, 1]
        let distances = ops::abs(&ops::subtract(&rotated_expanded, &key_centroids)?)?;
        let mse_indices = argmin_axis!(&distances, -1)?; // [H, T, D]

        // Norm correction for keys
        let norms = if self.config.norm_correction {
            let mse_indices_i32 = mse_indices.as_dtype(Dtype::Int32)?;
            let rotated_approx = key_centroids.take(&mse_indices_i32)?;
            let recon_l2 = ops::sqrt(&ops::sum_axis(&ops::square(&rotated_approx)?, -1, None)?)?;
            let safe_recon = ops::maximum(&recon_l2, &eps)?;
            ops::divide(&norms, &safe_recon)?
        } else {
            norms
        };

        // Eval everything before CPU readout
        mse_indices.eval()?;
        norms.eval()?;

        let indices_flat = mse_indices.as_slice::<u32>();
        let norms_flat = norms.as_slice::<f32>();

        let ht = usize::try_from(h).unwrap() * usize::try_from(t).unwrap();
        let dim = usize::try_from(self.head_dim).unwrap();
        let key_code_bytes = usize::try_from(self.key_code_bytes).unwrap();

        let mut packed_codes = vec![0_u8; ht * key_code_bytes];
        for row in 0..ht {
            let row_start = row * dim;
            let code_start = row * key_code_bytes;
            pack_u32_indices(
                &indices_flat[row_start..row_start + dim],
                key_bits,
                &mut packed_codes[code_start..code_start + key_code_bytes],
            );
        }

        Ok(BatchQuantizedKeys {
            norms: norms_flat.to_vec(),
            gammas: vec![0.0; ht],
            packed_codes,
            key_code_bytes,
        })
    }

    /// Batch-quantize values entirely on GPU, returning lazy Arrays.
    ///
    /// Input: `[H, T, D]` f32 tensor.
    /// Returns `(norms: [H, T] f32, packed_codes: [H, T, value_code_words] u32)`.
    /// No `eval()` or CPU readback — the entire graph stays lazy.
    pub fn quantize_values_gpu(&self, values: &Array) -> Result<(Array, Array), Exception> {
        let shape = values.shape();
        let h = shape[0];
        let t = shape[1];
        let n = h * t;

        // norms: [H, T]
        let norms = ops::sqrt(&ops::sum_axis(&ops::square(values)?, -1, None)?)?;
        let eps = Array::from_f32(f32::EPSILON);
        let safe_norms = ops::maximum(&norms, &eps)?;
        let normalized = ops::divide(values, &safe_norms.expand_dims(-1)?)?;

        // rotated: [H, T, D] = hadamard(normalized)
        let rotated = normalized.hadamard_transform(None)?;

        // distances: [H, T, D, C] = |rotated[..., None] - centroids|
        let centroids = self.value_centroids_array()?;
        let distances = ops::abs(&ops::subtract(&rotated.expand_dims(-1)?, &centroids)?)?;

        // indices: [H, T, D] u32
        let indices = argmin_axis!(&distances, -1)?;

        // Norm correction on GPU: gather centroids → L2 → divide norms
        let corrected_norms = if self.config.norm_correction {
            // reconstruction[h,t,d] = centroids[indices[h,t,d]]
            let indices_i32 = indices.as_dtype(Dtype::Int32)?;
            let gathered = centroids.take(&indices_i32)?;
            // reconstruction_l2[h,t] = sqrt(sum(gathered^2, axis=-1))
            let recon_l2 = ops::sqrt(&ops::sum_axis(&ops::square(&gathered)?, -1, None)?)?;
            let safe_recon = ops::maximum(&recon_l2, &eps)?;
            ops::divide(&norms, &safe_recon)?
        } else {
            norms
        };

        // Pack on GPU: [H*T, D] u32 → [H*T, code_words] u32
        let indices_flat = indices.reshape(&[n, self.head_dim])?;
        let packed_flat = pack_indices_gpu(
            &indices_flat,
            n,
            self.head_dim,
            self.config.value_bits(),
            self.value_code_words,
        )?;
        let packed = packed_flat.reshape(&[h, t, self.value_code_words])?;

        Ok((corrected_norms, packed))
    }

    /// Batch-quantize keys entirely on GPU, returning lazy Arrays.
    ///
    /// Input: `[H, T, D]` f32 tensor.
    /// Returns `(norms [H,T], gammas [H,T], packed_codes [H,T,key_code_words] u32)`.
    pub fn quantize_keys_gpu(&self, keys: &Array) -> Result<(Array, Array, Array), Exception> {
        let shape = keys.shape();
        let h = shape[0];
        let t = shape[1];
        let n = h * t;
        let key_bits = self.config.key_bits();

        // norms: [H, T]
        let norms = ops::sqrt(&ops::sum_axis(&ops::square(keys)?, -1, None)?)?;
        let eps = Array::from_f32(f32::EPSILON);
        let safe_norms = ops::maximum(&norms, &eps)?;
        let normalized = ops::divide(keys, &safe_norms.expand_dims(-1)?)?;

        // rotated: [H, T, D] = hadamard(normalized)
        let rotated = normalized.hadamard_transform(None)?;

        // MSE quantize: find nearest centroid in rotated space
        let key_centroids = self.key_centroids_array()?;
        let distances = ops::abs(&ops::subtract(&rotated.expand_dims(-1)?, &key_centroids)?)?;
        let mse_indices = argmin_axis!(&distances, -1)?;

        // Norm correction for keys: use rotated_approx L2
        let corrected_norms = if self.config.norm_correction {
            let mse_i32 = mse_indices.as_dtype(Dtype::Int32)?;
            let rotated_approx = key_centroids.take(&mse_i32)?;
            let recon_l2 = ops::sqrt(&ops::sum_axis(&ops::square(&rotated_approx)?, -1, None)?)?;
            let safe_recon = ops::maximum(&recon_l2, &eps)?;
            ops::divide(&norms, &safe_recon)?
        } else {
            norms
        };

        // Gammas are zeroed out (QJL removed)
        let gammas = ops::zeros_dtype(&[h, t], Dtype::Float32)?;

        // Pack on GPU
        let indices_flat = mse_indices.reshape(&[n, self.head_dim])?;

        let packed_codes_flat = pack_indices_gpu(
            &indices_flat,
            n,
            self.head_dim,
            key_bits,
            self.key_code_words,
        )?;

        let packed_codes = packed_codes_flat.reshape(&[h, t, self.key_code_words])?;

        Ok((corrected_norms, gammas, packed_codes))
    }
}

/// Result of batch value quantization.
pub struct BatchQuantizedValues {
    pub norms: Vec<f32>,
    pub packed_codes: Vec<u8>,
    pub code_bytes: usize,
}

/// Result of batch key quantization.
pub struct BatchQuantizedKeys {
    pub norms: Vec<f32>,
    pub gammas: Vec<f32>,
    pub packed_codes: Vec<u8>,
    pub key_code_bytes: usize,
}

/// Pack u32 index values into a byte buffer at the given bit width.
fn pack_u32_indices(indices: &[u32], bits: u8, out: &mut [u8]) {
    let required_bytes = (indices.len() * usize::from(bits) + 7) / 8;
    assert!(
        out.len() >= required_bytes,
        "pack_u32_indices: output buffer too small ({} < {required_bytes})",
        out.len()
    );
    out.fill(0);
    let mask = (1_u16 << bits) - 1;
    for (index, &code) in indices.iter().enumerate() {
        let bit_index = index * usize::from(bits);
        let byte_index = bit_index / 8;
        let shift = bit_index % 8;
        let value = (code as u16) & mask;
        out[byte_index] |= (value << shift) as u8;
        if shift + usize::from(bits) > 8 {
            out[byte_index + 1] |= (value >> (8 - shift)) as u8;
        }
    }
}

#[allow(unsafe_code)]
pub(crate) fn decode_scores(
    q_rot: &Array,
    key_codes: &Array,
    key_norms: &Array,
    key_centroids: &Array,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    capacity: i32,
    seq_len: i32,
    key_bits: u8,
    key_code_words: i32,
) -> Result<Array, Exception> {
    ensure_ffi_error_handler();

    let stream = Stream::task_local_or_default();
    let kernel = SCORE_KERNEL.get_or_init(|| CachedMetalKernel(create_scores_kernel()));
    let config = configure_scores_kernel(
        num_heads,
        num_kv_heads,
        head_dim,
        seq_len,
        key_bits,
        key_code_words,
    );

    let capacity_scalar = unsafe { mlx_sys::mlx_array_new_int(capacity) };
    let seq_scalar = unsafe { mlx_sys::mlx_array_new_int(seq_len) };
    let input_ptrs = [
        q_rot.as_ptr(),
        key_codes.as_ptr(),
        key_norms.as_ptr(),
        key_centroids.as_ptr(),
        capacity_scalar,
        seq_scalar,
    ];
    let inputs_vec =
        unsafe { mlx_sys::mlx_vector_array_new_data(input_ptrs.as_ptr(), input_ptrs.len()) };

    let mut outputs_vec = unsafe { mlx_sys::mlx_vector_array_new() };
    let status = unsafe {
        mlx_sys::mlx_fast_metal_kernel_apply(
            &raw mut outputs_vec,
            kernel.0,
            inputs_vec,
            config,
            stream.as_ptr(),
        )
    };

    let result = extract_single_output(status, outputs_vec, "turboquant_scores");

    unsafe {
        mlx_sys::mlx_fast_metal_kernel_config_free(config);
        mlx_sys::mlx_vector_array_free(inputs_vec);
        mlx_sys::mlx_vector_array_free(outputs_vec);
        mlx_sys::mlx_array_free(capacity_scalar);
        mlx_sys::mlx_array_free(seq_scalar);
    }

    result
}

#[allow(unsafe_code)]
pub(crate) fn decode_weighted_values(
    weights: &Array,
    value_codes: &Array,
    value_norms: &Array,
    value_centroids: &Array,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    capacity: i32,
    seq_len: i32,
    value_bits: u8,
    value_code_words: i32,
) -> Result<Array, Exception> {
    ensure_ffi_error_handler();

    let stream = Stream::task_local_or_default();
    let kernel = VALUE_KERNEL.get_or_init(|| CachedMetalKernel(create_values_kernel()));
    let config = configure_values_kernel(
        num_heads,
        num_kv_heads,
        head_dim,
        value_bits,
        value_code_words,
    );

    let capacity_scalar = unsafe { mlx_sys::mlx_array_new_int(capacity) };
    let seq_scalar = unsafe { mlx_sys::mlx_array_new_int(seq_len) };
    let input_ptrs = [
        weights.as_ptr(),
        value_codes.as_ptr(),
        value_norms.as_ptr(),
        value_centroids.as_ptr(),
        capacity_scalar,
        seq_scalar,
    ];
    let inputs_vec =
        unsafe { mlx_sys::mlx_vector_array_new_data(input_ptrs.as_ptr(), input_ptrs.len()) };

    let mut outputs_vec = unsafe { mlx_sys::mlx_vector_array_new() };
    let status = unsafe {
        mlx_sys::mlx_fast_metal_kernel_apply(
            &raw mut outputs_vec,
            kernel.0,
            inputs_vec,
            config,
            stream.as_ptr(),
        )
    };

    let result = extract_single_output(status, outputs_vec, "turboquant_values");

    unsafe {
        mlx_sys::mlx_fast_metal_kernel_config_free(config);
        mlx_sys::mlx_vector_array_free(inputs_vec);
        mlx_sys::mlx_vector_array_free(outputs_vec);
        mlx_sys::mlx_array_free(capacity_scalar);
        mlx_sys::mlx_array_free(seq_scalar);
    }

    result
}

fn scaled_centroids(bits: u8, scale: f32) -> Result<Vec<f32>, Exception> {
    let base: &[f32] = match bits {
        1 => &ONE_BIT_CENTROIDS,
        2 => &TWO_BIT_CENTROIDS,
        3 => &THREE_BIT_CENTROIDS,
        4 => &FOUR_BIT_CENTROIDS,
        _ => {
            return Err(Exception::custom(format!(
                "TurboQuant centroids only implemented for 1-4 bits, got {}",
                bits
            )));
        }
    };
    Ok(base.iter().map(|value| *value * scale).collect())
}

fn packed_bytes(dim: usize, bits: u8) -> Result<i32, Exception> {
    let total_bits = dim
        .checked_mul(usize::from(bits))
        .ok_or_else(|| Exception::custom("TurboQuant packed byte overflow"))?;
    i32::try_from(total_bits.div_ceil(8))
        .map_err(|_| Exception::custom("TurboQuant packed byte overflow"))
}

fn packed_words(dim: usize, bits: u8) -> Result<i32, Exception> {
    let total_bits = dim
        .checked_mul(usize::from(bits))
        .ok_or_else(|| Exception::custom("TurboQuant packed word overflow"))?;
    i32::try_from(total_bits.div_ceil(32))
        .map_err(|_| Exception::custom("TurboQuant packed word overflow"))
}

fn l2_norm(values: &[f32]) -> f32 {
    values.iter().map(|value| value * value).sum::<f32>().sqrt()
}

/// L2 norm of the quantized reconstruction: each index maps to a centroid value.
fn reconstruction_norm(indices: &[u8], centroids: &[f32]) -> f32 {
    indices
        .iter()
        .map(|&idx| {
            let c = centroids.get(usize::from(idx)).copied().unwrap_or(0.0);
            c * c
        })
        .sum::<f32>()
        .sqrt()
}

fn quantize_rotated(values: &[f32], centroids: &[f32]) -> Vec<u8> {
    values
        .iter()
        .map(|value| nearest_centroid(*value, centroids))
        .collect()
}

fn nearest_centroid(value: f32, centroids: &[f32]) -> u8 {
    let mut best_index = 0usize;
    let mut best_distance = f32::INFINITY;
    for (index, centroid) in centroids.iter().enumerate() {
        let distance = (value - *centroid).abs();
        if distance < best_distance {
            best_distance = distance;
            best_index = index;
        }
    }
    u8::try_from(best_index).unwrap_or(0)
}

fn dequantize_rotated(
    codes: &[u8],
    bits: u8,
    centroids: &[f32],
    dim: usize,
    norm: f32,
) -> Vec<f32> {
    (0..dim)
        .map(|index| centroids[usize::from(unpack_index(codes, index, bits))] * norm)
        .collect()
}

fn pack_indices(indices: &[u8], bits: u8) -> Vec<u8> {
    let total_bits = indices.len() * usize::from(bits);
    let mut packed = vec![0_u8; total_bits.div_ceil(8)];
    let mask = (1_u16 << bits) - 1;
    for (index, code) in indices.iter().enumerate() {
        let bit_index = index * usize::from(bits);
        let byte_index = bit_index / 8;
        let shift = bit_index % 8;
        let value = u16::from(*code) & mask;
        packed[byte_index] |= (value << shift) as u8;
        if shift + usize::from(bits) > 8 {
            packed[byte_index + 1] |= (value >> (8 - shift)) as u8;
        }
    }
    packed
}

fn unpack_index(data: &[u8], index: usize, bits: u8) -> u8 {
    let bit_index = index * usize::from(bits);
    let byte_index = bit_index / 8;
    let shift = bit_index % 8;
    let mask = (1_u16 << bits) - 1;
    let mut value = u16::from(data[byte_index] >> shift);
    if shift + usize::from(bits) > 8 {
        assert!(
            byte_index + 1 < data.len(),
            "unpack_index: read past end of buffer"
        );
        value |= u16::from(data[byte_index + 1]) << (8 - shift);
    }
    u8::try_from(value & mask).unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Fast Walsh-Hadamard Transform (FWHT) — replaces dense orthogonal rotation.
// ---------------------------------------------------------------------------

/// In-place iterative FWHT on a `&mut [f32]` slice. Length must be a power of 2.
///
/// Computes `H @ x` (unnormalized Hadamard). The caller is responsible for
/// scaling by `1/sqrt(D)` to make the transform orthonormal.
///
/// `H` is the Sylvester-constructed Hadamard matrix of order `D`. Since
/// `H² = D·I`, the orthonormal version `H/√D` is self-inverse, meaning
/// `fwht(fwht(x)) / D = x`.
pub fn fwht(x: &mut [f32]) {
    let n = x.len();
    debug_assert!(n.is_power_of_two(), "FWHT requires power-of-2 length, got {n}");
    let mut stride = n >> 1;
    while stride > 0 {
        for i in 0..n {
            if (i / stride) % 2 == 0 {
                let j = i + stride;
                let a = x[i];
                let b = x[j];
                x[i] = a + b;
                x[j] = a - b;
            }
        }
        stride >>= 1;
    }
}

/// Apply orthonormal FWHT: `H/√D @ x`, which is self-inverse.
///
/// For a vector of length D, this computes `fwht(x) / √D`.
/// Applying twice recovers the original: `fwht_normalized(fwht_normalized(x)) = x`.
pub fn fwht_normalized(x: &mut [f32]) {
    fwht(x);
    let scale = 1.0 / (x.len() as f32).sqrt();
    for val in x.iter_mut() {
        *val *= scale;
    }
}

fn random_normal<R: Rng>(rng: &mut R) -> f32 {
    let u1 = rng.random::<f32>().max(1.0e-7);
    let u2 = rng.random::<f32>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

#[cfg(test)]
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < f32::EPSILON || norm_b < f32::EPSILON {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

// ---------------------------------------------------------------------------
// Metal kernel plumbing.
// ---------------------------------------------------------------------------

/// Per-thread FFI error capture — avoids cross-contamination between threads.
thread_local! {
    static FFI_LAST_ERROR: RefCell<Option<String>> = const { RefCell::new(None) };
}

#[allow(unsafe_code)]
unsafe extern "C" fn ffi_error_handler(msg: *const c_char, _data: *mut c_void) {
    let message = unsafe { CStr::from_ptr(msg) }
        .to_string_lossy()
        .into_owned();
    FFI_LAST_ERROR.with(|cell| {
        *cell.borrow_mut() = Some(message);
    });
}

fn ensure_ffi_error_handler() {
    static REGISTERED: OnceLock<()> = OnceLock::new();
    REGISTERED.get_or_init(|| {
        #[allow(unsafe_code)]
        unsafe {
            mlx_sys::mlx_set_error_handler(Some(ffi_error_handler), std::ptr::null_mut(), None);
        }
    });
}

struct CachedMetalKernel(mlx_sys::mlx_fast_metal_kernel);

// SAFETY: The kernel handle is created once during initialization and used
// read-only thereafter (only passed as an argument to `mlx_fast_metal_kernel_apply`).
// No mutable state is shared across threads.
#[allow(unsafe_code)]
unsafe impl Send for CachedMetalKernel {}
#[allow(unsafe_code)]
unsafe impl Sync for CachedMetalKernel {}

impl Drop for CachedMetalKernel {
    fn drop(&mut self) {
        #[allow(unsafe_code)]
        unsafe {
            mlx_sys::mlx_fast_metal_kernel_free(self.0);
        }
    }
}

static SCORE_KERNEL: OnceLock<CachedMetalKernel> = OnceLock::new();
static VALUE_KERNEL: OnceLock<CachedMetalKernel> = OnceLock::new();
static PACK_KERNEL: OnceLock<CachedMetalKernel> = OnceLock::new();

const TURBOQUANT_SCORES_KERNEL_SOURCE: &str = r"
constexpr int GQA = H / HKV;
auto pos = thread_position_in_grid.x;
auto kv_head = thread_position_in_grid.y;
if (pos >= T) { return; }

auto packed_index = kv_head * Capacity + pos;
auto code_ptr = k_codes + packed_index * KWords;
float norm_val = float(k_norms[packed_index]);

float mse_acc[GQA];
for (int g = 0; g < GQA; ++g) {
  mse_acc[g] = 0.0f;
}

for (int j = 0; j < D; ++j) {
  auto bit_index = j * KBits;
  auto word_index = bit_index / 32;
  auto shift = bit_index % 32;
  uint code = code_ptr[word_index] >> shift;
  if (shift + KBits > 32) {
    code |= code_ptr[word_index + 1] << (32 - shift);
  }
  code &= ((1u << KBits) - 1u);
  float key_val = key_centroids[code];

  for (int g = 0; g < GQA; ++g) {
    int head = kv_head * GQA + g;
    mse_acc[g] += float(q_rot[head * D + j]) * key_val;
  }
}

for (int g = 0; g < GQA; ++g) {
  int head = kv_head * GQA + g;
  scores[head * T + pos] = mse_acc[g] * norm_val;
}
";

const TURBOQUANT_VALUES_KERNEL_SOURCE: &str = r"
auto dim = thread_position_in_grid.x;
auto head = thread_position_in_grid.y;
if (dim >= D) { return; }
auto kv_head = head / (H / HKV);

float acc = 0.0f;
for (int pos = 0; pos < T; ++pos) {
  float w = float(weights[head * T + pos]);
  if (w < 1e-6f) continue;
  auto packed_index = kv_head * Capacity + pos;
  auto bit_index = dim * VBits;
  auto word_index = bit_index / 32;
  auto shift = bit_index % 32;
  auto code_ptr = v_codes + packed_index * VWords;
  uint code = code_ptr[word_index] >> shift;
  if (shift + VBits > 32) {
    code |= code_ptr[word_index + 1] << (32 - shift);
  }
  code &= ((1u << VBits) - 1u);
  acc += w * value_centroids[code] * float(v_norms[packed_index]);
}
out_rot[head * D + dim] = acc;
";

const TURBOQUANT_PACK_KERNEL_SOURCE: &str = r"
int wi = int(thread_position_in_grid.x);
int ri = int(thread_position_in_grid.y);
if (wi >= CodeWords || ri >= N) return;

uint word = 0;
for (int b = 0; b < 32; ++b) {
    int gbit = wi * 32 + b;
    int dim = gbit / Bits;
    int bic = gbit % Bits;
    if (dim >= D) break;
    word |= ((uint(indices[ri * D + dim]) >> bic) & 1u) << b;
}
packed[ri * CodeWords + wi] = word;
";

#[allow(unsafe_code)]
fn create_scores_kernel() -> mlx_sys::mlx_fast_metal_kernel {
    let input_names: [&std::ffi::CStr; 6] = [
        c"q_rot",
        c"k_codes",
        c"k_norms",
        c"key_centroids",
        c"Capacity",
        c"T",
    ];
    let output_names: [&std::ffi::CStr; 1] = [c"scores"];
    let input_ptrs: Vec<*const c_char> = input_names.iter().map(|name| name.as_ptr()).collect();
    let output_ptrs: Vec<*const c_char> = output_names.iter().map(|name| name.as_ptr()).collect();
    let source =
        CString::new(TURBOQUANT_SCORES_KERNEL_SOURCE).unwrap_or_else(|_| CString::default());

    unsafe {
        let in_vec =
            mlx_sys::mlx_vector_string_new_data(input_ptrs.as_ptr().cast_mut(), input_ptrs.len());
        let out_vec =
            mlx_sys::mlx_vector_string_new_data(output_ptrs.as_ptr().cast_mut(), output_ptrs.len());
        let kernel = mlx_sys::mlx_fast_metal_kernel_new(
            c"turboquant_scores".as_ptr(),
            in_vec,
            out_vec,
            source.as_ptr(),
            c"".as_ptr(),
            true,
            false,
        );
        mlx_sys::mlx_vector_string_free(in_vec);
        mlx_sys::mlx_vector_string_free(out_vec);
        kernel
    }
}

#[allow(unsafe_code)]
fn create_values_kernel() -> mlx_sys::mlx_fast_metal_kernel {
    let input_names: [&std::ffi::CStr; 6] = [
        c"weights",
        c"v_codes",
        c"v_norms",
        c"value_centroids",
        c"Capacity",
        c"T",
    ];
    let output_names: [&std::ffi::CStr; 1] = [c"out_rot"];
    let input_ptrs: Vec<*const c_char> = input_names.iter().map(|name| name.as_ptr()).collect();
    let output_ptrs: Vec<*const c_char> = output_names.iter().map(|name| name.as_ptr()).collect();
    let source =
        CString::new(TURBOQUANT_VALUES_KERNEL_SOURCE).unwrap_or_else(|_| CString::default());

    unsafe {
        let in_vec =
            mlx_sys::mlx_vector_string_new_data(input_ptrs.as_ptr().cast_mut(), input_ptrs.len());
        let out_vec =
            mlx_sys::mlx_vector_string_new_data(output_ptrs.as_ptr().cast_mut(), output_ptrs.len());
        let kernel = mlx_sys::mlx_fast_metal_kernel_new(
            c"turboquant_values".as_ptr(),
            in_vec,
            out_vec,
            source.as_ptr(),
            c"".as_ptr(),
            true,
            false,
        );
        mlx_sys::mlx_vector_string_free(in_vec);
        mlx_sys::mlx_vector_string_free(out_vec);
        kernel
    }
}

#[allow(unsafe_code)]
fn configure_scores_kernel(
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    seq_len: i32,
    key_bits: u8,
    key_code_words: i32,
) -> mlx_sys::mlx_fast_metal_kernel_config {
    assert!(num_heads > 0, "num_heads must be positive");
    assert!(num_kv_heads > 0, "num_kv_heads must be positive");
    assert!(head_dim > 0, "head_dim must be positive");
    assert!(seq_len > 0, "seq_len must be positive");
    assert!(key_bits > 0, "key_bits must be positive");
    unsafe {
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(config, c"D".as_ptr(), head_dim);
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"H".as_ptr(),
            num_heads,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"HKV".as_ptr(),
            num_kv_heads,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"KBits".as_ptr(),
            i32::from(key_bits),
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"KWords".as_ptr(),
            key_code_words,
        );
        mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, seq_len, num_kv_heads, 1);
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, 32, 1, 1);
        let shape = [num_heads, seq_len];
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            shape.as_ptr(),
            shape.len(),
            mlx_sys::mlx_dtype__MLX_FLOAT32,
        );
        config
    }
}

#[allow(unsafe_code)]
fn configure_values_kernel(
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    value_bits: u8,
    value_code_words: i32,
) -> mlx_sys::mlx_fast_metal_kernel_config {
    assert!(num_heads > 0, "num_heads must be positive");
    assert!(num_kv_heads > 0, "num_kv_heads must be positive");
    assert!(head_dim > 0, "head_dim must be positive");
    assert!(value_bits > 0, "value_bits must be positive");
    unsafe {
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(config, c"D".as_ptr(), head_dim);
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"H".as_ptr(),
            num_heads,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"HKV".as_ptr(),
            num_kv_heads,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"VBits".as_ptr(),
            i32::from(value_bits),
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"VWords".as_ptr(),
            value_code_words,
        );
        mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, head_dim, num_heads, 1);
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, 32, 1, 1);
        let shape = [num_heads, head_dim];
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            shape.as_ptr(),
            shape.len(),
            mlx_sys::mlx_dtype__MLX_FLOAT32,
        );
        config
    }
}

#[allow(unsafe_code)]
fn extract_single_output(
    status: i32,
    outputs_vec: mlx_sys::mlx_vector_array,
    label: &str,
) -> Result<Array, Exception> {
    if status != 0 {
        let message = FFI_LAST_ERROR
            .with(|cell| cell.borrow_mut().take())
            .unwrap_or_default();
        if message.is_empty() {
            return Err(Exception::custom(format!("{label} kernel failed")));
        }
        return Err(Exception::custom(format!(
            "{label} kernel failed: {message}"
        )));
    }

    let mut out_ptr = unsafe { mlx_sys::mlx_array_new() };
    unsafe {
        mlx_sys::mlx_vector_array_get(&raw mut out_ptr, outputs_vec, 0);
        Ok(Array::from_ptr(out_ptr))
    }
}

// ---------------------------------------------------------------------------
// GPU bit-packing kernels: u32 indices → packed u8 bytes.
// ---------------------------------------------------------------------------

#[allow(unsafe_code)]
fn create_pack_kernel() -> mlx_sys::mlx_fast_metal_kernel {
    let input_names: [&std::ffi::CStr; 2] = [c"indices", c"N"];
    let output_names: [&std::ffi::CStr; 1] = [c"packed"];
    let input_ptrs: Vec<*const c_char> = input_names.iter().map(|name| name.as_ptr()).collect();
    let output_ptrs: Vec<*const c_char> = output_names.iter().map(|name| name.as_ptr()).collect();
    let source = CString::new(TURBOQUANT_PACK_KERNEL_SOURCE).unwrap_or_else(|_| CString::default());

    unsafe {
        let in_vec =
            mlx_sys::mlx_vector_string_new_data(input_ptrs.as_ptr().cast_mut(), input_ptrs.len());
        let out_vec =
            mlx_sys::mlx_vector_string_new_data(output_ptrs.as_ptr().cast_mut(), output_ptrs.len());
        let kernel = mlx_sys::mlx_fast_metal_kernel_new(
            c"turboquant_pack".as_ptr(),
            in_vec,
            out_vec,
            source.as_ptr(),
            c"".as_ptr(),
            true,
            false,
        );
        mlx_sys::mlx_vector_string_free(in_vec);
        mlx_sys::mlx_vector_string_free(out_vec);
        kernel
    }
}

#[allow(unsafe_code)]
fn configure_pack_kernel(
    n: i32,
    head_dim: i32,
    bits: u8,
    code_words: i32,
) -> mlx_sys::mlx_fast_metal_kernel_config {
    unsafe {
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(config, c"D".as_ptr(), head_dim);
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"Bits".as_ptr(),
            i32::from(bits),
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"CodeWords".as_ptr(),
            code_words,
        );
        mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, code_words, n, 1);
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, 32, 1, 1);
        let shape = [n, code_words];
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            shape.as_ptr(),
            shape.len(),
            mlx_sys::mlx_dtype__MLX_UINT32,
        );
        config
    }
}

/// Pack u32 indices into bit-packed u32 words on GPU.
///
/// Input: `indices` `[N, D]` uint32.
/// Output: `[N, code_words]` uint32 with the same bit layout as `pack_u32_indices`.
#[allow(unsafe_code)]
pub(crate) fn pack_indices_gpu(
    indices: &Array,
    n: i32,
    head_dim: i32,
    bits: u8,
    code_words: i32,
) -> Result<Array, Exception> {
    ensure_ffi_error_handler();

    let stream = Stream::task_local_or_default();
    let kernel = PACK_KERNEL.get_or_init(|| CachedMetalKernel(create_pack_kernel()));
    let config = configure_pack_kernel(n, head_dim, bits, code_words);

    let n_scalar = unsafe { mlx_sys::mlx_array_new_int(n) };
    let input_ptrs = [indices.as_ptr(), n_scalar];
    let inputs_vec =
        unsafe { mlx_sys::mlx_vector_array_new_data(input_ptrs.as_ptr(), input_ptrs.len()) };

    let mut outputs_vec = unsafe { mlx_sys::mlx_vector_array_new() };
    let status = unsafe {
        mlx_sys::mlx_fast_metal_kernel_apply(
            &raw mut outputs_vec,
            kernel.0,
            inputs_vec,
            config,
            stream.as_ptr(),
        )
    };

    let result = extract_single_output(status, outputs_vec, "turboquant_pack");

    unsafe {
        mlx_sys::mlx_fast_metal_kernel_config_free(config);
        mlx_sys::mlx_vector_array_free(inputs_vec);
        mlx_sys::mlx_vector_array_free(outputs_vec);
        mlx_sys::mlx_array_free(n_scalar);
    }

    result
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn make_context(bits: u8, head_dim: i32) -> TurboQuantContext {
        let config = KvCacheConfig {
            mode: KvCacheMode::Turboquant,
            bits,
            seed: 42,
            ..Default::default()
        };
        TurboQuantContext::new(config, head_dim, 1).unwrap()
    }

    fn random_vector(dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        (0..dim).map(|_| random_normal(&mut rng)).collect()
    }

    // ── Bit-packing roundtrips ──────────────────────────────────────────

    #[test]
    fn test_pack_unpack_indices_roundtrip() {
        for bits in [2_u8, 3, 4] {
            let max_val = (1 << bits) - 1;
            let indices: Vec<u8> = (0..128).map(|i| (i % (max_val + 1)) as u8).collect();
            let packed = pack_indices(&indices, bits);
            let unpacked: Vec<u8> = (0..128).map(|i| unpack_index(&packed, i, bits)).collect();
            assert_eq!(indices, unpacked, "roundtrip failed for {bits}-bit");
        }
    }

    // ── Value quantize/dequantize roundtrip ─────────────────────────────

    #[test]
    fn test_value_roundtrip_cosine_similarity() {
        for bits in [2_u8, 3, 4] {
            let ctx = make_context(bits, 128);
            let mut total_cos = 0.0_f64;
            let n = 100;
            for seed in 0..n {
                let original = random_vector(128, seed);
                let quantized = ctx.quantize_value(&original).unwrap();
                let recovered = ctx.dequantize_value(&quantized).unwrap();
                let cos = cosine_similarity(&original, &recovered);
                total_cos += f64::from(cos);
            }
            let avg_cos = total_cos / n as f64;
            let min_cos = match bits {
                2 => 0.80,
                3 => 0.90,
                4 => 0.95,
                _ => unreachable!(),
            };
            assert!(
                avg_cos > min_cos,
                "value roundtrip {bits}-bit: avg cos {avg_cos:.4} < {min_cos}"
            );
        }
    }

    // ── Key quantize/dequantize roundtrip ───────────────────────────────

    #[test]
    fn test_key_roundtrip_cosine_similarity() {
        for bits in [2_u8, 3, 4] {
            let ctx = make_context(bits, 128);
            let mut total_cos = 0.0_f64;
            let n = 100;
            for seed in 0..n {
                let original = random_vector(128, seed);
                let quantized = ctx.quantize_key(&original).unwrap();
                let recovered = ctx.dequantize_key(&quantized).unwrap();
                let cos = cosine_similarity(&original, &recovered);
                total_cos += f64::from(cos);
            }
            let avg_cos = total_cos / n as f64;
            // Key uses bits-1 for codes (MSE-only, no QJL correction)
            let min_cos = match bits {
                2 => 0.50,
                3 => 0.75,
                4 => 0.88,
                _ => unreachable!(),
            };
            assert!(
                avg_cos > min_cos,
                "key roundtrip {bits}-bit: avg cos {avg_cos:.4} < {min_cos}"
            );
        }
    }

    // ── Zero vector handling ────────────────────────────────────────────

    #[test]
    fn test_quantize_zero_vector() {
        let ctx = make_context(3, 64);
        let zeros = vec![0.0_f32; 64];

        let qv = ctx.quantize_value(&zeros).unwrap();
        assert_eq!(qv.norm, 0.0);
        let dv = ctx.dequantize_value(&qv).unwrap();
        assert!(dv.iter().all(|v| *v == 0.0));

        let qk = ctx.quantize_key(&zeros).unwrap();
        assert_eq!(qk.norm, 0.0);
        assert_eq!(qk.gamma, 0.0);
        let dk = ctx.dequantize_key(&qk).unwrap();
        assert!(dk.iter().all(|v| *v == 0.0));
    }

    // ── Centroid scaling ────────────────────────────────────────────────

    #[test]
    fn test_scaled_centroids_preserve_relative_order() {
        for bits in [1_u8, 2, 3, 4] {
            let centroids = scaled_centroids(bits, 0.5).unwrap();
            for pair in centroids.windows(2) {
                assert!(pair[0] < pair[1], "{bits}-bit centroids not sorted");
            }
        }
    }

    // ── Config validation ───────────────────────────────────────────────

    #[test]
    fn test_config_validates_bit_range() {
        // value_bits = 10 (too high)
        let bad = KvCacheConfig {
            mode: KvCacheMode::Turboquant,
            bits: 3,
            value_bits_override: Some(10),
            ..Default::default()
        };
        assert!(bad.validate().is_err());

        // key_bits = 0 (too low)
        let bad_key = KvCacheConfig {
            mode: KvCacheMode::Turboquant,
            bits: 3,
            key_bits_override: Some(0),
            ..Default::default()
        };
        assert!(bad_key.validate().is_err());

        // Default bits=3: key=2, value=3 — both valid
        let good = KvCacheConfig {
            mode: KvCacheMode::Turboquant,
            bits: 3,
            seed: 0,
            ..Default::default()
        };
        assert!(good.validate().is_ok());

        // Asymmetric: key=4, value=3 — both valid
        let asym = KvCacheConfig {
            mode: KvCacheMode::Turboquant,
            bits: 3,
            key_bits_override: Some(4),
            value_bits_override: Some(3),
            ..Default::default()
        };
        assert!(asym.validate().is_ok());
    }

    #[test]
    fn test_key_bits_is_bits_minus_one() {
        let config = KvCacheConfig {
            mode: KvCacheMode::Turboquant,
            bits: 3,
            ..Default::default()
        };
        assert_eq!(config.key_bits(), 2);
        assert_eq!(config.value_bits(), 3);

        // With overrides
        let config_asym = KvCacheConfig {
            mode: KvCacheMode::Turboquant,
            bits: 3,
            key_bits_override: Some(4),
            value_bits_override: Some(2),
            ..Default::default()
        };
        assert_eq!(config_asym.key_bits(), 4);
        assert_eq!(config_asym.value_bits(), 2);
    }

    // ── Deterministic with seed ─────────────────────────────────────────

    #[test]
    fn test_quantization_is_deterministic() {
        let ctx1 = make_context(3, 64);
        let ctx2 = make_context(3, 64);
        let vec = random_vector(64, 99);

        let q1 = ctx1.quantize_value(&vec).unwrap();
        let q2 = ctx2.quantize_value(&vec).unwrap();
        assert_eq!(q1.codes, q2.codes);
        assert_eq!(q1.norm, q2.norm);
    }

    // ── Batch vs scalar equivalence ──────────────────────────────────────

    #[test]
    fn test_batch_value_quantize_matches_scalar() {
        for bits in [2_u8, 3, 4] {
            let dim = 64;
            let heads = 2;
            let tokens = 3;
            let ctx = TurboQuantContext::new(
                KvCacheConfig {
                    mode: KvCacheMode::Turboquant,
                    bits,
                    seed: 42,
                    ..Default::default()
                },
                dim,
                heads,
            )
            .unwrap();

            let data: Vec<f32> = (0..(heads * tokens) as usize)
                .flat_map(|seed| random_vector(dim as usize, seed as u64 + 100))
                .collect();
            let arr = Array::from_slice(&data, &[heads, tokens, dim]);

            let batch = ctx.quantize_values_batch(&arr).unwrap();

            // Compare each head×token against scalar quantize_value
            for h in 0..heads as usize {
                for t in 0..tokens as usize {
                    let idx = h * tokens as usize + t;
                    let start = idx * dim as usize;
                    let end = start + dim as usize;
                    let scalar = ctx.quantize_value(&data[start..end]).unwrap();

                    assert!(
                        (batch.norms[idx] - scalar.norm).abs() < 1e-5,
                        "{bits}b h={h} t={t}: norm mismatch {} vs {}",
                        batch.norms[idx],
                        scalar.norm,
                    );

                    let code_start = idx * batch.code_bytes;
                    let code_end = code_start + batch.code_bytes;
                    assert_eq!(
                        &batch.packed_codes[code_start..code_end],
                        &scalar.codes[..],
                        "{bits}b h={h} t={t}: value codes mismatch",
                    );
                }
            }
        }
    }

    #[test]
    fn test_batch_key_quantize_matches_scalar() {
        for bits in [2_u8, 3, 4] {
            let dim = 64;
            let heads = 2;
            let tokens = 3;
            let ctx = TurboQuantContext::new(
                KvCacheConfig {
                    mode: KvCacheMode::Turboquant,
                    bits,
                    seed: 42,
                    ..Default::default()
                },
                dim,
                heads,
            )
            .unwrap();

            let data: Vec<f32> = (0..(heads * tokens) as usize)
                .flat_map(|seed| random_vector(dim as usize, seed as u64 + 200))
                .collect();
            let arr = Array::from_slice(&data, &[heads, tokens, dim]);

            let batch = ctx.quantize_keys_batch(&arr).unwrap();

            for h in 0..heads as usize {
                for t in 0..tokens as usize {
                    let idx = h * tokens as usize + t;
                    let start = idx * dim as usize;
                    let end = start + dim as usize;
                    let scalar = ctx.quantize_key(&data[start..end]).unwrap();

                    assert!(
                        (batch.norms[idx] - scalar.norm).abs() < 1e-5,
                        "{bits}b h={h} t={t}: key norm mismatch {} vs {}",
                        batch.norms[idx],
                        scalar.norm,
                    );

                    let code_start = idx * batch.key_code_bytes;
                    let code_end = code_start + batch.key_code_bytes;
                    assert_eq!(
                        &batch.packed_codes[code_start..code_end],
                        &scalar.codes[..],
                        "{bits}b h={h} t={t}: key codes mismatch",
                    );
                }
            }
        }
    }

    // ── GPU pack kernel correctness ──────────────────────────────────────

    #[test]
    fn test_gpu_pack_indices_matches_cpu() {
        for bits in [2_u8, 3, 4] {
            let dim = 128_i32;
            let n = 16_i32;
            let max_val = (1u32 << bits) - 1;
            let code_bytes = packed_bytes(dim as usize, bits).unwrap();
            let code_words = packed_words(dim as usize, bits).unwrap();

            // Generate deterministic indices
            let indices_vec: Vec<u32> = (0..(n * dim) as u32).map(|i| i % (max_val + 1)).collect();

            // CPU pack (into bytes, then reinterpret as u32 words)
            let mut cpu_packed = vec![0_u8; (n as usize) * (code_bytes as usize)];
            for row in 0..n as usize {
                let row_start = row * dim as usize;
                let out_start = row * code_bytes as usize;
                pack_u32_indices(
                    &indices_vec[row_start..row_start + dim as usize],
                    bits,
                    &mut cpu_packed[out_start..out_start + code_bytes as usize],
                );
            }
            // Reinterpret CPU bytes as u32 words for comparison
            let cpu_words: Vec<u32> = cpu_packed
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();

            // GPU pack (outputs u32 words)
            let indices_arr = Array::from_slice(&indices_vec, &[n, dim]);
            let gpu_packed = pack_indices_gpu(&indices_arr, n, dim, bits, code_words).unwrap();
            gpu_packed.eval().unwrap();
            let gpu_data = gpu_packed.as_slice::<u32>();

            assert_eq!(gpu_data, &cpu_words[..], "GPU pack mismatch for {bits}-bit",);
        }
    }

    #[test]
    fn test_gpu_quantize_values_matches_cpu_batch() {
        for bits in [2_u8, 3, 4] {
            let ctx = make_context(bits, 128);
            let h = 2_i32;
            let t = 4_i32;

            // Generate random data
            let data: Vec<f32> = (0..(h * t * 128) as u64)
                .map(|i| random_normal(&mut rand::rngs::StdRng::seed_from_u64(i + 100)))
                .collect();
            let values = Array::from_slice(&data, &[h, t, 128]);

            // CPU path (existing batch quantize)
            let cpu_result = ctx.quantize_values_batch(&values).unwrap();

            // GPU path (outputs u32 words)
            let (gpu_norms, gpu_codes) = ctx.quantize_values_gpu(&values).unwrap();
            gpu_norms.eval().unwrap();
            gpu_codes.eval().unwrap();

            let gpu_norms_data = gpu_norms.as_slice::<f32>();

            // Compare norms
            for (i, (gpu, cpu)) in gpu_norms_data
                .iter()
                .zip(cpu_result.norms.iter())
                .enumerate()
            {
                assert!(
                    (gpu - cpu).abs() < 1e-5,
                    "{bits}b norm[{i}]: gpu={gpu} vs cpu={cpu}",
                );
            }

            // Reinterpret CPU bytes as u32 words for comparison
            let cpu_words: Vec<u32> = cpu_result
                .packed_codes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let gpu_codes_data = gpu_codes.as_slice::<u32>();
            assert_eq!(
                gpu_codes_data,
                &cpu_words[..],
                "{bits}b: GPU value codes mismatch",
            );
        }
    }

    #[test]
    fn test_gpu_quantize_keys_matches_cpu_batch() {
        for bits in [2_u8, 3, 4] {
            let ctx = make_context(bits, 128);
            let h = 2_i32;
            let t = 4_i32;

            let data: Vec<f32> = (0..(h * t * 128) as u64)
                .map(|i| random_normal(&mut rand::rngs::StdRng::seed_from_u64(i + 200)))
                .collect();
            let keys = Array::from_slice(&data, &[h, t, 128]);

            // CPU path
            let cpu_result = ctx.quantize_keys_batch(&keys).unwrap();

            // GPU path (codes are u32 words now)
            let (gpu_norms, _gpu_gammas, gpu_codes) = ctx.quantize_keys_gpu(&keys).unwrap();
            gpu_norms.eval().unwrap();
            gpu_codes.eval().unwrap();

            // Compare norms
            for (i, (gpu, cpu)) in gpu_norms
                .as_slice::<f32>()
                .iter()
                .zip(cpu_result.norms.iter())
                .enumerate()
            {
                assert!(
                    (gpu - cpu).abs() < 1e-5,
                    "{bits}b key norm[{i}]: gpu={gpu} vs cpu={cpu}",
                );
            }

            // Reinterpret CPU bytes as u32 words for comparison
            let cpu_code_words: Vec<u32> = cpu_result
                .packed_codes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            assert_eq!(
                gpu_codes.as_slice::<u32>(),
                &cpu_code_words[..],
                "{bits}b: GPU key codes mismatch",
            );
        }
    }

    // ── FWHT (Fast Walsh-Hadamard Transform) ──────────────────────────────

    #[test]
    fn test_fwht_known_vector() {
        // H2 @ [a, b] = [a+b, a-b]
        let mut x = vec![3.0_f32, 1.0];
        fwht(&mut x);
        assert_eq!(x[0], 4.0);
        assert_eq!(x[1], 2.0);

        // H4 @ [1,2,3,4]: apply H2 pairwise, then stride-2 butterfly
        let mut v = vec![1.0_f32, 2.0, 3.0, 4.0];
        fwht(&mut v);
        // H4 = [[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]]
        // [1+2+3+4, 1-2+3-4, 1+2-3-4, 1-2-3+4] = [10, -2, -4, 0]
        assert_eq!(v, vec![10.0, -2.0, -4.0, 0.0]);
    }

    #[test]
    fn test_fwht_self_inverse_normalized() {
        // fwht_normalized(fwht_normalized(x)) == x for any x
        for dim in [4, 8, 16, 32, 64, 128] {
            let original = random_vector(dim, 42);
            let mut x = original.clone();
            fwht_normalized(&mut x);
            fwht_normalized(&mut x);
            for (i, (got, expected)) in x.iter().zip(original.iter()).enumerate() {
                assert!(
                    (got - expected).abs() < 1e-4,
                    "FWHT roundtrip failed at dim={dim} idx={i}: got {got}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn test_fwht_norm_preservation_normalized() {
        // Orthonormal FWHT preserves L2 norm
        for dim in [4, 8, 32, 64, 128] {
            let original = random_vector(dim, 99);
            let norm_before = l2_norm(&original);
            let mut x = original.clone();
            fwht_normalized(&mut x);
            let norm_after = l2_norm(&x);
            assert!(
                (norm_before - norm_after).abs() < 1e-4 * norm_before,
                "Norm not preserved at dim={dim}: {norm_before} vs {norm_after}"
            );
        }
    }

    #[test]
    fn test_fwht_involution_property() {
        // fwht(fwht(x)) = D * x (unnormalized)
        for dim in [4, 8, 16, 64, 128] {
            let original = random_vector(dim, 7);
            let mut x = original.clone();
            fwht(&mut x);
            fwht(&mut x);
            let d = dim as f32;
            for (i, (got, expected)) in x.iter().zip(original.iter()).enumerate() {
                assert!(
                    (got - d * expected).abs() < 1e-3 * d,
                    "FWHT² != D*I at dim={dim} idx={i}: got {got}, expected {}",
                    d * expected
                );
            }
        }
    }

    #[test]
    fn test_fwht_gpu_matches_cpu() {
        // MLX hadamard_transform with default scale = 1/sqrt(D) should match
        // our CPU fwht_normalized
        for dim in [4_i32, 8, 16, 32, 64, 128] {
            let original = random_vector(dim as usize, 42);

            // CPU path
            let mut cpu = original.clone();
            fwht_normalized(&mut cpu);

            // GPU path via MLX hadamard_transform
            let arr = Array::from_slice(&original, &[dim]);
            let gpu = arr.hadamard_transform(None).unwrap();
            gpu.eval().unwrap();
            let gpu_data = gpu.as_slice::<f32>();

            for (i, (c, g)) in cpu.iter().zip(gpu_data.iter()).enumerate() {
                assert!(
                    (c - g).abs() < 1e-3,
                    "CPU/GPU mismatch at dim={dim} idx={i}: cpu={c}, gpu={g}"
                );
            }
        }
    }

    #[test]
    fn test_fwht_gpu_batch_matches_cpu() {
        // Test [H, T, D] batch — matches the shape used in quantize_keys/values_gpu
        let h = 2_i32;
        let t = 4_i32;
        let dim = 128_i32;
        let total = (h * t * dim) as usize;
        let original: Vec<f32> = (0..total)
            .map(|i| random_normal(&mut rand::rngs::StdRng::seed_from_u64(i as u64 + 500)))
            .collect();

        // CPU: fwht_normalized per row
        let mut cpu = original.clone();
        for row in 0..(h * t) as usize {
            let start = row * dim as usize;
            let end = start + dim as usize;
            fwht_normalized(&mut cpu[start..end]);
        }

        // GPU: MLX hadamard_transform on [H, T, D]
        let arr = Array::from_slice(&original, &[h, t, dim]);
        let gpu = arr.hadamard_transform(None).unwrap();
        gpu.eval().unwrap();
        let gpu_data = gpu.as_slice::<f32>();

        let mut max_err = 0.0_f32;
        for (i, (c, g)) in cpu.iter().zip(gpu_data.iter()).enumerate() {
            let err = (c - g).abs();
            max_err = max_err.max(err);
            assert!(
                err < 1e-3,
                "Batch CPU/GPU mismatch at idx={i}: cpu={c}, gpu={g}"
            );
        }
    }
}
