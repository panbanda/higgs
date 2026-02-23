//! `SigLIP` vision encoder for vision-language models.
//!
//! Implements the `SigLIP` Vision Transformer used in `nanoLLaVA` and similar
//! VLMs. Processes images into patch embeddings via `Conv2d` + learned position
//! embeddings, then through standard transformer encoder layers (`LayerNorm`
//! pre-norm, self-attention, MLP with GELU activation).

use mlx_rs::{
    Array, arange,
    builder::Builder,
    error::Exception,
    module::{Module, Param},
    nn, ops,
    transforms::eval,
};
use serde::Deserialize;

/// Configuration for the `SigLIP` vision encoder, read from `vision_config`
/// in the model's config.json.
#[derive(Debug, Clone, Deserialize)]
pub struct SigLipVisionConfig {
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub num_channels: i32,
    pub patch_size: i32,
    pub image_size: i32,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f32,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
}

const fn default_layer_norm_eps() -> f32 {
    1e-6
}
fn default_hidden_act() -> String {
    "gelu_pytorch_tanh".to_owned()
}

impl SigLipVisionConfig {
    pub const fn num_patches(&self) -> i32 {
        (self.image_size / self.patch_size) * (self.image_size / self.patch_size)
    }

    pub const fn head_dim(&self) -> i32 {
        self.hidden_size / self.num_attention_heads
    }
}

// ---------------------------------------------------------------------------
// SigLIP Attention
// ---------------------------------------------------------------------------

struct SigLipAttention {
    q_proj: nn::Linear,
    k_proj: nn::Linear,
    v_proj: nn::Linear,
    out_proj: nn::Linear,
    num_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl SigLipAttention {
    fn new(config: &SigLipVisionConfig) -> Result<Self, Exception> {
        let dim = config.hidden_size;
        let head_dim = config.head_dim();
        #[allow(clippy::as_conversions, clippy::cast_precision_loss)]
        let scale = (head_dim as f32).powf(-0.5);
        Ok(Self {
            q_proj: nn::LinearBuilder::new(dim, dim).build()?,
            k_proj: nn::LinearBuilder::new(dim, dim).build()?,
            v_proj: nn::LinearBuilder::new(dim, dim).build()?,
            out_proj: nn::LinearBuilder::new(dim, dim).build()?,
            num_heads: config.num_attention_heads,
            head_dim,
            scale,
        })
    }

    fn forward(&mut self, hidden_states: &Array) -> Result<Array, Exception> {
        let &[b, seq_len, ..] = hidden_states.shape() else {
            return Err(Exception::custom(
                "SigLipAttention: expected at least 2D input",
            ));
        };

        let queries = self
            .q_proj
            .forward(hidden_states)?
            .reshape(&[b, seq_len, self.num_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let keys = self
            .k_proj
            .forward(hidden_states)?
            .reshape(&[b, seq_len, self.num_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let values = self
            .v_proj
            .forward(hidden_states)?
            .reshape(&[b, seq_len, self.num_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        // Scaled dot-product attention
        let scores = ops::matmul(&queries, &keys.transpose_axes(&[0, 1, 3, 2])?)?
            .multiply(mlx_rs::array!(self.scale))?;
        let weights = ops::softmax_axis(&scores, -1, None)?;
        let attn_output = ops::matmul(&weights, &values)?
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, seq_len, self.num_heads * self.head_dim])?;

        self.out_proj.forward(&attn_output)
    }
}

// ---------------------------------------------------------------------------
// SigLIP MLP
// ---------------------------------------------------------------------------

struct SigLipMlp {
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl SigLipMlp {
    fn new(config: &SigLipVisionConfig) -> Result<Self, Exception> {
        Ok(Self {
            fc1: nn::LinearBuilder::new(config.hidden_size, config.intermediate_size).build()?,
            fc2: nn::LinearBuilder::new(config.intermediate_size, config.hidden_size).build()?,
        })
    }

    fn forward(&mut self, input: &Array) -> Result<Array, Exception> {
        let hidden = self.fc1.forward(input)?;
        let activated = nn::gelu_approximate(&hidden)?;
        self.fc2.forward(&activated)
    }
}

// ---------------------------------------------------------------------------
// SigLIP Encoder Layer
// ---------------------------------------------------------------------------

struct SigLipEncoderLayer {
    self_attn: SigLipAttention,
    layer_norm1: nn::LayerNorm,
    mlp: SigLipMlp,
    layer_norm2: nn::LayerNorm,
}

impl SigLipEncoderLayer {
    fn new(config: &SigLipVisionConfig) -> Result<Self, Exception> {
        Ok(Self {
            self_attn: SigLipAttention::new(config)?,
            layer_norm1: nn::LayerNormBuilder::new(config.hidden_size)
                .eps(config.layer_norm_eps)
                .build()?,
            mlp: SigLipMlp::new(config)?,
            layer_norm2: nn::LayerNormBuilder::new(config.hidden_size)
                .eps(config.layer_norm_eps)
                .build()?,
        })
    }

    fn forward(&mut self, hidden_states: &Array) -> Result<Array, Exception> {
        let normed = self.layer_norm1.forward(hidden_states)?;
        let attn_out = self.self_attn.forward(&normed)?;
        let after_attn = hidden_states.add(&attn_out)?;

        let normed2 = self.layer_norm2.forward(&after_attn)?;
        let mlp_out = self.mlp.forward(&normed2)?;
        after_attn.add(&mlp_out)
    }
}

// ---------------------------------------------------------------------------
// SigLIP Vision Encoder
// ---------------------------------------------------------------------------

/// Complete `SigLIP` vision encoder.
///
/// Takes pixel values `[B, H, W, C]` (NHWC, channels last for MLX) and
/// produces hidden states `[B, num_patches, hidden_size]`.
pub struct SigLipVisionModel {
    patch_embedding: nn::Conv2d,
    position_embedding: nn::Embedding,
    layers: Vec<SigLipEncoderLayer>,
    num_patches: i32,
}

impl SigLipVisionModel {
    pub fn new(config: &SigLipVisionConfig) -> Result<Self, Exception> {
        let num_patches = config.num_patches();

        let patch_embedding =
            nn::Conv2dBuilder::new(config.num_channels, config.hidden_size, config.patch_size)
                .stride(config.patch_size)
                .bias(true)
                .build()?;

        let position_embedding = nn::Embedding::new(num_patches, config.hidden_size)?;

        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            layers.push(SigLipEncoderLayer::new(config)?);
        }

        Ok(Self {
            patch_embedding,
            position_embedding,
            layers,
            num_patches,
        })
    }

    /// Forward pass.
    ///
    /// Input: `pixel_values` with shape `[B, H, W, C]` (NHWC, channels last).
    /// Output: hidden states `[B, num_patches, hidden_size]`.
    pub fn forward(&mut self, pixel_values: &Array) -> Result<Array, Exception> {
        // Conv2d expects NHWC, outputs NHWC: [B, grid_h, grid_w, hidden_size]
        let patch_embeds = self.patch_embedding.forward(pixel_values)?;

        // Flatten spatial dims: [B, grid_h * grid_w, hidden_size]
        let &[b, h, w, c] = patch_embeds.shape() else {
            return Err(Exception::custom("SigLIP: expected 4D output from Conv2d"));
        };
        let embeddings = patch_embeds.reshape(&[b, h * w, c])?;

        // Add position embeddings
        let position_ids = arange!(stop = self.num_patches)?;
        let pos_embeds = self.position_embedding.forward(&position_ids)?;
        let mut hidden_states = embeddings.add(&pos_embeds)?;

        for layer in &mut self.layers {
            hidden_states = layer.forward(&hidden_states)?;
        }

        Ok(hidden_states)
    }

    /// Forward pass returning all hidden states (for extracting intermediate layers).
    pub fn forward_with_hidden_states(
        &mut self,
        pixel_values: &Array,
    ) -> Result<Vec<Array>, Exception> {
        let patch_embeds = self.patch_embedding.forward(pixel_values)?;

        let &[b, h, w, c] = patch_embeds.shape() else {
            return Err(Exception::custom("SigLIP: expected 4D output from Conv2d"));
        };
        let embeddings = patch_embeds.reshape(&[b, h * w, c])?;

        let position_ids = arange!(stop = self.num_patches)?;
        let pos_embeds = self.position_embedding.forward(&position_ids)?;
        let mut hidden_states = embeddings.add(&pos_embeds)?;

        let mut all_hidden_states = vec![hidden_states.clone()];

        for layer in &mut self.layers {
            hidden_states = layer.forward(&hidden_states)?;
            all_hidden_states.push(hidden_states.clone());
        }

        Ok(all_hidden_states)
    }
}

// ---------------------------------------------------------------------------
// Image preprocessing
// ---------------------------------------------------------------------------

/// Preprocess an image for the `SigLIP` vision encoder.
///
/// Steps:
/// 1. Resize to `image_size x image_size`
/// 2. Convert to float32 and normalize to [0, 1]
/// 3. Apply `SigLIP` normalization (maps [0, 1] to [-1, 1])
/// 4. Return as `[1, H, W, 3]` NHWC array
pub fn preprocess_image(
    image_bytes: &[u8],
    image_size: u32,
) -> Result<Array, crate::error::ModelError> {
    use image::imageops::FilterType;
    let img = image::load_from_memory(image_bytes)
        .map_err(|e| crate::error::ModelError::Io(std::io::Error::other(e)))?;

    let resized = img.resize_exact(image_size, image_size, FilterType::Lanczos3);
    let rgb = resized.to_rgb8();

    let (w, h) = rgb.dimensions();
    let pixels = rgb.into_raw();

    let mut float_pixels: Vec<f32> = pixels.iter().map(|&p| f32::from(p) / 255.0).collect();

    // mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]
    for pixel in &mut float_pixels {
        *pixel = (*pixel - 0.5) / 0.5;
    }

    #[allow(clippy::as_conversions, clippy::cast_possible_wrap)]
    let array = Array::from_slice(&float_pixels, &[1, h as i32, w as i32, 3]);

    Ok(array)
}

/// Preprocess an image from a file path.
pub fn preprocess_image_from_path(
    path: &std::path::Path,
    image_size: u32,
) -> Result<Array, crate::error::ModelError> {
    let bytes = std::fs::read(path)?;
    preprocess_image(&bytes, image_size)
}

// ---------------------------------------------------------------------------
// Weight loading helpers
// ---------------------------------------------------------------------------

/// Load `SigLIP` vision model weights from safetensors.
///
/// Expects weights prefixed with `vision_tower.vision_tower.vision_model.`
#[allow(clippy::implicit_hasher)]
pub fn load_siglip_weights(
    model: &mut SigLipVisionModel,
    weights: &std::collections::HashMap<String, Array>,
    prefix: &str,
) -> Result<(), crate::error::ModelError> {
    let get = |name: &str| -> Result<Array, crate::error::ModelError> {
        weights.get(name).cloned().ok_or_else(|| {
            crate::error::ModelError::MissingWeight(format!("Missing weight: {name}"))
        })
    };

    // Patch embedding (Conv2d)
    let pe_prefix = format!("{prefix}embeddings.patch_embedding");
    model.patch_embedding.weight = Param::new(get(&format!("{pe_prefix}.weight"))?);
    model.patch_embedding.bias = Param::new(Some(get(&format!("{pe_prefix}.bias"))?));

    // Position embedding
    let pos_prefix = format!("{prefix}embeddings.position_embedding");
    model.position_embedding.weight = Param::new(get(&format!("{pos_prefix}.weight"))?);

    // Encoder layers
    for (i, layer) in model.layers.iter_mut().enumerate() {
        let lp = format!("{prefix}encoder.layers.{i}");

        layer.layer_norm1.weight = Param::new(Some(get(&format!("{lp}.layer_norm1.weight"))?));
        layer.layer_norm1.bias = Param::new(Some(get(&format!("{lp}.layer_norm1.bias"))?));

        layer.layer_norm2.weight = Param::new(Some(get(&format!("{lp}.layer_norm2.weight"))?));
        layer.layer_norm2.bias = Param::new(Some(get(&format!("{lp}.layer_norm2.bias"))?));

        let attn_p = format!("{lp}.self_attn");
        layer.self_attn.q_proj.weight = Param::new(get(&format!("{attn_p}.q_proj.weight"))?);
        layer.self_attn.q_proj.bias = Param::new(Some(get(&format!("{attn_p}.q_proj.bias"))?));
        layer.self_attn.k_proj.weight = Param::new(get(&format!("{attn_p}.k_proj.weight"))?);
        layer.self_attn.k_proj.bias = Param::new(Some(get(&format!("{attn_p}.k_proj.bias"))?));
        layer.self_attn.v_proj.weight = Param::new(get(&format!("{attn_p}.v_proj.weight"))?);
        layer.self_attn.v_proj.bias = Param::new(Some(get(&format!("{attn_p}.v_proj.bias"))?));
        layer.self_attn.out_proj.weight = Param::new(get(&format!("{attn_p}.out_proj.weight"))?);
        layer.self_attn.out_proj.bias = Param::new(Some(get(&format!("{attn_p}.out_proj.bias"))?));

        let mlp_p = format!("{lp}.mlp");
        layer.mlp.fc1.weight = Param::new(get(&format!("{mlp_p}.fc1.weight"))?);
        layer.mlp.fc1.bias = Param::new(Some(get(&format!("{mlp_p}.fc1.bias"))?));
        layer.mlp.fc2.weight = Param::new(get(&format!("{mlp_p}.fc2.weight"))?);
        layer.mlp.fc2.bias = Param::new(Some(get(&format!("{mlp_p}.fc2.bias"))?));
    }

    // Force evaluation to load weights to GPU
    let mut all_params: Vec<&Array> = Vec::new();
    all_params.push(model.patch_embedding.weight.as_ref());
    if let Some(ref b) = model.patch_embedding.bias.value {
        all_params.push(b);
    }
    all_params.push(model.position_embedding.weight.as_ref());
    eval(all_params)?;

    Ok(())
}
