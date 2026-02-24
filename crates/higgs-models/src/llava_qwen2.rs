//! LLaVA-Qwen2 vision-language model (nanoLLaVA architecture).
//!
//! Combines a `SigLIP` vision encoder with a Qwen2 language model through an
//! MLP projector. The vision encoder processes images into patch embeddings,
//! which are projected into the LLM's embedding space and concatenated with
//! text token embeddings at positions marked by `IMAGE_TOKEN_INDEX`.

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::{
    Array,
    builder::Builder,
    error::Exception,
    module::{Module, Param},
    nn,
    ops::indexing::{IndexOp, NewAxis},
    transforms::eval,
};
use serde::Deserialize;

use crate::cache::KeyValueCache;
use crate::error::ModelError;
use crate::siglip::{SigLipVisionConfig, SigLipVisionModel, load_siglip_weights};
use crate::transformer;

/// Token ID used as a placeholder for image positions in the input sequence.
pub const IMAGE_TOKEN_INDEX: i32 = -200;

/// Full LLaVA-Qwen2 config from config.json.
#[derive(Debug, Deserialize)]
pub struct LlavaQwen2Config {
    pub hidden_size: i32,
    pub mm_hidden_size: i32,
    pub mm_projector_type: String,
    pub num_hidden_layers: i32,
    pub vision_config: SigLipVisionConfig,
    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
}

#[derive(Debug, Deserialize)]
pub struct QuantizationConfig {
    pub group_size: i32,
    pub bits: i32,
}

// ---------------------------------------------------------------------------
// Multimodal Projector (MLP)
// ---------------------------------------------------------------------------

/// MLP projector mapping vision hidden states to the LLM's embedding space.
/// For `mlp2x_gelu`: Linear -> GELU -> Linear
pub struct MmProjector {
    linear_1: nn::Linear,
    linear_2: nn::Linear,
}

impl MmProjector {
    fn new(vision_dim: i32, lm_dim: i32) -> Result<Self, Exception> {
        Ok(Self {
            linear_1: nn::LinearBuilder::new(vision_dim, lm_dim).build()?,
            linear_2: nn::LinearBuilder::new(lm_dim, lm_dim).build()?,
        })
    }

    fn forward(&mut self, input: &Array) -> Result<Array, Exception> {
        let hidden = self.linear_1.forward(input)?;
        let activated = nn::gelu(&hidden)?;
        self.linear_2.forward(&activated)
    }
}

// ---------------------------------------------------------------------------
// LLaVA-Qwen2 Model
// ---------------------------------------------------------------------------

/// Combined vision-language model.
pub struct LlavaQwen2Model {
    vision_tower: SigLipVisionModel,
    mm_projector: MmProjector,
    language_model: transformer::Model,
    image_size: i32,
}

impl LlavaQwen2Model {
    /// Get the hidden size of the language model.
    pub const fn hidden_size(&self) -> i32 {
        self.language_model.args.hidden_size
    }

    /// Number of transformer layers.
    pub const fn num_hidden_layers(&self) -> i32 {
        self.language_model.args.num_hidden_layers
    }

    /// Get the image size expected by the vision encoder.
    pub const fn image_size(&self) -> i32 {
        self.image_size
    }

    /// Forward pass for text-only (no image).
    pub fn forward_text<C: KeyValueCache>(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut Vec<Option<C>>,
    ) -> Result<Array, Exception> {
        self.language_model.forward(inputs, mask, cache)
    }

    /// Forward pass for text-only, returning hidden states.
    pub fn forward_text_hidden<C: KeyValueCache>(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut Vec<Option<C>>,
    ) -> Result<Array, Exception> {
        self.language_model.forward_hidden(inputs, mask, cache)
    }

    /// Encode an image through the vision tower and projector.
    ///
    /// Input: `pixel_values` with shape `[1, H, W, 3]` (NHWC).
    /// Output: projected features `[1, num_patches, lm_hidden_size]`.
    pub fn encode_image(&mut self, pixel_values: &Array) -> Result<Array, Exception> {
        let hidden_states = self.vision_tower.forward_with_hidden_states(pixel_values)?;

        // nanoLLaVA uses the second-to-last layer output
        let num_states = hidden_states.len();
        let vision_features = hidden_states
            .get(num_states.saturating_sub(2))
            .or_else(|| hidden_states.last())
            .ok_or_else(|| Exception::custom("empty hidden states from vision encoder"))?;

        self.mm_projector.forward(vision_features)
    }

    /// Forward pass with an image.
    ///
    /// `input_ids`: token IDs `[1, seq_len]` with `IMAGE_TOKEN_INDEX` at image positions.
    /// `pixel_values`: preprocessed image `[1, H, W, 3]`.
    /// `cache`: KV cache for the language model.
    ///
    /// Replaces `IMAGE_TOKEN_INDEX` positions with projected image features,
    /// then runs the combined sequence through the language model.
    pub fn forward_multimodal<C: KeyValueCache>(
        &mut self,
        input_ids: &Array,
        pixel_values: &Array,
        cache: &mut Vec<Option<C>>,
    ) -> Result<Array, Exception> {
        // Validate batch=1 assumption
        let batch = input_ids
            .shape()
            .first()
            .copied()
            .ok_or_else(|| Exception::custom("input_ids must have >= 2 dims"))?;
        if batch != 1 {
            return Err(Exception::custom(format!(
                "LLaVA-Qwen2 only supports batch_size=1, got {batch}"
            )));
        }
        let image_features = self.encode_image(pixel_values)?;
        // Replace IMAGE_TOKEN_INDEX sentinel with 0 before embedding lookup to
        // avoid out-of-bounds access. merge_embeddings overwrites these positions.
        let sentinel = Array::from_slice(&[IMAGE_TOKEN_INDEX], &[1]);
        let is_sentinel = input_ids.eq(&sentinel)?;
        let zero = Array::from_slice(&[0_i32], &[1]);
        let safe_ids = mlx_rs::ops::r#where(&is_sentinel, &zero, input_ids)?;
        let text_embeddings = self.language_model.embed_tokens(&safe_ids)?;
        let combined = merge_embeddings(input_ids, &text_embeddings, &image_features)?;
        self.language_model
            .forward_from_embeddings(&combined, None, cache)
    }
}

/// Merge text embeddings and image features at `IMAGE_TOKEN_INDEX` positions.
#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]
fn merge_embeddings(
    input_ids: &Array,
    text_embeddings: &Array,
    image_features: &Array,
) -> Result<Array, Exception> {
    // input_ids: [1, seq_len], text_embeddings: [1, seq_len, hidden_size]
    // image_features: [1, num_patches, hidden_size]
    eval([input_ids])?;
    let ids: Vec<i32> = input_ids.index(0).as_slice::<i32>().to_vec();

    let image_positions: Vec<usize> = ids
        .iter()
        .enumerate()
        .filter(|(_, id)| **id == IMAGE_TOKEN_INDEX)
        .map(|(i, _)| i)
        .collect();

    if image_positions.is_empty() {
        return Ok(text_embeddings.clone());
    }

    // Only single-image is supported; multiple IMAGE_TOKEN_INDEX positions
    // would duplicate the same features, inflating the sequence length.
    if image_positions.len() > 1 {
        return Err(Exception::custom(format!(
            "Expected 1 image token position, found {}",
            image_positions.len()
        )));
    }

    let image_feats = image_features.index(0); // [num_patches, hidden_size]

    let mut segments: Vec<Array> = Vec::new();
    let mut text_start: usize = 0;

    for &img_pos in &image_positions {
        if img_pos > text_start {
            let text_seg = text_embeddings.index((.., text_start as i32..img_pos as i32, ..));
            segments.push(text_seg);
        }
        segments.push(image_feats.index(NewAxis));
        text_start = img_pos + 1;
    }

    let seq_len = ids.len();
    if text_start < seq_len {
        let text_seg = text_embeddings.index((.., text_start as i32..seq_len as i32, ..));
        segments.push(text_seg);
    }

    if segments.len() == 1 {
        return segments
            .into_iter()
            .next()
            .ok_or_else(|| Exception::custom("internal error: empty segments"));
    }

    let seg_refs: Vec<&Array> = segments.iter().collect();
    mlx_rs::ops::concatenate_axis(&seg_refs, 1)
}

// ---------------------------------------------------------------------------
// Weight loading
// ---------------------------------------------------------------------------

/// Load a LLaVA-Qwen2 model from a directory.
pub fn load_llava_qwen2_model(model_dir: &Path) -> Result<LlavaQwen2Model, ModelError> {
    let config_path = model_dir.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)?;
    let config: LlavaQwen2Config = serde_json::from_str(&config_str)?;

    tracing::info!(
        image_size = config.vision_config.image_size,
        vision_layers = config.vision_config.num_hidden_layers,
        vision_hidden = config.vision_config.hidden_size,
        lm_hidden = config.hidden_size,
        lm_layers = config.num_hidden_layers,
        projector = %config.mm_projector_type,
        "Loading LLaVA-Qwen2 model"
    );

    // Build vision encoder
    let mut vision_tower = SigLipVisionModel::new(&config.vision_config)?;

    // Build projector
    let mut mm_projector = MmProjector::new(config.mm_hidden_size, config.hidden_size)?;

    // Build language model (reads text_config, strips language_model. prefix)
    let language_model = transformer::load_vlm_language_model(model_dir)?;

    // Load all safetensor weights for vision and projector
    let weights = load_safetensor_weights(model_dir)?;

    let vision_prefix = "vision_tower.vision_tower.vision_model.";
    load_siglip_weights(&mut vision_tower, &weights, vision_prefix)?;
    load_projector_weights(&mut mm_projector, &weights)?;

    let image_size = config.vision_config.image_size;

    tracing::info!("LLaVA-Qwen2 model loaded successfully");

    Ok(LlavaQwen2Model {
        vision_tower,
        mm_projector,
        language_model,
        image_size,
    })
}

fn load_projector_weights(
    projector: &mut MmProjector,
    weights: &HashMap<String, Array>,
) -> Result<(), ModelError> {
    let get = |name: &str| -> Result<Array, ModelError> {
        weights
            .get(name)
            .cloned()
            .ok_or_else(|| ModelError::MissingWeight(format!("Missing projector weight: {name}")))
    };

    projector.linear_1.weight = Param::new(get("mm_projector.linear_1.weight")?);
    projector.linear_1.bias = Param::new(Some(get("mm_projector.linear_1.bias")?));
    projector.linear_2.weight = Param::new(get("mm_projector.linear_2.weight")?);
    projector.linear_2.bias = Param::new(Some(get("mm_projector.linear_2.bias")?));

    Ok(())
}

/// Load all safetensor weights from a model directory into a `HashMap`.
fn load_safetensor_weights(model_dir: &Path) -> Result<HashMap<String, Array>, ModelError> {
    let index_path = model_dir.join("model.safetensors.index.json");
    let single_path = model_dir.join("model.safetensors");

    let files: Vec<std::path::PathBuf> = if index_path.exists() {
        let index_str = std::fs::read_to_string(&index_path)?;
        let index: serde_json::Value = serde_json::from_str(&index_str)?;

        let weight_map = index
            .get("weight_map")
            .and_then(|v| v.as_object())
            .ok_or_else(|| ModelError::MissingWeight("Missing weight_map in index".to_owned()))?;

        let mut shard_files: Vec<String> = weight_map
            .values()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();
        shard_files.sort();
        shard_files.dedup();
        shard_files.into_iter().map(|f| model_dir.join(f)).collect()
    } else if single_path.exists() {
        vec![single_path]
    } else {
        return Err(ModelError::MissingWeight(
            "No safetensors file found".to_owned(),
        ));
    };

    let mut all_weights = HashMap::new();
    for path in &files {
        let loaded = Array::load_safetensors(path)
            .map_err(|e| ModelError::Io(std::io::Error::other(e.to_string())))?;
        all_weights.extend(loaded);
    }
    Ok(all_weights)
}
