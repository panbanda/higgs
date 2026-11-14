//! Per-layer cache state for the hybrid KV and SSM layers.

use crate::cache::SteppingKeyValueCache;
use mlx_rs::{
    Array,
    error::Exception,
    ops::{self, indexing::IndexOp},
    utils::Updatable,
};

/// Rotate a convolution ring buffer `[B, K - 1, C]` into chronological order
/// (oldest first), zero-filling the front when fewer than `K - 1` tokens have
/// been seen. `conv_pos` is the slot holding the newest entry and `offset` the
/// number of tokens processed.
pub fn chronological_conv_history(
    state: &Array,
    conv_pos: i32,
    offset: i32,
) -> Result<Array, Exception> {
    let &[batch, n_keep, conv_dim] = state.shape() else {
        return Err(Exception::custom("conv state must be [B, K - 1, C]"));
    };
    if n_keep <= 0 {
        return Ok(state.clone());
    }

    let available = offset.clamp(0, n_keep);
    if available == 0 {
        return Ok(state.clone());
    }
    if available == n_keep && conv_pos == n_keep - 1 {
        return Ok(state.clone());
    }

    let start = (conv_pos - available + 1).rem_euclid(n_keep);
    let ordered_tail = if start + available <= n_keep {
        state.index((.., start..start + available, ..))
    } else {
        let first = state.index((.., start.., ..));
        let second = state.index((.., ..(start + available - n_keep), ..));
        ops::concatenate(&[&first, &second], 1)?
    };

    if available == n_keep {
        return Ok(ordered_tail);
    }

    let pad = ops::zeros_dtype(&[batch, n_keep - available, conv_dim], state.dtype())?;
    ops::concatenate(&[&pad, &ordered_tail], 1)
}

/// Cache state for a `GatedDeltaNet` layer.
#[derive(Debug, Clone)]
pub struct ArraysCache {
    pub conv_state: Option<Array>,
    pub ssm_state: Option<Array>,
    pub conv_pos: i32,
    pub offset: i32,
}

impl ArraysCache {
    pub const fn new() -> Self {
        Self {
            conv_state: None,
            ssm_state: None,
            conv_pos: -1,
            offset: 0,
        }
    }
}

impl Default for ArraysCache {
    fn default() -> Self {
        Self::new()
    }
}

impl ArraysCache {
    /// Evaluate lazy arrays so a subsequent `clone()` captures values.
    pub fn eval_arrays(&self) -> Result<(), mlx_rs::error::Exception> {
        if let Some(cs) = &self.conv_state {
            cs.eval()?;
        }
        if let Some(ss) = &self.ssm_state {
            ss.eval()?;
        }
        Ok(())
    }
}

impl Updatable for ArraysCache {
    fn state_projection(
        &mut self,
    ) -> Result<mlx_rs::utils::StateProjection<'_>, mlx_rs::error::StateProjectionError> {
        let mut projection = mlx_rs::utils::StateProjection::new();
        projection.optional("ssm_state", &mut self.ssm_state)?;
        Ok(projection)
    }
}

/// Per-layer cache: either KV cache (full attention) or arrays (SSM).
#[derive(Debug, Clone)]
pub enum LayerCache {
    KV(SteppingKeyValueCache),
    Arrays(ArraysCache),
}
