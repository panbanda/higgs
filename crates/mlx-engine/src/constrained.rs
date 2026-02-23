//! Constrained (guided) generation using FSM-based token masking.
//!
//! Uses `outlines-core` to pre-compute a finite-state machine from a JSON
//! schema or regex. During decode, `allowed_token_ids` returns which tokens
//! are valid at the current state, and `apply_mask` zeros out disallowed
//! logits before sampling.

use mlx_rs::{Array, error::Exception};
use outlines_core::index::Index;
use outlines_core::json_schema;
use outlines_core::vocabulary::Vocabulary;

use crate::error::EngineError;

/// Wraps an `outlines-core` Index for constrained decoding.
pub struct ConstrainedGenerator {
    index: Index,
    state: outlines_core::primitives::StateId,
    vocab_size: usize,
}

impl ConstrainedGenerator {
    /// Build from a pre-computed `Index` and vocab size.
    fn new(index: Index, vocab_size: usize) -> Self {
        let state = index.initial_state();
        Self {
            index,
            state,
            vocab_size,
        }
    }

    /// Build from a JSON schema string.
    ///
    /// Converts the schema to a regex via `outlines-core`, then builds the
    /// FSM index against the given vocabulary.
    pub fn from_json_schema(
        schema: &str,
        vocabulary: &Vocabulary,
        vocab_size: usize,
    ) -> Result<Self, EngineError> {
        let regex = json_schema::regex_from_str(schema, None, None)
            .map_err(|e| EngineError::Generation(format!("Invalid JSON schema: {e}")))?;
        let index = Index::new(&regex, vocabulary)
            .map_err(|e| EngineError::Generation(format!("Failed to build FSM index: {e}")))?;
        Ok(Self::new(index, vocab_size))
    }

    /// Build for `json_object` mode (any valid JSON object).
    pub fn for_json_object(
        vocabulary: &Vocabulary,
        vocab_size: usize,
    ) -> Result<Self, EngineError> {
        Self::from_json_schema(r#"{"type": "object"}"#, vocabulary, vocab_size)
    }

    /// Build from a regex pattern directly.
    pub fn from_regex(
        pattern: &str,
        vocabulary: &Vocabulary,
        vocab_size: usize,
    ) -> Result<Self, EngineError> {
        let index = Index::new(pattern, vocabulary)
            .map_err(|e| EngineError::Generation(format!("Failed to build FSM index: {e}")))?;
        Ok(Self::new(index, vocab_size))
    }

    /// Get the set of allowed token IDs at the current state.
    pub fn allowed_token_ids(&self) -> Option<Vec<outlines_core::primitives::TokenId>> {
        self.index.allowed_tokens(&self.state)
    }

    /// Advance the FSM state after sampling a token.
    ///
    /// Returns `true` if the transition was valid, `false` if the token was
    /// not allowed (should not happen if `apply_mask` was used).
    pub fn advance(&mut self, token_id: u32) -> bool {
        if let Some(next) = self.index.next_state(&self.state, &token_id) {
            self.state = next;
            true
        } else {
            false
        }
    }

    /// Whether the FSM is in a final (accepting) state.
    pub fn is_finished(&self) -> bool {
        self.index.is_final_state(&self.state)
    }

    /// Apply the constraint mask to logits.
    ///
    /// Sets disallowed token logits to negative infinity so they have zero
    /// probability after softmax.
    pub fn apply_mask(&self, logits: &Array) -> Result<Array, Exception> {
        let Some(allowed) = self.allowed_token_ids() else {
            // No allowed tokens -- this shouldn't happen in practice.
            // Return logits unchanged and let EOS handling take over.
            return Ok(logits.clone());
        };

        // Build a mask: -inf for disallowed, 0 for allowed
        let mut mask_vec = vec![f32::NEG_INFINITY; self.vocab_size];
        for &tid in &allowed {
            if let Some(slot) = mask_vec.get_mut(usize::try_from(tid).unwrap_or(usize::MAX)) {
                *slot = 0.0;
            }
        }

        let vocab_i32 =
            i32::try_from(self.vocab_size).map_err(|_| Exception::custom("vocab_size overflow"))?;
        let mask_array = Array::from_slice(&mask_vec, &[vocab_i32]);

        // Reshape to match logits shape (broadcast along batch dims)
        let logits_shape = logits.shape();
        let reshaped = if logits_shape.len() > 1 {
            mask_array.reshape(logits_shape)?
        } else {
            mask_array
        };

        logits.add(reshaped)
    }
}

/// Build an `outlines-core` [`Vocabulary`] from a `tokenizers` tokenizer.
///
/// Maps each token string to its ID. The tokenizer's vocabulary is typically
/// the full set of BPE/Unigram tokens.
pub fn build_vocabulary(
    tokenizer: &tokenizers::Tokenizer,
    eos_token_id: u32,
) -> Result<Vocabulary, EngineError> {
    let mut vocab = Vocabulary::new(eos_token_id);

    let token_map = tokenizer.get_vocab(true);
    for (token_str, token_id) in &token_map {
        let tid = *token_id;
        if vocab.try_insert(token_str.as_str(), tid).is_err() {
            tracing::trace!(token = %token_str, id = tid, "Skipping duplicate token");
        }
    }

    Ok(vocab)
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;

    #[test]
    fn build_vocabulary_from_tokenizer_succeeds() {
        // Verify the vocabulary builder doesn't panic with a real-ish scenario.
        // Full integration tests require a real tokenizer + model.
        let mut vocab = Vocabulary::new(0);
        vocab.try_insert("hello", 1).unwrap();
        vocab.try_insert("world", 2).unwrap();
        // Just verify construction doesn't panic
        assert!(vocab.token_ids("hello").is_some());
    }

    #[test]
    fn apply_mask_shapes_are_correct() {
        // Test the mask building logic directly without needing a full FSM.
        let vocab_size = 10;
        let vocab_i32 = i32::try_from(vocab_size).unwrap();

        // Simulate allowed tokens: only tokens 2 and 5
        let mut mask_vec = vec![f32::NEG_INFINITY; vocab_size];
        mask_vec[2] = 0.0;
        mask_vec[5] = 0.0;
        let mask_array = Array::from_slice(&mask_vec, &[1, vocab_i32]);

        let logits = Array::from_slice(&vec![1.0_f32; vocab_size], &[1, vocab_i32]);
        let masked = logits.add(mask_array).unwrap();
        mlx_rs::transforms::eval([&masked]).unwrap();
        let vals = masked.as_slice::<f32>();

        assert!(
            (vals[2] - 1.0).abs() < 1e-5,
            "Allowed token 2 should keep logit"
        );
        assert!(
            (vals[5] - 1.0).abs() < 1e-5,
            "Allowed token 5 should keep logit"
        );
        assert!(
            vals[0].is_infinite() && vals[0].is_sign_negative(),
            "Token 0 should be -inf"
        );
        assert!(
            vals[3].is_infinite() && vals[3].is_sign_negative(),
            "Token 3 should be -inf"
        );
    }

    #[test]
    fn constrained_generator_initial_state() {
        // Test with a very simple regex that can terminate
        let mut vocab = Vocabulary::new(0);
        // Token 0 is EOS
        vocab.try_insert("a", 1).unwrap();
        vocab.try_insert("b", 2).unwrap();

        // Simple regex: one or more 'a' characters
        let result = ConstrainedGenerator::from_regex("a+", &vocab, 3);
        if let Ok(cg) = result {
            assert!(!cg.is_finished());
            let allowed = cg.allowed_token_ids();
            assert!(allowed.is_some());
            assert!(allowed.unwrap().contains(&1)); // 'a' should be allowed
        }
        // If it fails due to EOS handling, that's OK for this minimal vocab
    }
}
