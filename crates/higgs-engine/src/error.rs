use higgs_models::error::ModelError;

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("Model error: {0}")]
    Model(#[from] ModelError),

    #[error("MLX error: {0}")]
    Mlx(#[from] mlx_rs::error::Exception),

    #[error("Tokenization error: {0}")]
    Tokenization(String),

    #[error("Template error: {0}")]
    Template(String),

    #[error("Generation error: {0}")]
    Generation(String),
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_error_display_tokenization() {
        let err = EngineError::Tokenization("bad token".to_owned());
        assert!(err.to_string().contains("bad token"));
    }

    #[test]
    fn test_engine_error_display_template() {
        let err = EngineError::Template("syntax error".to_owned());
        assert!(err.to_string().contains("syntax error"));
    }

    #[test]
    fn test_engine_error_display_generation() {
        let err = EngineError::Generation("out of memory".to_owned());
        assert!(err.to_string().contains("out of memory"));
    }

    #[test]
    fn test_engine_error_display_model() {
        let model_err = higgs_models::error::ModelError::UnsupportedModel("gpt5".to_owned());
        let err = EngineError::Model(model_err);
        assert!(err.to_string().contains("gpt5"));
        assert!(err.to_string().contains("Model error"));
    }

    #[test]
    fn test_engine_error_display_mlx() {
        let exc = mlx_rs::error::Exception::custom("tensor shape mismatch");
        let err = EngineError::Mlx(exc);
        assert!(err.to_string().contains("tensor shape mismatch"));
        assert!(err.to_string().contains("MLX error"));
    }

    #[test]
    fn test_from_model_error_to_engine_error() {
        let model_err = higgs_models::error::ModelError::MissingWeight("layer.0".to_owned());
        let engine_err: EngineError = model_err.into();
        assert!(matches!(engine_err, EngineError::Model(_)));
        assert!(engine_err.to_string().contains("layer.0"));
    }

    #[test]
    fn test_from_exception_to_engine_error() {
        let exc = mlx_rs::error::Exception::custom("mlx computation failed");
        let engine_err: EngineError = exc.into();
        assert!(matches!(engine_err, EngineError::Mlx(_)));
        assert!(engine_err.to_string().contains("mlx computation failed"));
    }

    #[test]
    fn test_error_message_content_preserved_through_model_conversion() {
        let specific_msg = "very specific: layer 42, head 7, dim mismatch [128] vs [256]";
        let model_err = higgs_models::error::ModelError::ShapeMismatch(specific_msg.to_owned());
        let engine_err: EngineError = model_err.into();
        assert!(engine_err.to_string().contains(specific_msg));
    }

    #[test]
    fn test_error_message_content_preserved_through_exception_conversion() {
        let specific_msg = "bad dtype: expected float32, got bfloat16 at position 3";
        let exc = mlx_rs::error::Exception::custom(specific_msg);
        let engine_err: EngineError = exc.into();
        assert!(engine_err.to_string().contains(specific_msg));
    }
}
