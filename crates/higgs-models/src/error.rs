use mlx_rs::error::{Exception, IoError};

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("MLX error: {0}")]
    Mlx(#[from] Exception),

    #[error("MLX IO error: {0}")]
    MlxIo(#[from] IoError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Unsupported model type: {0}")]
    UnsupportedModel(String),

    #[error("Missing weight: {0}")]
    MissingWeight(String),

    #[error("Shape mismatch: {0}")]
    ShapeMismatch(String),
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn display_unsupported_model() {
        let err = ModelError::UnsupportedModel("gpt5".to_owned());
        assert_eq!(err.to_string(), "Unsupported model type: gpt5");
    }

    #[test]
    fn display_missing_weight() {
        let err = ModelError::MissingWeight("layer.0.weight".to_owned());
        assert_eq!(err.to_string(), "Missing weight: layer.0.weight");
    }

    #[test]
    fn display_shape_mismatch() {
        let err = ModelError::ShapeMismatch("expected [4,4] got [3,3]".to_owned());
        assert_eq!(err.to_string(), "Shape mismatch: expected [4,4] got [3,3]");
    }

    #[test]
    fn display_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file gone");
        let err = ModelError::Io(io_err);
        assert_eq!(err.to_string(), "IO error: file gone");
    }

    #[test]
    fn display_json_error() {
        let json_err = serde_json::from_str::<serde_json::Value>("not json").unwrap_err();
        let msg = json_err.to_string();
        let err = ModelError::Json(json_err);
        assert_eq!(err.to_string(), format!("JSON error: {msg}"));
    }

    #[test]
    fn display_mlx_error() {
        let exc = Exception::custom("bad tensor");
        let err = ModelError::Mlx(exc);
        assert!(err.to_string().contains("bad tensor"));
    }

    #[test]
    fn from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
        let model_err: ModelError = io_err.into();
        assert!(matches!(model_err, ModelError::Io(_)));
        assert!(model_err.to_string().contains("denied"));
    }

    #[test]
    fn from_json_error() {
        let json_err = serde_json::from_str::<serde_json::Value>("{bad}").unwrap_err();
        let model_err: ModelError = json_err.into();
        assert!(matches!(model_err, ModelError::Json(_)));
    }

    #[test]
    fn from_exception() {
        let exc = Exception::custom("mlx failure");
        let model_err: ModelError = exc.into();
        assert!(matches!(model_err, ModelError::Mlx(_)));
        assert!(model_err.to_string().contains("mlx failure"));
    }

    #[test]
    fn error_message_preserves_content() {
        let msg = "very specific error about layer 42 weight mismatch";
        let err = ModelError::MissingWeight(msg.to_owned());
        assert!(err.to_string().contains(msg));

        let err2 = ModelError::UnsupportedModel("my_custom_arch_v99".to_owned());
        assert!(err2.to_string().contains("my_custom_arch_v99"));
    }
}
