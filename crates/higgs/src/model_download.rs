use std::io::{BufRead, Write};

/// Prompt the user to confirm downloading `model_id`, then call `run_download`.
///
/// - `is_interactive`: if `false`, returns an error immediately with a hint to
///   run `huggingface-cli download` manually (e.g. when stdin is not a TTY).
/// - `out`: where the prompt is written (typically stderr).
/// - `input`: where the user's response is read from (typically locked stdin).
/// - `run_download`: called only when the user confirms; should execute the download.
pub fn offer_download<W, R, F>(
    model_id: &str,
    is_interactive: bool,
    out: &mut W,
    mut input: R,
    run_download: F,
) -> Result<(), String>
where
    W: Write,
    R: BufRead,
    F: FnOnce() -> Result<(), String>,
{
    if !is_interactive {
        return Err(format!(
            "model '{model_id}' not found in HuggingFace cache; run: huggingface-cli download {model_id}"
        ));
    }

    write!(
        out,
        "Model '{model_id}' not found in HuggingFace cache. Download now? [y/N] "
    )
    .map_err(|e| e.to_string())?;
    out.flush().map_err(|e| e.to_string())?;

    let mut line = String::new();
    input.read_line(&mut line).map_err(|e| e.to_string())?;

    if !line.trim().eq_ignore_ascii_case("y") {
        return Err(format!("model '{model_id}' not downloaded; aborting"));
    }

    run_download()
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    // Needs to return Result<(), String> to satisfy F: FnOnce() -> Result<(), String>.
    #[allow(clippy::unnecessary_wraps)]
    fn ok_download() -> Result<(), String> {
        Ok(())
    }

    fn err_download() -> Result<(), String> {
        Err("download failed".to_owned())
    }

    #[test]
    fn non_interactive_returns_error_with_hint() {
        let mut out = Vec::new();
        let result = offer_download("org/model", false, &mut out, b"".as_slice(), ok_download);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("org/model"));
        assert!(err.contains("huggingface-cli download"));
    }

    #[test]
    fn non_interactive_writes_no_prompt() {
        let mut out = Vec::new();
        let _ = offer_download("org/model", false, &mut out, b"".as_slice(), ok_download);
        assert!(out.is_empty());
    }

    #[test]
    fn user_confirms_lowercase_succeeds() {
        let mut out = Vec::new();
        let result = offer_download("org/model", true, &mut out, b"y\n".as_slice(), ok_download);
        assert!(result.is_ok());
    }

    #[test]
    fn user_confirms_uppercase_succeeds() {
        let mut out = Vec::new();
        let result = offer_download("org/model", true, &mut out, b"Y\n".as_slice(), ok_download);
        assert!(result.is_ok());
    }

    #[test]
    fn user_declines_does_not_call_download() {
        let mut out = Vec::new();
        let mut called = false;
        let result = offer_download("org/model", true, &mut out, b"n\n".as_slice(), || {
            called = true;
            Ok(())
        });
        assert!(result.is_err());
        assert!(!called);
    }

    #[test]
    fn empty_input_treated_as_no() {
        let mut out = Vec::new();
        let result = offer_download("org/model", true, &mut out, b"\n".as_slice(), ok_download);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("not downloaded"));
    }

    #[test]
    fn download_failure_propagates() {
        let mut out = Vec::new();
        let result = offer_download("org/model", true, &mut out, b"y\n".as_slice(), err_download);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("download failed"));
    }

    #[test]
    fn prompt_contains_model_id() {
        let mut out = Vec::new();
        let _ = offer_download(
            "myorg/mymodel",
            true,
            &mut out,
            b"n\n".as_slice(),
            ok_download,
        );
        let prompt = String::from_utf8(out).unwrap();
        assert!(prompt.contains("myorg/mymodel"));
    }

    #[test]
    fn non_interactive_error_contains_exact_command() {
        let mut out = Vec::new();
        let result = offer_download(
            "myorg/mymodel",
            false,
            &mut out,
            b"".as_slice(),
            ok_download,
        );
        let err = result.unwrap_err();
        assert!(err.contains("huggingface-cli download myorg/mymodel"));
    }
}
