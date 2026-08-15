#![allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::print_stderr,
    clippy::unwrap_used
)]

#[cfg(test)]
mod tests {
    use std::process::Command;

    #[test]
    #[ignore = "requires model files on disk"]
    fn record_then_check_is_self_consistent_and_perturbation_is_measured() {
        let Some(model_dir) = std::env::var_os("HIGGS_MODEL_PATH") else {
            eprintln!("Skipping: set HIGGS_MODEL_PATH to a local model directory");
            return;
        };
        let temp = tempfile::tempdir().unwrap();
        let prompts = temp.path().join("prompts.json");
        let fixture = temp.path().join("fixture.json");
        std::fs::write(&prompts, r#"["The answer is"]"#).unwrap();

        let binary = env!("CARGO_BIN_EXE_quality_gate");
        let record = Command::new(binary)
            .args([
                "record",
                "--model-dir",
                model_dir.to_str().unwrap(),
                "--prompts",
                prompts.to_str().unwrap(),
                "--max-tokens",
                "4",
                "--out",
                fixture.to_str().unwrap(),
            ])
            .status()
            .unwrap();
        assert!(record.success());

        let check = Command::new(binary)
            .args([
                "check",
                "--model-dir",
                model_dir.to_str().unwrap(),
                "--fixture",
                fixture.to_str().unwrap(),
            ])
            .output()
            .unwrap();
        assert!(check.status.success());
        let summary: serde_json::Value = serde_json::from_slice(&check.stdout).unwrap();
        assert!(
            summary["prompts"]
                .as_array()
                .unwrap()
                .iter()
                .all(|prompt| prompt["token_exact"] == true)
        );

        let perturbed = Command::new(binary)
            .args([
                "check",
                "--model-dir",
                model_dir.to_str().unwrap(),
                "--fixture",
                fixture.to_str().unwrap(),
                "--perturb-logits",
                "0.5",
            ])
            .output()
            .unwrap();
        assert_eq!(perturbed.status.code(), Some(1));
        let perturbed_summary: serde_json::Value =
            serde_json::from_slice(&perturbed.stdout).unwrap();
        assert_eq!(perturbed_summary["passed"], false);
        assert!(
            perturbed_summary["prompts"]
                .as_array()
                .unwrap()
                .iter()
                .all(|prompt| prompt["max_abs_logprob_delta"].as_f64().unwrap() > 0.0)
        );
    }
}
