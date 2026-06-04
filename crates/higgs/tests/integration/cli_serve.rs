//! CLI integration tests for `higgs serve`.

#![allow(clippy::panic, clippy::unwrap_used, clippy::tests_outside_test_module)]

use std::process::Command;

fn higgs_bin() -> std::path::PathBuf {
    // cargo sets this during `cargo test`
    let mut path = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    path.push("higgs");
    path
}

#[test]
fn serve_with_non_cached_hf_model() {
    let non_cached_model = "test/non-cached-hf-model";

    let output = Command::new(higgs_bin())
        .args(["serve", "--model", non_cached_model])
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains(&format!("run: hf download {non_cached_model}")),
        "expected 'run: hf download {non_cached_model}' in stderr, got: {stderr}"
    );
}
