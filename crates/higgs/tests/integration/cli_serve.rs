//! CLI integration tests for `higgs serve`.

#![allow(clippy::panic, clippy::unwrap_used, clippy::tests_outside_test_module)]

use super::higgs_bin;
use std::process::Command;

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
