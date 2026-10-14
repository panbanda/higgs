use clap::Parser;
use higgs::config::Cli;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::process::{Command, Output};

#[test]
fn accepts_typed_mesh_generate_command() {
    assert!(
        Cli::try_parse_from([
            "higgs",
            "mesh",
            "generate",
            "--input",
            "/private/input.png",
            "--output-dir",
            "/private/job",
            "--seed",
            "0",
        ])
        .is_ok()
    );
}

#[test]
fn rejects_arbitrary_mesh_command_and_model_with_all_required_arguments() {
    for option in ["--command", "--model"] {
        let error = Cli::try_parse_from([
            "higgs",
            "mesh",
            "generate",
            "--input",
            "/private/input.png",
            "--output-dir",
            "/private/job",
            option,
            "remote",
        ])
        .unwrap_err();
        assert_eq!(error.kind(), clap::error::ErrorKind::UnknownArgument);
    }
}

#[test]
fn missing_generation_arguments_are_distinct_from_unknown_options() {
    let error = Cli::try_parse_from(["higgs", "mesh", "generate"]).unwrap_err();
    assert_eq!(
        error.kind(),
        clap::error::ErrorKind::MissingRequiredArgument
    );
}

fn invoke_adapter(script: &str) -> Output {
    let home = tempfile::tempdir().unwrap();
    let runtime = home.path().join(".local/share/higgs/mesh-runtime");
    fs::create_dir_all(&runtime).unwrap();
    let executable = runtime.join("mesh");
    fs::write(&executable, format!("#!/bin/sh\n{script}\n")).unwrap();
    fs::set_permissions(&executable, fs::Permissions::from_mode(0o700)).unwrap();
    Command::new(env!("CARGO_BIN_EXE_higgs"))
        .args(["mesh", "capabilities"])
        .env_clear()
        .env("HOME", home.path())
        .env("PATH", "/usr/bin:/bin")
        .output()
        .unwrap()
}

#[test]
fn forwards_adapter_success_and_output() {
    let output = invoke_adapter("printf '%s\n' 'local-adapter-output'; exit 0");
    assert_eq!(output.status.code(), Some(0));
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        "local-adapter-output\n"
    );
}

#[test]
fn preserves_adapter_failure_exit_code() {
    assert_eq!(invoke_adapter("exit 37").status.code(), Some(37));
}

#[test]
fn preserves_signaled_adapter_using_unix_exit_convention() {
    assert_eq!(invoke_adapter("kill -TERM $$").status.code(), Some(143));
}
