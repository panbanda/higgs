use clap::Parser;
use higgs::config::Cli;

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
            "0"
        ])
        .is_ok()
    );
}

#[test]
fn rejects_arbitrary_mesh_command_and_model() {
    assert!(Cli::try_parse_from(["higgs", "mesh", "generate", "--command", "sh"]).is_err());
    assert!(Cli::try_parse_from(["higgs", "mesh", "generate", "--model", "remote"]).is_err());
}
