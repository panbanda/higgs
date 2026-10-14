//! Opt-in fixed native image-to-mesh adapter, separate from language inference.
use clap::{Args, Subcommand};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};

/// Typed mesh operations. Models and executable commands cannot be supplied.
#[derive(Args, Debug)]
pub struct MeshArgs {
    /// Native adapter operation.
    #[command(subcommand)]
    pub action: MeshAction,
}

/// Operations exposed by the installed fixed mesh runtime.
#[derive(Subcommand, Debug)]
pub enum MeshAction {
    /// Report locally installed model provenance and readiness; never download.
    Capabilities,
    /// Generate a mesh from a previously staged local PNG.
    Generate {
        /// PNG staged in the installed runtime's inputs directory.
        #[arg(long)]
        input: PathBuf,
        /// UUID job directory in the installed runtime's jobs directory.
        #[arg(long)]
        output_dir: PathBuf,
        /// Deterministic sampling seed.
        #[arg(long, default_value_t = 0)]
        seed: u32,
    },
}

/// Create a fixed adapter invocation, without inherited credentials or config.
pub fn command(runtime: &Path, action: &MeshAction) -> Command {
    let mut cmd = Command::new(runtime.join("mesh"));
    cmd.env_clear().env("PATH", "/usr/bin:/bin");
    match action {
        MeshAction::Capabilities => {
            cmd.arg("capabilities");
        }
        MeshAction::Generate {
            input,
            output_dir,
            seed,
        } => {
            cmd.arg("generate")
                .arg("--input")
                .arg(input)
                .arg("--output-dir")
                .arg(output_dir)
                .arg("--seed")
                .arg(seed.to_string());
        }
    }
    cmd
}

/// Run the explicitly installed runtime. This command never loads LLM providers.
pub fn run(args: &MeshArgs) -> std::io::Result<ExitStatus> {
    let home =
        std::env::var_os("HOME").ok_or_else(|| std::io::Error::other("HOME is unavailable"))?;
    let runtime = PathBuf::from(home).join(".local/share/higgs/mesh-runtime");
    command(&runtime, &args.action).status()
}
