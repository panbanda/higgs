use std::{
    env, fs,
    path::{Path, PathBuf},
};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let is_ci = env::var("CI").is_ok_and(|v| v == "true");
    let is_macos = env::var("CARGO_CFG_TARGET_OS").is_ok_and(|os| os == "macos");

    if let Err(reason) = copy_metallib() {
        if is_macos && !is_ci {
            println!("cargo:warning=failed to set up mlx.metallib: {reason}");
        }
    }
}

/// Copy the metallib of the mlx-sys build this binary links next to the binary.
/// MLX's runtime uses dladdr to look for mlx.metallib next to the executable.
fn copy_metallib() -> Result<(), String> {
    // mlx-sys publishes compile_commands.json from its CMake build directory, which also holds
    // the compiled lib/mlx.metallib. Scanning target/*/build/mlx-sys-* instead can pick up a
    // lingering build of another mlx-sys version.
    let compile_commands = env::var_os("DEP_MLX_COMPILE_COMMANDS")
        .ok_or("mlx-sys did not publish DEP_MLX_COMPILE_COMMANDS")?;
    let metallib = Path::new(&compile_commands)
        .with_file_name("lib")
        .join("mlx.metallib");
    if !metallib.is_file() {
        return Err(format!("{} does not exist", metallib.display()));
    }

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").ok_or("OUT_DIR not set")?);
    let directory = profile_dir(&out_dir)?;
    let destination = directory.join("mlx.metallib");
    fs::copy(&metallib, &destination).map_err(|error| {
        format!(
            "failed to copy {} to {}: {error}",
            metallib.display(),
            destination.display()
        )
    })?;
    println!(
        "cargo:warning=Copied mlx.metallib to {}",
        destination.display()
    );
    Ok(())
}

/// The directory that holds the binary. `OUT_DIR` is `<profile>/build/<crate>-<hash>/out` with older
/// Cargo and `<profile>/build/<crate>/<hash>/out` with the newer layout, so the profile directory
/// is the parent of the nearest `build` ancestor.
///
/// A custom `build-dir` puts that `build` directory away from the binary. The binary then lives
/// under `CARGO_TARGET_DIR`, when that variable is set, in a directory of the same name.
fn profile_dir(out_dir: &Path) -> Result<PathBuf, String> {
    let profile_dir = out_dir
        .ancestors()
        .find(|ancestor| ancestor.file_name().is_some_and(|name| name == "build"))
        .and_then(Path::parent)
        .ok_or("OUT_DIR is not inside a build directory")?;
    if let Some(target_dir) = env::var_os("CARGO_TARGET_DIR").map(PathBuf::from)
        && !profile_dir.starts_with(&target_dir)
        && let Some(name) = profile_dir.file_name()
    {
        let relocated = target_dir.join(name);
        if relocated.is_dir() {
            return Ok(relocated);
        }
    }
    Ok(profile_dir.to_path_buf())
}
