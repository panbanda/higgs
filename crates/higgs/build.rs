use std::{env, fs, path::PathBuf, process};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let is_macos = env::var("CARGO_CFG_TARGET_OS").is_ok_and(|os| os == "macos");

    if let Err(reason) = copy_metallib() {
        if is_macos {
            println!("cargo:warning=failed to set up mlx.metallib: {reason}");
            process::exit(1);
        }
    }
}

/// Search the mlx-sys build output for mlx.metallib and copy it next to the binary.
/// MLX's runtime uses dladdr to look for mlx.metallib next to the executable.
fn copy_metallib() -> Result<(), &'static str> {
    let Some(out_dir) = env::var("OUT_DIR").map(PathBuf::from).ok() else {
        return Err("OUT_DIR not set");
    };

    // OUT_DIR is target/<profile>/build/<crate>-<hash>/out
    // Walk up to target/<profile>/build/ to search mlx-sys-*/out/
    let Some(build_dir) = out_dir.ancestors().nth(2) else {
        return Err("could not derive build dir from OUT_DIR");
    };

    let Ok(entries) = fs::read_dir(build_dir) else {
        return Err("could not read build dir");
    };

    for entry in entries.flatten() {
        let entry_name = entry.file_name();
        let is_mlx_sys = entry_name
            .to_str()
            .is_some_and(|s| s.starts_with("mlx-sys-"));
        if !is_mlx_sys {
            continue;
        }

        let metallib = entry.path().join("out/build/lib/mlx.metallib");
        if !metallib.exists() {
            continue;
        }

        println!("cargo:rerun-if-changed={}", metallib.display());

        // Copy to target profile dir (e.g. target/release/) so the binary finds it via dladdr
        let Some(profile_dir) = out_dir.ancestors().nth(3) else {
            continue;
        };
        let dst = profile_dir.join("mlx.metallib");
        if let Err(err) = fs::copy(&metallib, &dst) {
            println!(
                "cargo:warning=failed to copy {} to {}: {}",
                metallib.display(),
                dst.display(),
                err
            );
            return Err("copy failed (see warning above)");
        }
        println!("cargo:warning=Copied mlx.metallib to {}", dst.display());
        return Ok(());
    }

    Err("mlx.metallib not found in mlx-sys build output")
}
