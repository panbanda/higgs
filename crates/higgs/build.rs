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

    // Collect candidates and pick the most recently modified, since read_dir
    // iteration order is unspecified and stale build dirs may linger.
    let mut candidates: Vec<PathBuf> = entries
        .flatten()
        .filter_map(|entry| {
            let is_mlx_sys = entry
                .file_name()
                .to_str()
                .is_some_and(|s| s.starts_with("mlx-sys-"));
            if !is_mlx_sys {
                return None;
            }
            let metallib = entry.path().join("out/build/lib/mlx.metallib");
            metallib.exists().then_some(metallib)
        })
        .collect();

    candidates.sort_by_key(|p| fs::metadata(p).and_then(|m| m.modified()).ok());
    let Some(metallib) = candidates.pop() else {
        return Err("mlx.metallib not found in mlx-sys build output");
    };

    // NOTE: don't emit `cargo:rerun-if-changed` for this specific metallib path.
    // When mlx-sys rebuilds in a new build directory (different hash), the old
    // path wouldn't change, so cargo wouldn't re-run this script, leaving a
    // stale metallib that can cause silent Metal failures.

    // Copy to target profile dir (e.g. target/release/) so the binary finds it via dladdr
    let Some(profile_dir) = out_dir.ancestors().nth(3) else {
        return Err("could not derive target profile dir from OUT_DIR");
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
    Ok(())
}
