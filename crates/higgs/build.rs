use std::{env, fs, path::PathBuf};

fn main() {
    // Find mlx.metallib in the mlx-sys build output and copy it next to the binary.
    // MLX's runtime uses dladdr to look for mlx.metallib next to the executable.
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // OUT_DIR is target/<profile>/build/<crate>-<hash>/out
    // Walk up to target/<profile>/build/ to search mlx-sys-*/out/
    let Some(build_dir) = out_dir.ancestors().nth(2) else {
        return;
    };

    let Ok(entries) = fs::read_dir(build_dir) else {
        return;
    };

    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if !name.starts_with("mlx-sys-") {
            continue;
        }

        let metallib = entry.path().join("out/build/lib/mlx.metallib");
        if !metallib.exists() {
            continue;
        }

        // Copy to target profile dir (e.g. target/release/) so the binary finds it via dladdr
        if let Some(profile_dir) = out_dir.ancestors().nth(3) {
            let _ = fs::copy(&metallib, profile_dir.join("mlx.metallib"));
            println!(
                "cargo:warning=Copied mlx.metallib to {}",
                profile_dir.join("mlx.metallib").display()
            );
        }

        break;
    }
}
