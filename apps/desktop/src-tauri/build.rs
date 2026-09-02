fn main() {
    ensure_bundled_resource_placeholders();
    tauri_build::build();
}

/// `tauri.conf.json`'s `bundle.resources` names `resources/bin/higgs` and
/// `resources/bin/mlx.metallib`, and `tauri-build` fails the compile step if
/// either is missing, even though only `tauri build` (packaging) actually
/// reads their contents. The release workflow copies the real CLI and
/// metallib into that directory before packaging; for every other build
/// (`cargo check`/`build`, `pnpm tauri dev`), create empty placeholders so
/// the resource paths exist without committing real binaries to the repo.
fn ensure_bundled_resource_placeholders() {
    let dir = std::path::Path::new("resources/bin");
    if let Err(error) = std::fs::create_dir_all(dir) {
        println!("cargo:warning=failed to create {}: {error}", dir.display());
        return;
    }
    for name in ["higgs", "mlx.metallib"] {
        let path = dir.join(name);
        if path.exists() {
            continue;
        }
        if let Err(error) = std::fs::write(&path, b"") {
            println!("cargo:warning=failed to create {}: {error}", path.display());
        }
    }
}
