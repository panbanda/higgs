# Higgs

## Project Structure

- `crates/higgs/` -- main binary crate (config, router, doctor, daemon, CLI)
- `crates/higgs-engine/` -- inference engine
- `crates/higgs-models/` -- model architectures
- `apps/desktop/` -- Tauri desktop chat client (separate cargo workspace under `src-tauri/`, excluded from the root workspace)

## Development Rules

### Doctor Validation

When adding or changing config fields, always update `crates/higgs/src/doctor.rs` to validate the new field. The doctor should catch misconfiguration before the server starts. Run `cargo test -p higgs -- --test-threads=1` to verify.

### Documentation

When changing user-facing behavior (config fields, CLI flags, API surface), update:

1. `README.md` -- config examples and reference tables
2. `crates/higgs/src/daemon.rs` -- the `higgs init` config template
3. Doc comments on public structs/fields if applicable

### Testing

- `cargo clippy -p higgs` -- must be clean (nursery lints enabled)
- `cargo fmt -p higgs -- --check` -- must pass
- `cargo test -p higgs -- --test-threads=1` -- all tests must pass (thread limit required due to shared port bindings)
