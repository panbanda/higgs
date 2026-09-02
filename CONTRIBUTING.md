# Contributing

## Project Structure

- `crates/higgs/`: main binary crate, config, router, doctor, daemon, CLI, HTTP routes, and TUI
- `crates/higgs-engine/`: inference engine and runtime behavior
- `crates/higgs-models/`: model loaders and architecture implementations
- `docs/`: user-facing reference docs linked from the README

## Development Commands

Run these before submitting changes:

```bash
cargo test -- --test-threads=1
cargo clippy
cargo fmt --check
```

The test thread limit is required because some tests share port bindings.

## Contributor Expectations

- Keep changes scoped and documented.
- Update user-facing docs when changing CLI flags, config fields, routing behavior, API surface, or benchmark-backed claims.
- Do not describe unsupported behavior in docs.

## Documentation Updates

When changing user-facing behavior, update the relevant sources of truth:

1. `README.md` for landing-page level behavior and examples
2. `docs/configuration.md`, `docs/models.md`, or `docs/benchmarking.md` for detailed reference
3. `crates/higgs/src/daemon.rs` for the `higgs init` config template
4. Public doc comments when applicable

## Validation Rules

When adding or changing config fields:

- update `crates/higgs/src/doctor.rs` so misconfiguration is caught before startup
- update the config template in `crates/higgs/src/daemon.rs`
- update the relevant configuration docs

## Pull Request Quality

- Prefer clear user-visible examples over internal-only wording.
- Keep the README focused on value, quick start, and proof.
- Move long reference material into dedicated docs instead of expanding the landing page.

## Release validation

- Run `scripts/release_smoke_cached_models.sh` to validate the cached MLX models already
  present on the machine without downloading anything.
- Set `HIGGS_SMOKE_INCLUDE_OPTIONAL_MODELS=1` to include optional large or private cached
  models such as `mlx-community/Qwen3.6-35B-A3B-4bit`.
- The harness covers single-model serve, streaming and non-streaming requests, multi-model
  startup, routing precedence, daemon start/attach/stop, and the batch-support guardrails.
