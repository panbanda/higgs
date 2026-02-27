# Changelog

## [0.1.16](https://github.com/panbanda/higgs/compare/higgs-v0.1.15...higgs-v0.1.16) (2026-02-27)


### Features

* add `higgs exec -- <command>` subcommand ([#49](https://github.com/panbanda/higgs/issues/49)) ([b61dbf5](https://github.com/panbanda/higgs/commit/b61dbf575ff2ec944e81751f654a65b1b746a90f))

## [0.1.15](https://github.com/panbanda/higgs/compare/higgs-v0.1.14...higgs-v0.1.15) (2026-02-27)


### Features

* add `higgs run -- <command>` subcommand ([#47](https://github.com/panbanda/higgs/issues/47)) ([72339d2](https://github.com/panbanda/higgs/commit/72339d270a3c71db4c0ba052a8fddfa91b62c953))

## [0.1.14](https://github.com/panbanda/higgs/compare/higgs-v0.1.13...higgs-v0.1.14) (2026-02-27)


### Features

* add --profile CLI flag, TUI routing tab, and config visibility ([#45](https://github.com/panbanda/higgs/issues/45)) ([95dda05](https://github.com/panbanda/higgs/commit/95dda054071942979aaa2a9c5611a067c0f227d5))

## [0.1.13](https://github.com/panbanda/higgs/compare/higgs-v0.1.12...higgs-v0.1.13) (2026-02-27)


### Features

* add name field to ModelConfig ([#43](https://github.com/panbanda/higgs/issues/43)) ([54147e0](https://github.com/panbanda/higgs/commit/54147e01ec1e1d9d0f308aff1d5228716207b9c7))

## [0.1.12](https://github.com/panbanda/higgs/compare/higgs-v0.1.11...higgs-v0.1.12) (2026-02-25)


### Bug Fixes

* resolve auto_router model by basename when config uses full path ([#41](https://github.com/panbanda/higgs/issues/41)) ([7220fa2](https://github.com/panbanda/higgs/commit/7220fa27ce4f9f67b587ea9a5a3785735d3cfb1d))

## [0.1.11](https://github.com/panbanda/higgs/compare/higgs-v0.1.10...higgs-v0.1.11) (2026-02-25)


### Features

* ship mlx.metallib alongside the higgs binary ([#39](https://github.com/panbanda/higgs/issues/39)) ([deaa322](https://github.com/panbanda/higgs/commit/deaa32275c8ef8236846cc1adc2edf24d698fbe5))

## [0.1.10](https://github.com/panbanda/higgs/compare/higgs-v0.1.9...higgs-v0.1.10) (2026-02-25)


### Features

* unified AI gateway with proxy routing and format translation ([#38](https://github.com/panbanda/higgs/issues/38)) ([7c5668b](https://github.com/panbanda/higgs/commit/7c5668b67a4de113447dce9297c56ace23f5f017))

## [0.1.9](https://github.com/panbanda/mlx-server/compare/mlx-server-v0.1.8...mlx-server-v0.1.9) (2026-02-23)


### Features

* feature parity with vllm-mlx ([#32](https://github.com/panbanda/mlx-server/issues/32)) ([cd71a42](https://github.com/panbanda/mlx-server/commit/cd71a42db4bc0034c93f0412a155b165f5130dda))

## [0.1.8](https://github.com/panbanda/mlx-server/compare/mlx-server-v0.1.7...mlx-server-v0.1.8) (2026-02-22)


### Features

* prompt to download missing HF models via huggingface-cli ([#21](https://github.com/panbanda/mlx-server/issues/21)) ([091058a](https://github.com/panbanda/mlx-server/commit/091058a852529e6703fc4d6fa5edced88ecaa5fd))

## [0.1.7](https://github.com/panbanda/mlx-server/compare/mlx-server-v0.1.6...mlx-server-v0.1.7) (2026-02-20)


### Features

* resolve HuggingFace model IDs from local cache ([#12](https://github.com/panbanda/mlx-server/issues/12)) ([5ed1949](https://github.com/panbanda/mlx-server/commit/5ed1949a358f4a954bb406c8f4fc8e0c1e3f302e))


### Bug Fixes

* handle edge cases in model resolver ([#13](https://github.com/panbanda/mlx-server/issues/13)) ([b1a212f](https://github.com/panbanda/mlx-server/commit/b1a212fcccedd58b55321c52fce2fae417253996))

## [0.1.6](https://github.com/panbanda/mlx-server/compare/mlx-server-v0.1.5...mlx-server-v0.1.6) (2026-02-19)


### Bug Fixes

* correct CLI about text to include Anthropic compatibility ([40fc5c5](https://github.com/panbanda/mlx-server/commit/40fc5c5bd9e337ca4a5b90fe2e97542099caa0ab))

## [0.1.5](https://github.com/panbanda/mlx-server/compare/mlx-server-v0.1.4...mlx-server-v0.1.5) (2026-02-18)


### Features

* serve multiple models via repeated --model flags ([#7](https://github.com/panbanda/mlx-server/issues/7)) ([cc972c3](https://github.com/panbanda/mlx-server/commit/cc972c39c8af961103884e9dc08f835f3822b091))

## [0.1.4](https://github.com/panbanda/mlx-server/compare/mlx-server-v0.1.3...mlx-server-v0.1.4) (2026-02-18)


### Bug Fixes

* add missing doc comments on AppState fields ([6b7e862](https://github.com/panbanda/mlx-server/commit/6b7e8628640765239c8be9b943d7de1e40cd44dd))

## [0.1.3](https://github.com/panbanda/mlx-server/compare/mlx-server-v0.1.2...mlx-server-v0.1.3) (2026-02-18)


### Bug Fixes

* add missing doc comment on ErrorDetail ([ee23882](https://github.com/panbanda/mlx-server/commit/ee238826fe903fdd11ef96dce1ce4641d4234614))

## [0.1.2](https://github.com/panbanda/mlx-server/compare/mlx-server-v0.1.1...mlx-server-v0.1.2) (2026-02-18)


### Features

* publish crates to crates.io on release ([5363b1a](https://github.com/panbanda/mlx-server/commit/5363b1a45c2aadecc3538803b7340adc9d975b7c))

## [0.1.1](https://github.com/panbanda/mlx-server/compare/mlx-server-v0.1.0...mlx-server-v0.1.1) (2026-02-18)


### Bug Fixes

* **release:** use explicit versions instead of version.workspace = true ([ee353bd](https://github.com/panbanda/mlx-server/commit/ee353bd05ded9ab01b6efdc45b56037949096560))
