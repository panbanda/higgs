# Changelog

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
