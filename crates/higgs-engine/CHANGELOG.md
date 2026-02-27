# Changelog

## [0.1.16](https://github.com/panbanda/higgs/compare/higgs-engine-v0.1.15...higgs-engine-v0.1.16) (2026-02-27)


### Miscellaneous Chores

* **higgs-engine:** Synchronize workspace versions

## [0.1.15](https://github.com/panbanda/higgs/compare/higgs-engine-v0.1.14...higgs-engine-v0.1.15) (2026-02-27)


### Miscellaneous Chores

* **higgs-engine:** Synchronize workspace versions

## [0.1.14](https://github.com/panbanda/higgs/compare/higgs-engine-v0.1.13...higgs-engine-v0.1.14) (2026-02-27)


### Miscellaneous Chores

* **higgs-engine:** Synchronize workspace versions

## [0.1.13](https://github.com/panbanda/higgs/compare/higgs-engine-v0.1.12...higgs-engine-v0.1.13) (2026-02-27)


### Miscellaneous Chores

* **higgs-engine:** Synchronize workspace versions

## [0.1.12](https://github.com/panbanda/higgs/compare/higgs-engine-v0.1.11...higgs-engine-v0.1.12) (2026-02-25)


### Miscellaneous Chores

* **higgs-engine:** Synchronize workspace versions

## [0.1.11](https://github.com/panbanda/higgs/compare/higgs-engine-v0.1.10...higgs-engine-v0.1.11) (2026-02-25)


### Miscellaneous Chores

* **higgs-engine:** Synchronize workspace versions

## [0.1.10](https://github.com/panbanda/higgs/compare/higgs-engine-v0.1.9...higgs-engine-v0.1.10) (2026-02-25)


### Miscellaneous Chores

* **higgs-engine:** Synchronize workspace versions

## [0.1.9](https://github.com/panbanda/mlx-server/compare/mlx-engine-v0.1.6...mlx-engine-v0.1.9) (2026-02-23)


### Features

* add Qwen3-MoE architecture support ([#29](https://github.com/panbanda/mlx-server/issues/29)) ([bf36aee](https://github.com/panbanda/mlx-server/commit/bf36aeeaeebb2d3355f048d6dc698fa2e0c94250))
* feature parity with vllm-mlx ([#32](https://github.com/panbanda/mlx-server/issues/32)) ([cd71a42](https://github.com/panbanda/mlx-server/commit/cd71a42db4bc0034c93f0412a155b165f5130dda))

## [0.1.6](https://github.com/panbanda/mlx-server/compare/mlx-engine-v0.1.5...mlx-engine-v0.1.6) (2026-02-22)


### Bug Fixes

* qwen3 correctness (attention_bias, QK norm, RoPE bug) ([#23](https://github.com/panbanda/mlx-server/issues/23)) ([2831957](https://github.com/panbanda/mlx-server/commit/2831957f7cfcbda20d191d88a7fd52e1af1cf4a0))

## [0.1.5](https://github.com/panbanda/mlx-server/compare/mlx-engine-v0.1.4...mlx-engine-v0.1.5) (2026-02-22)


### Performance Improvements

* fused GPU kernels + dtype fix for 4x speedup (18.6 -&gt; 75 tok/s) ([#18](https://github.com/panbanda/mlx-server/issues/18)) ([8ece387](https://github.com/panbanda/mlx-server/commit/8ece387a3d825972996ef8cb654dbbb3b75f75a3))

## [0.1.4](https://github.com/panbanda/mlx-server/compare/mlx-engine-v0.1.3...mlx-engine-v0.1.4) (2026-02-20)


### Performance Improvements

* skip token decoding in generate loop when no stop sequences ([#16](https://github.com/panbanda/mlx-server/issues/16)) ([cd8dbd0](https://github.com/panbanda/mlx-server/commit/cd8dbd079518e94566a48b7609fd4e491962f0f4))
* use async_eval to pipeline GPU execution in decode loop ([#15](https://github.com/panbanda/mlx-server/issues/15)) ([f4a6042](https://github.com/panbanda/mlx-server/commit/f4a60422fa5e9fd67f2487d61f0fdcc7d5885e39))

## [0.1.3](https://github.com/panbanda/mlx-server/compare/mlx-engine-v0.1.2...mlx-engine-v0.1.3) (2026-02-20)


### Features

* resolve HuggingFace model IDs from local cache ([#12](https://github.com/panbanda/mlx-server/issues/12)) ([5ed1949](https://github.com/panbanda/mlx-server/commit/5ed1949a358f4a954bb406c8f4fc8e0c1e3f302e))


### Bug Fixes

* derive readable model names from HuggingFace cache paths ([caf85e9](https://github.com/panbanda/mlx-server/commit/caf85e9f0f2fa08afcce6d13454a2a7871674ffc))

## [0.1.2](https://github.com/panbanda/mlx-server/compare/mlx-engine-v0.1.1...mlx-engine-v0.1.2) (2026-02-18)


### Features

* publish crates to crates.io on release ([5363b1a](https://github.com/panbanda/mlx-server/commit/5363b1a45c2aadecc3538803b7340adc9d975b7c))

## [0.1.1](https://github.com/panbanda/mlx-server/compare/mlx-engine-v0.1.0...mlx-engine-v0.1.1) (2026-02-18)


### Bug Fixes

* **release:** use explicit versions instead of version.workspace = true ([ee353bd](https://github.com/panbanda/mlx-server/commit/ee353bd05ded9ab01b6efdc45b56037949096560))
