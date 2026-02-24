# Changelog

## [0.1.9](https://github.com/panbanda/mlx-server/compare/mlx-models-v0.1.5...mlx-models-v0.1.9) (2026-02-23)


### Features

* add Qwen3-MoE architecture support ([#29](https://github.com/panbanda/mlx-server/issues/29)) ([bf36aee](https://github.com/panbanda/mlx-server/commit/bf36aeeaeebb2d3355f048d6dc698fa2e0c94250))
* feature parity with vllm-mlx ([#32](https://github.com/panbanda/mlx-server/issues/32)) ([cd71a42](https://github.com/panbanda/mlx-server/commit/cd71a42db4bc0034c93f0412a155b165f5130dda))

## [0.1.5](https://github.com/panbanda/mlx-server/compare/mlx-models-v0.1.4...mlx-models-v0.1.5) (2026-02-22)


### Bug Fixes

* qwen3 correctness (attention_bias, QK norm, RoPE bug) ([#23](https://github.com/panbanda/mlx-server/issues/23)) ([2831957](https://github.com/panbanda/mlx-server/commit/2831957f7cfcbda20d191d88a7fd52e1af1cf4a0))

## [0.1.4](https://github.com/panbanda/mlx-server/compare/mlx-models-v0.1.3...mlx-models-v0.1.4) (2026-02-22)


### Performance Improvements

* fused GPU kernels + dtype fix for 4x speedup (18.6 -&gt; 75 tok/s) ([#18](https://github.com/panbanda/mlx-server/issues/18)) ([8ece387](https://github.com/panbanda/mlx-server/commit/8ece387a3d825972996ef8cb654dbbb3b75f75a3))
* sort expert indices for gather_qmm coalescing (77 -&gt; 80 tok/s) ([#20](https://github.com/panbanda/mlx-server/issues/20)) ([3143fa9](https://github.com/panbanda/mlx-server/commit/3143fa9a3c06edee87162de2fae263e20f34c5b6))

## [0.1.3](https://github.com/panbanda/mlx-server/compare/mlx-models-v0.1.2...mlx-models-v0.1.3) (2026-02-20)


### Features

* resolve HuggingFace model IDs from local cache ([#12](https://github.com/panbanda/mlx-server/issues/12)) ([5ed1949](https://github.com/panbanda/mlx-server/commit/5ed1949a358f4a954bb406c8f4fc8e0c1e3f302e))


### Bug Fixes

* derive readable model names from HuggingFace cache paths ([caf85e9](https://github.com/panbanda/mlx-server/commit/caf85e9f0f2fa08afcce6d13454a2a7871674ffc))

## [0.1.2](https://github.com/panbanda/mlx-server/compare/mlx-models-v0.1.1...mlx-models-v0.1.2) (2026-02-18)


### Features

* publish crates to crates.io on release ([5363b1a](https://github.com/panbanda/mlx-server/commit/5363b1a45c2aadecc3538803b7340adc9d975b7c))


### Bug Fixes

* add doc comments to AnyModel variants and WeightMapIndex fields ([b6a8f0e](https://github.com/panbanda/mlx-server/commit/b6a8f0ea86373a0bf7aeb0218da314a8de89010d))

## [0.1.1](https://github.com/panbanda/mlx-server/compare/mlx-models-v0.1.0...mlx-models-v0.1.1) (2026-02-18)


### Bug Fixes

* **release:** use explicit versions instead of version.workspace = true ([ee353bd](https://github.com/panbanda/mlx-server/commit/ee353bd05ded9ab01b6efdc45b56037949096560))
