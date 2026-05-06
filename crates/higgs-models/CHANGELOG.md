# Changelog

## [1.2.0](https://github.com/panbanda/higgs/compare/higgs-models-v1.1.1...higgs-models-v1.2.0) (2026-05-06)


### Features

* **bonsai-q1:** packed engine scaffold with upstream MLX guard ([#142](https://github.com/panbanda/higgs/issues/142)) ([fe43aab](https://github.com/panbanda/higgs/commit/fe43aabe44104ab285e19e8ff73c724c0875cbe0))
* **cache:** AnyCache::trim_by dispatcher for spec-decode rollback ([#143](https://github.com/panbanda/higgs/issues/143)) ([229c111](https://github.com/panbanda/higgs/commit/229c1110f2da4c3d93ac79c8f28c2435e0c97aad))
* **qwen3_next:** mixed-bit Qwen3.5 GDN BA loading fallback ([#148](https://github.com/panbanda/higgs/issues/148)) ([cc18616](https://github.com/panbanda/higgs/commit/cc186160756c9d5741fc18baecbacd3666eeacf3))


### Bug Fixes

* **deps:** update rust crate safetensors to 0.7 ([#151](https://github.com/panbanda/higgs/issues/151)) ([a177ef6](https://github.com/panbanda/higgs/commit/a177ef6ce08508a08fa20ab9be4175d3a3a77476))


### Performance Improvements

* **models:** opt-in fused MoE gate+up — 3→2 expert matmuls per layer ([#141](https://github.com/panbanda/higgs/issues/141)) ([60d7cb4](https://github.com/panbanda/higgs/commit/60d7cb48f28ddff5254f8aece898a24a1a475d66))

## [1.1.1](https://github.com/panbanda/higgs/compare/higgs-models-v1.1.0...higgs-models-v1.1.1) (2026-04-28)


### Miscellaneous Chores

* **higgs-models:** Synchronize workspace versions

## [1.1.0](https://github.com/panbanda/higgs/compare/higgs-models-v1.0.1...higgs-models-v1.1.0) (2026-04-28)


### Performance Improvements

* **sampling:** partial-sort path for small top-k ([67a5a06](https://github.com/panbanda/higgs/commit/67a5a06f479a9c0ded06af00a07ccafab392ef4c))
* **sampling:** partial-sort path for small top-k ([bab4519](https://github.com/panbanda/higgs/commit/bab4519bdc5bb4d4f2ad7bc90086c054e55131b0))

## [1.0.1](https://github.com/panbanda/higgs/compare/higgs-models-v1.0.0...higgs-models-v1.0.1) (2026-04-27)


### Performance Improvements

* **dtype:** preserve fp16 through scalar multiply in deepseek_v2 + siglip ([5dfbd15](https://github.com/panbanda/higgs/commit/5dfbd159926113e5a53a620933f701d7dd3c6ff2))
* **dtype:** preserve fp16 through scalar multiply in deepseek_v2 + siglip ([90aec29](https://github.com/panbanda/higgs/commit/90aec296ada971231dff254cad1095eac1c92646))

## [1.0.0](https://github.com/panbanda/higgs/compare/higgs-models-v0.1.22...higgs-models-v1.0.0) (2026-04-24)


### ⚠ BREAKING CHANGES

* higgs start is now config/profile-only, attach is a strict daemon dashboard, shellenv/exec fail fast on invalid or unreachable targets, and exact local model matches now take precedence over regex routes.

### Features

* harden CLI, dashboard, routing, and MLX runtime ([4dfc930](https://github.com/panbanda/higgs/commit/4dfc930365ec1d8eb8143508fe63c41b21001ba1))

## [0.1.22](https://github.com/panbanda/higgs/compare/higgs-models-v0.1.21...higgs-models-v0.1.22) (2026-04-24)


### Bug Fixes

* align turboquant tests with rand 0.10 ([d0617c4](https://github.com/panbanda/higgs/commit/d0617c495f4d81f941492c247e45c4b91c996236))
* gate rng import to tests ([1161eaa](https://github.com/panbanda/higgs/commit/1161eaa5c81f11fd736527525abcf2327e604e48))
* restore rng import and checkout pin ([8771fa0](https://github.com/panbanda/higgs/commit/8771fa00c7d0a3f1fd83fd55594c441c95e214ba))

## [0.1.21](https://github.com/panbanda/higgs/compare/higgs-models-v0.1.20...higgs-models-v0.1.21) (2026-04-24)


### Features

* add qwen3.5/qwen3.6 turboquant stack ([4f165ee](https://github.com/panbanda/higgs/commit/4f165eeca1166ac1d1b62cbaf523b83d344b9a22))
* Qwen3.5 model architecture, TurboQuant KV cache, and model-level optimizations ([1514737](https://github.com/panbanda/higgs/commit/1514737d4e1fb108c68b3150e9104aae5b20f400))


### Bug Fixes

* address coderabbit issues ([6f10bc6](https://github.com/panbanda/higgs/commit/6f10bc6d733c192d30c5ab3aed76e4a2e2a69818))
* address review items across models and engine — PR [#74](https://github.com/panbanda/higgs/issues/74) ([8863e2d](https://github.com/panbanda/higgs/commit/8863e2d11dacbdada6e560556813c051bf265d8b))
* cargo fmt + restore higgs crate build on PR [#74](https://github.com/panbanda/higgs/issues/74) ([050af95](https://github.com/panbanda/higgs/commit/050af9590968c129dc8f0a5003d4af47f6591da5))
* clear lint and review blockers ([9c8b878](https://github.com/panbanda/higgs/commit/9c8b878bb654ba8865f3455c91ba457b83d1161b))
* **engine,models:** cache config propagation, ndim guards, FSM ordering ([37d2ddd](https://github.com/panbanda/higgs/commit/37d2ddd3c222614373f9ee3c49b43e5e69e35c62))
* **engine,models:** clippy clean lib targets on PR [#74](https://github.com/panbanda/higgs/issues/74) (51→0 errors) ([ebf3192](https://github.com/panbanda/higgs/commit/ebf319238ed9a79c8efc18bba6b0fbfc6b98f3e3))
* honor qwen3 next gate quantization ([00958e1](https://github.com/panbanda/higgs/commit/00958e1687201e9ed02875dd606bcfb406ab08ed))
* make pre-push pass on dust stack ([2695165](https://github.com/panbanda/higgs/commit/26951659aa8ff56d0362a6916827dbe9cf6fae88))
* **models:** clippy clean for higgs-models on PR [#74](https://github.com/panbanda/higgs/issues/74) (part 2/2) ([1b4470b](https://github.com/panbanda/higgs/commit/1b4470bd7e286c259cf03a63f34fa1a773d9642a))
* **models:** clippy clean in 5/7 files on PR [#74](https://github.com/panbanda/higgs/issues/74) (part 1/2) ([af57c32](https://github.com/panbanda/higgs/commit/af57c3223e81c0328d8217794358e52ec3aadd35))
* satisfy clippy on qwen3.6 tests ([ab2920a](https://github.com/panbanda/higgs/commit/ab2920a182c2a76f8a2e54b67479a044d9513024))
* stabilize dust stack CI ([8e629e3](https://github.com/panbanda/higgs/commit/8e629e35e53bb65b9a6dd718be5d2d50a79e297c))
* support mlx qwen3.6 smoke ([7046849](https://github.com/panbanda/higgs/commit/704684970b3cf43ad9bc3644e74422e378dacfd4))

## [0.1.20](https://github.com/panbanda/higgs/compare/higgs-models-v0.1.19...higgs-models-v0.1.20) (2026-02-28)


### Miscellaneous Chores

* **higgs-models:** Synchronize workspace versions

## [0.1.19](https://github.com/panbanda/higgs/compare/higgs-models-v0.1.18...higgs-models-v0.1.19) (2026-02-28)


### Miscellaneous Chores

* **higgs-models:** Synchronize workspace versions

## [0.1.18](https://github.com/panbanda/higgs/compare/higgs-models-v0.1.17...higgs-models-v0.1.18) (2026-02-28)


### Miscellaneous Chores

* **higgs-models:** Synchronize workspace versions

## [0.1.17](https://github.com/panbanda/higgs/compare/higgs-models-v0.1.16...higgs-models-v0.1.17) (2026-02-27)


### Miscellaneous Chores

* **higgs-models:** Synchronize workspace versions

## [0.1.16](https://github.com/panbanda/higgs/compare/higgs-models-v0.1.15...higgs-models-v0.1.16) (2026-02-27)


### Miscellaneous Chores

* **higgs-models:** Synchronize workspace versions

## [0.1.15](https://github.com/panbanda/higgs/compare/higgs-models-v0.1.14...higgs-models-v0.1.15) (2026-02-27)


### Miscellaneous Chores

* **higgs-models:** Synchronize workspace versions

## [0.1.14](https://github.com/panbanda/higgs/compare/higgs-models-v0.1.13...higgs-models-v0.1.14) (2026-02-27)


### Miscellaneous Chores

* **higgs-models:** Synchronize workspace versions

## [0.1.13](https://github.com/panbanda/higgs/compare/higgs-models-v0.1.12...higgs-models-v0.1.13) (2026-02-27)


### Miscellaneous Chores

* **higgs-models:** Synchronize workspace versions

## [0.1.12](https://github.com/panbanda/higgs/compare/higgs-models-v0.1.11...higgs-models-v0.1.12) (2026-02-25)


### Miscellaneous Chores

* **higgs-models:** Synchronize workspace versions

## [0.1.11](https://github.com/panbanda/higgs/compare/higgs-models-v0.1.10...higgs-models-v0.1.11) (2026-02-25)


### Miscellaneous Chores

* **higgs-models:** Synchronize workspace versions

## [0.1.10](https://github.com/panbanda/higgs/compare/higgs-models-v0.1.9...higgs-models-v0.1.10) (2026-02-25)


### Miscellaneous Chores

* **higgs-models:** Synchronize workspace versions

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
