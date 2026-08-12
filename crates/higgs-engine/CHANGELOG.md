# Changelog

## [1.6.1](https://github.com/panbanda/higgs/compare/higgs-engine-v1.6.0...higgs-engine-v1.6.1) (2026-08-11)


### Miscellaneous Chores

* **higgs-engine:** Synchronize workspace versions

## [1.6.0](https://github.com/panbanda/higgs/compare/higgs-engine-v1.5.0...higgs-engine-v1.6.0) (2026-06-21)


### Features

* **bonsai:** run Bonsai-Q1 bits=1 on vanilla MLX via JIT qmv_fast kernels (no mlx-rs fork) ([#182](https://github.com/panbanda/higgs/issues/182)) ([b604754](https://github.com/panbanda/higgs/commit/b604754cab79f3f91a95d863324db46e8765e14a))
* enable MiniCPM5-1B (explicit head_dim + special-token decode + tojson kwarg + tool parser) ([#176](https://github.com/panbanda/higgs/issues/176)) ([05464e2](https://github.com/panbanda/higgs/commit/05464e22fd67a713ccf8a42d6547fab260d401cb))

## [1.5.0](https://github.com/panbanda/higgs/compare/higgs-engine-v1.4.0...higgs-engine-v1.5.0) (2026-06-21)


### Features

* **chat:** streaming tool-call deltas + Qwen-friendly arg normalisation ([#164](https://github.com/panbanda/higgs/issues/164)) ([8f01163](https://github.com/panbanda/higgs/commit/8f01163a0a60950c011048f8bd3c6d906b3e8a01))
* **qwen35:** Qwen3.6 MoE MTP speculative decode + checkpoint memory-safety fix ([#183](https://github.com/panbanda/higgs/issues/183)) ([4989955](https://github.com/panbanda/higgs/commit/49899557c5e4fef561c6d79ab957ad0656f486ab))
* **serve:** stream prefill progress (llama.cpp-compatible prompt_progress) ([#184](https://github.com/panbanda/higgs/issues/184)) ([e7b2c8a](https://github.com/panbanda/higgs/commit/e7b2c8a41689c7e8e1f1fc4d657a4fa17bba2a96))

## [1.4.0](https://github.com/panbanda/higgs/compare/higgs-engine-v1.3.0...higgs-engine-v1.4.0) (2026-06-21)


### Features

* harden default security posture ([1b41a5c](https://github.com/panbanda/higgs/commit/1b41a5c93696a7492ef4bf7e941982341edf50ab))
* **models:** add Gemma 3 and Gemma 4 inference support ([4f301fc](https://github.com/panbanda/higgs/commit/4f301fcc0d45e2922df4203149f05a24a9f46ae6))


### Bug Fixes

* doc_markdown lints in detok test comments ([deefbe3](https://github.com/panbanda/higgs/commit/deefbe39345669e101b9f1fc5caa90de0aaaa973))
* rename first-token detok bindings to avoid shadowing ([8d16696](https://github.com/panbanda/higgs/commit/8d16696c2d59750d3d742f203ab7316a4785b2be))
* route first streamed token through incremental detokenizer ([722218d](https://github.com/panbanda/higgs/commit/722218df3cfb10c8671721cf45888ec45e31187c))
* split long first doc paragraphs for clippy ([c195312](https://github.com/panbanda/higgs/commit/c1953125d8399f899236d7917f26b94051fccb4b))


### Performance Improvements

* incremental detokenization for streaming generation ([a949e50](https://github.com/panbanda/higgs/commit/a949e50fb3a5c2d650c2d7de367eb4afea444138))

## [1.3.0](https://github.com/panbanda/higgs/compare/higgs-engine-v1.2.0...higgs-engine-v1.3.0) (2026-05-31)


### Features

* **mtp:** release speculative decoding optimizations ([0e5e458](https://github.com/panbanda/higgs/commit/0e5e458ce2b59ab5f1a6aaaa1fe91af6f298fe39))
* **mtp:** release speculative decoding optimizations ([1d15599](https://github.com/panbanda/higgs/commit/1d15599ea86fccb022d786a56b2904131078b150))

## [1.2.0](https://github.com/panbanda/higgs/compare/higgs-engine-v1.1.1...higgs-engine-v1.2.0) (2026-05-06)


### Features

* **bonsai-q1:** packed engine scaffold with upstream MLX guard ([#142](https://github.com/panbanda/higgs/issues/142)) ([fe43aab](https://github.com/panbanda/higgs/commit/fe43aabe44104ab285e19e8ff73c724c0875cbe0))

## [1.1.1](https://github.com/panbanda/higgs/compare/higgs-engine-v1.1.0...higgs-engine-v1.1.1) (2026-04-28)


### Miscellaneous Chores

* **higgs-engine:** Synchronize workspace versions

## [1.1.0](https://github.com/panbanda/higgs/compare/higgs-engine-v1.0.1...higgs-engine-v1.1.0) (2026-04-28)


### Miscellaneous Chores

* **higgs-engine:** Synchronize workspace versions

## [1.0.1](https://github.com/panbanda/higgs/compare/higgs-engine-v1.0.0...higgs-engine-v1.0.1) (2026-04-27)


### Miscellaneous Chores

* **higgs-engine:** Synchronize workspace versions

## [1.0.0](https://github.com/panbanda/higgs/compare/higgs-engine-v0.1.22...higgs-engine-v1.0.0) (2026-04-24)


### ⚠ BREAKING CHANGES

* higgs start is now config/profile-only, attach is a strict daemon dashboard, shellenv/exec fail fast on invalid or unreachable targets, and exact local model matches now take precedence over regex routes.

### Features

* harden CLI, dashboard, routing, and MLX runtime ([4dfc930](https://github.com/panbanda/higgs/commit/4dfc930365ec1d8eb8143508fe63c41b21001ba1))


### Bug Fixes

* address follow-up review issues ([5c857ce](https://github.com/panbanda/higgs/commit/5c857ce32f5eeab4fe020c9760a2bd5a0e984c05))

## [0.1.22](https://github.com/panbanda/higgs/compare/higgs-engine-v0.1.21...higgs-engine-v0.1.22) (2026-04-24)


### Miscellaneous Chores

* **higgs-engine:** Synchronize workspace versions

## [0.1.21](https://github.com/panbanda/higgs/compare/higgs-engine-v0.1.20...higgs-engine-v0.1.21) (2026-04-24)


### Features

* add qwen3.5/qwen3.6 turboquant stack ([4f165ee](https://github.com/panbanda/higgs/commit/4f165eeca1166ac1d1b62cbaf523b83d344b9a22))
* inference engine — chunked prefill, prefix cache, thinking budget, MTP ([4fa0971](https://github.com/panbanda/higgs/commit/4fa09711096dc40cff8b2b33b6c5ad1b87e4b128))
* Qwen3.5 model architecture, TurboQuant KV cache, and model-level optimizations ([1514737](https://github.com/panbanda/higgs/commit/1514737d4e1fb108c68b3150e9104aae5b20f400))
* thinking mode support — reasoning parser and chat template ([a3cc52e](https://github.com/panbanda/higgs/commit/a3cc52e7e1f3ef12118b011156cc7b7bad6c05fb))
* tune qwen3.6 thinking defaults ([6969834](https://github.com/panbanda/higgs/commit/696983431b7c404bf5efea4978398251603ac6ec))


### Bug Fixes

* address coderabbit issues ([6f10bc6](https://github.com/panbanda/higgs/commit/6f10bc6d733c192d30c5ab3aed76e4a2e2a69818))
* address remaining draft review issues ([4396e88](https://github.com/panbanda/higgs/commit/4396e8840e399af4abb60c84dfc4412ba8b2131f))
* address review items across models and engine — PR [#74](https://github.com/panbanda/higgs/issues/74) ([8863e2d](https://github.com/panbanda/higgs/commit/8863e2d11dacbdada6e560556813c051bf265d8b))
* cargo fmt + restore higgs crate build on PR [#74](https://github.com/panbanda/higgs/issues/74) ([050af95](https://github.com/panbanda/higgs/commit/050af9590968c129dc8f0a5003d4af47f6591da5))
* clear lint and review blockers ([9c8b878](https://github.com/panbanda/higgs/commit/9c8b878bb654ba8865f3455c91ba457b83d1161b))
* **engine,models:** cache config propagation, ndim guards, FSM ordering ([37d2ddd](https://github.com/panbanda/higgs/commit/37d2ddd3c222614373f9ee3c49b43e5e69e35c62))
* **engine,models:** clippy clean lib targets on PR [#74](https://github.com/panbanda/higgs/issues/74) (51→0 errors) ([ebf3192](https://github.com/panbanda/higgs/commit/ebf319238ed9a79c8efc18bba6b0fbfc6b98f3e3))
* **engine:** clippy clean 6 files on PR [#74](https://github.com/panbanda/higgs/issues/74) (85→51 errors) ([7dc7370](https://github.com/panbanda/higgs/commit/7dc737052edea2fdf96c4ab7a097051c0ba34562))
* **engine:** prefix cache full-hit, scheduler leak, gather bounds check ([3e35b3f](https://github.com/panbanda/higgs/commit/3e35b3f8e3fe83604883418e408bae09e3533c39))
* make pre-push pass on dust stack ([2695165](https://github.com/panbanda/higgs/commit/26951659aa8ff56d0362a6916827dbe9cf6fae88))
* stabilize dust stack CI ([8e629e3](https://github.com/panbanda/higgs/commit/8e629e35e53bb65b9a6dd718be5d2d50a79e297c))

## [0.1.20](https://github.com/panbanda/higgs/compare/higgs-engine-v0.1.19...higgs-engine-v0.1.20) (2026-02-28)


### Miscellaneous Chores

* **higgs-engine:** Synchronize workspace versions

## [0.1.19](https://github.com/panbanda/higgs/compare/higgs-engine-v0.1.18...higgs-engine-v0.1.19) (2026-02-28)


### Miscellaneous Chores

* **higgs-engine:** Synchronize workspace versions

## [0.1.18](https://github.com/panbanda/higgs/compare/higgs-engine-v0.1.17...higgs-engine-v0.1.18) (2026-02-28)


### Miscellaneous Chores

* **higgs-engine:** Synchronize workspace versions

## [0.1.17](https://github.com/panbanda/higgs/compare/higgs-engine-v0.1.16...higgs-engine-v0.1.17) (2026-02-27)


### Miscellaneous Chores

* **higgs-engine:** Synchronize workspace versions

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
