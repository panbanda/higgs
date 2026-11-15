# Changelog

## [1.9.0](https://github.com/panbanda/higgs/compare/higgs-engine-v1.8.2...higgs-engine-v1.9.0) (2026-11-15)


### Features

* **api:** report cached tokens, reasoning tokens and timings in usage ([0caa206](https://github.com/panbanda/higgs/commit/0caa2063e83f3424dd60ecdfeb4a36cecf9fcc53))
* **batch:** reach feature parity with the simple engine ([d2fae4c](https://github.com/panbanda/higgs/commit/d2fae4c0e99f1db6d235993cbf88f88618f8b914))
* **config:** choose the engine per model with batch = "auto" ([907b473](https://github.com/panbanda/higgs/commit/907b47303113f38cb46778228846d51fd9f7a806))
* **config:** make prefix-cache sizing and hybrid checkpoints configurable ([d16f815](https://github.com/panbanda/higgs/commit/d16f81549847b2b6d155adfad37cba41f26afeae))
* **engine:** add scheduling plan and speculative decode state for the worker ([5131c24](https://github.com/panbanda/higgs/commit/5131c2453915a81c846b9f67e23d36a9973156de))
* **engine:** batched decode for Phi-3 and Starcoder2 through a generic batched cache ([61da6c8](https://github.com/panbanda/higgs/commit/61da6c8ce4575451afd2e0d3a0ed799cea916dfe))
* **engine:** make the thinking budget configurable ([9431c34](https://github.com/panbanda/higgs/commit/9431c344da030f4615440722b9447234c5b8bf8a))
* **engine:** per-model tool-call formats and parsers ([f803bce](https://github.com/panbanda/higgs/commit/f803bce0f2706078084a4e3a6a91fc3e95a397b3))
* **engine:** serve a diverging prompt from the shared blocks of a cached entry ([59bd5b5](https://github.com/panbanda/higgs/commit/59bd5b5fae4258b9ccf1d33a4e15299f6522d676))
* **engine:** serve vision, disk prefix store, TurboQuant and speculative decode from the worker ([e727b26](https://github.com/panbanda/higgs/commit/e727b26184bba144a72176eaadfb1a699a60fd76))
* **engine:** shut an engine down and wait for its MLX state to be freed ([c192c86](https://github.com/panbanda/higgs/commit/c192c869d50aee7ec6a0b72a1c69fb287cafc99b))


### Bug Fixes

* **ci:** declare rust-version on every crate and satisfy Rust 1.100 clippy ([060d1dd](https://github.com/panbanda/higgs/commit/060d1ddcb601d7f6adca07c19166f94645528334))
* **ci:** satisfy the clippy and rustfmt of the CI toolchain ([c48e2f8](https://github.com/panbanda/higgs/commit/c48e2f85a6fa881b674cb03e1817d7cc8b0e99b7))
* **engine:** build no disk prefix tier when the cache keeps no entries ([5246a93](https://github.com/panbanda/higgs/commit/5246a93b6c69a7b018ed72e921cff6769a04734c))
* **engine:** keep Gemma 4 string delimiters through detokenization ([7f22283](https://github.com/panbanda/higgs/commit/7f222830b6153ede0ef275d6fcaf8767c1d6e7eb))
* **engine:** leave the disk prefix directory alone when the cache keeps no entries ([74e4cfc](https://github.com/panbanda/higgs/commit/74e4cfc13fe69accbe55e0117d3707b5ace8c39b))
* **engine:** penalize the in-flight token during pipelined decode ([c1eb041](https://github.com/panbanda/higgs/commit/c1eb041c4f49116bbd2133dd7a9aa9796ce70b31))
* **engine:** plan no checkpoints when the prefix cache keeps no entries ([fa50747](https://github.com/panbanda/higgs/commit/fa507470d50de5c196a88540c342158d802a1d07))
* **engine:** report worker failures to the caller instead of an empty success ([06400db](https://github.com/panbanda/higgs/commit/06400db76499858b3fa5a950896e7c834b48d5fc))
* **engine:** stop the simple engine emitting the forced &lt;/think&gt; twice ([f8d5467](https://github.com/panbanda/higgs/commit/f8d54677f5ec1169948091b6b06c706f452eef77))
* **engine:** take logprob bytes from the detokenizer's text and drop the re-decode ([22b6573](https://github.com/panbanda/higgs/commit/22b65734af209a995774f5ec8c51602b24d3488d))
* **engine:** tool-call format accessor for batch engines without a chat template ([95465f9](https://github.com/panbanda/higgs/commit/95465f9b82d0900bd93cb050a5989235e69ba31c))
* **engine:** trust model_type only for families with unmistakable output ([8da1d42](https://github.com/panbanda/higgs/commit/8da1d4273202bd77da9c3017f308b850821dbc79))


### Performance Improvements

* **engine:** batched decode for the Qwen3.5/3.6 hybrid models ([f58543d](https://github.com/panbanda/higgs/commit/f58543d32c018f99da603051212c22a85805e436))
* **engine:** persistent batched KV cache and pipelined batch decode ([4574fa4](https://github.com/panbanda/higgs/commit/4574fa432e96984be5a43c18a8b19d4296f36241))
* **engine:** resume hybrid prefix-cache entries from recurrent checkpoints ([65f34fe](https://github.com/panbanda/higgs/commit/65f34fe7b7f681031eb862f0383775537fc6375f))
* faster than oMLX on follow-up turns and concurrency, feature parity, one engine ([5522af9](https://github.com/panbanda/higgs/commit/5522af96fb2bcf75d3c0d3a69f2e841a602751a2))

## [1.8.2](https://github.com/panbanda/higgs/compare/higgs-engine-v1.8.1...higgs-engine-v1.8.2) (2026-11-10)


### Miscellaneous Chores

* **higgs-engine:** Synchronize workspace versions

## [1.8.1](https://github.com/panbanda/higgs/compare/higgs-engine-v1.8.0...higgs-engine-v1.8.1) (2026-11-10)


### Bug Fixes

* **engine:** keep MLX state on the thread that owns it ([#310](https://github.com/panbanda/higgs/issues/310)) ([3a1eb0a](https://github.com/panbanda/higgs/commit/3a1eb0a83d448a619dcfdff5f8139c49ae791aeb))
* satisfy clippy lints added in Rust 1.99 ([#302](https://github.com/panbanda/higgs/issues/302)) ([837f1c2](https://github.com/panbanda/higgs/commit/837f1c2ac055e763bfc8f46eb4a263d0bd1582eb))

## [1.8.0](https://github.com/panbanda/higgs/compare/higgs-engine-v1.7.0...higgs-engine-v1.8.0) (2026-09-02)


### Miscellaneous Chores

* **higgs-engine:** Synchronize workspace versions

## [1.7.0](https://github.com/panbanda/higgs/compare/higgs-engine-v1.6.1...higgs-engine-v1.7.0) (2026-08-18)


### Features

* **bench:** context-frontier benchmark with shared cache rollback helper (PR2 / ds4 P8) ([#253](https://github.com/panbanda/higgs/issues/253)) ([4510835](https://github.com/panbanda/higgs/commit/451083523816e33cf636d796e240c6bb21a0a62d))
* **engine:** chunked prefill yield quantum for the batch serving loop (PR8 / ds4 P9) ([#259](https://github.com/panbanda/higgs/issues/259)) ([247147e](https://github.com/panbanda/higgs/commit/247147eacfa04ad97af6d1b5aca974594e8da736))
* **engine:** disk-backed KV prefix store for restart-resume (PR6 / ds4 P4) ([#257](https://github.com/panbanda/higgs/issues/257)) ([0435d11](https://github.com/panbanda/higgs/commit/0435d112ff8abf3de65ed554a95e3e3a9b69d690))
* **models:** MLA latent KV cache with weight absorption for DeepSeek-V2 (PR3 / ds4 P1) ([#254](https://github.com/panbanda/higgs/issues/254)) ([482d2ec](https://github.com/panbanda/higgs/commit/482d2eceab6d616f9e6e79c37d6af06624c42410))


### Bug Fixes

* **deps:** migrate to sha2 0.11 ([#273](https://github.com/panbanda/higgs/issues/273)) ([eb817f9](https://github.com/panbanda/higgs/commit/eb817f97f6f5fd27c6f6d1f1b99c358fe53c6724))

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
