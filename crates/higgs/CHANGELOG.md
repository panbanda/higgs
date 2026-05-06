# Changelog

## [1.2.0](https://github.com/panbanda/higgs/compare/higgs-v1.1.1...higgs-v1.2.0) (2026-05-06)


### Bug Fixes

* default Qwen3.6 to non-thinking mode ([#150](https://github.com/panbanda/higgs/issues/150)) ([372020e](https://github.com/panbanda/higgs/commit/372020ec5198b47bfb29babbaf60b4250507a76e))

## [1.1.1](https://github.com/panbanda/higgs/compare/higgs-v1.1.0...higgs-v1.1.1) (2026-04-28)


### Performance Improvements

* **sse:** pre-serialize static chunk fields for streaming routes ([056f4b4](https://github.com/panbanda/higgs/commit/056f4b4e316d27ab5c960b954619996f63653942))
* **sse:** pre-serialize static chunk fields for streaming routes ([d8e02ba](https://github.com/panbanda/higgs/commit/d8e02ba9114f799cf47ebc6074385c14b7cf44cf))

## [1.1.0](https://github.com/panbanda/higgs/compare/higgs-v1.0.1...higgs-v1.1.0) (2026-04-28)


### Features

* **doctor:** validate ServerSection fields ([0250c42](https://github.com/panbanda/higgs/commit/0250c422d022e72a00095cd2def1ed5996dfb762))
* **doctor:** validate ServerSection fields ([554371f](https://github.com/panbanda/higgs/commit/554371f890cf23a7e2fd1141109e51cb9f26c2d5))


### Bug Fixes

* **doctor:** bracket IPv6 host when binding port; add api_key/rate_limit positive-path tests ([7177476](https://github.com/panbanda/higgs/commit/717747662379fe1cf89b01b5db5fdf50246504a3))

## [1.0.1](https://github.com/panbanda/higgs/compare/higgs-v1.0.0...higgs-v1.0.1) (2026-04-27)


### Bug Fixes

* clarify metallib recovery hint ([645dbb0](https://github.com/panbanda/higgs/commit/645dbb0201b3a54e3be754a6165ca4bbd951141b))

## [1.0.0](https://github.com/panbanda/higgs/compare/higgs-v0.1.22...higgs-v1.0.0) (2026-04-24)


### ⚠ BREAKING CHANGES

* higgs start is now config/profile-only, attach is a strict daemon dashboard, shellenv/exec fail fast on invalid or unreachable targets, and exact local model matches now take precedence over regex routes.

### Features

* harden CLI, dashboard, routing, and MLX runtime ([4dfc930](https://github.com/panbanda/higgs/commit/4dfc930365ec1d8eb8143508fe63c41b21001ba1))


### Bug Fixes

* address follow-up review issues ([5c857ce](https://github.com/panbanda/higgs/commit/5c857ce32f5eeab4fe020c9760a2bd5a0e984c05))
* address review follow-ups ([d2b312e](https://github.com/panbanda/higgs/commit/d2b312e0c447b33b65505e8b34adfa74d240da96))
* satisfy clippy duration lint ([20c6aae](https://github.com/panbanda/higgs/commit/20c6aaebdafba342e62e33ed4b904dcc5620dc0a))

## [0.1.22](https://github.com/panbanda/higgs/compare/higgs-v0.1.21...higgs-v0.1.22) (2026-04-24)


### Miscellaneous Chores

* **higgs:** Synchronize workspace versions

## [0.1.21](https://github.com/panbanda/higgs/compare/higgs-v0.1.20...higgs-v0.1.21) (2026-04-24)


### Features

* add qwen3.5/qwen3.6 turboquant stack ([4f165ee](https://github.com/panbanda/higgs/commit/4f165eeca1166ac1d1b62cbaf523b83d344b9a22))
* API routes — thinking mode streaming and tool call validation ([5f7e62c](https://github.com/panbanda/higgs/commit/5f7e62cb8eeb168dfb36d98eea2d39c54cf4e950))
* server config — TurboQuant, thinking budget, chunked prefill settings ([5b63e4c](https://github.com/panbanda/higgs/commit/5b63e4c9bedee401adf6f23b319316f55727c7f1))
* tune qwen3.6 thinking defaults ([6969834](https://github.com/panbanda/higgs/commit/696983431b7c404bf5efea4978398251603ac6ec))


### Bug Fixes

* address coderabbit issues ([6f10bc6](https://github.com/panbanda/higgs/commit/6f10bc6d733c192d30c5ab3aed76e4a2e2a69818))
* cargo fmt + restore higgs crate build on PR [#74](https://github.com/panbanda/higgs/issues/74) ([050af95](https://github.com/panbanda/higgs/commit/050af9590968c129dc8f0a5003d4af47f6591da5))
* clear lint and review blockers ([9c8b878](https://github.com/panbanda/higgs/commit/9c8b878bb654ba8865f3455c91ba457b83d1161b))
* **config:** timeout validation and figment model merge ([441884d](https://github.com/panbanda/higgs/commit/441884d6450e2ce486ff3e131b37033296ea3cc9))
* make pre-push pass on dust stack ([2695165](https://github.com/panbanda/higgs/commit/26951659aa8ff56d0362a6916827dbe9cf6fae88))
* remove incorrect default_value on Option&lt;u8&gt; CLI arg — PR [#75](https://github.com/panbanda/higgs/issues/75) ([bf87bbd](https://github.com/panbanda/higgs/commit/bf87bbd4b9f5b2a88e386f2bb7b03beb0f07534d))
* restore higgs crate build on PR [#75](https://github.com/panbanda/higgs/issues/75) — chat.rs usage stubs ([f92d139](https://github.com/panbanda/higgs/commit/f92d139e4d8b1319169d168c73c92db52bf01a4d))
* **routes:** close think tag on length-stopped reasoning in OpenAI path ([5c2ea56](https://github.com/panbanda/higgs/commit/5c2ea56566fafbf8198f2bca34d3454ea70c2ca2))
* stabilize dust stack CI ([8e629e3](https://github.com/panbanda/higgs/commit/8e629e35e53bb65b9a6dd718be5d2d50a79e297c))
* support clippy across toolchains ([abc316d](https://github.com/panbanda/higgs/commit/abc316d1c573db20a1921530782752f41ee745f4))
* support mlx qwen3.6 smoke ([7046849](https://github.com/panbanda/higgs/commit/704684970b3cf43ad9bc3644e74422e378dacfd4))

## [0.1.20](https://github.com/panbanda/higgs/compare/higgs-v0.1.19...higgs-v0.1.20) (2026-02-28)


### Bug Fixes

* restore pattern matching fallback in force routing mode ([c1bb2c3](https://github.com/panbanda/higgs/commit/c1bb2c39c2fc8a377a7fab2a0c792f3ec6794899))
* restore pattern matching fallback in force routing mode ([217249f](https://github.com/panbanda/higgs/commit/217249f2612cde2bf92ac81d10228ee6c6b9b4f8))

## [0.1.19](https://github.com/panbanda/higgs/compare/higgs-v0.1.18...higgs-v0.1.19) (2026-02-28)


### Bug Fixes

* strip thinking tags from Anthropic route, fix force routing, add proxy usage tracking ([89f8db2](https://github.com/panbanda/higgs/commit/89f8db2c889971b8c2ae21dec8d63ea70308491e))
* strip thinking tags, fix force routing, add proxy usage metrics ([0ffbabb](https://github.com/panbanda/higgs/commit/0ffbabb1cb0abbca70d5c01bea007028b21d6ff4))

## [0.1.18](https://github.com/panbanda/higgs/compare/higgs-v0.1.17...higgs-v0.1.18) (2026-02-28)


### Bug Fixes

* address review comments ([36c6484](https://github.com/panbanda/higgs/commit/36c6484acd486af7744507ebb974e5a62ecf1f93))
* backtick TcpListener in doc comment for clippy doc_markdown ([94acea7](https://github.com/panbanda/higgs/commit/94acea7730105895be598bb04d9580de4ea3195a))
* correct comment about signal handler behavior ([81cbd71](https://github.com/panbanda/higgs/commit/81cbd71e38957287cdc81c1e098bc4c8d81a7801))
* improve Anthropic API compatibility ([16ee50d](https://github.com/panbanda/higgs/commit/16ee50d5b4d1e7e24bdbd3e7e97a2f8943b15a23))
* improve Anthropic API compatibility ([ff7e96c](https://github.com/panbanda/higgs/commit/ff7e96c94b308cfe5d83ea8d162e96e2e50d7e0e))
* log warning instead of silently discarding signal handler error ([6946974](https://github.com/panbanda/higgs/commit/6946974df001540b071ebc97bf7dc3fecfa68371))
* use 128+signal convention for signal-killed child exit codes ([89b7ae0](https://github.com/panbanda/higgs/commit/89b7ae0cb98b3bf372d49b83f65ef8e23d5d722c))

## [0.1.17](https://github.com/panbanda/higgs/compare/higgs-v0.1.16...higgs-v0.1.17) (2026-02-27)


### Bug Fixes

* normalize auto_router.model to a name like routes do ([1e741c5](https://github.com/panbanda/higgs/commit/1e741c5c83133923565b59260e9bf64d41cd3e87))
* normalize auto_router.model to a name like routes do ([3b52db4](https://github.com/panbanda/higgs/commit/3b52db488c4099e44b22712640f82f3f57f8efaa))

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
