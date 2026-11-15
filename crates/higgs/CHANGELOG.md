# Changelog

## [1.9.0](https://github.com/panbanda/higgs/compare/higgs-v1.8.2...higgs-v1.9.0) (2026-11-15)


### Features

* **api:** accept regex and guided_* output constraints on chat ([20c8404](https://github.com/panbanda/higgs/commit/20c84043e3cd2607d4b6e94ecddda0f993c57f61))
* **api:** add /v1/tokenize and /v1/detokenize ([6e92c69](https://github.com/panbanda/higgs/commit/6e92c69e4f84f1137e78a2d226804b2903bb91f5))
* **api:** add Responses API types, translation, stream builder and store ([69ef43e](https://github.com/panbanda/higgs/commit/69ef43e1fdb67becaec5b58c5598e843f5461d00))
* **api:** forced tool_choice and parallel_tool_calls on chat, responses and anthropic ([443693e](https://github.com/panbanda/higgs/commit/443693ea74dc12390c67227c4a2344f3e4631c56))
* **api:** map Anthropic tool_use/tool_result history into the prompt ([e953d44](https://github.com/panbanda/higgs/commit/e953d4402c8b14dbbf0ad778daca0ab19f97a133))
* **api:** report cached tokens, reasoning tokens and timings in usage ([0caa206](https://github.com/panbanda/higgs/commit/0caa2063e83f3424dd60ecdfeb4a36cecf9fcc53))
* **api:** serve POST/GET/DELETE /v1/responses ([f93fd04](https://github.com/panbanda/higgs/commit/f93fd0499c0652be732321ab50666ddfdb8e33c0))
* **api:** support tool_choice none/auto and accept seed ([1c18647](https://github.com/panbanda/higgs/commit/1c1864774e63aae75864b481b01a9785965c6219))
* **batch:** reach feature parity with the simple engine ([d2fae4c](https://github.com/panbanda/higgs/commit/d2fae4c0e99f1db6d235993cbf88f88618f8b914))
* **cli:** add higgs bench and higgs selftest ([186ecbb](https://github.com/panbanda/higgs/commit/186ecbb1425e74ef41f4a18504d2c62294296073))
* **cli:** find a running server without a config in shellenv and exec ([653e237](https://github.com/panbanda/higgs/commit/653e237e0155f4d40a44fd6f589e81664569cc50))
* **config:** choose the engine per model with batch = "auto" ([907b473](https://github.com/panbanda/higgs/commit/907b47303113f38cb46778228846d51fd9f7a806))
* **config:** make prefix-cache sizing and hybrid checkpoints configurable ([d16f815](https://github.com/panbanda/higgs/commit/d16f81549847b2b6d155adfad37cba41f26afeae))
* **engine:** batched decode for Phi-3 and Starcoder2 through a generic batched cache ([61da6c8](https://github.com/panbanda/higgs/commit/61da6c8ce4575451afd2e0d3a0ed799cea916dfe))
* **engine:** make the thinking budget configurable ([9431c34](https://github.com/panbanda/higgs/commit/9431c344da030f4615440722b9447234c5b8bf8a))
* **engine:** serve vision, disk prefix store, TurboQuant and speculative decode from the worker ([e727b26](https://github.com/panbanda/higgs/commit/e727b26184bba144a72176eaadfb1a699a60fd76))
* **models:** batched decode and prefill for Gemma 2, 3 and 4 ([f86677b](https://github.com/panbanda/higgs/commit/f86677be30a476c3a5b9448c1f59f67829711cbd))
* **router:** serve unknown model names with the only local model ([589b7f4](https://github.com/panbanda/higgs/commit/589b7f4223be849dffbace3fcfd652cee03bc50d))
* **server:** add a model pool with single-flight loading and LRU eviction ([705259a](https://github.com/panbanda/higgs/commit/705259af2cd674daa7e2c8bd0fb358003a543265))
* **server:** add MCP client and /v1/mcp endpoints ([da16565](https://github.com/panbanda/higgs/commit/da165657e2c3e9dc29f6ef39aa0a0d73a3e51b07))
* **server:** add model lifecycle config, serve flags and doctor checks ([27ae7aa](https://github.com/panbanda/higgs/commit/27ae7aaa74600d85c9137fd98ea11eee5e9a5382))
* **server:** add opt-in server-side tool loop for MCP and web tools ([9657fac](https://github.com/panbanda/higgs/commit/9657fac490e65f3906206420996fc22ef16c159c))
* **server:** add web search and fetch endpoints ([3d75bb2](https://github.com/panbanda/higgs/commit/3d75bb271d0365ab646ff30f0ca3ceba7c8546a7))
* **server:** serve local models from the pool and expose load, unload and status ([a270d1c](https://github.com/panbanda/higgs/commit/a270d1c6ca5564a6675a5edab45ba8f2549f6505))


### Bug Fixes

* **build:** do not shadow the entry name in mlx_sys_output_dirs ([309a381](https://github.com/panbanda/higgs/commit/309a3813b29feed0b9288bacca8abcf00d233b4b))
* **build:** find the profile directory under both Cargo build layouts ([66b5a05](https://github.com/panbanda/higgs/commit/66b5a0553514a8a8f788af45aed8f3ab3aa2bc35))
* **build:** quote OUT_DIR in the build script doc comment ([2e7a7c9](https://github.com/panbanda/higgs/commit/2e7a7c93f081172ccd9fd1750b6d0eb9e2998f3f))
* **ci:** declare rust-version on every crate and satisfy Rust 1.100 clippy ([060d1dd](https://github.com/panbanda/higgs/commit/060d1ddcb601d7f6adca07c19166f94645528334))
* **ci:** satisfy the clippy and rustfmt of the CI toolchain ([c48e2f8](https://github.com/panbanda/higgs/commit/c48e2f85a6fa881b674cb03e1817d7cc8b0e99b7))
* **cli:** confirm the private server's identity before measuring ([1d08f9e](https://github.com/panbanda/higgs/commit/1d08f9eff061edb34a0716f60fee75450491480c))
* **cli:** follow HIGGS_HOST and HIGGS_PORT when probing without a config ([9416ed5](https://github.com/panbanda/higgs/commit/9416ed5fc24ecc0c1e90c53f1e1fc3548a7c086c))
* **cli:** ignore HIGGS_PORT when --port is given ([6982eb1](https://github.com/panbanda/higgs/commit/6982eb1a7bdc9e6f5334a85d9c44ef7ba7f9e767))
* **cli:** isolate the bench and selftest server from the user's config and daemon ([2210a5b](https://github.com/panbanda/higgs/commit/2210a5b11f2269d7091cb30fe61f5ab76d7a69fd))
* **cli:** keep the parent-death watcher off the global stdin lock ([61365c2](https://github.com/panbanda/higgs/commit/61365c272bb263f42e034036a4cea6815e89f027))
* **cli:** map bench --trials to concurrency rounds ([253c6ae](https://github.com/panbanda/higgs/commit/253c6aefbd41770d14663acdae329bd391f40c17))
* **cli:** print the shellenv error with its hint on separate lines ([f83c016](https://github.com/panbanda/higgs/commit/f83c016cf0d67bc7f815e76961b1b0eea0c7ecef))
* **cli:** stop the private server when the bench or selftest parent dies ([c2cfd64](https://github.com/panbanda/higgs/commit/c2cfd646e5480c3368d554dbff9b018895bd9ecb))
* **cli:** try every resolved address when probing the server ([eb9b2ce](https://github.com/panbanda/higgs/commit/eb9b2ce9bb1fa3bf967c931e54f4fd3c0958f2d6))
* **config:** validate a provider's url after layers are merged ([8e3ad65](https://github.com/panbanda/higgs/commit/8e3ad65e8cd6ad8cb3d839b61bf9b395ee35da3e))
* **engine:** build no disk prefix tier when the cache keeps no entries ([5246a93](https://github.com/panbanda/higgs/commit/5246a93b6c69a7b018ed72e921cff6769a04734c))
* **engine:** take logprob bytes from the detokenizer's text and drop the re-decode ([22b6573](https://github.com/panbanda/higgs/commit/22b65734af209a995774f5ec8c51602b24d3488d))
* **generation:** stream each token's logprobs with the text it produced ([351f389](https://github.com/panbanda/higgs/commit/351f3896d49bc4b6f81b00718f1f2e61d679557b))
* **generation:** stream the logprobs of tokens whose text a tracker holds back ([6485850](https://github.com/panbanda/higgs/commit/64858504ef472d8b1181d18d1987d2a49928b42b))
* **higgs:** do not think when the request is constrained ([af1be19](https://github.com/panbanda/higgs/commit/af1be192bb255cc6e0f5b9c30ae7350eb627d35d))
* **higgs:** forward completion text deltas immediately ([9fa36eb](https://github.com/panbanda/higgs/commit/9fa36eb7c2d7c85d7845946907bb06baabacf335))
* **higgs:** report engine failures as errors in every API, keep tool-loop results in stored responses ([c41fb00](https://github.com/panbanda/higgs/commit/c41fb004c9d30f6e995b132f937e877c295f1c67))
* **lint:** avoid shadowed names in the MCP env filter and metallib search ([6ef652f](https://github.com/panbanda/higgs/commit/6ef652f8a9dc4d8132ac7dd7978f4a1ee8ed984f))
* **mcp:** bound sending a request and delivering its cancellation ([03d90f9](https://github.com/panbanda/higgs/commit/03d90f93d054fb6adc33999ffcaff775a95ca726))
* **mcp:** clean up and cancel requests dropped before completion ([236463d](https://github.com/panbanda/higgs/commit/236463d41270d6ebe2e220caffbb0ab9198a3eea))
* **mcp:** do not replay a tool call whose outcome is unknown ([ed1a46f](https://github.com/panbanda/higgs/commit/ed1a46f932c39cfec6435726a2228a208b334bcd))
* **mcp:** do not shadow the variable name in is_inherited ([6304d4c](https://github.com/panbanda/higgs/commit/6304d4c23048a4349d8199d7d4144b954b885617))
* **mcp:** keep multibyte characters split across HTTP chunks ([a626601](https://github.com/panbanda/higgs/commit/a626601e39085523bfd6300463f5c2cbdf3b8cb0))
* **mcp:** reap finished HTTP transport tasks ([1a4869f](https://github.com/panbanda/higgs/commit/1a4869f3c974d0c171988d4a5ad2248a7eec66d5))
* **mcp:** start stdio servers with a cleared environment ([80a11d6](https://github.com/panbanda/higgs/commit/80a11d632b6192d452e25ae1741b3edd22ebdf2b))
* **responses:** bound the response store by bytes ([b42149c](https://github.com/panbanda/higgs/commit/b42149cc3efae895753481429ec0c54f6ea198df))
* **responses:** store each tool-loop round's text once for streamed responses ([d8bcdab](https://github.com/panbanda/higgs/commit/d8bcdabbfa4c3e901e079082052d292fdb488270))
* **responses:** store the tool-augmented transcript for previous_response_id ([1662cb2](https://github.com/panbanda/higgs/commit/1662cb2d91d16f7813ca8b26d9ee275cd5864326))
* **selftest:** rebuild the logprob reference text from token bytes ([4864eb2](https://github.com/panbanda/higgs/commit/4864eb2f814f25f6e3ee3011cbcc8afa6e712a7c))
* **selftest:** skip the prefix cache check when the cache is off ([3db06ac](https://github.com/panbanda/higgs/commit/3db06acd5235ab6c240bd632973475444a0afb64))
* **test:** locate the higgs binary through CARGO_BIN_EXE_higgs ([c5cb398](https://github.com/panbanda/higgs/commit/c5cb398ae12771d8200e09dcc9aba50d8027bfdc))
* **web:** bound the entity search so a page of ampersands decodes in linear time ([e4f743b](https://github.com/panbanda/higgs/commit/e4f743becf8fdfddc9e94dfbeb8f59efdc8176d2))
* **web:** extract fetched pages on a blocking thread ([3a94c0b](https://github.com/panbanda/higgs/commit/3a94c0bd22c62e840a995c8c1a8db2f64bad8ca2))
* **web:** extract heading text without recursing ([2f3ea34](https://github.com/panbanda/higgs/commit/2f3ea3420b0f299b45dc3eca16a0386dcb833306))
* **web:** search for closing tags and trim leading lines in linear time ([0136584](https://github.com/panbanda/higgs/commit/013658456c216f8af66a1296f168c77b3edf82b1))


### Performance Improvements

* faster than oMLX on follow-up turns and concurrency, feature parity, one engine ([5522af9](https://github.com/panbanda/higgs/commit/5522af96fb2bcf75d3c0d3a69f2e841a602751a2))
* **models:** cheaper MoE routing and batched decode for Qwen3 MoE ([93e8fd2](https://github.com/panbanda/higgs/commit/93e8fd25ee89fc7646d8ac98b3f984068c81ae8e))

## [1.8.2](https://github.com/panbanda/higgs/compare/higgs-v1.8.1...higgs-v1.8.2) (2026-11-10)


### Bug Fixes

* **build:** copy the metallib of the linked mlx-sys build ([#325](https://github.com/panbanda/higgs/issues/325)) ([d79ae29](https://github.com/panbanda/higgs/commit/d79ae2931f5860fe29096740adbea50a3af94056))

## [1.8.1](https://github.com/panbanda/higgs/compare/higgs-v1.8.0...higgs-v1.8.1) (2026-11-10)


### Bug Fixes

* **engine:** keep MLX state on the thread that owns it ([#310](https://github.com/panbanda/higgs/issues/310)) ([3a1eb0a](https://github.com/panbanda/higgs/commit/3a1eb0a83d448a619dcfdff5f8139c49ae791aeb))
* satisfy clippy lints added in Rust 1.99 ([#302](https://github.com/panbanda/higgs/issues/302)) ([837f1c2](https://github.com/panbanda/higgs/commit/837f1c2ac055e763bfc8f46eb4a263d0bd1582eb))

## [1.8.0](https://github.com/panbanda/higgs/compare/higgs-v1.7.0...higgs-v1.8.0) (2026-09-02)


### Features

* **cli:** higgs ui opens the desktop app ([09a5b4c](https://github.com/panbanda/higgs/commit/09a5b4ce99360e76c866916e973ab847f5c93404))
* **desktop:** Tauri dashboard app, request tracing, and richer server metrics ([72d8809](https://github.com/panbanda/higgs/commit/72d88094ed70fbd23fba3f83357bd8165c8c5c18))
* **desktop:** Tauri dashboard app, request tracing, and richer server metrics ([487716f](https://github.com/panbanda/higgs/commit/487716f91797d85ae7429d090a79de4ee52abd42))


### Bug Fixes

* **desktop:** address review findings and bundle the CLI in the app ([bb9a5c3](https://github.com/panbanda/higgs/commit/bb9a5c3dd474655e68a545562a9b3f6bfee13ac5))

## [1.7.0](https://github.com/panbanda/higgs/compare/higgs-v1.6.1...higgs-v1.7.0) (2026-08-18)


### Features

* **config:** first-class config surface for the MLA latent KV cache ([#263](https://github.com/panbanda/higgs/issues/263)) ([cdb3c4e](https://github.com/panbanda/higgs/commit/cdb3c4e74677be2471a89ca6c4d826c2b095d662))
* **engine:** chunked prefill yield quantum for the batch serving loop (PR8 / ds4 P9) ([#259](https://github.com/panbanda/higgs/issues/259)) ([247147e](https://github.com/panbanda/higgs/commit/247147eacfa04ad97af6d1b5aca974594e8da736))
* **engine:** disk-backed KV prefix store for restart-resume (PR6 / ds4 P4) ([#257](https://github.com/panbanda/higgs/issues/257)) ([0435d11](https://github.com/panbanda/higgs/commit/0435d112ff8abf3de65ed554a95e3e3a9b69d690))


### Bug Fixes

* **higgs:** validate config writes, and make metrics see failed requests ([#272](https://github.com/panbanda/higgs/issues/272)) ([cfe0ab0](https://github.com/panbanda/higgs/commit/cfe0ab033705a2f819c1eb4d59b78db6c6d2f6f7))

## [1.6.1](https://github.com/panbanda/higgs/compare/higgs-v1.6.0...higgs-v1.6.1) (2026-08-11)


### Bug Fixes

* suffix float literals for new float_literal_f32_fallback lint ([#230](https://github.com/panbanda/higgs/issues/230)) ([83029e4](https://github.com/panbanda/higgs/commit/83029e409cf0bbfbc03d363590edba920de3b1f4))

## [1.6.0](https://github.com/panbanda/higgs/compare/higgs-v1.5.0...higgs-v1.6.0) (2026-06-21)


### Miscellaneous Chores

* **higgs:** Synchronize workspace versions

## [1.5.0](https://github.com/panbanda/higgs/compare/higgs-v1.4.0...higgs-v1.5.0) (2026-06-21)


### Features

* **chat:** streaming tool-call deltas + Qwen-friendly arg normalisation ([#164](https://github.com/panbanda/higgs/issues/164)) ([8f01163](https://github.com/panbanda/higgs/commit/8f01163a0a60950c011048f8bd3c6d906b3e8a01))
* **higgs:** migrate from huggingface-cli to hf ([#180](https://github.com/panbanda/higgs/issues/180)) ([aaecdb4](https://github.com/panbanda/higgs/commit/aaecdb4fa7225dc7fc5375982bc2f6c7945eaf2f))
* **serve:** stream prefill progress (llama.cpp-compatible prompt_progress) ([#184](https://github.com/panbanda/higgs/issues/184)) ([e7b2c8a](https://github.com/panbanda/higgs/commit/e7b2c8a41689c7e8e1f1fc4d657a4fa17bba2a96))

## [1.4.0](https://github.com/panbanda/higgs/compare/higgs-v1.3.0...higgs-v1.4.0) (2026-06-21)


### Features

* harden default security posture ([1b41a5c](https://github.com/panbanda/higgs/commit/1b41a5c93696a7492ef4bf7e941982341edf50ab))


### Bug Fixes

* address CI lint and test failures ([0d8e259](https://github.com/panbanda/higgs/commit/0d8e2594809589fe405a3b8091e669164b2e8e52))
* route first streamed token through incremental detokenizer ([722218d](https://github.com/panbanda/higgs/commit/722218df3cfb10c8671721cf45888ec45e31187c))
* split long first doc paragraphs for clippy ([c195312](https://github.com/panbanda/higgs/commit/c1953125d8399f899236d7917f26b94051fccb4b))
* type alias for build_router signature assertion ([d084592](https://github.com/panbanda/higgs/commit/d0845926a64c0179646855b628ef0da737e31b2f))

## [1.3.0](https://github.com/panbanda/higgs/compare/higgs-v1.2.0...higgs-v1.3.0) (2026-05-31)


### Miscellaneous Chores

* **higgs:** Synchronize workspace versions

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
