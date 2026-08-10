# Handoff — Unsloth UD mix-bit, Step 4 landed (2026-04-26)

**Branch:** `unsloth-ud-mixbit` in worktree `/Users/peppi/Dev/higgs-unsloth-ud`
**Step 4 commit:** `afdc9211 feat(qwen3_5): BF16-dense attention output projections via bits=0 (Step 4)`
**Working tree:** clean (after committing this RECAP)
**Approved plan:** `~/.claude/plans/generic-fluttering-peach.md`
**Prior handoffs:**
- `.planning/RECAP-2026-04-26-unsloth-ud-mixbit-step3-handoff.md`
- `.planning/RECAP-2026-04-26-unsloth-ud-mixbit-step2-handoff.md`

## What's done

- **Step 1** (`e74fe83`): scaffolded `args.quant_overrides` + `resolve_quant_for`.
- **Step 2** (`94785e7`): `collect_quant_overrides` wired in both loaders.
- **Step 3** (`68d2c12`): canonical paths threaded through every `QLinear` site.
- **Step 4** (`afdc921`): BF16-dense attention outputs via `bits == 0`.
  - `init_unquantized_params()` (weight=`Param[1]`, scales=biases=`Param[0]`)
    next to `init_quantized_params`. Shape `[0]` bypasses
    `placeholder_param_names`' shape-`[1]` filter naturally.
  - `QLinear::new` / `QEmbedding::new` branch on `bits == 0`.
  - `QLinear::forward` / `QEmbedding::as_linear` early-return with
    `x.matmul(weight.transpose())`. `forward_decode_fast` falls through to
    `forward` for `bits != 4` so it picks up the new path automatically.
  - New field `Qwen3NextModelArgs::dense_attention_outputs: bool`
    (`#[serde(default)]`, default `false`). Set to `true` in
    `load_qwen3_5_moe_text_config_args`. Untouched in
    `load_qwen3_next_args_from_value` to preserve original arch behavior.
  - Five wired sites: `Qwen3NextAttention::new` o_proj (1431-1440); and
    in `GatedDeltaNet::new` (2206) the helper `resolve_maybe_dense` covers
    `out_proj`, `in_proj_ba`, `in_proj_a`, `in_proj_b`.
  - Test `test_o_proj_and_out_proj_are_bf16_in_qwen3_5` covers both layer
    types (idx 3 full-attention, idx 0 GDN with `use_separate_gdn_projections=true`)
    and asserts the scales/biases land at shape `[0]` for dense sites.
  - 334/334 release lib tests green. fmt clean. clippy clean (one
    pre-existing `MoE`-doc warning unchanged from clean tree).

## What's left — Commit C — Step 5: integration test + smoke

Two layers of verification per `~/.claude/plans/generic-fluttering-peach.md`
§"Commit C — Step 5":

### Layer 1 — synthetic safetensors integration test

**Goal:** load a tiny mix-bit fixture end-to-end, assert no `[1]` placeholders
remain, bit widths land where expected, and forward emits finite logits.

**Spec:**
- 2 layers: idx 0 (GDN linear), idx 3 (full attention) — `full_attention_interval=4`.
- `hidden_size=64`, `vocab_size=128`, `linear_*` modest; mirror `qwen35_dense_text_config()` (line 13214) but smaller.
- Overrides matching real shape: `lm_head=5-bit`, `embed=4-bit`,
  `in_proj_qkv=4-bit`, `mlp.down_proj=3-bit`, default `(64, 2)`.
- Set `dense_attention_outputs=true` in the fixture config (the qwen3_5
  loader does this automatically).
- Place fixture under `crates/higgs-models/tests/fixtures/qwen3_5_mixbit/`
  OR generate into `tempfile::tempdir()` per test (cleaner — no on-disk
  artifact, follows existing test pattern at line 13276).

**Building blocks already in tree:**
- `qwen3_next.rs:13207` `write_qwen35_config(dir, text_config_json)`.
- `qwen3_next.rs:13264` `write_weight_index(dir, &keys)`.
- `qwen3_next.rs:13214` `qwen35_dense_text_config()` template.
- `Array::load_safetensors` used at `qwen3_next.rs:3787`, `:4145`, `:4217`.

**Missing piece:** a `save_safetensors` helper. `mlx-rs` ships one — see
`/Users/peppi/Dev/mlx-rs/mlx-rs/src/utils/io.rs` and
`mlx-rs/src/ops/io.rs`. Need to confirm whether
`Array::save_safetensors(path, map)` or a free function. Check via:

```bash
rg -n "save_safetensors|fn save\b" /Users/peppi/Dev/mlx-rs/mlx-rs/src/utils/io.rs
```

**Tensor inventory to write** (per layer + global):
- Globals: `language_model.model.embed_tokens.{weight,scales,biases}`,
  `language_model.lm_head.{weight,scales,biases}`,
  `language_model.model.norm.weight`.
- Layer 0 (GDN, idx 0): all of
  `language_model.model.layers.0.{input_layernorm.weight,
   post_attention_layernorm.weight,
   linear_attn.{in_proj_qkvz,in_proj_ba}.{weight[,scales,biases]},
   linear_attn.out_proj.weight (BF16-dense, no scales/biases),
   linear_attn.{conv1d.weight,A_log,dt_bias,norm.weight},
   mlp.{gate,up,down}_proj.{weight,scales,biases}}`.
- Layer 3 (self_attn): substitute `self_attn.{q,k,v}_proj` (quantized) +
  `self_attn.o_proj.weight` (BF16-dense) + `self_attn.{q,k}_norm.weight`
  + `self_attn.rope` is computed, not loaded.
- Use `mlx_rs::ops::quantize(weight, group_size, bits)` to generate
  `(qweight, scales, biases)` tuples for each quantized tensor.

**Assertions:**
1. `load_qwen3_5_model(dir)` returns `Ok(model)`.
2. After loading, every tensor in the parameter tree has a non-`[1]` shape.
3. `attn.o_proj.bits == 0`, `gdn.out_proj.bits == 0`, `gdn.in_proj_ba.bits == 0`.
4. Override-keyed sites match: `lm_head.bits == 5`, `embed.bits == 4`,
   `mlp.down_proj.bits == 3`.
5. One forward pass on a `[1, 4]` token-id input produces `[1, 4, vocab]`
   logits, all finite (`is_finite().all()`).

This deserves its own test module file (e.g.
`crates/higgs-models/src/qwen3_next.rs` adds `mod mixbit_fixture_tests` near
the end, keeping the giant tests-mod intact). One test, comprehensive, per
project rule "minimal new tests, highest coverage".

### Layer 2 — real-checkpoint smoke run

**CLI discovery — left for next session:**

```bash
cd /Users/peppi/Dev/higgs-unsloth-ud
cargo run --release -p higgs -- --help
rg -n "Subcommand|#\[command" crates/higgs/src/main.rs crates/higgs/src/cli.rs 2>/dev/null
```

Expect a `generate` subcommand or similar. If not present, may need to
write a small bin target in `examples/`. The plan suggests:

```bash
cargo run --release -p higgs -- generate \
    --model /Users/peppi/.cache/lm-studio/models/Brooooooklyn/Qwen3.5-27B-UD-Q2_K_XL-mlx \
    --prompt "The capital of France is" --max-tokens 32

cargo run --release -p higgs -- generate \
    --model /Users/peppi/.cache/lm-studio/models/Brooooooklyn/Qwen3.6-35B-A3B-UD-Q3_K_XL-mlx \
    --prompt "The capital of France is" --max-tokens 32
```

Both must produce coherent English continuations. Garbage = a missed
override path or a missed BF16-dense site.

**Debug recipe if smoke fails:**
1. After model construction, walk every `QLinear` field and log
   `(canonical_name, bits)` into a `BTreeMap<String, i32>`. Diff against
   `args.quant_overrides`.
2. Check `ensure_all_model_params_loaded` for any leftover shape-`[1]`
   placeholders (already errors on load — but log the exact names if
   present).
3. Compare logits of the first few decode steps against an MLX-Python
   reference if available (out of scope for this commit).

## Verification before each commit

```bash
cd /Users/peppi/Dev/higgs-unsloth-ud
cargo fmt -p higgs-models -- --check
cargo test --release -p higgs-models --lib -- qwen3_next::tests
cargo clippy -p higgs-models --release --lib --tests
```

Note: a single occasional GPU-flaky failure of
`test_gated_delta_kernel_matches_ops_bfloat16` was observed during Step 4
verification. Re-running clears it. If it persists, investigate Metal
kernel state; do not silence.

## How to resume

```bash
cd /Users/peppi/Dev/higgs-unsloth-ud
git log -3 --oneline    # confirm afdc9211 at HEAD (after this RECAP commit)
cat ~/.claude/plans/generic-fluttering-peach.md
cat .planning/RECAP-2026-04-26-unsloth-ud-mixbit-step4-handoff.md
```

Then start Commit C. Suggested order:
1. Confirm `mlx-rs` `save_safetensors` API. Smallest probe: write a 1-tensor
   safetensors file in a throwaway test, load it back, compare shapes.
2. Build a fixture-writer helper (private to the test module): takes the
   config JSON and an override map, generates random weights at the right
   shapes, runs `mlx_rs::ops::quantize` per (group_size, bits), writes
   `model.safetensors` + `model.safetensors.index.json` + `config.json`.
3. Write the integration test using the helper.
4. Run + fix until green.
5. Commit Step 5a (synthetic fixture).
6. Run smoke against the two real checkpoints. Commit Step 5b only if
   output is coherent. If garbage, debug per recipe above; do NOT commit
   passing tests with broken smoke.

## Things deliberately NOT done

- **MTP** — disabled at load for both target checkpoints.
- **Original `qwen3_next` model_type** — gated off via
  `dense_attention_outputs=false` default.
- **`gate_quantization` removal** — kept as fallback in `resolve_gate_quant`.
- **GitNexus impact analysis** — index stale (additive struct field +
  constructor body changes only).
- **Smoke run** — deferred to Step 5b.
