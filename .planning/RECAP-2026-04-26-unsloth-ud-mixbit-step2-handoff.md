# Handoff — Unsloth UD mix-bit, Step 2 landed (2026-04-26)

**Branch:** `unsloth-ud-mixbit` in worktree `/Users/peppi/Dev/higgs-unsloth-ud`
**Step 2 commit:** `94785e7b feat(qwen3_5): lift Unsloth UD per-tensor mix-bit overrides into args`
**Working tree:** clean (after committing this RECAP)
**Approved plan:** `~/.claude/plans/generic-fluttering-peach.md` — read this first.

## What's done

- **Step 1** (`e74fe83`): scaffolded `args.quant_overrides: BTreeMap<String, QuantizationConfig>` + `resolve_quant_for(args, path)` helper. 1 red TDD test, 1 green.
- **Step 2** (`94785e7`): `collect_quant_overrides(config)` walks `quantization` (or sibling `quantization_config`), promotes every nested `(group_size, bits)` object, drops `mode`, skips scalars. Wired from BOTH `load_qwen3_next_args_from_value` AND `load_qwen3_5_moe_text_config_args`. 331/331 release-mode lib tests green. The override map IS now populated correctly for both target checkpoints — verified by reading `args.quant_overrides` in tests.

## What's left

The Step-1 handoff sketched Steps 3-5; the approved plan extends that with MoE additions and a concrete BF16-dense strategy. Three commits, in order:

### Commit A — Step 3 (path-aware QLinear sites)

The override map exists but is unread. Every `QLinear::new(ql, qb)` call site uses the global default. Refactor to call `resolve_quant_for(args, &canonical_path)`.

**Critical inventory** (verified by Explore agent — file:line refs valid as of `94785e7`):

| Constructor | File:line | Has `args`? | Has `layer_idx`? | QLinear sites | Canonical path template |
|---|---|---|---|---|---|
| `Qwen3NextAttention::new` | 1381 | yes | NO | q/k/v/o_proj @ 1395-1398 | `language_model.model.layers.{i}.self_attn.{q,k,v,o}_proj` |
| `new_mlp_projections` | 1558 | NO | NO | gate/down/up @ 1563-1565 | varies — see callers |
| `Qwen3NextMLP::new` | 1570 | NO | NO | delegates | `…/mlp` (dense) or `…/mlp.shared_expert` (MoE) |
| `SwitchMlpWeights::new` | 1714 | NO | NO | delegates | `…/mlp.switch_mlp.{gate,down,up}_proj` |
| `SparseMoeBlock::new` | 1903 | yes | NO | gate @ 1921, shared_expert_gate @ 1924 + delegates | `…/mlp.gate`, `…/mlp.shared_expert_gate`, `…/mlp.shared_expert.*`, `…/mlp.switch_mlp.*` |
| `GatedDeltaNet::new` | 2133 | yes | NO | in_proj_qkvz @ 2145, in_proj_ba @ 2146, conditional in_proj_{qkv,z,a,b} @ 2148/2153/2158/2163, out_proj @ 2175 | `…/linear_attn.{in_proj_qkvz, in_proj_ba, in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, out_proj}` |
| `FfnBlock::new_dense` | 2597 | NO | NO | gate/up/down @ 2603-2605 | `…/mlp.{gate,up,down}_proj` |
| `FfnBlock::new_moe` | 2580 | yes | NO | delegates to SparseMoeBlock | — |
| `Qwen3NextInner::new` | 2889 | yes | top-level | embed_tokens (QEmbedding) @ 2895 | `language_model.model.embed_tokens` |
| `Qwen3NextCausalLM::new` | 3010 | yes (owns args) | top-level | lm_head @ 3028 | `language_model.lm_head` |
| `DecoderLayer::new` | 2794 | yes | YES | delegates @ 2798/2805/2809-2811 | — |

**`QLinear` exposes `pub(crate) bits: i32`** (line 260) — readable in tests for assertions.

**Existing `gate_quantization` mechanism** at `SparseMoeBlock::new` 1916-1919 reads `args.gate_quantization` as override for both `mlp.gate` and `mlp.shared_expert_gate`. Plan keeps this as a fallback after `quant_overrides[path]` lookup. Resolution order: `quant_overrides[path]` → `gate_quantization` → `args.quantization` → `(64, 4)`.

**Refactor strategy** (per approved plan):
- Drop `ql, qb` from `DecoderLayer::new` and propagate `args, layer_idx` down.
- For path-builder sites without `layer_idx` (like `Qwen3NextMLP::new`), pass `prefix: &str` (e.g. `"language_model.model.layers.3.mlp"` or `".../mlp.shared_expert"`).
- `Qwen3NextCausalLM::new` constructs `lm_head` with hardcoded `"language_model.lm_head"` path.
- `Qwen3NextInner::new` uses hardcoded `"language_model.model.embed_tokens"`.

**Skip for this commit:** `MtpHead::new` (1664). MTP is disabled by `maybe_disable_mtp_without_checkpoint_weights` for both target checkpoints.

**Test 3** to add (handoff Step 3):
```rust
#[test]
fn test_decoder_layer_routes_overrides_to_qlinears() {
    // overrides: mlp.down_proj=3-bit, self_attn.q_proj=4-bit
    // build DecoderLayer for a full-attn layer index (e.g. 3)
    // assert layer.attn.q_proj.bits == 4, layer.ffn.gate_proj.bits == 2 (default), etc.
}
```

Add a parallel MoE test asserting `shared_expert.down_proj` and `switch_mlp.down_proj` get their overrides.

### Commit B — Step 4 (BF16-dense via bits=0)

**Verified ground truth** (read directly from safetensors of both checkpoints):

| Tensor | 27B Q2_K_XL | 35B-A3B Q3_K_XL |
|---|---|---|
| `linear_attn.in_proj_a` | `.weight` only | `.weight` only |
| `linear_attn.in_proj_b` | `.weight` only | `.weight` only |
| `linear_attn.out_proj` | `.weight` only | `.weight` only |
| `linear_attn.in_proj_qkv` | weight+scales+biases | weight+scales+biases |
| `linear_attn.in_proj_z` | weight+scales+biases | weight+scales+biases |
| `self_attn.o_proj` | `.weight` only | `.weight` only |
| `self_attn.{q,k,v}_proj` | weight+scales+biases | weight+scales+biases |
| `mlp.{gate,down,up}_proj` (27B dense) | weight+scales+biases | — |
| `mlp.{shared_expert,switch_mlp}.*` (35B MoE) | — | weight+scales+biases |
| `mlp.gate`, `mlp.shared_expert_gate` (35B) | — | weight+scales+biases |
| `embed_tokens`, `lm_head` | weight+scales+biases | weight+scales+biases |

**The four BF16-dense classes are the same in both checkpoints**: `linear_attn.{in_proj_a, in_proj_b, out_proj}` and `self_attn.o_proj`. In fused-GDN mode (default), `linear_attn.in_proj_b` + `in_proj_a` are concatenated into `in_proj_ba` — also BF16-dense.

**Implementation per plan:**

1. **`QLinear::forward` (line 275)** — early-return on `bits == 0`:
   ```rust
   if self.bits == 0 {
       // BF16-dense: x @ weight.T (mlx weight stored as [out, in])
       return ops::matmul(x, &self.weight.transpose_axes(&[-2, -1])?);
   }
   ```
   Same for `forward_decode_fast` (290) and `QEmbedding::as_linear` (340) — the latter unlikely to hit bits=0 but keep consistent.

2. **`init_quantized_params` (line 230)** — keep as-is (returns `[1]`-shape Param). Add new helper `init_unquantized_params()` returning `(weight=Param[1], scales=Param[0], biases=Param[0])`. The shape-`[0]` arrays bypass `placeholder_param_names` (which filters on shape == `[1]`).

3. **`QLinear::new` (line 264)** — branch on `bits`:
   ```rust
   pub(crate) fn new(group_size: i32, bits: i32) -> Result<Self, Exception> {
       let (weight, scales, biases) = if bits == 0 {
           init_unquantized_params()?
       } else {
           init_quantized_params()?
       };
       ...
   }
   ```

4. **Add `args.dense_attention_outputs: bool`** (default false). Set true in `load_qwen3_5_moe_text_config_args`. Leave `load_qwen3_next_args_from_value` untouched (preserves original arch behavior).

5. **At call sites for `o_proj`, `out_proj`, `in_proj_a`, `in_proj_b`, and `in_proj_ba`**: when `args.dense_attention_outputs`, use `QLinear::new(group_size, 0)`. The `group_size` value doesn't matter for bits=0 (just pass `args.quantization.group_size` or `64`).

**Test 4**: build a `DecoderLayer` with `dense_attention_outputs=true` from a full-attn layer index. Assert `layer.attn.o_proj.bits == 0` and `layer.gdn.out_proj.bits == 0`.

### Commit C — Step 5 (integration test + smoke)

**Synthetic safetensors fixture** (handoff Step 5):
- 2 layers (1 GDN at idx 0, 1 self_attn at idx 3), `hidden_size=64`, `vocab_size=128`, `full_attention_interval=4`, `linear_*=4`.
- Build via `mlx_rs::ops::quantize` inside an `#[ignore]`-gated test that writes once.
- Place under `crates/higgs-models/tests/fixtures/qwen3_5_mixbit/`.
- Assertions: model loads, no `[1]` placeholders remain, bit widths match, forward produces finite logits.

**Smoke run on real checkpoints**:
- The CLI is `cargo run --release -p higgs --` with `clap` subcommands defined in `crates/higgs/src/main.rs`. Discovery left to next session — `rg "Subcommand|#\[command" crates/higgs/src/` and `cargo run --release -p higgs -- --help`.
- Both checkpoints must produce coherent English continuations.
- 27B-Q2_K_XL: `/Users/peppi/.cache/lm-studio/models/Brooooooklyn/Qwen3.5-27B-UD-Q2_K_XL-mlx`
- 35B-A3B-Q3_K_XL: `/Users/peppi/.cache/lm-studio/models/Brooooooklyn/Qwen3.6-35B-A3B-UD-Q3_K_XL-mlx`

If output is garbage, debug by listing every `QLinear` field's `(name, bits)` after construction and diffing against the override map.

## Verification before each commit

```bash
cd /Users/peppi/Dev/higgs-unsloth-ud
cargo fmt -p higgs-models -- --check
cargo test --release -p higgs-models --lib -- qwen3_next::tests
```

Note: clippy was hitting an ENOSPC issue with this session's tool-output buffer. Disk was 100% full (only 120Mi free); `cargo clean` in this worktree freed 4.2 GiB. **Run `cargo clean` before resuming if disk pressure returns.**

## Things deliberately NOT done

- **`MtpHead::new` refactor** — MTP disabled at load for both target checkpoints. Out of scope.
- **Original `qwen3_next` model_type** — gated off via `dense_attention_outputs=false` default.
- **`gate_quantization` removal** — kept as fallback for backward compat.
- **GitNexus impact analysis** — index stale at session start.
- **`.DS_Store` and empty `~/.codex/worktrees/0eb0/`** — still pending user authorization to delete (carryover from Step 1 handoff).

## How to resume

```bash
cd /Users/peppi/Dev/higgs-unsloth-ud
git log -3 --oneline    # confirm 94785e7 at HEAD (after this RECAP commit)
cat ~/.claude/plans/generic-fluttering-peach.md   # read approved plan
```

Then start Commit A: refactor `Qwen3NextAttention::new` first (smallest, isolated), wire `&args` + derive paths from `layer_idx` (which must come from `DecoderLayer::new` callsite at line 2805). That cascades naturally into the other constructors.
