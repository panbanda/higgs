# Handoff — Unsloth UD mix-bit, Step 3 landed (2026-04-26)

**Branch:** `unsloth-ud-mixbit` in worktree `/Users/peppi/Dev/higgs-unsloth-ud`
**Step 3 commit:** `68d2c128 feat(qwen3_5): thread canonical paths through every QLinear site (Step 3)`
**Working tree:** clean (after committing this RECAP)
**Approved plan:** `~/.claude/plans/generic-fluttering-peach.md`
**Prior handoff:** `.planning/RECAP-2026-04-26-unsloth-ud-mixbit-step2-handoff.md`

## What's done

- **Step 1** (`e74fe83`): scaffolded `args.quant_overrides` + `resolve_quant_for`.
- **Step 2** (`94785e7`): `collect_quant_overrides` wired in both loaders.
- **Step 3** (`68d2c128`): every `QLinear::new` / `QEmbedding::new` now calls
  `resolve_quant_for(args, &canonical_path)`. Constructor signatures updated
  to thread paths through. 333/333 release lib tests green. fmt clean.

### Step 3 refactor map (file:line in qwen3_next.rs after `68d2c128`)

| Constructor | New signature |
|---|---|
| `Qwen3NextAttention::new` (1381) | `(args, attn_prefix: &str)` |
| `new_mlp_projections` (1563) | `(args, mlp_prefix: &str)` |
| `new_mlp_projections_from_quant` (NEW) | `(ql, qb)` — for qwen3_moe / deepseek_v2 |
| `Qwen3NextMLP::new` (1578) | `(args, mlp_prefix: &str)` |
| `SwitchMlpWeights::new` (1733) | `(args, prefix: &str)` |
| `SwitchMlpWeights::from_quant` (NEW) | `(ql, qb)` — for qwen3_moe / deepseek_v2 |
| `SparseMoeBlock::new` (~1920) | `(args, mlp_prefix: &str)` — uses `resolve_gate_quant` for gate + shared_expert_gate |
| `GatedDeltaNet::new` (~2150) | `(args, gdn_prefix: &str)` |
| `FfnBlock::new_moe` / `new_dense` (~2580/2600) | `(args, mlp_prefix: &str)` |
| `DecoderLayer::new` (~2810) | `(args, layer_idx)` — builds `language_model.model.layers.{i}` |
| `Qwen3NextInner::new` (~2900) | `(args)` — uses `language_model.model.embed_tokens` |
| `Qwen3NextCausalLM::new` (~3025) | `(args)` — uses `language_model.lm_head` |
| `MtpHead::new` (1675) | `(args)` — uses `language_model.mtp.layers.{i}` |

### `resolve_gate_quant` helper (NEW, near `resolve_quant_for`)
Resolution order: `quant_overrides[path]` → `gate_quantization` → global → `(64, 4)`.
Preserves backward compat for checkpoints carrying only `gate_quantization`.

### Backward-compat shims for sister architectures
`qwen3_moe.rs` and `deepseek_v2.rs` carry different args types so they can't
participate in path-aware overrides. They now call `SwitchMlpWeights::from_quant(ql, qb)`
and `new_mlp_projections_from_quant(ql, qb)` instead.

## What's left

### Commit B — Step 4: BF16-dense tensors via `bits == 0`

**Verified ground truth (read from safetensors, both checkpoints):**
- `linear_attn.in_proj_a.weight` only (no scales/biases)
- `linear_attn.in_proj_b.weight` only
- `linear_attn.out_proj.weight` only
- `self_attn.o_proj.weight` only
- Fused-GDN mode: `linear_attn.in_proj_ba` is BF16-dense too (concat of `b` + `a`)

**Implementation per plan (`~/.claude/plans/generic-fluttering-peach.md` §"Commit B"):**

1. **`QLinear::forward` (line 275) / `forward_decode_fast` (290) / `QEmbedding::as_linear` (340)** — early-return on `bits == 0`:
   ```rust
   if self.bits == 0 {
       // BF16-dense: x @ weight.T (mlx weight stored as [out, in])
       return ops::matmul(x, &self.weight.transpose_axes(&[-2, -1])?);
   }
   ```

2. **`init_quantized_params` (line ~230)** — keep as-is. Add `init_unquantized_params()` returning `(weight=Param[1], scales=Param[0], biases=Param[0])`. Shape `[0]` bypasses `placeholder_param_names` (filters on shape == `[1]`).

3. **`QLinear::new` (line 264)** — branch on `bits`:
   ```rust
   pub(crate) fn new(group_size: i32, bits: i32) -> Result<Self, Exception> {
       let (weight, scales, biases) = if bits == 0 {
           init_unquantized_params()?
       } else {
           init_quantized_params()?
       };
       Ok(Self { weight, scales, biases, group_size, bits })
   }
   ```

4. **Add `args.dense_attention_outputs: bool`** (default false). Set to true in `load_qwen3_5_moe_text_config_args` (~3738). Leave `load_qwen3_next_args_from_value` (~3591) at false to preserve original arch behavior.

5. **At call sites for `o_proj`, `out_proj`, `in_proj_a`, `in_proj_b`, `in_proj_ba`**:
   when `args.dense_attention_outputs`, use `QLinear::new(group_size, 0)`. Group size doesn't matter for bits=0 — pass `args.quantization.group_size` or 64.

   Specific sites in qwen3_next.rs (post-Step 3):
   - `Qwen3NextAttention::new` o_proj — currently uses `resolve_quant_for(...self_attn.o_proj)`; gate on `args.dense_attention_outputs` and force `(g, 0)`.
   - `GatedDeltaNet::new` out_proj, in_proj_ba (always present), in_proj_a/b (when `use_separate_projections`) — same gate.

**Test 4** (handoff Step 4):
```rust
#[test]
fn test_o_proj_and_out_proj_are_bf16_in_qwen3_5() {
    let mut args = valid_causal_lm_args();
    args.dense_attention_outputs = true;
    let layer = DecoderLayer::new(&args, 3).unwrap();
    assert_eq!(layer.self_attn.as_ref().unwrap().o_proj.bits, 0);
    // For a linear layer (idx 0):
    let gdn_layer = DecoderLayer::new(&args, 0).unwrap();
    assert_eq!(gdn_layer.linear_attn.as_ref().unwrap().out_proj.bits, 0);
}
```

### Commit C — Step 5: integration test fixture + smoke run

See `.planning/RECAP-2026-04-26-unsloth-ud-mixbit-step2-handoff.md` §"Commit C — Step 5"
for fixture spec + smoke command details. Unchanged from prior recap.

## Verification before each commit

```bash
cd /Users/peppi/Dev/higgs-unsloth-ud
cargo fmt -p higgs-models -- --check
cargo test --release -p higgs-models --lib -- qwen3_next::tests
```

## How to resume

```bash
cd /Users/peppi/Dev/higgs-unsloth-ud
git log -3 --oneline    # confirm 68d2c128 at HEAD (after this RECAP commit)
cat ~/.claude/plans/generic-fluttering-peach.md
cat .planning/RECAP-2026-04-26-unsloth-ud-mixbit-step3-handoff.md
```

Then start Commit B. Suggested order:
1. Touch `QLinear::new`, `forward`, `forward_decode_fast`, `QEmbedding::as_linear` to handle `bits == 0`.
2. Add `init_unquantized_params()` next to `init_quantized_params`.
3. Add `args.dense_attention_outputs: bool` field; default false; serde-default tag.
4. Set true in `load_qwen3_5_moe_text_config_args`. Verify `load_qwen3_next_args_from_value` stays false.
5. Wire 5 call sites in `Qwen3NextAttention::new` (o_proj) and `GatedDeltaNet::new` (out_proj, in_proj_ba, in_proj_a, in_proj_b).
6. Add Test 4 above. Run release tests + fmt.
7. Commit. Then proceed to Commit C.

## Things deliberately NOT done

- **MtpHead path-aware** — refactored to use `language_model.mtp.layers.{i}` paths but MTP is still disabled at load for both target checkpoints. No bit-width verification needed.
- **gate_quantization removal** — kept as fallback in `resolve_gate_quant`.
- **GitNexus index** — stale; skipped (additive constructor changes only).
