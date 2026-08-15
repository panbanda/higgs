# MoE calibration and asymmetric quantization recipes (ds4 P2)

This directory holds the calibration half of ds4 P2. The other half --
*loading* a checkpoint whose `config.json` carries a per-tensor
`"quantization"` map (mixed group sizes/bit widths, including dense
`false` overrides) -- already lives in the higgs Rust codebase; see
`crates/higgs-models/src/quant_config.rs` and the
`deepseek_expert_projection_quantization` helper in
`crates/higgs-models/src/deepseek_v2.rs`.

**Why Python, not Rust:** the actual weight quantization (computing scales,
packing bits, writing safetensors) is done by `mlx_lm.convert`, which is
part of the mlx_lm ecosystem and has no Rust equivalent higgs could
reasonably reimplement. higgs's job is to *load* mixed-quant checkpoints
fast; producing them is squarely an `mlx_lm` conversion-time concern. This
directory is deliberately the home for that tooling -- it depends on
`mlx`/`mlx_lm` (Python), not the Rust crates, and is not built or tested by
`cargo`.

## ds4 rationale

DeepSeek-V2-style MoE models route each token to a small subset of experts
(e.g. top-6 of 64). Two consequences matter for quantization:

1. **Routed experts individually see a small, noisy slice of tokens.**
   Aggressive quantization (3-bit) tends to hurt them less than it hurts
   dense, always-active tensors (attention projections, embeddings,
   lm_head), which see every token and accumulate error every layer.
2. **Routing is not uniform.** Within a layer, some experts are picked far
   more often than others -- these "hot" experts carry disproportionate
   weight in the model's output and are worth protecting with a higher bit
   width, even while the long tail of rarely-used experts stays at the
   aggressive default.

This is the same idea as GGUF's importance-matrix (imatrix) calibration,
applied at expert granularity instead of per-layer: measure salience from
real forward passes, then spend the bit budget where it matters.

## Workflow

```
1. collect_imatrix.py    -- run calibration texts through the model,
                             record per-expert routing frequency/weight
                             and per-layer input activation magnitude.
2. make_recipe.py         -- turn those statistics into a recipe.json:
                             which tensors get which (group_size, bits).
3. convert_with_recipe.py -- feed the recipe into mlx_lm.convert via a
                             quant_predicate, producing the mixed-quant
                             checkpoint.
4. Load the result in higgs -- the per-tensor loader (PR #260) reads the
   resulting config.json "quantization" map directly; no higgs changes
   needed.
```

### 1. Collect routing/activation statistics

```
python3 collect_imatrix.py \
  --model-dir /path/to/DeepSeek-V2-Lite-bf16 \
  --texts calibration_texts.txt \
  --max-tokens-per-text 512 \
  --out imatrix.json
```

`--texts` accepts either a `.txt` file with one prompt per line (multi-line
prompts encode their newlines as a literal `\n` inside that one line -- see
`calibration_texts.txt` for examples with embedded code) or a `.json` file
containing a list of strings.

This runs prefill-only forward passes (no sampling/generation needed --
routing decisions and activation magnitudes are visible from a single
forward pass over the prompt). It works against either a bf16 source
checkpoint or an already-quantized one (routing statistics for a given
architecture are close enough either way for calibration purposes, and
being able to calibrate against a model you've already shipped is
convenient); we validated it against the cached
`mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx` checkpoint.

Output shape:

```json
{
  "model": "...",
  "n_tokens": 3072,
  "config": { ...the source model's config.json... },
  "layers": [
    {
      "layer": 1,
      "expert_topk_freq": [0.14, 0.02, ...],
      "expert_mean_weight": [0.31, 0.18, ...],
      "input_sq_mean": 0.183
    },
    ...
  ]
}
```

`expert_topk_freq[e]` is the fraction of routed tokens for which expert
`e` was in the top-k; it sums to `top_k` across each layer's experts (each
routed token contributes to exactly `top_k` experts). `expert_mean_weight`
is the mean router weight *conditional on* the expert being selected.
`input_sq_mean` is `mean(x^2)` over both tokens and hidden dims at the MoE
block's input -- a coarse proxy for how much numerical range that layer's
activations need, borrowed from the same idea GGUF's imatrix uses to weight
per-channel quantization error.

### 2. Generate a recipe

```
python3 make_recipe.py \
  --imatrix imatrix.json \
  --target-avg-bits 3.75 \
  --expert-bits-low 3 --expert-bits-high 4 \
  --other-bits 6 \
  --group-size 64 \
  --out recipe.json
```

Routed expert projections default to `--expert-bits-low`, except the
most-frequently-routed experts in each layer (by `expert_topk_freq`),
which get `--expert-bits-high`. Everything else -- attention, shared
experts, embeddings, lm_head -- gets `--other-bits` via the recipe's
`"default"` bucket (router gate weights are never quantized by mlx_lm in
the first place: `MoEGate` stores its weight as a plain `mx.array`, not an
`nn.Linear`, so it has no `to_quantized` method and `nn.quantize` skips it
automatically).

The fraction of "high" experts per layer is solved so the
**parameter-weighted** average bits across the whole model is
`<= target-avg-bits`. Per-tensor parameter counts are estimated from
`config.json` dims rather than read from actual safetensors (acceptable
per the ds4 P2 spec); see the docstrings on `expert_param_count`,
`attention_param_count`, `dense_mlp_param_count`, and
`shared_expert_param_count` in `make_recipe.py` for the exact formulas
(MLA attention: `q_proj` or `q_a_proj`/`q_b_proj` depending on
`q_lora_rank`, plus `kv_a_proj_with_mqa`, `kv_b_proj`, `o_proj`; MoE:
`3 * hidden_size * moe_intermediate_size` per expert for
`gate_proj + up_proj + down_proj`).

Output (`recipe.json`):

```json
{
  "default": {"group_size": 64, "bits": 6},
  "rules": [
    {"path": "model.layers.1.mlp.experts.0.gate_proj", "group_size": 64, "bits": 4},
    {"path": "model.layers.1.mlp.experts.0.up_proj", "group_size": 64, "bits": 4},
    {"path": "model.layers.1.mlp.experts.0.down_proj", "group_size": 64, "bits": 4},
    ...
  ]
}
```

Tensor paths use higgs's own per-expert convention
(`model.layers.N.mlp.experts.M.{gate,up,down}_proj`) -- the same one
`crates/higgs-models/src/deepseek_v2.rs` constructs when resolving
per-tensor overrides, and the same one exercised by
`crates/higgs-models/tests/fixtures/mixed_quant_config.json`.

**Known granularity limitation.** mlx_lm's DeepSeek-V2 implementation
stores all routed experts for one layer/projection as a single *fused*
`switch_mlp.{gate,up,down}_proj` tensor (see `SwitchGLU` in
`mlx_lm.models.deepseek_v2`), and `mlx.nn.quantize`'s `class_predicate` is
invoked once per leaf module. As of the installed mlx_lm (checked against
mlx 0.32 and the cached
`mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx` checkpoint, whose
`model.safetensors.index.json` only has `switch_mlp.*` keys, never
per-expert ones), there is **no supported way to apply two different bit
widths within one fused tensor at conversion time** -- `quant_predicate`
sees one path per layer/projection covering all 64 experts, not one path
per expert. `recipe.json` is still generated at true per-expert
granularity because that's the more useful long-term artifact (it matches
higgs's own loader convention and documents the intended salience-guided
split precisely), but `convert_with_recipe.py` has to *collapse* per-expert
rules into one per-layer decision before handing them to
`mlx_lm.convert` -- see its docstring. If a future mlx_lm version (or a
custom unfused conversion path) exposes per-expert leaf modules, the
collapse step becomes unnecessary and `recipe.json` can be applied as-is.

### 3. Convert

```
python3 convert_with_recipe.py \
  --hf-path /path/to/DeepSeek-V2-Lite-bf16 \
  --mlx-path /path/to/DeepSeek-V2-Lite-mixed \
  --recipe recipe.json
```

Calls `mlx_lm.convert(..., quantize=True, q_group_size=<default group_size>,
q_bits=<default bits>, quant_predicate=<fn>)`. The predicate returns the
recipe's exact `{group_size, bits}` dict for tensors it has a rule for
(non-MoE tensors match directly; fused `switch_mlp.*` tensors get the
majority-vote collapse described above, ties broken toward the higher bit
width), and falls back to `True` (the scalar defaults) for everything else.
Prints a summary of tensor counts per bits bucket and an (unweighted)
average bits estimate afterward.

Pass `--dry-run` to see the predicate's decisions (and the collapse count)
without downloading or converting anything -- note this only reports
useful counts against an **unquantized** source checkpoint, since an
already-quantized model's Linear layers no longer expose `to_quantized()`.

### 4. Load in higgs

The output directory from step 3 is a normal mlx model directory with a
`config.json` containing a per-tensor `"quantization"` map. No higgs
changes are needed -- point a `[[models]]` entry at it and the per-tensor
loader (PR #260) does the rest. `crates/higgs/src/doctor.rs` already
validates `quantization` map shapes at startup.

## Calibration text set

`calibration_texts.txt` has 24 diverse, hand-written prompts (100-300
words each, no copyrighted material): English prose, Python/JavaScript/Go
code, math proofs and derivations, Chinese/Spanish/German/Japanese/Russian
prose, and tool-call-shaped JSON (function-call requests/responses,
webhook payloads, a higgs `/v1/chat/completions` transcript). One prompt
per line; multi-line entries (code) encode embedded newlines as a literal
`\n` within that single line, which `collect_imatrix.py` unescapes.

## Constraints and dependencies

Python 3 stdlib + `mlx`/`mlx_lm` only. No `torch`/`transformers`/`numpy` --
`mlx.core` arrays cover everything needed here (elementwise ops, softmax,
top-k via `argpartition`, reductions); where a small amount of data needs
python-side aggregation (per-expert routing counts), we materialize it with
`.tolist()` rather than reaching for numpy, since the calibration batches
involved are tiny (a few thousand tokens total).
