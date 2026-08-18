# Supported Models

Higgs detects local model support from `config.json` `model_type`. The tables below are representative rather than exhaustive.

## Supported Architectures

| Architecture | `model_type` | Examples |
|---|---|---|
| LLaMA | `llama` | Llama 3 and CodeLlama |
| Mistral | `mistral` | Mistral 7B |
| Qwen2 | `qwen2` | Qwen2 and Qwen2.5 |
| Qwen3 | `qwen3` | Qwen3 |
| Qwen3.5+ (dense) | `qwen3_5`, `qwen3_5_text` | Qwen3.5 dense checkpoints; Qwen3.8-27B |
| Qwen3.5+ (MoE) | `qwen3_5_moe`, `qwen3_5_text_moe` | Qwen3.5-35B-A3B, Qwen3.6-35B-A3B |
| Qwen3-Next | `qwen3_next` | Qwen3-Coder hybrid checkpoints |
| Qwen3-MoE | `qwen3_moe` | Qwen3-30B-A3B |
| Gemma 2 | `gemma2` | Gemma 2 2B, 9B, and 27B |
| Gemma 3 | `gemma3`, `gemma3_text` | Gemma 3 1B, 4B, 12B, and 27B |
| Gemma 4 | `gemma4`, `gemma4_text`, `gemma4_unified` | Gemma 4 E2B, E4B (edge); 12B, 31B; 26B-A4B (MoE) |
| Phi-3 | `phi3` | Phi-3 Mini, Small, and Medium |
| Starcoder2 | `starcoder2` | Starcoder2 3B, 7B, and 15B |
| DeepSeek-V2 | `deepseek_v2` | DeepSeek-V2-Lite |
| LLaVA-Qwen2 | `llava-qwen2` | nanoLLaVA-1.5 |

### Gemma 3 / Gemma 4 notes

- These are text-language-model implementations. Multimodal checkpoints
  (`gemma3`, `gemma4`) load fine — their vision/audio tower weights are skipped
  and only the text model runs. The text weights may be nested under
  `language_model.` in such checkpoints; Higgs strips that prefix automatically.
- Gemma 4 E2B/E4B (per-layer-input embeddings + cross-layer KV sharing) and dense
  text variants are supported. The MoE variant (`gemma4` with
  `enable_moe_block`, e.g. 26B-A4B) is supported only with **unquantized** expert
  weights; a checkpoint with quantized experts is rejected at load with a clear
  error rather than producing incorrect output.

## Continuous Batching Support

`batch=true` enables true batched decode only for these `model_type` values:

- `llama`
- `mistral`
- `qwen2`
- `qwen3`

Other supported architectures still serve normally in simple mode, but Higgs now rejects `batch=true` during config load, `doctor`, and server startup.

## Representative Working MLX Model IDs

| Family | Example model IDs |
|---|---|
| LLaMA | `mlx-community/Llama-3.2-1B-Instruct-4bit` |
| Qwen2.5 | `mlx-community/Qwen2.5-3B-Instruct-4bit` |
| Qwen3 | `mlx-community/Qwen3-1.7B-4bit` |
| Qwen3-Next | `mlx-community/Qwen3-Coder-Next-4bit` |
| Qwen3.5 dense | `mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit` |
| Qwen3.5 MoE | `NexVeridian/Qwen3.5-35B-A3B-3bit` |
| Qwen3.6 MoE | `mlx-community/Qwen3.6-35B-A3B-4bit` |
| Qwen3.8 dense | `mlx-community/Qwen3.8-27B-4bit` |
| DeepSeek-V2 | `mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx` |

## Qwen 3.5+ Adapter and Version Notes

- Qwen 3.5, 3.6, and 3.8 dense and MoE checkpoints use the Qwen 3.5 adapters. This includes `*ForConditionalGeneration` wrapper configs: Higgs detects the top-level wrapper and consumes the nested `text_config` used by the text loader.
- `_text` aliases such as `qwen3_5_text` and `qwen3_5_text_moe` resolve to the corresponding dense or MoE adapter.
- Unknown newer versions within a supported family can use the nearest adapter only after the resolved config passes structural validation. Higgs logs an untested-version warning when it takes this tolerant path. A missing or invalid required field is rejected by name; an unknown family is rejected with the supported family/version list.
- `mlx-community/Qwen3.8-27B-4bit` (27B dense, 4-bit) is verified working through its top-level `qwen3_5` wrapper and nested `qwen3_5_text` config.
- The cached-model smoke matrix covered `mlx-community/Qwen3.6-35B-A3B-4bit` plus `mlx-community/Llama-3.2-1B-Instruct-4bit`, `mlx-community/Qwen2.5-3B-Instruct-4bit`, `mlx-community/Qwen3-1.7B-4bit`, and `mlx-community/Qwen3-Coder-Next-4bit`.
- OpenAI-style chat requests use non-thinking mode by default for `Qwen3.6` unless the request explicitly opts into reasoning.

## Model Input Requirements

- Local models can be referenced by Hugging Face model ID or local path.
- The model must be in MLX `safetensors` format.
- The checkpoint must use a supported `config.json` `model_type`.
- macOS local serving requires `mlx.metallib` next to the executable. Release artifacts bundle it, and source builds restore it from Cargo build output when possible.

For configuration details, see [configuration.md](configuration.md).
