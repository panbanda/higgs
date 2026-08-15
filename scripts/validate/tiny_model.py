#!/usr/bin/env python3
"""Create a tiny random dense Qwen3 checkpoint for quality-gate plumbing tests.

Requires: pip install torch transformers tokenizers
The output is deliberately random and is useful only for local/CI harness
plumbing; `make_tiny_fixture.sh` converts it to quantized MLX format.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # A byte-level tokenizer keeps arbitrary raw prompts encodable while the
    # intentionally small vocabulary keeps the checkpoint and conversion quick.
    vocab = {"<|endoftext|>": 0, "<|unk|>": 1}
    vocab.update({bytes([index]).decode("latin-1"): index + 2 for index in range(256)})
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        unk_token="<|unk|>",
    )
    fast_tokenizer.save_pretrained(args.out)

    config = Qwen3Config(
        vocab_size=1024,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=512,
        bos_token_id=0,
        eos_token_id=0,
    )
    Qwen3ForCausalLM(config).save_pretrained(args.out, safe_serialization=True)


if __name__ == "__main__":
    main()
