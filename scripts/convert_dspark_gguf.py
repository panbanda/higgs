#!/usr/bin/env python3
"""Convert a Prism dSpark GGUF sidecar into Higgs/MLX safetensors.

The converter intentionally omits ``token_embd.weight`` because dSpark is tied
to the target Bonsai model and Higgs uses the target embedding for the draft
block.  It keeps dSpark's higher-precision output projection and low-rank
Markov head, which are part of Prism's published inference algorithm.

Dependencies are deliberately kept outside the Rust workspace::

    python -m venv /tmp/dspark-convert
    /tmp/dspark-convert/bin/pip install gguf mlx safetensors numpy

Example::

    python scripts/convert_dspark_gguf.py \
      Bonsai-27B-dspark-Q4_1.gguf /tmp/Bonsai-27B-dspark-mlx \
      --target-dir ~/.cache/lm-studio/models/prism-ml/Bonsai-27B-mlx-1bit

``--reuse-target-head`` omits dSpark's frozen Q4 output copy and uses the
paired Bonsai target's packed Q1 head for proposals. This experimental compact
profile keeps final generation exact through verification, but its proposals
are not the trained Prism distribution and may have lower acceptance. The
default preserves the frozen Q4 proposal head.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from gguf import GGMLQuantizationType, GGUFReader, dequantize


def _selected_target_files(target_dir: Path) -> list[Path]:
    """Mirror Higgs' base-checkpoint selection, excluding MTP sidecars."""
    index_path = target_dir / "model.safetensors.index.json"
    single_path = target_dir / "model.safetensors"
    if index_path.exists():
        with index_path.open(encoding="utf-8") as file:
            index = json.load(file)
        files = sorted(
            {target_dir / name for name in index.get("weight_map", {}).values()}
        )
        if files and not all(path.exists() for path in files) and single_path.exists():
            files = [single_path]
    elif single_path.exists():
        files = [single_path]
    else:
        raise FileNotFoundError(f"no target safetensors found in {target_dir}")
    if not files or not all(path.is_file() for path in files):
        missing = [str(path) for path in files if not path.is_file()]
        raise FileNotFoundError(f"target checkpoint shards are missing: {missing}")
    return files


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _target_binding(target_dir: Path) -> dict[str, Any]:
    config_path = target_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"target config missing: {config_path}")
    selected = [config_path, *_selected_target_files(target_dir)]
    files = []
    for path in sorted(selected, key=lambda item: item.relative_to(target_dir).as_posix()):
        relative = path.relative_to(target_dir).as_posix()
        files.append(
            {"path": relative, "size": path.stat().st_size, "sha256": _sha256(path)}
        )
    return {"format": "higgs-target-artifact-v1", "files": files}


def _field_values(reader: GGUFReader, name: str) -> list[Any]:
    field = reader.fields[name]
    values: list[Any] = []
    for part_index in field.data:
        part = field.parts[part_index]
        if part.size == 1:
            values.append(part.item())
        else:
            values.append(part.tolist())
    return values


def _scalar(reader: GGUFReader, name: str) -> Any:
    values = _field_values(reader, name)
    if len(values) != 1:
        raise ValueError(f"expected scalar GGUF field {name}, got {values!r}")
    return values[0]


def _target_layers(reader: GGUFReader) -> list[int]:
    return [int(value) for value in _field_values(reader, "dspark.dspark.target_layers")]


def _tensor_mapping(name: str) -> tuple[str | None, bool]:
    """Return ``(Higgs name/base, quantize_as_qlinear)``."""
    exact: dict[str, tuple[str | None, bool]] = {
        "token_embd.weight": (None, False),
        "dspark.confidence_head.weight": (None, False),
        "dspark.confidence_head.bias": (None, False),
        "dspark.fc.weight": ("fc", True),
        "dspark.hidden_norm.weight": ("hidden_norm.weight", False),
        "output.weight": ("dspark.output", True),
        "output_norm.weight": ("norm.weight", False),
        "dspark.log_snr_fc1.weight": ("dspark.log_snr_fc1.weight", False),
        "dspark.log_snr_fc1.bias": ("dspark.log_snr_fc1.bias", False),
        "dspark.log_snr_fc2.weight": ("dspark.log_snr_fc2.weight", False),
        "dspark.log_snr_fc2.bias": ("dspark.log_snr_fc2.bias", False),
        "dspark.markov_head_a.weight": ("dspark.markov_head_a", False),
        "dspark.markov_head_b.weight": ("dspark.markov_head_b", True),
    }
    if name in exact:
        return exact[name]

    parts = name.split(".")
    if len(parts) != 4 or parts[0] != "blk" or parts[3] != "weight":
        raise ValueError(f"unmapped dSpark GGUF tensor: {name}")
    layer = int(parts[1])
    leaf = parts[2]
    prefix = f"layers.{layer}"
    layer_map: dict[str, tuple[str, bool]] = {
        "attn_norm": (f"{prefix}.input_layernorm.weight", False),
        "ffn_norm": (f"{prefix}.post_attention_layernorm.weight", False),
        "attn_q_norm": (f"{prefix}.self_attn.q_norm.weight", False),
        "attn_k_norm": (f"{prefix}.self_attn.k_norm.weight", False),
        "attn_q": (f"{prefix}.self_attn.q_proj", True),
        "attn_k": (f"{prefix}.self_attn.k_proj", True),
        "attn_v": (f"{prefix}.self_attn.v_proj", True),
        "attn_output": (f"{prefix}.self_attn.o_proj", True),
        "ffn_gate": (f"{prefix}.mlp.gate_proj", True),
        "ffn_up": (f"{prefix}.mlp.up_proj", True),
        "ffn_down": (f"{prefix}.mlp.down_proj", True),
    }
    try:
        return layer_map[leaf]
    except KeyError as exc:
        raise ValueError(f"unmapped dSpark layer tensor: {name}") from exc


def _dense_tensor(tensor: Any) -> np.ndarray:
    dense = dequantize(tensor.data, tensor.tensor_type)
    return np.asarray(dense)


def _q4_1_to_mlx(tensor: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Repack GGUF Q4_1/group32 losslessly into MLX affine Q4/group32.

    GGUF stores 16 bytes whose logical dequant order is their 16 low nibbles
    followed by their 16 high nibbles. MLX packs each consecutive run of eight
    logical values into one little-endian u32. The FP16 scale/min values map
    directly to MLX scale/bias.
    """
    input_width = int(tensor.shape[0])
    output_width = int(tensor.shape[1])
    if input_width % 32:
        raise ValueError(f"Q4_1 input width is not group32 aligned: {tensor.name}")
    blocks_per_row = input_width // 32
    raw = np.asarray(tensor.data).reshape(output_width, blocks_per_row, 20)
    scales = raw[..., 0:2].copy().view("<f2").reshape(output_width, blocks_per_row)
    biases = raw[..., 2:4].copy().view("<f2").reshape(output_width, blocks_per_row)
    nibbles = raw[..., 4:20]
    values = np.concatenate((nibbles & 0x0F, nibbles >> 4), axis=-1).astype(
        np.uint32
    )
    values = values.reshape(output_width, blocks_per_row, 4, 8)
    shifts = (np.arange(8, dtype=np.uint32) * 4).reshape(1, 1, 1, 8)
    packed = np.bitwise_or.reduce(values << shifts, axis=-1).reshape(
        output_width, input_width // 8
    )
    return packed, scales, biases


def convert(
    source: Path,
    output_dir: Path,
    target_dir: Path,
    group_size: int,
    reuse_target_head: bool,
) -> None:
    reader = GGUFReader(source)
    if int(_scalar(reader, "dspark.dspark.block_size")) <= 0:
        raise ValueError("invalid dSpark block size")
    if not bool(_scalar(reader, "dspark.dspark.log_snr_conditioning")):
        raise ValueError("this converter currently requires Prism log-SNR conditioning")

    output_dir.mkdir(parents=True, exist_ok=True)
    converted: dict[str, mx.array] = {}

    for index, tensor in enumerate(reader.tensors, start=1):
        mapped, quantized = _tensor_mapping(tensor.name)
        if reuse_target_head and tensor.name == "output.weight":
            mapped = None
        if mapped is None:
            print(f"[{index:02d}/{len(reader.tensors)}] skip {tensor.name}", flush=True)
            continue

        print(
            f"[{index:02d}/{len(reader.tensors)}] {tensor.name} -> {mapped}"
            f" ({'q4 affine' if quantized else 'dense'})",
            flush=True,
        )
        if quantized:
            if group_size == 32 and tensor.tensor_type == GGMLQuantizationType.Q4_1:
                packed, scale_values, bias_values = _q4_1_to_mlx(tensor)
                weight = mx.array(packed)
                scales = mx.array(scale_values)
                biases = mx.array(bias_values)
                del packed, scale_values, bias_values
            else:
                dense = _dense_tensor(tensor)
                if dense.ndim != 2:
                    raise ValueError(
                        f"quantized tensor must be 2D: {tensor.name} {dense.shape}"
                    )
                if dense.shape[-1] % group_size:
                    raise ValueError(
                        f"{tensor.name} input width {dense.shape[-1]} is not divisible by {group_size}"
                    )
                dense_mx = mx.array(dense).astype(mx.float16)
                weight, scales, biases = mx.quantize(
                    dense_mx, group_size=group_size, bits=4
                )
                del dense, dense_mx
            mx.eval(weight, scales, biases)
            converted[f"{mapped}.weight"] = weight
            converted[f"{mapped}.scales"] = scales
            converted[f"{mapped}.biases"] = biases
        else:
            dense = _dense_tensor(tensor)
            value = mx.array(dense)
            # GGUF BF16 weights are dequantized to f32 by gguf-py. Restore the
            # checkpoint dtype; keep native F32 norms/biases as F32.
            if tensor.tensor_type == GGMLQuantizationType.BF16:
                value = value.astype(mx.bfloat16)
            mx.eval(value)
            converted[mapped] = value
            del dense
        gc.collect()

    model_path = output_dir / "model.safetensors"
    mx.save_safetensors(
        str(model_path),
        converted,
        metadata={
            "format": "mlx",
            "source": f"{source.name}:Prism-dSpark-GGUF",
        },
    )

    config = {
        "model_type": "dspark",
        "hidden_size": int(_scalar(reader, "dspark.embedding_length")),
        "num_hidden_layers": int(_scalar(reader, "dspark.block_count")),
        "num_attention_heads": int(_scalar(reader, "dspark.attention.head_count")),
        "num_key_value_heads": int(
            _scalar(reader, "dspark.attention.head_count_kv")
        ),
        "head_dim": int(_scalar(reader, "dspark.attention.key_length")),
        "intermediate_size": int(_scalar(reader, "dspark.feed_forward_length")),
        "rms_norm_eps": float(
            _scalar(reader, "dspark.attention.layer_norm_rms_epsilon")
        ),
        "rope_theta": float(_scalar(reader, "dspark.rope.freq_base")),
        "block_size": int(_scalar(reader, "dspark.dspark.block_size")),
        "vocab_size": int(_scalar(reader, "dspark.vocab_size")),
        "quantization": {"group_size": group_size, "bits": 4, "mode": "affine"},
        "dflash_config": {
            "target_layer_ids": _target_layers(reader),
            "tap_semantics": "post_layer_residual_v1",
            "mask_token_id": int(_scalar(reader, "dspark.dspark.mask_token_id")),
            "dspark": True,
            "markov_rank": int(_scalar(reader, "dspark.dspark.markov_rank")),
            "log_snr_conditioning": True,
            "min_log_snr": float(_scalar(reader, "dspark.dspark.min_log_snr")),
            "max_log_snr": float(_scalar(reader, "dspark.dspark.max_log_snr")),
            "reuse_target_head": reuse_target_head,
            "target_binding": _target_binding(target_dir),
        },
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, sort_keys=True)
        file.write("\n")

    total_bytes = sum(array.nbytes for array in converted.values())
    print(f"wrote {model_path} ({total_bytes / 2**20:.1f} MiB tensors)")
    print(f"wrote {output_dir / 'config.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Prism dSpark GGUF")
    parser.add_argument("output_dir", type=Path, help="output MLX sidecar directory")
    parser.add_argument(
        "--target-dir",
        type=Path,
        required=True,
        help="exact paired target checkpoint directory to bind into the sidecar",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=32,
        choices=(32, 64, 128),
        help="MLX affine Q4 group size (default: preserve GGUF Q4_1's 32)",
    )
    parser.add_argument(
        "--reuse-target-head",
        action="store_true",
        help="omit dSpark's Q4 output copy and use the paired target head",
    )
    args = parser.parse_args()
    convert(
        args.source,
        args.output_dir,
        args.target_dir,
        args.group_size,
        args.reuse_target_head,
    )


if __name__ == "__main__":
    main()
