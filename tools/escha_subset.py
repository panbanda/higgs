#!/usr/bin/env python3
"""Assemble a small but real eschamoe checkpoint for loader/forward testing.

The published checkpoint is 12.3GB, which is more disk than a dev box
necessarily has spare. Almost none of it is needed to exercise the loader: the
layers are homogeneous and the experts are interchangeable. This builds a
truncated checkpoint holding the first N layers (enough to cover both a GDN
layer and a full-attention one) with the expert axis sliced down, plus the
embeddings, final norm and lm_head.

Every tensor is copied verbatim, so the trellis codes, scales and int8 weights
are real data in the real layout -- only the counts shrink. Output is a few
hundred MB.

The expert axis is leading on every per-expert tensor, so slicing it is a
contiguous byte prefix: a subset costs one ranged GET per tensor and no
server-side work.

Usage:
    python3 tools/escha_subset.py <out_dir> [layers] [experts] [source_dir]

Give `source_dir` to read a local copy of the full checkpoint. Then the tool
makes the subset with no network.

Defaults to 4 layers (indices 0-3, so `full_attention_interval=4` yields three
GDN layers and one full-attention layer) and 32 experts.
"""

from __future__ import annotations

import json
import os
import struct
import sys
import urllib.request

from escha_ref import ESCHA_REPO, hf_url, weight_map

# Per-expert tensors: leading axis is the expert, so a subset is a byte prefix.
EXPERT_AXIS_KEYS = (".escha_code", ".escha_rin", ".escha_rout",
                    ".escha_s_in", ".escha_s_out", ".mlp.gate.weight")
# Index of `num_experts` within the 9-element escha_config vector.
CONFIG_NUM_EXPERTS = 4

DTYPE_SIZE = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2,
              "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json",
                   "merges.txt", "generation_config.json", "chat_template.jinja")


class Source:
    """A checkpoint on a local disk or on the Hugging Face hub.

    A local directory is much faster. Use it when the full checkpoint is
    already available. The tool then makes the subset with no network.
    """

    def __init__(self, root: str | None = None):
        self.root = os.path.expanduser(root) if root else None
        self._headers: dict[str, tuple[dict, int]] = {}

    @property
    def local(self) -> bool:
        return self.root is not None

    def index(self) -> dict[str, str]:
        """Give the map from a tensor name to its shard file."""
        if not self.local:
            return weight_map(ESCHA_REPO)
        with open(os.path.join(self.root, "model.safetensors.index.json")) as f:
            return json.load(f)["weight_map"]

    def text(self, name: str) -> bytes:
        """Read a small file, for example `config.json`."""
        if not self.local:
            with urllib.request.urlopen(hf_url(ESCHA_REPO, name)) as r:
                return r.read()
        with open(os.path.join(self.root, name), "rb") as f:
            return f.read()

    def header(self, filename: str) -> tuple[dict, int]:
        """Give the safetensors header and the start of the data."""
        if filename not in self._headers:
            size = struct.unpack("<Q", self.read(filename, 0, 8))[0]
            header = json.loads(self.read(filename, 8, size))
            self._headers[filename] = (header, 8 + size)
        return self._headers[filename]

    def read(self, filename: str, start: int, length: int) -> bytes:
        """Read `length` bytes at offset `start`.

        The function tries the read four times. A short read makes an
        incorrect tensor, and the error is difficult to find later.
        """
        if self.local:
            with open(os.path.join(self.root, filename), "rb") as f:
                f.seek(start)
                data = f.read(length)
            if len(data) != length:
                raise OSError(f"short read: {len(data)} != {length}")
            return data

        url = hf_url(ESCHA_REPO, filename)
        for attempt in range(4):
            try:
                req = urllib.request.Request(
                    url, headers={"Range": f"bytes={start}-{start + length - 1}"}
                )
                data = urllib.request.urlopen(req, timeout=120).read()
                if len(data) == length:
                    return data
                raise OSError(f"short read: {len(data)} != {length}")
            except Exception as exc:  # noqa: BLE001 - the code tries again
                if attempt == 3:
                    raise
                print(f"    retry {attempt + 1} ({exc})", file=sys.stderr)
        raise AssertionError("unreachable")


def rewrite_config_experts(data: bytes, experts: int) -> bytes:
    """Clamp `num_experts` inside a packed 9-element escha_config vector."""
    vals = list(struct.unpack(f"<{len(data) // 4}i", data))
    vals[CONFIG_NUM_EXPERTS] = min(vals[CONFIG_NUM_EXPERTS], experts)
    return struct.pack(f"<{len(vals)}i", *vals)


def wanted(key: str, layers: int) -> bool:
    if key.startswith("mtp."):
        return False  # partial head in this checkpoint; the loader disables it
    if ".layers." not in key:
        return True
    index = int(key.split(".layers.")[1].split(".")[0])
    return index < layers


def subset(out_dir: str, layers: int = 4, experts: int = 32,
           source: str | None = None) -> int:
    src = Source(source)
    os.makedirs(out_dir, exist_ok=True)
    print(f"source: {src.root or ESCHA_REPO}")
    wmap = src.index()
    keys = sorted(k for k in wmap if wanted(k, layers))
    print(f"{len(keys)} tensors from {len(set(wmap[k] for k in keys))} shards")

    # Plan every tensor first: the safetensors header carries all offsets, so it
    # must be complete before a single byte of payload is written.
    plan, offset = [], 0
    for key in keys:
        filename = wmap[key]
        header, data_start = src.header(filename)
        meta = header[key]
        shape = list(meta["shape"])
        begin, end = meta["data_offsets"]
        nbytes = end - begin

        if any(key.endswith(s) for s in EXPERT_AXIS_KEYS):
            # Track the true expert count the tensors hold: a requested
            # `experts` above the checkpoint's count would make config.json
            # advertise experts the payload does not have, and escha_config
            # would disagree with it.
            experts = min(experts, shape[0])
            if shape[0] > experts:
                nbytes = nbytes * experts // shape[0]
                shape[0] = experts

        plan.append({
            "key": key, "dtype": meta["dtype"], "shape": shape,
            "src": (filename, data_start + begin, nbytes),
            "data_offsets": [offset, offset + nbytes],
        })
        offset += nbytes

    header = {p["key"]: {"dtype": p["dtype"], "shape": p["shape"],
                         "data_offsets": p["data_offsets"]} for p in plan}
    blob = json.dumps(header).encode()
    pad = (-len(blob)) % 8
    total = offset

    out_path = os.path.join(out_dir, "model.safetensors")
    print(f"writing {out_path} ({total / 2**30:.2f} GiB payload)")
    with open(out_path, "wb") as f:
        f.write(struct.pack("<Q", len(blob) + pad))
        f.write(blob + b" " * pad)
        for i, p in enumerate(plan, 1):
            filename, start, nbytes = p["src"]
            data = src.read(filename, start, nbytes)
            if p["key"].endswith(".escha_config"):
                # Rewrite num_experts so the spec matches the sliced tensors.
                # (`p["shape"]` here is the config vector's own length, not an
                # expert count -- it must not be used for this.)
                data = rewrite_config_experts(data, experts)
            f.write(data)
            if i % 25 == 0 or i == len(plan):
                print(f"  {i}/{len(plan)}")

    write_configs(out_dir, layers, experts, src)
    return 0


def write_configs(out_dir: str, layers: int, experts: int,
                  src: "Source | None" = None) -> None:
    src = src or Source()
    cfg = json.loads(src.text("config.json"))

    text = cfg["text_config"]
    text["num_hidden_layers"] = layers
    text["layer_types"] = text["layer_types"][:layers]
    text["num_experts"] = experts
    text["num_experts_per_tok"] = min(text["num_experts_per_tok"], experts)
    # No `mtp.layers.*` survive the subset, so don't advertise a draft head.
    text["mtp_num_hidden_layers"] = 0
    # Text-only: the published checkpoint ships no vision weights anyway.
    cfg.pop("vision_config", None)
    cfg["architectures"] = ["Qwen3_5MoeForCausalLM"]

    quant = cfg.get("quantization_config", {})
    if "layer_meta" in quant:
        quant["layer_meta"] = {
            k: v for k, v in quant["layer_meta"].items() if wanted(k, layers)
        }
        for meta in quant["layer_meta"].values():
            meta["num_experts"] = min(meta.get("num_experts", experts), experts)
    if "global_config" in quant:
        quant["global_config"]["num_experts"] = experts

    # Target the same affine layout as the reference build
    # (mlx-community/Qwen3.6-35B-A3B-4bit): 4-bit g=64 throughout, with the two
    # router gates per layer at 8-bit. The loader dequantizes the trellis and
    # requantizes into this, so a Phase-1 build is layout-identical to the
    # baseline and any quality delta is attributable to escha's 2-bit source.
    quantization = {"group_size": 64, "bits": 4, "mode": "affine"}
    for layer in range(layers):
        for gate in ("mlp.gate", "mlp.shared_expert_gate"):
            quantization[f"language_model.model.layers.{layer}.{gate}"] = {
                "group_size": 64,
                "bits": 8,
            }
    cfg["quantization"] = quantization

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    with open(os.path.join(out_dir, "quantize_config.json"), "w") as f:
        json.dump({"quant_method": "eschamoe", "bits": 2.0}, f, indent=2)

    for name in TOKENIZER_FILES:
        try:
            body = src.text(name)
        except Exception:  # noqa: BLE001 - these files are optional
            continue
        with open(os.path.join(out_dir, name), "wb") as f:
            f.write(body)

    print(f"wrote config for {layers} layers x {experts} experts -> {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    args = [int(a) if a.isdigit() else a for a in sys.argv[1:]]
    sys.exit(subset(*args))
