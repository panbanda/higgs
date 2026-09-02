#!/usr/bin/env python3
"""Non-expert forensics: escha (int8/bf16) vs mlx-community 4-bit, tensor by tensor.

Offline, numpy only, one tensor at a time via safetensors byte ranges.
Reuses tools/escha_ref.py's Shard reader. Import-only helper code — run the
analysis by importing and calling :func:`cos_table` / :func:`layout` from a
REPL or script.

Environment:
    ESCHA_LOCAL_MODEL   path to the eschamoe checkpoint
                        (default: the EschaLabs Qwen3.6-35B-A3B-Escha-W2 cache)
    BASE_LOCAL_MODEL    path to the reference 4-bit checkpoint
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import escha_ref as E  # noqa: E402

ESCHA = os.environ.get(
    "ESCHA_LOCAL_MODEL",
    "/Users/peppi/.cache/lm-studio/models/EschaLabs/Qwen3.6-35B-A3B-Escha-W2",
)
BASE = os.environ.get(
    "BASE_LOCAL_MODEL",
    "/Users/peppi/.cache/lm-studio/models/mlx-community/Qwen3.6-35B-A3B-4bit",
)

E._DTYPES.setdefault("U32", np.dtype("<u4"))
E._DTYPES.setdefault("U16", np.dtype("<u2"))

_IDX: dict[str, dict] = {}
_SH: dict[tuple[str, str], E.Shard] = {}


def wmap(root: str) -> dict[str, str]:
    if root not in _IDX:
        _IDX[root] = json.load(open(f"{root}/model.safetensors.index.json"))["weight_map"]
    return _IDX[root]


def shard(root: str, key: str) -> E.Shard:
    fn = wmap(root)[key]
    if (root, fn) not in _SH:
        _SH[(root, fn)] = E.Shard(f"{root}/{fn}")
    return _SH[(root, fn)]


def get(root: str, key: str, rows: slice | None = None) -> np.ndarray:
    """Read a tensor, optionally only `rows` of the leading axis."""
    s = shard(root, key)
    meta = s.header[key]
    dt = E._DTYPES[meta["dtype"]]
    shape = list(meta["shape"])
    begin, end = meta["data_offsets"]
    if rows is not None:
        stride = int(np.prod(shape[1:])) * dt.itemsize
        lo, hi, _ = rows.indices(shape[0])
        begin, end = begin + lo * stride, begin + hi * stride
        shape[0] = hi - lo
    raw = s._read(s.data_start + begin, end - begin)
    a = np.frombuffer(raw, dtype=dt).reshape(shape)
    if meta["dtype"] == "BF16":
        a = (a.astype(np.uint32) << 16).view(np.float32)
    return a.astype(np.float32) if a.dtype != np.float32 else a


def raw_get(root: str, key: str, rows: slice | None = None) -> np.ndarray:
    """Same but keeps the on-disk integer dtype (for packed uint32 / int8)."""
    s = shard(root, key)
    meta = s.header[key]
    dt = E._DTYPES[meta["dtype"]]
    shape = list(meta["shape"])
    begin, end = meta["data_offsets"]
    if rows is not None:
        stride = int(np.prod(shape[1:])) * dt.itemsize
        lo, hi, _ = rows.indices(shape[0])
        begin, end = begin + lo * stride, begin + hi * stride
        shape[0] = hi - lo
    raw = s._read(s.data_start + begin, end - begin)
    return np.frombuffer(raw, dtype=dt).reshape(shape)


_QCFG: dict | None = None


def _qcfg() -> dict:
    """Load the reference checkpoint's quantization block on first use, so an
    import needs no checkpoint on disk."""
    global _QCFG
    if _QCFG is None:
        _QCFG = json.load(open(f"{BASE}/config.json"))["quantization"]
    return _QCFG


def qbits(module: str) -> tuple[int, int]:
    v = _qcfg().get(module)
    if isinstance(v, dict):
        return int(v["group_size"]), int(v["bits"])
    cfg = _qcfg()
    return int(cfg["group_size"]), int(cfg["bits"])


def deq_affine(root: str, module: str, rows: slice | None = None) -> np.ndarray:
    """MLX affine dequant: w = q * scale + bias, groups along the last axis."""
    g, b = qbits(module)
    packed = raw_get(root, f"{module}.weight", rows).view(np.uint32)
    scales = get(root, f"{module}.scales", rows)
    biases = get(root, f"{module}.biases", rows)
    per = 32 // b
    shifts = (np.arange(per, dtype=np.uint32) * b)
    q = (packed[..., None] >> shifts) & np.uint32((1 << b) - 1)
    q = q.reshape(*packed.shape[:-1], packed.shape[-1] * per).astype(np.float32)
    n = q.shape[-1]
    q = q.reshape(*q.shape[:-1], n // g, g)
    w = q * scales[..., None] + biases[..., None]
    return w.reshape(*packed.shape[:-1], n)


def deq_int8(root: str, module: str, rows: slice | None = None) -> np.ndarray:
    q = raw_get(root, f"{module}.weight_int8", rows).astype(np.float32)
    s = get(root, f"{module}.weight_scale", rows)
    return q * s.reshape(-1, *([1] * (q.ndim - 1)))


def cos(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def rowcos(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(a.shape[0], -1).astype(np.float64)
    b = b.reshape(b.shape[0], -1).astype(np.float64)
    n = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    ok = n > 0
    return float(((a * b).sum(axis=1)[ok] / n[ok]).mean()) if ok.any() else 0.0


def escha_key(module: str) -> str:
    """De-normalize back to the escha checkpoint's own naming."""
    if module.startswith("language_model.model."):
        return "model.language_model." + module[len("language_model.model."):]
    if module.startswith("language_model.lm_head"):
        return module[len("language_model."):]
    return module


def escha_side(module: str, rows: slice | None = None) -> tuple[np.ndarray, str]:
    k = escha_key(module)
    wm = wmap(ESCHA)
    if f"{k}.weight_int8" in wm:
        return deq_int8(ESCHA, k, rows), "int8"
    if f"{k}.weight" in wm:
        return get(ESCHA, f"{k}.weight", rows), "f16"
    return get(ESCHA, k, rows), "f16"


def base_side(module: str, rows: slice | None = None) -> tuple[np.ndarray, str]:
    wm = wmap(BASE)
    if f"{module}.scales" in wm:
        g, b = qbits(module)
        return deq_affine(BASE, module, rows), f"q{b}g{g}"
    if f"{module}.weight" in wm:
        return get(BASE, f"{module}.weight", rows), "bf16"
    return get(BASE, module, rows), "bf16"
