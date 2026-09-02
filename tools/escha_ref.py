#!/usr/bin/env python3
"""NumPy reference decoder for EschaLabs' `eschamoe` weight format.

Phase 0 oracle for native eschamoe support in higgs. This is deliberately slow
and dependency-light (numpy only) -- it exists to pin the format and to serve as
ground truth for the Metal kernel, not to run a model.

The trellis layout is exllamav3's EXL3 (MIT), batched over experts:

  escha_code  I16 [E, in/16, out/16, 16*K]   <- EXL3 `trellis`
  escha_rin   F16 [E, in]                    <- EXL3 `suh`
  escha_rout  F16 [E, out]                   <- EXL3 `svh`
  escha_s_in  F32 [E, in]                    escha addition
  escha_s_out F32 [E, out]                   escha addition
  escha_config I32 [9]                       escha addition

Bit packing and the tile permutation are ported literally from
exllamav3_ext/quant/pack.cu (`unpack_trellis_kernel`) and
exl3_lib/quantize.py (`tensor_core_perm`). The 3INST codebook is
codebook.cuh `decode_3inst<1>`.

Pinned empirically against the unquantized `Qwen/Qwen3.6-35B-A3B`, by scoring
every plausible ordering (`probe`) rather than reversing the shipped wheel --
the right one wins by a landslide (0.97 vs 0.00 cosine):

  escha_config = [tile=16, K, bits, mcg, num_experts, in, out, in_p, out_p]
  W[out, in]   = (Had_128 . Ŵ . Had_128 * r_in[:,None] * r_out[None,:]).T
                 with tiles decoding to [in, out]; both scales OUTSIDE the
                 Hadamard.
  escha_s_in / escha_s_out are identically 1.0 in this checkpoint -- carried
  through anyway, since nothing guarantees that for future ones.

Verified over layers {0,1,3,20,39} x experts {0,7} x both projections
(K=2 and K=3): global cosine 0.945 - 0.990 against the base model.

Usage:
    python3 tools/escha_ref.py gate [layers] [experts]        # Phase 0 gate
    python3 tools/escha_ref.py probe [layer] [proj] [expert]  # score orderings
    python3 tools/escha_ref.py dump <layer> <proj> <expert> <out.npy>

`layers` and `experts` are comma-separated index specs, each item a single
index or an inclusive range: `0-39`, `0,7,15`, `0-3,20`. Both default to the
sample above.

All three read the checkpoints over HTTP with byte ranges, so a run costs a few
MB rather than a full shard download.
"""

from __future__ import annotations

import json
import struct
import sys
import urllib.request

import numpy as np

ESCHA_REPO = "EschaLabs/Qwen3.6-35B-A3B-Escha-W2"
BASE_REPO = "Qwen/Qwen3.6-35B-A3B"

MCG_MULT = np.uint32(0xCBAC1FED)
LOP3_AND = np.uint32(0x8FFF8FFF)
LOP3_XOR = np.uint32(0x3B603B60)
HAD_BLOCK = 128


# ---------------------------------------------------------------------------
# safetensors reading (header parse + byte ranges; no dependency, works on both
# a local file and an HTTP URL so the Phase 0 gate needs ~10MB, not 3GB)
# ---------------------------------------------------------------------------

_DTYPES = {
    "F64": np.dtype("<f8"), "F32": np.dtype("<f4"), "F16": np.dtype("<f2"),
    "I64": np.dtype("<i8"), "I32": np.dtype("<i4"), "I16": np.dtype("<i2"),
    "I8": np.dtype("<i1"), "U8": np.dtype("<u1"), "BOOL": np.dtype("?"),
    "BF16": np.dtype("<u2"),  # widened to f32 on read
}


class Shard:
    """Random-access reader for one safetensors file, local path or URL."""

    def __init__(self, src: str):
        self.src = src
        self.remote = src.startswith("http")
        n = struct.unpack("<Q", self._read(0, 8))[0]
        self.header = json.loads(self._read(8, n))
        self.data_start = 8 + n

    def _read(self, offset: int, length: int) -> bytes:
        if not self.remote:
            with open(self.src, "rb") as f:
                f.seek(offset)
                data = f.read(length)
            if len(data) != length:
                raise OSError(f"short read: {len(data)} != {length}")
            return data
        req = urllib.request.Request(
            self.src, headers={"Range": f"bytes={offset}-{offset + length - 1}"}
        )
        # HTTPS only: the repo URL is fixed, but a redirected or spoofed
        # scheme would otherwise pass straight through to urlopen.
        if not req.full_url.startswith("https://"):
            raise ValueError(f"refusing non-https URL: {req.full_url}")
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        if len(data) != length:
            raise OSError(f"short range read: {len(data)} != {length}")
        return data

    def __contains__(self, key: str) -> bool:
        return key in self.header

    def keys(self):
        return [k for k in self.header if k != "__metadata__"]

    def get(self, key: str, index: int | None = None) -> np.ndarray:
        """Read a tensor. `index` slices the leading (expert) axis, so only that
        slice is transferred -- the point of the whole range-read design."""
        meta = self.header[key]
        dt = _DTYPES[meta["dtype"]]
        shape = list(meta["shape"])
        begin, end = meta["data_offsets"]

        if index is not None:
            if not 0 <= index < shape[0]:
                raise IndexError(
                    f"{key}: index {index} out of range for axis 0 of {shape[0]}"
                )
            stride = int(np.prod(shape[1:])) * dt.itemsize
            begin += index * stride
            end = begin + stride
            shape = shape[1:]

        raw = self._read(self.data_start + begin, end - begin)
        arr = np.frombuffer(raw, dtype=dt).reshape(shape)
        if meta["dtype"] == "BF16":
            arr = (arr.astype(np.uint32) << 16).view(np.float32)
        return arr


def hf_url(repo: str, filename: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{filename}"


_INDEX: dict[str, dict[str, str]] = {}
_SHARDS: dict[tuple[str, str], "Shard"] = {}


def weight_map(repo: str) -> dict[str, str]:
    if repo not in _INDEX:
        with urllib.request.urlopen(hf_url(repo, "model.safetensors.index.json")) as r:
            _INDEX[repo] = json.load(r)["weight_map"]
    return _INDEX[repo]


def shard_for(repo: str, key: str) -> "Shard":
    """The shard holding `key`. A single module's tensors can straddle a shard
    boundary, so this must be resolved per tensor, not per prefix."""
    fn = weight_map(repo)[key]
    if (repo, fn) not in _SHARDS:
        _SHARDS[(repo, fn)] = Shard(hf_url(repo, fn))
    return _SHARDS[(repo, fn)]


def resolve_shard(repo: str, key: str) -> str:
    return hf_url(repo, weight_map(repo)[key])


# ---------------------------------------------------------------------------
# Trellis unpack -- literal port of exllamav3_ext/quant/pack.cu
# ---------------------------------------------------------------------------

def unpack_trellis(packed: np.ndarray, k: int) -> np.ndarray:
    """(..., 16*K) int16 -> (..., 256) uint16 tail-biting trellis codes.

    Each code is a 16-bit window into a circular 256*K-bit stream at stride K;
    consecutive codes overlap by 16-K bits. Thread `t` of the CUDA kernel emits
    codes 2t and 2t+1, so this vectorizes over t in [0, 128).
    """
    assert packed.shape[-1] == 16 * k, f"expected last dim {16 * k}, got {packed.shape[-1]}"
    lead = packed.shape[:-1]
    u32 = packed.reshape(-1, 16 * k).view(np.uint32)  # (N, 8*K)
    n_words = k * 256 // 32

    t = np.arange(128)
    b0 = t * 2 * k + k - 16 + 256 * k
    b2 = b0 + k + 16
    i0 = b0 // 32
    i1 = (b2 - 1) // 32
    s1 = (i1 + 1) * 32 - b2

    a = u32[:, i0 % n_words].astype(np.uint64)
    b = u32[:, i1 % n_words].astype(np.uint64)

    # __funnelshift_r(lo=b, hi=a, s1): bits [s1+31:s1] of the 64-bit {a, b}
    w1 = (((a << np.uint64(32)) | b) >> s1.astype(np.uint64)).astype(np.uint32)
    w0 = (w1 >> np.uint32(k)) & np.uint32(0xFFFF)
    w1 = w1 & np.uint32(0xFFFF)

    codes = np.empty((u32.shape[0], 256), dtype=np.uint16)
    codes[:, 0::2] = w0
    codes[:, 1::2] = w1
    return codes.reshape(*lead, 256)


def tensor_core_perm() -> np.ndarray:
    """256-entry 16x16 tile permutation (exl3_lib/quantize.py)."""
    perm = np.empty(256, dtype=np.int64)
    for t in range(32):
        r0 = (t % 4) * 2
        c0 = t // 4
        rows = (r0, r0 + 1, r0 + 8, r0 + 9)
        for j, c in enumerate((c0, c0 + 8)):
            for i, r in enumerate(rows):
                perm[t * 8 + j * 4 + i] = r * 16 + c
    return perm


PERM = tensor_core_perm()


def decode_3inst(codes: np.ndarray) -> np.ndarray:
    """codebook.cuh decode_3inst<1>: MCG multiply, mask/xor into two fp16 lanes,
    sum them. `lop3.b32 ... 0x6a` is (a & b) ^ c."""
    x = codes.astype(np.uint32) * MCG_MULT
    x = (x & LOP3_AND) ^ LOP3_XOR
    halves = x.view(np.uint16).reshape(*x.shape, 2).astype(np.uint16)
    return halves.view(np.float16).astype(np.float32).sum(axis=-1)


def decode_tiles(code: np.ndarray, k: int) -> np.ndarray:
    """[tk, tn, 16*K] int16 -> [tk*16, tn*16] float32, permutation undone."""
    tk, tn = code.shape[0], code.shape[1]
    vals = decode_3inst(unpack_trellis(code, k))          # [tk, tn, 256]
    tiles = np.empty_like(vals)
    tiles[..., PERM] = vals                                # scatter to row-major
    return tiles.reshape(tk, tn, 16, 16).transpose(0, 2, 1, 3).reshape(tk * 16, tn * 16)


def had128(x: np.ndarray, axis: int) -> np.ndarray:
    """Orthonormal blockwise Sylvester Hadamard over `axis` in 128-wide blocks."""
    x = np.moveaxis(x, axis, -1)
    shape = x.shape
    y = x.reshape(-1, shape[-1] // HAD_BLOCK, HAD_BLOCK).astype(np.float32)
    h = 1
    while h < HAD_BLOCK:
        y = y.reshape(y.shape[0], y.shape[1], -1, 2, h)
        lo, hi = y[..., 0, :], y[..., 1, :]
        y = np.stack([lo + hi, lo - hi], axis=-2)
        h *= 2
    y = y.reshape(shape) / np.sqrt(HAD_BLOCK)
    return np.moveaxis(y, -1, axis)


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------

# (name, fp16 scale inside the Hadamard?, fp32 scale inside?)
VARIANTS = [
    ("both_outside", False, False),
    ("r_inside", True, False),
    ("both_inside", True, True),
    ("s_inside", False, True),
]


def reconstruct(t: dict[str, np.ndarray], k: int, variant: str) -> np.ndarray:
    """Rebuild one expert's weight as [out, in] float32.

    Tiles decode to [in, out]; scales are per-input-channel (`*_in`) and
    per-output-channel (`*_out`). The Hadamard is blockwise-128 on both axes.
    """
    _, r_inside, s_inside = next(v for v in VARIANTS if v[0] == variant)
    w = decode_tiles(t["code"], k)                                     # [in, out]

    def sc(inside: bool, which: str) -> np.ndarray:
        keys = [key for key, ins in (("r", r_inside), ("s", s_inside)) if ins == inside]
        out = np.ones(w.shape[0 if which == "in" else 1], dtype=np.float32)
        for key in keys:
            out = out * t[f"{key}_{which}"].astype(np.float32)
        return out

    w = w * sc(True, "in")[:, None] * sc(True, "out")[None, :]
    w = had128(had128(w, axis=0), axis=1)
    w = w * sc(False, "in")[:, None] * sc(False, "out")[None, :]
    return w.T


TENSORS = (("code", "code"), ("r_in", "rin"), ("r_out", "rout"),
           ("s_in", "s_in"), ("s_out", "s_out"))


def load_expert(prefix: str, expert: int, repo: str = ESCHA_REPO) -> dict[str, np.ndarray]:
    return {
        short: shard_for(repo, f"{prefix}.escha_{name}").get(
            f"{prefix}.escha_{name}", index=expert
        )
        for short, name in TENSORS
    }


def escha_k(prefix: str, repo: str = ESCHA_REPO) -> int:
    key = f"{prefix}.escha_code"
    return int(shard_for(repo, key).header[key]["shape"][-1]) // 16


def live_mask(ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rows/columns of `ref` that are not structurally dead.

    Qwen3.6's experts have pruned intermediate channels: whole `gate_up` output
    rows are exactly zero, and so are the matching `down_proj` input columns
    (280 of 512 in layer 0 expert 0). Escha only encodes the dead structure on
    the output side, via exact zeros in `escha_rout` -- on the input side the
    trellis codes for dead columns are unconstrained don't-cares, because the
    activation feeding them is exactly zero at inference (SwiGLU of a zero row).
    Scoring that garbage would report a correct decode as broken.
    """
    return np.linalg.norm(ref, axis=1) > 0, np.linalg.norm(ref, axis=0) > 0


def cosine(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(global cosine, mean per-row cosine), both over live entries of `b`."""
    rows, cols = live_mask(b)
    a, b = a[np.ix_(rows, cols)], b[np.ix_(rows, cols)]
    if b.size == 0:
        return 0.0, 0.0
    glob = float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    per_row = float(((a * b).sum(axis=1) / (na * nb + 1e-12)).mean())
    return glob, per_row


# ---------------------------------------------------------------------------

def prefix_for(layer: int, proj: str) -> str:
    return f"model.language_model.layers.{layer}.mlp.experts.{proj}"


def base_ref(prefix: str, expert: int, like: np.ndarray) -> np.ndarray:
    """The unquantized expert, oriented to match `like`. The base checkpoint
    stores stacked experts under the bare module name (no `.weight`)."""
    ref = shard_for(BASE_REPO, prefix).get(prefix, index=expert)
    return ref.T if ref.shape != like.shape else ref


def score(layer: int, proj: str, expert: int, variant: str = "both_outside"):
    prefix = prefix_for(layer, proj)
    w = reconstruct(load_expert(prefix, expert), escha_k(prefix), variant)
    ref = base_ref(prefix, expert, w)
    glob, rows = cosine(w, ref)
    live = np.ix_(*live_mask(ref))
    rel = float(np.linalg.norm(w[live] - ref[live]) / (np.linalg.norm(ref[live]) + 1e-12))
    return glob, rows, rel


def probe(layer: int = 0, proj: str = "gate_up_proj", expert: int = 0) -> int:
    """Pin the scale/Hadamard ordering: score every variant against the
    unquantized base model. Only the correct ordering scores high."""
    prefix = prefix_for(layer, proj)
    cfg = shard_for(ESCHA_REPO, f"{prefix}.escha_config").get(f"{prefix}.escha_config")
    print(f"{prefix}\nescha_config = {cfg.tolist()}   K = {escha_k(prefix)}")

    print("\n  variant         global   per-row   rel.fro")
    best = ("", -1.0)
    for name, _, _ in VARIANTS:
        glob, rows, rel = score(layer, proj, expert, name)
        print(f"  {name:<14}  {glob:+.4f}  {rows:+.4f}   {rel:.4f}")
        if glob > best[1]:
            best = (name, glob)
    print(f"\nbest: {best[0]}  global cosine={best[1]:+.4f}")
    return 0 if best[1] >= 0.9 else 1


GATE_LAYERS = (0, 1, 3, 20, 39)
GATE_EXPERTS = (0, 7)


def index_set(spec: str, default: tuple[int, ...]) -> tuple[int, ...]:
    """Parse an index spec: comma-separated items, each a single index `N` or
    an inclusive range `A-B`. An empty spec keeps `default`."""
    if not spec:
        return default
    out: list[int] = []
    for item in spec.split(","):
        lo, _, hi = item.partition("-")
        out.extend(range(int(lo), int(hi or lo) + 1))
    return tuple(out)


def gate(layers: str = "", experts: str = "") -> int:
    """Phase 0 gate: decode must track the unquantized base model everywhere.

    `layers` and `experts` are index specs (see `index_set`). Each empty one
    keeps the sample the format was pinned on: layers {0,1,3,20,39} x experts
    {0,7}. A run costs two range reads per cell, so a wide sweep is slow but
    still cheap in bytes.
    """
    print("layer  expert  gate_up   down")
    worst, worst_at = 1.0, ""
    for layer in index_set(layers, GATE_LAYERS):
        for expert in index_set(experts, GATE_EXPERTS):
            cols = [score(layer, proj, expert)[0] for proj in ("gate_up_proj", "down_proj")]
            print(f"  {layer:3}  {expert:5}   {cols[0]:+.4f}  {cols[1]:+.4f}")
            for proj, c in zip(("gate_up_proj", "down_proj"), cols):
                if c < worst:
                    worst, worst_at = c, f"L{layer}.{proj}.e{expert}"
    print(f"\nworst: {worst:+.4f} at {worst_at}")
    if worst >= 0.9:
        print("GATE PASS")
        return 0
    print("GATE FAIL (need >= 0.9)")
    return 1


def write_safetensors(path: str, tensors: dict[str, np.ndarray]) -> None:
    """Minimal safetensors writer (the `safetensors` package is not a dep here)."""
    rev = {v: k for k, v in _DTYPES.items() if k != "BF16"}
    header, blobs, offset = {}, [], 0
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr)
        header[name] = {
            "dtype": rev[arr.dtype],
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + arr.nbytes],
        }
        offset += arr.nbytes
        blobs.append(arr.tobytes())
    blob = json.dumps(header).encode()
    pad = (-len(blob)) % 8
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob) + pad))
        f.write(blob + b" " * pad)
        for b in blobs:
            f.write(b)


def fixture(out: str = "/tmp/escha_fixture.safetensors", layer: int = 0,
            proj: str = "gate_up_proj", expert: int = 0) -> int:
    """Dump one expert plus its expected dequantization, for the Rust test to
    check `dequant_expert` against this validated reference end to end."""
    prefix = prefix_for(layer, proj)
    t = load_expert(prefix, expert)
    w = reconstruct(t, escha_k(prefix), "both_outside")
    write_safetensors(out, {
        "code": t["code"].astype("<i2"),
        "rin": t["r_in"].astype("<f2"), "rout": t["r_out"].astype("<f2"),
        "s_in": t["s_in"].astype("<f4"), "s_out": t["s_out"].astype("<f4"),
        "expected": w.astype("<f4"),
    })
    glob, _ = cosine(w, base_ref(prefix, expert, w))
    print(f"wrote {out}  expected{list(w.shape)}  (cosine vs base {glob:+.4f})")
    return 0


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else ""
    args = [int(a) if a.lstrip("-").isdigit() else a for a in argv[2:]]
    if cmd == "probe":
        return probe(*args)
    if cmd == "gate":
        return gate(*argv[2:])
    if cmd == "fixture":
        return fixture(*args)
    if cmd == "dump" and len(args) == 4:
        layer, proj, expert, out = args
        prefix = prefix_for(layer, proj)
        w = reconstruct(load_expert(prefix, expert), escha_k(prefix), "both_outside")
        np.save(out, w)
        print(f"wrote {out} {list(w.shape)}")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
