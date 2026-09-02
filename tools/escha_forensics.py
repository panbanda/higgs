#!/usr/bin/env python3
"""Plausibility floor for the escha decode: our decoded weight vs naive RTN
baselines applied to the *same* base tensor, plus residual structure tests.

Reuses tools/escha_ref.py verbatim for loading + reconstruction; the escha
side is served from the local LM Studio cache, the base side over HTTP byte
ranges (one expert slice, a few MB).

    python3 tools/escha_forensics.py [layer] [proj] [expert]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import escha_ref as E  # noqa: E402

LOCAL_ESCHA = os.environ.get(
    "ESCHA_LOCAL_MODEL",
    "/Users/peppi/.cache/lm-studio/models/EschaLabs/Qwen3.6-35B-A3B-Escha-W2",
)

_orig_hf_url = E.hf_url


def hf_url(repo: str, filename: str) -> str:
    if repo == E.ESCHA_REPO:
        return f"{LOCAL_ESCHA}/{filename}"
    return _orig_hf_url(repo, filename)


E.hf_url = hf_url


def weight_map(repo: str):
    if repo not in E._INDEX:
        if repo == E.ESCHA_REPO:
            import json
            with open(f"{LOCAL_ESCHA}/model.safetensors.index.json") as f:
                E._INDEX[repo] = json.load(f)["weight_map"]
        else:
            return _orig_weight_map(repo)
    return E._INDEX[repo]


_orig_weight_map = E.weight_map
E.weight_map = weight_map


# ---------------------------------------------------------------------------
# naive RTN baselines, applied to the BASE tensor
# ---------------------------------------------------------------------------

def rtn_sym(w: np.ndarray, bits: int, group: int | None) -> np.ndarray:
    """Symmetric absmax RTN over `w` [out, in].

    group=None -> one scale per output channel (row).
    group=g    -> one scale per contiguous run of g input channels within a row.
    Levels: {-2^(b-1) .. 2^(b-1)-1}, s = absmax / 2^(b-1).
    """
    out, inn = w.shape
    g = inn if group is None else group
    assert inn % g == 0
    x = w.reshape(out, inn // g, g)
    qmax = 2 ** (bits - 1)
    s = np.abs(x).max(axis=-1, keepdims=True) / qmax
    s = np.where(s == 0, 1.0, s)
    q = np.clip(np.rint(x / s), -qmax, qmax - 1)
    return (q * s).reshape(out, inn)


def rtn_asym(w: np.ndarray, bits: int, group: int | None) -> np.ndarray:
    """Asymmetric min/max RTN (the usual GPTQ/AWQ baseline), same grouping."""
    out, inn = w.shape
    g = inn if group is None else group
    assert inn % g == 0
    x = w.reshape(out, inn // g, g)
    lo = x.min(axis=-1, keepdims=True)
    hi = x.max(axis=-1, keepdims=True)
    s = (hi - lo) / (2 ** bits - 1)
    s = np.where(s == 0, 1.0, s)
    q = np.clip(np.rint((x - lo) / s), 0, 2 ** bits - 1)
    return (q * s + lo).reshape(out, inn)


# ---------------------------------------------------------------------------
# residual structure tests
# ---------------------------------------------------------------------------

def mod_profile(v: np.ndarray, keep: np.ndarray, stride: int) -> np.ndarray:
    """Mean of `v` grouped by index mod `stride`, over live channels only.

    `v` is indexed in the ORIGINAL (unmasked) channel space so that `i % 16` and
    `i % 128` still mean tile row and Hadamard-block lane; `keep` marks which
    entries of `v` are meaningful.
    """
    idx = np.arange(len(v)) % stride
    num = np.bincount(idx[keep], weights=v[keep], minlength=stride)
    den = np.bincount(idx[keep], minlength=stride)
    return num[den > 0] / den[den > 0]


def structure(name: str, r: np.ndarray, ref: np.ndarray,
              rows: np.ndarray, cols: np.ndarray) -> dict:
    """r, ref are FULL [out, in]; rows/cols are the live masks (index space kept)."""
    live = np.ix_(rows, cols)
    a = np.zeros_like(ref)
    w = np.zeros_like(ref)
    a[live] = np.abs(r[live])
    w[live] = np.abs(ref[live])
    nr, nc = rows.sum(), cols.sum()
    # per-channel RELATIVE error: |R| profile normalised by the base tensor's own
    # per-channel magnitude, so a channel that is simply large in W does not read
    # as "structure". Structure = the decode being worse on some channels.
    ch = (a.sum(axis=1) / nc) / (w.sum(axis=1) / nc + 1e-30)   # per output channel
    ci = (a.sum(axis=0) / nr) / (w.sum(axis=0) / nr + 1e-30)   # per input channel
    rl, rf = r[live], ref[live]
    d = {"name": name, "rel_fro": float(np.linalg.norm(rl) / np.linalg.norm(rf))}
    for tag, prof, keep in (("out", ch, rows), ("in", ci, cols)):
        live_prof = prof[keep]
        d[f"{tag}_min"] = float(live_prof.min())
        d[f"{tag}_max"] = float(live_prof.max())
        d[f"{tag}_ratio"] = float(live_prof.max() / (live_prof.min() + 1e-30))
        d[f"{tag}_cv"] = float(live_prof.std() / (live_prof.mean() + 1e-30))
        for stride in (16, 128):
            p = mod_profile(prof, keep, stride)
            d[f"{tag}_mod{stride}"] = float(p.max() / (p.min() + 1e-30))
    return d


def fmt_struct(d: dict) -> str:
    return (f"  {d['name']:<26} relF={d['rel_fro']:.4f}  "
            f"|R| per-out ch min/max/ratio = {d['out_min']:.3e}/{d['out_max']:.3e}/"
            f"{d['out_ratio']:8.2f}  cv={d['out_cv']:.3f}\n"
            f"  {'':<26} |R| per-in  ch min/max/ratio = {d['in_min']:.3e}/{d['in_max']:.3e}/"
            f"{d['in_ratio']:8.2f}  cv={d['in_cv']:.3f}\n"
            f"  {'':<26} mod-16 peak/trough: out={d['out_mod16']:.3f} in={d['in_mod16']:.3f}   "
            f"mod-128: out={d['out_mod128']:.3f} in={d['in_mod128']:.3f}")


def runs(idx: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous runs in a sorted index array, as (start, length)."""
    if len(idx) == 0:
        return []
    brk = np.where(np.diff(idx) != 1)[0]
    starts = np.r_[0, brk + 1]
    ends = np.r_[brk, len(idx) - 1]
    return [(int(idx[s]), int(e - s + 1)) for s, e in zip(starts, ends)]


def dead(layer: int = 0, expert: int = 0) -> int:
    """What does our decode put in the structurally-dead channels?

    Every cosine in the table above was live-MASKED. This measures the part that
    was masked out, and asks whether the checkpoint itself signals deadness.
    """
    print(f"===== layer {layer}, expert {expert} =====")
    info = {}
    for proj in ("gate_up_proj", "down_proj"):
        prefix = E.prefix_for(layer, proj)
        k = E.escha_k(prefix)
        cfg = E.shard_for(E.ESCHA_REPO, f"{prefix}.escha_config").get(
            f"{prefix}.escha_config").tolist()
        t = E.load_expert(prefix, expert)
        ours = E.reconstruct(t, k, "both_outside").astype(np.float32)
        ref = E.base_ref(prefix, expert, ours).astype(np.float32)
        rows, cols = E.live_mask(ref)          # <- derived from the BASE tensor
        dr = np.where(~rows)[0]
        dc = np.where(~cols)[0]

        # NOTE cfg layout: [tile, K, bits, mcg, num_experts, in, out, in_p, out_p]
        print(f"\n{proj}  shape={list(ours.shape)} [out, in]  K={k}")
        print(f"  escha_config in={cfg[5]} out={cfg[6]} in_p={cfg[7]} out_p={cfg[8]}"
              f"   -> padded? in:{cfg[7] != cfg[5]}  out:{cfg[8] != cfg[6]}")
        print(f"  live rows(out) {rows.sum()}/{len(rows)}   dead {len(dr)}")
        print(f"  live cols(in)  {cols.sum()}/{len(cols)}   dead {len(dc)}")

        # (2) magnitude of OUR decode in the dead output rows
        for tag, idx, ax in (("row(out)", dr, 1), ("col(in)", dc, 0)):
            if len(idx) == 0:
                continue
            keep = np.ones(ours.shape[1 - ax], dtype=bool)  # unused axis, all
            o_rms = np.sqrt((ours ** 2).mean(axis=ax))
            live_idx = np.setdiff1d(np.arange(len(o_rms)), idx)
            dmean = float(o_rms[idx].mean())
            lmean = float(o_rms[live_idx].mean())
            print(f"  OURS dead {tag}: RMS mean={dmean:.4e} max={o_rms[idx].max():.4e}"
                  f"  | live RMS mean={lmean:.4e}  -> dead/live = {dmean / (lmean + 1e-30):.4f}")
            print(f"       dead {tag} exactly zero? {np.all(np.take(ours, idx, axis=1 - ax) == 0)}"
                  f"   max|ours| on dead = {np.abs(np.take(ours, idx, axis=1 - ax)).max():.4e}")
            # (3) what the BASE holds there
            b = np.take(ref, idx, axis=1 - ax)
            print(f"       BASE on dead {tag}: all exact zero? {np.all(b == 0)}"
                  f"   max|base| = {np.abs(b).max():.4e}")

        # (4) does the checkpoint signal deadness?
        for nm in ("r_out", "s_out", "r_in", "s_in"):
            v = np.asarray(t[nm]).astype(np.float32)
            nz = int((v == 0).sum())
            print(f"  escha {nm:5} len={len(v):5}  #exact-zero={nz:5}"
                  f"  min|nonzero|={np.abs(v[v != 0]).min() if nz < len(v) else 0:.3e}")
        rz = np.where(np.asarray(t["r_out"]).astype(np.float32) == 0)[0]
        if len(dr):
            print(f"  r_out zero-set == dead out-row set? "
                  f"{len(rz) == len(dr) and bool(np.all(rz == dr))}  "
                  f"(|r_out==0|={len(rz)}, |dead rows|={len(dr)})")
        info[proj] = (ours.shape, dr, dc, rows, cols)

    # (5) index pattern / gate-up split arithmetic
    gu_shape, gu_dr, _, _, _ = info["gate_up_proj"]
    _, _, dp_dc, _, dp_cols = info["down_proj"]
    half = gu_shape[0] // 2
    g_dead = gu_dr[gu_dr < half]
    u_dead = gu_dr[gu_dr >= half] - half
    print(f"\n-- index pattern --")
    print(f"  gate_up out={gu_shape[0]} -> gate[0:{half}] + up[{half}:{gu_shape[0]}]"
          f"   moe_intermediate_size={half}")
    print(f"  dead in gate half: {len(g_dead)}   dead in up half: {len(u_dead)}")
    print(f"  gate-dead set == up-dead set? {len(g_dead) == len(u_dead) and bool(np.all(g_dead == u_dead))}")
    print(f"  down_proj dead in-cols: {len(dp_dc)}   == gate-dead set? "
          f"{len(dp_dc) == len(g_dead) and bool(np.all(dp_dc == g_dead))}")
    print(f"  first 24 dead gate idx: {g_dead[:24].tolist()}")
    print(f"  contiguous runs (start,len), first 8: {runs(g_dead)[:8]}   total runs={len(runs(g_dead))}")
    if len(g_dead):
        print(f"  dead idx mod 16 histogram: {np.bincount(g_dead % 16, minlength=16).tolist()}")
    return 0


def main(layer: int = 0, proj: str = "gate_up_proj", expert: int = 0) -> int:
    prefix = E.prefix_for(layer, proj)
    k = E.escha_k(prefix)
    cfg = E.shard_for(E.ESCHA_REPO, f"{prefix}.escha_config").get(f"{prefix}.escha_config")
    ours = E.reconstruct(E.load_expert(prefix, expert), k, "both_outside")
    ref = E.base_ref(prefix, expert, ours)
    print(f"tensor: {prefix}  expert={expert}  K={k}  bits={cfg.tolist()[2]}  shape={list(ours.shape)}")
    print(f"escha_config = {cfg.tolist()}")

    rows, cols = E.live_mask(ref)
    live = np.ix_(rows, cols)
    print(f"live: {rows.sum()}/{len(rows)} out rows, {cols.sum()}/{len(cols)} in cols")

    ref = ref.astype(np.float32)
    refl = ref[live]
    full = {
        "escha decode (ours)": ours.astype(np.float32),
        "RTN 2-bit, per-out-ch (absmax)": rtn_sym(ref, 2, None),
        "RTN 2-bit, group-64 (absmax)": rtn_sym(ref, 2, 64),
        "RTN 2-bit, per-out-ch (asym)": rtn_asym(ref, 2, None),
        "RTN 2-bit, group-64 (asym)": rtn_asym(ref, 2, 64),
        "RTN 4-bit, group-64 (absmax)": rtn_sym(ref, 4, 64),
        "RTN 4-bit, group-64 (asym)": rtn_asym(ref, 4, 64),
        "RTN 8-bit, group-64 (asym)": rtn_asym(ref, 8, 64),
    }
    cands = {k2: v[live] for k2, v in full.items()}

    print("\n  method                          cosine    per-row   rel.fro  sqrt(1-rel^2)  <R,W>/|R||W|")
    for name, a in cands.items():
        glob = float((a * refl).sum() / (np.linalg.norm(a) * np.linalg.norm(refl) + 1e-12))
        na = np.linalg.norm(a, axis=1)
        nb = np.linalg.norm(refl, axis=1)
        per_row = float(((a * refl).sum(axis=1) / (na * nb + 1e-12)).mean())
        r = refl - a
        rel = float(np.linalg.norm(r) / np.linalg.norm(refl))
        orth = float((r * refl).sum() / (np.linalg.norm(r) * np.linalg.norm(refl) + 1e-12))
        print(f"  {name:<32} {glob:+.4f}   {per_row:+.4f}   {rel:.4f}   "
              f"{np.sqrt(max(0.0, 1 - rel * rel)):.4f}        {orth:+.4f}")
    print(f"\n  Gaussian rate-distortion bound at R bits: rel.fro = 2^-R  ->  "
          f"R=2: 0.2500   R=3: 0.1250   (this tensor's trellis rate K={k})")

    print("\nresidual structure  (R = base - method)")
    for name in ("escha decode (ours)", "RTN 2-bit, group-64 (absmax)"):
        print(fmt_struct(structure(name, ref - full[name], ref, rows, cols)))
    return 0


if __name__ == "__main__":
    a = [int(x) if x.lstrip("-").isdigit() else x for x in sys.argv[1:]]
    if a and a[0] == "dead":
        sys.exit(dead(*a[1:]))
    sys.exit(main(*a))
