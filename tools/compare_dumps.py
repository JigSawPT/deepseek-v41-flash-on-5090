# -*- coding: utf-8 -*-
"""Puts the sub-steps of one block side by side: llama.cpp against the reference.

Reads the `llama-logits --dump` files (header DSV4DMP1 + ne[4] as int64 + float32) and the
reference's .npy files, and prints one line per sub-step in the order the block produces them.

The reading is always the same: the first sub-step whose correlation drops is the one that
contains the defect, because everything before it was already matching. This is how the
per-head q norm was found (norm 181.019 = sqrt(64 x 512): every head at RMS 1).

The reference .npy files were captured with forward hooks on the reference model through the
house's Python probe (not published); the dump names below are the `cb()` names of the fork's
graph, which `llama-logits --dump` writes without extra instrumentation.

  python compare_dumps.py --cpp <dump prefix> --ref <directory of .npy> --layer 0
"""

import argparse
import struct
from pathlib import Path

import numpy as np

# the order the block produces them; any missing name is simply skipped
ORDER = [
    "hc_init",
    "hc_attn_pre",
    "attn_norm",
    "qr_norm",
    "q",
    "kv",
    "attn_raw",
    "attn_derope",
    "attn_wo_a",
    "attn_out",
    "hc_attn_post",
    "hc_ffn_pre",
    "ffn_norm",
    "ffn_moe_out",
    "ffn_shexp",
    "ffn_out",
    "l_last",
]

# dumps whose token axis is not the last: attn_wo_a leaves the mul_mat as
# [o_lora_rank, tokens, groups], because the group is the batch axis
TOKEN_AXIS_1 = {"attn_wo_a"}


def read_dump(path, name=""):
    """Returns the last token of the dumped tensor, shaped [hc, d] or [d]."""
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != b"DSV4DMP1":
            raise ValueError(f"{path}: unexpected header {magic!r}")
        ne = struct.unpack("<4q", f.read(32))
        v = np.fromfile(f, dtype=np.float32)
    # ggml varies ne[0] fastest
    v = v.reshape(ne[3], ne[2], ne[1], ne[0])
    if name in TOKEN_AXIS_1:   # [d, tokens, batch]: keep the batch, pick the token
        return v[0, :, -1, :]
    if ne[2] > 1:              # [d, hc, tokens]: the last position, with the hc copies
        return v[0, -1]
    return v[0, 0, -1]         # [d, tokens]: the last position


def metrics(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size != b.size:
        return None
    d = a - b
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    cos = float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")
    corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
    return corr, cos, float(np.abs(d).max()), float(np.abs(d).mean()), float(na), float(nb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpp", required=True, help="prefix of the --dump files")
    ap.add_argument("--ref", required=True, help="directory of the reference .npy files")
    ap.add_argument("--layer", type=int, default=0)
    a = ap.parse_args()

    ref = Path(a.ref)
    print(f"=== layer {a.layer}: sub-step by sub-step ===")
    print(f" {'sub-step':16s} {'correlation':>11s} {'cos':>7s} {'max|d|':>9s} "
          f"{'mean|d|':>9s} {'norm cpp':>10s} {'norm ref':>10s}")
    print(" " + "-" * 80)

    previous_ok = True
    first_bad = None
    for name in ORDER:
        # hc_init carries no layer suffix: it is produced once, before the loop
        suffix = "" if name == "hc_init" else f"-{a.layer}"
        f_cpp = Path(f"{a.cpp}.{name}{suffix}.bin")
        f_ref = ref / f"{name}.npy"
        if not f_cpp.exists() or not f_ref.exists():
            print(f" {name:16s} {'(missing)':>11s}")
            continue
        m = metrics(read_dump(f_cpp, name), np.load(f_ref))
        if m is None:
            print(f" {name:16s} {'(shapes differ)':>11s}")
            continue
        corr, cos, mx, md, na, nb = m
        mark = ""
        if corr < 0.9999 and previous_ok:
            mark = "  <- OPENS HERE"
            first_bad = name
            previous_ok = False
        print(f" {name:16s} {corr:11.6f} {cos:7.3f} {mx:9.4f} {md:9.4f} "
              f"{na:10.3f} {nb:10.3f}{mark}")

    print()
    if first_bad is None:
        print("No sub-step opens: either the defect is in another layer, or in something none")
        print("of these capture points crosses.")
    else:
        print(f"First sub-step to diverge: {first_bad}")
        print("What comes before it matches, so the defect is in the operation that produces")
        print("it -- not in what it receives.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
