# -*- coding: utf-8 -*-
"""How many of the 6 chosen experts agree between the port and the reference, layer by layer.

Why it matters: at layer 2 the shared expert -- same input, same arithmetic, no choice -- loses
0.0022 of correlation from its input, and the routed MoE loses 0.0304. Fourteen times more. The
only difference between the two is that the second one chooses.

The choice is discrete: a 1 % perturbation of the scores does not change the output by 1 %, it
changes it entirely if it swaps an expert. That is why a 1.5 % floating-point error per linear
turns into 10 % of divergence after 40 layers -- not linear accumulation, a switch.

Reads the llama.cpp side from `llama-logits --dump ffn_moe_topk-<il>,...` files and the reference
side from an .npz of per-layer top-k indices (one int array per layer, keyed by the layer index
as a string). The reference .npz was produced by hooking every `Gate` of the reference model with
the house's Python probe, which is not published; producing one needs the reference model
resident. The numbers in the report (208/240, 86.7 %) come from that run.

  python routing_switch.py --cpp <dump prefix> --ref results/routes_reference.npz --layers 0,2,5,...
"""

import argparse
import struct
from pathlib import Path

import numpy as np


def read_dump_i32(path):
    """ffn_moe_topk comes out as [n_expert_used, tokens]; returns the last token."""
    with open(path, "rb") as f:
        assert f.read(8) == b"DSV4DMP1"
        ne = struct.unpack("<4q", f.read(32))
        v = np.fromfile(f, dtype=np.float32)
    v = v.reshape(ne[3], ne[2], ne[1], ne[0])
    return v[0, 0, -1].astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpp", required=True, help="dump prefix: <prefix>.ffn_moe_topk-<il>.bin")
    ap.add_argument("--ref", required=True, help=".npz with one int array of chosen experts per layer, keyed by layer index")
    ap.add_argument("--layers", required=True, help="comma-separated layer indices")
    a = ap.parse_args()

    ref = np.load(a.ref)
    layers = [int(c) for c in a.layers.split(",") if c.strip()]
    print(" layer    equal/6   only in cpp              only in reference")
    print(" " + "-" * 72)
    total_equal = total = 0
    for il in layers:
        f = Path(f"{a.cpp}.ffn_moe_topk-{il}.bin")
        if not f.exists() or str(il) not in ref:
            print(f" {il:6d}   (missing)")
            continue
        c = set(read_dump_i32(f).tolist())
        r = set(ref[str(il)].tolist())
        equal = len(c & r)
        total_equal += equal
        total += len(r)
        print(f" {il:6d}   {equal}/{len(r)}        {str(sorted(c - r)):24s} {sorted(r - c)}")
    if total:
        print()
        print(f" overall agreement: {total_equal}/{total} = {100.0*total_equal/total:.1f}%")
        print()
        print(" A swapped expert is not a small error: it is another function applied to the")
        print(" same token. It is what turns floating-point noise into divergence, and it is not")
        print(" fixed by more precision on one side only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
