# -*- coding: utf-8 -*-
"""The divergence floor between the reference and a port that dequantizes.

The reference runs its linear layers in fp8 with 32x32 blocks and ue8m0 scales, and quantizes the
activation as well. llama.cpp dequantizes the same weights and computes in another precision. Two
different arithmetics over the same numbers do not give the same result, and the difference is
not a defect -- it is the floor.

This probe measures it on a single linear, with the true activation vector captured at layer 0
(a `llama-logits --dump attn_norm-0` file). If the correlation it returns is the one the port
already reaches at that point, the port is at the floor and there is nothing to fix there.

Needs the reference checkpoint (its `inference/` package provides the fp8 GEMM) and a CUDA GPU.

  python fp8_floor.py --ref-dir <DeepSeek-V4.1-Flash> --vector <dump of attn_norm-0>
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch


def read_dump(path):
    """A llama-logits dump: 8-byte magic, ne[4] as int64, then float32 data."""
    with open(path, "rb") as f:
        assert f.read(8) == b"DSV4DMP1"
        ne = struct.unpack("<4q", f.read(32))
        v = np.fromfile(f, dtype=np.float32)
    v = v.reshape(ne[3], ne[2], ne[1], ne[0])
    return v[0, 0, -1] if ne[2] == 1 else v[0, -1]


def metrics(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    d = a - b
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return (float(np.corrcoef(a, b)[0, 1]),
            float(a @ b / (na * nb)),
            float(np.linalg.norm(d) / nb),
            na, nb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-dir", required=True, help="the Hugging Face checkpoint directory")
    ap.add_argument("--vector", required=True, help="llama-logits dump of attn_norm-0")
    ap.add_argument("--layer", type=int, default=0)
    a = ap.parse_args()

    ref = Path(a.ref_dir)
    sys.path.insert(0, str(ref / "inference"))
    from safetensors.torch import load_file

    # the fp8 GEMM allocates its output in the default dtype and expects bf16
    torch.set_default_dtype(torch.bfloat16)
    import model as M

    idx = json.load(open(ref / "model.safetensors.index.json"))["weight_map"]
    x = torch.tensor(read_dump(a.vector), dtype=torch.bfloat16, device="cuda").unsqueeze(0)
    print(f"activation: {tuple(x.shape)} norm {x.float().norm().item():.4f}")

    print()
    print(f" {'linear':10s} {'correlation':>11s} {'cos':>7s} {'rel err':>9s} "
          f"{'norm fp8':>10s} {'norm exact':>12s}")
    print(" " + "-" * 66)

    for name in ("wq_a", "wkv"):
        key = f"layers.{a.layer}.attn.{name}.weight"
        if key not in idx:
            continue
        t = load_file(str(ref / idx[key]))
        w = t[key].cuda()
        s = t[f"layers.{a.layer}.attn.{name}.scale"].cuda()
        if w.dtype != torch.float8_e4m3fn:
            print(f" {name:10s} (not fp8: {w.dtype})")
            continue
        w.scale = s

        # the reference path: quantize the activation and call the fp8 GEMM
        y_fp8 = M.linear(x, w)
        # the port's path: dequantize the weight and compute in high precision. The scale is
        # ue8m0, so its value is 2^(bits-127) -- the same arithmetic the converter does
        bits = s.view(torch.uint8).float()
        scale = torch.exp2(bits - 127.0)
        scale = scale.repeat_interleave(32, 0).repeat_interleave(32, 1)[: w.shape[0], : w.shape[1]]
        w_deq = w.float() * scale
        y_exact = (x.float() @ w_deq.T)

        corr, cos, rel, n1, n2 = metrics(y_fp8.float().cpu().numpy(), y_exact.cpu().numpy())
        print(f" {name:10s} {corr:11.6f} {cos:7.4f} {rel:9.4f} {n1:10.3f} {n2:12.3f}")

    print()
    print("Reading: the correlation above is the best an exact port can reach at that point.")
    print("If the port is already there, the residue is arithmetic, not a defect.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
