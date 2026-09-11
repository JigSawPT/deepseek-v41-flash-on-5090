# -*- coding: utf-8 -*-
"""The floor the reference pays on its ACTIVATIONS, and the port does not.

fp8_floor.py measured the floor of the weights. The activations have their own, and on this
model it is not small:

  _window_kv        act_quant(kv, 32, ue8m0, inplace=True)     -> fp8 round trip
  _compress_kv      fp4_act_quant(latent, 16, True, e4m3)      -> fp4 round trip

These are roundings the reference applies to its own state and llama.cpp does not (it keeps the
KV in f16). So a layer that compresses diverges more than one that does not, with nothing wrong
-- and layer 2 diverges more than layer 0.

Measures the round trip on the true vectors captured from the graph.

  python kv_floor.py --ref-dir <DeepSeek-V4.1-Flash> --kv <dump of kv-0>
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch


def read_dump(path):
    with open(path, "rb") as f:
        assert f.read(8) == b"DSV4DMP1"
        ne = struct.unpack("<4q", f.read(32))
        v = np.fromfile(f, dtype=np.float32)
    v = v.reshape(ne[3], ne[2], ne[1], ne[0])
    return v[0, 0, -1] if ne[2] == 1 else v[0, -1]


def metrics(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    nb = np.linalg.norm(b)
    return (float(np.corrcoef(a, b)[0, 1]),
            float(np.linalg.norm(a - b) / nb),
            float(np.abs(a - b).max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-dir", required=True, help="the Hugging Face checkpoint directory")
    ap.add_argument("--kv", required=True, help="llama-logits dump of kv-<il>, the window KV")
    a = ap.parse_args()

    sys.path.insert(0, str(Path(a.ref_dir) / "inference"))
    torch.set_default_dtype(torch.bfloat16)
    import model as M
    from kernel import act_quant, fp4_act_quant

    v = read_dump(a.kv).astype(np.float32)
    print(f"vector: {v.shape}  norm {np.linalg.norm(v):.4f}")
    print()
    print(f" {'round trip':26s} {'correlation':>11s} {'rel err':>9s} {'max|d|':>9s}")
    print(" " + "-" * 59)

    # fp8 with blocks of 32 and ue8m0 scales, exactly as _window_kv does
    x = torch.tensor(v, dtype=torch.bfloat16, device="cuda").unsqueeze(0).contiguous()
    orig = x.float().cpu().numpy().copy()
    act_quant(x, M.fp8_block_size, M.scale_fmt, M.scale_dtype, True)
    corr, rel, mx = metrics(x.float().cpu().numpy(), orig)
    print(f" {'fp8 block 32 (window)':26s} {corr:11.6f} {rel:9.4f} {mx:9.4f}")

    # fp4 with blocks of 16 and e4m3 scales: _compress_kv passes the 16 by hand, it does not use
    # fp4_block_size (32) -- the compressed latent is the coarsest state in the model
    y = torch.tensor(v, dtype=torch.bfloat16, device="cuda").unsqueeze(0).contiguous()
    fp4_act_quant(y, 16, True, scale_dtype=torch.float8_e4m3fn)
    corr, rel, mx = metrics(y.float().cpu().numpy(), orig)
    print(f" {'fp4 block 16 (latent)':26s} {corr:11.6f} {rel:9.4f} {mx:9.4f}")

    # fp4 with blocks of 32 and the default scale: what the indexer does to q and k
    z = torch.tensor(v, dtype=torch.bfloat16, device="cuda").unsqueeze(0).contiguous()
    fp4_act_quant(z, M.fp4_block_size, True)
    corr, rel, mx = metrics(z.float().cpu().numpy(), orig)
    print(f" {'fp4 block 32 (indexer)':26s} {corr:11.6f} {rel:9.4f} {mx:9.4f}")

    print()
    print("Reading: this is what the reference does to its own state and the port does not.")
    print("The layer that compresses pays the lower lines; the one that does not, does not.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
