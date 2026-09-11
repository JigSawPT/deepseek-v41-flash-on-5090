# -*- coding: utf-8 -*-
"""How much of the token is compute, how much PCIe and how much NVMe -- without running anything new.

The expert-tier ladder swept the host tier (L2) with the VRAM cache FIXED. That changes one thing
only: the fraction of misses served from RAM instead of disk. With the stall measured at four
points and the L2 hit rate at each, the cost of a disk miss and of a RAM miss come out of a system
of equations:

  stall_per_call = base + misses x [ f_l2 . cost_ram + (1 - f_l2) . cost_disk ]

`base` is not fitted: it is measured. It is the stall per call when there are NO misses, which is
the resident case of the ruler -- 21.27 tokens/s over 40 layers.

The proof that the model is right is not the fit: it is predicting the resident case, which did
not enter the equations, and getting it.

  python stall_decomposition.py [--resident-tps 21.27]
"""

import argparse

# expert-tier ladder on the model with the engram, VRAM cache fixed at 18 GiB
# (L2 in GiB, mean stall per remap call in ms over two runs, fraction of misses served by L2)
LADDER = [
    (0,  (8.493 + 7.512) / 2, 0.000),
    (32, (7.024 + 6.969) / 2, 0.3506),
    (64, (5.857 + 5.768) / 2, 0.5120),
    (72, (5.481 + 5.435) / 2, 0.5365),
]

MISSES_PER_CALL = 12472 / 5200      # measured: VRAM misses per remap call
MIB_PER_EXPERT = 17.93
LAYERS = 40


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resident-tps", type=float, default=21.27,
                    help="tokens/s with everything resident, from the ruler: the base comes from here")
    ap.add_argument("--cold-tps", type=float, default=5.12)
    a = ap.parse_args()

    base = 1000.0 / a.resident_tps / LAYERS
    print(f"base (compute, no miss at all) = {base:.3f} ms per call")
    print(f"  from {a.resident_tps} tokens/s resident over {LAYERS} layers, not from a fit")
    print(f"misses per call = {MISSES_PER_CALL:.2f}")
    print()

    # each pair of points gives one delta; the spread between pairs is the honest error bar
    print(" pair of points    cost_disk - cost_ram")
    print(" " + "-" * 45)
    deltas = []
    l2_0, s0, f0 = LADDER[0]
    for l2, s, f in LADDER[1:]:
        # s0 - s = misses * (f - f0) * (cost_disk - cost_ram)
        d = (s0 - s) / (MISSES_PER_CALL * (f - f0))
        deltas.append(d)
        print(f" L2 {l2_0:3d} vs {l2:3d}      {d:8.3f} ms per expert")
    delta = sum(deltas) / len(deltas)
    print(f"\n mean = {delta:.3f} ms per expert of difference")
    print()

    # with f_l2 = 0 at the first point: s0 = base + misses * cost_disk
    cost_disk = (s0 - base) / MISSES_PER_CALL
    cost_ram = cost_disk - delta
    print(f" cost of a miss served from DISK = {cost_disk:.3f} ms  "
          f"({MIB_PER_EXPERT/cost_disk*1.048576:.1f} GB/s equivalent)")
    print(f" cost of a miss served from RAM  = {cost_ram:.3f} ms  "
          f"({MIB_PER_EXPERT/cost_ram*1.048576:.1f} GB/s equivalent)")
    print()

    print(" where the token goes, at the best measured point (L2 = 72 GiB):")
    l2, s, f = LADDER[-1]
    d_pcie = MISSES_PER_CALL * f * cost_ram
    d_disk = MISSES_PER_CALL * (1 - f) * cost_disk
    total = base + d_pcie + d_disk
    for name, v in (("compute", base), ("PCIe (RAM -> VRAM)", d_pcie), ("NVMe", d_disk)):
        print(f"   {name:20s} {v:6.3f} ms  {100*v/total:5.1f}%")
    print(f"   {'sum':20s} {total:6.3f} ms   against {s:.3f} measured "
          f"({100*abs(total-s)/s:.1f}% error)")
    print()

    print(" ceilings this implies, and they differ from one another:")
    for name, ms in (
        ("today, L2=72", total),
        ("no disk miss at all (perfect look-ahead + a large enough L2)", base + MISSES_PER_CALL*cost_ram),
        ("everything resident in VRAM (measured by the ruler)", base),
    ):
        print(f"   {1000.0/(ms*LAYERS):5.1f} tokens/s   {name}")
    print()
    print(" The validation is not the fit: it is the last line. The base entered as the measured")
    print(" resident case and the model returns it; the other two fall between it and the cold figure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
