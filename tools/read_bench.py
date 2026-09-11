# -*- coding: utf-8 -*-
"""Reads a bench_server.py JSON separating NEW content from REPEATED content.

The usual rule "discard the first request of every start-up" was written for models where the
first request is cold because of the START-UP. On this model what is cold is the CONTENT: experts
are brought from disk as the text asks for them, and repeating the same prompt with greedy decoding
generates exactly the same tokens, which touch exactly the same experts, already resident.

Measured: round 1 gives 4.6-6.0 tokens/s and rounds 2 and 3 give 20-24, with the server's cold-miss
counter stopped. Discarding round 1 would publish four times the speed a real conversation sees.

So this reader never merges the two. They are two quantities:

  cold       content seen for the first time -- what a conversation does
  resident   the same content again -- the compute ceiling, with the I/O out of the way

  python read_bench.py --json results/bench_server.json
"""

import argparse
import json
import statistics
from pathlib import Path


def summary(vals):
    if not vals:
        return None
    v = sorted(vals)
    return statistics.median(v), v[0], v[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    a = ap.parse_args()

    d = json.loads(Path(a.json).read_text(encoding="utf-8"))
    print(f"model    {d.get('model')}")
    print(f"config   L2={d.get('l2_gib')} GiB  cache={d.get('cache_gib')} GiB  "
          f"pin={d.get('pin')}  mode={d.get('mode')}  ctx={d.get('ctx')}  extra={d.get('extra') or '-'}")
    print(f"startup  {d.get('startup_s')} s")
    print()

    print(f" {'prompt':12s} | {'COLD (new content)':>26s} | {'RESIDENT (repeated)':>26s}")
    print(f" {'':12s} | {'t/s':>10s} {'TTFT':>14s} | {'t/s':>10s} {'TTFT':>14s}")
    print(" " + "-" * 72)

    tps_cold, tps_res, ttft_cold, ttft_res = [], [], [], []
    for name, r in (d.get("results") or {}).items():
        v = r["rounds"]
        f, q = v[0], v[1:]
        tps_cold.append(f["tps"])
        ttft_cold.append(f["ttft_s"])
        tq = [x["tps"] for x in q]
        ttq = [x["ttft_s"] for x in q]
        tps_res += tq
        ttft_res += ttq
        mq = statistics.median(tq) if tq else float("nan")
        mtq = statistics.median(ttq) if ttq else float("nan")
        print(f" {name:12s} | {f['tps']:10.2f} {f['ttft_s']:13.2f}s | {mq:10.2f} {mtq:13.2f}s")

    print()
    rc, rr = summary(tps_cold), summary(tps_res)
    if rc:
        print(f" COLD       decode median {rc[0]:5.2f} t/s   (worst {rc[1]:.2f}, best {rc[2]:.2f})   "
              f"TTFT median {statistics.median(ttft_cold):.2f} s")
    if rr:
        print(f" RESIDENT   decode median {rr[0]:5.2f} t/s   (worst {rr[1]:.2f}, best {rr[2]:.2f})   "
              f"TTFT median {statistics.median(ttft_res):.2f} s")
    if rc and rr:
        print()
        print(f" I/O costs {rr[0]/max(rc[0],1e-9):.1f}x: the distance between what the card can compute")
        print(f" and what the machine can feed it. {rr[0]:.0f} t/s is the measured compute ceiling,")
        print(f" not a projection -- and {rc[0]:.1f} t/s is what a conversation sees.")
        print()
        print(" Publishing the median of all rounds would give "
              f"{statistics.median(sorted(tps_cold + tps_res)):.1f} t/s, which is neither.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
