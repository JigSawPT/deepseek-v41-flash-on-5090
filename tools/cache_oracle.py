# -*- coding: utf-8 -*-
"""How much the cache POLICY still has to give, and how much is capacity.

The one question that separates "worth programming better" from "only more RAM helps": with the
same capacity and the same trace, how much does the optimal policy (MIN/Belady, which sees the
future) save over the one running today?

  - if MIN is close to LRU, policy is not the problem. It is capacity or I/O, and writing a
    predictor would be work without a prize.
  - if MIN is far below, there is a predictor worth writing, and the oracle says how much it is
    worth BEFORE it is written.

Belady is not implementable (it needs the future). It is an upper bound, not a proposal.

Policies:
  lru      the classic, what the expert tier does today
  lfu      frequency, to separate "recent" from "popular"
  heat     a weight that decays once per token
  min      Belady: evicts what is needed again latest
  random   baseline: any serious policy has to beat this

Trace: one JSON line per (token, layer) with the list of experts, in access ORDER; a histogram
does not serve, which is why this file exists. Fields: "token", "layer", "experts".

  python cache_oracle.py --trace results/trace.jsonl --capacities 1028,4112,5140
"""

import argparse
import json
import heapq
import random
from collections import defaultdict, OrderedDict

BYTES_PER_EXPERT = 18_800_640          # measured, uniform on this model
GIB = 1024 ** 3


def read_trace(path):
    """Returns (accesses, token_of_access).

    An access is (layer, expert): the same expert index in different layers is a different
    weight, so the cache key carries the layer. The token number comes from the trace itself;
    inferring it from the layer order broke on any trace where a layer does not route.
    """
    accesses, tokens = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            layer = d.get("layer")
            tk = d.get("token", 0)
            for e in d.get("experts", []):
                accesses.append((layer, e))
                tokens.append(tk)
    return accesses, tokens


def run_lru(accesses, cap):
    cache = OrderedDict()
    misses = 0
    for a in accesses:
        if a in cache:
            cache.move_to_end(a)
            continue
        misses += 1
        cache[a] = True
        if len(cache) > cap:
            cache.popitem(last=False)
    return misses


def run_lfu(accesses, cap):
    cache = {}
    count = defaultdict(int)
    misses = 0
    for a in accesses:
        count[a] += 1
        if a in cache:
            continue
        misses += 1
        cache[a] = True
        if len(cache) > cap:
            # evict the least frequent; arbitrary tie-break, enough for a bound
            victim = min(cache, key=lambda k: count[k])
            del cache[victim]
    return misses


def run_heat(accesses, cap, decay=0.97, per_token=None):
    """A weight that decays ONCE per token, not once per layer. Decaying per layer
    (0.97^40 = 0.296 per token) erased the profile in ten tokens."""
    cache = {}
    heat = defaultdict(float)
    misses = 0
    for i, a in enumerate(accesses):
        if per_token is not None and i > 0 and per_token[i] != per_token[i - 1]:
            for k in list(heat):
                heat[k] *= decay
        heat[a] += 1.0
        if a in cache:
            continue
        misses += 1
        cache[a] = True
        if len(cache) > cap:
            victim = min(cache, key=lambda k: heat[k])
            del cache[victim]
    return misses


def run_min(accesses, cap):
    """Belady. Evicts what is needed again latest -- or never."""
    n = len(accesses)
    next_use = [n] * n
    last = {}
    for i in range(n - 1, -1, -1):
        a = accesses[i]
        next_use[i] = last.get(a, n)
        last[a] = i

    cache = set()
    heap = []                 # keyed by -next_use: the top is what comes back latest
    valid_next = {}
    misses = 0
    for i, a in enumerate(accesses):
        if a in cache:
            valid_next[a] = next_use[i]
            heapq.heappush(heap, (-next_use[i], a))
            continue
        misses += 1
        if len(cache) >= cap:
            while heap:
                nnext, victim = heapq.heappop(heap)
                if victim in cache and valid_next.get(victim) == -nnext:
                    cache.discard(victim)
                    valid_next.pop(victim, None)
                    break
        cache.add(a)
        valid_next[a] = next_use[i]
        heapq.heappush(heap, (-next_use[i], a))
    return misses


def run_random(accesses, cap, seed=0):
    r = random.Random(seed)
    cache = []
    inside = set()
    misses = 0
    for a in accesses:
        if a in inside:
            continue
        misses += 1
        if len(cache) >= cap:
            j = r.randrange(len(cache))
            inside.discard(cache[j])
            cache[j] = a
        else:
            cache.append(a)
        inside.add(a)
    return misses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--capacities", default="1028,4112,5140",
                    help="in NUMBER OF EXPERTS: 18 GiB=1028, 72 GiB=4112, both=5140")
    ap.add_argument("--decay", type=float, default=0.97)
    a = ap.parse_args()

    accesses, per_token = read_trace(a.trace)
    if not accesses:
        print("empty trace")
        return 1
    distinct = len(set(accesses))
    print(f"trace: {len(accesses):,} accesses, {distinct:,} distinct experts")
    print(f"       {distinct * BYTES_PER_EXPERT / GIB:.1f} GiB of distinct experts touched")
    print(f"       {len(set(per_token)):,} tokens in the trace")

    print()
    print(f" {'capacity':>10s} {'GiB':>6s} | {'LRU':>8s} {'LFU':>8s} {'heat':>8s} "
          f"{'random':>8s} {'MIN':>8s} | {'LRU vs MIN':>11s}")
    print(" " + "-" * 78)
    for cap_s in a.capacities.split(","):
        cap = int(cap_s)
        lru = run_lru(accesses, cap)
        lfu = run_lfu(accesses, cap)
        heat = run_heat(accesses, cap, a.decay, per_token)
        rnd = run_random(accesses, cap)
        mn = run_min(accesses, cap)
        margin = (lru - mn) / max(lru, 1) * 100
        print(f" {cap:10d} {cap*BYTES_PER_EXPERT/GIB:6.1f} | {lru:8d} {lfu:8d} {heat:8d} "
              f"{rnd:8d} {mn:8d} | {margin:10.1f}%")

    print()
    print(" The right-hand column is the decision: how much a PERFECT policy would save over")
    print(" the one running today, at the same capacity. Below ~10 % no predictor pays for the")
    print(" work -- the deficit is in capacity or in I/O.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
