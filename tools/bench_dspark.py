# -*- coding: utf-8 -*-
"""The DSpark draft head on the port: one server, two prompts, cold and resident rounds.

The server of this branch does not accept speculative.* per request (server-schema.cpp, #if 0),
so the control is a second run of the same script with --no-draft: same target, same cache, same
context. Each case has two rounds, cold (empty expert cache) and resident. Throughput is the
server's own predicted_per_second; acceptance comes from draft_n / draft_n_accepted.
The draft's experts stay in RAM (--spec-draft-n-cpu-moe) so the target keeps its VRAM cache and
the control is exactly the target of the ruler.

  python bench_dspark.py --server <llama-server> --gguf <first shard> --draft <draft.gguf> --n-max 2
  python bench_dspark.py --server ... --gguf ... --no-draft --compare results/dspark_draft.json
"""

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(errors="replace")
sys.path.insert(0, str(Path(__file__).parent))
from bench_server import machine_load, wait_ready  # noqa: E402

CASES = ("verbatim", "code")
ROUNDS = ("cold", "resident")
ERROR = re.compile(r"inconsistent sequence positions|GGML_ASSERT|error|abort", re.I)

# Raw prompts in the model's own chat template (chat mode, no reasoning). "verbatim" is the
# mechanism's best case (the answer is in the prompt); "code" is a realistic one.
PROMPTS = {
    "verbatim": (
        "<｜User｜>Repeat the following paragraph exactly, word for word: \"A database index "
        "speeds up a query by letting the engine find rows without scanning the whole table. It does "
        "not help when the query touches most of the table, when the column has few distinct values, "
        "or when the index does not match the predicate.\"<｜Assistant｜></think>"
    ),
    "code": (
        "<｜User｜>Write a Python function that prints the numbers from 1 to 20, and explain "
        "how it works.<｜Assistant｜></think>"
    ),
}


def complete(port, prompt, n_new):
    body = {"prompt": prompt, "n_predict": n_new, "temperature": 0.0, "cache_prompt": False}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion",
                                 data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        j = json.loads(r.read().decode("utf-8", "replace"))
    t = j.get("timings", {})
    return {
        "tps": round(t.get("predicted_per_second", 0.0), 3),
        "generated": t.get("predicted_n"),
        "ttft_s": round(t.get("prompt_ms", 0.0) / 1000.0, 2),
        "wall_s": round(time.time() - t0, 2),
        "drafted": t.get("draft_n", 0),
        "accepted": t.get("draft_n_accepted", 0),
        "text": j.get("content", ""),
    }


def errors_in_log(log):
    try:
        return sum(1 for l in log.read_text(encoding="utf-8", errors="replace").splitlines() if ERROR.search(l))
    except FileNotFoundError:
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--draft", default="", help="the DSpark draft GGUF (required unless --no-draft)")
    ap.add_argument("--port", type=int, default=18412)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--cache", type=int, default=18)
    ap.add_argument("--l2", type=int, default=72)
    ap.add_argument("--n-max", type=int, default=2, help="draft length; the trained block is 5")
    ap.add_argument("--ncmoed", type=int, default=3, help="draft layers whose experts stay in RAM (0 = all in VRAM)")
    ap.add_argument("--p-min", type=float, default=-1.0, help="draft confidence threshold; -1 = default")
    ap.add_argument("--n-new", type=int, default=96)
    ap.add_argument("--no-draft", action="store_true", help="the control: the same server without the draft")
    ap.add_argument("--smoke-only", action="store_true")
    ap.add_argument("--compare", default="", help="JSON of the other run, to compare the texts")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    label = "control" if a.no_draft else "draft"
    out = Path(a.out) if a.out else Path("results") / f"dspark_{label}.json"
    log = out.with_suffix(".server.log")
    out.parent.mkdir(parents=True, exist_ok=True)

    cmd = [a.server, "-m", a.gguf, "-ngl", "99", "-c", str(a.ctx),
           "--moe-stream", "--moe-stream-cache", str(a.cache), "--moe-stream-l2", str(a.l2),
           "--host", "127.0.0.1", "--port", str(a.port), "--no-webui", "--reasoning", "off"]
    if not a.no_draft:
        if not a.draft:
            ap.error("--draft is required unless --no-draft")
        cmd += ["-md", a.draft, "--spec-type", "draft-dspark", "--spec-draft-n-max", str(a.n_max),
                "-ngld", "99", "--spec-draft-n-cpu-moe", str(a.ncmoed)]
        if a.p_min >= 0:
            cmd += ["--spec-draft-p-min", str(a.p_min)]
    rec = {"label": label, "command": " ".join(cmd), "load_start": machine_load(), "cases": {}}
    print(f"{label}: " + " ".join(cmd[1:]))
    fh = open(log, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
    try:
        rec["startup_s"] = round(wait_ready(a.port, 2400, proc), 1)
        print(f"  ready in {rec['startup_s']} s; errors in the log so far: {errors_in_log(log)}")

        smoke = complete(a.port, "<｜User｜>Count to three.<｜Assistant｜></think>", 16)
        print(f"  smoke: {smoke['generated']} tokens, {smoke['tps']} t/s, drafted {smoke['drafted']}, "
              f"accepted {smoke['accepted']}, errors in the log {errors_in_log(log)}")
        print("  text: " + smoke["text"][:160].replace("\n", " | "))
        rec["smoke"] = smoke
        if a.smoke_only:
            return

        print(f"\n  {'case':9s} {'round':9s} {'t/s':>7s} {'ttft s':>7s} {'drafted':>8s} {'accepted':>8s} {'accept.':>8s} {'errors':>6s}")
        for case in CASES:
            rounds = {}
            for name in ROUNDS:
                r = complete(a.port, PROMPTS[case], a.n_new)
                r["log_errors"] = errors_in_log(log)
                rounds[name] = r
                acc = f"{100.0 * r['accepted'] / r['drafted']:.1f} %" if r["drafted"] else "-"
                print(f"  {case:9s} {name:9s} {r['tps']:7.2f} {r['ttft_s']:7.2f} {r['drafted']:8d} "
                      f"{r['accepted']:8d} {acc:>8s} {r['log_errors']:6d}")
            rec["cases"][case] = rounds

        if a.compare and Path(a.compare).exists():
            other = json.loads(Path(a.compare).read_text(encoding="utf-8"))
            for case in CASES:
                mine = rec["cases"][case]["resident"]["text"]
                theirs = other.get("cases", {}).get(case, {}).get("resident", {}).get("text")
                print(f"  {case:9s} resident text == {other.get('label')}: {mine == theirs}")
                rec["cases"][case]["text_equal_to"] = {other.get("label"): mine == theirs}
    finally:
        proc.terminate()
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            proc.kill()
        fh.close()
        rec["load_end"] = machine_load()
        out.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n  written: {out}")


if __name__ == "__main__":
    main()
