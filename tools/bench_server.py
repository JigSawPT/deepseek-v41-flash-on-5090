# -*- coding: utf-8 -*-
"""The house ruler applied to the llama.cpp port: four prompts, several rounds, one resident server.

Protocol:
  - the model loads ONCE and stays resident; a run must not pay the start-up, or it measures
    the disk and not the decode;
  - the first request after start-up is thrown away, with a prompt that is NOT in the list
    (using the first listed prompt left it half-warm: prefill cached, part of the experts
    resident, and something else was being measured);
  - several rounds per prompt; median and worst are reported, never the best;
  - the machine's load is recorded at the start and at the end;
  - the reasoning mode is recorded, otherwise the number is not comparable.

TTFT comes from streaming (the instant of the first piece); decode is the time between the first
piece and the last. The wall clock is not used, it mixes the two.

Read the JSON with read_bench.py: on this model the first round of every prompt is the "cold"
number (content seen for the first time) and the later rounds are "resident" (the same tokens
touch the same experts, already in the cache). They are two different quantities.

  python bench_server.py --server <llama-server.exe> --gguf <first shard> --l2 72
  python bench_server.py ... --extra "-md draft.gguf --spec-type draft-dspark --spec-draft-n-max 2 --spec-draft-n-cpu-moe 3"
"""

import argparse
import json
import os
import shlex
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# The four prompts of the report, verbatim: they are the measured ones, and a ruler that measures
# different prompts compares nothing. Three are Portuguese by design (the model is served to
# Portuguese users); "ptpt" tests European-Portuguese prose specifically.
PROMPTS = [
    ("short",      "Explica em duas frases o que e a entropia em termodinamica."),
    ("code",       "Escreve uma funcao Python que recebe uma lista de inteiros e devolve o segundo maior valor distinto. Explica a complexidade."),
    ("prose_pt",   "Escreve um paragrafo em portugues de Portugal sobre a importancia da agua nos ecossistemas de montanha. Nao uses portugues do Brasil."),
    ("reasoning",  "Tres caixas: uma so com macas, uma so com laranjas, uma com ambas. Todas as etiquetas estao trocadas. Quantas frutas precisas de tirar, e de que caixa, para corrigir todas as etiquetas?"),
]


def machine_load():
    """Heavy processes that are not ours: what invalidates a speed measurement. Windows only;
    elsewhere it returns a note and the run goes on."""
    if os.name != "nt":
        return "(load not sampled on this OS)"
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-Process | Where-Object { $_.WorkingSet64 -gt 1GB -or $_.CPU -gt 60 } | "
             "Select-Object -First 8 ProcessName, @{n='RAM_GB';e={[math]::Round($_.WorkingSet64/1GB,1)}}, "
             "@{n='CPU_s';e={[math]::Round($_.CPU)}} | ConvertTo-Json -Compress"],
            capture_output=True, text=True, timeout=30)
        return out.stdout.strip()[:400]
    except Exception as e:  # noqa: BLE001
        return f"(could not read: {e})"


def wait_ready(port, deadline_s, proc):
    """/health answers 200 only once the model is loaded."""
    url = f"http://127.0.0.1:{port}/health"
    start = time.time()
    while time.time() - start < deadline_s:
        if proc.poll() is not None:
            raise RuntimeError(f"the server exited with code {proc.returncode} before it was ready")
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return time.time() - start
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2)
    raise TimeoutError(f"the server was not ready within {deadline_s} s")


def request(port, text, n_new, temp):
    """Returns (ttft_s, decode_s, n_pieces, text). Streaming, so the TTFT is real."""
    body = {
        "messages": [{"role": "user", "content": text}],
        "max_tokens": n_new,
        "temperature": temp,
        "stream": True,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    ttft = None
    last = t0
    pieces = 0
    out = []
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                j = json.loads(data)
            except Exception:  # noqa: BLE001
                continue
            delta = (j.get("choices") or [{}])[0].get("delta") or {}
            piece = delta.get("content") or delta.get("reasoning_content") or ""
            if not piece:
                continue
            now = time.time()
            if ttft is None:
                ttft = now - t0
            last = now
            pieces += 1
            out.append(piece)
    if ttft is None:
        return None
    # decode is the interval between the first piece and the last: n-1 intervals
    decode = max(last - t0 - ttft, 1e-9)
    return ttft, decode, pieces, "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True, help="path to llama-server(.exe) from the dsv41-porte branch")
    ap.add_argument("--gguf", required=True, help="the target GGUF (first shard of a split file)")
    ap.add_argument("--port", type=int, default=18411)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--cache", type=int, default=18, help="VRAM expert cache, GiB")
    ap.add_argument("--l2", type=int, default=72, help="pinned host tier, GiB")
    ap.add_argument("--pin", type=int, default=0, help="experts pinned by JigSaw (needs --hotlist)")
    ap.add_argument("--hotlist", default="", help="hot list for --pin")
    ap.add_argument("--n-new", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--mode", default="chat", choices=["chat", "thinking"])
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--start-deadline", type=int, default=2400)
    ap.add_argument("--out", default="results/bench_server.json")
    ap.add_argument("--server-log", default="results/bench_server.server.log")
    ap.add_argument("--extra", default="", help="extra server arguments, one string (e.g. the draft)")
    a = ap.parse_args()

    env = dict(os.environ)
    if a.pin > 0:
        env["LLAMA_MOE_STREAM_PIN_LIST"] = a.hotlist
        env["LLAMA_MOE_STREAM_PIN_N"] = str(a.pin)
    else:
        env.pop("LLAMA_MOE_STREAM_PIN_LIST", None)
        env.pop("LLAMA_MOE_STREAM_PIN_N", None)

    cmd = [a.server, "-m", a.gguf, "-ngl", "99", "-c", str(a.ctx),
           "--moe-stream", "--moe-stream-cache", str(a.cache), "--moe-stream-l2", str(a.l2),
           "--host", "127.0.0.1", "--port", str(a.port), "--no-webui",
           "--reasoning", "on" if a.mode == "thinking" else "off"]
    if a.extra:
        cmd += shlex.split(a.extra)

    rec = {
        "model": Path(a.gguf).name,
        "l2_gib": a.l2, "cache_gib": a.cache, "pin": a.pin,
        "mode": a.mode, "n_new": a.n_new, "rounds": a.rounds,
        "temp": a.temp, "ctx": a.ctx, "extra": a.extra,
        "load_start": machine_load(),
        "command": " ".join(cmd),
    }
    print(f"ruler on the port: {rec['model']}")
    print(f"  L2={a.l2} GiB  cache={a.cache} GiB  pin={a.pin}  mode={a.mode}  n_new={a.n_new}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    log = open(a.server_log, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    try:
        t_load = wait_ready(a.port, a.start_deadline, proc)
        rec["startup_s"] = round(t_load, 1)
        print(f"  server ready in {t_load:.1f} s")

        cold = request(a.port, "Count to three.", 16, a.temp)
        print(f"  warm-up request discarded ({cold[2] if cold else 0} pieces)")

        results = {}
        for name, text in PROMPTS:
            rounds = []
            for v in range(a.rounds):
                r = request(a.port, text, a.n_new, a.temp)
                if r is None:
                    print(f"  {name} r{v+1}: no answer")
                    continue
                ttft, decode, pieces, _ = r
                tps = (pieces - 1) / decode if pieces > 1 else 0.0
                rounds.append({"ttft_s": round(ttft, 3), "decode_s": round(decode, 3),
                               "pieces": pieces, "tps": round(tps, 2)})
                print(f"  {name:11s} r{v+1}: {tps:5.2f} t/s   TTFT {ttft:6.2f} s   {pieces} pieces")
            if rounds:
                tps = sorted(x["tps"] for x in rounds)
                tt = sorted(x["ttft_s"] for x in rounds)
                results[name] = {
                    "rounds": rounds,
                    "tps_median": round(statistics.median(tps), 2),
                    "tps_worst": round(tps[0], 2),
                    "ttft_median": round(statistics.median(tt), 2),
                }
        rec["results"] = results
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()

    rec["load_end"] = machine_load()
    Path(a.out).write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print(f" {'prompt':12s} {'median':>9s} {'worst':>7s} {'TTFT':>8s}")
    print(" " + "-" * 40)
    for name, r in rec.get("results", {}).items():
        print(f" {name:12s} {r['tps_median']:9.2f} {r['tps_worst']:7.2f} {r['ttft_median']:8.2f}")
    print(f"\nwritten to {a.out}; read it with read_bench.py, which separates cold from resident")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
