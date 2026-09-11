# -*- coding: utf-8 -*-
"""One-off: the archived result JSONs were written by the Portuguese-named tools. This renames
their keys and values to the English names the published tools use, and renames the files.
Kept in the repository so the mapping is on record; running it twice is a no-op."""
import json
from pathlib import Path

HERE = Path(__file__).parent

KEYS = {
    "modelo": "model", "modo": "mode", "novos": "n_new", "repeticoes": "rounds",
    "carga_inicio": "load_start", "carga_fim": "load_end", "comando": "command",
    "arranque_s": "startup_s", "resultados": "results", "voltas": "rounds", "pecas": "pieces",
    "tps_mediana": "tps_median", "tps_p10": "tps_worst", "ttft_mediana": "ttft_median",
    "etiqueta": "label", "casos": "cases", "fumo": "smoke", "gerados": "generated",
    "parede_s": "wall_s", "propostos": "drafted", "aceites": "accepted", "texto": "text",
    "erros_log": "log_errors", "texto_igual_a": "text_equal_to",
    "fria": "cold", "residente": "resident", "repete": "verbatim", "codigo": "code",
    "curto": "short", "ptpt": "prose_pt", "raciocinio": "reasoning",
}
VALUES = {"dspark": "draft", "controlo": "control"}

FILES = {
    "REGUA_PORTE_V41_engram.json":     "bench_server_baseline_morning.json",
    "REGUA_PORTE_V41_engram_v2.json":  "bench_server_baseline.json",
    "REGUA_PORTE_V41_DSPARK.json":     "bench_server_draft2.json",
    "DSPARK_V41_dspark.json":          "dspark_draft5.json",
    "DSPARK_V41_controlo.json":        "dspark_control.json",
    "DSPARK_V41_nmax2.json":           "dspark_draft2.json",
    "DSPARK_V41_nmax3.json":           "dspark_draft3.json",
    "DSPARK_V41_pmin0.5.json":         "dspark_draft2_pmin05.json",
    "DSPARK_V41_pmin0.8.json":         "dspark_draft2_pmin08.json",
    "DSPARK_V41_vram_cache13.json":    "dspark_draft2_vram_cache13.json",
    "DSPARK_V41_controlo_cache13.json": "dspark_control_cache13.json",
    "DSPARK_V41_split_fumo.json":      "split_smoke.json",
}


def rename(obj):
    if isinstance(obj, dict):
        return {KEYS.get(k, k): rename(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rename(v) for v in obj]
    if isinstance(obj, str) and obj in VALUES:
        return VALUES[obj]
    return obj


def main():
    for old, new in FILES.items():
        src = HERE / old
        if not src.exists():
            continue
        d = json.loads(src.read_text(encoding="utf-8"))
        d = rename(d)
        (HERE / new).write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
        src.unlink()
        print(f"  {old} -> {new}")


if __name__ == "__main__":
    main()
