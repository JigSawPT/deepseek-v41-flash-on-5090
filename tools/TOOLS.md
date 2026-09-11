# Tools

The measurement tools behind the numbers in the report. They are the subset of the house's
toolbox that produced published figures, rewritten in English with paths as arguments; the rest
(the Python probe of the reference implementation, the trace recorders, the ladder scripts) stays
private because it depends on the house's infrastructure. Where a tool needs an input that only
the probe can produce, it says so.

All tools are Python 3.11, standard library plus `numpy`; the two floor probes also need `torch`
and the `inference/` package of the released checkpoint. The server tools run the
`dsv41-porte` branch of llama.cpp on Windows; the disk tool is Windows-only.

| tool | measures | in the report |
|---|---|---|
| `bench_server.py` | the house ruler on a resident `llama-server`: four prompts, three rounds, TTFT from streaming, decode between first and last piece | 5.12 / 21.27 tokens/s; the ruler with the draft |
| `read_bench.py` | reads the ruler JSON separating cold (first round) from resident (later rounds) — the two are different quantities on this model | the cold/resident table |
| `bench_dspark.py` | the DSpark draft head: acceptance and throughput per case, cold and resident, with the control on the same target | the draft-length sweep, the p_min and VRAM checks |
| `stall_decomposition.py` | compute / PCIe / NVMe share of a remap call from the expert-tier ladder; validates against the resident case | 20 / 26 / 54 %; the 6.2 and 21.3 ceilings |
| `fp8_floor.py` | the round-trip correlation of one fp8 linear against its exact dequantized product, on a true activation | 0.999896, 1.45 % |
| `kv_floor.py` | the round trips the reference applies to its own state (fp8 window KV, fp4 latent, fp4 indexer) | 2.80 / 10.39 / 15.64 % |
| `routing_switch.py` | top-6 agreement per layer between the port's `ffn_moe_topk` dumps and the reference | 208/240 = 86.7 % |
| `compare_dumps.py` | sub-step by sub-step correlation of one block, port vs. reference; the first drop is the defect | the q-norm finding (181.019) |
| `cache_oracle.py` | LRU / LFU / heat / random / MIN(Belady) misses over a recorded routing trace at given capacities | policy margin 0 % at 72 GiB |
| `disk_physical_io.py` | logical vs. physical bytes read while a process runs (Windows counters) | 86.7 % of logical reads are physical |

`llama-logits` (in the fork, `tools/logits`) produces the dumps the comparison tools read:
`--dump <node>,<node>` writes any `cb()`-named graph node as `DSV4DMP1` files;
`--tokens-file` and `-ub` set the prompt and the batch size.

Result files in [`../results/`](../results/): the ruler JSONs (`bench_server_baseline.json`,
`bench_server_baseline_morning.json` — the earlier run whose "short" prompt was half-warm —
and `bench_server_draft2.json`), the DSpark runs (`dspark_draft{2,3,5}.json`, the p_min and VRAM
variants, the controls), the block-verification sweep (`block_verification_ubatch.tsv`), the split
smoke test, the JigSaw hot list, and `SHA256SUMS.txt` for the published GGUF files. The JSONs were
produced by the house's Portuguese-named tools and renamed to the field names used here by
`rename_fields.py`, which is kept in the directory so the mapping is on record.
