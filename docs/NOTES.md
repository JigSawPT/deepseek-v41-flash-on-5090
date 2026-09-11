# Working notes

The long form of the report: what the port had to change, every experiment in the plan with its
result (the negative ones included), and the lessons kept for the next model. Written from the
day-by-day record; dates are 2026.

## 1. What is proven, with the proof

- **The MXFP4 conversion is a bijection.** `safetensors → GGUF MXFP4 blocks → safetensors` returns
  the same bytes on 480 of 480 blocks sampled across layers and experts. The routed experts in the
  GGUF are the released weights, not a requantization.
- **The engram hash in C++ matches the reference.** 384/384 indices on a probe that includes a dead
  token and the pad token; the compressed vocabulary rebuilt from the tokenizer has exactly the size
  the checkpoint declares (99 092), and the primes sum to the row counts of both tables.
- **The engram tables live on disk, not in RAM or VRAM.** Same command before and after the
  conversion with the tables: resident memory unchanged, the loader maps the tensors
  (`TENSOR_HOST`) and the host gathers 56 rows per token.
- **The engram is on and pulls in the right direction.** A probe compares the port's layer-1 state
  with the table zeroed and with the table live against the reference: live is closer.
- **The model loads and runs.** 1 012 tensors, 467.3 GiB of weights plus the tables, `--moe-stream`,
  exit 0. First light at 3.6 tokens/s, 5.1 after the fixes below.
- **The split loads.** Eleven shards from `llama-gguf-split`; the server loads the first, resolves
  the engram tables from their own shards, and answers identically to the monolithic file.

## 2. What the port had to change from V4

The branch inherits `deepseek4.cpp`. Nine things were different, and none of them raised an error:
shapes matched, the model loaded, the text was fluent.

1. **The fp8 scale block is 32×32, not 128×128.** The converter hard-coded 128; the
   `[:out_features]` slice kept every shape valid and every dequantized weight came out mis-scaled.
   Root cause of the first wrong generation. Now read from `quantization_config`.
2. **Compression ratios were compile-time constants** (`DSV4_CSA_RATIO = 4`); V4.1 uses 2 and 1
   per layer. Now runtime, per layer.
3. **The V4.1 compressor is a different module.** A ratio-1 compressor degenerates to
   `norm(wkv(x))`: no gate, and V4.1 has no `ape` tensor.
4. **The compressed caches are shared.** Four `kv_source_layers` produce them; every other layer
   reads the nearest preceding source. V4 gave every layer its own.
5. **The indexer keys are projected directly** (`attn.indexer.wk`, `k_norm`), not through a second
   compressor.
6. **The indexer does not use the Hadamard rotation.**
7. **The hyper-connection mix is threaded one sub-layer ahead.** Each sub-layer computes the
   coefficients the *next* one applies; the final collapse uses the mix the last FFN produced. V4
   computes in place and has dedicated `hc_head` weights. The reference's docstring is the proof;
   the A/B switch `LLAMA_DSV41_NO_HC_THREAD` measures the difference.
8. **The first level of the hierarchical indexer** (block candidates before positions) was not in
   the base; without it the context refused to open above 16 384 rather than being silently wrong.
   Ported; the candidate mask above 16 384 positions is untested.
9. **The engram had five silent defects** in the first pass (hash-state sizing, the mirrored index
   vector sized before the one it mirrors, history following the sequence rather than the batch,
   two in the table read path). All found by comparing dumps, none by a crash.

And the one found last, after everything ran: **V4 normalises each head of `q` after `wq_b`;
V4.1 does not.** Norm 181.019 = √(64 × 512) identified it. Gated on the architecture.

## 3. The plan and its results

The plan had two routes — A, the vendor's Python reference with a two-tier expert cache (the probe),
and B, the llama.cpp port — plus a verification block and nine hypotheses. Every item below has a
measurement; the ones that failed are kept because they close a door.

### Route A, the probe

- **A0.1 CUDA-core utilisation.** The batch-1 expert GEMM ran at 5.2 % of the card's bandwidth
  (200 µs per 18.8 MB expert against 10.5 µs at peak). This reordered the plan: the "20 tokens/s
  software ceiling" was a 19× inefficiency, and the llama.cpp `mul_mat_id` kernel, which delivers
  72 % at n = 1, was the route.
- **A1 twelve defects** in the probe's cache (heat decay 40× per token, a host round-trip per expert,
  an inclusive two-tier cache that did not add capacity, wrong chunk size for the NVMe queue, …).
  All exact. 3.37 → 4.34–4.67 tokens/s.
- **A0.5 sequential trace** of every expert access with request, completion and need times — the
  input for every oracle.
- **The loop costs more than the compute it orchestrates.** 6 669 kernel launches per token; the
  MoE loop alone bounded the probe at ~8 tokens/s.
- **Grouping `w1`/`w3` is exact and gives nothing** — the kernel was not the limit.
- **Layer-ahead routing prediction works** (`gate_{L+1}(h_L)`: recall@6 70.9 %, 58.6 % of misses
  covered) **and solves a problem the probe did not have**: its I/O wait was already hidden under
  compute. Archived, then un-archived for the port, where disk is 94 % of the token — and finally
  killed by the A4 curve (see below).
- **A2 more residency does not give throughput** on the probe: two independent experiments, +0 %
  and +0.5 %.
- **B1 the GGUF conversion is a bijection**, 480/480.
- **Non-reproducibility investigation.** Identical greedy runs gave different texts. Eleven
  suspects excluded with proof on the probe; the cause was found on the port (below).

### Route B, the port

- **First light**, then the nine fixes above, then the q-norm.
- **The expert-tier ladder** (host tier 0 → 88 GiB): optimum at 72 GiB; 88 is slower because it
  steals page cache from the engram tables.
- **The house ruler on the port:** 5.12 tokens/s cold, 21.27 resident. The resident number is the
  compute ceiling and the ruler's "discard the first request" rule would have published it as the
  conversation number.
- **Working set and eviction policy (A3, A0.3).** At 72 GiB, LRU, LFU, heat, random and MIN/Belady
  give the same misses: policy has 0 % of margin. 86.7 % of logical reads are physical. The working
  set grows 26 experts per token in the tail: 105 GiB at 128 tokens.
- **Where the missing 6 GB/s are.** The NVMe delivers 10.04 GB/s at queue depth 6; the port
  extracts 4.33; the I/O-thread sweep plateaus at 4 workers because only 3.78 disk requests exist
  per layer. The deficit is the graph's serialisation, not the queue.
- **H8 prefix cache** was already delivered by the server: 32× on time to first token.
- **A4 prefetch oracle.** A speculative queue fed by a recorded trace: +30 % at 200 remap calls
  (5 tokens) ahead, within 2 % of the no-disk ceiling the decomposition predicted. At 1, 2 and 4
  calls ahead: −19 / −7 / −16 %, because the speculative read is still in flight when the demand
  arrives. This kills every layer-level predictor (A5).
- **The token decomposed:** 20 % compute, 26 % PCIe, 54 % NVMe per remap call at 72 GiB; ceilings
  4.3 today, 6.2 without disk, 21.3 all-resident.
- **H4 engram prefetch:** 10.44 → 0.92 ms per token, 11.4×, exact.
- **CED (encoder-only prefill):** worth 23.9 % of prefill bytes, but not exact below 2 432 tokens
  (the receptive field of 19 sliding windows of 128), and the reference does not implement it.
  Closed.
- **The non-determinism above 1 024 tokens has a cause and a switch:** expert-cache slot assignment
  depends on I/O timing; one I/O thread makes runs bit-identical.
- **H1 block verification with a perfect draft:** +17–20 %, plateau from K = 4.
- **H2 n-gram draft:** null best case (+2 %), −33 % realistic; and it exposed the rollback gap
  (`need_n_rs_seq`) — fixed for every draft type.
- **H2 DSpark draft head:** exported (78 tensors, 7.97 GB), acceptance 51–97 % by content, neutral
  on the house ruler (−4 % cold, 0 % resident), +12–15 % on verbatim repetition only; best draft
  length 2, not the trained 5. Confidence threshold and draft experts in VRAM: no change.
- **The 502 GB GGUF splits into 11 shards and loads split.**

### Verification

Four separate requirements, all met: identity of weights (bijection), identity of routes (86.7 %,
with the margin explaining the rest), numerical comparison at the floor of the reference's own
arithmetic, and equivalence of distribution at long context (0.9967 vs. 0.9959 self).

## 4. Lessons kept for the next model

1. A batch-1 MoE kernel can be using 5 % of the card — and nothing shows it.
2. llama.cpp's `mul_mat_id` delivers 72 % of bandwidth at n = 1 — 13.8× a naive kernel.
3. A vendor's "activated parameters" promise FLOPs, not bytes. On a machine with offload the number
   that matters is distinct bytes read per token at your memory tier.
4. A flat top-k is not a sparse router; conflating them costs a plan.
5. An inclusive two-tier cache does not add capacity.
6. The pinned-memory allocator rounds to the next power of two above 8 GiB (1.79×);
   `cudaHostRegister` does not (1.01×).
7. Building a model on `device='meta'` loses information, silently.
8. Freezing a cache "during prefill" must key on sequence length, not batch rows.
9. NVMe queue depth is set by the chunk size, not the thread count.
10. Logical reads are not physical reads.
11. A KV cache of 890 B/token changes what residency means: context stops competing with experts.
12. A vendor's reference implementation is not its production engine.
13. A number inherited from an audit is a hypothesis, not a measurement.
14. A trace cited in a report must be reproducible, or the report has a debt.
15. An offload measurement carries ±8 % of noise nobody was counting.
16. A compiler warning about a race is a defect, not start-up noise.
17. Before hunting a reproducibility bug, prove each piece clean — and publish the list.
18. A blocking counter is not a critical-path counter.
19. A feasibility gate is not a value gate.
20. In a Python engine, measure the loop before optimising what it calls.
21. If the model is not reproducible, an A/B compares loads, not implementations.
22. Grouping calls and overlapping I/O are in tension.
23. A safety guard can be the bomb.
24. A tensor that undergoes no transformation does not have to pass through RAM.
25. Shapes that divide hide wrong scales.
26. A dispatch condition on an optional tensor turns subsystems off silently.
27. ggml broadcasts whenever dimensions divide.
28. The reference's docstring is proof; intuition is not.
29. Not every ggml op has a kernel on every backend.
30. In graph code, compiling proves nothing.
31. A cache policy can be null alone and positive in combination.
32. Converting a large model: the GGUF writer holds everything until the end.
33. A small allocation error points at the accumulated total, not the request.
34. A buffered log lies about where the crash was.
35. A workaround outlives the cause that motivated it — look for it after the fix.
36. An inherited architecture brings its sibling's assumptions, silently.
37. A probe that produces an identifiable constant finds the defect by itself (181.019).
38. Before calling a difference a defect, measure the floor.
39. In a MoE, the divergence between two implementations is not a continuous quantity.
40. A number that contradicts one already measured indicts the instrument first.
41. Repeating the prompt does not warm the engine — it warms the answer.
42. The warm-up cannot use a prompt from the list.
43. A patch that prints "success" does not prove the file compiles.
44. A hypothesis must not evaluate itself.
45. Two counters printed as a ratio must share a unit.
46. Look-ahead with overlap measures duplication, not depth.
47. "Cannot delay" has to be true at execution, not only at dispatch.
48. Five independent reviewers catching the same defect is the signal.
49. A draft pays by tokens accepted per step and costs by the expert union of the block; the draft
    length is a property of the machine.

## 5. Open items

- The candidate mask of the hierarchical indexer above 16 384 positions is implemented but
  untested against the reference.
- The A/B switch `LLAMA_DSV41_NO_HC_THREAD` stays in the fork as an instrument; it will not be part
  of an upstream PR.
- The speculative prefetch queue stays off by default; it needs a source of future tokens to pay.
- The 22-GiB VRAM cache point and the 88-GiB host-tier point were measured once each.
- Upstream's conversion PR (#28696) stores the engram differently; the two layouts need
  reconciling before a runtime PR.
