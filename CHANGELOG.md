# Changelog

## Builds

Every build, total tok/s on two RTX 3090s: Q4_K_M and Q8_0, 1 and 8 parallel requests, coding prompt, 512 tokens each,
greedy, `-d 4`. The newest build is the last row; what each one changed is under its version below.

| build | version | change | Q4_K_M 1 req | Q4_K_M 8 req | Q8_0 1 req | Q8_0 8 req |
|---|---|---|---|---|---|---|
| h | v0.2.0 | Q8_0 tensor-core rows up to 32 (baseline) | | | 142 | 561 |
| i | v0.2.0 | K-quants on dp4a, capped at 10 rows per cycle | 174 | 298 | | 560 |
| k | v0.2.0 | K-quant tensor-core rows (16/32 by slot count), swizzled activation layout | 174 | 433 | | 500 |
| l | v0.2.0 | K-quant weights in 16-row tiles | 163 | 506 | | 518 |
| m | v0.2.0 | Q6_K whole-block path; FFN gate/up 4 rows per step | 180 | 513 | | 506 |
| n | v0.2.0 | Q4_K/Q5_K whole-block path | 179 | 542 | | 509 |
| o | v0.2.0 | bf16 exchange between the cards, only rows in use cross PCIe, one fence per block | 183 | 585 | 146 | 561 |
| p | v0.2.0 | one GEMV instance per weight type (type checked once per phase) | 183 | 583 | 146 | 570 |
| q | v0.2.0 | int8 exchange with a scale per 32 values | 182 | 603 | 139 | 587 |
| r | v0.2.0 | 3 K-quant groups in flight in the residual GEMV: slower, reverted | 182 | 585 | 139 | 584 |
| s/t | v0.2.0 | profiler step record; faster code generation, confirmed by a back-to-back drift check | 186 | 624 | 141 | 595 |
| w | v0.2.0 | release store for the exchange flags without a full system fence | 190 | 627 | 144 | 599 |
| y | v0.2.0 | projections: the two warps of a pair split each weight slice | | 621 | | 606 |
| za | v0.2.0 | w + y | 190 | 625 | 144 | 610 |
| zc | v0.3.0 | shared paged KV pool, prefix cache, preemption | **190** | **622** | **143** | **610** |

## v0.3.2: no crash with sampling at 5+ parallel requests

- **Fixed an illegal memory access with sampling at 5 or more parallel requests**: with temperature above 0 and 17 to 24 draft rows in one cycle, the sampled draft head runs in two passes, and the second pass overwrote shared memory the first was still merging its top-k lists in. Garbage token ids then indexed the draft vocabulary out of bounds. A barrier between the passes fixes it; greedy decoding never took this path and is unchanged.
- **`bench/tool_bench.py`**: long-context tool-call accuracy at 8K to 190K tokens, to compare `--kv f16` and `--kv q8` case by case (`--compare`).

## v0.3.1: running requests keep their speed, request ids, relocatable builds

Server and build only; the kernel is build zc unchanged.

- **Prefill no longer stalls running requests** (`--prefill-share`, `--prefill-chunk`): while others decode, prompts are prefilled in chunks of 512 tokens (was 2048, and one chunk for every waiting prompt per cycle) for at most a quarter of the time, so running requests keep about three quarters of their speed. The shortest remaining prompt goes first. With nothing decoding, prompts still go in whole.
- **Shared prompts arriving together are prefilled once**: a request whose prompt shares a snapshot point (the end of the system message, the start of the last user message, the whole prompt) with a prompt still being prefilled waits for it, then starts from its snapshot instead of prefilling the same tokens again.
- **Request ids**: the client's `X-Request-Id` or else the response id tags every server log line of a request, is echoed in the `X-Request-Id` response header and shows per slot in `/status`.
- **`/status` `kv_pool`** adds `used_blocks` (held by running requests) and `updated`; it is read after every decode cycle.
- **Relocatable builds**: `make` copies libllama and its ggml libraries next to the binaries and links every one of them directly with an `$ORIGIN` rpath, so a build directory or the release tarball runs from any path on any machine without `LD_LIBRARY_PATH` (libllama's own rpath points at the machine it was built on and is no longer searched).

## v0.3.0: shared KV pool, prefix cache, preemption

All requests now take their KV from one paged pool instead of a fixed share per slot, conversations and shared system prompts are cached across requests, and a full pool pauses the newest request instead of failing. Same VRAM, same quality, 0.1 to 0.6 % decode tok/s.

### What changed

- **Paged KV** (`qwen38_mk.cu`, `batch.cuh`, `prefill.cuh`): KV lives in blocks of 64 tokens (fp16) or 128 (q8_0), block-major over all 17 KV layers, with a block table per slot. Attention tiles that straddle two blocks are read from both, so results are bit-identical to the contiguous layout (same KL divergence on both quants).
- **One pool, any split**: `--kv-mem` sets the pool per card (default: the memory the fixed split used). `-c` is now the maximum of one request, not a per-slot reservation. Q8_0 with fp16 KV and 8 slots holds 208,704 tokens in total (the fixed split: 8 x 24K), and one 117K-token request runs next to four 10K ones.
- **Prefix cache** (`kv_cache.h`): refcounted blocks with copy-on-write, a radix tree over 64-token blocks, and DeltaNet state snapshots at the end of the system prompt, the start of the last user message, the start of the generation prompt and the end of the prompt (exact positions, never inside drafts). Snapshots take at most a quarter of the pool; eviction is least recently used. A follow-up turn prefilled 57 of 76 tokens from cache: 146 instead of 268 ms.
- **Preemption** (`server/mk_server.py`): requests are admitted when their prompt fits. When the pool runs full, cached blocks are evicted first, then the most recently admitted request is paused with its tokens cached and resumed at the head of the queue; its stream only pauses. A request that alone outgrows the pool ends with `length`, and only that request.
- **Server**: `GET /status` adds `kv_pool`, `preempted` and per-slot preemption counts; usage reports `prompt_tokens_details.cached_tokens`; `--think-budget` stops exactly at the budget, also across a pause.
- **Fixes**: KV rows written by the draft rows past the context end (`KV_AHEAD` = 2 x drafts + 1 rows are now reserved); a thinking stop that overran the context failed the whole batch; a second engine in one process failed on already-enabled peer access.
- **Tests** (`tests/`): pool fuzz and adversarial scenarios against a content-checking fake device, and a scheduler simulation (816 preemptions, no hang, no lost token). On the GPUs: a debug build that checks the whole pool after every operation ran all server tests and 1,500 random engine operations without a fault.
- **`MK_HOST_PROF=1`**: host time per decode cycle (KV bookkeeping, row prep + launch, wait for the cards, rest).

### Cost

Two RTX 3090s, fp16 KV, `-d 4`, 512 tokens, mean of 3 runs alternated with the previous build:

| | previous | paged | |
|---|---|---|---|
| Q4_K_M, 1 request | 190.9 | 190.4 | -0.3 % |
| Q4_K_M, 8 requests | 625.7 | 621.9 | -0.6 % |
| Q8_0, 1 request | 143.5 | 143.2 | -0.2 % |
| Q8_0, 8 requests | 611.3 | 609.9 | -0.2 % |
| prefill, 40,634 tokens, Q4_K_M / Q8_0 | 1,596 / 1,565 | 1,593 / 1,563 | -0.2 / -0.1 % |

KV bookkeeping costs under 1 µs per cycle. What remains is about 55 µs per cycle in the kernel at 1 request (block-table lookups) and about 125 µs of host time per batched cycle; both are open.

Greedy output after a pause, or in a different batch, continues correctly but is not always token-identical to an uninterrupted run: the draft/verify path takes a slightly different numeric route (the previous build already differed between batched and solo runs).

## v0.2.0: batched decode, up to 8 requests in one cycle

Two RTX 3090s, 8 parallel requests, total tok/s: **Q4_K_M 298 to 625 (2.1x)**, Q8_0 561 to 610. One request: Q4_K_M 190 tok/s, Q8_0 144.

Builds h to za ([Builds](#builds)) were found and checked with the per-phase profiler (`MK_PROFILE=batch`): time per phase per cycle, block arrival at each barrier, a "min-composed" estimate that filters out other GPU users, and a per-step record of block 0 (activation loads, compute, reduce, send, peer wait). Each build ran the same sweep (Q4_K_M and Q8_0, 1 to 8 parallel requests, coding prompt, 512 tokens each, greedy, `-d 4`).

Time per cycle at 8 requests, Q4_K_M: 62.5 ms (k) to 42.7 ms (za). Build i was fast per cycle (33 ms) but carried at most 10 rows, so it could not batch more than about 2 requests' drafts.

### What moved the numbers

- **Tensor cores past 10 rows** (k to n): the dp4a GEMVs keep every row in registers and spill above 10 rows. int8 `mma.m16n8k32` straight from the GGUF blocks lifts the cap to 32 rows; tiling the weights and decoding whole Q4_K/Q5_K/Q6_K superblocks at once took Q4_K_M from 433 to 542.
- **Cheaper exchange** (o, p, q): the three phases that sum across the cards (attention output, DeltaNet output, FFN down) went from 24.8 ms (m) to 15.8 ms (t) per cycle on Q4_K_M. Partials now cross PCIe as int8 with a scale per 32 values, only for rows in use, with one system fence per block instead of one per item. Quality is unchanged: KL divergence against llama.cpp 0.00424 (int8) vs 0.00410 (fp32).
- **One code path per weight type** (p): the GEMV no longer switches on the type inside its inner loop. Net speed was flat at 8 requests because the residual phases gained what the projections lost; the projection loss is still open.
- **Measured, not guessed** (r): keeping three weight groups in flight looked like a latency fix but made the residual phases 10 to 14 µs slower per call, so it was reverted.
- **Lighter fence** (w): the flag that tells the other card a partial has arrived is now a release store (`st.release.sys`) without a sequentially consistent system fence before it. Time from written to sent per residual call: 28 to 24 µs.
- **Pairs split the projections** (y): the projections have fewer 16-row tiles than the grid has warps, so the two warps of a pair now split each slice and add through shared memory. Attention projections 2.49 to 2.34 ms per cycle, DeltaNet projections 8.68 to 8.30.
- **Tried and dropped**: a fence-free send in the style of NCCL's LL protocol (data and flag in one 8-byte store) was slower from 2 requests up; moving the DeltaNet alpha/beta rows into the projection loop cost more in register pressure than it saved.
- **Drift check** (s): q, s and t were rerun back to back to separate a real gain from GPU clock drift (q 603.6, s 622.5, t 623.2 tok/s).

### Per phase, Q4_K_M, 8 requests, ms per cycle

| phase | m | o | q | t | za |
|---|---|---|---|---|---|
| FFN down (sum across cards) | 14.54 | 12.46 | 10.66 | 10.09 | 9.94 |
| FFN gate/up | 8.92 | 8.14 | 8.37 | 8.18 | 8.15 |
| DeltaNet projections | 8.04 | 7.51 | 8.93 | 8.68 | 8.30 |
| DeltaNet output (sum across cards) | 7.81 | 5.69 | 4.52 | 4.41 | 4.24 |
| DeltaNet core | 3.81 | 3.83 | 3.91 | 3.92 | 3.83 |
| attention projections | 2.15 | 1.93 | 2.53 | 2.49 | 2.34 |
| attention output (sum across cards) | 2.43 | 1.82 | 1.39 | 1.34 | 1.28 |
| everything else | 4.80 | 4.61 | 4.59 | 4.54 | 4.46 |
| **cycle** | 52.49 | 45.99 | 44.89 | 43.64 | 42.52 |

### Next

Sending the partials still takes about 24 µs per residual call from written to sent, against a transfer floor of about 14 µs (184 KB per card per call at 13.2 GB/s), while waiting for the other card costs 1 to 2 µs. The projections still give work to fewer than half of the warps. Both are open.

### Also in this release

- Project name Geezuz; binaries `geezuz`, `geezuz_tp2`, `libgeezuz.so`, `libgeezuz_tp2.so`; prebuilt sm_86 tarball (`make release`).
- `mk_server.py --slots N`: scheduler thread, request queue, slot assignment by longest cached prefix, prefill between cycles, per-slot streams and `/status`.
- CLI `--slots N` with a per-slot comparison against a solo run; `bench/batch_bench.py` (server or llama-server `--parallel`); `bench/kq_mma_bench.cu`, `bench/q6_dp4a_bench.cu`.
- `MK_OPT`, `MK_BATCH_ROWS`, `MK_PROFILE=batch` (see the README's [Options](README.md#options)).
