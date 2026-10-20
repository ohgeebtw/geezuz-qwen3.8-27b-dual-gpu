# Geezuz

**Geezuz, it's fast.** Qwen3.8-27B on two GPUs in one CUDA megakernel.

| two RTX 3090s power-limited to 280 W (stock 350 W), coding prompt, greedy | Q4_K_M | Q8_0 |
|---|---|---|
| 1 request | **190 tok/s** | 143 tok/s |
| 8 parallel requests (total) | **622 tok/s** | 610 tok/s |

- 1.64x llama.cpp's fastest two-GPU setup, same output quality ([Results](#results), [Quality](#quality)).
- Up to 8 requests decoded together in one cycle, sharing one KV pool with a prefix cache ([Parallel requests](#parallel-requests)).
- Shared KV pool: one long request next to several short ones, cached system prompts and conversations, the newest request pauses when the pool is full ([Shared KV pool](#shared-kv-pool-and-prefix-cache)).
- Plain PCIe, no NVLink. Q4_K / Q5_K / Q6_K / Q8_0 weights in any per-tensor mix.
- OpenAI-compatible server.

Built on L-Forster's single-GPU megakernel from [open-jet](https://github.com/L-Forster/open-jet) ([credits](#license-and-credits)). How it got this fast, build by build: [CHANGELOG](CHANGELOG.md).

## Contents

- [Quick start](#quick-start)
- [Results](#results): [one request](#one-request-vs-llamacpp), [long context](#long-context), [parallel requests](#parallel-requests), [one GPU](#one-gpu)
- [Quality](#quality)
- [How it works](#how-it-works): [two GPUs](#two-gpus), [batched decode](#batched-decode), [the megakernel](#the-megakernel), [prefill](#prefill)
- [Server](#server), [Options](#options), [Supported GGUF files](#supported-gguf-files)
- [FAQ](#faq), [Limitations](#limitations), [Files](#files), [License and credits](#license-and-credits)

## Quick start

### Prebuilt (RTX 30)

[Releases](https://github.com/ohgeebtw/geezuz-qwen3.8-27b-dual-gpu/releases) has `geezuz-v0.3.2-sm_86.tar.gz` for RTX 30 cards (sm_86): the one-GPU and two-GPU CLI and server libraries, libllama for the tokenizer, the server and the draft vocabulary. It needs Linux x86-64 with glibc 2.35+ (Ubuntu 22.04 or newer), NVIDIA driver 575.57+, Python 3 with `jinja2` for the server, and P2P for two GPUs (see [Two GPUs](#two-gpus)).

```sh
sha256sum -c geezuz-v0.3.2-sm_86.tar.gz.sha256
tar xzf geezuz-v0.3.2-sm_86.tar.gz && cd geezuz-v0.3.2-sm_86
MK_DEVICES=0,1 ./geezuz_tp2 -m Qwen3.8-27B-Q4_K_M.gguf -p "Write quicksort in C" -n 512 -d 4 --draft-vocab draft_vocab_32k.bin
python3 server/mk_server.py -m Qwen3.8-27B-Q4_K_M.gguf --lib ./libgeezuz_tp2.so -c 16384 --slots 8
```

### From source

You need:

- CUDA 12+ (12.8+ for RTX 50 series)
- one or two 24 GB GPUs: sm_86 (RTX 30), sm_89 (RTX 40) or sm_120 (RTX 50)
- for two GPUs: PCIe P2P between them (see [Two GPUs](#two-gpus))
- the Qwen3.8-27B GGUF (Q4_K_M; Q8_0 needs two cards)
- llama.cpp next to this directory, only for the tokenizer (a CPU-only build is enough)
- for the two-GPU build: about 55 min of compile time and up to 60 GB of host RAM

```sh
# llama.cpp (tag b10246) as ../llama.cpp, CPU-only shared libllama
git clone --depth 1 --branch b10246 https://github.com/ggml-org/llama.cpp ../llama.cpp
cmake -S ../llama.cpp -B ../llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=OFF
cmake --build ../llama.cpp/build -j

make   # one-GPU and two-GPU CLI + server libraries, tools; ARCH=sm_89 overrides the detected GPU
       # libllama and its ggml libraries are copied next to them: the directory runs from any path or machine

# one GPU
./geezuz -m ../models/Qwen3.8-27B-Q4_K_M.gguf -p "Write quicksort in C" -n 512 -d 4 --draft-vocab draft_vocab_32k.bin
# two GPUs (devices 0,1 by default)
MK_DEVICES=0,1 ./geezuz_tp2 -m ../models/Qwen3.8-27B-Q4_K_M.gguf -p "Write quicksort in C" -n 512 -d 4 --draft-vocab draft_vocab_32k.bin
# OpenAI-compatible server on two GPUs
python3 server/mk_server.py -m ../models/Qwen3.8-27B-Q4_K_M.gguf --lib ./libgeezuz_tp2.so -c 100000
# the same, up to 8 requests at once
python3 server/mk_server.py -m ../models/Qwen3.8-27B-Q4_K_M.gguf --lib ./libgeezuz_tp2.so -c 16384 --slots 8
```

`make release` builds the tarball in `dist/` (build llama.cpp with `-DGGML_NATIVE=OFF` and pass `LLAMA_BIN=<its bin dir>`; `docker/build.Dockerfile` with `--build-arg UBUNTU=22.04` gives the glibc 2.35 baseline). To build on a remote GPU host inside a CUDA container: `docker/build.Dockerfile`, `scripts/box-sync.sh <ssh-host> <dir>`, `scripts/box-build.sh <ssh-host> <dir>`.

## Results

Setup: two RTX 3090s (24 GB, power-limited to 280 W instead of 350 W), each on PCIe 4.0 x8 through the CPU, no NVLink; Ryzen 7 5800X; driver 595.84 (open, P2P patch); CUDA 12.9; unsloth's Qwen3.8-27B GGUF; greedy decoding.

### One request vs llama.cpp

4 chat prompts, 512 tokens each, best of 2 runs, tok/s. llama.cpp b10246 with CUDA: `llama-server --flash-attn on -ctk q8_0 -ctv q8_0 -ngl 99 --spec-type draft-mtp --spec-draft-n-max N -sm <layer|tensor>`. Geezuz: `-d 4`, fp16 KV. Script: `bench/tp2_bench.py`.

| prompt | llama.cpp layer split, MTP 2 | llama.cpp layer split, MTP 4 | llama.cpp tensor split, MTP 4 | Geezuz |
|---|---|---|---|---|
| Python function (code) | 84.5 | 91.9 | 128.0 | **213.9** (1.67x) |
| hash map explanation | 75.5 | 74.7 | 102.8 | **168.2** (1.64x) |
| short story | 61.6 | 49.0 | 65.0 | **109.2** (1.68x) |
| Flask to FastAPI plan | 74.6 | 75.6 | 102.9 | **164.8** (1.60x) |
| **mean** | 74.1 | 72.8 | 99.7 | **164.0** (1.64x) |

Speed-ups are against llama.cpp's fastest setup (tensor split, 4 drafts). Speed depends on how predictable the text is: code accepts the most drafts, prose the fewest. This table was measured before batched decode and not rerun.

Drafts accepted per prompt:

| prompt | llama.cpp: drafted tokens accepted | Geezuz: tokens per cycle (max 5) |
|---|---|---|
| Python function | 93.9% | 4.46 |
| hash map | 69.7% | 3.43 |
| short story | 34.0% | 2.16 |
| Flask plan | 69.8% | 3.36 |

Both engines draft with the model's own MTP head. Geezuz's lead does not come from accepting more drafts: checking 5 tokens costs it almost the same as checking 1, because the weights are read once per pass. Without drafts both engines are near the memory-bandwidth limit (one GPU: llama.cpp 40.4 tok/s, Geezuz 42.9).

### Long context

A source file in the prompt, then a coding task (256 tokens), tok/s. llama.cpp prompt processing is its best setup (layer split, 2 drafts).

| context (tokens) | decode, llama.cpp | decode, Geezuz | prompt processing, llama.cpp | prompt processing, Geezuz |
|---|---|---|---|---|
| 3,599 | 83.4 | **133.2** (1.60x) | 1,470 | **1,872** (1.27x) |
| 15,751 | 80.2 | **136.3** (1.70x) | 1,623 | **1,787** (1.10x) |
| 29,586 | 76.9 | **128.8** (1.67x) | 1,549 | **1,662** (1.07x) |
| 56,017 | 66.6 | **105.4** (1.58x) | 1,385 | **1,457** (1.05x) |

The decode lead holds at long context. The prompt-processing lead shrinks as attention over the long prompt takes over.

### Parallel requests

CLI `--slots N`: N variants of one coding prompt, 512 tokens each, `-d 4`, fp16 KV, `-c 16384`, one run each.

| parallel requests | Q4_K_M total tok/s | Q4_K_M ms per cycle | Q8_0 total tok/s | Q8_0 ms per cycle |
|---|---|---|---|---|
| 1 | 190 | 19.6 | 144 | 24.9 |
| 2 | 266 | 27.3 | 233 | 30.3 |
| 3 | 370 | 29.4 | 332 | 32.3 |
| 4 | 442 | 30.2 | 407 | 33.2 |
| 5 | 484 | 37.5 | 482 | 38.4 |
| 6 | 526 | 41.1 | 515 | 42.2 |
| 7 | 574 | 43.0 | 556 | 44.4 |
| 8 | 625 | 42.7 | 610 | 43.8 |

Total tok/s is the sum over all requests; at 8 requests each runs at about 75 to 78 tok/s. A cycle serves every request at once, so ms per cycle is also the gap between two token bursts of one stream. `bench/batch_bench.py` runs the same sweep through the server at 10K and 60K context (`--slots 1,2,3,4,6,8 --ctx 10000,60000`; `--engine llama` for llama-server `--parallel N`).

### One GPU

Upstream's measurements on one RTX 3090 against `llama-server` with open-jet's flags (MTP 2 drafts). This fork's one-GPU build runs at the same speed (105.7 vs 106.7 tok/s mean on the 4 prompts, measured back to back on the setup above).

| workload | llama.cpp | megakernel | |
|---|---|---|---|
| greedy, 4 chat prompts x 512 tokens (mean) | 63.1 | 106.7 | 1.69x |
| greedy, Python function (code) | 72.5 | 139.7 | 1.93x |
| reasoning mode through the HTTP servers (thinking on, temp 1.0, top_p 0.95, top_k 20; 5 coding prompts x 2 seeds, up to 1,024 tokens) | 50.9 | 78.0 | 1.53x |
| decode with 3.6K tokens of code in context | 56.3 | 95.3 | 1.69x |
| decode with 29.5K tokens of code in context | 52.3 | 72.7 | 1.39x |
| prompt processing, 1.8K to 16K tokens | 1,050 to 1,140 | 1,466 to 1,595 | ~1.4x |
| pasting 2.7K / 12K new tokens into a running conversation | 2.41 / 10.45 s | 1.68 / 8.10 s | |

The context rows predate two later changes (the sequence-token draft vocabulary and double-buffered attention), which add 1-8% decode speed there. With the q8_0 KV cache (picked automatically above ~80K tokens on one card) decode is ~1-2% and prefill 2-7% slower than with fp16. Scripts: `bench/code_bench.py`, `bench/think_bench.py`, `bench/bench.py`.

## Quality

The output is not byte-identical to llama.cpp, and the two-GPU output is not byte-identical to the one-GPU output: the math is ordered and rounded differently, so when two tokens are almost tied the other one can win. To check that this costs no quality, each build is scored against llama.cpp's next-token distributions: top-token agreement, mean KL divergence and perplexity.

**Geezuz vs upstream.** Reference: llama.cpp b10246 batched on the CPU, q8_0 KV, 4,092 positions of this repository's code and README, each with at least 1,024 tokens of context (`--score-path mixed --kv q8`).

| engine | top token agrees | KL divergence (mean, nats) | perplexity (reference 4.1372) |
|---|---|---|---|
| upstream, one GPU | 96.75% | 0.00406 | 4.1356 |
| Geezuz, one GPU | 96.75% | 0.00406 | 4.1356 |
| Geezuz, two GPUs | 96.75% | 0.00413 | 4.1313 |

The one-GPU build matches upstream to the last digit. The two-GPU build disagrees at the same 133 positions with the same perplexity within noise: two cards cost no quality. (Agreement is lower than in upstream's study below because this reference ran on the CPU, which rounds differently; what matters is the comparison between the rows.)

**All-reduce format** (two GPUs, same reference; Q8_0 is scored against the Q4_K_M reference, so its rows include the gap between the quants):

| model | partials sent between the cards | top token agrees | KL divergence (mean, nats) | perplexity (reference 4.1372) |
|---|---|---|---|---|
| Q4_K_M | fp32 | 96.750% | 0.00410 | 4.1314 |
| Q4_K_M | bf16 (`MK_OPT=0`) | 96.799% | 0.00424 | 4.1307 |
| Q4_K_M | int8 with a scale per 32 values (default) | 96.945% | 0.00424 | 4.1337 |
| Q8_0 | fp32 | 94.208% | 0.0154 | 4.1251 |
| Q8_0 | bf16 (`MK_OPT=0`) | 94.453% | 0.0155 | 4.1216 |
| Q8_0 | int8 with a scale per 32 values (default) | 94.355% | 0.0156 | 4.1234 |

All three formats are within run-to-run noise. A request decoded next to others can differ from the same request decoded alone (rows are packed and rounded differently), the same kind of near-tie flip as between `-d` settings.

**Upstream's study against llama.cpp generating token by token** (their tooling, `tools/llama_ref` and `--kld`; one GPU; 13,299 positions of code and prose, each after a 1,024-token prompt; reference: llama.cpp on the GPU, q8_0 KV):

| against llama.cpp generating token by token | top token agrees | KL divergence (mean, nats) | perplexity (reference 2.4263) |
|---|---|---|---|
| llama.cpp, batched instead | 98.68% | 0.0019 | 2.4283 |
| llama.cpp, fp16 KV cache instead | 99.17% | 0.0007 | 2.4284 |
| megakernel (prefill kernels, then the decode kernel), q8_0 KV | 99.08% | 0.0009 | 2.4274 |
| megakernel, fp16 KV | 99.11% | 0.0008 | 2.4273 |
| megakernel, prefill kernels only, q8_0 KV | 99.01% | 0.0009 | 2.4275 |

The megakernel is as close to llama.cpp as llama.cpp's own setups are to each other; its log-loss differs by +0.0005 +- 0.0004 (not significant). Most disagreements are near-ties (median lead of llama.cpp's token 0.03-0.04 nats); in 2-3 of ~120 llama.cpp's token led by more than 0.5 nats, as in llama.cpp against itself (4 of 176). Long context, one sequence of code each:

- 30K (1,999 positions after 28,000 tokens): 99.65% agreement, KL 0.0007 (fp16 KV: 99.55%, 0.0007; llama.cpp batched: 99.35%, 0.0009), log-loss +0.0018 +- 0.001 (llama.cpp batched +0.0017).
- 96K (1,999 positions after 94,000 tokens, q8_0 KV): 99.60%, KL 0.0008, log-loss +0.0012 +- 0.001 (llama.cpp batched: 99.45%, 0.0014, +0.0015 +- 0.002).

Speculative decoding is exact: greedy output is the main model's greedy choice, and with sampling the token distribution is the same as without drafts (2,000-sample test per position). Checking 1 vs 4 drafts per pass rounds slightly differently, so near-ties can differ between `-d` settings.

To reproduce:

```sh
make tools/llama_ref
tools/llama_ref -m model.gguf -f code.txt -f prose.txt --score-from 1024 -b 1 --ref-out llama.ref
tools/llama_ref -m model.gguf --kld llama.ref                                 # llama.cpp batched: the noise floor
./geezuz     -m model.gguf --kld llama.ref --score-path mixed --kv q8      # one GPU
./geezuz_tp2 -m model.gguf --kld llama.ref --score-path mixed --kv q8      # two GPUs
```

`llama_ref` teacher-forces 2,048-token chunks of the text (`--chunk`; `-b 1` for token by token; `--max-chunks` to limit) and stores the 256 most likely tokens per scored position (`logit_stats.h`). `--score-path`: `prefill` (default, the batched prefill kernels), `decode` (the decode kernel), or `mixed` (prefill for the unscored prompt, decode for the scored positions, as a generation runs).

## How it works

### Two GPUs

`geezuz_tp2` / `libgeezuz_tp2.so` (`-DMK_TP=2`) split the model over two identical GPUs (tensor parallel). Each card runs its own copy of the megakernel on half of the weights. The cards send partial sums straight to each other over PCIe P2P, with a flag per item (no shared grid barrier), and add them in the same order, so both cards keep a bit-identical copy of the residual stream. Prefill runs one host thread per card with the same exchange.

Each card holds its attention q / KV heads, its DeltaNet value heads (card c owns heads h with `(h % 16) / 8 == c`), half of the FFN rows (8,704, padded to 9,216), half of the lm_head vocabulary (top-k / argmax merged across cards), its K half of the MTP `eh_proj` and half of the MTP draft vocabulary. Embeddings and norms are on both. Weights take 8.7 GB per card instead of 16.7 GB on one, which leaves room for long contexts with an fp16 KV cache.

**P2P is required**; the engine refuses to start without it. GeForce cards have P2P disabled in NVIDIA's stock driver. The results here use the 595.84 open kernel module with the community P2P patch (tinygrad's, packaged in aikitoria/open-gpu-kernel-modules), Resizable BAR on and `iommu=pt`. Check with `nvidia-smi topo -p2p r` (both directions `OK`). Workstation and data-center cards have P2P with the stock driver.

Two-GPU tuning, each step measured with `MK_PROFILE=1`:

- **Overlapped exchange**: the projections that end in a sum across cards (attention output, DeltaNet output, FFN down) compute all their items, park them in shared memory and send them, then wait and add, instead of waiting item by item.
- **Warp-pair FFN gate/up**: two warps share one activation load per 16-row item, cutting that phase from 102.5 to 84.5 µs.
- **Balanced tails**: when rows don't divide evenly over the 1,312 warps, the last round is spread one row per warp.
- **Cross-phase L2 prefetch**: before each barrier, blocks prefetch the first weight rows of the next phase.
- **Split draft vocabulary**: each card scores half of the MTP draft rows, cutting the MTP lm_head from 220 to 132 µs per cycle.

### Batched decode

`--slots N` (up to 8, server and CLI) gives each request its own slot: DeltaNet state, MTP state and position; its KV comes from the shared pool (below). On two GPUs all active slots share one cycle:

- **One weight read for all**: each slot checks its own drafts (1 to 5 rows); the rows of all slots are packed into one batch of up to 32, so each weight is read once per cycle. Up to 4 slots a cycle takes 16 rows, from 5 slots 32 (a wider cycle costs more than it returns at few slots).
- **Tensor cores**: past 10 rows the dp4a path runs out of registers, so 16-, 24- and 32-row cycles use int8 tensor-core MMAs straight from Q8_0, Q4_K, Q5_K and Q6_K blocks, with one code path per weight type (`mma.cuh`; Q8_0 rows in 16-row tiles, `q8_tile.h`).
- **Per-slot work**: attention, DeltaNet, draft checking and sampling run per slot, each with its own draft depth.
- **Smaller exchange**: partial sums cross PCIe as int8 with a scale per 32 values (bf16 with `MK_OPT=0`), and only rows in use are sent.

The one-GPU library accepts `--slots` too, but decodes the slots one after another.

### Shared KV pool and prefix cache

- **One pool for all requests**: KV lives in blocks of 64 tokens (128 with q8_0 KV) that requests take as they grow, so one request can use up to `-c` while the others are short. Default `--kv-mem` takes the memory the fixed per-slot split used to take: Q8_0 with fp16 KV and 8 slots holds 208,704 tokens in total, and one 117K-token request runs next to four 10K ones.
- **Prefix cache**: finished conversations stay in the pool as a radix tree of blocks plus DeltaNet state snapshots (at the end of the system prompt, the start of the last user message, the start of the generation prompt and the end of the prompt). A request that shares a prefix with anything cached prefills only the rest (`usage.prompt_tokens_details.cached_tokens`). Snapshots take at most a quarter of the pool; cached blocks are evicted least recently used first.
- **Preemption**: requests are admitted when their prompt fits. When the pool runs full, cached blocks go first, then the most recently admitted request is paused (its tokens stay cached) and resumes when room frees up; its stream just pauses. A request that alone outgrows the pool ends with `finish_reason` `length`.
- **Cost**: 0.1 to 0.6 % decode tok/s against the fixed split, quality unchanged (same KL divergence).
- Greedy output after a pause or in a different batch is a valid continuation but not always token-identical: the draft/verify path takes a slightly different numeric route.

### The megakernel

L-Forster's design from open-jet. This fork adds the per-tensor quant types and the two-GPU and batched paths above.

Each decode step runs a whole speculative cycle in one persistent kernel launch:

```
main model over T = 1 + n_draft tokens  ->  lm_head + draft check on the GPU (greedy, or exact
speculative sampling)  ->  MTP catch-up for the accepted tokens  ->  n_draft MTP draft steps
```

The ~340 phases of a cycle are separated by a software grid barrier; the host only reads back the accepted tokens. Where the speed comes from:

- **Weights stay in their GGUF encoding** (Q4_K / Q5_K / Q6_K / Q8_0) and are read once per cycle, whatever T is. Q6_K and Q8_0 are re-laid-out at load (values unchanged) for aligned 16/8-byte loads.
- **dp4a GEMV with shared activations**: activations are int8 (per-16/32 scales, like llama.cpp's q8_1); each warp streams 2-4 weight rows and reuses every activation load across them, so T = 5 costs almost the same memory time as T = 1. Weight loads skip L1 and are marked L2 evict-first.
- **No launches, few round trips**: each phase re-quantizes its output (including the next RMS norm's weight), so the next phase only copies it in (`cp.async`).
- **DeltaNet rollback by lazy replay**: the recurrent state is committed only up to the surely accepted token; accepted drafts are replayed from a small ring at the start of the next cycle, so rejected drafts cost nothing.
- **Attention**: split-K flash decoding on fp16 tensor cores (fp32 accumulation, online softmax), double-buffered 32-position K/V tiles (~5% faster cycles at 30K context).
- **Drafting on the GPU**: the MTP draft head scores only the 32,768 most frequent tokens (`draft_vocab_32k.bin`, built by `tools/vocab_freq`, covers ~99.4% of tokens) plus the tokens of the current sequence (+3-8% decode speed with pasted code). The check always uses the full vocabulary, so this changes acceptance, never output.
- **Sampling** keeps the 32 best logits per row; a draft is accepted with probability p(draft), a rejected one is resampled from the rest, so the sampled distribution stays exact.
- **L2 warm-up**: before each barrier, blocks prefetch what their next latency-bound phase reads first.

### Prefill

Prompts go through ordinary kernel launches (`prefill.cuh`) in chunks of up to 2,048 tokens, which leave exactly the state the megakernel expects; the last prompt token then goes through the megakernel, which produces the first token and drafts. The same path appends to a running conversation: it continues from the committed DeltaNet position, and the output is token-identical to processing the whole conversation as one prompt.

- **GEMMs on fp16 tensor cores straight from the k-quants** (`prefill_gemm.cuh`, ~85% of prefill time): 128 x 128 tiles, weights dequantized to fp16 in registers, ~100-108 TFLOPS on all four quant types (`bench/mmq_bench.cu`). FFN gate and up share one GEMM that writes `silu(gate) * up`.
- **DeltaNet in chunked (WY) form on tensor cores**: 32-token chunks, the state kept in fp32 MMA accumulators; ~2% of prefill time, 3.5e-4 relative error against a float64 reference.
- **Attention**: causal flash attention over the KV cache on tensor cores.

## Server

`server/mk_server.py`: OpenAI-compatible `/v1/chat/completions` (streaming or not), `/health`, `/v1/models`, `GET /status`, through `libgeezuz.so` or, with `--lib`, `libgeezuz_tp2.so` (C API in `capi.cpp`).

| flag | default | |
|---|---|---|
| `-m` | | GGUF file |
| `--host`, `--port` | 127.0.0.1, 18080 | |
| `-c` | 32768 | maximum context of one request |
| `-d` | 4 | drafts per cycle |
| `--draft-vocab` | | draft vocabulary file |
| `--lib` | `libgeezuz.so` | engine library |
| `--kv` | auto | KV cache: `f16`, `q8`, or `auto` (q8_0 only when the fp16 pool holds less than one `-c` request) |
| `--kv-mem` | 0 | KV pool MB per card (0: the free memory after the slots and a reserve) |
| `--slots` | 1 | requests decoded at once; more requests queue |
| `--prefill-chunk` | 512 | prompt tokens fed at a time while other requests decode (they stall one chunk at most) |
| `--prefill-share` | 0.25 | share of the time prompts are prefilled while other requests decode: those keep the rest of their speed, a prompt takes 1/share times as long |
| `--think-budget` | 0 | reasoning tokens per request before `</think>` is forced (0: unlimited; a request's `thinking_budget` overrides it) |

- Prompts use the chat template stored in the GGUF: `chat_template_kwargs` `enable_thinking` / `preserve_thinking` / `reasoning_effort`, OpenAI `tools`. Reasoning streams as `reasoning_content`; tool calls come back as `tool_calls`.
- Follow-up requests of a conversation, and requests sharing a system prompt, only prefill what is not in the prefix cache. Requests that arrive together with the same uncached system prompt prefill it once: the others wait until it is in, then start from its snapshot.
- With nothing decoding, prompts are prefilled whole. While others decode, prompts go in `--prefill-chunk` chunks, shortest remaining prompt first, for `--prefill-share` of the time.
- Request ids: a client's `X-Request-Id` header (letters, digits, `._:-`, up to 128) or else the response id (`chatcmpl-...`) tags every log line of the request (`slot N [id]: ...`: admission, cache hit, preemption, thinking budget, finish, errors), is echoed in the response's `X-Request-Id` header and shows as `id` per slot in `/status`.
- Sampling: temperature, top_p, top_k (<= 32), min_p, seed. Presence / repetition penalties are ignored (with a warning).
- `GET /status` reports each slot (with its request `id` and preemption count), `kv_pool` (free, total, cached, evictable and `used_blocks` held by running requests, tokens per block, `updated`: read after every decode cycle) and `preempted`; `/props` reports `total_slots`.

open-jet uses this server in place of `llama-server` (`openjet setup` builds the one-GPU engine).

## Options

CLI (`geezuz`, `geezuz_tp2`):

| flag | |
|---|---|
| `-d N` | drafts per cycle, 0-4 (with 4, a rejected draft drops the next cycle to 3) |
| `-c N` | maximum context of one request |
| `--kv f16\|q8\|auto` | KV cache format |
| `--kv-mem MB` | KV pool per card (default: the free memory after the slots and a reserve) |
| `--slots N` | N prompt variants decoded together, each compared with its own solo run |
| `--think [xhigh\|medium\|low]` | reasoning mode (template default: xhigh) |
| `--temp`, `--top-p`, `--top-k`, `--min-p`, `--seed` | sampling (default greedy) |
| `--raw` | no chat template |
| `--ids`, `--prompt-ids f` | print token ids; prompt as token ids |
| `--turn2 text`, `--turn2-file f` | a second user turn appended to the conversation |
| `--score ids.txt` | teacher-forced agreement |
| `--kld ref`, `--score-path` | KL divergence, perplexity and agreement against a reference (see [Quality](#quality)) |

Environment:

| variable | |
|---|---|
| `MK_DEVICES=a,b` | the two cards |
| `MK_PF_CAP=1024` | prefill chunk in tokens (multiple of 128, default 2048); 1024 frees ~0.34 GB per card at ~4% slower prefill |
| `MK_EMBD_DEVICE=1` | keep `token_embd` in VRAM (default: pinned host memory, 0.72 GB for Q4_K_M and 1.35 GB for Q8_0 per card saved) |
| `MK_DRAFT_SAMPLING=greedy` | at temperature > 0, take the draft head's argmax instead of sampling it; both keep the output distribution exact, the default accepts more drafts |
| `MK_PREFILL=mk` | prefill through the decode megakernel |
| `MK_PROFILE=1` | per-phase timing, with a "min-composed" estimate that filters out other GPU users |
| `MK_PROFILE=batch` | also block arrival per phase and a per-step record of block 0 (activation loads, compute, exchange, peer wait) |
| `MK_PF_PROFILE=1` | per-kernel prefill timing |
| `MK_HOST_PROF=1` | host time per decode cycle at exit: KV block bookkeeping, row prep + launch, wait for the cards, rest |
| `MK_OPT=n` | batched-cycle switches, default 4: bit 1 one type-switching GEMV, bit 2 no next-phase L2 prefetch, bit 4 int8 exchange (else bf16) |
| `MK_BATCH_ROWS=n` | fixed row cap per batched cycle (default 16 up to 4 slots, 32 above) |

## Supported GGUF files

Qwen3.8-27B, llama.cpp arch `qwen35`: 48 Gated-DeltaNet layers, 16 gated-attention layers, one MTP block. Every quantized weight tensor may be Q4_K, Q5_K, Q6_K or Q8_0, chosen per tensor (this fork's addition; upstream takes unsloth's Q4_K_M layout):

| tensors | types |
|---|---|
| `attn_q`, `attn_k`, `attn_v`, `attn_output`, `attn_qkv`, `attn_gate`, `ssm_out`, `ffn_gate`, `ffn_up`, `ffn_down`, `token_embd`, `output`, `nextn.eh_proj` | Q4_K, Q5_K, Q6_K or Q8_0 |
| `ffn_gate` and `ffn_up` of one layer | same type |
| `ssm_alpha`, `ssm_beta` | Q4_K, Q5_K, Q6_K or Q8_0, converted to F32 at load |

Other types (IQ*, Q2/Q3, ...) are refused at load, naming the tensor. A plain `llama-quantize` from the BF16 GGUF works:

```sh
llama-quantize Qwen3.8-27B-BF16-00001-of-00002.gguf Qwen3.8-27B-Q4_K_M.gguf Q4_K_M
```

## FAQ

**One instance on two cards, or two one-GPU instances?** One instance on two cards gives one request up to 190 tok/s and twice the KV memory, and serves up to 8 requests at once with `--slots`.

**More than two GPUs?** No, the split is written for exactly two.

**Parallel requests?** Yes, up to 8 on two GPUs: 622 tok/s in total on Q4_K_M. A task split into 8 independent parts finishes about 3.4x faster than as one request. All slots share one KV pool, so a long request and several short ones fit together; when the pool runs full, the newest request pauses until room frees up.

**Which quants?** Q4_K_M (unsloth) is tuned, benchmarked and checked for quality; batched decode is also tuned on Q8_0. Q5_K_M, Q6_K and UD-Q4_K_M load but are not tuned or checked. Q8_0 (28.7 GB) only fits on two cards.

**How long can the context be?** The model's native context is 262,144 tokens. One GPU: fp16 KV up to ~80K tokens, q8_0 above (147K per upstream). Two GPUs: the full 262,144 tokens fit with fp16 KV. The longest context benchmarked here is 56K.

**Is the llama.cpp comparison fair?** llama.cpp ran in three setups, including tensor split (both cards on every layer, like Geezuz) with the same 4 drafts; speed-ups are against the fastest. Tensor split with 2 drafts was not measured; with layer split, 2 drafts beat 4 by 2%.

**RTX 4090 / 5090?** It builds for sm_89 and sm_120 and runs on 24 GB+ cards, but is tuned and measured on 3090s only. Two GPUs need P2P, which GeForce cards only have with the patched driver.

**NVLink?** Not used, not needed: everything here runs over PCIe 4.0 x8 per card.

**Vision (mmproj)?** No, text only.

**Pass rate on coding tasks, other engines (HyperQwen, ninfer, ...)?** Not measured yet.

## Limitations

- Built for this model's shapes only.
- Tuned and measured on RTX 3090s (sm_86); builds for sm_89 and sm_120.
- At most two identical GPUs, with PCIe P2P. Batched decode: up to 8 slots, two-GPU build only.
- Sampling: temperature, top-k (<= 32), top-p, min-p (llama.cpp's order); no presence / repetition penalties.
- Text only. The prefix cache resumes at its snapshots: editing an earlier part (e.g. context compaction) re-prefills from the last snapshot before the edit. No cache save / restore across restarts.

## Files

| file | |
|---|---|
| `qwen38_mk.cu` / `qwen38_mk.h` | loader, repacking, the megakernel (one and two GPUs), the `MKEngine` API |
| `batch.cuh` / `mma.cuh` / `q8_tile.h` | batched decode cycle, its tensor-core GEMVs, the Q8_0 tile layout |
| `kv_cache.h` | KV block pool: refcounts, copy-on-write, radix-tree prefix cache, DeltaNet snapshots, LRU eviction |
| `tests/kv_cache_fuzz.cpp`, `tests/kv_cache_scen.cpp` | pool fuzz and scenario tests against a content-checking fake device (`g++ -std=c++17 -O2 -I.. <file>`) |
| `tests/sched_sim.py`, `tests/sched_sim_identity.py` | server scheduler simulation (admission, preemption, thinking budget) on a fake engine |
| `prefill.cuh` / `prefill_gemm.cuh` | prefill kernels and their fp16 GEMM |
| `capi.cpp` | C API (engine + libllama tokenizer) for `libgeezuz.so` / `libgeezuz_tp2.so` |
| `server/mk_server.py` | OpenAI-compatible server |
| `main.cu` | CLI |
| `gguf.h` | minimal mmap GGUF reader |
| `draft_vocab_32k.bin`, `tools/vocab_freq.cpp` | MTP draft vocabulary (int32 ids) and its builder |
| `tools/llama_ref.cpp` / `logit_stats.h` | llama.cpp reference distributions and the comparison with them |
| `bench/tp2_bench.py`, `bench/batch_bench.py` | two-GPU benchmark vs llama-server; parallel-request sweep (Geezuz, llama.cpp, vLLM) |
| `bench/quality_bench.py`, `bench/gsm8k_test.jsonl` | GSM8K accuracy through each engine's server (test set from openai/grade-school-math) |
| `bench/tool_bench.py` | long-context tool-call accuracy, fp16 vs q8 KV |
| `bench/code_bench.py`, `bench/think_bench.py`, `bench/bench.py` | one-GPU benchmarks |
| `bench/*_bench.cu` | GEMV / GEMM microbenchmarks |
| `docker/build.Dockerfile`, `scripts/box-*.sh` | containerized build on a remote GPU host |

## License and credits

The megakernel is the work of **L-Forster** in [open-jet](https://github.com/L-Forster/open-jet): the decode megakernel, the prefill kernels, the loader, the server and CLI, the one-GPU benchmarks (`bench/code_bench.py`, `bench/think_bench.py`, `bench/bench.py`, `bench/{ffn,gemv,mmq,resid}_bench.cu`), the quality tooling (`tools/llama_ref`, `logit_stats.h`, `--kld`) and the draft vocabulary. This repository is a modified version of `megakernel/` at commit `2993f47`, imported with its git history, licensed under the GNU Affero General Public License v3.0 and distributed under the same license (see `LICENSE`).

Modifications, October 2026: sources moved to the repository root; per-tensor Q4_K / Q5_K / Q6_K / Q8_0 weights in decode and prefill; tensor parallelism over two GPUs via PCIe P2P (decode and prefill) with the two-GPU tuning; batched decode of up to 8 slots (tensor-core GEMVs, int8 exchange, server scheduler); shared paged KV pool with prefix cache and preemption (`kv_cache.h`, `tests/`); `bench/tp2_bench.py`, `bench/batch_bench.py`, `bench/quality_bench.py`, `bench/tool_bench.py`, `bench/kq_mma_bench.cu`, `bench/q6_dp4a_bench.cu`; the benchmarks and quality checks above.
