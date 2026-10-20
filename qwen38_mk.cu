// Persistent cooperative "megakernel" for Qwen3.8-27B (llama.cpp arch "qwen35", shared with Qwen3.5/3.6;
// 64 layers: 48 Gated-DeltaNet + 16 gated full-attention, plus one MTP block), tuned for
// RTX 3090 (sm_86, 82 SMs).
//
// One kernel launch executes an entire speculative decoding cycle:
//   main forward over T = 1 + n_draft tokens  ->  lm_head + greedy verify on device
//   -> MTP catch-up rows  ->  n_draft MTP draft steps.
// Phases inside the launch are separated by a software grid barrier.  Weights stay in their
// GGUF k-quant encodings (values unchanged; rows stored in 16-row tiles, Q6_K / Q8_0 blocks re-laid-out for aligned loads)
// and are streamed once per cycle with int8 dp4a dot products against q8 activations.
//
// Gated-DeltaNet rollback after rejected drafts is done by lazy replay: the committed state
// only ever includes surely-accepted tokens; accepted drafts are replayed from a small ring
// of (k, v, g, beta) at the start of the next cycle, so no state snapshots are written.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <map>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <thread>
#include <vector>

#include "ggml.h"
#include "gguf.h"
#include "kv_cache.h"
#include "qwen38_mk.h"
#include "q8_tile.h"

#define CK(x)                                                                                   \
    do {                                                                                        \
        cudaError_t e_ = (x);                                                                   \
        if (e_ != cudaSuccess) {                                                                \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__,             \
                    cudaGetErrorString(e_));                                                    \
            exit(1);                                                                            \
        }                                                                                       \
    } while (0)

#ifndef MK_TP
#define MK_TP 1
#endif
static_assert(MK_TP == 1 || MK_TP == 2);

namespace {

constexpr int D = 5120, FF = 17408, VOCAB = 248320, NL = 64, NLA = 65;
constexpr int NH = 24, NKV = 4, HD = 256;
constexpr int NKH = 16, NVH = 48, HK = 128;
constexpr int QKV = 10240, VD = 6144, QG = 12288, KVD = 1024;
constexpr int MAXT = 5;
// a decode cycle at position n writes K/V below n + KV_AHEAD: rows up to n + MAXT - 1, MTP drafts up to
// n + MAXT (bonus) + MAXT - 2; the batched MTP catch-up's padding stays inside only while mtp_valid = n + 1,
// which every prefill establishes (step refuses a sequence that never had one)
constexpr int KV_AHEAD = 2 * MAXT - 1;
constexpr int MAXR = 32;    // GEMV rows of a batched cycle (all slots' rows; per-slot work stays <= MAXT)
constexpr int MAXR_DP = 10; // of those, the widest the dp4a GEMVs take (more rows need every matrix Q8_0: mma.cuh)
constexpr int NT = 512, NW = NT / 32;
constexpr float EPS = 1e-6f;
constexpr float ROPE_BASE = 10000000.0f;
constexpr int CONV_RING = 8, KV_RING = 16, H_RING = 16;
constexpr int KVGB = NKH * HK + NVH * HK + 2 * NVH; // k | v | g | beta
constexpr int KVGB_V = NKH * HK, KVGB_G = KVGB_V + NVH * HK, KVGB_B = KVGB_G + NVH;
constexpr int N_ATT_SLOTS = 17; // 16 trunk attention layers + MTP block
constexpr int MAX_CH = 32;      // attention split-K chunks per kv head
constexpr int DV_EXTRA = 4096;  // draft-vocabulary rows reserved for tokens of the current sequence
constexpr int AB_SPLIT = 4;     // K-chunks per delta-net alpha/beta row
constexpr int U_D = D / 32, U_FF = FF / 32, U_VD = VD / 32, U_EH = 2 * D / 32;
constexpr int UMAX = U_FF;
// per-card dims: TP=1 equals the full values; TP=2 halves, except FF padded 8704 -> 9216 (272 units not a multiple of 32)
constexpr int NH_L = NH / MK_TP, NKV_L = NKV / MK_TP;
constexpr int NKH_L = NKH / MK_TP, NVH_L = NVH / MK_TP;
constexpr int QKV_L = QKV / MK_TP, VD_L = VD / MK_TP, QG_L = QG / MK_TP, KVD_L = KVD / MK_TP;
constexpr int KVGB_V_L = NKH_L * HK, KVGB_G_L = KVGB_V_L + NVH_L * HK, KVGB_B_L = KVGB_G_L + NVH_L;
constexpr int KVGB_L = KVGB_B_L + NVH_L;
constexpr int FF_L = (MK_TP == 2) ? 9216 : FF;
constexpr int VOCAB_L = VOCAB / MK_TP;
constexpr int U_VD_L = VD_L / 32, U_FF_L = FF_L / 32, U_EH_L = U_EH / MK_TP;
static_assert((U_VD_L % 32) == 0 && (U_FF_L % 32) == 0 && (U_EH_L % 32) == 0);
static_assert((QG_L % 128) == 0 && (KVD_L % 128) == 0 && (QKV_L % 128) == 0);
static_assert((VD_L % 128) == 0 && (FF_L % 128) == 0 && (VOCAB_L % 128) == 0);

constexpr int AR = 6 * MAXT;   // attention query rows per kv head (6 q heads x tokens)
constexpr int SM_XU = 2304; // activation units (x tokens) shared memory holds: TP=2 ffn_down up to 8 rows; TP=1 T = 5 ffn_down fills in two halves
constexpr int SM_XA = 0;
constexpr int SM_XB = SM_XA + SM_XU * 16;
constexpr int SM_XM = SM_XB + SM_XU * 16;
constexpr int SM_MISC = SM_XM + SM_XU * 8;
constexpr int SM_PARK = SM_MISC + 2048; // reduced partial sums parked while their all-reduce is in flight
constexpr int ATP = 32; // attention: positions per K/V tile (two tiles in flight)
constexpr int KVQ_ROW = 256 + 16; // staged q8 K or V row: 256 int8 values + 8 fp16 scales
// K/V tile region: fp16 cache: 2 stages of fp16 K/V tiles; q8 cache: one fp16 K/V tile + 2 stages of q8 tiles
constexpr int SM_ATTN_KV = (4 * ATP * (256 + 8) * 2 > 2 * ATP * (256 + 8) * 2 + 4 * ATP * KVQ_ROW)
                               ? 4 * ATP * (256 + 8) * 2 : 2 * ATP * (256 + 8) * 2 + 4 * ATP * KVQ_ROW;
constexpr int SM_ATTN = 32 * (256 + 8) * 2 + SM_ATTN_KV + 32 * ATP * 4 + 32 * (ATP + 8) * 2 + 96 * 4; // Q, K/V, S, P, stats
constexpr int SM_TOTAL = (SM_PARK + 4096 > SM_ATTN ? SM_PARK + 4096 : SM_ATTN);
static_assert(SM_TOTAL <= 99 * 1024, "shared memory budget");
// KV arena geometry: blocks of NP positions, [blk][att layer][kv head][K|V][NP][HD]; a q8 K or V
// sub-block holds its int8 rows [NP][HD] then their fp16 scales [NP][HD / 32]
template <bool Q8>
struct KvL {
    static constexpr int NP = Q8 ? 128 : 64, SH = Q8 ? 7 : 6; // positions per block, log2
    static constexpr int RB = Q8 ? HD : HD * 2;                // bytes of one K or V row's values
    static constexpr size_t SUB = (size_t) NP * (RB + (Q8 ? HD / 32 * 2 : 0)); // one (layer, head, K|V)
    static constexpr size_t BLK = (size_t) N_ATT_SLOTS * NKV_L * 2 * SUB;
    static_assert((1 << SH) == NP, "kv_row / kv_tile index with pos >> SH and mask with NP - 1");
};
// the decode tile straddles at most one block boundary only while a tile fits in a block
static_assert(ATP <= KvL<false>::NP && ATP <= KvL<true>::NP && KvL<false>::NP % ATP == 0 && KvL<true>::NP % ATP == 0);
static_assert(KvL<false>::SUB % 16 == 0 && KvL<true>::SUB % 16 == 0);
// one lane's DeltaNet state (floats), and what a snapshot keeps besides it: the conv rows of the 3 positions
// before it and the h ring
constexpr size_t DS_N = (size_t) 48 * NVH_L * 128 * 128;
constexpr size_t SNAP_STAGE = (size_t) 3 * 48 * QKV_L + (size_t) H_RING * D;

enum { T_F32 = 0, T_Q8_0 = 8, T_Q4K = 12, T_Q5K = 13, T_Q6K = 14 };
// MK_OPT bits for A/B runs without a rebuild: switching gemv_mma, no next-phase L2 weight prefetch in
// batched cycles, int8 all-reduce partials (one scale per 32 values) instead of bf16
enum { OPT_GEMV_SW = 1, OPT_NO_PF = 2, OPT_XR8 = 4 };
enum { MODE_PREFILL = 0, MODE_DECODE = 1, MODE_BENCH = 2 };

struct Mat {
    const uint8_t * w;
    const __half * d; // per-row block scales for repacked Q6_K / Q8_0
    int type, K, N, rb, nd;
};

struct LayerW {
    int attn;
    const float *attn_norm, *post_norm;
    // gated delta net
    Mat qkv, gate, sout;
    const float *ab, *conv, *dt, *A, *snorm;
    // gated attention
    Mat q, k, v, o;
    const float *qn, *kn;
    // ffn
    Mat fg, fu, fd;
};

struct ModelW {
    LayerW L[NLA];
    Mat eh, head;
    Mat head_d;            // MTP draft head: subset of lm_head rows (== head when no subset is used)
    const int * draft_ids; // row -> token id for head_d (nullptr: identity)
    const float *enorm, *hnorm, *shnorm, *onorm;
    const uint8_t * embd;
    int embd_type; // raw GGUF type of one embedding row, with its row bytes (embd_rb)
    int embd_rb;
};

// Quantised activation vector(s) in the "unit" layout consumed by the GEMV:
// unit u = (super-block b, 64-group p, half h) covers elements A=[256b+64p+16h,+16) and
// B=[256b+64p+32+16h,+16).  xa/xb hold the int8 values, d the per-32 scales of the natural
// blocks the A/B halves live in, qs the int sums of each 16-chunk, ssq per natural block x^2.
struct QBuf {
    int8_t *xa, *xb;
    uint2 * meta; // per unit: half2 (scale of A / B chunk), short2 (int sums of A / B chunk)
    float * ssq;
    int U;
};

struct Ctl {
    int n;         // position of tok[0]
    int tok[8];    // tokens fed to the main model this cycle
    int dn_c;      // delta-net committed position (state covers positions < dn_c)
    int mtp_valid; // MTP kv valid for positions < mtp_valid
    int n_out, out[8];
    int n_acc;
    int mtp_R, mtp_logits, mtp_P;
    int mtp_pos[8], mtp_tok[8];
    int tgt[8];
    int pad[8];
};

constexpr int MAXB = 8; // sequence slots

// State that persists across cycles, one per sequence slot.
struct SeqBufs {
    Ctl * ctl;
    float *conv, *kvgb, *dstate;
    const int * bt; // block table: arena block of positions [i * NP, (i + 1) * NP) (KvL), ceil(ctx / NP) entries
    float *h_hist, *hm;
    float2 * dcand; // [MAXT][TOPK_MAX] each MTP draft's normalised distribution q, descending (q[0] = 0: point-mass draft)
};

struct Bufs {
    float *x, *xm;
    QBuf qn, qo, qa, qe;
    float *z, *ab, *qbuf, *kbuf, *vbuf;
    float * qkvs; // [MAXR][QKV_L] delta-net qkv rows of a batched cycle (dn_core copies them into each slot's conv ring)
    SeqBufs sq[MAXB];
    int nseq;
    int kvq8;
    uint8_t * kv; // KV arena (KvL<kvq8>): fp16 K/V, or int8 (q8_0) with one fp16 scale per 32 values
    float * apart;
    float2 * aml;
    const float2 * rope; // [ctx][32] (cos, sin) of pos * base^(-2i/64)
    float2 * blk_best; // per block: best (logit, id) per row, or the TOPK_MAX best when sampling
    float2 * cand;     // [MAXR][TOPK_MAX] best (logit, id) of the main lm_head rows, descending (sampling)
    float * logits;    // [MAXT][VOCAB] main lm_head logits of a scoring launch (Launch::score == 2)
    unsigned * bar;
    unsigned * cnt;
    unsigned long long * prof; // optional per-barrier timestamps (block 0)
    int ctx;
    int opt;               // MK_OPT experiment bits (OPT_*)
#if MK_TP > 1
    int rank;              // this card's tensor-parallel rank
    __nv_bfloat16 * xr[2]; // own reduce receive buffers [D/32 items][MAXR*32], double-buffered by k
    unsigned * fl;         // own per-item reduce flags
    __nv_bfloat16 * peer_xr[2]; // peer's reduce receive buffers, P2P-mapped
    unsigned * peer_fl;    // peer's per-item reduce flags
    float2 * lx;           // lm_head exchange slot [MAXR*TOPK_MAX] float2
    unsigned * lfl;        // lm_head exchange flag
    float2 * peer_lx;      // peer's lm_head exchange slot, P2P-mapped
    unsigned * peer_lfl;   // peer's lm_head exchange flag
#endif
};

__constant__ ModelW cW;
__constant__ Bufs cB;
// The selected slot's buffers, copied in by the host between calls: plain constant loads as before slots existed.
__constant__ SeqBufs cS;
__device__ __forceinline__ const SeqBufs & sqb() { return cS; }
// BS (batched cycle): slot s's buffers from cB; otherwise the selected slot's
template <int BS>
__device__ __forceinline__ const SeqBufs & sqx(int s) {
    if constexpr (BS) return cB.sq[s];
    else return cS;
}

// K / V rows of (attention layer, kv head, position) in sequence S's blocks; ks / vs: their q8 scales
struct KvRow {
    uint8_t *k, *v;
    __half *ks, *vs;
};
template <bool Q8>
__device__ __forceinline__ KvRow kv_row(const SeqBufs & S, int layer, int g, int pos) {
    using K = KvL<Q8>;
    const int r = pos & (K::NP - 1);
    uint8_t * sub = cB.kv + (size_t) __ldg(S.bt + (pos >> K::SH)) * K::BLK + (size_t) (layer * NKV_L + g) * 2 * K::SUB;
    __half * sc = reinterpret_cast<__half *>(sub + K::NP * K::RB) + r * (HD / 32);
    return {sub + r * K::RB, sub + K::SUB + r * K::RB, sc, reinterpret_cast<__half *>(reinterpret_cast<uint8_t *>(sc) + K::SUB)};
}

// rows of a tile of positions tb .. tb + np - 1 (np <= NP, so at most one block boundary)
struct KvTile {
    const uint8_t *a, *b; // K sub-blocks holding the rows before / from the boundary
    int o, split;         // row of tb in a; tile rows < split lie in a
};
template <bool Q8>
__device__ __forceinline__ KvTile kv_tile(const SeqBufs & S, int layer, int g, int tb, int np) {
    using K = KvL<Q8>;
    const size_t sub = (size_t) (layer * NKV_L + g) * 2 * K::SUB;
    const int o = tb & (K::NP - 1);
    return {cB.kv + (size_t) __ldg(S.bt + (tb >> K::SH)) * K::BLK + sub,
            cB.kv + (size_t) __ldg(S.bt + ((tb + max(np, 1) - 1) >> K::SH)) * K::BLK + sub, o, K::NP - o};
}
// values of tile row p of K (which 0) or V (which 1)
template <bool Q8>
__device__ __forceinline__ const uint8_t * kv_tile_row(const KvTile & t, int p, int which) {
    using K = KvL<Q8>;
    const uint8_t * sub = (p < t.split ? t.a : t.b) + which * K::SUB;
    return sub + (size_t) (p < t.split ? t.o + p : p - t.split) * K::RB;
}
// fp16 scales of q8 tile row p of K (which 0) or V (which 1)
__device__ __forceinline__ const __half * kv_tile_scales(const KvTile & t, int p, int which) {
    using K = KvL<true>;
    const uint8_t * sub = (p < t.split ? t.a : t.b) + which * K::SUB + K::NP * K::RB;
    return reinterpret_cast<const __half *>(sub) + (p < t.split ? t.o + p : p - t.split) * (HD / 32);
}

struct Launch {
    int mode, T, last, ndraft;
    int score; // prefill only: greedy argmax for every row (teacher-forced evaluation), no MTP; 2: also their logits
    // sampling (temp > 0; greedy otherwise), applied in llama.cpp's order: top_k, top_p, min_p, temperature
    float temp, top_p, min_p;
    int top_k; // <= TOPK_MAX
    unsigned long long seed;
    int draft_prob; // sample the MTP draft, verify by rejection (default; MK_DRAFT_SAMPLING=greedy: argmax draft)
#if MK_TP > 1
    unsigned gen; // host bumps per launch; folded into the cross-GPU flag tags
#endif
};
constexpr int TOPK_MAX = 32; // candidates kept per main lm_head row when sampling
#if MK_TP > 1
__constant__ Launch L; // host writes gen each launch
#endif

// ------------------------------------------------------------------------------------------
// small device helpers
// ------------------------------------------------------------------------------------------

// Weights are streamed once per cycle: bypass L1 and mark them evict-first in L2 so that data
// prefetched with evict-last (recurrent state, KV cache) survives the stream.
__device__ __forceinline__ unsigned long long l2_evict_first() {
    unsigned long long pol;
    asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
    return pol;
}
__device__ __forceinline__ int4 ldw4(const void * p) {
    int4 r;
    asm("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.s32 {%0,%1,%2,%3}, [%4], %5;"
        : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
        : "l"(p), "l"(l2_evict_first()));
    return r;
}
__device__ __forceinline__ int2 ldw2(const void * p) {
    int2 r;
    asm("ld.global.nc.L1::no_allocate.L2::cache_hint.v2.s32 {%0,%1}, [%2], %3;" : "=r"(r.x), "=r"(r.y) : "l"(p), "l"(l2_evict_first()));
    return r;
}
__device__ __forceinline__ int ldw1(const void * p) {
    int r;
    asm("ld.global.nc.L1::no_allocate.L2::cache_hint.s32 %0, [%1], %2;" : "=r"(r) : "l"(p), "l"(l2_evict_first()));
    return r;
}
__device__ __forceinline__ void pf_l2(const void * p) { asm volatile("prefetch.global.L2::evict_last [%0];" ::"l"(p)); }
// prefetch `bytes` starting at p (128-byte lines) with the threads of the calling block
__device__ __forceinline__ void pf_block(const void * p, int bytes) {
    for (int off = threadIdx.x * 128; off < bytes; off += NT * 128) pf_l2(reinterpret_cast<const uint8_t *>(p) + off);
}
__device__ __forceinline__ float4 ldw4f(const float * p) {
    int4 r = ldw4(p);
    return make_float4(__int_as_float(r.x), __int_as_float(r.y), __int_as_float(r.z), __int_as_float(r.w));
}
__device__ __forceinline__ unsigned short ldwu16(const void * p) {
    unsigned short r;
    asm("ld.global.nc.L1::no_allocate.u16 %0, [%1];" : "=h"(r) : "l"(p));
    return r;
}

__device__ __forceinline__ float wsum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float wmax(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }

enum {
    PH_EMBED, PH_ATTN_PROJ, PH_ATTN_CORE, PH_ATTN_COMB, PH_RESID_O, PH_DN_PROJ, PH_DN_CORE, PH_RESID_SOUT, PH_FFN_GU,
    PH_FFN_DOWN, PH_LMHEAD, PH_MTP_M0, PH_MTP_EH, PH_MTP_ATTN_PROJ, PH_MTP_ATTN_CORE, PH_MTP_ATTN_COMB, PH_MTP_RESID_O,
    PH_MTP_FFN_GU, PH_MTP_FFN_DOWN, PH_MTP_LMHEAD, PH_COUNT
};

__device__ __forceinline__ unsigned long long gtimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__shared__ unsigned s_bar_k; // index of the last grid barrier passed (profiling only)
// MK_PROFILE step record of block 0 per phase (clock64 cycles, the host scales them by the phase's globaltimer
// length): [0] activation fill cycles, [1] fills, [2] peer wait cycles, [3..7] marks after phase start
// (compute done, parts reduced, partials written, flags released after the fence, first peer partial in),
// [8] phase cycles.
constexpr int PROF_ST = 4096 + 1024 * 128, PROF_SN = 12, PROF_N = PROF_ST + 1024 * PROF_SN, PS_NV = 8;
enum { PS_COMPUTE = 0, PS_RED = 1, PS_WRITTEN = 2, PS_SENT = 3, PS_PEER = 4 };
__shared__ long long s_pc0, s_pf0, s_pw0;
__shared__ unsigned s_pst[PS_NV];
#define PROF_B0 (cB.prof && blockIdx.x == 0 && threadIdx.x == 0)

// Profiling hooks: an inline test, the work out of line so the many inlined callers stay small for ptxas.
__device__ __noinline__ void prof_fill_begin() { s_pf0 = clock64(); }
__device__ __noinline__ void prof_fill_end() {
    if (s_bar_k + 1 < 2048) cB.prof[2048 + s_bar_k + 1] = gtimer();
    s_pst[0] += (unsigned) (clock64() - s_pf0);
    s_pst[1]++;
}
__device__ __noinline__ void prof_step(int i) {
    if (!s_pst[3 + i]) s_pst[3 + i] = max(1u, (unsigned) (clock64() - s_pc0));
}
__device__ __noinline__ void prof_wait_begin() { s_pw0 = clock64(); }
__device__ __noinline__ void prof_wait_end() { s_pst[2] += (unsigned) (clock64() - s_pw0); }
__device__ __forceinline__ void fill_begin() {
    if (PROF_B0) prof_fill_begin();
}
// end of an activation fill: block 0's last fill end per phase and the summed fill time
__device__ __forceinline__ void mark_fill() {
    if (PROF_B0) prof_fill_end();
}
// first call per phase wins (block 0, thread 0's view)
__device__ __forceinline__ void mark_step(int i) {
    if (PROF_B0) prof_step(i);
}

extern __shared__ __align__(16) uint8_t smem[]; // dynamic shared memory of the persistent block

// Grid barrier: release-add on arrival, acquire-poll until every block of this generation arrived.
__device__ __forceinline__ void gsync(unsigned & epoch, int tag) {
    __syncthreads();
    if (threadIdx.x == 0) {
        if (cB.prof) {
            const unsigned k = epoch / gridDim.x + 1;
            if (k < 1024) cB.prof[4096 + k * 128 + blockIdx.x] = gtimer();
        }
        epoch += gridDim.x;
        asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(cB.bar) : "memory");
        unsigned v;
        do {
            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(cB.bar) : "memory");
        } while (v < epoch);
        if (cB.prof && blockIdx.x == 0) {
            const unsigned k = epoch / gridDim.x;
            s_bar_k = k;
            if (k < 2048) cB.prof[k] = (gtimer() << 8) | (unsigned long long) tag;
            const long long c = clock64();
            if (k < 1024) {
                unsigned long long * st = cB.prof + PROF_ST + k * PROF_SN;
                for (int i = 0; i < PS_NV; i++) st[i] = s_pst[i];
                st[PS_NV] = (unsigned long long) (c - s_pc0);
            }
            for (int i = 0; i < PS_NV; i++) s_pst[i] = 0;
            s_pc0 = c;
        }
    }
    __syncthreads();
}

#if MK_TP > 1
// Cross-GPU reduce/flag primitives. Flags are never reset per launch: the tag embeds Launch.gen,
// so a stale flag from an earlier launch simply has an old gen and is not accepted.
__device__ __forceinline__ unsigned tag_of(int k) {
    // k is the reduction index within a launch; keep k < 4096 (1<<12) so it fits the low 12 bits
    return (L.gen << 12) | (unsigned) k;
}
// The peer may already have sent its next reduction on this flag (it only needs my data, sent
// before I wait), so a later tag also means mine arrived; the signed difference survives wrap.
__device__ __forceinline__ bool tag_reached(unsigned v, unsigned t) { return (int) (v - t) >= 0; }
__device__ __forceinline__ void flag_release_sys(unsigned * p, unsigned v) {
    asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ unsigned flag_acquire_sys(const unsigned * p) {
    unsigned v;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
// send this block's partial for item j to the peer, then release the peer's flag j
// The PCIe link is the all-reduce bottleneck, so partials cross it as bf16, or with OPT_XR8 as int8 per
// 32 values (item slot: MAXR * 32 int8, then MAXR float scales); n is a multiple of 32.  The own partial
// is rounded in place the same way, so both cards sum identical values.
__device__ __forceinline__ void xr_send(int j, int k, float * part, int n) {
    if (cB.opt & OPT_XR8) {
        int8_t * dq = reinterpret_cast<int8_t *>(cB.peer_xr[k & 1] + j * MAXR * 32);
        float * ds = reinterpret_cast<float *>(dq + MAXR * 32);
        for (int i = threadIdx.x; i < n / 8; i += NT) {
            float * p = part + 8 * i;
            float am = 0.f;
#pragma unroll
            for (int e = 0; e < 8; e++) am = fmaxf(am, fabsf(p[e]));
            // the 4 threads of a 32-value group are consecutive lanes, all active together
            const unsigned mk = __activemask();
            am = fmaxf(am, __shfl_xor_sync(mk, am, 1));
            am = fmaxf(am, __shfl_xor_sync(mk, am, 2));
            const float d = am / 127.f, inv = am > 0.f ? 127.f / am : 0.f;
            char4 q[2];
            int8_t * qb = reinterpret_cast<int8_t *>(q);
#pragma unroll
            for (int e = 0; e < 8; e++) {
                qb[e] = (int8_t) __float2int_rn(p[e] * inv);
                p[e] = qb[e] * d;
            }
            reinterpret_cast<int2 *>(dq)[i] = *reinterpret_cast<const int2 *>(q);
            if ((i & 3) == 0) ds[i >> 2] = d;
        }
        return;
    }
    uint4 * dst = reinterpret_cast<uint4 *>(cB.peer_xr[k & 1] + j * MAXR * 32);
    for (int i = threadIdx.x; i < n / 8; i += NT) {
        float * p = part + 8 * i;
        __nv_bfloat162 h[4];
#pragma unroll
        for (int e = 0; e < 4; e++) {
            h[e] = __floats2bfloat162_rn(p[2 * e], p[2 * e + 1]);
            p[2 * e] = __low2float(h[e]);
            p[2 * e + 1] = __high2float(h[e]);
        }
        dst[i] = *reinterpret_cast<const uint4 *>(h);
    }
}
// release the peer's flags of this block's n_items items; the release store orders the block's sends
// (made visible to thread 0 by the barrier) without a sequentially consistent fence.sc.sys
__device__ __forceinline__ void xr_release(int n_items, int k) {
    __syncthreads();
    mark_step(PS_WRITTEN);
    if (threadIdx.x == 0) {
        for (int i = 0; i < n_items; i++) flag_release_sys(cB.peer_fl + blockIdx.x + i * gridDim.x, tag_of(k));
        mark_step(PS_SENT);
    }
}
__device__ __forceinline__ void xr_put(int j, int k, float * part, int n) {
    xr_send(j, k, part, n);
    __syncthreads();
    mark_step(PS_WRITTEN);
    if (threadIdx.x == 0) flag_release_sys(cB.peer_fl + j, tag_of(k));
    mark_step(PS_SENT);
}
// own partial (rounded by xr_send) plus the peer's
__device__ __forceinline__ float xr_sum(float own, const __nv_bfloat16 * peer, int i) {
    float p;
    if (cB.opt & OPT_XR8) {
        const int8_t * q = reinterpret_cast<const int8_t *>(peer);
        p = __ldcg(q + i) * __ldcg(reinterpret_cast<const float *>(q + MAXR * 32) + (i >> 5));
    } else p = __bfloat162float(__ldcg(peer + i));
    return cB.rank == 0 ? own + p : p + own;
}
// wait for the peer's partial for item j to arrive; the caller then reads cB.xr[k & 1] + j*MAXR*32
__device__ __forceinline__ void xr_wait(int j, int k) {
    if (threadIdx.x == 0) {
        if (PROF_B0) prof_wait_begin();
        while (!tag_reached(flag_acquire_sys(cB.fl + j), tag_of(k))) __nanosleep(32);
        if (PROF_B0) prof_wait_end();
    }
    __syncthreads();
    mark_step(PS_PEER);
}
// same send/flag exchange for the single lm_head slot
__device__ __forceinline__ void lx_put(int k, const float2 * part, int n) {
    float2 * dst = cB.peer_lx;
    for (int i = threadIdx.x; i < n; i += NT) dst[i] = part[i];
    __syncthreads();
    if (threadIdx.x == 0) flag_release_sys(cB.peer_lfl, tag_of(k));
    mark_step(PS_SENT);
}
__device__ __forceinline__ void lx_wait(int k) {
    if (threadIdx.x == 0) {
        if (PROF_B0) prof_wait_begin();
        while (!tag_reached(flag_acquire_sys(cB.lfl), tag_of(k))) __nanosleep(32);
        if (PROF_B0) prof_wait_end();
    }
    __syncthreads();
    mark_step(PS_PEER);
}
#endif

// block-wide sum, result broadcast. red must hold >= NW floats.
__device__ __forceinline__ float bsum(float v, float * red) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = wsum(v);
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float r = lane < NW ? red[lane] : 0.f;
    r = wsum(r);
    return r;
}

// lane i holds element 32*j+i of row t; quantise the natural 32-block j into the unit layout.
__device__ __forceinline__ void q32_store(const QBuf & q, int t, int j, float v, float ss, bool wss) {
    const int lane = threadIdx.x & 31;
    const float amax = wmax(fabsf(v));
    const float d = amax / 127.0f;
    const int qi = amax == 0.0f ? 0 : (int) roundf(v / d);
    const int b = j >> 3, p = (j >> 1) & 3, part = j & 1, hh = lane >> 4, byte = lane & 15;
    const int gi = t * q.U + 8 * b + 2 * p + hh;
    (part ? q.xb : q.xa)[gi * 16 + byte] = (int8_t) qi;
    int qs = qi;
    qs += __shfl_xor_sync(0xffffffffu, qs, 8);
    qs += __shfl_xor_sync(0xffffffffu, qs, 4);
    qs += __shfl_xor_sync(0xffffffffu, qs, 2);
    qs += __shfl_xor_sync(0xffffffffu, qs, 1);
    if (byte == 0) {
        reinterpret_cast<__half *>(q.meta + gi)[part] = __float2half_rn(d);
        reinterpret_cast<short *>(q.meta + gi)[2 + part] = (short) qs;
    }
    if (wss) {
        const float s = wsum(ss);
        if (lane == 0) q.ssq[t * q.U + j] = s;
    }
}

struct SX {
    int4 * xa;
    int4 * xb;
    uint2 * xm; // per unit: half2 scales of the A / B 16-chunks, short2 int sums of the chunks
    int U;
};

__device__ __forceinline__ SX make_sx(uint8_t * /*sm*/, int U) {
    uint8_t * const sm = smem;
    SX s;
    s.xa = reinterpret_cast<int4 *>(sm + SM_XA);
    s.xb = reinterpret_cast<int4 *>(sm + SM_XB);
    s.xm = reinterpret_cast<uint2 *>(sm + SM_XM);
    s.U = U;
    return s;
}

__device__ __forceinline__ float * misc(uint8_t * /*sm*/) {
    uint8_t * const sm = smem; return reinterpret_cast<float *>(sm + SM_MISC); }

__device__ __forceinline__ void cp_async16(void * smem_dst, const void * gsrc) {
    const unsigned d = (unsigned) __cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(d), "l"(gsrc) : "memory");
}

// Load T rows (starting at row t0) of a quantised activation buffer into shared memory with
// async copies.  norm: s_rs[t] receives the RMS-norm scale of row t (from the producer's
// per-block sums of squares); consumers apply it to their dot products, otherwise s_rs = 1.
template <int T>
__device__ void fill_xs(const QBuf & q, int t0, bool norm, const SX & s, float * s_rs) {
    fill_begin();
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int U = q.U;
    const int8_t * ga = q.xa + (size_t) t0 * U * 16;
    const int8_t * gb = q.xb + (size_t) t0 * U * 16;
    const uint2 * gm = q.meta + (size_t) t0 * U;
    for (int i = tid; i < T * U; i += NT) {
        cp_async16(s.xa + i, ga + (size_t) i * 16);
        cp_async16(s.xb + i, gb + (size_t) i * 16);
    }
    for (int i = tid; i < T * U / 2; i += NT) cp_async16(s.xm + 2 * i, gm + 2 * i);
    asm volatile("cp.async.commit_group;" ::: "memory");
    if (norm) { // normed consumers always read the K = D residual stream (U_D = 160 partial sums)
        if (warp < T) {
            float p[U_D / 32];
#pragma unroll
            for (int k = 0; k < U_D / 32; k++) p[k] = __ldcg(q.ssq + (t0 + warp) * U_D + lane + 32 * k);
            float acc = 0.f;
#pragma unroll
            for (int k = 0; k < U_D / 32; k++) acc += p[k];
            acc = wsum(acc);
            if (lane == 0) s_rs[warp] = rsqrtf(acc / (float) D + EPS);
        }
    } else if (tid < T) {
        s_rs[tid] = 1.0f;
    }
    asm volatile("cp.async.wait_all;" ::: "memory");
    __syncthreads();
    mark_fill();
}

// Async copy of units [u0, u0 + US) of T activation rows (one K-slice) into shared memory.
template <int T, int US>
__device__ void fill_xs_part(const QBuf & q, int u0, const SX & s) {
    const int tid = threadIdx.x, U = q.U;
    __syncthreads(); // the previous slice may still be in use
    for (int i = tid; i < T * US; i += NT) {
        const int t = i / US, u = i - t * US;
        const size_t gi = (size_t) t * U + u0 + u;
        cp_async16(s.xa + i, q.xa + gi * 16);
        cp_async16(s.xb + i, q.xb + gi * 16);
    }
    for (int i = tid; i < T * US / 2; i += NT) {
        const int t = 2 * i / US, u = 2 * i - t * US;
        cp_async16(s.xm + 2 * i, q.meta + (size_t) t * U + u0 + u);
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_all;" ::: "memory");
    __syncthreads();
}

// ------------------------------------------------------------------------------------------
// quantised GEMV: R rows per warp share every activation load (per-lane partial sums;
// the caller reduces across the warp).  Lane handles units u = lane + 32 i.
// ------------------------------------------------------------------------------------------

constexpr int M4 = 0x0F0F0F0F;

__device__ __forceinline__ int gwarp() { return (threadIdx.x >> 5) * gridDim.x + blockIdx.x; }
__device__ __forceinline__ int nwarps() { return NW * gridDim.x; }

template <int T, int R>
__device__ __forceinline__ void zero_acc(float (&a)[R][T]) {
#pragma unroll
    for (int k = 0; k < R; k++)
#pragma unroll
        for (int t = 0; t < T; t++) a[k][t] = 0.f;
}

template <int T, int R>
__device__ __forceinline__ void wsum_acc(float (&a)[R][T]) {
#pragma unroll
    for (int k = 0; k < R; k++)
#pragma unroll
        for (int t = 0; t < T; t++) a[k][t] = wsum(a[k][t]);
}

template <int T, int R, int UU>
__device__ __forceinline__ void accR(const SX & s, int u, const int (&A)[R][4], const int (&B)[R][4],
                                     const float (&fA)[R], const float (&gA)[R], const float (&fB)[R],
                                     const float (&gB)[R], float (&acc)[R][T]) {
#pragma unroll
    for (int t = 0; t < T; t++) {
        const int ix = t * UU + u;
        const int4 va = s.xa[ix], vb = s.xb[ix];
        const uint2 mm = s.xm[ix];
        const float2 dx = __half22float2(*reinterpret_cast<const __half2 *>(&mm.x));
        const int qq = (int) mm.y;
        const float sqA = dx.x * (float) (short) (qq & 0xffff), sqB = dx.y * (float) (short) (qq >> 16);
#pragma unroll
        for (int k = 0; k < R; k++) {
            int sa = __dp4a(A[k][0], va.x, 0);
            sa = __dp4a(A[k][1], va.y, sa);
            sa = __dp4a(A[k][2], va.z, sa);
            sa = __dp4a(A[k][3], va.w, sa);
            int sb = __dp4a(B[k][0], vb.x, 0);
            sb = __dp4a(B[k][1], vb.y, sb);
            sb = __dp4a(B[k][2], vb.z, sb);
            sb = __dp4a(B[k][3], vb.w, sb);
            acc[k][t] += fA[k] * (dx.x * (float) sa) + fB[k] * (dx.y * (float) sb) - gA[k] * sqA - gB[k] * sqB;
        }
    }
}

__device__ __forceinline__ void q4k_scales(const int4 & h, int p, float & fA, float & gA, float & fB, float & gB) {
    const float2 dm = __half22float2(*reinterpret_cast<const __half2 *>(&h.x));
    const int sh = (p & 1) * 16;
    const unsigned y = (unsigned) h.y >> sh, z = (unsigned) h.z >> sh, w = (unsigned) h.w >> sh;
    unsigned sc0, sc1, m0, m1;
    if (p < 2) {
        sc0 = y & 63;
        sc1 = (y >> 8) & 63;
        m0 = z & 63;
        m1 = (z >> 8) & 63;
    } else {
        sc0 = (w & 0xF) | ((y >> 2) & 0x30);
        sc1 = ((w >> 8) & 0xF) | ((y >> 10) & 0x30);
        m0 = ((w >> 4) & 0xF) | ((z >> 2) & 0x30);
        m1 = ((w >> 12) & 0xF) | ((z >> 10) & 0x30);
    }
    fA = dm.x * (float) sc0;
    fB = dm.x * (float) sc1;
    gA = dm.y * (float) m0;
    gB = dm.y * (float) m1;
}

// QR: lane octet lane >> 3 reads its own row (rp per lane) and the warp walks the blocks in order, so with
// 16-row tiles one load covers the same block of 4 adjacent rows (contiguous) instead of 4 blocks of one row.
template <int T, int IT, int R, bool QR = false>
__device__ __forceinline__ void rows_q4k(const uint8_t * const (&rp)[R], const SX & s, float (&acc)[R][T]) {
    const int lane = threadIdx.x & 31;
#pragma unroll(IT > 8 ? (T >= 5 ? 1 : 4) : IT)
    for (int i = 0; i < IT; i++) {
        const int u = QR ? 8 * i + (lane & 7) : lane + 32 * i, b = u >> 3, p = (u >> 1) & 3, hh = u & 1;
        int4 h[R], q[R];
#pragma unroll
        for (int k = 0; k < R; k++) {
            const uint8_t * blk = rp[k] + b * (KQ_T * 144);
            h[k] = ldw4(blk);
            q[k] = ldw4(blk + 16 + 32 * p + 16 * hh);
        }
        int A[R][4], B[R][4];
        float fA[R], gA[R], fB[R], gB[R];
#pragma unroll
        for (int k = 0; k < R; k++) {
            q4k_scales(h[k], p, fA[k], gA[k], fB[k], gB[k]);
            A[k][0] = q[k].x & M4; A[k][1] = q[k].y & M4; A[k][2] = q[k].z & M4; A[k][3] = q[k].w & M4;
            B[k][0] = (q[k].x >> 4) & M4; B[k][1] = (q[k].y >> 4) & M4;
            B[k][2] = (q[k].z >> 4) & M4; B[k][3] = (q[k].w >> 4) & M4;
        }
        accR<T, R, (QR ? 8 : 32) * IT>(s, u, A, B, fA, gA, fB, gB, acc);
    }
}

template <int T, int IT, int R, bool QR = false>
__device__ __forceinline__ void rows_q5k(const uint8_t * const (&rp)[R], const SX & s, float (&acc)[R][T]) {
    const int lane = threadIdx.x & 31;
#pragma unroll(IT > 8 ? (T >= 5 ? 1 : 4) : IT)
    for (int i = 0; i < IT; i++) {
        const int u = QR ? 8 * i + (lane & 7) : lane + 32 * i, b = u >> 3, p = (u >> 1) & 3, hh = u & 1;
        int4 h[R], qh[R], q[R];
#pragma unroll
        for (int k = 0; k < R; k++) {
            const uint8_t * blk = rp[k] + b * (KQ_T * 176);
            h[k] = ldw4(blk);
            qh[k] = ldw4(blk + 16 + 16 * hh);
            q[k] = ldw4(blk + 48 + 32 * p + 16 * hh);
        }
        int A[R][4], B[R][4];
        float fA[R], gA[R], fB[R], gB[R];
        const int s0 = 2 * p, s1 = 2 * p + 1;
#pragma unroll
        for (int k = 0; k < R; k++) {
            q4k_scales(h[k], p, fA[k], gA[k], fB[k], gB[k]);
            A[k][0] = (q[k].x & M4) | (((qh[k].x >> s0) & 0x01010101) << 4);
            A[k][1] = (q[k].y & M4) | (((qh[k].y >> s0) & 0x01010101) << 4);
            A[k][2] = (q[k].z & M4) | (((qh[k].z >> s0) & 0x01010101) << 4);
            A[k][3] = (q[k].w & M4) | (((qh[k].w >> s0) & 0x01010101) << 4);
            B[k][0] = ((q[k].x >> 4) & M4) | (((qh[k].x >> s1) & 0x01010101) << 4);
            B[k][1] = ((q[k].y >> 4) & M4) | (((qh[k].y >> s1) & 0x01010101) << 4);
            B[k][2] = ((q[k].z >> 4) & M4) | (((qh[k].z >> s1) & 0x01010101) << 4);
            B[k][3] = ((q[k].w >> 4) & M4) | (((qh[k].w >> s1) & 0x01010101) << 4);
        }
        accR<T, R, (QR ? 8 : 32) * IT>(s, u, A, B, fA, gA, fB, gB, acc);
    }
}

// repacked Q6_K block (224 B): qlo[128] (q4k-style nibbles) | qhi[64] (8 B per unit) |
// meta[32]: for each 64-group p: sc[4p..4p+3], d (fp16), 2 pad bytes  -> one 8-byte load per unit
constexpr int Q6_BLK = 224;
template <int T, int IT, int R, bool QR = false>
__device__ __forceinline__ void rows_q6k(const uint8_t * const (&rp)[R], const SX & s, float (&acc)[R][T]) {
    const int lane = threadIdx.x & 31;
#pragma unroll(IT > 8 ? (T >= 5 ? 1 : 4) : IT)
    for (int i = 0; i < IT; i++) {
        const int u = QR ? 8 * i + (lane & 7) : lane + 32 * i, b = u >> 3, p = (u >> 1) & 3, hh = u & 1;
        int4 q[R];
        int2 hb[R], meta[R];
#pragma unroll
        for (int k = 0; k < R; k++) {
            const uint8_t * blk = rp[k] + b * (KQ_T * Q6_BLK);
            q[k] = ldw4(blk + 32 * p + 16 * hh);
            hb[k] = ldw2(blk + 128 + 8 * (u & 7));
            meta[k] = ldw2(blk + 192 + 8 * p);
        }
        int A[R][4], B[R][4];
        float fA[R], gA[R], fB[R], gB[R];
#pragma unroll
        for (int k = 0; k < R; k++) {
            const float d = __half2float(__ushort_as_half((unsigned short) (meta[k].y & 0xffff)));
            fA[k] = d * (float) (int8_t) (meta[k].x >> (8 * hh));
            fB[k] = d * (float) (int8_t) (meta[k].x >> (16 + 8 * hh));
            gA[k] = 32.0f * fA[k];
            gB[k] = 32.0f * fB[k];
            const unsigned hx = (unsigned) hb[k].x, hy = (unsigned) hb[k].y;
            A[k][0] = (q[k].x & M4) | (int) (((hx >> 0) & 0x03030303u) << 4);
            A[k][1] = (q[k].y & M4) | (int) (((hx >> 2) & 0x03030303u) << 4);
            A[k][2] = (q[k].z & M4) | (int) (((hx >> 4) & 0x03030303u) << 4);
            A[k][3] = (q[k].w & M4) | (int) (((hx >> 6) & 0x03030303u) << 4);
            B[k][0] = ((q[k].x >> 4) & M4) | (int) (((hy >> 0) & 0x03030303u) << 4);
            B[k][1] = ((q[k].y >> 4) & M4) | (int) (((hy >> 2) & 0x03030303u) << 4);
            B[k][2] = ((q[k].z >> 4) & M4) | (int) (((hy >> 4) & 0x03030303u) << 4);
            B[k][3] = ((q[k].w >> 4) & M4) | (int) (((hy >> 6) & 0x03030303u) << 4);
        }
        accR<T, R, (QR ? 8 : 32) * IT>(s, u, A, B, fA, gA, fB, gB, acc);
    }
}

// repacked Q8_0 (16-row tiles, q8_tile.h): per unit 32 B (A 16 int8 | B 16 int8), scales per natural 32-block
template <int T, int IT, int R>
__device__ __forceinline__ void rows_q80(const uint8_t * const (&rp)[R], const __half * const (&dp)[R], const SX & s,
                                         float (&acc)[R][T]) {
    const int lane = threadIdx.x & 31;
#pragma unroll(IT > 8 ? (T >= 5 ? 1 : 4) : IT)
    for (int i = 0; i < IT; i++) {
        const int u = lane + 32 * i;
        int A[R][4], B[R][4];
        float fA[R], gA[R], fB[R], gB[R];
#pragma unroll
        for (int k = 0; k < R; k++) {
            const uint8_t * p = rp[k] + Q8_GW * (u >> 1) + 32 * (u & 1);
            const int4 a = ldw4(p), bq = ldw4(p + 16);
            const int dd = ldw1(dp[k] + Q8_GD * (u >> 1));
            const float2 df = __half22float2(*reinterpret_cast<const __half2 *>(&dd));
            A[k][0] = a.x; A[k][1] = a.y; A[k][2] = a.z; A[k][3] = a.w;
            B[k][0] = bq.x; B[k][1] = bq.y; B[k][2] = bq.z; B[k][3] = bq.w;
            fA[k] = df.x; fB[k] = df.y; gA[k] = 0.f; gB[k] = 0.f;
        }
        accR<T, R, 32 * IT>(s, u, A, B, fA, gA, fB, gB, acc);
    }
}

// RG (8 or 4) consecutive rows of a Q8_0 tile per warp (NM matrices sharing the row index): lane (g, c),
// g = lane / (32 / RG), reads 16-chunk c & 3 of row g's 64-value group q + (c >> 2), so each load
// instruction is 512 bytes in 64 * RG contiguous pieces (one row per warp reads 32 B pieces 1 KB apart,
// which re-opens the tile's DRAM pages per row).  Chunk c & 3 is the A (even) or B half of unit
// 2 (q + (c >> 2)) + ((c & 3) >> 1); acc[m][t] is row g's sum after the reduction over c.  RG = 4 doubles
// the items of a phase, for phases with too few 8-row items to keep every warp loading.
template <int T, int NM, int UU, int RG = 8>
__device__ __forceinline__ void rows8_q80(const uint8_t * const (&wp)[NM], const __half * const (&dp)[NM], const SX & s,
                                          float (&acc)[NM][T], int q0 = 0, int q1 = UU / 2) {
    constexpr int LP = 32 / RG, GL = LP / 4, PD = 4; // q1 - q0 must be a multiple of PD * GL
    static_assert(RG == 8 || RG == 4, "rows8_q80 width");
    const int lane = threadIdx.x & 31, c = lane & (LP - 1), part = c & 1, cq = c >> 2, ch = c & 3;
    const int4 * xs = part ? s.xb : s.xa;
#pragma unroll 1
    for (int q = q0 + cq; q < q1; q += PD * GL) {
        int4 w[PD][NM];
        float dw[PD][NM];
#pragma unroll
        for (int p = 0; p < PD; p++)
#pragma unroll
            for (int m = 0; m < NM; m++) {
                w[p][m] = ldw4(wp[m] + Q8_GW * (q + GL * p) + 16 * ch);
                const int dd = ldw1(dp[m] + Q8_GD * (q + GL * p));
                const float2 df = __half22float2(*reinterpret_cast<const __half2 *>(&dd));
                dw[p][m] = part ? df.y : df.x;
            }
#pragma unroll
        for (int p = 0; p < PD; p++) {
            const int u = 2 * (q + GL * p) + (ch >> 1);
#pragma unroll
            for (int t = 0; t < T; t++) {
                const int4 v = xs[t * UU + u];
                const uint2 mm = s.xm[t * UU + u];
                const float2 dx2 = __half22float2(*reinterpret_cast<const __half2 *>(&mm.x));
                const float dx = part ? dx2.y : dx2.x;
#pragma unroll
                for (int m = 0; m < NM; m++) {
                    int sd = __dp4a(w[p][m].x, v.x, 0);
                    sd = __dp4a(w[p][m].y, v.y, sd);
                    sd = __dp4a(w[p][m].z, v.z, sd);
                    sd = __dp4a(w[p][m].w, v.w, sd);
                    acc[m][t] += dw[p][m] * (dx * (float) sd);
                }
            }
        }
    }
#pragma unroll
    for (int m = 0; m < NM; m++)
#pragma unroll
        for (int t = 0; t < T; t++) {
            acc[m][t] += __shfl_xor_sync(0xffffffffu, acc[m][t], 1);
            acc[m][t] += __shfl_xor_sync(0xffffffffu, acc[m][t], 2);
            if constexpr (RG == 4) acc[m][t] += __shfl_xor_sync(0xffffffffu, acc[m][t], 4);
        }
}

// rows row0 .. row0+R-1 of m, weight type fixed at compile time
// u0: first unit of a K-slice (multiple of 32) when the activations in s hold only that slice
template <int TYPE>
__device__ __forceinline__ int unit_bytes(int u0) {
    return TYPE == T_Q8_0 ? 32 * u0 : (u0 >> 3) * KQ_T * kq_bb(TYPE);
}
// One entry for the per-type row decoders; dp (Q8_0 scales) unused for the other types.
template <int T, int IT, int R, int TYPE, bool QR = false>
__device__ __forceinline__ void rows_any(const uint8_t * const (&rp)[R], const __half * const (&dp)[R],
                                         const SX & s, float (&acc)[R][T]) {
    if constexpr (TYPE == T_Q4K) rows_q4k<T, IT, R, QR>(rp, s, acc);
    else if constexpr (TYPE == T_Q5K) rows_q5k<T, IT, R, QR>(rp, s, acc);
    else if constexpr (TYPE == T_Q6K) rows_q6k<T, IT, R, QR>(rp, s, acc);
    else rows_q80<T, IT, R>(rp, dp, s, acc);
}

template <int T, int IT, int R, int TYPE>
__device__ __forceinline__ void mrows(const Mat & m, int row0, const SX & s, float (&acc)[R][T], int u0 = 0) {
    const uint8_t * rp[R];
    const __half * dp[R];
#pragma unroll
    for (int k = 0; k < R; k++) {
        rp[k] = TYPE == T_Q8_0 ? m.w + q8_wo(row0 + k, m.rb) + Q8_GW / 2 * u0 : m.w + kq_wo(row0 + k, m.rb, kq_bb(TYPE)) + unit_bytes<TYPE>(u0);
        dp[k] = TYPE == T_Q8_0 ? m.d + q8_do(row0 + k, m.nd) + Q8_GD / 2 * u0 : m.d + (size_t) (row0 + k) * m.nd + u0;
    }
    rows_any<T, IT, R, TYPE>(rp, dp, s, acc);
}

// Noinline so the type switch's register cost stays out of the callers (cf. gemv_pairs_t).
template <int T, int IT, int R>
__device__ __noinline__ void mrows_t(int type, const Mat & m, int row0, const SX & s, float (&acc)[R][T], int u0 = 0) {
    switch (type) {
    case T_Q4K: mrows<T, IT, R, T_Q4K>(m, row0, s, acc, u0); break;
    case T_Q5K: mrows<T, IT, R, T_Q5K>(m, row0, s, acc, u0); break;
    case T_Q6K: mrows<T, IT, R, T_Q6K>(m, row0, s, acc, u0); break;
    default: mrows<T, IT, R, T_Q8_0>(m, row0, s, acc, u0); break;
    }
}

// Row pairs [lo, hi) of a single-matrix item range, distributed over all warps of the grid; tot
// ends the phase's whole item chain.  out(row0, acc, nr) receives the reduced sums for rows
// row0 .. row0 + nr - 1 (nr 1 or 2).
template <int T, int IT, int TYPE, class Out>
__device__ __noinline__ void gemv_pairs(const Mat m, int lo, int hi, int tot, Out out) {
    const SX sx = make_sx(nullptr, 32 * IT); // activations already in shared memory
    const int W = nwarps(), gw = gwarp();
    if constexpr (TYPE == T_Q8_0) {
        // 4-row items (rows8_q80) when the range is whole row quads
        if ((hi - lo) % 2 == 0) {
            const int lane = threadIdx.x & 31, g = lane >> 3;
            for (int it = lo + (((gw - lo) % W) + W) % W; it < lo + (hi - lo) / 2; it += W) {
                const int r0 = 4 * (it - lo);
                const uint8_t * wp[1] = {m.w + q8_wo(r0 + g, m.rb)};
                const __half * dp[1] = {m.d + q8_do(r0 + g, m.nd)};
                float a4[1][T];
                zero_acc<T, 1>(a4);
                rows8_q80<T, 1, 32 * IT, 4>(wp, dp, sx, a4);
#pragma unroll
                for (int p = 0; p < 2; p++) {
                    float a[2][T];
#pragma unroll
                    for (int t = 0; t < T; t++) {
                        a[0][t] = __shfl_sync(0xffffffffu, a4[0][t], 16 * p);
                        a[1][t] = __shfl_sync(0xffffffffu, a4[0][t], 16 * p + 8);
                    }
                    out(r0 + 2 * p, a, 2);
                }
            }
            return;
        }
    }
    // A last partial round of at most W/2 pairs runs as single rows on twice the warps, which
    // halves the phase's tail (the other warps are idle then anyway).
    const int tail = (2 * (tot % W) <= W) ? tot - tot % W : tot;
    const int first = lo + (((gw - lo) % W) + W) % W;
    for (int it = first; it < min(hi, max(lo, tail)); it += W) {
        const int r = 2 * (it - lo);
        float a[2][T];
        zero_acc<T, 2>(a);
        mrows<T, IT, 2, TYPE>(m, r, sx, a);
        wsum_acc<T, 2>(a);
        out(r, a, 2);
    }
    const int it = tail + (gw >> 1);
    if (gw < 2 * (tot - tail) && it >= lo && it < hi) {
        const int r = 2 * (it - lo) + (gw & 1);
        float b[1][T];
        zero_acc<T, 1>(b);
        mrows<T, IT, 1, TYPE>(m, r, sx, b);
        wsum_acc<T, 1>(b);
        float a[2][T];
#pragma unroll
        for (int t = 0; t < T; t++) a[0][t] = a[1][t] = b[0][t];
        out(r, a, 1);
    }
}

// Noinline dispatch so the type switch's register cost stays out of the callers.
template <int T, int IT, class Out>
__device__ __noinline__ void gemv_pairs_t(int type, const Mat m, int lo, int hi, int tot, Out out) {
    switch (type) {
    case T_Q4K: gemv_pairs<T, IT, T_Q4K>(m, lo, hi, tot, out); break;
    case T_Q5K: gemv_pairs<T, IT, T_Q5K>(m, lo, hi, tot, out); break;
    case T_Q6K: gemv_pairs<T, IT, T_Q6K>(m, lo, hi, tot, out); break;
    default: gemv_pairs<T, IT, T_Q8_0>(m, lo, hi, tot, out); break;
    }
}

// Prefetch the first `bytes` of rows row0..row0+n-1 into L2 so the DRAM stream of a phase starts
// while its activations are still being loaded into shared memory.
__device__ __forceinline__ void pf_rows(const Mat & m, int row0, int n, int bytes) {
    const int lane = threadIdx.x & 31;
    for (int k = 0; k < n; k++) {
        if (m.type == T_Q8_0) {
            const uint8_t * p = m.w + q8_wo(row0 + k, m.rb);
            for (int q = lane; q < bytes / 64; q += 32) asm volatile("prefetch.global.L2 [%0];" ::"l"(p + Q8_GW * q));
            continue;
        }
        const int bb = kq_bb(m.type);
        const uint8_t * p = m.w + kq_wo(row0 + k, m.rb, bb);
        for (int off = lane * 128; off < bytes; off += 32 * 128)
            asm volatile("prefetch.global.L2 [%0];" ::"l"(p + KQ_T * bb * (off / bb) + off % bb));
    }
}

// First weight lines of a phase's first item.  main_forward also issues them before the
// preceding grid barrier, so DRAM streams while the barrier and the activation fill complete.
__device__ __forceinline__ void pf_resid(const Mat & W, int it) {
    if (blockIdx.x < D / 32) pf_rows(W, 32 * blockIdx.x + 2 * (threadIdx.x >> 5), 2, W.rb / it);
}
__device__ __forceinline__ void pf_ffn_gu(const LayerW & W) {
#if MK_TP > 1
    const int row = 16 * ((threadIdx.x >> 6) * gridDim.x + blockIdx.x) + 8 * ((threadIdx.x >> 5) & 1);
#else
    const int row = 16 * gwarp();
#endif
    if (row < FF_L) {
        pf_rows(W.fg, row, 2, 576);
        pf_rows(W.fu, row, 2, 576);
    }
}
__device__ __forceinline__ void pf_attn_proj(const LayerW & W) {
    if (gwarp() < QG_L / 2) pf_rows(W.q, 2 * gwarp(), 2, 576);
}
__device__ __forceinline__ void pf_dn_proj(const LayerW & W) {
    const int gw = gwarp(), W_ = nwarps(), lo = AB_SPLIT * 2 * NVH_L;
    const int first = lo + (((gw - lo) % W_) + W_) % W_;
    if (first < lo + QKV_L / 2) pf_rows(W.qkv, 2 * (first - lo), 2, 1024);
}

// lanes 0-15 quantise 16-chunk c of row t, lanes 16-31 the same chunk of row t + 1 (if valid)
__device__ __forceinline__ void q16_store(const QBuf & q, int t, int c, float v, bool valid) {
    const int lane = threadIdx.x & 31;
    float amax = fabsf(v);
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 8));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    const float d = amax / 127.0f;
    const int qi = amax == 0.0f ? 0 : (int) roundf(v / d);
    int qs = qi;
    qs += __shfl_xor_sync(0xffffffffu, qs, 8);
    qs += __shfl_xor_sync(0xffffffffu, qs, 4);
    qs += __shfl_xor_sync(0xffffffffu, qs, 2);
    qs += __shfl_xor_sync(0xffffffffu, qs, 1);
    if (!valid) return;
    const int b = c >> 4, p = (c >> 2) & 3, part = (c >> 1) & 1, hh = c & 1, byte = lane & 15;
    const int gi = t * q.U + 8 * b + 2 * p + hh;
    (part ? q.xb : q.xa)[gi * 16 + byte] = (int8_t) qi;
    if (byte == 0) {
        reinterpret_cast<__half *>(q.meta + gi)[part] = __float2half_rn(d);
        reinterpret_cast<short *>(q.meta + gi)[2 + part] = (short) qs;
    }
}

template <int T>
__device__ __forceinline__ void wsum_all(float (&a)[T]) {
#pragma unroll
    for (int t = 0; t < T; t++) a[t] = wsum(a[t]);
}

// ------------------------------------------------------------------------------------------
// phases
// ------------------------------------------------------------------------------------------

__device__ __forceinline__ float dequant_q4k(const uint8_t * row, int e) {
    const int b = e >> 8, w = e & 255, sb = w >> 5, l = w & 31;
    const uint8_t * blk = row + b * 144;
    const float d = __half2float(*reinterpret_cast<const __half *>(blk));
    const float dmin = __half2float(*reinterpret_cast<const __half *>(blk + 2));
    const uint8_t * sc = blk + 4;
    int s, m;
    if (sb < 4) {
        s = sc[sb] & 63;
        m = sc[sb + 4] & 63;
    } else {
        s = (sc[sb + 4] & 0xF) | ((sc[sb - 4] >> 6) << 4);
        m = (sc[sb + 4] >> 4) | ((sc[sb] >> 6) << 4);
    }
    const uint8_t qb = blk[16 + 32 * (sb >> 1) + l];
    const int qv = (sb & 1) ? (qb >> 4) : (qb & 0xF);
    return d * (float) s * (float) qv - dmin * (float) m;
}

// one element of a raw GGUF row for the four embedding types; math mirrors ggml-quants.c
__device__ __forceinline__ float dequant_elem(const uint8_t * row, int type, int e) {
    if (type == 8) { // Q8_0: 34 B per 32
        const uint8_t * blk = row + (e >> 5) * 34;
        const float d = __half2float(*reinterpret_cast<const __half *>(blk));
        return d * (float) (int8_t) blk[2 + (e & 31)];
    }
    if (type == 12) return dequant_q4k(row, e);
    if (type == 13) { // Q5_K: 176 B per 256, 5th bit stored in qh
        const int b = e >> 8, w = e & 255, g = w >> 6, l = w & 63, lo = l < 32;
        const uint8_t * blk = row + b * 176;
        const float d = __half2float(*reinterpret_cast<const __half *>(blk));
        const float dmin = __half2float(*reinterpret_cast<const __half *>(blk + 2));
        const uint8_t * sc = blk + 4;
        const int sb = 2 * g + !lo;
        int s, m;
        if (sb < 4) {
            s = sc[sb] & 63;
            m = sc[sb + 4] & 63;
        } else {
            s = (sc[sb + 4] & 0xF) | ((sc[sb - 4] >> 6) << 4);
            m = (sc[sb + 4] >> 4) | ((sc[sb] >> 6) << 4);
        }
        const int qb = blk[48 + 32 * g + (l & 31)];
        const int q = (lo ? qb & 0xF : qb >> 4) + ((blk[16 + (l & 31)] >> (2 * g + !lo) & 1) ? 16 : 0);
        return d * (float) s * (float) q - dmin * (float) m;
    }
    // Q6_K: 210 B per 256, ql/qh low+high bits, int8 scales
    const int b = e >> 8, n = e & 255, l = n & 31, t = n >> 5, h = n >> 7;
    const uint8_t * blk = row + b * 210;
    const float d = __half2float(*reinterpret_cast<const __half *>(blk + 208));
    const int8_t * sc = reinterpret_cast<const int8_t *>(blk + 192) + h * 8 + (l >> 4) + 2 * t;
    const int qb = blk[h * 64 + (t & 1) * 32 + l];
    const int q = ((t < 2 ? qb & 0xF : qb >> 4) | (((blk[128 + h * 32 + l] >> (2 * t)) & 3) << 4)) - 32;
    return d * (float) *sc * (float) q;
}

// token embedding rows -> x, plus quantised attn_norm input for layer 0
// token_embd row -> shared memory in one coalesced read: the table lives in mapped host memory, where every
// dependent access pays a PCIe round trip (row bytes of all four types are multiples of 8)
__device__ __forceinline__ const uint8_t * stage_embd_row(int tok, uint8_t * dst) {
    const uint8_t * src = cW.embd + (size_t) tok * cW.embd_rb;
    const int rb = cW.embd_rb;
    if (!(rb & 15))
        for (int i = threadIdx.x; i < rb / 16; i += blockDim.x) reinterpret_cast<uint4 *>(dst)[i] = reinterpret_cast<const uint4 *>(src)[i];
    else
        for (int i = threadIdx.x; i < rb / 8; i += blockDim.x) reinterpret_cast<uint2 *>(dst)[i] = reinterpret_cast<const uint2 *>(src)[i];
    __syncthreads();
    return dst;
}
constexpr int EMBD_RB_MAX = D / 32 * 34; // Q8_0, the widest embedding row

template <int T>
__device__ __noinline__ void embed_prologue() {
    const int t = blockIdx.x;
    if (t >= T) return;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int tok = ((volatile Ctl *) sqb().ctl)->tok[t];
    const uint8_t * row = stage_embd_row(tok, smem);
    const float * w0 = cW.L[0].attn_norm;
    for (int j = warp; j < U_D; j += NW) {
        const int e = 32 * j + lane;
        const float v = dequant_elem(row, cW.embd_type, e);
        cB.x[t * D + e] = v;
        q32_store(cB.qn, t, j, v * w0[e], v * v, true);
    }
}

// Output rows (32 consecutive rows per block item) with optional residual add; the new
// residual value is re-quantised for the next RMS-normed consumer (weight wnext).  k numbers
// this reduction within the launch (< 4096, packed into the all-reduce flag tag).
template <int T, int IT, int TYPE>
__device__ __noinline__ void resid32(const Mat W, const QBuf in, float * x, bool add, const float * wnext, uint8_t * /*sm*/, int k) {
    uint8_t * const sm = smem;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    float * s_rs = misc(sm);
    float * s_res = misc(sm) + 16;
    pf_resid(W, IT);
    if (threadIdx.x < 4 && blockIdx.x + threadIdx.x * gridDim.x < D / 32) pf_l2(wnext + 32 * (blockIdx.x + threadIdx.x * gridDim.x));
    auto epilogue = [&](int j, float (&a)[2][T]) {
        wsum_acc<T, 2>(a);
        if (lane == 0) {
#pragma unroll
            for (int t = 0; t < T; t++) {
                s_res[t * 32 + 2 * warp] = a[0][t];
                s_res[t * 32 + 2 * warp + 1] = a[1][t];
            }
        }
        __syncthreads();
#if MK_TP > 1
        // K-sharded weights leave s_res a partial; all-reduce it before it touches x, summed
        // p0 + p1 in rank order so both cards get bit-identical values
        {
            const __nv_bfloat16 * peer = cB.xr[k & 1] + j * MAXR * 32;
            xr_put(j, k, s_res, T * 32);
            xr_wait(j, k);
            for (int i = threadIdx.x; i < T * 32; i += NT)
                s_res[i] = xr_sum(s_res[i], peer, i);
            __syncthreads();
        }
#endif
        if (warp < T) {
            const int t = warp, row = 32 * j + lane;
            float xv = s_res[t * 32 + lane];
            if (add) xv += __ldcg(x + t * D + row);
            x[t * D + row] = xv;
            q32_store(cB.qn, t, j, xv * wnext[row], xv * xv, true);
        }
        __syncthreads();
    };
    static_assert(MK_TP == 1 || T * 32 * IT <= SM_XU, "TP=2 must not take the K-half split");
    if constexpr (MK_TP == 1 && T * 32 * IT > SM_XU) {
        // The activations do not fit (T = 5 ffn_down): fill one K-half at a time.  A block's two row
        // groups are processed one after the other; the first group's per-lane partial sums of the
        // first half are parked in shared memory while the second group is computed.
        constexpr int U0 = 32 * ((IT + 1) / 2), U1 = 32 * IT - U0;
        static_assert(T * U0 * 40 + 2 * T * NT * 4 <= SM_MISC, "T = 5 ffn_down shared memory");
        auto sxh = [&](int US) {
            SX r;
            r.xa = reinterpret_cast<int4 *>(sm);
            r.xb = reinterpret_cast<int4 *>(sm + T * US * 16);
            r.xm = reinterpret_cast<uint2 *>(sm + 2 * T * US * 16);
            r.U = US;
            return r;
        };
        float * park = reinterpret_cast<float *>(sm + T * U0 * 40); // [2 * T][NT]
        const SX s0 = sxh(U0), s1 = sxh(U1);
        for (int j0 = blockIdx.x; j0 < D / 32; j0 += 2 * gridDim.x) {
            const int j1 = j0 + gridDim.x;
            fill_xs_part<T, U0>(in, 0, s0);
            if (j0 == blockIdx.x) mark_fill();
            float a[2][T];
            zero_acc<T, 2>(a);
            mrows<T, U0 / 32, 2, TYPE>(W, 32 * j0 + 2 * warp, s0, a);
#pragma unroll
            for (int t = 0; t < T; t++) {
                park[t * NT + threadIdx.x] = a[0][t];
                park[(T + t) * NT + threadIdx.x] = a[1][t];
            }
            if (j1 < D / 32) {
                zero_acc<T, 2>(a);
                mrows<T, U0 / 32, 2, TYPE>(W, 32 * j1 + 2 * warp, s0, a);
            }
            fill_xs_part<T, U1>(in, U0, s1);
            if (j1 < D / 32) {
                mrows<T, U1 / 32, 2, TYPE>(W, 32 * j1 + 2 * warp, s1, a, U0);
                epilogue(j1, a);
            }
#pragma unroll
            for (int t = 0; t < T; t++) {
                a[0][t] = park[t * NT + threadIdx.x];
                a[1][t] = park[(T + t) * NT + threadIdx.x];
            }
            mrows<T, U1 / 32, 2, TYPE>(W, 32 * j0 + 2 * warp, s1, a, U0);
            epilogue(j0, a);
        }
    } else {
        const SX sx = make_sx(sm, in.U);
        fill_xs<T>(in, 0, false, sx, s_rs);
#if MK_TP > 1
        // Send every item's partial before waiting on any, so the PCIe round trips overlap the
        // remaining rows instead of serialising per item; partials park past the activations.
        const int park_off = (T * in.U * 16 + 15) & ~15;
        const int n_items = (D / 32 - blockIdx.x + gridDim.x - 1) / gridDim.x;
        const bool tail = n_items * T * 32 * 4 <= SM_XU * 16 - park_off;
        float * park = reinterpret_cast<float *>(sm + (tail ? SM_XA + park_off : SM_PARK));
        if (tail || n_items * T * 32 * 4 <= 4096) {
            for (int i = 0; i < n_items; i++) {
                const int j = blockIdx.x + i * gridDim.x;
                float * pk = park + i * T * 32;
                if constexpr (TYPE == T_Q8_0) {
                    // warp = (K-quarter kp, tile half h8) of the item's 32 rows (rows8_q80); the quarters add
                    // into pk in a fixed order
                    constexpr int QP = IT * 16 / 4;
                    static_assert(QP % 4 == 0, "resid32 K-quarters");
                    const int h8 = warp & 3, kp = warp >> 2, rr = 8 * h8 + (lane >> 2);
                    const uint8_t * wp[1] = {W.w + q8_wo(32 * j + rr, W.rb)};
                    const __half * dp[1] = {W.d + q8_do(32 * j + rr, W.nd)};
                    float a8[1][T];
                    zero_acc<T, 1>(a8);
                    rows8_q80<T, 1, 32 * IT>(wp, dp, sx, a8, kp * QP, (kp + 1) * QP);
#pragma unroll 1
                    for (int p = 0; p < 4; p++) {
                        if (kp == p && (lane & 3) == 0) {
#pragma unroll
                            for (int t = 0; t < T; t++) pk[t * 32 + rr] = p ? pk[t * 32 + rr] + a8[0][t] : a8[0][t];
                        }
                        __syncthreads();
                    }
                } else {
                    float a[2][T];
                    zero_acc<T, 2>(a);
                    mrows<T, IT, 2, TYPE>(W, 32 * j + 2 * warp, sx, a);
                    wsum_acc<T, 2>(a);
                    if (lane == 0) {
#pragma unroll
                        for (int t = 0; t < T; t++) {
                            pk[t * 32 + 2 * warp] = a[0][t];
                            pk[t * 32 + 2 * warp + 1] = a[1][t];
                        }
                    }
                    __syncthreads();
                }
                xr_put(j, k, pk, T * 32);
            }
            for (int i = 0; i < n_items; i++) {
                const int j = blockIdx.x + i * gridDim.x;
                const float * pk = park + i * T * 32;
                const __nv_bfloat16 * peer = cB.xr[k & 1] + j * MAXR * 32;
                xr_wait(j, k);
                if (warp < T) {
                    const int t = warp, row = 32 * j + lane;
                    float xv = xr_sum(pk[t * 32 + lane], peer, t * 32 + lane);
                    if (add) xv += __ldcg(x + t * D + row);
                    x[t * D + row] = xv;
                    q32_store(cB.qn, t, j, xv * wnext[row], xv * xv, true);
                }
            }
            return;
        }
#endif
        for (int j = blockIdx.x; j < D / 32; j += gridDim.x) {
            float a[2][T];
            zero_acc<T, 2>(a);
            mrows<T, IT, 2, TYPE>(W, 32 * j + 2 * warp, sx, a);
            epilogue(j, a);
        }
    }
}

// Noinline dispatch so the type switch's register cost stays out of the callers.
template <int T, int IT>
__device__ __noinline__ void resid32_t(int type, const Mat W, const QBuf in, float * x, bool add, const float * wnext, uint8_t * sm, int k) {
    switch (type) {
    case T_Q4K: resid32<T, IT, T_Q4K>(W, in, x, add, wnext, sm, k); break;
    case T_Q5K: resid32<T, IT, T_Q5K>(W, in, x, add, wnext, sm, k); break;
    case T_Q6K: resid32<T, IT, T_Q6K>(W, in, x, add, wnext, sm, k); break;
    default: resid32<T, IT, T_Q8_0>(W, in, x, add, wnext, sm, k); break;
    }
}

// FFN gate/up: warp item = 16 consecutive rows of both matrices (4 rows in flight sharing the
// activations) -> silu(g)*u quantised per 16-chunk for the down projection.
template <int T, int TYPE>
__device__ __noinline__ void ffn_gu_q(const LayerW & W, uint8_t * /*sm*/) {
    uint8_t * const sm = smem;
    const int lane = threadIdx.x & 31;
    const SX sx = make_sx(sm, U_D);
    const float * s_rs = misc(sm);
    pf_ffn_gu(W);
    fill_xs<T>(cB.qn, 0, true, sx, misc(sm));
    const uint8_t * gw = W.fg.w;
    const uint8_t * uw = W.fu.w;
    const __half * gd = W.fg.d;
    const __half * ud = W.fu.d;
    const int rb = W.fg.rb;
    const int nd = W.fg.nd;
#if MK_TP > 1
    // A card holds too few 16-row items to busy every warp, so a warp pair splits each item
    // (8 rows each) and the lower warp quantises the 16-chunk from both halves.
    constexpr int TS = T > MAXT ? T : MAXT; // exchange rows (MAXT keeps the single-sequence layout)
    static_assert(TS * U_D * 16 + NW * TS * 16 * 4 <= SM_XU * 16, "ffn_gu pair exchange buffer");
    float * s_act = reinterpret_cast<float *>(sm + SM_XA + TS * U_D * 16) + (threadIdx.x >> 6) * 2 * TS * 16;
    const int half = (threadIdx.x >> 5) & 1, bar_id = 1 + (threadIdx.x >> 6);
    int buf = 0;
    for (int j = (threadIdx.x >> 6) * gridDim.x + blockIdx.x; j < FF_L / 16; j += nwarps() / 2, buf ^= 1) {
        float act[T];
#pragma unroll
        for (int t = 0; t < T; t++) act[t] = 0.f;
        if constexpr (TYPE == T_Q8_0) {
            const int row = 16 * j + 8 * half + (lane >> 2);
            const uint8_t * wp[2] = {gw + q8_wo(row, rb), uw + q8_wo(row, rb)};
            const __half * dp[2] = {gd + q8_do(row, nd), ud + q8_do(row, nd)};
            float a[2][T];
            zero_acc<T, 2>(a);
            rows8_q80<T, 2, U_D>(wp, dp, sx, a);
#pragma unroll
            for (int t = 0; t < T; t++) act[t] = silu(a[0][t] * s_rs[t]) * (a[1][t] * s_rs[t]);
        }
        // k-quants: 4 rows per step, lane octet o on row r + o (QR), so loads stay contiguous in the 16-row tiles
#pragma unroll 1
        for (int r = 0; r < (TYPE == T_Q8_0 ? 0 : 8); r += 4) {
            const int row = 16 * j + 8 * half + r + (lane >> 3);
            const size_t wo = kq_wo(row, rb, kq_bb(TYPE));
            const uint8_t * rp[2] = {gw + wo, uw + wo};
            const __half * dp[2] = {gd, ud};
            float a[2][T];
            zero_acc<T, 2>(a);
            rows_any<T, 20, 2, TYPE, true>(rp, dp, sx, a);
#pragma unroll
            for (int k = 0; k < 2; k++)
#pragma unroll
                for (int t = 0; t < T; t++)
#pragma unroll
                    for (int o = 1; o < 8; o <<= 1) a[k][t] += __shfl_xor_sync(0xffffffffu, a[k][t], o);
#pragma unroll
            for (int t = 0; t < T; t++) {
                const float g = __shfl_sync(0xffffffffu, a[0][t], 8 * ((lane - r) & 3));
                const float u = __shfl_sync(0xffffffffu, a[1][t], 8 * ((lane - r) & 3));
                if (lane >= r && lane < r + 4) act[t] = silu(g * s_rs[t]) * (u * s_rs[t]);
            }
        }
        float * sa = s_act + buf * TS * 16;
        if (TYPE == T_Q8_0 ? (lane & 3) == 0 : lane < 8) {
#pragma unroll
            for (int t = 0; t < T; t++) sa[t * 16 + 8 * half + (TYPE == T_Q8_0 ? lane >> 2 : lane)] = act[t];
        }
        // double-buffered sa: the next write to this buffer comes after the next pair barrier
        asm volatile("bar.sync %0, 64;" ::"r"(bar_id) : "memory");
        if (half == 0) {
#pragma unroll
            for (int t = 0; t < T; t += 2) {
                const int tt = lane < 16 ? t : (t + 1 < T ? t + 1 : t);
                q16_store(cB.qa, tt, j, sa[tt * 16 + (lane & 15)], lane < 16 || t + 1 < T);
            }
        }
    }
#else
    for (int j = gwarp(); j < FF_L / 16; j += nwarps()) {
        float act[T];
#pragma unroll
        for (int t = 0; t < T; t++) act[t] = 0.f;
        constexpr int RG = (T >= 3 || TYPE != T_Q4K) ? 1 : 2; // rows per matrix per step (fewer at large T or non-Q4K: register budget)
#pragma unroll 1
        for (int r = 0; r < 16; r += RG) {
            const int row = 16 * j + r;
            const uint8_t * rp[2 * RG];
            const __half * dp[2 * RG];
#pragma unroll
            for (int k = 0; k < RG; k++) {
                const size_t wo = TYPE == T_Q8_0 ? q8_wo(row + k, rb) : kq_wo(row + k, rb, kq_bb(TYPE));
                const size_t dq = TYPE == T_Q8_0 ? q8_do(row + k, nd) : (size_t) (row + k) * nd;
                rp[k] = gw + wo;
                rp[RG + k] = uw + wo;
                dp[k] = gd + dq;
                dp[RG + k] = ud + dq;
            }
            float a[2 * RG][T];
            zero_acc<T, 2 * RG>(a);
            rows_any<T, 5, 2 * RG, TYPE>(rp, dp, sx, a);
            wsum_acc<T, 2 * RG>(a);
#pragma unroll
            for (int k = 0; k < RG; k++)
                if (lane == r + k) {
#pragma unroll
                    for (int t = 0; t < T; t++) act[t] = silu(a[k][t] * s_rs[t]) * (a[RG + k][t] * s_rs[t]);
                }
        }
#pragma unroll
        for (int t = 0; t < T; t += 2) {
            const float other = __shfl_sync(0xffffffffu, act[t + 1 < T ? t + 1 : t], lane & 15);
            const float v = lane < 16 ? act[t] : other;
            q16_store(cB.qa, lane < 16 ? t : t + 1, j, v, lane < 16 || t + 1 < T);
        }
    }
#endif
}

// Noinline dispatch that reads the gate type itself so the load stays out of the inlined callers' register budget.
template <int T>
__device__ __noinline__ void ffn_gu(const LayerW & W, uint8_t * sm) {
    const int type = W.fg.type;
    switch (type) {
    case T_Q4K: ffn_gu_q<T, T_Q4K>(W, sm); break;
    case T_Q5K: ffn_gu_q<T, T_Q5K>(W, sm); break;
    case T_Q6K: ffn_gu_q<T, T_Q6K>(W, sm); break;
    default: ffn_gu_q<T, T_Q8_0>(W, sm); break;
    }
}

template <int T>
__device__ __forceinline__ void store_lane_t(float * base, int stride, int idx, const float (&a)[T]) {
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int t = 0; t < T; t++)
        if (lane == t) base[t * stride + idx] = a[t];
}

// a[k][t] -> base[t * stride + idx + k]
template <int T>
__device__ __forceinline__ void store_pair_t(float * base, int stride, int idx, const float (&a)[2][T], int nr) {
    const int lane = threadIdx.x & 31;
    const float * s_rs = misc(nullptr);
#pragma unroll
    for (int t = 0; t < T; t++)
        if ((lane >> 1) == t && (lane & 1) < nr) base[t * stride + idx + (lane & 1)] = ((lane & 1) ? a[1][t] : a[0][t]) * s_rs[t];
}
// gemv_pairs output: rows -> base (one type for every caller, so the GEMV instances are shared)
template <int T>
struct StorePair {
    float * base;
    int stride;
    __device__ __forceinline__ void operator()(int r, const float (&a)[2][T], int nr) const { store_pair_t<T>(base, stride, r, a, nr); }
};

// delta-net input projections: alpha/beta (fp32), qkv -> conv ring, z
template <int T>
__device__ __noinline__ void dn_proj(const LayerW & W, int dl, int n, int dn_c, uint8_t * /*sm*/) {
    uint8_t * const sm = smem;
    const int lane = threadIdx.x & 31;
    const SX sx = make_sx(sm, U_D);
    float * s_rs = misc(sm);
    pf_dn_proj(W);
    fill_xs<T>(cB.qn, 0, true, sx, s_rs);
    float * conv = sqb().conv + (size_t) dl * CONV_RING * QKV_L;
    const int n_ab = AB_SPLIT * 2 * NVH_L, n_qkv = QKV_L / 2, n_z = VD_L / 2;
    // alpha / beta: fp32 rows with the norm weight folded in (dot with the raw residual), each
    // row split into AB_SPLIT K-chunks so these heavier items do not form a tail
    for (int it = gwarp(); it < n_ab; it += nwarps()) {
        const int row = it % (2 * NVH_L), ch = it / (2 * NVH_L);
        const float * wr = W.ab + (size_t) row * D;
        float acc[T];
#pragma unroll
        for (int t = 0; t < T; t++) acc[t] = 0.f;
        for (int k = ch * (D / AB_SPLIT) + lane * 4; k < (ch + 1) * (D / AB_SPLIT); k += 128) {
            const float4 w = ldw4f(wr + k);
#pragma unroll
            for (int t = 0; t < T; t++) {
                const float4 xv = __ldcg(reinterpret_cast<const float4 *>(cB.x + t * D + k));
                acc[t] += w.x * xv.x + w.y * xv.y + w.z * xv.z + w.w * xv.w;
            }
        }
#pragma unroll
        for (int t = 0; t < T; t++) acc[t] = wsum(acc[t]) * s_rs[t];
        store_lane_t<T>(cB.ab + ch * MAXR * 2 * NVH_L, 2 * NVH_L, row, acc);
    }
    const int n_all = n_ab + n_qkv + n_z;
    gemv_pairs_t<T, 5>(W.qkv.type, W.qkv, n_ab, n_ab + n_qkv, n_all, [=](int r, const float (&a)[2][T], int nr) {
        const int lane = threadIdx.x & 31;
#pragma unroll
        for (int t = 0; t < T; t++)
            if ((lane >> 1) == t && (lane & 1) < nr)
                conv[((n + t) % CONV_RING) * QKV_L + r + (lane & 1)] = ((lane & 1) ? a[1][t] : a[0][t]) * misc(nullptr)[t];
    });
    gemv_pairs_t<T, 5>(W.gate.type, W.gate, n_ab + n_qkv, n_all, n_all, StorePair<T>{cB.z, VD_L});
    // Warm L2 with what dn_core will read, while the other blocks are still projecting:
    // the head's recurrent state (64 KB), its conv weights, older conv rows and replay entries.
    if (blockIdx.x < NVH_L) {
        const int h = blockIdx.x, kh = h % NKH_L, tid = threadIdx.x;
        pf_l2(sqb().dstate + ((size_t) dl * NVH_L + h) * 128 * 128 + tid * 32);
        if (tid == 0) {
            pf_l2(W.dt + (h & ~31));
            pf_l2(W.A + (h & ~31));
            pf_l2(W.snorm);
            pf_l2(W.snorm + 32);
            pf_l2(W.snorm + 64);
            pf_l2(W.snorm + 96);
        }
        if (tid < 384) {
            const int ch = tid < 128 ? kh * 128 + tid : (tid < 256 ? NKH_L * HK + kh * 128 + (tid - 128) : 2 * NKH_L * HK + h * 128 + (tid - 256));
            if ((tid & 7) == 0) pf_l2(W.conv + ch * 4);
            if ((tid & 31) == 0)
                for (int j = 0; j < 3; j++)
                    if (n - 3 + j >= 0) pf_l2(conv + ((n - 3 + j) % CONV_RING) * QKV_L + ch);
        }
        const float * ring = sqb().kvgb + (size_t) dl * KV_RING * KVGB_L;
        for (int i = tid; i < (n - dn_c) * 8; i += NT) {
            const float * slot = ring + (size_t) ((dn_c + (i >> 3)) % KV_RING) * KVGB_L;
            const int c = (i & 7);
            pf_l2(c < 4 ? slot + kh * 128 + c * 32 : slot + KVGB_V_L + h * 128 + (c - 4) * 32);
        }
    }
}

// delta-net core of value head h of one sequence (its buffers SQ, its rows r0 .. r0 + nr - 1; tokens past
// nr repeat the last row and write no output): conv1d+silu, l2-norm, gated delta rule (with replay of
// pending accepted tokens), gated rms-norm, quantise for ssm_out.
// BS (batched cycle): this cycle's conv inputs come from cB.qkvs, and this head copies them into SQ's ring.
template <int T, int BS = 0>
__device__ __forceinline__ void dn_head(const LayerW & W, int dl, int n, int dn_c, int n_commit, int h, const SeqBufs & SQ, int r0_, int nr = T) {
    uint8_t * const sm = smem;
    const int r0 = BS ? r0_ : 0;
    const int kh = h % NKH_L;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    // thread owns state column `col`, rows [32 r, 32 r + 32)  (S stored as [col][row])
    const int col = tid >> 2, r = tid & 3;
    constexpr int NE = T + MAXT - 1;     // max events: up to MAXT - 1 replayed accepted drafts + T new tokens
    constexpr int QS = 4 * 36;           // q / k with a 4-float pad per 32-row quarter (bank spread)
    float * s_q = reinterpret_cast<float *>(sm); // [NE][QS]
    float * s_k = s_q + NE * QS;                 // [NE][QS]
    float * s_v = s_k + NE * QS;                 // [NE][128]
    float * s_o = s_v + NE * 128;                // [T][128]
    float * s_z = s_o + T * 128;                 // [T][128]
    float * s_gb = s_z + T * 128;                // [NE][2]
    float * s_raw = s_gb + 32;                   // [T][384] conv outputs (silu), 16-byte aligned

    float * Sg = SQ.dstate + ((size_t) dl * NVH_L + h) * 128 * 128 + col * 128 + 32 * r;
    float4 S[8];
#pragma unroll
    for (int i = 0; i < 8; i++) S[i] = __ldcg(reinterpret_cast<const float4 *>(Sg) + i);

    const int np = n - dn_c; // replayed (accepted, not yet committed) tokens
    float * ring = SQ.kvgb + (size_t) dl * KV_RING * KVGB_L;
    float * conv = SQ.conv + (size_t) dl * CONV_RING * QKV_L;

    // ---- prologue: every input of every event, one round of loads ----
    if (tid < 384) {
        const int ch = tid < 128 ? kh * 128 + tid : (tid < 256 ? NKH_L * HK + kh * 128 + (tid - 128) : 2 * NKH_L * HK + h * 128 + (tid - 256));
        const float4 w = *reinterpret_cast<const float4 *>(W.conv + ch * 4);
        float raw[T + 3];
#pragma unroll
        for (int j = 0; j < T + 3; j++) {
            const int p = n - 3 + j;
            if (BS && j >= 3) raw[j] = __ldcg(cB.qkvs + (size_t) (r0 + min(j - 3, nr - 1)) * QKV_L + ch);
            else raw[j] = p >= 0 ? __ldcg(conv + (p % CONV_RING) * QKV_L + ch) : 0.f;
        }
        // the ring slots of positions n .. n+T-1 are not among those the other heads read (n-3 .. n-1);
        // q / k channels are shared by NVH_L / NKH_L heads, the first of them writes
        if (BS && (h < NKH_L || tid >= 256))
#pragma unroll
            for (int t = 0; t < T; t++) conv[((n + t) % CONV_RING) * QKV_L + ch] = raw[t + 3];
#pragma unroll
        for (int t = 0; t < T; t++)
            s_raw[t * 384 + tid] = silu(w.x * raw[t] + w.y * raw[t + 1] + w.z * raw[t + 2] + w.w * raw[t + 3]);
    } else if (tid < 384 + T) {
        const int t = tid - 384;
        float al = 0.f, be = 0.f;
#pragma unroll
        for (int c = 0; c < AB_SPLIT; c++) {
            al += __ldcg(cB.ab + (c * MAXR + r0 + min(t, nr - 1)) * 2 * NVH_L + h);
            be += __ldcg(cB.ab + (c * MAXR + r0 + min(t, nr - 1)) * 2 * NVH_L + NVH_L + h);
        }
        float sp = al + W.dt[h];
        sp = sp > 20.0f ? sp : logf(1.0f + expf(sp));
        s_gb[2 * (np + t)] = expf(sp * W.A[h]);
        s_gb[2 * (np + t) + 1] = 1.0f / (1.0f + expf(-be));
    } else if (tid >= 448 && tid < 448 + np) {
        const int e = tid - 448;
        const float * slot = ring + (size_t) ((dn_c + e) % KV_RING) * KVGB_L;
        s_gb[2 * e] = __ldcg(slot + KVGB_G_L + h);
        s_gb[2 * e + 1] = __ldcg(slot + KVGB_B_L + h);
    }
    for (int i = tid; i < np * 128; i += NT) { // replayed k / v
        const int e = i >> 7, c = i & 127;
        const float * slot = ring + (size_t) ((dn_c + e) % KV_RING) * KVGB_L;
        s_k[e * QS + (c >> 5) * 36 + (c & 31)] = __ldcg(slot + kh * 128 + c);
        s_v[e * 128 + c] = __ldcg(slot + KVGB_V_L + h * 128 + c);
    }
    for (int i = tid; i < T * 128; i += NT) s_z[i] = __ldcg(cB.z + (r0 + min(i >> 7, nr - 1)) * VD_L + h * 128 + (i & 127));
    __syncthreads();

    // l2-normalise q and k of the new tokens (warp 2t: q of token t, warp 2t+1: k), copy v
    if (warp < 2 * T) {
        const int t = warp >> 1, isk = warp & 1;
        const float4 v = *reinterpret_cast<const float4 *>(s_raw + t * 384 + isk * 128 + lane * 4);
        const float ss = wsum(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
        float sc = rsqrtf(fmaxf(ss, EPS * EPS));
        if (!isk) sc *= 0.08838834764831845f; // 1/sqrt(128)
        float * dst = (isk ? s_k : s_q) + (np + t) * QS + (lane >> 3) * 36 + (lane & 7) * 4;
        *reinterpret_cast<float4 *>(dst) = make_float4(v.x * sc, v.y * sc, v.z * sc, v.w * sc);
    }
    for (int i = tid; i < T * 128; i += NT) s_v[(np + (i >> 7)) * 128 + (i & 127)] = s_raw[(i >> 7) * 384 + 256 + (i & 127)];
    __syncthreads();

    // save the new tokens' (k, v, g, beta) for a possible replay next cycle
    for (int i = tid; i < T * 128; i += NT) {
        const int t = i >> 7, c = i & 127;
        float * slot = ring + (size_t) ((n + t) % KV_RING) * KVGB_L;
        if (h < NKH_L) slot[h * 128 + c] = s_k[(np + t) * QS + (c >> 5) * 36 + (c & 31)];
        slot[KVGB_V_L + h * 128 + c] = s_v[(np + t) * 128 + c];
        if (c == 0) {
            slot[KVGB_G_L + h] = s_gb[2 * (np + t)];
            slot[KVGB_B_L + h] = s_gb[2 * (np + t) + 1];
        }
    }

    // ---- gated delta rule over all events; inputs are in shared memory, no barriers needed ----
    for (int e = 0; e < np + T; e++) {
        const float g = s_gb[2 * e], beta = s_gb[2 * e + 1];
        const float4 * kq = reinterpret_cast<const float4 *>(s_k + e * QS + r * 36);
        float kv = 0.f;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const float4 k4 = kq[i];
            kv += S[i].x * k4.x + S[i].y * k4.y + S[i].z * k4.z + S[i].w * k4.w;
        }
        kv += __shfl_xor_sync(0xffffffffu, kv, 1);
        kv += __shfl_xor_sync(0xffffffffu, kv, 2);
        const float dlt = (s_v[e * 128 + col] - g * kv) * beta;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const float4 k4 = kq[i];
            S[i].x = g * S[i].x + k4.x * dlt;
            S[i].y = g * S[i].y + k4.y * dlt;
            S[i].z = g * S[i].z + k4.z * dlt;
            S[i].w = g * S[i].w + k4.w * dlt;
        }
        if (e >= np) {
            const float4 * qq = reinterpret_cast<const float4 *>(s_q + e * QS + r * 36);
            float o = 0.f;
#pragma unroll
            for (int i = 0; i < 8; i++) {
                const float4 q4 = qq[i];
                o += S[i].x * q4.x + S[i].y * q4.y + S[i].z * q4.z + S[i].w * q4.w;
            }
            o += __shfl_xor_sync(0xffffffffu, o, 1);
            o += __shfl_xor_sync(0xffffffffu, o, 2);
            if (r == 0) s_o[(e - np) * 128 + col] = o;
        }
        if (e == np + n_commit - 1) {
#pragma unroll
            for (int i = 0; i < 8; i++) reinterpret_cast<float4 *>(Sg)[i] = S[i];
        }
    }
    __syncthreads();

    // gated rms-norm + quantise: warp w -> token w / 4, 32-element block w % 4 of this head
    for (int w = warp; w < 4 * nr; w += NW) {
        const int t = w >> 2, jb = w & 3;
        const float4 v = *reinterpret_cast<const float4 *>(s_o + t * 128 + lane * 4);
        const float rs = rsqrtf(wsum(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w) / 128.0f + EPS);
        const int c = jb * 32 + lane;
        const float y = s_o[t * 128 + c] * rs * W.snorm[c] * silu(s_z[t * 128 + c]);
        q32_store(cB.qo, r0 + t, 4 * h + jb, y, 0.f, false);
    }
}

// delta-net core: one block per value head (48 blocks).
template <int T>
__device__ __noinline__ void dn_core(const LayerW & W, int dl, int n, int dn_c, int n_commit, uint8_t * /*sm*/) {
    if (blockIdx.x >= NVH_L) return;
    dn_head<T>(W, dl, n, dn_c, n_commit, blockIdx.x, sqb(), 0);
}

// q8_0 store of one KV row: lane holds values 8 lane .. 8 lane + 7 (4 lanes per 32-value block)
__device__ __forceinline__ void kv_store_q8(int8_t * q, __half * sc, const float (&v)[8]) {
    const int lane = threadIdx.x & 31;
    float am = 0.f;
#pragma unroll
    for (int k = 0; k < 8; k++) am = fmaxf(am, fabsf(v[k]));
    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 1));
    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 2));
    const float id = am > 0.f ? 127.0f / am : 0.f;
    unsigned w[2];
#pragma unroll
    for (int h = 0; h < 2; h++) {
        unsigned x = 0;
#pragma unroll
        for (int k = 0; k < 4; k++) x |= ((unsigned) __float2int_rn(v[4 * h + k] * id) & 0xffu) << (8 * k);
        w[h] = x;
    }
    *reinterpret_cast<uint2 *>(q + lane * 8) = make_uint2(w[0], w[1]);
    if ((lane & 3) == 0) sc[lane >> 2] = __float2half_rn(am / 127.0f);
}

// fp16 store of one KV row (same lane layout)
__device__ __forceinline__ void kv_store_f16(__half * dst, const float (&v)[8]) {
    __half2 h[4];
#pragma unroll
    for (int k = 0; k < 4; k++) h[k] = __floats2half2_rn(v[2 * k], v[2 * k + 1]);
    *reinterpret_cast<int4 *>(dst + (threadIdx.x & 31) * 8) = *reinterpret_cast<int4 *>(h);
}

// 8 q8 values + scale -> 4 fp16 pairs
__device__ __forceinline__ void q8_to_f16(uint2 q, float d, __half2 (&h)[4]) {
#pragma unroll
    for (int k = 0; k < 4; k++) {
        const unsigned x = k < 2 ? q.x : q.y;
        const int sh = 16 * (k & 1);
        h[k] = __floats2half2_rn(d * (float) (int8_t) (x >> sh), d * (float) (int8_t) (x >> (sh + 8)));
    }
}

// staged q8 K/V rows (KVQ_ROW bytes: values, then scales) -> fp16 rows of stride ld; rows x 2 tensors
__device__ __forceinline__ void kv_tile_to_f16(const uint8_t * src, __half * dK, __half * dV, int rows, int ld) {
    for (int i = threadIdx.x; i < 2 * rows * 32; i += blockDim.x) {
        const int which = i / (rows * 32), r = (i >> 5) % rows, c = i & 31;
        const uint8_t * row = src + (size_t) (which * rows + r) * KVQ_ROW;
        __half2 h[4];
        q8_to_f16(*reinterpret_cast<const uint2 *>(row + 8 * c), __half2float(reinterpret_cast<const __half *>(row + 256)[c >> 2]), h);
        *reinterpret_cast<int4 *>((which ? dV : dK) + (size_t) r * ld + 8 * c) = *reinterpret_cast<int4 *>(h);
    }
}

// async copy of q8 K/V tile rows r < np into the staging layout (rows >= np: zero scales)
__device__ __forceinline__ void kv_stage(const KvTile & t, uint8_t * dst, int rows, int np) {
    for (int i = threadIdx.x; i < 2 * rows * 17; i += blockDim.x) {
        const int which = i / (rows * 17), r = (i / 17) % rows, c = i % 17;
        uint8_t * d = dst + (size_t) (which * rows + r) * KVQ_ROW + 16 * c;
        if (r < np) {
            if (c < 16) cp_async16(d, kv_tile_row<true>(t, r, which) + 16 * c);
            else cp_async16(d, kv_tile_scales(t, r, which));
        } else if (c == 16) {
            *reinterpret_cast<int4 *>(d) = make_int4(0, 0, 0, 0);
        }
    }
}

// split-K over the context: at most 20 chunks per kv head so all NKV_L x nch items run in one wave
__device__ __forceinline__ void attn_dims(int L_ctx, int & nch, int & cs) {
    nch = max(1, min(20, (L_ctx + 63) / 64));
    cs = (L_ctx + nch - 1) / nch;
}

// first K / V tile (positions < n) of the attention items of one sequence that global item i0 + k * gridDim.x
// covers, its items numbered from ibase
template <bool Q8>
__device__ __forceinline__ void pf_attn_kv_t(const SeqBufs & S, int slot, int n, int T, int i0, int ibase) {
    int nch, CS;
    attn_dims(n + T, nch, CS);
    constexpr int lines = KvL<Q8>::RB / 128; // 128-byte lines per K / V row
    static_assert(KvL<Q8>::NP >= 64, "the prefetched span straddles at most one block boundary");
    const int G = gridDim.x;
    for (int item = (((i0 - ibase) % G) + G) % G; item < NKV_L * nch; item += G) {
        const int g = item / nch, c0 = (item % nch) * CS, c1 = min(n, min(c0 + CS, c0 + 64));
        const KvTile t = kv_tile<Q8>(S, slot, g, c0, c1 - c0);
        for (int i = threadIdx.x; i < (c1 - c0) * lines; i += NT) {
            const uint8_t * k = kv_tile_row<Q8>(t, i / lines, 0) + (i % lines) * 128;
            pf_l2(k);
            pf_l2(k + KvL<Q8>::SUB);
        }
    }
}
__device__ __forceinline__ void pf_attn_kv(const SeqBufs & S, int slot, int n, int T, int i0, int ibase) {
    if (cB.kvq8) pf_attn_kv_t<true>(S, slot, n, T, i0, ibase);
    else pf_attn_kv_t<false>(S, slot, n, T, i0, ibase);
}

// attention q/k/v projection GEMVs (activations already in shared memory)
template <int T>
__device__ __forceinline__ void attn_qkv(const LayerW & W) {
    const int nq = QG_L / 2, nk = KVD_L / 2;
    const int n_all = nq + 2 * nk;
    gemv_pairs_t<T, 5>(W.q.type, W.q, 0, nq, n_all, StorePair<T>{cB.qbuf, QG_L});
    gemv_pairs_t<T, 5>(W.k.type, W.k, nq, nq + nk, n_all, StorePair<T>{cB.kbuf, KVD_L});
    gemv_pairs_t<T, 5>(W.v.type, W.v, nq + nk, n_all, n_all, StorePair<T>{cB.vbuf, KVD_L});
}

// attention q/k/v projections
template <int T>
__device__ __noinline__ void attn_proj(const LayerW & W, int slot, int n, uint8_t * /*sm*/) {
    uint8_t * const sm = smem;
    const int lane = threadIdx.x & 31;
    (void) lane;
    const SX sx = make_sx(sm, U_D);
    pf_attn_proj(W);
    fill_xs<T>(cB.qn, 0, true, sx, misc(sm));
    attn_qkv<T>(W);
    // Warm L2 with what attn_core reads first: q/k norm weights, RoPE rows of the new positions,
    // and the first K / V tile of this block's attention item (positions < n).
    pf_block(W.qn, HD * 4);
    pf_block(W.kn, HD * 4);
    pf_block(cB.rope + (size_t) n * 32, T * 32 * 8);
    pf_attn_kv(sqb(), slot, n, T, blockIdx.x, 0);
}

__device__ __forceinline__ void rope8(float (&v)[8], int lane, int pos) {
    float o[8];
#pragma unroll
    for (int k = 0; k < 8; k++) o[k] = __shfl_xor_sync(0xffffffffu, v[k], 4);
    if (lane < 8) {
        const bool first = lane < 4;
        const float2 * tab = cB.rope + (size_t) pos * 32 + (lane & 3) * 8;
#pragma unroll
        for (int k = 0; k < 8; k++) {
            const float2 cs = __ldg(tab + k);
            v[k] = first ? v[k] * cs.x - o[k] * cs.y : o[k] * cs.y + v[k] * cs.x;
        }
    }
}

__device__ __forceinline__ void load8(const float * p, float (&v)[8]) {
    const float4 a = __ldcg(reinterpret_cast<const float4 *>(p));
    const float4 b = __ldcg(reinterpret_cast<const float4 *>(p + 4));
    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w;
    v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
}


// split-K flash decoding over (kv head, chunk) items; also writes this cycle's K/V.
__device__ __forceinline__ void mma16816(float (&c)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                         unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Split-K flash decoding, one (kv head g, chunk c) item of one sequence (buffers S, query rows r0 ..);
// partials go to item index oitem.  The 6*T query rows of a kv head form the M dimension of
// tensor-core MMAs: S = Q K^T and O += P V per 64-position tile of K / V staged in shared memory
// (fp16 operands, fp32 accumulation, online softmax).  Also writes this cycle's K/V.
template <int T, bool Q8>
__device__ __forceinline__ void attn_item(const LayerW & W, int slot, int n, int g, int c, int CS, int oitem, const SeqBufs & S, int r0,
                                          int nr = T) {
    uint8_t * const sm = smem;
    constexpr int R = 6 * T;
    constexpr int MT = (R + 15) / 16; // 16-row MMA tiles of query rows
    constexpr int TP = ATP;           // positions per K/V tile; tiles are double-buffered
    constexpr int LDH = HD + 8;       // smem row stride in halves (conflict-free fragment loads)
    constexpr int LDP = TP + 8;
    const int L_ctx = n + T;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g8 = lane >> 2, tig = lane & 3;
    __half * sQ = reinterpret_cast<__half *>(sm); // [32][LDH]
    // fp16 cache: 2 stages x (K [TP][LDH], V [TP][LDH]); q8 cache: one fp16 K/V tile + 2 q8 stages
    __half * sKV = sQ + 32 * LDH;
    uint8_t * sKVq = reinterpret_cast<uint8_t *>(sKV + 2 * TP * LDH);
    float * sS = reinterpret_cast<float *>(reinterpret_cast<uint8_t *>(sKV) + SM_ATTN_KV); // [32][TP]
    __half * sP = reinterpret_cast<__half *>(sS + 32 * TP); // [32][LDP]
    float * sM = reinterpret_cast<float *>(sP + 32 * LDP);
    float * sL = sM + 32;
    float * sAl = sL + 32;
    const int mt = warp >> 3, wn = warp & 7; // MMA work split: warp -> (row tile, column group)

    auto issue_tile = [&](int g, int tb, int np_, int st) {
        // split-K chunk starts are not tile aligned, so a tile may cross one block boundary
        const KvTile kt = kv_tile<Q8>(S, slot, g, tb, np_);
        if constexpr (Q8) {
            kv_stage(kt, sKVq + st * 2 * TP * KVQ_ROW, TP, np_);
            asm volatile("cp.async.commit_group;" ::: "memory");
            return;
        }
        __half * sK = sKV + st * 2 * TP * LDH, * sV = sK + TP * LDH;
        for (int i = tid; i < TP * (HD / 8); i += NT) {
            const int p = i >> 5, cc = (i & 31) * 8;
            if (p < np_) {
                const __half * gK = reinterpret_cast<const __half *>(kv_tile_row<false>(kt, p, 0));
                cp_async16(sK + p * LDH + cc, gK + cc);
                cp_async16(sV + p * LDH + cc, reinterpret_cast<const __half *>(reinterpret_cast<const uint8_t *>(gK) + KvL<false>::SUB) + cc);
            } else {
                *reinterpret_cast<int4 *>(sK + p * LDH + cc) = make_int4(0, 0, 0, 0);
                *reinterpret_cast<int4 *>(sV + p * LDH + cc) = make_int4(0, 0, 0, 0);
            }
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    };
    {
        const int c0 = c * CS, c1 = min(L_ctx, c0 + CS);
        // one round of small loads (q rows, norm weights, RoPE rows) before any bulk K/V traffic:
        // per-SM requests are served roughly in order
        float qv[2][8], qw[8];
        float2 rc[2][8];
        load8(W.qn + lane * 8, qw);
#pragma unroll
        for (int k2 = 0; k2 < 2; k2++) {
            const int r = warp + 16 * k2;
            if (r < R) {
                const int hq = r / T, t = r % T, head = g * 6 + hq;
                load8(cB.qbuf + (r0 + min(t, nr - 1)) * QG_L + head * 512 + lane * 8, qv[k2]);
                if (lane < 8) {
                    const float2 * tab = cB.rope + (size_t) (n + t) * 32 + (lane & 3) * 8;
#pragma unroll
                    for (int k = 0; k < 8; k++) rc[k2][k] = __ldg(tab + k);
                }
            }
        }
#pragma unroll
        for (int k2 = 0; k2 < 2; k2++) {
            const int r = warp + 16 * k2;
            if (r < R) {
                float * v = qv[k2];
                float ss = 0.f;
#pragma unroll
                for (int k = 0; k < 8; k++) ss += v[k] * v[k];
                const float rs = rsqrtf(wsum(ss) / 256.0f + EPS);
#pragma unroll
                for (int k = 0; k < 8; k++) v[k] *= rs * qw[k];
                float o[8];
#pragma unroll
                for (int k = 0; k < 8; k++) o[k] = __shfl_xor_sync(0xffffffffu, v[k], 4);
                if (lane < 8) {
#pragma unroll
                    for (int k = 0; k < 8; k++)
                        v[k] = lane < 4 ? v[k] * rc[k2][k].x - o[k] * rc[k2][k].y : o[k] * rc[k2][k].y + v[k] * rc[k2][k].x;
                }
                __half2 hq2[4];
#pragma unroll
                for (int k = 0; k < 4; k++) hq2[k] = __floats2half2_rn(v[2 * k] * 0.0625f, v[2 * k + 1] * 0.0625f);
                *reinterpret_cast<int4 *>(sQ + r * LDH + lane * 8) = *reinterpret_cast<int4 *>(hq2);
            } else {
                *reinterpret_cast<int4 *>(sQ + r * LDH + lane * 8) = make_int4(0, 0, 0, 0);
                if (2 * lane < TP) *reinterpret_cast<unsigned *>(sP + r * LDP + 2 * lane) = 0u; // padded rows of P stay 0
            }
        }
        // this cycle's K / V rows that fall into the chunk (positions >= n)
        const int pn0 = max(c0, n), pn1 = min(c1, n + T);
        for (int pp = pn0 + warp; pp < pn1; pp += NW) {
            const int t = pp - n;
            float v[8];
            load8(cB.kbuf + (r0 + min(t, nr - 1)) * KVD_L + g * 256 + lane * 8, v);
            float ss = 0.f;
#pragma unroll
            for (int k = 0; k < 8; k++) ss += v[k] * v[k];
            const float rs = rsqrtf(wsum(ss) / 256.0f + EPS);
#pragma unroll
            for (int k = 0; k < 8; k++) v[k] *= rs * W.kn[lane * 8 + k];
            rope8(v, lane, pp);
            const KvRow kr = kv_row<Q8>(S, slot, g, pp);
            if constexpr (Q8) kv_store_q8(reinterpret_cast<int8_t *>(kr.k), kr.ks, v);
            else kv_store_f16(reinterpret_cast<__half *>(kr.k), v);
            load8(cB.vbuf + (r0 + min(t, nr - 1)) * KVD_L + g * 256 + lane * 8, v);
            if constexpr (Q8) kv_store_q8(reinterpret_cast<int8_t *>(kr.v), kr.vs, v);
            else kv_store_f16(reinterpret_cast<__half *>(kr.v), v);
        }
        if (tid < 32) {
            sM[tid] = -INFINITY;
            sL[tid] = 0.f;
        }
        __threadfence_block();
        __syncthreads();

        float acc[4][4]; // O tiles: rows 16*mt + {g8, g8+8}, dims 8*(wn + 8j) + {2tig, 2tig+1}
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int k = 0; k < 4; k++) acc[j][k] = 0.f;

        issue_tile(g, c0, min(TP, c1 - c0), 0);
        int st = 0;
        for (int tb = c0; tb < c1; tb += TP, st ^= 1) {
            const int np_ = min(TP, c1 - tb);
            const __half * sK = Q8 ? sKV : sKV + st * 2 * TP * LDH, * sV = sK + TP * LDH;
            // the next tile streams into the other stage while this one is processed
            if (tb + TP < c1) {
                issue_tile(g, tb + TP, min(TP, c1 - tb - TP), st ^ 1);
                asm volatile("cp.async.wait_group 1;" ::: "memory");
            } else {
                asm volatile("cp.async.wait_group 0;" ::: "memory");
            }
            __syncthreads();
            if constexpr (Q8) { // q8 stage -> the fp16 tile
                kv_tile_to_f16(sKVq + st * 2 * TP * KVQ_ROW, sKV, sKV + TP * LDH, TP, LDH);
                __syncthreads();
            }
            // S = Q K^T : warp -> row tile mt, positions 8*wn .. 8*wn+7
            if (mt < MT && wn < TP / 8) {
                float s[4] = {0.f, 0.f, 0.f, 0.f};
                const __half * qa = sQ + (16 * mt + g8) * LDH + 2 * tig;
                const __half * kb = sK + (8 * wn + g8) * LDH + 2 * tig;
#pragma unroll
                for (int ks = 0; ks < HD / 16; ks++) {
                    const unsigned a0 = *reinterpret_cast<const unsigned *>(qa + 16 * ks);
                    const unsigned a1 = *reinterpret_cast<const unsigned *>(qa + 8 * LDH + 16 * ks);
                    const unsigned a2 = *reinterpret_cast<const unsigned *>(qa + 16 * ks + 8);
                    const unsigned a3 = *reinterpret_cast<const unsigned *>(qa + 8 * LDH + 16 * ks + 8);
                    const unsigned b0 = *reinterpret_cast<const unsigned *>(kb + 16 * ks);
                    const unsigned b1 = *reinterpret_cast<const unsigned *>(kb + 16 * ks + 8);
                    mma16816(s, a0, a1, a2, a3, b0, b1);
                }
                float * so = sS + (16 * mt + g8) * TP + 8 * wn + 2 * tig;
                so[0] = s[0];
                so[1] = s[1];
                so[8 * TP] = s[2];
                so[8 * TP + 1] = s[3];
            }
            __syncthreads();
            // online softmax per query row (causal mask: row token t sees positions <= n + t)
            for (int r = warp; r < R; r += NW) {
                const int lim = n + (r % T);
                float s0 = sS[r * TP + lane], s1 = TP > 32 ? sS[r * TP + (lane + 32) % TP] : -INFINITY;
                if (lane >= np_ || tb + lane > lim) s0 = -INFINITY;
                if (lane + 32 >= np_ || tb + lane + 32 > lim) s1 = -INFINITY;
                const float mt_ = wmax(fmaxf(s0, s1));
                const float mold = sM[r];
                const float mnew = fmaxf(mold, mt_);
                float p0 = 0.f, p1 = 0.f, al = 1.f;
                if (mnew != -INFINITY) {
                    p0 = expf(s0 - mnew);
                    p1 = expf(s1 - mnew);
                    al = expf(mold - mnew);
                }
                const float ps = wsum(p0 + p1);
                sP[r * LDP + lane] = __float2half_rn(p0);
                if (TP > 32) sP[r * LDP + (lane + 32) % TP] = __float2half_rn(p1);
                if (lane == 0) {
                    sL[r] = sL[r] * al + ps;
                    sM[r] = mnew;
                    sAl[r] = al;
                }
            }
            __syncthreads();
            // O = O * alpha + P V : warp -> row tile mt, dims 8 * (wn + 8 j)
            if (mt < MT) {
                const float al0 = sAl[16 * mt + g8], al1 = sAl[16 * mt + g8 + 8];
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    acc[j][0] *= al0;
                    acc[j][1] *= al0;
                    acc[j][2] *= al1;
                    acc[j][3] *= al1;
                }
                const __half * pa = sP + (16 * mt + g8) * LDP + 2 * tig;
#pragma unroll
                for (int ks = 0; ks < TP / 16; ks++) {
                    const unsigned a0 = *reinterpret_cast<const unsigned *>(pa + 16 * ks);
                    const unsigned a1 = *reinterpret_cast<const unsigned *>(pa + 8 * LDP + 16 * ks);
                    const unsigned a2 = *reinterpret_cast<const unsigned *>(pa + 16 * ks + 8);
                    const unsigned a3 = *reinterpret_cast<const unsigned *>(pa + 8 * LDP + 16 * ks + 8);
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        const __half * vb = sV + (16 * ks + (lane & 15)) * LDH + 8 * (wn + 8 * j);
                        unsigned b0, b1;
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
                                     : "=r"(b0), "=r"(b1)
                                     : "r"((unsigned) __cvta_generic_to_shared(vb)));
                        mma16816(acc[j], a0, a1, a2, a3, b0, b1);
                    }
                }
            }
            __syncthreads();
        }
        if (mt < MT) {
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int rr = 16 * mt + g8, d0 = 8 * (wn + 8 * j) + 2 * tig;
                if (rr < R) {
                    float * o = cB.apart + ((size_t) oitem * AR + rr) * 256 + d0;
                    o[0] = acc[j][0];
                    o[1] = acc[j][1];
                }
                if (rr + 8 < R) {
                    float * o = cB.apart + ((size_t) oitem * AR + rr + 8) * 256 + d0;
                    o[0] = acc[j][2];
                    o[1] = acc[j][3];
                }
            }
        }
        if (tid < R) cB.aml[oitem * AR + tid] = make_float2(sM[tid], sL[tid]);
        __syncthreads();
    }
}

// split-K flash decoding over the (kv head, chunk) items of the selected sequence
template <int T, bool Q8>
__device__ __forceinline__ void attn_core_t(const LayerW & W, int slot, int n) {
    int nch, CS;
    attn_dims(n + T, nch, CS);
    for (int item = blockIdx.x; item < NKV_L * nch; item += gridDim.x) attn_item<T, Q8>(W, slot, n, item / nch, item % nch, CS, item, sqb(), 0);
}

template <int T>
__device__ __noinline__ void attn_core(const LayerW & W, int slot, int n, uint8_t * /*sm*/) {
    if (cB.kvq8) attn_core_t<T, true>(W, slot, n);
    else attn_core_t<T, false>(W, slot, n);
}

// combine split-K partials, apply sigmoid output gate, quantise for o_proj
template <int T>
__device__ __noinline__ void attn_combine(int n) {
    const int L_ctx = n + T;
    int nch, CS;
    attn_dims(L_ctx, nch, CS);
    const int tid = threadIdx.x, d = tid & 255, wh = (tid & 255) >> 5;
    for (int it = blockIdx.x * 2 + (tid >> 8); it < NH_L * T; it += 2 * gridDim.x) {
        const int h = it / T, t = it % T, g = h / 6, r = (h % 6) * T + t;
        float M = -INFINITY;
        for (int c = 0; c < nch; c++) M = fmaxf(M, __ldcg(&cB.aml[(g * nch + c) * AR + r].x));
        float num = 0.f, den = 0.f;
        for (int c = 0; c < nch; c++) {
            const float2 ml = __ldcg(&cB.aml[(g * nch + c) * AR + r]);
            const float w = ml.x == -INFINITY ? 0.f : expf(ml.x - M);
            num += w * __ldcg(cB.apart + ((size_t) (g * nch + c) * AR + r) * 256 + d);
            den += w * ml.y;
        }
        const float gate = __ldcg(cB.qbuf + t * QG_L + h * 512 + 256 + d);
        const float o = num / den / (1.0f + expf(-gate));
        q32_store(cB.qo, t, 8 * h + wh, o, 0.f, false);
    }
}

// final RMS norm rows -> h (fp32) for the MTP head.
template <int T>
__device__ __noinline__ void write_h_rows(const float * x, const QBuf & q, const float * w, float * dst_base, int pos0, bool ring,
                             uint8_t * /*sm*/) {
    uint8_t * const sm = smem;
    if (blockIdx.x != 0) return;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    float * s_rs = misc(sm);
    if (warp < T) {
        float acc = 0.f;
        for (int j = lane; j < U_D; j += 32) acc += __ldcg(q.ssq + warp * U_D + j);
        acc = wsum(acc);
        if (lane == 0) s_rs[warp] = rsqrtf(acc / (float) D + EPS);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < T * D; i += NT) {
        const int t = i / D, e = i - t * D;
        float * dst = ring ? dst_base + (size_t) ((pos0 + t) % H_RING) * D : dst_base;
        if (!ring && t != T - 1) continue;
        dst[e] = __ldcg(x + i) * s_rs[t] * w[e];
    }
    __syncthreads();
}

#include "mma.cuh"

// warp-wide sorted (descending) list of the 32 best (value, id), lane i holding rank i; v is warp-uniform.
// Returns false (and inserts nothing) when v does not beat the current 32nd value.
__device__ __forceinline__ bool topk_insert(float & tv, int & ti, float v, int id) {
    const float mn = __shfl_sync(0xffffffffu, tv, 31);
    if (!(v > mn)) return false;
    const int lane = threadIdx.x & 31;
    const int pos = __popc(__ballot_sync(0xffffffffu, tv >= v));
    const float uv = __shfl_up_sync(0xffffffffu, tv, 1);
    const int ui = __shfl_up_sync(0xffffffffu, ti, 1);
    if (lane == pos) { tv = v; ti = id; }
    else if (lane > pos) { tv = uv; ti = ui; }
    return true;
}

// This block's share of lm_head over rows t0 .. t0 + TL - 1 -> blk_best rows bo .. bo + TL - 1.
template <int TL, bool TOPK, bool FULL>
__device__ __forceinline__ void lm_block(int t0, int bo, bool draft) {
    uint8_t * const sm = smem;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const SX sx = make_sx(sm, U_D);
    float * m = misc(sm);
#if MK_TP > 1
    const int off = cB.rank * VOCAB_L; // this card holds output rows [rank*VOCAB_L, +VOCAB_L)
#else
    const int off = 0;
#endif
    float bv[TL];
    int bi[TL];
#pragma unroll
    for (int t = 0; t < TL; t++) {
        bv[t] = -INFINITY;
        bi[t] = TOPK ? -1 : 0;
    }
    const Mat head = draft ? cW.head_d : cW.head;
    if constexpr (TL > MAXR_DP) {
        static_assert(!FULL && (!TOPK || TL == MR), "tensor-core lm_head serves batched decode only");
        constexpr int NS = mma_ns<TL, D>(), KS = D / NS, NC = TL / 8;
        const int g = lane >> 2, tig = lane & 3, o = draft ? 0 : off;
        float lv[NC][2]; // greedy: this lane's best of activation row 8 nt + 2 tig + c
        int li[NC][2];
#pragma unroll
        for (int nt = 0; nt < NC; nt++)
#pragma unroll
            for (int c = 0; c < 2; c++) {
                lv[nt][c] = -INFINITY;
                li[nt][c] = 0;
            }
        // block-uniform rounds of one 16-row tile per warp, the slices walked as in gemv_mma
        const int W = nwarps(), rounds = (head.N + 16 * W - 1) / (16 * W);
        XC xc{};
        int cur = -1;
        for (int rd = 0; rd < rounds; rd++) {
            // the draft head's row count is only even: the last tile clamps its loads and drops rows >= N
            const int r0 = 16 * (gwarp() + rd * W), ra = r0 + g, rb = r0 + g + 8;
            const bool va = ra < head.N, vb = rb < head.N;
            float acc[1][NC][4];
            zero_mma<1, NC>(acc);
            for (int i = 0; i < NS; i++) {
                const int s = (rd & 1) ? NS - 1 - i : i;
                if (s != cur) {
                    xc = fill_xc<TL, KS>(cB.qn, t0, s * KS / 32, true, m);
                    cur = s;
                }
                if (r0 < head.N) {
                    mma_tile<4, TL>(head, min(ra, head.N - 1), min(rb, head.N - 1), s * KS, xc, 0, KS / 64, acc);
                }
            }
            if (r0 >= head.N) continue;
            if constexpr (TOPK) {
                // the lanes holding row t offer the values that beat the list's 32nd, in ascending row order
#pragma unroll
                for (int t = 0; t < TL; t++) {
                    const int nt = t >> 3, c = t & 1;
                    const bool mine = tig == ((t & 7) >> 1);
                    const float v0 = va ? acc[0][nt][c] * m[t] : -INFINITY, v1 = vb ? acc[0][nt][2 + c] * m[t] : -INFINITY;
                    const float mn = __shfl_sync(0xffffffffu, bv[t], 31);
                    unsigned b0 = __ballot_sync(0xffffffffu, mine && v0 > mn), b1 = __ballot_sync(0xffffffffu, mine && v1 > mn);
                    while (b0) {
                        const int src = __ffs(b0) - 1;
                        topk_insert(bv[t], bi[t], __shfl_sync(0xffffffffu, v0, src), r0 + (src >> 2) + o);
                        b0 &= b0 - 1;
                    }
                    while (b1) {
                        const int src = __ffs(b1) - 1;
                        topk_insert(bv[t], bi[t], __shfl_sync(0xffffffffu, v1, src), r0 + 8 + (src >> 2) + o);
                        b1 &= b1 - 1;
                    }
                }
            } else {
#pragma unroll
                for (int nt = 0; nt < NC; nt++)
#pragma unroll
                    for (int c = 0; c < 2; c++) {
                        if (va && acc[0][nt][c] > lv[nt][c]) { lv[nt][c] = acc[0][nt][c]; li[nt][c] = ra + o; }
                        if (vb && acc[0][nt][2 + c] > lv[nt][c]) { lv[nt][c] = acc[0][nt][2 + c]; li[nt][c] = rb + o; }
                    }
            }
        }
        if constexpr (!TOPK) {
            // best over the 8 lanes sharing an activation row (lowest id on ties), then warp-uniform per row
#pragma unroll
            for (int nt = 0; nt < NC; nt++)
#pragma unroll
                for (int c = 0; c < 2; c++)
#pragma unroll
                    for (int sh = 4; sh < 32; sh <<= 1) {
                        const float ov = __shfl_xor_sync(0xffffffffu, lv[nt][c], sh);
                        const int oi = __shfl_xor_sync(0xffffffffu, li[nt][c], sh);
                        if (ov > lv[nt][c] || (ov == lv[nt][c] && oi < li[nt][c])) { lv[nt][c] = ov; li[nt][c] = oi; }
                    }
#pragma unroll
            for (int t = 0; t < TL; t++) {
                bv[t] = __shfl_sync(0xffffffffu, lv[t >> 3][t & 1], (t & 7) >> 1);
                bi[t] = __shfl_sync(0xffffffffu, li[t >> 3][t & 1], (t & 7) >> 1);
            }
        }
    } else {
        fill_xs<TL>(cB.qn, t0, true, sx, m);
        auto take = [&](int r, const float (&a)[2][TL]) {
            if constexpr (FULL) {
                if (lane == 0) {
    #pragma unroll
                    for (int t = 0; t < TL; t++) {
                        cB.logits[(size_t) (t0 + t) * VOCAB_L + r] = a[0][t] * m[t];
                        cB.logits[(size_t) (t0 + t) * VOCAB_L + r + 1] = a[1][t] * m[t];
                    }
                }
            }
    #pragma unroll
            for (int t = 0; t < TL; t++) {
                if constexpr (TOPK) { // true logits: apply the final RMS norm scale (irrelevant to the argmax)
                    const int o = draft ? 0 : off; // draft rows stay local until the block merge
                    topk_insert(bv[t], bi[t], a[0][t] * m[t], r + o);
                    topk_insert(bv[t], bi[t], a[1][t] * m[t], r + 1 + o);
                } else if (draft) { // draft greedy keeps local ids; the winner is converted after the block merge
                    if (a[0][t] > bv[t]) { bv[t] = a[0][t]; bi[t] = r; }
                    if (a[1][t] > bv[t]) { bv[t] = a[1][t]; bi[t] = r + 1; }
                } else {
                    if (a[0][t] > bv[t]) { bv[t] = a[0][t]; bi[t] = r + off; }
                    if (a[1][t] > bv[t]) { bv[t] = a[1][t]; bi[t] = r + 1 + off; }
                }
            }
        };
        if (head.type == T_Q8_0) {
            // 8-row items (rows8_q80); the tile padding covers the loads past an N that is only even
            const int g = lane >> 2;
            for (int r8 = 8 * gwarp(); r8 < head.N; r8 += 8 * nwarps()) {
                const uint8_t * wp[1] = {head.w + q8_wo(r8 + g, head.rb)};
                const __half * dp[1] = {head.d + q8_do(r8 + g, head.nd)};
                float a8[1][TL];
                zero_acc<TL, 1>(a8);
                rows8_q80<TL, 1, U_D>(wp, dp, sx, a8);
#pragma unroll
                for (int p = 0; p < 4; p++) {
                    float a[2][TL];
#pragma unroll
                    for (int t = 0; t < TL; t++) {
                        a[0][t] = __shfl_sync(0xffffffffu, a8[0][t], 8 * p);
                        a[1][t] = __shfl_sync(0xffffffffu, a8[0][t], 8 * p + 4);
                    }
                    if (r8 + 2 * p < head.N) take(r8 + 2 * p, a);
                }
            }
        } else {
            for (int r = 2 * gwarp(); r < head.N; r += 2 * nwarps()) {
                float a[2][TL];
                zero_acc<TL, 2>(a);
                mrows_t<TL, 5, 2>(head.type, head, r, sx, a);
                wsum_acc<TL, 2>(a);
                take(r, a);
            }
        }
    }
    if constexpr (TOPK) {
        // merge the warps' lists (shared memory, the activations are no longer needed), then the blocks'
        __syncthreads();
        float2 * wl = reinterpret_cast<float2 *>(sm); // [NW][TL][32]
#pragma unroll
        for (int t = 0; t < TL; t++) wl[(warp * TL + t) * 32 + lane] = make_float2(bv[t], __int_as_float(bi[t]));
        __syncthreads();
        if (warp < TL) {
            float v = -INFINITY;
            int id = -1;
            for (int w = 0; w < NW; w++)
                for (int j = 0; j < 32; j++) {
                    const float2 e = wl[(w * TL + warp) * 32 + j];
                    if (!topk_insert(v, id, e.x, __float_as_int(e.y))) break;
                }
            cB.blk_best[((size_t) blockIdx.x * MAXR + bo + warp) * TOPK_MAX + lane] = make_float2(v, __int_as_float(id));
        }
    } else {
        float * wv = reinterpret_cast<float *>(sm + SM_PARK);  // [NW][MAXR] (misc is too small at MAXR rows)
        int * wi = reinterpret_cast<int *>(wv + NW * MAXR);   // [NW][MAXR]
        if (lane == 0) {
#pragma unroll
            for (int t = 0; t < TL; t++) {
                wv[warp * MAXR + t] = bv[t];
                wi[warp * MAXR + t] = bi[t];
            }
        }
        __syncthreads();
        if (tid < TL) {
            float b = -INFINITY;
            int ix = 0;
            for (int w = 0; w < NW; w++) {
                const float v = wv[w * MAXR + tid];
                const int id = wi[w * MAXR + tid];
                if (v > b || (v == b && id < ix)) { b = v; ix = id; }
            }
            cB.blk_best[blockIdx.x * MAXR + bo + tid] = make_float2(b, __int_as_float(ix));
        }
    }
}

// lm_head over TL rows (qn rows t0..t0+TL-1) with greedy argmax; returns true in the single
// block that finished last, which holds the reduced argmax in s_tok.  TOPK: additionally keeps the
// TOPK_MAX best logits of each row in out, default cB.cand (for sampling; draft rows carry token ids).  FULL: additionally stores every
// logit of each row in cB.logits (scoring).
template <int TL, bool TOPK = false, bool FULL = false>
__device__ __noinline__ bool lm_head(int t0, bool draft, int kx, uint8_t * /*sm*/, int * s_tok, float2 * out = nullptr) {
    uint8_t * const sm = smem;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    float * m = misc(sm);
#if MK_TP > 1
    const int off = cB.rank * VOCAB_L;
#else
    const int off = 0;
#endif
    pf_rows(draft ? cW.head_d : cW.head, 2 * gwarp(), 2, 896);
    if constexpr (TOPK && TL > MR) {
        // a warp's top-k lists of TL rows do not fit in registers: two passes over the head
        lm_block<MR, true, FULL>(t0, 0, draft);
        __syncthreads(); // the second pass's activation fill overwrites the warp lists the first pass is still merging
        lm_block<TL - MR, true, FULL>(t0 + MR, MR, draft);
    } else {
        lm_block<TL, TOPK, FULL>(t0, 0, draft);
    }
    __threadfence();
    __syncthreads();
    int * s_last = reinterpret_cast<int *>(m + 384);
    if (tid == 0) s_last[0] = (atomicAdd(cB.cnt, 1u) == gridDim.x - 1);
    __syncthreads();
    if (!s_last[0]) return false;
    __threadfence();
    if constexpr (TOPK) {
#if MK_TP > 1
        // single lx slot: consecutive lm_head exchanges are separated by a resid32 all-reduce
        float2 * own = reinterpret_cast<float2 *>(sm); // [TL][TOPK_MAX], reuses the dead wl
        for (int t = warp; t < TL; t += NW) {
            float v = -INFINITY;
            int id = -1;
            for (int k = 0; k < (int) gridDim.x; k++)
                for (int j = 0; j < 32; j++) {
                    const float2 e = __ldcg(&cB.blk_best[((size_t) k * MAXR + t) * TOPK_MAX + j]);
                    if (!topk_insert(v, id, e.x, __float_as_int(e.y))) break;
                }
            if (draft && id >= 0) id = cW.draft_ids ? cW.draft_ids[id] : id + off;
            own[t * TOPK_MAX + lane] = make_float2(v, __int_as_float(id));
        }
        __syncthreads();
        lx_put(kx, own, TL * TOPK_MAX);
        lx_wait(kx);
        for (int t = warp; t < TL; t += NW) {
            const float2 * pl = cB.lx + t * TOPK_MAX; // peer data: read past L1
            const float2 * ol = own + t * TOPK_MAX;
            float v = -INFINITY;
            int id = -1;
            for (int j = 0; j < TOPK_MAX; j++) {
                const float2 e = (cB.rank == 0) ? ol[j] : __ldcg(pl + j);
                if (!topk_insert(v, id, e.x, __float_as_int(e.y))) break;
            }
            for (int j = 0; j < TOPK_MAX; j++) {
                const float2 e = (cB.rank == 0) ? __ldcg(pl + j) : ol[j];
                if (!topk_insert(v, id, e.x, __float_as_int(e.y))) break;
            }
            (out ? out : cB.cand)[t * TOPK_MAX + lane] = make_float2(v, __int_as_float(id));
            __syncwarp();
            const int best = __shfl_sync(0xffffffffu, id, 0);
            if (lane == 0) s_tok[t] = best;
        }
#else
        if (warp < TL) {
            float v = -INFINITY;
            int id = -1;
            for (int k = 0; k < (int) gridDim.x; k++)
                for (int j = 0; j < 32; j++) {
                    const float2 e = __ldcg(&cB.blk_best[((size_t) k * MAXR + warp) * TOPK_MAX + j]);
                    if (!topk_insert(v, id, e.x, __float_as_int(e.y))) break;
                }
            if (draft && id >= 0) id = cW.draft_ids ? cW.draft_ids[id] : id;
            (out ? out : cB.cand)[warp * TOPK_MAX + lane] = make_float2(v, __int_as_float(id));
            __syncwarp();
            const int best = __shfl_sync(0xffffffffu, id, 0);
            if (lane == 0) s_tok[warp] = best;
        }
#endif
    } else {
        float b = -INFINITY;
        int ix = 0;
#if MK_TP > 1
        // single lx slot: consecutive lm_head exchanges are separated by a resid32 all-reduce
        float2 * own = reinterpret_cast<float2 *>(sm); // [TL], reuses the dead region at offset 0
#endif
        if (tid < TL) {
            for (int k = 0; k < (int) gridDim.x; k++) {
                const float2 e = __ldcg(&cB.blk_best[k * MAXR + tid]);
                const int id = __float_as_int(e.y);
                if (e.x > b || (e.x == b && id < ix)) { b = e.x; ix = id; }
            }
            if (draft) ix = cW.draft_ids ? cW.draft_ids[ix] : ix + off;
#if MK_TP > 1
            own[tid] = make_float2(b, __int_as_float(ix));
#endif
        }
#if MK_TP > 1
        __syncthreads();
        lx_put(kx, own, TL);
        lx_wait(kx);
        if (tid < TL) {
            const float2 pe = __ldcg(cB.lx + tid);
            const float2 p0 = (cB.rank == 0) ? own[tid] : pe;
            const float2 p1 = (cB.rank == 0) ? pe : own[tid];
            ix = __float_as_int(p0.y);
            const int p1i = __float_as_int(p1.y);
            if (p1.x > p0.x || (p1.x == p0.x && p1i < ix)) ix = p1i;
            s_tok[tid] = ix;
        }
#else
        if (tid < TL) s_tok[tid] = ix;
#endif
    }
    if (tid == 0) *cB.cnt = 0;
    __syncthreads();
    return true;
}

// ---- sampling (single thread) ----
// counter-based uniform random number in [0, 1)
__device__ __forceinline__ float urand(unsigned long long seed, unsigned long long ctr) {
    unsigned long long z = seed + (ctr + 1) * 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    return (float) (z >> 40) * (1.0f / 16777216.0f);
}

// sampling distribution over one row's candidates (descending), llama.cpp order: top_k, top_p, min_p,
// temperature; returns the number kept, p[0..K) normalised
__device__ int cand_probs(const float2 * c, const Launch & P, float * p) {
    int K = P.top_k > 0 ? min(P.top_k, TOPK_MAX) : TOPK_MAX;
    while (K > 1 && !(c[K - 1].x > -INFINITY)) K--;
    const float mx = c[0].x;
    float s = 0.f;
    for (int j = 0; j < K; j++) s += (p[j] = expf(c[j].x - mx));
    if (P.top_p < 1.0f) {
        float cum = 0.f;
        for (int j = 0; j < K; j++) {
            cum += p[j] / s;
            if (cum >= P.top_p) { K = j + 1; break; }
        }
    }
    if (P.min_p > 0.0f) {
        int k = 1;
        while (k < K && p[k] >= P.min_p * p[0]) k++;
        K = k;
    }
    s = 0.f;
    for (int j = 0; j < K; j++) s += (p[j] = expf((c[j].x - mx) / P.temp));
    for (int j = 0; j < K; j++) p[j] /= s;
    return K;
}

// draw from p (optionally without token excl, renormalised)
__device__ int cand_draw(const float2 * c, const float * p, int K, int excl, float u) {
    float s = 0.f;
    for (int j = 0; j < K; j++)
        if (__float_as_int(c[j].y) != excl) s += p[j];
    float x = u * s;
    int last = __float_as_int(c[0].y);
    for (int j = 0; j < K; j++) {
        const int id = __float_as_int(c[j].y);
        if (id == excl) continue;
        last = id;
        x -= p[j];
        if (x < 0.f) return id;
    }
    return last;
}

// q_j of token id in a stored draft distribution (descending, zero-terminated)
__device__ float q_of(const float2 * qd, int id) {
#pragma unroll 1
    for (int k = 0; k < TOPK_MAX && qd[k].x > 0.f; k++)
        if (__float_as_int(qd[k].y) == id) return qd[k].x;
    return 0.f;
}

// draw from norm(max(0, p - q)): the residual of rejection sampling with a sampled draft.  Only p's
// candidates matter: tokens outside p's support have p = 0 and cannot be drawn.
__device__ __noinline__ int residual_draw(const float2 * c, const float * p, int K, const float2 * qd, float u) {
    float s = 0.f;
    for (int j = 0; j < K; j++) s += fmaxf(p[j] - q_of(qd, __float_as_int(c[j].y)), 0.f);
    if (s <= 0.f) return cand_draw(c, p, K, -1, u); // q dominates p here: the target's own draw
    float x = u * s;
    int last = __float_as_int(c[0].y);
    for (int j = 0; j < K; j++) {
        const int id = __float_as_int(c[j].y);
        const float d = p[j] - q_of(qd, id);
        if (d <= 0.f) continue;
        last = id;
        x -= d;
        if (x < 0.f) return last;
    }
    return last;
}

// Sampled targets for the verified rows: draft token d at row i is accepted with probability
// p_i(d) for a point-mass draft, or with min(1, p_i(d)/q_i(d)) when the draft was sampled from q;
// on rejection the row's token is drawn from the residual distribution (p_i without d, or
// norm(max(0, p_i - q_i))), and the rows after it are dropped.  Rows with no draft to verify (the
// bonus row, the last prompt row) are drawn from p_i.  tg then drives the greedy acceptance logic
// of main_control unchanged.
// BS: slot s of a batched cycle, its rows from r0.
template <int BS = 0>
__device__ void sample_targets(const Launch & P, int T, int tl, int * tg, int s = 0, int r0_ = 0) {
    const int r0 = BS ? r0_ : 0;
    const Ctl * c = sqx<BS>(s).ctl;
    const unsigned long long n = (unsigned long long) c->n;
    float p[TOPK_MAX];
    for (int i = 0; i < tl; i++) {
        const float2 * cd = cB.cand + (r0 + i) * TOPK_MAX;
        const int K = cand_probs(cd, P, p);
        if (P.mode == MODE_DECODE && i < T - 1) {
            const int d = c->tok[i + 1];
            float pd = 0.f;
            for (int j = 0; j < K; j++)
                if (__float_as_int(cd[j].y) == d) pd += p[j];
            const float2 * qd = sqx<BS>(s).dcand + (size_t) i * TOPK_MAX;
            if (P.draft_prob && qd[0].x > 0.f) {
                const float qx = q_of(qd, d);
                if (qx > 0.f && (pd >= qx || urand(P.seed, 2 * (n + i)) < pd / qx)) {
                    tg[i] = d;
                    continue;
                }
                tg[i] = residual_draw(cd, p, K, qd, urand(P.seed, 2 * (n + i) + 1));
                return;
            }
            if (urand(P.seed, 2 * (n + i)) < pd) {
                tg[i] = d;
                continue;
            }
            tg[i] = cand_draw(cd, p, K, d, urand(P.seed, 2 * (n + i) + 1));
            return;
        }
        tg[i] = cand_draw(cd, p, K, -1, urand(P.seed, 2 * (n + i) + 1));
        return;
    }
}

// ---- main-model control (single thread) ----
template <int BS = 0>
__device__ void main_control(int mode, int last, int T, const int * tgt, int s = 0) {
    Ctl * c = sqx<BS>(s).ctl;
    const int n = c->n;
    int a, bonus = -1;
    if (mode == MODE_DECODE) {
        a = 0;
        while (a < T - 1 && tgt[a] == c->tok[a + 1]) a++;
        for (int i = 0; i < a; i++) c->out[i] = c->tok[i + 1];
        c->out[a] = tgt[a];
        c->n_out = a + 1;
        c->n_acc = a;
        bonus = tgt[a];
        c->dn_c = n + 1;
    } else {
        a = T - 1;
        c->dn_c = n + T;
        c->n_acc = 0;
        if (last) {
            bonus = tgt[0];
            c->out[0] = bonus;
            c->n_out = 1;
        } else {
            c->n_out = 0;
        }
    }
    const int m = c->mtp_valid;
    const int P = bonus >= 0 ? n + a + 1 : n + T - 1;
    int R = 0;
    for (int p = m; p <= P && R < 8; p++, R++) {
        c->mtp_pos[R] = p;
        c->mtp_tok[R] = (p >= n && p <= n + a) ? c->tok[p - n] : bonus;
    }
    c->mtp_R = R;
    c->mtp_P = P;
    c->mtp_logits = bonus >= 0;
    c->mtp_valid = P + 1;
    if (bonus >= 0) {
        c->n = P;
        c->tok[0] = bonus;
    }
}

template <int T>
__device__ void main_forward(const Launch & P, uint8_t * /*sm*/, unsigned & epoch) {
    uint8_t * const sm = smem;
    volatile Ctl * vc = sqb().ctl;
    const int n = vc->n;
    const int dn_c = vc->dn_c;
    const int n_commit = P.mode == MODE_DECODE ? 1 : T;

    embed_prologue<T>();
    gsync(epoch, PH_EMBED);
    int dl = 0;
    for (int L = 0; L < NL; L++) {
        const LayerW & W = cW.L[L];
        const float * wnext = (L + 1 < NL) ? cW.L[L + 1].attn_norm : cW.onorm;
        if (W.attn) {
            attn_proj<T>(W, L / 4, n, sm);
            gsync(epoch, PH_ATTN_PROJ);
            attn_core<T>(W, L / 4, n, sm);
            gsync(epoch, PH_ATTN_CORE);
            attn_combine<T>(n);
            pf_resid(W.o, U_VD_L / 32);
            gsync(epoch, PH_ATTN_COMB);
            resid32_t<T, U_VD_L / 32>(W.o.type, W.o, cB.qo, cB.x, true, W.post_norm, sm, 2 * L);
            pf_ffn_gu(W);
            gsync(epoch, PH_RESID_O);
        } else {
            dn_proj<T>(W, dl, n, dn_c, sm);
            gsync(epoch, PH_DN_PROJ);
            dn_core<T>(W, dl, n, dn_c, n_commit, sm);
            pf_resid(W.sout, U_VD_L / 32);
            gsync(epoch, PH_DN_CORE);
            resid32_t<T, U_VD_L / 32>(W.sout.type, W.sout, cB.qo, cB.x, true, W.post_norm, sm, 2 * L);
            pf_ffn_gu(W);
            gsync(epoch, PH_RESID_SOUT);
            dl++;
        }
        ffn_gu<T>(W, sm);
        pf_resid(W.fd, U_FF_L / 32);
        gsync(epoch, PH_FFN_GU);
        resid32_t<T, U_FF_L / 32>(W.fd.type, W.fd, cB.qa, cB.x, true, wnext, sm, 2 * L + 1);
        if (L + 1 < NL) {
            if (cW.L[L + 1].attn) pf_attn_proj(cW.L[L + 1]);
            else pf_dn_proj(cW.L[L + 1]);
        }
        gsync(epoch, PH_FFN_DOWN);
    }
    write_h_rows<T>(cB.x, cB.qn, cW.onorm, sqb().h_hist, n, true, sm);

    int * s_tok = reinterpret_cast<int *>(misc(sm) + 400);
    bool am_last;
    int tl = 0;
    const bool sampling = P.temp > 0.0f && !P.score;
    if (P.mode == MODE_DECODE || P.score) {
        if (sampling) am_last = lm_head<T, true>(0, false, 4000, sm, s_tok);
        else if (P.score == 2) am_last = lm_head<T, false, true>(0, false, 4000, sm, s_tok);
        else am_last = lm_head<T>(0, false, 4000, sm, s_tok);
        tl = T;
    } else if (P.last) {
        am_last = sampling ? lm_head<1, true>(T - 1, false, 4000, sm, s_tok) : lm_head<1>(T - 1, false, 4000, sm, s_tok);
        tl = 1;
    } else {
        am_last = blockIdx.x == 0;
    }
    if (am_last && threadIdx.x == 0) {
        int tg[MAXT];
        for (int i = 0; i < tl; i++) tg[i] = s_tok[i];
        if (sampling) sample_targets(P, T, tl, tg);
        for (int i = 0; i < tl; i++) sqb().ctl->tgt[i] = tg[i];
        main_control(P.mode, P.last, T, P.score ? s_tok + T - 1 : tg);
        __threadfence();
    }
    gsync(epoch, PH_LMHEAD);
}

// one MTP pass over R consecutive positions; rows read from ctl (catch-up) or a draft row.
template <int R>
__device__ __noinline__ void mtp_pass(const Launch & P, int row0, bool draft_row, int draft_idx, bool logits, int p, uint8_t * /*sm*/, unsigned & epoch) {
    uint8_t * const sm = smem;
    volatile Ctl * vc = sqb().ctl;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    int pos0;
    if (draft_row) pos0 = vc->mtp_P + draft_idx;
    else pos0 = vc->mtp_pos[row0];

    // M0: concat(enorm(embed(tok)), hnorm(h_prev)) -> qe
    if (blockIdx.x < R) {
        const int r = blockIdx.x;
        const int pos = pos0 + r;
        const int tok = draft_row ? vc->tok[draft_idx] : vc->mtp_tok[row0 + r];
        const float * hsrc = draft_row ? sqb().hm : (pos >= 1 ? sqb().h_hist + (size_t) ((pos - 1) % H_RING) * D : nullptr);
        float * se = reinterpret_cast<float *>(sm);
        float * sh = se + D;
        float * red = sh + D;
        const uint8_t * erow = stage_embd_row(tok, reinterpret_cast<uint8_t *>(red + 32));
        float ae = 0.f, ah = 0.f;
        for (int e = tid; e < D; e += NT) {
            const float v = dequant_elem(erow, cW.embd_type, e);
            const float hv = hsrc ? __ldcg(hsrc + e) : 0.f;
            se[e] = v;
            sh[e] = hv;
            ae += v * v;
            ah += hv * hv;
        }
        ae = bsum(ae, red);
        ah = bsum(ah, red + 16);
        const float rse = rsqrtf(ae / (float) D + EPS), rsh = rsqrtf(ah / (float) D + EPS);
#if MK_TP > 1
        // eh_proj is K-split: rank 0 holds the enorm(embed) columns, rank 1 the hnorm(h) columns
        for (int j = warp; j < U_D; j += NW) {
            const int e = 32 * j + lane;
            const float v = cB.rank == 0 ? se[e] * rse * cW.enorm[e] : sh[e] * rsh * cW.hnorm[e];
            q32_store(cB.qe, r, j, v, 0.f, false);
        }
#else
        for (int j = warp; j < 2 * U_D; j += NW) {
            const int e = 32 * (j % U_D) + lane;
            const float v = j < U_D ? se[e] * rse * cW.enorm[e] : sh[e] * rsh * cW.hnorm[e];
            q32_store(cB.qe, r, j, v, 0.f, false);
        }
#endif
    }
    gsync(epoch, PH_MTP_M0);
    const LayerW & W = cW.L[NL];
    const int k0 = 128 + 3 * p; // eh | o | fd of pass p; p <= 8 here, so k stays far below the 4096 tag bound
    resid32_t<R, U_EH_L / 32>(cW.eh.type, cW.eh, cB.qe, cB.xm, false, W.attn_norm, sm, k0);
    gsync(epoch, PH_MTP_EH);
    attn_proj<R>(W, 16, pos0, sm);
    gsync(epoch, PH_MTP_ATTN_PROJ);
    attn_core<R>(W, 16, pos0, sm);
    gsync(epoch, PH_MTP_ATTN_CORE);
    attn_combine<R>(pos0);
    gsync(epoch, PH_MTP_ATTN_COMB);
    resid32_t<R, U_VD_L / 32>(W.o.type, W.o, cB.qo, cB.xm, true, W.post_norm, sm, k0 + 1);
    gsync(epoch, PH_MTP_RESID_O);
    ffn_gu<R>(W, sm);
    gsync(epoch, PH_MTP_FFN_GU);
    resid32_t<R, U_FF_L / 32>(W.fd.type, W.fd, cB.qa, cB.xm, true, cW.shnorm, sm, k0 + 2);
    gsync(epoch, PH_MTP_FFN_DOWN);
    if (logits) {
        write_h_rows<R>(cB.xm, cB.qn, cW.shnorm, sqb().hm, 0, false, sm);
        int * s_tok = reinterpret_cast<int *>(misc(sm) + 400);
        const int di = draft_row ? draft_idx : 0;
        const bool prob = P.draft_prob && P.temp > 0.0f;
        float2 * out = sqb().dcand + (size_t) di * TOPK_MAX;
        const bool last = prob ? lm_head<1, true>(R - 1, true, 4001 + p, sm, s_tok, out)
                               : lm_head<1>(R - 1, true, 4001 + p, sm, s_tok);
        if (last && threadIdx.x == 0) {
            int tok = s_tok[0];
            if (prob) {
                // dv_add pads the draft head with duplicate rows: keep each token once, or q(d) halves
                int k = 0;
                for (int j = 0; j < TOPK_MAX; j++) {
                    const int id = __float_as_int(out[j].y);
                    bool dup = false;
                    for (int i = 0; i < k; i++) dup |= __float_as_int(out[i].y) == id;
                    if (!dup) out[k++] = out[j];
                }
                for (int j = k; j < TOPK_MAX; j++) out[j] = make_float2(-INFINITY, __int_as_float(-1));
                // sample the draft from q as the target samples and keep q for next cycle's rejection test
                float * qs = misc(sm) + 448;
                const int K = cand_probs(out, P, qs);
                tok = cand_draw(out, qs, K, -1, urand(P.seed ^ 0x85ebca6bULL, 3ull * (vc->mtp_P + di + 1)));
                for (int j = 0; j < K; j++) out[j].x = qs[j];
                for (int j = K; j < TOPK_MAX; j++) out[j].x = 0.f;
            } else {
                out[0].x = 0.f; // point-mass draft: stale q from an earlier sampled draft must not apply
            }
            sqb().ctl->tok[1 + di] = tok;
            __threadfence();
        }
        gsync(epoch, PH_MTP_LMHEAD);
    }
}

template <int T>
__device__ void run_main(const Launch & P, uint8_t * /*sm*/, unsigned & epoch) {
    uint8_t * const sm = smem;
    main_forward<T>(P, sm, epoch);
}

template <int T>
__device__ void bench_phase(int which, int layer, uint8_t * /*sm*/, unsigned & epoch) {
    uint8_t * const sm = smem;
    for (int it = 0; it < 4; it++) {
        const int L = layer + 4 * it; // different layers: weights are not L2-resident, code is
        const LayerW & W = cW.L[L];
        if (which == 0) ffn_gu<T>(W, sm);
        else if (which == 1) {
            resid32_t<T, U_FF_L / 32>(W.fd.type, W.fd, cB.qa, cB.x, true, W.attn_norm, sm, 2 * L + 1);
        } else if (which == 2) {
            if (!W.attn) dn_proj<T>(W, 0, 0, 0, sm);
        }
        gsync(epoch, PH_FFN_GU);
    }
}

__global__ void __launch_bounds__(NT, 1) mk_cycle(Launch P) {
    uint8_t * const sm = smem;
    unsigned epoch = 0;
    if (P.mode == MODE_BENCH) {
        switch (P.T) {
            case 1: bench_phase<1>(P.last, P.ndraft, sm, epoch); break;
            case 4: bench_phase<4>(P.last, P.ndraft, sm, epoch); break;
        }
        return;
    }
    if (cB.prof && blockIdx.x == 0 && threadIdx.x == 0) {
        cB.prof[0] = gtimer() << 8;
        s_bar_k = 0;
        for (int i = 0; i < PS_NV; i++) s_pst[i] = 0;
        s_pc0 = clock64();
    }
    switch (P.T) {
        case 1: run_main<1>(P, sm, epoch); break;
        case 2: run_main<2>(P, sm, epoch); break;
        case 3: run_main<3>(P, sm, epoch); break;
        case 4: run_main<4>(P, sm, epoch); break;
        case 5: run_main<5>(P, sm, epoch); break;
    }
    if (P.ndraft == 0 || P.score) return;
    volatile Ctl * vc = sqb().ctl;
    const int Rtot = vc->mtp_R;
    const bool lg = vc->mtp_logits;
    int done = 0;
    int p = 0; // MTP pass index; the pass's reductions take k = 128 + 3 * p + {0, 1, 2}
    while (done < Rtot) {
        const int R = min(MAXT, Rtot - done);
        const bool last_pass = done + R == Rtot;
        const bool want = lg && last_pass && P.ndraft > 0;
        switch (R) {
            case 1: mtp_pass<1>(P, done, false, 0, want, p, sm, epoch); break;
            case 2: mtp_pass<2>(P, done, false, 0, want, p, sm, epoch); break;
            case 3: mtp_pass<3>(P, done, false, 0, want, p, sm, epoch); break;
            case 4: mtp_pass<4>(P, done, false, 0, want, p, sm, epoch); break;
            case 5: mtp_pass<5>(P, done, false, 0, want, p, sm, epoch); break;
        }
        done += R;
        p++;
    }
    if (lg) {
        for (int i = 1; i < P.ndraft; i++) {
            mtp_pass<1>(P, 0, true, i, true, p, sm, epoch);
            p++;
        }
    }
}

#if MK_TP > 1
#include "batch.cuh"
#endif

// ids are global; src0 / src1 hold the output rows of vocab half 0 / 1 (the peer card's for the other half)
__global__ void gather_rows(const uint8_t * src0, const __half * dsrc0, const uint8_t * src1, const __half * dsrc1,
                            uint8_t * dst, __half * ddst, const int * ids, int n0, int rb, int nd, bool q8, int bb) {
    const int r = n0 + blockIdx.x, owner = ids[r] / VOCAB_L, id = ids[r] - owner * VOCAB_L;
    const uint8_t * src = owner ? src1 : src0;
    const __half * dsrc = owner ? dsrc1 : dsrc0;
    if (q8) {
        for (int i = threadIdx.x; i < rb / 16; i += blockDim.x)
            *reinterpret_cast<int4 *>(dst + q8_wo(r, rb) + Q8_GW * (i >> 2) + 16 * (i & 3)) =
                *reinterpret_cast<const int4 *>(src + q8_wo(id, rb) + Q8_GW * (i >> 2) + 16 * (i & 3));
        for (int i = threadIdx.x; i < nd; i += blockDim.x)
            ddst[q8_do(r, nd) + Q8_GD * (i >> 1) + (i & 1)] = dsrc[q8_do(id, nd) + Q8_GD * (i >> 1) + (i & 1)];
        return;
    }
    // k-quant tiles: 16-byte chunk i of a row is chunk i % (bb / 16) of its block i / (bb / 16)
    for (int i = threadIdx.x; i < rb / 16; i += blockDim.x) {
        const size_t o = (size_t) KQ_T * bb * (i / (bb / 16)) + 16 * (i % (bb / 16));
        *reinterpret_cast<int4 *>(dst + kq_wo(r, rb, bb) + o) = *reinterpret_cast<const int4 *>(src + kq_wo(id, rb, bb) + o);
    }
    for (int i = threadIdx.x; i < nd; i += blockDim.x) ddst[(size_t) r * nd + i] = dsrc[(size_t) id * nd + i];
}

#include "prefill.cuh"

// ------------------------------------------------------------------------------------------
// host side: loading / repacking
// ------------------------------------------------------------------------------------------

uint16_t f2h_bits(float f) {
    __half h = __float2half(f);
    return *reinterpret_cast<uint16_t *>(&h);
}

template <class F>
void parallel_rows(int64_t n, F f) {
    int nt = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> th;
    for (int i = 0; i < nt; i++) {
        th.emplace_back([&, i] {
            for (int64_t r = i; r < n; r += nt) f(r);
        });
    }
    for (auto & t : th) t.join();
}

// Card of tensor-parallel rank r: MK_DEVICES="a,b" or devices 0, 1.
int mk_device(int rank) {
    const char * s = getenv("MK_DEVICES");
    if (!s) return rank;
    int d[2] = {0, 1};
    if (sscanf(s, "%d,%d", &d[0], &d[1]) < MK_TP) throw std::runtime_error("MK_DEVICES must list one device per rank, e.g. 0,1");
    return d[rank];
}

// Host barrier of the two per-card prefill threads; breaks (throws) for everyone once a thread fails.
struct HostBarrier {
    std::mutex m;
    std::condition_variable cv;
    int n, waiting = 0;
    unsigned long phase = 0;
    bool broken = false;
    explicit HostBarrier(int n_) : n(n_) {}
    void wait() {
        std::unique_lock<std::mutex> lk(m);
        if (broken) throw std::runtime_error("prefill: the other card's thread failed");
        const unsigned long ph = phase;
        if (++waiting == n) { waiting = 0; phase++; cv.notify_all(); return; }
        cv.wait(lk, [&] { return phase != ph || broken; });
        if (phase == ph) throw std::runtime_error("prefill: the other card's thread failed");
    }
    void abort() {
        std::lock_guard<std::mutex> lk(m);
        broken = true;
        cv.notify_all();
    }
};

} // namespace

struct MKEngine::Impl : KvDev {
    // TP > 1: one Impl per card; rank 0 (the engine's) owns rank 1's and drives both.
    int rank = 0, dev = 0;
    Impl * peer = nullptr;
    std::unique_ptr<Impl> sub;
    std::shared_ptr<HostBarrier> hbar;
    unsigned gen = 0; // launch generation, folded into the cross-card flag tags
    void use() const { CK(cudaSetDevice(dev)); }
    gguf::File file;
    ModelW hw{};
    Bufs hb{};
    std::vector<void *> allocs;
    void * embd_host = nullptr;
    int ctx;
    int nsm = 0;
    // Host state of one open sequence (rank 0): bound to a lane, whose device buffers hold its recurrent state.
    struct Seq {
        float smp_temp = 0.f, smp_top_p = 1.f, smp_min_p = 0.f; // sampling (temp 0: greedy)
        int smp_top_k = TOPK_MAX;
        unsigned long long smp_seed = 0;
        int ndraft = 0;       // MTP drafts the next launch produces
        int nd_avail = 0;     // drafts produced by the previous launch (verified by the next decode cycle)
        bool all_acc = true;  // the last decode cycle accepted all of its drafts
        std::vector<int> seq; // the sequence so far: prompt(s) and emitted tokens (position = index)
        Ctl hctl{};
        std::vector<int> blk; // KV pool blocks of positions [i * NP, (i + 1) * NP)
        int lane = -1;
    };
    int nseq = 1;    // lanes (sequences open at once)
    int cur = -1;    // the sequence every call acts on
    int cs_lane = 0; // the lane cS points at on this card
    std::map<int, Seq> seqs;
    int seq_next = 0;
    Seq dflt; // sampling and draft depth of sequences opened from now on
    Seq & S() {
        const auto it = seqs.find(cur);
        if (it == seqs.end()) throw std::runtime_error("no sequence selected");
        return it->second;
    }
    const Seq & S() const { return const_cast<Impl *>(this)->S(); }
    SeqBufs & sb() { return hb.sq[cs_lane]; }
    // Points this card's next launches at lane l (cS); the peer follows.
    void select_lane(int l) {
        if (sub) sub->select_lane(l);
        use();
        cs_lane = l;
        CK(cudaMemcpyToSymbol(cS, &hb.sq[l], sizeof(SeqBufs)));
    }
    // the selected sequence, with every card pointed at its lane
    Seq & focus() {
        Seq & q = S();
        if (q.lane != cs_lane) select_lane(q.lane);
        use();
        return q;
    }
    // focus(), opening a sequence first when none is selected (single-sequence callers never open one)
    Seq & need() {
        if (cur < 0) {
            int r;
            open_seq({}, r, false);
        }
        return focus();
    }
    // Clears lane l's recurrent state, h ring and Ctl on every card.
    void lane_reset(int l) {
        if (sub) sub->lane_reset(l);
        use();
        const SeqBufs & b = hb.sq[l];
        CK(cudaMemset(b.dstate, 0, sizeof(float) * DS_N));
        CK(cudaMemset(b.h_hist, 0, sizeof(float) * (size_t) H_RING * D));
        CK(cudaMemset(b.ctl, 0, sizeof(Ctl)));
        CK(cudaMemset(hb.cnt, 0, 16));
    }
    // Opens a sequence on a free lane and selects it; look: resume from the prefix cache (reused tokens of prompt).
    int open_seq(const std::vector<int> & prompt, int & reused, bool look) {
        Seq q = dflt;
        const auto o = kc->open(prompt.data(), (int) prompt.size(), look, q.blk);
        q.lane = o.lane;
        select_lane(o.lane);
        if (o.pos == 0) {
            lane_reset(o.lane);
        } else {
            q.hctl = o.meta;
            q.hctl.n = q.hctl.dn_c = o.pos;
            q.hctl.mtp_valid = std::min(q.hctl.mtp_valid, o.pos);
            q.seq.assign(prompt.begin(), prompt.begin() + o.pos);
            put_ctl(q.hctl);
            dv_add(q.seq.data(), q.seq.size());
        }
        reused = o.pos;
        const int id = seq_next++;
        seqs[id] = std::move(q);
        cur = id;
        return id;
    }
    // Snapshot of the selected sequence's state at its committed position (just put on the device), for reuse.
    void snap_take(Seq & q) {
        Ctl m = q.hctl;
        m.n = m.dn_c;
        m.mtp_valid = std::min(m.mtp_valid, m.dn_c);
        kc->snap_take(q.seq.data(), m.dn_c, q.blk, q.lane, m);
    }
    void dv_rebuild() {
        dv_reset();
        for (const auto & kv : seqs) dv_add(kv.second.seq.data(), kv.second.seq.size());
    }

    // ---------------- KV pool ----------------
    // Rank 0 owns the bookkeeping (kv_cache.h); every card holds the same block ids, so one host table feeds both.
    std::unique_ptr<KvCache<Ctl>> kc;
    int nblk_lane = 0;      // block-table entries per lane: ceil(ctx / NP)
    int * bt_dev = nullptr;  // this card's block tables [nseq][nblk_lane]
    int * bt_host = nullptr; // rank 0: pinned mirror, the source of the async uploads
    float * snap_stage = nullptr; // a snapshot's conv rows and h ring on their way to / from the pool
    std::vector<cudaEvent_t> bt_ev; // per lane: its last block-table upload on this card (bt_host is its source)
    int kv_np() const { return hb.kvq8 ? KvL<true>::NP : KvL<false>::NP; }
    size_t blk_bytes() const { return hb.kvq8 ? KvL<true>::BLK : KvL<false>::BLK; }

    // Sizes the pool on every card (the smallest card decides the block count) and builds empty tables.
    void pool_init() {
        // allocated lazily besides the pool: prefill scratch and runtime (~1.5 GB)
        const size_t reserve = (size_t) 1536 * 1024 * 1024;
        std::vector<Impl *> cards{this};
        if (peer) cards.push_back(peer);
        auto fit = [&](size_t blk_b) {
            size_t nb = SIZE_MAX;
            for (Impl * e : cards) {
                e->use();
                size_t free_b = 0, total_b = 0;
                CK(cudaMemGetInfo(&free_b, &total_b));
                const size_t avail = free_b > reserve ? free_b - reserve : 0;
                if (kv_mem > 0 && (size_t) kv_mem * 1024 * 1024 > avail)
                    throw std::runtime_error("kv_mem " + std::to_string(kv_mem) + " MB exceeds the free memory of device " +
                                             std::to_string(e->dev) + " (" + std::to_string(avail >> 20) + " MB after the reserve)");
                nb = std::min(nb, (kv_mem > 0 ? (size_t) kv_mem * 1024 * 1024 : avail) / blk_b);
            }
            return nb;
        };
        const int lane16 = (ctx + KvL<false>::NP - 1) / KvL<false>::NP;
        const bool q8 = kv_mode == 1 || (kv_mode < 0 && fit(KvL<false>::BLK) < (size_t) lane16);
        const size_t blk_b = q8 ? KvL<true>::BLK : KvL<false>::BLK;
        const int np = q8 ? KvL<true>::NP : KvL<false>::NP;
        const size_t nbz = std::min(fit(blk_b), (size_t) INT32_MAX);
        const int nb = (int) nbz;
        nblk_lane = (ctx + np - 1) / np;
        if (nb < nblk_lane)
            throw std::runtime_error("KV pool holds " + std::to_string((size_t) nb * np) + " tokens, less than one ctx sequence (" +
                                     std::to_string(ctx) + "): lower -c or raise kv_mem");
        for (Impl * e : cards) {
            e->use();
            e->nblk_lane = nblk_lane;
            e->hb.kvq8 = q8;
            e->hb.kv = e->dalloc<uint8_t>((size_t) nb * blk_b);
            e->bt_dev = e->dalloc<int>((size_t) nseq * nblk_lane); // zeroed: unused entries still name a valid block
            for (int s = 0; s < nseq; s++) e->hb.sq[s].bt = e->bt_dev + (size_t) s * nblk_lane;
            e->bt_ev.resize(nseq);
            for (cudaEvent_t & ev : e->bt_ev) CK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
            CK(cudaMemcpyToSymbol(cB, &e->hb, sizeof(Bufs)));
            CK(cudaMemcpyToSymbol(cS, &e->hb.sq[e->cs_lane], sizeof(SeqBufs)));
        }
        use();
        CK(cudaHostAlloc(&bt_host, (size_t) nseq * nblk_lane * sizeof(int), cudaHostAllocDefault));
        memset(bt_host, 0, (size_t) nseq * nblk_lane * sizeof(int));
        const int snap_nb = (int) (((DS_N + SNAP_STAGE) * 4 + blk_b - 1) / blk_b);
        kc = std::make_unique<KvCache<Ctl>>(nb, np, nblk_lane, snap_nb, nseq, this);
        fprintf(stderr, "[mk] KV pool %s, %.2f GB = %d blocks of %d tokens (max %zu tokens), %d lanes x %d ctx; a DeltaNet "
                        "snapshot takes %d blocks, snapshots at most %d blocks%s\n",
                q8 ? "q8_0" : "fp16", (double) nb * blk_b / 1e9, nb, np, (size_t) nb * np, nseq, ctx, snap_nb, kc->snap_cap(),
                snap_nb > kc->snap_cap() ? " (too small: no prefix cache)" : "");
    }

    // KvDev: the cache's device work, stream-ordered on every card before the next launch
    void copy_block(int src, int dst) override {
        const size_t bb = blk_bytes();
        for (Impl * e : {this, peer}) {
            if (!e) continue;
            e->use();
            CK(cudaMemcpyAsync(e->hb.kv + (size_t) dst * bb, e->hb.kv + (size_t) src * bb, bb, cudaMemcpyDeviceToDevice, 0));
        }
        use();
    }
    void upload_bt(int lane, int i0, const int * ids, int k) override {
        // pinned source, so truly async: the lane's previous upload must have read it before it is rewritten
        for (Impl * e : {this, peer})
            if (e) {
                e->use();
                CK(cudaEventSynchronize(e->bt_ev[lane]));
            }
        int * h = bt_host + (size_t) lane * nblk_lane + i0;
        std::copy(ids, ids + k, h);
        for (Impl * e : {this, peer}) {
            if (!e) continue;
            e->use();
            CK(cudaMemcpyAsync(e->bt_dev + (size_t) lane * nblk_lane + i0, h, k * sizeof(int), cudaMemcpyHostToDevice, 0));
            CK(cudaEventRecord(e->bt_ev[lane], 0));
        }
        use();
    }
    void snap_save(int lane, int pos, const std::vector<int> & blks) override { snap_copy(lane, pos, blks, true); }
    void snap_load(int lane, int pos, const std::vector<int> & blks) override { snap_copy(lane, pos, blks, false); }
    // Lane l's DeltaNet state at position p (recurrent state, conv rows of positions p-3 .. p-1, h ring) as one byte
    // stream laid over pool blocks blks: saved into them or loaded from them, on every card.
    void snap_copy(int l, int p, const std::vector<int> & blks, bool save) {
        if (sub) sub->snap_copy(l, p, blks, save);
        use();
        const SeqBufs & b = hb.sq[l];
        const size_t bb = blk_bytes(), row = (size_t) QKV_L * 4;
        float * cv = snap_stage;
        float * hh = snap_stage + (size_t) 3 * 48 * QKV_L;
        auto ring = [&](int j) { return b.conv + (size_t) (((p - 3 + j) % CONV_RING + CONV_RING) % CONV_RING) * QKV_L; };
        if (save) {
            for (int j = 0; j < 3; j++)
                CK(cudaMemcpy2DAsync(cv + (size_t) j * 48 * QKV_L, row, ring(j), CONV_RING * row, row, 48, cudaMemcpyDeviceToDevice, 0));
            CK(cudaMemcpyAsync(hh, b.h_hist, (size_t) H_RING * D * 4, cudaMemcpyDeviceToDevice, 0));
        }
        const std::pair<uint8_t *, size_t> parts[2] = {{reinterpret_cast<uint8_t *>(b.dstate), DS_N * 4},
                                                       {reinterpret_cast<uint8_t *>(snap_stage), SNAP_STAGE * 4}};
        size_t off = 0;
        for (const auto & pc : parts)
            for (size_t o = 0; o < pc.second;) {
                const size_t k = std::min(pc.second - o, bb - off % bb);
                uint8_t * dst = hb.kv + (size_t) blks.at(off / bb) * bb + off % bb;
                CK(cudaMemcpyAsync(save ? dst : pc.first + o, save ? pc.first + o : dst, k, cudaMemcpyDeviceToDevice, 0));
                o += k;
                off += k;
            }
        if (!save) {
            for (int j = 0; j < 3; j++)
                CK(cudaMemcpy2DAsync(ring(j), CONV_RING * row, cv + (size_t) j * 48 * QKV_L, row, row, 48, cudaMemcpyDeviceToDevice, 0));
            CK(cudaMemcpyAsync(b.h_hist, hh, (size_t) H_RING * D * 4, cudaMemcpyDeviceToDevice, 0));
        }
    }
    // MK_DEBUG_POOL builds: every refcount, the free list and the block tables against their holders after each call.
    void pool_check() {
#ifdef MK_DEBUG_POOL
        std::vector<const std::vector<int> *> v;
        std::vector<int> dev_bt(nblk_lane);
        for (const auto & kv : seqs) {
            const Seq & q = kv.second;
            v.push_back(&q.blk);
            for (const auto & kw : seqs)
                if (kw.first != kv.first && kw.second.lane == q.lane) throw std::logic_error("KV pool check: two sequences on one lane");
            for (Impl * e : {this, peer}) {
                if (!e || q.blk.empty()) continue;
                e->use();
                CK(cudaMemcpy(dev_bt.data(), e->bt_dev + (size_t) q.lane * nblk_lane, q.blk.size() * sizeof(int), cudaMemcpyDeviceToHost));
                for (size_t i = 0; i < q.blk.size(); i++)
                    if (dev_bt[i] != q.blk[i] || bt_host[(size_t) q.lane * nblk_lane + i] != q.blk[i])
                        throw std::logic_error("KV pool check: block table of sequence " + std::to_string(kv.first) + " entry " + std::to_string(i));
            }
            use();
        }
        kc->check(v);
#endif
    }

    // draft vocabulary: ids of head_d rows (base list + tokens added from the sequence)
    std::vector<int> dv_ids;
    std::vector<uint8_t> dv_in;
    int dv_base = 0;

    // Adds the tokens not yet in the draft head's vocabulary (copies their lm_head rows).
    void dv_add(const int * toks, size_t n) {
        if (sub) sub->dv_add(toks, n);
        if (dv_in.empty()) return; // full-vocabulary draft head
        use();
        const int n0 = (int) dv_ids.size();
        for (size_t i = 0; i < n && (int) dv_ids.size() < dv_base + DV_EXTRA - 1; i++)
            if (toks[i] % MK_TP == rank && !dv_in[toks[i]]) {
                dv_in[toks[i]] = 1;
                dv_ids.push_back(toks[i]);
            }
        if ((int) dv_ids.size() == n0) return;
        if (dv_ids.size() & 1) dv_ids.push_back(dv_ids.back()); // the draft lm_head takes rows in pairs
        const int k = (int) dv_ids.size() - n0;
        Mat & m = hw.head_d;
        CK(cudaMemcpy(const_cast<int *>(hw.draft_ids) + n0, dv_ids.data() + n0, k * 4, cudaMemcpyHostToDevice));
        draft_gather(n0, k);
        m.N = (int) dv_ids.size();
        CK(cudaMemcpyToSymbol(cW, &m.N, sizeof(int), offsetof(ModelW, head_d) + offsetof(Mat, N)));
    }
    // Copies head_d rows [n0, n0 + k) from the lm_head halves (the peer's over P2P for its half).
    void draft_gather(int n0, int k) {
        use();
        const Impl * h0 = (rank == 0 || !peer) ? this : peer;
        const Impl * h1 = (rank == 1 || !peer) ? this : peer;
        const Mat & m = hw.head_d;
        gather_rows<<<k, 256>>>(h0->hw.head.w, h0->hw.head.d, h1->hw.head.w, h1->hw.head.d, const_cast<uint8_t *>(m.w),
                                const_cast<__half *>(m.d), hw.draft_ids, n0, m.rb, m.nd, m.type == T_Q8_0, kq_bb(m.type));
        CK(cudaGetLastError());
    }
    void dv_reset() {
        if (sub) sub->dv_reset();
        if (dv_in.empty() || (int) dv_ids.size() == dv_base) return;
        use();
        for (size_t i = dv_base; i < dv_ids.size(); i++) dv_in[dv_ids[i]] = 0;
        for (int i = 0; i < dv_base; i++) dv_in[dv_ids[i]] = 1; // a padding duplicate may have been cleared
        dv_ids.resize(dv_base);
        hw.head_d.N = dv_base;
        CK(cudaMemcpyToSymbol(cW, &hw.head_d.N, sizeof(int), offsetof(ModelW, head_d) + offsetof(Mat, N)));
    }
    size_t wbytes = 0;
    bool profiling = false;
    bool prof_batch = false; // MK_PROFILE=batch: profile the batched launches instead of the single-slot ones
    int draft_prob = 1; // MK_DRAFT_SAMPLING=greedy clears it
    int row_rot = 0;    // batched cycles: first slot offered a spare draft row (rotates for fairness)
    // Row cap of a batched cycle: the tensor-core rows past MAXR_DP need every GEMV matrix in a type mma_tile
    // decodes and at most 2 residual items per block (resid_mma).
    int batch_rows() const {
        auto tc = [](const Mat & m) { return m.type == T_Q8_0 || m.type == T_Q4K || m.type == T_Q5K || m.type == T_Q6K; };
        bool ok = nsm >= 80 && tc(hw.eh) && tc(hw.head);
        for (int i = 0; i <= NL; i++) {
            const LayerW & W = hw.L[i];
            ok = ok && tc(W.fg) && tc(W.fu) && tc(W.fd) &&
                 (W.attn ? tc(W.q) && tc(W.k) && tc(W.v) && tc(W.o) : tc(W.qkv) && tc(W.gate) && tc(W.sout));
        }
        return ok ? MAXR : MAXR_DP;
    }
    std::vector<double> ph_ns = std::vector<double>(PH_COUNT, 0.0);
    std::vector<long> ph_n = std::vector<long>(PH_COUNT, 0);
    double launch_ns = 0;
    long launches = 0;
    // min over launches of each barrier-to-barrier interval, keyed by (T, barrier index); batched: rows per slot
    std::vector<std::vector<double>> best_ns = std::vector<std::vector<double>>(MAXR + 1);
    std::vector<std::vector<int>> best_tag = std::vector<std::vector<int>>(MAXR + 1);
    std::vector<std::vector<double>> best_fill = std::vector<std::vector<double>>(MAXR + 1);
    std::vector<double> arr_first = std::vector<double>(PH_COUNT, 0.0), arr_med = std::vector<double>(PH_COUNT, 0.0),
                        arr_last = std::vector<double>(PH_COUNT, 0.0);
    std::vector<long> arr_n = std::vector<long>(PH_COUNT, 0);
    // step record sums per phase: fill ns, fills, peer wait ns, marks (compute, sent, peer) ns and their counts
    std::vector<double> st_sum = std::vector<double>(PH_COUNT * PS_NV, 0.0);
    std::vector<long> st_cnt = std::vector<long>(PH_COUNT * PS_NV, 0);

    template <class T> T * dalloc(size_t n, bool zero = true) {
        void * p;
        CK(cudaMalloc(&p, n * sizeof(T) + 16));
        if (zero) CK(cudaMemset(p, 0, n * sizeof(T) + 16));
        allocs.push_back(p);
        return (T *) p;
    }

    // A tensor of the file in the quant type `to` its kernel is built for. ssm_alpha / ssm_beta are
    // folded to fp32 at load, so a file that stores them in any quant type still loads: they are
    // dequantized here by ggml, a few seconds.
    struct Src {
        uint32_t type = 0;
        std::vector<uint64_t> ne;
        const uint8_t * data = nullptr;
        uint64_t nbytes = 0;
        std::vector<uint8_t> own; // the re-encoded copy, when there is one
    };
    int n_conv = 0;
    double conv_s = 0;

    Src want(const std::string & name, int to) {
        const auto & t = file.get(name);
        Src s;
        s.type = t.type;
        s.ne = t.ne;
        s.data = t.data;
        s.nbytes = t.nbytes;
        if ((int) t.type == to || (to != T_F32 && t.type != gguf::Q4_K)) return s; // other mixes are refused by need() below
        const int64_t K = (int64_t) t.ne[0];
        int64_t rows = 1;
        for (size_t d = 1; d < t.ne.size(); d++) rows *= (int64_t) t.ne[d];
        const size_t rs_from = ggml_row_size((ggml_type) t.type, K), rs_to = ggml_row_size((ggml_type) to, K);
        const auto t0 = std::chrono::steady_clock::now();
        s.own.resize(rs_to * (size_t) rows);
        if (to != T_F32) ggml_quantize_init((ggml_type) to);
        const auto * tr = ggml_get_type_traits((ggml_type) t.type);
        parallel_rows(rows, [&](int64_t r) {
            thread_local std::vector<float> f;
            f.resize((size_t) K);
            tr->to_float(t.data + (size_t) r * rs_from, f.data(), K);
            if (to == T_F32) memcpy(s.own.data() + (size_t) r * rs_to, f.data(), (size_t) K * 4);
            else ggml_quantize_chunk((ggml_type) to, f.data(), s.own.data() + (size_t) r * rs_to, 0, 1, K, nullptr);
        });
        s.type = (uint32_t) to;
        s.data = s.own.data();
        s.nbytes = s.own.size();
        n_conv++;
        conv_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return s;
    }

    // This card's part of a weight: row ranges and K-column ranges [first, count) of the GGUF tensor
    // (block aligned), then zero rows / columns appended (zero bytes decode to 0 in every quant type).
    struct Slice {
        std::vector<std::pair<int, int>> rows, cols;
        int pad_rows = 0, pad_cols = 0;
    };
    static Src slice(const Src & s, const Slice & sl) {
        const int K = (int) s.ne[0], N = (int) s.ne[1];
        const auto rows = sl.rows.empty() ? std::vector<std::pair<int, int>>{{0, N}} : sl.rows;
        const auto cols = sl.cols.empty() ? std::vector<std::pair<int, int>>{{0, K}} : sl.cols;
        const ggml_type ty = (ggml_type) s.type;
        const int bs = (int) ggml_blck_size(ty);
        int K2 = sl.pad_cols, N2 = sl.pad_rows;
        for (const auto & c : cols) {
            if (c.first % bs || c.second % bs || c.first + c.second > K) throw std::runtime_error("slice: K range not block aligned");
            K2 += c.second;
        }
        for (const auto & r : rows) {
            if (r.first + r.second > N) throw std::runtime_error("slice: row range out of bounds");
            N2 += r.second;
        }
        if (sl.pad_cols % bs) throw std::runtime_error("slice: K padding not block aligned");
        const size_t rs = ggml_row_size(ty, K), rs2 = ggml_row_size(ty, K2);
        Src o;
        o.type = s.type;
        o.ne = {(uint64_t) K2, (uint64_t) N2};
        o.own.assign(rs2 * (size_t) N2, 0);
        size_t dr = 0;
        for (const auto & r : rows)
            for (int i = 0; i < r.second; i++, dr++) {
                const uint8_t * src = s.data + (size_t) (r.first + i) * rs;
                uint8_t * dst = o.own.data() + dr * rs2;
                for (const auto & c : cols) {
                    const size_t b = ggml_row_size(ty, c.second);
                    memcpy(dst, src + ggml_row_size(ty, c.first), b);
                    dst += b;
                }
            }
        o.data = o.own.data();
        o.nbytes = o.own.size();
        return o;
    }

    // f32 tensor restricted to element ranges [first, count) (this card's heads / channels); empty: all
    const float * up_f32_sel(const std::string & name, size_t expect, const std::vector<std::pair<int, int>> & sel) {
        if (sel.empty()) return up_f32(name, expect);
        const auto & t = file.get(name);
        if (t.type != gguf::F32) throw std::runtime_error(name + " not f32");
        if (t.nbytes / 4 != expect) throw std::runtime_error(name + " bad size");
        const float * f = (const float *) t.data;
        std::vector<float> v;
        for (const auto & r : sel) v.insert(v.end(), f + r.first, f + r.first + r.second);
        float * d = dalloc<float>(v.size(), false);
        CK(cudaMemcpy(d, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
        wbytes += v.size() * 4;
        return d;
    }

    const float * up_f32(const std::string & name, size_t expect) {
        const auto & t = file.get(name);
        if (t.type != gguf::F32) throw std::runtime_error(name + " not f32");
        size_t n = t.nbytes / 4;
        if (expect && n != expect) throw std::runtime_error(name + " bad size");
        float * d = dalloc<float>(n, false);
        CK(cudaMemcpy(d, t.data, t.nbytes, cudaMemcpyHostToDevice));
        wbytes += t.nbytes;
        return d;
    }

    Mat up_mat(const std::string & name, int K, int N, int to = -1, const Slice * sl = nullptr) {
        Src t = want(name, to < 0 ? (int) file.get(name).type : to);
        if ((int) t.ne[0] != K || (int) t.ne[1] != N) throw std::runtime_error(name + " bad shape");
        if (sl) {
            t = slice(t, *sl);
            K = (int) t.ne[0];
            N = (int) t.ne[1];
        }
        Mat m{};
        m.K = K;
        m.N = N;
        m.type = (int) t.type;
        const int nb = K / 256;
        if (t.type == gguf::Q4_K || t.type == gguf::Q5_K) {
            m.rb = (int) gguf::type_row_bytes(t.type, K);
            const int bb = kq_bb((int) t.type);
            std::vector<uint8_t> q((size_t) q8_rows(N) * m.rb, 0);
            const uint8_t * src = t.data;
            parallel_rows(N, [&](int64_t r) {
                for (int b = 0; b < nb; b++)
                    memcpy(q.data() + kq_wo((int) r, m.rb, bb) + (size_t) KQ_T * bb * b, src + (size_t) r * m.rb + (size_t) b * bb, bb);
            });
            uint8_t * d = dalloc<uint8_t>(q.size(), false);
            CK(cudaMemcpy(d, q.data(), q.size(), cudaMemcpyHostToDevice));
            m.w = d;
            wbytes += q.size();
        } else if (t.type == gguf::Q6_K) {
            m.rb = nb * 224;
            m.nd = 0;
            std::vector<uint8_t> q((size_t) q8_rows(N) * m.rb, 0);
            const uint8_t * src = t.data;
            parallel_rows(N, [&](int64_t r) {
                for (int b = 0; b < nb; b++) {
                    const uint8_t * blk = src + ((size_t) r * nb + b) * 210;
                    const uint8_t * ql = blk;
                    const uint8_t * qh = blk + 128;
                    const int8_t * sc = (const int8_t *) (blk + 192);
                    uint16_t dbits;
                    memcpy(&dbits, blk + 208, 2);
                    uint8_t vals[256];
                    for (int nn = 0; nn < 2; nn++)
                        for (int l = 0; l < 32; l++) {
                            const uint8_t * L = ql + 64 * nn;
                            const uint8_t * H = qh + 32 * nn;
                            vals[128 * nn + l + 0] = (L[l] & 0xF) | (((H[l] >> 0) & 3) << 4);
                            vals[128 * nn + l + 32] = (L[l + 32] & 0xF) | (((H[l] >> 2) & 3) << 4);
                            vals[128 * nn + l + 64] = (L[l] >> 4) | (((H[l] >> 4) & 3) << 4);
                            vals[128 * nn + l + 96] = (L[l + 32] >> 4) | (((H[l] >> 6) & 3) << 4);
                        }
                    uint8_t * dst = q.data() + kq_wo((int) r, m.rb, 224) + (size_t) KQ_T * 224 * b;
                    for (int p = 0; p < 4; p++)
                        for (int i = 0; i < 32; i++)
                            dst[32 * p + i] = (vals[64 * p + i] & 0xF) | ((vals[64 * p + 32 + i] & 0xF) << 4);
                    for (int p = 0; p < 4; p++)
                        for (int hh = 0; hh < 2; hh++) {
                            uint32_t hA = 0, hB = 0;
                            for (int i = 0; i < 16; i++) {
                                const int sh = 8 * (i % 4) + 2 * (i / 4);
                                hA |= (uint32_t) (vals[64 * p + 16 * hh + i] >> 4) << sh;
                                hB |= (uint32_t) (vals[64 * p + 32 + 16 * hh + i] >> 4) << sh;
                            }
                            memcpy(dst + 128 + 8 * (2 * p + hh), &hA, 4);
                            memcpy(dst + 128 + 8 * (2 * p + hh) + 4, &hB, 4);
                        }
                    for (int p = 0; p < 4; p++) {
                        memcpy(dst + 192 + 8 * p, sc + 4 * p, 4);
                        memcpy(dst + 192 + 8 * p + 4, &dbits, 2);
                    }
                }
            });
            uint8_t * dq = dalloc<uint8_t>(q.size(), false);
            CK(cudaMemcpy(dq, q.data(), q.size(), cudaMemcpyHostToDevice));
            m.w = dq;
            m.d = nullptr;
            wbytes += q.size();
        } else if (t.type == gguf::Q8_0) {
            const int nb32 = K / 32;
            m.rb = K;
            m.nd = nb32;
            std::vector<uint8_t> q((size_t) q8_rows(N) * K, 0);
            std::vector<uint16_t> ds((size_t) q8_rows(N) * nb32, 0);
            const uint8_t * src = t.data;
            parallel_rows(N, [&](int64_t r) {
                const uint8_t * row = src + (size_t) r * nb32 * 34;
                std::vector<int8_t> vals(K);
                for (int j = 0; j < nb32; j++) {
                    memcpy(&ds[q8_do((int) r, nb32) + Q8_GD * (j >> 1) + (j & 1)], row + 34 * j, 2);
                    memcpy(&vals[32 * j], row + 34 * j + 2, 32);
                }
                uint8_t * dst = q.data() + q8_wo((int) r, K);
                for (int u = 0; u < K / 32; u++) {
                    const int b = u >> 3, p = (u >> 1) & 3, hh = u & 1;
                    uint8_t * du = dst + Q8_GW * (u >> 1) + 32 * (u & 1);
                    memcpy(du, &vals[256 * b + 64 * p + 16 * hh], 16);
                    memcpy(du + 16, &vals[256 * b + 64 * p + 32 + 16 * hh], 16);
                }
            });
            uint8_t * dq = dalloc<uint8_t>(q.size(), false);
            CK(cudaMemcpy(dq, q.data(), q.size(), cudaMemcpyHostToDevice));
            uint16_t * dd = dalloc<uint16_t>(ds.size(), false);
            CK(cudaMemcpy(dd, ds.data(), ds.size() * 2, cudaMemcpyHostToDevice));
            m.w = dq;
            m.d = (const __half *) dd;
            wbytes += q.size() + ds.size() * 2;
        } else {
            throw std::runtime_error(name + ": unsupported quant type " + std::to_string(t.type));
        }
        return m;
    }

    QBuf qbuf(int U) {
        QBuf q;
        q.U = U;
        q.xa = dalloc<int8_t>((size_t) MAXR * U * 16);
        q.xb = dalloc<int8_t>((size_t) MAXR * U * 16);
        q.meta = dalloc<uint2>((size_t) MAXR * U);
        q.ssq = dalloc<float>((size_t) MAXR * U);
        return q;
    }

    int n_draft_max;
    int kv_mode; // KV cache format: 0 fp16, 1 q8_0, -1 auto (q8_0 when the fp16 pool holds less than one ctx sequence)
    int kv_mem;  // KV pool MB per card, 0: the free memory after everything else
    Impl(const std::string & path, int ctx_, int ndraft_, const std::string & draft_vocab, int kv_mode_, int nseq_, int kv_mem_,
         int rank_ = 0)
        : rank(rank_), dev(mk_device(rank_)), file(path), ctx(ctx_), n_draft_max(ndraft_), kv_mode(kv_mode_), kv_mem(kv_mem_) {
        if (kv_mem_ < 0) throw std::runtime_error("kv_mem must be >= 0 (MB per card, 0: all free memory)");
        if (nseq_ < 1 || nseq_ > MAXB) throw std::runtime_error("lanes (n_seq) must be in [1, " + std::to_string(MAXB) + "]");
        nseq = nseq_;
        dflt.ndraft = dflt.nd_avail = ndraft_;
        if (file.strs["general.architecture"] != "qwen35") throw std::runtime_error("expected qwen35 architecture");
        if (file.ints["qwen35.block_count"] != NLA || file.ints["qwen35.embedding_length"] != D)
            throw std::runtime_error("unexpected model dimensions (this kernel is specialised for the 27B)");
#if MK_TP > 1
        if (rank == 0) {
            const int d1 = mk_device(1);
            int a = 0, b = 0;
            if (d1 != dev) {
                CK(cudaDeviceCanAccessPeer(&a, dev, d1));
                CK(cudaDeviceCanAccessPeer(&b, d1, dev));
            }
            if (!a || !b)
                throw std::runtime_error("TP=2 needs two distinct GPUs with peer-to-peer access both ways (devices " +
                                         std::to_string(dev) + ", " + std::to_string(d1) + ")");
            // Peer access outlives an engine, so a second engine in the same process finds it already enabled.
            auto peer = [](int to) {
                cudaError_t e = cudaDeviceEnablePeerAccess(to, 0);
                if (e == cudaErrorPeerAccessAlreadyEnabled) {
                    cudaGetLastError();
                    e = cudaSuccess;
                }
                CK(e);
            };
            CK(cudaSetDevice(d1));
            peer(dev);
            CK(cudaSetDevice(dev));
            peer(d1);
        }
#endif
        use();
        if (rank) pf_prof = false;
        cudaDeviceProp prop;
        CK(cudaGetDeviceProperties(&prop, dev));
        nsm = prop.multiProcessorCount;

        // TP > 1: this card's rows / K columns of each sharded weight (design: README, tensor parallel).
        // Delta-net value heads are interleaved: card r owns {h : (h % 16) / 8 == r}, so the local
        // value head hl -> h = (hl / 8) * 16 + 8 r + hl % 8 keeps key head hl % 8 = (h % 16) - 8 r.
        const bool tp = MK_TP > 1;
        const int r = rank;
        std::vector<std::pair<int, int>> vh, vrows;     // owned value heads; their 128-row blocks
        for (int g = 0; g < NVH / 16 && tp; g++) {
            vh.push_back({g * 16 + 8 * r, 8});
            vrows.push_back({(g * 16 + 8 * r) * HK, 8 * HK});
        }
        Slice s_q, s_kv, s_o, s_qkv, s_gate, s_sout, s_gu, s_down, s_eh, s_head;
        std::vector<std::pair<int, int>> conv_sel;
        if (tp) {
            s_q.rows = {{r * QG_L, QG_L}};
            s_kv.rows = {{r * KVD_L, KVD_L}};
            s_o.cols = {{r * VD_L, VD_L}};
            s_qkv.rows = {{r * NKH_L * HK, NKH_L * HK}, {NKH * HK + r * NKH_L * HK, NKH_L * HK}};
            for (const auto & v : vrows) s_qkv.rows.push_back({2 * NKH * HK + v.first, v.second});
            s_gate.rows = vrows;
            s_sout.cols = vrows;
            for (const auto & c : s_qkv.rows) conv_sel.push_back({4 * c.first, 4 * c.second});
            const int ffc = FF / MK_TP; // 8704 = 34 super-blocks; padded to FF_L with zero rows / columns
            s_gu.rows = {{r * ffc, ffc}};
            s_gu.pad_rows = FF_L - ffc;
            s_down.cols = {{r * ffc, ffc}};
            s_down.pad_cols = FF_L - ffc;
            s_eh.cols = {{r * D, D}};
            s_head.rows = {{r * VOCAB_L, VOCAB_L}};
        }
        auto S = [&](const Slice & x) { return tp ? &x : nullptr; };

        auto blk = [](int i, const char * s) { return "blk." + std::to_string(i) + "." + s; };
        // The decode kernels are compiled for one quant type per tensor role (resid32 / gemv_pairs
        // take it as a template argument), so a GGUF quantized with a different per-tensor recipe
        // loads without complaint and then reads past its weights on the first token. Check every
        // role against the type its kernel was built for.
        auto tname = [](int t) -> std::string {
            switch (t) {
                case gguf::F32: return "F32";
                case gguf::F16: return "F16";
                case gguf::Q8_0: return "Q8_0";
                case gguf::Q4_K: return "Q4_K";
                case gguf::Q5_K: return "Q5_K";
                case gguf::Q6_K: return "Q6_K";
                default: return "type " + std::to_string(t);
            }
        };
        const std::initializer_list<int> kquant = {T_Q4K, T_Q5K, T_Q6K, T_Q8_0};
        auto need = [&](const std::string & name, int got, std::initializer_list<int> ok) {
            for (int t : ok)
                if (got == t) return;
            std::string want;
            for (int t : ok) want += (want.empty() ? "" : " or ") + tname(t);
            throw std::runtime_error(name + " is " + tname(got) + ", the kernel needs " + want +
                                     " (see README: the GGUF)");
        };
        for (int i = 0; i < NLA; i++) {
            LayerW & L = hw.L[i];
            const bool mtp = i == NL;
            L.attn = mtp || ((i + 1) % 4 == 0);
            L.attn_norm = up_f32(blk(i, "attn_norm.weight"), D);
            L.post_norm = up_f32(blk(i, "post_attention_norm.weight"), D);
            if (L.attn) {
                L.q = up_mat(blk(i, "attn_q.weight"), D, QG, -1, S(s_q));
                L.k = up_mat(blk(i, "attn_k.weight"), D, KVD, -1, S(s_kv));
                L.v = up_mat(blk(i, "attn_v.weight"), D, KVD, -1, S(s_kv));
                L.o = up_mat(blk(i, "attn_output.weight"), VD, D, -1, S(s_o));
                need(blk(i, "attn_q.weight"), L.q.type, kquant);
                need(blk(i, "attn_k.weight"), L.k.type, kquant);
                need(blk(i, "attn_v.weight"), L.v.type, kquant);
                need(blk(i, "attn_output.weight"), L.o.type, kquant);
                L.qn = up_f32(blk(i, "attn_q_norm.weight"), HD);
                L.kn = up_f32(blk(i, "attn_k_norm.weight"), HD);
            } else {
                L.qkv = up_mat(blk(i, "attn_qkv.weight"), D, QKV, -1, S(s_qkv));
                L.gate = up_mat(blk(i, "attn_gate.weight"), D, VD, -1, S(s_gate));
                L.sout = up_mat(blk(i, "ssm_out.weight"), VD, D, -1, S(s_sout));
                need(blk(i, "attn_qkv.weight"), L.qkv.type, kquant);
                need(blk(i, "attn_gate.weight"), L.gate.type, kquant);
                need(blk(i, "ssm_out.weight"), L.sout.type, kquant);
                L.conv = up_f32_sel(blk(i, "ssm_conv1d.weight"), 4 * QKV, conv_sel);
                L.dt = up_f32_sel(blk(i, "ssm_dt.bias"), NVH, vh);
                L.A = up_f32_sel(blk(i, "ssm_a"), NVH, vh);
                L.snorm = up_f32(blk(i, "ssm_norm.weight"), HK);
                // alpha/beta projections in fp32 with the attn_norm weight folded in
                const Src ta = want(blk(i, "ssm_alpha.weight"), T_F32);
                const Src tb = want(blk(i, "ssm_beta.weight"), T_F32);
                const auto & tn = file.get(blk(i, "attn_norm.weight"));
                // read below as raw floats, so anything but F32 would be reinterpreted silently
                need(blk(i, "ssm_alpha.weight"), (int) ta.type, {T_F32});
                need(blk(i, "ssm_beta.weight"), (int) tb.type, {T_F32});
                if (ta.ne[0] != D || ta.ne[1] != NVH || tb.ne[0] != D || tb.ne[1] != NVH)
                    throw std::runtime_error(blk(i, "ssm_alpha/beta.weight") + " bad shape");
                std::vector<float> ab((size_t) 2 * NVH_L * D);
                const float * fa = (const float *) ta.data;
                const float * fb = (const float *) tb.data;
                const float * fn = (const float *) tn.data;
                for (int hl = 0; hl < NVH_L; hl++) {
                    const int h = tp ? (hl / 8) * 16 + 8 * r + hl % 8 : hl;
                    for (int e = 0; e < D; e++) {
                        ab[(size_t) hl * D + e] = fa[(size_t) h * D + e] * fn[e];
                        ab[(size_t) (NVH_L + hl) * D + e] = fb[(size_t) h * D + e] * fn[e];
                    }
                }
                float * d = dalloc<float>(ab.size(), false);
                CK(cudaMemcpy(d, ab.data(), ab.size() * 4, cudaMemcpyHostToDevice));
                L.ab = d;
                wbytes += ab.size() * 4;
            }
            L.fg = up_mat(blk(i, "ffn_gate.weight"), D, FF, -1, S(s_gu));
            L.fu = up_mat(blk(i, "ffn_up.weight"), D, FF, -1, S(s_gu));
            L.fd = up_mat(blk(i, "ffn_down.weight"), FF, D, -1, S(s_down));
            need(blk(i, "ffn_gate.weight"), L.fg.type, kquant);
            need(blk(i, "ffn_up.weight"), L.fu.type, kquant);
            if (L.fg.type != L.fu.type)
                throw std::runtime_error("blk " + std::to_string(i) + " ffn gate/up have different types: " +
                                         tname(L.fg.type) + " vs " + tname(L.fu.type));
            need(blk(i, "ffn_down.weight"), L.fd.type, kquant);
        }
        hw.eh = up_mat(blk(NL, "nextn.eh_proj.weight"), 2 * D, D, -1, S(s_eh));
        need(blk(NL, "nextn.eh_proj.weight"), hw.eh.type, kquant);
        hw.enorm = up_f32(blk(NL, "nextn.enorm.weight"), D);
        hw.hnorm = up_f32(blk(NL, "nextn.hnorm.weight"), D);
        hw.shnorm = up_f32(blk(NL, "nextn.shared_head_norm.weight"), D);
        hw.onorm = up_f32("output_norm.weight", D);
        hw.head = up_mat("output.weight", D, VOCAB, -1, S(s_head));
        need("output.weight", hw.head.type, kquant);
        hw.head_d = hw.head;
        hw.draft_ids = nullptr;
        if (!draft_vocab.empty()) {
            FILE * f = fopen(draft_vocab.c_str(), "rb");
            if (!f) throw std::runtime_error("cannot open draft vocab " + draft_vocab);
            std::vector<int> all, ids;
            int id;
            while (fread(&id, 4, 1, f) == 1) {
                if (id < 0 || id >= VOCAB) throw std::runtime_error("bad id in draft vocab");
                all.push_back(id);
            }
            fclose(f);
            // every MK_TP-th entry so both cards draft over equally many rows; tokens added later go
            // to the card of their id parity (dv_add)
            for (size_t i = rank; i < all.size(); i += MK_TP) ids.push_back(all[i]);
            if (ids.size() & 1) ids.push_back(ids.back());
            // room for DV_EXTRA more rows: tokens of the current sequence that are not in the list
            // are added on the fly (identifiers of pasted code, see dv_add)
            const int cap = (int) ids.size() + DV_EXTRA;
            int * did = dalloc<int>(cap, false);
            CK(cudaMemcpy(did, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
            Mat m = hw.head;
            m.N = (int) ids.size();
            uint8_t * w = dalloc<uint8_t>((size_t) q8_rows(cap) * m.rb, false);
            __half * d = m.nd ? dalloc<__half>((size_t) q8_rows(cap) * m.nd, false) : nullptr;
            m.w = w;
            m.d = d;
            hw.head_d = m;
            hw.draft_ids = did;
            dv_ids = ids;
            dv_base = (int) ids.size();
            if (MK_TP == 1 && m.N) draft_gather(0, m.N); // TP=2 gathers once the peer exists
            CK(cudaDeviceSynchronize());
            dv_in.assign(VOCAB, 0);
            for (int t : all) dv_in[t] = 1; // also the other card's base rows, so dv_add never duplicates them
            fprintf(stderr, "[mk] MTP draft head restricted to %d tokens (%s)\n", m.N, draft_vocab.c_str());
        }
        {
            const auto & t = file.get("token_embd.weight");
            need("token_embd.weight", (int) t.type, kquant);
            hw.embd_type = (int) t.type;
            hw.embd_rb = (int) ggml_row_size((ggml_type) t.type, D);
            // mapped pinned host memory: a cycle reads only a few rows over PCIe, and the table leaves VRAM
            const char * ed = getenv("MK_EMBD_DEVICE");
            if (!(ed && ed[0] == '1') &&
                cudaHostAlloc(&embd_host, t.nbytes, cudaHostAllocMapped | cudaHostAllocWriteCombined) == cudaSuccess) {
                memcpy(embd_host, t.data, t.nbytes);
                void * d = nullptr;
                CK(cudaHostGetDevicePointer(&d, embd_host, 0));
                hw.embd = (const uint8_t *) d;
                fprintf(stderr, "[mk] token_embd in mapped host memory (%.2f GB)\n", t.nbytes / 1e9);
            } else {
                cudaGetLastError();
                embd_host = nullptr;
                uint8_t * d = dalloc<uint8_t>(t.nbytes, false);
                CK(cudaMemcpy(d, t.data, t.nbytes, cudaMemcpyHostToDevice));
                hw.embd = d;
            }
        }
        CK(cudaMemcpyToSymbol(cW, &hw, sizeof(ModelW)));
        if (n_conv)
            fprintf(stderr, "[mk] %d ssm alpha/beta tensors converted to F32 in %.0f s; quantized weights accept Q4_K, Q5_K, Q6_K or Q8_0 "
                            "(README: the GGUF)\n", n_conv, conv_s);

        hb.x = dalloc<float>(MAXR * D);
        hb.xm = dalloc<float>(MAXR * D);
        hb.qn = qbuf(U_D);
        hb.qo = qbuf(U_VD_L);
        hb.qa = qbuf(U_FF_L);
        hb.qe = qbuf(U_EH_L);
        hb.z = dalloc<float>(MAXR * VD_L);
        hb.ab = dalloc<float>(AB_SPLIT * MAXR * 2 * NVH_L);
        hb.qbuf = dalloc<float>(MAXR * QG_L);
        hb.kbuf = dalloc<float>(MAXR * KVD_L);
        hb.vbuf = dalloc<float>(MAXR * KVD_L);
        hb.qkvs = dalloc<float>((size_t) MAXR * QKV_L);
        {
            // the KV pool and block tables come later (pool_init), once both cards' free memory is known
            for (int s = 0; s < nseq; s++) {
                SeqBufs & b = hb.sq[s];
                b.conv = dalloc<float>((size_t) 48 * CONV_RING * QKV_L);
                b.kvgb = dalloc<float>((size_t) 48 * KV_RING * KVGB_L);
                b.dstate = dalloc<float>((size_t) 48 * NVH_L * 128 * 128);
                b.h_hist = dalloc<float>((size_t) H_RING * D);
                b.hm = dalloc<float>(D);
                b.ctl = dalloc<Ctl>(1);
                b.dcand = dalloc<float2>((size_t) MAXT * TOPK_MAX);
            }
            hb.nseq = nseq;
            snap_stage = dalloc<float>(SNAP_STAGE, false);
        }
        // a batched cycle's attention items of all slots: up to MAXB x NKV_L x MAX_CH
        hb.apart = dalloc<float>((size_t) MAXB * NKV_L * MAX_CH * AR * 256);
        hb.aml = dalloc<float2>((size_t) MAXB * NKV_L * MAX_CH * AR);
        {
            const int rope_n = (ctx + KvL<true>::NP - 1) / KvL<true>::NP * KvL<true>::NP; // whole blocks of either format
            std::vector<float2> rt((size_t) rope_n * 32);
            for (int p = 0; p < rope_n; p++)
                for (int i = 0; i < 32; i++) {
                    const double th = (double) p * std::pow((double) ROPE_BASE, -2.0 * i / 64.0);
                    rt[(size_t) p * 32 + i] = make_float2((float) std::cos(th), (float) std::sin(th));
                }
            float2 * d = dalloc<float2>(rt.size(), false);
            CK(cudaMemcpy(d, rt.data(), rt.size() * sizeof(float2), cudaMemcpyHostToDevice));
            hb.rope = d;
        }
        hb.blk_best = dalloc<float2>((size_t) nsm * MAXR * TOPK_MAX);
        hb.cand = dalloc<float2>((size_t) MAXR * TOPK_MAX);
        hb.logits = dalloc<float>((size_t) MAXT * VOCAB_L);
        hb.bar = dalloc<unsigned>(4);
        hb.cnt = dalloc<unsigned>(4);
        hb.ctx = ctx;
        hb.opt = getenv("MK_OPT") ? atoi(getenv("MK_OPT")) : OPT_XR8;
        if (rank == 0 && hb.opt) fprintf(stderr, "[mk] MK_OPT %d\n", hb.opt);
        profiling = getenv("MK_PROFILE") != nullptr;
        prof_batch = profiling && !strcmp(getenv("MK_PROFILE"), "batch");
        {
            const char * ds = getenv("MK_DRAFT_SAMPLING");
            draft_prob = ds && !strcmp(ds, "greedy") ? 0 : 1;
            if (rank == 0) fprintf(stderr, "[mk] draft sampling: %s\n", draft_prob ? "probabilistic" : "greedy");
        }
        hb.prof = profiling ? dalloc<unsigned long long>(PROF_N) : nullptr;
#if MK_TP > 1
        // flags are zeroed once here and never per launch: tags carry the launch generation
        hb.rank = rank;
        for (auto & b : hb.xr) b = dalloc<__nv_bfloat16>((size_t) D * MAXR);
        hb.fl = dalloc<unsigned>(D / 32);
        hb.lx = dalloc<float2>((size_t) MAXR * TOPK_MAX);
        hb.lfl = dalloc<unsigned>(1);
#endif
        CK(cudaMemcpyToSymbol(cB, &hb, sizeof(Bufs)));

        CK(cudaFuncSetAttribute(mk_cycle, cudaFuncAttributeMaxDynamicSharedMemorySize, SM_TOTAL));
#if MK_TP > 1
        CK(cudaFuncSetAttribute(mk_bcycle, cudaFuncAttributeMaxDynamicSharedMemorySize, SM_TOTAL));
#endif
        int nb = 0;
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, mk_cycle, NT, SM_TOTAL));
        if (nb < 1) throw std::runtime_error("megakernel does not fit on an SM");
        fprintf(stderr, "[mk] weights %.2f GB on device %d (TP rank %d of %d), %d SMs, ctx %d, n_draft %d\n", wbytes / 1e9, dev,
                rank, MK_TP, nsm, ctx, n_draft_max);
#if MK_TP > 1
        if (rank == 0) {
            sub = std::make_unique<Impl>(path, ctx_, ndraft_, draft_vocab, kv_mode_, nseq_, kv_mem_, 1);
            if (sub->nsm != nsm) throw std::runtime_error("TP=2 needs two GPUs with the same SM count (equal grids)");
            peer = sub.get();
            sub->peer = this;
            hbar = std::make_shared<HostBarrier>(MK_TP);
            sub->hbar = hbar;
            for (Impl * e : {this, peer}) {
                const Impl * o = e->peer;
                for (int i = 0; i < 2; i++) e->hb.peer_xr[i] = o->hb.xr[i];
                e->hb.peer_fl = o->hb.fl;
                e->hb.peer_lx = o->hb.lx;
                e->hb.peer_lfl = o->hb.lfl;
                e->use();
                CK(cudaMemcpyToSymbol(cB, &e->hb, sizeof(Bufs)));
            }
            for (Impl * e : {this, peer})
                if (e->hw.draft_ids && e->hw.head_d.N) {
                    e->draft_gather(0, e->hw.head_d.N);
                    CK(cudaDeviceSynchronize());
                }
            use();
        }
#endif
        if (rank == 0) pool_init();
    }

    ~Impl() {
        sub.reset();
        use();
        print_profile();
        if (bt_host) cudaFreeHost(bt_host);
        for (cudaEvent_t e : ev_done)
            if (e) cudaEventDestroy(e);
        for (cudaEvent_t e : bt_ev) cudaEventDestroy(e);
        for (void * p : allocs) cudaFree(p);
        if (embd_host) cudaFreeHost(embd_host);
    }

    // One cooperative launch per card, back to back with no blocking call between them: the cards
    // wait on each other's flags inside the kernel.
    void coop(Launch & P) {
#if MK_TP > 1
        P.gen = ++gen; // starts at 1, so the zeroed flags never match a tag
        for (Impl * e : {this, peer}) {
            e->use();
            CK(cudaMemcpyToSymbolAsync(L, &P, sizeof(Launch), 0, cudaMemcpyHostToDevice, 0));
            CK(cudaMemsetAsync(e->hb.bar, 0, 16));
            void * args[] = {&P};
            CK(cudaLaunchCooperativeKernel((void *) mk_cycle, dim3(nsm), dim3(NT), args, SM_TOTAL, 0));
        }
        use();
#else
        void * args[] = {&P};
        CK(cudaMemsetAsync(hb.bar, 0, 16));
        CK(cudaLaunchCooperativeKernel((void *) mk_cycle, dim3(nsm), dim3(NT), args, SM_TOTAL, 0));
#endif
    }

#if MK_TP > 1
    // A batched cycle (mk_bcycle) on both cards; the slots' Ctl stay on the device.
    void launch_batch(BLaunch & P) {
        Launch G{};
        G.gen = ++gen;
        for (Impl * e : {this, peer}) {
            e->use();
            CK(cudaMemcpyToSymbolAsync(L, &G, sizeof(Launch), 0, cudaMemcpyHostToDevice, 0));
            CK(cudaMemsetAsync(e->hb.bar, 0, 16));
            void * args[] = {&P};
            if (e == this && prof_batch) CK(cudaMemsetAsync(hb.prof, 0, PROF_N * 8));
            CK(cudaLaunchCooperativeKernel((void *) mk_bcycle, dim3(nsm), dim3(NT), args, SM_TOTAL, 0));
        }
        use();
        if (prof_batch) prof_collect(P.Tb);
    }
#endif
    // slot s's Ctl from rank 0, after both cards finished
    void get_ctl_slot(int s, Ctl & c) {
        if (sub) {
            sub->use();
            CK(cudaDeviceSynchronize());
        }
        use();
        CK(cudaMemcpy(&c, hb.sq[s].ctl, sizeof(Ctl), cudaMemcpyDeviceToHost));
    }
    // host -> every card's Ctl (the cards keep identical copies); reads come from rank 0 after both finished
    void put_ctl(const Ctl & c) {
        if (sub) sub->put_ctl(c);
        use();
        CK(cudaMemcpy(sb().ctl, &c, sizeof(Ctl), cudaMemcpyHostToDevice));
    }
    void get_ctl(Ctl & c) {
        if (sub) {
            sub->use();
            CK(cudaDeviceSynchronize());
        }
        use();
        CK(cudaMemcpy(&c, sb().ctl, sizeof(Ctl), cudaMemcpyDeviceToHost));
    }
    // rows s .. s + n - 1 of the last scoring launch's logits, both vocab halves -> dst [n][VOCAB]
    void get_logits(float * dst, int s, int n) {
        for (int r = 0; r < MK_TP; r++) {
            const Impl * e = r ? peer : this;
            e->use();
            CK(cudaMemcpy2D(dst + (size_t) r * VOCAB_L, (size_t) VOCAB * 4, e->hb.logits + (size_t) s * VOCAB_L, (size_t) VOCAB_L * 4,
                            (size_t) VOCAB_L * 4, n, cudaMemcpyDeviceToHost));
        }
        use();
    }

    void launch(int mode, int T, int last, int score = 0) {
        use();
        const Seq & q = S();
        Launch P{mode, T, last, q.ndraft, score, q.smp_temp, q.smp_top_p, q.smp_min_p, q.smp_top_k, q.smp_seed, draft_prob};
        if (profiling) CK(cudaMemsetAsync(hb.prof, 0, PROF_N * 8));
        coop(P);
        if (profiling && !prof_batch && mode == MODE_DECODE) prof_collect(T);
    }

    // folds the last launch's barrier timestamps into the profile
    void prof_collect(int T) {
        std::vector<unsigned long long> pr(PROF_N);
        CK(cudaMemcpy(pr.data(), hb.prof, pr.size() * 8, cudaMemcpyDeviceToHost));
        unsigned long long prev = pr[0] >> 8;
        auto & bn = best_ns[T];
        auto & bt = best_tag[T];
        auto & bf = best_fill[T];
        for (int k = 1; k < 4096 && pr[k]; k++) {
            const unsigned long long t = pr[k] >> 8;
            const int tag = (int) (pr[k] & 0xff);
            const double dt = (double) (t - prev);
            ph_ns[tag] += dt;
            ph_n[tag]++;
            launch_ns += dt;
            prev = t;
            const unsigned long long mark = pr[2048 + k] & ((1ull << 56) - 1);
            const double fill = mark ? (double) ((mark - (t - (unsigned long long) dt)) & ((1ull << 56) - 1)) : 0.0;
            if ((int) bn.size() < k) { bn.resize(k, 1e30); bt.resize(k, -1); bf.resize(k, 0.0); }
            if (k < 1024 && pr[PROF_ST + k * PROF_SN + PS_NV]) {
                const unsigned long long * st = &pr[PROF_ST + k * PROF_SN];
                const double ns_per_cyc = dt / (double) st[PS_NV];
                st_sum[tag * PS_NV + 0] += st[0] * ns_per_cyc;
                st_sum[tag * PS_NV + 1] += (double) st[1];
                st_sum[tag * PS_NV + 2] += st[2] * ns_per_cyc;
                st_cnt[tag * PS_NV]++;
                for (int i = 3; i < PS_NV; i++)
                    if (st[i]) { st_sum[tag * PS_NV + i] += st[i] * ns_per_cyc; st_cnt[tag * PS_NV + i]++; }
            }
            if (k < 1024) {
                std::vector<double> arr;
                const unsigned long long t0 = t - (unsigned long long) dt;
                for (int b = 0; b < nsm; b++) {
                    const unsigned long long a = pr[4096 + k * 128 + b];
                    if (a) arr.push_back((double) (((a & ((1ull << 56) - 1)) - t0) & ((1ull << 56) - 1)));
                }
                if (arr.size() == (size_t) nsm) {
                    std::sort(arr.begin(), arr.end());
                    arr_first[tag] += arr.front();
                    arr_med[tag] += arr[arr.size() / 2];
                    arr_last[tag] += arr.back();
                    arr_n[tag]++;
                }
            }
            if (bt[k - 1] != tag) { bt[k - 1] = tag; bn[k - 1] = dt; bf[k - 1] = fill; } // sequence changed
            else if (dt < bn[k - 1]) { bn[k - 1] = dt; bf[k - 1] = fill; }
        }
        launches++;
    }

    void print_profile() {
        if (!profiling || !launches) return;
        static const char * names[PH_COUNT] = {"embed", "attn_proj", "attn_core", "attn_comb", "resid_o", "dn_proj",
                                               "dn_core", "resid_sout", "ffn_gu", "ffn_down", "lm_head", "mtp_m0",
                                               "mtp_eh", "mtp_attn_proj", "mtp_attn_core", "mtp_attn_comb",
                                               "mtp_resid_o", "mtp_ffn_gu", "mtp_ffn_down", "mtp_lm_head"};
        fprintf(stderr, "[mk-profile] %ld decode launches, mean %.3f ms in barriers-to-barrier phases\n", launches,
                launch_ns / launches / 1e6);
        for (int i = 0; i < PH_COUNT; i++) {
            if (!ph_n[i]) continue;
            fprintf(stderr, "  %-16s total/launch %8.3f ms   per call %8.2f us   calls/launch %6.1f\n", names[i],
                    ph_ns[i] / launches / 1e6, ph_ns[i] / ph_n[i] / 1e3, (double) ph_n[i] / launches);
        }
        fprintf(stderr, "[mk-profile] block arrival at barrier (us after phase start; mean over launches): first / median / last\n");
        for (int i = 0; i < PH_COUNT; i++)
            if (arr_n[i])
                fprintf(stderr, "  %-16s %8.2f %8.2f %8.2f\n", names[i], arr_first[i] / arr_n[i] / 1e3, arr_med[i] / arr_n[i] / 1e3,
                        arr_last[i] / arr_n[i] / 1e3);
        fprintf(stderr, "[mk-profile] steps of block 0 (us per call, mean): fill total / fills / peer wait / compute done / "
                        "reduced / written / sent / peer in\n");
        for (int i = 0; i < PH_COUNT; i++) {
            const long n = st_cnt[i * PS_NV];
            if (!n) continue;
            auto mk = [&](int j) { return st_cnt[i * PS_NV + j] ? st_sum[i * PS_NV + j] / st_cnt[i * PS_NV + j] / 1e3 : -1.0; };
            fprintf(stderr, "  %-16s %8.2f %6.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f\n", names[i], st_sum[i * PS_NV] / n / 1e3,
                    st_sum[i * PS_NV + 1] / n, st_sum[i * PS_NV + 2] / n / 1e3, mk(3), mk(4), mk(5), mk(6), mk(7));
        }
        for (int T = 1; T <= MAXR; T++) {
            if (best_ns[T].empty()) continue;
            std::vector<double> tot(PH_COUNT, 0.0), fil(PH_COUNT, 0.0);
            std::vector<int> cnt(PH_COUNT, 0);
            double all = 0;
            for (size_t k = 0; k < best_ns[T].size(); k++) {
                if (best_tag[T][k] < 0) continue;
                tot[best_tag[T][k]] += best_ns[T][k];
                fil[best_tag[T][k]] += best_fill[T][k];
                cnt[best_tag[T][k]]++;
                all += best_ns[T][k];
            }
            fprintf(stderr, "[mk-profile] T=%d min-composed launch %.3f ms (interference-free estimate)\n", T, all / 1e6);
            for (int i = 0; i < PH_COUNT; i++)
                if (cnt[i])
                    fprintf(stderr, "  %-16s %8.3f ms  %8.2f us/call (fill %5.2f us)  x%d\n", names[i], tot[i] / 1e6,
                            tot[i] / cnt[i] / 1e3, fil[i] / cnt[i] / 1e3, cnt[i]);
        }
    }

    void bench() {
        use();
        const char * names[3] = {"ffn_gu", "ffn_down", "dn_proj"};
        if (profiling) {
            // i-cache test: 4 back-to-back iterations of one phase inside one launch
            for (int which = 0; which < 3; which++) {
                std::vector<double> best(5, 1e30);
                for (int rep = 0; rep < 60; rep++) {
                    Launch P{MODE_BENCH, 4, which, which == 2 ? 0 : 1, 0};
                    CK(cudaMemsetAsync(hb.prof, 0, 4096 * 8));
                    coop(P);
                    std::vector<unsigned long long> pr(8);
                    CK(cudaMemcpy(pr.data(), hb.prof, 8 * 8, cudaMemcpyDeviceToHost));
                    for (int k = 1; k <= 4; k++) best[k] = std::min(best[k], (double) ((pr[k] >> 8) - (pr[k - 1] >> 8)));
                }
                fprintf(stderr, "[mk-bench] %-8s T=4 back-to-back iterations: %.1f / %.1f / %.1f / %.1f us\n", names[which],
                        best[1] / 1e3, best[2] / 1e3, best[3] / 1e3, best[4] / 1e3);
            }
            return;
        }
        for (int which = 0; which < 3; which++)
            for (int T : {1, 4}) {
                float best = 1e30f;
                cudaEvent_t a, b;
                cudaEventCreate(&a);
                cudaEventCreate(&b);
                for (int rep = 0; rep < 400; rep++) {
                    cudaEventRecord(a);
                    Launch P{MODE_BENCH, T, which, which == 2 ? 0 : (rep % 16), 0};
                    coop(P);
                    cudaEventRecord(b);
                    CK(cudaEventSynchronize(b));
                    float ms;
                    cudaEventElapsedTime(&ms, a, b);
                    best = std::min(best, ms);
                }
                fprintf(stderr, "[mk-bench] %-8s T=%d: %.1f us/launch (1 layer + barrier + launch overhead)\n", names[which], T, best * 1000);
            }
    }

    // ---------------- batched prefill (prefill.cuh) ----------------
    PBufs hp{};
    bool pf_ready = false;
    cudaEvent_t ev_done[2] = {nullptr, nullptr}; // TP > 1: this card's partial GEMM output of a reduction is ready
    int ar_n = 0;                                 // prefill reductions so far (selects the part / event buffer)

    void pf_init() {
        if (pf_ready) return;
        // MK_PF_CAP: tokens per prefill chunk (multiple of 128); scratch is ~0.33 MB per token
        int cap = 2048;
        if (const char * e = getenv("MK_PF_CAP")) cap = std::max(128, std::min(2048, atoi(e) / 128 * 128));
        hp.cap = cap;
        hp.x = dalloc<float>((size_t) cap * D);
        hp.xm = dalloc<float>((size_t) cap * D);
        hp.rs = dalloc<float>(cap);
        hp.xh = dalloc<__half>((size_t) cap * FF_L);
        hp.xh2 = dalloc<__half>((size_t) cap * FF_L);
        hp.y0 = dalloc<float>((size_t) cap * FF_L);
        hp.y1 = dalloc<float>((size_t) cap * FF_L);
        hp.y2 = dalloc<float>((size_t) cap * VD_L);
        hp.dq = dalloc<__half>((size_t) cap * QKV_L);
        hp.gb = dalloc<float>((size_t) cap * 2 * NVH_L);
        hp.ab = dalloc<float>((size_t) 8 * cap * 2 * NVH_L);
        hp.h = dalloc<float>((size_t) (cap + 1) * D);
        hp.qh = dalloc<__half>((size_t) cap * NH_L * HD);
        hp.tok = dalloc<int>(cap);
        hp.amax = dalloc<int>(cap);
        hp.amaxv = dalloc<float>(cap);
#if MK_TP > 1
        for (int b = 0; b < 2; b++) {
            hp.part[b] = dalloc<float>((size_t) cap * D);
            CK(cudaEventCreateWithFlags(&ev_done[b], cudaEventDisableTiming));
        }
#endif
        CK(cudaMemcpyToSymbol(cP, &hp, sizeof(PBufs)));
        CK(cudaFuncSetAttribute(pf::mmh<pf::TQ4K>, cudaFuncAttributeMaxDynamicSharedMemorySize, pf::hm_smem<pf::TQ4K>()));
        CK(cudaFuncSetAttribute(pf::mmh<pf::TQ5K>, cudaFuncAttributeMaxDynamicSharedMemorySize, pf::hm_smem<pf::TQ5K>()));
        CK(cudaFuncSetAttribute(pf::mmh<pf::TQ6K>, cudaFuncAttributeMaxDynamicSharedMemorySize, pf::hm_smem<pf::TQ6K>()));
        CK(cudaFuncSetAttribute(pf::mmh<pf::TQ80>, cudaFuncAttributeMaxDynamicSharedMemorySize, pf::hm_smem<pf::TQ80>()));
        CK(cudaFuncSetAttribute(pf::mmh<pf::TQ4K, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, pf::hm_smem<pf::TQ4K>()));
        CK(cudaFuncSetAttribute(pf::mmh<pf::TQ5K, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, pf::hm_smem<pf::TQ5K>()));
        CK(cudaFuncSetAttribute(pf::mmh<pf::TQ6K, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, pf::hm_smem<pf::TQ6K>()));
        CK(cudaFuncSetAttribute(pf::mmh<pf::TQ80, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, pf::hm_smem<pf::TQ80>()));
        CK(cudaFuncSetAttribute(pk_attn<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, AT_SMEM));
        CK(cudaFuncSetAttribute(pk_attn<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, AT_SMEM));
        pf_ready = true;
    }

    // y[t] (+)= W x_t over D outputs for a K-sharded weight: at TP > 1 each card's GEMM writes its partial
    // into part, then both cards sum p0 + p1 (rank order) into y.  part and the events alternate per
    // reduction: the peer reuses a buffer two reductions later, only after it got my next partial,
    // which my stream produces after this all-reduce read the peer's buffer.
    void pf_out(const Mat & m, int T, float * y, int mode, const __half * xin = nullptr) {
#if MK_TP > 1
        const int b = ar_n++ & 1;
        gemm(m, T, hp.part[b], D, pf::OUT_STORE, 0, xin);
        CK(cudaEventRecord(ev_done[b], 0));
        hbar->wait(); // the peer's record is enqueued before this card waits on it
        CK(cudaStreamWaitEvent(0, peer->ev_done[b], 0));
        const int n = T * D;
        pk_allreduce<<<std::min((n + 255) / 256, 8 * nsm), 256>>>(y, hp.part[b], peer->hp.part[b], n, rank, mode == pf::OUT_ADD);
#else
        gemm(m, T, y, D, mode, 0, xin);
#endif
    }

    // y[t][row] (+)= W x_t for the first ntok fp16 rows of xh (token offset t0)
    void gemm(const Mat & m, int ntok, float * y, int ldy, int mode, int t0 = 0, const __half * xin = nullptr) {
        const pf::WMat W{m.w, m.d, m.type, m.K, m.N, m.rb, m.nd};
        const __half * X = (xin ? xin : hp.xh) + (size_t) t0 * m.K;
        const pf::Out O{y, ldy, mode, nullptr};
        const dim3 grid(m.N / pf::BM, (ntok + pf::BN - 1) / pf::BN);
        switch (m.type) {
            case T_Q4K: pf::mmh<pf::TQ4K><<<grid, 256, pf::hm_smem<pf::TQ4K>()>>>(W, X, ntok, O); break;
            case T_Q5K: pf::mmh<pf::TQ5K><<<grid, 256, pf::hm_smem<pf::TQ5K>()>>>(W, X, ntok, O); break;
            case T_Q6K: pf::mmh<pf::TQ6K><<<grid, 256, pf::hm_smem<pf::TQ6K>()>>>(W, X, ntok, O); break;
            case T_Q8_0: pf::mmh<pf::TQ80><<<grid, 256, pf::hm_smem<pf::TQ80>()>>>(W, X, ntok, O); break;
            default: throw std::runtime_error("gemm: unsupported weight type");
        }
    }

    // xh2[t] = silu(Wg x_t) * (Wu x_t) for the first ntok rows of xh (gate / up same type)
    void gemm_glu(const Mat & g, const Mat & u, int ntok) {
        if (g.type != u.type || u.rb != g.rb) throw std::runtime_error("gemm_glu: gate / up type or row stride mismatch");
        const pf::WMat W{g.w, g.d, g.type, g.K, g.N, g.rb, g.nd};
        const pf::Out O{nullptr, g.N, pf::OUT_STORE, hp.xh2};
        const dim3 grid(g.N / (pf::BM / 2), (ntok + pf::BN - 1) / pf::BN);
        switch (g.type) {
            case T_Q4K: pf::mmh<pf::TQ4K, true><<<grid, 256, pf::hm_smem<pf::TQ4K>()>>>(W, hp.xh, ntok, O, u.w, nullptr); break;
            case T_Q5K: pf::mmh<pf::TQ5K, true><<<grid, 256, pf::hm_smem<pf::TQ5K>()>>>(W, hp.xh, ntok, O, u.w, nullptr); break;
            case T_Q6K: pf::mmh<pf::TQ6K, true><<<grid, 256, pf::hm_smem<pf::TQ6K>()>>>(W, hp.xh, ntok, O, u.w, nullptr); break;
            case T_Q8_0: pf::mmh<pf::TQ80, true><<<grid, 256, pf::hm_smem<pf::TQ80>()>>>(W, hp.xh, ntok, O, u.w, u.d); break;
            default: throw std::runtime_error("gemm_glu: unsupported weight type");
        }
    }

    // MK_PF_PROFILE=1: per-category GPU time of the batched prefill (events around each launch)
    struct PfEv { const char * cat; cudaEvent_t a, b; };
    std::vector<PfEv> pf_ev;
    bool pf_prof = getenv("MK_PF_PROFILE") != nullptr;
    template <class F> void pf_time(const char * cat, F f) {
        if (!pf_prof) { f(); return; }
        // MK_PF_PROFILE=2: idempotent kernels are timed alone (5 repeats, min)
        static const char * idem[] = {"attn_prep", "attn_out", "dn_out", "norm_quant", "dn_ab", "attn", "embed", "gemm_ffn_gu", "gemm_dn_in", "gemm_attn_qkv"};
        if (getenv("MK_PF_PROFILE")[0] == '2' && std::any_of(std::begin(idem), std::end(idem), [&](const char * c) { return !strcmp(c, cat); })) {
            float best = 1e30f;
            for (int r = 0; r < 5; r++) {
                cudaEvent_t a, b;
                CK(cudaEventCreate(&a));
                CK(cudaEventCreate(&b));
                CK(cudaDeviceSynchronize());
                CK(cudaEventRecord(a));
                f();
                CK(cudaEventRecord(b));
                CK(cudaEventSynchronize(b));
                float ms;
                CK(cudaEventElapsedTime(&ms, a, b));
                best = std::min(best, ms);
                cudaEventDestroy(a);
                cudaEventDestroy(b);
            }
            pf_iso[cat] += best;
            return;
        }
        PfEv e{cat, nullptr, nullptr};
        CK(cudaEventCreate(&e.a));
        CK(cudaEventCreate(&e.b));
        CK(cudaEventRecord(e.a));
        f();
        CK(cudaEventRecord(e.b));
        pf_ev.push_back(e);
    }
    std::map<std::string, double> pf_iso;
    void pf_report() {
        if (!pf_prof || pf_ev.empty()) return;
        CK(cudaDeviceSynchronize());
        // per category: sum, and median-per-launch x launches (robust to time-slicing by other GPU users)
        std::vector<std::string> cats;
        std::vector<std::vector<float>> ts;
        for (auto & e : pf_ev) {
            float ms;
            CK(cudaEventElapsedTime(&ms, e.a, e.b));
            auto it = std::find(cats.begin(), cats.end(), std::string(e.cat));
            if (it == cats.end()) { cats.push_back(e.cat); ts.push_back({}); it = cats.end() - 1; }
            ts[it - cats.begin()].push_back(ms);
            cudaEventDestroy(e.a);
            cudaEventDestroy(e.b);
        }
        pf_ev.clear();
        double all = 0, all_med = 0;
        std::vector<std::tuple<double, double, std::string, size_t>> rows;
        for (size_t i = 0; i < cats.size(); i++) {
            auto v = ts[i];
            double sum = 0;
            for (float x : v) sum += x;
            std::sort(v.begin(), v.end());
            const double med = v[v.size() / 2] * v.size();
            rows.emplace_back(med, sum, cats[i], v.size());
            all += sum;
            all_med += med;
        }
        for (auto & kv : pf_iso) fprintf(stderr, "  [isolated] %-12s %8.1f ms\n", kv.first.c_str(), kv.second);
        pf_iso.clear();
        std::sort(rows.rbegin(), rows.rend());
        fprintf(stderr, "[mk-prefill-profile] %.1f ms in kernels (median-based %.1f ms)\n", all, all_med);
        for (auto & r : rows)
            fprintf(stderr, "  %-14s median-based %8.1f ms %5.1f%%   sum %8.1f ms   x%zu\n", std::get<2>(r).c_str(), std::get<0>(r),
                    100 * std::get<0>(r) / all_med, std::get<1>(r), std::get<3>(r));
    }

    // Runs positions n0 .. n0 + toks.size() - 1 through the trunk (and the MTP block's K/V).
    // score: instead of MTP, greedy argmax of every position -> pred, and / or the logits of the
    // positions >= first -> sink.
    void pf_run(const std::vector<int> & toks, int n0, std::vector<int> * pred, const MKEngine::LogitSink * sink = nullptr,
                int first = 0) {
#if MK_TP > 1
        ar_n = peer->ar_n = 0;
        std::exception_ptr err;
        std::thread t([&] {
            try {
                peer->pf_rank(toks, n0, nullptr, nullptr, first, pred != nullptr, sink != nullptr);
            } catch (...) {
                err = std::current_exception();
                hbar->abort();
            }
        });
        try {
            pf_rank(toks, n0, pred, sink, first, pred != nullptr, sink != nullptr);
        } catch (...) {
            hbar->abort();
            t.join();
            throw;
        }
        t.join();
        if (err) std::rethrow_exception(err);
        use();
#else
        pf_rank(toks, n0, pred, sink, first, pred != nullptr, sink != nullptr);
#endif
    }

    // One card's prefill (its own thread at TP > 1; only rank 0 gets pred / sink, want_* tell rank 1).
    void pf_rank(const std::vector<int> & toks, int n0, std::vector<int> * pred, const MKEngine::LogitSink * sink, int first,
                 bool want_pred, bool want_sink) {
        use();
        pf_init();
        const int N = (int) toks.size(), cap = hp.cap;
        if (n0 == 0) CK(cudaMemset(hp.h, 0, D * sizeof(float)));
        else CK(cudaMemcpy(hp.h, sb().h_hist + (size_t) ((n0 - 1) % H_RING) * D, D * sizeof(float), cudaMemcpyDeviceToDevice));
        for (int c0 = 0; c0 < N; c0 += cap) {
            const int T = std::min(cap, N - c0), n = n0 + c0;
            CK(cudaMemcpy(hp.tok, toks.data() + c0, T * sizeof(int), cudaMemcpyHostToDevice));
            pf_time("embed", [&] { pk_embed<<<T, 256>>>(T); });
            int dl = 0;
            for (int L = 0; L < NL; L++) {
                const LayerW & W = hw.L[L];
                pf_time("norm_quant", [&] { pk_norm_quant<<<T, 256>>>(hp.x, W.attn_norm); });
                if (W.attn) {
                    pf_time("gemm_attn_qkv", [&] { gemm(W.q, T, hp.y0, QG_L, pf::OUT_STORE); });
                    pf_time("gemm_attn_qkv", [&] { gemm(W.k, T, hp.y1, 2 * KVD_L, pf::OUT_STORE); gemm(W.v, T, hp.y1 + KVD_L, 2 * KVD_L, pf::OUT_STORE); });
                    pf_time("attn_prep", [&] { pk_attn_prep<<<T, 256>>>(n, L / 4, W.qn, W.kn, 1); });
                    pf_time("attn", [&] {
                        if (hb.kvq8) pk_attn<true><<<dim3((T + AT_Q - 1) / AT_Q, NKV_L), 192, AT_SMEM>>>(T, n, L / 4);
                        else pk_attn<false><<<dim3((T + AT_Q - 1) / AT_Q, NKV_L), 192, AT_SMEM>>>(T, n, L / 4);
                    });
                    pf_time("attn_out", [&] { pk_attn_out<<<T, 256>>>(); });
                    pf_time("gemm_attn_o", [&] { pf_out(W.o, T, hp.x, pf::OUT_ADD); });
                } else {
                    pf_time("gemm_dn_in", [&] { gemm(W.qkv, T, hp.y0, QKV_L, pf::OUT_STORE); gemm(W.gate, T, hp.y1, VD_L, pf::OUT_STORE); });
                    pf_time("dn_ab", [&] { pk_ab<<<dim3((T + 63) / 64, AB_SPLITS), 256>>>(W.ab, T); });
                    pf_time("dn_prep", [&] { pk_dn_prep<<<T, 256>>>(T, n, dl, W.conv, W.dt, W.A); pk_conv_ring<<<std::min(T, CONV_RING), 256>>>(T, n, dl); });
                    pf_time("dn_scan", [&] { pk_dn_scan<<<dim3(NVH_L, 2), 256>>>(T, dl); });
                    pf_time("dn_out", [&] { pk_dn_out<<<T, 256>>>(W.snorm); });
                    pf_time("gemm_dn_out", [&] { pf_out(W.sout, T, hp.x, pf::OUT_ADD); });
                    dl++;
                }
                pf_time("norm_quant", [&] { pk_norm_quant<<<T, 256>>>(hp.x, W.post_norm); });
                pf_time("gemm_ffn_gu", [&] { gemm_glu(W.fg, W.fu, T); });
                pf_time("gemm_ffn_down", [&] { pf_out(W.fd, T, hp.x, pf::OUT_ADD, hp.xh2); });
            }
            if (want_pred || want_sink) {
                // each card scores its vocab half; rank 0 joins the halves (logits) and merges the argmax
                pk_norm_quant<<<T, 256>>>(hp.x, hw.onorm);
                std::vector<float> rows;
                const int tile = std::min(128, cap * FF_L / VOCAB_L); // logits rows that fit in y0
                for (int s = 0; s < T; s += tile) {
                    const int nt = std::min(tile, T - s);
                    const int lo = std::max(s, first - n); // rows from lo on are at positions >= first
                    if (!want_pred && lo >= s + nt) continue;
                    gemm(hw.head, nt, hp.y0, VOCAB_L, pf::OUT_STORE, s);
                    if (want_pred) pk_argmax<<<nt, 256>>>(s);
                    if (want_sink && lo < s + nt) {
                        if (MK_TP > 1) {
                            CK(cudaDeviceSynchronize());
                            hbar->wait(); // both halves written
                        }
                        if (rank == 0) {
                            rows.resize((size_t) (s + nt - lo) * VOCAB);
                            for (int r = 0; r < MK_TP; r++) {
                                const Impl * e = r ? peer : this;
                                CK(cudaMemcpy2D(rows.data() + (size_t) r * VOCAB_L, (size_t) VOCAB * 4, e->hp.y0 + (size_t) (lo - s) * VOCAB_L,
                                                (size_t) VOCAB_L * 4, (size_t) VOCAB_L * 4, s + nt - lo, cudaMemcpyDeviceToHost));
                            }
                            (*sink)(n + lo, s + nt - lo, rows.data());
                        }
                        if (MK_TP > 1) hbar->wait(); // rank 1 overwrites y0 only after rank 0 copied it
                    }
                }
                if (want_pred) {
                    if (MK_TP > 1) {
                        CK(cudaDeviceSynchronize());
                        hbar->wait();
                    }
                    if (rank == 0) {
                        std::vector<int> am(T), ai(T);
                        std::vector<float> av(T), bv(T);
                        CK(cudaMemcpy(am.data(), hp.amax, T * sizeof(int), cudaMemcpyDeviceToHost));
                        CK(cudaMemcpy(av.data(), hp.amaxv, T * sizeof(float), cudaMemcpyDeviceToHost));
                        for (int r = 1; r < MK_TP; r++) { // higher logit wins, a tie keeps the smaller (rank 0) id
                            CK(cudaMemcpy(ai.data(), peer->hp.amax, T * sizeof(int), cudaMemcpyDeviceToHost));
                            CK(cudaMemcpy(bv.data(), peer->hp.amaxv, T * sizeof(float), cudaMemcpyDeviceToHost));
                            for (int t = 0; t < T; t++)
                                if (bv[t] > av[t]) { av[t] = bv[t]; am[t] = ai[t] + r * VOCAB_L; }
                        }
                        pred->insert(pred->end(), am.begin(), am.end());
                    }
                    if (MK_TP > 1) hbar->wait(); // rank 1 overwrites amax only after rank 0 read it
                }
                continue;
            }
            pf_time("mtp", [&] {
                pk_final_h<<<T, 256>>>(T, n);
                pk_mtp_in<<<T, 256>>>();
                pf_out(hw.eh, T, hp.xm, pf::OUT_STORE);
                const LayerW & M = hw.L[NL];
                pk_norm_quant<<<T, 256>>>(hp.xm, M.attn_norm);
                gemm(M.k, T, hp.y1, 2 * KVD_L, pf::OUT_STORE);
                gemm(M.v, T, hp.y1 + KVD_L, 2 * KVD_L, pf::OUT_STORE);
                pk_attn_prep<<<T, 256>>>(n, 16, M.qn, M.kn, 0);
            });
            CK(cudaMemcpyAsync(hp.h, hp.h + (size_t) T * D, D * sizeof(float), cudaMemcpyDeviceToDevice));
        }
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        if (rank == 0) pf_report();
    }
};

MKEngine::MKEngine(const std::string & path, int ctx, int n_draft, const std::string & draft_vocab, int kv_mode, int n_seq,
                   int kv_mem)
    : p_(nullptr), n_draft_(n_draft) {
    if (n_draft < 0 || n_draft > MAXT - 1) throw std::runtime_error("n_draft must be in [0, " + std::to_string(MAXT - 1) + "]");
    p_ = new Impl(path, ctx, n_draft, draft_vocab, kv_mode, n_seq, kv_mem);
    for (int l = 0; l < n_seq; l++) p_->lane_reset(l);
}

int MKEngine::n_seq() const { return p_->nseq; }
int MKEngine::selected() const { return p_->cur; }
void MKEngine::select(int id) {
    if (!p_->seqs.count(id)) throw std::runtime_error("select: no open sequence " + std::to_string(id));
    p_->cur = id;
    p_->focus();
}

MKEngine::~MKEngine() { delete p_; }

void MKEngine::bench_phases() {
    p_->need();
    p_->bench();
}

int MKEngine::seq_open(const std::vector<int> & prompt, int & reused) {
    const int id = p_->open_seq(prompt, reused, true);
    p_->pool_check();
    return id;
}

void MKEngine::seq_close(int id, bool keep) {
    const auto it = p_->seqs.find(id);
    if (it == p_->seqs.end()) throw std::runtime_error("seq_close: no open sequence " + std::to_string(id));
    Impl::Seq & q = it->second;
    // the lane's live state covers the committed positions only; the KV rows below them are final
    const int pos = std::min(q.hctl.dn_c, (int) q.seq.size());
    Ctl m = q.hctl;
    m.n = m.dn_c = pos;
    m.mtp_valid = std::min(m.mtp_valid, pos);
    p_->kc->close(q.lane, q.seq.data(), pos, q.blk, keep, m);
    p_->seqs.erase(it);
    if (p_->cur == id) p_->cur = -1;
    // the draft head's added rows are shared by all sequences: rebuild them from the open ones
    p_->dv_rebuild();
    p_->pool_check();
}

void MKEngine::seq_free(int id) { seq_close(id, false); }

int MKEngine::seq_blocks(int id) const {
    const auto it = p_->seqs.find(id);
    if (it == p_->seqs.end()) throw std::runtime_error("seq_blocks: no open sequence " + std::to_string(id));
    return (int) it->second.blk.size();
}

void MKEngine::reset() {
    Impl::Seq & q = p_->need();
    p_->lane_reset(q.lane);
    q.hctl = Ctl{};
    q.seq.clear();
    p_->kc->trim(q.blk, 0);
    p_->dv_rebuild();
    if (p_->nseq == 1) stats = MKStats();
    p_->pool_check();
}

void MKEngine::pool_stats(int & free_blocks, int & total_blocks, int & block_tokens, int & cached_blocks, int & evictable_blocks) const {
    const KvCache<Ctl> & k = *p_->kc;
    free_blocks = k.free_blocks();
    total_blocks = k.total();
    block_tokens = p_->kv_np();
    cached_blocks = k.cached_blocks();
    evictable_blocks = k.evictable_blocks();
}

int MKEngine::kv_ahead() { return KV_AHEAD; }

int MKEngine::position() const { return p_->cur < 0 ? 0 : p_->S().hctl.n; }

int MKEngine::prefill(const std::vector<int> & tokens, bool snapshot_end, const std::vector<int> & snap_pos) {
    if (tokens.empty()) throw std::runtime_error("empty prompt");
    Impl::Seq & q = p_->need();
    Ctl & c = q.hctl;
    p_->get_ctl(c);
    std::vector<int> & seq = q.seq;
    const int len = (int) (seq.size() + tokens.size());
    if (len + KV_AHEAD > p_->ctx) throw std::runtime_error("prompt exceeds context"); // room for one decode cycle
    // every chunk's blocks at once, before any state changes: a short pool leaves the sequence as it was; rows are
    // rewritten from the committed position on, and the last launch's MTP drafts write up to position len + MAXT - 2
    p_->kc->ensure(q.blk, q.lane, std::min(c.dn_c, c.mtp_valid), len + MAXT);
    seq.insert(seq.end(), tokens.begin(), tokens.end());
    p_->dv_add(tokens.data(), tokens.size());
    // Everything from the committed delta-net position on is fed: on a continued sequence that is
    // the last emitted token (not yet processed) and any accepted drafts not yet folded into the
    // recurrent state, followed by the new tokens.  Their KV / MTP rows are simply rewritten.
    const int n0 = c.dn_c;
    const std::vector<int> feed(seq.begin() + n0, seq.end());
    c.mtp_valid = std::min(c.mtp_valid, n0);
    size_t i = 0;
    bool snapped = false;
    const char * mode = getenv("MK_PREFILL");
    const bool batched = !(mode && std::string(mode) == "mk");
    if (batched && feed.size() >= 2) {
        // all but the last token in batched kernels, split at the hinted positions, whose states are saved for
        // other requests; the last token goes through the megakernel, which produces the first output token
        // and the MTP drafts
        std::vector<int> cuts;
        for (int h : snap_pos)
            if (h >= std::max(n0, 1) && h < len - 1) cuts.push_back(h);
        std::sort(cuts.begin(), cuts.end());
        cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
        int pos = n0;
        for (int h : cuts) {
            if (h > pos) {
                p_->pf_run(std::vector<int>(seq.begin() + pos, seq.begin() + h), pos, nullptr);
                pos = c.dn_c = c.mtp_valid = h;
            }
            p_->snap_take(q);
        }
        p_->pf_run(std::vector<int>(seq.begin() + pos, seq.end() - 1), pos, nullptr);
        i = feed.size() - 1;
        c.dn_c = len - 1;
        c.mtp_valid = c.dn_c;
        // Snapshot before the last prompt token: the next request's re-rendered history re-tokenizes
        // the generation prompt's tail, so a snapshot after it would never lie inside the common prefix.
        c.n = c.dn_c;
        p_->put_ctl(c);
        if (snapshot_end) p_->snap_take(q);
        snapped = true;
    }
    while (i < feed.size()) {
        const int T = (int) std::min<size_t>(MAXT, feed.size() - i);
        const bool last = i + T == feed.size();
        if (last && snapshot_end && !snapped) p_->snap_take(q); // before the last token(s), as the batched path
        c.n = c.dn_c;
        for (int k = 0; k < T; k++) c.tok[k] = feed[i + k];
        p_->put_ctl(c);
        q.ndraft = p_->n_draft_max;
        p_->launch(MODE_PREFILL, T, last ? 1 : 0);
        p_->get_ctl(c);
        i += T;
    }
    q.nd_avail = q.ndraft;
    q.all_acc = true;
    CK(cudaGetLastError());
    seq.push_back(c.out[0]);
    p_->pool_check();
    return c.out[0];
}

void MKEngine::set_sampling(float temp, float top_p, int top_k, float min_p, unsigned long long seed) {
    if (temp > 0.f && (top_k <= 0 || top_k > TOPK_MAX))
        throw std::runtime_error("sampling needs 1 <= top_k <= " + std::to_string(TOPK_MAX));
    Impl::Seq & q = p_->cur < 0 ? p_->dflt : p_->S();
    q.smp_temp = temp;
    q.smp_top_p = top_p;
    q.smp_top_k = top_k;
    q.smp_min_p = min_p;
    q.smp_seed = seed;
}

const std::vector<int> & MKEngine::sequence() const {
    static const std::vector<int> none;
    return p_->cur < 0 ? none : p_->S().seq;
}
int MKEngine::committed() const { return p_->cur < 0 ? 0 : p_->S().hctl.dn_c; }

void MKEngine::truncate(int len) {
    Impl::Seq & q = p_->need();
    if (len < q.hctl.dn_c || len > (int) q.seq.size())
        throw std::runtime_error("truncate: can only drop tokens emitted after the committed position");
    q.seq.resize(len);
    p_->kc->trim(q.blk, len);
    p_->pool_check();
}

std::vector<int> MKEngine::score(const std::vector<int> & tokens) {
    reset();
    Impl::Seq & q = p_->S();
    p_->kc->ensure(q.blk, q.lane, 0, (int) tokens.size()); // scoring launches run no MTP drafts
    std::vector<int> pred;
    const char * mode = getenv("MK_PREFILL");
    if (!(mode && std::string(mode) == "mk")) {
        p_->pf_run(tokens, 0, &pred);
        reset();
        return pred;
    }
    Ctl & c = q.hctl;
    p_->get_ctl(c);
    size_t i = 0;
    while (i < tokens.size()) {
        const int T = (int) std::min<size_t>(MAXT, tokens.size() - i);
        c.n = (int) i;
        for (int k = 0; k < T; k++) c.tok[k] = tokens[i + k];
        p_->put_ctl(c);
        p_->launch(MODE_PREFILL, T, 0, 1);
        p_->get_ctl(c);
        for (int k = 0; k < T; k++) pred.push_back(c.tgt[k]);
        i += T;
    }
    return pred;
}

int MKEngine::n_vocab() { return VOCAB; }

void MKEngine::score_logits(const std::vector<int> & tokens, int mk_from, int first, const LogitSink & sink) {
    const int N = (int) tokens.size();
    if (N + 8 > p_->ctx) throw std::runtime_error("sequence exceeds context");
    reset();
    Impl::Seq & q = p_->S();
    p_->kc->ensure(q.blk, q.lane, 0, N);
    mk_from = std::min(std::max(mk_from, 0), N);
    if (mk_from > 0)
        p_->pf_run(std::vector<int>(tokens.begin(), tokens.begin() + mk_from), 0, nullptr, first < mk_from ? &sink : nullptr, first);
    Ctl & c = q.hctl;
    p_->get_ctl(c);
    c.dn_c = mk_from;
    std::vector<float> rows((size_t) MAXT * VOCAB);
    for (int i = mk_from; i < N;) {
        const int T = std::min(MAXT, N - i);
        c.n = i;
        for (int k = 0; k < T; k++) c.tok[k] = tokens[i + k];
        p_->put_ctl(c);
        p_->launch(MODE_PREFILL, T, 0, 2);
        p_->get_ctl(c);
        const int s = std::max(0, first - i);
        if (s < T) {
            p_->get_logits(rows.data(), s, T - s);
            sink(i + s, T - s, rows.data());
        }
        i += T;
    }
    reset();
}

// MK_HOST_PROF: host time per decode cycle split into KV block bookkeeping, row prep + launch submit, wait for the
// cards + Ctl readback, and the rest; printed at exit
static struct HostProf {
    using C = std::chrono::steady_clock;
    const bool on = getenv("MK_HOST_PROF") != nullptr;
    double ns[4] = {};
    long n = 0;
    C::time_point t;
    void start() {
        if (on) t = C::now();
    }
    void lap(int k) {
        if (!on) return;
        const C::time_point u = C::now();
        ns[k] += std::chrono::duration<double, std::nano>(u - t).count();
        t = u;
        n += k == 3;
    }
    ~HostProf() {
        if (on && n)
            fprintf(stderr, "[mk-host] %ld cycles, us/cycle: kv blocks %.1f, prep+submit %.1f, card wait %.1f, rest %.1f\n", n,
                    ns[0] / n / 1e3, ns[1] / n / 1e3, ns[2] / n / 1e3, ns[3] / n / 1e3);
    }
} host_prof;

int MKEngine::step(std::vector<int> & out) {
    Impl::Seq & q = p_->need();
    Ctl & c = q.hctl;
    if ((int) q.seq.size() <= c.n) throw std::runtime_error("step: the sequence has no pending token (prefill first)");
    if (c.n + KV_AHEAD > p_->ctx) throw std::runtime_error("context full");
    host_prof.start();
    p_->kc->ensure(q.blk, q.lane, std::min(c.dn_c, c.mtp_valid), c.n + KV_AHEAD);
    host_prof.lap(0);
    const int T = 1 + q.nd_avail;
    // Draft depth: after a cycle that rejected a draft, draft one token less (but at least 3), since a
    // deeper draft is then unlikely to be accepted; the full depth again after a fully accepted cycle.
    q.ndraft = (!q.all_acc && n_draft_ > 3) ? n_draft_ - 1 : n_draft_;
    p_->launch(MODE_DECODE, T, 0);
    host_prof.lap(1);
    q.nd_avail = q.ndraft;
    p_->get_ctl(c);
    host_prof.lap(2);
    for (int i = 0; i < c.n_out; i++) out.push_back(c.out[i]);
    q.seq.insert(q.seq.end(), c.out, c.out + c.n_out);
    p_->dv_add(c.out, c.n_out);
    q.all_acc = c.n_acc == T - 1;
    stats.cycles++;
    stats.drafted += T - 1;
    stats.accepted += c.n_acc;
    p_->pool_check();
    host_prof.lap(3);
    return c.n_out;
}

int MKEngine::max_batch_rows() { return MK_TP > 1 ? MAXR : MAXT; }

void MKEngine::step_batch(const std::vector<int> & ids, std::vector<std::vector<int>> & outs) {
    const int nb = (int) ids.size();
    outs.resize(nb);
    if (nb == 0) return;
    for (int b = 0; b < nb; b++) {
        if (!p_->seqs.count(ids[b])) throw std::runtime_error("step_batch: no open sequence " + std::to_string(ids[b]));
        for (int k = 0; k < b; k++)
            if (ids[k] == ids[b]) throw std::runtime_error("step_batch: sequence listed twice");
    }
    host_prof.start();
    {
        // every sequence's blocks before any of them steps: a short pool leaves all of them as they were
        std::vector<KvCache<Ctl>::Req> rs;
        for (int b = 0; b < nb; b++) {
            Impl::Seq & q = p_->seqs.at(ids[b]);
            const Ctl & c = q.hctl;
            if ((int) q.seq.size() <= c.n)
                throw std::runtime_error("step_batch: sequence " + std::to_string(ids[b]) + " has no pending token (prefill first)");
            if (c.n + KV_AHEAD > p_->ctx) throw std::runtime_error("context full in sequence " + std::to_string(ids[b]));
            rs.push_back({&q.blk, q.lane, std::min(c.dn_c, c.mtp_valid), c.n + KV_AHEAD});
        }
        p_->kc->ensure_all(rs);
    }
    host_prof.lap(0);
#if MK_TP > 1
    const bool batched = nb > 1 && p_->nsm >= 54; // resid32_ks parks at most 3 items per block
#else
    const bool batched = false;
#endif
    if (!batched) {
        const int cur = p_->cur;
        for (int b = 0; b < nb; b++) {
            select(ids[b]);
            step(outs[b]);
        }
        p_->cur = cur;
        if (cur >= 0) p_->focus();
        p_->pool_check();
        return;
    }
#if MK_TP > 1
    // rows per slot (packed): each verifies the drafts it made last cycle, and the next cycle's drafts
    // share out the row cap's spare rows
    BLaunch P{};
    static const int row_cap = [] { // MK_BATCH_ROWS=n: fixed cap for A/B runs (else by slot count)
        const char * e = getenv("MK_BATCH_ROWS");
        return e ? std::max(MAXB, atoi(e)) : 0;
    }();
    // a wider cycle costs more than its extra drafts return at few slots (measured: 16 rows win at 4 slots, 32 at 5+)
    const int by_slots = nb <= 4 ? MR : MAXR;
    P.maxr = std::min(p_->batch_rows(), row_cap ? row_cap : by_slots);
    P.nb = nb;
    P.Tb = 1;
    std::vector<int> T(nb), nd(nb, 0), want(nb);
    int tot = 0;
    for (int b = 0; b < nb; b++) {
        const Impl::Seq & q = p_->seqs.at(ids[b]);
        T[b] = std::min(1 + q.nd_avail, MAXT);
        tot += T[b];
        want[b] = std::min((!q.all_acc && n_draft_ > 3) ? n_draft_ - 1 : n_draft_, MAXT - 1);
    }
    while (tot > P.maxr) { // slots joined since the drafts were made: drop drafts of the longest
        int m = 0;
        for (int b = 1; b < nb; b++)
            if (T[b] > T[m]) m = b;
        T[m]--;
        tot--;
    }
    for (int spare = P.maxr - nb, more = 1; more && spare > 0;) {
        more = 0;
        for (int i = 0; i < nb && spare > 0; i++) {
            const int b = (p_->row_rot + i) % nb;
            if (nd[b] < want[b]) nd[b]++, spare--, more = 1;
        }
    }
    p_->row_rot++;
    P.nr = 0;
    for (int b = 0; b < nb; b++) {
        Impl::Seq & q = p_->seqs.at(ids[b]);
        P.Tb = std::max(P.Tb, T[b]);
        P.r0[b] = P.nr;
        P.nr += T[b];
        q.ndraft = nd[b];
        P.slot[b] = q.lane;
        P.mtp_on[b] = n_draft_ > 0;
        P.sp[b] = Launch{MODE_DECODE, T[b], 0, q.ndraft, 0, q.smp_temp, q.smp_top_p, q.smp_min_p, q.smp_top_k, q.smp_seed,
                         p_->draft_prob};
    }
    p_->launch_batch(P);
    host_prof.lap(1);
    for (int b = 0; b < nb; b++) {
        Impl::Seq & q = p_->seqs.at(ids[b]);
        Ctl & c = q.hctl;
        p_->get_ctl_slot(q.lane, c);
        if (b == 0) host_prof.lap(2);
        q.nd_avail = q.ndraft;
        outs[b].insert(outs[b].end(), c.out, c.out + c.n_out);
        q.seq.insert(q.seq.end(), c.out, c.out + c.n_out);
        p_->dv_add(c.out, c.n_out);
        q.all_acc = c.n_acc == T[b] - 1;
        stats.drafted += T[b] - 1;
        stats.accepted += c.n_acc;
    }
    stats.cycles++;
    CK(cudaGetLastError());
    p_->pool_check();
    host_prof.lap(3);
#endif
}

std::vector<float> MKEngine::last_hidden() {
    Impl::Seq & q = p_->need();
    std::vector<float> h(D);
    const int pos = q.hctl.dn_c - 1;
    CK(cudaMemcpy(h.data(), p_->sb().h_hist + (size_t) (pos % H_RING) * D, D * 4, cudaMemcpyDeviceToHost));
    return h;
}
