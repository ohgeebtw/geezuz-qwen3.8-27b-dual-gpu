// Qwen3.8-27B (llama.cpp arch "qwen35", shared with Qwen3.5/3.6) single-launch-per-cycle CUDA megakernel decoder.
#pragma once

#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

struct MKStats {
    long cycles = 0;
    long drafted = 0;
    long accepted = 0;
};

// Thrown when the shared KV pool has too few free blocks for a call; the call changed nothing.
struct MKPoolFull : std::runtime_error {
    using std::runtime_error::runtime_error;
};

class MKEngine {
  public:
    // n_draft: MTP draft tokens per cycle (0..4). ctx: positions per sequence (the per-request maximum).
    // draft_vocab: optional file of int32 token ids; the MTP draft head is restricted to them.
    // kv_mode: KV cache format, 0 fp16, 1 q8_0 (half the memory, ~1-2% slower), -1 auto (q8_0 only
    // when the fp16 pool would hold less than one ctx sequence).
    // n_seq: lanes (1..8), the sequences open at once, each lane with its own recurrent state; their KV comes from
    // one shared pool of blocks, taken as a sequence grows (MKPoolFull when none is left).  The pool also holds the
    // prefix cache: KV blocks and DeltaNet state snapshots of closed sequences, reused by later ones and evicted
    // (least recently used first) when a sequence needs the blocks.
    // kv_mem: KV pool MB per card; 0 = the free memory left after the lanes and a reserve.
    MKEngine(const std::string & gguf_path, int ctx, int n_draft, const std::string & draft_vocab = "", int kv_mode = -1,
             int n_seq = 1, int kv_mem = 0);
    ~MKEngine();

    // Opens a sequence on a free lane (throws when every lane is busy) and selects it.  It starts from the longest
    // cached prefix of `prompt` with a stored state: `reused` = its length (0: empty sequence); feed the rest with
    // prefill(prompt[reused:]).
    int seq_open(const std::vector<int> & prompt, int & reused);
    // Closes sequence id and frees its lane.  keep: its tokens, KV and state stay in the prefix cache for later
    // sequences (evicted when the pool needs the blocks); otherwise only blocks other holders share survive.
    void seq_close(int id, bool keep);
    void seq_free(int id); // seq_close(id, false)
    // KV pool blocks sequence id holds.
    int seq_blocks(int id) const;

    // Every call below acts on the selected sequence (-1: none; single-sequence callers need not open one, the
    // first call opens an empty one).
    void select(int id);
    int selected() const;
    int n_seq() const;

    // Clears the selected sequence (KV, recurrent state, positions) and returns its KV blocks to the pool.
    void reset();
    // KV pool: free and total blocks, tokens per block, blocks only the prefix cache holds and those of them it
    // can evict (free for a sequence that needs them).
    void pool_stats(int & free_blocks, int & total_blocks, int & block_tokens, int & cached_blocks, int & evictable_blocks) const;
    // A sequence at position n needs KV blocks for positions < n + kv_ahead() before its next cycle.
    static int kv_ahead();
    // Processes the prompt, or appends `tokens` to the sequence so far (prompt + emitted tokens,
    // e.g. the next user turn); returns the first generated token (greedy).  snapshot_end: store the state just
    // before the prompt's last token in the prefix cache (false for tokens forced into a running generation);
    // snap_pos: sequence positions inside the prompt (e.g. message boundaries) whose states are stored too.
    int prefill(const std::vector<int> & tokens, bool snapshot_end = true, const std::vector<int> & snap_pos = {});
    // Sampling: temp 0 = greedy (default); otherwise llama.cpp's order top_k (1..32), top_p, min_p,
    // temperature.  Speculative verification keeps the sampled distribution exact.
    void set_sampling(float temp, float top_p, int top_k, float min_p, unsigned long long seed);
    // Drops emitted tokens beyond `len` sequence positions (e.g. those generated after an
    // end-of-turn token within the last cycle) before the next prefill; frees the KV blocks beyond them.
    void truncate(int len);
    // The sequence so far (prompts and emitted tokens; position = index) and the number of its
    // positions folded into the recurrent state (truncate() can only go back to there).
    const std::vector<int> & sequence() const;
    int committed() const;
    // Runs one verify+draft cycle; appends 1..n_draft+1 accepted tokens to `out`.
    int step(std::vector<int> & out);
    // One cycle for several distinct open sequences at once; outs[i] receives sequence ids[i]'s tokens (appended).
    // TP=2 builds run them in one launch whose GEMVs share every weight read across the sequences' rows (at most
    // max_batch_rows() rows, 10 unless every matrix is Q8_0: fewer drafts per sequence as sequences are added);
    // otherwise they step one by one.
    void step_batch(const std::vector<int> & ids, std::vector<std::vector<int>> & outs);
    static int max_batch_rows();
    // Debug helper: copy of the last main-model logits row selection (argmax ids) is not kept;
    // instead expose the final normalised hidden row of the last processed position.
    std::vector<float> last_hidden();

    int position() const;
    // Progress of the latest prefill(): (number of prefill() calls so far, mod 2^31) << 32 | the sequence position it
    // has processed up to, advancing per 2,048-token chunk of the batched kernels.  Lock-free: safe to call from
    // another thread while that prefill() runs.
    long long prefill_progress() const;
    // Teacher-forced greedy predictions: pred[i] = argmax of the next-token distribution after
    // tokens[0..i] (resets the sequence).
    std::vector<int> score(const std::vector<int> & tokens);
    // Teacher-forced next-token logits (resets the sequence): sink(i0, n, logits) receives the logits
    // ([n][n_vocab()], valid during the call) after tokens[0..i] for i = i0 .. i0 + n - 1, in position
    // order, for all positions >= first.  Positions below mk_from are processed by the batched
    // prefill kernels, the others by the decode megakernel.
    using LogitSink = std::function<void(int i0, int n, const float * logits)>;
    void score_logits(const std::vector<int> & tokens, int mk_from, int first, const LogitSink & sink);
    static int n_vocab();
    void bench_phases();
    int n_draft() const { return n_draft_; }
    MKStats stats;

  private:
    struct Impl;
    Impl * p_;
    int n_draft_;
};
