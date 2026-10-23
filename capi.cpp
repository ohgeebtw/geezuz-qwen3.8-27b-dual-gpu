// C API over MKEngine plus libllama's tokenizer (vocab only), for the HTTP server (server/mk_server.py).
#include <algorithm>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

#include "llama.h"
#include "qwen38_mk.h"

namespace {
struct Handle {
    MKEngine * eng = nullptr;
    llama_model * lm = nullptr;
    const llama_vocab * vocab = nullptr;
    std::string err;
};
thread_local std::string g_err;
// error codes of the calls that can grow a sequence: -1 any error, MK_POOL_FULL too few free KV blocks
// (the call changed nothing; retry once blocks are freed)
constexpr int MK_POOL_FULL = -2;

// a call on sequence id: selects it first; returns f()'s result, -1 on an error, MK_POOL_FULL when the pool is short
template <class F> int on_seq(void * hp, int id, F f) {
    try {
        MKEngine & e = *static_cast<Handle *>(hp)->eng;
        e.select(id);
        return f(e);
    } catch (const MKPoolFull & e) {
        g_err = e.what();
        return MK_POOL_FULL;
    } catch (const std::exception & e) {
        g_err = e.what();
        return -1;
    }
}
} // namespace

extern "C" {

const char * mk_last_error() { return g_err.c_str(); }

// kv_mem: KV pool MB per card, 0 = the free memory after the lanes and a reserve
void * mk_new(const char * model, int ctx, int n_draft, const char * draft_vocab, int kv_mode, int n_seq, int kv_mem) {
    try {
        llama_backend_init();
        llama_model_params mp = llama_model_default_params();
        mp.vocab_only = true;
        auto * h = new Handle;
        h->lm = llama_model_load_from_file(model, mp);
        if (!h->lm) throw std::runtime_error("failed to load the vocabulary");
        h->vocab = llama_model_get_vocab(h->lm);
        h->eng = new MKEngine(model, ctx, n_draft, draft_vocab ? draft_vocab : "", kv_mode, n_seq, kv_mem);
        return h;
    } catch (const std::exception & e) {
        g_err = e.what();
        return nullptr;
    }
}

void mk_free(void * hp) {
    auto * h = static_cast<Handle *>(hp);
    delete h->eng;
    llama_model_free(h->lm);
    delete h;
}

// chat template stored in the GGUF (tokenizer.chat_template); returns its length, copies up to cap
int mk_chat_template(void * hp, char * buf, int cap) {
    auto * h = static_cast<Handle *>(hp);
    const char * t = llama_model_chat_template(h->lm, nullptr);
    if (!t) return -1;
    const int n = (int) strlen(t);
    if (buf && cap > 0) memcpy(buf, t, std::min(n, cap));
    return n;
}

// text -> token ids (special tokens such as <|im_start|> parsed); returns the count (negative: -needed)
int mk_tokenize(void * hp, const char * text, int len, int * out, int cap) {
    auto * h = static_cast<Handle *>(hp);
    return llama_tokenize(h->vocab, text, len, out, cap, false, true);
}

// token -> its bytes (special tokens rendered as text); returns the byte count
int mk_piece(void * hp, int tok, char * buf, int cap) {
    auto * h = static_cast<Handle *>(hp);
    return llama_token_to_piece(h->vocab, tok, buf, cap, 0, true);
}

int mk_is_eog(void * hp, int tok) { return llama_vocab_is_eog(static_cast<Handle *>(hp)->vocab, tok) ? 1 : 0; }

// opens a sequence starting from the longest cached prefix of toks[0..n) with a stored state; *reused = its length
// (feed toks[*reused..n) with mk_prefill); returns the id, or -1 (e.g. every lane busy)
int mk_seq_open(void * hp, const int * toks, int n, int * reused) {
    try {
        int r = 0;
        const int id = static_cast<Handle *>(hp)->eng->seq_open(std::vector<int>(toks, toks + n), r);
        *reused = r;
        return id;
    } catch (const std::exception & e) {
        g_err = e.what();
        return -1;
    }
}

// closes sequence id and frees its lane; keep: its prefix stays cached for later sequences
int mk_seq_close(void * hp, int id, int keep) {
    try {
        static_cast<Handle *>(hp)->eng->seq_close(id, keep != 0);
        return 0;
    } catch (const std::exception & e) {
        g_err = e.what();
        return -1;
    }
}

int mk_seq_free(void * hp, int id) { return mk_seq_close(hp, id, 0); }

// KV pool blocks sequence id holds (negative: no such sequence)
int mk_seq_blocks(void * hp, int id) {
    try {
        return static_cast<Handle *>(hp)->eng->seq_blocks(id);
    } catch (const std::exception & e) {
        g_err = e.what();
        return -1;
    }
}

int mk_set_sampling(void * hp, int id, float temp, float top_p, int top_k, float min_p, unsigned long long seed) {
    return on_seq(hp, id, [&](MKEngine & e) {
        e.set_sampling(temp, top_p, top_k, min_p, seed);
        return 0;
    });
}

// appends tokens to sequence id; snap: n_snap sequence positions inside them whose states are cached too;
// returns the first generated token (negative on error, MK_POOL_FULL)
int mk_prefill(void * hp, int id, const int * toks, int n, const int * snap, int n_snap) {
    return on_seq(hp, id, [&](MKEngine & e) {
        return e.prefill(std::vector<int>(toks, toks + n), true, std::vector<int>(snap, snap + (snap ? n_snap : 0)));
    });
}

// mk_prefill without the snapshot before the last token: for tokens forced into a running generation and for
// all but the last chunk of a chunked prompt
int mk_append(void * hp, int id, const int * toks, int n, const int * snap, int n_snap) {
    return on_seq(hp, id, [&](MKEngine & e) {
        return e.prefill(std::vector<int>(toks, toks + n), false, std::vector<int>(snap, snap + (snap ? n_snap : 0)));
    });
}

int mk_n_seq(void * hp) { return static_cast<Handle *>(hp)->eng->n_seq(); }

// progress of the latest mk_prefill / mk_append: (calls so far, mod 2^31) << 32 | sequence position processed up to;
// callable from another thread during that call (selects no sequence)
long long mk_prefill_progress(void * hp) { return static_cast<Handle *>(hp)->eng->prefill_progress(); }

// one decode cycle for n distinct sequences; sequence ids[i] writes counts[i] tokens to out[i * cap ..]; 0, -1 or
// MK_POOL_FULL (none stepped)
int mk_step_batch(void * hp, const int * ids, int n, int * out, int cap, int * counts) {
    try {
        std::vector<std::vector<int>> outs(n);
        static_cast<Handle *>(hp)->eng->step_batch(std::vector<int>(ids, ids + n), outs);
        for (int b = 0; b < n; b++) {
            counts[b] = std::min((int) outs[b].size(), cap);
            for (int i = 0; i < counts[b]; i++) out[b * cap + i] = outs[b][i];
        }
        return 0;
    } catch (const MKPoolFull & e) {
        g_err = e.what();
        return MK_POOL_FULL;
    } catch (const std::exception & e) {
        g_err = e.what();
        return -1;
    }
}

// a sequence at position n needs KV blocks for positions < n + mk_kv_ahead() before its next cycle
int mk_kv_ahead() { return MKEngine::kv_ahead(); }

// KV pool: free and total blocks, tokens per block, blocks only the prefix cache holds, those of them it can evict
void mk_pool_stats(void * hp, int * free_blocks, int * total_blocks, int * block_tokens, int * cached_blocks, int * evictable_blocks) {
    static_cast<Handle *>(hp)->eng->pool_stats(*free_blocks, *total_blocks, *block_tokens, *cached_blocks, *evictable_blocks);
}

int mk_truncate(void * hp, int id, int len) {
    return on_seq(hp, id, [&](MKEngine & e) {
        e.truncate(len);
        return 0;
    });
}

// copies sequence id's tokens (up to cap); returns its length (negative: no such sequence)
int mk_sequence(void * hp, int id, int * out, int cap) {
    return on_seq(hp, id, [&](MKEngine & e) {
        const auto & s = e.sequence();
        const int n = (int) s.size();
        if (out)
            for (int i = 0; i < std::min(n, cap); i++) out[i] = s[i];
        return n;
    });
}

void mk_stats(void * hp, long * cycles, long * drafted, long * accepted) {
    const auto & s = static_cast<Handle *>(hp)->eng->stats;
    *cycles = s.cycles;
    *drafted = s.drafted;
    *accepted = s.accepted;
}

} // extern "C"
