// Host bookkeeping of the shared KV pool: refcounted blocks with copy-on-write, the prefix index (a radix tree
// whose nodes are pool blocks) and the DeltaNet state snapshots that make a cached prefix resumable.  No CUDA
// here: copies go through KvDev, which runs them on every card (a block id names the same block on each).
#pragma once

#include <algorithm>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "qwen38_mk.h"

struct KvDev {
    virtual ~KvDev() = default;
    virtual void copy_block(int src, int dst) = 0;
    // DeltaNet state of `lane` at position pos (covering positions < pos) into blocks / from blocks into the lane
    virtual void snap_save(int lane, int pos, const std::vector<int> & blks) = 0;
    virtual void snap_load(int lane, int pos, const std::vector<int> & blks) = 0;
    // block-table entries i0 .. i0 + k - 1 of lane
    virtual void upload_bt(int lane, int i0, const int * ids, int k) = 0;
};

// Meta: what a snapshot keeps besides the device state (the engine's Ctl).
template <class Meta>
class KvCache {
  public:
    struct Open {
        int lane = -1, pos = 0; // pos: reused prompt tokens (0: the lane needs a fresh state)
        Meta meta{};
    };

    KvCache(int nblocks, int np_, int nblk_lane_, int snap_nb_, int lanes, KvDev * dev_)
        : np(np_), nblk_lane(nblk_lane_), snap_nb(snap_nb_), cap(nblocks / 4), dev(dev_), ref(nblocks, 0), act(nblocks, 0),
          parked(lanes, -1), busy(lanes, 0), guard(lanes, -1) {
        free_.reserve(nblocks);
        for (int i = nblocks - 1; i >= 0; i--) free_.push_back(i); // low ids first
        nodes[0].hash = 0xcbf29ce484222325ULL;
    }

    int total() const { return (int) ref.size(); }
    int free_blocks() const { return (int) free_.size(); }
    int cached_blocks() const { return n_cached; }
    int evictable_blocks() const { return n_evict; }
    int snap_blocks() const { return snap_nb; }
    int snap_cap() const { return cap; }

    // ---------------- sequences' blocks ----------------
    // Blocks a sequence must gain before writing positions [from, upto): new ones past its end plus a copy of
    // each shared one in that range.
    int need(const std::vector<int> & blk, int from, int upto) const {
        const int want = (upto + np - 1) / np;
        if (want > nblk_lane) throw std::runtime_error("sequence exceeds context");
        int k = std::max(0, want - (int) blk.size());
        for (int i = std::max(from, 0) / np; i < std::min((int) blk.size(), want); i++) k += ref[blk[i]] > 1;
        return k;
    }
    // Makes positions [from, upto) of a sequence on `lane` writable and uploads the changed table entries;
    // evicts cached prefixes when the pool is short, then throws MKPoolFull (changing nothing) if still short.
    void ensure(std::vector<int> & blk, int lane, int from, int upto) {
        const int k = need(blk, from, upto);
        if (k == 0) return;
        if (!room(k)) throw MKPoolFull("KV pool full");
        const int want = (upto + np - 1) / np;
        int lo = (int) blk.size();
        for (int i = std::max(from, 0) / np; i < std::min((int) blk.size(), want); i++)
            if (ref[blk[i]] > 1) {
                const int b = alloc(true);
                dev->copy_block(blk[i], b);
                change(blk[i], -1, -1);
                blk[i] = b;
                lo = std::min(lo, i);
            }
        while ((int) blk.size() < want) blk.push_back(alloc(true));
        if (lo < (int) blk.size()) dev->upload_bt(lane, lo, blk.data() + lo, (int) blk.size() - lo);
    }
    struct Req {
        std::vector<int> * blk;
        int lane, from, upto;
    };
    // ensure() for several sequences, all or nothing
    void ensure_all(const std::vector<Req> & rs) {
        // a block shared by c listed sequences and ref holders in all needs min(c, ref - 1) copies: the last
        // listed holder of an otherwise unshared block writes it in place
        auto total_need = [&] {
            int k = 0;
            std::unordered_map<int, int> shared;
            for (const Req & r : rs) {
                const int want = (r.upto + np - 1) / np;
                if (want > nblk_lane) throw std::runtime_error("sequence exceeds context");
                k += std::max(0, want - (int) r.blk->size());
                for (int i = std::max(r.from, 0) / np; i < std::min((int) r.blk->size(), want); i++)
                    if (ref[(*r.blk)[i]] > 1) shared[(*r.blk)[i]]++;
            }
            for (const auto & kv : shared) k += std::min(kv.second, ref[kv.first] - 1);
            return k;
        };
        if (!room(total_need()) && total_need() > (int) free_.size()) throw MKPoolFull("KV pool full");
        for (const Req & r : rs) ensure(*r.blk, r.lane, r.from, r.upto);
    }
    // Drops a sequence's blocks that hold only positions >= keep.
    void trim(std::vector<int> & blk, int keep) {
        const size_t n = (size_t) ((std::max(keep, 0) + np - 1) / np);
        while (blk.size() > n) {
            change(blk.back(), -1, -1);
            blk.pop_back();
        }
    }

    // ---------------- lanes ----------------
    // Binds a new sequence with prompt toks[0, n) to a lane.  look: reuse the deepest cached snapshot at or below
    // the prompt's common prefix with the cache (at most n - 1, one token is always fed); blk receives the shared
    // blocks below it and the snapshot is restored into the lane.  Throws when every lane is busy.
    Open open(const int * toks, int n, bool look, std::vector<int> & blk) {
        std::pair<int, int> m{-1, 0};
        if (look && n >= 2) m = lookup(toks, n);
        const int sid = m.first;
        const int lane = (sid >= 0 && snaps.at(sid).lane >= 0) ? snaps.at(sid).lane : free_lane();
        Open o;
        o.lane = lane;
        o.pos = m.second;
        trim(blk, 0);
        if (sid >= 0) {
            Snap & s = snaps.at(sid);
            s.used = ++tick;
            o.meta = s.meta;
            path(s.node, blk);
            for (int b : blk) change(b, 1, 1);
        }
        const bool inplace = sid >= 0 && parked[lane] == sid; // the lane still holds that state
        pin = sid;
        if (parked[lane] >= 0) stash(parked[lane]);
        busy[lane] = 1;
        guard[lane] = sid;
        if (sid >= 0 && !inplace) dev->snap_load(lane, o.pos, snaps.at(sid).blks);
        pin = -1;
        if (!blk.empty()) dev->upload_bt(lane, 0, blk.data(), (int) blk.size());
        return o;
    }
    // Unbinds a sequence (tokens toks, live DeltaNet state at position pos in its lane).  keep: its prefix below
    // pos joins the index and the lane's state stays parked there as a snapshot, copied to blocks only when the
    // lane is reused.  Releases the sequence's blocks either way.
    void close(int lane, const int * toks, int pos, std::vector<int> & blk, bool keep, const Meta & meta) {
        if (keep && pos > 0) {
            const int node = publish(toks, pos, blk, false);
            if (node >= 0) {
                int dup = -1;
                for (int s : nodes.at(node).snaps)
                    if (snaps.at(s).pos == pos) dup = s;
                if (dup >= 0) snaps.at(dup).used = ++tick;
                else {
                    const int id = snap_next++;
                    Snap & s = snaps[id];
                    s.pos = pos;
                    s.node = node;
                    s.lane = lane;
                    s.meta = meta;
                    s.used = ++tick;
                    nodes.at(node).snaps.push_back(id);
                    parked[lane] = id;
                }
            }
        }
        trim(blk, 0);
        busy[lane] = 0;
        guard[lane] = -1;
    }

    // ---------------- snapshots ----------------
    // Saves the state of the sequence on `lane` at position pos (tokens toks, blocks blk) as a snapshot and
    // indexes its prefix; skipped (false) when one exists there already or the pool cannot hold it.
    bool snap_take(const int * toks, int pos, const std::vector<int> & blk, int lane, const Meta & meta) {
        if (pos <= 0 || snap_nb > cap) return false;
        const int have = find(toks, pos);
        if (have >= 0) {
            snaps.at(have).used = ++tick;
            if (busy[lane]) guard[lane] = have;
            return false;
        }
        // the partial block holding pos is copied: the sequence keeps writing its own one
        if (!fit_snap(-1) || !room(snap_nb + (pos % np ? 1 : 0))) return false;
        const int node = publish(toks, pos, blk, true);
        if (node < 0) return false;
        const int id = snap_next++;
        Snap & s = snaps[id];
        s.pos = pos;
        s.node = node;
        s.meta = meta;
        s.used = ++tick;
        for (int i = 0; i < snap_nb; i++) s.blks.push_back(alloc(false));
        dev->snap_save(lane, pos, s.blks);
        snap_used += snap_nb;
        nodes.at(node).snaps.push_back(id);
        if (busy[lane]) guard[lane] = id;
        return true;
    }

    // Consistency check of every count against the holders (debug builds): seqs = the open sequences' blocks.
    void check(const std::vector<const std::vector<int> *> & seqs) const {
        auto fail = [](const std::string & m) { throw std::logic_error("KV pool check: " + m); };
        const int nb = total();
        std::vector<int> r(nb, 0), a(nb, 0);
        for (const auto * v : seqs)
            for (int b : *v) {
                if (b < 0 || b >= nb) fail("sequence names block " + std::to_string(b));
                r[b]++, a[b]++;
            }
        int used = 0;
        for (const auto & kv : nodes) {
            const int id = kv.first;
            const Node & c = kv.second;
            if (id == 0) continue;
            r[c.blk]++;
            if (!nodes.count(c.parent)) fail("node " + std::to_string(id) + " lost its parent");
            const Node & p = nodes.at(c.parent);
            const bool full = (int) c.toks.size() == np;
            if (full ? (!p.kids.count(c.hash) || p.kids.at(c.hash) != id)
                     : std::find(p.parts.begin(), p.parts.end(), id) == p.parts.end())
                fail("node " + std::to_string(id) + " is not linked from its parent");
            if (c.start != p.start + (c.parent ? np : 0) || c.toks.empty() || (int) c.toks.size() > np)
                fail("node " + std::to_string(id) + " has a bad range");
            if (c.kids.empty() && c.parts.empty() && c.snaps.empty()) fail("node " + std::to_string(id) + " holds no snapshot");
            if (!full && (!c.kids.empty() || !c.parts.empty())) fail("partial node " + std::to_string(id) + " has children");
        }
        for (const auto & kv : snaps) {
            const Snap & s = kv.second;
            if (!nodes.count(s.node)) fail("snapshot " + std::to_string(kv.first) + " lost its node");
            const Node & c = nodes.at(s.node);
            if (std::find(c.snaps.begin(), c.snaps.end(), kv.first) == c.snaps.end()) fail("snapshot not linked");
            if (s.pos <= c.start || s.pos > c.start + (int) c.toks.size()) fail("snapshot outside its node");
            if (s.lane >= 0) {
                if (!s.blks.empty() || parked[s.lane] != kv.first || busy[s.lane]) fail("parked snapshot mismatch");
            } else {
                if ((int) s.blks.size() != snap_nb) fail("snapshot block count");
                for (int b : s.blks) r[b]++;
                used += snap_nb;
            }
        }
        for (size_t l = 0; l < parked.size(); l++)
            if (parked[l] >= 0 && (!snaps.count(parked[l]) || snaps.at(parked[l]).lane != (int) l)) fail("lane park mismatch");
        if (used != snap_used) fail("snapshot block total");
        int nc = 0, ne = 0, nfree = 0;
        std::vector<char> in_free(nb, 0);
        for (int b : free_) {
            if (in_free[b]) fail("block " + std::to_string(b) + " freed twice");
            in_free[b] = 1;
        }
        for (int b = 0; b < nb; b++) {
            if (r[b] != ref[b] || a[b] != act[b])
                fail("block " + std::to_string(b) + " refs " + std::to_string(ref[b]) + "/" + std::to_string(act[b]) + ", holders " +
                     std::to_string(r[b]) + "/" + std::to_string(a[b]));
            if ((ref[b] == 0) != (bool) in_free[b]) fail("block " + std::to_string(b) + " free list mismatch");
            nfree += ref[b] == 0;
            nc += ref[b] > act[b];
            ne += ref[b] > 0 && act[b] == 0;
        }
        if (nfree != (int) free_.size() || nc != n_cached || ne != n_evict) fail("block counters");
    }

  private:
    struct Node {
        int parent = -1, blk = -1, start = 0; // block of positions [start, start + toks.size())
        uint64_t hash = 0;                    // chain hash over all tokens up to this full block
        std::vector<int> toks;                // np tokens, or fewer in a partial leaf
        std::unordered_map<uint64_t, int> kids; // full children by hash
        std::vector<int> parts;                 // partial children
        std::vector<int> snaps;                 // snapshots at positions in (start, start + toks.size()]
    };
    struct Snap {
        int pos = 0, node = -1; // node: holds position pos - 1
        int lane = -1;          // >= 0: still the live state of that idle lane (no blocks yet)
        std::vector<int> blks;
        Meta meta{};
        uint64_t used = 0; // LRU stamp
    };

    int np, nblk_lane, snap_nb, cap; // tokens per block, table entries per lane, blocks per snapshot, snapshot block cap
    KvDev * dev;
    std::vector<int> free_, ref, act; // act: the part of ref held by open sequences (the rest by the cache)
    int n_cached = 0, n_evict = 0;   // blocks the cache holds a ref on / only the cache holds
    std::unordered_map<int, Node> nodes; // 0: root (no block)
    int node_next = 1;
    std::map<int, Snap> snaps;
    int snap_next = 0, snap_used = 0;
    std::vector<int> parked; // per lane: its parked snapshot or -1
    std::vector<char> busy;  // per lane: bound to an open sequence
    // per busy lane: the deepest snapshot of its sequence's prompt (the one it resumed from or took last).  A long
    // generation outlives the LRU order of the other lanes' turns; this snapshot is where its conversation's next
    // request falls back to when the turn just generated does not match, so eviction takes it only as a last resort.
    std::vector<int> guard;
    uint64_t tick = 0;
    int pin = -1; // snapshot an open() is about to restore: never evicted meanwhile

    // every ref change goes through here, so the counters and the free list stay exact
    void change(int b, int dref, int dact) {
        const int c0 = ref[b] > act[b], e0 = ref[b] > 0 && act[b] == 0;
        ref[b] += dref;
        act[b] += dact;
        n_cached += (ref[b] > act[b]) - c0;
        n_evict += (ref[b] > 0 && act[b] == 0) - e0;
        if (ref[b] == 0) free_.push_back(b);
    }
    int alloc(bool seq) {
        const int b = free_.back();
        free_.pop_back();
        change(b, 1, seq ? 1 : 0);
        return b;
    }

    static uint64_t chain(uint64_t h, const int * t, int n) {
        for (int i = 0; i < n; i++) {
            h = (h ^ (uint32_t) t[i]) * 0x100000001b3ULL;
            h ^= h >> 29;
        }
        return h;
    }
    // full child of node with tokens t[0, np): its id, -1 none, -2 another block under the same hash
    int full_kid(int node, const int * t) const {
        const Node & p = nodes.at(node);
        const auto it = p.kids.find(chain(p.hash, t, np));
        if (it == p.kids.end()) return -1;
        const Node & c = nodes.at(it->second);
        return std::equal(c.toks.begin(), c.toks.end(), t) ? it->second : -2;
    }
    int part_kid(int node, const int * t, int n) const {
        for (int c : nodes.at(node).parts) {
            const Node & k = nodes.at(c);
            if ((int) k.toks.size() == n && std::equal(k.toks.begin(), k.toks.end(), t)) return c;
        }
        return -1;
    }
    int add_node(int parent, int start, const int * t, int n, int b, bool take_ref) {
        const int id = node_next++;
        Node & c = nodes[id];
        Node & p = nodes.at(parent);
        c.parent = parent;
        c.blk = b;
        c.start = start;
        c.toks.assign(t, t + n);
        if (n == np) {
            c.hash = chain(p.hash, t, n);
            p.kids[c.hash] = id;
        } else {
            p.parts.push_back(id);
        }
        if (take_ref) change(b, 1, 0);
        return id;
    }
    // Indexes toks[0, pos) held in blk; returns the node holding position pos - 1, or -1 (hash collision or no
    // block for the copy).  copy_partial: a partial last block is copied, else shared.
    int publish(const int * toks, int pos, const std::vector<int> & blk, bool copy_partial) {
        int node = 0, made = -1;
        for (int i = 0; i * np < pos; i++) {
            const int s = i * np, n = std::min(np, pos - s);
            int next = n == np ? full_kid(node, toks + s) : part_kid(node, toks + s, n);
            if (next == -2 || (next < 0 && n < np && copy_partial && free_.empty())) {
                if (made >= 0) prune(made);
                return -1;
            }
            if (next < 0) {
                if (n < np && copy_partial) {
                    const int b = alloc(false);
                    dev->copy_block(blk[i], b);
                    next = add_node(node, s, toks + s, n, b, false);
                } else {
                    next = add_node(node, s, toks + s, n, blk[i], true);
                }
                made = next;
            }
            node = next;
        }
        return node;
    }
    // node holding position pos - 1 of toks if indexed, then its snapshot at pos (-1 if none)
    int find(const int * toks, int pos) const {
        int node = 0;
        for (int i = 0; i * np < pos; i++) {
            const int s = i * np, n = std::min(np, pos - s);
            node = n == np ? full_kid(node, toks + s) : part_kid(node, toks + s, n);
            if (node < 0) return -1;
        }
        for (int s : nodes.at(node).snaps)
            if (snaps.at(s).pos == pos) return s;
        return -1;
    }
    // deepest snapshot at a position <= min(common prefix, n - 1): (id, pos), or (-1, 0).  Every child is scanned
    // for its in-block common prefix, so a snapshot on a sibling branch that diverges after it also counts.
    std::pair<int, int> lookup(const int * toks, int n) const {
        int best = -1, bp = 0, node = 0, pos = 0;
        auto consider = [&](int c) {
            const Node & k = nodes.at(c);
            const int lim = std::min((int) k.toks.size(), n - pos);
            int m = 0;
            while (m < lim && k.toks[m] == toks[pos + m]) m++;
            for (int s : k.snaps) {
                const int sp = snaps.at(s).pos;
                if (sp <= k.start + m && sp <= n - 1 && sp > bp) best = s, bp = sp;
            }
            return m;
        };
        while (pos < n) {
            const Node & p = nodes.at(node);
            int next = -1;
            for (const auto & kv : p.kids)
                if (consider(kv.second) == np) next = kv.second;
            for (int c : p.parts) consider(c);
            if (next < 0) break;
            node = next;
            pos += np;
        }
        return {best, bp};
    }
    // blocks of the path root .. node, in position order
    void path(int node, std::vector<int> & out) const {
        out.clear();
        for (; node > 0; node = nodes.at(node).parent) out.push_back(nodes.at(node).blk);
        std::reverse(out.begin(), out.end());
    }
    // drops nodes that no longer lead to a snapshot (their KV is unusable without one), from node upwards
    void prune(int id) {
        while (id > 0) {
            Node & c = nodes.at(id);
            if (!c.snaps.empty() || !c.kids.empty() || !c.parts.empty()) return;
            Node & p = nodes.at(c.parent);
            if ((int) c.toks.size() == np) p.kids.erase(c.hash);
            else p.parts.erase(std::find(p.parts.begin(), p.parts.end(), id));
            change(c.blk, -1, 0);
            const int up = c.parent;
            nodes.erase(id);
            id = up;
        }
    }
    void remove(int sid) {
        for (int & g : guard)
            if (g == sid) g = -1;
        Snap & s = snaps.at(sid);
        if (s.lane >= 0) {
            parked[s.lane] = -1;
        } else {
            for (int b : s.blks) change(b, -1, 0);
            snap_used -= snap_nb;
        }
        const int node = s.node;
        std::vector<int> & v = nodes.at(node).snaps;
        v.erase(std::find(v.begin(), v.end(), sid));
        snaps.erase(sid);
        prune(node);
    }
    // least recently used snapshot other than pin and keep (real_only: one holding blocks), a busy lane's guard
    // only when nothing else is left; -1 if none
    int lru(bool real_only, int keep) const {
        auto guarded = [&](int id) {
            for (size_t l = 0; l < guard.size(); l++)
                if (guard[l] == id && busy[l]) return true;
            return false;
        };
        for (const bool spare : {true, false}) {
            int v = -1;
            uint64_t u = UINT64_MAX;
            for (const auto & kv : snaps)
                if (kv.first != pin && kv.first != keep && (!real_only || kv.second.lane < 0) && kv.second.used < u &&
                    !(spare && guarded(kv.first)))
                    v = kv.first, u = kv.second.used;
            if (v >= 0) return v;
        }
        return -1;
    }
    // evicts LRU snapshots (and the nodes only they kept) until k blocks are free
    bool room(int k, int keep = -1) {
        while ((int) free_.size() < k) {
            const int v = lru(false, keep);
            if (v < 0) return false;
            remove(v);
        }
        return true;
    }
    // makes room for one more snapshot under the cap and in the pool
    bool fit_snap(int keep) {
        if (snap_nb > cap) return false;
        while (snap_used + snap_nb > cap) {
            const int v = lru(true, keep);
            if (v < 0) return false;
            remove(v);
        }
        return room(snap_nb, keep);
    }
    // a parked snapshot whose lane is about to be reused: copied to blocks, or dropped when they do not fit
    void stash(int sid) {
        if (!fit_snap(sid)) {
            remove(sid);
            return;
        }
        Snap & s = snaps.at(sid);
        for (int i = 0; i < snap_nb; i++) s.blks.push_back(alloc(false));
        dev->snap_save(s.lane, s.pos, s.blks);
        snap_used += snap_nb;
        parked[s.lane] = -1;
        s.lane = -1;
    }
    int free_lane() const {
        int best = -1;
        for (int l = 0; l < (int) busy.size(); l++) {
            if (busy[l]) continue;
            if (parked[l] < 0) return l;
            if (best < 0 || snaps.at(parked[l]).used < snaps.at(parked[best]).used) best = l;
        }
        if (best < 0) throw std::runtime_error("every lane is busy (close a sequence first)");
        return best;
    }
};
