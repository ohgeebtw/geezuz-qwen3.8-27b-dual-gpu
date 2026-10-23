"""Fake-engine simulation of mk_server.Server's scheduler (admission, preemption, hold, think stop)."""
import random, sys, types, queue, collections, importlib.util, threading, time, os

# stub jinja2
j = types.ModuleType("jinja2"); j.exceptions = types.SimpleNamespace(TemplateError=Exception)
sb = types.ModuleType("jinja2.sandbox"); sb.ImmutableSandboxedEnvironment = object
sys.modules["jinja2"] = j; sys.modules["jinja2.sandbox"] = sb
spec = importlib.util.spec_from_file_location("mk", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server", "mk_server.py"))
mk = importlib.util.module_from_spec(spec); spec.loader.exec_module(mk)
mk.log = lambda *a: None

NP, MAXT, KV_AHEAD, EOG = 16, 5, 9, 0
THINK_TOKS = list(range(900, 920))


class FakeEng:
    def __init__(self, nblocks, ctx, rng):
        self.total, self.ctx, self.rng = nblocks, ctx, rng
        self.seqs = {}  # id -> dict(toks, blocks, dn)
        self.cache = collections.OrderedDict()  # tuple(prefix) -> blocks (LRU order)
        self.nid = 0
        self.calls, self.pos = 0, 0
        self.closed_keep = 0

    def used(self):
        return sum(s["blocks"] for s in self.seqs.values()) + sum(self.cache.values())

    def free(self):
        return self.total - self.used()

    def room(self, k):
        while self.free() < k and self.cache:
            self.cache.popitem(last=False)
        return self.free() >= k

    def pool_stats(self):
        return dict(free_blocks=self.free(), total_blocks=self.total, block_tokens=NP,
                    cached_blocks=sum(self.cache.values()), evictable_blocks=sum(self.cache.values()))

    def kv_ahead(self):
        return KV_AHEAD

    def prefill_progress(self):
        return self.calls, self.pos

    def seq_open(self, toks):
        if len(self.seqs) >= SLOTS:
            raise RuntimeError("every lane busy")
        best = 0
        for k in list(self.cache):
            if len(k) <= len(toks) - 1 and list(k) == toks[:len(k)] and len(k) > best:
                best = len(k)
        if best:
            self.cache.move_to_end(tuple(toks[:best]))
            # shared blocks: modelled as freed from the cache entry into the seq
            self.cache.pop(tuple(toks[:best]))
        i = self.nid; self.nid += 1
        self.seqs[i] = dict(toks=list(toks[:best]), blocks=-(-best // NP), dn=best)
        return i, best

    def seq_close(self, i, keep=True):
        s = self.seqs.pop(i)
        if keep and s["dn"] > 0:
            self.cache[tuple(s["toks"][:s["dn"]])] = s["blocks"]
            self.closed_keep += 1

    def _grow(self, s, upto):
        want = -(-upto // NP)
        if want > -(-self.ctx // NP):
            raise RuntimeError("sequence exceeds context")
        k = max(0, want - s["blocks"])
        if k and not self.room(k):
            raise mk.PoolFull("full")
        s["blocks"] = max(s["blocks"], want)

    def _next(self, s):
        return EOG if self.rng.random() < 0.002 else self.rng.randrange(1, 800)

    def prefill(self, i, toks, snaps=()):
        return self.append(i, toks, snaps)

    def append(self, i, toks, snaps=()):
        s = self.seqs[i]
        self.calls, self.pos = self.calls + 1, s["dn"]
        if not toks:
            raise RuntimeError("empty prompt")
        L = len(s["toks"]) + len(toks)
        if L + KV_AHEAD > self.ctx:
            raise RuntimeError("prompt exceeds context")
        self._grow(s, L + MAXT)
        s["toks"] += toks
        s["dn"] = self.pos = L
        t = self._next(s)
        s["toks"].append(t)
        return t

    def truncate(self, i, n):
        s = self.seqs[i]
        assert s["dn"] <= n <= len(s["toks"]), (s["dn"], n, len(s["toks"]))
        s["toks"] = s["toks"][:n]

    def step_batch(self, ids):
        assert len(set(ids)) == len(ids)
        need = 0
        for i in ids:
            s = self.seqs[i]
            n = len(s["toks"]) - 1
            if n + KV_AHEAD > self.ctx:
                raise RuntimeError("context full")
            need += max(0, -(-(n + KV_AHEAD) // NP) - s["blocks"])
        if need and not self.room(need):
            raise mk.PoolFull("full")
        outs = []
        for i in ids:
            s = self.seqs[i]
            n = len(s["toks"]) - 1
            s["blocks"] = max(s["blocks"], -(-(n + KV_AHEAD) // NP))
            k = self.rng.randint(1, MAXT)
            o = [self._next(s) for _ in range(k)]
            s["dn"] = len(s["toks"])  # engine: dn_c = n + 1, the first output's position
            s["toks"] += o
            outs.append(o)
        return outs

    def sequence(self, i):
        return list(self.seqs[i]["toks"])

    def seq_len(self, i):
        return len(self.seqs[i]["toks"])

    def seq_blocks(self, i):
        return self.seqs[i]["blocks"]

    def tokenize(self, text):
        return list(THINK_TOKS)

    def piece(self, t):
        return b"x"

    def is_eog(self, t):
        return t == EOG

    def set_sampling(self, *a):
        pass


def run(seed, nblocks, slots, njobs, ctx=2048):
    global SLOTS
    SLOTS = slots
    rng = random.Random(seed)
    srv = object.__new__(mk.Server)
    srv.args = types.SimpleNamespace(slots=slots, prefill_chunk=64, prefill_share=rng.choice([0.25, 1.0]), ctx=ctx,
                                     think_budget=0, temperature=1.0, top_p=0.95, temperature_instruct=0.7,
                                     top_p_instruct=0.8, top_k=20, min_p=0.0)
    srv.eng = FakeEng(nblocks, ctx, rng)
    srv.warned = set(); srv.queue = queue.Queue(); srv.waiting = collections.deque()
    srv.pool = srv.pool_stats(); srv.prefill_owed = 0.0; srv.kv_ahead = KV_AHEAD; srv.active = {}
    srv.admissions = 0; srv.preempted = 0; srv.think_stop_n = len(THINK_TOKS)
    srv.slot_status = [{"slot": s, "busy": False, "phase": "idle", "action": None, "updated": 0} for s in range(slots)]
    srv.pf, srv.pf_on, srv.pf_seen = None, threading.Event(), None
    srv.last = None
    jobs = []
    system = [rng.randrange(1, 800) for _ in range(rng.randint(20, 200))]  # a prefix some prompts share
    for k in range(njobs):
        toks = [rng.randrange(1, 800) for _ in range(rng.randint(5, 400))]
        shared = rng.random() < 0.4
        if shared:
            toks = system + toks
        max_new = min(rng.randint(50, 1200), ctx - len(toks) - 16)
        budget = rng.choice([0, 0, 100, 300])
        job = mk.Job({"temperature": 0}, toks, budget > 0, max_new, budget)
        job.snaps = [len(system)] if shared else []
        job.cancelled = rng.random() < 0.05
        jobs.append(job); srv.queue.put(job)
    # drive run() manually: copy of the loop body with an iteration bound
    it = 0
    while any(j.events.empty() or not _done(j) for j in jobs):
        it += 1
        if it > 200000:
            return "HANG", srv, jobs
        if not srv.active and not srv.waiting:
            if srv.queue.empty():
                break
            srv.waiting.append(srv.queue.get())
        srv.schedule()
        # invariants
        for s, j in srv.active.items():
            assert j.slot == s and j.seq in srv.eng.seqs, "active job without sequence"
        assert len(srv.eng.seqs) == len(srv.active), ("leaked seqs", len(srv.eng.seqs), len(srv.active))
        for st in srv.status()["slots"]:
            if st["busy"] and st["phase"] in ("prefill", "resume"):
                j = srv.active[st["slot"]]
                assert st["prefill_from"] <= st["prefilled"] <= len(j.feed), ("prefill progress", st, len(j.feed))
    res = collections.Counter()
    for j in jobs:
        ev = list(j.events.queue)
        last = ev[-1]
        res[last[0] if last[0] != "done" else last[1][0]] += 1
        if last[0] == "error":
            res["err:" + str(last[1])[:40]] += 1
        if last[0] == "done" and not j.cancelled and j.n_gen > j.max_new:
            res["over_max"] += 1
    return res, srv, jobs


def _done(j):
    return any(e[0] in ("done", "error") for e in list(j.events.queue))


if __name__ == "__main__":
    agg = collections.Counter(); pre = 0
    for seed in range(300):
        nb = random.Random(seed).choice([140, 200, 300, 600])
        r, srv, jobs = run(seed, nb, random.Random(seed + 1).choice([2, 4, 8]), 20)
        if r == "HANG":
            print("HANG seed", seed); continue
        agg.update(r); pre += srv.preempted
    print(agg, "preemptions", pre)
