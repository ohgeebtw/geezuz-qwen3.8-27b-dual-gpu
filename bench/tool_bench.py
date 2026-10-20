#!/usr/bin/env python3
"""Long-context tool-call accuracy: an agent session of read() calls over real source files fills the context to
each depth, and facts planted at spread positions must come back as exact tool-call arguments.  Every run builds
the same cases (fixed seeds), so a run with --kv f16 and one with --kv q8 compare one to one (--compare).

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2 python3 bench/tool_bench.py --kv f16|q8
      [--depths 8000,32000,64000,128000,190000] [--cases 20] [--src DIR --src DIR ...] [--out tool_out]
  python3 bench/tool_bench.py --compare tool_out/tools_mk_..._f16.jsonl tool_out/tools_mk_..._q8.jsonl

Two kinds of case, alternating, each with three near-miss decoys planted elsewhere in the session:
  read  "open the file that defines resolve_x_y": passes when the first tool call is read with that file's path
  grep  "search for the error code logs/build_N.log reported": passes when the first tool call is grep with a
        pattern containing that code
The target sits at an evenly spread fraction of the session (case i of n at (i + 0.5) / n).  Greedy, thinking off,
one request at a time; each case records the server's prompt_tokens.
"""
import argparse
import json
import os
import random
import urllib.request

from batch_bench import CHARS_PER_TOK, LIB, MKDIR, MODEL, ROOT, Server

EXT = (".cu", ".cuh", ".h", ".cpp", ".c", ".py", ".md", ".sh")
FILE_CHARS = 4000  # reads are cut like a read tool with a default limit; short reads place planted files finely
WORDS = ("amber anchor apple arrow aspen badge basin beacon birch blade bloom bolt branch breeze brick bridge brook "
         "cable canal candle canyon cargo cedar chalk chart cider cliff clock cloud clover coast comet coral cotton "
         "crane crest crystal dawn delta dune eagle ember falcon fern field flame flint forest fossil frost garnet "
         "glacier grove harbor hazel heron hollow island ivory jade jasper kettle lagoon lantern lava ledger lemon "
         "linen lotus maple marble meadow mesa mint mirror moss nectar oak ocean olive onyx orbit orchid otter "
         "paddle palm pebble pepper pine planet plume pollen prairie quartz quill raven reed ridge river robin "
         "saddle sage salmon sand shadow shell signal silver slate spark spruce stone summit swan thistle thunder "
         "timber topaz tulip valley velvet violet walnut willow winter zephyr").split()
VERBS = ["resolve", "compute", "merge", "parse", "rebuild", "flush", "encode", "validate"]
TOOLS = [
    {"type": "function", "function": {"name": "read", "description": "Read a file of the repository.",
                                      "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                                                     "required": ["path"]}}},
    {"type": "function", "function": {"name": "grep", "description": "Search file contents for a regex pattern.",
                                      "parameters": {"type": "object",
                                                     "properties": {"pattern": {"type": "string"},
                                                                    "path": {"type": "string"}},
                                                     "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "find", "description": "Find files by name pattern.",
                                      "parameters": {"type": "object", "properties": {"name": {"type": "string"}},
                                                     "required": ["name"]}}},
    {"type": "function", "function": {"name": "ls", "description": "List a directory.",
                                      "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                                                     "required": ["path"]}}},
]
SYSTEM = ("You are a coding agent working in a repository; paths are relative to its root. Act with the tools "
          "directly, without explaining first.")


def corpus(dirs):
    """(path, text) of every source file under dirs, in a fixed order"""
    out = []
    for d in dirs:
        for root, subs, files in os.walk(d):
            subs[:] = sorted(s for s in subs if not s.startswith(".") and s not in ("build", "dist", "node_modules"))
            for f in sorted(files):
                if f.endswith(EXT):
                    p = os.path.join(root, f)
                    try:
                        text = open(p, encoding="utf-8").read()
                    except (UnicodeDecodeError, OSError):
                        continue
                    if len(text) > 200:
                        out.append((os.path.relpath(p, os.path.dirname(d.rstrip("/"))), text[:FILE_CHARS]))
    return out


def py_file(fn, rng):
    a, b = rng.sample(WORDS, 2)
    return (f'"""Helpers for the {a} {b} stage."""\n\n\ndef {fn}(items, limit=None):\n'
            f'    """Returns the {a} entries of items, at most limit of them."""\n    out = []\n'
            f'    for it in items:\n        if it.get("{a}") and not it.get("{b}"):\n            out.append(it)\n'
            f'        if limit is not None and len(out) >= limit:\n            break\n    return out\n')


def log_file(code, rng):
    lines = [f"[{i:3d}/212] Building CXX object src/CMakeFiles/core.dir/{rng.choice(WORDS)}.cpp.o" for i in range(1, 25)]
    lines.insert(rng.randrange(10, 20), f"error {code}: undefined reference to '{rng.choice(WORDS)}_{rng.choice(WORDS)}'")
    return "\n".join(lines + ["make: *** [Makefile:142: all] Error 2"]) + "\n"


def case(files, depth, i, n):
    """one case: (messages, expected tool, expected argument, planted positions)"""
    rng = random.Random(depth * 1000 + i)
    kind = "read" if i % 2 == 0 else "grep"
    verb, w1 = rng.choice(VERBS), rng.choice(WORDS)
    w2s = rng.sample([w for w in WORDS if w != w1], 4)
    planted = []  # (path, text); the first one is the target
    if kind == "read":
        for w2 in w2s:
            path = f"src/{rng.choice(WORDS)}/{rng.choice(WORDS)}_{rng.choice(WORDS)}.py"
            planted.append((path, py_file(f"{verb}_{w1}_{w2}", rng)))
        question = f"Open the file that defines `{verb}_{w1}_{w2s[0]}` so we can change it."
        expect = planted[0][0]
    else:
        nums = rng.sample(range(100, 1000), 4)
        codes = [f"E{rng.randrange(10000, 99999)}" for _ in nums]
        for num, code in zip(nums, codes):
            planted.append((f"logs/build_{num}.log", log_file(code, rng)))
        question = f"Search the repository for the error code that logs/build_{nums[0]}.log reported."
        expect = codes[0]
    budget = int(depth * CHARS_PER_TOK) - sum(len(t) for _, t in planted)
    order = list(range(len(files)))
    random.Random(depth).shuffle(order)  # one session per depth: the cases share it up to their first planted file
    filler, used = [], 0
    for k in order:
        if used >= budget:
            break
        p, t = files[k]
        t = t[: budget - used]
        filler.append((p, t))
        used += len(t)
    if used < budget:
        raise SystemExit(f"the sources hold {used} characters, depth {depth} needs {budget}: add --src directories")
    slots = [round((i + 0.5) / n * len(filler))] + rng.sample(range(len(filler) + 1), 3)
    items = list(filler)
    for pos, pl in sorted(zip(slots, planted), key=lambda x: -x[0]):  # back to front keeps the earlier indices
        items.insert(pos, pl)
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Read through the repository first; I will give you a task afterwards."}]
    for k, (p, t) in enumerate(items):
        cid = f"call_{k}"
        msgs.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": cid, "type": "function", "function": {"name": "read", "arguments": json.dumps({"path": p})}}]})
        msgs.append({"role": "tool", "tool_call_id": cid, "content": t})
    msgs.append({"role": "user", "content": question})
    return msgs, kind, expect, slots[0] / max(1, len(filler))


def norm(p):
    p = p.strip()
    while p.startswith(("./", "/")):
        p = p[1:] if p.startswith("/") else p[2:]
    return p


def judge(kind, expect, msg):
    calls = msg.get("tool_calls") or []
    if not calls:
        return False, None
    f = calls[0]["function"]
    try:
        a = json.loads(f.get("arguments") or "{}")
    except json.JSONDecodeError:
        return False, f
    if kind == "read":
        return f.get("name") == "read" and norm(str(a.get("path", ""))) == expect, f
    return f.get("name") == "grep" and expect in str(a.get("pattern", "")), f


def ask(url, msgs, gen):
    body = {"messages": msgs, "tools": TOOLS, "max_tokens": gen, "temperature": 0.0, "seed": 1,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=7200) as r:
        return json.load(r)


def compare(a, b):
    ra, rb = ([json.loads(x) for x in open(p)] for p in (a, b))
    print(f"{'depth':>7} {'A pass':>8} {'B pass':>8} {'same call':>10}")
    for d in sorted({r["depth"] for r in ra}):
        xa = [r for r in ra if r["depth"] == d]
        xb = {r["case"]: r for r in rb if r["depth"] == d}
        same = sum(1 for r in xa if r["case"] in xb and r["call"] == xb[r["case"]]["call"])
        print(f"{d:>7} {sum(r['ok'] for r in xa):>4}/{len(xa):<3} {sum(r['ok'] for r in xb.values()):>4}/{len(xb):<3} "
              f"{same:>6}/{len(xa)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="mk", choices=["mk", "llama", "vllm"])
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--lib", default=LIB)
    ap.add_argument("--kv", default="f16", choices=["f16", "q8"])
    ap.add_argument("--depths", default="8000,32000,64000,128000,190000")
    ap.add_argument("--cases", type=int, default=20)
    ap.add_argument("--ctx", type=int, default=200000, help="server context: the deepest prompt plus the reply")
    ap.add_argument("--gen", type=int, default=256)
    ap.add_argument("--src", action="append", help="source directories for the session (default: this repo and ../llama.cpp)")
    ap.add_argument("--port", type=int, default=18098)
    ap.add_argument("--drafts", type=int, default=4)
    ap.add_argument("--out", default="tool_out")
    ap.add_argument("--compare", nargs=2, metavar="JSONL")
    args = ap.parse_args()
    if args.compare:
        return compare(*args.compare)
    args.temp = 0.0
    files = corpus(args.src or [d for d in (MKDIR, os.path.join(ROOT, "llama.cpp")) if os.path.isdir(d)])
    depths = [int(x) for x in args.depths.split(",")]
    cases = [(d, i, *case(files, d, i, args.cases)) for d in depths for i in range(args.cases)]  # fail before loading
    tag = f"{args.engine}_{os.path.basename(args.model.rstrip('/'))}_{args.kv}"
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"tools_{tag}.jsonl")
    srv = Server(args, 1, args.ctx)
    try:
        with open(path, "w") as f:
            for d, i, msgs, kind, expect, frac in cases:
                r = ask(srv.url, msgs, args.gen)
                ok, call = judge(kind, expect, r["choices"][0]["message"])
                row = {"depth": d, "case": i, "kind": kind, "at": round(frac, 3), "expect": expect, "ok": ok,
                       "call": call, "content": r["choices"][0]["message"].get("content"),
                       "prompt_tokens": r.get("usage", {}).get("prompt_tokens")}
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(f"depth {d} case {i:2d} {kind} at {frac:.2f}: {'ok' if ok else 'FAIL'} "
                      f"({row['prompt_tokens']} tok)", flush=True)
        print(srv.kv_report())
    finally:
        srv.stop()
    rows = [json.loads(x) for x in open(path)]
    for d in depths:
        x = [r for r in rows if r["depth"] == d]
        print(f"{tag} depth {d}: {sum(r['ok'] for r in x)}/{len(x)}, prompt {min(r['prompt_tokens'] or 0 for r in x)} to "
              f"{max(r['prompt_tokens'] or 0 for r in x)} tok")


if __name__ == "__main__":
    main()
