#!/usr/bin/env python3
"""OpenAI-compatible chat server for the Qwen3.8-27B megakernel (a drop-in for the parts of
llama-server's API that open-jet uses).

  python3 server/mk_server.py -m model.gguf --port 18080 [--ctx 32768] [--draft 4]

POST /v1/chat/completions (streaming or not): the request is rendered with the chat template stored
in the GGUF (enable_thinking / preserve_thinking / reasoning_effort via chat_template_kwargs, tools
via `tools`), reasoning is returned as `reasoning_content` (a tool call opened inside thinking, `<tool_call>` or
`<function=`, ends it), Qwen's XML tool calls as structured
`tool_calls`.  Sampling: temperature, top_p, top_k (<= 32), min_p, seed.  Requests reuse processed prompt
prefixes across requests and conversations (the engine's prefix cache in the KV pool): a request starts from
the longest cached prefix with a stored DeltaNet state (a finished request's end, the state before a prompt's
last token, the end of a leading system message and the start of the last user message) and only the rest is
prefilled; usage.prompt_tokens_details.cached_tokens reports the reused tokens.
GET /health, GET /v1/models. --slots N decodes up to N requests at once; more wait for a free slot. A prompt is
prefilled between decode cycles; while others decode it goes in chunks (--prefill-chunk), rounds alternating between
shortest remaining prompt first and longest waiting first, for a share of the time (--prefill-share), so running
requests keep the rest of their speed and a long prompt is not starved by a stream of short ones. A prompt that
shares a snapshot point (e.g. the system message) with a prompt still being prefilled waits for it and starts from
its snapshot. Every log line of a request carries its id: the client's X-Request-Id (letters, digits, ._:-, at most
128) or else the response id; the response echoes it in an X-Request-Id header. The requests share one KV pool
(--kv-mem) with the prefix cache: a request is admitted when the free blocks plus the cached ones the engine can
evict (least recently used first) cover its prompt and a little growth. When the pool runs full anyway, the most
recently admitted request is preempted: its sequence stays cached, it waits at the head of the queue and resumes by
prefilling its prompt plus what it generated (mostly cached), its stream continuing seamlessly; a request running
alone that cannot grow ends with finish_reason "length".
GET /status: what the engine is doing now (phase, live token count, tail of the reasoning / text / tool call being
generated, per slot under `slots` with the request `id`), the KV pool (`kv_pool`, read after every decode cycle;
`used_blocks` = held by running requests, `updated` = read time), preemptions (`preempted`, per slot `preempts`),
the MTP draft counters since load (`draft`: cycles, drafted, accepted, accept_rate)
and the stats of the last finished request.  A slot reading its prompt (phase prefill / resume) shows `prefilled`
(sequence position reached, live per 2,048-token chunk) and `prefill_tok_s` (prompt tokens processed since
admission, cached ones excluded, per second up to when that position was reached).
"""
import argparse
import array
import bisect
import codecs
import collections
import ctypes
import hashlib
import json
import os
import queue
import random
import re
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment

HERE = os.path.dirname(os.path.abspath(__file__))
# Forced into the reasoning when the thinking budget runs out, so the model answers instead of hitting max_tokens.
THINK_STOP = "\n\nI have thought enough and will answer now based on the reasoning so far.\n</think>\n\n"
TOPK_MAX = 32
STATUS_TAIL = 400  # characters of reasoning / text / tool call (or of a tool argument) GET /status shows


def log(*a):
    print("[mk-server]", *a, file=sys.stderr, flush=True)


MK_POOL_FULL = -2  # capi.cpp: too few free KV blocks; the call changed nothing


class PoolFull(RuntimeError):
    pass


class Engine:
    """ctypes binding of capi.cpp"""

    def __init__(self, lib, model, ctx, n_draft, draft_vocab, kv_mode=-1, n_seq=1, kv_mem=0):
        L = self.L = ctypes.CDLL(lib)
        vp, ip, cp = ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_char_p
        lp = ctypes.POINTER(ctypes.c_long)
        sig = {
            "mk_new": (vp, [cp, ctypes.c_int, ctypes.c_int, cp, ctypes.c_int, ctypes.c_int, ctypes.c_int]),
            "mk_pool_stats": (None, [vp, ip, ip, ip, ip, ip]),
            "mk_stats": (None, [vp, lp, lp, lp]),
            "mk_seq_open": (ctypes.c_int, [vp, ip, ctypes.c_int, ip]),
            "mk_seq_close": (ctypes.c_int, [vp, ctypes.c_int, ctypes.c_int]),
            "mk_seq_blocks": (ctypes.c_int, [vp, ctypes.c_int]),
            "mk_kv_ahead": (ctypes.c_int, []),
            "mk_last_error": (cp, []),
            "mk_chat_template": (ctypes.c_int, [vp, cp, ctypes.c_int]),
            "mk_tokenize": (ctypes.c_int, [vp, cp, ctypes.c_int, ip, ctypes.c_int, ctypes.c_int]),
            "mk_piece": (ctypes.c_int, [vp, ctypes.c_int, cp, ctypes.c_int]),
            "mk_is_eog": (ctypes.c_int, [vp, ctypes.c_int]),
            "mk_set_sampling": (ctypes.c_int, [vp, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_int,
                                               ctypes.c_float, ctypes.c_ulonglong]),
            "mk_prefill": (ctypes.c_int, [vp, ctypes.c_int, ip, ctypes.c_int, ip, ctypes.c_int]),
            "mk_append": (ctypes.c_int, [vp, ctypes.c_int, ip, ctypes.c_int, ip, ctypes.c_int]),
            "mk_step_batch": (ctypes.c_int, [vp, ip, ctypes.c_int, ip, ctypes.c_int, ip]),
            "mk_truncate": (ctypes.c_int, [vp, ctypes.c_int, ctypes.c_int]),
            "mk_sequence": (ctypes.c_int, [vp, ctypes.c_int, ip, ctypes.c_int]),
            "mk_prefill_progress": (ctypes.c_longlong, [vp]),
        }
        for name, (res, args) in sig.items():
            f = getattr(L, name)
            f.restype, f.argtypes = res, args
        self.h = L.mk_new(model.encode(), ctx, n_draft, draft_vocab.encode() if draft_vocab else None, kv_mode,
                          n_seq, kv_mem)
        if not self.h:
            raise RuntimeError(L.mk_last_error().decode())
        self.ctx = ctx
        self._pieces = {}

    def _err(self, what, code=-1):
        msg = f"{what}: {self.L.mk_last_error().decode()}"
        return PoolFull(msg) if code == MK_POOL_FULL else RuntimeError(msg)

    def chat_template(self):
        n = self.L.mk_chat_template(self.h, None, 0)
        buf = ctypes.create_string_buffer(n + 1)
        self.L.mk_chat_template(self.h, buf, n)
        return buf.raw[:n].decode()

    def tokenize(self, text, special=True):
        b = text.encode()
        cap = len(b) + 16
        out = (ctypes.c_int * cap)()
        n = self.L.mk_tokenize(self.h, b, len(b), out, cap, int(special))
        if n < 0:
            cap = -n
            out = (ctypes.c_int * cap)()
            n = self.L.mk_tokenize(self.h, b, len(b), out, cap, int(special))
        return list(out[:n])

    def piece(self, tok):
        p = self._pieces.get(tok)
        if p is None:
            buf = ctypes.create_string_buffer(256)
            n = self.L.mk_piece(self.h, tok, buf, 256)
            p = self._pieces[tok] = buf.raw[:max(n, 0)]
        return p

    def is_eog(self, tok):
        return bool(self.L.mk_is_eog(self.h, tok))

    def seq_open(self, toks):
        """opens a sequence from the longest cached prefix of toks with a stored state; returns (id, reused tokens)"""
        reused = ctypes.c_int()
        i = self.L.mk_seq_open(self.h, (ctypes.c_int * max(len(toks), 1))(*toks), len(toks), ctypes.byref(reused))
        if i < 0:
            raise self._err("seq_open")
        return i, reused.value

    def seq_close(self, i, keep=True):
        """keep: the sequence's tokens and state stay cached for later requests"""
        if self.L.mk_seq_close(self.h, i, int(keep)) < 0:
            raise self._err("seq_close")

    def seq_blocks(self, i):
        n = self.L.mk_seq_blocks(self.h, i)
        if n < 0:
            raise self._err("seq_blocks")
        return n

    def set_sampling(self, i, temp, top_p, top_k, min_p, seed):
        if self.L.mk_set_sampling(self.h, i, temp, top_p, top_k, min_p, seed) < 0:
            raise self._err("sampling")

    def _feed(self, f, what, i, toks, snaps):
        arr = (ctypes.c_int * len(toks))(*toks)
        r = f(self.h, i, arr, len(toks), (ctypes.c_int * max(len(snaps), 1))(*snaps), len(snaps))
        if r < 0:
            raise self._err(what, r)
        return r

    def prefill(self, i, toks, snaps=()):
        """feeds toks to sequence i, caching the states at the sequence positions snaps and before the last token;
        returns the next token"""
        return self._feed(self.L.mk_prefill, "prefill", i, toks, snaps)

    def append(self, i, toks, snaps=()):
        """prefill() without the state before the last token (forced tokens, intermediate prompt chunks)"""
        return self._feed(self.L.mk_append, "append", i, toks, snaps)

    def prefill_progress(self):
        """(prefill() / append() calls so far, mod 2^31; sequence position the latest one has processed up to), live
        while it runs: the one engine call a handler may make while the scheduler drives the engine (lock-free)"""
        v = self.L.mk_prefill_progress(self.h)
        return v >> 32, v & 0xFFFFFFFF

    def step_batch(self, ids):
        """one cycle of the given sequences; returns each one's tokens"""
        n, cap = len(ids), 8
        out, counts = (ctypes.c_int * (n * cap))(), (ctypes.c_int * n)()
        r = self.L.mk_step_batch(self.h, (ctypes.c_int * n)(*ids), n, out, cap, counts)
        if r < 0:
            raise self._err("decode", r)
        return [list(out[b * cap:b * cap + counts[b]]) for b in range(n)]

    def pool_stats(self):
        v = [ctypes.c_int() for _ in range(5)]
        self.L.mk_pool_stats(self.h, *(ctypes.byref(x) for x in v))
        return dict(zip(("free_blocks", "total_blocks", "block_tokens", "cached_blocks", "evictable_blocks"),
                        (x.value for x in v)))

    def draft_stats(self):
        """decode cycles, drafts verified and drafts accepted since load (summed over the batch's sequences)"""
        v = [ctypes.c_long() for _ in range(3)]
        self.L.mk_stats(self.h, *(ctypes.byref(x) for x in v))
        return dict(zip(("cycles", "drafted", "accepted"), (x.value for x in v)))

    def kv_ahead(self):
        return self.L.mk_kv_ahead()

    def truncate(self, i, n):
        if self.L.mk_truncate(self.h, i, n) < 0:
            raise self._err("truncate")

    def seq_len(self, i):
        n = self.L.mk_sequence(self.h, i, None, 0)
        if n < 0:
            raise self._err("sequence")
        return n

    def sequence(self, i):
        """sequence i's tokens: prompt(s) and emitted tokens"""
        n = self.L.mk_sequence(self.h, i, None, 0)
        if n < 0:
            raise self._err("sequence")
        out = (ctypes.c_int * max(n, 1))()
        self.L.mk_sequence(self.h, i, out, n)
        return list(out[:n])


# ---------------------------------------------------------------------------------------------
# chat template and output parsing


def make_template(src):
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])

    def raise_exception(msg):
        raise jinja2.exceptions.TemplateError(msg)

    env.globals["raise_exception"] = raise_exception
    env.filters["tojson"] = lambda x, indent=None, ensure_ascii=False, sort_keys=False: json.dumps(
        x, indent=indent, ensure_ascii=ensure_ascii, sort_keys=sort_keys)
    return env.from_string(src)


class BadRequest(Exception):
    pass


def normalize_messages(messages):
    out = []
    for m in messages:
        m = dict(m)
        content = m.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") not in (None, "text"):
                    raise BadRequest(f"unsupported content part type {part.get('type')!r} (text only)")
        if content is None:
            m["content"] = ""
        if m.get("role") == "assistant" and m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                tc = dict(tc)
                fn = dict(tc.get("function") or {})
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args.strip() else {}
                    except json.JSONDecodeError:
                        args = {}
                fn["arguments"] = args if isinstance(args, dict) else {}
                tc["function"] = fn
                calls.append(tc)
            m["tool_calls"] = calls
        out.append(m)
    return out


def protect_spans(messages, tools, mark, spans):
    def wrap(s, escape=False):
        core = s.strip()
        if not core:
            return s
        lead, trail = s[:len(s) - len(s.lstrip())], s[len(s.rstrip()):]
        spans.append(json.dumps(core, ensure_ascii=False)[1:-1] if escape else core)
        return lead + mark + str(len(spans) - 1) + mark + trail

    out = []
    for m in messages:
        m = dict(m)
        c = m.get("content")
        if isinstance(c, str):
            m["content"] = wrap(c)
        elif isinstance(c, list):
            parts = []
            for part in c:
                if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                    part = dict(part)
                    part["text"] = wrap(part["text"])
                parts.append(part)
            m["content"] = parts
        if isinstance(m.get("reasoning_content"), str):
            m["reasoning_content"] = wrap(m["reasoning_content"])
        if m.get("role") == "assistant" and m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                tc = dict(tc)
                fn = dict(tc.get("function") or {})
                if isinstance(fn.get("arguments"), dict):
                    fn["arguments"] = {k: wrap(v) if isinstance(v, str) else v for k, v in fn["arguments"].items()}
                tc["function"] = fn
                calls.append(tc)
            m["tool_calls"] = calls
        out.append(m)
    new_tools = None
    if tools:
        new_tools = []
        for t in tools:
            if isinstance(t, dict):
                t = dict(t)
                if isinstance(t.get("function"), dict):
                    f = dict(t["function"])
                    if isinstance(f.get("description"), str):
                        f["description"] = wrap(f["description"], True)
                    t["function"] = f
                elif isinstance(t.get("description"), str):
                    t["description"] = wrap(t["description"], True)
            new_tools.append(t)
    return out, new_tools


TOOL_CALL_RE = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)(?:</function>\s*(?:</tool_call>|$)|$)", re.S)
PARAM_RE = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)


def parse_tool_calls(text, tools):
    schemas = {}
    for t in tools or []:
        fn = t.get("function", t)
        schemas[fn.get("name")] = (fn.get("parameters") or {}).get("properties") or {}
    calls = []
    for m in TOOL_CALL_RE.finditer(text):
        name = m.group(1).strip()
        props = schemas.get(name, {})
        args = {}
        for p in PARAM_RE.finditer(m.group(2)):
            key, val = p.group(1).strip(), p.group(2)
            typ = (props.get(key) or {}).get("type")
            if typ == "string" or (typ is None and not val.strip()[:1] in "[{0123456789-tfn"):
                args[key] = val
            else:
                try:
                    args[key] = json.loads(val)
                except json.JSONDecodeError:
                    args[key] = val
        calls.append({"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                      "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
    return calls


class OutputParser:
    """Splits the generated text into reasoning (before </think>), content and tool-call markup."""

    def __init__(self, thinking, tools):
        self.mode = "reasoning" if thinking else "content_start"
        self.tools = tools
        self.buf = ""
        self.tool_text = ""

    def feed(self, text, final=False):
        """returns [(kind, text)] with kind in {reasoning, content}"""
        out = []
        self.buf += text
        while True:
            if self.mode == "reasoning":
                i = self.buf.find("</think>")
                if self.tools:
                    js = [x for x in (self.buf.find("<tool_call>"), self.buf.find("<function=")) if x >= 0]
                    j = min(js) if js else -1
                    if j >= 0 and (i < 0 or j < i):
                        out.append(("reasoning", self.buf[:j].rstrip()))
                        self.tool_text = self.buf[j:] if self.buf.startswith("<tool_call>", j) \
                            else "<tool_call>\n" + self.buf[j:]
                        self.buf = ""
                        self.mode = "tools"
                        break
                if i >= 0:
                    out.append(("reasoning", self.buf[:i]))
                    self.buf = self.buf[i + len("</think>"):]
                    self.mode = "content_start"
                    continue
                keep = 0 if final else len("<tool_call>" if self.tools else "</think>") - 1
                if len(self.buf) > keep:
                    out.append(("reasoning", self.buf[:len(self.buf) - keep]))
                    self.buf = self.buf[len(self.buf) - keep:]
                break
            if self.mode == "content_start":
                self.buf = self.buf.lstrip()
                if not self.buf:
                    break
                self.mode = "content"
                continue
            if self.mode == "content":
                if self.tools:
                    i = self.buf.find("<tool_call>")
                    if i >= 0:
                        out.append(("content", self.buf[:i].rstrip()))
                        self.tool_text = self.buf[i:]
                        self.buf = ""
                        self.mode = "tools"
                        break
                    keep = 0 if final else len("<tool_call>") - 1
                else:
                    keep = 0
                if final:
                    out.append(("content", self.buf.rstrip()))
                    self.buf = ""
                elif len(self.buf) > keep:
                    # trailing whitespace waits for more text: it is dropped before a tool call or the end
                    seg = self.buf[:len(self.buf) - keep]
                    head = seg.rstrip()
                    out.append(("content", head))
                    self.buf = seg[len(head):] + self.buf[len(self.buf) - keep:]
                break
            # tools: collect everything
            self.tool_text += self.buf
            self.buf = ""
            break
        return [(k, t) for k, t in out if t]


# ---------------------------------------------------------------------------------------------


class TurnReplay:
    """Generated assistant turns, put back into later prompts in place of their template rendering.

    A client sends a turn back as parsed text and the chat template renders it again.  Whitespace the parser dropped
    and markup it normalized (a tool call opened inside thinking gets a </think>) make that rendering differ from the
    tokens the model generated, and on this hybrid model a prompt that differs from the cache once is prefilled from
    there on: the DeltaNet state cannot be cut back.  A finished request leaves its prompt and generated turn; the next
    prompt that starts with that prompt and renders the turn with the same text (whitespace and the markup above
    aside) gets the generated tokens instead, and every later prompt of the conversation gets them through the stored
    (rendered -> generated) pair.  A turn whose text differs (edited by the client) keeps its rendering."""

    MARKUP = ("<think>", "</think>", "<tool_call>")

    def __init__(self, im_start, im_end, text, cap=4096, recent=64):
        self.im_start, self.im_end = im_start, im_end  # token ids
        self.text = text  # tokens -> str
        self.cap = cap
        self.subs = collections.OrderedDict()  # (generation prompt, rendered turn) -> generated turn, LRU order
        self.gps = collections.Counter()  # generation prompts of the stored pairs
        self.pending = collections.deque(maxlen=recent)  # finished turns no later prompt has met yet
        self.lock = threading.Lock()

    @staticmethod
    def digest(toks):
        return hashlib.blake2b(array.array("i", toks).tobytes(), digest_size=16).digest()

    def finished(self, rid, prompt, gen):
        """scheduler side: a request ended its turn with an end-of-generation token; gen = the turn's tokens"""
        try:
            i = len(prompt) - 1 - prompt[::-1].index(self.im_start)
        except ValueError:
            return
        if gen[-1] != self.im_end:  # the rendering closes the turn with <|im_end|>
            gen = gen[:-1] + [self.im_end]
        entry = (rid, len(prompt), tuple(prompt[-16:]), self.digest(prompt), tuple(prompt[i:]), gen)
        with self.lock:
            self.pending.append(entry)

    def apply(self, toks, rid, log=None):
        """handler side: the prompt with every known turn replayed and the turn of a finished request it continues"""
        with self.lock:
            toks = self.substitute(toks)
            for entry in reversed(self.pending):
                prev, n, tail, h, gp, gen = entry
                if n >= len(toks) or tuple(toks[max(0, n - 16):n]) != tail or self.digest(toks[:n]) != h:
                    continue
                self.pending.remove(entry)
                try:
                    e = toks.index(self.im_end, n)
                except ValueError:
                    break
                rendered = toks[n:e + 1]
                if rendered == gen:
                    break
                k = next((k for k, (a, b) in enumerate(zip(rendered, gen)) if a != b), min(len(rendered), len(gen)))
                same = self.norm(rendered) == self.norm(gen)
                if log:
                    log(f"[{rid}]: turn of [{prev}] rendered as {len(rendered)} tok, generated as {len(gen)} tok, "
                        f"first difference at token {k}: {self.text(rendered[k:k + 6])!r} vs {self.text(gen[k:k + 6])!r}"
                        + (", generated tokens replayed" if same else ", text differs: rendering kept"))
                if same:
                    self.store(gp, rendered, gen)
                    toks = toks[:n] + gen + toks[e + 1:]
                break
        return toks

    def norm(self, toks):
        s = re.sub(r"\s+", "", self.text(toks))
        for m in self.MARKUP:
            s = s.replace(m, "")
        return s

    def store(self, gp, rendered, gen):
        key = (gp, tuple(rendered))
        if key not in self.subs:
            self.gps[gp] += 1
        self.subs[key] = gen
        self.subs.move_to_end(key)
        while len(self.subs) > self.cap:
            (old, _), _ = self.subs.popitem(last=False)
            self.gps[old] -= 1
            if not self.gps[old]:
                del self.gps[old]

    def substitute(self, toks):
        """replaces every rendered turn of a stored pair by its generated tokens"""
        if not self.subs:
            return toks
        ends = [i for i, t in enumerate(toks) if t == self.im_end]
        out, at = [], 0
        for i, t in enumerate(toks):
            if t != self.im_start or i < at:
                continue
            for gp in self.gps:
                if tuple(toks[i:i + len(gp)]) != gp:
                    continue
                start = i + len(gp)
                j = bisect.bisect_left(ends, start)
                if j == len(ends):
                    break
                key = (gp, tuple(toks[start:ends[j] + 1]))
                gen = self.subs.get(key)
                if gen is not None:
                    self.subs.move_to_end(key)
                    out += toks[at:start] + gen
                    at = ends[j] + 1
                break
        return out + toks[at:] if at else toks


class Job:
    """one chat completion; the scheduler thread runs it, the handler thread reads its events"""

    def __init__(self, body, toks, thinking, max_new, budget, rid=None):
        self.body, self.toks, self.max_new, self.budget = body, toks, max_new, budget
        self.cid = "chatcmpl-" + uuid.uuid4().hex  # the response id
        self.rid = rid or self.cid  # the client's X-Request-Id or the response id: in every log line and echoed
        self.tools = body.get("tools") or []
        self.thinking = thinking  # the rendered prompt's mode: the sampling defaults pick their set by it
        self.parser = OutputParser(thinking, self.tools)
        self.dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.tail, self.shown = {"reasoning": "", "content": ""}, 0.0
        self.n_gen, self.n_think, self.finish, self.reused = 0, 0, "length", 0
        self.cycles, self.cycle_toks = 0, 0  # decode cycles it took part in and the tokens they emitted
        self.finish_tok = None
        self.events = queue.Queue()
        self.cancelled = False  # set by the handler when the client is gone
        self.slot = None  # the /status seat it runs in
        self.seq = None  # its engine sequence
        self.snaps = []  # prompt positions (message starts) whose states the engine caches for later requests
        self.feed = toks  # tokens to prefill on admission: the prompt, after a preemption also what it generated
        self.fed = 0  # tokens of feed its sequence holds
        self.stopped = False  # the think stop was forced into the sequence
        self.admitted = 0  # admission order stamp (0: never admitted); the latest admitted job is preempted first
        self.preempts = 0  # times it left for a full KV pool and resumed by recompute
        self.held = 0  # KV blocks it held when last preempted (most stay cached: not room it can count on)
        self.holding = False  # waits for another prompt to prefill their shared prefix (logged once)
        self.seed = None  # sampling seed, fixed at the first admission
        self.decoding = False  # its first token is out (a resume continues the stream and the decode timing)
        self.t0 = self.t1 = time.time()

    def show(self, kind, delta):
        self.tail[kind] = (self.tail[kind] + delta)[-STATUS_TAIL:]
        self.events.put(("delta", kind, delta))

    def wait(self, emit):
        """handler side: passes the deltas to emit until the job ends; returns (finish, calls, usage, timings)"""
        while True:
            ev = self.events.get()
            if ev[0] == "delta":
                if not self.cancelled and emit(ev[1], ev[2]) is False:
                    self.cancelled = True
            elif ev[0] == "done":
                return ev[1]
            else:
                raise ev[1]


class Server:
    def __init__(self, args):
        self.args = args
        t0 = time.time()
        self.eng = Engine(args.lib, args.model, args.ctx, args.draft, args.draft_vocab,
                          {"f16": 0, "q8": 1, "auto": -1}[args.kv], args.slots, args.kv_mem)
        self.template = make_template(self.eng.chat_template())
        self.warm_up()
        self.model_id = os.path.basename(args.model)
        self.warned = set()
        self.queue = queue.Queue()
        self.waiting = collections.deque()  # jobs not admitted yet for lack of a slot or KV blocks, in arrival order
        self.pool = self.pool_stats()  # refreshed by the scheduler thread; handlers only read eng.prefill_progress()
        self.draft = self.draft_stats()  # MTP draft counters since load, refreshed with the pool
        self.pf = None  # (slot, engine call count before it) while prefill() feeds that slot's prompt
        self.pf_on = threading.Event()  # set while self.pf is
        self.pf_seen = None  # (slot, engine call count, position, time the position was first seen)
        self.prefill_owed = 0.0  # decode seconds owed to running requests before the next prompt chunk
        self.pf_round = 0  # prompt chunk rounds; odd: shortest remaining prompt first, even: longest waiting first
        self.kv_ahead = self.eng.kv_ahead()
        self.im_start = self.eng.tokenize("<|im_start|>")
        im_end = self.eng.tokenize("<|im_end|>")
        self.replay = TurnReplay(self.im_start[0], im_end[0], self.detok) \
            if len(self.im_start) == 1 and len(im_end) == 1 else None
        self.think_stop_n = len(self.eng.tokenize(THINK_STOP))
        self.active = {}  # slot -> Job, touched by the scheduler thread only
        self.admissions = 0
        self.preempted = 0  # preemptions since start
        now = time.time()
        self.slot_status = [{"slot": s, "busy": False, "phase": "idle", "action": None, "updated": now}
                            for s in range(args.slots)]
        self.last = None
        threading.Thread(target=self.run, name="scheduler", daemon=True).start()
        threading.Thread(target=self.watch_prefill, name="prefill-watch", daemon=True).start()
        log(f"ready in {time.time() - t0:.1f} s: {self.model_id}, ctx {args.ctx}, {args.draft} drafts, "
            f"{args.slots} slot(s)")

    def warm_up(self):
        """Runs one prefill and one decode cycle before the server listens.

        Loading allocates the weights and the KV cache, but the first kernel launches still need
        device memory of their own. When the context is too large for that, loading succeeds, /health
        answers, and the first request ends the process with a CUDA out-of-memory (CK exits), which
        the client sees as a dropped connection. Warming up here moves that failure to startup.
        The prompt must be at least two tokens: only then does prefill take the batched path, whose
        buffers are allocated on first use; with several slots the cycle is a batched one."""
        toks = self.eng.tokenize("Hello, how are you today?")
        assert len(toks) >= 2
        ids = [self.eng.seq_open([])[0] for _ in range(min(self.args.slots, 2))]
        for i in ids:
            self.eng.append(i, toks)  # no state cached for later requests
        self.eng.step_batch(ids)
        for i in ids:
            self.eng.seq_close(i, keep=False)

    def detok(self, toks):
        return b"".join(self.eng.piece(t) for t in toks).decode(errors="replace")

    def warn_once(self, key, msg):
        if key not in self.warned:
            self.warned.add(key)
            log("warning:", msg)

    def render(self, body):
        kw = dict(body.get("chat_template_kwargs") or {})
        mark = next(chr(c) for c in range(0xE000, 0xF900) if chr(c) not in json.dumps(body, ensure_ascii=False))
        spans = []
        msgs, tools = protect_spans(normalize_messages(body.get("messages") or []), body.get("tools"), mark, spans)
        ctx = dict(messages=msgs, add_generation_prompt=True)
        if body.get("tools"):
            ctx["tools"] = tools
        for k in ("enable_thinking", "preserve_thinking", "reasoning_effort"):
            if k in kw:
                ctx[k] = kw[k]
        try:
            text = self.template.render(**ctx)
        except jinja2.exceptions.TemplateError as e:
            raise BadRequest(f"chat template: {e}")
        thinking = text.endswith("<think>\n")
        toks = []
        for n, part in enumerate(re.split("(" + re.escape(mark) + r"\d+" + re.escape(mark) + ")", text)):
            if not part:
                continue
            if n % 2:
                toks += self.eng.tokenize(spans[int(part.strip(mark))], special=False)
            else:
                toks += self.eng.tokenize(part)
        return toks, thinking

    def prepare(self, body, rid=None):
        """handler side: renders and tokenizes the request (BadRequest before anything is streamed)"""
        toks, thinking = self.render(body)
        if self.replay:
            toks = self.replay.apply(toks, rid, log)
        max_new =int(body.get("max_tokens") or body.get("max_completion_tokens") or body.get("n_predict") or -1)
        budget = body.get("thinking_budget", (body.get("chat_template_kwargs") or {}).get("thinking_budget",
                                                                                         self.args.think_budget))
        budget = int(budget or 0) if thinking else 0
        # a forced think stop enters the sequence without counting as generated tokens
        limit = self.args.ctx - len(toks) - 16 - (self.think_stop_n if budget else 0)
        if limit <= 0:
            raise BadRequest(f"prompt of {len(toks)} tokens does not fit the context of {self.args.ctx}")
        max_new = limit if max_new <= 0 else min(max_new, limit)
        job = Job(body, toks, thinking, max_new, budget, rid)
        job.snaps = self.snap_hints(toks)
        return job

    def snap_hints(self, toks):
        """prompt positions where a later request is likely to diverge: the end of a leading system message (with
        the tools), the start of the last user message and the start of the generation prompt (a later turn may
        render it differently, e.g. without <think>); the engine also caches the state before the last token"""
        if len(self.im_start) != 1:
            return []
        starts = [i for i, t in enumerate(toks[:-1]) if t == self.im_start[0]]
        role = lambda i: self.eng.piece(toks[i + 1])
        hints = set()
        if len(starts) > 1 and role(starts[0]) == b"system":
            hints.add(starts[1])
        users = [i for i in starts if role(i) == b"user"]
        if users:
            hints.add(users[-1])
        if starts:
            hints.add(starts[-1])
        return sorted(h for h in hints if 0 < h < len(toks) - 1)

    def submit(self, job):
        self.queue.put(job)

    def sampling(self, job):
        """sets the job's sequence's sampling from the request"""
        body = job.body
        # mode-aware server defaults: a non-thinking request uses the -instruct
        # flags; a request value always wins over both sets
        d_temp, d_top_p = ((self.args.temperature, self.args.top_p) if job.thinking
                           else (self.args.temperature_instruct, self.args.top_p_instruct))
        temp = float(body["temperature"]) if body.get("temperature") is not None else d_temp
        top_p = float(body["top_p"]) if body.get("top_p") is not None else d_top_p
        top_k = int(body["top_k"]) if body.get("top_k") is not None else self.args.top_k
        min_p = float(body["min_p"]) if body.get("min_p") is not None else self.args.min_p
        if job.seed is None:  # drawn once: the sampler is keyed by (seed, position), so a resume continues the same draws
            seed = body.get("seed")
            job.seed = random.getrandbits(63) if seed is None or int(seed) < 0 else int(seed)
        seed, i = job.seed, job.seq
        if top_k <= 0 or top_k > TOPK_MAX:
            self.warn_once("top_k", f"top_k {top_k} -> {TOPK_MAX} (the megakernel samples from at most {TOPK_MAX} candidates)")
            top_k = TOPK_MAX
        for k in ("presence_penalty", "frequency_penalty"):
            if float(body.get(k) or 0) != 0:
                self.warn_once(k, f"{k} is not supported by the megakernel and is ignored")
        if float(body.get("repetition_penalty") or body.get("repeat_penalty") or 1.0) != 1.0:
            self.warn_once("rep", "repetition_penalty is not supported by the megakernel and is ignored")
        if temp <= 0:
            self.eng.set_sampling(i, 0.0, 1.0, TOPK_MAX, 0.0, seed)
        else:
            self.eng.set_sampling(i, temp, top_p, top_k, min_p, seed)

    # ---- /status

    def track(self, s, **kw):
        """updates slot s's /status snapshot; a new dict each time, so a reader never sees it half-written"""
        self.slot_status[s] = {**self.slot_status[s], **kw, "updated": time.time()}

    def status(self):
        """the most recently updated busy slot at the top level (one-slot layout), every slot under `slots`"""
        slots = [self.prefill_status(st) for st in self.slot_status]
        busy = [st for st in slots if st["busy"]]
        top = max(busy or slots, key=lambda st: st["updated"])
        return {**top, "busy": bool(busy), "last": self.last, "queued": self.queue.qsize() + len(self.waiting),
                "kv_pool": self.pool, "preempted": self.preempted, "draft": self.draft, "slots": slots}

    def prefill_status(self, st):
        """slot snapshot st with its live prompt progress while it reads its prompt"""
        if not st["busy"] or st["phase"] not in ("prefill", "resume") or st.get("prefill_from") is None:
            return st
        pos, at = st.get("prefilled") or st["prefill_from"], st.get("prefilled_t") or st["prefill_t"]
        pf, seen = self.pf, self.pf_seen
        if pf and seen and pf[0] == seen[0] == st["slot"] and seen[1] != pf[1] and seen[2] > pos:
            pos, at = seen[2], seen[3]
        elapsed = at - st["prefill_t"]
        return {**st, "prefilled": pos,
                "prefill_tok_s": (pos - st["prefill_from"]) / elapsed if elapsed > 0 else 0.0}

    def watch_prefill(self):
        """while prefill() runs, notes when the engine's position moves (per 2,048-token chunk), so the prompt rate
        is tokens over the time they took, not over the time since (which saws between chunks)"""
        while True:
            self.pf_on.wait()
            pf = self.pf
            if pf:
                calls, at = self.eng.prefill_progress()
                seen = self.pf_seen
                # a count still at pf[1] is the previous call's value: the engine has not entered this one yet
                if calls != pf[1] and (not seen or seen[:3] != (pf[0], calls, at)):
                    self.pf_seen = (pf[0], calls, at, time.time())
            time.sleep(0.05)

    @staticmethod
    def rate(n_gen, elapsed):
        """decode tok/s: the first token comes out of the prefill, so it is not part of the decode time"""
        return (n_gen - 1) / elapsed if n_gen > 1 and elapsed > 0 else 0.0

    def live(self, job, elapsed):
        """/status fields while tokens are generated: the phase and the tail of what it is producing"""
        if job.parser.mode == "tools":
            phase, text = "tool_call", job.parser.tool_text[-STATUS_TAIL:]
        elif job.parser.mode == "reasoning":
            phase, text = "thinking", job.tail["reasoning"]
        else:
            phase, text = "writing", job.tail["content"]
        return {"phase": phase, "gen": job.n_gen, "tok_s": self.rate(job.n_gen, elapsed),
                "action": {"kind": phase, "text": text}}

    # ---- scheduler thread: the only one that drives the engine

    def run(self):
        while True:
            if not self.active and not self.waiting:
                self.waiting.append(self.queue.get())
            self.schedule()

    def schedule(self):
        """one scheduler pass: admissions, prompt feeding, one decode cycle"""
        # in arrival order, while a slot is free and the KV pool covers the job's feed (alone it always fits)
        held = []
        while len(self.active) < self.args.slots:
            if not self.waiting:
                try:
                    self.waiting.append(self.queue.get_nowait())
                except queue.Empty:
                    break
            job = self.waiting[0]
            if not job.cancelled and self.held(job):
                held.append(self.waiting.popleft())
                continue
            if self.active and not job.cancelled and not self.fits(job):
                break
            self.admit(self.waiting.popleft())
        self.waiting.extendleft(reversed(held))
        filling = [j for j in self.active.values() if j.fed < len(j.feed)]
        if len(filling) == len(self.active):  # nothing decodes: the prompts go in whole
            for job in filling:
                self.prefill(job)
        elif filling and self.prefill_owed <= 0:
            # one chunk's worth, then decode cycles until the decoders got their share back.  Rounds alternate between
            # shortest remaining prompts first (short prompts arriving together start together) and longest waiting
            # first, so a long prompt keeps half the prefill time while other conversations keep sending short turns
            self.pf_round += 1
            order = (lambda j: (len(j.feed) - j.fed, j.admitted)) if self.pf_round % 2 else (lambda j: (j.t0, j.admitted))
            budget = self.args.prefill_chunk
            t = time.time()
            for job in sorted(filling, key=order):
                rest = len(job.feed) - job.fed
                if rest > budget and budget < self.args.prefill_chunk:
                    break
                self.prefill(job, budget)
                budget -= rest
                if budget <= 0:
                    break
            share = self.args.prefill_share
            self.prefill_owed += (time.time() - t) * (1 - share) / share
        t = time.time()
        self.cycle()
        self.prefill_owed = max(0.0, self.prefill_owed - (time.time() - t))
        self.pool = self.pool_stats()
        self.draft = self.draft_stats()

    def admit(self, job):
        """opens the job's sequence from the longest cached prefix; the rest of its feed is prefilled by prefill()"""
        if job.cancelled:
            job.events.put(("done", ("length", [], {}, {})))
            return
        s = job.slot = min(set(range(self.args.slots)) - set(self.active))
        self.active[s] = job
        first = not job.admitted
        self.admissions += 1
        job.admitted = self.admissions
        if first:
            job.t0 = time.time()
        # every field, so a resume on another seat shows none of its previous job's
        self.track(s, busy=True, id=job.rid, phase="prefill" if first else "resume", gen=job.n_gen,
                   tok_s=self.rate(job.n_gen, time.time() - job.t1) if job.decoding else 0.0, ctx=len(job.toks),
                   started=job.t0, prefill_s=job.t1 - job.t0 if job.decoding else None,
                   prefill_n=len(job.toks) - job.reused if job.decoding else None, prefilled=None, preempts=job.preempts)
        try:
            job.seq, job.fed = self.eng.seq_open(job.feed)
            if first:
                job.reused = job.fed
            self.track(s, prefilled=job.fed, prefilled_t=None, prefill_from=job.fed, prefill_t=time.time())
            self.sampling(job)
            log(f"slot {s} [{job.rid}]: {'prompt' if first else 'resume'} {len(job.feed)} tok, {job.fed} cached")
        except Exception as e:
            self.fail(job, e)

    def held(self, job):
        """a prompt that shares a snapshot point with a prompt still being prefilled waits until that point is
        prefilled, then starts from its snapshot instead of prefilling the shared part a second time"""
        for other in self.active.values():
            if other.fed >= len(other.feed):
                continue
            for h in [*other.snaps, len(other.feed) - 1]:
                if other.fed < h < len(job.feed) and job.feed[:h] == other.feed[:h]:
                    if not job.holding:
                        job.holding = True
                        log(f"[{job.rid}]: waits for the shared prefix of {h} tok being prefilled in slot {other.slot}")
                    return True
        return False

    def pool_stats(self):
        """the engine's pool counts plus the blocks running requests hold and when they were read"""
        p = self.eng.pool_stats()
        used = p["total_blocks"] - p["free_blocks"] - p["evictable_blocks"]
        return {**p, "used_blocks": used, "updated": time.time()}

    def draft_stats(self):
        """the engine's draft counters plus the share of verified drafts accepted"""
        d = self.eng.draft_stats()
        return {**d, "accept_rate": d["accepted"] / d["drafted"] if d["drafted"] else 0.0}

    def fits(self, job):
        """optimistic admission: free blocks plus the cached ones the engine can evict cover the job's feed, one
        cycle's lookahead and one block of growth per running job.  A preempted job does not count the blocks it
        left cached: re-admitting it by evicting its own prefix would only preempt it again."""
        self.pool = self.pool_stats()
        avail = self.pool["free_blocks"] + self.pool["evictable_blocks"] - job.held
        need = -(-(len(job.feed) + self.kv_ahead) // self.pool["block_tokens"]) + len(self.active)
        return need <= avail

    def preempt(self, job):
        """the KV pool is full: the job leaves (its sequence stays cached as far as snapshots allow) and resumes by
        recompute from the head of the queue, its stream, parser and counters continuing where they stopped"""
        s = job.slot
        del self.active[s]
        try:
            held = self.eng.seq_blocks(job.seq)
            if job.fed >= len(job.feed):  # decoding: what it generated becomes part of what the resume prefills
                feed = self.eng.sequence(job.seq)
            else:
                feed = job.feed
            self.eng.seq_close(job.seq, keep=True)
        except Exception as e:
            self.fail(job, e)  # closes the sequence (no lane leak)
            return
        job.seq, job.feed, job.held = None, feed, held
        job.preempts += 1
        self.preempted += 1
        self.track(s, busy=False, phase="idle")
        self.waiting.appendleft(job)
        log(f"slot {s} [{job.rid}]: KV pool full, preempted after {job.n_gen} tok (preemption {job.preempts}, "
            f"{len(job.feed)} tok to resume, {len(self.waiting)} waiting)")

    def grow(self, job, call):
        """runs an engine call that grows job's sequence; on a full pool (cached blocks already evicted) preempts the
        most recently admitted job and retries.  None: job itself was preempted.  PoolFull: job is the only one."""
        while True:
            try:
                return call()
            except PoolFull:
                if len(self.active) <= 1:
                    raise
                victim = max(self.active.values(), key=lambda j: j.admitted)
                self.preempt(victim)
                if victim is job:
                    return None

    def prefill(self, job, chunk=None):
        """feeds the job's feed, at most chunk tokens of it while other jobs are decoding (they stall one chunk at most)"""
        if self.active.get(job.slot) is not job:  # preempted earlier in this loop
            return
        self.pf = (job.slot, self.eng.prefill_progress()[0])
        self.pf_on.set()
        try:
            rest = job.feed[job.fed:]
            if chunk is not None and len(rest) > chunk:
                # an intermediate chunk ends with a generated token, which the next chunk's position overwrites
                if self.grow(job, lambda: self.eng.append(job.seq, rest[:chunk], job.snaps)) is None:
                    return
                job.fed += chunk
                self.eng.truncate(job.seq, job.fed)
                self.track(job.slot, prefilled=job.fed, prefilled_t=time.time())
                return
            tok = self.grow(job, lambda: self.eng.prefill(job.seq, rest, job.snaps))
            if tok is None:
                return
            job.fed = len(job.feed)
            self.track(job.slot, prefilled=job.fed, prefilled_t=time.time())
            if job.decoding and self.stop_due(job):
                # preempted when its think stop was due: an unpreempted run forces the stop right after the token
                # the feed ends with, so the resume's own token is dropped and the next cycle forces the stop
                self.eng.truncate(job.seq, len(job.feed))
                return
        except PoolFull:
            self.out_of_pool(job)
            return
        except Exception as e:
            self.fail(job, e)
            return
        finally:
            self.pf = None
            self.pf_on.clear()
        if not job.decoding:
            job.decoding = True
            job.t1 = time.time()
            self.track(job.slot, prefill_s=job.t1 - job.t0, prefill_n=len(job.toks) - job.reused)
        self.consume(job, [tok])

    def cycle(self):
        """one decode cycle of every active job: forced thinking stops one by one, the rest in one batched step"""
        stepping = []
        for s, job in list(self.active.items()):
            if self.active.get(s) is not job:  # preempted by an earlier job's think stop
                continue
            if job.cancelled:
                self.complete(job)
            elif job.fed < len(job.feed):
                continue
            elif self.stop_due(job):
                try:
                    tok = self.grow(job, lambda job=job: self.eng.append(job.seq, self.eng.tokenize(THINK_STOP)))
                except PoolFull:
                    self.out_of_pool(job)
                    continue
                except Exception as e:
                    self.fail(job, e)
                    continue
                if tok is None:  # preempted: the stop is forced after the resume
                    continue
                for kind, delta in job.parser.feed(THINK_STOP):
                    job.show(kind, delta)
                log(f"slot {s} [{job.rid}]: thinking budget of {job.budget} tokens reached")
                job.budget, job.stopped = 0, True
                self.consume(job, [tok])
            else:
                stepping.append(s)
        for s in [s for s in stepping if s in self.active]:
            # a cycle needs its lookahead inside the context; the engine would fail the whole batch otherwise
            job = self.active[s]
            try:
                full = self.eng.seq_len(job.seq) - 1 + self.kv_ahead > self.args.ctx
            except Exception as e:
                self.fail(job, e)
                continue
            if full:
                log(f"slot {s} [{job.rid}]: context full, finishing after {job.n_gen} tok")
                job.finish = "length"
                self.complete(job)
        stepping = [s for s in stepping if s in self.active]
        while stepping:
            try:
                outs = self.eng.step_batch([self.active[s].seq for s in stepping])
                break
            except PoolFull:
                if len(self.active) <= 1:
                    self.out_of_pool(self.active[stepping[0]])
                    return
                self.preempt(max(self.active.values(), key=lambda j: j.admitted))
                stepping = [s for s in stepping if s in self.active]
            except Exception as e:
                for s in stepping:
                    self.fail(self.active[s], e)
                return
        else:
            return
        for s, toks in zip(stepping, outs):
            job = self.active[s]
            job.cycles += 1
            job.cycle_toks += len(toks)
            self.consume(job, toks)

    @staticmethod
    def stop_due(job):
        """the thinking budget is used up and the stop not forced yet"""
        return job.budget and job.n_think >= job.budget and job.parser.mode == "reasoning"

    def out_of_pool(self, job):
        """the job runs alone and still cannot grow: it ends as if max_tokens were reached"""
        log(f"slot {job.slot} [{job.rid}]: KV pool full with no other job to preempt, finishing after {job.n_gen} tok")
        job.finish = "length"
        self.complete(job)

    def consume(self, job, toks):
        """feeds a cycle's tokens to the job; tokens after its end stay in its sequence (never matched by a later
        prompt that differs there)"""
        done = False
        for k, t in enumerate(toks):
            if job.n_gen >= job.max_new:
                break
            job.n_gen += 1
            if job.parser.mode == "reasoning":
                job.n_think += 1
            if self.eng.is_eog(t):
                job.finish, done = "stop", True
                job.finish_tok = t
                break
            for kind, delta in job.parser.feed(job.dec.decode(self.eng.piece(t))):
                job.show(kind, delta)
            if k < len(toks) - 1 and self.stop_due(job):
                # the stop goes right after the budget's last token, not at the cycle's end, so where it lands does
                # not depend on how tokens fell into cycles (batch size, a preemption's resume)
                try:
                    self.eng.truncate(job.seq, self.eng.seq_len(job.seq) - (len(toks) - 1 - k))
                    break
                except Exception as e:
                    log(f"slot {job.slot} [{job.rid}]: think stop after the cycle's last token:", repr(e))
        if time.time() - job.shown >= 0.25:
            job.shown = time.time()
            self.track(job.slot, **self.live(job, job.shown - job.t1))
        if done or job.n_gen >= job.max_new:
            self.complete(job)

    def release(self, job):
        """the job's tokens and state stay cached for its conversation's next request"""
        del self.active[job.slot]
        if job.seq is not None:
            try:
                self.eng.seq_close(job.seq, keep=True)
            except Exception as e:
                log(f"slot {job.slot} [{job.rid}]: close failed:", repr(e))
                try:
                    self.eng.seq_close(job.seq, keep=False)
                except Exception as e2:
                    log(f"slot {job.slot} [{job.rid}]: close without caching failed:", repr(e2))
            job.seq = None

    def complete(self, job):
        for kind, delta in job.parser.feed(job.dec.decode(b"", final=True), final=True):
            job.show(kind, delta)
        calls = parse_tool_calls(job.parser.tool_text, job.tools) if job.parser.tool_text else []
        if calls:
            job.finish = "tool_calls"
        t2 = time.time()
        n_gen, n_prompt = job.n_gen, len(job.toks)
        end = "cancelled" if job.cancelled else job.finish
        tail = ""
        if job.finish_tok is not None:
            tail = f" tok={job.finish_tok} {self.eng.piece(job.finish_tok).decode(errors='replace')!r}"
        log(f"slot {job.slot} [{job.rid}]: prompt {n_prompt} tok ({job.reused} reused) in {job.t1 - job.t0:.2f} s, "
            f"{n_gen} tok in {t2 - job.t1:.2f} s ({self.rate(n_gen, t2 - job.t1):.1f} tok/s), "
            f"{end}{tail}" + (f", {job.cycle_toks / job.cycles:.2f} tok/cycle" if job.cycles else "")
            + (f", preempted {job.preempts}x" if job.preempts else ""))
        usage = {"prompt_tokens": n_prompt, "completion_tokens": n_gen, "total_tokens": n_prompt + n_gen,
                 "prompt_tokens_details": {"cached_tokens": job.reused}}
        timings = {"prompt_n": n_prompt - job.reused, "prompt_ms": (job.t1 - job.t0) * 1e3, "predicted_n": n_gen,
                   "predicted_ms": (t2 - job.t1) * 1e3, "predicted_per_second": self.rate(n_gen, t2 - job.t1)}
        action = self.live(job, t2 - job.t1)["action"]
        if calls:
            action = {"kind": "tool_call", "calls": [
                {"name": c["function"]["name"],
                 "arguments": {k: v[:STATUS_TAIL] if isinstance(v, str) else v
                               for k, v in json.loads(c["function"]["arguments"]).items()}} for c in calls]}
        if n_gen > 1:  # a 1-token request (warm-up, title) has no decode speed: keep the last real job's figures
            self.last = {"slot": job.slot, "id": job.rid, "tok_s": timings["predicted_per_second"], "gen": n_gen,
                         "ctx": n_prompt,
                         "reused": job.reused, "prefill_s": job.t1 - job.t0, "finish": end, "preempts": job.preempts,
                         "at": t2}
        self.track(job.slot, busy=False, phase="idle", action=action)
        if self.replay and job.finish_tok is not None and not job.cancelled and job.seq is not None:
            try:
                seq = self.eng.sequence(job.seq)
                self.replay.finished(job.rid, job.toks, seq[len(job.toks):seq.index(job.finish_tok, len(job.toks)) + 1])
            except Exception as e:
                log(f"slot {job.slot} [{job.rid}]: turn not kept for replay:", repr(e))
        self.release(job)
        job.events.put(("done", (job.finish, calls, usage, timings)))

    def fail(self, job, e):
        """engine error: the job reports it, its sequence is dropped (what other holders share stays cached)"""
        log(f"slot {job.slot} [{job.rid}] error:", repr(e))
        s = job.slot
        self.active.pop(s, None)
        if job.seq is not None:
            try:
                self.eng.seq_close(job.seq, keep=False)
            except Exception as e2:
                log(f"slot {s} [{job.rid}]: closing the sequence after an error failed:", repr(e2))
            job.seq = None
        self.track(s, busy=False, phase="idle")
        job.events.put(("error", e))


def make_handler(srv):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _json(self, code, obj, rid=None):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            if rid:
                self.send_header("X-Request-Id", rid)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/health", "/v1/health"):
                self._json(200, {"status": "ok"})
            elif path in ("/v1/models", "/models"):
                self._json(200, {"object": "list", "data": [{"id": srv.model_id, "object": "model", "owned_by": "openjet"}]})
            elif path == "/status":
                self._json(200, srv.status())
            elif path == "/props":
                self._json(200, {"default_generation_settings": {"n_ctx": srv.args.ctx}, "total_slots": srv.args.slots})
            else:
                self._json(404, {"error": {"message": "not found", "type": "not_found_error"}})

        def do_POST(self):
            path = self.path.split("?")[0]
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else b""
            if path not in ("/v1/chat/completions", "/chat/completions"):
                self._json(404 if not path.startswith("/slots") else 501,
                           {"error": {"message": f"{path} is not supported by the megakernel server", "type": "not_supported"}})
                return
            try:
                body = json.loads(raw or b"{}")
            except json.JSONDecodeError:
                self._json(400, {"error": {"message": "invalid JSON", "type": "invalid_request_error"}})
                return
            # only a plain token is taken: it is logged and echoed in a header
            rid = self.headers.get("X-Request-Id", "").strip()
            rid = rid if re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", rid) else None
            try:
                job = srv.prepare(body, rid)
            except BadRequest as e:
                self._json(400, {"error": {"message": str(e), "type": "invalid_request_error"}}, rid)
                return
            srv.submit(job)
            try:
                if body.get("stream"):
                    self._stream(job)
                else:
                    self._complete(job)
            except (BrokenPipeError, ConnectionResetError):
                job.cancelled = True
            except Exception as e:  # engine failure: report it, keep serving
                try:
                    self._json(500, {"error": {"message": str(e), "type": "server_error"}}, job.rid)
                except Exception:
                    pass

        def _complete(self, job):
            parts = {"reasoning": [], "content": []}
            finish, calls, usage, timings = job.wait(lambda k, t: parts[k].append(t))
            msg = {"role": "assistant", "content": "".join(parts["content"]) or (None if calls else "")}
            if parts["reasoning"]:
                msg["reasoning_content"] = "".join(parts["reasoning"]).strip()
            if calls:
                msg["tool_calls"] = calls
            self._json(200, {"id": job.cid, "object": "chat.completion", "created": int(time.time()),
                             "model": srv.model_id, "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                             "usage": usage, "timings": timings}, job.rid)

        def _stream(self, job):
            cid, created = job.cid, int(time.time())
            started = [False]

            def chunk(delta, finish=None, usage=None, timings=None):
                obj = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": srv.model_id,
                       "choices": [] if delta is None else [{"index": 0, "delta": delta, "finish_reason": finish}]}
                if usage is not None:
                    obj["usage"] = usage
                if timings is not None:
                    obj["timings"] = timings
                data = b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n"
                self.wfile.write(b"%x\r\n" % len(data) + data + b"\r\n")
                self.wfile.flush()

            def head():
                if not started[0]:
                    started[0] = True
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Transfer-Encoding", "chunked")
                    self.send_header("X-Request-Id", job.rid)
                    self.end_headers()
                    chunk({"role": "assistant", "content": None})

            def emit(kind, text):
                try:
                    head()
                    chunk({"reasoning_content": text} if kind == "reasoning" else {"content": text})
                except (BrokenPipeError, ConnectionResetError):
                    return False
                return True

            finish, calls, usage, timings = job.wait(emit)
            head()
            for i, c in enumerate(calls):
                chunk({"tool_calls": [{"index": i, **c}]})
            chunk({}, finish=finish, timings=timings)
            if (job.body.get("stream_options") or {}).get("include_usage"):
                chunk(None, usage=usage)
            data = b"data: [DONE]\n\n"
            self.wfile.write(b"%x\r\n" % len(data) + data + b"\r\n0\r\n\r\n")
            self.wfile.flush()

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("-c", "--ctx", type=int, default=32768)
    ap.add_argument("-d", "--draft", type=int, default=4)
    ap.add_argument("--draft-vocab", default=os.path.join(os.path.dirname(HERE), "draft_vocab_32k.bin"))
    ap.add_argument("--lib", default=os.path.join(os.path.dirname(HERE), "libgeezuz.so"),
                    help="engine library; libgeezuz_tp2.so runs tensor-parallel on two GPUs (MK_DEVICES=a,b)")
    ap.add_argument("--kv", choices=["auto", "f16", "q8"], default="auto",
                    help="KV cache format (auto: q8_0 only when the fp16 pool holds less than one -c sequence)")
    ap.add_argument("--kv-mem", type=int, default=0,
                    help="KV pool MB per card (0: the free memory after the slots and a reserve); -c stays the "
                         "per-request maximum")
    ap.add_argument("--slots", type=int, default=1,
                    help="requests decoded at once (1..8), their KV taken from the shared pool as they grow; with the "
                         "TP2 library they share one batched cycle")
    ap.add_argument("--prefill-chunk", type=int, default=2048,
                    help="prompt tokens fed at a time while other requests decode (they stall one chunk at most; "
                         "smaller chunks re-read the cached context more often, which slows deep prompts)")
    ap.add_argument("--prefill-share", type=float, default=0.25,
                    help="share of the time prompts are prefilled while other requests decode (0..1]: the decoders "
                         "keep the rest of their speed, a prompt takes 1/share times as long")
    ap.add_argument("--think-budget", type=int, default=0,
                    help="reasoning tokens per request before </think> is forced (0: unlimited); "
                         "a request's thinking_budget overrides it")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="default temperature for thinking requests that send none "
                         "(Qwen3.8-27B model-card rec; <= 0: greedy)")
    ap.add_argument("--top-p", type=float, default=0.95,
                    help="default top_p for thinking requests that send none (model-card rec)")
    ap.add_argument("--top-k", type=int, default=20,
                    help="default top_k for requests that send none (model-card rec; the engine samples "
                         "from at most 32)")
    ap.add_argument("--min-p", type=float, default=0.0,
                    help="default min_p for requests that send none (model-card rec: off)")
    ap.add_argument("--temperature-instruct", type=float, default=0.7,
                    help="default temperature for non-thinking requests that send none (model-card rec)")
    ap.add_argument("--top-p-instruct", type=float, default=0.8,
                    help="default top_p for non-thinking requests that send none (model-card rec)")
    args = ap.parse_args()
    if not 0 < args.prefill_share <= 1:
        ap.error("--prefill-share must be in (0, 1]")
    srv = Server(args)
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(srv))
    httpd.daemon_threads = True
    log(f"listening on http://{args.host}:{args.port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
