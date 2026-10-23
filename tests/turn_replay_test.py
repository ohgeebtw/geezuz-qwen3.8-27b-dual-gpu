"""TurnReplay: a generated turn whose rendering differs only in whitespace or normalized markup is replayed in the
next prompt and every later one of its conversation; a turn the client changed keeps its rendering."""
import os, sys, types, importlib.util

j = types.ModuleType("jinja2"); j.exceptions = types.SimpleNamespace(TemplateError=Exception)
sb = types.ModuleType("jinja2.sandbox"); sb.ImmutableSandboxedEnvironment = object
sys.modules["jinja2"] = j; sys.modules["jinja2.sandbox"] = sb
spec = importlib.util.spec_from_file_location("mk", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server", "mk_server.py"))
mk = importlib.util.module_from_spec(spec); spec.loader.exec_module(mk)

V = {1: "<|im_start|>", 2: "<|im_end|>", 3: "assistant\n", 4: "<think>\n", 5: "user\n", 6: "</think>\n\n", 7: "\n",
     8: "\n\n", 9: "<tool_call>\n", 99: "<|endoftext|>"}
text = lambda toks: "".join(V.get(t, f"w{t} ") for t in toks)
GP = [1, 3, 4]  # generation prompt: <|im_start|>assistant\n<think>\n
logs = []


def user(w):
    return [1, 5, w, 2, 7]


def check(cond, msg):
    if not cond:
        raise SystemExit("FAIL: " + str(msg))


r = mk.TurnReplay(1, 2, text)
P = [1, 5, 10, 11, 2, 7] + [20 + k for k in range(30)] + GP
G = [12, 8, 6, 9, 13, 2]   # model: two newlines before </think>, a tool call
R1 = [12, 7, 6, 9, 13, 2]  # template: one newline
r.finished("a1", P, G)
Q = P + R1 + [7] + user(14) + GP
out = r.apply(Q, "a2", logs.append)
check(out == P + G + [7] + user(14) + GP, "next prompt replays the generated turn")
check(len(logs) == 1 and "replayed" in logs[0], logs)

# the next turn: the template renders the first turn the same way again, plus a second generated turn
G2 = [15, 7, 6, 16, 99]  # ended with another end-of-generation token
r.finished("a2", out, G2)
Q2 = Q + [15, 7, 6, 16, 2] + [7] + user(17) + GP
out2 = r.apply(Q2, "a3", logs.append)
check(out2 == P + G + [7] + user(14) + GP + [15, 7, 6, 16, 2] + [7] + user(17) + GP,
      "first turn replayed from the stored pair, second matches its rendering")
check(len(logs) == 1, "a turn matching its rendering (end token normalized) logs nothing")

# a client that edits the turn's text keeps its rendering
r.finished("b1", P, G)
edited = [12, 7, 6, 9, 18, 2]
out3 = r.apply(P + edited + [7] + user(14) + GP, "b2", logs.append)
check(out3 == P + edited + [7] + user(14) + GP, "edited turn not replayed")
check("rendering kept" in logs[-1], logs[-1])

# unrelated prompts pass unchanged, and a pending turn is met only once
unrelated = [1, 5, 40, 2, 7] + GP
check(r.apply(unrelated, "c", logs.append) == unrelated, "unrelated prompt changed")
r.finished("d1", P, G)
r.apply(Q, "d2", logs.append)
n = len(logs)
r.apply(Q, "d3", logs.append)
check(len(logs) == n, "a pending turn matched twice")
print("turn replay: ok")
