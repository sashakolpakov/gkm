"""Counterexample-guided SMT synthesis and the identical data-only LLM task.

The SMT variables describe a general forward control graph, not a traversal
sketch. Recursion is the only backward edge. Width/output effects of acquired
fragments are exact summaries; no tree grammar or reference program is used.
"""
from __future__ import annotations

import itertools
import resource
import sys
import time

import z3

from modular_machine import Machine as FragmentMachine
from recursive_machine import Machine, canonical, node


def score(library, name, pairs, bindings, *, trace_limit=0, disabled=()):
    machine = Machine(library)
    records = []
    for source, expected in pairs:
        result = machine.run(name, source, bindings, trace_limit=trace_limit, disabled=disabled)
        records.append(dict(source=list(source), expected=list(expected),
                            exact=result["ok"] and result["output"] == list(expected), **result))
    return {"exact": all(r["exact"] for r in records), "total": len(records),
            "correct": sum(r["exact"] for r in records), "records": records}


def brief(result):
    return {k: result[k] for k in ("exact", "correct", "total")}


def prompt(library, pairs, bits, maximum, bindings, feedback=None):
    return {
        "instruction": "Infer a prefix procedure from input/output examples. Return only the JSON program. "
                       "You may use recursion and procedure parameters. Reuse compatible retained procedures "
                       "when useful. You may not introduce new token operations or modify retained bodies.",
        "library_hash": library.digest,
        "fragments": {name: {k: c[k] for k in ("width", "program", "hash")}
                      for name, c in library.base.cells.items()},
        "retained_prefix_procedures": library.cells,
        "interface": {"context_bits": bits, "initial_context": 0,
                      "parameters": list(bindings), "parameter_width": 2,
                      "max_control_nodes": maximum},
        "semantics": {
            "tokens": "Opaque integers. Only equality may inspect identities. No arithmetic on tokens.",
            "word_actions": {"0": "advance one token", "1": "emit current token",
                             "10/11": "save token to local register 0/1", "30/31": "emit local register 0/1"},
            "nodes": "Root is node 0. Every node has op,arg,cell,a,b,bindings. Unused fields are null or []. "
                     "Ordinary a/b edges must point to later nodes. A call returns before continuing at a.",
            "eq": "If the next two input tokens exist and are equal go to a, else b.",
            "bit": "If bit arg of this invocation's context is set go to a, else b.",
            "call": "Execute acquired fragment cell on its fixed input window, then go to a.",
            "param": "Execute width-two fragment supplied in parameter slot arg (0 or 1), then go to a.",
            "recur": "Recursively call THIS procedure at node 0, with context XOR arg and unchanged parameters. "
                     "On return keep the advanced cursor/output, restore the caller's context, then go to a. "
                     "Allowed only after this invocation has consumed at least one input token.",
            "invoke": "Call retained prefix procedure cell with initial context arg, supplying the two "
                      "width-two fragment names in bindings. Restore the caller's context/parameters on return; go to a.",
            "ret": "Return immediately, possibly leaving input for the caller. The root must consume ALL input.",
            "limits": "100000 execution steps, 8192 output tokens, 160 stack frames. No host parser or subtree operation."},
        "training": [[list(s), list(t)] for s, t in pairs],
        "feedback": feedback,
    }


def rss_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2 if sys.platform == "darwin" else 1024)


class Deadline(Exception):
    pass


class Synthesizer:
    """One shared unknown CFG, constrained by bounded executions of examples."""
    OPS = ("ret", "eq", "bit", "recur", "param", "call")

    def __init__(self, library, bits, maximum, bindings, deadline):
        self.library, self.bits, self.n = library, bits, maximum
        self.bindings, self.deadline = bindings, deadline
        self.names = list(library.base.cells)
        self.solver = z3.SolverFor("QF_BV")
        self.solver.set(random_seed=0)
        self.sort = z3.BitVecSort(8)
        self.op, self.arg, self.a, self.b = [[z3.BitVec(f"{key}_{i}", 8) for i in range(maximum)]
                                           for key in ("op", "arg", "a", "b")]
        self.examples = 0
        self.transitions = 0
        for i in range(maximum):
            op, arg, a, b = [v[i] for v in (self.op, self.arg, self.a, self.b)]
            self.solver.add(op >= 0, op < len(self.OPS))
            self.solver.add(z3.Implies(op == 0, z3.And(arg == 0, a == 0, b == 0)))
            self.solver.add(z3.Implies(op != 0, z3.And(a > i, a < maximum)))
            self.solver.add(z3.Implies(z3.Or(op == 1, op == 2), z3.And(b > i, b < maximum)))
            self.solver.add(z3.Implies(z3.And(op != 1, op != 2), b == 0))
            self.solver.add(z3.Implies(op == 1, arg == 0))
            for kind, bound in ((2, bits), (3, 2 ** bits), (4, 2), (5, len(self.names))):
                self.solver.add(z3.Implies(op == kind, z3.And(arg >= 0, arg < bound)))
        self.effects = []
        machine = FragmentMachine(library.base)
        for name in self.names:
            width = library.base.cells[name]["width"]
            result = machine.run(name, list(range(width)))
            if not result["ok"]:
                raise ValueError("invalid acquired fragment summary")
            self.effects.append((width, tuple(result["output"])))

    def check_time(self):
        if time.monotonic() >= self.deadline:
            raise Deadline("wall-time limit")
        if rss_mib() > 640:
            raise Deadline("mechanical process memory limit")

    def add_example(self, source, expected):
        self.check_time()
        tag = f"e{self.examples}"
        self.examples += 1
        length, outlength = len(source), len(expected)
        if max(length, outlength) >= 120:
            raise ValueError("SMT training encoding supports fewer than 120 input/output tokens")
        # Equality-only alpha renaming: no token identities or arithmetic are
        # introduced. Eight-bit bounded indices avoid nonlinear integer/array
        # reasoning while leaving room for every possible fragment increment.
        identities = {v: i + 1 for i, v in enumerate(dict.fromkeys((*source, *expected)))}
        source, expected = [identities[v] for v in source], [identities[v] for v in expected]
        def pick(values, index):
            result = z3.BitVecVal(0, 8)
            for i, value in enumerate(values):
                if isinstance(value, int):
                    value = z3.BitVecVal(value, 8)
                result = z3.If(index == i, value, result)
            return result
        widths = [w for w, _ in self.effects]
        sizes = [len(mapping) for _, mapping in self.effects]
        # This is a bounded encoding, not a claim of complete recursive synthesis.
        # Search short executions first. This explicit horizon excludes some
        # legal programs; admission still uses the full graph interpreter.
        bound = 4 * (length + 1)
        old = None
        for t in range(bound + 1):
            if t % 16 == 0:
                self.check_time()
            pc, cur, out, ctx, sp, start = z3.BitVecs(" ".join(f"{tag}_{k}_{t}" for k in ("pc", "cur", "out", "ctx", "sp", "start")), 8)
            retpc, retctx, retstart = [[z3.BitVec(f"{tag}_{k}_{t}_{j}", 8) for j in range(length + 1)]
                                       for k in ("retpc", "retctx", "retstart")]
            self.solver.add(pc >= 0, pc <= self.n, cur >= 0, cur <= length,
                            out >= 0, out <= outlength, ctx >= 0, ctx < 2 ** self.bits,
                            sp >= 0, sp <= length, start >= 0, start <= cur)
            current = (pc, cur, out, ctx, sp, start, retpc, retctx, retstart)
            if old is None:
                self.solver.add(pc == 0, cur == 0, out == 0, ctx == 0, sp == 0, start == 0)
                for stack in (retpc, retctx, retstart):
                    self.solver.add(*[value == 0 for value in stack])
                old = current
                continue
            p, c, q, k, d, s, rp, rc, rs = old
            op, arg, a, b = [pick(v, p) for v in (self.op, self.arg, self.a, self.b)]
            active = p != self.n
            isret, isrecur = z3.And(active, op == 0), z3.And(active, op == 3)
            emit = z3.And(active, z3.Or(op == 4, op == 5))
            which = z3.If(op == 4, z3.If(arg == 0, z3.BitVecVal(self.names.index(self.bindings[0]), 8),
                                        z3.BitVecVal(self.names.index(self.bindings[1]), 8)), arg)
            conditions = []
            for f, (width, mapping) in enumerate(self.effects):
                conditions.append(z3.And(which == f, c + width <= length, q + len(mapping) <= outlength,
                                         *[pick(source, c + off) == pick(expected, q + j)
                                           for j, off in enumerate(mapping)]))
            self.solver.add(z3.Implies(emit, z3.Or(*conditions)))
            self.solver.add(z3.Implies(isrecur, z3.And(c > s, d < length)))
            same = z3.And(c + 1 < length, pick(source, c) == pick(source, c + 1))
            bit = (z3.LShR(k, arg) & 1) == 1
            nextpc = z3.If(op == 0, z3.If(d == 0, self.n, pick(rp, d - 1)),
                          z3.If(op == 1, z3.If(same, a, b), z3.If(op == 2, z3.If(bit, a, b),
                                z3.If(op == 3, 0, a))))
            self.solver.add(pc == z3.If(active, nextpc, p),
                            cur == z3.If(emit, c + pick(widths, which), c),
                            out == z3.If(emit, q + pick(sizes, which), q),
                            sp == z3.If(isrecur, d + 1, z3.If(z3.And(isret, d > 0), d - 1, d)),
                            ctx == z3.If(isrecur, k ^ arg,
                                         z3.If(z3.And(isret, d > 0), pick(rc, d - 1), k)),
                            start == z3.If(isrecur, c, z3.If(z3.And(isret, d > 0), pick(rs, d - 1), s)))
            for new_stack, old_stack, value in ((retpc, rp, a), (retctx, rc, k), (retstart, rs, s)):
                self.solver.add(*[new_stack[j] == z3.If(z3.And(isrecur, d == j), value, old_stack[j])
                                  for j in range(length + 1)])
            old = current
            self.transitions += 1
        self.solver.add(pc == self.n, cur == length, out == outlength)

    def candidate(self):
        self.check_time()
        self.solver.set(timeout=max(1, int(1000 * (self.deadline - time.monotonic()))))
        result = self.solver.check()
        if result != z3.sat:
            return None, str(result) + (":" + self.solver.reason_unknown() if result == z3.unknown else "")
        model = self.solver.model()
        values = lambda arr, i: model.eval(arr[i]).as_long()
        nodes = []
        for i in range(self.n):
            op = self.OPS[values(self.op, i)]
            arg, a, b = (values(arr, i) for arr in (self.arg, self.a, self.b))
            nodes.append(node(op, arg=arg if op in ("bit", "recur", "param") else None,
                              cell=self.names[arg] if op == "call" else None,
                              a=a if op != "ret" else None, b=b if op in ("eq", "bit") else None))
        return canonical(nodes), "sat"


def mechanical(library, pairs, bits, maximum, bindings, seconds=120):
    started = time.monotonic()
    deadline = started + seconds
    effort = {"method": "reuse-first adapters, then counterexample-guided SMT over forward CFGs",
              "adapter_candidates": 0, "smt_candidates": 0, "counterexamples": [], "solver_status": None}
    answer = None
    # Ordinary reuse preference; no free-energy or complexity admission formula.
    callback_cells = [name for name, c in library.base.cells.items() if c["width"] == 2]
    for name, cell in library.cells.items():
        if cell["bits"] != bits:
            continue
        for context, a, b in itertools.product(range(2 ** bits), callback_cells, callback_cells):
            if time.monotonic() >= deadline:
                break
            nodes = [node("invoke", arg=context, cell=name, a=1, bindings=(a, b)), node("ret")]
            built, _ = library.attach("Candidate", nodes, bits)
            effort["adapter_candidates"] += 1
            if score(built, "Candidate", pairs, bindings)["exact"]:
                answer = nodes
                break
        if answer:
            break
    synth = None
    if answer is None:
        try:
            synth = Synthesizer(library, bits, maximum, bindings, deadline)
            used = set()
            # All examples are available to both arms. Start small; add the
            # shortest failing supplied example after each complete candidate.
            index = min(range(len(pairs)), key=lambda i: len(pairs[i][0]))
            while True:
                synth.add_example(*pairs[index])
                used.add(index)
                effort["counterexamples"].append(index)
                nodes, status = synth.candidate()
                effort["solver_status"] = status
                if nodes is None:
                    break
                effort["smt_candidates"] += 1
                built, _ = library.attach("Candidate", nodes, bits)
                result = score(built, "Candidate", pairs, bindings)
                if result["exact"]:
                    answer = nodes
                    break
                failures = [i for i, r in enumerate(result["records"]) if not r["exact"]]
                index = min(failures, key=lambda i: (len(pairs[i][0]), i))
                if index in used:
                    raise ValueError("SMT candidate disagrees with graph semantics on a constrained example")
        except Deadline as exc:
            effort["solver_status"] = str(exc)
    effort.update(seconds=time.monotonic() - started, peak_process_rss_mib=rss_mib(),
                  constrained_examples=synth.examples if synth else 0,
                  symbolic_transitions=synth.transitions if synth else 0)
    return answer, effort
