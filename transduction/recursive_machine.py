"""Recursive prefix procedures attached to unchanged acquired token machines.

New control only: forward control flow, recursive calls, lexically restored
context bits and two width-two procedure parameters. There is no tree parser,
subtree instruction, task oracle, or new token instruction in this interpreter.
"""
from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass

from cofibration import Morphism, pushout, read_graph
from modular_machine import (BODY, ENTER, EXIT, FIRST, RESUME, RETURN, SECOND,
                             Builder, Library, digest)
from pattern_fsa import PatternRule

PREFIX, PARAM, RECURSE, BIT, INVOKE, RET, EQ, FRAG = range(56, 64)
LABELS = {"ret": RET, "eq": EQ, "bit": BIT, "recur": RECURSE,
          "param": PARAM, "call": FRAG, "invoke": INVOKE}
FIELDS = {"op", "arg", "cell", "a", "b", "bindings"}
STEP_LIMIT, OUTPUT_LIMIT, DEPTH_LIMIT = 100000, 8192, 160


def node(op, arg=None, cell=None, a=None, b=None, bindings=()):
    return dict(op=op, arg=arg, cell=cell, a=a, b=b, bindings=list(bindings))


def canonical(nodes):
    reachable = set()
    def visit(i):
        if i in reachable:
            return
        reachable.add(i)
        for key in ("a", "b"):
            if nodes[i][key] is not None:
                visit(nodes[i][key])
    visit(0)
    order = sorted(reachable)
    index = {old: new for new, old in enumerate(order)}
    return [{**copy.deepcopy(nodes[i]), **{key: index[nodes[i][key]]
             for key in ("a", "b") if nodes[i][key] is not None}} for i in order]


def parse(payload, library, bits, maximum):
    if (not isinstance(payload, dict) or set(payload) != {"library_hash", "nodes"}
            or payload["library_hash"] != library.digest):
        raise ValueError("proposal is not bound to the retained library")
    nodes = payload["nodes"]
    if not isinstance(nodes, list) or not 1 <= len(nodes) <= maximum:
        raise ValueError("control-node limit")
    if bits not in (1, 2):
        raise ValueError("invalid context interface")
    for i, n in enumerate(nodes):
        if not isinstance(n, dict) or set(n) != FIELDS or n["op"] not in LABELS:
            raise ValueError("invalid control instruction")
        op = n["op"]
        keys = (("a", "b") if op in ("eq", "bit") else () if op == "ret" else ("a",))
        for key in ("a", "b"):
            value = n[key]
            if key in keys:
                if type(value) is not int or not i < value < len(nodes):
                    raise ValueError("ordinary control edges must point forward")
            elif value is not None:
                raise ValueError("unused edge must be null")
        bound = {"bit": bits, "recur": 2 ** bits, "param": 2, "invoke": 2 ** bits}.get(op)
        if bound is not None:
            if type(n["arg"]) is not int or not 0 <= n["arg"] < bound:
                raise ValueError("invalid context or parameter argument")
        elif n["arg"] is not None:
            raise ValueError("unused argument must be null")
        if op == "call":
            if n["cell"] not in library.base.cells:
                raise ValueError("call needs an acquired fragment")
        elif op == "invoke":
            if n["cell"] not in library.cells or library.cells[n["cell"]]["bits"] != bits:
                raise ValueError("invoke needs a compatible retained prefix procedure")
        elif n["cell"] is not None:
            raise ValueError("unused cell must be null")
        if (not isinstance(n["bindings"], list)
                or len(n["bindings"]) != (2 if op == "invoke" else 0)):
            raise ValueError("invalid callback binding interface")
        for cell in n["bindings"]:
            if cell not in library.base.cells or library.base.cells[cell]["width"] != 2:
                raise ValueError("callback must consume exactly two tokens")
    return canonical(nodes)


@dataclass(frozen=True)
class RecursiveLibrary:
    base: Library
    graph: object
    cells: dict

    @classmethod
    def from_base(cls, base):
        base.verify()
        if any(cell["kind"] != "fragment" for cell in base.cells.values()):
            raise ValueError("recursive experiment imports only acquired fragments")
        return cls(base, base.graph, {})

    def payload(self):
        return {"base": self.base.payload(), "graph": self.graph.payload(), "cells": self.cells}

    @property
    def digest(self):
        return digest(self.payload())

    @classmethod
    def read(cls, value):
        result = cls(Library.read(value["base"]), read_graph(value["graph"]), value["cells"])
        result.verify()
        return result

    def attach(self, name, nodes, bits):
        if name in self.base.cells or name in self.cells:
            raise ValueError("cannot replace a retained cell")
        nodes = parse({"library_hash": self.digest, "nodes": nodes}, self, bits, 32)
        deps = sorted({n["cell"] for n in nodes if n["cell"] is not None}
                      | {c for n in nodes for c in n["bindings"]})
        old = {**self.base.cells, **self.cells}
        boundary, glue = Builder(), Builder()
        ls, rs, ports = [], [], {}
        labels = {r.state: r.actions for r in self.graph.rules if r.observation == 0}
        for dep in deps:
            cell = old[dep]
            a, z = boundary.node(labels[cell["entry"]]), boundary.node((RETURN,))
            boundary.edge(a, EXIT, z)
            ga, gz = glue.node(labels[cell["entry"]]), glue.node((RETURN,))
            glue.edge(ga, EXIT, gz)
            ls.extend((cell["entry"], cell["exit"]))
            rs.extend((ga, gz))
            ports[dep] = (ga, gz)
        indices = {r: i for i, r in enumerate(self.graph.rules)}
        le = [indices[PatternRule(ls[r.state], r.observation, r.actions, ls[r.next_state])]
              for r in boundary.rules]
        entry, end = glue.node((PREFIX, bits, 2)), glue.node((RETURN,))
        glue.edge(entry, EXIT, end)
        vertices = [glue.node((LABELS[n["op"]],) + (() if n["arg"] is None else (n["arg"],)))
                    for n in nodes]
        glue.edge(entry, BODY, vertices[0])
        for i, n in enumerate(nodes):
            v = vertices[i]
            for key, role in (("a", FIRST), ("b", SECOND)):
                if n[key] is not None:
                    glue.edge(v, role, vertices[n[key]])
            if n["op"] in ("call", "invoke", "recur"):
                a, z = (entry, end) if n["op"] == "recur" else ports[n["cell"]]
                glue.edge(v, ENTER, a)
                glue.edge(v, RESUME, z)
            for slot, binding in enumerate(n["bindings"]):
                glue.edge(v, slot, ports[binding][0])
                glue.edge(v, slot + 2, ports[binding][1])
        square = pushout(boundary.graph(), self.graph, glue.graph(),
                         Morphism(tuple(ls), tuple(le)),
                         Morphism(tuple(rs), tuple(range(len(boundary.rules)))))
        cell = {"bits": bits, "nodes": nodes,
                "dependencies": {dep: old[dep]["hash"] for dep in deps},
                "entry": square.from_right.states[entry], "exit": square.from_right.states[end]}
        cell["hash"] = digest({k: cell[k] for k in ("bits", "nodes", "dependencies")})
        return RecursiveLibrary(self.base, square.graph, {**self.cells, name: cell}), square

    def verify(self):
        # Recompilation verifies every executable edge, return port and hash,
        # including recursive cycles, without trusting serialized AST metadata.
        rebuilt = RecursiveLibrary.from_base(self.base)
        for name, cell in self.cells.items():
            rebuilt, _ = rebuilt.attach(name, cell["nodes"], cell["bits"])
            if rebuilt.cells[name] != cell:
                raise ValueError("changed recursive cell or dependency")
        if rebuilt.graph != self.graph:
            raise ValueError("executable quotient differs from its cell derivation")


class Machine:
    def __init__(self, library):
        self.library = library
        self.labels, self.edges = {}, {}
        self.names = {cell["entry"]: name for name, cell in
                      {**library.base.cells, **library.cells}.items()}
        for rule in library.graph.rules:
            if rule.observation == 0:
                if rule.state != rule.next_state or rule.state in self.labels:
                    raise ValueError("invalid node label")
                self.labels[rule.state] = tuple(rule.actions)
            else:
                role = rule.actions[0]
                if role in self.edges.setdefault(rule.state, {}):
                    raise ValueError("duplicate executable edge")
                self.edges[rule.state][role] = rule.next_state

    def run(self, name, source, bindings, context=0, *, trace_limit=0, disabled=(),
            step_limit=STEP_LIMIT, depth_limit=DEPTH_LIMIT):
        source = tuple(source)
        output, trace, visits = [], [], Counter()
        steps = calls = returns = maximum_depth = 0
        cursor = 0
        def tick():
            nonlocal steps
            steps += 1
            if steps > step_limit or len(output) > OUTPUT_LIMIT:
                raise ValueError("execution resource limit")
        def event(kind, entry, position, mode, depth):
            if len(trace) < trace_limit:
                trace.append([kind, self.names[entry], position, mode, depth])
        def fragment(entry, pos):
            # Read the original word from the shared quotient, not its AST.
            width = self.labels[entry][1]
            if pos + width > len(source):
                raise ValueError("fragment window underflow")
            regs = [None, None]
            local = 0
            actions = self.labels[self.edges[entry][BODY]][1:]
            for action in actions:
                tick()
                token = source[pos + local] if local < width else None
                emit = None
                if action == 0:
                    local = min(local + 1, width)
                elif action == 1:
                    emit = token
                elif action in (10, 11):
                    if token is not None:
                        regs[action - 10] = token
                elif action in (30, 31):
                    emit = regs[action - 30]
                else:
                    raise ValueError("unknown token instruction")
                if emit is not None:
                    output.append(emit)
            if local != width:
                raise ValueError("fragment failed its consumption contract")
            return pos + width
        def procedure(entry, pos, mode, args, depth):
            nonlocal calls, returns, maximum_depth
            tick()
            if depth > depth_limit:
                raise ValueError("call depth limit")
            maximum_depth = max(maximum_depth, depth)
            if self.names[entry] in disabled:
                event("elided", entry, pos, mode, depth)
                return pos
            calls += 1
            visits[self.names[entry]] += 1
            event("enter", entry, pos, mode, depth)
            if self.labels[entry][0] != PREFIX:
                pos = fragment(entry, pos)
            else:
                start = pos
                pc = self.edges[entry][BODY]
                while True:
                    tick()
                    label, edges = self.labels[pc], self.edges.get(pc, {})
                    op = label[0]
                    if op == RET:
                        break
                    if op == EQ:
                        yes = pos + 1 < len(source) and source[pos] == source[pos + 1]
                        pc = edges[FIRST if yes else SECOND]
                        continue
                    if op == BIT:
                        pc = edges[FIRST if mode & (1 << label[1]) else SECOND]
                        continue
                    if op in (FRAG, RECURSE, INVOKE):
                        child = edges[ENTER]
                        if self.edges[child][EXIT] != edges[RESUME]:
                            raise ValueError("call return interface changed")
                    if op == FRAG:
                        pos = procedure(child, pos, mode, args, depth + 1)
                    elif op == PARAM:
                        pos = procedure(args[label[1]], pos, mode, args, depth + 1)
                    elif op == RECURSE:
                        if pos <= start:
                            raise ValueError("recursive call requires strict input progress")
                        pos = procedure(child, pos, mode ^ label[1], args, depth + 1)
                    elif op == INVOKE:
                        new_args = tuple(edges[slot] for slot in range(2))
                        if any(self.edges[a][EXIT] != edges[slot + 2] for slot, a in enumerate(new_args)):
                            raise ValueError("callback return interface changed")
                        pos = procedure(child, pos, label[1], new_args, depth + 1)
                    else:
                        raise ValueError("invalid control node")
                    pc = edges[FIRST]
            returns += 1
            event("return", entry, pos, mode, depth)
            return pos
        error = None
        try:
            if any(type(t) is not int for t in source):
                raise ValueError("opaque integer tokens required")
            if len(bindings) != 2 or any(self.library.base.cells[c]["width"] != 2 for c in bindings):
                raise ValueError("two width-two callbacks required")
            args = tuple(self.library.base.cells[c]["entry"] for c in bindings)
            cell = self.library.cells[name]
            if type(context) is not int or not 0 <= context < 2 ** cell["bits"]:
                raise ValueError("context outside interface")
            cursor = procedure(cell["entry"], 0, context, args, 1)
            tick()
            if cursor != len(source):
                raise ValueError("top-level call left unread input")
        except (ValueError, RecursionError) as exc:
            error = str(exc)
        return dict(ok=error is None, error=error, output=output, cursor=cursor, steps=steps,
                    calls=calls, returns=returns, max_call_depth=maximum_depth,
                    visits=dict(visits), trace=trace)


def schema(library, maximum):
    nullable = {"type": ["integer", "null"], "minimum": 0, "maximum": maximum - 1}
    item = {"type": "object", "additionalProperties": False,
            "properties": {"op": {"type": "string", "enum": list(LABELS)},
                           "arg": nullable, "cell": {"type": ["string", "null"]},
                           "a": nullable, "b": nullable,
                           "bindings": {"type": "array", "items": {"type": "string"}, "maxItems": 2}},
            "required": sorted(FIELDS)}
    return {"type": "object", "additionalProperties": False,
            "properties": {"library_hash": {"type": "string", "enum": [library.digest]},
                           "nodes": {"type": "array", "items": item, "minItems": 1, "maxItems": maximum}},
            "required": ["library_hash", "nodes"]}
