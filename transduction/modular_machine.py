"""Return-aware program graphs and exact attachment along procedure interfaces.

This is a NEW executable category, not a relaxation of the tail-only transducer
gate. Its fixed interpreter supplies sequencing, bounded-window instructions,
calls/returns, equality guards, progress-checked loops and whole-tape pipelines.
The underlying labelled multigraph pushout is the existing generic quotient.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass

from cofibration import Graph, Morphism, pushout, read_graph
from pattern_fsa import PatternRule, PrimitiveSet


SIGNATURE = PrimitiveSet("return-aware-program-graphs-v1", tuple(range(64)))
FRAGMENT, WHOLE, RETURN, WORD, CALL, SEQ, WHILE, IF, PIPE = range(40, 49)
BODY, EXIT, FIRST, SECOND, ENTER, RESUME = range(50, 56)
INSTRUCTIONS = (0, 1, 10, 30, 11, 31)
OPS = {"seq": SEQ, "while": WHILE, "if": IF, "pipe": PIPE}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def call(cell):
    return {"op": "call", "cell": cell}


def dependencies(program):
    if program["op"] == "call":
        return {program["cell"]}
    return set().union(*(dependencies(child) for child in program.get("children", [])))


def validate_program(program, cells, *, allow_word=False):
    if not isinstance(program, dict):
        raise ValueError("program must be an object")
    op = program.get("op")
    if op == "call":
        if set(program) != {"op", "cell"} or program["cell"] not in cells:
            raise ValueError("unbound call")
    elif op == "word":
        word = program.get("actions")
        if (not allow_word or set(program) != {"op", "actions"} or not isinstance(word, list)
                or not 1 <= len(word) <= 8
                or any(type(a) is not int or a not in INSTRUCTIONS for a in word)):
            raise ValueError("primitive words are acquired separately, not glue")
    elif op in OPS:
        keys = {"op", "children"} | ({"n"} if op == "while" else set())
        children = program.get("children")
        if set(program) != keys or not isinstance(children, list) or len(children) != (1 if op == "while" else 2):
            raise ValueError("invalid constructor")
        if op == "while" and (type(program["n"]) is not int or not 1 <= program["n"] <= 3):
            raise ValueError("invalid loop threshold")
        for child in children:
            validate_program(child, cells, allow_word=False)
        if op == "pipe" and any(child["op"] != "call" or cells[child["cell"]]["kind"] != "whole"
                                for child in children):
            raise ValueError("pipeline requires two whole-input procedures")
    else:
        raise ValueError("unknown constructor")


class Builder:
    def __init__(self):
        self.labels = []
        self.rules = []

    def node(self, label):
        index = len(self.labels)
        self.labels.append(tuple(label))
        self.rules.append(PatternRule(index, 0, tuple(label), index))
        return index

    def edge(self, source, role, target):
        self.rules.append(PatternRule(source, 1, (role,), target))

    def graph(self):
        return Graph(len(self.labels), tuple(self.rules), SIGNATURE)


@dataclass(frozen=True)
class Library:
    graph: Graph
    cells: dict

    @classmethod
    def empty(cls):
        return cls(Graph(0, (), SIGNATURE), {})

    @property
    def digest(self):
        return digest(self.payload())

    def payload(self):
        return {"graph": self.graph.payload(), "cells": self.cells}

    @classmethod
    def read(cls, payload):
        library = cls(read_graph(payload["graph"]), payload["cells"])
        library.verify()
        return library

    def attach(self, cell_id, program, *, kind="whole", width=0):
        if cell_id in self.cells or not isinstance(cell_id, str):
            raise ValueError("cell ID is already bound")
        if kind not in {"fragment", "whole"} or type(width) is not int:
            raise ValueError("invalid procedure interface")
        if (kind == "fragment" and not 1 <= width <= 3) or (kind == "whole" and width != 0):
            raise ValueError("invalid window width")
        validate_program(program, self.cells, allow_word=kind == "fragment")
        if kind == "fragment" and program["op"] != "word":
            raise ValueError("seed fragments must be primitive words")
        deps = sorted(dependencies(program))
        boundary, glue = Builder(), Builder()
        left_states, right_states, left_rules, right_rules = [], [], [], []
        ports = {}
        index = {rule: i for i, rule in enumerate(self.graph.rules)}
        for dep in deps:
            old = self.cells[dep]
            label = (FRAGMENT, old["width"]) if old["kind"] == "fragment" else (WHOLE,)
            a, z = boundary.node(label), boundary.node((RETURN,))
            boundary.edge(a, EXIT, z)
            ga, gz = glue.node(label), glue.node((RETURN,))
            glue.edge(ga, EXIT, gz)
            left_states.extend((old["entry"], old["exit"]))
            right_states.extend((ga, gz))
            ports[dep] = (ga, gz)
        for rule in boundary.rules:
            image = PatternRule(left_states[rule.state], rule.observation, rule.actions,
                                left_states[rule.next_state])
            left_rules.append(index[image])
        right_rules = list(range(len(boundary.rules)))
        entry = glue.node((FRAGMENT, width) if kind == "fragment" else (WHOLE,))
        exit_node = glue.node((RETURN,))
        glue.edge(entry, EXIT, exit_node)

        def compile_term(term):
            op = term["op"]
            if op == "call":
                node = glue.node((CALL,))
                ga, gz = ports[term["cell"]]
                glue.edge(node, ENTER, ga)
                glue.edge(node, RESUME, gz)
            elif op == "word":
                node = glue.node((WORD, *term["actions"]))
            else:
                node = glue.node((OPS[op], term["n"]) if op == "while" else (OPS[op],))
                for role, child in zip((FIRST, SECOND), term["children"]):
                    glue.edge(node, role, compile_term(child))
            return node

        glue.edge(entry, BODY, compile_term(program))
        square = pushout(boundary.graph(), self.graph, glue.graph(),
                         Morphism(tuple(left_states), tuple(left_rules)),
                         Morphism(tuple(right_states), tuple(right_rules)))
        cell = {"kind": kind, "width": width, "program": program,
                "dependencies": {dep: self.cells[dep]["hash"] for dep in deps},
                "entry": square.from_right.states[entry],
                "exit": square.from_right.states[exit_node]}
        cell["hash"] = content_hash(cell)
        return Library(square.graph, {**self.cells, cell_id: cell}), square

    def verify(self):
        if self.graph.primitives != SIGNATURE:
            raise ValueError("wrong module semantics signature")
        machine = Machine(self)
        known = {}
        for name, cell in self.cells.items():
            validate_program(cell["program"], known, allow_word=cell["kind"] == "fragment")
            if cell["hash"] != content_hash(cell):
                raise ValueError("changed cell content")
            if cell["dependencies"] != {dep: known[dep]["hash"] for dep in sorted(dependencies(cell["program"]))}:
                raise ValueError("unbound dependency version")
            if machine.decode(cell["entry"]) != cell["program"]:
                raise ValueError("AST does not match executable program graph")
            expected = (FRAGMENT, cell["width"]) if cell["kind"] == "fragment" else (WHOLE,)
            if machine.labels[cell["entry"]] != expected or machine.children[cell["entry"]] != {
                EXIT: cell["exit"], BODY: machine.children[cell["entry"]][BODY]
            } or machine.labels[cell["exit"]] != (RETURN,):
                raise ValueError("changed return interface")
            known[name] = cell


def content_hash(cell):
    return digest({key: cell[key] for key in ("kind", "width", "program", "dependencies")})


class ExecutionError(ValueError):
    pass


class Machine:
    """Execute the quotient graph itself. No separately trusted solution AST."""
    def __init__(self, library):
        self.library = library
        self.labels, self.children = {}, {}
        self.names = {cell["entry"]: name for name, cell in library.cells.items()}
        if len(self.names) != len(library.cells):
            raise ValueError("aliased procedure entries")
        for rule in library.graph.rules:
            if rule.observation == 0 and rule.state == rule.next_state:
                if rule.state in self.labels:
                    raise ValueError("duplicate node label")
                self.labels[rule.state] = rule.actions
            elif rule.observation == 1 and len(rule.actions) == 1:
                edges = self.children.setdefault(rule.state, {})
                if rule.actions[0] in edges:
                    raise ValueError("duplicate child role")
                edges[rule.actions[0]] = rule.next_state
            else:
                raise ValueError("malformed program graph edge")
        if len(self.labels) != library.graph.states:
            raise ValueError("missing program node label")

    def decode(self, entry):
        active = set()

        def term(node):
            if node in active:
                raise ValueError("cyclic term")
            active.add(node)
            label, edges = self.labels[node], self.children.get(node, {})
            op = label[0]
            if op == CALL:
                if len(label) != 1 or set(edges) != {ENTER, RESUME}:
                    raise ValueError("invalid call interface")
                name = self.names[edges[ENTER]]
                if edges[RESUME] != self.library.cells[name]["exit"]:
                    raise ValueError("call return port mismatch")
                result = call(name)
            elif op == WORD:
                if edges:
                    raise ValueError("word with children")
                result = {"op": "word", "actions": list(label[1:])}
            elif op in OPS.values():
                kind = next(k for k, v in OPS.items() if v == op)
                roles = (FIRST,) if op == WHILE else (FIRST, SECOND)
                if set(edges) != set(roles) or len(label) != (2 if op == WHILE else 1):
                    raise ValueError("invalid constructor edges")
                result = {"op": kind, "children": [term(edges[role]) for role in roles]}
                if op == WHILE:
                    result["n"] = label[1]
            else:
                raise ValueError("not a term node")
            active.remove(node)
            return result

        return term(self.children[entry][BODY])

    def run(self, cell_id, source, *, disabled=(), step_limit=50000, output_limit=4096,
            depth_limit=64, trace_limit=0):
        if not isinstance(source, (tuple, list)) or any(type(t) is not int for t in source):
            raise ValueError("input must be opaque integer tokens")
        steps = primitive_steps = calls = returns = peak_depth = peak_buffer = 0
        visits, trace = Counter(), []

        def event(kind, name, position, depth):
            if len(trace) < trace_limit:
                trace.append([kind, name, position, depth])

        def tick(primitive=False):
            nonlocal steps, primitive_steps
            steps += 1
            primitive_steps += int(primitive)
            if steps > step_limit:
                raise ExecutionError("step limit")

        def bounded(output):
            nonlocal peak_buffer
            peak_buffer = max(peak_buffer, len(output))
            if len(output) > output_limit:
                raise ExecutionError("output/buffer limit")

        def procedure(entry, tape, cursor, output, depth):
            nonlocal calls, returns, peak_depth
            tick()
            name = self.names[entry]
            if name in disabled:
                # Counterfactual identity stub, not an automatic failure. Elide
                # BOTH output and cursor effects; callers' checks still apply.
                event("elided", name, cursor, depth)
                return cursor
            if depth > depth_limit:
                raise ExecutionError("call depth limit")
            peak_depth = max(peak_depth, depth)
            calls += 1
            visits[name] += 1
            event("enter", name, cursor, depth)
            label = self.labels[entry]
            start = cursor
            if label[0] == FRAGMENT:
                width = label[1]
                if len(tape) - cursor < width:
                    raise ExecutionError("fragment window underflow")
                window = tape[cursor:cursor + width]
                local = execute(self.children[entry][BODY], window, 0, output, depth)
                if local != width:
                    raise ExecutionError("fragment did not consume its declared window")
                cursor += width
            elif label[0] == WHOLE:
                cursor = execute(self.children[entry][BODY], tape, cursor, output, depth)
                if cursor != len(tape):
                    raise ExecutionError("whole-input procedure left unread tokens")
            else:
                raise ExecutionError("call target is not a procedure")
            if self.labels[self.children[entry][EXIT]] != (RETURN,):
                raise ExecutionError("invalid return")
            returns += 1
            event("return", name, cursor, depth)
            if cursor < start:
                raise ExecutionError("backward consumption")
            return cursor

        def execute(node, tape, cursor, output, depth):
            tick()
            label, edges = self.labels[node], self.children.get(node, {})
            op = label[0]
            if op == WORD:
                registers = [None, None]  # local to each invocation, never inherited
                for action in label[1:]:
                    tick(True)
                    token = tape[cursor] if cursor < len(tape) else None
                    emit = None
                    if action == 0:
                        cursor = min(cursor + 1, len(tape))
                    elif action == 1:
                        emit = token
                    elif action in (10, 11):
                        if token is not None:
                            registers[action - 10] = token
                    elif action in (30, 31):
                        emit = registers[action - 30]
                    else:
                        raise ExecutionError("unknown primitive")
                    if emit is not None:
                        output.append(emit)
                        bounded(output)
            elif op == CALL:
                if self.children[edges[ENTER]][EXIT] != edges[RESUME]:
                    raise ExecutionError("return interface mismatch")
                cursor = procedure(edges[ENTER], tape, cursor, output, depth + 1)
            elif op == SEQ:
                cursor = execute(edges[FIRST], tape, cursor, output, depth)
                cursor = execute(edges[SECOND], tape, cursor, output, depth)
            elif op == WHILE:
                while len(tape) - cursor >= label[1]:
                    previous = cursor
                    cursor = execute(edges[FIRST], tape, cursor, output, depth)
                    if cursor <= previous:
                        raise ExecutionError("loop failed strict cursor progress")
                    tick()
            elif op == IF:
                same = cursor + 1 < len(tape) and tape[cursor] == tape[cursor + 1]
                cursor = execute(edges[FIRST if same else SECOND], tape, cursor, output, depth)
            elif op == PIPE:
                intermediate, final = [], []
                end = execute(edges[FIRST], tape, cursor, intermediate, depth)
                if end != len(tape):
                    raise ExecutionError("pipeline stage left unread input")
                second_end = execute(edges[SECOND], tuple(intermediate), 0, final, depth)
                if second_end != len(intermediate):
                    raise ExecutionError("pipeline stage left unread buffer")
                bounded(intermediate)
                output.extend(final)
                bounded(output)
                cursor = end
            else:
                raise ExecutionError("invalid executable term")
            return cursor

        output = []
        error = None
        cursor = None
        try:
            cursor = procedure(self.library.cells[cell_id]["entry"], tuple(source), 0, output, 1)
        except ExecutionError as exc:
            error = str(exc)
        return {"output": output, "cursor": cursor, "ok": error is None, "error": error,
                "steps": steps, "primitive_steps": primitive_steps, "calls": calls,
                "returns": returns, "max_call_depth": peak_depth, "max_buffer": peak_buffer,
                "visits": dict(visits), "trace": trace}


def program_size(program):
    return 1 + len(program.get("actions", [])) + sum(program_size(c) for c in program.get("children", []))


def expanded_size(library, cell_id):
    """Diagnostic literal inlining size; loop bodies are NOT execution-unrolled."""
    def size(term):
        if term["op"] == "call":
            return 1 + expanded_size(library, term["cell"])  # inlined procedure/window interface
        return 1 + len(term.get("actions", [])) + sum(size(c) for c in term.get("children", []))
    return 2 + size(library.cells[cell_id]["program"])


def reachable_cells(library, cell_id):
    result = {cell_id}
    for dep in library.cells[cell_id]["dependencies"]:
        result.update(reachable_cells(library, dep))
    return result


def structural_metrics(library, cell_id):
    reachable = reachable_cells(library, cell_id)
    shared = sum(2 + program_size(library.cells[name]["program"]) for name in reachable)
    return {"reachable_cells": len(reachable), "reachable_shared_term_units": shared,
            "single_root_inlined_term_units": expanded_size(library, cell_id),
            "library_stored_term_units": sum(2 + program_size(c["program"]) for c in library.cells.values()),
            "all_roots_separately_inlined_term_units": sum(expanded_size(library, name) for name in library.cells),
            "graph_nodes": library.graph.states, "graph_edges": len(library.graph.rules)}


def unshare(library, cell_id):
    """Copy each syntactic call occurrence's procedure subtree, with no sharing.

    Loops stay loops and call/window semantics stay fixed. This is a storage and
    execution control, NOT a separately synthesized no-library baseline.
    """
    result = Library.empty()

    def clone_cell(name):
        nonlocal result
        cell = library.cells[name]
        body = clone_term(cell["program"])
        fresh = f"U{len(result.cells):04}"
        result, _ = result.attach(fresh, body, kind=cell["kind"], width=cell["width"])
        return fresh

    def clone_term(term):
        if term["op"] == "call":
            return call(clone_cell(term["cell"]))
        return {**term, **({"children": [clone_term(c) for c in term["children"]]} if "children" in term else {})}

    entry = clone_cell(cell_id)
    result.verify()
    return result, entry
