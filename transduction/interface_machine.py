"""Typed, independently chosen interfaces over the unchanged token basis.

The new control language has lexical Boolean values and nullary procedure
values. A new procedure chooses 0..4 arguments and a Boolean or unit result.
There is no tree operation, callback count, context register or task parser.
Every expression and call boundary is compiled into the executable pushout.
"""
from __future__ import annotations

import copy
import json
import re
from collections import Counter
from dataclasses import dataclass

from cofibration import Morphism, pushout, read_graph
from modular_machine import (BODY, ENTER, EXIT, RESUME, RETURN, Builder, Library,
                             digest)
from pattern_fsa import PatternRule


PROC = 20
TAGS = {name: i for i, name in enumerate(
    ("atom", "ref", "call", "if", "seq", "let", "not", "xor", "and", "or"), 21)}
REVERSE_TAGS = {v: k for k, v in TAGS.items()}
TYPES = {"U": 0, "B": 1, "F": 2}
MAX_NODES, MAX_ARGUMENTS, MAX_DEFINITIONS = 180, 4, 4
IDENT = re.compile(r"[A-Za-z_][A-Za-z_0-9]*\Z")


def parse(text):
    if not isinstance(text, str) or len(text) > 16000:
        raise ValueError("body must be a bounded S-expression string")
    tokens = re.findall(r"\(|\)|[^\s()]+", text)
    if len(tokens) > 1000:
        raise ValueError("expression token bound")
    index = 0
    def expression(depth=0):
        nonlocal index
        if depth > 48 or index == len(tokens):
            raise ValueError("incomplete or excessively nested expression")
        token = tokens[index]
        index += 1
        if token == "(":
            items = []
            while index < len(tokens) and tokens[index] != ")":
                items.append(expression(depth + 1))
            if index == len(tokens) or not items:
                raise ValueError("invalid parentheses")
            index += 1
            return tuple(items)
        if token == ")" or not IDENT.fullmatch(token):
            raise ValueError("invalid symbol")
        return token
    result = expression()
    if index != len(tokens) or size(result) > MAX_NODES:
        raise ValueError("trailing text or expression node bound")
    return result


def render(term):
    return term if isinstance(term, str) else "(" + " ".join(map(render, term)) + ")"


def size(term):
    return 1 if isinstance(term, str) else 1 + sum(size(t) for t in term[1:])


def signature(definition):
    return ([p["type"] for p in definition["params"]], definition["returns"])


def typecheck(term, env, signatures, own):
    if isinstance(term, str):
        if term == "unit":
            return "U"
        if term in ("true", "false", "eq"):
            return "B"
        if term not in env:
            raise ValueError("unbound variable: " + term)
        return env[term]
    op, *args = term
    if op == "ref" and len(args) == 1:
        if signatures.get(args[0]) != ([], "U"):
            raise ValueError("ref requires a nullary unit procedure")
        return "F"
    if op == "call" and 1 <= len(args) <= MAX_ARGUMENTS + 1:
        callee = args[0]
        sig = (([], "U") if env.get(callee) == "F"
               else signatures.get(own if callee == "self" else callee))
        if sig is None or len(sig[0]) != len(args) - 1:
            raise ValueError("call interface mismatch: " + str(callee))
        actual = [typecheck(t, env, signatures, own) for t in args[1:]]
        if actual != sig[0]:
            raise ValueError("call argument types disagree with boundary")
        return sig[1]
    if op == "if" and len(args) == 3:
        test = typecheck(args[0], env, signatures, own)
        a, b = [typecheck(t, env, signatures, own) for t in args[1:]]
        if test != "B" or a != b:
            raise ValueError("ill-typed conditional")
        return a
    if op == "seq" and 1 <= len(args) <= 12:
        return [typecheck(t, env, signatures, own) for t in args][-1]
    if op == "let" and len(args) == 3:
        var, value, body = args
        if (not isinstance(var, str) or not IDENT.fullmatch(var) or var in env
                or var in {"unit", "eq", "self", "true", "false"} or var in signatures):
            raise ValueError("invalid or shadowing local binder")
        typ = typecheck(value, env, signatures, own)
        return typecheck(body, {**env, var: typ}, signatures, own)
    if op in ("not", "xor", "and", "or") and len(args) == (1 if op == "not" else 2):
        if any(typecheck(t, env, signatures, own) != "B" for t in args):
            raise ValueError("Boolean operands required")
        return "B"
    raise ValueError("unknown expression or arity: " + str(op))


def dependencies(term, env, own):
    if isinstance(term, str):
        return set()
    op, *args = term
    if op == "ref":
        return set() if args[0] == own else {args[0]}
    if op == "call":
        dep = set() if args[0] in env or args[0] in ("self", own) else {args[0]}
        return dep | set().union(*(dependencies(t, env, own) for t in args[1:]))
    if op == "let":
        return dependencies(args[1], env, own) | dependencies(args[2], env | {args[0]}, own)
    return set().union(*(dependencies(t, env, own) for t in args))


def _pack(value):
    return tuple(n for b in json.dumps(value, separators=(",", ":")).encode()
                 for n in (b // 16, b % 16))


def _unpack(values):
    if len(values) % 2 or any(not 0 <= v < 16 for v in values):
        raise ValueError("invalid graph label encoding")
    return json.loads(bytes(16 * a + b for a, b in zip(values[::2], values[1::2])))


@dataclass(frozen=True)
class InterfaceLibrary:
    base: Library
    graph: object
    cells: dict

    @classmethod
    def from_base(cls, base):
        base.verify()
        if any(c["kind"] != "fragment" for c in base.cells.values()):
            raise ValueError("only fixed acquired token fragments may be imported")
        return cls(base, base.graph, {})

    def payload(self):
        return dict(base=self.base.payload(), graph=self.graph.payload(), cells=self.cells,
                    order=list(self.cells))

    @property
    def digest(self):
        return digest(self.payload())

    def signatures(self):
        return {**{k: ([], "U") for k in self.base.cells},
                **{k: signature(c) for k, c in self.cells.items()}}

    @classmethod
    def read(cls, value):
        if len(value["order"]) != len(value["cells"]) or set(value["order"]) != set(value["cells"]):
            raise ValueError("invalid cell order")
        obj = cls(Library.read(value["base"]), read_graph(value["graph"]),
                  {name: value["cells"][name] for name in value["order"]})
        obj.verify()
        return obj

    def attach(self, definition):
        if not isinstance(definition, dict) or set(definition) != {"name", "params", "returns", "body"}:
            raise ValueError("invalid definition fields")
        name, params = definition["name"], definition["params"]
        old = {**self.base.cells, **self.cells}
        if (not isinstance(name, str) or not IDENT.fullmatch(name) or name in old
                or name in {"unit", "eq", "self", "true", "false"}):
            raise ValueError("invalid or retained definition name")
        if (not isinstance(params, list) or len(params) > MAX_ARGUMENTS
                or definition["returns"] not in ("U", "B")):
            raise ValueError("invalid interface")
        env = {}
        for p in params:
            if (not isinstance(p, dict) or set(p) != {"name", "type"}
                    or p["type"] not in ("B", "F") or not isinstance(p["name"], str)
                    or not IDENT.fullmatch(p["name"]) or p["name"] in env
                    or p["name"] in old or p["name"] in {name, "unit", "eq", "self", "true", "false"}):
                raise ValueError("invalid argument")
            env[p["name"]] = p["type"]
        ast = parse(definition["body"])
        sigs = {**self.signatures(), name: signature(definition)}
        if typecheck(ast, env, sigs, name) != definition["returns"]:
            raise ValueError("body result disagrees with return interface")
        deps = sorted(dependencies(ast, set(env), name))
        boundary, glue = Builder(), Builder()
        labels = {r.state: r.actions for r in self.graph.rules if r.observation == 0}
        ls, rs, ports = [], [], {}
        for dep in deps:
            cell = old[dep]
            a, z = boundary.node(labels[cell["entry"]]), boundary.node((RETURN,))
            boundary.edge(a, EXIT, z)
            ga, gz = glue.node(labels[cell["entry"]]), glue.node((RETURN,))
            glue.edge(ga, EXIT, gz)
            ls.extend((cell["entry"], cell["exit"]))
            rs.extend((ga, gz))
            ports[dep] = ga, gz
        indices = {r: i for i, r in enumerate(self.graph.rules)}
        le = [indices[PatternRule(ls[r.state], r.observation, r.actions, ls[r.next_state])]
              for r in boundary.rules]
        entry = glue.node((PROC, *_pack([params, definition["returns"]])))
        end = glue.node((RETURN,))
        glue.edge(entry, EXIT, end)
        ports[name] = ports["self"] = entry, end

        def compile_expr(term, scope):
            if isinstance(term, str):
                return glue.node((TAGS["atom"], *_pack(term)))
            op, *args = term
            literal = args[0] if op in ("ref", "call", "let") else None
            children = args[1:] if literal is not None else args
            v = glue.node((TAGS[op], *_pack(literal)))
            if op == "ref" or op == "call" and args[0] not in scope:
                a, z = ports[args[0]]
                glue.edge(v, ENTER, a)
                glue.edge(v, RESUME, z)
            for i, child in enumerate(children):
                local = scope | {args[0]} if op == "let" and i == 1 else scope
                glue.edge(v, i, compile_expr(child, local))
            return v

        glue.edge(entry, BODY, compile_expr(ast, set(env)))
        square = pushout(boundary.graph(), self.graph, glue.graph(),
                         Morphism(tuple(ls), tuple(le)),
                         Morphism(tuple(rs), tuple(range(len(boundary.rules)))))
        clean = {**copy.deepcopy(definition), "body": render(ast)}
        cell = {**clean, "dependencies": {d: old[d]["hash"] for d in deps},
                "entry": square.from_right.states[entry], "exit": square.from_right.states[end]}
        cell["hash"] = digest({k: cell[k] for k in ("params", "returns", "body", "dependencies")})
        return InterfaceLibrary(self.base, square.graph, {**self.cells, name: cell}), square

    def verify(self):
        rebuilt = InterfaceLibrary.from_base(self.base)
        for cell in self.cells.values():
            definition = {k: cell[k] for k in ("name", "params", "returns", "body")}
            rebuilt, _ = rebuilt.attach(definition)
            if rebuilt.cells[cell["name"]] != cell:
                raise ValueError("changed cell or interface")
        if rebuilt.graph != self.graph:
            raise ValueError("graph differs from executable derivation")


def attach_proposal(library, payload, root):
    if (not isinstance(payload, dict) or set(payload) != {"library_hash", "definitions"}
            or payload["library_hash"] != library.digest):
        raise ValueError("proposal is not bound to this library")
    defs = payload["definitions"]
    if not isinstance(defs, list) or not 1 <= len(defs) <= MAX_DEFINITIONS:
        raise ValueError("definition count bound")
    certificates = []
    for d in defs:
        library, square = library.attach(d)
        certificates.append(square.certificate())
    if root != defs[-1]["name"] or signature(defs[-1]) != ([], "U"):
        raise ValueError("last definition must be the requested nullary unit root")
    validate_composition(parse(defs[-1]["body"]), root)
    return library, certificates


def validate_composition(term, root):
    """Shared library/player split, not a size or interface-discovery gate.

    A task root binds and composes procedures. Input inspection and recursion
    belong in helpers. Helpers remain free to choose their own interfaces;
    merely placing specialized code in a helper does not establish transfer.
    """
    if isinstance(term, str):
        if term == "eq":
            raise ValueError("task root must compose helpers; input inspection belongs in helpers")
        return
    if term[0] in ("call", "ref") and term[1] in ("self", root):
        raise ValueError("task root must compose helpers; recursion belongs in helpers")
    children = term[2:] if term[0] in ("call", "ref", "let") else term[1:]
    for child in children:
        validate_composition(child, root)


class Machine:
    """Execute code decoded from the quotient's expression and call edges."""
    def __init__(self, library):
        self.library = library
        labels, edges = {}, {}
        for r in library.graph.rules:
            if r.observation == 0:
                if r.state != r.next_state or r.state in labels:
                    raise ValueError("invalid expression label")
                labels[r.state] = tuple(r.actions)
            else:
                role = r.actions[0]
                if role in edges.setdefault(r.state, {}):
                    raise ValueError("duplicate expression edge")
                edges[r.state][role] = r.next_state
        self.entries = {c["entry"]: n for n, c in {**library.base.cells, **library.cells}.items()}
        def decode(v):
            tag, *literal = labels[v]
            op, value = REVERSE_TAGS[tag], _unpack(literal)
            if op == "atom":
                return value
            kids = edges.get(v, {})
            if ENTER in kids:
                if edges[kids[ENTER]][EXIT] != kids[RESUME]:
                    raise ValueError("call changed return boundary")
                # The target comes from the glued edge, not the textual name.
                value = self.entries[kids[ENTER]]
            children = [decode(kids[i]) for i in sorted(k for k in kids if k < 12)]
            return (op, value, *children) if op in ("ref", "call", "let") else (op, *children)
        self.definitions, self.words = {}, {}
        for n, c in library.cells.items():
            ps, ret = _unpack(labels[c["entry"]][1:])
            self.definitions[n] = dict(name=n, params=ps, returns=ret, ast=decode(edges[c["entry"]][BODY]))
        for n, c in library.base.cells.items():
            self.words[n] = (labels[c["entry"]][1], labels[edges[c["entry"]][BODY]][1:])

    def run(self, name, source, args=(), *, disabled=(), trace_limit=0, step_limit=60000, depth_limit=150):
        return execute(self.definitions, self.words, name, source, args, disabled=disabled,
                       trace_limit=trace_limit, step_limit=step_limit, depth_limit=depth_limit)


def execute(definitions, words, name, source, args=(), *, disabled=(), trace_limit=0,
            step_limit=60000, depth_limit=150, expected=None):
    """Also used by search on ASTs; admission always recompiles to graph code."""
    cursor, steps, depth_peak = 0, 0, 0
    output, trace, visits, active = [], [], Counter(), {}
    def tick():
        nonlocal steps
        steps += 1
        if steps > step_limit or len(output) > 8192:
            raise ValueError("execution resource limit")
    def invoke(callee, values, depth):
        nonlocal cursor, depth_peak
        tick()
        if depth > depth_limit:
            raise ValueError("call depth limit")
        depth_peak = max(depth_peak, depth)
        visits[callee] += 1
        if callee in disabled:
            return False if definitions.get(callee, {}).get("returns") == "B" else None
        if len(trace) < trace_limit:
            trace.append(["enter", callee, cursor, list(values), depth])
        if callee in words:
            width, actions = words[callee]
            if values or cursor + width > len(source):
                raise ValueError("fragment window or arity")
            regs, offset, start = [None, None], 0, cursor
            for action in actions:
                tick()
                token = source[start + offset] if offset < width else None
                emit = None
                if action == 0:
                    offset = min(offset + 1, width)
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
                    if expected is not None and (len(output) >= len(expected) or expected[len(output)] != emit):
                        raise ValueError("irreversible output mismatch")
                    output.append(emit)
            if offset != width:
                raise ValueError("fragment consumption contract")
            cursor += width
            result = None
        else:
            d = definitions[callee]
            if len(values) != len(d["params"]):
                raise ValueError("runtime call arity")
            start = cursor
            if callee in active and cursor <= active[callee]:
                raise ValueError("recursive call requires strict input progress")
            previous = active.get(callee)
            active[callee] = start
            env = dict(zip((p["name"] for p in d["params"]), values))
            result = evaluate(d["ast"], env, callee, depth)
            if previous is None:
                del active[callee]
            else:
                active[callee] = previous
        if len(trace) < trace_limit:
            trace.append(["return", callee, cursor, result, depth])
        return result
    def evaluate(term, env, own, depth):
        tick()
        if isinstance(term, str):
            if term == "unit":
                return None
            if term in ("true", "false"):
                return term == "true"
            if term == "eq":
                return cursor + 1 < len(source) and source[cursor] == source[cursor + 1]
            return env[term]
        op, *items = term
        if op == "ref":
            return items[0]
        if op == "call":
            target = own if items[0] == "self" else env.get(items[0], items[0])
            values = tuple(evaluate(t, env, own, depth) for t in items[1:])
            return invoke(target, values, depth + 1)
        if op == "if":
            return evaluate(items[1] if evaluate(items[0], env, own, depth) else items[2], env, own, depth)
        if op == "let":
            val = evaluate(items[1], env, own, depth)
            return evaluate(items[2], {**env, items[0]: val}, own, depth)
        if op == "seq":
            result = None
            for t in items:
                result = evaluate(t, env, own, depth)
            return result
        a = evaluate(items[0], env, own, depth)
        if op == "not":
            return not a
        b = evaluate(items[1], env, own, depth)
        return {"xor": lambda: a != b, "and": lambda: a and b, "or": lambda: a or b}[op]()
    error, result = None, None
    try:
        result = invoke(name, tuple(args), 1)
        if cursor != len(source):
            raise ValueError("top-level procedure left unread input")
    except (ValueError, RecursionError, KeyError, TypeError) as exc:
        error = str(exc)
    return dict(ok=error is None, error=error, output=output, cursor=cursor, result=result,
                steps=steps, visits=dict(visits), max_call_depth=depth_peak, trace=trace)


def schema(library):
    param = {"type": "object", "additionalProperties": False,
             "properties": {"name": {"type": "string"}, "type": {"enum": ["B", "F"], "type": "string"}},
             "required": ["name", "type"]}
    definition = {"type": "object", "additionalProperties": False,
                  "properties": {"name": {"type": "string"},
                                 "params": {"type": "array", "items": param, "maxItems": MAX_ARGUMENTS},
                                 "returns": {"type": "string", "enum": ["B", "U"]},
                                 "body": {"type": "string"}},
                  "required": ["name", "params", "returns", "body"]}
    return {"type": "object", "additionalProperties": False,
            "properties": {"library_hash": {"type": "string", "enum": [library.digest]},
                           "definitions": {"type": "array", "items": definition,
                                           "minItems": 1, "maxItems": MAX_DEFINITIONS}},
            "required": ["library_hash", "definitions"]}
