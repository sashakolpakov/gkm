"""Fixed-basis synthesis over recursively composed whole-input procedures.

No interpreter or primitive changes: nested pipeline operands are lifted into
ordinary local procedures and attached with the existing certified pushout.
"""
from __future__ import annotations

import json
import resource
import sys
import time
from collections import OrderedDict
from pathlib import Path

from cofibration import verify_certificate
from growth_benchmark import source_hashes as growth_hashes
from modular_machine import Library, Machine, call


CELLS = tuple(f"M{i:02}" for i in range(1, 7))
OPS = ("pipe", "if")
OUTPUT_LIMIT = 4096
STEP_LIMIT = 200000
MAX_LEAVES = 6


def leaves(term):
    return 1 if term[0] == "call" else leaves(term[1]) + leaves(term[2])


def normalize(term):
    """Exact laws for pure whole-input functions, not sample equivalence pruning.

    Right-associate pipelines. If children test the same original input, only
    their reachable branch matters. Identical branches need no outer test.
    No two programs are merged merely because training fingerprints agree.
    """
    if term[0] == "call":
        return term
    op, a, b = term[0], normalize(term[1]), normalize(term[2])
    if op == "pipe":
        if a[0] == "pipe":
            return normalize(("pipe", a[1], ("pipe", a[2], b)))
    elif op == "if":
        if a[0] == "if":
            a = a[1]
        if b[0] == "if":
            b = b[2]
        if a == b:
            return a
    else:
        raise ValueError("unknown constructor")
    return (op, a, b)


def canonical_node(op, a, b):
    if op == "pipe" and a[0] == "pipe":
        return False
    if op == "if" and (a[0] == "if" or b[0] == "if" or a == b):
        return False
    return True


def enumerate_terms(maximum):
    """Increasing leaf count, then operator, split size and retained cell order.

    Layers are materialized only if the next layer is reached. The active layer
    streams; candidate/memory/time limits are imposed by the consuming search.
    """
    if not 1 <= maximum <= MAX_LEAVES:
        raise ValueError("invalid leaf bound")
    layers = {1: [("call", cell) for cell in CELLS]}
    yield from layers[1]
    for count in range(2, maximum + 1):
        current = []
        for op in OPS:
            for left_count in range(1, count):
                for a in layers[left_count]:
                    if op == "pipe" and a[0] == "pipe" or op == "if" and a[0] == "if":
                        continue
                    for b in layers[count - left_count]:
                        if canonical_node(op, a, b):
                            term = (op, a, b)
                            if count < maximum:
                                current.append(term)
                            yield term
        layers[count] = current


def syntax_counts(maximum):
    """Counts before exact algebraic normalization; not a difficulty guarantee."""
    counts = [0, len(CELLS)]
    for n in range(2, maximum + 1):
        counts.append(2 * sum(counts[k] * counts[n - k] for k in range(1, n)))
    return counts[1:]


def encode(term, library_hash):
    nodes = []
    def visit(t):
        if t[0] == "call":
            node = {"op": "call", "cell": t[1], "left": None, "right": None}
        else:
            left, right = visit(t[1]), visit(t[2])
            node = {"op": t[0], "cell": None, "left": left, "right": right}
        nodes.append(node)
        return len(nodes) - 1
    root = visit(term)
    return {"library_hash": library_hash, "nodes": nodes, "root": root}


def decode(payload, library_hash, maximum):
    if not isinstance(payload, dict) or set(payload) != {"library_hash", "nodes", "root"}:
        raise ValueError("invalid proposal fields")
    if payload["library_hash"] != library_hash:
        raise ValueError("proposal changed the fixed library")
    nodes = payload["nodes"]
    if not isinstance(nodes, list) or not 1 <= len(nodes) <= 2 * maximum - 1:
        raise ValueError("node budget exceeded")
    if type(payload["root"]) is not int or payload["root"] != len(nodes) - 1:
        raise ValueError("root must be the last node")
    terms, uses = [], [0] * len(nodes)
    for index, node in enumerate(nodes):
        if not isinstance(node, dict) or set(node) != {"op", "cell", "left", "right"}:
            raise ValueError("invalid node fields")
        if node["op"] == "call":
            if node["cell"] not in CELLS or node["left"] is not None or node["right"] is not None:
                raise ValueError("invalid retained call")
            terms.append(("call", node["cell"]))
        elif node["op"] in OPS:
            if node["cell"] is not None:
                raise ValueError("constructor cell must be null")
            for edge in (node["left"], node["right"]):
                if type(edge) is not int or not 0 <= edge < index:
                    raise ValueError("children must precede their parent")
                uses[edge] += 1
            terms.append((node["op"], terms[node["left"]], terms[node["right"]]))
        else:
            raise ValueError("outside the shared grammar")
    if uses[-1] != 0 or any(count != 1 for count in uses[:-1]):
        raise ValueError("proposal must be one tree, with no unused nodes or hidden DAG sharing")
    if leaves(terms[-1]) > maximum:
        raise ValueError("call budget exceeded")
    return normalize(terms[-1])


def schema(library_hash, maximum):
    nullable_index = {"type": ["integer", "null"], "minimum": 0, "maximum": 2 * maximum - 2}
    node = {"type": "object", "additionalProperties": False,
            "properties": {"op": {"type": "string", "enum": ["call", *OPS]},
                           "cell": {"type": ["string", "null"], "enum": [*CELLS, None]},
                           "left": nullable_index, "right": nullable_index},
            "required": ["op", "cell", "left", "right"]}
    return {"type": "object", "additionalProperties": False,
            "properties": {"library_hash": {"type": "string", "enum": [library_hash]},
                           "nodes": {"type": "array", "items": node, "minItems": 1, "maxItems": 2 * maximum - 1},
                           "root": {"type": "integer", "minimum": 0, "maximum": 2 * maximum - 2}},
            "required": ["library_hash", "nodes", "root"]}


def load_basis(folder):
    """Rebuild the old common acquisition, not any of its later task solutions."""
    protocol = json.loads((folder / "protocol.json").read_text())
    if protocol["source_hashes"] != growth_hashes():
        raise ValueError("source of the original acquisition changed")
    library = Library.empty()
    paths = [folder / "s1-acquisition" / f"F{i:03}.json" for i in range(1, 6)]
    paths += [folder / f"s1-{i:02}-mechanical" / "selection.json" for i in range(1, 7)]
    for path in paths:
        record = json.loads(path.read_text())
        rebuilt = verify_certificate(record["certificate"], library.graph)
        library = Library.read(record["library"])
        if library.graph != rebuilt.graph or "row" in record and not record["row"]["admitted"]:
            raise ValueError("invalid basis provenance")
    if set(library.cells) != {*(f"F{i:03}" for i in range(1, 6)), *CELLS}:
        raise ValueError("unexpected extra basis cells")
    return library


def attach_term(library, term, root="Q"):
    """Mechanical procedure lifting, with no choice of behavior by the compiler.

    The old interpreter accepts pipelines between named whole procedures. For
    a nested operand we name that exact subtree and use its unchanged body.
    Helpers are private to this candidate; the whole joint object is promoted.
    """
    term = normalize(term)
    certificates = []
    counter = 0
    def named(t):
        nonlocal library, counter
        if t[0] == "call":
            return call(t[1])
        body = lower(t)
        name = f"{root}H{counter:02}"
        counter += 1
        library, square = library.attach(name, body)
        certificates.append({"cell": name, "certificate": square.certificate()})
        return call(name)
    def lower(t):
        if t[0] == "call":
            return call(t[1])
        children = ([named(t[1]), named(t[2])] if t[0] == "pipe"
                    else [lower(t[1]), lower(t[2])])
        return {"op": t[0], "children": children}
    body = lower(term)
    library, square = library.attach(root, body)
    certificates.append({"cell": root, "certificate": square.certificate()})
    library.verify()
    return library, certificates


class FastEvaluator:
    """Pure-functional search evaluator, checked against the graph interpreter.

    Only retained calls are memoized by exact (cell, input). No equivalence of
    different programs is inferred from their answers on the original samples:
    a pipeline may feed a later program entirely different intermediate inputs.
    """
    def __init__(self, library, cache_limit=10000):
        self.machine = Machine(library)
        self.cache = OrderedDict()
        self.cache_limit = cache_limit
        self.base_executions = 0
        self.hits = 0

    def run(self, term, source):
        source = tuple(source)
        if term[0] == "call":
            key = term[1], source
            if key in self.cache:
                self.hits += 1
                self.cache.move_to_end(key)
                return self.cache[key]
            result = self.machine.run(term[1], source, step_limit=STEP_LIMIT, output_limit=OUTPUT_LIMIT)
            self.base_executions += 1
            output = tuple(result["output"]) if result["ok"] else None
            self.cache[key] = output
            if len(self.cache) > self.cache_limit:
                self.cache.popitem(last=False)
            return output
        if term[0] == "if":
            same = len(source) >= 2 and source[0] == source[1]
            return self.run(term[1 if same else 2], source)
        middle = self.run(term[1], source)
        return None if middle is None else self.run(term[2], middle)

    def score(self, term, pairs, fail_fast=False):
        correct = tested = 0
        failures = []
        for source, expected in pairs:
            output = self.run(term, source)
            tested += 1
            good = output == tuple(expected)
            correct += int(good)
            if not good:
                if len(failures) < 4:
                    failures.append({"input": source, "expected": expected, "actual": output})
                if fail_fast:
                    break
        return {"correct": correct, "total": len(pairs), "tested": tested,
                "exact": correct == len(pairs), "counterexamples": failures}


def rss_mib():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024 if sys.platform == "darwin" else 1024)


def mechanical(library, pairs, maximum, *, budget=10000000, seconds=120, memory_mib=512):
    evaluator = FastEvaluator(library)
    started = time.monotonic()
    count = 0
    layers = {}
    reason = "grammar_exhausted"
    selected = None
    for term in enumerate_terms(maximum):
        if count >= budget:
            reason = "candidate_limit"
            break
        if time.monotonic() - started >= seconds:
            reason = "time_limit"
            break
        if count % 256 == 0 and rss_mib() > memory_mib:
            reason = "memory_limit"
            break
        count += 1
        size = leaves(term)
        layers[size] = layers.get(size, 0) + 1
        score = evaluator.score(term, pairs, fail_fast=True)
        if score["exact"]:
            selected, reason = term, "training_exact"
            break
    return selected, {"evaluated": count, "seconds": time.monotonic() - started,
                      "stop_reason": reason, "layers": layers,
                      "base_executions": evaluator.base_executions, "cache_hits": evaluator.hits,
                      "cache_entries": len(evaluator.cache), "process_peak_mib": rss_mib()}


def graph_score(library, root, pairs, trace_limit=0):
    machine, records = Machine(library), []
    for source, expected in pairs:
        result = machine.run(root, source, step_limit=STEP_LIMIT, output_limit=OUTPUT_LIMIT, trace_limit=trace_limit)
        records.append({"source": list(source), "expected": list(expected), **result,
                        "exact": result["ok"] and result["output"] == list(expected)})
    correct = sum(record["exact"] for record in records)
    return {"correct": correct, "total": len(records), "exact": correct == len(records), "records": records}


def prompt(library, pairs, maximum, feedback=None):
    return {
        "instructions": (
            "Infer the opaque-token transformation from training examples and the fixed retained library. "
            "Return ONLY the JSON program, with at most the stated number of Call leaves. No tools. "
            "The shared grammar is recursive, not a list of fixed templates: T = Call(cell) | Pipe(T,T) | If(T,T). "
            "A call may target only M01..M06. Pipe(a,b) runs a on the entire input then b on a's output. "
            "If(a,b) runs a when the next TWO input tokens exist and are equal, otherwise b; the selected "
            "branch receives the unchanged whole input. Nest these constructors freely. No new primitive "
            "instructions, token constants, altered retained bodies or extra operations are permitted. "
            "Output a postorder node array: each child index is less than its parent's index; the last node "
            "is root. A call has its cell ID and null left/right; a constructor has null cell and two child "
            "indices. Every non-root node must be used exactly once; no unused nodes or hidden DAG sharing. "
            "Library body conventions: seq executes its children in order on the same cursor/output; "
            "while repeats its child while at least n tokens remain; fragments consume their declared "
            "width and RETURN, with fresh registers and a bounded input window. Whole procedures consume "
            "all remaining input. Fragment actions: 0=advance; 1=append current token; 10/11=store current "
            "token in R0/R1; 30/31=append R0/R1. Missing tokens/empty registers produce no output. "
            "Prefer a small correct composition. You have at most two proposals within one shared wall "
            "budget; a retry receives training counterexamples only. You never receive the target program, "
            "validation or hidden data. Pipeline associativity and redundant identical-input branches "
            "are normalized identically for both proposers."
        ),
        "max_call_leaves": maximum, "library_hash": library.digest,
        "library": {name: {k: cell[k] for k in ("kind", "width", "program", "hash", "dependencies")}
                    for name, cell in library.cells.items()},
        "training_examples": pairs, "previous_training_feedback": feedback,
    }
