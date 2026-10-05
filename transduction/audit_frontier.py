"""Independent finite shape and nonterminal-return audit after selection."""
import argparse
import hashlib
import itertools
import json
import random
import sys
from pathlib import Path

from frontier_benchmark import encode, materialize, reference, save, shapes, sources
from interface_machine import InterfaceLibrary, Machine, parse, render
from interface_search import definition


def _specialize_body(cell, kept, fixed, old_names):
    """Conservative syntactic check; fixed recursive arguments must stay fixed.

    Erased actual arguments are literal refs/Booleans, so cannot hide effects.
    No execution, equation guessing, program repair or induction search occurs.
    let is preserved without commuting effects; capture under parameter renaming
    is rejected. Self-references as values remain outside this small check.
    """
    params = [p["name"] for p in cell["params"]]
    env = {**fixed, **{params[i]: name for i, name in zip(kept, old_names)}}
    renamed = {v for v in env.values() if isinstance(v, str)}
    def walk(term, bound=frozenset(params)):
        if isinstance(term, str):
            return env.get(term, term)
        if term[0] == "let":
            variable, value, body = term[1:]
            if variable in bound or variable in renamed:
                raise ValueError("specialization would shadow or capture a variable")
            return ("let", variable, walk(value, bound), walk(body, bound | {variable}))
        if term[0] == "ref" and term[1] == cell["name"]:
            raise ValueError("specialization outside conservative checker")
        if term[0] == "ref":
            return term
        if term[0] == "call":
            target, args = term[1], tuple(walk(a, bound) for a in term[2:])
            if target in ("self", cell["name"]):
                if len(args) != len(params) or any(args[i] != fixed[p] for i, p in enumerate(params) if p in fixed):
                    raise ValueError("recursive specialization is not closed")
                return ("call", "self", *(args[i] for i in kept))
            target = env.get(target, target)
            if isinstance(target, tuple):
                if target[0] != "ref":
                    raise ValueError("non-reference call target")
                target = target[1]
            return ("call", target, *args)
        return (term[0], *(walk(a, bound) for a in term[1:]))
    return walk(parse(cell["body"]))


def exact_extension(old_name, old_cell, new_name, new_cell, signatures):
    """Does fixing extra new arguments reproduce the old recursive AST exactly?

    This is a post-selection interface-extension diagnostic, not immutable-cell
    reuse and not a new categorical attachment certificate.
    """
    old, new = dict(old_cell, name=old_name), dict(new_cell, name=new_name)
    if old["returns"] != new["returns"] or len(new["params"]) <= len(old["params"]):
        return None
    names = [p["name"] for p in old["params"]]
    constants = set()
    def collect(term):
        if isinstance(term, tuple):
            if term[0] in ("call", "ref") and term[1] not in names:
                sig = signatures.get(term[1])
                if sig and not sig[0] and sig[1] == "U":
                    constants.add(("ref", term[1]))
            for child in term[1:]:
                collect(child)
    collect(parse(old["body"]))
    try:
        expected = _specialize_body(old, tuple(range(len(names))), {}, names)
    except ValueError:
        return None
    # Keep argument evaluation order: never commute effectful actual arguments.
    for kept in itertools.combinations(range(len(new["params"])), len(names)):
        if any(old["params"][j]["type"] != new["params"][i]["type"] for j, i in enumerate(kept)):
            continue
        extra = [p for i, p in enumerate(new["params"]) if i not in kept]
        domains = [sorted(constants) if p["type"] == "F" else ["false", "true"] for p in extra]
        for values in itertools.product(*domains):
            fixed = {p["name"]: value for p, value in zip(extra, values)}
            try:
                actual = _specialize_body(new, kept, fixed, names)
            except ValueError:
                continue
            if actual == expected:
                return dict(old=old_name, new=new_name, old_hash=old["hash"], new_hash=new["hash"],
                    retained_arguments={new["params"][i]["name"]: name for i, name in zip(kept, names)},
                    fixed_arguments={name: render(value) for name, value in fixed.items()},
                    recovered_body=render(actual), exact_ast_identity=True,
                    recursive_fixed_arguments_preserved=True,
                    equal_interpreter_step_cost_claimed=False)
    return None


def audit(output, maximum_leaves=8, assignments=4):
    protocol = json.loads((output / "protocol.json").read_text())
    if protocol["source_hashes"] != sources():
        raise ValueError("use frozen source archive")
    frozen = json.loads((output / "frozen.json").read_text())
    rng = random.Random(77331)
    forms = [s for n in range(1, maximum_leaves + 1) for s in shapes(n)]
    trees = [materialize(s, rng, 9000) for s in forms for _ in range(assignments)]
    small = [materialize(s, rng, 10000) for n in range(1, 5) for s in shapes(n)]
    records = []
    for row in frozen["rows"]:
        if not row["admitted"]:
            continue
        folder = output / row["family"] / row["arm"] / f"stage-{row['stage']}"
        record = json.loads((folder / "selection.json").read_text())
        library = InterfaceLibrary.read(record["library"])
        root, family, stage = f"T{row['stage']}", row["family"], row["stage"]
        machine = Machine(library)
        correct, failures = 0, []
        for tree in trees:
            source, expected = encode(tree, family), reference(tree, family, stage)[0]
            execution = machine.run(root, source)
            good = execution["ok"] and execution["output"] == list(expected)
            correct += good
            if not good and len(failures) < 2:
                failures.append(dict(source=source, expected=expected, actual=execution))
        twice, _ = library.attach(definition("AuditTwice", [], "U", ("seq", ("call", root), ("call", root))))
        repeated = Machine(twice)
        continuation_correct = 0
        for a, b in itertools.product(small, repeat=2):
            source = encode(a, family) + encode(b, family)
            expected = reference(a, family, stage)[0] + reference(b, family, stage)[0]
            execution = repeated.run("AuditTwice", source)
            continuation_correct += execution["ok"] and execution["output"] == list(expected)
        extensions = []
        if stage == 2:
            old = InterfaceLibrary.read(record["input_library"])
            base = InterfaceLibrary.read(json.loads((output / "basis/corpus.json").read_text())["library"])
            acquired = old.cells.keys() - base.cells.keys() - {"T1"}
            added = library.cells.keys() - old.cells.keys() - {root}
            for old_name, new_name in itertools.product(sorted(acquired), sorted(added)):
                proof = exact_extension(old_name, old.cells[old_name], new_name,
                                        library.cells[new_name], library.signatures())
                if proof:
                    extensions.append(proof)
        records.append(dict(family=family, arm=row["arm"], stage=stage,
            shapes_correct=correct, shapes_total=len(trees), continuations_correct=continuation_correct,
            continuations_total=len(small) ** 2, failures=failures, interface_extensions=extensions))
    report = dict(auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        maximum_leaves=maximum_leaves, assignments_per_shape=assignments, shape_count=len(forms), rows=records,
        admitted_roots=len(records),
        all_exact=None if not records else all(r["shapes_correct"] == r["shapes_total"] and
                      r["continuations_correct"] == r["continuations_total"] for r in records))
    save(output / "shape-audit.json", report)
    return report


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    result = audit(p.parse_args().output)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))
