"""Post-freeze shape/continuation audit; never feedback for a proposer."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import sys
from pathlib import Path

from interface_benchmark import (effects_for, encode_tree, materialize, reference,
                                 save, shapes, sources)
from interface_machine import InterfaceLibrary, Machine
from interface_search import (definition, discover, lifted_skeleton, parse,
                              render, substitute)


def audit_derivation(record):
    field = "reuse" if record["row"]["selected_by"] == "host_reuse" else "proposer"
    origin = record["row"]["effort"][field].get("derivation")
    if not origin:
        return None
    library = InterfaceLibrary.read(record["input_library"])
    if origin["kind"] == "retained_call":
        cell = {**library.base.cells, **library.cells}[origin["callee"]]
        call = ("call", origin["callee"], *(parse(s) for s in origin["arguments"]))
        body = ("seq", call, "unit") if library.signatures()[origin["callee"]][1] == "B" else call
        definitions = record["candidate"]["definitions"]
        if len(definitions) != 1 or parse(definitions[0]["body"]) != body or cell["hash"] != origin["callee_hash"]:
            raise ValueError("retained-call binding derivation changed")
        return dict(kind=origin["kind"], callee=origin["callee"], retained_hash_verified=True,
                    bindings_verified=True)
    matches = [a for a in discover(library) if a["sources"] == origin["sources"]]
    if len(matches) != 1:
        raise ValueError("missing source abstraction")
    abstract = matches[0]
    if abstract["specializations"] != origin["exact_base_specializations"]:
        raise ValueError("source specialization proof changed")
    proto = lifted_skeleton(abstract, *origin["signature"])
    if proto["holes"] != origin["holes"] or render(proto["body"]) != origin["prototype"]:
        raise ValueError("dataflow lifting derivation changed")
    filling = [parse(s) for s in origin["filling"]]
    for domain, value in zip(proto["domains"], filling):
        if value not in domain:
            raise ValueError("hole filled outside search grammar")
    recovered = substitute(proto["body"], filling)
    helper, root = record["candidate"]["definitions"]
    if render(recovered) != helper["body"]:
        raise ValueError("mechanical helper differs from derived code")
    call = ("call", helper["name"], *filling[len(proto["domains"]):])
    expected = ("seq", call, "unit") if proto["returns"] == "B" else call
    if parse(root["body"]) != expected:
        raise ValueError("mechanical caller differs from derived bindings")
    return dict(kind=origin["kind"], exact_source_specializations_verified=True,
                typed_dataflow_derivation_verified=True)


def audit(output, maximum_leaves=8, assignments=4):
    protocol = json.loads((output / "protocol.json").read_text())
    if protocol["source_hashes"] != sources():
        raise ValueError("use source archive matching this run")
    frozen = json.loads((output / "frozen.json").read_text())
    forms = [s for n in range(1, maximum_leaves + 1) for s in shapes(n)]
    rng = random.Random(77331)
    trees = [materialize(s, rng, 9000) for s in forms for _ in range(assignments)]
    # Independent pairs exercise calls in a nonterminal position: the first
    # root must return at its region boundary, before the second tree starts.
    small = [materialize(s, rng, 10000) for n in range(1, 5) for s in shapes(n)]
    continuations = [(a, b) for a, b in itertools.product(small, repeat=2)]
    records = []
    for row in frozen["rows"]:
        if not row["admitted"]:
            continue
        folder = output / row["family"] / row["arm"] / f"stage-{row['stage']}"
        selection = json.loads((folder / "selection.json").read_text())
        library = InterfaceLibrary.read(selection["library"])
        root = f"T{row['stage']}"
        machine = Machine(library)
        effects = effects_for(row["family"], row["stage"])
        correct, failures = 0, []
        for tree in trees:
            source = encode_tree(tree)
            expected = reference(tree, row["family"], effects)[0]
            execution = machine.run(root, source)
            good = execution["ok"] and execution["output"] == list(expected)
            correct += good
            if not good and len(failures) < 2:
                failures.append(dict(source=source, expected=expected, actual=execution))
        twice, square = library.attach(definition("AuditTwice", [], "U", ("seq", ("call", root), ("call", root))))
        repeated = Machine(twice)
        continuation_correct = 0
        for a, b in continuations:
            source = encode_tree(a) + encode_tree(b)
            expected = reference(a, row["family"], effects)[0] + reference(b, row["family"], effects)[0]
            execution = repeated.run("AuditTwice", source)
            continuation_correct += execution["ok"] and execution["output"] == list(expected)
        result = dict(family=row["family"], arm=row["arm"], stage=row["stage"],
                      shapes_correct=correct, shapes_total=len(trees),
                      continuations_correct=continuation_correct, continuations_total=len(continuations),
                      derivation=audit_derivation(selection), failures=failures,
                      continuation_attachment_hash=square.graph.digest)
        records.append(result)
    report = dict(auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  maximum_leaves=maximum_leaves, assignments_per_shape=assignments,
                  shape_count=len(forms), rows=records,
                  all_exact=all(r["shapes_correct"] == r["shapes_total"] and
                                r["continuations_correct"] == r["continuations_total"] for r in records))
    save(output / "shape-audit.json", report)
    return report


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.output)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}))
