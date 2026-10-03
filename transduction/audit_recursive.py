"""Post-freeze shape and call/return audits; never proposal feedback."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from recursive_benchmark import (DEFAULT_BINDINGS, encode_tree, materialize, reference,
                                 save, shapes, specification)
from recursive_machine import RecursiveLibrary, node
from recursive_search import brief, score


def audit(output, maximum_leaves=8, assignments=4):
    summary = json.loads((output / "summary.json").read_text())
    frozen = json.loads((output / "frozen.json").read_text())
    if any(row["admitted"] != frozen["rows"][i]["admitted"] for i, row in enumerate(summary["rows"])):
        raise ValueError("admission changed after freeze")
    rng = random.Random(90210)
    trees = [materialize(shape, rng, 3000) for count in range(1, maximum_leaves + 1)
             for shape in shapes(count) for _ in range(assignments)]
    rows = []
    for row in summary["rows"]:
        if not row["admitted"]:
            continue
        family, stage = row["family"], row["stage"]
        folder = output / family / row["arm"] / f"stage-{stage}"
        record = json.loads((folder / "selection.json").read_text())
        library = RecursiveLibrary.read(record["library"])
        root = f"R{stage:02}"
        spec = specification(family)
        pairs = [(encode_tree(tree), reference(tree, spec, stage)) for tree in trees]
        result = score(library, root, pairs, DEFAULT_BINDINGS)
        # This is an evaluator-created wrapper, not an acquired solution or
        # feedback to either proposer. It independently tests prefix boundaries:
        # two calls must consume exactly their own trees from one input stream.
        wrapper = [node("invoke", cell=root, arg=0, a=1, bindings=DEFAULT_BINDINGS),
                   node("invoke", cell=root, arg=0, a=2, bindings=DEFAULT_BINDINGS), node("ret")]
        joined, square = library.attach("AuditTwoCalls", wrapper, spec["bits"])
        two = [(pairs[i][0] + pairs[-i - 1][0], pairs[i][1] + pairs[-i - 1][1])
               for i in range(min(256, len(pairs)))]
        continuation = score(joined, "AuditTwoCalls", two, DEFAULT_BINDINGS)
        item = dict(family=family, stage=stage, arm=row["arm"], shapes=brief(result),
                    two_sequential_prefix_calls=brief(continuation),
                    callback_calls_balanced=all(r["calls"] == r["returns"] for r in result["records"]),
                    continuation_certificate=square.certificate())
        save(folder / "shape-audit.json", item)
        rows.append(item)
    report = dict(maximum_leaves=maximum_leaves, assignments_per_shape=assignments,
                  shapes=sum(len(shapes(i)) for i in range(1, maximum_leaves + 1)),
                  audit_source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  all_exact=all(r["shapes"]["exact"] and r["two_sequential_prefix_calls"]["exact"] for r in rows),
                  admitted_conditions=len(rows), rows=rows)
    save(output / "shape-audit.json", report)
    return {k: v for k, v in report.items() if k != "rows"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--maximum-leaves", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(audit(args.output, args.maximum_leaves), indent=2))
