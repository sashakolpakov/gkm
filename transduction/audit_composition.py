"""Post-freeze exhaustive short-input audit of the composition comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from audit_growth import equality_patterns
from composition_benchmark import reference, sources
from composition_search import OUTPUT_LIMIT, STEP_LIMIT
from modular_machine import Library, Machine, digest


def audit(folder, maximum=8):
    protocol = json.loads((folder / "protocol.json").read_text())
    frozen = json.loads((folder / "frozen.json").read_text())
    tasks = json.loads((folder / "private-tasks.json").read_text())
    if protocol["source_hashes"] != sources() or digest(tasks) != protocol["tasks_hash"]:
        raise ValueError("source or task provenance changed")
    cache, rows = {}, []
    for task in tasks:
        for arm in protocol["arms"]:
            record = json.loads((folder / task["id"] / arm / "selection.json").read_text())
            row = next(r for r in frozen["rows"] if r["task"] == task["id"] and r["arm"] == arm)
            if row != record["row"]:
                raise ValueError("selection differs from freeze")
            if not row["admitted"]:
                rows.append({"task": task["id"], "arm": arm, "audited": False, "reason": "no admitted program"})
                continue
            library = Library.read(record["library"])
            key = library.digest, digest(task["target"])
            if key not in cache:
                machine = Machine(library)
                counts, failures, checksum = [], [], hashlib.sha256()
                for length in range(maximum + 1):
                    correct = total = 0
                    for source in equality_patterns(length):
                        expected = list(reference(task["target"], source))
                        run = machine.run("Q", source, step_limit=STEP_LIMIT, output_limit=OUTPUT_LIMIT)
                        good = run["ok"] and run["output"] == expected
                        total += 1
                        correct += int(good)
                        checksum.update(json.dumps([source, expected, run["ok"], run["output"]], separators=(",", ":")).encode() + b"\n")
                        if not good and len(failures) < 5:
                            failures.append({"source": source, "expected": expected, "execution": run})
                    counts.append({"length": length, "correct": correct, "total": total})
                cache[key] = {"counts": counts, "counterexamples": failures,
                              "exact": all(c["correct"] == c["total"] for c in counts),
                              "replay_sha256": checksum.hexdigest()}
            result = {"task": task["id"], "arm": arm, "audited": True, "library_hash": library.digest, **cache[key]}
            rows.append(result)
            print(json.dumps({k: result[k] for k in ("task", "arm", "exact")}), flush=True)
    return {"maximum_length": maximum, "patterns_per_task": sum(1 for n in range(maximum + 1) for _ in equality_patterns(n)),
            "unique_program_target_pairs": len(cache), "rows": rows,
            "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=8)
    args = parser.parse_args()
    if not 0 <= args.max_length <= 8:
        parser.error("maximum length must be 0..8")
    result = audit(args.output, args.max_length)
    with (args.output / f"equality-audit-{args.max_length}.json").open("x") as stream:
        json.dump(result, stream, indent=2)
