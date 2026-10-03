"""Post-freeze audit over every token-equality pattern up to a bounded length.

Restricted-growth strings represent set partitions, so (given this language's
token-renaming equivariance) these cover all token identities up to renaming.
No results are returned to either proposer. Identical frozen libraries share
audit work, with their hashes recorded explicitly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from growth_benchmark import source_hashes, target
from modular_machine import Library, Machine


def equality_patterns(length):
    if length == 0:
        yield ()
        return

    def extend(prefix, largest):
        if len(prefix) == length:
            yield prefix
            return
        for token in range(largest + 2):
            yield from extend(prefix + (token,), max(largest, token))

    yield from extend((0,), 0)


def audit(folder, maximum=8):
    protocol = json.loads((folder / "protocol.json").read_text())
    frozen = json.loads((folder / "frozen.json").read_text())
    if source_hashes() != protocol["source_hashes"]:
        raise ValueError("experiment source changed")
    cache, rows = {}, []
    for seed in protocol["seeds"]:
        for arm in protocol["arms"]:
            record = json.loads((folder / f"s{seed}-{arm}-frozen.json").read_text())
            library = Library.read(record["library"])
            if library.digest != frozen["libraries"][f"{seed}-{arm}"]:
                raise ValueError("library differs from selection freeze")
            machine = Machine(library)
            for cell, task, _ in record["admitted"]:
                key = (library.digest, cell, task)
                if key not in cache:
                    counts, failures, checksum = [], [], hashlib.sha256()
                    for length in range(maximum + 1):
                        count = correct = 0
                        for source in equality_patterns(length):
                            result = machine.run(cell, source)
                            expected = list(target(task, source))
                            good = result["ok"] and result["output"] == expected
                            count += 1
                            correct += int(good)
                            checksum.update(json.dumps([source, expected, result["ok"], result["output"]], separators=(",", ":")).encode() + b"\n")
                            if not good and len(failures) < 5:
                                failures.append({"source": source, "expected": expected, "execution": result})
                        counts.append({"length": length, "correct": correct, "total": count})
                    cache[key] = {"counts": counts, "counterexamples": failures, "replay_sha256": checksum.hexdigest(),
                                  "exact": all(c["correct"] == c["total"] for c in counts)}
                row = {"seed": seed, "arm": arm, "cell": cell, "task": task,
                       "library_hash": library.digest, **cache[key]}
                rows.append(row)
                print(json.dumps({k: row[k] for k in ("seed", "arm", "cell", "exact")}), flush=True)
    return {"maximum_length": maximum, "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "patterns_per_root": sum(1 for n in range(maximum + 1) for _ in equality_patterns(n)),
            "unique_library_root_tasks": len(cache), "rows": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=8)
    args = parser.parse_args()
    if not 0 <= args.max_length <= 8:
        parser.error("maximum length must be 0..8")
    result = audit(args.output, args.max_length)
    destination = args.output / f"equality-audit-{args.max_length}.json"
    # Exclusive creation: never silently replace a completed audit.
    with destination.open("x") as stream:
        json.dump(result, stream, indent=2)
