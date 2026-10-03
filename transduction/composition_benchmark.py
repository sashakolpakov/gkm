"""A/B difficulty ladder: fixed acquired basis, increasingly broad composition."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

from codex_glue import MODEL, extract, propose
from cofibration import verify_certificate
from composition_search import (CELLS, MAX_LEAVES, OUTPUT_LIMIT, STEP_LIMIT, FastEvaluator,
                                attach_term, decode, encode, graph_score, leaves, load_basis,
                                mechanical, normalize, prompt, schema, syntax_counts)
from growth_benchmark import FRAGMENTS, TASKS, dataset, target
from modular_machine import Library, Machine, digest, structural_metrics, unshare


DEFAULT_BASIS = Path("output/transduction_growth/20261003-returning-ab")
FAMILIES = ((7, "pipeline"), (19, "mixed"))
BOUNDS = (2, 3, 4, 5, 6)


def sources():
    names = ("composition_benchmark.py", "composition_search.py", "codex_glue.py", "cofibration.py",
             "modular_machine.py", "growth_benchmark.py", "glue_ab.py", "pattern_fsa.py",
             "run_cofibration_experiment.py")
    return {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names}


def save(path, value):
    path.write_text(json.dumps(value, indent=2))


def reference(term, source):
    """Independent IO oracle using reference transformations, not Machine.run."""
    source = tuple(source)
    if term[0] == "call":
        return target(TASKS[CELLS.index(term[1])], source)
    if term[0] == "pipe":
        return reference(term[2], reference(term[1], source))
    same = len(source) >= 2 and source[0] == source[1]
    return reference(term[1 if same else 2], source)


def maximum_lengths(term, length):
    """Conservative resource screen independent of any held-out examples."""
    if term[0] == "call":
        out = 2 * length if term[1] == "M02" else (length + 1) // 2 if term[1] == "M05" else length
        return out, max(length, out)
    if term[0] == "pipe":
        middle, peak_a = maximum_lengths(term[1], length)
        output, peak_b = maximum_lengths(term[2], middle)
        return output, max(peak_a, peak_b)
    a, b = maximum_lengths(term[1], length), maximum_lengths(term[2], length)
    return max(a[0], b[0]), max(a[1], b[1])


def leaf_paths(term, prefix=()):
    if term[0] == "call":
        yield prefix
    else:
        yield from leaf_paths(term[1], prefix + (1,))
        yield from leaf_paths(term[2], prefix + (2,))


def replace_leaf(term, path):
    if not path:
        return ("call", "M01")  # identity intervention in the reference task
    pieces = list(term)
    pieces[path[0]] = replace_leaf(pieces[path[0]], path[1:])
    return tuple(pieces)


def make_task(seed, family, bound):
    """Sample before proposals; never select tasks using an A/B outcome.

    Reject only syntactically reduced size, unsafe expansion, equality to a
    single basis call on training inputs, and leaves with no observable training
    effect under an identity intervention. These are not minimality proofs.
    """
    rng = random.Random(seed * 1009 + bound)
    inputs = [source for source, _ in dataset("copy_all", seed, "train")]
    def sample(count):
        if count == 1:
            return ("call", rng.choice(CELLS[1:]))  # target leaves omit identity padding
        split = 1 if family == "pipeline" else rng.randrange(1, count)
        op = "pipe" if family == "pipeline" else rng.choice(("pipe", "if"))
        return op, sample(split), sample(count - split)
    for attempt in range(1, 4001):
        term = normalize(sample(bound))
        if leaves(term) != bound or maximum_lengths(term, 512)[1] > OUTPUT_LIMIT:
            continue
        if family == "mixed" and '"if"' not in json.dumps(term):
            continue
        outputs = [reference(term, source) for source in inputs]
        if any(outputs == [reference(("call", cell), source) for source in inputs] for cell in CELLS):
            continue
        if any(all(reference(replace_leaf(term, path), source) == expected
                   for source, expected in zip(inputs, outputs)) for path in leaf_paths(term)):
            continue
        return {"id": f"{family}-s{seed}-l{bound}", "seed": seed, "family": family,
                "max_leaves": bound, "target": term, "generation_attempts": attempt,
                "generation_note": "witness size, not a proven minimum program size"}
    raise RuntimeError("fixed task-generation bound exhausted")


def pairs_for(task, split):
    pairs = [(source, reference(task["target"], source))
             for source, _ in dataset("copy_all", task["seed"], split)]
    # Same training order for both arms; long examples usually expose errors
    # sooner than the repeated empty-input examples. No target-specific ranking.
    return sorted(pairs, key=lambda pair: len(pair[0]), reverse=True) if split == "train" else pairs


def brief(score):
    return {key: score[key] for key in ("correct", "total", "exact")}


def basis_retention(library, seed):
    results = {}
    for index, (name, width) in enumerate(FRAGMENTS, 1):
        cell = f"F{index:03}"
        results[cell] = brief(graph_score(library, cell, dataset(name, seed, "validation", width)))
    for cell, task in zip(CELLS, TASKS):
        results[cell] = brief(graph_score(library, cell, dataset(task, seed, "validation")))
    return results


def codex_search(library, pairs, maximum, folder, seconds):
    started = time.monotonic()
    attempts = []
    feedback = None
    chosen = None
    evaluator = FastEvaluator(library)
    reason = "proposal_limit"
    for index in range(1, 3):
        remaining = seconds - (time.monotonic() - started)
        if remaining < 2:
            reason = "time_limit"
            break
        record = {}
        try:
            payload = propose(prompt(library, pairs, maximum, feedback), folder / f"proposal-{index}",
                              timeout=remaining, memory_mib=768, response_schema=schema(library.digest, maximum))
            record["payload"] = payload
            term = decode(payload, library.digest, maximum)
            score = evaluator.score(term, pairs)
            record.update(normalized=encode(term, library.digest), training=score)
            if score["exact"]:
                chosen, reason = term, "training_exact"
                attempts.append(record)
                break
            feedback = {"proposal": payload, "counterexamples": score["counterexamples"]}
        except (ValueError, RuntimeError) as exc:
            record["error"] = str(exc)
            if "payload" not in record:
                attempts.append(record)
                reason = "transport_or_resource_failure"
                break
            feedback = {"proposal": record["payload"], "error": str(exc)}
        attempts.append(record)
    return chosen, {"seconds": time.monotonic() - started, "stop_reason": reason,
                    "proposals": len(attempts), "attempts": attempts}


def evaluate_frozen(output, library, tasks, rows):
    # These private reference programs are never sent to either proposer. After
    # the freeze, verify that even unsolved tasks have an executable witness in
    # the SAME grammar, basis, interpreter and resource envelope.
    for task in tasks:
        witness, certificates = attach_term(library, task["target"], root="W")
        checks = {split: graph_score(witness, "W", pairs_for(task, split))
                  for split in ("train", "validation", "hidden", "stress")}
        save(output / task["id"] / "private-witness-check.json",
             {"candidate": encode(task["target"], library.digest), "library": witness.payload(),
              "certificates": certificates, "scores": checks,
              "all_exact": all(s["exact"] for s in checks.values())})
    for row in rows:
        folder = output / row["task"] / row["arm"]
        task = next(t for t in tasks if t["id"] == row["task"])
        if not row["admitted"]:
            for split in ("hidden", "stress"):
                row[split] = {"correct": 0, "total": len(pairs_for(task, split)), "exact": False,
                              "reason": "no admitted program"}
            continue
        selection = json.loads((folder / "selection.json").read_text())
        built = Library.read(selection["library"])
        copied, copied_entry = unshare(built, "Q")
        controls = {"library": copied.payload(), "entry": copied_entry, "scores": {}}
        for split in ("hidden", "stress"):
            pairs = pairs_for(task, split)
            score = graph_score(built, "Q", pairs, trace_limit=32)
            other = graph_score(copied, copied_entry, pairs)
            if any((a["output"], a["ok"], a["cursor"]) != (b["output"], b["ok"], b["cursor"])
                   for a, b in zip(score["records"], other["records"])):
                raise ValueError("unshared control changed semantics")
            row[split] = brief(score)
            row[split + "_maxima"] = {key: max(r[key] for r in score["records"]) for key in
                                      ("steps", "calls", "returns", "max_call_depth", "max_buffer")}
            controls["scores"][split] = other
            save(folder / f"{split}.json", score)
        row["unshared_control_matches"] = True
        save(folder / "unshared-control.json", controls)
    save(output / "summary.json", {"rows": rows})


def run(output, basis_folder=DEFAULT_BASIS, *, bounds=BOUNDS, families=FAMILIES,
        arms=("mechanical", "codex"), budget=10000000, seconds=120):
    output.mkdir(parents=True, exist_ok=False)
    library = load_basis(basis_folder)
    tasks = [make_task(seed, family, bound) for bound in bounds for seed, family in families]
    protocol = {"version": 1, "source_hashes": sources(), "basis_folder": str(basis_folder.resolve()),
                "basis_hash": library.digest, "grammar": "T = Call(M01..M06) | Pipe(T,T) | IfNextEqual(T,T)",
                "bounds": bounds, "families": families, "arms": arms, "tasks_hash": digest(tasks),
                "syntax_counts_before_normalization": syntax_counts(max(bounds)),
                "seconds_per_arm_condition": seconds, "mechanical_candidate_limit": budget,
                "mechanical_peak_memory_stop_mib": 512, "model_proposals_per_condition": 2,
                "model": MODEL, "model_process_group_memory_stop_mib": 768,
                "output_limit": OUTPUT_LIMIT, "execution_step_limit": STEP_LIMIT,
                "selection": "first training-exact program, then fixed validation and retention gates",
                "hidden_policy": "freeze every choice before scoring any hidden or stress example",
                "task_generation": "fixed random seeds; no A/B-outcome-based filtering",
                "basis_policy": "same frozen library on every task; no cross-task promotion shortcuts"}
    save(output / "protocol.json", protocol)
    save(output / "private-tasks.json", tasks)
    save(output / "basis.json", library.payload())
    # Check only training consistency of the separate evaluator before synthesis.
    evaluator = FastEvaluator(library)
    for task in tasks:
        if not evaluator.score(task["target"], pairs_for(task, "train"))["exact"]:
            raise ValueError("task oracle and fixed acquired basis disagree on training data")
    rows = []
    for index, task in enumerate(tasks):
        for arm in (arms if index % 2 == 0 else tuple(reversed(arms))):
            folder = output / task["id"] / arm
            folder.mkdir(parents=True)
            train = pairs_for(task, "train")
            if arm == "mechanical":
                term, effort = mechanical(library, train, task["max_leaves"], budget=budget, seconds=seconds)
            else:
                term, effort = codex_search(library, train, task["max_leaves"], folder, seconds)
            row = {"task": task["id"], "arm": arm, "family": task["family"], "seed": task["seed"],
                   "max_leaves": task["max_leaves"], "admitted": False, "effort": effort,
                   "proposal_leaves": leaves(term) if term else None}
            receipt = {"row": row, "candidate": encode(term, library.digest) if term else None}
            if term:
                built, certificates = attach_term(library, term)
                previous = library.graph
                for record in certificates:
                    square = verify_certificate(record["certificate"], previous)
                    previous = square.graph
                if previous != built.graph:
                    raise ValueError("compiled attachment differs from certified quotient")
                training = graph_score(built, "Q", train, trace_limit=32)
                if not training["exact"]:
                    # Never silently score a fast-evaluator false positive as a success.
                    row["compiler_gate_failure"] = True
                validation = graph_score(built, "Q", pairs_for(task, "validation"), trace_limit=32)
                retention = basis_retention(built, task["seed"])
                row.update(training=brief(training), validation=brief(validation),
                           retention_exact=all(s["exact"] for s in retention.values()),
                           new_cells=len(certificates), structure=structural_metrics(built, "Q"))
                row["admitted"] = training["exact"] and validation["exact"] and row["retention_exact"]
                receipt.update(library=built.payload(), certificates=certificates, training=training,
                               validation=validation, retention=retention)
            save(folder / "selection.json", receipt)
            rows.append(row)
            print(json.dumps({key: row[key] for key in ("task", "arm", "admitted", "proposal_leaves")}
                             | {"effort": {k: v for k, v in effort.items() if k != "attempts"}}), flush=True)
    save(output / "frozen.json", {"basis_hash": library.digest, "tasks_hash": digest(tasks), "rows": rows})
    evaluate_frozen(output, library, tasks, rows)
    return rows


def replay(output, reproduce_search=False):
    protocol = json.loads((output / "protocol.json").read_text())
    if sources() != protocol["source_hashes"]:
        raise ValueError("source hashes differ from frozen protocol")
    library = load_basis(Path(protocol["basis_folder"]))
    if library.digest != protocol["basis_hash"] or Library.read(json.loads((output / "basis.json").read_text())).digest != library.digest:
        raise ValueError("basis changed")
    tasks = [make_task(seed, family, bound) for bound in protocol["bounds"] for seed, family in protocol["families"]]
    if digest(tasks) != protocol["tasks_hash"] or digest(json.loads((output / "private-tasks.json").read_text())) != digest(tasks):
        raise ValueError("task generation not reproducible")
    frozen = json.loads((output / "frozen.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    if frozen["basis_hash"] != library.digest or frozen["tasks_hash"] != digest(tasks):
        raise ValueError("invalid selection freeze")
    verified = matched = rejected = 0
    for task in tasks:
        private = json.loads((output / task["id"] / "private-witness-check.json").read_text())
        witness, certificates = attach_term(library, task["target"], root="W")
        if (json.loads(json.dumps(certificates)) != private["certificates"]
                or witness.digest != Library.read(private["library"]).digest):
            raise ValueError("private expressibility witness changed")
        for split in ("train", "validation", "hidden", "stress"):
            if graph_score(witness, "W", pairs_for(task, split)) != private["scores"][split]:
                raise ValueError("private expressibility replay differs")
        for arm in protocol["arms"]:
            folder = output / task["id"] / arm
            record = json.loads((folder / "selection.json").read_text())
            row = record["row"]
            frozen_row = next(r for r in frozen["rows"] if r["task"] == task["id"] and r["arm"] == arm)
            if row != frozen_row:
                raise ValueError("selection changed after freeze")
            if arm == "codex":
                feedback, evaluator = None, FastEvaluator(library)
                for index, attempt in enumerate(row["effort"]["attempts"], 1):
                    call_folder = folder / f"proposal-{index}"
                    request = json.loads((call_folder / "request.json").read_text())
                    if request != json.loads(json.dumps(prompt(library, pairs_for(task, "train"), task["max_leaves"], feedback))):
                        raise ValueError("model received an unintended request")
                    if "payload" not in attempt:
                        continue
                    events = [json.loads(line) for line in (call_folder / "events.jsonl").read_text().splitlines() if line]
                    original, _ = extract(events)
                    if original != attempt["payload"]:
                        raise ValueError("model proposal was replaced")
                    if "training" in attempt:
                        term = decode(original, library.digest, task["max_leaves"])
                        score = evaluator.score(term, pairs_for(task, "train"))
                        if json.loads(json.dumps(score)) != attempt["training"]:
                            raise ValueError("model feedback was not the recorded training result")
                        feedback = {"proposal": original, "counterexamples": score["counterexamples"]}
                        if score["exact"] and record["candidate"] != encode(term, library.digest):
                            raise ValueError("selected candidate was not the successful model answer")
                    else:
                        feedback = {"proposal": original, "error": attempt["error"]}
            elif reproduce_search:
                term, effort = mechanical(library, pairs_for(task, "train"), task["max_leaves"],
                                          budget=protocol["mechanical_candidate_limit"], seconds=protocol["seconds_per_arm_condition"])
                candidate = encode(term, library.digest) if term else None
                if candidate != record["candidate"]:
                    raise ValueError("mechanical result not reproduced within bounds")
                if term and effort["evaluated"] != row["effort"]["evaluated"]:
                    raise ValueError("mechanical candidate order changed")
            if not record["candidate"]:
                rejected += 1
                continue
            term = decode(record["candidate"], library.digest, task["max_leaves"])
            built, certificates = attach_term(library, term)
            if json.loads(json.dumps(certificates)) != record["certificates"] or built.digest != Library.read(record["library"]).digest:
                raise ValueError("lowered program or certificate changed")
            previous = library.graph
            for certificate in certificates:
                previous = verify_certificate(certificate["certificate"], previous).graph
                verified += 1
            for field, split in (("training", "train"), ("validation", "validation")):
                if graph_score(built, "Q", pairs_for(task, split), trace_limit=32) != record[field]:
                    raise ValueError("admission replay differs")
            if basis_retention(built, task["seed"]) != record["retention"]:
                raise ValueError("retention replay differs")
            if row["admitted"] != (record["training"]["exact"] and record["validation"]["exact"] and all(s["exact"] for s in record["retention"].values())):
                raise ValueError("incorrect admission decision")
            if not row["admitted"]:
                rejected += 1
                continue
            copied, copied_entry = unshare(built, "Q")
            controls = json.loads((folder / "unshared-control.json").read_text())
            if copied.digest != Library.read(controls["library"]).digest or copied_entry != controls["entry"]:
                raise ValueError("unshared control changed")
            final_row = next(r for r in summary["rows"] if r["task"] == task["id"] and r["arm"] == arm)
            for split in ("hidden", "stress"):
                score = graph_score(built, "Q", pairs_for(task, split), trace_limit=32)
                if score != json.loads((folder / f"{split}.json").read_text()) or brief(score) != final_row[split]:
                    raise ValueError("hidden replay differs")
                if graph_score(copied, copied_entry, pairs_for(task, split)) != controls["scores"][split]:
                    raise ValueError("unshared replay differs")
            matched += 1
    result = {"verified_pushouts": verified, "admitted_conditions_replayed": matched,
              "nonadmitted_conditions": rejected, "source_hashes_match": True,
              "task_generation_reproduced": True, "mechanical_search_reproduced": reproduce_search}
    save(output / "verification.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--basis", type=Path, default=DEFAULT_BASIS)
    parser.add_argument("--arms", nargs="+", choices=("mechanical", "codex"), default=["mechanical", "codex"])
    parser.add_argument("--bounds", nargs="+", type=int, default=list(BOUNDS))
    parser.add_argument("--budget", type=int, default=10000000)
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--reproduce-search", action="store_true")
    args = parser.parse_args()
    if (any(not 2 <= n <= MAX_LEAVES for n in args.bounds) or len(set(args.bounds)) != len(args.bounds)
            or not 1 <= args.budget <= 10000000 or not 1 <= args.seconds <= 120):
        parser.error("invalid resource or grammar bound")
    if args.replay:
        print(json.dumps(replay(args.output, args.reproduce_search)))
    else:
        run(args.output, args.basis, bounds=tuple(args.bounds), arms=tuple(args.arms), budget=args.budget, seconds=args.seconds)
