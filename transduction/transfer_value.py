"""Prospective transfer assay and matched fresh-library description controls.

No new helper is manufactured by this harness. Retained arms enumerate the
existing single-call adapter grammar; cold controls use the existing proposer.
The private tree oracle supplies examples and audits, never proposal code.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import sys
import tarfile
from pathlib import Path

from audit_frontier import exact_extension
from codex_glue import MODEL, extract
from cofibration import verify_certificate
import frontier_benchmark as frontier
from interface_benchmark import (MEMORY_MIB, local_effect, save, search_process,
                                 seed_library)
from interface_machine import InterfaceLibrary, Machine, attach_proposal, parse, size
from interface_search import GROWTH_CONTRACT, brief, definition, prompt, reuse_search, score
from modular_machine import digest


# The three unused leaf operations in the fixed five-operation basis, in order.
# Earlier tasks copied and duplicated leaves; all payload pairs remain swapped.
TASKS = ((3, 1), (4, 3), (5, 4))
SOURCES = (*frontier.SOURCES, "audit_frontier.py", "transfer_value.py")


def sources():
    return {n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in SOURCES}


def reference(tree, effect):
    if "pair" in tree:
        return local_effect(tree["pair"], effect)
    return (local_effect(tree["pre"], 1) + reference(tree["left"], effect)
            + local_effect(tree["mid"], 1) + reference(tree["right"], effect)
            + local_effect(tree["post"], 1))


def pairs_for(effect, split, seed):
    offset, pool = {"train": (7, 3000), "validation": (101, 4000),
                    "hidden": (503, 5000), "stress": (997, 6000)}[split]
    rng = random.Random(seed * 1009 + offset)
    if split in ("train", "validation"):
        forms = [s for n in range(1, 5) for s in frontier.shapes(n) for _ in range(2)]
        forms += [frontier.random_shape(5, rng) for _ in range(8)]
    elif split == "hidden":
        forms = [frontier.random_shape(n, rng) for n in (6, 9, 16, 32, 64) for _ in range(8)]
    else:
        forms = [frontier.comb(n, side) for n in (16, 32, 64) for side in ("left", "right", "alternating")]
        forms += [frontier.random_shape(128, rng) for _ in range(4)]
    trees = [frontier.materialize(form, rng, pool) for form in forms]
    return [(frontier.encode(tree, "framed"), reference(tree, effect)) for tree in trees]


def node_cost(raw):
    return sum(size(parse(d["body"])) for d in raw["definitions"])


def certificate_replay(library, raw, expected_library, certificates, root):
    built, actual = attach_proposal(library, raw, root)
    if digest(actual) != digest(certificates) or built.digest != InterfaceLibrary.read(expected_library).digest:
        raise ValueError("attachment or immutable library changed")
    graph = library.graph
    for cert in certificates:
        graph = verify_certificate(cert, graph).graph
    return built


def prior_inputs(prior):
    """Import every planned trial, checking its two actual admitted attachments."""
    manifest = json.loads((prior / "trials.json").read_text())
    plan = manifest["plan"]
    if digest(plan) != manifest["hash"] or plan["source_hashes"] != frontier.sources():
        raise ValueError("prior experiment sources/manifest changed")
    result = []
    for trial in plan["trials"]:
        path = prior / trial["directory"]
        base = InterfaceLibrary.read(json.loads((path / "basis/corpus.json").read_text())["library"])
        previous, stages = base, []
        frozen = json.loads((path / "frozen.json").read_text())
        if digest(frozen["rows"]) != frozen["hash"]:
            raise ValueError("prior frozen selections changed")
        for stage in (1, 2):
            record = json.loads((path / f"framed/codex/stage-{stage}/selection.json").read_text())
            if not record["row"]["admitted"] or record["row"] not in frozen["rows"]:
                raise ValueError("this fixed study requires both earlier tasks admitted")
            if InterfaceLibrary.read(record["input_library"]).digest != previous.digest:
                raise ValueError("prior lineage changed")
            built = certificate_replay(previous, record["candidate"], record["library"],
                                       record["certificates"], f"T{stage}")
            for split in ("train", "validation"):
                if score(built, f"T{stage}", frontier.pairs_for("framed", stage, split, trial["seed"]),
                         trace_limit=100) != record["checks"][split]:
                    raise ValueError("prior admission changed")
            stages.append(record)
            previous = built
        before = InterfaceLibrary.read(stages[0]["library"])
        after = InterfaceLibrary.read(stages[1]["library"])
        extensions = []
        for old, new in itertools.product(stages[0]["row"]["new_helpers"], stages[1]["row"]["new_helpers"]):
            proof = exact_extension(old, before.cells[old], new, after.cells[new], after.signatures())
            if proof:
                extensions.append(proof)
        result.append(dict(trial=trial, base=base.payload(), stages=stages, extensions=extensions))
    return result


def schedule(inputs, seeds):
    rows = []
    for seed in seeds:
        # Cold tasks are independent; an earlier cold failure blocks no later task.
        for stage, effect in TASKS:
            rows.append(dict(arm="cold", seed=seed, stage=stage, effect=effect,
                             lineage=None, directory=f"seed-{seed}/cold/task-{stage}"))
        for item in inputs:
            lineage = item["trial"]["directory"]
            for arm in ("before", "after"):
                for stage, effect in TASKS:
                    rows.append(dict(arm=arm, seed=seed, stage=stage, effect=effect,
                        lineage=lineage, directory=f"seed-{seed}/{lineage}/{arm}/task-{stage}"))
    return rows


def retention(library, history):
    return {h["root"]: brief(score(library, h["root"], h["validation"])) for h in history}


def initial_state(row, inputs, base):
    if row["arm"] == "cold":
        return base, []
    item = next(x for x in inputs if x["trial"]["directory"] == row["lineage"])
    count = 1 if row["arm"] == "before" else 2
    library = InterfaceLibrary.read(item["stages"][count - 1]["library"])
    history = [dict(root=f"T{stage}", validation=frontier.pairs_for(
        "framed", stage, "validation", item["trial"]["seed"])) for stage in range(1, count + 1)]
    return library, history


def run(output, prior, seeds=(21, 22, 23), seconds=300, replies=4):
    if not seeds or len(set(seeds)) != len(seeds) or seconds <= 0 or replies < 1:
        raise ValueError("distinct seeds and positive budgets required")
    inputs = prior_inputs(prior)
    output.mkdir(parents=True, exist_ok=False)
    base = seed_library(output / "basis")
    if any(InterfaceLibrary.read(i["base"]).digest != base.digest for i in inputs):
        raise ValueError("the fixed elementary basis changed")
    save(output / "inputs.json", inputs)
    plan = dict(version=1, source_hashes=sources(), input_hash=digest(inputs), basis_hash=base.digest,
        seeds=list(seeds), tasks=list(TASKS), model=MODEL, seconds=seconds, replies=replies,
        reuse_seconds=10, reuse_candidate_limit=10000, process_group_memory_mib=MEMORY_MIB,
        growth_contract=GROWTH_CONTRACT, cold_candidate_limit=8000000,
        rows=schedule(inputs, seeds), diagnostic="new expression nodes; never used for selection",
        intervention="before/after task 2, no new helpers in retained arms",
        audit=dict(maximum_leaves=7, assignments=2, continuation_maximum_leaves=4))
    save(output / "plan.json", dict(plan=plan, hash=digest(plan)))
    with tarfile.open(output / "sources.tar.gz", "w:gz") as archive:
        for name in SOURCES:
            archive.add(Path(__file__).with_name(name), arcname=name)
    heads, rows = {}, []
    for task in plan["rows"]:
        folder, root = output / task["directory"], f"T{task['stage']}"
        key = (task["seed"], task["lineage"], task["arm"])
        library, history = (initial_state(task, inputs, base) if task["arm"] == "cold" else
                            heads.get(key, initial_state(task, inputs, base)))
        pairs = pairs_for(task["effect"], "train", task["seed"])
        print(json.dumps(dict(event="start", **task)), flush=True)
        if task["arm"] == "cold":
            raw, effort, by = frontier.search(library, pairs, root, folder, seconds, "codex", [],
                                             replies=replies, candidate_limit=8000000)
        else:
            raw, detail = search_process(library, pairs, root, folder / "reuse", 10, phase="reuse")
            effort, by = dict(reuse=detail), "host_reuse" if raw else None
        row = dict(**task, admitted=False, selected_by=by, status="no candidate", marginal_nodes=None)
        record = dict(row=row, input_library=library.payload(), history=history, candidate=raw, effort=effort)
        if raw:
            built, certs = attach_proposal(library, raw, root)
            checks = {s: brief(score(built, root, pairs_for(task["effect"], s, task["seed"])))
                      for s in ("train", "validation")}
            preserved = retention(built, history)
            ok = all(r["exact"] for r in (*checks.values(), *preserved.values()))
            row.update(admitted=ok, status="promoted" if ok else "gate rejected",
                marginal_nodes=node_cost(raw) if ok else None,
                new_interface_arguments=sum(len(d["params"]) for d in raw["definitions"]))
            record.update(library=built.payload(), certificates=certs, checks=checks, retention=preserved)
            if ok and task["arm"] != "cold":
                heads[key] = built, [*history, dict(root=root, validation=pairs_for(
                    task["effect"], "validation", task["seed"]))]
        save(folder / "selection.json", record)
        rows.append(row)
        save(output / "progress.json", dict(completed=rows, planned=len(plan["rows"])))
        print(json.dumps(dict(event="selected", **row,
            stopped=effort.get("reuse", {}).get("stopped"))), flush=True)
    save(output / "frozen.json", dict(rows=rows, hash=digest(rows)))
    return replay(output)


def verify_proposer(record, folder, task, plan):
    library = InterfaceLibrary.read(record["input_library"])
    pairs, root = pairs_for(task["effect"], "train", task["seed"]), f"T{task['stage']}"
    effort = record["effort"]
    reuse_seconds = effort["reuse_budget_seconds"] if task["arm"] == "cold" else plan["reuse_seconds"]
    expected = dict(library=library.payload(), pairs=pairs, root=root, seconds=reuse_seconds,
                    phase="reuse", growth_contract=GROWTH_CONTRACT, history=[], reuse=None)
    if digest(json.loads((folder / "reuse/search-request.json").read_text())) != digest(expected):
        raise ValueError("adapter request changed")
    response = folder / "reuse/search-response.json"
    if record["row"]["selected_by"] == "host_reuse":
        if json.loads(response.read_text())["candidate"] != record["candidate"]:
            raise ValueError("adapter provenance changed")
        if len(record["candidate"]["definitions"]) != 1:
            raise ValueError("retained adapter added a helper")
    elif task["arm"] != "cold":
        if response.exists() and json.loads(response.read_text())["candidate"] != record["candidate"]:
            raise ValueError("failed adapter outcome changed")
        if effort["reuse"].get("stopped") == "enumerated_subset_exhausted":
            candidate, replayed = reuse_search(library, pairs, root, 60, plan["reuse_candidate_limit"])
            if (candidate is not None or replayed["stopped"] != "enumerated_subset_exhausted"
                    or replayed["candidates"] != effort["reuse"]["candidates"]):
                raise ValueError("finite adapter exhaustion did not replay")
    if task["arm"] == "cold":
        reuse = frontier.reuse_context(effort["reuse"])
        feedback = None
        attempts = effort["proposer"].get("attempts", [])
        if len(attempts) > plan["replies"]:
            raise ValueError("reply budget exceeded")
        for number, attempt in enumerate(attempts, 1):
            path = folder / "proposer" / f"proposal-{number}"
            if digest(json.loads((path / "request.json").read_text())) != digest(prompt(library, pairs, root, feedback, reuse, [])):
                raise ValueError("model request changed")
            if "payload" in attempt:
                events = [json.loads(line) for line in (path / "events.jsonl").read_text().splitlines() if line]
                raw, _ = extract(events)
                if raw != attempt["payload"]:
                    raise ValueError("model response changed")
                if "training" in attempt:
                    built, _ = attach_proposal(library, raw, root)
                    measured = score(built, root, pairs)
                    if brief(measured) != attempt["training"]:
                        raise ValueError("public feedback changed")
                    if measured["exact"] and raw != record["candidate"]:
                        raise ValueError("first exact candidate changed")
                    if not measured["exact"] and attempt.get("feedback") != dict(proposal=raw,
                            counterexamples=[r for r in measured["records"] if not r["exact"]][:2]):
                        raise ValueError("counterexamples changed")
            feedback = attempt.get("feedback")


def behavior_audit(library, root, effect, settings):
    rng = random.Random(114901)
    trees = [frontier.materialize(s, rng, 9000) for n in range(1, settings["maximum_leaves"] + 1)
             for s in frontier.shapes(n) for _ in range(settings["assignments"])]
    machine = Machine(library)
    correct = 0
    for tree in trees:
        execution = machine.run(root, frontier.encode(tree, "framed"))
        correct += execution["ok"] and execution["output"] == list(reference(tree, effect))
    small = [frontier.materialize(s, rng, 10000) for n in range(1, settings["continuation_maximum_leaves"] + 1)
             for s in frontier.shapes(n)]
    twice, _ = library.attach(definition("AuditTwice", [], "U", ("seq", ("call", root), ("call", root))))
    machine = Machine(twice)
    continuations = 0
    for a, b in itertools.product(small, repeat=2):
        execution = machine.run("AuditTwice", frontier.encode(a, "framed") + frontier.encode(b, "framed"))
        continuations += execution["ok"] and execution["output"] == list(reference(a, effect) + reference(b, effect))
    return dict(shapes_correct=correct, shapes_total=len(trees), continuations_correct=continuations,
                continuations_total=len(small) ** 2,
                exact=correct == len(trees) and continuations == len(small) ** 2)


def replay(output):
    manifest = json.loads((output / "plan.json").read_text())
    plan = manifest["plan"]
    inputs = json.loads((output / "inputs.json").read_text())
    frozen = json.loads((output / "frozen.json").read_text())
    base = InterfaceLibrary.read(json.loads((output / "basis/corpus.json").read_text())["library"])
    if (digest(plan) != manifest["hash"] or plan["source_hashes"] != sources()
            or digest(inputs) != plan["input_hash"] or base.digest != plan["basis_hash"]
            or digest(frozen["rows"]) != frozen["hash"]
            or plan["rows"] != schedule(inputs, plan["seeds"]) or len(plan["rows"]) != len(frozen["rows"])):
        raise ValueError("fixed plan, source, input or outcome changed")
    heads, measured, certificates = {}, [], 0
    for task, row in zip(plan["rows"], frozen["rows"]):
        if any(row[k] != v for k, v in task.items()):
            raise ValueError("planned outcome omitted or reordered")
        key = (task["seed"], task["lineage"], task["arm"])
        expected, history = (initial_state(task, inputs, base) if task["arm"] == "cold" else
                             heads.get(key, initial_state(task, inputs, base)))
        folder, root = output / task["directory"], f"T{task['stage']}"
        record = json.loads((folder / "selection.json").read_text())
        if (record["row"] != row or InterfaceLibrary.read(record["input_library"]).digest != expected.digest
                or digest(record["history"]) != digest(history)):
            raise ValueError("selection or lineage changed")
        verify_proposer(record, folder, task, plan)
        result = dict(row, reuse_stopped=record["effort"]["reuse"].get("stopped"))
        if record["candidate"]:
            built = certificate_replay(expected, record["candidate"], record["library"], record["certificates"], root)
            certificates += len(record["certificates"])
            checks = {s: brief(score(built, root, pairs_for(task["effect"], s, task["seed"])))
                      for s in ("train", "validation")}
            preserved = retention(built, history)
            ok = all(r["exact"] for r in (*checks.values(), *preserved.values()))
            if checks != record["checks"] or preserved != record["retention"] or ok != row["admitted"]:
                raise ValueError("admission changed")
            if row["marginal_nodes"] != (node_cost(record["candidate"]) if ok else None):
                raise ValueError("diagnostic changed")
        elif row["admitted"] or row["marginal_nodes"] is not None:
            raise ValueError("missing proposal counted as success or zero cost")
        if row["admitted"]:
            if task["arm"] != "cold":
                heads[key] = built, [*history, dict(root=root, validation=pairs_for(
                    task["effect"], "validation", task["seed"]))]
            for split in ("hidden", "stress"):
                result[split] = brief(score(built, root, pairs_for(task["effect"], split, task["seed"])))
            result["audit"] = behavior_audit(built, root, task["effect"], plan["audit"])
            if task["arm"] != "cold":
                initial, _ = initial_state(task, inputs, base)
                helpers = initial.cells.keys() - base.cells.keys() - {"T1", "T2"}
                samples = pairs_for(task["effect"], "hidden", task["seed"])
                observed = score(built, root, samples)
                result["reuse"] = {name: dict(hash_unchanged=initial.cells[name]["hash"] == built.cells[name]["hash"],
                    used_cases=sum(r["visits"].get(name, 0) > 0 for r in observed["records"]),
                    elided_correct=score(built, root, samples, disabled=(name,))["correct"], total=len(samples))
                    for name in sorted(helpers)}
        measured.append(result)
    summary = dict(rows=measured, verified_pushouts=certificates,
        all_planned_outcomes_replayed=True, provenance_verified=True)
    existing = output / "summary.json"
    if existing.exists() and json.loads(existing.read_text()) != summary:
        raise ValueError("post-selection result changed")
    save(existing, summary)
    print(json.dumps(dict(event="complete", verified_pushouts=certificates,
        outcomes=len(measured), admitted=sum(r["admitted"] for r in measured))), flush=True)
    return summary


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    if args.replay:
        replay(args.output)
    elif args.prior is None:
        parser.error("--prior required")
    else:
        run(args.output, args.prior)
