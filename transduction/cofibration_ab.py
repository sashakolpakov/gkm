"""Paired cumulative attachment discovery: model proposals versus search.

Unlike the library-value assay, the model authors every model-arm attachment.
The two arms never exchange acquired cells or proposals.
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

from codex_glue import MODEL, extract
from cofibration import verify_certificate
import frontier_benchmark as frontier
from frontier_search import verify_derivation
from interface_benchmark import MEMORY_MIB, model_search, save, seed_library
from interface_machine import InterfaceLibrary, Machine, attach_proposal, parse, size
from interface_search import GROWTH_CONTRACT, brief, definition, prompt, score
from modular_machine import digest


TASKS = [(family, phase) for family in ("infix", "framed", "feedback") for phase in (1, 2)]
ARMS = ("mechanical", "codex")
SOURCES = (*frontier.SOURCES, "cofibration_ab.py")


def sources():
    return {n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in SOURCES}


def schedule(seeds, tasks):
    return [dict(seed=seed, arm=arm, task=i, family=family, phase=phase,
                 directory=f"seed-{seed}/{arm}/task-{i}")
            for seed in seeds for arm in ARMS for i, (family, phase) in enumerate(tasks, 1)]


def prepare(output, seeds=(31, 32), seconds=300, replies=4, tasks=TASKS):
    if len(set(seeds)) != len(seeds) or not seeds or seconds <= 0 or replies < 1:
        raise ValueError("distinct seeds and positive limits required")
    if any(f not in frontier.FAMILIES or p not in (1, 2) for f, p in tasks):
        raise ValueError("invalid task sequence")
    output.mkdir(parents=True, exist_ok=False)
    base = seed_library(output / "basis")
    plan = dict(version=1, seeds=list(seeds), tasks=list(tasks), arms=list(ARMS),
        source_hashes=sources(), basis_hash=base.digest, seconds=seconds, replies=replies,
        candidate_limit=8000000, model=MODEL, process_group_memory_mib=MEMORY_MIB,
        growth_contract=GROWTH_CONTRACT, model_arm_host_reuse=False,
        selection="first training-exact proposal, then validation and prior-task retention",
        complexity="new expression nodes, diagnostic only", rows=schedule(seeds, tasks))
    save(output / "plan.json", dict(plan=plan, hash=digest(plan)))
    with tarfile.open(output / "sources.tar.gz", "w:gz") as archive:
        for name in SOURCES:
            archive.add(Path(__file__).with_name(name), arcname=name)
    return plan


def load_plan(output):
    manifest = json.loads((output / "plan.json").read_text())
    plan = manifest["plan"]
    base = InterfaceLibrary.read(json.loads((output / "basis/corpus.json").read_text())["library"])
    if (digest(plan) != manifest["hash"] or plan["source_hashes"] != sources()
            or base.digest != plan["basis_hash"] or plan["model_arm_host_reuse"] is not False
            or plan["rows"] != schedule(plan["seeds"], plan["tasks"])):
        raise ValueError("fixed protocol, sources, basis or model/search separation changed")
    return plan, base


def validation_history(library, completed):
    return {f"T{t['task']}": brief(score(library, f"T{t['task']}", frontier.pairs_for(
        t["family"], t["phase"], "validation", t["seed"]))) for t in completed}


def run_arm(output, arm):
    plan, base = load_plan(output)
    if arm not in ARMS or (output / f"{arm}-frozen.json").exists():
        raise ValueError("unknown arm or completed arm cannot be rerun")
    planned = [t for t in plan["rows"] if t["arm"] == arm]
    if any((output / t["directory"] / "selection.json").exists() for t in planned):
        raise ValueError("partial arm exists; do not silently repeat sampled proposals")
    heads, histories, completed, stopped, rows = {}, {}, {}, set(), []
    for task in planned:
        seed, root, folder = task["seed"], f"T{task['task']}", output / task["directory"]
        library = heads.get(seed, base)
        history = histories.setdefault(seed, [])
        old_tasks = completed.setdefault(seed, [])
        row = dict(**task, admitted=False, marginal_nodes=None, selected_by=None)
        record = dict(row=row, input_library=library.payload(), history=list(history), candidate=None)
        if seed in stopped:
            row["status"] = "blocked by preceding failed attachment"
            record["effort"] = {}
        else:
            print(json.dumps(dict(event="start", **task)), flush=True)
            pairs = frontier.pairs_for(task["family"], task["phase"], "train", seed)
            if arm == "codex":
                # No host reuse probe and no mechanical proposal in this arm.
                raw, effort = model_search(library, pairs, root, folder / "proposer", plan["seconds"],
                    reuse=None, history=history, max_replies=plan["replies"], log_progress=True)
                by = "codex" if raw else None
            else:
                raw, effort, by = frontier.search(library, pairs, root, folder, plan["seconds"],
                    "mechanical", history, replies=plan["replies"], candidate_limit=plan["candidate_limit"])
            record.update(candidate=raw, effort=effort)
            row.update(status="no candidate", selected_by=by)
            if raw:
                built, certs = attach_proposal(library, raw, root)
                checks = {s: brief(score(built, root, frontier.pairs_for(task["family"], task["phase"], s, seed)))
                          for s in ("train", "validation")}
                retained = validation_history(built, old_tasks)
                admitted = all(r["exact"] for r in (*checks.values(), *retained.values()))
                row.update(admitted=admitted, status="promoted" if admitted else "gate rejected",
                    marginal_nodes=sum(size(parse(d["body"])) for d in raw["definitions"]) if admitted else None,
                    new_helpers=[d["name"] for d in raw["definitions"][:-1]])
                record.update(library=built.payload(), certificates=certs, checks=checks, retention=retained)
                if admitted:
                    heads[seed] = built
                    history.append(dict(root=root, training_examples=pairs, library_hash=built.digest))
                    old_tasks.append(task)
            if not row["admitted"]:
                stopped.add(seed)
        save(folder / "selection.json", record)
        rows.append(row)
        save(output / f"{arm}-progress.json", dict(rows=rows, planned=len(planned)))
        print(json.dumps(dict(event="selected", **row)), flush=True)
    save(output / f"{arm}-frozen.json", dict(rows=rows, hash=digest(rows)))
    return rows


def verify_provenance(record, folder, task, plan):
    library = InterfaceLibrary.read(record["input_library"])
    pairs = frontier.pairs_for(task["family"], task["phase"], "train", task["seed"])
    root, history, effort = f"T{task['task']}", record["history"], record["effort"]
    if task["arm"] == "codex":
        if (folder / "reuse").exists() or record["row"]["selected_by"] not in (None, "codex"):
            raise ValueError("mechanical glue entered the model arm")
        feedback = None
        if len(effort["attempts"]) > plan["replies"]:
            raise ValueError("reply cap exceeded")
        for number, attempt in enumerate(effort["attempts"], 1):
            path = folder / "proposer" / f"proposal-{number}"
            expected = prompt(library, pairs, root, feedback, None, history)
            if digest(json.loads((path / "request.json").read_text())) != digest(expected):
                raise ValueError("model input changed")
            if "payload" in attempt:
                events = [json.loads(line) for line in (path / "events.jsonl").read_text().splitlines() if line]
                raw, _ = extract(events)
                if raw != attempt["payload"]:
                    raise ValueError("model reply changed")
                if "training" in attempt:
                    candidate, _ = attach_proposal(library, raw, root)
                    measured = score(candidate, root, pairs)
                    if brief(measured) != attempt["training"]:
                        raise ValueError("model public score changed")
                    if measured["exact"] and raw != record["candidate"]:
                        raise ValueError("first exact proposal was not selected")
                    if not measured["exact"] and attempt.get("feedback") != dict(proposal=raw,
                        counterexamples=[r for r in measured["records"] if not r["exact"]][:2]):
                        raise ValueError("model feedback changed")
            feedback = attempt.get("feedback")
    else:
        expected = dict(library=library.payload(), pairs=pairs, root=root, seconds=effort["reuse_budget_seconds"],
            phase="reuse", growth_contract=GROWTH_CONTRACT, history=history, reuse=None)
        if digest(json.loads((folder / "reuse/search-request.json").read_text())) != digest(expected):
            raise ValueError("mechanical reuse input changed")
        if record["row"]["selected_by"] == "host_reuse":
            response = json.loads((folder / "reuse/search-response.json").read_text())
            if response["candidate"] != record["candidate"] or effort["proposer"]:
                raise ValueError("mechanical reuse provenance changed")
        if effort["proposer"]:
            expected = dict(library=library.payload(), pairs=pairs, root=root,
                seconds=effort["proposer_budget_seconds"], growth_contract=GROWTH_CONTRACT,
                history=history, reuse=frontier.reuse_context(effort["reuse"]), candidate_limit=plan["candidate_limit"])
            if digest(json.loads((folder / "proposer/request.json").read_text())) != digest(expected):
                raise ValueError("mechanical synthesis input changed")
            response = folder / "proposer/response.json"
            if response.exists() and json.loads(response.read_text())["candidate"] != record["candidate"]:
                raise ValueError("mechanical synthesis output changed")
        if record["row"]["selected_by"] == "mechanical":
            verify_derivation(library, record["candidate"], effort["proposer"]["derivation"], root)


def behavior_audit(library, root, family, phase, maximum_leaves=7, assignments=2):
    rng = random.Random(793541)
    trees = [frontier.materialize(s, rng, 9000) for n in range(1, maximum_leaves + 1)
             for s in frontier.shapes(n) for _ in range(assignments)]
    machine = Machine(library)
    correct = 0
    for tree in trees:
        execution = machine.run(root, frontier.encode(tree, family))
        correct += execution["ok"] and execution["output"] == list(frontier.reference(tree, family, phase)[0])
    small = [frontier.materialize(s, rng, 10000) for n in range(1, 5) for s in frontier.shapes(n)]
    twice, _ = library.attach(definition("AuditTwice", [], "U", ("seq", ("call", root), ("call", root))))
    machine, continuations = Machine(twice), 0
    for a, b in itertools.product(small, repeat=2):
        execution = machine.run("AuditTwice", frontier.encode(a, family) + frontier.encode(b, family))
        target = frontier.reference(a, family, phase)[0] + frontier.reference(b, family, phase)[0]
        continuations += execution["ok"] and execution["output"] == list(target)
    return dict(shapes_correct=correct, shapes_total=len(trees), continuations_correct=continuations,
                continuations_total=len(small) ** 2,
                exact=correct == len(trees) and continuations == len(small) ** 2)


def replay(output):
    plan, base = load_plan(output)
    frozen = {}
    for arm in ARMS:
        payload = json.loads((output / f"{arm}-frozen.json").read_text())
        if digest(payload["rows"]) != payload["hash"]:
            raise ValueError("frozen arm changed")
        expected = [t for t in plan["rows"] if t["arm"] == arm]
        if len(payload["rows"]) != len(expected):
            raise ValueError("planned trial omitted")
        for row, task in zip(payload["rows"], expected):
            if any(row[k] != v for k, v in task.items()):
                raise ValueError("planned trial reordered")
            frozen[task["directory"]] = row
    heads, histories, completed, blocked, rows, ncerts = {}, {}, {}, set(), [], 0
    for task in plan["rows"]:
        key = task["seed"], task["arm"]
        library = heads.get(key, base)
        history = histories.setdefault(key, [])
        old_tasks = completed.setdefault(key, [])
        folder, root = output / task["directory"], f"T{task['task']}"
        record = json.loads((folder / "selection.json").read_text())
        row = frozen[task["directory"]]
        if (record["row"] != row or InterfaceLibrary.read(record["input_library"]).digest != library.digest
                or digest(record["history"]) != digest(history)):
            raise ValueError("independent library lineage or history changed")
        result = dict(row)
        if key in blocked:
            if record["candidate"] is not None or record["effort"] or row["admitted"] or row["marginal_nodes"] is not None:
                raise ValueError("failed prefix was silently restarted")
            rows.append(result)
            continue
        verify_provenance(record, folder, task, plan)
        if record["candidate"]:
            built, certs = attach_proposal(library, record["candidate"], root)
            if digest(certs) != digest(record["certificates"]) or built.digest != InterfaceLibrary.read(record["library"]).digest:
                raise ValueError("attachment changed")
            graph = library.graph
            for cert in certs:
                graph = verify_certificate(cert, graph).graph
                ncerts += 1
            checks = {s: brief(score(built, root, frontier.pairs_for(task["family"], task["phase"], s, task["seed"])))
                      for s in ("train", "validation")}
            retained = validation_history(built, old_tasks)
            admitted = all(r["exact"] for r in (*checks.values(), *retained.values()))
            cost = sum(size(parse(d["body"])) for d in record["candidate"]["definitions"]) if admitted else None
            if (checks != record["checks"] or retained != record["retention"] or admitted != row["admitted"]
                    or cost != row["marginal_nodes"]):
                raise ValueError("admission or complexity diagnostic changed")
        elif row["admitted"] or row["marginal_nodes"] is not None:
            raise ValueError("missing proposal counted as success or free code")
        if not row["admitted"]:
            blocked.add(key)
        else:
            heads[key] = built
            history.append(dict(root=root, training_examples=frontier.pairs_for(
                task["family"], task["phase"], "train", task["seed"]), library_hash=built.digest))
            old_tasks.append(task)
            for split in ("hidden", "stress"):
                result[split] = brief(score(built, root, frontier.pairs_for(task["family"], task["phase"], split, task["seed"])))
            result["audit"] = behavior_audit(built, root, task["family"], task["phase"])
            names = library.cells.keys() - base.cells.keys() - {f"T{i}" for i in range(1, task["task"])}
            samples = frontier.pairs_for(task["family"], task["phase"], "hidden", task["seed"])
            executed = score(built, root, samples)
            result["reuse"] = {name: dict(hash_unchanged=library.cells[name]["hash"] == built.cells[name]["hash"],
                used_cases=sum(r["visits"].get(name, 0) > 0 for r in executed["records"]),
                elided_correct=score(built, root, samples, disabled=(name,))["correct"], total=len(samples))
                for name in sorted(names)}
            result["verified_reuse"] = executed["exact"] and any(v["hash_unchanged"] and v["used_cases"] > 0
                and v["elided_correct"] < v["total"] for v in result["reuse"].values())
        rows.append(result)
    coverage = []
    # Private grammar witnesses are constructed only after BOTH arms freeze.
    for seed in plan["seeds"]:
        for family, phase in plan["tasks"]:
            raw, origin = frontier.witness(base, family, phase)
            built, _ = attach_proposal(base, raw, "W")
            checks = {s: brief(score(built, "W", frontier.pairs_for(family, phase, s, seed)))
                      for s in ("train", "validation", "hidden", "stress")}
            if not all(c["exact"] for c in checks.values()):
                raise ValueError("mechanical grammar coverage failed")
            coverage.append(dict(seed=seed, family=family, phase=phase, candidate=raw, derivation=origin, checks=checks))
    summary = dict(rows=rows, verified_pushouts=ncerts, all_planned_outcomes_replayed=True,
                   no_host_glue_in_model_arm=True, independent_libraries_verified=True)
    old = output / "summary.json"
    if old.exists() and json.loads(old.read_text()) != summary:
        raise ValueError("fresh evaluation changed")
    save(output / "private-grammar-coverage.json", coverage)
    save(old, summary)
    print(json.dumps(dict(event="complete", verified_pushouts=ncerts,
        admitted={a: sum(r["admitted"] and r["arm"] == a for r in rows) for a in ARMS})), flush=True)
    return summary


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    if sum((args.prepare, args.arm is not None, args.replay)) != 1:
        parser.error("choose exactly one of --prepare, --arm, --replay")
    if args.prepare:
        print(json.dumps(prepare(args.output), indent=2))
    elif args.arm:
        run_arm(args.output, args.arm)
    else:
        replay(args.output)
