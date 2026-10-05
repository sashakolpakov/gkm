"""Control-structure frontier with the unchanged elementary token basis."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from codex_glue import MODEL, _rss_kib, extract
from cofibration import verify_certificate
from frontier_search import (insert_effects, insertion_sites, mechanical,
                             verify_derivation)
from interface_benchmark import (MEMORY_MIB, SOURCE_NAMES, local_effect, model_search,
                                 pairs_for as old_pairs, reuse_context, save,
                                 search_process, seed_library)
from interface_machine import InterfaceLibrary, Machine, attach_proposal, parse, render, signature, size
from interface_search import (GROWTH_CONTRACT, brief, definition, discover,
                              lifted_skeleton, prompt, score, substitute)
from modular_machine import digest
from recursive_benchmark import comb, random_shape, shapes


FAMILIES = ("control", "infix", "framed", "feedback")
SOURCES = (*SOURCE_NAMES, "frontier_search.py", "frontier_benchmark.py")


def sources():
    return {n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in SOURCES}


def materialize(shape, rng, pool):
    pair = lambda: tuple(rng.sample(range(pool, pool + 32), 2))
    if shape is None:
        return dict(pair=pair())
    return dict(marker=rng.randrange(pool, pool + 32), pre=pair(), mid=pair(), post=pair(),
                left=materialize(shape[0], rng, pool), right=materialize(shape[1], rng, pool))


def encode(tree, family):
    if "pair" in tree:
        return tree["pair"]
    before = tree["pre"] if family == "framed" else ()
    middle = tree["mid"] if family != "control" else ()
    after = tree["post"] if family == "framed" else ()
    return ((tree["marker"],) * 2 + before + encode(tree["left"], family)
            + middle + encode(tree["right"], family) + after)


def reference(tree, family, stage):
    # Independent tree oracle. No DSL or graph is executed here.
    leaf_effect = (3 if stage == 1 else 4) if family == "control" else (0 if stage == 1 else 2)
    if "pair" in tree:
        return local_effect(tree["pair"], leaf_effect), 1
    left, nl = reference(tree["left"], family, stage)
    right, nr = reference(tree["right"], family, stage)
    middle_effect = 3 if family == "feedback" and nl % 2 == 0 else 1
    middle = local_effect(tree["mid"], middle_effect) if family != "control" else ()
    before = local_effect(tree["pre"], 1) if family == "framed" else ()
    after = local_effect(tree["post"], 1) if family == "framed" else ()
    return before + left + middle + right + after, nl + nr


def pairs_for(family, stage, split, seed=1):
    offset, pool = {"train": (7, 0), "validation": (101, 100),
                    "hidden": (503, 1000), "stress": (997, 2000)}[split]
    rng = random.Random(seed * 1009 + offset)
    if split in ("train", "validation"):
        forms = [s for n in range(1, 5) for s in shapes(n) for _ in range(2)]
        forms += [random_shape(5, rng) for _ in range(8)]
    elif split == "hidden":
        forms = [random_shape(n, rng) for n in (6, 9, 16, 32, 64) for _ in range(8)]
    else:
        forms = [comb(n, side) for n in (16, 32, 64) for side in ("left", "right", "alternating")]
        forms += [random_shape(128, rng) for _ in range(4)]
    trees = [materialize(form, rng, pool) for form in forms]
    return [(encode(t, family), reference(t, family, stage)[0]) for t in trees]


def witness(library, family, stage, root="W"):
    """Private coverage certificate, generated inside the mechanical grammar.

    Used in tests and after freezing. Neither proposer receives this construction,
    its signature, its edit locations, or the family label.
    """
    abstract = discover(library)[0]
    spec = (0, 1 if family == "control" else 3 if family == "feedback" else 2,
            "B" if family == "feedback" else "U")
    original = lifted_skeleton(abstract, *spec)
    sites = insertion_sites(original, library.signatures())
    def at(path):
        term = original["body"]
        for i in path:
            term = term[i]
        return term
    recursive = [s for s in sites if s["side"] == "before" and isinstance(at(s["path"]), tuple)
                 and at(s["path"])[:2] == ("call", "self")]
    chosen = []
    if family != "control":
        chosen.append(recursive[1])
    if family == "framed":
        chosen.insert(0, recursive[0])
        # The sequence containing both recursive calls is a different original
        # expression from either call; its after-site handles the trailer.
        parent = recursive[0]["path"][:-1]
        chosen.append(next(s for s in sites if s["path"] == parent and s["side"] == "after"))
    proto = insert_effects(original, chosen)
    filling, insertion_holes = [], {}
    for hole, domain in zip(proto["holes"], proto["domains"]):
        if hole["role"] == "result":
            env = hole["available"]
            value = ("xor", env[-2], env[-1]) if len(env) >= 2 else "true"
        elif hole["role"] == "inserted_procedure_selection":
            number = insertion_holes.get(hole["insertion"], 0)
            insertion_holes[hole["insertion"]] = number + 1
            value = "true" if number == 0 else ("not", hole["available"][-1])
        else:
            value = "false"
        if value not in domain:
            raise ValueError("coverage witness outside generic hole grammar")
        filling.append(value)
    leaf = ("C03" if stage == 1 else "C04") if family == "control" else ("C01" if stage == 1 else "C02")
    refs = [leaf] + (["F004", "C03"] if family == "feedback" else ["F004"] if family != "control" else [])
    filling += [("ref", name) for name in refs]
    body = substitute(proto["body"], filling)
    call = ("call", root + "H", *(("ref", name) for name in refs))
    raw = dict(library_hash=library.digest, definitions=[definition(root + "H", proto["params"], proto["returns"], body),
        definition(root, [], "U", ("seq", call, "unit") if proto["returns"] == "B" else call)])
    origin = dict(kind="typed_control_edits", sources=abstract["sources"], signature=list(spec), edits=chosen,
                  exact_base_specializations=abstract["specializations"], filling=[render(f) for f in filling])
    verify_derivation(library, raw, origin, root)
    return raw, origin


def worker(library, pairs, root, folder, seconds, history, reuse, *, candidate_limit=2000000):
    request = folder / "request.json"
    save(request, dict(library=library.payload(), pairs=pairs, root=root, seconds=seconds,
                       growth_contract=GROWTH_CONTRACT, history=history, reuse=reuse,
                       candidate_limit=candidate_limit))
    started, peak, stopped = time.monotonic(), 0, None
    proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", str(request)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    try:
        while proc.poll() is None:
            peak = max(peak, _rss_kib(proc.pid))
            if peak > MEMORY_MIB * 1024 or time.monotonic() - started > seconds + 10:
                stopped = "memory limit" if peak > MEMORY_MIB * 1024 else "outer wall-time limit"
                os.killpg(proc.pid, signal.SIGTERM)
                break
            time.sleep(0.25)
        try:
            stdout, stderr = proc.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            stdout, stderr = proc.communicate(timeout=3)
        (folder / "stdout.log").write_bytes(stdout)
        (folder / "stderr.log").write_bytes(stderr)
        response = folder / "response.json"
        if stopped or proc.returncode or not response.exists():
            return None, dict(error=stopped or "worker failed", exit_code=proc.returncode,
                              seconds=time.monotonic() - started, peak_process_group_rss_kib=peak)
        result = json.loads(response.read_text())
        result["effort"].update(outer_seconds=time.monotonic() - started, peak_process_group_rss_kib=peak)
        return result["candidate"], result["effort"]
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=3)


def search(library, pairs, root, folder, seconds, arm, history, *, replies=2, candidate_limit=2000000):
    started = time.monotonic()
    budget = min(10, seconds)
    raw, reused = search_process(library, pairs, root, folder / "reuse", budget, phase="reuse", history=history)
    remaining = max(0, seconds - (time.monotonic() - started))
    effort = dict(reuse=reused, reuse_budget_seconds=budget, proposer_budget_seconds=remaining, proposer={})
    by = "host_reuse" if raw else None
    if raw is None and remaining >= 2:
        method = model_search if arm == "codex" else worker
        options = dict(max_replies=replies, log_progress=True) if arm == "codex" else dict(candidate_limit=candidate_limit)
        raw, effort["proposer"] = method(library, pairs, root, folder / "proposer", remaining,
                                          history=history, reuse=reuse_context(reused), **options)
        by = arm if raw else None
    effort["seconds"] = time.monotonic() - started
    return raw, effort, by


def retention(library, family, stage, seed):
    old = {"S" + str(i): score(library, "S" + str(i), old_pairs("local", 1, "validation", effects=(e, e)))
           for i, e in enumerate((0, 1, 2))}
    if stage == 2:
        old["T1"] = score(library, "T1", pairs_for(family, 1, "validation", seed))
    return old


def run(output, families=FAMILIES, arms=("mechanical", "codex"), seconds=180, seed=1,
        *, replies=2, candidate_limit=2000000):
    if seconds <= 0 or type(replies) is not int or replies < 1 or type(candidate_limit) is not int or candidate_limit < 1:
        raise ValueError("positive time, reply and candidate limits required")
    output.mkdir(parents=True, exist_ok=False)
    base = seed_library(output / "basis")
    protocol = dict(version=2, families=list(families), arms=list(arms), seconds=seconds, seed=seed,
        model=MODEL, source_hashes=sources(), basis_hash=base.digest, growth_contract=GROWTH_CONTRACT,
        max_control_edits=3, candidate_limit=candidate_limit, active_edited_sketch_limit=512,
        zero_edit_search="never evicted, at least one third of batches", replies=replies,
        process_group_memory_mib=MEMORY_MIB, no_private_feedback=True,
        selection="first training-exact proposal, then private validation and retention; size is diagnostic")
    save(output / "protocol.json", protocol)
    with tarfile.open(output / "sources.tar.gz", "w:gz") as archive:
        for name in SOURCES:
            archive.add(Path(__file__).with_name(name), arcname=name)
    rows = []
    for family in families:
        for arm in arms:
            library, history = base, []
            for stage in (1, 2):
                folder, root = output / family / arm / f"stage-{stage}", f"T{stage}"
                folder.mkdir(parents=True)
                row = dict(family=family, arm=arm, stage=stage, admitted=False)
                if stage == 2 and "T1" not in library.cells:
                    row.update(status="blocked: acquisition failed", effort={})
                    save(folder / "selection.json", dict(row=row, candidate=None))
                    rows.append(row)
                    continue
                print(json.dumps(dict(event="start", **row)), flush=True)
                pairs = pairs_for(family, stage, "train", seed)
                raw, effort, by = search(library, pairs, root, folder, seconds, arm, history,
                                         replies=replies, candidate_limit=candidate_limit)
                row.update(effort=effort, selected_by=by, status="no candidate")
                record = dict(row=row, candidate=raw, input_library=library.payload(), history=list(history))
                if raw:
                    built, certificates = attach_proposal(library, raw, root)
                    built.verify()
                    checks = {split: score(built, root, pairs_for(family, stage, split, seed), trace_limit=100)
                              for split in ("train", "validation")}
                    retained = retention(built, family, stage, seed)
                    ok = all(r["exact"] for r in (*checks.values(), *retained.values()))
                    row.update(admitted=ok, status="promoted" if ok else "gate rejected",
                        signatures={d["name"]: signature(d) for d in raw["definitions"]},
                        new_helpers=[d["name"] for d in raw["definitions"][:-1]],
                        new_expression_nodes=sum(size(parse(d["body"])) for d in raw["definitions"]),
                        checks={s: brief(r) for s, r in checks.items()})
                    record.update(library=built.payload(), certificates=certificates, checks=checks, retention=retained)
                    if ok:
                        library = built
                        history.append(dict(root=root, training_examples=pairs, library_hash=built.digest))
                save(folder / "selection.json", record)
                rows.append(row)
                print(json.dumps(dict(event="selected", family=family, arm=arm, stage=stage,
                    admitted=row["admitted"], selected_by=by, seconds=effort["seconds"])), flush=True)
    save(output / "frozen.json", dict(rows=rows, hash=digest(rows)))
    measured = evaluate(output, rows, base, seed)
    save(output / "summary.json", dict(rows=measured))
    return measured


def evaluate(output, rows, base, seed):
    results = []
    for row in rows:
        result = dict(row)
        folder = output / row["family"] / row["arm"] / f"stage-{row['stage']}"
        record = json.loads((folder / "selection.json").read_text())
        if row["admitted"]:
            library, root = InterfaceLibrary.read(record["library"]), f"T{row['stage']}"
            for split in ("hidden", "stress"):
                measured = score(library, root, pairs_for(row["family"], row["stage"], split, seed), trace_limit=180)
                save(folder / f"{split}.json", measured)
                result[split] = brief(measured)
            if row["stage"] == 2:
                old = InterfaceLibrary.read(record["input_library"])
                samples = pairs_for(row["family"], 2, "hidden", seed)
                executions = score(library, root, samples)
                transfer = {}
                for name in old.cells:
                    if name in base.cells or name == "T1":
                        continue
                    elided = score(library, root, samples, disabled=(name,))
                    transfer[name] = dict(hash_unchanged=old.cells[name]["hash"] == library.cells[name]["hash"],
                        used_cases=sum(r["visits"].get(name, 0) > 0 for r in executions["records"]),
                        elided_correct=elided["correct"], total=len(samples))
                    save(folder / f"elide-{name}.json", elided)
                result["transfer"] = transfer
                result["genuine_transfer"] = executions["exact"] and any(
                    t["hash_unchanged"] and t["used_cases"] > 0 and t["elided_correct"] < t["total"] for t in transfer.values())
        results.append(result)
    for family in sorted({r["family"] for r in rows}):
        controls = {}
        for stage in (1, 2):
            raw, origin = witness(base, family, stage)
            built, _ = attach_proposal(base, raw, "W")
            checks = {s: brief(score(built, "W", pairs_for(family, stage, s, seed)))
                      for s in ("train", "validation", "hidden", "stress")}
            if not all(c["exact"] for c in checks.values()):
                raise ValueError("private grammar coverage certificate failed")
            controls[str(stage)] = dict(candidate=raw, derivation=origin, checks=checks)
        save(output / family / "private-grammar-coverage.json", controls)
    return results


def replay(output):
    protocol = json.loads((output / "protocol.json").read_text())
    if sources() != protocol["source_hashes"]:
        raise ValueError("use frozen source archive")
    base = InterfaceLibrary.read(json.loads((output / "basis/corpus.json").read_text())["library"])
    frozen = json.loads((output / "frozen.json").read_text())
    if base.digest != protocol["basis_hash"] or digest(frozen["rows"]) != frozen["hash"]:
        raise ValueError("frozen basis or selections changed")
    heads, histories, count = {}, {}, 0
    for row in frozen["rows"]:
        folder = output / row["family"] / row["arm"] / f"stage-{row['stage']}"
        record = json.loads((folder / "selection.json").read_text())
        if record["row"] != row:
            raise ValueError("selection record changed")
        if "input_library" not in record:
            continue
        key, root = (row["family"], row["arm"]), f"T{row['stage']}"
        library = InterfaceLibrary.read(record["input_library"])
        history = histories.setdefault(key, [])
        if library.digest != heads.get(key, base.digest) or record["history"] != history:
            raise ValueError("retained lineage or public history changed")
        pairs = pairs_for(row["family"], row["stage"], "train", protocol["seed"])
        expected = dict(library=library.payload(), pairs=pairs, root=root, seconds=row["effort"]["reuse_budget_seconds"],
            phase="reuse", growth_contract=GROWTH_CONTRACT, history=history, reuse=None)
        if json.loads((folder / "reuse/search-request.json").read_text()) != json.loads(json.dumps(expected)):
            raise ValueError("shared reuse request changed")
        reuse = reuse_context(row["effort"]["reuse"])
        if row["selected_by"] == "host_reuse":
            reply = json.loads((folder / "reuse/search-response.json").read_text())
            if reply["candidate"] != record["candidate"] or row["effort"]["proposer"]:
                raise ValueError("shared reuse provenance changed")
        if row["arm"] == "codex":
            feedback = None
            attempts = row["effort"]["proposer"].get("attempts", [])
            if len(attempts) > protocol["replies"]:
                raise ValueError("model reply budget exceeded")
            for i, attempt in enumerate(attempts, 1):
                path = folder / "proposer" / f"proposal-{i}"
                if json.loads((path / "request.json").read_text()) != json.loads(json.dumps(prompt(library, pairs, root, feedback, reuse, history))):
                    raise ValueError("model input changed")
                if "payload" in attempt:
                    events = [json.loads(line) for line in (path / "events.jsonl").read_text().splitlines() if line]
                    raw, _ = extract(events)
                    if raw != attempt["payload"] or (attempt.get("training", {}).get("exact") and raw != record["candidate"]):
                        raise ValueError("model reply changed")
                    if "training" in attempt:
                        attempted, _ = attach_proposal(library, raw, root)
                        measured = score(attempted, root, pairs)
                        if brief(measured) != attempt["training"]:
                            raise ValueError("intermediate training score changed")
                        if not measured["exact"] and attempt.get("feedback") != dict(proposal=raw,
                                counterexamples=[r for r in measured["records"] if not r["exact"]][:2]):
                            raise ValueError("intermediate public feedback changed")
                feedback = attempt.get("feedback")
        elif row["effort"]["proposer"]:
            expected = dict(library=library.payload(), pairs=pairs, root=root, seconds=row["effort"]["proposer_budget_seconds"],
                growth_contract=GROWTH_CONTRACT, history=history, reuse=reuse,
                candidate_limit=protocol["candidate_limit"])
            if json.loads((folder / "proposer/request.json").read_text()) != json.loads(json.dumps(expected)):
                raise ValueError("mechanical request changed")
            response = folder / "proposer/response.json"
            if response.exists() and json.loads(response.read_text())["candidate"] != record["candidate"]:
                raise ValueError("mechanical reply changed")
        if record["candidate"] is None:
            continue
        built, certs = attach_proposal(library, record["candidate"], root)
        if (json.loads(json.dumps(certs)) != record["certificates"] or
                built.digest != InterfaceLibrary.read(record["library"]).digest):
            raise ValueError("attachment changed")
        if row["selected_by"] == "mechanical":
            verify_derivation(library, record["candidate"], row["effort"]["proposer"]["derivation"], root)
        graph = library.graph
        for cert in certs:
            graph = verify_certificate(cert, graph).graph
            count += 1
        for split in ("train", "validation"):
            if score(built, root, pairs_for(row["family"], row["stage"], split, protocol["seed"]), trace_limit=100) != record["checks"][split]:
                raise ValueError("admission replay changed")
        if retention(built, row["family"], row["stage"], protocol["seed"]) != record["retention"]:
            raise ValueError("old-task retention changed")
        admitted = all(r["exact"] for r in (*record["checks"].values(), *record["retention"].values()))
        if admitted != row["admitted"]:
            raise ValueError("admission decision changed")
        if admitted:
            heads[key] = built.digest
            history.append(dict(root=root, training_examples=json.loads(json.dumps(pairs)), library_hash=built.digest))
    old = json.loads((output / "summary.json").read_text())
    if dict(rows=evaluate(output, frozen["rows"], base, protocol["seed"])) != old:
        raise ValueError("hidden evaluation changed")
    result = dict(verified_pushouts=count, exact_fresh_replay=True, source_and_proposal_provenance_verified=True)
    save(output / "verification.json", result)
    return result


def run_batch(output, seeds, repeats, **options):
    """Fixed independent trials. Never carry a learned library between trials."""
    if repeats < 1 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("positive repeat count and distinct seeds required")
    output.mkdir(parents=True, exist_ok=False)
    plan = dict(version=1, source_hashes=sources(), options=options,
        trials=[dict(seed=seed, repeat=repeat, directory=f"seed-{seed}-trial-{repeat}")
                for seed in seeds for repeat in range(1, repeats + 1)])
    save(output / "trials.json", dict(plan=plan, hash=digest(plan)))
    records = []
    for trial in plan["trials"]:
        print(json.dumps(dict(event="trial_start", **trial)), flush=True)
        rows = run(output / trial["directory"], seed=trial["seed"], **options)
        records.append(dict(**trial, rows=rows))
        save(output / "progress.json", dict(completed=records, planned=len(plan["trials"])))
    save(output / "batch-summary.json", dict(trials=records))
    return records


def replay_batch(output):
    manifest = json.loads((output / "trials.json").read_text())
    plan = manifest["plan"]
    if digest(plan) != manifest["hash"] or plan["source_hashes"] != sources():
        raise ValueError("trial plan or sources changed")
    records, verified = [], []
    for trial in plan["trials"]:
        path = output / trial["directory"]
        if Path(trial["directory"]).name != trial["directory"]:
            raise ValueError("trial path outside batch")
        protocol = json.loads((path / "protocol.json").read_text())
        if protocol["seed"] != trial["seed"] or any(protocol[k] != v for k, v in plan["options"].items()):
            raise ValueError("trial differs from fixed batch plan")
        verified.append(dict(**trial, verification=replay(path)))
        records.append(dict(**trial, rows=json.loads((path / "summary.json").read_text())["rows"]))
    if json.loads((output / "batch-summary.json").read_text()) != dict(trials=records):
        raise ValueError("trial omitted or aggregate results changed")
    result = dict(trials=verified, all_trials_replayed=True)
    save(output / "batch-verification.json", result)
    return result


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path)
    p.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    p.add_argument("--arms", nargs="+", choices=("mechanical", "codex"), default=["mechanical", "codex"])
    p.add_argument("--seconds", type=float, default=180)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--seeds", type=int, nargs="+", help="fixed batch of independent data seeds")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--replies", type=int, default=2)
    p.add_argument("--candidate-limit", type=int, default=2000000)
    p.add_argument("--worker", type=Path)
    p.add_argument("--replay", action="store_true")
    args = p.parse_args()
    if args.worker:
        req = json.loads(args.worker.read_text())
        try:
            raw, effort = mechanical(InterfaceLibrary.read(req["library"]), req["pairs"], req["root"], req["seconds"],
                                     candidate_limit=req["candidate_limit"])
        except Exception as exc:
            raw, effort = None, dict(error=type(exc).__name__ + ": " + str(exc))
        save(args.worker.with_name("response.json"), dict(candidate=raw, effort=effort))
    elif args.output is None:
        p.error("--output required")
    elif args.replay:
        method = replay_batch if (args.output / "trials.json").exists() else replay
        print(json.dumps(method(args.output), indent=2))
    else:
        options = dict(families=args.families, arms=args.arms, seconds=args.seconds,
                       replies=args.replies, candidate_limit=args.candidate_limit)
        if args.seeds:
            run_batch(args.output, args.seeds, args.repeats, **options)
        elif args.repeats != 1:
            p.error("--repeats requires --seeds")
        else:
            run(args.output, seed=args.seed, **options)
