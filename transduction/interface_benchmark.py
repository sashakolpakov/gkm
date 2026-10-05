"""Paired interface-discovery experiment with immutable executable attachments."""
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

from codex_glue import MODEL, _rss_kib, extract, propose
from cofibration import verify_certificate
from growth_benchmark import acquire_word
from interface_machine import InterfaceLibrary, Machine, attach_proposal, parse, schema, signature, size
from interface_search import GROWTH_CONTRACT, brief, definition, mechanical, prompt, reuse_search, score
from modular_machine import digest
from recursive_benchmark import (acquire_basis, comb, encode_tree, materialize,
                                 random_shape, shapes)


FAMILIES = ("local", "inherited", "returned")
MEMORY_MIB = 640
REUSE_SECONDS = 10
SOURCE_NAMES = ("interface_machine.py", "interface_search.py", "interface_benchmark.py",
                "codex_glue.py", "cofibration.py", "modular_machine.py", "pattern_fsa.py",
                "growth_benchmark.py", "recursive_benchmark.py", "recursive_machine.py",
                "recursive_search.py", "glue_ab.py", "run_cofibration_experiment.py")


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True))


def sources():
    return {n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in SOURCE_NAMES}


def local_effect(pair, code):
    a, b = pair
    return {0: (a, b), 1: (b, a), 2: (a, a, b, b), 3: (a,), 4: (b,)}[code]


def reference(tree, family, effects, context=False):
    # Private tree oracle, independent of proposal syntax and graph execution.
    if "pair" in tree:
        return local_effect(tree["pair"], effects[int(context)]), True
    left, left_summary = reference(tree["left"], family, effects, context)
    right_context = (not context if family == "inherited" else
                     context != left_summary if family == "returned" else context)
    right, right_summary = reference(tree["right"], family, effects, right_context)
    return left + right, left_summary != right_summary


def effects_for(family, stage):
    return ((3, 3) if stage == 1 else (4, 4)) if family == "local" else (
        (0, 1) if stage == 1 else (2, 3) if family == "inherited" else (2, 4))


def pairs_for(family, stage, split, seed=1, effects=None):
    offsets = {"train": (7, 0), "validation": (101, 100), "hidden": (503, 1000), "stress": (997, 2000)}
    offset, pool = offsets[split]
    rng = random.Random(seed * 1009 + offset)
    if split in ("train", "validation"):
        forms = [s for n in range(1, 5) for s in shapes(n) for _ in range(2)]
        forms += [random_shape(5, rng) for _ in range(8)]
    elif split == "hidden":
        forms = [random_shape(n, rng) for n in (6, 9, 16, 32, 64) for _ in range(8)]
    else:
        forms = [comb(n, side) for n in (16, 32, 64) for side in ("left", "right", "alternating")]
        forms += [random_shape(128, rng) for _ in range(4)]
    result = []
    for shape in forms:
        tree = materialize(shape, rng, pool)
        result.append((encode_tree(tree), reference(tree, family, effects or effects_for(family, stage))[0]))
    return result


def seed_library(folder):
    base = acquire_basis(folder / "acquired").base
    pairs = [((a, b), (b,)) for a, b in itertools.product(range(3), repeat=2)]
    word, effort = acquire_word(pairs)
    base, sq = base.attach("C04", word, kind="fragment", width=2)
    save(folder / "C04.json", dict(program=word, effort=effort, training=pairs, certificate=sq.certificate()))
    library = InterfaceLibrary.from_base(base)
    evidence = []
    for index, (callee, effect) in enumerate((("C01", 0), ("F004", 1), ("C02", 2))):
        name = "S" + str(index)
        body = ("if", "eq", ("seq", ("call", "F002"), ("call", "F002"),
                              ("call", "self"), ("call", "self")), ("call", callee))
        library, square = library.attach(definition(name, [], "U", body))
        result = score(library, name, pairs_for("local", 1, "validation", effects=(effect, effect)))
        if not result["exact"]:
            raise ValueError("concrete starting corpus failed independent validation")
        evidence.append(dict(name=name, certificate=square.certificate(), validation=result))
    library.verify()
    save(folder / "corpus.json", dict(library=library.payload(), evidence=evidence,
                                     provenance="three explicitly supplied concrete source programs"))
    return library


def witness(library, family, stage, root):
    """Private language-expressibility control; never a proposer success."""
    helper = root + "H"
    ps = [("f", "F")] if family == "local" else [("b", "B"), ("f", "F"), ("g", "F")]
    if family == "local":
        core = ("seq", ("call", "F002"), ("call", "F002"), ("call", "self", "f"), ("call", "self", "f"))
        leaf, ret = ("call", "f"), "U"
    elif family == "inherited":
        core = ("seq", ("call", "F002"), ("call", "F002"),
                ("call", "self", "b", "f", "g"), ("call", "self", ("not", "b"), "f", "g"))
        leaf, ret = ("if", "b", ("call", "g"), ("call", "f")), "U"
    else:
        core = ("seq", ("call", "F002"), ("call", "F002"),
                ("let", "left", ("call", "self", "b", "f", "g"),
                 ("let", "right", ("call", "self", ("xor", "b", "left"), "f", "g"),
                  ("xor", "left", "right"))))
        leaf, ret = ("seq", ("if", "b", ("call", "g"), ("call", "f")), "true"), "B"
    cells = ["C01", "F004", "C02", "C03", "C04"]
    chosen = effects_for(family, stage)
    args = [("ref", cells[chosen[0]])] if family == "local" else ["false", *( ("ref", cells[c]) for c in chosen)]
    main = ("call", helper, *args)
    if ret == "B":
        main = ("seq", main, "unit")
    return dict(library_hash=library.digest, definitions=[
        definition(helper, ps, ret, ("if", "eq", core, leaf)), definition(root, [], "U", main)])


def model_search(library, pairs, root, folder, seconds, reuse=None, history=None,
                 *, max_replies=2, log_progress=False):
    if type(max_replies) is not int or max_replies < 1:
        raise ValueError("max_replies must be a positive integer")
    started, attempts, feedback, winner = time.monotonic(), [], None, None
    for number in range(1, max_replies + 1):
        remaining = seconds - (time.monotonic() - started)
        if remaining < 2:
            break
        attempt = {}
        attempts.append(attempt)
        attempt_started = time.monotonic()
        try:
            raw = propose(prompt(library, pairs, root, feedback, reuse, history), folder / f"proposal-{number}",
                          timeout=remaining, memory_mib=MEMORY_MIB, response_schema=schema(library))
            attempt["payload"] = raw
            built, _ = attach_proposal(library, raw, root)
            result = score(built, root, pairs)
            attempt["training"] = brief(result)
            if result["exact"]:
                winner = raw
            else:
                feedback = dict(proposal=raw, counterexamples=[r for r in result["records"] if not r["exact"]][:2])
        except (RuntimeError, ValueError) as exc:
            attempt["error"] = str(exc)
            feedback = dict(proposal=attempt.get("payload"), error=str(exc))
        if winner is None:
            attempt["feedback"] = feedback
        attempt.update(seconds=time.monotonic() - attempt_started,
                       elapsed_seconds=time.monotonic() - started)
        if log_progress:
            save(folder / "progress.json", dict(attempts=attempts, max_replies=max_replies,
                                                elapsed_seconds=attempt["elapsed_seconds"]))
            print(json.dumps(dict(event="model_reply", root=root, number=number,
                training=attempt.get("training"), error=attempt.get("error"),
                elapsed_seconds=attempt["elapsed_seconds"])), flush=True)
        if winner is not None:
            break
    return winner, dict(seconds=time.monotonic() - started, attempts=attempts,
        max_replies=max_replies, stopped="solved" if winner else
        "reply_limit" if len(attempts) == max_replies else "time_limit")


def search_process(library, pairs, root, folder, seconds, *, phase, history=None, reuse=None):
    folder.mkdir(parents=True, exist_ok=True)
    request = folder / "search-request.json"
    save(request, dict(library=library.payload(), pairs=pairs, root=root, seconds=seconds,
                       phase=phase, growth_contract=GROWTH_CONTRACT, history=history or [], reuse=reuse))
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
        response = folder / "search-response.json"
        if stopped or proc.returncode or not response.exists():
            return None, dict(seconds=time.monotonic() - started, error=stopped or "worker failed",
                              exit_code=proc.returncode, peak_process_group_rss_kib=peak)
        result = json.loads(response.read_text())
        result["effort"].update(outer_seconds=time.monotonic() - started, peak_process_group_rss_kib=peak)
        return result["candidate"], result["effort"]
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=3)


def reuse_context(effort):
    return {k: effort.get(k) for k in ("candidates", "best", "stopped", "error")}


def search_stage(library, pairs, root, folder, seconds, arm, history):
    started = time.monotonic()
    reuse_budget = min(REUSE_SECONDS, seconds)
    candidate, reused = search_process(library, pairs, root, folder / "reuse", reuse_budget,
                                       phase="reuse", history=history)
    remaining = max(0, seconds - (time.monotonic() - started))
    effort = dict(reuse=reused, reuse_budget_seconds=reuse_budget,
                  proposer_budget_seconds=remaining, proposer={})
    selected_by = "host_reuse" if candidate else None
    if candidate is None and remaining >= 2:
        context = reuse_context(reused)
        if arm == "codex":
            candidate, proposed = model_search(library, pairs, root, folder / "proposer", remaining,
                                                reuse=context, history=history)
        else:
            candidate, proposed = search_process(library, pairs, root, folder / "proposer", remaining,
                                                 phase="synthesis", history=history, reuse=context)
        effort["proposer"] = proposed
        selected_by = arm if candidate else None
    effort["seconds"] = time.monotonic() - started
    return candidate, effort, selected_by


def run(output, *, arms=("mechanical", "codex"), families=FAMILIES, seconds=180, seed=1):
    output.mkdir(parents=True, exist_ok=False)
    base = seed_library(output / "basis")
    protocol = dict(version=2, arms=list(arms), families=list(families),
                    seconds=seconds, seed=seed, model=MODEL, process_group_memory_mib=MEMORY_MIB,
                    source_hashes=sources(), basis_hash=base.digest,
                    selection="first training-exact candidate; then validation and preservation; no hard complexity gate",
                    growth_contract=GROWTH_CONTRACT, reuse_seconds=REUSE_SECONDS,
                    reuse_candidate_limit=10000, synthesis_candidate_limit=200000, model_reply_limit=2,
                    interface_hints=False, shared_reuse_first=True,
                    mechanical_scope="typed anti-unification and dataflow liftings of corpus control skeletons",
                    no_hidden_feedback=True)
    save(output / "protocol.json", protocol)
    with tarfile.open(output / "sources.tar.gz", "w:gz") as archive:
        for name in SOURCE_NAMES:
            archive.add(Path(__file__).with_name(name), arcname=name)
    rows = []
    for family in families:
        for arm in arms:
            library, history = base, []
            for stage in (1, 2):
                folder = output / family / arm / f"stage-{stage}"
                folder.mkdir(parents=True)
                row = dict(family=family, arm=arm, stage=stage, admitted=False)
                if stage == 2 and "T1" not in library.cells:
                    row.update(status="blocked: acquisition failed", effort={})
                    save(folder / "selection.json", dict(row=row, candidate=None))
                    rows.append(row)
                    continue
                print(json.dumps(dict(event="start", **row)), flush=True)
                pairs = pairs_for(family, stage, "train", seed)
                candidate, effort, selected_by = search_stage(library, pairs, f"T{stage}", folder, seconds, arm, history)
                row.update(status="no candidate", effort=effort, selected_by=selected_by)
                record = dict(row=row, candidate=candidate, input_library=library.payload(), history=list(history))
                if candidate:
                    built, certificates = attach_proposal(library, candidate, f"T{stage}")
                    built.verify()
                    training = score(built, f"T{stage}", pairs, trace_limit=100)
                    validation = score(built, f"T{stage}", pairs_for(family, stage, "validation", seed), trace_limit=100)
                    retention = {"S" + str(i): score(built, "S" + str(i), pairs_for("local", 1, "validation", effects=(e, e)))
                                 for i, e in enumerate((0, 1, 2))}
                    if stage == 2:
                        retention["T1"] = score(built, "T1", pairs_for(family, 1, "validation", seed))
                    admitted = training["exact"] and validation["exact"] and all(s["exact"] for s in retention.values())
                    helpers = [d for d in candidate["definitions"] if d["name"] != f"T{stage}"]
                    row.update(admitted=admitted, status="promoted" if admitted else "gate rejected",
                               training=brief(training), validation=brief(validation),
                               signatures={d["name"]: signature(d) for d in candidate["definitions"]},
                               new_helpers=[d["name"] for d in helpers],
                               parameterized_helpers=[d["name"] for d in helpers if d["params"]],
                               new_expression_nodes=sum(size(parse(d["body"])) for d in candidate["definitions"]))
                    record.update(library=built.payload(), certificates=certificates,
                                  training=training, validation=validation, retention=retention)
                    if admitted:
                        library = built
                        history.append(dict(root=f"T{stage}", training_examples=pairs, library_hash=library.digest))
                save(folder / "selection.json", record)
                rows.append(row)
                print(json.dumps(dict(event="selected", family=family, arm=arm, stage=stage,
                                      admitted=row["admitted"], selected_by=selected_by,
                                      seconds=effort.get("seconds"))), flush=True)
    save(output / "frozen.json", dict(rows=rows, hash=digest(rows)))
    result = evaluate_frozen(output, rows, base, seed)
    save(output / "summary.json", dict(rows=result))
    return result


def evaluate_frozen(output, rows, base, seed):
    final = []
    for row in rows:
        result = dict(row)
        folder = output / row["family"] / row["arm"] / f"stage-{row['stage']}"
        record = json.loads((folder / "selection.json").read_text())
        if row["admitted"]:
            library = InterfaceLibrary.read(record["library"])
            root = f"T{row['stage']}"
            for split in ("hidden", "stress"):
                measured = score(library, root, pairs_for(row["family"], row["stage"], split, seed), trace_limit=180)
                save(folder / f"{split}.json", measured)
                result[split] = brief(measured)
            if row["stage"] == 2:
                old = InterfaceLibrary.read(record["input_library"])
                helpers = [n for n in old.cells if n not in base.cells and n != "T1"]
                samples = pairs_for(row["family"], 2, "hidden", seed)
                executions = score(library, root, samples)
                transfer = {}
                for helper in helpers:
                    elided = score(library, root, samples, disabled=(helper,))
                    transfer[helper] = dict(hash_unchanged=old.cells[helper]["hash"] == library.cells[helper]["hash"],
                                            used_cases=sum(r["visits"].get(helper, 0) > 0 for r in executions["records"]),
                                            elided_correct=elided["correct"], total=len(samples))
                    save(folder / f"elide-{helper}.json", elided)
                result["transfer"] = transfer
                result["genuine_transfer"] = executions["exact"] and any(
                    t["hash_unchanged"] and t["used_cases"] > 0 and t["elided_correct"] < t["total"]
                    for t in transfer.values())
        final.append(result)
    # Check expressibility separately, only after selection is frozen.
    for family in sorted({r["family"] for r in rows}):
        checks = {}
        for stage in (1, 2):
            raw = witness(base, family, stage, "W")
            built, _ = attach_proposal(base, raw, "W")
            for split in ("train", "validation", "hidden", "stress"):
                result = score(built, "W", pairs_for(family, stage, split, seed))
                checks[f"{stage}-{split}"] = brief(result)
                if not result["exact"]:
                    raise ValueError("private expressibility control failed")
        save(output / family / "private-expressibility.json", checks)
    return final


def replay(output):
    protocol = json.loads((output / "protocol.json").read_text())
    if sources() != protocol["source_hashes"]:
        raise ValueError("frozen source hashes changed; use archived sources")
    corpus = json.loads((output / "basis" / "corpus.json").read_text())
    base = InterfaceLibrary.read(corpus["library"])
    if base.digest != protocol["basis_hash"]:
        raise ValueError("basis changed")
    frozen = json.loads((output / "frozen.json").read_text())
    if digest(frozen["rows"]) != frozen["hash"]:
        raise ValueError("frozen selections changed")
    count, heads, histories = 0, {}, {}
    for row in frozen["rows"]:
        folder = output / row["family"] / row["arm"] / f"stage-{row['stage']}"
        record = json.loads((folder / "selection.json").read_text())
        if row != record["row"]:
            raise ValueError("selection changed")
        if "input_library" not in record:
            continue
        library = InterfaceLibrary.read(record["input_library"])
        root = f"T{row['stage']}"
        key = (row["family"], row["arm"])
        history = histories.setdefault(key, [])
        if library.digest != heads.get(key, base.digest) or record["history"] != history:
            raise ValueError("retained lineage or public history changed")
        pairs = pairs_for(row["family"], row["stage"], "train", protocol["seed"])
        reuse = reuse_context(row["effort"]["reuse"])
        # Verify the shared probe's input and chosen candidate for both arms.
        for phase, budget, context in (("reuse", row["effort"]["reuse_budget_seconds"], None),
                                       ("synthesis", row["effort"]["proposer_budget_seconds"], reuse)):
            if phase == "synthesis" and (row["arm"] != "mechanical" or not row["effort"]["proposer"]):
                continue
            path = folder / ("reuse" if phase == "reuse" else "proposer")
            expected = dict(library=library.payload(), pairs=pairs, root=root, seconds=budget,
                            phase=phase, growth_contract=GROWTH_CONTRACT, history=history, reuse=context)
            if json.loads((path / "search-request.json").read_text()) != json.loads(json.dumps(expected)):
                raise ValueError("mechanical or shared probe input changed")
            response = path / "search-response.json"
            if response.exists():
                chosen = json.loads(response.read_text())["candidate"]
                if chosen and (chosen != record["candidate"] or row["selected_by"] !=
                               ("host_reuse" if phase == "reuse" else "mechanical")):
                    raise ValueError("mechanical or shared probe proposal changed")
        if row["arm"] == "codex":
            feedback = None
            for i, attempt in enumerate(row["effort"]["proposer"].get("attempts", []), 1):
                path = folder / "proposer" / f"proposal-{i}"
                actual = json.loads((path / "request.json").read_text())
                if actual != json.loads(json.dumps(prompt(library, pairs, root, feedback, reuse, history))):
                    raise ValueError("model received unintended information")
                if "payload" in attempt:
                    events = [json.loads(line) for line in (path / "events.jsonl").read_text().splitlines() if line]
                    raw, _ = extract(events)
                    if raw != attempt["payload"]:
                        raise ValueError("model proposal changed")
                    if attempt.get("training", {}).get("exact") and raw != record["candidate"]:
                        raise ValueError("selected model proposal changed")
                feedback = attempt.get("feedback")
        if not record["candidate"]:
            continue
        built, certificates = attach_proposal(library, record["candidate"], root)
        if json.loads(json.dumps(certificates)) != record["certificates"] or built.digest != InterfaceLibrary.read(record["library"]).digest:
            raise ValueError("attachment changed")
        graph = library.graph
        for cert in certificates:
            graph = verify_certificate(cert, graph).graph
            count += 1
        for split, field in (("train", "training"), ("validation", "validation")):
            if score(built, root, pairs_for(row["family"], row["stage"], split, protocol["seed"]), trace_limit=100) != record[field]:
                raise ValueError("admission replay changed")
        retained = {"S" + str(i): pairs_for("local", 1, "validation", effects=(e, e)) for i, e in enumerate((0, 1, 2))}
        if row["stage"] == 2:
            retained["T1"] = pairs_for(row["family"], 1, "validation", protocol["seed"])
        for name, data in retained.items():
            if score(built, name, data) != record["retention"][name]:
                raise ValueError("retention replay changed")
        admitted = record["training"]["exact"] and record["validation"]["exact"] and all(r["exact"] for r in record["retention"].values())
        if row["admitted"] != admitted:
            raise ValueError("admission changed")
        if admitted:
            heads[key] = built.digest
            history.append(dict(root=root, training_examples=json.loads(json.dumps(pairs)), library_hash=built.digest))
    old = json.loads((output / "summary.json").read_text())
    fresh = dict(rows=evaluate_frozen(output, frozen["rows"], base, protocol["seed"]))
    if old != fresh:
        raise ValueError("hidden evaluation changed")
    result = dict(verified_pushouts=count, sources_and_receipts_verified=True, fresh_execution_replay=True)
    save(output / "verification.json", result)
    return result


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--arms", nargs="+", choices=("mechanical", "codex"), default=["mechanical", "codex"])
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--seconds", type=float, default=180)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    if args.worker:
        req = json.loads(args.worker.read_text())
        try:
            search = reuse_search if req["phase"] == "reuse" else mechanical
            raw, effort = search(InterfaceLibrary.read(req["library"]), req["pairs"], req["root"], req["seconds"])
        except Exception as exc:
            raw, effort = None, dict(error=type(exc).__name__ + ": " + str(exc))
        save(args.worker.with_name("search-response.json"), dict(candidate=raw, effort=effort))
    elif args.output is None:
        parser.error("--output required")
    elif args.replay:
        print(json.dumps(replay(args.output), indent=2))
    else:
        run(args.output, arms=args.arms, families=args.families, seconds=args.seconds, seed=args.seed)
