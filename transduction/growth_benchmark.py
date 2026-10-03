"""Cumulative mechanical/Codex A/B: return-aware growth from acquired pieces."""
from __future__ import annotations

import argparse
import itertools
import json
import random
import time
from collections import deque
from pathlib import Path

from codex_glue import MODEL, extract, propose
from cofibration import verify_certificate
from glue_ab import advance
from modular_machine import (INSTRUCTIONS, Library, Machine, call, digest,
                             structural_metrics, unshare)


TASKS = ("copy_all", "duplicate_all", "swap_pairs", "reverse_triples", "keep_odd",
         "dedupe_runs", "swap_then_dedupe", "dedupe_then_swap", "reverse_then_keep",
         "reverse_swap_dedupe", "reverse_swap_dedupe_keep", "guarded_pipeline")
SEEDS = (1, 11)
FRAGMENTS = (("copy_one", 1), ("skip_one", 1), ("duplicate_one", 1),
             ("swap_pair", 2), ("reverse_triple", 3))
# Grammar order prefers smaller fresh terms; it is not an energy admission gate.
SHAPES = {"call": (1, False), "while": (1, True), "seq": (2, False),
          "if": (2, False), "pipe": (2, False), "while_then": (2, True),
          "then_while": (2, True), "while_seq": (2, True), "while_if": (2, True),
          "while_seq_then": (3, True)}


def target(name, source):
    s = tuple(source)
    if name in {"copy_one", "copy_all"}:
        return s
    if name == "skip_one":
        return ()
    if name in {"duplicate_one", "duplicate_all"}:
        return tuple(t for value in s for t in (value, value))
    if name in {"swap_pair", "swap_pairs", "reverse_triple", "reverse_triples"}:
        width = 2 if name in {"swap_pair", "swap_pairs"} else 3
        stop = len(s) // width * width
        return tuple(t for i in range(0, stop, width) for t in reversed(s[i:i + width])) + s[stop:]
    if name == "keep_odd":
        return s[::2]
    if name == "dedupe_runs":
        return tuple(t for i, t in enumerate(s) if i == 0 or t != s[i - 1])
    chains = {"swap_then_dedupe": ("swap_pairs", "dedupe_runs"),
              "dedupe_then_swap": ("dedupe_runs", "swap_pairs"),
              "reverse_then_keep": ("reverse_triples", "keep_odd"),
              "reverse_swap_dedupe": ("reverse_triples", "swap_then_dedupe"),
              "reverse_swap_dedupe_keep": ("reverse_swap_dedupe", "keep_odd")}
    if name in chains:
        for step in chains[name]:
            s = target(step, s)
        return s
    if name == "guarded_pipeline":
        return target("reverse_swap_dedupe_keep" if len(s) >= 2 and s[0] == s[1]
                      else "dedupe_then_swap", s)
    raise ValueError("unknown reference task")


def dataset(name, seed, split, width=None):
    offset, pool, lengths = {
        "train": (0, 0, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12)),
        "validation": (1001, 16, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12)),
        "hidden": (2001, 32, (0, 1, 2, 3, 4, 5, 7, 11, 17, 31, 64, 97)),
        "stress": (3001, 48, (128, 257, 512)),
    }[split]
    rng = random.Random(seed + offset)
    pairs = []
    for length in ((width,) * 4 if width else lengths):
        for pattern in range(4):
            tokens = list(range(pool, pool + 16))
            rng.shuffle(tokens)
            if pattern == 0:
                source = tuple(rng.choice(tokens[:8]) for _ in range(length))
            elif pattern == 1:
                source = tuple(tokens[(i // 2) % 4] for i in range(length))
            elif pattern == 2:
                source = (tokens[0],) * length
            else:
                source = tuple(tokens[i % 7] for i in range(length))
            pairs.append((source, target(name, source)))
    return pairs


def acquire_word(pairs, budget=30000):
    initial = tuple((0, None, None, 0) for _ in pairs)
    queue = deque([((), initial)])
    seen = {initial}
    evaluated = 0
    while queue and evaluated < budget:
        word, states = queue.popleft()
        if len(word) == 8:
            continue
        for action in INSTRUCTIONS:
            updated = advance(states, action, pairs)
            if updated is None or updated in seen:
                continue
            evaluated += 1
            seen.add(updated)
            child = word + (action,)
            if all(cursor == len(source) and written == len(out)
                   for (cursor, _, _, written), (source, out) in zip(updated, pairs)):
                return {"op": "word", "actions": list(child)}, {"evaluated": evaluated, "seen": len(seen)}
            queue.append((child, updated))
    raise RuntimeError("fragment acquisition exhausted its fixed bound")


def evaluate(library, cell, pairs, *, disabled=(), trace_limit=0, fail_fast=False):
    machine = Machine(library)
    exact = 0
    records = []
    for source, expected in pairs:
        run = machine.run(cell, source, disabled=disabled, trace_limit=trace_limit)
        good = run["ok"] and run["output"] == list(expected)
        exact += int(good)
        records.append({"source": list(source), "expected": list(expected), "exact": good, **run})
        if fail_fast and not good:
            break
    return {"correct": exact, "total": len(pairs), "tested": len(records),
            "exact": exact == len(pairs), "records": records}


def brief(score):
    return {key: score[key] for key in ("correct", "total", "tested", "exact")}


def expression(shape, a, b=None, c=None, n=None):
    A, B, C = call(a), call(b), call(c)
    def binary(op, left, right):
        return {"op": op, "children": [left, right]}
    def loop(body):
        return {"op": "while", "n": n, "children": [body]}
    if shape == "call":
        return A
    if shape == "while":
        return loop(A)
    if shape in {"seq", "if", "pipe"}:
        return binary(shape, A, B)
    if shape == "while_then":
        return binary("seq", loop(A), B)
    if shape == "then_while":
        return binary("seq", A, loop(B))
    if shape == "while_seq":
        return loop(binary("seq", A, B))
    if shape == "while_if":
        return loop(binary("if", A, B))
    if shape == "while_seq_then":
        return binary("seq", loop(binary("seq", A, B)), C)
    raise ValueError("unknown proposal shape")


def candidate_payload(library, shape, ids, n=None):
    return {"library_hash": library.digest, "shape": shape,
            **dict(zip(("a", "b", "c"), list(ids) + [None] * (3 - len(ids)))), "n": n}


def parse_proposal(library, payload):
    if not isinstance(payload, dict) or set(payload) != {"library_hash", "shape", "a", "b", "c", "n"}:
        raise ValueError("invalid proposal fields")
    if payload["library_hash"] != library.digest or payload["shape"] not in SHAPES:
        raise ValueError("unbound library or constructor")
    arity, loop = SHAPES[payload["shape"]]
    for index, key in enumerate(("a", "b", "c")):
        if index < arity:
            if not isinstance(payload[key], str) or payload[key] not in library.cells:
                raise ValueError("invalid cell binding")
        elif payload[key] is not None:
            raise ValueError("unused binding must be null")
    if (loop and (type(payload["n"]) is not int or not 1 <= payload["n"] <= 3)
            or not loop and payload["n"] is not None):
        raise ValueError("invalid loop binding")
    return expression(**{k: payload[k] for k in ("shape", "a", "b", "c", "n")})


def candidates(library):
    names = list(library.cells)
    library_hash = library.digest
    for shape, (arity, loop) in SHAPES.items():
        choices = [name for name in names if shape != "pipe" or library.cells[name]["kind"] == "whole"]
        for ids in itertools.product(choices, repeat=arity):
            for n in ((1, 2, 3) if loop else (None,)):
                yield {"library_hash": library_hash, "shape": shape,
                       **dict(zip(("a", "b", "c"), list(ids) + [None] * (3 - len(ids)))), "n": n}


def mechanical(library, cell_id, pairs, budget=30000, seconds=120):
    started = time.monotonic()
    evaluated = 0
    for payload in candidates(library):
        if evaluated >= budget or time.monotonic() - started > seconds:
            break
        evaluated += 1
        program = parse_proposal(library, payload)
        built, _ = library.attach(cell_id, program)
        score = evaluate(built, cell_id, pairs, fail_fast=True)
        if score["exact"]:
            return payload, {"evaluated": evaluated, "seconds": time.monotonic() - started, "exact_found": True}
    return None, {"evaluated": evaluated, "seconds": time.monotonic() - started, "exact_found": False}


def response_schema(library):
    cell = {"type": ["string", "null"], "enum": [*library.cells, None]}
    properties = {"library_hash": {"type": "string", "enum": [library.digest]},
                  "shape": {"type": "string", "enum": list(SHAPES)},
                  "a": {"type": "string", "enum": list(library.cells)}, "b": cell, "c": cell,
                  "n": {"type": ["integer", "null"], "enum": [1, 2, 3, None]}}
    return {"type": "object", "additionalProperties": False, "properties": properties,
            "required": list(properties)}


def prompt_for(library, pairs, feedback=None):
    return {
        "instructions": (
            "Infer the transformation ONLY from training examples and propose glue from the shared finite grammar. "
            "Tokens are opaque: only equality, not their magnitudes/identities, may matter. No tools. "
            "Return one JSON proposal bound to library_hash. Use null for unused b/c/n. "
            "Every a/b/c is an immutable retained procedure ID. New glue may not contain primitive instructions. "
            "Calls RETURN to their caller. Seq passes the current cursor and appends output. "
            "A fragment requires width remaining tokens, runs on exactly that window, consumes it, and returns. "
            "Its registers are fresh at each invocation. A whole procedure consumes the entire remaining tape. "
            "While(n,body) repeats while remaining>=n, with mandatory cursor progress. n is 1,2 or3. "
            "If(a,b) chooses a if the next TWO input tokens exist and are equal, otherwise b. "
            "Pipe(a,b) requires whole procedures: apply a to remaining tape; feed its output as a fresh tape to b; "
            "append b's output, consuming the original remaining input. "
            "Fragment primitive actions: 0=move right capped at window EOS; 1=append current token; "
            "10/11=store current token in local R0/R1; 30/31=append R0/R1. Stores at EOS and empty writes do nothing. "
            "Prefer a small attachment reusing retained behavior. Admission uses exact execution, not an energy score. "
            "You have at most two proposals; any second proposal sees only training counterexamples, never validation or hidden data."
        ),
        "grammar": {
            "call": "Call(a)", "while": "While(n,Call(a))", "seq": "Seq(Call(a),Call(b))",
            "if": "IfNextEqual(Call(a),Call(b))", "pipe": "Pipe(Call(a),Call(b))",
            "while_then": "Seq(While(n,Call(a)),Call(b))", "then_while": "Seq(Call(a),While(n,Call(b)))",
            "while_seq": "While(n,Seq(Call(a),Call(b)))", "while_if": "While(n,IfNextEqual(Call(a),Call(b)))",
            "while_seq_then": "Seq(While(n,Seq(Call(a),Call(b))),Call(c))",
        },
        "library_hash": library.digest,
        "library": {name: {k: cell[k] for k in ("kind", "width", "program", "hash", "dependencies")}
                    for name, cell in library.cells.items()},
        "training_examples": pairs, "previous_training_feedback": feedback,
    }


def save(path, data):
    path.write_text(json.dumps(data, indent=2))


def source_hashes():
    import hashlib
    return {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("growth_benchmark.py", "modular_machine.py", "cofibration.py", "codex_glue.py",
                         "pattern_fsa.py", "glue_ab.py", "run_cofibration_experiment.py")}


def acquire_library(seed, folder):
    library = Library.empty()
    folder.mkdir(parents=True)
    for index, (name, width) in enumerate(FRAGMENTS, 1):
        train = dataset(name, seed, "train", width)
        validation = dataset(name, seed, "validation", width)
        program, effort = acquire_word(train)
        cell = f"F{index:03}"
        library, square = library.attach(cell, program, kind="fragment", width=width)
        library.verify()
        scores = [evaluate(library, cell, pairs) for pairs in (train, validation)]
        if not all(score["exact"] for score in scores):
            raise RuntimeError("fragment failed fixed admission split")
        save(folder / f"{cell}.json", {"cell": cell, "name": name, "effort": effort,
                                     "certificate": square.certificate(), "library": library.payload(),
                                     "training": scores[0], "validation": scores[1]})
    save(folder / "library.json", library.payload())
    return library


def run(output, *, seeds=SEEDS, arms=("mechanical", "codex"), budget=30000, stages=12):
    output.mkdir(parents=True, exist_ok=False)
    protocol = {"version": 2, "seeds": seeds, "arms": arms, "tasks": TASKS[:stages],
                "acquired_fragments": FRAGMENTS, "grammar": SHAPES, "mechanical_budget": budget,
                "wall_seconds_per_search_or_model_call": 120, "model_proposals_per_stage": 2,
                "model": MODEL, "max_codex_process_group_mib": 768,
                "execution_limits": {"steps": 50000, "output_tokens": 4096, "call_depth": 64},
                "selection": "first training-exact proposal, then fixed validation and retention gates",
                "hidden_evaluation": "after all libraries have been frozen; no hidden feedback",
                "ablation": "elide a dependency body, leaving cursor and output unchanged; keep caller checks",
                "unshared_control": "one private procedure subtree per syntactic Call; no execution unrolling",
                "source_hashes": source_hashes()}
    save(output / "protocol.json", protocol)
    rows, final_libraries = [], {}
    for seed in seeds:
        initial = acquire_library(seed, output / f"s{seed}-acquisition")
        libraries = {arm: initial for arm in arms}
        admitted = {arm: [] for arm in arms}
        for stage, name in enumerate(TASKS[:stages], 1):
            for arm in (arms if stage % 2 else tuple(reversed(arms))):
                library, cell = libraries[arm], f"M{stage:02}"
                folder = output / f"s{seed}-{stage:02}-{arm}"
                folder.mkdir()
                train, val = dataset(name, seed, "train"), dataset(name, seed, "validation")
                attempt_records = []
                selected = None
                started = time.monotonic()
                if arm == "mechanical":
                    payload, effort = mechanical(library, cell, train, budget=budget)
                    attempt_payloads = [(payload, effort)]
                else:
                    attempt_payloads = []
                feedback = None
                for attempt in range(1 if arm == "mechanical" else 2):
                    if arm == "codex":
                        try:
                            payload = propose(prompt_for(library, train, feedback), folder / f"proposal-{attempt + 1}",
                                              response_schema=response_schema(library))
                        except RuntimeError as exc:
                            attempt_records.append({"error": str(exc), "transport_failure": True})
                            break
                    else:
                        payload, _ = attempt_payloads[0]
                    record = {"payload": payload}
                    if payload is None:
                        attempt_records.append(record)
                        break
                    try:
                        program = parse_proposal(library, payload)
                        built, square = library.attach(cell, program)
                        training = evaluate(built, cell, train, trace_limit=64)
                        record["training"] = training
                        if training["exact"]:
                            selected = built, square, payload, training
                            attempt_records.append(record)
                            break
                        feedback = {"proposal": payload, "counterexamples": [r for r in training["records"] if not r["exact"]][:4]}
                    except ValueError as exc:
                        record["error"] = str(exc)
                        feedback = {"proposal": payload, "error": str(exc)}
                    attempt_records.append(record)
                row = {"seed": seed, "stage": stage, "task": name, "arm": arm,
                       "admitted": False, "attempts": len(attempt_records), "seconds": time.monotonic() - started,
                       "before_library_hash": library.digest}
                if arm == "mechanical":
                    row["effort"] = effort
                receipt = {"row": row, "attempts": attempt_records}
                if selected:
                    built, square, payload, training = selected
                    rebuilt = verify_certificate(square.certificate(), library.graph)
                    if rebuilt.graph != built.graph:
                        raise ValueError("quotient differs from execution graph")
                    built.verify()
                    validation = evaluate(built, cell, val, trace_limit=64)
                    retention = {}
                    for old, old_name, width in [(f"F{i:03}", n, w) for i, (n, w) in enumerate(FRAGMENTS, 1)] + admitted[arm]:
                        retention[old] = brief(evaluate(built, old, dataset(old_name, seed, "validation", width)))
                    ablation = {dep: brief(evaluate(built, cell, train, disabled=(dep,)))
                                for dep in built.cells[cell]["dependencies"]}
                    row.update(training=brief(training), validation=brief(validation),
                               retention_exact=all(s["exact"] for s in retention.values()),
                               dependency_ablation=ablation, structure=structural_metrics(built, cell))
                    # Every named direct dependency must affect at least one training case.
                    row["admitted"] = validation["exact"] and row["retention_exact"] and all(not a["exact"] for a in ablation.values())
                    receipt.update(candidate=payload, certificate=square.certificate(), library=built.payload(),
                                   training=training, validation=validation, retention=retention)
                    if row["admitted"]:
                        libraries[arm] = built
                        admitted[arm].append((cell, name, None))
                save(folder / "selection.json", receipt)
                rows.append(row)
                print(json.dumps(row), flush=True)
        for arm in arms:
            final_libraries[(seed, arm)] = (libraries[arm], admitted[arm])
            save(output / f"s{seed}-{arm}-frozen.json", {"library": libraries[arm].payload(), "admitted": admitted[arm]})
    # Freeze ALL selections before looking at ANY hidden/stress outcomes.
    save(output / "frozen.json", {"rows": rows, "libraries": {f"{s}-{a}": lib.digest for (s, a), (lib, _) in final_libraries.items()}})
    for row in rows:
        if not row["admitted"]:
            continue
        library, _ = final_libraries[(row["seed"], row["arm"])]
        cell = f"M{row['stage']:02}"
        folder = output / f"s{row['seed']}-{row['stage']:02}-{row['arm']}"
        unshared, unshared_entry = unshare(library, cell)
        control = {"library": unshared.payload(), "entry": unshared_entry, "splits": {}}
        for split in ("hidden", "stress"):
            pairs = dataset(row["task"], row["seed"], split)
            score = evaluate(library, cell, pairs, trace_limit=64)
            other = evaluate(unshared, unshared_entry, pairs)
            keys = ("output", "cursor", "ok", "error", "steps", "primitive_steps", "calls", "returns", "max_call_depth", "max_buffer")
            if any(any(a[key] != b[key] for key in keys) for a, b in zip(score["records"], other["records"])):
                raise ValueError("shared and unshared executions disagree")
            control["splits"][split] = other
            row[split] = brief(score)
            row[split + "_maxima"] = {key: max(r[key] for r in score["records"]) for key in
                                      ("steps", "primitive_steps", "calls", "returns", "max_call_depth", "max_buffer")}
            save(folder / f"{split}.json", score)
        row["unshared_control_exact"] = True
        save(folder / "unshared-control.json", control)
    save(output / "summary.json", {"protocol": protocol, "rows": rows})
    return rows


def replay(output):
    protocol = json.loads((output / "protocol.json").read_text())
    if protocol["source_hashes"] != source_hashes():
        raise ValueError("experiment source has changed")
    verified = 0
    for seed in protocol["seeds"]:
        library = Library.empty()
        for index in range(1, len(FRAGMENTS) + 1):
            receipt = json.loads((output / f"s{seed}-acquisition" / f"F{index:03}.json").read_text())
            square = verify_certificate(receipt["certificate"], library.graph)
            library = Library.read(receipt["library"])
            if library.graph != square.graph:
                raise ValueError("acquisition graph mismatch")
            for split in ("training", "validation"):
                records = receipt[split]["records"]
                pairs = [(r["source"], r["expected"]) for r in records]
                if evaluate(library, receipt["cell"], pairs) != receipt[split]:
                    raise ValueError("acquisition replay mismatch")
            learned, _ = acquire_word([(r["source"], r["expected"]) for r in receipt["training"]["records"]])
            if learned != library.cells[receipt["cell"]]["program"]:
                raise ValueError("acquisition not reproducible")
            verified += 1
        initial = library
        for arm in protocol["arms"]:
            library = initial
            for stage, name in enumerate(protocol["tasks"], 1):
                folder = output / f"s{seed}-{stage:02}-{arm}"
                receipt = json.loads((folder / "selection.json").read_text())
                if receipt["row"]["before_library_hash"] != library.digest:
                    raise ValueError("lineage substitution")
                if not receipt["row"]["admitted"]:
                    continue
                cell = f"M{stage:02}"
                square = verify_certificate(receipt["certificate"], library.graph)
                built = Library.read(receipt["library"])
                expected, _ = library.attach(cell, parse_proposal(library, receipt["candidate"]))
                if built.digest != expected.digest or square.graph != built.graph:
                    raise ValueError("proposal-to-pushout mismatch")
                if arm == "codex":
                    last = len(receipt["attempts"])
                    raw = (folder / f"proposal-{last}" / "events.jsonl").read_text()
                    actual, _ = extract([json.loads(line) for line in raw.splitlines() if line])
                    if actual != receipt["candidate"]:
                        raise ValueError("candidate not from recorded model")
                    feedback = None
                    for index, attempt in enumerate(receipt["attempts"], 1):
                        request = json.loads((folder / f"proposal-{index}" / "request.json").read_text())
                        intended = json.loads(json.dumps(prompt_for(library, dataset(name, seed, "train"), feedback)))
                        if request != intended:
                            raise ValueError("model request differs from prescribed training-only prompt")
                        if "training" in attempt:
                            feedback = {"proposal": attempt["payload"], "counterexamples": [r for r in attempt["training"]["records"] if not r["exact"]][:4]}
                        else:
                            feedback = {"proposal": attempt["payload"], "error": attempt["error"]}
                else:
                    reproduced, effort = mechanical(library, cell, dataset(name, seed, "train"), budget=protocol["mechanical_budget"])
                    if reproduced != receipt["candidate"] or effort["evaluated"] != receipt["row"]["effort"]["evaluated"]:
                        raise ValueError("mechanical selection not reproducible")
                for field, split in (("training", "train"), ("validation", "validation")):
                    if evaluate(built, cell, dataset(name, seed, split), trace_limit=64) != receipt[field]:
                        raise ValueError("selected execution mismatch")
                for dep, score in receipt["row"]["dependency_ablation"].items():
                    if brief(evaluate(built, cell, dataset(name, seed, "train"), disabled=(dep,))) != score:
                        raise ValueError("dependency ablation mismatch")
                library = built
                verified += 1
            frozen = json.loads((output / f"s{seed}-{arm}-frozen.json").read_text())
            if Library.read(frozen["library"]).digest != library.digest:
                raise ValueError("frozen library mismatch")
            for cell, name, _ in frozen["admitted"]:
                stage = int(cell[1:])
                control = json.loads((output / f"s{seed}-{stage:02}-{arm}" / "unshared-control.json").read_text())
                unshared, control_entry = unshare(library, cell)
                if unshared.digest != Library.read(control["library"]).digest or control_entry != control["entry"]:
                    raise ValueError("unshared control mismatch")
                for split in ("hidden", "stress"):
                    saved = json.loads((output / f"s{seed}-{stage:02}-{arm}" / f"{split}.json").read_text())
                    if evaluate(library, cell, dataset(name, seed, split), trace_limit=64) != saved:
                        raise ValueError("final-library hidden/retention replay mismatch")
                    if evaluate(unshared, control_entry, dataset(name, seed, split)) != control["splits"][split]:
                        raise ValueError("unshared replay mismatch")
    return {"verified_pushouts": verified, "source_hashes_match": True, "all_recorded_replays_match": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", choices=("mechanical", "codex"), default=["mechanical", "codex"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--stages", type=int, default=12)
    parser.add_argument("--budget", type=int, default=30000)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.stages <= len(TASKS) or not 1 <= args.budget <= 30000:
        parser.error("invalid stage or candidate bound")
    if args.replay:
        print(json.dumps(replay(args.output)))
    else:
        run(args.output, seeds=tuple(args.seeds), arms=tuple(args.arms), budget=args.budget, stages=args.stages)
