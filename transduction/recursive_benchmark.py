"""Paired recursive acquisition and transfer with exact graph attachments."""
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
import time
from pathlib import Path

from codex_glue import MODEL, _rss_kib, extract, propose
from cofibration import verify_certificate
from growth_benchmark import acquire_library, acquire_word
from modular_machine import Machine as FragmentMachine, digest
from recursive_machine import Machine, RecursiveLibrary, node, parse, schema
from recursive_search import brief, mechanical, prompt, score


FAMILIES = ("right_parity", "two_contexts")
DEFAULT_BINDINGS = ("C01", "F004")
TRANSFER_BINDINGS = ("C02", "C03")


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True))


def sources():
    return {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("recursive_machine.py", "recursive_search.py", "recursive_benchmark.py",
                         "modular_machine.py", "growth_benchmark.py", "cofibration.py",
                         "codex_glue.py", "glue_ab.py", "pattern_fsa.py")}


def acquire_basis(folder):
    base = acquire_library(1, folder / "original-fragments")
    # Additional machines, not additional primitive instructions. Bodies are
    # discovered by the same old word synthesizer, never supplied as solutions.
    for name, function in (("C01", lambda a, b: (a, b)),
                           ("C02", lambda a, b: (a, a, b, b)),
                           ("C03", lambda a, b: (a,))):
        train = [((a, b), function(a, b)) for a, b in itertools.product(range(3), repeat=2)]
        validation = [((a, b), function(a, b)) for a, b in itertools.product(range(11, 14), repeat=2)]
        program, effort = acquire_word(train)
        base, square = base.attach(name, program, kind="fragment", width=2)
        base.verify()
        machine = FragmentMachine(base)
        if not all(machine.run(name, s)["output"] == list(t) for s, t in train + validation):
            raise ValueError("pair fragment acquisition failed")
        save(folder / f"{name}.json", dict(program=program, effort=effort, training=train,
                                         validation=validation, certificate=square.certificate(),
                                         library=base.payload()))
    result = RecursiveLibrary.from_base(base)
    save(folder / "library.json", result.payload())
    return result


def specification(family):
    if family == "right_parity":
        return dict(bits=1, left_mask=0, right_mask=1, truth=[0, 1], maximum=12)
    if family == "two_contexts":
        return dict(bits=2, left_mask=1, right_mask=2, truth=[0, 0, 0, 1], maximum=15)
    raise ValueError("unknown family")


def shapes(leaves):
    if leaves == 1:
        return [None]
    return [(a, b) for n in range(1, leaves) for a in shapes(n) for b in shapes(leaves - n)]


def random_shape(leaves, rng):
    if leaves == 1:
        return None
    n = rng.randrange(1, leaves)
    return (random_shape(n, rng), random_shape(leaves - n, rng))


def comb(depth, direction):
    value = None
    for i in range(depth):
        value = (None, value) if direction == "right" or direction == "alternating" and i % 2 else (value, None)
    return value


def materialize(shape, rng, pool):
    if shape is None:
        a, b = rng.sample(range(pool, pool + 32), 2)
        return {"pair": (a, b)}
    return {"marker": rng.randrange(pool, pool + 32),
            "left": materialize(shape[0], rng, pool), "right": materialize(shape[1], rng, pool)}


def encode_tree(tree):
    if "pair" in tree:
        return tuple(tree["pair"])
    return (tree["marker"],) * 2 + encode_tree(tree["left"]) + encode_tree(tree["right"])


def reference(tree, spec, stage, context=0):
    # Independent tree oracle: never executes the proposal grammar or a machine.
    if "pair" in tree:
        a, b = tree["pair"]
        selected = spec["truth"][context]
        return ((b, a) if selected else (a, b)) if stage == 1 else ((a,) if selected else (a, a, b, b))
    return (reference(tree["left"], spec, stage, context ^ spec["left_mask"])
            + reference(tree["right"], spec, stage, context ^ spec["right_mask"]))


def pairs_for(family, stage, split):
    offsets = {"train": (7, 0), "validation": (101, 100), "hidden": (503, 1000), "stress": (997, 2000)}
    seed, pool = offsets[split]
    rng = random.Random(seed)
    if split in ("train", "validation"):
        forms = [shape for count in range(1, 5) for shape in shapes(count) for _ in range(2)]
        forms += [random_shape(5, rng) for _ in range(8)]
    elif split == "hidden":
        forms = [random_shape(count, rng) for count in (6, 9, 16, 32, 64) for _ in range(8)]
    else:
        forms = [comb(depth, side) for depth in (16, 48, 96, 120) for side in ("left", "right", "alternating")]
        forms += [random_shape(128, rng) for _ in range(4)]
    trees = [materialize(shape, rng, pool) for shape in forms]
    spec = specification(family)
    return [(encode_tree(tree), reference(tree, spec, stage)) for tree in trees]


def witness(family):
    """Private expressibility certificate, never provided to either proposer."""
    spec = specification(family)
    nodes = [node("eq", a=1, b=5), node("call", cell="F002", a=2),
             node("call", cell="F002", a=3), node("recur", arg=spec["left_mask"], a=4),
             node("recur", arg=spec["right_mask"], a=0)]
    def decision(bit, choices):
        if len(set(choices)) == 1:
            return -1 - choices[0]
        here = len(nodes)
        nodes.append(None)
        low = decision(bit + 1, choices[::2])
        high = decision(bit + 1, choices[1::2])
        nodes[here] = node("bit", arg=bit, a=high, b=low)
        return here
    leaf = decision(0, spec["truth"])
    nodes[0]["b"] = leaf
    params = [len(nodes), len(nodes) + 1]
    end = len(nodes) + 2
    nodes.extend([node("param", arg=0, a=end), node("param", arg=1, a=end), node("ret")])
    nodes[4]["a"] = end
    for n in nodes:
        for key in ("a", "b"):
            if n[key] is not None and n[key] < 0:
                n[key] = params[-n[key] - 1]
    return nodes


def model_search(library, pairs, bits, maximum, folder, seconds):
    started = time.monotonic()
    feedback, answer, attempts = None, None, []
    for index in range(1, 3):
        remaining = seconds - (time.monotonic() - started)
        if remaining < 2:
            break
        item = {}
        attempts.append(item)
        try:
            raw = propose(prompt(library, pairs, bits, maximum, DEFAULT_BINDINGS, feedback),
                          folder / f"proposal-{index}", timeout=remaining, memory_mib=640,
                          response_schema=schema(library, maximum))
            item["payload"] = raw
            nodes = parse(raw, library, bits, maximum)
            built, _ = library.attach("Candidate", nodes, bits)
            result = score(built, "Candidate", pairs, DEFAULT_BINDINGS)
            item["training"] = brief(result)
            if result["exact"]:
                answer = nodes
                break
            feedback = {"proposal": raw, "counterexamples": [r for r in result["records"] if not r["exact"]][:3]}
            item["feedback"] = feedback
        except (ValueError, RuntimeError) as exc:
            item["error"] = str(exc)
            feedback = {"proposal": item.get("payload"), "error": str(exc)}
    return answer, dict(seconds=time.monotonic() - started, attempts=attempts)


def _mechanical_process_once(library, pairs, bits, maximum, folder, seconds):
    folder.mkdir(parents=True, exist_ok=True)
    request = folder / "mechanical-request.json"
    response = folder / "mechanical-response.json"
    save(request, dict(library=library.payload(), pairs=pairs, bits=bits, maximum=maximum,
                       bindings=DEFAULT_BINDINGS, seconds=seconds))
    started, peak = time.monotonic(), 0
    proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", str(request)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    stopped = None
    try:
        while proc.poll() is None:
            peak = max(peak, _rss_kib(proc.pid))
            if peak > 640 * 1024 or time.monotonic() - started > seconds + 5:
                stopped = "memory limit" if peak > 640 * 1024 else "outer wall-time limit"
                os.killpg(proc.pid, signal.SIGTERM)
                break
            time.sleep(0.25)
        try:
            stdout, stderr = proc.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            stdout, stderr = proc.communicate(timeout=3)
        (folder / "mechanical-stdout.log").write_bytes(stdout)
        (folder / "mechanical-stderr.log").write_bytes(stderr)
        if stopped or proc.returncode != 0 or not response.exists():
            return None, dict(seconds=time.monotonic() - started, error=stopped or "worker failed",
                              exit_code=proc.returncode, peak_process_group_rss_kib=peak)
        result = json.loads(response.read_text())
        result["effort"]["outer_seconds"] = time.monotonic() - started
        result["effort"]["peak_process_group_rss_kib"] = peak
        return result["nodes"], result["effort"]
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=3)


def mechanical_process(library, pairs, bits, maximum, folder, seconds):
    """Fresh workers for increasing program bounds; bounded failures survive."""
    started = time.monotonic()
    save(folder / "mechanical-request.json", dict(library=library.payload(), pairs=pairs, bits=bits,
                                                 maximum=maximum, bindings=DEFAULT_BINDINGS, seconds=seconds))
    attempts, answer = [], None
    sizes = sorted({min(n, maximum) for n in (2, 4, 8, maximum)})
    for size in sizes:
        remaining = seconds - (time.monotonic() - started)
        if remaining < 2:
            break
        allowance = remaining if size == maximum else min(30, remaining / 2)
        nodes, effort = _mechanical_process_once(library, pairs, bits, size,
                                                 folder / f"nodes-{size}", allowance)
        attempts.append(dict(maximum_nodes=size, effort=effort))
        if nodes:
            answer = nodes
            break
    effort = dict(seconds=time.monotonic() - started, attempts=attempts,
                  trace_horizon="4 * (input_tokens + 1)",
                  peak_process_group_rss_kib=max((a["effort"].get("peak_process_group_rss_kib", 0)
                                                 for a in attempts), default=0))
    save(folder / "mechanical-response.json", dict(nodes=answer, effort=effort))
    return answer, effort


def run(output, *, arms=("mechanical", "codex"), families=FAMILIES, seconds=120):
    output.mkdir(parents=True, exist_ok=False)
    base = acquire_basis(output / "basis")
    protocol = dict(version=2, arms=list(arms), families=list(families), seconds=seconds,
                    model=MODEL, max_process_group_mib=640, source_hashes=sources(),
                    basis_hash=base.digest, default_bindings=DEFAULT_BINDINGS,
                    shared_grammar="forward CFG, equality/context tests, fragment/parameter calls, recursive calls, retained invocation",
                    mechanical="reuse-first adapters then bounded CEGIS/SMT; fresh workers at 2,4,8,max nodes; no traversal sketch",
                    smt_trace_horizon="4 * (input_tokens + 1)",
                    selection="first training-exact candidate, then fixed validation and retention gates",
                    no_hidden_feedback=True)
    save(output / "protocol.json", protocol)
    save(output / "private-specifications.json", {f: specification(f) for f in families})
    rows = []
    for family in families:
        spec = specification(family)
        for arm in arms:
            library = base
            for stage in (1, 2):
                folder = output / family / arm / f"stage-{stage}"
                folder.mkdir(parents=True)
                print(json.dumps(dict(event="start", family=family, arm=arm, stage=stage)), flush=True)
                if stage == 2 and "R01" not in library.cells:
                    row = dict(family=family, arm=arm, stage=stage, admitted=False,
                               status="blocked: acquisition did not pass", effort={})
                    save(folder / "selection.json", dict(row=row, candidate=None))
                    rows.append(row)
                    continue
                pairs = pairs_for(family, stage, "train")
                search = mechanical_process if arm == "mechanical" else model_search
                nodes, effort = search(library, pairs, spec["bits"], spec["maximum"], folder, seconds)
                row = dict(family=family, arm=arm, stage=stage, admitted=False,
                           status="no candidate", effort=effort)
                record = dict(row=row, candidate=nodes, input_library=library.payload())
                if nodes:
                    root = f"R{stage:02}"
                    built, square = library.attach(root, nodes, spec["bits"])
                    built.verify()
                    training = score(built, root, pairs, DEFAULT_BINDINGS, trace_limit=96)
                    validation = score(built, root, pairs_for(family, stage, "validation"), DEFAULT_BINDINGS, trace_limit=96)
                    retention = (score(built, "R01", pairs_for(family, 1, "validation"), DEFAULT_BINDINGS)
                                 if stage == 2 else None)
                    row.update(admitted=training["exact"] and validation["exact"] and (retention is None or retention["exact"]),
                               nodes=len(nodes), dependencies=built.cells[root]["dependencies"],
                               training=brief(training), validation=brief(validation))
                    row["status"] = "promoted" if row["admitted"] else "gate rejected"
                    record.update(library=built.payload(), certificate=square.certificate(),
                                  training=training, validation=validation, retention=retention)
                    if row["admitted"]:
                        library = built
                save(folder / "selection.json", record)
                rows.append(row)
                print(json.dumps(dict(event="selected", family=family, arm=arm, stage=stage,
                                      admitted=row["admitted"], status=row["status"],
                                      seconds=effort.get("seconds"))), flush=True)
    save(output / "frozen.json", dict(rows=rows, basis_hash=base.digest))
    final = evaluate_frozen(output, rows, base)
    save(output / "summary.json", dict(rows=final))
    return final


def evaluate_frozen(output, rows, base):
    final = []
    for row in rows:
        result = dict(row)
        folder = output / row["family"] / row["arm"] / f"stage-{row['stage']}"
        record = json.loads((folder / "selection.json").read_text())
        if row["admitted"]:
            library = RecursiveLibrary.read(record["library"])
            root = f"R{row['stage']:02}"
            for split in ("hidden", "stress"):
                evaluated = score(library, root, pairs_for(row["family"], row["stage"], split),
                                  DEFAULT_BINDINGS, trace_limit=128)
                save(folder / f"{split}.json", evaluated)
                result[split] = brief(evaluated)
                if split == "stress":
                    result["max_call_depth"] = max(r["max_call_depth"] for r in evaluated["records"])
                    result["max_steps"] = max(r["steps"] for r in evaluated["records"])
            if row["stage"] == 2:
                samples = pairs_for(row["family"], 2, "hidden")
                elided = score(library, root, samples, DEFAULT_BINDINGS, disabled=("R01",))
                save(folder / "reuse-ablation.json", elided)
                executions = score(library, root, samples, DEFAULT_BINDINGS, trace_limit=256)
                uses = all(r["visits"].get("R01", 0) > 0 for r in executions["records"])
                result["transfer"] = dict(calls_retained_traversal=uses,
                                          old_traversal_hash_unchanged=library.cells["R01"]["hash"] == record["input_library"]["cells"]["R01"]["hash"],
                                          elided_correct=elided["correct"], total=elided["total"])
        final.append(result)
    # Expressibility witnesses are checked only after selection has been frozen.
    for family in sorted({r["family"] for r in rows}):
        spec = specification(family)
        library, first = base.attach("W01", witness(family), spec["bits"])
        adapter = [node("invoke", arg=0, cell="W01", a=1, bindings=TRANSFER_BINDINGS), node("ret")]
        library, second = library.attach("W02", adapter, spec["bits"])
        checks = {f"stage-{stage}-{split}": brief(score(library, f"W{stage:02}", pairs_for(family, stage, split), DEFAULT_BINDINGS))
                  for stage in (1, 2) for split in ("train", "validation", "hidden", "stress")}
        if not all(value["exact"] for value in checks.values()):
            raise ValueError("private expressibility witness failed")
        save(output / family / "private-witness.json", dict(library=library.payload(), checks=checks,
                                                           certificates=[first.certificate(), second.certificate()]))
    return final


def replay(output):
    protocol = json.loads((output / "protocol.json").read_text())
    if protocol["source_hashes"] != sources():
        raise ValueError("frozen experiment source hashes changed")
    base = RecursiveLibrary.read(json.loads((output / "basis" / "library.json").read_text()))
    if base.digest != protocol["basis_hash"]:
        raise ValueError("basis changed")
    frozen = json.loads((output / "frozen.json").read_text())
    count = 0
    for row in frozen["rows"]:
        folder = output / row["family"] / row["arm"] / f"stage-{row['stage']}"
        record = json.loads((folder / "selection.json").read_text())
        if row != record["row"]:
            raise ValueError("selection changed after freeze")
        if "input_library" not in record:
            continue
        library = RecursiveLibrary.read(record["input_library"])
        spec = specification(row["family"])
        pairs = pairs_for(row["family"], row["stage"], "train")
        if row["arm"] == "codex":
            feedback = None
            for i, attempt in enumerate(row["effort"]["attempts"], 1):
                call = folder / f"proposal-{i}"
                request = json.loads((call / "request.json").read_text())
                if request != json.loads(json.dumps(prompt(library, pairs, spec["bits"], spec["maximum"], DEFAULT_BINDINGS, feedback))):
                    raise ValueError("model request included unintended information")
                if "payload" in attempt:
                    events = [json.loads(line) for line in (call / "events.jsonl").read_text().splitlines() if line]
                    raw, _ = extract(events)
                    if raw != attempt["payload"]:
                        raise ValueError("model proposal replaced")
                    if attempt.get("training", {}).get("exact") and parse(raw, library, spec["bits"], spec["maximum"]) != record["candidate"]:
                        raise ValueError("model winner changed")
                feedback = attempt.get("feedback", {"proposal": attempt.get("payload"), "error": attempt.get("error")})
        else:
            request = json.loads((folder / "mechanical-request.json").read_text())
            expected = dict(library=library.payload(), pairs=pairs, bits=spec["bits"], maximum=spec["maximum"],
                            bindings=DEFAULT_BINDINGS, seconds=protocol["seconds"])
            if request != json.loads(json.dumps(expected)):
                raise ValueError("mechanical request included unintended information")
            response = folder / "mechanical-response.json"
            if response.exists() and json.loads(response.read_text())["nodes"] != record["candidate"]:
                raise ValueError("mechanical proposal replaced")
        if not record["candidate"]:
            continue
        root = f"R{row['stage']:02}"
        built, square = library.attach(root, record["candidate"], spec["bits"])
        if (json.loads(json.dumps(square.certificate())) != record["certificate"]
                or built.digest != RecursiveLibrary.read(record["library"]).digest):
            raise ValueError("attachment derivation changed")
        verify_certificate(record["certificate"], library.graph)
        count += 1
        for split, field in (("train", "training"), ("validation", "validation")):
            if score(built, root, pairs_for(row["family"], row["stage"], split), DEFAULT_BINDINGS, trace_limit=96) != record[field]:
                raise ValueError("admission execution changed")
        if row["stage"] == 2 and score(built, "R01", pairs_for(row["family"], 1, "validation"), DEFAULT_BINDINGS) != record["retention"]:
            raise ValueError("retention changed")
        admitted = record["training"]["exact"] and record["validation"]["exact"] and (
            record["retention"] is None or record["retention"]["exact"])
        if row["admitted"] != admitted:
            raise ValueError("incorrect admission decision")
    original = json.loads((output / "summary.json").read_text())
    final = evaluate_frozen(output, frozen["rows"], base)
    if original != dict(rows=final):
        raise ValueError("hidden evaluation changed")
    result = dict(verified_pushouts=count, source_hashes_match=True, model_requests_and_receipts_verified=True,
                  mechanical_requests_and_receipts_verified=True, fresh_replay=True)
    save(output / "verification.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--arms", nargs="+", choices=("mechanical", "codex"), default=["mechanical", "codex"])
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    if args.worker:
        request = json.loads(args.worker.read_text())
        library = RecursiveLibrary.read(request["library"])
        try:
            nodes, effort = mechanical(library, request["pairs"], request["bits"], request["maximum"],
                                       request["bindings"], seconds=request["seconds"])
        except Exception as exc:
            nodes, effort = None, dict(error=type(exc).__name__ + ": " + str(exc))
        save(args.worker.with_name("mechanical-response.json"), dict(nodes=nodes, effort=effort))
    elif args.output is None:
        parser.error("--output is required")
    elif args.replay:
        print(json.dumps(replay(args.output), indent=2))
    else:
        run(args.output, arms=args.arms, families=args.families, seconds=args.seconds)
