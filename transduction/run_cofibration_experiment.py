"""Bounded, held-out comparison of cold evolution and retained-machine attachment.

This is a new variable-length curriculum, not a rerun of the old ten-row matrix.
All labels live in the evaluator. Proposers see training examples only.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import time
from pathlib import Path

from cofibration import Graph, parse_attachment, read_graph, verify_certificate
from pattern_fsa import (OBS_TOKEN, PatternRule, PatternTask, evolve_solver,
                         normalized_edit_distance, register_primitives,
                         run_transducer, transform_sequence)


TASKS = ("copy", "duplicate_first", "swap", "triple_first")


def examples(task, seed, count, pool_start, lengths):
    rng = random.Random(seed)
    pairs = []
    for index in range(count):
        length = lengths[index % len(lengths)]
        source = tuple(rng.randrange(pool_start, pool_start + 16) for _ in range(length))
        target = (source[:1] * 2 + source if task == "triple_first"
                  else transform_sequence(task, source, 16))
        pairs.append((source, target))
    return tuple(pairs)


def make_task(name, seed):
    return PatternTask(
        name, 48,
        examples(name, seed, 12, 0, (2, 3, 4, 5)),
        examples(name, seed + 101, 12, 16, (2, 3, 4, 5)),
        examples(name, seed + 202, 48, 32, tuple(range(2, 14))),
    )


def evaluate(graph, entry, pairs, retained_states=0):
    genome = graph.genome()
    traces = []
    for source, target in pairs:
        run = run_transducer(genome, source, graph.primitives, max_steps=128,
                             entry_state=entry)
        traces.append({"source": source, "target": target, "output": run.output,
                       "halted": run.halted, "steps": run.steps,
                       "active_rules": run.active_rules,
                       "reused_rules": sum(state < retained_states
                                           for state, _ in run.active_rules)})
    return {
        "loss": sum(normalized_edit_distance(t["output"], t["target"])
                    for t in traces) / len(traces),
        "exact": sum(t["output"] == t["target"] and t["halted"]
                     for t in traces) / len(traces),
        "reuse_cases": sum(t["reused_rules"] > 0 for t in traces),
        "traces": traces,
    }


def seed_machine(task, primitives):
    """Acquire the initial machine by generic enumeration, not a planted solver."""
    tested = 0
    for length in range(1, 5):
        for actions in itertools.product(primitives.actions, repeat=length):
            graph = Graph(1, (PatternRule(0, OBS_TOKEN, actions, 0),), primitives)
            score = evaluate(graph, 0, task.train_pairs)
            tested += 1
            if score["exact"] == 1:
                if evaluate(graph, 0, task.val_pairs)["exact"] != 1:
                    raise RuntimeError("seed failed validation; no promotion")
                return graph, tested
    raise RuntimeError("seed grammar did not acquire the first task")


def rigid_proposals(library, budget):
    tested = 0
    # Exhaust all action words in length order and every retained state binding.
    # There are no task-specific action templates or manually authored solutions.
    for length in range(1, 7):
        for actions in itertools.product(library.primitives.actions, repeat=length):
            for port in range(library.states):
                if tested >= budget:
                    return
                tested += 1
                yield {"library_hash": library.digest, "fresh_states": 1,
                       "ports": [port], "rules": [
                           {"state": 0, "observation": OBS_TOKEN,
                            "actions": list(actions), "next_state": 1}]}


def search_rigid(library, task, budget):
    best = None
    tested = 0
    for payload in rigid_proposals(library, budget):
        square, entry = parse_attachment(payload, library)
        score = evaluate(square.graph, entry, task.train_pairs, library.states)
        tested += 1
        objective = score["loss"] + 0.002 * square.right.complexity
        if best is None or objective < best[0]:
            best = (objective, payload, square, entry, score)
        if score["exact"] == 1 and necessary_reuse(square.graph, entry, task.train_pairs, library.states):
            # Enumeration is in increasing glue size. No test/validation feedback.
            best = (objective, payload, square, entry, score)
            break
    return best, tested


def necessary_reuse(graph, entry, pairs, retained_states):
    """Old transitions must change task accuracy, not merely appear in a trace."""
    ablated = Graph(graph.states, tuple(rule for rule in graph.rules
                                       if rule.state >= retained_states), graph.primitives)
    return evaluate(ablated, entry, pairs)["exact"] < evaluate(graph, entry, pairs)["exact"]


def cold_search(task, primitives, seed, budget):
    population = 32
    generations = max(0, budget // population - 1)
    best, _, _, _ = evolve_solver(
        task, primitives, seed=seed, generations=generations,
        population_size=population, state_count=4, max_states=4,
        initial_rule_count=4, max_rules=8, max_rule_length=6,
        mutation_rate=0.12, lambda_value=0.002, max_steps=128, report_every=0,
    )
    return Graph(best.state_count, tuple(best.rules), primitives), (generations + 1) * population


def proposal_prompt(library, task, feedback):
    return {
        "instructions": (
            "Infer the sequence transformation from the training examples. Propose ONLY "
            "a JSON attachment, not executable Python. Reuse retained transitions with "
            "small new glue. Do not modify retained states or rules. Each execution "
            "starts with head at input index 0, empty output, registers empty. A rule "
            "is selected by (state, observation), then all its actions execute in order "
            "before next_state is entered. Missing rules halt. Observations are 1=TOKEN "
            "and 0=EOS. Actions: 0 moves head right (capped at EOS); 1 appends the "
            "current token if present; 2 halts immediately; 10 stores current token "
            "in R0; 30 appends R0 if set. Token values are opaque and cannot appear "
            "as constants in rules. Local states [0,fresh_states) are new; port j "
            "has local state fresh_states+j, identified with library state ports[j]. "
            "Only fresh states may have new outgoing rules. Local state 0 is the "
            "new entry. Retained rules must contribute to solving the task: removing "
            "them must worsen training accuracy. Some inputs may terminate before "
            "entering the retained machine. Maximum 8 fresh states, 64 rules, "
            "8 actions per rule. Prefer "
            "the smallest correct attachment. Do not use tools or inspect files."
        ),
        "library_hash": library.digest, "library": library.payload(),
        "training_examples": task.train_pairs, "prior_attempts": feedback,
    }


def search_codex(library, task, attempts, artifact_dir):
    from codex_glue import propose
    feedback = []
    best = None
    for index in range(attempts):
        attempt_dir = artifact_dir / f"attempt-{index + 1:02d}"
        payload = propose(proposal_prompt(library, task, feedback), attempt_dir)
        try:
            square, entry = parse_attachment(payload, library)
            score = evaluate(square.graph, entry, task.train_pairs, library.states)
            objective = score["loss"] + 0.002 * square.right.complexity
            if best is None or objective < best[0]:
                best = (objective, payload, square, entry, score)
            feedback.append({"proposal": payload, "training": score})
            if score["exact"] == 1 and necessary_reuse(square.graph, entry, task.train_pairs, library.states):
                best = (objective, payload, square, entry, score)
                break
        except ValueError as exc:
            feedback.append({"proposal": payload, "rejection": str(exc)})
    if best is None:
        raise RuntimeError("all Codex proposals were structurally invalid")
    return best, len(feedback)


def summary(evaluation):
    return {key: value for key, value in evaluation.items() if key != "traces"}


def run(seed, budget, output, with_codex=False):
    primitives = register_primitives(48, register_count=1)
    tasks = [make_task(name, seed) for name in TASKS]
    anchor, acquisition_evaluations = seed_machine(tasks[0], primitives)
    output.mkdir(parents=True, exist_ok=False)
    (output / "acquisition.json").write_text(json.dumps({
        "graph": anchor.payload(), "hash": anchor.digest,
        "candidate_evaluations": acquisition_evaluations,
        "training": evaluate(anchor, 0, tasks[0].train_pairs),
        "validation": evaluate(anchor, 0, tasks[0].val_pairs),
    }, indent=2))
    rows = []
    frozen = []  # Hidden labels are evaluated only after every selection is frozen.
    for method in ("cold", "rigid") + (("codex",) if with_codex else ()):
        library = anchor
        retained = [(0, tasks[0])]
        for number, task in enumerate(tasks[1:], 2):
            started = time.monotonic()
            artifact_dir = output / f"{method}-T{number:02d}"
            artifact_dir.mkdir()
            if method == "cold":
                graph, tested = cold_search(task, primitives, seed, budget)
                entry = 0
                reuse_states = 0
                certificate = None
                candidate = None
            else:
                if method == "rigid":
                    best, tested = search_rigid(library, task, budget)
                else:
                    best, tested = search_codex(library, task, 3, artifact_dir)
                _, candidate, square, entry, _ = best
                graph = square.graph
                reuse_states = library.states
                certificate = square.certificate()
            training = evaluate(graph, entry, task.train_pairs, reuse_states)
            validation = evaluate(graph, entry, task.val_pairs, reuse_states)
            retention = [evaluate(graph, old_entry, old_task.train_pairs)["exact"]
                         for old_entry, old_task in retained] if method != "cold" else []
            promoted = training["exact"] == validation["exact"] == 1 and all(v == 1 for v in retention)
            ablated_training_exact = None
            if method != "cold":
                ablated = Graph(graph.states, tuple(rule for rule in graph.rules
                                                   if rule.state >= reuse_states), primitives)
                ablated_training_exact = evaluate(ablated, entry, task.train_pairs)["exact"]
                promoted = promoted and ablated_training_exact < training["exact"]
                # Reconstruct from proposal bytes and replay rather than trusting the search object.
                replay, replay_entry = parse_attachment(json.loads(json.dumps(candidate)), library)
                if replay.graph.digest != graph.digest or replay_entry != entry:
                    raise RuntimeError("attachment reconstruction mismatch")
                if evaluate(replay.graph, replay_entry, task.train_pairs, reuse_states) != training:
                    raise RuntimeError("fresh execution mismatch")
            row = {"seed": seed, "method": method, "task": task.name,
                   "candidate_evaluations": tested, "budget": budget if method != "codex" else 3,
                   "seconds": time.monotonic() - started, "entry": entry,
                   "states": graph.states, "complexity": graph.complexity,
                   "new_complexity": graph.complexity - (library.complexity if method != "cold" else 0),
                   "training": summary(training), "validation": summary(validation),
                   "without_retained_transitions_training_exact": ablated_training_exact,
                   "retention": retention, "promoted": promoted, "hash": graph.digest}
            (artifact_dir / "selection.json").write_text(json.dumps({
                **row, "candidate": candidate, "certificate": certificate,
                "graph": graph.payload(), "training_replay": training,
                "validation_replay": validation,
            }, indent=2))
            rows.append(row)
            frozen.append((row, graph, entry, task, reuse_states, artifact_dir))
            print(json.dumps(row), flush=True)
            if promoted and method != "cold":
                library = graph
                retained.append((entry, task))
        if method != "cold":
            (output / f"{method}-library.json").write_text(json.dumps({
                "graph": library.payload(), "hash": library.digest,
                "entries": [{"entry": entry, "task": task.name} for entry, task in retained],
            }, indent=2))
    for row, graph, entry, task, reuse_states, artifact_dir in frozen:
        test = evaluate(graph, entry, task.test_pairs, reuse_states)
        row["test"] = summary(test)
        (artifact_dir / "hidden-replay.json").write_text(json.dumps(test, indent=2))
    (output / "summary.json").write_text(json.dumps({
        "protocol": "frozen selections before hidden evaluation; old entries preserved",
        "acquisition_evaluations": acquisition_evaluations, "rows": rows,
    }, indent=2))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--budget", type=int, default=8192)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--codex", action="store_true", help="use existing Codex login and gpt-5.6-sol")
    parser.add_argument("--replay", action="store_true", help="verify an existing run without search or model calls")
    args = parser.parse_args()
    if args.budget < 32 or args.budget > 32768:
        parser.error("budget must be in 32..32768")
    if args.replay:
        print(json.dumps(replay_run(args.output), indent=2))
    else:
        run(args.seed, args.budget, args.output, args.codex)


def replay_run(output):
    """Independent receipt replay, including earlier tasks in the final grown graph."""
    anchor_receipt = json.loads((output / "acquisition.json").read_text())
    anchor = read_graph(anchor_receipt["graph"])
    if anchor.digest != anchor_receipt["hash"]:
        raise ValueError("acquisition hash mismatch")
    report = json.loads((output / "summary.json").read_text())
    libraries, entries = {}, {}
    verified = 0

    def equal(left, right):
        if json.dumps(left, sort_keys=True) != json.dumps(right, sort_keys=True):
            raise ValueError("recorded replay disagrees with fresh execution")

    for row in report["rows"]:
        method, name, seed = row["method"], row["task"], row["seed"]
        if method not in {"cold", "rigid", "codex"} or name not in TASKS[1:]:
            raise ValueError("unknown experiment condition")
        task = make_task(name, seed)
        folder = output / f"{method}-T{TASKS.index(name) + 1:02d}"
        receipt = json.loads((folder / "selection.json").read_text())
        library = libraries.setdefault(method, anchor)
        retained = entries.setdefault(method, [(0, make_task("copy", seed))])
        graph = read_graph(receipt["graph"])
        entry = receipt["entry"]
        if graph.digest != receipt["hash"] or graph.digest != row["hash"]:
            raise ValueError("selected graph hash mismatch")
        old_states = 0
        if method != "cold":
            square = verify_certificate(receipt["certificate"], library)
            rebuilt, rebuilt_entry = parse_attachment(receipt["candidate"], library)
            if square.graph != graph or rebuilt.graph != graph or rebuilt_entry != entry:
                raise ValueError("candidate and certificate disagree")
            old_states = library.states
            verified += 1
        training = evaluate(graph, entry, task.train_pairs, old_states)
        validation = evaluate(graph, entry, task.val_pairs, old_states)
        equal(training, receipt["training_replay"])
        equal(validation, receipt["validation_replay"])
        equal(summary(training), row["training"])
        equal(summary(validation), row["validation"])
        test = evaluate(graph, entry, task.test_pairs, old_states)
        equal(test, json.loads((folder / "hidden-replay.json").read_text()))
        equal(summary(test), row["test"])
        promotion = training["exact"] == validation["exact"] == 1
        if method != "cold":
            retention = [evaluate(graph, old_entry, old_task.train_pairs)["exact"]
                         for old_entry, old_task in retained]
            equal(retention, receipt["retention"])
            promotion = promotion and all(value == 1 for value in retention)
            promotion = promotion and necessary_reuse(graph, entry, task.train_pairs, old_states)
        if promotion != receipt["promoted"] or promotion != row["promoted"]:
            raise ValueError("incorrect promotion decision")
        if promotion and method != "cold":
            libraries[method] = graph
            retained.append((entry, task))
    final_retention = {}
    for method, library in libraries.items():
        if method == "cold":
            continue
        stored = json.loads((output / f"{method}-library.json").read_text())
        if library.digest != stored["hash"] or read_graph(stored["graph"]) != library:
            raise ValueError("final library mismatch")
        equal(stored["entries"], [{"entry": entry, "task": task.name}
                                  for entry, task in entries[method]])
        final_retention[method] = {task.name: evaluate(library, entry, task.test_pairs)["exact"]
                                   for entry, task in entries[method]}
    return {"verified_pushouts": verified, "verified_rows": len(report["rows"]),
            "final_library_hidden_accuracy": final_retention}


if __name__ == "__main__":
    main()
