"""Paired mechanical/Codex A/B test with an identical attachment grammar."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import deque
from pathlib import Path

from codex_glue import propose, schema
from cofibration import parse_attachment, verify_certificate
from pattern_fsa import PatternTask, register_primitives
from run_cofibration_experiment import evaluate, examples, necessary_reuse, seed_machine, summary


CASES = ("duplicate_first", "swap_first_pair", "rotate_first_three", "reverse_first_three")
SEEDS = (1, 11, 23)
ACTIONS = (0, 1, 10, 30, 11, 31)
MAX_WORD = 8


def task_for(name, seed):
    def split(start, offset, count, lengths):
        inputs = examples("copy", seed + offset, count, start, lengths)
        result = []
        for source, _ in inputs:
            if name == "duplicate_first":
                target = source[:1] + source
            elif name == "swap_first_pair":
                target = (source[1], source[0]) + source[2:]
            elif name == "rotate_first_three":
                target = source[1:3] + source[:1] + source[3:]
            elif name == "reverse_first_three":
                target = tuple(reversed(source[:3])) + source[3:]
            else:
                raise ValueError("unknown A/B task")
            result.append((source, target))
        return tuple(result)
    return PatternTask(name, 48, split(0, 0, 12, (3, 4, 5, 6)),
                       split(16, 101, 12, (3, 4, 5, 6)),
                       split(32, 202, 48, tuple(range(3, 15))))


def payload_for(word, library):
    return {"library_hash": library.digest, "fresh_states": 1, "ports": [0],
            "rules": [{"state": 0, "observation": 1, "actions": list(word), "next_state": 1}]}


def check_common_grammar(payload, library):
    square, entry = parse_attachment(payload, library)
    word = payload["rules"][0]["actions"] if len(payload["rules"]) == 1 else []
    if not 1 <= len(word) <= MAX_WORD or any(action not in ACTIONS for action in word):
        raise ValueError("outside shared instruction grammar")
    if payload != payload_for(word, library):
        raise ValueError("outside shared attachment grammar")
    return square, entry


def response_schema(library):
    result = schema()
    properties = result["properties"]
    properties["library_hash"]["enum"] = [library.digest]
    properties["fresh_states"]["enum"] = [1]
    properties["ports"].update(minItems=1, maxItems=1, items={"type": "integer", "enum": [0]})
    properties["rules"].update(minItems=1, maxItems=1)
    rule = properties["rules"]["items"]["properties"]
    # Assign fresh dicts: the base schema intentionally shares its integer schema.
    for name, value in (("state", 0), ("observation", 1), ("next_state", 1)):
        rule[name] = {"type": "integer", "enum": [value]}
    rule["actions"] = {"type": "array", "minItems": 1, "maxItems": MAX_WORD,
                       "items": {"type": "integer", "enum": list(ACTIONS)}}
    return result


def prompt_for(library, pairs):
    return {
        "instructions": (
            "Infer the token-sequence transformation from the training pairs. Return "
            "only the structured attachment. The permitted glue is IDENTICAL to the "
            "mechanical comparison: one fresh state, one TOKEN transition, 1..8 "
            "primitive instructions, then enter retained state 0. Use fresh_states=1, "
            "ports=[0], one rule with state=0, observation=1, next_state=1. The port's "
            "local index 1 is identified with retained state 0. Start at input index "
            "0, empty output and two empty registers. Actions: 0=MOVE_RIGHT (capped "
            "at EOS); 1=WRITE_CURRENT (append current token if present); 10=STORE_R0; "
            "11=STORE_R1; 30=WRITE_R0; 31=WRITE_R1. Stores at EOS and writes of empty "
            "registers do nothing. Instructions execute in order before entering "
            "the retained state. Observations: 1=TOKEN, 0=EOS; absent rules halt. "
            "Token identities are opaque, not constants you may embed. Infer only "
            "from these pairs and the retained graph. Prefer fewer instructions. "
            "You get one proposal, no tools and no evaluation feedback."
        ),
        "library_hash": library.digest, "library": library.payload(),
        "training_examples": pairs,
    }


def advance(configurations, action, pairs):
    """Apply one instruction to training configurations; reject irreparable output.

    Output is append-only. Once a written token disagrees with its target prefix,
    no continuation can make this instruction word exact on that example.
    """
    updated = []
    for (cursor, r0, r1, written), (source, target) in zip(configurations, pairs):
        registers = [r0, r1]
        token = source[cursor] if cursor < len(source) else None
        emit = None
        if action == 0:
            cursor = min(cursor + 1, len(source))
        elif action == 1:
            emit = token
        elif action in (10, 11):
            if token is not None:
                registers[action - 10] = token
        elif action in (30, 31):
            emit = registers[action - 30]
        else:
            raise ValueError("unknown shared instruction")
        if emit is not None:
            if written == len(target) or target[written] != emit:
                return None
            written += 1
        updated.append((cursor, *registers, written))
    return tuple(updated)


def mechanical(library, pairs, budget=8192, seconds=120):
    initial = tuple((0, None, None, 0) for _ in pairs)
    queue = deque([((), initial)])
    visited = {initial}
    best, best_key = None, None
    evaluated = expanded = rejected = 0
    started = time.monotonic()
    while queue and evaluated < budget and time.monotonic() - started < seconds:
        word, configurations = queue.popleft()
        expanded += 1
        if len(word) == MAX_WORD:
            continue
        for action in ACTIONS:
            if evaluated >= budget or time.monotonic() - started >= seconds:
                break
            successor = advance(configurations, action, pairs)
            if successor is None or successor in visited:
                rejected += 1
                continue
            visited.add(successor)
            child = word + (action,)
            queue.append((child, successor))
            candidate = payload_for(child, library)
            square, entry = check_common_grammar(candidate, library)
            score = evaluate(square.graph, entry, pairs, library.states)
            evaluated += 1
            key = (score["loss"], len(child))
            if best_key is None or key < best_key:
                best, best_key = candidate, key
            if score["exact"] == 1 and necessary_reuse(square.graph, entry, pairs, library.states):
                return candidate, {"evaluated": evaluated, "expanded": expanded,
                                   "pruned": rejected, "visited": len(visited),
                                   "seconds": time.monotonic() - started, "exact_found": True}
    return best, {"evaluated": evaluated, "expanded": expanded, "pruned": rejected,
                  "visited": len(visited), "seconds": time.monotonic() - started,
                  "exact_found": False}


def run(output, budget=8192):
    output.mkdir(parents=True, exist_ok=False)
    source_hashes = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                     for name in ("glue_ab.py", "cofibration.py", "codex_glue.py", "pattern_fsa.py",
                                  "run_cofibration_experiment.py")}
    (output / "protocol.json").write_text(json.dumps({
        "seeds": SEEDS, "tasks": CASES, "shared_grammar": "one new TOKEN rule, 1..8 instructions, then retained copy",
        "mechanical_max_evaluations": budget, "codex_proposals_per_condition": 1,
        "wall_seconds_per_arm_condition": 120, "model": "gpt-5.6-sol",
        "source_hashes": source_hashes, "hidden_evaluation": "after all choices are frozen",
    }, indent=2))
    frozen = []
    for seed in SEEDS:
        anchor_task = PatternTask("anchor", 48,
                                 examples("copy", seed, 12, 0, (3, 4, 5, 6)),
                                 examples("copy", seed + 101, 12, 16, (3, 4, 5, 6)), ())
        library, seed_cost = seed_machine(anchor_task, register_primitives(48, 2))
        for task_index, name in enumerate(CASES):
            task = task_for(name, seed)
            # Alternate execution order; neither arm receives the other's output.
            arms = ("mechanical", "codex") if (seed + task_index) % 2 else ("codex", "mechanical")
            for arm in arms:
                folder = output / f"s{seed}-{name}-{arm}"
                folder.mkdir()
                started = time.monotonic()
                if arm == "mechanical":
                    candidate, effort = mechanical(library, task.train_pairs, budget)
                else:
                    candidate = propose(prompt_for(library, task.train_pairs), folder / "proposal",
                                        response_schema=response_schema(library))
                    effort = {"proposals": 1, "seconds": time.monotonic() - started}
                square, entry = check_common_grammar(candidate, library)
                training = evaluate(square.graph, entry, task.train_pairs, library.states)
                validation = evaluate(square.graph, entry, task.val_pairs, library.states)
                reuse = necessary_reuse(square.graph, entry, task.train_pairs, library.states)
                row = {"seed": seed, "task": name, "arm": arm, "effort": effort,
                       "common_acquisition_evaluations": seed_cost, "library_hash": library.digest,
                       "training": summary(training), "validation": summary(validation),
                       "necessary_reuse": reuse, "word_length": len(candidate["rules"][0]["actions"]),
                       "admitted": training["exact"] == validation["exact"] == 1 and reuse}
                (folder / "selection.json").write_text(json.dumps({
                    **row, "candidate": candidate, "certificate": square.certificate(),
                    "entry": entry, "training_replay": training, "validation_replay": validation,
                }, indent=2))
                frozen.append((row, task, square, entry, folder))
                print(json.dumps(row), flush=True)
    rows = []
    for row, task, square, entry, folder in frozen:
        # Independent reconstruction from disk before hidden scoring.
        receipt = json.loads((folder / "selection.json").read_text())
        rebuilt = verify_certificate(receipt["certificate"], square.left)
        test = evaluate(rebuilt.graph, entry, task.test_pairs, square.left.states)
        row["test"] = summary(test)
        (folder / "hidden-replay.json").write_text(json.dumps(test, indent=2))
        rows.append(row)
    (output / "summary.json").write_text(json.dumps({"rows": rows}, indent=2))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=8192)
    args = parser.parse_args()
    if not 32 <= args.budget <= 32768:
        parser.error("budget must be in 32..32768")
    run(args.output, args.budget)
