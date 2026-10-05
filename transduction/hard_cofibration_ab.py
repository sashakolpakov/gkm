"""Harder paired discovery, reusing the frozen A/B engine without editing it.

Only the private task adapter and predeclared budgets change. Neither proposer
receives the adapter, reference program, family label or coverage witness.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import cofibration_ab as engine
import frontier_benchmark as previous
from frontier_search import insert_effects, insertion_sites, verify_derivation
from interface_benchmark import local_effect, save
from interface_machine import render
from interface_search import definition, discover, lifted_skeleton, substitute
from modular_machine import digest


FAMILIES = ("framed_return",)
TASKS = [(FAMILIES[0], 1), (FAMILIES[0], 2)]
SEEDS = (41, 42)
SECONDS = 1200
REPLIES = 10
CANDIDATES = 32000000
SOURCES = (*engine.SOURCES, "hard_cofibration_ab.py")


def encode(tree, family):
    if family not in FAMILIES:
        raise ValueError("unknown hard task")
    return previous.encode(tree, "framed")


def reference(tree, family, phase):
    """Private tree oracle: pre, left-dependent middle, total-dependent trailer."""
    if family not in FAMILIES or phase not in (1, 2):
        raise ValueError("unknown hard task")
    if "pair" in tree:
        return local_effect(tree["pair"], 0 if phase == 1 else 2), 1
    left, nl = reference(tree["left"], family, phase)
    right, nr = reference(tree["right"], family, phase)
    before = local_effect(tree["pre"], 1)
    middle = local_effect(tree["mid"], 1 if nl % 2 else 3)
    after = local_effect(tree["post"], 3 if (nl + nr) % 2 else 1)
    return before + left + middle + right + after, nl + nr


def pairs_for(family, phase, split, seed=1):
    offset, pool = {"train": (7, 0), "validation": (101, 100),
                    "hidden": (503, 1000), "stress": (997, 2000)}[split]
    rng = random.Random(seed * 1009 + offset)
    if split in ("train", "validation"):
        forms = [s for n in range(1, 6) for s in previous.shapes(n)]
        forms += [previous.comb(n, side) for n in (6, 7, 8)
                  for side in ("left", "right", "alternating")]
        forms += [previous.random_shape(n, rng) for n in (6, 8, 10) for _ in range(4)]
    elif split == "hidden":
        forms = [previous.random_shape(n, rng) for n in (12, 20, 32, 64, 96) for _ in range(8)]
    else:
        forms = [previous.comb(n, side) for n in (16, 32, 64)
                  for side in ("left", "right", "alternating")]
        forms += [previous.random_shape(128, rng) for _ in range(4)]
    trees = [previous.materialize(form, rng, pool) for form in forms]
    return [(encode(t, family), reference(t, family, phase)[0]) for t in trees]


def witness(library, family, phase, root="W"):
    """Private grammar-membership control, never a discovery or proposer input."""
    if family not in FAMILIES or phase not in (1, 2):
        raise ValueError("unknown hard task")
    abstract = discover(library)[0]
    spec = (0, 3, "B")
    original = lifted_skeleton(abstract, *spec)
    sites = insertion_sites(original, library.signatures())
    def at(path):
        term = original["body"]
        for index in path:
            term = term[index]
        return term
    recursive = [s for s in sites if s["side"] == "before" and isinstance(at(s["path"]), tuple)
                 and at(s["path"])[:2] == ("call", "self")]
    trailer = next(s for s in sites if s["path"] == [2] and s["side"] == "after")
    selected = [recursive[0], recursive[1], trailer]
    proto = insert_effects(original, selected)
    filling, seen = [], {}
    for hole, domain in zip(proto["holes"], proto["domains"]):
        available = hole["available"]
        if hole["role"] == "result":
            value = ("xor", available[-2], available[-1]) if len(available) >= 2 else "true"
        elif hole["role"] == "inserted_procedure_selection":
            index = hole["insertion"]
            number = seen.get(index, 0)
            seen[index] = number + 1
            value = ("true" if number == 0 else "false" if index == 0 else
                     ("not", available[-1]) if index == 1 else available[-1])
        else:
            value = "false"
        if value not in domain:
            raise ValueError("coverage witness outside the unchanged mechanical grammar")
        filling.append(value)
    refs = ["C01" if phase == 1 else "C02", "F004", "C03"]
    filling += [("ref", name) for name in refs]
    body = substitute(proto["body"], filling)
    call = ("call", root + "H", *(("ref", name) for name in refs))
    raw = dict(library_hash=library.digest, definitions=[
        definition(root + "H", proto["params"], "B", body),
        definition(root, [], "U", ("seq", call, "unit"))])
    origin = dict(kind="typed_control_edits", sources=abstract["sources"], signature=list(spec),
        edits=selected, exact_base_specializations=abstract["specializations"], filling=[render(f) for f in filling])
    verify_derivation(library, raw, origin, root)
    return raw, origin


@contextmanager
def configured_engine():
    """Scoped task dependency injection; the original experiment stays replayable."""
    old_frontier, old_sources = engine.frontier, engine.SOURCES
    engine.frontier = SimpleNamespace(FAMILIES=FAMILIES, pairs_for=pairs_for, encode=encode,
        reference=reference, witness=witness, materialize=previous.materialize, shapes=previous.shapes,
        search=previous.search, reuse_context=previous.reuse_context)
    engine.SOURCES = SOURCES
    try:
        yield engine
    finally:
        engine.frontier, engine.SOURCES = old_frontier, old_sources


def prepare(output, seeds=SEEDS):
    with configured_engine() as experiment:
        plan = experiment.prepare(output, seeds=seeds, seconds=SECONDS, replies=REPLIES, tasks=TASKS)
        plan.update(candidate_limit=CANDIDATES, task_specification="framed-return-v1",
                    example_counts=dict(train=44, validation=44, hidden=40, stress=13))
        save(output / "plan.json", dict(plan=plan, hash=digest(plan)))
        return plan


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--arm", choices=engine.ARMS)
    mode.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        print(json.dumps(prepare(args.output), indent=2))
    else:
        with configured_engine() as experiment:
            if args.arm:
                experiment.run_arm(args.output, args.arm)
            else:
                experiment.replay(args.output)
