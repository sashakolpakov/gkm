"""Generic typed effect insertion beyond retained control skeletons.

No task names, encodings, reference functions or solution bodies enter this
module. The bounded search is deliberately documented, not claimed complete.
"""
from __future__ import annotations

import copy
import itertools
import time
from collections import Counter, deque

from interface_machine import MAX_NODES, attach_proposal, parse, render, size, typecheck
from interface_search import (bool_terms, brief, definition, diagonal_product,
                              discover, lifted_skeleton, ordered_refs, runtime,
                              score, substitute, test_ast)


def insertion_sites(proto, signatures):
    """Before/after any typed expression, including lexical continuations."""
    env = dict(proto["params"])
    env.update({"HOLE" + str(i): "B" for i in range(len(proto["domains"]))})
    sigs = {**signatures, "H": ([t for _, t in proto["params"]], proto["returns"])}
    result = []
    def visit(term, path, scope):
        typ = typecheck(term, scope, sigs, "H")
        if typ in ("U", "B"):
            available = [n for n, t in scope.items() if t == "B" and not n.startswith("HOLE")]
            for side in ("before", "after"):
                result.append(dict(path=list(path), side=side, result=typ, available=available))
        if isinstance(term, str) or term[0] == "ref":
            return
        if term[0] == "let":
            visit(term[2], (*path, 2), scope)
            visit(term[3], (*path, 3), {**scope, term[1]: typecheck(term[2], scope, sigs, "H")})
        else:
            for i in range(2 if term[0] == "call" else 1, len(term)):
                visit(term[i], (*path, i), scope)
    visit(proto["body"], (), env)
    return result


def insert_effects(proto, selected):
    """Uniform, type-preserving AST transformation, independent of the task."""
    updated = copy.deepcopy(proto)
    functions = [n for n, t in proto["params"] if t == "F"]
    by_path = {}
    for index, site in enumerate(selected):
        path = tuple(site["path"])
        if path in by_path:
            raise ValueError("at most one insertion at each original expression")
        by_path[path] = index, site
    def effect(available, site_index):
        def choose(i):
            if i == len(functions) - 1:
                return ("call", functions[i])
            h = len(updated["domains"])
            updated["domains"].append(bool_terms(available))
            updated["holes"].append(dict(index=h, role="inserted_procedure_selection",
                                         available=available, insertion=site_index))
            return ("if", "HOLE" + str(h), choose(i + 1), ("call", functions[i]))
        return choose(0)
    def walk(term, path=()):
        if isinstance(term, str):
            changed = term
        else:
            begin = 2 if term[0] in ("call", "ref", "let") else 1
            changed = (*term[:begin], *(walk(term[i], (*path, i)) for i in range(begin, len(term))))
        if path not in by_path:
            return changed
        index, site = by_path[path]
        available = site["available"]
        if site["side"] == "before":
            return ("seq", effect(available, index), changed)
        if site["result"] == "U":
            return ("seq", changed, effect(available, index))
        # Preserve the original result while allowing the inserted action to
        # depend on it. This is generic let insertion, not a parity rule.
        local = "inserted_result_" + str(index)
        return ("let", local, changed, ("seq", effect([*available, local], index), local))
    updated["body"] = walk(proto["body"])
    return updated


def variants(proto, refs, root):
    root_domains = [["false", "true"] if t == "B" else [("ref", n) for n in refs]
                    for _, t in proto["params"]]
    helper = root + "H"
    for choices in diagonal_product(proto["domains"] + root_domains):
        body = substitute(proto["body"], choices)
        if size(body) > MAX_NODES:
            continue
        call = ("call", helper, *choices[len(proto["domains"]):])
        root_body = ("seq", call, "unit") if proto["returns"] == "B" else call
        yield [definition(helper, proto["params"], proto["returns"], body),
               definition(root, [], "U", root_body)], [render(c) for c in choices]


def sketch_streams(library, max_edits=3, min_edits=0):
    """Round robin across signatures and edit counts; no task-specific order."""
    specs = [(nb, nf, ret) for nb in range(3) for nf in range(1, 5)
             if nb + nf <= 4 for ret in ("U", "B")]
    specs.sort(key=lambda s: (s[0] + s[1] + (s[2] == "B"), s))
    streams = deque()
    def edited(abstract, spec, count):
        original = lifted_skeleton(abstract, *spec)
        sites = insertion_sites(original, library.signatures())
        for selected in itertools.combinations(sites, count):
            if len({tuple(s["path"]) for s in selected}) != count:
                continue
            proto = insert_effects(original, selected)
            yield proto, dict(kind="typed_control_edits", sources=abstract["sources"],
                signature=list(spec), edits=list(selected),
                exact_base_specializations=abstract["specializations"])
    for abstract in discover(library):
        for spec in specs:
            for count in range(min_edits, max_edits + 1):
                streams.append(edited(abstract, spec, count))
    while streams:
        stream = streams.popleft()
        try:
            yield next(stream)
            streams.append(stream)
        except StopIteration:
            pass


def mechanical(library, pairs, root, seconds=180, candidate_limit=2000000,
               max_edits=3, active_limit=512):
    started = time.monotonic()
    machine = runtime(library)
    pairs = sorted(pairs, key=lambda p: len(p[0]))
    refs = ordered_refs(library, machine, pairs)
    pending = sketch_streams(library, max_edits, min_edits=1)
    plain = deque((variants(proto, refs, root), origin, 0)
                  for proto, origin in sketch_streams(library, max_edits=0))
    active = deque()
    attempts, opened, rounds = 0, 0, 0
    best, winner, derivation = None, None, None
    exhausted, counts = False, Counter()
    while attempts < candidate_limit and time.monotonic() - started < seconds:
        # Preserve the old zero-edit search: it is never evicted and receives
        # at least one third of batches. Widen the edit pool with bounded memory;
        # no sketch is evicted before testing 256 of its bindings.
        can_open = len(active) < active_limit or any(v >= 256 for _, _, v in active)
        if not exhausted and can_open and (not active or rounds % 4 == 0):
            try:
                proto, origin = next(pending)
                if len(active) >= active_limit:
                    victim = next(i for i, (_, _, v) in enumerate(active) if v >= 256)
                    del active[victim]
                active.append((variants(proto, refs, root), origin, 0))
                opened += 1
            except StopIteration:
                exhausted = True
        pool = plain if plain and (rounds % 3 == 0 or not active) else active
        if not pool:
            break
        stream, origin, visits = pool.popleft()
        done = False
        for _ in range(16):
            if attempts >= candidate_limit or time.monotonic() - started >= seconds:
                break
            try:
                definitions, filling = next(stream)
            except StopIteration:
                done = True
                break
            attempts += 1
            visits += 1
            counts[str(len(origin["edits"]))] += 1
            result = test_ast(machine, definitions, root, pairs)
            if best is None or result["correct"] > best["correct"]:
                best = {**brief(result), "definitions": definitions}
            if result["exact"]:
                raw = dict(library_hash=library.digest, definitions=definitions)
                built, _ = attach_proposal(library, raw, root)
                if not score(built, root, pairs)["exact"]:
                    raise ValueError("AST and compiled graph disagree")
                winner, derivation = raw, {**origin, "filling": filling}
                break
        if winner:
            break
        if not done:
            pool.append((stream, origin, visits))
        rounds += 1
    return winner, dict(seconds=time.monotonic() - started, candidates=attempts,
        sketches_opened=opened, active_sketch_limit=active_limit, candidates_by_edit_count=dict(counts),
        best=best, derivation=derivation,
        stopped="solved" if winner else "candidate_limit" if attempts >= candidate_limit else
        "time_limit" if time.monotonic() - started >= seconds else "enumerated_subset_exhausted")


def verify_derivation(library, payload, origin, root):
    candidates = [a for a in discover(library) if a["sources"] == origin["sources"]]
    if len(candidates) != 1 or candidates[0]["specializations"] != origin["exact_base_specializations"]:
        raise ValueError("source abstraction changed")
    original = lifted_skeleton(candidates[0], *origin["signature"])
    legal = insertion_sites(original, library.signatures())
    if any(site not in legal for site in origin["edits"]):
        raise ValueError("edit outside generic grammar")
    proto = insert_effects(original, origin["edits"])
    filling = [parse(s) for s in origin["filling"]]
    if len(filling) != len(proto["domains"]) + len(proto["params"]):
        raise ValueError("wrong filling length")
    for domain, value in zip(proto["domains"], filling):
        if value not in domain:
            raise ValueError("choice outside typed grammar")
    body = substitute(proto["body"], filling)
    call = ("call", root + "H", *filling[len(proto["domains"]):])
    root_body = ("seq", call, "unit") if proto["returns"] == "B" else call
    expected = [definition(root + "H", proto["params"], proto["returns"], body),
                definition(root, [], "U", root_body)]
    if payload["definitions"] != expected:
        raise ValueError("candidate differs from mechanical derivation")
    attach_proposal(library, payload, root)
    return True
