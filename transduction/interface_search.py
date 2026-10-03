"""Corpus-guided abstraction extraction and typed dataflow lifting.

The search never receives a family name or reference evaluator. It uses source
programs, types and public training pairs. Its skeleton bias is explicit: this
is not an exhaustive synthesizer for the accepted language.
"""
from __future__ import annotations

import copy
import itertools
import json
import time
from collections import Counter, deque

from interface_machine import (Machine, attach_proposal, execute, parse, render,
                               signature, size, typecheck)


GROWTH_CONTRACT = {
    "objective": "Grow a retained library across successive related tasks, not a collection of independent solvers.",
    "layout": "The final task root only composes and binds procedures. Put input inspection and recursion in named helpers. "
              "The root may use calls, references, let, sequence, conditionals and Boolean operations, but not eq or self-reference.",
    "preference": "First reuse retained helpers. When they are insufficient, factor common structure in the visible programs "
                  "into a helper and expose observed differences as arguments rather than copying a specialized solver. "
                  "Choose useful helper boundaries and argument/result interfaces yourself. Do not branch on task identity. "
                  "Prefer minimal new glue, but do not sacrifice a useful interface merely to shorten the first solution.",
    "retention": "The entire first admissible proposal is retained for subsequent tasks, including its helpers. "
                 "No scalar complexity score selects or deletes helpers. No future examples or desired signatures are supplied.",
}


def definition(name, params, returns, body):
    return dict(name=name, params=[dict(name=n, type=t) for n, t in params],
                returns=returns, body=render(body))


def runtime(library):
    return Machine(library)


def test_ast(machine, definitions, root, pairs, *, stop=True):
    defs = {**machine.definitions, **{d["name"]: {**d, "ast": parse(d["body"])} for d in definitions}}
    correct, first = 0, None
    for source, target in pairs:
        run = execute(defs, machine.words, root, source, expected=target, step_limit=12000)
        good = run["ok"] and run["output"] == list(target)
        correct += int(good)
        if not good and first is None:
            first = dict(source=list(source), expected=list(target), actual=run)
            if stop:
                break
    return dict(exact=correct == len(pairs), correct=correct, total=len(pairs), first_failure=first)


def score(library, root, pairs, *, disabled=(), trace_limit=0):
    machine, records = Machine(library), []
    for source, expected in pairs:
        run = machine.run(root, source, disabled=disabled, trace_limit=trace_limit)
        records.append(dict(source=list(source), expected=list(expected),
                            exact=run["ok"] and run["output"] == list(expected), **run))
    count = sum(r["exact"] for r in records)
    return dict(exact=count == len(pairs), correct=count, total=len(pairs), records=records)


def brief(result):
    return {k: result[k] for k in ("exact", "correct", "total")}


def abstract_pair(left, right, signatures):
    """Typed anti-unification of differing nullary calls; repeated holes share.

    Recursive calls propagate the newly inferred parameters. The specialization
    check below reconstructs both concrete source bodies exactly, not by tests.
    """
    holes = {}
    def go(a, b):
        if a == b:
            return a
        if (isinstance(a, tuple) and isinstance(b, tuple) and len(a) == len(b) == 2
                and a[0] == b[0] == "call" and a[1] != "self" and b[1] != "self"
                and signatures.get(a[1]) == signatures.get(b[1]) == ([], "U")):
            slot = holes.setdefault((a[1], b[1]), "f" + str(len(holes)))
            return ("call", slot)
        if (not isinstance(a, tuple) or not isinstance(b, tuple)
                or len(a) != len(b) or a[0] != b[0]):
            raise ValueError("not an applicable source generalization")
        return (a[0], *(go(x, y) for x, y in zip(a[1:], b[1:])))
    term = go(left, right)
    if not holes or len(holes) > 3:
        raise ValueError("no bounded procedure-parameter abstraction")
    names = list(holes.values())
    def lift(t):
        if isinstance(t, str):
            return t
        if t == ("call", "self"):
            return ("call", "self", *names)
        return tuple(lift(x) for x in t)
    term = lift(term)
    specializations = []
    for side, original in enumerate((left, right)):
        mapping = {v: pair[side] for pair, v in holes.items()}
        def specialize(t):
            if isinstance(t, str):
                return t
            if t[:2] == ("call", "self"):
                return ("call", "self")
            if t[0] == "call" and t[1] in mapping:
                return ("call", mapping[t[1]])
            return tuple(specialize(x) for x in t)
        if specialize(term) != original:
            raise ValueError("abstraction does not specialize exactly")
        specializations.append(dict(bindings=mapping, recovered_body=render(original), exact_syntax=True))
    return term, names, specializations


def discover(library):
    sources = [c for c in library.cells.values() if signature(c) == ([], "U")]
    results, seen = [], set()
    for a, b in itertools.combinations(sources, 2):
        try:
            term, names, evidence = abstract_pair(parse(a["body"]), parse(b["body"]), library.signatures())
        except ValueError:
            continue
        if render(term) in seen:
            continue
        seen.add(render(term))
        results.append(dict(body=term, params=names, sources=[a["name"], b["name"]],
                            source_hashes=[a["hash"], b["hash"]], specializations=evidence))
    return results


def bool_terms(variables):
    # Pure expressions; truth-table dedup is exact on all Boolean assignments,
    # unlike equating whole programs only on a small training set.
    variables = list(variables)
    candidates = [*variables, "false", "true"]
    candidates += [("not", v) for v in variables]
    for op in ("xor", "and", "or"):
        for a, b in reversed(list(itertools.combinations(variables, 2))):
            candidates.append((op, a, b))
    def value(t, env):
        if isinstance(t, str):
            return t == "true" if t in ("true", "false") else env[t]
        vals = [value(x, env) for x in t[1:]]
        if t[0] == "not":
            return not vals[0]
        return vals[0] != vals[1] if t[0] == "xor" else (
            vals[0] and vals[1] if t[0] == "and" else vals[0] or vals[1])
    seen, answer = set(), []
    assignments = list(itertools.product((False, True), repeat=len(variables)))
    for term in candidates:
        table = tuple(value(term, dict(zip(variables, row))) for row in assignments)
        if table not in seen:
            seen.add(table)
            answer.append(term)
    return answer


def lifted_skeleton(abstract, boolean_count, function_count, returns):
    """Uniform grammar transformation, not a tree/task-specific constructor.

    Recursive calls acquire arguments. If their new type returns a Boolean,
    sequencing binds it, making it available to subsequent calls and the return.
    Parameter calls can be selected by a Boolean expression. Holes are typed.
    """
    bvars = ["b" + str(i) for i in range(boolean_count)]
    fvars = ["f" + str(i) for i in range(function_count)]
    domains, hole_info = [], []
    counter = itertools.count()
    def hole(env, role):
        index = len(domains)
        domains.append(bool_terms(env))
        hole_info.append(dict(index=index, role=role, available=list(env)))
        return "HOLE" + str(index)
    def leaf_call(env):
        if not fvars:
            raise ValueError("at least one abstracted callee required")
        def choose(i):
            return (("call", fvars[i]) if i == len(fvars) - 1 else
                    ("if", hole(env, "procedure_selection"), choose(i + 1), ("call", fvars[i])))
        return choose(0)
    def call(t, env):
        if t[1] == "self":
            return ("call", "self", *(hole(env, "recursive_argument") for _ in bvars), *fvars)
        if t[1] in abstract["params"]:
            return leaf_call(env)
        return t
    def terminal(effect, env):
        return ("seq", effect, hole(env, "result")) if returns == "B" else effect
    def sequence(items, env):
        if not items:
            return hole(env, "result") if returns == "B" else "unit"
        first, *tail = items
        if isinstance(first, tuple) and first[:2] == ("call", "self"):
            term = call(first, env)
            if returns == "B":
                var = "r" + str(next(counter))
                return ("let", var, term, sequence(tail, [*env, var]))
            return ("seq", term, sequence(tail, env)) if tail else term
        if not tail:
            return walk(first, env)
        if isinstance(first, tuple) and first[0] == "call":
            return ("seq", call(first, env), sequence(tail, env))
        # General nested expressions are lifted and sequenced as well.
        return ("seq", walk(first, env), sequence(tail, env))
    def walk(t, env):
        if isinstance(t, str):
            return hole(env, "result") if returns == "B" and t == "unit" else t
        if t[0] == "if":
            return ("if", t[1], walk(t[2], env), walk(t[3], env))
        if t[0] == "seq":
            return sequence(list(t[1:]), env)
        if t[0] == "call":
            if t[1] == "self" and returns == "B":
                return sequence([t], env)
            return terminal(call(t, env), env)
        raise ValueError("source skeleton outside dataflow-lifting subset")
    body = walk(abstract["body"], bvars)
    return dict(body=body, params=[*( (b, "B") for b in bvars), *((f, "F") for f in fvars)],
                returns=returns, domains=domains, holes=hole_info)


def substitute(term, choices):
    if isinstance(term, str):
        return choices[int(term[4:])] if term.startswith("HOLE") else term
    return tuple(substitute(x, choices) for x in term)


def diagonal_product(domains):
    """Bounded-memory short-index-first enumeration, not a giant BFS queue."""
    if not domains:
        yield ()
        return
    maximum = sum(len(d) - 1 for d in domains)
    suffix = [0] * (len(domains) + 1)
    for i in reversed(range(len(domains))):
        suffix[i] = suffix[i + 1] + len(domains[i]) - 1
    def at(i, remaining, prefix):
        if i == len(domains):
            if remaining == 0:
                yield tuple(prefix)
            return
        low = max(0, remaining - suffix[i + 1])
        for j in range(low, min(remaining, len(domains[i]) - 1) + 1):
            prefix.append(domains[i][j])
            yield from at(i + 1, remaining - j, prefix)
            prefix.pop()
    for weight in range(maximum + 1):
        yield from at(0, weight, [])


def ordered_refs(library, machine, pairs):
    refs = [n for n, sig in library.signatures().items() if sig == ([], "U")]
    tiny = sorted(pairs, key=lambda p: len(p[0]))[:2]
    def preference(n):
        good = sum(machine.run(n, s)["output"] == list(t) for s, t in tiny)
        c = library.base.cells.get(n)
        return (-good, 0 if c and c["width"] == 2 else 1,
                len(c["program"]["actions"]) if c else 99, n)
    return sorted(refs, key=preference)


def adapters(library, refs, root):
    cells = {**library.base.cells, **library.cells}
    for name, sig in library.signatures().items():
        domains = [["false", "true"] if typ == "B" else [("ref", n) for n in refs] for typ in sig[0]]
        for args in diagonal_product(domains):
            body = ("call", name, *args)
            if sig[1] == "B":
                body = ("seq", body, "unit")
            yield [definition(root, [], "U", body)], dict(kind="retained_call", callee=name,
                 callee_hash=cells[name]["hash"], arguments=[render(a) for a in args])


def reuse_search(library, pairs, root, seconds=10, candidate_limit=10000):
    """Identical host-side retained-call probe before either proposer."""
    started = time.monotonic()
    machine = runtime(library)
    pairs = sorted(pairs, key=lambda p: len(p[0]))
    refs = ordered_refs(library, machine, pairs)
    attempts, best, winner, origin = 0, None, None, None
    stopped = "enumerated_subset_exhausted"
    for definitions, derivation in adapters(library, refs, root):
        if attempts >= candidate_limit or time.monotonic() - started >= seconds:
            stopped = "candidate_limit" if attempts >= candidate_limit else "time_limit"
            break
        attempts += 1
        result = test_ast(machine, definitions, root, pairs)
        if best is None or result["correct"] > best["correct"]:
            best = brief(result)
        if result["exact"]:
            winner = dict(library_hash=library.digest, definitions=definitions)
            built, _ = attach_proposal(library, winner, root)
            if not score(built, root, pairs)["exact"]:
                raise ValueError("retained-call AST and graph execution disagree")
            origin, stopped = derivation, "solved"
            break
    return winner, dict(seconds=time.monotonic() - started, candidates=attempts,
                        best=best, derivation=origin, stopped=stopped)


def mechanical(library, pairs, root, seconds=90, candidate_limit=200000):
    started, deadline = time.monotonic(), time.monotonic() + seconds
    machine = runtime(library)
    pairs = sorted(pairs, key=lambda p: len(p[0]))
    refs = ordered_refs(library, machine, pairs)
    abstracts = discover(library)
    attempts, best, provenance = 0, None, None
    per_stream = Counter()
    streams = deque()
    signatures = [(nb, nf, ret) for nb in range(3) for nf in range(1, 4)
                  if nb + nf <= 4 for ret in ("U", "B")]
    signatures.sort(key=lambda s: (s[0] + s[1] + (s[2] == "B"), s))

    def variants(abstract, spec):
        proto = lifted_skeleton(abstract, *spec)
        helper = root + "H"
        root_domains = [["false", "true"] if typ == "B" else [("ref", n) for n in refs]
                        for _, typ in proto["params"]]
        for choice in diagonal_product(proto["domains"] + root_domains):
            body = substitute(proto["body"], choice)
            definition_h = definition(helper, proto["params"], proto["returns"], body)
            args = choice[len(proto["domains"]):]
            call = ("call", helper, *args)
            main = ("seq", call, "unit") if proto["returns"] == "B" else call
            yield [definition_h, definition(root, [], "U", main)], dict(
                kind="corpus_abstraction_and_dataflow_lifting", sources=abstract["sources"],
                exact_base_specializations=abstract["specializations"],
                signature=spec, holes=proto["holes"], filling=[render(c) for c in choice],
                prototype=render(proto["body"]))

    for index, abstract in enumerate(abstracts):
        for spec in signatures:
            streams.append((str((index, *spec)), variants(abstract, spec)))
    winner = None
    while streams and attempts < candidate_limit and time.monotonic() < deadline:
        label, stream = streams.popleft()
        exhausted = False
        for _ in range(16):
            if attempts >= candidate_limit or time.monotonic() >= deadline:
                break
            try:
                definitions, origin = next(stream)
            except StopIteration:
                exhausted = True
                break
            attempts += 1
            per_stream[label] += 1
            result = test_ast(machine, definitions, root, pairs)
            if best is None or result["correct"] > best["correct"]:
                best = {**brief(result), "definitions": definitions}
            if result["exact"]:
                payload = dict(library_hash=library.digest, definitions=definitions)
                built, _ = attach_proposal(library, payload, root)
                if not score(built, root, pairs)["exact"]:
                    raise ValueError("AST search and graph execution disagree")
                winner, provenance = payload, origin
                break
        if winner:
            break
        if not exhausted:
            streams.append((label, stream))
    return winner, dict(seconds=time.monotonic() - started, candidates=attempts,
                        candidates_by_stream=dict(per_stream), extracted_abstractions=len(abstracts),
                        derivation=provenance, best=best,
                        stopped="solved" if winner else "candidate_limit" if attempts >= candidate_limit else
                        "time_limit" if time.monotonic() >= deadline else "enumerated_subset_exhausted")


def prompt(library, pairs, root, feedback=None, reuse=None, history=None):
    return dict(
        instruction="Infer the transformation only from these training examples and existing programs. "
        "Propose up to four typed definitions, ending in the specified nullary U root. "
        "You are growing a LEG LIBRARY across tasks. Follow the shared growth contract below. "
        "You choose helper boundaries, argument counts/types and return types; correctness is a hard requirement. "
        "Definitions may call themselves or earlier definitions; retained cells cannot be changed. "
        "Do not use tools. Return only the JSON object matching the schema.",
        library_hash=library.digest, root=root, growth_contract=GROWTH_CONTRACT,
        language={
            "types": "B = Boolean; U = unit; F = reference to a nullary U procedure. 0..4 B/F arguments; result U/B.",
            "atoms": "unit, true, false, eq, or a lexical variable. eq tests whether the next two input tokens exist and are equal.",
            "expressions": "(ref NAME); (call NAME args...); (call self args...); (call FUNCTION_PARAMETER); "
            "(if BOOL THEN ELSE); (seq EXPR...); (let VARIABLE VALUE BODY); (not B); (xor B B); (and B B); (or B B).",
            "semantics": "seq evaluates left to right and returns its last value. if evaluates only the chosen branch; "
            "branches have identical types. let is lexical; shadowing is forbidden. Boolean operands evaluate left to right. "
            "Calls retain advanced input and appended output, but not the callee's lexical locals. Recursive reentry to "
            "any active procedure requires strictly greater input position. The root must consume all input. "
            "There is no subtree operation or rewind. Opaque token identities cannot appear in programs.",
            "fragments": "Each fragment consumes its fixed width with two fresh local token registers. "
            "0=advance, 1=emit current, 10/11=store R0/R1, 30/31=emit R0/R1; emits append. "
            "Fragment windows must fit. Fragment calls have no arguments and return unit.",
            "bounds": dict(definitions=4, arguments=4, expression_nodes_per_body=180,
                           runtime_steps=60000, runtime_depth=150)},
        fragments={n: dict(width=c["width"], actions=c["program"]["actions"], hash=c["hash"])
                   for n, c in library.base.cells.items()},
        retained={n: {k: c[k] for k in ("params", "returns", "body", "hash")} for n, c in library.cells.items()},
        reuse_probe=reuse, prior_tasks=history or [], training_examples=pairs, feedback=feedback)
