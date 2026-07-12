"""General semantic-term coverage audit.

This module decides whether an executable cone is allowed to claim its own
declared semantic terms.  It contains NO per-concept lexicon: nothing in the
harness knows what a bird, a pinwheel or a triangle is.  Coverage is judged
mechanically against the leg registry.

Each content token of a declared term is classified:

- *covered*   — it stem-matches a witness type or leg inside the score's
  dependency cone, a used leg's contract vocabulary (``proxy_for``), or a
  proposer-declared gluing;
- *violation* — it stem-matches structure the registry CAN express (a
  witness type or leg exists for it) but the score path does not execute
  it.  This is real weakening: "triangle" scored by a bounding box.
- *unknown*   — the registry has no concept of it ("notch", "spine",
  "bird").  Ordinary descriptive words are harmless as long as the term is
  anchored by at least one covered token.

A term FAILS iff it has a violation token, or it has no covered token at
all (a purely-unknown claim is inexpressible and must surface as
MISSING_LEG rather than ride on an unrelated scalar).  Quantity/measure
words are threshold content (handled by the fitted rule and the declared
order) and are skipped.  Suggestions in the failure are derived from the
registry, never from a concept table.

Admissibility stays a hard gate; Kolmogorov/MDL selection happens after.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

_STOPWORDS = frozenset({
    "a", "along", "an", "and", "any", "are", "among", "as", "at", "basic",
    "be", "between", "by", "each", "exactly", "figure", "for", "from",
    "has", "have", "in", "into", "is", "it", "its", "like", "location",
    "made", "main", "negative", "no", "not", "object", "objects", "of",
    "on", "only", "or", "other", "own", "panel", "per", "plain", "point",
    "positive", "principal", "scene", "self", "shape", "shaped", "simple",
    "single", "spot", "start", "that", "the", "their", "there", "to",
    "unified", "versus", "which", "with", "within", "without",
})

_NUMBER_WORDS = frozenset({
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "several", "many", "few", "multiple", "pair",
})

# Closed-class quantity/measure vocabulary: these words describe the
# direction or magnitude of a measurement (handled by the fitted threshold
# rule and the declared order), not structure to be witnessed.
_MEASURE_WORDS = frozenset({
    "amount", "average", "count", "degree", "deviation", "error", "fit",
    "fitted", "fraction", "high", "large", "least", "less", "level", "long",
    "low", "max", "maximum", "mean", "measure", "measurement", "min",
    "minimum", "more", "most", "number", "overall", "ratio", "relative",
    "residual", "roughly", "score", "short", "small", "total", "value",
})


@dataclass(frozen=True)
class MissingLeg:
    semantic_term: str
    required_witness_types: tuple[str, ...]
    available_terminal_types: tuple[str, ...]
    unresolved_relation: str | None = None
    attempted_paths: tuple[str, ...] = ()
    missing_legs: tuple[str, ...] = ()
    uncovered_tokens: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)

    def describe(self) -> str:
        rel = f"\nunresolved relation: {self.unresolved_relation}" if self.unresolved_relation else ""
        missing = "\nmissing:\n- " + "\n- ".join(self.missing_legs) if self.missing_legs else ""
        paths = "\navailable paths terminate at:\n- " + "\n- ".join(self.available_terminal_types)
        tokens = (f"\nuncovered tokens: {', '.join(self.uncovered_tokens)} "
                  "(each content word must map to a witness/leg in the score "
                  "path, a leg's covers vocabulary, or a declared gluing; "
                  "rephrase or add typed structure)"
                  if self.uncovered_tokens else "")
        return (
            "MISSING_LEG\n"
            f"semantic term: {self.semantic_term}\n"
            f"required: {' + '.join(self.required_witness_types) or '(no registry match; new legs or a gluing needed)'}"
            f"{tokens}{paths}{rel}{missing}"
        )


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _singular(token: str) -> str:
    if token.endswith(("ss", "us", "is")):
        return token
    if len(token) > 3 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def term_tokens(term: str) -> tuple[str, ...]:
    tokens = []
    for raw in _norm(term).split():
        tok = _singular(raw)
        if len(tok) < 3 or tok in _STOPWORDS:
            continue
        tokens.append(tok)
    return tuple(dict.fromkeys(tokens))


def _token_variants(token: str) -> tuple[str, ...]:
    variants = [token]
    for prefix in ("un", "non"):
        if token.startswith(prefix) and len(token) - len(prefix) >= 3:
            variants.append(token[len(prefix):])
    return tuple(variants)


def _tokens_match(a: str, b: str) -> bool:
    """Prefix-tolerant stem match: 'decomposition' matches 'decompose'."""
    if a == b:
        return True
    return (len(a) >= 4 and b.startswith(a)) or (len(b) >= 4 and a.startswith(b))


def _is_quantity_word(token: str) -> bool:
    return token in _NUMBER_WORDS or token in _MEASURE_WORDS or token.isdigit()


def explicit_terms(hypothesis) -> tuple[str, ...]:
    values = []
    for attr in ("semantic_requirements", "relations"):
        values.extend(str(x) for x in getattr(hypothesis, attr, ()) if str(x).strip())
    return tuple(dict.fromkeys(values))


def _name_tokens(name: str) -> tuple[str, ...]:
    return tuple(_singular(w) for w in _norm(name).split())


def _stem_in(token: str, name: str) -> bool:
    if token in _norm(name).replace(" ", ""):
        return True
    return any(_tokens_match(token, w) for w in _name_tokens(name))


def witness_type_suggestions(token: str, registry) -> tuple[str, ...]:
    return tuple(sorted(
        t for t in registry.terminal_types()
        if t.endswith("Witness") and _stem_in(token, t)
    ))


def leg_suggestions(token: str, registry) -> tuple[str, ...]:
    names = []
    for contract in registry.contracts():
        if _stem_in(token, contract.name) or _stem_in(token, contract.codomain):
            names.append(contract.name)
    return tuple(sorted(dict.fromkeys(names)))


def proxy_covered_tokens(used_leg_contracts) -> frozenset[str]:
    covered = set()
    for contract in used_leg_contracts:
        for claim in contract.proxy_for:
            covered.update(term_tokens(claim))
            covered.add(_singular(_norm(claim).replace(" ", "")))
    return frozenset(covered)


def _proxy_match(token: str, proxy_tokens: frozenset[str]) -> bool:
    return any(_tokens_match(token, p) for p in proxy_tokens)


def audit_term_coverage(hypothesis, node_types: dict[str, str],
                        score_dep_nodes: frozenset[str],
                        used_legs: tuple[str, ...],
                        registry,
                        extra_terms: tuple[str, ...] = ()) -> tuple[MissingLeg, ...]:
    """Return one MissingLeg per declared term the cone does not witness."""
    dep_types = {node_types[n] for n in score_dep_nodes if n in node_types}
    dep_leg_names = {n for n in score_dep_nodes if n in set(used_legs)}
    used_contracts = [registry.get(name) for name in dict.fromkeys(used_legs)]
    proxy_tokens = proxy_covered_tokens(
        registry.get(name) for name in dep_leg_names) if dep_leg_names else \
        proxy_covered_tokens(used_contracts)
    gluing_names: list[str] = []
    for spec in getattr(hypothesis, "cofibrations", ()):
        gluing_names.extend((spec.name, spec.source_type, spec.target_type,
                             spec.attachment_leg))

    available = tuple(sorted({t for t in node_types.values()}))
    failures: list[MissingLeg] = []
    for term in explicit_terms(hypothesis) + tuple(extra_terms):
        covered_any = False
        violations: list[str] = []
        unknown: list[str] = []
        skipped_all = True
        for token in term_tokens(term):
            if _is_quantity_word(token):
                continue
            skipped_all = False
            variants = _token_variants(token)
            is_covered = any(
                _proxy_match(v, proxy_tokens)
                or any(_stem_in(v, t) for t in dep_types)
                or any(_stem_in(v, leg) for leg in dep_leg_names)
                or any(_stem_in(v, name) for name in gluing_names if name)
                for v in variants
            )
            if is_covered:
                covered_any = True
                continue
            known = any(
                witness_type_suggestions(v, registry)
                or leg_suggestions(v, registry)
                for v in variants
            )
            if known:
                violations.append(token)
            else:
                unknown.append(token)
        if skipped_all or (covered_any and not violations):
            continue
        failing = violations or unknown
        suggested_types: list[str] = []
        suggested_legs: list[str] = []
        for token in failing:
            for v in _token_variants(token):
                suggested_types.extend(witness_type_suggestions(v, registry))
                suggested_legs.extend(leg_suggestions(v, registry))
        failures.append(MissingLeg(
            semantic_term=term,
            required_witness_types=tuple(dict.fromkeys(suggested_types)),
            available_terminal_types=available,
            unresolved_relation=(
                "term names registry-expressible structure the score does not execute"
                if violations else
                "no registry structure matches this term; request new legs or declare a gluing"),
            attempted_paths=tuple(used_legs),
            missing_legs=tuple(dict.fromkeys(suggested_legs)),
            uncovered_tokens=tuple(failing),
        ))
    return tuple(failures)


def score_depends_on_witness(node_types: dict[str, str],
                             score_dep_nodes: frozenset[str]) -> bool:
    return any(
        node_types.get(n, "").endswith("Witness") for n in score_dep_nodes)
