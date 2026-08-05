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
- *unknown*   — the registry has no concept of it ("spine", "bird").  Unknown
  content is an unsupported semantic claim even when a neighboring token is
  covered; `bird-like connected component` may not ride on `object_count`.

A term FAILS iff any content token is a violation or is unknown. Closed-class
operator words are handled by the canonical score parser; metric identity is
retained and checked against the executing score contract. Suggestions in the
failure are derived from the registry, never from a concept table.

Admissibility stays a hard gate; Kolmogorov/MDL selection happens after.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from semantic_legs import is_witness_codomain

_STOPWORDS = frozenset({
    "a", "along", "an", "and", "any", "are", "among", "as", "at",
    "be", "between", "but", "by", "contain", "consist", "each", "exactly",
    "figure", "for", "from",
    "has", "have", "in", "into", "is", "it", "its", "like", "location",
    "made", "main", "negative", "no", "none", "not", "object", "objects", "of",
    "on", "only", "or", "other", "own", "panel", "per",
    "image", "positive", "principal", "scene", "self", "shape",
    "single", "spot", "start", "that", "than", "the", "their", "there", "to",
    "unified", "versus", "which", "with", "within", "without",
})
_RELATION_WORDS = frozenset({
    "along", "among", "between", "by", "contain", "consist", "from", "in",
    "into", "made", "on", "to", "with", "within",
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
    "minimum", "more", "most", "number", "overall", "aggregate", "ratio", "relative",
    "residual", "roughly", "score", "short", "small", "total", "value",
    "higher", "lower", "greater", "fewer", "larger", "smaller", "longer",
    "shorter", "above", "below", "over", "under", "density", "occupancy",
    "confidence", "length", "angle", "uniformity", "elongation", "aspect",
})

CARDINAL_VALUES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "single": 1,
}
HIGH_DIRECTION_WORDS = frozenset({
    "more", "higher", "greater", "larger", "longer", "above", "over",
})
LOW_DIRECTION_WORDS = frozenset({
    "less", "lower", "fewer", "smaller", "shorter", "below", "under",
})
ABSENCE_WORDS = frozenset({"no", "none", "without", "zero"})

# Measurement identity is deliberately separate from comparison grammar.
# A previous version skipped both as generic "measure words", which allowed
# prose about line length to be discharged by line residual (and aspect ratio
# by occupancy).  These canonical identities are used by both the compiler
# and verifier; aliases that genuinely denote the same scalar share a value.
_METRIC_IDENTITY = {
    "aggregate": "total",
    "angle": "angle",
    "aspect": "aspect",
    "confidence": "confidence",
    "count": "count",
    "density": "occupancy",
    "deviation": "residual",
    "elongation": "elongation",
    "error": "residual",
    "fraction": "fraction",
    "ink": "ink",
    "largest": "largest",
    "length": "length",
    "number": "count",
    "occupancy": "occupancy",
    "overall": "total",
    "ratio": "ratio",
    "residual": "residual",
    "total": "total",
    "uniformity": "uniformity",
}
_GENERIC_METRIC_WORDS = frozenset({
    "amount", "fit", "measure", "measurement", "metric", "score", "value",
})
_COMPARISON_GRAMMAR = frozenset({
    "above", "at", "below", "exactly", "fewer", "greater", "higher",
    "least", "less", "lower", "maximum", "minimum", "more", "most",
    "not", "or", "over", "than", "under",
})
_VAGUE_QUANTIFIERS = frozenset({
    "few", "many", "several",
})
_VAGUE_MAGNITUDES = frozenset({
    "high", "large", "long", "low", "short", "small",
})
_APPROXIMATION_WORDS = frozenset({
    "about", "approximately", "around", "roughly",
})
_DISTRIBUTIVE_QUANTIFIERS = frozenset({
    "all", "any", "both", "each", "either", "every", "exclusively",
    "neither", "only", "per", "sole",
})
_COMPARATIVE_NORMALIZATION = {
    "larger": ("high", "size"),
    "smaller": ("low", "size"),
    "longer": ("high", "length"),
    "shorter": ("low", "length"),
    "straighter": ("high", "straight"),
    "smoother": ("high", "smooth"),
}
_NEGATABLE_PROXY_BASES = frozenset({"closed", "open", "smooth", "straight"})
_NEGATABLE_LESS_BASES = frozenset({
    "attachment", "branch", "circle", "component", "contact", "contour",
    "crossing", "cycle", "endpoint", "ink", "intersection", "loop",
    "object", "part", "triangle",
})
_LEXICALIZED_LESS_WORDS = frozenset({
    "boundless", "countless", "endless", "numberless", "peerless",
    "priceless", "regardless", "timeless",
})


@dataclass(frozen=True)
class ScoreOperator:
    """Canonical positive-side operator carried by one semantic phrase.

    ``direction`` is the direction of the *named semantic quantity* before a
    directional proxy is composed with its scalar polarity.  Thus "more
    straight" is semantically high but maps to a low line residual.
    """

    mode: str | None
    target: float | None = None
    direction: str | None = None
    negated: bool = False


@dataclass(frozen=True)
class CalibratedClaim:
    """An operator bound to the nearest claim anchor in its clause."""

    anchor: tuple[str, ...]
    operator: ScoreOperator

    def signature(self) -> tuple:
        target = self.operator.target
        if target is not None and float(target).is_integer():
            target = int(target)
        return (
            self.anchor,
            self.operator.mode,
            target,
            self.operator.direction,
            self.operator.negated,
        )


_LEXICAL_TOKEN_RE = re.compile(
    r">=|<=|!=|==|>|<|=|"
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?\+?|"
    r"[a-z]+(?:'[a-z]+)?",
    re.IGNORECASE,
)


def _canonical_text(text: str) -> str:
    canonical = (str(text).lower().replace("≥", ">=").replace("≤", "<=")
                 .replace("≠", "!=").replace("−", "-"))
    return re.sub(
        r"\b(un|non)[\s-]+(closed|open|smooth|straight)\b",
        lambda match: match.group(1) + match.group(2),
        canonical,
    )


def _lexical_tokens(text: str) -> tuple[str, ...]:
    canonical = _canonical_text(text)
    return tuple(match.group(0).lower()
                 for match in _LEXICAL_TOKEN_RE.finditer(canonical))


def _discarded_surface_tokens(text: str) -> tuple[str, ...]:
    """Meaningful surface that the ASCII registry cannot silently erase."""
    canonical = _canonical_text(text)
    markers: list[str] = []
    for character in canonical:
        if character.isascii() and (
                character.isalnum() or character.isspace()
                or character in ".,;:()'\"-_"):
            continue
        markers.append(f"surface_{ord(character):x}")
    return tuple(dict.fromkeys(markers))


def _numeric_token_value(token: str) -> int | None:
    """Return a bounded plain nonnegative integer, else ``None``.

    Decimal/scientific/signed/symbolic cardinal syntax is intentionally not
    accepted yet.  Crucially, it is recognized as unsupported instead of
    being punctuation-stripped into a different integer claim.
    """
    if not token.isdigit() or len(token) > 9:
        return None
    value = int(token)
    return value if value <= 1_000_000_000 else None


def _number_occurrences(tokens: tuple[str, ...]
                        ) -> tuple[tuple[int, int, str], ...]:
    values: list[tuple[int, int, str]] = []
    for index, token in enumerate(tokens):
        if token in CARDINAL_VALUES:
            values.append((index, CARDINAL_VALUES[token], token))
        elif token == "pair":
            values.append((index, 2, token))
        elif token == "multiple":
            values.append((index, 2, token))
        elif token and (token[0].isdigit() or token[0] in "+-."):
            value = _numeric_token_value(token)
            if value is None:
                values.append((index, -1, "<unsupported-number>"))
            else:
                values.append((index, value, token))
    return tuple(values)


def parse_score_operator(text: str) -> ScoreOperator:
    """Parse one bounded score operator without silently changing syntax.

    The function is intentionally conservative.  Spatial ``above/under`` is
    a comparator only when a numeric operand is present; vague quantifiers and
    magnitude adjectives remain unsupported until they have fixed semantics.
    """
    tokens = _lexical_tokens(text)
    words = set(tokens)
    joined = " ".join(tokens)
    numbers = _number_occurrences(tokens)

    canonical_surface = _canonical_text(text)
    if _discarded_surface_tokens(text):
        return ScoreOperator("unsupported")
    if any(symbol in canonical_surface for symbol in "~≈?%±#/"):
        return ScoreOperator("unsupported")
    if re.search(r"(?:[0-9][a-z_]|[a-z_][0-9])", canonical_surface):
        return ScoreOperator("unsupported")
    if any(token in {">=", "<=", "!=", "==", ">", "<", "="}
           or token.endswith("+") for token in tokens):
        return ScoreOperator("unsupported")
    if any(source == "<unsupported-number>" for _, _, source in numbers):
        return ScoreOperator("unsupported")
    productive_prefixed_negations = sum(
        token.startswith(prefix)
        and token[len(prefix):] in _NEGATABLE_PROXY_BASES
        for token in tokens for prefix in ("non", "un")
    )
    productive_suffix_negations = sum(
        token.endswith("less") and token[:-4] in _NEGATABLE_LESS_BASES
        for token in tokens
    )
    negation_cues = (
        sum(token in {"no", "none", "not", "without"} for token in tokens)
        + productive_prefixed_negations
        + productive_suffix_negations
        + int("other than" in joined)
    )
    if negation_cues > 1:
        # Nested negation needs an explicit compositional grammar.  Collapsing
        # `not contactless`, `not without contact`, or `not other than two`
        # to one absence/inequality operator reverses the stated predicate.
        return ScoreOperator("unsupported")
    if len(numbers) > 1:
        return ScoreOperator("unsupported")
    if words & (_VAGUE_QUANTIFIERS | _VAGUE_MAGNITUDES
                | _APPROXIMATION_WORDS | _LEXICALIZED_LESS_WORDS):
        return ScoreOperator("unsupported")
    if words & _DISTRIBUTIVE_QUANTIFIERS:
        # The current cone IR has no universal/per-carrier aggregation.  A
        # selector for one principal object cannot establish `each`/`per`.
        return ScoreOperator("unsupported")

    if numbers:
        _index, value, source = numbers[0]
        if "other than" in joined:
            return ScoreOperator(
                "not_exact", float(value), negated=True)
        # `no single component` / `without one component` have ambiguous
        # scope.  Comparator phrases below consume no/not before this guard.
        at_most = (
            ("at most" in joined and "not at most" not in joined)
            or "up to" in joined
            or "no more than" in joined
            or "not more than" in joined
            or "no greater than" in joined
            or "not greater than" in joined
            or "no higher than" in joined
            or "not higher than" in joined
            or "no above" in joined or "not above" in joined
            or "no over" in joined or "not over" in joined
            or f"{source} or fewer" in joined
            or f"{source} or less" in joined
            or f"{source} and fewer" in joined
            or f"{source} and less" in joined
            or "at or below" in joined or "at or under" in joined
            or f"{source} or below" in joined
            or f"{source} and below" in joined
            or f"{source} or under" in joined
            or f"{source} and under" in joined
        )
        at_least = (
            ("at least" in joined and "not at least" not in joined)
            or "no less than" in joined
            or "not less than" in joined
            or "no fewer than" in joined
            or "not fewer than" in joined
            or "no lower than" in joined
            or "not lower than" in joined
            or "no below" in joined or "not below" in joined
            or "no under" in joined or "not under" in joined
            or f"{source} or more" in joined
            or f"{source} and more" in joined
            or "at or above" in joined or "at or over" in joined
            or f"{source} or above" in joined
            or f"{source} and above" in joined
            or f"{source} or over" in joined
            or f"{source} and over" in joined
        )
        greater = ("more than" in joined or "greater than" in joined
                   or "higher than" in joined) \
            and not ("no more than" in joined
                     or "not more than" in joined
                     or "no greater than" in joined
                     or "not greater than" in joined
                     or "no higher than" in joined
                     or "not higher than" in joined)
        less = ("less than" in joined or "fewer than" in joined
                or "lower than" in joined) \
            and not ("no less than" in joined
                     or "not less than" in joined
                     or "no fewer than" in joined
                     or "not fewer than" in joined
                     or "no lower than" in joined
                     or "not lower than" in joined)
        greater = greater or "not at most" in joined
        less = less or "not at least" in joined
        numeric_high = bool(words & {"above", "over"}) and not any(
            phrase in joined for phrase in (
                "no above", "not above", "no over", "not over",
                "at or above", "at or over",
                f"{source} or above", f"{source} and above",
                f"{source} or over", f"{source} and over"))
        numeric_low = bool(words & {"below", "under"}) and not any(
            phrase in joined for phrase in (
                "no below", "not below", "no under", "not under",
                "at or below", "at or under",
                f"{source} or below", f"{source} and below",
                f"{source} or under", f"{source} and under"))
        recognized = sum(bool(item) for item in (
            at_most, at_least, greater or numeric_high,
            less or numeric_low,
        ))
        if recognized > 1:
            return ScoreOperator("unsupported")
        if at_most:
            return ScoreOperator("at_most", float(value), "low")
        if source == "multiple" and words & {"no", "none", "not", "without"}:
            return ScoreOperator("unsupported")
        if at_least or source == "multiple":
            return ScoreOperator("at_least", float(value), "high")
        if greater or numeric_high:
            return ScoreOperator("greater_than", float(value), "high")
        if less or numeric_low:
            return ScoreOperator("less_than", float(value), "low")
        if words & {
                "more", "higher", "greater", "less", "lower", "fewer",
                "above", "over", "below", "under", "up"}:
            return ScoreOperator("unsupported")
        if words & {"max", "maximum", "min", "minimum", "least", "most"}:
            return ScoreOperator("unsupported")
        if (words & {"no", "none", "without"}) and value != 0:
            return ScoreOperator("unsupported")
        if "not" in words:
            return ScoreOperator("not_exact", float(value), negated=True)
        return ScoreOperator("exact", float(value))

    if "exactly" in words or words & {"average", "mean", "relative"}:
        return ScoreOperator("unsupported")
    if words & {"no", "none", "not", "without"} and words & {
            "more", "higher", "greater", "less", "lower", "fewer",
            "larger", "smaller", "longer", "shorter"}:
        return ScoreOperator("unsupported")
    if words & ({"max", "maximum", "min", "minimum", "least", "most"}
                | _VAGUE_MAGNITUDES):
        return ScoreOperator("unsupported")
    prefixed_negation = any(
        token.startswith(prefix)
        and token[len(prefix):] in _NEGATABLE_PROXY_BASES
        for token in tokens for prefix in ("non", "un"))
    suffix_negation = any(
        token.endswith("less") and token[:-4] in _NEGATABLE_LESS_BASES
        for token in tokens)
    if words & {"no", "none", "without", "not"} \
            or prefixed_negation or suffix_negation:
        return ScoreOperator("absence", negated=True)

    high = bool(words & {"more", "higher", "greater"})
    low = bool(words & {"less", "lower", "fewer"})
    comparative = [
        _COMPARATIVE_NORMALIZATION[token][0]
        for token in tokens if token in _COMPARATIVE_NORMALIZATION
    ]
    high = high or "high" in comparative
    low = low or "low" in comparative
    if high and low:
        return ScoreOperator("unsupported")
    if high:
        return ScoreOperator("relative", direction="high")
    if low:
        return ScoreOperator("relative", direction="low")
    # Without a numeric operand these are potentially spatial relations, not
    # scalar order.  Never let part_count "prove" that one part is above one.
    if words & {"above", "below", "over", "under"}:
        return ScoreOperator("unsupported")
    return ScoreOperator(None)


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
    return re.sub(r"[^a-z0-9]+", " ", _canonical_text(text)).strip()


def _singular(token: str) -> str:
    irregular = {
        "indices": "index",
        "matrices": "matrix",
        "vertices": "vertex",
    }
    if token in irregular:
        return irregular[token]
    if token.endswith(("ss", "us", "is")):
        return token
    if len(token) > 3 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith(
            ("ches", "shes", "sses", "xes", "zes")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def term_tokens(term: str) -> tuple[str, ...]:
    tokens = []
    normalized = _norm(term).split()
    consumed_positions: set[int] = set()
    operator = parse_score_operator(term)
    if operator.mode == "at_most":
        for index in range(len(normalized) - 1):
            if normalized[index:index + 2] == ["up", "to"]:
                consumed_positions.update((index, index + 1))
    for index, raw in enumerate(normalized):
        if index in consumed_positions:
            continue
        tok = _singular(raw)
        if tok in _STOPWORDS and tok not in _RELATION_WORDS:
            continue
        tokens.append(tok)
    tokens.extend(_discarded_surface_tokens(term))
    return tuple(dict.fromkeys(tokens))


def raw_word_tokens(text: str) -> tuple[str, ...]:
    """Lower-case lexical tokens without semantic stopword deletion.

    Unlike the former punctuation-stripping tokenizer, this preserves signs,
    decimal/exponent notation and comparator symbols so callers can reject
    unsupported syntax rather than accidentally reinterpret it.
    """
    return _lexical_tokens(text)


def plural_content_tokens(text: str) -> tuple[str, ...]:
    """Singular heads whose lexical surface is genuinely plural."""
    plurals: list[str] = []
    for token in _lexical_tokens(text):
        singular = _singular(token)
        if singular != token:
            plurals.append(singular)
    return tuple(dict.fromkeys(plurals))


def _claim_base(token: str) -> tuple[str, ...]:
    if token in _COMPARATIVE_NORMALIZATION:
        _direction, derived = _COMPARATIVE_NORMALIZATION[token]
        return (derived,)
    for prefix in ("non", "un"):
        if token.startswith(prefix) \
                and token[len(prefix):] in _NEGATABLE_PROXY_BASES:
            return (_singular(token[len(prefix):]),)
    if token.endswith("less") and token[:-4] in _NEGATABLE_LESS_BASES:
        return (_singular(token[:-4]),)
    if token == "triangular":
        return ("triangle",)
    if token == "circular":
        return ("circle",)
    return (_singular(token),)


def claim_tokens(text: str) -> tuple[str, ...]:
    """Content tokens used to associate a phrase with a score contract.

    Claim tokenization retains measurement identity (``length``, ``residual``)
    and semantic nouns that ordinary prose coverage may treat as framing.
    ``object`` is retained here so quantified objects bind to object_count.
    """
    raw = _lexical_tokens(text)
    skipped = _STOPWORDS | _COMPARISON_GRAMMAR | _VAGUE_QUANTIFIERS \
        | _VAGUE_MAGNITUDES | _APPROXIMATION_WORDS | {"pair", "multiple"}
    tokens: list[str] = []
    for token in raw:
        if token in {">=", "<=", "!=", "==", ">", "<", "="}:
            continue
        if _number_occurrences((token,)):
            continue
        if token in skipped and _singular(token) not in _RELATION_WORDS:
            # `object` matters only as the qualifier in object count.  In all
            # other contexts it is proposal framing, not a scalar claim.
            if token in {"object", "objects"}:
                tokens.append("object")
            continue
        for base in _claim_base(token):
            if len(base) >= 3:
                tokens.append(base)
    return tuple(dict.fromkeys(tokens))


def metric_identities(text: str, *, include_generic: bool = True
                      ) -> tuple[str, ...]:
    """Canonical measurement identities explicitly named by ``text``."""
    raw = _lexical_tokens(text)
    identities: list[str] = []
    derived_size = False
    for token in raw:
        if token in _COMPARATIVE_NORMALIZATION:
            derived = _COMPARATIVE_NORMALIZATION[token][1]
            if derived == "length":
                identities.append(derived)
            elif derived == "size":
                derived_size = True
        canonical = _METRIC_IDENTITY.get(_singular(token))
        if canonical:
            identities.append(canonical)
        elif include_generic and _singular(token) in _GENERIC_METRIC_WORDS:
            identities.append(_singular(token))
    if derived_size and not identities:
        identities.append("size")
    identities = list(dict.fromkeys(identities))
    # "aspect ratio" is one metric, while an unqualified ratio remains its
    # own identity.  Density and occupancy are intentional registry aliases.
    if "aspect" in identities and "ratio" in identities:
        identities.remove("ratio")
    # Generic nouns do not create a second metric when a specific one is
    # present ("ink amount", "fit confidence").
    if any(item not in _GENERIC_METRIC_WORDS for item in identities):
        identities = [item for item in identities
                      if item not in _GENERIC_METRIC_WORDS]
    return tuple(identities)


def contract_metric_identities(contract) -> tuple[str, ...]:
    text = " ".join((contract.name, *contract.proxy_for))
    return metric_identities(text, include_generic=True)


def term_metric_compatible(term: str, contract) -> bool:
    named = set(metric_identities(term))
    if not named:
        return True
    available = set(contract_metric_identities(contract))
    if named <= _GENERIC_METRIC_WORDS:
        return named <= available
    return named <= available


def _anchor_for(text: str, operator: ScoreOperator) -> tuple[str, ...]:
    tokens = _lexical_tokens(text)
    number_occurrences = _number_occurrences(tokens)
    content_positions: list[tuple[int, str]] = []
    for index, token in enumerate(tokens):
        if token in {">=", "<=", "!=", "==", ">", "<", "="}:
            continue
        if _number_occurrences((token,)):
            continue
        if _singular(token) in _STOPWORDS or token in _COMPARISON_GRAMMAR \
                or token in _VAGUE_QUANTIFIERS \
                or token in _VAGUE_MAGNITUDES \
                or token in _APPROXIMATION_WORDS:
            if token in {"object", "objects"}:
                content_positions.append((index, "object"))
            continue
        for base in _claim_base(token):
            if len(base) >= 3 and base not in _GENERIC_METRIC_WORDS:
                content_positions.append((index, base))

    metrics = list(metric_identities(text, include_generic=False))
    if metrics == ["count"]:
        metric_indices = [i for i, token in enumerate(tokens)
                          if token in {"count", "number"}]
        if metric_indices:
            count_index = metric_indices[0]
            number_index = (number_occurrences[0][0]
                            if number_occurrences else len(tokens))
            if count_index + 1 < len(tokens) \
                    and tokens[count_index + 1] == "of":
                # In "number of connected components is two", the measured
                # head is the final token of the noun phrase, not the modifier
                # nearest `number`.  Stop at a postpositive relation or
                # participle: "number of parts contacting others" counts
                # parts, while "number of intersecting circles" counts
                # circles because the participle precedes the first head.
                boundary = number_index
                seen_head = False
                content_indices = {index for index, _ in content_positions}
                for index in range(count_index + 2, number_index):
                    token = tokens[index]
                    if seen_head and (
                            token in _RELATION_WORDS
                            or token in {"that", "which", "who", "whose"}
                            or token.endswith(("ing", "ed"))):
                        boundary = index
                        break
                    if index in content_indices:
                        seen_head = True
                heads = [
                    (index, value) for index, value in content_positions
                    if count_index < index < boundary
                    and value not in {"count", "number"}
                ]
                if heads:
                    return (heads[-1][1], "count")
            qualifiers = [
                (abs(index - count_index), index, value)
                for index, value in content_positions
                if value not in {"count", "number"}
            ]
            if qualifiers:
                _distance, _index, qualifier = min(qualifiers)
                return (qualifier, "count")
    if number_occurrences and metrics and metrics != ["count"]:
        return tuple(metrics)

    if number_occurrences:
        number_index, number_value, _source = number_occurrences[0]
        after = [item for item in content_positions if item[0] > number_index]
        if after:
            # Bind to the quantified noun immediately following the number,
            # never a later predicate ("two parts in contact").  A leading
            # participle is treated as an attributive modifier, so the useful
            # established phrase "two intersecting circles" binds `circle`.
            candidates = after
            if len(after) > 1:
                non_participles = [
                    item for item in after if not tokens[item[0]].endswith("ing")
                ]
                if non_participles:
                    candidates = non_participles
            if number_value != 1:
                plural_heads = [
                    item for item in candidates
                    if len(tokens[item[0]]) > 3
                    and tokens[item[0]].endswith("s")
                    and not tokens[item[0]].endswith(("ss", "us", "is"))
                ]
                if plural_heads:
                    candidates = plural_heads
            return (candidates[0][1],)
        before = [item for item in content_positions if item[0] < number_index]
        return (before[-1][1],) if before else ()

    if metrics:
        # Count/number alone loses the counted noun, so retain its nearest
        # qualifier.  This distinguishes part_count from contact_count.
        return tuple(metrics)

    if operator.mode == "absence" and content_positions:
        negation_indices = [
            index for index, token in enumerate(tokens)
            if token in {"no", "none", "without", "not"}
            or any(token.startswith(prefix)
                   and token[len(prefix):] in _NEGATABLE_PROXY_BASES
                   for prefix in ("non", "un"))
            or (token.endswith("less")
                and token[:-4] in _NEGATABLE_LESS_BASES)
        ]
        if negation_indices:
            anchored = [item for item in content_positions
                        if item[0] >= negation_indices[-1]]
            if anchored:
                return (anchored[0][1],)
        return (content_positions[0][1],)
    if content_positions:
        return (content_positions[-1][1],)
    generics = metric_identities(text)
    return tuple(generics)


def calibrated_claims(text: str) -> tuple[CalibratedClaim, ...]:
    """Extract clause-bound score operators for header/compiler checks."""
    # Structured semantic terms are expected to be short.  Splitting at
    # conjunctions prevents numbers in separate claims from being pooled;
    # ambiguous multi-cardinal clauses still fail closed in the parser.
    number_source = "|".join(
        sorted((*CARDINAL_VALUES, "pair", "multiple", r"\d+"),
               key=len, reverse=True))
    protected = re.sub(
        rf"\b({number_source})\s+and\s+"
        r"(more|less|fewer|above|below|over|under)\b",
        r"\1 __score_bound_and__ \2",
        str(text), flags=re.IGNORECASE,
    )
    clauses = [
        clause.replace("__score_bound_and__", "and").strip()
        for clause in re.split(r"(?:[.;,]|\b(?:and|but)\b)", protected,
                               flags=re.IGNORECASE)
        if clause.strip()
    ]
    claims: list[CalibratedClaim] = []
    for clause in clauses or [str(text)]:
        operator = parse_score_operator(clause)
        if operator.mode is None:
            continue
        claims.append(CalibratedClaim(_anchor_for(clause, operator), operator))
    nonempty_anchors = {
        claim.anchor for claim in claims if claim.anchor
    }
    if len(nonempty_anchors) == 1 and any(not claim.anchor for claim in claims):
        # Shared-head conjunction: "at least two and at most four
        # components" binds both numeric clauses to `components`.  Distinct
        # explicit heads are never pooled.
        shared_anchor = next(iter(nonempty_anchors))
        claims = [
            claim if claim.anchor else CalibratedClaim(
                shared_anchor, claim.operator)
            for claim in claims
        ]
    return tuple(claims)


def calibrated_claim_signature(text: str) -> tuple[tuple, ...]:
    return tuple(sorted((claim.signature() for claim in calibrated_claims(text)),
                        key=repr))


def calibration_markers(text: str) -> tuple[str, ...]:
    """Backward-compatible rendering of canonical, bound claim signatures."""
    return tuple(repr(item) for item in calibrated_claim_signature(text))


def _token_variants(token: str) -> tuple[str, ...]:
    variants = [token]
    for prefix in ("un", "non"):
        if token.startswith(prefix) and len(token) - len(prefix) >= 3:
            variants.append(token[len(prefix):])
    if token.endswith("less") and token[:-4] in _NEGATABLE_LESS_BASES:
        variants.append(token[:-4])
    if token == "triangular":
        variants.append("triangle")
    if token == "circular":
        variants.append("circle")
    return tuple(variants)


def _tokens_match(a: str, b: str) -> bool:
    """Prefix-tolerant stem match: 'decomposition' matches 'decompose'."""
    if a == b:
        return True
    def stem(token: str) -> str:
        # Productive -pose/-position alternation (compose/composition,
        # decompose/decomposition).  A blind -tion deletion produces the
        # false stem ``decomposi`` and contradicts the declared matching
        # contract; keep the repair narrowly morphological.
        if token.endswith("position") and len(token) > len("position"):
            return token[:-len("position")] + "pose"
        for suffix in ("ation", "tion", "ion", "ment", "ness", "ical",
                       "ing", "ial", "ed", "ity", "ic", "al"):
            if token.endswith(suffix) and len(token) - len(suffix) >= 4:
                return token[:-len(suffix)]
        return token

    if stem(a) == stem(b):
        return True
    suffixes = ("s", "es", "ed", "ing", "ion", "tion", "ation",
                "ment", "ness", "ity", "al", "ial", "ic", "ical")
    if len(a) >= 4 and b.startswith(a):
        return b[len(a):] in suffixes
    if len(b) >= 4 and a.startswith(b):
        return a[len(b):] in suffixes
    return False


def term_matches_contract_claim(term: str, contract) -> bool:
    """Whether a term names the score contract or one of its declared aliases."""
    if not term_metric_compatible(term, contract):
        return False
    # For a count, the registry aliases name the entity being counted.  The
    # implementation name often also names its carrier (polygon_side_count,
    # endpoint_count), which must not let "three polygons" mean "three sides"
    # or "two paths" mean "two endpoints".  Custom count contracts without
    # aliases retain the exact-name fallback.
    contract_tokens = set()
    if contract.measurement_kind != "count" or not contract.proxy_for:
        contract_tokens.update(_name_tokens(contract.name))
    for claim in contract.proxy_for:
        aliases = claim_tokens(claim)
        # A multiword count alias names a counted compound whose head is the
        # final noun: count_curve_parts counts parts, not curves.  Structural
        # coverage still requires the whole phrase when its modifier matters.
        if contract.measurement_kind == "count" and len(aliases) > 1:
            contract_tokens.add(aliases[-1])
        else:
            contract_tokens.update(aliases)
    for domain in contract.domain:
        if contract.measurement_kind == "continuous" \
                and is_witness_codomain(domain):
            contract_tokens.update(_name_tokens(
                domain.removesuffix("Witness")))
    term_claim_tokens = claim_tokens(term)
    if contract.measurement_kind == "count" \
            and "count" in metric_identities(term, include_generic=False):
        generic = {"count", "number", "amount", "measure", "measurement",
                   "metric", "score", "value"}
        term_qualifiers = set(term_claim_tokens) - generic
        contract_qualifiers = contract_tokens - generic
        if term_qualifiers and not any(
                _tokens_match(left, right)
                for left in term_qualifiers for right in contract_qualifiers):
            return False
    return any(
        _tokens_match(token, claim)
        for token in term_claim_tokens
        for claim in contract_tokens
    )


def term_matches_phrase(term: str, phrase: str) -> bool:
    """Stem-tolerant match used for directional proxy calibration."""
    return any(
        _tokens_match(token, claim)
        for token in claim_tokens(term)
        for claim in claim_tokens(phrase)
    )


def proxy_score_direction(term: str, contract) -> tuple[str | None, str]:
    """Resolve directional aliases clause-locally, including negation."""
    directions: set[str] = set()
    clauses = [
        clause.strip()
        for clause in re.split(r"(?:[.;,]|\b(?:and|but)\b)", str(term),
                               flags=re.IGNORECASE)
        if clause.strip()
    ] or [str(term)]
    for clause in clauses:
        operator = parse_score_operator(clause)
        for proxy, base_direction in contract.proxy_directions:
            if not term_matches_phrase(clause, proxy):
                continue
            direction = base_direction
            if operator.mode == "absence" or operator.direction == "low":
                direction = "low" if direction == "high" else "high"
            directions.add(direction)
    if len(directions) > 1:
        return None, "semantic_score_direction_conflict"
    if directions:
        return next(iter(directions)), ""
    return parse_score_operator(term).direction, ""


def term_matches_produced_claim(term: str, contract) -> bool:
    """Match only what a witness-producing leg itself establishes.

    A producer's *domain* is context, not its conclusion: ``classify_triangle``
    proves TriangleWitness, not the upstream PolygonWitness.  Keeping this
    separate from score-contract matching closes the polygon→triangle
    witness-presence bypass.
    """
    phrases = list(contract.proxy_for)
    if is_witness_codomain(contract.codomain):
        phrases.append(contract.codomain.removesuffix("Witness"))
    return any(term_matches_phrase(term, phrase) for phrase in phrases)


def is_quantity_word(token: str) -> bool:
    return (token in _NUMBER_WORDS or token in _MEASURE_WORDS
            or token in _COMPARATIVE_NORMALIZATION or token.isdigit())


def explicit_terms(hypothesis) -> tuple[str, ...]:
    values = []
    for attr in ("semantic_requirements", "relations"):
        values.extend(str(x) for x in getattr(hypothesis, attr, ()) if str(x).strip())
    return tuple(dict.fromkeys(values))


def _name_tokens(name: str) -> tuple[str, ...]:
    # Registry names are snake_case while witness types are CamelCase.  Split
    # both into semantic atoms before morphology; concatenated substring
    # matching lets arbitrary fragments such as `art`, `wit`, or `gra` ride
    # on PartGraphWitness and defeats the unknown-token hard gate.
    expanded = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(name))
    return tuple(_singular(w) for w in _norm(expanded).split())


def _stem_in(token: str, name: str) -> bool:
    return any(_tokens_match(token, w) for w in _name_tokens(name))


def witness_type_suggestions(token: str, registry) -> tuple[str, ...]:
    return tuple(sorted(
        t for t in registry.terminal_types()
        if is_witness_codomain(t) and _stem_in(token, t)
    ))


def leg_suggestions(token: str, registry) -> tuple[str, ...]:
    names = []
    for contract in registry.contracts():
        if _stem_in(token, contract.name) or _stem_in(token, contract.codomain):
            names.append(contract.name)
    return tuple(sorted(dict.fromkeys(names)))


def proxy_coverage(used_leg_contracts
                   ) -> tuple[frozenset[str], tuple[tuple[str, ...], ...]]:
    """Return atomic aliases and load-bearing multiword aliases separately."""
    covered: set[str] = set()
    phrases: list[tuple[str, ...]] = []
    for contract in used_leg_contracts:
        for claim in contract.proxy_for:
            tokens = term_tokens(claim)
            if len(tokens) > 1:
                phrases.append(tokens)
            else:
                covered.update(tokens)
    return frozenset(covered), tuple(dict.fromkeys(phrases))


def _proxy_match(token: str, proxy_tokens: frozenset[str]) -> bool:
    return any(_tokens_match(token, p) for p in proxy_tokens)


def audit_term_coverage(hypothesis, node_types: dict[str, str],
                        score_dep_nodes: frozenset[str],
                        score_dep_legs: frozenset[str],
                        used_legs: tuple[str, ...],
                        registry,
                        extra_terms: tuple[str, ...] = ()) -> tuple[MissingLeg, ...]:
    """Return one MissingLeg per declared term the cone does not witness."""
    dep_types = {node_types[n] for n in score_dep_nodes if n in node_types}
    dep_leg_names = set(score_dep_legs)
    proxy_tokens, proxy_phrases = proxy_coverage(
        registry.get(name) for name in dep_leg_names)
    gluing_structure: list[str] = []
    for spec in getattr(hypothesis, "cofibrations", ()):
        # A proposer-chosen gluing *name* is prose, not evidence.  Only the
        # typed endpoints and the implemented attachment arrow may discharge
        # vocabulary, after the compiler has validated the full spec.
        gluing_structure.extend(
            (spec.source_type, spec.target_type, spec.attachment_leg))

    available = tuple(sorted({t for t in node_types.values()}))
    failures: list[MissingLeg] = []
    for term in explicit_terms(hypothesis) + tuple(extra_terms):
        tokens = term_tokens(term)
        compound_proxy_positions: set[int] = set()
        for phrase in proxy_phrases:
            for start in range(len(tokens) - len(phrase) + 1):
                if all(_tokens_match(tokens[start + offset], wanted)
                       for offset, wanted in enumerate(phrase)):
                    compound_proxy_positions.update(
                        range(start, start + len(phrase)))
        covered_any = False
        violations: list[str] = []
        unknown: list[str] = []
        skipped_all = True
        for token_index, token in enumerate(tokens):
            if is_quantity_word(token):
                continue
            skipped_all = False
            variants = _token_variants(token)
            is_covered = token_index in compound_proxy_positions or any(
                _proxy_match(v, proxy_tokens)
                or any(_stem_in(v, t) for t in dep_types)
                or any(_stem_in(v, leg) for leg in dep_leg_names)
                or any(_stem_in(v, name) for name in gluing_structure if name)
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
        if skipped_all:
            # A declaration made only of quantity/stop words is legal only
            # when its actual phrase is supported by the load-bearing path
            # (for example `object count` -> object_count).  Empty prose such
            # as `the` or `more` cannot become semantics by being skipped.
            raw_tokens = tuple(
                token for token in _name_tokens(term) if len(token) >= 3)
            auditable = any(
                _proxy_match(token, proxy_tokens)
                or any(_stem_in(token, typ) for typ in dep_types)
                or any(_stem_in(token, leg) for leg in dep_leg_names)
                or any(_stem_in(token, name)
                       for name in gluing_structure if name)
                for token in raw_tokens
            )
            if auditable:
                continue
            unknown.extend(raw_tokens or ("<empty>",))
        elif covered_any and not violations and not unknown:
            continue
        # Report every uncovered content token.  Choosing the first nonempty
        # bucket hid novel claims whenever the same phrase also named a known
        # but decorative concept (for example ``bird-like connected``).
        failing = violations + unknown
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
                "term mixes registry-expressible structure absent from the "
                "score path with unsupported structure"
                if violations and unknown else
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
        is_witness_codomain(node_types.get(n, ""))
        for n in score_dep_nodes)
