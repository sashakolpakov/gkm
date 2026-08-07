"""Closed mixed proposal language for deterministic and soft visual atoms.

This module is the untrusted-proposer boundary.  A verifier freezes a
``RegisteredAtomCatalog`` and one scorer-protocol digest before support pixels
are presented.  The model may then select zero to three exact deterministic
catalog options and may describe at most one affirmative soft claim.  It does
not write executable code, invent thresholds, choose polarity or weights, or
name atoms and cues.

The raw model formula refers only to array positions.  Parsing assigns stable
``atom-00`` and ``cue-00`` identifiers, validates that every atom is used once
in a conjunction, and emits a canonical, content-addressed value suitable for
cold replay.  A soft claim is a specification for a separately calibrated
empirical scorer; its prose is not evidence about pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import math
from pathlib import Path
import re
import unicodedata
from typing import Any, Mapping

from bongard.canonical import canonical_digest, canonical_json
from bongard.canonical_cache import (
    cached_content_bytes,
    cached_content_data,
    cached_content_digest,
)


TYPED_VISUAL_PROPOSAL_SCHEMA_VERSION = "gkm.bongard-typed-visual-proposal.v1"
TYPED_VISUAL_PROPOSER_GRAMMAR_ID = "typed-visual-proposal-grammar-v1"
TYPED_VISUAL_PROPOSER_PROMPT_ID = "typed-visual-proposal-prompt-v1"
REGISTERED_ATOM_CATALOG_SCHEMA_VERSION = "gkm.bongard-registered-atom-catalog.v1"
SOFT_AGGREGATION = "min"
VIEWS = frozenset({"literal_ink", "carrier_shape", "relational"})
MAX_DETERMINISTIC_ATOMS = 3
MAX_SOFT_CUES = 4
# These are UTF-8 byte limits, enforced by the canonical Python parser.  The
# Codex strict structured-output subset used by ``typed_visual_transport`` does
# not currently admit ``minLength``/``maxLength``; consequently these limits
# intentionally do not appear in ``typed_visual_proposal_schema``.  They are
# still prompt-published and source/digest bound through the parser artifact.
MAX_POSITIVE_DESCRIPTION_UTF8_BYTES = 320
MAX_SOFT_CUE_DESCRIPTION_UTF8_BYTES = 192
MAX_PANEL_DESCRIPTION_UTF8_BYTES = 384
PANEL_DESCRIPTION_KEYS = tuple(
    [f"pos_{index}" for index in range(6)]
    + [f"neg_{index}" for index in range(6)]
)

_IDENTIFIER = re.compile(r"[a-z][a-z0-9_.-]{0,127}\Z")
_ARGUMENT_NAME = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ASSIGNED_ID = re.compile(r"(?:atom|cue)-[0-9]+", re.IGNORECASE)
_ALLOWED_NATURAL_PROSE_PUNCTUATION = frozenset(" .,'()-/;?")


class TypedVisualProposalError(ValueError):
    """Untrusted proposal data lies outside the closed positive language."""


class TypedVisualProposalIntegrityError(TypedVisualProposalError):
    """A canonical proposal differs from verifier-frozen configuration."""


class ArgumentKind(str, Enum):
    """JSON scalar types admitted by verifier-owned atom options."""

    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"


JsonScalar = str | int | float | bool


def _strict_keys(
    value: object, expected: set[str], name: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypedVisualProposalError(f"{name} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise TypedVisualProposalError(f"{name} keys must be strings")
    actual = set(value)
    if actual != expected:
        raise TypedVisualProposalError(
            f"{name} fields differ from schema: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return value


def _exact_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypedVisualProposalError(f"{name} must be a non-empty string")
    if value != value.strip():
        raise TypedVisualProposalError(
            f"{name} must not contain surrounding whitespace"
        )
    if "\x00" in value:
        raise TypedVisualProposalError(f"{name} contains a NUL byte")
    return value


def _exact_sha256(value: object, name: str) -> str:
    text = _exact_text(value, name)
    if _SHA256.fullmatch(text) is None:
        raise TypedVisualProposalError(f"{name} must be a lowercase sha256")
    return text


def _identifier(value: object, name: str) -> str:
    text = _exact_text(value, name)
    if _IDENTIFIER.fullmatch(text) is None:
        raise TypedVisualProposalError(f"invalid {name} {text!r}")
    return text


def _argument_name(value: object, name: str) -> str:
    text = _exact_text(value, name)
    if _ARGUMENT_NAME.fullmatch(text) is None:
        raise TypedVisualProposalError(f"invalid {name} {text!r}")
    return text


def _normalise_text(value: str) -> str:
    return (
        unicodedata.normalize("NFKC", value)
        .casefold()
        .replace("’", "'")
        .replace("‘", "'")
        .replace("‐", "-")
        .replace("‑", "-")
        .replace("–", "-")
        .replace("—", "-")
    )


def _bounded_natural_prose(
    value: object,
    name: str,
    *,
    maximum_utf8_bytes: int,
) -> str:
    """Return exact, bounded English prose from an untrusted model.

    This deliberately small character language prevents control characters,
    multiline role blocks, JSON/markup delimiters, bidi controls, and Unicode
    compatibility spellings from entering a later prompt.  It is a transport
    hardening rule, not a proof that the surviving natural language denotes a
    positive visual property.
    """

    text = _exact_text(value, name)
    try:
        encoded = text.encode("utf-8")
    except UnicodeError as exc:
        raise TypedVisualProposalError(f"{name} is not valid UTF-8") from exc
    if len(encoded) > maximum_utf8_bytes:
        raise TypedVisualProposalError(
            f"{name} exceeds {maximum_utf8_bytes} UTF-8 bytes"
        )
    if unicodedata.normalize("NFKC", text) != text:
        raise TypedVisualProposalError(f"{name} must use canonical NFKC text")
    for character in text:
        if not (
            character.isascii()
            and (
                character.isalnum()
                or character in _ALLOWED_NATURAL_PROSE_PUNCTUATION
            )
        ):
            raise TypedVisualProposalError(
                f"{name} contains a forbidden prose character U+{ord(character):04X}"
            )
    return text


# These are explicit logical/absence constructions.  Prefixes such as
# ``dis-``, ``a-`` and ``un-`` are deliberately not rejected: disconnected,
# asymmetric and unequal can denote constructive, directly visible
# organization.  Likewise ``separated`` is an affirmative spatial relation.
_NEGATION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("no", re.compile(r"\bno\b")),
    ("not", re.compile(r"\bnot\b")),
    ("without", re.compile(r"\bwithout\b")),
    ("lack", re.compile(r"\black(?:s|ed|ing)?\b")),
    ("absence", re.compile(r"\babsence\b")),
    ("absent", re.compile(r"\babsent\b")),
    ("missing", re.compile(r"\bmiss(?:ing|es|ed)?\b")),
    ("neither", re.compile(r"\bneither\b")),
    ("nor", re.compile(r"\bnor\b")),
    ("none", re.compile(r"\bnone\b")),
    ("never", re.compile(r"\bnever\b")),
    ("cannot", re.compile(r"\bcannot\b")),
    ("false", re.compile(r"\bfalse\b")),
    ("contraction", re.compile(r"\b[a-z]+n't\b")),
    (
        "apostrophe-free contraction",
        re.compile(
            r"\b(?:isnt|arent|wasnt|werent|doesnt|dont|hasnt|havent|cant|wont)\b"
        ),
    ),
    ("non-", re.compile(r"\bnon(?:[- ]|(?=[a-z]))")),
    ("-less", re.compile(r"\b[a-z]+less\b")),
    ("free of", re.compile(r"\bfree\s+of\b")),
    ("devoid of", re.compile(r"\bdevoid\s+of\b")),
    ("omit", re.compile(r"\bomit(?:s|ted|ting)?\b")),
    ("exclude", re.compile(r"\bexclud(?:e|es|ed|ing)\b")),
    ("except", re.compile(r"\bexcept(?:ing)?\b")),
    ("zero", re.compile(r"(?:\bzero\b|\b0\b)")),
    ("logical negation symbol", re.compile(r"(?:¬|!=|≠)")),
)

_SUPPORT_RELATIVE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "labelled support item",
        re.compile(
            r"\b(?:positive|negative)(?:[\s-]+support)?[\s-]+"
            r"(?:panel|example|image|side|set|class)s?\b"
        ),
    ),
    ("support class plural", re.compile(r"\b(?:positives|negatives)\b")),
    (
        "support set",
        re.compile(r"\b(?:support|training)[\s-]+(?:panel|example|image|side|set)s?\b"),
    ),
    ("support label", re.compile(r"\bclass[\s-]+(?:a|b|0|1)\b")),
    (
        "explicit support label",
        re.compile(
            r"\b(?:label(?:led)?|target|class(?:ified)?)\s*(?:is|=|:)?\s*"
            r"(?:positive|negative|a|b|0|1)\b"
        ),
    ),
    (
        "panel support label",
        re.compile(
            r"\b(?:this|the)[\s-]+(?:panel|example|image)[\s-]+"
            r"(?:is|belongs[\s-]+to)[\s-]+(?:the[\s-]+)?"
            r"(?:positive|negative|target|other)\b"
        ),
    ),
    ("named support label", re.compile(r"\b(?:positive|negative)[\s-]+label\b")),
    (
        "opposite class",
        re.compile(r"\b(?:other|opposite)[\s-]+(?:class|side|examples?|panels?)\b"),
    ),
    ("panel label", re.compile(r"\b(?:pos|neg)[_-][0-9]+\b")),
    (
        "support comparison",
        re.compile(
            r"\b(?:compared|relative|set)[\s-]+(?:to|with|against)[\s-]+"
            r"(?:the[\s-]+)?(?:positive|negative|support|other|opposite)\b"
        ),
    ),
    (
        "support contrast",
        re.compile(
            r"\b(?:versus|vs\.?|in[\s-]+contrast[\s-]+to)[\s-]+"
            r"(?:the[\s-]+)?(?:positive|negative|support|other|opposite)\b"
        ),
    ),
)

# These references can be harmless in the mandatory per-panel audit channel
# (whose keys already bind presentation order), but they are forbidden in a
# proposed rule or cue because they make that predicate support-set specific.
_SUPPORT_INDEX_COUNT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "indexed support item",
        re.compile(
            r"\b(?:(?:first|second|third|fourth|fifth|sixth|last)|"
            r"[0-9]+(?:st|nd|rd|th)?)\s+(?:positive|negative|target|other|"
            r"support|training)?[\s-]*(?:panel|example|image|presentation)\b"
        ),
    ),
    (
        "numbered support item",
        re.compile(
            r"\b(?:positive|negative|target|other|support|training)?[\s-]*"
            r"(?:panel|example|image|presentation)\s*(?:number|index|#)?\s*[0-9]+\b"
        ),
    ),
    (
        "support-set cardinality",
        re.compile(
            r"\b(?:(?:all|each|every|both)(?:\s+(?:[0-9]+|one|two|three|"
            r"four|five|six|twelve))?|[0-9]+|one|two|three|four|five|six|"
            r"seven|eight|nine|ten|eleven|twelve)\s+"
            r"(?:(?:positive|negative|target|other|support|training)[\s-]+)?"
            r"(?:panels|examples|images|presentations)\b"
        ),
    ),
    (
        "support-set fraction",
        re.compile(
            r"\b(?:[0-9]+|one|two|three|four|five|six)\s*(?:/|of)\s*"
            r"(?:[0-9]+|one|two|three|four|five|six|twelve)\s+"
            r"(?:panels|examples|images|presentations)\b"
        ),
    ),
)

_CONTROL_TEXT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("assigned atom/cue ID", _ASSIGNED_ID),
    ("threshold", re.compile(r"\bthresholds?\b")),
    ("weight", re.compile(r"\bweights?\b")),
    ("polarity", re.compile(r"\bpolarity\b")),
    ("score", re.compile(r"\bscor(?:e|es|ed|ing)\b")),
    ("probability", re.compile(r"\bprobabilit(?:y|ies)\b")),
    ("cue priority", re.compile(r"\b(?:priority|prioriti[sz]e[ds]?|importance)\b")),
    ("code fence", re.compile(r"```")),
    ("code instruction", re.compile(r"\bcode\b")),
    ("code definition", re.compile(r"\b(?:def|lambda|import|eval|exec)\b")),
    (
        "code return",
        re.compile(r"\breturn[\s-]+(?:true|false|present|absent|[01])\b"),
    ),
    ("comparison expression", re.compile(r"(?:<=|>=|==|<|>)")),
    ("function call", re.compile(r"\b[a-z_][a-z0-9_.]*\s*\([^)]*\)")),
    (
        "prompt-role declaration",
        re.compile(r"(?:^|\s)(?:system|developer|assistant|user|tool)\s*:"),
    ),
    (
        "instruction override",
        re.compile(
            r"\b(?:ignore|disregard|override|bypass|forget)\b.{0,48}"
            r"\b(?:instruction|prompt|policy|schema|rule|message)s?\b"
        ),
    ),
    (
        "instruction payload",
        re.compile(
            r"\b(?:new|previous|prior|above|following|hidden|system|developer|"
            r"assistant|user|tool)"
            r"[\s-]+(?:instruction|prompt|message|role)s?\b"
        ),
    ),
    (
        "instruction execution",
        re.compile(
            r"\b(?:follow|obey|execute)\b.{0,32}"
            r"\b(?:instruction|prompt|message|command)s?\b"
        ),
    ),
    (
        "role impersonation",
        re.compile(r"\b(?:act as|you are now|switch (?:to )?role)\b"),
    ),
    (
        "output command",
        re.compile(
            r"\b(?:return|output|emit|respond|reply|write)\b.{0,40}"
            r"\b(?:json|schema|cue_judgments|supported|ambiguous|unsupported)\b"
        ),
    ),
    ("prompt vocabulary", re.compile(r"\bprompt[\s-]+injection\b")),
)

AFFIRMATIVE_PROSE_SURFACE_POLICY_SCHEMA = (
    "gkm.bongard-affirmative-prose-surface-policy.v1"
)
_AFFIRMATIVE_PROSE_PATTERN_FAMILIES: tuple[
    tuple[str, tuple[tuple[str, re.Pattern[str]], ...]], ...
] = (
    ("explicit-negation", _NEGATION_PATTERNS),
    ("support-relative", _SUPPORT_RELATIVE_PATTERNS),
    ("support-index-count", _SUPPORT_INDEX_COUNT_PATTERNS),
    ("control-text", _CONTROL_TEXT_PATTERNS),
)


def affirmative_prose_surface_policy_data() -> dict[str, object]:
    """Return every exact regex applied to rule and soft-cue prose."""

    return {
        "schema": AFFIRMATIVE_PROSE_SURFACE_POLICY_SCHEMA,
        "matching_normalization": "NFKC-casefold-with-canonical-punctuation-folds",
        "closed_families": [
            {
                "family": family,
                "patterns": [
                    {
                        "name": name,
                        "regex": pattern.pattern,
                        "flags": pattern.flags,
                    }
                    for name, pattern in patterns
                ],
            }
            for family, patterns in _AFFIRMATIVE_PROSE_PATTERN_FAMILIES
        ],
    }


def affirmative_prose_surface_policy_description() -> str:
    """Describe every downstream TypedSoftCue forbidden surface family."""

    families = "; ".join(
        family + " [" + ", ".join(name for name, _ in patterns) + "]"
        for family, patterns in _AFFIRMATIVE_PROSE_PATTERN_FAMILIES
    )
    return (
        "The downstream affirmative-prose guard additionally rejects the exact "
        "closed surface families "
        + families
        + "."
    )


def _require_affirmative_text(
    value: object,
    name: str,
    *,
    maximum_utf8_bytes: int = MAX_POSITIVE_DESCRIPTION_UTF8_BYTES,
) -> str:
    text = _exact_text(value, name)
    normalised = _normalise_text(text)
    for _family, patterns in _AFFIRMATIVE_PROSE_PATTERN_FAMILIES:
        for label, pattern in patterns:
            if pattern.search(normalised) is not None:
                raise TypedVisualProposalError(
                    f"{name} contains forbidden {label}; proposals must be "
                    "affirmative and single-panel checkable"
                )
    return _bounded_natural_prose(
        text,
        name,
        maximum_utf8_bytes=maximum_utf8_bytes,
    )


def _require_audit_panel_text(value: object, name: str) -> str:
    """Validate literal audit prose without converting it into a predicate.

    A concrete panel can literally have no loop or a missing-looking part, so
    the positive-rule negation filter does not apply here.  The audit channel
    still cannot refer to support labels/classes or carry executable controls.
    """

    text = _exact_text(value, name)
    normalised = _normalise_text(text)
    for label, pattern in (*_SUPPORT_RELATIVE_PATTERNS, *_CONTROL_TEXT_PATTERNS):
        if pattern.search(normalised) is not None:
            raise TypedVisualProposalError(
                f"{name} contains forbidden {label}; panel descriptions are "
                "literal audit prose, not support comparisons or executable controls"
            )
    return _bounded_natural_prose(
        text,
        name,
        maximum_utf8_bytes=MAX_PANEL_DESCRIPTION_UTF8_BYTES,
    )


def _parse_panel_descriptions(
    value: object, name: str = "panel_descriptions"
) -> tuple[tuple[str, str], ...]:
    raw = _strict_keys(value, set(PANEL_DESCRIPTION_KEYS), name)
    return tuple(
        (
            panel_key,
            _require_audit_panel_text(raw[panel_key], f"{name}.{panel_key}"),
        )
        for panel_key in PANEL_DESCRIPTION_KEYS
    )


def _validate_scalar(value: object, kind: ArgumentKind, name: str) -> JsonScalar:
    if kind is ArgumentKind.STRING:
        return _exact_text(value, name)
    if kind is ArgumentKind.BOOLEAN:
        if not isinstance(value, bool):
            raise TypedVisualProposalError(f"{name} must be a boolean")
        return value
    if kind is ArgumentKind.INTEGER:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypedVisualProposalError(f"{name} must be an integer")
        return value
    if kind is ArgumentKind.NUMBER:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypedVisualProposalError(f"{name} must be a number")
        if isinstance(value, float) and not math.isfinite(value):
            raise TypedVisualProposalError(f"{name} must be finite")
        return value
    raise TypeError(f"unsupported argument kind {kind!r}")


@dataclass(frozen=True, order=True)
class AtomArgument:
    """The static type of one registered atom argument."""

    name: str
    kind: ArgumentKind

    def __post_init__(self) -> None:
        _argument_name(self.name, "atom argument name")
        if not isinstance(self.kind, ArgumentKind):
            raise TypeError("atom argument kind must be ArgumentKind")

    def to_data(self) -> dict[str, str]:
        return {"name": self.name, "type": self.kind.value}


@dataclass(frozen=True)
class RegisteredAtomOption:
    """One exact comparison/argument vector precommitted by the verifier."""

    comparison: str
    arguments: tuple[tuple[str, JsonScalar], ...]

    def __post_init__(self) -> None:
        _identifier(self.comparison, "comparison")
        if not isinstance(self.arguments, tuple):
            raise TypeError("registered option arguments must be a tuple")
        names: list[str] = []
        for pair in self.arguments:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise TypeError("registered option arguments must be name/value pairs")
            name, value = pair
            names.append(_argument_name(name, "option argument name"))
            if value is None or not isinstance(value, (str, int, float, bool)):
                raise TypeError(f"option argument {name!r} must be a JSON scalar")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"option argument {name!r} must be finite")
        if len(names) != len(set(names)):
            raise ValueError("registered option contains duplicate argument names")

    @classmethod
    def from_mapping(
        cls, comparison: str, arguments: Mapping[str, JsonScalar]
    ) -> "RegisteredAtomOption":
        if not isinstance(arguments, Mapping):
            raise TypeError("registered option arguments must be a mapping")
        return cls(comparison, tuple(sorted(arguments.items())))

    def to_data(self) -> dict[str, object]:
        return {
            "comparison": self.comparison,
            "arguments": dict(sorted(self.arguments)),
        }


@dataclass(frozen=True)
class RegisteredAtomSpec:
    """Verifier-owned typed atom family and its complete finite option grid."""

    catalog_key: str
    affirmative_description: str
    arguments: tuple[AtomArgument, ...]
    allowed_options: tuple[RegisteredAtomOption, ...]

    def __post_init__(self) -> None:
        _identifier(self.catalog_key, "catalog key")
        _require_affirmative_text(
            self.affirmative_description, "registered atom description"
        )
        if not isinstance(self.arguments, tuple) or not all(
            isinstance(argument, AtomArgument) for argument in self.arguments
        ):
            raise TypeError("registered atom arguments must be AtomArgument values")
        names = tuple(argument.name for argument in self.arguments)
        if len(names) != len(set(names)):
            raise ValueError("registered atom has duplicate argument names")
        if not isinstance(self.allowed_options, tuple) or not self.allowed_options:
            raise ValueError("registered atom requires at least one allowed option")
        if not all(
            isinstance(option, RegisteredAtomOption)
            for option in self.allowed_options
        ):
            raise TypeError(
                "registered atom options must be RegisteredAtomOption values"
            )
        option_keys: list[bytes] = []
        argument_by_name = {argument.name: argument for argument in self.arguments}
        expected = set(argument_by_name)
        for index, option in enumerate(self.allowed_options):
            option_mapping = dict(option.arguments)
            if set(option_mapping) != expected:
                raise ValueError(
                    f"allowed option {index} arguments differ from registered types"
                )
            for name, value in option_mapping.items():
                _validate_scalar(
                    value,
                    argument_by_name[name].kind,
                    f"allowed option {index}.{name}",
                )
            option_keys.append(canonical_json(option.to_data()))
        if len(option_keys) != len(set(option_keys)):
            raise ValueError("registered atom contains duplicate allowed options")

    def to_data(self) -> dict[str, object]:
        return {
            "catalog_key": self.catalog_key,
            "affirmative_description": self.affirmative_description,
            "arguments": [
                argument.to_data()
                for argument in sorted(self.arguments, key=lambda item: item.name)
            ],
            "allowed_options": [
                option.to_data()
                for option in sorted(
                    self.allowed_options, key=lambda item: canonical_json(item.to_data())
                )
            ],
        }

    def canonical_selection(
        self, comparison: object, arguments: object, name: str
    ) -> tuple[str, tuple[tuple[str, JsonScalar], ...]]:
        comparison_text = _identifier(comparison, f"{name}.comparison")
        if not isinstance(arguments, Mapping):
            raise TypedVisualProposalError(f"{name}.arguments must be an object")
        if any(not isinstance(key, str) for key in arguments):
            raise TypedVisualProposalError(f"{name}.argument keys must be strings")
        argument_by_name = {argument.name: argument for argument in self.arguments}
        expected = set(argument_by_name)
        if set(arguments) != expected:
            raise TypedVisualProposalError(
                f"{name}.arguments differ from catalog: "
                f"missing={sorted(expected - set(arguments))}, "
                f"extra={sorted(set(arguments) - expected)}"
            )
        canonical_arguments: list[tuple[str, JsonScalar]] = []
        for argument_name in sorted(expected):
            value = _validate_scalar(
                arguments[argument_name],
                argument_by_name[argument_name].kind,
                f"{name}.arguments.{argument_name}",
            )
            canonical_arguments.append((argument_name, value))
        candidate = RegisteredAtomOption(
            comparison_text, tuple(canonical_arguments)
        )
        candidate_key = canonical_json(candidate.to_data())
        allowed = {
            canonical_json(option.to_data()) for option in self.allowed_options
        }
        if candidate_key not in allowed:
            raise TypedVisualProposalError(
                f"{name} comparison/arguments are outside the verifier grid"
            )
        return comparison_text, tuple(canonical_arguments)


@dataclass(frozen=True)
class RegisteredAtomCatalog:
    """Finite catalog frozen before the proposer sees support panels."""

    atoms: tuple[RegisteredAtomSpec, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.atoms, tuple) or not all(
            isinstance(atom, RegisteredAtomSpec) for atom in self.atoms
        ):
            raise TypeError("catalog atoms must be RegisteredAtomSpec values")
        keys = tuple(atom.catalog_key for atom in self.atoms)
        if len(keys) != len(set(keys)):
            raise ValueError("registered atom catalog contains duplicate keys")

    def to_data(self) -> dict[str, object]:
        return {
            "schema": REGISTERED_ATOM_CATALOG_SCHEMA_VERSION,
            "atoms": [
                atom.to_data()
                for atom in sorted(self.atoms, key=lambda item: item.catalog_key)
            ],
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self.to_data())

    def get(self, key: str) -> RegisteredAtomSpec:
        for atom in self.atoms:
            if atom.catalog_key == key:
                return atom
        raise TypedVisualProposalError(f"unknown registered atom {key!r}")


@dataclass(frozen=True)
class TypedDeterministicAtom:
    """A catalog selection after verifier assignment of its atom ID."""

    atom_id: str
    catalog_key: str
    comparison: str
    arguments: tuple[tuple[str, JsonScalar], ...]

    def __post_init__(self) -> None:
        if re.fullmatch(r"atom-[0-9]{2}", self.atom_id) is None:
            raise TypedVisualProposalError(f"invalid assigned atom ID {self.atom_id!r}")
        _identifier(self.catalog_key, "catalog key")
        _identifier(self.comparison, "comparison")
        if not isinstance(self.arguments, tuple):
            raise TypeError("typed atom arguments must be an immutable tuple")
        names: list[str] = []
        for pair in self.arguments:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise TypeError("typed atom arguments must be name/value pairs")
            name, value = pair
            names.append(_argument_name(name, "typed atom argument name"))
            if value is None or not isinstance(value, (str, int, float, bool)):
                raise TypeError(f"typed atom argument {name!r} must be a JSON scalar")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"typed atom argument {name!r} must be finite")
        names_tuple = tuple(names)
        if names_tuple != tuple(sorted(names_tuple)) or len(names_tuple) != len(
            set(names_tuple)
        ):
            raise TypedVisualProposalError(
                "typed atom arguments must be unique and sorted"
            )

    def to_data(self) -> dict[str, object]:
        return {
            "atom_id": self.atom_id,
            "catalog_key": self.catalog_key,
            "comparison": self.comparison,
            "arguments": dict(self.arguments),
        }


@dataclass(frozen=True)
class TypedSoftCue:
    """One affirmative cue with a parser-assigned stable identity."""

    cue_id: str
    positive_description: str

    def __post_init__(self) -> None:
        if re.fullmatch(r"cue-[0-9]{2}", self.cue_id) is None:
            raise TypedVisualProposalError(f"invalid assigned cue ID {self.cue_id!r}")
        self.assert_prose_policy()

    def assert_prose_policy(self) -> None:
        """Recheck model-visible prose, including after hostile mutation."""

        _require_affirmative_text(
            self.positive_description,
            "soft cue positive_description",
            maximum_utf8_bytes=MAX_SOFT_CUE_DESCRIPTION_UTF8_BYTES,
        )

    def to_data(self) -> dict[str, str]:
        return {
            "cue_id": self.cue_id,
            "positive_description": self.positive_description,
        }


@dataclass(frozen=True)
class TypedSoftClaim:
    """Positive prose specification bound to one frozen scorer protocol."""

    atom_id: str
    positive_description: str
    cues: tuple[TypedSoftCue, ...]
    aggregation: str
    scorer_protocol_digest: str

    def __post_init__(self) -> None:
        if re.fullmatch(r"atom-[0-9]{2}", self.atom_id) is None:
            raise TypedVisualProposalError(f"invalid soft atom ID {self.atom_id!r}")
        _require_affirmative_text(
            self.positive_description,
            "soft claim positive_description",
            maximum_utf8_bytes=MAX_POSITIVE_DESCRIPTION_UTF8_BYTES,
        )
        if not isinstance(self.cues, tuple) or not 1 <= len(self.cues) <= MAX_SOFT_CUES:
            raise TypedVisualProposalError(
                f"soft claim must contain 1..{MAX_SOFT_CUES} cues"
            )
        if not all(isinstance(cue, TypedSoftCue) for cue in self.cues):
            raise TypeError("soft claim cues must be TypedSoftCue values")
        for cue in self.cues:
            cue.assert_prose_policy()
        expected_cue_ids = tuple(f"cue-{index:02d}" for index in range(len(self.cues)))
        if tuple(cue.cue_id for cue in self.cues) != expected_cue_ids:
            raise TypedVisualProposalError("soft cue IDs are not canonical")
        descriptions = tuple(
            _normalise_text(cue.positive_description).strip() for cue in self.cues
        )
        if len(descriptions) != len(set(descriptions)):
            raise TypedVisualProposalError("soft claim contains duplicate cue descriptions")
        if self.aggregation != SOFT_AGGREGATION:
            raise TypedVisualProposalError(
                f"soft aggregation must be frozen as {SOFT_AGGREGATION!r}"
            )
        _exact_sha256(self.scorer_protocol_digest, "scorer_protocol_digest")

    def assert_prose_policy(self) -> None:
        """Recheck every string immediately before it enters a scorer prompt."""

        _require_affirmative_text(
            self.positive_description,
            "soft claim positive_description",
            maximum_utf8_bytes=MAX_POSITIVE_DESCRIPTION_UTF8_BYTES,
        )
        if not isinstance(self.cues, tuple) or not all(
            isinstance(cue, TypedSoftCue) for cue in self.cues
        ):
            raise TypedVisualProposalError("soft claim cues are not typed")
        for cue in self.cues:
            cue.assert_prose_policy()

    def to_data(self) -> dict[str, object]:
        return {
            "atom_id": self.atom_id,
            "positive_description": self.positive_description,
            "cues": [cue.to_data() for cue in self.cues],
            "aggregation": self.aggregation,
            "scorer_protocol_digest": self.scorer_protocol_digest,
        }


@dataclass(frozen=True)
class TypedConjunction:
    """Closed formula: all assigned atoms exactly once, in canonical order."""

    atom_ids: tuple[str, ...]
    kind: str = "all"

    def __post_init__(self) -> None:
        if self.kind != "all":
            raise TypedVisualProposalError("formula.kind must be 'all'")
        if not self.atom_ids:
            raise TypedVisualProposalError("formula must contain at least one atom")
        if not isinstance(self.atom_ids, tuple):
            raise TypeError("formula atom_ids must be an immutable tuple")
        if len(self.atom_ids) != len(set(self.atom_ids)):
            raise TypedVisualProposalError("formula contains duplicate atom IDs")
        for atom_id in self.atom_ids:
            if not isinstance(atom_id, str) or re.fullmatch(
                r"atom-[0-9]{2}", atom_id
            ) is None:
                raise TypedVisualProposalError(
                    f"formula contains invalid assigned atom ID {atom_id!r}"
                )

    def to_data(self) -> dict[str, object]:
        return {"kind": self.kind, "atom_ids": list(self.atom_ids)}


@dataclass(frozen=True)
class TypedVisualProposal:
    """Canonical mixed proposal after all model-controlled choices are closed."""

    catalog_digest: str
    positive_description: str
    panel_descriptions: tuple[tuple[str, str], ...]
    view: str
    deterministic_atoms: tuple[TypedDeterministicAtom, ...]
    soft_claim: TypedSoftClaim | None
    formula: TypedConjunction

    def __post_init__(self) -> None:
        _exact_sha256(self.catalog_digest, "catalog_digest")
        _require_affirmative_text(self.positive_description, "positive_description")
        if not isinstance(self.panel_descriptions, tuple):
            raise TypeError("panel_descriptions must be an immutable tuple")
        if len(self.panel_descriptions) != len(PANEL_DESCRIPTION_KEYS) or any(
            not isinstance(item, tuple) or len(item) != 2
            for item in self.panel_descriptions
        ):
            raise TypedVisualProposalError(
                "panel_descriptions must contain the exact canonical 6+6 entries"
            )
        panel_keys = tuple(item[0] for item in self.panel_descriptions)
        if panel_keys != PANEL_DESCRIPTION_KEYS:
            raise TypedVisualProposalError(
                "panel_descriptions are not in canonical pos_0..neg_5 order"
            )
        for panel_key, description in self.panel_descriptions:
            _require_audit_panel_text(
                description, f"panel_descriptions.{panel_key}"
            )
        if self.view not in VIEWS:
            raise TypedVisualProposalError(f"unknown view {self.view!r}")
        if not isinstance(self.deterministic_atoms, tuple) or not 0 <= len(
            self.deterministic_atoms
        ) <= MAX_DETERMINISTIC_ATOMS:
            raise TypedVisualProposalError(
                f"deterministic_atoms must contain 0..{MAX_DETERMINISTIC_ATOMS} items"
            )
        if not all(
            isinstance(atom, TypedDeterministicAtom)
            for atom in self.deterministic_atoms
        ):
            raise TypeError(
                "deterministic_atoms must contain TypedDeterministicAtom values"
            )
        if self.soft_claim is not None and not isinstance(
            self.soft_claim, TypedSoftClaim
        ):
            raise TypeError("soft_claim must be TypedSoftClaim or None")
        expected_ids = tuple(
            f"atom-{index:02d}"
            for index in range(
                len(self.deterministic_atoms) + (self.soft_claim is not None)
            )
        )
        if not expected_ids:
            raise TypedVisualProposalError("proposal must contain at least one atom")
        actual_ids = tuple(atom.atom_id for atom in self.deterministic_atoms)
        if self.soft_claim is not None:
            actual_ids += (self.soft_claim.atom_id,)
        if actual_ids != expected_ids:
            raise TypedVisualProposalError("parser-assigned atom IDs are not canonical")
        selections = tuple(
            canonical_json(
                {
                    "catalog_key": atom.catalog_key,
                    "comparison": atom.comparison,
                    "arguments": dict(atom.arguments),
                }
            )
            for atom in self.deterministic_atoms
        )
        if len(selections) != len(set(selections)):
            raise TypedVisualProposalError("proposal contains duplicate deterministic atoms")
        if not isinstance(self.formula, TypedConjunction):
            raise TypeError("formula must be TypedConjunction")
        if self.formula.atom_ids != expected_ids:
            raise TypedVisualProposalError(
                "formula must reference every assigned atom exactly once in canonical order"
            )

    def _canonical_anchor(self) -> tuple[object, ...]:
        soft = self.soft_claim
        return (
            self.catalog_digest,
            self.positive_description,
            self.panel_descriptions,
            self.view,
            tuple(
                (
                    atom.atom_id,
                    atom.catalog_key,
                    atom.comparison,
                    atom.arguments,
                )
                for atom in self.deterministic_atoms
            ),
            (
                None
                if soft is None
                else (
                    soft.atom_id,
                    soft.positive_description,
                    tuple(
                        (cue.cue_id, cue.positive_description) for cue in soft.cues
                    ),
                    soft.aggregation,
                    soft.scorer_protocol_digest,
                )
            ),
            (self.formula.kind, self.formula.atom_ids),
        )

    def _uncached_data(self) -> dict[str, object]:
        return {
            "schema": TYPED_VISUAL_PROPOSAL_SCHEMA_VERSION,
            "catalog_digest": self.catalog_digest,
            "positive_description": self.positive_description,
            "panel_descriptions": dict(self.panel_descriptions),
            "view": self.view,
            "deterministic_atoms": [
                atom.to_data() for atom in self.deterministic_atoms
            ],
            "soft_claim": (
                None if self.soft_claim is None else self.soft_claim.to_data()
            ),
            "formula": self.formula.to_data(),
        }

    def to_data(self) -> dict[str, object]:
        return cached_content_data(
            self,
            self._canonical_anchor(),
            self._uncached_data,
        )

    @property
    def digest(self) -> str:
        return cached_content_digest(
            self,
            self._canonical_anchor(),
            self._uncached_data,
        )

    def canonical_bytes(self) -> bytes:
        return cached_content_bytes(
            self,
            self._canonical_anchor(),
            self._uncached_data,
        )

    @classmethod
    def from_data(
        cls,
        data: Mapping[str, Any],
        *,
        catalog: RegisteredAtomCatalog,
        expected_scorer_protocol_digest: str | None = None,
    ) -> "TypedVisualProposal":
        """Cold-decode canonical data against verifier-frozen dependencies."""

        if not isinstance(catalog, RegisteredAtomCatalog):
            raise TypeError("catalog must be RegisteredAtomCatalog")
        raw = _strict_keys(
            data,
            {
                "schema",
                "catalog_digest",
                "positive_description",
                "panel_descriptions",
                "view",
                "deterministic_atoms",
                "soft_claim",
                "formula",
            },
            "canonical proposal",
        )
        if raw["schema"] != TYPED_VISUAL_PROPOSAL_SCHEMA_VERSION:
            raise TypedVisualProposalIntegrityError(
                f"unsupported typed proposal schema {raw['schema']!r}"
            )
        catalog_digest = _exact_sha256(raw["catalog_digest"], "catalog_digest")
        if catalog_digest != catalog.digest:
            raise TypedVisualProposalIntegrityError(
                "proposal belongs to a different registered atom catalog"
            )
        positive_description = _require_affirmative_text(
            raw["positive_description"], "positive_description"
        )
        panel_descriptions = _parse_panel_descriptions(raw["panel_descriptions"])
        view = _exact_text(raw["view"], "view")
        if view not in VIEWS:
            raise TypedVisualProposalError(f"unknown view {view!r}")

        deterministic_raw = raw["deterministic_atoms"]
        if not isinstance(deterministic_raw, list) or not 0 <= len(
            deterministic_raw
        ) <= MAX_DETERMINISTIC_ATOMS:
            raise TypedVisualProposalError(
                f"deterministic_atoms must be a list of 0..{MAX_DETERMINISTIC_ATOMS} items"
            )
        deterministic_atoms: list[TypedDeterministicAtom] = []
        for index, item in enumerate(deterministic_raw):
            item_raw = _strict_keys(
                item,
                {"atom_id", "catalog_key", "comparison", "arguments"},
                f"deterministic_atoms[{index}]",
            )
            expected_id = f"atom-{index:02d}"
            if item_raw["atom_id"] != expected_id:
                raise TypedVisualProposalIntegrityError(
                    f"deterministic_atoms[{index}].atom_id must be {expected_id!r}"
                )
            key = _identifier(
                item_raw["catalog_key"],
                f"deterministic_atoms[{index}].catalog_key",
            )
            spec = catalog.get(key)
            comparison, arguments = spec.canonical_selection(
                item_raw["comparison"],
                item_raw["arguments"],
                f"deterministic_atoms[{index}]",
            )
            deterministic_atoms.append(
                TypedDeterministicAtom(expected_id, key, comparison, arguments)
            )

        soft_claim = _decode_canonical_soft_claim(
            raw["soft_claim"],
            atom_index=len(deterministic_atoms),
            expected_scorer_protocol_digest=expected_scorer_protocol_digest,
        )
        total_atoms = len(deterministic_atoms) + (soft_claim is not None)
        if total_atoms == 0:
            raise TypedVisualProposalError("proposal must contain at least one atom")
        expected_ids = tuple(f"atom-{index:02d}" for index in range(total_atoms))
        formula_raw = _strict_keys(raw["formula"], {"kind", "atom_ids"}, "formula")
        if formula_raw["kind"] != "all":
            raise TypedVisualProposalError("formula.kind must be 'all'")
        atom_ids = formula_raw["atom_ids"]
        if not isinstance(atom_ids, list) or any(
            not isinstance(item, str) for item in atom_ids
        ):
            raise TypedVisualProposalError("formula.atom_ids must be a list of strings")
        if tuple(atom_ids) != expected_ids:
            raise TypedVisualProposalIntegrityError(
                "canonical formula must contain every assigned atom exactly once in order"
            )
        proposal = cls(
            catalog_digest=catalog_digest,
            positive_description=positive_description,
            panel_descriptions=panel_descriptions,
            view=view,
            deterministic_atoms=tuple(deterministic_atoms),
            soft_claim=soft_claim,
            formula=TypedConjunction(expected_ids),
        )
        if proposal.to_data() != dict(data):
            raise TypedVisualProposalIntegrityError(
                "proposal data is not the exact canonical representation"
            )
        return proposal


def _decode_canonical_soft_claim(
    value: object,
    *,
    atom_index: int,
    expected_scorer_protocol_digest: str | None,
) -> TypedSoftClaim | None:
    if value is None:
        return None
    raw = _strict_keys(
        value,
        {
            "atom_id",
            "positive_description",
            "cues",
            "aggregation",
            "scorer_protocol_digest",
        },
        "soft_claim",
    )
    expected_atom_id = f"atom-{atom_index:02d}"
    if raw["atom_id"] != expected_atom_id:
        raise TypedVisualProposalIntegrityError(
            f"soft_claim.atom_id must be {expected_atom_id!r}"
        )
    if raw["aggregation"] != SOFT_AGGREGATION:
        raise TypedVisualProposalIntegrityError(
            f"soft aggregation must be {SOFT_AGGREGATION!r}"
        )
    scorer_digest = _exact_sha256(
        raw["scorer_protocol_digest"], "soft_claim.scorer_protocol_digest"
    )
    if expected_scorer_protocol_digest is None:
        raise TypedVisualProposalIntegrityError(
            "cold decoding a soft claim requires the frozen scorer-protocol digest"
        )
    expected_digest = _exact_sha256(
        expected_scorer_protocol_digest, "expected_scorer_protocol_digest"
    )
    if scorer_digest != expected_digest:
        raise TypedVisualProposalIntegrityError(
            "soft claim belongs to a different scorer protocol"
        )
    cues_raw = raw["cues"]
    if not isinstance(cues_raw, list) or not 1 <= len(cues_raw) <= MAX_SOFT_CUES:
        raise TypedVisualProposalError(
            f"soft_claim.cues must be a list of 1..{MAX_SOFT_CUES} items"
        )
    cues: list[TypedSoftCue] = []
    for index, item in enumerate(cues_raw):
        cue_raw = _strict_keys(
            item, {"cue_id", "positive_description"}, f"soft_claim.cues[{index}]"
        )
        expected_cue_id = f"cue-{index:02d}"
        if cue_raw["cue_id"] != expected_cue_id:
            raise TypedVisualProposalIntegrityError(
                f"soft_claim.cues[{index}].cue_id must be {expected_cue_id!r}"
            )
        cues.append(
            TypedSoftCue(
                expected_cue_id,
                _require_affirmative_text(
                    cue_raw["positive_description"],
                    f"soft_claim.cues[{index}].positive_description",
                    maximum_utf8_bytes=MAX_SOFT_CUE_DESCRIPTION_UTF8_BYTES,
                ),
            )
        )
    return TypedSoftClaim(
        atom_id=expected_atom_id,
        positive_description=_require_affirmative_text(
            raw["positive_description"], "soft_claim.positive_description"
        ),
        cues=tuple(cues),
        aggregation=SOFT_AGGREGATION,
        scorer_protocol_digest=scorer_digest,
    )


def _option_schema(
    spec: RegisteredAtomSpec, option: RegisteredAtomOption
) -> dict[str, object]:
    argument_by_name = {argument.name: argument for argument in spec.arguments}
    properties: dict[str, object] = {}
    for name, value in sorted(option.arguments):
        properties[name] = {
            "type": argument_by_name[name].kind.value,
            "enum": [value],
        }
    return {
        "type": "object",
        "properties": {
            "catalog_key": {"type": "string", "enum": [spec.catalog_key]},
            "comparison": {"type": "string", "enum": [option.comparison]},
            "arguments": {
                "type": "object",
                "properties": properties,
                "required": sorted(properties),
                "additionalProperties": False,
            },
        },
        "required": ["catalog_key", "comparison", "arguments"],
        "additionalProperties": False,
    }


def typed_visual_proposal_schema(
    catalog: RegisteredAtomCatalog,
) -> dict[str, object]:
    """Generate the exact structured-output schema for raw model choices.

    The deployed Codex strict-schema subset rejects string length keywords, so
    ``maxLength`` is intentionally absent.  The canonical parser applies the
    source-bound UTF-8 byte and character policy after structured decoding.
    """

    if not isinstance(catalog, RegisteredAtomCatalog):
        raise TypeError("catalog must be RegisteredAtomCatalog")
    options = [
        _option_schema(spec, option)
        for spec in sorted(catalog.atoms, key=lambda item: item.catalog_key)
        for option in sorted(
            spec.allowed_options, key=lambda item: canonical_json(item.to_data())
        )
    ]
    deterministic_items: dict[str, object]
    if options:
        deterministic_items = {"anyOf": options}
    else:
        deterministic_items = {"not": {}}
    soft_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "positive_description": {"type": "string"},
            "cue_descriptions": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["positive_description", "cue_descriptions"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "positive_description": {"type": "string"},
            "panel_descriptions": {
                "type": "object",
                "properties": {
                    panel_key: {"type": "string"}
                    for panel_key in PANEL_DESCRIPTION_KEYS
                },
                "required": list(PANEL_DESCRIPTION_KEYS),
                "additionalProperties": False,
            },
            "view": {"type": "string", "enum": sorted(VIEWS)},
            "deterministic_atoms": {
                "type": "array",
                "items": deterministic_items,
            },
            "soft_claim": {"anyOf": [{"type": "null"}, soft_schema]},
            "formula": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["all"]},
                    "atom_indices": {
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                },
                "required": ["kind", "atom_indices"],
                "additionalProperties": False,
            },
        },
        "required": [
            "positive_description",
            "panel_descriptions",
            "view",
            "deterministic_atoms",
            "soft_claim",
            "formula",
        ],
        "additionalProperties": False,
    }


def parse_typed_visual_proposal(
    payload: Mapping[str, Any],
    *,
    catalog: RegisteredAtomCatalog,
    scorer_protocol_digest: str | None = None,
) -> TypedVisualProposal:
    """Parse one untrusted model payload into the canonical mixed contract."""

    if not isinstance(catalog, RegisteredAtomCatalog):
        raise TypeError("catalog must be RegisteredAtomCatalog")
    raw = _strict_keys(
        payload,
        {
            "positive_description",
            "panel_descriptions",
            "view",
            "deterministic_atoms",
            "soft_claim",
            "formula",
        },
        "proposal",
    )
    positive_description = _require_affirmative_text(
        raw["positive_description"], "positive_description"
    )
    panel_descriptions = _parse_panel_descriptions(raw["panel_descriptions"])
    view = _exact_text(raw["view"], "view")
    if view not in VIEWS:
        raise TypedVisualProposalError(f"unknown view {view!r}")

    deterministic_raw = raw["deterministic_atoms"]
    if not isinstance(deterministic_raw, list) or not 0 <= len(
        deterministic_raw
    ) <= MAX_DETERMINISTIC_ATOMS:
        raise TypedVisualProposalError(
            f"deterministic_atoms must be a list of 0..{MAX_DETERMINISTIC_ATOMS} items"
        )
    deterministic_atoms: list[TypedDeterministicAtom] = []
    for index, item in enumerate(deterministic_raw):
        atom_raw = _strict_keys(
            item,
            {"catalog_key", "comparison", "arguments"},
            f"deterministic_atoms[{index}]",
        )
        key = _identifier(
            atom_raw["catalog_key"], f"deterministic_atoms[{index}].catalog_key"
        )
        spec = catalog.get(key)
        comparison, arguments = spec.canonical_selection(
            atom_raw["comparison"],
            atom_raw["arguments"],
            f"deterministic_atoms[{index}]",
        )
        deterministic_atoms.append(
            TypedDeterministicAtom(
                atom_id=f"atom-{index:02d}",
                catalog_key=key,
                comparison=comparison,
                arguments=arguments,
            )
        )

    selection_keys = tuple(
        canonical_json(
            {
                "catalog_key": atom.catalog_key,
                "comparison": atom.comparison,
                "arguments": dict(atom.arguments),
            }
        )
        for atom in deterministic_atoms
    )
    if len(selection_keys) != len(set(selection_keys)):
        raise TypedVisualProposalError(
            "deterministic_atoms contains a duplicate catalog selection"
        )

    soft_claim: TypedSoftClaim | None
    if raw["soft_claim"] is None:
        soft_claim = None
    else:
        soft_raw = _strict_keys(
            raw["soft_claim"],
            {"positive_description", "cue_descriptions"},
            "soft_claim",
        )
        if scorer_protocol_digest is None:
            raise TypedVisualProposalIntegrityError(
                "a soft claim requires the verifier-frozen scorer-protocol digest"
            )
        frozen_scorer_digest = _exact_sha256(
            scorer_protocol_digest, "scorer_protocol_digest"
        )
        cue_descriptions = soft_raw["cue_descriptions"]
        if not isinstance(cue_descriptions, list) or not 1 <= len(
            cue_descriptions
        ) <= MAX_SOFT_CUES:
            raise TypedVisualProposalError(
                f"soft_claim.cue_descriptions must be a list of 1..{MAX_SOFT_CUES} strings"
            )
        cues = tuple(
            TypedSoftCue(
                f"cue-{index:02d}",
                _require_affirmative_text(
                    description,
                    f"soft_claim.cue_descriptions[{index}]",
                    maximum_utf8_bytes=MAX_SOFT_CUE_DESCRIPTION_UTF8_BYTES,
                ),
            )
            for index, description in enumerate(cue_descriptions)
        )
        soft_claim = TypedSoftClaim(
            atom_id=f"atom-{len(deterministic_atoms):02d}",
            positive_description=_require_affirmative_text(
                soft_raw["positive_description"],
                "soft_claim.positive_description",
            ),
            cues=cues,
            aggregation=SOFT_AGGREGATION,
            scorer_protocol_digest=frozen_scorer_digest,
        )

    total_atoms = len(deterministic_atoms) + (soft_claim is not None)
    if total_atoms == 0:
        raise TypedVisualProposalError("proposal must contain at least one atom")

    formula_raw = _strict_keys(raw["formula"], {"kind", "atom_indices"}, "formula")
    if formula_raw["kind"] != "all":
        raise TypedVisualProposalError("formula.kind must be 'all'; Not/Or are forbidden")
    indices = formula_raw["atom_indices"]
    if not isinstance(indices, list) or any(
        isinstance(index, bool) or not isinstance(index, int) for index in indices
    ):
        raise TypedVisualProposalError(
            "formula.atom_indices must be a list of integer positions"
        )
    expected_indices = set(range(total_atoms))
    if len(indices) != total_atoms or set(indices) != expected_indices:
        raise TypedVisualProposalError(
            "formula must reference every assigned atom exactly once; "
            f"expected positions={sorted(expected_indices)}, received={indices!r}"
        )
    atom_ids = tuple(f"atom-{index:02d}" for index in range(total_atoms))
    return TypedVisualProposal(
        catalog_digest=catalog.digest,
        positive_description=positive_description,
        panel_descriptions=panel_descriptions,
        view=view,
        deterministic_atoms=tuple(deterministic_atoms),
        soft_claim=soft_claim,
        formula=TypedConjunction(atom_ids),
    )


def typed_visual_proposal_prompt(
    catalog: RegisteredAtomCatalog,
) -> str:
    """Build the support-only instruction for the closed mixed proposal turn."""

    if not isinstance(catalog, RegisteredAtomCatalog):
        raise TypeError("catalog must be RegisteredAtomCatalog")
    # The prospective SoftScorerProtocol binds this exact prompt's digest.
    # Its own digest cannot be an input here: doing so would create an
    # impossible SHA-256 fixed point.  The parser injects and validates that
    # protocol identity after the model turn, where it belongs.
    return f"""You are the support-only visual proposer for one Bongard problem.

State the smallest affirmative visual rule shared by the target examples.  The
output is a typed specification, not executable code and not evidence that the
rule is true of any pixels.  Choose exactly one view: literal_ink,
carrier_shape, or relational.

Before selecting any atom, concretely describe every presented panel in
panel_descriptions using exactly pos_0 through pos_5 and neg_0 through neg_5.
These twelve strings are mandatory audit prose for diagnosing whether an error
came from vision or compilation.  Describe what is literally visible in that
single panel; literal phrases such as "no enclosed loop" are allowed here.
Never use these descriptions as formula atoms, evidence, scores, labels, or
support-side comparisons.  The verifier binds them to the proposal digest but
does not execute them.

You may select 0..{MAX_DETERMINISTIC_ATOMS} deterministic atoms.  Each selection
must exactly copy one catalog_key, comparison, and complete arguments object
from a single allowed_options row in the frozen catalog below.  Do not invent
an argument, comparison, threshold, or value.  Do not supply atom IDs: the
verifier assigns atom-00, atom-01, and so on by array position.

You may additionally supply zero or one positive soft_claim for open visual
semantics such as bird-like organization.  It must contain one short positive
description and 1..{MAX_SOFT_CUES} concrete affirmative single-panel cue
descriptions.  Do not supply cue IDs, scores, thresholds, weights, aggregation,
or a scorer identifier.  The verifier assigns cue-00 onward, freezes minimum
cue aggregation, and binds the precommitted scorer protocol after this turn.

All rule and soft-claim descriptions are limited to
{MAX_POSITIVE_DESCRIPTION_UTF8_BYTES} UTF-8 bytes; each cue is limited to
{MAX_SOFT_CUE_DESCRIPTION_UTF8_BYTES} UTF-8 bytes; each panel audit description
is limited to {MAX_PANEL_DESCRIPTION_UTF8_BYTES} UTF-8 bytes.  Use one-line
NFKC ASCII prose containing only letters, digits, spaces, and the punctuation
period, comma, apostrophe, parentheses, hyphen, slash, semicolon, and question
mark.  Do not write role headers, prompts, instructions, JSON, markup, support
item indices, support-set counts, or support-set fractions in rule or cue prose.

At least one deterministic or soft atom is required.  formula.kind must be
exactly "all".  formula.atom_indices must contain each overall atom position
exactly once (deterministic atoms first, then the optional soft atom).  Use
[0,1,...] in that order.  Not, Or, polarity reversal, code, probabilities,
support labels, side-specific descriptions, and comparisons with either
support class are forbidden.  Rule and cue prose must be positive and
checkable in one isolated panel; literal panel audit prose has the explicit
negation exception described above.  Constructive visible terms such as
separated, disconnected, asymmetric, and unequal are allowed; phrases such as
"not connected" or "unlike the negative examples" are not valid rules.

Frozen catalog digest: {catalog.digest}
Frozen registered atom catalog:
{canonical_json(catalog.to_data()).decode("utf-8")}
"""


def typed_visual_proposal_prompt_digest(
    catalog: RegisteredAtomCatalog,
) -> str:
    """Bind the exact policy-static proposer prompt without a digest cycle."""

    return hashlib.sha256(
        typed_visual_proposal_prompt(catalog).encode("utf-8")
    ).hexdigest()


def typed_visual_proposal_grammar_digest(
    catalog: RegisteredAtomCatalog,
) -> str:
    """Bind the closed schema and the source of its canonical Python parser."""

    if not isinstance(catalog, RegisteredAtomCatalog):
        raise TypeError("catalog must be RegisteredAtomCatalog")
    return canonical_digest(
        {
            "schema": "gkm.bongard-typed-visual-proposer-grammar-artifact.v1",
            "grammar_id": TYPED_VISUAL_PROPOSER_GRAMMAR_ID,
            "catalog_digest": catalog.digest,
            "output_schema": typed_visual_proposal_schema(catalog),
            "parser_source_digest": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        }
    )


# Descriptive aliases retained within this new API; both names denote the same
# strict raw structured-output contract.
typed_visual_proposal_json_schema = typed_visual_proposal_schema
build_typed_visual_proposal_prompt = typed_visual_proposal_prompt


__all__ = [
    "AFFIRMATIVE_PROSE_SURFACE_POLICY_SCHEMA",
    "ArgumentKind",
    "AtomArgument",
    "MAX_DETERMINISTIC_ATOMS",
    "MAX_PANEL_DESCRIPTION_UTF8_BYTES",
    "MAX_POSITIVE_DESCRIPTION_UTF8_BYTES",
    "MAX_SOFT_CUES",
    "MAX_SOFT_CUE_DESCRIPTION_UTF8_BYTES",
    "PANEL_DESCRIPTION_KEYS",
    "REGISTERED_ATOM_CATALOG_SCHEMA_VERSION",
    "RegisteredAtomCatalog",
    "RegisteredAtomOption",
    "RegisteredAtomSpec",
    "SOFT_AGGREGATION",
    "TYPED_VISUAL_PROPOSAL_SCHEMA_VERSION",
    "TYPED_VISUAL_PROPOSER_GRAMMAR_ID",
    "TYPED_VISUAL_PROPOSER_PROMPT_ID",
    "TypedConjunction",
    "TypedDeterministicAtom",
    "TypedSoftClaim",
    "TypedSoftCue",
    "TypedVisualProposal",
    "TypedVisualProposalError",
    "TypedVisualProposalIntegrityError",
    "affirmative_prose_surface_policy_data",
    "affirmative_prose_surface_policy_description",
    "VIEWS",
    "build_typed_visual_proposal_prompt",
    "parse_typed_visual_proposal",
    "typed_visual_proposal_json_schema",
    "typed_visual_proposal_grammar_digest",
    "typed_visual_proposal_prompt",
    "typed_visual_proposal_prompt_digest",
    "typed_visual_proposal_schema",
]
