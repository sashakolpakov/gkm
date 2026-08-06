"""Headless visual proposer and frozen soft-predicate observer.

This module keeps the epistemic boundary explicit:

``support PNGs -> visual prose -> affirmative operational claim``

The prose is not a theorem about pixels.  It is either compiled against a
registered deterministic observable, or frozen as a conditionally checkable
HYBRID empirical measurement procedure whose prompt, model receipt, image
bytes, stable cue identifiers, and categorical outcome are archived.  Query
observation happens only after the proposal has been frozen by
:mod:`bongard.artifacts` or :mod:`bongard.benchmark`.
"""

from __future__ import annotations

from contextlib import contextmanager
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
import unicodedata
from typing import Any, Callable, Iterator, Mapping, Sequence

from bongard.transport import (
    DEFAULT_CODEX_MODEL,
    DEFAULT_REASONING_EFFORT,
    CloudPolicyCacheSnapshot,
    CodexProposerFailure,
    CodexReceipt,
    CodexStructuredResult,
    run_codex_named_images_structured,
    run_codex_structured,
    snapshot_cloud_policy_cache,
    validate_codex_receipt,
)
from bongard.evidence import Disposition, Evidence, Provenance


PROPOSAL_SCHEMA_VERSION = "gkm.bongard-visual-proposal.v3"
REJECTED_PROPOSAL_ATTEMPT_SCHEMA_VERSION = (
    "gkm.bongard-rejected-visual-proposal-attempt.v1"
)
HEADLESS_EPISODE_SCHEMA_VERSION = "gkm.bongard-headless-codex-episode.v4"
HYBRID_OBSERVATION_SCHEMA_VERSION = "gkm.bongard-hybrid-observation.v4"
HYBRID_EPISTEMIC_STATUS = "hybrid_empirical_conditionally_checkable"
HYBRID_NONMATCH = "nonmatch"
VIEWS = frozenset({"literal_ink", "carrier_shape", "relational"})
CONFIDENCE_LEVELS = frozenset({"high", "medium", "low"})
_OBSERVABLE_ID = re.compile(r"[a-z][a-z0-9_.-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PANEL_NAMES = tuple(
    [f"pos_{index}" for index in range(6)]
    + [f"neg_{index}" for index in range(6)]
)


class ProposalError(ValueError):
    """A visual proposal lies outside the frozen affirmative language."""


class TransportIdentityError(ProposalError):
    """A later observer was not executed under the proposal transport identity."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _nonempty_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProposalError(f"{name} must be a non-empty string")
    if "\x00" in value:
        raise ProposalError(f"{name} contains a NUL byte")
    return value.strip()


def _string_tuple(value: object, name: str, *, nonempty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ProposalError(f"{name} must be a JSON list")
    result = tuple(_nonempty_text(item, f"{name} item") for item in value)
    if nonempty and not result:
        raise ProposalError(f"{name} must not be empty")
    if len(result) != len(set(result)):
        raise ProposalError(f"{name} contains duplicates")
    return result


def _exact_cue_id(value: object, name: str) -> str:
    cue_id = _nonempty_text(value, name)
    if value != cue_id:
        raise ProposalError(f"{name} must not contain surrounding whitespace")
    if _OBSERVABLE_ID.fullmatch(cue_id) is None:
        raise ProposalError(f"{name} contains invalid cue ID {cue_id!r}")
    return cue_id


def _cue_id_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ProposalError(f"{name} must be a JSON list")
    result = tuple(
        _exact_cue_id(item, f"{name}[{index}]")
        for index, item in enumerate(value)
    )
    if len(result) != len(set(result)):
        raise ProposalError(f"{name} contains duplicates")
    return result


def _strict_keys(value: object, expected: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProposalError(f"{name} must be an object")
    actual = set(value)
    if actual != expected:
        raise ProposalError(
            f"{name} fields differ from schema: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return value


# HYBRID claims are a deliberately affirmative language.  Explicit logical
# negation, absence claims, polarity reversals, and support-side comparisons
# cannot be certified by a generic vision judgment: they need a dedicated
# registered certifier with an explicit contract.  Intrinsic visual properties
# such as ``asymmetric`` and ``irregular`` are constructive descriptors, not
# logical complements: they remain admissible only as a positive HYBRID claim
# whose concrete cue witnesses must all be observed.  This lexical filter is
# intentionally conservative and is not advertised as a complete
# natural-language theorem.
_NEGATION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("no", re.compile(r"\bno\b")),
    ("not", re.compile(r"\bnot\b")),
    ("without", re.compile(r"\bwithout\b")),
    ("lack", re.compile(r"\black(?:s|ed|ing)?\b")),
    ("absence", re.compile(r"\babsence\b")),
    ("absent", re.compile(r"\babsent\b")),
    ("missing", re.compile(r"\bmiss(?:ing|es|ed)\b")),
    ("neither", re.compile(r"\bneither\b")),
    ("nor", re.compile(r"\bnor\b")),
    ("none", re.compile(r"\bnone\b")),
    ("never", re.compile(r"\bnever\b")),
    ("cannot", re.compile(r"\bcannot\b")),
    ("false", re.compile(r"\bfalse\b")),
    ("contraction", re.compile(r"\b[a-z]+n't\b")),
    (
        "apostrophe-free contraction",
        re.compile(r"\b(?:isnt|arent|wasnt|werent|doesnt|dont|hasnt|havent|cant|wont)\b"),
    ),
    ("non-", re.compile(r"\bnon(?:[- ]|(?=[a-z]))")),
    ("-less", re.compile(r"\b[a-z]+less\b")),
    ("free of", re.compile(r"\bfree\s+of\b")),
    ("devoid of", re.compile(r"\bdevoid\s+of\b")),
    ("absence synonym", re.compile(r"\b(?:empty|vacant|void|bare)\b")),
    (
        "relational absence synonym",
        re.compile(
            r"\b(?:separate|separated|detached|isolated|disjoint|"
            r"disconnected|unconnected)\b"
        ),
    ),
    (
        "explicit un- absence complement",
        re.compile(
            r"\b(?:unfilled|unclosed|unfinished|unattached|unmarked|"
            r"unshaded|uncolou?red|unbroken|unpaired)\b"
        ),
    ),
    ("omit", re.compile(r"\bomit(?:s|ted|ting)?\b")),
    ("exclude", re.compile(r"\bexclud(?:e|es|ed|ing)\b")),
    ("except", re.compile(r"\bexcept(?:ing)?\b")),
    ("avoid", re.compile(r"\bavoid(?:s|ed|ing)?\b")),
    ("fail", re.compile(r"\bfail(?:s|ed|ing)?\b")),
    ("fewer", re.compile(r"\bfewer\b")),
    ("less", re.compile(r"\bless\b")),
    ("zero", re.compile(r"(?:\bzero\b|\b0\b)")),
    ("only", re.compile(r"\b(?:only|sole|alone)\b")),
    ("at most", re.compile(r"\bat(?:\s+|-)+most\b")),
    ("not as", re.compile(r"\bnot\s+as\b")),
    ("unlike", re.compile(r"\bunlike\b")),
    ("rather than", re.compile(r"\brather\s+than\b")),
    ("instead of", re.compile(r"\binstead\s+of\b")),
    ("other than", re.compile(r"\bother\s+than\b")),
    ("different from", re.compile(r"\bdifferent\s+from\b")),
    ("outside", re.compile(r"\boutside\b")),
    ("opposite of", re.compile(r"\bopposite\s+of\b")),
    (
        "support-side comparison",
        re.compile(
            r"\b(?:positive|negative)(?:[\s-]+support)?[\s-]+"
            r"(?:panel|example|side|image)s?\b"
        ),
    ),
    (
        "labelled support reference",
        re.compile(r"\b(?:positive|negative)[\s-]+supports?\b"),
    ),
    ("support class plural", re.compile(r"\b(?:positives|negatives)\b")),
    (
        "support class label",
        re.compile(r"\b(?:class[\s-]+[ab01]|(?:other|opposite)[\s-]+class)\b"),
    ),
    ("support comparison", re.compile(r"\bcompar(?:ed|ison)\s+(?:to|with)\b")),
    ("support contrast", re.compile(r"\b(?:in\s+contrast\s+to|versus|vs\.?)\b")),
    (
        "support-relative comparison",
        re.compile(r"\b(?:relative\s+to|set\s+against)\b"),
    ),
    (
        "negative inequality",
        re.compile(
            r"\b(?:smaller|shorter|narrower|lower|weaker)\b"
            r"(?:\W+\w+){0,5}\W+\bthan\b"
        ),
    ),
    (
        "negative count comparison",
        re.compile(r"\b(?:smaller|lower|reduced)\s+(?:number|count|amount)\b"),
    ),
    ("negative inequality symbol", re.compile(r"(?:<=|≤|<|!=|≠)")),
)


def _normalise_semantic_text(value: str) -> str:
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


def _require_affirmative_hybrid_text(value: str, name: str) -> None:
    normalised = _normalise_semantic_text(value)
    for label, pattern in _NEGATION_PATTERNS:
        if pattern.search(normalised) is not None:
            raise ProposalError(
                f"{name} contains forbidden semantic negation ({label!r}); "
                "absence concepts require a dedicated registered certifier"
            )


@dataclass(frozen=True)
class ObservableRequest:
    observable_id: str
    affirmative_interpretation: str
    arguments: tuple[tuple[str, str | int | float | bool | None], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "observable_id": self.observable_id,
            "affirmative_interpretation": self.affirmative_interpretation,
            "arguments": dict(self.arguments),
        }


@dataclass(frozen=True)
class HybridCue:
    """One immutable, affirmative cue in a HYBRID empirical procedure."""

    cue_id: str
    positive_description: str

    def __post_init__(self) -> None:
        _exact_cue_id(self.cue_id, "HybridCue.cue_id")
        _require_affirmative_hybrid_text(
            self.cue_id.replace("_", " ").replace(".", " ").replace("-", " "),
            "HybridCue.cue_id",
        )
        description = _nonempty_text(
            self.positive_description, "HybridCue.positive_description"
        )
        if description != self.positive_description:
            raise ProposalError(
                "HybridCue.positive_description must not contain surrounding whitespace"
            )
        _require_affirmative_hybrid_text(
            self.positive_description, "HybridCue.positive_description"
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "cue_id": self.cue_id,
            "positive_description": self.positive_description,
        }


@dataclass(frozen=True)
class HybridClaim:
    """A conditionally checkable empirical claim, never a pixel-level truth."""

    phrase: str
    operational_definition: str
    required_visual_cues: tuple[HybridCue, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("HybridClaim.phrase", self.phrase),
            ("HybridClaim.operational_definition", self.operational_definition),
        ):
            canonical = _nonempty_text(value, name)
            if canonical != value:
                raise ProposalError(f"{name} must not contain surrounding whitespace")
            _require_affirmative_hybrid_text(value, name)
        if not 1 <= len(self.required_visual_cues) <= 12:
            raise ProposalError("HybridClaim requires 1..12 structured visual cues")
        if not all(isinstance(cue, HybridCue) for cue in self.required_visual_cues):
            raise ProposalError("HybridClaim cues must be HybridCue values")
        cue_ids = self.required_cue_ids
        if len(cue_ids) != len(set(cue_ids)):
            raise ProposalError("HybridClaim contains duplicate cue IDs")
        description_keys = tuple(
            _normalise_semantic_text(cue.positive_description).strip()
            for cue in self.required_visual_cues
        )
        if len(description_keys) != len(set(description_keys)):
            raise ProposalError("HybridClaim contains duplicate positive cue descriptions")

    @property
    def epistemic_status(self) -> str:
        return HYBRID_EPISTEMIC_STATUS

    @property
    def required_cue_ids(self) -> tuple[str, ...]:
        return tuple(cue.cue_id for cue in self.required_visual_cues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "epistemic_status": self.epistemic_status,
            "phrase": self.phrase,
            "operational_definition": self.operational_definition,
            "required_visual_cues": [cue.to_dict() for cue in self.required_visual_cues],
        }


@dataclass(frozen=True)
class RuleProposal:
    """One support-only, positive-orientation proposal and its causal receipt."""

    positive_description: str
    panel_descriptions: tuple[tuple[str, str], ...]
    view: str
    observable_requests: tuple[ObservableRequest, ...]
    formula_atoms: tuple[str, ...]
    hybrid_claim: HybridClaim | None
    confidence: str
    receipt: CodexReceipt
    model_payload: Mapping[str, Any]

    @property
    def digest(self) -> str:
        return _digest(self.to_dict())

    @property
    def is_hybrid(self) -> bool:
        return self.hybrid_claim is not None

    def formula_data(self) -> dict[str, Any]:
        return {"kind": "all", "atoms": list(self.formula_atoms)}

    def content_dict(self) -> dict[str, Any]:
        return {
            "schema": PROPOSAL_SCHEMA_VERSION,
            "positive_description": self.positive_description,
            "panel_descriptions": dict(self.panel_descriptions),
            "view": self.view,
            "observable_requests": [item.to_dict() for item in self.observable_requests],
            "formula_template": self.formula_data(),
            "hybrid_claim": self.hybrid_claim.to_dict() if self.hybrid_claim else None,
            "confidence": self.confidence,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.content_dict(),
            "model_payload": dict(self.model_payload),
            "receipt": self.receipt.to_dict(),
        }


@dataclass(frozen=True)
class SupportPanelIdentity:
    """One proposer-visible support byte identity, in presentation order."""

    name: str
    byte_count: int
    content_digest: str

    def __post_init__(self) -> None:
        if self.name not in tuple(f"{panel_name}.png" for panel_name in _PANEL_NAMES):
            raise ProposalError(f"invalid support presentation name {self.name!r}")
        if isinstance(self.byte_count, bool) or not isinstance(self.byte_count, int) \
                or self.byte_count <= 0:
            raise ProposalError("support presentation byte_count must be positive")
        if not isinstance(self.content_digest, str) \
                or _SHA256.fullmatch(self.content_digest) is None:
            raise ProposalError(
                "support presentation content_digest must be lowercase SHA-256"
            )

    def to_dict(self) -> dict[str, str | int]:
        return {
            "name": self.name,
            "byte_count": self.byte_count,
            "content_digest": self.content_digest,
        }


@dataclass(frozen=True)
class RejectedProposalAttempt:
    """A semantically rejected, receipt-bound proposer result.

    This is deliberately not a :class:`RuleProposal`: no formula, registry, or
    accepted proposer digest can be derived from it.  It preserves enough of
    the failed turn to reproduce the canonical catalog-free parser rejection
    and receipt input envelope without pretending that a rejected rule was
    admitted.  Nonempty observable catalogs are outside the canonical CLI
    rejected-run verifier profile.
    """

    model_payload: Mapping[str, Any]
    receipt: CodexReceipt
    support_presentation: tuple[SupportPanelIdentity, ...]
    parse_error_type: str
    parse_error_reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.model_payload, Mapping) or any(
            not isinstance(key, str) for key in self.model_payload
        ):
            raise ProposalError("rejected proposal payload must be a JSON object")
        try:
            payload_copy = json.loads(_canonical_json(dict(self.model_payload)))
        except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
            raise ProposalError(
                f"rejected proposal payload is not canonical finite JSON: {exc}"
            ) from exc
        object.__setattr__(self, "model_payload", payload_copy)
        if not isinstance(self.receipt, CodexReceipt):
            raise ProposalError("rejected proposal receipt is not a CodexReceipt")
        try:
            validate_codex_receipt(self.receipt.to_dict())
        except (CodexProposerFailure, TypeError, ValueError) as exc:
            raise ProposalError(f"rejected proposal receipt is invalid: {exc}") from exc
        expected_names = tuple(f"{name}.png" for name in _PANEL_NAMES)
        if not all(
            isinstance(item, SupportPanelIdentity)
            for item in self.support_presentation
        ):
            raise ProposalError(
                "rejected proposal support presentation contains invalid identities"
            )
        if tuple(item.name for item in self.support_presentation) != expected_names:
            raise ProposalError(
                "rejected proposal support presentation must be canonical 6+6 order"
            )
        for name, value in (
            ("parse_error_type", self.parse_error_type),
            ("parse_error_reason", self.parse_error_reason),
        ):
            canonical = _nonempty_text(value, f"RejectedProposalAttempt.{name}")
            if canonical != value:
                raise ProposalError(
                    f"RejectedProposalAttempt.{name} must not contain surrounding whitespace"
                )

    def content_dict(self) -> dict[str, Any]:
        return {
            "schema": REJECTED_PROPOSAL_ATTEMPT_SCHEMA_VERSION,
            "proposal_schema": PROPOSAL_SCHEMA_VERSION,
            "model_payload": json.loads(_canonical_json(dict(self.model_payload))),
            "receipt": self.receipt.to_dict(),
            "support_presentation": [
                item.to_dict() for item in self.support_presentation
            ],
            "parse_error": {
                "error_type": self.parse_error_type,
                "reason": self.parse_error_reason,
            },
        }

    @property
    def digest(self) -> str:
        return _digest(self.content_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**self.content_dict(), "attempt_digest": self.digest}


class RejectedProposalError(ProposalError):
    """A transport succeeded, but its content was not an admissible proposal."""

    def __init__(self, attempt: RejectedProposalAttempt):
        self.attempt = attempt
        super().__init__(attempt.parse_error_reason)


_STRING = {"type": "string", "minLength": 1, "maxLength": 20_000}
_PANEL_DESCRIPTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {name: _STRING for name in _PANEL_NAMES},
    "required": list(_PANEL_NAMES),
}
_OBSERVABLE_REQUEST_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "observable_id": {
            "type": "string",
            "pattern": "^[a-z][a-z0-9_.-]{0,127}$",
        },
        "affirmative_interpretation": _STRING,
        "arguments": {
            "type": "object",
            "additionalProperties": False,
            "properties": {},
            "required": [],
        },
    },
    "required": ["observable_id", "affirmative_interpretation", "arguments"],
}
_HYBRID_CUE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "cue_id": {
            "type": "string",
            "pattern": "^[a-z][a-z0-9_.-]{0,127}$",
        },
        "positive_description": _STRING,
    },
    "required": ["cue_id", "positive_description"],
}
_HYBRID_CLAIM_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "epistemic_status": {
            "type": "string",
            "const": HYBRID_EPISTEMIC_STATUS,
        },
        "phrase": _STRING,
        "operational_definition": _STRING,
        "required_visual_cues": {
            "type": "array",
            "items": _HYBRID_CUE_SCHEMA,
            "minItems": 1,
            "maxItems": 12,
        },
    },
    "required": [
        "epistemic_status",
        "phrase",
        "operational_definition",
        "required_visual_cues",
    ],
}

RULE_PROPOSAL_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "positive_description": _STRING,
        "panel_descriptions": _PANEL_DESCRIPTION_SCHEMA,
        "view": {"type": "string", "enum": sorted(VIEWS)},
        "observable_requests": {
            "type": "array",
            "items": _OBSERVABLE_REQUEST_SCHEMA,
            "maxItems": 4,
        },
        "formula_template": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "kind": {"type": "string", "const": "all"},
                "atoms": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "minItems": 1,
                    "maxItems": 4,
                },
            },
            "required": ["kind", "atoms"],
        },
        "hybrid_claim": {"anyOf": [{"type": "null"}, _HYBRID_CLAIM_SCHEMA]},
        "confidence": {"type": "string", "enum": sorted(CONFIDENCE_LEVELS)},
    },
    "required": [
        "positive_description",
        "panel_descriptions",
        "view",
        "observable_requests",
        "formula_template",
        "hybrid_claim",
        "confidence",
    ],
}

# The canonical CLI currently supplies no PURE observable catalog.  Give the
# structured-output service the narrower language we actually accept in that
# case, so an otherwise useful visual analysis cannot fail merely because the
# model copied each cue ID into the formula.  Cues live inside one task-local
# empirical claim; they are not independently executable atoms.
_hybrid_only_schema = copy.deepcopy(RULE_PROPOSAL_SCHEMA)
_hybrid_only_schema["properties"]["observable_requests"]["maxItems"] = 0
_hybrid_only_schema["properties"]["formula_template"]["properties"]["atoms"] = {
    "type": "array",
    "items": {"type": "string", "const": "hybrid_claim"},
    "minItems": 1,
    "maxItems": 1,
}
_hybrid_only_schema["properties"]["hybrid_claim"] = _HYBRID_CLAIM_SCHEMA
HYBRID_ONLY_RULE_PROPOSAL_SCHEMA: Mapping[str, Any] = _hybrid_only_schema


def parse_rule_proposal(
    payload: Mapping[str, Any],
    *,
    receipt: CodexReceipt,
    observable_catalog: Mapping[str, str],
) -> RuleProposal:
    """Validate the semantic constraints JSON Schema cannot express."""

    raw = _strict_keys(
        payload,
        {
            "positive_description",
            "panel_descriptions",
            "view",
            "observable_requests",
            "formula_template",
            "hybrid_claim",
            "confidence",
        },
        "proposal",
    )
    positive_description = _nonempty_text(
        raw["positive_description"], "positive_description"
    )
    panel_raw = _strict_keys(
        raw["panel_descriptions"], set(_PANEL_NAMES), "panel_descriptions"
    )
    panel_descriptions = tuple(
        (name, _nonempty_text(panel_raw[name], f"panel_descriptions.{name}"))
        for name in _PANEL_NAMES
    )
    view = _nonempty_text(raw["view"], "view")
    if view not in VIEWS:
        raise ProposalError(f"unknown visual view {view!r}")
    confidence = _nonempty_text(raw["confidence"], "confidence")
    if confidence not in CONFIDENCE_LEVELS:
        raise ProposalError(f"unknown confidence {confidence!r}")

    request_raw = raw["observable_requests"]
    if not isinstance(request_raw, list) or len(request_raw) > 4:
        raise ProposalError("observable_requests must be a list of at most four items")
    requests: list[ObservableRequest] = []
    for index, item in enumerate(request_raw):
        request = _strict_keys(
            item,
            {"observable_id", "affirmative_interpretation", "arguments"},
            f"observable_requests[{index}]",
        )
        observable_id = _nonempty_text(
            request["observable_id"], f"observable_requests[{index}].observable_id"
        )
        if _OBSERVABLE_ID.fullmatch(observable_id) is None:
            raise ProposalError(f"invalid observable id {observable_id!r}")
        if observable_id not in observable_catalog:
            raise ProposalError(f"unregistered observable {observable_id!r}")
        arguments = request["arguments"]
        if not isinstance(arguments, Mapping):
            raise ProposalError(f"arguments for {observable_id} must be an object")
        canonical_arguments: list[tuple[str, str | int | float | bool | None]] = []
        for name, value in sorted(arguments.items()):
            if not isinstance(name, str) or not name:
                raise ProposalError("observable argument names must be non-empty strings")
            if value is not None and not isinstance(value, (str, int, float, bool)):
                raise ProposalError(f"argument {name!r} is not a JSON literal")
            canonical_arguments.append((name, value))
        requests.append(
            ObservableRequest(
                observable_id,
                _nonempty_text(
                    request["affirmative_interpretation"],
                    f"observable_requests[{index}].affirmative_interpretation",
                ),
                tuple(canonical_arguments),
            )
        )
    request_ids = tuple(item.observable_id for item in requests)
    if len(request_ids) != len(set(request_ids)):
        raise ProposalError("observable_requests contains duplicate IDs")

    formula = _strict_keys(raw["formula_template"], {"kind", "atoms"}, "formula_template")
    if formula["kind"] != "all":
        raise ProposalError("formula_template.kind must be 'all'")
    formula_atoms = _string_tuple(
        formula["atoms"], "formula_template.atoms", nonempty=True
    )

    hybrid: HybridClaim | None
    if raw["hybrid_claim"] is None:
        hybrid = None
        if not requests:
            raise ProposalError("a PURE proposal requires at least one registered observable")
        if set(formula_atoms) != set(request_ids):
            raise ProposalError(
                "every requested observable must be load-bearing and every atom registered"
            )
    else:
        hybrid_raw = _strict_keys(
            raw["hybrid_claim"],
            {
                "epistemic_status",
                "phrase",
                "operational_definition",
                "required_visual_cues",
            },
            "hybrid_claim",
        )
        if hybrid_raw["epistemic_status"] != HYBRID_EPISTEMIC_STATUS:
            raise ProposalError(
                "hybrid_claim.epistemic_status must mark the claim as "
                f"{HYBRID_EPISTEMIC_STATUS!r}"
            )
        if requests:
            raise ProposalError(
                "a HYBRID claim cannot be glued to PURE observables before admission"
            )
        if formula_atoms != ("hybrid_claim",):
            raise ProposalError(
                "a HYBRID formula must be exactly the positive hybrid_claim atom"
            )

        phrase = _nonempty_text(hybrid_raw["phrase"], "hybrid_claim.phrase")
        operational_definition = _nonempty_text(
            hybrid_raw["operational_definition"],
            "hybrid_claim.operational_definition",
        )
        _require_affirmative_hybrid_text(phrase, "hybrid_claim.phrase")
        _require_affirmative_hybrid_text(
            operational_definition, "hybrid_claim.operational_definition"
        )

        cue_raw = hybrid_raw["required_visual_cues"]
        if not isinstance(cue_raw, list) or not 1 <= len(cue_raw) <= 12:
            raise ProposalError(
                "hybrid_claim.required_visual_cues must contain 1..12 cue objects"
            )
        cues: list[HybridCue] = []
        for index, value in enumerate(cue_raw):
            cue = _strict_keys(
                value,
                {"cue_id", "positive_description"},
                f"hybrid_claim.required_visual_cues[{index}]",
            )
            cue_id = _exact_cue_id(
                cue["cue_id"],
                f"hybrid_claim.required_visual_cues[{index}].cue_id",
            )
            # Cue identifiers are also human-authored semantic labels.  Treat
            # separators as spaces so `missing_wing` cannot smuggle a negative
            # orientation around the prose filter.
            _require_affirmative_hybrid_text(
                cue_id.replace("_", " ").replace(".", " ").replace("-", " "),
                f"hybrid_claim.required_visual_cues[{index}].cue_id",
            )
            description = _nonempty_text(
                cue["positive_description"],
                f"hybrid_claim.required_visual_cues[{index}].positive_description",
            )
            _require_affirmative_hybrid_text(
                description,
                f"hybrid_claim.required_visual_cues[{index}].positive_description",
            )
            cues.append(HybridCue(cue_id=cue_id, positive_description=description))
        cue_ids = tuple(cue.cue_id for cue in cues)
        if len(cue_ids) != len(set(cue_ids)):
            raise ProposalError("hybrid_claim.required_visual_cues contains duplicate cue IDs")

        hybrid = HybridClaim(
            phrase=phrase,
            operational_definition=operational_definition,
            required_visual_cues=tuple(cues),
        )
        # The verifier-owned compiler maps this exact affirmative token to one
        # task-local PRESENT atom.  Arbitrary raw labels are rejected rather
        # than silently canonicalized, so `not_hybrid_claim` cannot be
        # laundered into the positive IR.

    return RuleProposal(
        positive_description=positive_description,
        panel_descriptions=panel_descriptions,
        view=view,
        observable_requests=tuple(requests),
        formula_atoms=formula_atoms,
        hybrid_claim=hybrid,
        confidence=confidence,
        receipt=receipt,
        model_payload=json.loads(_canonical_json(dict(payload))),
    )


def proposer_prompt(observable_catalog: Mapping[str, str]) -> str:
    catalog = [
        {"observable_id": name, "affirmative_meaning": description}
        for name, description in sorted(observable_catalog.items())
    ]
    return f"""You are the visual proposer for one Bongard problem.

You see exactly six labelled positive support panels (pos_0.png..pos_5.png)
and six labelled negative support panels (neg_0.png..neg_5.png).  Describe each
panel concretely before proposing the common positive rule.  Do not use file
identity, position on disk, task names, hidden metadata, or memorized dataset
labels.

Choose the view that carries the distinction:
- literal_ink: stroke texture and local appearance matter;
- carrier_shape: stroke style is nuisance but centerline geometry matters;
- relational: objects, contacts, containment, repetition, symmetry, ownership.

The rule must have the declared POSITIVE orientation.  Do not reverse a bad
feature, return Not, attach a polarity flag, or say merely that positives lack
a negative-side feature.  An affirmative absence concept is allowed only when
the catalog contains a dedicated observable that can certify that absence.

Use registered observables only when their stated affirmative meaning really
matches the rule.  Every requested observable must be load-bearing.  If the
catalog is insufficient, request no observables and return one HYBRID claim:
a short phrase, an operational definition visible in a single isolated panel,
and 1..12 concrete required visual cues.  Each cue must be an object with a
stable lowercase `cue_id` and an affirmative `positive_description`.  Use the
same cue IDs later; descriptions are not identifiers.  Mark the claim's
`epistemic_status` exactly as `{HYBRID_EPISTEMIC_STATUS}`.

HYBRID is a conditionally checkable empirical frozen vision measurement, not
pixel truth or mathematical truth.  Its phrase, operational definition, cue
IDs, and cue descriptions must all be affirmative.  Do not use explicit or
hidden negation such as no, not, non-, without, lacks, absent, missing,
neither/nor, fewer, less, at most, or negative-side comparisons.  A genuine
absence concept must wait for a dedicated registered certifier.  Never output
code, Lean, thresholds, probabilities, query policies, or arbitrary
expressions.

Intrinsic constructive descriptors such as asymmetric, irregular, unbalanced,
and unequal are allowed when they directly name visible organization in one
panel.  Ground each such descriptor in affirmative cue witnesses (for example,
distinct left and right extents or visibly unequal lobe sizes).  They do not
license phrases such as "not symmetric", absence claims, a Not formula, or a
comparison with the positive or negative support side.

For every HYBRID claim, `formula_template` must be exactly
`{{"kind":"all","atoms":["hybrid_claim"]}}`.  Required cue IDs belong inside
`hybrid_claim.required_visual_cues`; never copy them into formula atoms.

Registered observable catalog:
{json.dumps(catalog, sort_keys=True, indent=2)}
"""


@contextmanager
def _canonical_support_paths(
    positive_support: Sequence[str | Path],
    negative_support: Sequence[str | Path],
) -> Iterator[tuple[str, ...]]:
    if len(positive_support) != 6 or len(negative_support) != 6:
        raise ProposalError("the proposer requires exactly 6 positive and 6 negative supports")
    sources = tuple(Path(path).resolve() for path in (*positive_support, *negative_support))
    if len(set(sources)) != 12:
        raise ProposalError("support panel paths must be distinct")
    for source in sources:
        if not source.is_file():
            raise ProposalError(f"support panel is not a file: {source}")
    with tempfile.TemporaryDirectory(prefix="bongard-support-") as directory:
        root = Path(directory)
        targets: list[str] = []
        for name, source in zip(
            (f"{name}.png" for name in _PANEL_NAMES), sources, strict=True
        ):
            target = root / name
            shutil.copyfile(source, target)
            targets.append(str(target))
        yield tuple(targets)


def _support_presentation(
    paths: Sequence[str | Path],
) -> tuple[SupportPanelIdentity, ...]:
    if len(paths) != len(_PANEL_NAMES):
        raise ProposalError("support presentation requires exactly 12 panels")
    identities: list[SupportPanelIdentity] = []
    for name, path_value in zip(_PANEL_NAMES, paths, strict=True):
        path = Path(path_value)
        try:
            before = path.stat()
            payload = path.read_bytes()
            after = path.stat()
        except OSError as exc:
            raise ProposalError(f"cannot bind support presentation {name}: {exc}") from exc
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ProposalError(f"support presentation changed while reading: {name}")
        identities.append(
            SupportPanelIdentity(
                name=f"{name}.png",
                byte_count=len(payload),
                content_digest=hashlib.sha256(payload).hexdigest(),
            )
        )
    return tuple(identities)


StructuredTransport = Callable[..., CodexStructuredResult]

TRANSPORT_IDENTITY_FIELDS = (
    "codex_launcher_digest",
    "codex_cli_version",
    "cloud_config_bundle_cache_binding",
    "isolation_policy",
    "requested_model",
    "requested_reasoning_effort",
)


def codex_transport_identity(receipt: CodexReceipt) -> tuple[tuple[str, str], ...]:
    """Fields that must remain identical for every transport in an episode."""

    return tuple((name, str(getattr(receipt, name))) for name in TRANSPORT_IDENTITY_FIELDS)


def propose_rule(
    positive_support: Sequence[str | Path],
    negative_support: Sequence[str | Path],
    *,
    observable_catalog: Mapping[str, str],
    model: str = DEFAULT_CODEX_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    minutes: int = 15,
    verbose: bool = False,
    executable: str = "codex",
    cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot | None = None,
    transport: StructuredTransport = run_codex_structured,
) -> RuleProposal:
    """Make exactly one isolated support-only Codex proposal call."""

    for name, description in observable_catalog.items():
        if not isinstance(name, str) or _OBSERVABLE_ID.fullmatch(name) is None:
            raise ProposalError(f"invalid catalog observable id {name!r}")
        _nonempty_text(description, f"catalog description for {name}")
    with _canonical_support_paths(positive_support, negative_support) as paths:
        result = transport(
            proposer_prompt(observable_catalog),
            paths,
            (
                HYBRID_ONLY_RULE_PROPOSAL_SCHEMA
                if not observable_catalog
                else RULE_PROPOSAL_SCHEMA
            ),
            model=model,
            reasoning_effort=reasoning_effort,
            minutes=minutes,
            verbose=verbose,
            executable=executable,
            cloud_policy_cache_snapshot=cloud_policy_cache_snapshot,
        )
        support_presentation = _support_presentation(paths)
    try:
        return parse_rule_proposal(
            result.payload,
            receipt=result.receipt,
            observable_catalog=observable_catalog,
        )
    except ProposalError as exc:
        attempt = RejectedProposalAttempt(
            model_payload=result.payload,
            receipt=result.receipt,
            support_presentation=support_presentation,
            parse_error_type=type(exc).__name__,
            parse_error_reason=str(exc) or "proposal validation failed",
        )
        raise RejectedProposalError(attempt) from exc


HYBRID_OBSERVATION_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "epistemic_status": {
            "type": "string",
            "const": HYBRID_EPISTEMIC_STATUS,
        },
        "disposition": {
            "type": "string",
            "enum": [
                Disposition.PRESENT.value,
                HYBRID_NONMATCH,
                Disposition.INDETERMINATE.value,
                Disposition.ERROR.value,
            ],
        },
        "observed_cue_ids": {
            "type": "array",
            "items": {
                "type": "string",
                "pattern": "^[a-z][a-z0-9_.-]{0,127}$",
            },
            "maxItems": 12,
        },
        "missing_cue_ids": {
            "type": "array",
            "items": {
                "type": "string",
                "pattern": "^[a-z][a-z0-9_.-]{0,127}$",
            },
            "maxItems": 12,
        },
        "missing_cue_reasons": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "cue_id": {
                        "type": "string",
                        "pattern": "^[a-z][a-z0-9_.-]{0,127}$",
                    },
                    "finding": _STRING,
                },
                "required": ["cue_id", "finding"],
            },
            "maxItems": 12,
        },
        "visibility_certificate": {
            "type": ["string", "null"],
            "maxLength": 20_000,
        },
        "reason": {
            "type": ["string", "null"],
            "maxLength": 20_000,
            "description": (
                "For nonmatch, an optional overall model summary archived "
                "inside the empirical nonmatch certificate; it does not "
                "replace cue-keyed findings or visibility. Must be null for "
                "present and non-empty for indeterminate/error."
            ),
        },
        "error_type": {"type": ["string", "null"], "maxLength": 200},
    },
    "required": [
        "epistemic_status",
        "disposition",
        "observed_cue_ids",
        "missing_cue_ids",
        "missing_cue_reasons",
        "visibility_certificate",
        "reason",
        "error_type",
    ],
}


@dataclass(frozen=True)
class HybridObservation:
    proposal_digest: str
    payload: Mapping[str, Any]
    evidence: Evidence[tuple[str, ...]]
    receipt: CodexReceipt

    @property
    def digest(self) -> str:
        return _digest(self.to_dict())

    @property
    def epistemic_status(self) -> str:
        return HYBRID_EPISTEMIC_STATUS

    @property
    def observed_cue_ids(self) -> tuple[str, ...]:
        values = self.payload.get("observed_cue_ids", ())
        return tuple(values) if isinstance(values, (list, tuple)) else ()

    @property
    def missing_cue_ids(self) -> tuple[str, ...]:
        values = self.payload.get("missing_cue_ids", ())
        return tuple(values) if isinstance(values, (list, tuple)) else ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": HYBRID_OBSERVATION_SCHEMA_VERSION,
            "epistemic_status": self.epistemic_status,
            "proposal_digest": self.proposal_digest,
            "payload": dict(self.payload),
            # ``nonmatch`` is deliberately the model-facing outcome.  The
            # internal Boolean IR still represents it with CERTIFIED_ABSENT,
            # but its certificate says only that the archived model judged
            # this frozen operational claim a nonmatch.  It is not a
            # mechanical certificate that an image property is absent.
            "disposition": self.payload.get("disposition"),
            "provenance_digest": self.evidence.provenance.digest(),
            "receipt": self.receipt.to_dict(),
        }


def hybrid_observer_prompt(proposal: RuleProposal) -> str:
    if proposal.hybrid_claim is None:
        raise ProposalError("hybrid observation requires a HYBRID proposal")
    claim = proposal.hybrid_claim
    return f"""Inspect the single neutral image query.png against this frozen,
affirmative visual predicate.  Judge only the image and this definition; do not
infer a Bongard side and do not compare against any other image.

Phrase: {claim.phrase}
Operational definition: {claim.operational_definition}
Required visual cues: {json.dumps([cue.to_dict() for cue in claim.required_visual_cues])}
Chosen view: {proposal.view}
Epistemic status: {HYBRID_EPISTEMIC_STATUS}

This is a conditionally checkable HYBRID empirical observation, not pixel
truth.  Copy cue IDs exactly; never use cue descriptions as IDs and never
invent an ID.  `observed_cue_ids` and `missing_cue_ids` must be disjoint
subsets of the declared IDs.

Return `present` if and only if ALL required cue IDs are visibly supported;
list every required ID in `observed_cue_ids` and leave all absence fields
empty/null.
Return `nonmatch` only when the relevant region is fully visible and visible
image evidence defeats at least one declared required cue for this frozen
operational claim.  This is an archived model judgment, not a mechanically
certified absence of an image property.  List at least one such ID in
`missing_cue_ids`, give exactly one `missing_cue_reasons` object keyed by each
missing ID, and provide `visibility_certificate`.  A cue finding must explain
what visible image evidence defeats that particular positive cue.  `reason`
may be null or one non-empty overall summary of why the image is a nonmatch;
when supplied it is archived as part of the model-nonmatch certificate.  It
never substitutes for cue-keyed findings or the visibility statement and does
not turn the model judgment into a mechanically certified image property.
Return `indeterminate` for ambiguity, occlusion, borderline resemblance, or an
underspecified definition, with both cue ID lists empty; partial cue judgments
are not exported through an indeterminate result.  Return `error` only when the
image itself cannot be inspected, also with empty cue lists.  Set
`epistemic_status` exactly to
`{HYBRID_EPISTEMIC_STATUS}`.  Do not invent a probability or silently treat
uncertainty as false.
"""


def _parse_hybrid_observation(
    proposal: RuleProposal,
    payload: Mapping[str, Any],
    receipt: CodexReceipt,
) -> HybridObservation:
    if proposal.hybrid_claim is None:
        raise ProposalError("hybrid observation requires a HYBRID proposal")
    raw = _strict_keys(
        payload,
        {
            "epistemic_status",
            "disposition",
            "observed_cue_ids",
            "missing_cue_ids",
            "missing_cue_reasons",
            "visibility_certificate",
            "reason",
            "error_type",
        },
        "hybrid observation",
    )
    if raw["epistemic_status"] != HYBRID_EPISTEMIC_STATUS:
        raise ProposalError(
            "hybrid observation epistemic_status does not mark a conditionally "
            "checkable empirical observation"
        )
    wire_disposition = raw["disposition"]
    if wire_disposition == HYBRID_NONMATCH:
        disposition = Disposition.CERTIFIED_ABSENT
    else:
        try:
            disposition = Disposition(wire_disposition)
        except (TypeError, ValueError) as exc:
            raise ProposalError("hybrid observation has an unknown disposition") from exc
    observed = _cue_id_tuple(raw["observed_cue_ids"], "observed_cue_ids")
    missing = _cue_id_tuple(raw["missing_cue_ids"], "missing_cue_ids")
    required_ids = proposal.hybrid_claim.required_cue_ids
    required_set = set(required_ids)
    for field, values in (("observed_cue_ids", observed), ("missing_cue_ids", missing)):
        for cue_id in values:
            if cue_id not in required_set:
                raise ProposalError(f"{field} contains undeclared cue ID {cue_id!r}")
    overlap = set(observed) & set(missing)
    if overlap:
        raise ProposalError(
            f"observed and missing cue IDs must be disjoint: {sorted(overlap)}"
        )

    reasons_raw = raw["missing_cue_reasons"]
    if not isinstance(reasons_raw, list) or len(reasons_raw) > 12:
        raise ProposalError("missing_cue_reasons must be a list of at most 12 items")
    missing_reasons: list[tuple[str, str]] = []
    for index, value in enumerate(reasons_raw):
        item = _strict_keys(
            value,
            {"cue_id", "finding"},
            f"missing_cue_reasons[{index}]",
        )
        cue_id = _exact_cue_id(
            item["cue_id"], f"missing_cue_reasons[{index}].cue_id"
        )
        if cue_id not in required_set:
            raise ProposalError(
                f"missing_cue_reasons contains undeclared cue ID {cue_id!r}"
            )
        finding = _nonempty_text(
            item["finding"], f"missing_cue_reasons[{index}].finding"
        )
        missing_reasons.append((cue_id, finding))
    reason_ids = tuple(cue_id for cue_id, _finding in missing_reasons)
    if len(reason_ids) != len(set(reason_ids)):
        raise ProposalError("missing_cue_reasons contains duplicate cue IDs")

    visibility_certificate = raw["visibility_certificate"]
    reason = raw["reason"]
    error_type = raw["error_type"]
    for name, value in (
        ("visibility_certificate", visibility_certificate),
        ("reason", reason),
        ("error_type", error_type),
    ):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ProposalError(f"{name} must be null or a non-empty string")

    provenance = Provenance(
        producer="headless-codex-hybrid-observer",
        version="3",
        method="frozen-cue-id-single-panel-judgment-with-nonmatch-summary",
        input_digests=(proposal.digest, receipt.input_digest),
        artifact_digest=receipt.receipt_digest,
        run_id=receipt.thread_id,
        details=tuple(
            sorted(
                (
                    ("model", receipt.requested_model),
                    ("proposal_schema", PROPOSAL_SCHEMA_VERSION),
                    ("epistemic_status", HYBRID_EPISTEMIC_STATUS),
                    ("required_cue_ids_digest", _digest(list(required_ids))),
                    ("view", proposal.view),
                )
            )
        ),
    )
    if disposition is Disposition.PRESENT:
        if (
            set(observed) != required_set
            or missing
            or missing_reasons
            or visibility_certificate is not None
            or reason is not None
            or error_type is not None
        ):
            raise ProposalError(
                "present requires all and only the declared cue IDs, with no absence/error fields"
            )
        evidence = Evidence.present(required_ids, provenance)
    elif disposition is Disposition.CERTIFIED_ABSENT:
        if (
            not missing
            or set(reason_ids) != set(missing)
            or not isinstance(visibility_certificate, str)
            or error_type is not None
        ):
            raise ProposalError(
                "nonmatch requires declared missing cue IDs, exactly matched "
                "cue-keyed findings, and a visibility statement"
            )
        ordered_reasons = [
            {"cue_id": cue_id, "finding": dict(missing_reasons)[cue_id]}
            for cue_id in required_ids
            if cue_id in set(missing)
        ]
        certificate = _canonical_json(
            {
                "certificate_semantics": (
                    "archived_model_nonmatch_for_frozen_operational_claim"
                ),
                "epistemic_status": HYBRID_EPISTEMIC_STATUS,
                "missing_cue_reasons": ordered_reasons,
                "reason": reason,
                "visibility_certificate": visibility_certificate,
            }
        )
        evidence = Evidence.certified_absent(provenance, certificate)
    elif disposition is Disposition.INDETERMINATE:
        if (
            observed
            or missing
            or missing_reasons
            or visibility_certificate is not None
            or not isinstance(reason, str)
            or error_type is not None
        ):
            raise ProposalError(
                "indeterminate requires empty observed/missing cue ID and absence "
                "fields, plus a reason"
            )
        evidence = Evidence.indeterminate(provenance, reason)
    else:
        if (
            observed
            or missing
            or missing_reasons
            or visibility_certificate is not None
            or not isinstance(reason, str)
            or not isinstance(error_type, str)
        ):
            raise ProposalError(
                "error observation requires empty cue/absence fields, reason, and error_type"
            )
        evidence = Evidence.error(provenance, error_type, reason)
    return HybridObservation(proposal.digest, dict(raw), evidence, receipt)


def parse_hybrid_observation_or_error(
    proposal: RuleProposal,
    payload: Mapping[str, Any],
    receipt: CodexReceipt,
) -> HybridObservation:
    """Archive a transport-successful but semantically invalid reply as ERROR.

    The strict parser remains available for admission tests.  At the runtime
    boundary, however, discarding the payload and receipt would make an honest
    observer error impossible to audit or replay.  This wrapper preserves the
    exact raw response and turns only the semantic-parser failure into an
    explicit error disposition; it never guesses a negative label.
    """

    try:
        return _parse_hybrid_observation(proposal, payload, receipt)
    except (ProposalError, TypeError, ValueError) as exc:
        provenance = Provenance(
            producer="headless-codex-hybrid-observer",
            version="3",
            method="malformed-frozen-hybrid-observation",
            input_digests=(proposal.digest, receipt.input_digest),
            artifact_digest=receipt.receipt_digest,
            run_id=receipt.thread_id,
            details=tuple(
                sorted(
                    (
                        ("model", receipt.requested_model),
                        ("proposal_schema", PROPOSAL_SCHEMA_VERSION),
                        ("epistemic_status", HYBRID_EPISTEMIC_STATUS),
                        ("view", proposal.view),
                    )
                )
            ),
        )
        return HybridObservation(
            proposal.digest,
            dict(payload),
            Evidence.error(
                provenance,
                type(exc).__name__,
                str(exc) or "hybrid observation failed semantic validation",
            ),
            receipt,
        )


def observe_hybrid_panel(
    proposal: RuleProposal,
    panel: str | Path,
    *,
    model: str | None = None,
    reasoning_effort: str | None = None,
    minutes: int = 10,
    verbose: bool = False,
    executable: str = "codex",
    cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot | None = None,
    transport: StructuredTransport = run_codex_named_images_structured,
) -> HybridObservation:
    """Measure one already-released query against an exact frozen claim."""

    if proposal.hybrid_claim is None:
        raise ProposalError("registered-observable proposals need the PURE observer")
    path = Path(panel).resolve()
    if not path.is_file():
        raise ProposalError(f"query panel is not a file: {path}")
    result = transport(
        hybrid_observer_prompt(proposal),
        (str(path),),
        ("query.png",),
        HYBRID_OBSERVATION_SCHEMA,
        model=model or proposal.receipt.requested_model,
        reasoning_effort=reasoning_effort
        or proposal.receipt.requested_reasoning_effort,
        minutes=minutes,
        verbose=verbose,
        executable=executable,
        cloud_policy_cache_snapshot=cloud_policy_cache_snapshot,
    )
    return parse_hybrid_observation_or_error(
        proposal, result.payload, result.receipt
    )


class HeadlessCodexEpisode:
    """One stateful adapter implementing both benchmark callback boundaries.

    A fresh instance is used for one episode.  It permits one support-only
    proposal and then independent single-panel observations.  Raw proposer and
    observer receipts remain available through :meth:`artifact_data`; the
    benchmark freeze independently binds the proposal's canonical digest.
    """

    requires_empirical_support_gate = True

    def __init__(
        self,
        *,
        observable_catalog: Mapping[str, str] | None = None,
        model: str = DEFAULT_CODEX_MODEL,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        proposer_minutes: int = 15,
        observer_minutes: int = 10,
        verbose: bool = False,
        executable: str = "codex",
        proposer_transport: StructuredTransport = run_codex_structured,
        observer_transport: StructuredTransport = run_codex_named_images_structured,
        cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot | None = None,
    ) -> None:
        self.observable_catalog = dict(observable_catalog or {})
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.proposer_minutes = proposer_minutes
        self.observer_minutes = observer_minutes
        self.verbose = verbose
        self.executable = executable
        self.proposer_transport = proposer_transport
        self.observer_transport = observer_transport
        if cloud_policy_cache_snapshot is not None and not isinstance(
            cloud_policy_cache_snapshot, CloudPolicyCacheSnapshot
        ):
            raise ProposalError("episode cloud policy cache snapshot is invalid")
        self._cloud_policy_cache_snapshot = cloud_policy_cache_snapshot
        self._proposal_attempted = False
        self._proposal: RuleProposal | None = None
        self._rejected_proposal_attempt: RejectedProposalAttempt | None = None
        self._compiled: Any | None = None
        self._observations: dict[str, HybridObservation] = {}
        self._transport_identity: tuple[tuple[str, str], ...] | None = None

    @property
    def proposal(self) -> RuleProposal | None:
        return self._proposal

    def propose(self, support: Any) -> Any:
        """Implement :class:`bongard.benchmark.Proposer` without task metadata."""

        if self._proposal_attempted:
            raise ProposalError("one episode adapter permits exactly one proposer call")
        try:
            positive_paths = support.positive_paths
            negative_paths = support.negative_paths
        except AttributeError as exc:
            raise ProposalError("benchmark support input is malformed") from exc
        self._proposal_attempted = True
        try:
            proposal = propose_rule(
                positive_paths,
                negative_paths,
                observable_catalog=self.observable_catalog,
                model=self.model,
                reasoning_effort=self.reasoning_effort,
                minutes=self.proposer_minutes,
                verbose=self.verbose,
                executable=self.executable,
                cloud_policy_cache_snapshot=self._episode_policy_cache_snapshot(),
                transport=self.proposer_transport,
            )
        except RejectedProposalError as exc:
            self._rejected_proposal_attempt = exc.attempt
            raise
        if not proposal.is_hybrid:
            raise ProposalError(
                "the current episode adapter supports HYBRID claims; PURE "
                "catalog proposals require a verifier-owned catalog compiler"
            )
        from bongard.benchmark import ProposedRule
        from bongard.synthesis import compile_hybrid_proposal

        compiled = compile_hybrid_proposal(proposal)
        self._proposal = proposal
        self._compiled = compiled
        self._transport_identity = codex_transport_identity(proposal.receipt)
        return ProposedRule(
            proposal_id="proposal-" + compiled.proposer_digest[:16],
            proposer_digest=compiled.proposer_digest,
            formula=compiled.formula,
            registry=compiled.registry,
            attachment_contract=compiled.attachment_contract,
        )

    def _episode_policy_cache_snapshot(self) -> CloudPolicyCacheSnapshot:
        """Capture once, then reuse exact policy bytes for every episode call."""

        if self._cloud_policy_cache_snapshot is None:
            self._cloud_policy_cache_snapshot = snapshot_cloud_policy_cache()
        return self._cloud_policy_cache_snapshot

    def create_support_observer(self) -> "HeadlessCodexEpisode":
        """Return a fresh support observer with only the frozen proposal state."""

        if self._proposal is None or self._compiled is None:
            raise ProposalError("support replay cannot precede the fixed proposal")
        isolated = copy.deepcopy(self)
        if isolated is self:
            raise ProposalError("support observer copy retained object identity")
        isolated._observations = {}
        return isolated

    def _check_transport_identity(self, receipt: CodexReceipt) -> None:
        if self._transport_identity is None:
            raise ProposalError("proposal transport identity is unavailable")
        actual = codex_transport_identity(receipt)
        if actual != self._transport_identity:
            expected_map = dict(self._transport_identity)
            actual_map = dict(actual)
            changed = [
                name
                for name in TRANSPORT_IDENTITY_FIELDS
                if expected_map[name] != actual_map[name]
            ]
            raise TransportIdentityError(
                "observer transport identity differs from proposer: "
                + ", ".join(changed)
            )

    def observe_support(self, panel: Any) -> Any:
        """Measure one neutral support image without receiving its side label."""

        if self._proposal is None or self._compiled is None:
            raise ProposalError("support replay cannot precede the fixed proposal")
        if panel.query_id != "query" or panel.panel.blob_id != "query-panel":
            raise ProposalError("support replay input is not neutral")
        if panel.panel_path.name != "query.png":
            raise ProposalError("support replay image must be named query.png")
        observation = observe_hybrid_panel(
            self._proposal,
            panel.panel_path,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            minutes=self.observer_minutes,
            verbose=self.verbose,
            executable=self.executable,
            cloud_policy_cache_snapshot=self._episode_policy_cache_snapshot(),
            transport=self.observer_transport,
        )
        from bongard.benchmark import SupportGateMeasurement
        from bongard.synthesis import truth_from_hybrid_observation

        try:
            self._check_transport_identity(observation.receipt)
        except ProposalError as exc:
            provenance = Provenance(
                producer="headless-codex-support-gate",
                version="1",
                method="transport-identity-mismatch",
                input_digests=(self._proposal.digest, observation.digest),
                artifact_digest=observation.receipt.receipt_digest,
                run_id=observation.receipt.thread_id,
            )
            return SupportGateMeasurement(
                evidence=Evidence.error(provenance, type(exc).__name__, str(exc)),
                observer_artifact={
                    "schema": "support-transport-identity-error/v1",
                    "raw_observation": observation.to_dict(),
                    "reason": str(exc),
                },
            )

        return SupportGateMeasurement(
            evidence=truth_from_hybrid_observation(observation),
            observer_artifact=observation.to_dict(),
        )

    def observe(self, query: Any) -> Mapping[tuple[int, ...], Evidence[bool]]:
        """Implement one isolated post-freeze benchmark observation."""

        if self._proposal is None or self._compiled is None:
            raise ProposalError("query observation cannot precede proposal freeze")
        if query.query_id in self._observations:
            raise ProposalError(f"query {query.query_id!r} was already observed")
        if query.freeze.proposer_digest != self._compiled.proposer_digest:
            raise ProposalError("query freeze belongs to a different proposal")
        if query.registry.digest() != self._compiled.registry.digest():
            raise ProposalError("query registry differs from the compiled proposal")
        observation = observe_hybrid_panel(
            self._proposal,
            query.panel_path,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            minutes=self.observer_minutes,
            verbose=self.verbose,
            executable=self.executable,
            cloud_policy_cache_snapshot=self._episode_policy_cache_snapshot(),
            transport=self.observer_transport,
        )
        # Preserve the exact raw response even when the transport identity is
        # rejected below.  The runner will score the raised protocol failure
        # as ERROR, while the post-hoc artifact still exposes the substituted
        # receipt instead of silently discarding it.
        self._observations[query.query_id] = observation
        self._check_transport_identity(observation.receipt)
        from bongard.artifacts import atom_paths
        from bongard.synthesis import truth_from_hybrid_observation

        paths = atom_paths(query.freeze.formula)
        if paths != ((),):
            raise ProposalError("HYBRID episode formula must contain exactly one atom")
        return {(): truth_from_hybrid_observation(observation)}

    def artifact_data(self) -> dict[str, Any]:
        """Return the complete vision-side evidence available so far."""

        return {
            "schema": HEADLESS_EPISODE_SCHEMA_VERSION,
            "proposal": self._proposal.to_dict() if self._proposal else None,
            "rejected_proposal_attempt": (
                self._rejected_proposal_attempt.to_dict()
                if self._rejected_proposal_attempt is not None
                else None
            ),
            "observations": {
                query_id: observation.to_dict()
                for query_id, observation in sorted(self._observations.items())
            },
        }

    @property
    def artifact_digest(self) -> str:
        return _digest(self.artifact_data())


__all__ = [
    "CONFIDENCE_LEVELS",
    "HYBRID_EPISTEMIC_STATUS",
    "HEADLESS_EPISODE_SCHEMA_VERSION",
    "HYBRID_ONLY_RULE_PROPOSAL_SCHEMA",
    "HYBRID_OBSERVATION_SCHEMA",
    "HYBRID_OBSERVATION_SCHEMA_VERSION",
    "HybridClaim",
    "HybridCue",
    "HeadlessCodexEpisode",
    "HybridObservation",
    "ObservableRequest",
    "PROPOSAL_SCHEMA_VERSION",
    "ProposalError",
    "REJECTED_PROPOSAL_ATTEMPT_SCHEMA_VERSION",
    "RejectedProposalAttempt",
    "RejectedProposalError",
    "TransportIdentityError",
    "RULE_PROPOSAL_SCHEMA",
    "RuleProposal",
    "SupportPanelIdentity",
    "VIEWS",
    "TRANSPORT_IDENTITY_FIELDS",
    "codex_transport_identity",
    "hybrid_observer_prompt",
    "observe_hybrid_panel",
    "parse_rule_proposal",
    "parse_hybrid_observation_or_error",
    "propose_rule",
    "proposer_prompt",
]
