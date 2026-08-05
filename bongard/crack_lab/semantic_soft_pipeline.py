"""Prose-grounded soft Bongard predicates with a blind vision boundary.

The labelled support proposer may use joint vision to name a human concept.
It must commit a side-free, content-addressed rubric before any target image is
scored.  A fresh, stateless scorer then sees exactly one neutrally named image
and that frozen rubric; it never receives labels, neighbouring panels, problem
IDs, filenames, or score feedback.  The harness composes atomic cue scores and
fits/evaluates only the declared polarity.

This is an explicit ``SEMANTIC-SOFT`` track.  Its VLM observations are
operational semantic evidence, not the deterministic typed proof claimed by
``SEMANTIC-PURE``.  Downstream selection is exactly replayable from recorded
evidence; live VLM re-query stability remains a separately measured risk.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Protocol, Sequence

import codex_proposer
from soft_semantics import (
    SoftAbsent,
    SoftError,
    SoftEvidence,
    SoftResult,
    canonical_json_bytes,
    content_digest,
)


SOFT_PREDICATE_SCHEMA = "bongard.prose-soft-predicate/v1"
SOFT_PROPOSAL_SCHEMA = "bongard.prose-soft-proposal/v1"
SOFT_EVIDENCE_SCHEMA = "bongard.prose-soft-evidence/v1"
SOFT_VERIFICATION_SCHEMA = "bongard.prose-soft-verification/v1"
SOFT_TRACK = "SEMANTIC-SOFT"
MAX_CUES = 12
MAX_DISQUALIFIERS = 8
MAX_TEXT = 1200
MAX_ACCEPTED_UNCERTAINTY = 0.5

_IDENTIFIER_RE = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")
_SIDE_LANGUAGE_RE = re.compile(
    r"(?:\bpositives?\b|\bnegatives?\b|\b(?:left|right)\s+(?:side|column)\b|"
    r"\b(?:pos|neg)_[0-9]+\b|"
    r"\b(?:first|last)\s+(?:six|6|examples?|panels?|images?|group|half)\b|"
    r"\b(?:first|second|former|latter)\s+(?:group|set|half|batch)\b|"
    r"\bpresented\s+(?:first|last)\b)",
    re.IGNORECASE,
)
_FORBIDDEN_TEMPLATE_RE = re.compile(
    r"(?:sha-?256|pixel\s+hash|exact\s+coordinates?|filename|panel_[0-9]+)",
    re.IGNORECASE,
)
_SIDE_IDENTIFIER_RE = re.compile(
    r"(?:^|[-_])(?:pos|neg|positive|negative)(?:[-_]|$)")
_MORPHISMS = frozenset({
    "translation", "rotation", "reflection", "uniform_scale",
    "stroke_width",
})


def _text(value: Any, name: str, *, side_free: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be text")
    normalized = " ".join(value.split())
    if not normalized or len(normalized) > MAX_TEXT:
        raise ValueError(f"{name} must contain 1..{MAX_TEXT} normalized characters")
    if side_free and (_SIDE_LANGUAGE_RE.search(normalized)
                      or _FORBIDDEN_TEMPLATE_RE.search(normalized)):
        raise ValueError(
            f"{name} must be side-free and may not encode panel identities")
    return normalized


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase identifier")
    return value


def _unit(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return number


@dataclass(frozen=True)
class SoftCueSpec:
    cue_id: str
    description: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "cue_id", _identifier(self.cue_id, "cue_id"))
        if _SIDE_IDENTIFIER_RE.search(self.cue_id):
            raise ValueError("cue_id must not encode a Bongard side")
        object.__setattr__(
            self, "description",
            _text(self.description, f"cue {self.cue_id}", side_free=True),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SoftCueSpec":
        if not isinstance(value, Mapping) or set(value) != {
                "cue_id", "description"}:
            raise ValueError("soft cue fields must be cue_id and description")
        return cls(str(value["cue_id"]), str(value["description"]))


@dataclass(frozen=True)
class SoftPredicateSpec:
    hypothesis_id: str
    claim: str
    operational_definition: str
    order: str
    comparison: str
    aggregation: str
    required_cues: tuple[SoftCueSpec, ...]
    disqualifiers: tuple[SoftCueSpec, ...]
    preservation_morphisms: tuple[str, ...]
    version: str = SOFT_PREDICATE_SCHEMA

    def __post_init__(self) -> None:
        if self.version != SOFT_PREDICATE_SCHEMA:
            raise ValueError("unsupported soft predicate schema")
        object.__setattr__(
            self, "hypothesis_id",
            _identifier(self.hypothesis_id, "hypothesis_id"),
        )
        object.__setattr__(
            self, "claim", _text(self.claim, "claim", side_free=True))
        object.__setattr__(
            self, "operational_definition",
            _text(self.operational_definition, "operational_definition",
                  side_free=True),
        )
        # A scorer's value is membership in the literal affirmative claim.
        # Consequently every admissible rule is high-positive.  Concepts such
        # as "few objects" or "absence of curves" must be stated and scored
        # directly; a selector may never rescue a bad concept by negating it.
        if self.order != "high_positive":
            raise ValueError(
                "order must be high_positive; phrase the affirmative property "
                "itself instead of negating a score")
        if self.comparison not in {"absolute", "relative"}:
            raise ValueError("comparison must be absolute or relative")
        if self.aggregation not in {"all", "any", "mean"}:
            raise ValueError("aggregation must be all, any, or mean")
        if not 1 <= len(self.required_cues) <= MAX_CUES:
            raise ValueError(f"required_cues must contain 1..{MAX_CUES} cues")
        if len(self.disqualifiers) > MAX_DISQUALIFIERS:
            raise ValueError(
                f"disqualifiers may contain at most {MAX_DISQUALIFIERS} cues")
        cue_ids = tuple(
            cue.cue_id for cue in self.required_cues + self.disqualifiers)
        if len(cue_ids) != len(set(cue_ids)):
            raise ValueError("soft cue IDs must be unique")
        if not self.preservation_morphisms \
                or len(self.preservation_morphisms) != \
                len(set(self.preservation_morphisms)) \
                or set(self.preservation_morphisms) - _MORPHISMS:
            raise ValueError(
                "preservation_morphisms must be a nonempty unique supported set")

    @property
    def cue_ids(self) -> tuple[str, ...]:
        return tuple(
            cue.cue_id for cue in self.required_cues + self.disqualifiers)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def scoring_rubric(self) -> dict[str, Any]:
        """The only spec surface visible to the label-blind scorer.

        Selector direction, comparison policy, and hypothesis identity are
        intentionally excluded.  They remain committed by ``digest()`` but
        cannot tell the visual scorer which way the Bongard labels run.
        """
        return {
            "schema": SOFT_PREDICATE_SCHEMA,
            "claim": self.claim,
            "operational_definition": self.operational_definition,
            "aggregation": self.aggregation,
            "required_cues": [asdict(item) for item in self.required_cues],
            "disqualifiers": [asdict(item) for item in self.disqualifiers],
            "preservation_morphisms": list(self.preservation_morphisms),
        }

    def scoring_rubric_digest(self) -> str:
        """Digest only the side-blind surface actually shown to the scorer."""
        return content_digest(self.scoring_rubric())

    def digest(self) -> str:
        return content_digest(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SoftPredicateSpec":
        required = {
            "hypothesis_id", "claim", "operational_definition", "order",
            "comparison", "aggregation", "required_cues", "disqualifiers",
            "preservation_morphisms",
        }
        if not isinstance(value, Mapping) or set(value) not in (
                required, required | {"version"}):
            raise ValueError("soft predicate fields differ from the contract")
        required_cues = value["required_cues"]
        disqualifiers = value["disqualifiers"]
        morphisms = value["preservation_morphisms"]
        if not isinstance(required_cues, list) \
                or not isinstance(disqualifiers, list) \
                or not isinstance(morphisms, list):
            raise ValueError("soft cue/morphism fields must be arrays")
        return cls(
            hypothesis_id=str(value["hypothesis_id"]),
            claim=str(value["claim"]),
            operational_definition=str(value["operational_definition"]),
            order=str(value["order"]),
            comparison=str(value["comparison"]),
            aggregation=str(value["aggregation"]),
            required_cues=tuple(
                SoftCueSpec.from_dict(item) for item in required_cues),
            disqualifiers=tuple(
                SoftCueSpec.from_dict(item) for item in disqualifiers),
            preservation_morphisms=tuple(str(item) for item in morphisms),
            version=str(value.get("version", SOFT_PREDICATE_SCHEMA)),
        )


_CUE_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {
            "type": "string", "minLength": 1, "maxLength": MAX_TEXT,
        },
    },
    "required": ["description"],
    "additionalProperties": False,
}

SOFT_HYPOTHESES_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "analysis": {
            "type": "string", "minLength": 1, "maxLength": MAX_TEXT,
        },
        "hypotheses": {
            "type": "array",
            "minItems": 3,
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "claim": {
                        "type": "string", "minLength": 1,
                        "maxLength": MAX_TEXT,
                    },
                    "operational_definition": {
                        "type": "string", "minLength": 1,
                        "maxLength": MAX_TEXT,
                    },
                    "order": {
                        "type": "string",
                        "enum": ["high_positive"],
                    },
                    "comparison": {
                        "type": "string", "enum": ["absolute", "relative"],
                    },
                    "aggregation": {
                        "type": "string", "enum": ["all", "any", "mean"],
                    },
                    "required_cues": {
                        "type": "array", "minItems": 1,
                        "maxItems": MAX_CUES, "items": _CUE_SCHEMA,
                    },
                    "disqualifiers": {
                        "type": "array", "maxItems": MAX_DISQUALIFIERS,
                        "items": _CUE_SCHEMA,
                    },
                    "preservation_morphisms": {
                        "type": "array", "minItems": 1,
                        "items": {"type": "string", "enum": sorted(_MORPHISMS)},
                    },
                },
                "required": [
                    "claim", "operational_definition", "order",
                    "comparison", "aggregation", "required_cues",
                    "disqualifiers", "preservation_morphisms",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["analysis", "hypotheses"],
    "additionalProperties": False,
}


def build_soft_proposal_prompt(problem_id: str) -> str:
    _identifier(problem_id.replace("problem_", "problem-"), "problem_id")
    return """\
You see twelve Bongard panels: six labelled positive, then six labelled
negative. Use joint vision to propose 3-8 short human semantic concepts. Do
not write pixel code.

First perform a nearest-foil audit: find the negative panel that most closely
matches the positive panels' coarse object inventory and silhouette, then
isolate the smallest local difference that still separates it. If topology is
shared, inspect intrinsic attachment turns, boundary tangents, interior
angles, arc sweep, length/aspect ratios, curvature placement, and which
corners or endpoints meet. At least one candidate must operationalize that
local contrast with a bounded angle/range/ratio when the distinction is
quantitative. Do not submit 3-8 paraphrases of the same coarse silhouette.

For each concept, commit an operational, SIDE-FREE scoring rubric. The claim,
definition, cue descriptions, and disqualifiers must never mention positives,
negatives, panel indices, sides, filenames, hashes, exact coordinates, or
templates. They will be sent verbatim to a later scorer that has no labels.

Each required cue is one independently observable reason the concept applies.
Disqualifiers are independently observable reasons it does not. Choose a
fixed aggregation: `all` is non-compensating min, `any` is max, and `mean` is
the unweighted mean. A disqualifier must be a distinct visual veto, not merely
the grammatical negation of a required cue. No learned weights are allowed.
Use `absolute` for a
categorical claim whose fixed decision rule is strictly greater than 0.5;
the exactly-0.5 ambiguous state is never positive. Use `relative` only when
the literal claim is comparative. Membership must always be HIGHER for
the literal affirmative property. If the positive concept is "few objects"
or "absence of curves", score sparseness or curve-absence directly; never ask
the selector to negate a score. Declare only genuine nuisance-preserving
transformations.

Good examples include a decomposed `bird-like silhouette` rubric or an
intrinsic `oblique angle` rubric. Prefer typed geometric concepts when the
available visual evidence permits them, but open-vocabulary shape resemblance
is allowed here and will be reported as SEMANTIC-SOFT, never SEMANTIC-PURE.

The complete structured response is the scientific commitment. Later panel
scores cannot modify it. The harness assigns opaque hypothesis and cue IDs;
do not encode names or indices for them in prose."""


@dataclass(frozen=True)
class SoftProposalBundle:
    problem_id: str
    hypotheses: tuple[SoftPredicateSpec, ...]
    analysis: str
    receipt: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SOFT_PROPOSAL_SCHEMA,
            "problem_id": self.problem_id,
            "analysis": self.analysis,
            "hypotheses": [item.to_dict() for item in self.hypotheses],
            "receipt": dict(self.receipt),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SoftProposalBundle":
        if not isinstance(value, Mapping) or set(value) != {
                "schema", "problem_id", "analysis", "hypotheses", "receipt"}:
            raise ValueError("soft proposal artifact fields differ")
        if value["schema"] != SOFT_PROPOSAL_SCHEMA \
                or not isinstance(value["hypotheses"], list) \
                or not 3 <= len(value["hypotheses"]) <= 8 \
                or not isinstance(value["receipt"], Mapping):
            raise ValueError("soft proposal artifact violates its schema")
        problem_id = _text(value["problem_id"], "proposal problem_id")
        hypotheses = tuple(
            SoftPredicateSpec.from_dict(item) for item in value["hypotheses"])
        if len({item.hypothesis_id for item in hypotheses}) != len(hypotheses):
            raise ValueError("soft proposal artifact repeats a hypothesis ID")
        receipt = dict(value["receipt"])
        if not isinstance(receipt.get("receipt_digest"), str) \
                or not receipt["receipt_digest"]:
            raise ValueError("soft proposal artifact has no receipt digest")
        if receipt.get("schema") == codex_proposer.CODEX_RECEIPT_SCHEMA:
            codex_proposer.validate_codex_receipt(receipt)
        return cls(
            problem_id=problem_id,
            hypotheses=hypotheses,
            analysis=_text(value["analysis"], "proposal analysis"),
            receipt=receipt,
        )


class CodexSoftHypothesisProposer:
    """One labelled multimodal turn that freezes prose rubrics."""

    def __init__(
            self, model: str = codex_proposer.DEFAULT_CODEX_MODEL,
            *, minutes: int = 15,
            reasoning_effort: str = codex_proposer.DEFAULT_REASONING_EFFORT,
            executable: str = "codex") -> None:
        self.model = model
        self.minutes = minutes
        self.reasoning_effort = reasoning_effort
        self.executable = executable

    def propose(
            self, problem_id: str,
            panel_png_paths: Sequence[str]) -> SoftProposalBundle:
        result = codex_proposer.run_codex_structured(
            build_soft_proposal_prompt(problem_id),
            panel_png_paths,
            SOFT_HYPOTHESES_OUTPUT_SCHEMA,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            minutes=self.minutes,
            executable=self.executable,
        )
        payload = result.payload
        if set(payload) != {"analysis", "hypotheses"} \
                or not isinstance(payload["analysis"], str) \
                or not isinstance(payload["hypotheses"], list) \
                or not 3 <= len(payload["hypotheses"]) <= 8:
            raise ValueError("Codex soft proposal violates its output contract")
        hypotheses_list: list[SoftPredicateSpec] = []
        expected_fields = {
            "claim", "operational_definition", "order", "comparison",
            "aggregation", "required_cues", "disqualifiers",
            "preservation_morphisms",
        }
        for hypothesis_index, raw_item in enumerate(payload["hypotheses"]):
            if not isinstance(raw_item, Mapping) \
                    or set(raw_item) != expected_fields \
                    or not isinstance(raw_item["required_cues"], list) \
                    or not isinstance(raw_item["disqualifiers"], list):
                raise ValueError(
                    "Codex soft proposal fields differ from its output contract")

            def assign_cues(items: Sequence[Any], prefix: str) \
                    -> list[dict[str, str]]:
                output = []
                for cue_index, item in enumerate(items):
                    if not isinstance(item, Mapping) \
                            or set(item) != {"description"}:
                        raise ValueError(
                            "Codex soft cue fields differ from its output contract")
                    output.append({
                        "cue_id": f"{prefix}-{cue_index:02d}",
                        "description": item["description"],
                    })
                return output

            normalized = dict(raw_item)
            normalized["hypothesis_id"] = f"hypothesis-{hypothesis_index:02d}"
            normalized["required_cues"] = assign_cues(
                raw_item["required_cues"], "required")
            normalized["disqualifiers"] = assign_cues(
                raw_item["disqualifiers"], "veto")
            hypotheses_list.append(SoftPredicateSpec.from_dict(normalized))
        hypotheses = tuple(hypotheses_list)
        return SoftProposalBundle(
            problem_id=problem_id,
            hypotheses=hypotheses,
            analysis=_text(payload["analysis"], "analysis"),
            receipt=result.receipt.to_dict(),
        )


def _evidence_output_schema(spec: SoftPredicateSpec) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "atomic_scores": {
                "type": "array",
                "minItems": len(spec.cue_ids),
                "maxItems": len(spec.cue_ids),
                "items": {
                    "type": "object",
                    "properties": {
                        "cue_id": {"type": "string", "enum": list(spec.cue_ids)},
                        "score": {"type": "number", "minimum": 0, "maximum": 1},
                        "evidence": {
                            "type": "string", "minLength": 1,
                            "maxLength": MAX_TEXT,
                        },
                    },
                    "required": ["cue_id", "score", "evidence"],
                    "additionalProperties": False,
                },
            },
            "uncertainty": {
                "type": "number", "minimum": 0, "maximum": 1,
            },
            "abstain": {"type": "boolean"},
            "abstention_reason": {
                "type": "string", "maxLength": MAX_TEXT,
            },
        },
        "required": [
            "atomic_scores", "uncertainty", "abstain", "abstention_reason"],
        "additionalProperties": False,
    }


def build_blind_score_prompt(spec: SoftPredicateSpec) -> str:
    rubric = json.dumps(
        spec.scoring_rubric(), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False)
    return f"""\
Evaluate exactly one neutrally named image against the frozen rubric below.
You have no Bongard label, side, neighbouring panel, problem ID, or previous
score. Do not infer one. Inspect the image directly.

Return one independent score in [0,1] for every required cue and disqualifier:
0 means no visible evidence, 0.5 means genuinely ambiguous/partial evidence,
and 1 means strong visible evidence. Evidence text must identify what is
visually present without inventing labels or comparing with unseen images.
Set abstain=true only when the image cannot be assessed; uncertainty above
{MAX_ACCEPTED_UNCERTAINTY:.1f} must also abstain. Do not calculate an overall
score—the harness applies the frozen aggregation and polarity mechanically.
Exactly 0.5 remains ambiguous and does not satisfy an absolute predicate.

FROZEN_RUBRIC_DIGEST: {spec.scoring_rubric_digest()}
FROZEN_RUBRIC_JSON: {rubric}"""


@dataclass(frozen=True)
class PanelSoftScore:
    spec_digest: str
    result: SoftResult
    cue_scores: tuple[tuple[str, float], ...]
    cue_evidence: tuple[tuple[str, str], ...]
    uncertainty: float
    receipt: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SOFT_EVIDENCE_SCHEMA,
            "spec_digest": self.spec_digest,
            "result": self.result.to_dict(),
            "cue_scores": [list(item) for item in self.cue_scores],
            "cue_evidence": [list(item) for item in self.cue_evidence],
            "uncertainty": self.uncertainty,
            "receipt": dict(self.receipt),
        }


class BlindSoftScorer(Protocol):
    def score(self, spec: SoftPredicateSpec, panel_png_path: str) \
            -> PanelSoftScore:
        ...


class BlindSoftBatchScorer(Protocol):
    def score_many(
            self, specs: Sequence[SoftPredicateSpec], panel_png_path: str
            ) -> tuple[PanelSoftScore, ...]:
        ...


class CodexBlindSoftScorer:
    """Stateless single-panel scorer; every call is a fresh ephemeral turn."""

    def __init__(
            self, model: str = codex_proposer.DEFAULT_CODEX_MODEL,
            *, minutes: int = 15,
            reasoning_effort: str = codex_proposer.DEFAULT_REASONING_EFFORT,
            executable: str = "codex") -> None:
        self.model = model
        self.minutes = minutes
        self.reasoning_effort = reasoning_effort
        self.executable = executable
        self.producer_digest = content_digest({
            "schema": SOFT_EVIDENCE_SCHEMA,
            "transport": codex_proposer.CODEX_ISOLATION_POLICY,
            "input_schema": codex_proposer.NAMED_IMAGE_INPUT_DIGEST_SCHEMA,
            "model": model,
            "reasoning_effort": reasoning_effort,
            "aggregation": "harness-owned-unweighted/v1",
        })

    def score(self, spec: SoftPredicateSpec,
              panel_png_path: str) -> PanelSoftScore:
        result = codex_proposer.run_codex_named_images_structured(
            build_blind_score_prompt(spec),
            [panel_png_path],
            ["panel.png"],
            _evidence_output_schema(spec),
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            minutes=self.minutes,
            executable=self.executable,
        )
        return panel_soft_score_from_payload(
            spec, result.payload, result.receipt.to_dict(),
            producer_digest=self.producer_digest)


class CodexBlindSoftBatchScorer:
    """Score every frozen rubric in one stateless call per neutral panel.

    Batching concepts does not weaken the information firewall: the turn still
    receives exactly one image and no labels, other panels, selector direction,
    or score feedback.  It cuts a 3-8 hypothesis problem from 36-96 model calls
    to exactly twelve.
    """

    def __init__(
            self, model: str = codex_proposer.DEFAULT_CODEX_MODEL,
            *, minutes: int = 15,
            reasoning_effort: str = codex_proposer.DEFAULT_REASONING_EFFORT,
            executable: str = "codex") -> None:
        self.model = model
        self.minutes = minutes
        self.reasoning_effort = reasoning_effort
        self.executable = executable

    def score_many(
            self, specs: Sequence[SoftPredicateSpec],
            panel_png_path: str) -> tuple[PanelSoftScore, ...]:
        specs = tuple(specs)
        if not 1 <= len(specs) <= 8 \
                or len({spec.digest() for spec in specs}) != len(specs):
            raise ValueError("blind batch requires 1..8 distinct frozen rubrics")
        aliases = tuple(f"rubric_{index:02d}" for index in range(len(specs)))
        visible_cue_ids = sorted({
            cue_id for spec in specs for cue_id in spec.cue_ids})
        visible = [
            {"rubric_id": alias, "rubric": spec.scoring_rubric(),
             "rubric_digest": spec.scoring_rubric_digest()}
            for alias, spec in zip(aliases, specs)
        ]
        prompt = f"""\
Evaluate exactly one neutrally named image against each frozen rubric below.
You have no Bongard label, side, neighbouring panel, problem ID, selector
direction, or previous score. Do not infer one. For every rubric return every
atomic cue exactly once. Scores use 0=no visible evidence, 0.5=ambiguous or
partial, 1=strong visible evidence. Set abstain=true when the image cannot be
assessed, including uncertainty above {MAX_ACCEPTED_UNCERTAINTY:.1f}. Do not
calculate overall scores; the harness owns composition.

FROZEN_RUBRICS_JSON: {json.dumps(visible, sort_keys=True, separators=(",", ":"))}"""
        schema = {
            "type": "object",
            "properties": {
                "evaluations": {
                    "type": "array", "minItems": len(specs),
                    "maxItems": len(specs),
                    "items": {
                        "type": "object",
                        "properties": {
                            "rubric_id": {
                                "type": "string", "enum": list(aliases)},
                            "atomic_scores": {
                                "type": "array", "minItems": 1,
                                "maxItems": MAX_CUES + MAX_DISQUALIFIERS,
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "cue_id": {
                                            "type": "string",
                                            "enum": visible_cue_ids,
                                        },
                                        "score": {"type": "number",
                                                  "minimum": 0, "maximum": 1},
                                        "evidence": {
                                            "type": "string", "minLength": 1,
                                            "maxLength": MAX_TEXT,
                                        },
                                    },
                                    "required": ["cue_id", "score", "evidence"],
                                    "additionalProperties": False,
                                },
                            },
                            "uncertainty": {"type": "number",
                                            "minimum": 0, "maximum": 1},
                            "abstain": {"type": "boolean"},
                            "abstention_reason": {
                                "type": "string", "maxLength": MAX_TEXT,
                            },
                        },
                        "required": [
                            "rubric_id", "atomic_scores", "uncertainty",
                            "abstain", "abstention_reason"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["evaluations"],
            "additionalProperties": False,
        }
        result = codex_proposer.run_codex_named_images_structured(
            prompt, [panel_png_path], ["panel.png"], schema,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            minutes=self.minutes,
            executable=self.executable,
        )
        payload = result.payload
        evaluations = payload.get("evaluations") \
            if isinstance(payload, Mapping) else None
        if not isinstance(evaluations, list) or len(evaluations) != len(specs):
            raise ValueError("blind batch omitted a frozen rubric")
        by_alias: dict[str, Mapping[str, Any]] = {}
        for evaluation in evaluations:
            if not isinstance(evaluation, Mapping):
                raise ValueError("blind batch evaluation is not an object")
            alias = evaluation.get("rubric_id")
            if not isinstance(alias, str) or alias in by_alias:
                raise ValueError("blind batch repeats or mangles a rubric ID")
            by_alias[alias] = evaluation
        if set(by_alias) != set(aliases):
            raise ValueError("blind batch rubric identities differ")
        receipt = result.receipt.to_dict()
        output: list[PanelSoftScore] = []
        for alias, spec in zip(aliases, specs):
            evaluation = dict(by_alias[alias])
            evaluation.pop("rubric_id", None)
            producer_digest = content_digest({
                "schema": SOFT_EVIDENCE_SCHEMA,
                "transport": codex_proposer.CODEX_ISOLATION_POLICY,
                "input_schema": codex_proposer.NAMED_IMAGE_INPUT_DIGEST_SCHEMA,
                "model": self.model,
                "reasoning_effort": self.reasoning_effort,
                "rubric_digest": spec.digest(),
                "batch_policy": "one-panel-many-frozen-rubrics/v1",
            })
            output.append(panel_soft_score_from_payload(
                spec, evaluation, receipt,
                producer_digest=producer_digest))
        return tuple(output)


def _compose_membership(
        spec: SoftPredicateSpec,
        cue_scores: Mapping[str, float]) -> float:
    required = [cue_scores[cue.cue_id] for cue in spec.required_cues]
    if spec.aggregation == "all":
        base = min(required)
    elif spec.aggregation == "any":
        base = max(required)
    else:
        base = math.fsum(required) / len(required)
    disqualifier = max(
        (cue_scores[cue.cue_id] for cue in spec.disqualifiers),
        default=0.0,
    )
    return min(base, 1.0 - disqualifier)


def panel_soft_score_from_payload(
        spec: SoftPredicateSpec, payload: Mapping[str, Any],
        receipt: Mapping[str, Any], *, producer_digest: str,
        ) -> PanelSoftScore:
    if not isinstance(payload, Mapping) or set(payload) != {
            "atomic_scores", "uncertainty", "abstain", "abstention_reason"}:
        raise ValueError("blind score fields differ from the contract")
    entries = payload["atomic_scores"]
    if not isinstance(entries, list) or len(entries) != len(spec.cue_ids):
        raise ValueError("blind score must contain every cue exactly once")
    by_id: dict[str, tuple[float, str]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != {
                "cue_id", "score", "evidence"}:
            raise ValueError("atomic score fields differ from the contract")
        cue_id = _identifier(entry["cue_id"], "atomic cue_id")
        if cue_id in by_id:
            raise ValueError("blind score repeats a cue")
        by_id[cue_id] = (
            _unit(entry["score"], f"score for {cue_id}"),
            _text(entry["evidence"], f"evidence for {cue_id}"),
        )
    if set(by_id) != set(spec.cue_ids):
        raise ValueError("blind score cue identities differ from the rubric")
    uncertainty = _unit(payload["uncertainty"], "uncertainty")
    abstain = payload["abstain"]
    if not isinstance(abstain, bool) \
            or not isinstance(payload["abstention_reason"], str):
        raise ValueError("blind score abstention fields are malformed")
    ordered_scores = tuple((cue_id, by_id[cue_id][0]) for cue_id in spec.cue_ids)
    ordered_evidence = tuple(
        (cue_id, by_id[cue_id][1]) for cue_id in spec.cue_ids)
    if abstain or uncertainty > MAX_ACCEPTED_UNCERTAINTY:
        soft_result: SoftResult = SoftAbsent(
            concept_id=spec.hypothesis_id,
            reason_code="scorer-abstained" if abstain else "high-uncertainty",
            detail=_text(
                payload["abstention_reason"] or "uncertainty above admission cap",
                "abstention_reason"),
            provenance=(spec.digest(), producer_digest),
        )
    else:
        membership = _compose_membership(
            spec, {cue_id: score for cue_id, score in ordered_scores})
        soft_result = SoftEvidence(
            concept_id=spec.hypothesis_id,
            membership=membership,
            producer_digest=producer_digest,
            components=ordered_scores,
            provenance=(spec.digest(), "blind-single-panel/v1"),
        )
    return PanelSoftScore(
        spec_digest=spec.digest(),
        result=soft_result,
        cue_scores=ordered_scores,
        cue_evidence=ordered_evidence,
        uncertainty=uncertainty,
        receipt=dict(receipt),
    )


def _validate_panel_score(
        spec: SoftPredicateSpec, panel: PanelSoftScore) -> None:
    """Rebind stored evidence to its rubric before any numeric selection."""
    if not isinstance(panel, PanelSoftScore) \
            or panel.spec_digest != spec.digest():
        raise ValueError("soft evidence belongs to a different frozen rubric")
    if panel.result.concept_id != spec.hypothesis_id:
        raise ValueError("soft evidence concept differs from its frozen rubric")
    uncertainty = _unit(panel.uncertainty, "stored uncertainty")
    if uncertainty != panel.uncertainty:
        raise ValueError("stored uncertainty is not canonical")

    cue_scores = tuple(panel.cue_scores)
    cue_evidence = tuple(panel.cue_evidence)
    if isinstance(panel.result, SoftEvidence):
        if tuple(cue_id for cue_id, _score in cue_scores) != spec.cue_ids:
            raise ValueError("soft evidence cue identities/order differ")
        normalized_scores = tuple(
            (cue_id, _unit(score, f"stored score for {cue_id}"))
            for cue_id, score in cue_scores)
        if normalized_scores != cue_scores \
                or panel.result.components != cue_scores:
            raise ValueError("soft evidence components differ from atomic cues")
        if tuple(cue_id for cue_id, _text_value in cue_evidence) \
                != spec.cue_ids:
            raise ValueError("soft textual evidence differs from atomic cues")
        for cue_id, text_value in cue_evidence:
            if _text(text_value, f"stored evidence for {cue_id}") != text_value:
                raise ValueError("soft textual evidence is not canonical")
        recomputed = _compose_membership(
            spec, {cue_id: score for cue_id, score in cue_scores})
        if panel.result.membership != recomputed:
            raise ValueError(
                "soft membership is not the frozen aggregation of atomic cues")
        if spec.digest() not in panel.result.provenance:
            raise ValueError("soft evidence does not bind the frozen rubric")
        if not isinstance(panel.receipt, Mapping) or not panel.receipt:
            raise ValueError("present soft evidence requires a producer receipt")
        receipt_digest = panel.receipt.get("receipt_digest")
        if not isinstance(receipt_digest, str) or not receipt_digest:
            raise ValueError("present soft evidence receipt has no digest")
        if panel.receipt.get("schema") == codex_proposer.CODEX_RECEIPT_SCHEMA:
            codex_proposer.validate_codex_receipt(panel.receipt)
    elif isinstance(panel.result, SoftAbsent):
        # A scorer-produced abstention may retain its atomic observations, but
        # a synthetic structural absence is also allowed to have none.
        if cue_scores or cue_evidence:
            if tuple(cue_id for cue_id, _score in cue_scores) != spec.cue_ids \
                    or tuple(cue_id for cue_id, _value in cue_evidence) \
                    != spec.cue_ids:
                raise ValueError("soft absence has malformed atomic cues")
    elif isinstance(panel.result, SoftError):
        if cue_scores or cue_evidence or panel.receipt:
            raise ValueError("soft scorer errors may not carry trusted evidence")
    else:  # pragma: no cover - SoftResult is deliberately a closed runtime sum
        raise ValueError("unknown soft result state")


def _soft_result_from_dict(value: Mapping[str, Any]) -> SoftResult:
    if not isinstance(value, Mapping):
        raise ValueError("stored soft result must be an object")
    state = value.get("state")
    if state == "present":
        required = {
            "state", "concept_id", "membership", "producer_digest",
            "raw_value", "components", "prototype_digest", "input_digests",
            "provenance",
        }
        if set(value) != required \
                or not isinstance(value["components"], list) \
                or not isinstance(value["input_digests"], list) \
                or not isinstance(value["provenance"], list):
            raise ValueError("stored present soft result fields differ")
        try:
            components = tuple(
                (str(item[0]), item[1])
                for item in value["components"]
                if isinstance(item, list) and len(item) == 2)
        except (IndexError, TypeError) as exc:
            raise ValueError("stored soft components are malformed") from exc
        if len(components) != len(value["components"]):
            raise ValueError("stored soft components are malformed")
        return SoftEvidence(
            concept_id=value["concept_id"],
            membership=value["membership"],
            producer_digest=value["producer_digest"],
            raw_value=value["raw_value"],
            components=components,
            prototype_digest=value["prototype_digest"],
            input_digests=tuple(value["input_digests"]),
            provenance=tuple(value["provenance"]),
        )
    if state == "absent":
        if set(value) != {
                "state", "concept_id", "reason_code", "detail", "provenance"} \
                or not isinstance(value["provenance"], list):
            raise ValueError("stored absent soft result fields differ")
        return SoftAbsent(
            concept_id=value["concept_id"],
            reason_code=value["reason_code"],
            detail=value["detail"],
            provenance=tuple(value["provenance"]),
        )
    if state == "error":
        if set(value) != {
                "state", "concept_id", "error_code", "detail", "provenance"} \
                or not isinstance(value["provenance"], list):
            raise ValueError("stored error soft result fields differ")
        return SoftError(
            concept_id=value["concept_id"],
            error_code=value["error_code"],
            detail=value["detail"],
            provenance=tuple(value["provenance"]),
        )
    raise ValueError("stored soft result has an unknown state")


def panel_soft_score_from_dict(
        spec: SoftPredicateSpec, value: Mapping[str, Any]) -> PanelSoftScore:
    """Strictly reconstruct and validate one persisted panel observation."""
    if not isinstance(value, Mapping) or set(value) != {
            "schema", "spec_digest", "result", "cue_scores", "cue_evidence",
            "uncertainty", "receipt"}:
        raise ValueError("stored panel soft score fields differ")
    if value["schema"] != SOFT_EVIDENCE_SCHEMA \
            or not isinstance(value["result"], Mapping) \
            or not isinstance(value["cue_scores"], list) \
            or not isinstance(value["cue_evidence"], list) \
            or not isinstance(value["receipt"], Mapping):
        raise ValueError("stored panel soft score violates its schema")

    def pairs(items: Sequence[Any], name: str) -> tuple[tuple[Any, Any], ...]:
        if any(not isinstance(item, list) or len(item) != 2 for item in items):
            raise ValueError(f"stored {name} pairs are malformed")
        return tuple((item[0], item[1]) for item in items)

    panel = PanelSoftScore(
        spec_digest=value["spec_digest"],
        result=_soft_result_from_dict(value["result"]),
        cue_scores=pairs(value["cue_scores"], "cue score"),
        cue_evidence=pairs(value["cue_evidence"], "cue evidence"),
        uncertainty=value["uncertainty"],
        receipt=dict(value["receipt"]),
    )
    _validate_panel_score(spec, panel)
    return panel


@contextmanager
def _neutral_panel_view(panel_png_path: str):
    """Expose one immutable PNG as ``panel.png`` outside the repository."""
    panel_bytes = codex_proposer._read_regular_png(panel_png_path)
    parent = codex_proposer._safe_temp_parent()
    with tempfile.TemporaryDirectory(
            prefix="bongard-soft-panel-", dir=parent) as directory:
        target = os.path.join(directory, "panel.png")
        descriptor = os.open(
            target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            offset = 0
            while offset < len(panel_bytes):
                offset += os.write(descriptor, panel_bytes[offset:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        yield target


@dataclass(frozen=True)
class SoftVerification:
    hypothesis_id: str
    spec_digest: str
    scores: tuple[float | None, ...]
    states: tuple[str, ...]
    support_predictions: tuple[bool, ...]
    support_labels: tuple[bool, ...]
    support_errors: int
    rotated_loo_errors: int
    rotated_loo_checks: int
    threshold: float
    fold_thresholds: tuple[float, ...]
    invalid_measurements: int
    polarity_conflict: bool
    accepted: bool
    rule: str
    complexity: int
    evidence: tuple[PanelSoftScore, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SOFT_VERIFICATION_SCHEMA,
            **{
                key: value for key, value in asdict(self).items()
                if key != "evidence"
            },
            "evidence": [item.to_dict() for item in self.evidence],
            "track": SOFT_TRACK,
        }


def _predict(score: float | None, threshold: float, order: str) -> bool:
    if score is None:
        return False
    return score > threshold if order == "high_positive" else score < threshold


def _fit_threshold(
        scores: Sequence[float | None], labels: Sequence[bool], order: str,
        comparison: str) -> float:
    if comparison == "absolute":
        return 0.5
    finite = sorted({float(score) for score in scores if score is not None})
    if not finite:
        return 0.5
    candidates = [
        math.nextafter(finite[0], -math.inf),
        *(low / 2.0 + high / 2.0
          for low, high in zip(finite[:-1], finite[1:])),
        math.nextafter(finite[-1], math.inf),
    ]
    ranked = []
    for threshold in candidates:
        predictions = tuple(_predict(score, threshold, order) for score in scores)
        errors = sum(pred != label for pred, label in zip(predictions, labels))
        ranked.append((errors, abs(threshold - 0.5), threshold))
    return min(ranked)[2]


def _verification_from_evidence(
        spec: SoftPredicateSpec, evidence: Sequence[PanelSoftScore],
        labels: Sequence[bool]) -> SoftVerification:
    if len(evidence) != len(labels) or len(labels) != 12 \
            or sum(bool(label) for label in labels) != 6:
        raise ValueError("soft verification requires six positive and six negative panels")
    for item in evidence:
        _validate_panel_score(spec, item)
    scores = tuple(
        item.result.membership if isinstance(item.result, SoftEvidence) else None
        for item in evidence)
    states = tuple(item.result.to_dict()["state"] for item in evidence)
    threshold = _fit_threshold(scores, labels, spec.order, spec.comparison)
    predictions = tuple(_predict(score, threshold, spec.order) for score in scores)
    support_errors = sum(
        prediction != bool(label)
        for prediction, label in zip(predictions, labels))
    reverse_order = (
        "low_positive" if spec.order == "high_positive" else "high_positive")
    reverse_threshold = _fit_threshold(
        scores, labels, reverse_order, spec.comparison)
    reverse_errors = sum(
        _predict(score, reverse_threshold, reverse_order) != bool(label)
        for score, label in zip(scores, labels))
    polarity_conflict = reverse_errors < support_errors

    positive_indices = [index for index, label in enumerate(labels) if label]
    negative_indices = [index for index, label in enumerate(labels) if not label]
    loo_errors = 0
    loo_checks = 0
    fold_thresholds: list[float] = []
    for positive in positive_indices:
        for negative in negative_indices:
            held = {positive, negative}
            train_scores = [score for index, score in enumerate(scores)
                            if index not in held]
            train_labels = [bool(label) for index, label in enumerate(labels)
                            if index not in held]
            fold_threshold = _fit_threshold(
                train_scores, train_labels, spec.order, spec.comparison)
            fold_thresholds.append(fold_threshold)
            for index in (positive, negative):
                loo_errors += int(
                    _predict(scores[index], fold_threshold, spec.order)
                    != bool(labels[index]))
                loo_checks += 1
    invalid = sum(score is None for score in scores)
    accepted = (
        invalid == 0 and support_errors == 0 and loo_errors == 0
        and not polarity_conflict)
    operator = ">" if spec.order == "high_positive" else "<"
    rule = f"soft:{spec.hypothesis_id}{operator}{threshold:.6g}"
    return SoftVerification(
        hypothesis_id=spec.hypothesis_id,
        spec_digest=spec.digest(),
        scores=scores,
        states=states,
        support_predictions=predictions,
        support_labels=tuple(bool(item) for item in labels),
        support_errors=support_errors,
        rotated_loo_errors=loo_errors,
        rotated_loo_checks=loo_checks,
        threshold=threshold,
        fold_thresholds=tuple(fold_thresholds),
        invalid_measurements=invalid,
        polarity_conflict=polarity_conflict,
        accepted=accepted,
        rule=rule,
        complexity=len(canonical_json_bytes(spec.to_dict())),
        evidence=tuple(evidence),
    )


def verify_soft_predicate(
        spec: SoftPredicateSpec, panel_png_paths: Sequence[str],
        labels: Sequence[bool], scorer: BlindSoftScorer) -> SoftVerification:
    if len(panel_png_paths) != len(labels):
        raise ValueError("panel and label counts differ")
    # Calls are stateless and individually blind.  A content-derived order
    # prevents the API call sequence itself from revealing the side grouping.
    def rank_key(index: int) -> bytes:
        panel_bytes = codex_proposer._read_regular_png(
            panel_png_paths[index])
        return hashlib.sha256(
            spec.digest().encode("ascii")
            + panel_bytes
        ).digest()

    ranked_indices = sorted(
        range(len(panel_png_paths)),
        key=rank_key,
    )
    evidence: list[PanelSoftScore | None] = [None] * len(panel_png_paths)
    for index in ranked_indices:
        try:
            with _neutral_panel_view(panel_png_paths[index]) as neutral_path:
                evidence[index] = scorer.score(spec, neutral_path)
        except Exception as exc:
            evidence[index] = PanelSoftScore(
                spec_digest=spec.digest(),
                result=SoftError(
                    spec.hypothesis_id, "scorer-failure",
                    f"{type(exc).__name__}: {exc}",
                    provenance=(spec.digest(),)),
                cue_scores=(), cue_evidence=(), uncertainty=1.0, receipt={},
            )
    completed = tuple(item for item in evidence if item is not None)
    if len(completed) != len(panel_png_paths):  # pragma: no cover - defensive
        raise AssertionError("blind scorer left an unfilled panel slot")
    return _verification_from_evidence(spec, completed, labels)


def verify_soft_predicates_batched(
        specs: Sequence[SoftPredicateSpec], panel_png_paths: Sequence[str],
        labels: Sequence[bool], scorer: BlindSoftBatchScorer,
        *, max_workers: int = 1,
        ) -> tuple[SoftVerification, ...]:
    """Blind-score one panel per turn and replay every frozen rubric."""
    specs = tuple(specs)
    if not specs or len({spec.digest() for spec in specs}) != len(specs):
        raise ValueError("batched verification requires distinct soft rubrics")
    if len(panel_png_paths) != len(labels) or len(labels) != 12:
        raise ValueError("batched verification requires twelve labelled panels")
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) \
            or not 1 <= max_workers <= 12:
        raise ValueError("max_workers must be an integer in 1..12")

    def rank_key(index: int) -> bytes:
        panel_bytes = codex_proposer._read_regular_png(
            panel_png_paths[index])
        return hashlib.sha256(
            content_digest([spec.digest() for spec in specs]).encode("ascii")
            + panel_bytes
        ).digest()

    evidence: list[list[PanelSoftScore | None]] = [
        [None] * len(panel_png_paths) for _ in specs]

    def score_panel(panel_index: int) -> tuple[
            int, tuple[PanelSoftScore, ...] | None, Exception | None]:
        try:
            with _neutral_panel_view(
                    panel_png_paths[panel_index]) as neutral_path:
                scored = scorer.score_many(specs, neutral_path)
            if len(scored) != len(specs):
                raise ValueError("blind batch result count differs")
            if any(item.spec_digest != spec.digest()
                   for spec, item in zip(specs, scored)):
                raise ValueError("blind batch score uses a different rubric")
            return panel_index, tuple(scored), None
        except Exception as exc:
            return panel_index, None, exc

    ranked_indices = sorted(range(len(panel_png_paths)), key=rank_key)
    if max_workers == 1:
        results = map(score_panel, ranked_indices)
        executor = None
    else:
        executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="bongard-soft-scorer")
        results = executor.map(score_panel, ranked_indices)
    try:
        for panel_index, scored, failure in results:
            if failure is None and scored is not None:
                for spec_index, item in enumerate(scored):
                    evidence[spec_index][panel_index] = item
                continue
            for spec_index, spec in enumerate(specs):
                evidence[spec_index][panel_index] = PanelSoftScore(
                    spec_digest=spec.digest(),
                    result=SoftError(
                        spec.hypothesis_id, "scorer-failure",
                        f"{type(failure).__name__}: {failure}",
                        provenance=(spec.digest(),)),
                    cue_scores=(), cue_evidence=(), uncertainty=1.0,
                    receipt={},
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
    return tuple(
        _verification_from_evidence(
            spec,
            tuple(item for item in rows if item is not None),
            labels,
        )
        for spec, rows in zip(specs, evidence)
    )


def replay_soft_verification(
        spec: SoftPredicateSpec, evidence: Sequence[PanelSoftScore],
        labels: Sequence[bool]) -> SoftVerification:
    """Replay all downstream decisions without re-querying the VLM."""
    return _verification_from_evidence(spec, evidence, labels)


def select_soft_verification(
        candidates: Sequence[SoftVerification]) -> SoftVerification | None:
    if not candidates:
        return None
    return min(candidates, key=lambda item: (
        not item.accepted,
        item.invalid_measurements,
        item.rotated_loo_errors,
        item.support_errors,
        item.polarity_conflict,
        item.complexity,
        item.hypothesis_id,
    ))


__all__ = [
    "BlindSoftScorer",
    "BlindSoftBatchScorer",
    "CodexBlindSoftBatchScorer",
    "CodexBlindSoftScorer",
    "CodexSoftHypothesisProposer",
    "MAX_ACCEPTED_UNCERTAINTY",
    "PanelSoftScore",
    "SOFT_EVIDENCE_SCHEMA",
    "SOFT_HYPOTHESES_OUTPUT_SCHEMA",
    "SOFT_PREDICATE_SCHEMA",
    "SOFT_PROPOSAL_SCHEMA",
    "SOFT_TRACK",
    "SOFT_VERIFICATION_SCHEMA",
    "SoftCueSpec",
    "SoftPredicateSpec",
    "SoftProposalBundle",
    "SoftVerification",
    "build_blind_score_prompt",
    "build_soft_proposal_prompt",
    "panel_soft_score_from_payload",
    "panel_soft_score_from_dict",
    "replay_soft_verification",
    "select_soft_verification",
    "verify_soft_predicate",
    "verify_soft_predicates_batched",
]
