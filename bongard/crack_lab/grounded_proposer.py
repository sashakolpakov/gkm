"""Headless-Codex proposer for grounded observable intents.

The model uses joint vision to choose *which registered measurements matter*.
It cannot author executable code, redefine an observable, assign membership
scores, or choose numeric cutoffs.  The harness later fits simple atomic rules
on support only and synthesizes a small positive Boolean formula.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import codex_proposer
from grounded_observables import ObservableDescriptor
from grounded_predicate_ir import canonical_digest


GROUNDED_PROPOSAL_SCHEMA = "bongard.grounded-intent-proposal/v1"
MAX_INTENTS = 12
MAX_TEXT = 1600


@dataclass(frozen=True)
class GroundingIntent:
    intent_id: str
    observable_id: str
    shape: str
    rationale: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class GroundedProposalBundle:
    problem_id: str
    analysis: str
    intents: tuple[GroundingIntent, ...]
    catalog_digest: str
    receipt: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": GROUNDED_PROPOSAL_SCHEMA,
            "problem_id": self.problem_id,
            "analysis": self.analysis,
            "intents": [intent.to_dict() for intent in self.intents],
            "catalog_digest": self.catalog_digest,
            "receipt": dict(self.receipt),
        }


def _normal_text(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be text")
    result = " ".join(value.split())
    if not result or len(result) > MAX_TEXT:
        raise ValueError(f"{label} must contain 1..{MAX_TEXT} characters")
    return result


def _catalog_payload(
        descriptors: Sequence[ObservableDescriptor]) -> list[dict[str, Any]]:
    return [descriptor.prompt_dict() for descriptor in descriptors]


def grounded_catalog_digest(
        descriptors: Sequence[ObservableDescriptor]) -> str:
    return canonical_digest(_catalog_payload(descriptors))


def _output_schema(
        descriptors: Sequence[ObservableDescriptor]) -> dict[str, Any]:
    observable_ids = [item.contract.observable_id for item in descriptors]
    shapes = sorted({shape for item in descriptors
                     for shape in item.admissible_shapes})
    return {
        "type": "object",
        "properties": {
            "analysis": {
                "type": "string", "minLength": 1, "maxLength": MAX_TEXT,
            },
            "intents": {
                "type": "array", "minItems": 1, "maxItems": MAX_INTENTS,
                "items": {
                    "type": "object",
                    "properties": {
                        "observable_id": {
                            "type": "string", "enum": observable_ids,
                        },
                        "shape": {"type": "string", "enum": shapes},
                        "rationale": {
                            "type": "string", "minLength": 1,
                            "maxLength": MAX_TEXT,
                        },
                    },
                    "required": ["observable_id", "shape", "rationale"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["analysis", "intents"],
        "additionalProperties": False,
    }


def build_grounded_proposal_prompt(
        descriptors: Sequence[ObservableDescriptor]) -> str:
    import json
    catalog = _catalog_payload(descriptors)
    return f"""\
You see one Bongard support problem: the first six images are positive and the
last six are negative. Use direct joint vision to identify which measurements
from the CLOSED CATALOG below could express the distinction.

Return measurement intents, not finished classifiers. For each intent choose:
- `exact`: the positive property should equal one discrete value;
- `low`: smaller values support the positive property;
- `high`: larger values support the positive property;
- `band`: values in a bounded middle interval support it.

The harness—not you—will extract the registered witnesses, fit numeric bounds
using support labels, synthesize positive conjunctions, freeze the formula,
and evaluate unseen rerendered query panels. Do not invent observables, code,
thresholds, cue scores, fuzzy memberships, panel hashes, or negated rescue
rules. Cover the coarse distractors and the nearest foil; include multiple
non-paraphrased intents when their conjunction may be necessary. Rationale is
diagnostic prose only and has no executable meaning.

CLOSED_OBSERVABLE_CATALOG_DIGEST: {grounded_catalog_digest(descriptors)}
CLOSED_OBSERVABLE_CATALOG_JSON: {json.dumps(catalog, sort_keys=True, separators=(",", ":"))}"""


class CodexGroundedIntentProposer:
    def __init__(
            self, descriptors: Sequence[ObservableDescriptor],
            model: str = codex_proposer.DEFAULT_CODEX_MODEL,
            *, minutes: int = 15,
            reasoning_effort: str = codex_proposer.DEFAULT_REASONING_EFFORT,
            executable: str = "codex") -> None:
        self.descriptors = tuple(descriptors)
        if not self.descriptors:
            raise ValueError("grounded proposer requires a nonempty catalog")
        self.model = model
        self.minutes = minutes
        self.reasoning_effort = reasoning_effort
        self.executable = executable

    def propose(
            self, problem_id: str,
            panel_png_paths: Sequence[str]) -> GroundedProposalBundle:
        if len(panel_png_paths) != 12:
            raise ValueError("grounded proposer requires twelve support panels")
        result = codex_proposer.run_codex_structured(
            build_grounded_proposal_prompt(self.descriptors),
            panel_png_paths,
            _output_schema(self.descriptors),
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            minutes=self.minutes,
            executable=self.executable,
        )
        payload = result.payload
        if not isinstance(payload, Mapping) or set(payload) != {
                "analysis", "intents"} or not isinstance(payload["intents"], list):
            raise ValueError("grounded proposal differs from its output contract")
        allowed = {
            descriptor.contract.observable_id: set(descriptor.admissible_shapes)
            for descriptor in self.descriptors
        }
        intents: list[GroundingIntent] = []
        seen: set[tuple[str, str]] = set()
        for raw in payload["intents"]:
            if not isinstance(raw, Mapping) or set(raw) != {
                    "observable_id", "shape", "rationale"}:
                raise ValueError("grounded intent fields differ")
            observable_id = str(raw["observable_id"])
            shape = str(raw["shape"])
            if observable_id not in allowed or shape not in allowed[observable_id]:
                raise ValueError("grounded intent is outside the closed catalog")
            key = (observable_id, shape)
            if key in seen:
                continue
            seen.add(key)
            intents.append(GroundingIntent(
                intent_id=f"intent-{len(intents):02d}",
                observable_id=observable_id,
                shape=shape,
                rationale=_normal_text(raw["rationale"], "intent rationale"),
            ))
        if not intents:
            raise ValueError("grounded proposer returned no distinct intents")
        return GroundedProposalBundle(
            problem_id=_normal_text(problem_id, "problem_id"),
            analysis=_normal_text(payload["analysis"], "analysis"),
            intents=tuple(intents),
            catalog_digest=grounded_catalog_digest(self.descriptors),
            receipt=result.receipt.to_dict(),
        )


__all__ = [
    "CodexGroundedIntentProposer",
    "GROUNDED_PROPOSAL_SCHEMA",
    "GroundedProposalBundle",
    "GroundingIntent",
    "build_grounded_proposal_prompt",
    "grounded_catalog_digest",
]
