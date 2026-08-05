"""One-turn headless support proposer for the HYBRID contrastive track."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import codex_proposer
from hybrid_program_split import canonical_digest, canonical_json


CLAIM_PROPOSAL_SCHEMA = "bongard.hybrid-claim-proposal/v1"
OFFLINE_FIXTURE_RECEIPT_SCHEMA = "bongard.hybrid-offline-fixture-receipt/v1"
MAX_TEXT = 1600

_SIDE_LANGUAGE = re.compile(
    r"(?:\bpositives?\b|\bnegatives?\b|\b(?:left|right)\s+"
    r"(?:side|column)\b|\b(?:pos|neg)_[0-9]+\b|\bfirst\s+six\b|"
    r"\blast\s+six\b|\bfirst\s+group\b|\bsecond\s+group\b)",
    re.IGNORECASE,
)
_IDENTITY_LANGUAGE = re.compile(
    r"(?:sha-?256|pixel\s+hash|file(?:name)?|exact\s+coordinates?)",
    re.IGNORECASE,
)
_LEXICAL_NEGATION = re.compile(
    r"(?:\b(?:no|not|never|neither|without|lacks?|lacking|absence)\b|"
    r"\b(?:missing|excludes?|excluding|avoids?|cannot)\b|"
    r"\bfails?\s+to\b|\bnon[- ]?[a-z]|(?:n't)\b)",
    re.IGNORECASE,
)


CLAIM_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "analysis": {
            "type": "string", "minLength": 1, "maxLength": MAX_TEXT,
        },
        "claim": {
            "type": "string", "minLength": 1, "maxLength": MAX_TEXT,
        },
    },
    "required": ["analysis", "claim"],
    "additionalProperties": False,
}


def _text(value: Any, name: str, *, side_free: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be text")
    normalized = " ".join(value.split())
    if not normalized or len(normalized) > MAX_TEXT:
        raise ValueError(f"{name} must contain 1..{MAX_TEXT} characters")
    if side_free and (
        _SIDE_LANGUAGE.search(normalized)
        or _IDENTITY_LANGUAGE.search(normalized)
        or _LEXICAL_NEGATION.search(normalized)
    ):
        raise ValueError(
            f"{name} must be side-free, grammatical-affirmative, and "
            "content-general")
    return normalized


def build_claim_prompt() -> str:
    return """\
You see one Bongard support problem: the first six images are labelled
positive and the last six are labelled negative. Return exactly ONE visual
claim whose literal affirmative membership is intended to hold for positive
instances and not for negative instances.

The `claim` must be a short, side-free, grammatical-affirmative,
content-general description usable on
one new image at a time. It must not mention labels, groups, panel positions,
filenames, hashes, exact pixel coordinates, or this prompt. Do not return
alternatives, scores, thresholds, Boolean formulas, code, or a polarity-flip
instruction. Do not phrase it with `no`, `not`, `without`, `lacks`,
`absence`, or another lexical negation; name the visible affirmative property
itself. A later fixed contrastive oracle will content-select three frozen
anchor/foil pairs from six-per-role support pools, show each target with those
pairs in two fresh fully swapped presentations, and decide whether this exact
claim applies. Prefer the smallest stable semantic distinction, including an
open-vocabulary resemblance such as "bird-like object" when that is genuinely
the shared concept. Put diagnostic deliberation in `analysis`; only `claim`
becomes executable oracle context."""


def _raw_digest(value: Any) -> str:
    return hashlib.sha256(
        canonical_json(value).encode("utf-8")).hexdigest()


def make_offline_fixture_receipt(fixture_id: str) -> dict[str, str]:
    """Create the only non-Codex receipt admitted by injected unit tests."""
    if not isinstance(fixture_id, str) or not fixture_id \
            or len(fixture_id) > 200:
        raise ValueError("fixture_id must be short nonempty text")
    body = {
        "schema": OFFLINE_FIXTURE_RECEIPT_SCHEMA,
        "source": "offline-fixture",
        "fixture_id": fixture_id,
    }
    return {**body, "receipt_digest": canonical_digest(body)}


def _validate_offline_fixture_receipt(receipt: Mapping[str, Any]) -> None:
    if not isinstance(receipt, Mapping) or set(receipt) != {
            "schema", "source", "fixture_id", "receipt_digest"} \
            or receipt.get("schema") != OFFLINE_FIXTURE_RECEIPT_SCHEMA \
            or receipt.get("source") != "offline-fixture" \
            or not isinstance(receipt.get("fixture_id"), str) \
            or not receipt["fixture_id"] \
            or len(receipt["fixture_id"]) > 200:
        raise ValueError("offline fixture receipt fields differ")
    unsigned = {key: item for key, item in receipt.items()
                if key != "receipt_digest"}
    if receipt["receipt_digest"] != canonical_digest(unsigned):
        raise ValueError("offline fixture receipt digest does not reproduce")


def validate_claim_proposal_receipt(
    bundle: "ClaimProposalBundle",
    support_png_paths: Sequence[str],
    *,
    model: str,
    reasoning_effort: str,
    allow_offline_fixture: bool = False,
) -> None:
    """Bind a proposal to its exact prompt, schema, payload, and support view."""
    receipt = bundle.receipt
    if receipt.get("source") == "offline-fixture":
        if not allow_offline_fixture:
            raise ValueError("offline fixture proposal is forbidden in a live run")
        _validate_offline_fixture_receipt(receipt)
        return
    if receipt.get("source") != "codex-cli":
        raise ValueError("claim proposal receipt source is unsupported")
    codex_proposer.validate_codex_receipt(receipt)
    prompt = build_claim_prompt()
    output_schema_digest = _raw_digest(CLAIM_OUTPUT_SCHEMA)
    panel_view_digest = codex_proposer.ordered_panel_view_digest(
        support_png_paths)
    causal = codex_proposer._causal_input_metadata(
        prompt,
        support_png_paths,
        output_schema_digest,
        panel_view_digest,
        None,
    )
    expected = {
        **causal,
        "requested_model": model,
        "requested_reasoning_effort": reasoning_effort,
        "output_schema_digest": output_schema_digest,
        "structured_output_digest": _raw_digest({
            "analysis": bundle.analysis,
            "claim": bundle.claim,
        }),
        "proposed_source_digest": "",
        "proposed_log_digest": "",
    }
    for field, expected_value in expected.items():
        if receipt.get(field) != expected_value:
            raise ValueError(
                f"claim proposal receipt {field} does not bind exact input/output")


@dataclass(frozen=True)
class ClaimProposalBundle:
    problem_id: str
    analysis: str
    claim: str
    receipt: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "schema": CLAIM_PROPOSAL_SCHEMA,
            "problem_id": self.problem_id,
            "analysis": self.analysis,
            "claim": self.claim,
            "receipt": dict(self.receipt),
        }
        body["proposal_digest"] = canonical_digest(body)
        return body

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ClaimProposalBundle":
        keys = {
            "schema", "problem_id", "analysis", "claim", "receipt",
            "proposal_digest",
        }
        if not isinstance(value, Mapping) or set(value) != keys:
            raise ValueError("HYBRID claim proposal fields differ")
        unsigned = {key: item for key, item in value.items()
                    if key != "proposal_digest"}
        if value["schema"] != CLAIM_PROPOSAL_SCHEMA \
                or value["proposal_digest"] != canonical_digest(unsigned) \
                or not isinstance(value["receipt"], Mapping):
            raise ValueError("HYBRID claim proposal digest does not reproduce")
        bundle = cls(
            problem_id=_text(value["problem_id"], "problem_id"),
            analysis=_text(value["analysis"], "analysis"),
            claim=_text(value["claim"], "claim", side_free=True),
            receipt=dict(value["receipt"]),
        )
        receipt = bundle.receipt
        if receipt.get("source") == "codex-cli":
            codex_proposer.validate_codex_receipt(receipt)
        elif receipt.get("source") == "offline-fixture":
            _validate_offline_fixture_receipt(receipt)
        else:
            raise ValueError("HYBRID claim proposal receipt source is unsupported")
        return bundle


class CodexHybridClaimProposer:
    def __init__(
        self,
        model: str = codex_proposer.DEFAULT_CODEX_MODEL,
        *,
        minutes: int = 15,
        reasoning_effort: str = codex_proposer.DEFAULT_REASONING_EFFORT,
        executable: str = "codex",
    ) -> None:
        self.model = model
        self.minutes = minutes
        self.reasoning_effort = reasoning_effort
        self.executable = executable

    def propose(
        self, problem_id: str, support_png_paths: Sequence[str],
    ) -> ClaimProposalBundle:
        if len(support_png_paths) != 12:
            raise ValueError("HYBRID claim proposer requires twelve support panels")
        result = codex_proposer.run_codex_structured(
            build_claim_prompt(),
            support_png_paths,
            CLAIM_OUTPUT_SCHEMA,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            minutes=self.minutes,
            executable=self.executable,
        )
        payload = result.payload
        if not isinstance(payload, Mapping) or set(payload) != {
            "analysis", "claim",
        }:
            raise ValueError("HYBRID claim output differs from its schema")
        analysis = _text(payload["analysis"], "analysis")
        claim = _text(payload["claim"], "claim", side_free=True)
        if payload["analysis"] != analysis or payload["claim"] != claim:
            raise ValueError(
                "HYBRID claim payload must already be normalized exact text")
        bundle = ClaimProposalBundle(
            problem_id=_text(problem_id, "problem_id"),
            analysis=analysis,
            claim=claim,
            receipt=result.receipt.to_dict(),
        )
        # Exercise the persisted validator before returning live evidence.
        restored = ClaimProposalBundle.from_dict(bundle.to_dict())
        validate_claim_proposal_receipt(
            restored,
            support_png_paths,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
        )
        return restored


__all__ = [
    "CLAIM_OUTPUT_SCHEMA",
    "CLAIM_PROPOSAL_SCHEMA",
    "OFFLINE_FIXTURE_RECEIPT_SCHEMA",
    "ClaimProposalBundle",
    "CodexHybridClaimProposer",
    "build_claim_prompt",
    "make_offline_fixture_receipt",
    "validate_claim_proposal_receipt",
]
