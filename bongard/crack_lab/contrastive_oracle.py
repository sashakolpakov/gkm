"""Content-addressed contrastive visual oracle for open-vocabulary concepts.

This module is deliberately not an absolute ``birdness`` scorer.  A frozen
contract contains an affirmative prose claim, six affirmative image
prototypes, and six hard-negative foils.  For each neutral target a fresh
Codex turn makes six categorical, evidence-bearing pair comparisons.  The
harness, not the model, maps the balanced left/right placements back to the
two reference families and requires a fixed five-of-six supermajority.

The resulting observable is an explicit ``ORACLE`` leaf in the grounded IR,
so any predicate using it is ``HYBRID`` rather than ``PURE``.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import codex_proposer
import grounded_predicate_ir as G


CONTRACT_SCHEMA = "bongard.contrastive-oracle-contract/v1"
EVIDENCE_SCHEMA = "bongard.contrastive-oracle-evidence/v1"
EVALUATION_SCHEMA = "bongard.contrastive-oracle-evaluation/v1"
PROMPT_POLICY = "affirmative-claim-six-neutral-pair-comparisons/v1"
ORDER_POLICY = "content-bound-balanced-three-left-three-right/v1"
DECISION_POLICY = "five-of-six-supermajority-abstaining/v1"
OBSERVABLE_ID = "oracle.contrastive.open-vocabulary/v1"
MAX_CLAIM_LENGTH = 4_000
MAX_EVIDENCE_LENGTH = 2_000
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _raw_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _digest(value: Any) -> str:
    return "sha256:" + _raw_digest(value)


def _byte_digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _read_png(path: str) -> bytes:
    if not isinstance(path, str) or not path:
        raise ValueError("PNG path must be a nonempty string")
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise ValueError(f"cannot stat PNG: {exc}") from exc
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise ValueError("oracle images must be regular non-symlink files")
    if not 8 <= info.st_size <= codex_proposer.MAX_PANEL_PNG_BYTES:
        raise ValueError("oracle PNG size is outside the transport bound")
    try:
        with open(path, "rb") as handle:
            data = handle.read(codex_proposer.MAX_PANEL_PNG_BYTES + 1)
    except OSError as exc:
        raise ValueError(f"cannot read PNG: {exc}") from exc
    if len(data) != info.st_size or not data.startswith(PNG_SIGNATURE):
        raise ValueError("oracle image is not a stable PNG snapshot")
    return data


def _text(value: Any, field: str, *, allow_empty: bool = False,
          maximum: int = MAX_EVIDENCE_LENGTH) -> str:
    if not isinstance(value, str) or "\x00" in value \
            or len(value.encode("utf-8")) > maximum \
            or (not allow_empty and not value.strip()):
        raise ValueError(f"{field} is invalid")
    return value


@dataclass(frozen=True, order=True)
class ImageIdentity:
    content_digest: str
    byte_count: int

    def __post_init__(self) -> None:
        if re.fullmatch(r"sha256:[0-9a-f]{64}", self.content_digest) is None:
            raise ValueError("image content_digest is invalid")
        if isinstance(self.byte_count, bool) or not isinstance(
                self.byte_count, int) or self.byte_count < len(PNG_SIGNATURE):
            raise ValueError("image byte_count is invalid")

    @classmethod
    def from_path(cls, path: str) -> "ImageIdentity":
        data = _read_png(path)
        return cls(_byte_digest(data), len(data))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ImageIdentity":
        if not isinstance(value, Mapping) \
                or set(value) != {"content_digest", "byte_count"}:
            raise ValueError("image identity fields are invalid")
        return cls(value["content_digest"], value["byte_count"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "content_digest": self.content_digest,
            "byte_count": self.byte_count,
        }


def _family(paths: Sequence[str], field: str) -> tuple[ImageIdentity, ...]:
    if isinstance(paths, (str, bytes)) or len(paths) != 6:
        raise ValueError(f"{field} requires exactly six PNG paths")
    identities = tuple(sorted(ImageIdentity.from_path(path) for path in paths))
    if len({item.content_digest for item in identities}) != 6:
        raise ValueError(f"{field} images must have six distinct byte digests")
    return identities


@dataclass(frozen=True)
class ContrastiveOracleContract:
    affirmative_claim: str
    positive_prototypes: tuple[ImageIdentity, ...]
    hard_negative_foils: tuple[ImageIdentity, ...]
    model: str = codex_proposer.DEFAULT_CODEX_MODEL
    reasoning_effort: str = codex_proposer.DEFAULT_REASONING_EFFORT
    prompt_policy: str = PROMPT_POLICY
    output_schema_digest: str = ""
    order_policy: str = ORDER_POLICY
    decision_policy: str = DECISION_POLICY
    transport_policy: str = codex_proposer.CODEX_ISOLATION_POLICY

    def __post_init__(self) -> None:
        _text(self.affirmative_claim, "affirmative_claim",
              maximum=MAX_CLAIM_LENGTH)
        if self.affirmative_claim != self.affirmative_claim.strip():
            raise ValueError("affirmative_claim must not have edge whitespace")
        positives = tuple(self.positive_prototypes)
        foils = tuple(self.hard_negative_foils)
        if len(positives) != 6 or len(foils) != 6 \
                or tuple(sorted(positives)) != positives \
                or tuple(sorted(foils)) != foils:
            raise ValueError("contract families must be canonical six-image tuples")
        all_digests = [item.content_digest for item in positives + foils]
        if len(set(all_digests)) != 12:
            raise ValueError("positive prototypes and foils must all be distinct")
        if not isinstance(self.model, str) \
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", self.model) is None:
            raise ValueError("model is invalid")
        if self.reasoning_effort not in codex_proposer.REASONING_EFFORTS:
            raise ValueError("reasoning_effort is invalid")
        if self.prompt_policy != PROMPT_POLICY \
                or self.order_policy != ORDER_POLICY \
                or self.decision_policy != DECISION_POLICY \
                or self.transport_policy != codex_proposer.CODEX_ISOLATION_POLICY:
            raise ValueError("oracle policy identity is unsupported")
        expected_schema_digest = _digest(contrastive_output_schema())
        if self.output_schema_digest != expected_schema_digest:
            raise ValueError("oracle output schema digest differs")

    @classmethod
    def create(
            cls, affirmative_claim: str,
            positive_prototype_paths: Sequence[str],
            hard_negative_foil_paths: Sequence[str],
            *, model: str = codex_proposer.DEFAULT_CODEX_MODEL,
            reasoning_effort: str = codex_proposer.DEFAULT_REASONING_EFFORT,
            ) -> "ContrastiveOracleContract":
        return cls(
            affirmative_claim=affirmative_claim,
            positive_prototypes=_family(
                positive_prototype_paths, "positive prototypes"),
            hard_negative_foils=_family(
                hard_negative_foil_paths, "hard-negative foils"),
            model=model,
            reasoning_effort=reasoning_effort,
            output_schema_digest=_digest(contrastive_output_schema()),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ContrastiveOracleContract":
        required = {
            "schema", "affirmative_claim", "positive_prototypes",
            "hard_negative_foils", "model", "reasoning_effort",
            "prompt_policy", "output_schema_digest", "order_policy",
            "decision_policy", "transport_policy",
        }
        if not isinstance(value, Mapping) or set(value) != required \
                or value.get("schema") != CONTRACT_SCHEMA:
            raise ValueError("contrastive oracle contract fields are invalid")
        for field in ("positive_prototypes", "hard_negative_foils"):
            if not isinstance(value[field], list):
                raise ValueError(f"{field} must be a list")
        return cls(
            affirmative_claim=value["affirmative_claim"],
            positive_prototypes=tuple(
                ImageIdentity.from_dict(item)
                for item in value["positive_prototypes"]),
            hard_negative_foils=tuple(
                ImageIdentity.from_dict(item)
                for item in value["hard_negative_foils"]),
            model=value["model"],
            reasoning_effort=value["reasoning_effort"],
            prompt_policy=value["prompt_policy"],
            output_schema_digest=value["output_schema_digest"],
            order_policy=value["order_policy"],
            decision_policy=value["decision_policy"],
            transport_policy=value["transport_policy"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CONTRACT_SCHEMA,
            "affirmative_claim": self.affirmative_claim,
            "positive_prototypes": [item.to_dict() for item in self.positive_prototypes],
            "hard_negative_foils": [item.to_dict() for item in self.hard_negative_foils],
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "prompt_policy": self.prompt_policy,
            "output_schema_digest": self.output_schema_digest,
            "order_policy": self.order_policy,
            "decision_policy": self.decision_policy,
            "transport_policy": self.transport_policy,
        }

    def digest(self) -> str:
        return _digest(self.to_dict())


@dataclass(frozen=True)
class PairAssignment:
    pair_id: str
    left: ImageIdentity
    right: ImageIdentity
    affirmative_side: str
    left_name: str
    right_name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
            "affirmative_side": self.affirmative_side,
            "left_name": self.left_name,
            "right_name": self.right_name,
        }


@dataclass(frozen=True)
class Presentation:
    target: ImageIdentity
    target_name: str
    pairs: tuple[PairAssignment, ...]

    def ordered_names(self) -> tuple[str, ...]:
        return (self.target_name,) + tuple(
            name for pair in self.pairs for name in (pair.left_name, pair.right_name))

    def ordered_identities(self) -> tuple[ImageIdentity, ...]:
        return (self.target,) + tuple(
            item for pair in self.pairs for item in (pair.left, pair.right))

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target.to_dict(),
            "target_name": self.target_name,
            "pairs": [pair.to_dict() for pair in self.pairs],
        }


def _rank(seed: str, role: str, identity: ImageIdentity) -> str:
    return hashlib.sha256(
        f"{seed}|{role}|{identity.content_digest}|{identity.byte_count}".encode()
    ).hexdigest()


def derive_presentation(
        contract: ContrastiveOracleContract,
        target: ImageIdentity) -> Presentation:
    seed = _digest({
        "contract_digest": contract.digest(),
        "target": target.to_dict(),
        "order_policy": ORDER_POLICY,
    })
    positives = sorted(
        contract.positive_prototypes, key=lambda item: _rank(seed, "p", item))
    foils = sorted(
        contract.hard_negative_foils, key=lambda item: _rank(seed, "f", item))
    left_positive_indices = set(sorted(
        range(6), key=lambda index: hashlib.sha256(
            f"{seed}|side|{index}".encode()).hexdigest())[:3])
    pairs: list[PairAssignment] = []
    for index, (positive, foil) in enumerate(zip(positives, foils)):
        positive_left = index in left_positive_indices
        pairs.append(PairAssignment(
            pair_id=f"pair_{index:02d}",
            left=positive if positive_left else foil,
            right=foil if positive_left else positive,
            affirmative_side="left" if positive_left else "right",
            left_name=f"pair_{index:02d}_left.png",
            right_name=f"pair_{index:02d}_right.png",
        ))
    return Presentation(target, "subject.png", tuple(pairs))


def contrastive_output_schema() -> dict[str, Any]:
    pair_ids = [f"pair_{index:02d}" for index in range(6)]
    return {
        "type": "object",
        "properties": {
            "abstain": {"type": "boolean"},
            "abstention_reason": {"type": "string", "maxLength": MAX_EVIDENCE_LENGTH},
            "comparisons": {
                "type": "array", "minItems": 6, "maxItems": 6,
                "items": {
                    "type": "object",
                    "properties": {
                        "pair_id": {"type": "string", "enum": pair_ids},
                        "closer_to": {
                            "type": "string", "enum": ["left", "right", "unclear"]},
                        "abstain": {"type": "boolean"},
                        "abstention_reason": {
                            "type": "string", "maxLength": MAX_EVIDENCE_LENGTH},
                        "visible_evidence": {
                            "type": "string", "minLength": 1,
                            "maxLength": MAX_EVIDENCE_LENGTH},
                    },
                    "required": [
                        "pair_id", "closer_to", "abstain",
                        "abstention_reason", "visible_evidence"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["abstain", "abstention_reason", "comparisons"],
        "additionalProperties": False,
    }


def build_prompt(
        contract: ContrastiveOracleContract, presentation: Presentation) -> str:
    pair_lines = "\n".join(
        f"- {pair.pair_id}: left={pair.left_name}, right={pair.right_name}"
        for pair in presentation.pairs)
    return f"""\
Assess one neutrally named target only by contrast with six frozen reference
pairs. This is a comparative visual judgment, not an absolute membership
score. Each pair contains one affirmative prototype and one hard-negative
foil; their left/right placement is hidden and exactly balanced. Filenames,
pair order, and left/right position carry no semantic signal.

AFFIRMATIVE PROSE CLAIM: {contract.affirmative_claim}
TARGET: {presentation.target_name}
PAIRS:
{pair_lines}

For every pair, decide whether the target is closer in claim-relevant visible
structure to the left image, the right image, or whether the comparison is
unclear. Cite concrete visible evidence. Set that pair's abstain flag for
occlusion, ambiguity, or insufficient evidence and choose unclear. Set the
turn-level abstain flag if the target cannot be assessed at all. Do not emit a
probability, numeric score, Bongard label, global class, or inferred problem
identity. Return every pair exactly once. The harness alone maps sides back to
families and applies its frozen supermajority rule.

CONTRACT_DIGEST: {contract.digest()}
PROMPT_POLICY: {PROMPT_POLICY}
ORDER_POLICY: {ORDER_POLICY}
DECISION_POLICY: {DECISION_POLICY}"""


@dataclass(frozen=True)
class PairVote:
    pair_id: str
    closer_to: str
    abstain: bool
    abstention_reason: str
    visible_evidence: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PairVote":
        required = {
            "pair_id", "closer_to", "abstain", "abstention_reason",
            "visible_evidence",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError("pair vote fields are invalid")
        pair_id = value["pair_id"]
        if pair_id not in {f"pair_{index:02d}" for index in range(6)}:
            raise ValueError("pair vote ID is invalid")
        closer = value["closer_to"]
        if closer not in {"left", "right", "unclear"}:
            raise ValueError("pair vote category is invalid")
        abstain = value["abstain"]
        if not isinstance(abstain, bool):
            raise ValueError("pair abstain must be boolean")
        reason = _text(value["abstention_reason"], "pair abstention_reason",
                       allow_empty=not abstain)
        evidence = _text(value["visible_evidence"], "visible_evidence")
        if abstain != (closer == "unclear"):
            raise ValueError("unclear and pair abstention must coincide")
        if not abstain and reason:
            raise ValueError("non-abstaining pair must have no abstention reason")
        return cls(pair_id, closer, abstain, reason, evidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "closer_to": self.closer_to,
            "abstain": self.abstain,
            "abstention_reason": self.abstention_reason,
            "visible_evidence": self.visible_evidence,
        }


def _parse_payload(payload: Mapping[str, Any]) \
        -> tuple[bool, str, tuple[PairVote, ...]]:
    if not isinstance(payload, Mapping) or set(payload) != {
            "abstain", "abstention_reason", "comparisons"}:
        raise ValueError("oracle payload fields are invalid")
    abstain = payload["abstain"]
    if not isinstance(abstain, bool):
        raise ValueError("turn abstain must be boolean")
    reason = _text(payload["abstention_reason"], "turn abstention_reason",
                   allow_empty=not abstain)
    if not abstain and reason:
        raise ValueError("non-abstaining turn must have no abstention reason")
    comparisons = payload["comparisons"]
    if not isinstance(comparisons, list) or len(comparisons) != 6:
        raise ValueError("oracle must return six comparisons")
    votes = tuple(PairVote.from_dict(item) for item in comparisons)
    if {vote.pair_id for vote in votes} != {
            f"pair_{index:02d}" for index in range(6)}:
        raise ValueError("oracle pair IDs are duplicated or incomplete")
    return abstain, reason, tuple(sorted(votes, key=lambda item: item.pair_id))


def _observation(
        contract: ContrastiveOracleContract, presentation: Presentation,
        turn_abstain: bool, turn_reason: str,
        votes: Sequence[PairVote]) -> G.Observation:
    provenance = (contract.digest(), _digest({
        "presentation": presentation.to_dict(),
        "votes": [vote.to_dict() for vote in votes],
    }))
    if turn_abstain:
        return G.Indeterminate("oracle-abstained", turn_reason, provenance)
    pair_by_id = {pair.pair_id: pair for pair in presentation.pairs}
    affirmative = 0
    foil = 0
    for vote in votes:
        if vote.abstain:
            continue
        if vote.closer_to == pair_by_id[vote.pair_id].affirmative_side:
            affirmative += 1
        else:
            foil += 1
    if affirmative >= 5:
        return G.Present(True, G.Unit.BOOLEAN, provenance)
    if foil >= 5:
        return G.Present(False, G.Unit.BOOLEAN, provenance)
    return G.Indeterminate(
        "oracle-no-supermajority",
        f"affirmative={affirmative}, foil={foil}, abstained={6-affirmative-foil}",
        provenance,
    )


def _named_binding(prompt: str, schema: Mapping[str, Any],
                   presentation: Presentation) -> dict[str, str]:
    identities = [
        {
            "name": name,
            "byte_count": identity.byte_count,
            "content_digest": identity.content_digest.removeprefix("sha256:"),
        }
        for name, identity in zip(
            presentation.ordered_names(), presentation.ordered_identities())
    ]
    view_digest = _raw_digest(identities)
    set_digest = "sha256:" + _raw_digest({
        "schema": codex_proposer.NAMED_IMAGE_INPUT_DIGEST_SCHEMA,
        "images": identities,
    })
    prompt_digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    schema_digest = _raw_digest(schema)
    envelope = {
        "schema": codex_proposer.NAMED_IMAGE_INPUT_DIGEST_SCHEMA,
        "task": prompt,
        "ordered_image_identities": identities,
        "image_view_digest": view_digest,
        "image_set_digest": set_digest,
        "prompt_digest": prompt_digest,
        "output_schema_digest": schema_digest,
    }
    return {
        "prompt_digest": prompt_digest,
        "output_schema_digest": schema_digest,
        "panel_view_digest": view_digest,
        "panel_set_digest": set_digest,
        "input_digest": _raw_digest(envelope),
    }


def _validate_receipt(
        receipt: Mapping[str, Any], contract: ContrastiveOracleContract,
        payload: Mapping[str, Any], binding: Mapping[str, str]) -> None:
    codex_proposer.validate_codex_receipt(receipt)
    expected = {
        "requested_model": contract.model,
        "requested_reasoning_effort": contract.reasoning_effort,
        "input_digest_schema": codex_proposer.NAMED_IMAGE_INPUT_DIGEST_SCHEMA,
        **binding,
        "task_digest": binding["prompt_digest"],
        "structured_output_digest": _raw_digest(payload),
    }
    for field, value in expected.items():
        if receipt.get(field) != value:
            raise ValueError(f"oracle receipt {field} does not bind the input/output")


@dataclass(frozen=True)
class ContrastiveOracleEvidence:
    contract_digest: str
    presentation: Presentation
    prompt_digest: str
    output_schema_digest: str
    payload: Mapping[str, Any]
    receipt: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": EVIDENCE_SCHEMA,
            "contract_digest": self.contract_digest,
            "presentation": self.presentation.to_dict(),
            "prompt_digest": self.prompt_digest,
            "output_schema_digest": self.output_schema_digest,
            "payload": dict(self.payload),
            "receipt": dict(self.receipt),
            "evidence_digest": self.digest(),
        }

    def digest(self) -> str:
        return _digest({
            "schema": EVIDENCE_SCHEMA,
            "contract_digest": self.contract_digest,
            "presentation": self.presentation.to_dict(),
            "prompt_digest": self.prompt_digest,
            "output_schema_digest": self.output_schema_digest,
            "payload": dict(self.payload),
            "receipt": dict(self.receipt),
        })


def _observation_from_dict(value: Mapping[str, Any]) -> G.Observation:
    if not isinstance(value, Mapping):
        raise ValueError("observation must be an object")
    status = value.get("status")
    if status == "present" and set(value) == {
            "status", "value", "unit", "provenance"}:
        return G.Present(value["value"], value["unit"], tuple(value["provenance"]))
    if status == "indeterminate" and set(value) == {
            "status", "mode", "detail", "provenance"}:
        return G.Indeterminate(
            value["mode"], value["detail"], tuple(value["provenance"]))
    if status == "error" and set(value) == {
            "status", "code", "detail", "provenance"}:
        return G.Error(value["code"], value["detail"], tuple(value["provenance"]))
    raise ValueError("serialized oracle observation is invalid")


@dataclass(frozen=True)
class ContrastiveOracleEvaluation:
    contract_digest: str
    target: ImageIdentity
    observation: G.Observation
    evidence: ContrastiveOracleEvidence | None

    def to_dict(self) -> dict[str, Any]:
        body = {
            "schema": EVALUATION_SCHEMA,
            "contract_digest": self.contract_digest,
            "target": self.target.to_dict(),
            "observation": self.observation.to_dict(),
            "evidence": self.evidence.to_dict() if self.evidence else None,
        }
        body["evaluation_digest"] = _digest(body)
        return body

    @classmethod
    def from_dict(
            cls, value: Mapping[str, Any],
            contract: ContrastiveOracleContract,
            *, target_png_path: str | None = None,
            ) -> "ContrastiveOracleEvaluation":
        required = {
            "schema", "contract_digest", "target", "observation",
            "evidence", "evaluation_digest",
        }
        if not isinstance(value, Mapping) or set(value) != required \
                or value.get("schema") != EVALUATION_SCHEMA:
            raise ValueError("oracle evaluation fields are invalid")
        unsigned = {key: item for key, item in value.items()
                    if key != "evaluation_digest"}
        if value["evaluation_digest"] != _digest(unsigned) \
                or value["contract_digest"] != contract.digest():
            raise ValueError("oracle evaluation digest or contract differs")
        target = ImageIdentity.from_dict(value["target"])
        if target_png_path is not None \
                and ImageIdentity.from_path(target_png_path) != target:
            raise ValueError("oracle replay target bytes differ")
        if not isinstance(value["evidence"], Mapping):
            # Failed transports are typed but intentionally have no receipt to replay.
            observation = _observation_from_dict(value["observation"])
            if not isinstance(observation, G.Error) or value["evidence"] is not None:
                raise ValueError("receipt-free evaluation must be an Error")
            return cls(contract.digest(), target, observation, None)
        evidence_value = value["evidence"]
        evidence_required = {
            "schema", "contract_digest", "presentation", "prompt_digest",
            "output_schema_digest", "payload", "receipt", "evidence_digest",
        }
        if set(evidence_value) != evidence_required \
                or evidence_value.get("schema") != EVIDENCE_SCHEMA:
            raise ValueError("oracle evidence fields are invalid")
        presentation = derive_presentation(contract, target)
        if evidence_value["presentation"] != presentation.to_dict() \
                or evidence_value["contract_digest"] != contract.digest():
            raise ValueError("oracle evidence presentation differs")
        prompt = build_prompt(contract, presentation)
        schema = contrastive_output_schema()
        binding = _named_binding(prompt, schema, presentation)
        if evidence_value["prompt_digest"] != binding["prompt_digest"] \
                or evidence_value["output_schema_digest"] != \
                contract.output_schema_digest:
            raise ValueError("oracle evidence prompt/schema differs")
        payload = evidence_value["payload"]
        receipt = evidence_value["receipt"]
        if not isinstance(payload, Mapping) or not isinstance(receipt, Mapping):
            raise ValueError("oracle evidence payload/receipt is invalid")
        artifact = ContrastiveOracleEvidence(
            contract.digest(), presentation, binding["prompt_digest"],
            contract.output_schema_digest, dict(payload), dict(receipt))
        if evidence_value["evidence_digest"] != artifact.digest():
            raise ValueError("oracle evidence digest differs")
        _validate_receipt(receipt, contract, payload, binding)
        turn_abstain, turn_reason, votes = _parse_payload(payload)
        replayed = _observation(
            contract, presentation, turn_abstain, turn_reason, votes)
        stored = _observation_from_dict(value["observation"])
        if stored.to_dict() != replayed.to_dict():
            raise ValueError("oracle stored observation differs from replay")
        return cls(contract.digest(), target, replayed, artifact)


def replay_evaluation(
        contract: ContrastiveOracleContract, value: Mapping[str, Any],
        *, target_png_path: str | None = None) -> ContrastiveOracleEvaluation:
    return ContrastiveOracleEvaluation.from_dict(
        value, contract, target_png_path=target_png_path)


def _path_map(expected: Sequence[ImageIdentity], paths: Sequence[str],
              field: str) -> dict[str, str]:
    if isinstance(paths, (str, bytes)) or len(paths) != 6:
        raise ValueError(f"{field} requires exactly six paths")
    result: dict[str, str] = {}
    for path in paths:
        identity = ImageIdentity.from_path(path)
        if identity.content_digest in result:
            raise ValueError(f"{field} repeats an image")
        result[identity.content_digest] = path
    if set(result) != {item.content_digest for item in expected}:
        raise ValueError(f"{field} bytes differ from the frozen contract")
    return result


class CodexContrastiveOracle:
    """One-target-per-turn evaluator over a frozen contrastive contract."""

    def __init__(
            self, contract: ContrastiveOracleContract,
            positive_prototype_paths: Sequence[str],
            hard_negative_foil_paths: Sequence[str],
            *, minutes: int = 15, executable: str = "codex",
            verbose: bool = False) -> None:
        if not isinstance(contract, ContrastiveOracleContract):
            raise TypeError("contract must be a ContrastiveOracleContract")
        self.contract = contract
        self._paths = {
            **_path_map(contract.positive_prototypes,
                        positive_prototype_paths, "positive prototypes"),
            **_path_map(contract.hard_negative_foils,
                        hard_negative_foil_paths, "hard-negative foils"),
        }
        if isinstance(minutes, bool) or not isinstance(minutes, int) \
                or not 1 <= minutes <= 120:
            raise ValueError("minutes must be in [1, 120]")
        self.minutes = minutes
        self.executable = executable
        self.verbose = bool(verbose)

    def evaluate(self, target_png_path: str) -> ContrastiveOracleEvaluation:
        target = ImageIdentity.from_path(target_png_path)
        presentation = derive_presentation(self.contract, target)
        prompt = build_prompt(self.contract, presentation)
        schema = contrastive_output_schema()
        paths = [target_png_path] + [
            self._paths[item.content_digest]
            for pair in presentation.pairs for item in (pair.left, pair.right)
        ]
        names = list(presentation.ordered_names())
        try:
            result = codex_proposer.run_codex_named_images_structured(
                prompt, paths, names, schema,
                model=self.contract.model,
                reasoning_effort=self.contract.reasoning_effort,
                minutes=self.minutes,
                verbose=self.verbose,
                executable=self.executable,
            )
            payload = result.payload
            receipt = result.receipt.to_dict()
            binding = _named_binding(prompt, schema, presentation)
            _validate_receipt(receipt, self.contract, payload, binding)
            turn_abstain, turn_reason, votes = _parse_payload(payload)
            observation = _observation(
                self.contract, presentation, turn_abstain, turn_reason, votes)
            evidence = ContrastiveOracleEvidence(
                self.contract.digest(), presentation,
                binding["prompt_digest"], self.contract.output_schema_digest,
                dict(payload), receipt)
            return ContrastiveOracleEvaluation(
                self.contract.digest(), target, observation, evidence)
        except Exception as exc:
            return ContrastiveOracleEvaluation(
                self.contract.digest(), target,
                G.Error(
                    "oracle-evaluation-error",
                    f"{type(exc).__name__}: {exc}",
                    (self.contract.digest(), target.content_digest),
                ),
                None,
            )

    def observable_contract(
            self, observable_id: str = OBSERVABLE_ID) -> G.ObservableContract:
        return G.ObservableContract(
            observable_id=observable_id,
            value_type=G.ValueType.BOOLEAN,
            unit=G.Unit.BOOLEAN,
            referent="panel.open-vocabulary-concept",
            reducer=G.Reducer.IDENTITY,
            evaluator=lambda context: self.evaluate(context).observation,
            indeterminate_modes=(
                "oracle-abstained", "oracle-no-supermajority"),
            source=G.ObservableSource.ORACLE,
            version="v1",
        )


__all__ = [
    "CONTRACT_SCHEMA", "EVIDENCE_SCHEMA", "EVALUATION_SCHEMA",
    "PROMPT_POLICY", "ORDER_POLICY", "DECISION_POLICY", "OBSERVABLE_ID",
    "ImageIdentity", "ContrastiveOracleContract", "PairAssignment",
    "Presentation", "PairVote", "ContrastiveOracleEvidence",
    "ContrastiveOracleEvaluation", "CodexContrastiveOracle",
    "contrastive_output_schema", "derive_presentation", "build_prompt",
    "replay_evaluation",
]
