"""A frozen, abstaining contrastive oracle for open-vocabulary observables.

This is deliberately an exploratory HYBRID measurement, not a calibrated
probability and not a proof that pixels instantiate the prose ontology.  The
model sees one target and three neutral reference pairs twice.  It never sees
the reference roles; the second fresh turn reverses pair order and swaps every
left/right placement.  Only role-level agreement across the two turns counts.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import codex_proposer
import grounded_predicate_ir as G


CONTRACT_SCHEMA = "bongard.grounded-contrastive-oracle-contract/v2"
EVIDENCE_SCHEMA = "bongard.grounded-contrastive-oracle-evidence/v2"
EVALUATION_SCHEMA = "bongard.grounded-contrastive-oracle-evaluation/v2"
PROTOCOL_STATUS = "HYBRID-EXPLORATORY"
SELECTION_POLICY = "claim-seeded-content-rank-three-pairs/v1"
PRESENTATION_POLICY = "two-fresh-turns-full-side-and-order-swap/v1"
DECISION_POLICY = "two-of-three-unanimous-pair-role-zero-opposition/v1"
CALIBRATOR = None
OBSERVABLE_ID = "oracle/contrastive-open-vocabulary/v2"
PAIR_IDS = ("comparison_00", "comparison_01", "comparison_02")
CHOICES = ("left", "right", "tie", "unassessable")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MAX_CLAIM_BYTES = 4_000
MAX_EVIDENCE_BYTES = 2_000


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _raw_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _digest(value: Any) -> str:
    return "sha256:" + _raw_digest(value)


def _byte_digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _read_png(path: str) -> bytes:
    if not isinstance(path, str) or not path:
        raise ValueError("image path must be nonempty text")
    try:
        before = os.lstat(path)
    except OSError as exc:
        raise ValueError(f"cannot stat oracle image: {exc}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError("oracle images must be regular non-symlink files")
    if not 8 <= before.st_size <= codex_proposer.MAX_PANEL_PNG_BYTES:
        raise ValueError("oracle image size is outside the transport bound")
    try:
        with open(path, "rb") as handle:
            data = handle.read(codex_proposer.MAX_PANEL_PNG_BYTES + 1)
    except OSError as exc:
        raise ValueError(f"cannot read oracle image: {exc}") from exc
    after = os.lstat(path)
    identity = (before.st_dev, before.st_ino, before.st_size,
                before.st_mtime_ns, before.st_ctime_ns)
    if identity != (after.st_dev, after.st_ino, after.st_size,
                    after.st_mtime_ns, after.st_ctime_ns) \
            or len(data) != before.st_size or not data.startswith(PNG_SIGNATURE):
        raise ValueError("oracle image is not a stable PNG snapshot")
    return data


def _claim(value: Any) -> str:
    if not isinstance(value, str) or "\x00" in value:
        raise ValueError("claim must be text")
    value = " ".join(value.split())
    if not value or len(value.encode("utf-8")) > MAX_CLAIM_BYTES:
        raise ValueError("claim is empty or oversized")
    # A claim is model-visible, so it may not smuggle the harness role map.
    if re.search(r"\b(?:exemplar|prototype|anchor|foil|positive|negative)\b",
                 value, re.IGNORECASE):
        raise ValueError("claim must not disclose reference roles or labels")
    return value


@dataclass(frozen=True)
class ImageBinding:
    content_digest: str
    byte_count: int
    source_path: str | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if re.fullmatch(r"sha256:[0-9a-f]{64}", self.content_digest) is None:
            raise ValueError("invalid image content digest")
        if isinstance(self.byte_count, bool) or not isinstance(
                self.byte_count, int) or self.byte_count < 8:
            raise ValueError("invalid image byte count")

    @classmethod
    def from_path(cls, path: str) -> "ImageBinding":
        data = _read_png(path)
        return cls(_byte_digest(data), len(data), path)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ImageBinding":
        if not isinstance(value, Mapping) or set(value) != {
                "content_digest", "byte_count"}:
            raise ValueError("image binding fields differ")
        return cls(value["content_digest"], value["byte_count"])

    def to_dict(self) -> dict[str, Any]:
        return {"content_digest": self.content_digest,
                "byte_count": self.byte_count}

    def validate_path(self, path: str | None = None) -> str:
        resolved = self.source_path if path is None else path
        if resolved is None:
            raise ValueError("frozen image has no bound execution path")
        current = ImageBinding.from_path(resolved)
        if current.to_dict() != self.to_dict():
            raise ValueError("oracle image bytes differ from frozen binding")
        return resolved


@dataclass(frozen=True)
class FrozenPair:
    pair_key: str
    exemplar: ImageBinding
    foil: ImageBinding
    first_exemplar_side: str

    def __post_init__(self) -> None:
        if re.fullmatch(r"pair-[0-2]", self.pair_key) is None \
                or self.first_exemplar_side not in {"left", "right"}:
            raise ValueError("invalid frozen pair")

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_key": self.pair_key,
            "exemplar": self.exemplar.to_dict(),
            "foil": self.foil.to_dict(),
            "first_exemplar_side": self.first_exemplar_side,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FrozenPair":
        if not isinstance(value, Mapping) or set(value) != {
                "pair_key", "exemplar", "foil", "first_exemplar_side"}:
            raise ValueError("frozen pair fields differ")
        return cls(
            value["pair_key"], ImageBinding.from_dict(value["exemplar"]),
            ImageBinding.from_dict(value["foil"]),
            value["first_exemplar_side"])


def contrastive_output_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "comparisons": {
                "type": "array", "minItems": 3, "maxItems": 3,
                "items": {
                    "type": "object",
                    "properties": {
                        "pair_id": {"type": "string", "enum": list(PAIR_IDS)},
                        "choice": {"type": "string", "enum": list(CHOICES)},
                        "evidence": {"type": "string", "minLength": 1,
                                     "maxLength": MAX_EVIDENCE_BYTES},
                    },
                    "required": ["pair_id", "choice", "evidence"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["comparisons"],
        "additionalProperties": False,
    }


def _rank(seed: str, role: str, image: ImageBinding) -> str:
    return _raw_digest({"seed": seed, "role": role,
                        "image": image.to_dict()})


@dataclass(frozen=True)
class ContrastiveOracleContract:
    claim: str
    pairs: tuple[FrozenPair, ...]
    exemplar_pool_digest: str
    foil_pool_digest: str
    model: str
    reasoning_effort: str
    output_schema_digest: str
    protocol_status: str = PROTOCOL_STATUS
    calibrator: None = CALIBRATOR
    selection_policy: str = SELECTION_POLICY
    presentation_policy: str = PRESENTATION_POLICY
    decision_policy: str = DECISION_POLICY

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim", _claim(self.claim))
        if len(self.pairs) != 3 or tuple(
                pair.pair_key for pair in self.pairs) != (
                    "pair-0", "pair-1", "pair-2"):
            raise ValueError("contract must freeze exactly three ordered pairs")
        digests = [image.content_digest for pair in self.pairs
                   for image in (pair.exemplar, pair.foil)]
        if len(set(digests)) != 6:
            raise ValueError("selected references must be byte-distinct")
        if self.protocol_status != PROTOCOL_STATUS or self.calibrator is not None \
                or self.selection_policy != SELECTION_POLICY \
                or self.presentation_policy != PRESENTATION_POLICY \
                or self.decision_policy != DECISION_POLICY:
            raise ValueError("unsupported contrastive-oracle policy")
        if self.reasoning_effort not in codex_proposer.REASONING_EFFORTS:
            raise ValueError("invalid reasoning effort")
        if not isinstance(self.model, str) or re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", self.model) is None:
            raise ValueError("invalid model")
        if self.output_schema_digest != _digest(contrastive_output_schema()):
            raise ValueError("output schema digest differs")

    @classmethod
    def create(cls, claim: str, exemplar_paths: Sequence[str],
               foil_paths: Sequence[str],
               model: str = codex_proposer.DEFAULT_CODEX_MODEL,
               reasoning: str = codex_proposer.DEFAULT_REASONING_EFFORT,
               **kwargs: Any) -> "ContrastiveOracleContract":
        # Accept the repository's longer spelling without weakening ambiguity.
        if "reasoning_effort" in kwargs:
            if reasoning != codex_proposer.DEFAULT_REASONING_EFFORT:
                raise TypeError("reasoning and reasoning_effort both supplied")
            reasoning = kwargs.pop("reasoning_effort")
        if kwargs:
            raise TypeError(f"unexpected contract options: {sorted(kwargs)}")
        normalized_claim = _claim(claim)
        if isinstance(exemplar_paths, (str, bytes)) or len(exemplar_paths) != 6 \
                or isinstance(foil_paths, (str, bytes)) or len(foil_paths) != 6:
            raise ValueError("contract creation requires six exemplars and six foils")
        exemplars = tuple(ImageBinding.from_path(path) for path in exemplar_paths)
        foils = tuple(ImageBinding.from_path(path) for path in foil_paths)
        all_digests = [item.content_digest for item in exemplars + foils]
        if len(set(all_digests)) != 12:
            raise ValueError("all twelve reference images must be byte-distinct")
        exemplar_pool = sorted(
            (item.to_dict() for item in exemplars),
            key=lambda item: item["content_digest"])
        foil_pool = sorted(
            (item.to_dict() for item in foils),
            key=lambda item: item["content_digest"])
        seed = _digest({
            "schema": CONTRACT_SCHEMA,
            "claim": normalized_claim,
            "model": model,
            "reasoning_effort": reasoning,
            "selection_policy": SELECTION_POLICY,
            "exemplars": exemplar_pool,
            "foils": foil_pool,
        })
        selected_exemplars = sorted(
            exemplars, key=lambda item: _rank(seed, "family-a", item))[:3]
        selected_foils = sorted(
            foils, key=lambda item: _rank(seed, "family-b", item))[:3]
        pairs: list[FrozenPair] = []
        for index, (exemplar, foil) in enumerate(
                zip(selected_exemplars, selected_foils)):
            side = "left" if int(_raw_digest({
                "seed": seed, "pair": index, "side": "opaque",
            }), 16) % 2 == 0 else "right"
            pairs.append(FrozenPair(f"pair-{index}", exemplar, foil, side))
        return cls(
            normalized_claim, tuple(pairs), _digest(exemplar_pool),
            _digest(foil_pool), model, reasoning,
            _digest(contrastive_output_schema()))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) \
            -> "ContrastiveOracleContract":
        fields = {
            "schema", "claim", "pairs", "exemplar_pool_digest",
            "foil_pool_digest", "model", "reasoning_effort",
            "output_schema_digest", "protocol_status", "calibrator",
            "selection_policy", "presentation_policy", "decision_policy",
        }
        if not isinstance(value, Mapping) or set(value) != fields \
                or value.get("schema") != CONTRACT_SCHEMA \
                or not isinstance(value.get("pairs"), list):
            raise ValueError("contrastive oracle contract fields differ")
        return cls(
            claim=value["claim"],
            pairs=tuple(FrozenPair.from_dict(item) for item in value["pairs"]),
            exemplar_pool_digest=value["exemplar_pool_digest"],
            foil_pool_digest=value["foil_pool_digest"],
            model=value["model"],
            reasoning_effort=value["reasoning_effort"],
            output_schema_digest=value["output_schema_digest"],
            protocol_status=value["protocol_status"],
            calibrator=value["calibrator"],
            selection_policy=value["selection_policy"],
            presentation_policy=value["presentation_policy"],
            decision_policy=value["decision_policy"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CONTRACT_SCHEMA,
            "claim": self.claim,
            "pairs": [pair.to_dict() for pair in self.pairs],
            "exemplar_pool_digest": self.exemplar_pool_digest,
            "foil_pool_digest": self.foil_pool_digest,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "output_schema_digest": self.output_schema_digest,
            "protocol_status": self.protocol_status,
            "calibrator": self.calibrator,
            "selection_policy": self.selection_policy,
            "presentation_policy": self.presentation_policy,
            "decision_policy": self.decision_policy,
        }

    def digest(self) -> str:
        return _digest(self.to_dict())

    @property
    def bundle_digest(self) -> str:
        return self.digest()


@dataclass(frozen=True)
class PresentationPair:
    slot_id: str
    pair_key: str
    left: ImageBinding
    right: ImageBinding
    left_role: str

    @property
    def right_role(self) -> str:
        return "foil" if self.left_role == "anchor" else "anchor"


def _presentation(contract: ContrastiveOracleContract,
                  trial: int) -> tuple[PresentationPair, ...]:
    if trial not in {0, 1}:
        raise ValueError("trial must be zero or one")
    pairs = contract.pairs if trial == 0 else tuple(reversed(contract.pairs))
    result = []
    for slot, pair in enumerate(pairs):
        exemplar_left = pair.first_exemplar_side == "left"
        if trial == 1:
            exemplar_left = not exemplar_left
        result.append(PresentationPair(
            PAIR_IDS[slot], pair.pair_key,
            pair.exemplar if exemplar_left else pair.foil,
            pair.foil if exemplar_left else pair.exemplar,
            "anchor" if exemplar_left else "foil",
        ))
    return tuple(result)


def build_prompt(contract: ContrastiveOracleContract) -> str:
    pair_lines = "\n".join(
        f"- {pair_id}: compare {pair_id}_left.png with {pair_id}_right.png"
        for pair_id in PAIR_IDS)
    return f"""\
Compare one target with three pairs of neutral references under the exact
visible claim below. For each pair choose which reference the target more
closely resembles in claim-relevant visible structure. Choose tie when neither
is closer and unassessable when the images do not support a responsible
comparison. Cite only concrete visible evidence. Reference filenames,
left/right position, and pair order carry no semantic meaning. Do not infer a
Bongard label, reference role, class prior, probability, score, threshold,
candidate concept, or information about any image not shown in this turn.

FROZEN CLAIM: {contract.claim}
TARGET: target.png
PAIRS:
{pair_lines}

Return each comparison exactly once. This is an uncalibrated exploratory
contrastive measurement; do not turn it into an absolute confidence score.
PROTOCOL: {PROTOCOL_STATUS}; {PRESENTATION_POLICY}; {DECISION_POLICY}
CONTRACT_DIGEST: {contract.digest()}"""


@dataclass(frozen=True)
class PairResponse:
    slot_id: str
    choice: str
    evidence: str


def _parse_payload(payload: Mapping[str, Any]) -> tuple[PairResponse, ...]:
    if not isinstance(payload, Mapping) or set(payload) != {"comparisons"}:
        raise ValueError("oracle payload fields differ from schema")
    values = payload["comparisons"]
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError("oracle payload must contain three comparisons")
    parsed = []
    for value in values:
        if not isinstance(value, Mapping) or set(value) != {
                "pair_id", "choice", "evidence"}:
            raise ValueError("comparison fields differ from schema")
        slot = value["pair_id"]
        choice = value["choice"]
        evidence = value["evidence"]
        if slot not in PAIR_IDS or choice not in CHOICES \
                or not isinstance(evidence, str) or not evidence.strip() \
                or len(evidence.encode("utf-8")) > MAX_EVIDENCE_BYTES:
            raise ValueError("invalid comparison response")
        parsed.append(PairResponse(slot, choice, evidence))
    if {item.slot_id for item in parsed} != set(PAIR_IDS):
        raise ValueError("comparison IDs are duplicated or incomplete")
    return tuple(sorted(parsed, key=lambda item: item.slot_id))


def _named_binding(prompt: str, schema: Mapping[str, Any], target: ImageBinding,
                   presentation: Sequence[PresentationPair]) -> dict[str, str]:
    named = [("target.png", target)] + [
        (name, image)
        for pair in presentation
        for name, image in (
            (f"{pair.slot_id}_left.png", pair.left),
            (f"{pair.slot_id}_right.png", pair.right))
    ]
    identities = [{
        "name": name, "byte_count": image.byte_count,
        "content_digest": image.content_digest.removeprefix("sha256:"),
    } for name, image in named]
    view = _raw_digest(identities)
    image_set = "sha256:" + _raw_digest({
        "schema": codex_proposer.NAMED_IMAGE_INPUT_DIGEST_SCHEMA,
        "images": identities,
    })
    prompt_digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    schema_digest = _raw_digest(schema)
    envelope = {
        "schema": codex_proposer.NAMED_IMAGE_INPUT_DIGEST_SCHEMA,
        "task": prompt,
        "ordered_image_identities": identities,
        "image_view_digest": view,
        "image_set_digest": image_set,
        "prompt_digest": prompt_digest,
        "output_schema_digest": schema_digest,
    }
    return {
        "prompt_digest": prompt_digest,
        "output_schema_digest": schema_digest,
        "panel_view_digest": view,
        "panel_set_digest": image_set,
        "input_digest": _raw_digest(envelope),
    }


def _receipt_dict(receipt: Any) -> dict[str, Any]:
    if hasattr(receipt, "to_dict"):
        receipt = receipt.to_dict()
    if not isinstance(receipt, Mapping) or not receipt:
        raise ValueError("oracle transport receipt is missing")
    return dict(receipt)


def _validate_receipt(receipt: Mapping[str, Any], payload: Mapping[str, Any],
                      binding: Mapping[str, str],
                      contract: ContrastiveOracleContract) -> None:
    if receipt.get("source") == "codex-cli":
        codex_proposer.validate_codex_receipt(receipt)
    expected = {
        "requested_model": contract.model,
        "requested_reasoning_effort": contract.reasoning_effort,
        "input_digest_schema": codex_proposer.NAMED_IMAGE_INPUT_DIGEST_SCHEMA,
        "task_digest": binding["prompt_digest"],
        "structured_output_digest": _raw_digest(payload),
        **binding,
    }
    # Production receipts must bind everything.  Offline fixtures may provide
    # the same fields without manufacturing a complete Codex CLI receipt.
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"oracle receipt {key} does not bind the call")


def _validate_fresh_trial_receipts(
        receipts: Sequence[Mapping[str, Any]]) -> None:
    if len(receipts) != 2:
        raise ValueError("contrastive oracle requires exactly two trial receipts")
    sources = [receipt.get("source") for receipt in receipts]
    if sources == ["codex-cli", "codex-cli"]:
        thread_ids = [receipt.get("thread_id") for receipt in receipts]
        if any(not isinstance(item, str) or not item for item in thread_ids) \
                or len(set(thread_ids)) != 2:
            raise ValueError(
                "contrastive oracle production trials are not fresh threads")
    elif sources != ["offline-fixture", "offline-fixture"]:
        raise ValueError(
            "contrastive oracle trial receipt sources are unsupported or mixed")


def _normalize_and_decode(
        contract: ContrastiveOracleContract, target: ImageBinding,
        responses_by_trial: Sequence[Sequence[PairResponse]],
        ) -> tuple[tuple[tuple[str, str], ...], G.Observation]:
    if len(responses_by_trial) != 2:
        raise ValueError("decoder requires exactly two fresh trial payloads")
    role_votes: dict[str, list[str]] = {
        pair.pair_key: [] for pair in contract.pairs}
    for trial, responses in enumerate(responses_by_trial):
        presentation = _presentation(contract, trial)
        by_slot = {item.slot_id: item for item in presentation}
        if {response.slot_id for response in responses} != set(PAIR_IDS):
            raise ValueError("trial responses do not cover the presentation")
        for response in responses:
            placed = by_slot[response.slot_id]
            if response.choice in {"tie", "unassessable"}:
                role = "unknown"
            elif response.choice == "left":
                role = placed.left_role
            else:
                role = placed.right_role
            role_votes[placed.pair_key].append(role)
    normalized: list[tuple[str, str]] = []
    for pair in contract.pairs:
        votes = role_votes[pair.pair_key]
        vote = votes[0] if len(votes) == 2 and votes[0] == votes[1] \
            and votes[0] in {"anchor", "foil"} else "unknown"
        normalized.append((pair.pair_key, vote))
    frozen = tuple(normalized)
    anchor_count = sum(vote == "anchor" for _, vote in frozen)
    foil_count = sum(vote == "foil" for _, vote in frozen)
    provenance = (contract.digest(), _digest({
        "target": target.to_dict(), "normalized_votes": frozen}))
    if anchor_count >= 2 and foil_count == 0:
        observation: G.Observation = G.Present(
            True, G.Unit.BOOLEAN, provenance)
    elif foil_count >= 2 and anchor_count == 0:
        observation = G.Present(False, G.Unit.BOOLEAN, provenance)
    else:
        observation = G.Indeterminate(
            "contrastive-oracle-inconclusive",
            f"anchor={anchor_count}, foil={foil_count}, "
            f"unknown={3-anchor_count-foil_count}", provenance)
    return frozen, observation


@dataclass(frozen=True)
class ContrastiveOracleEvidence:
    contract_digest: str
    target: ImageBinding
    trials: tuple[Mapping[str, Any], Mapping[str, Any]]
    normalized_votes: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        body = {
            "schema": EVIDENCE_SCHEMA,
            "protocol_status": PROTOCOL_STATUS,
            "calibrator": None,
            "contract_digest": self.contract_digest,
            "target": self.target.to_dict(),
            "trials": [dict(item) for item in self.trials],
            "normalized_votes": [list(item) for item in self.normalized_votes],
        }
        body["evidence_digest"] = _digest(body)
        return body


@dataclass(frozen=True)
class ContrastiveOracleEvaluation:
    observation: G.Observation
    evidence: ContrastiveOracleEvidence | None
    contract_digest: str = ""
    target: ImageBinding | None = None

    def to_dict(self) -> dict[str, Any]:
        contract_digest = self.contract_digest or (
            self.evidence.contract_digest if self.evidence is not None else "")
        target = self.target or (
            self.evidence.target if self.evidence is not None else None)
        body = {
            "schema": EVALUATION_SCHEMA,
            "contract_digest": contract_digest,
            "target": target.to_dict() if target is not None else None,
            "observation": self.observation.to_dict(),
            "evidence": self.evidence.to_dict() if self.evidence is not None else None,
        }
        body["evaluation_digest"] = _digest(body)
        return body

    @classmethod
    def from_dict(
            cls, value: Mapping[str, Any],
            contract: ContrastiveOracleContract, *, target_png_path: str,
            ) -> "ContrastiveOracleEvaluation":
        return replay_evaluation(
            contract, value, target_png_path=target_png_path)


def _observation_from_dict(value: Mapping[str, Any]) -> G.Observation:
    if not isinstance(value, Mapping):
        raise ValueError("stored observation must be an object")
    status = value.get("status")
    if status == "present" and set(value) == {
            "status", "value", "unit", "provenance"}:
        return G.Present(
            value["value"], value["unit"], tuple(value["provenance"]))
    if status == "indeterminate" and set(value) == {
            "status", "mode", "detail", "provenance"}:
        return G.Indeterminate(
            value["mode"], value["detail"], tuple(value["provenance"]))
    if status == "error" and set(value) == {
            "status", "code", "detail", "provenance"}:
        return G.Error(
            value["code"], value["detail"], tuple(value["provenance"]))
    raise ValueError("stored oracle observation has invalid fields")


def _presentation_dict(
        presentation: Sequence[PresentationPair]) -> list[dict[str, Any]]:
    return [{
        "slot_id": pair.slot_id,
        "pair_key": pair.pair_key,
        "left": pair.left.to_dict(),
        "right": pair.right.to_dict(),
    } for pair in presentation]


def replay_evaluation(
        contract: ContrastiveOracleContract, value: Mapping[str, Any],
        *, target_png_path: str) -> ContrastiveOracleEvaluation:
    """Cold-replay two stored oracle turns without invoking a model."""
    fields = {
        "schema", "contract_digest", "target", "observation", "evidence",
        "evaluation_digest",
    }
    if not isinstance(value, Mapping) or set(value) != fields \
            or value.get("schema") != EVALUATION_SCHEMA:
        raise ValueError("stored oracle evaluation fields differ")
    unsigned = {key: item for key, item in value.items()
                if key != "evaluation_digest"}
    if value["evaluation_digest"] != _digest(unsigned) \
            or value["contract_digest"] != contract.digest():
        raise ValueError("stored oracle evaluation digest or contract differs")
    target = ImageBinding.from_path(target_png_path)
    if value["target"] != target.to_dict():
        raise ValueError("stored oracle target bytes differ")
    stored_observation = _observation_from_dict(value["observation"])
    evidence_value = value["evidence"]
    if evidence_value is None:
        if not isinstance(stored_observation, G.Error):
            raise ValueError("only a typed Error may omit oracle evidence")
        return ContrastiveOracleEvaluation(
            stored_observation, None, contract.digest(), target)
    evidence_fields = {
        "schema", "protocol_status", "calibrator", "contract_digest",
        "target", "trials", "normalized_votes", "evidence_digest",
    }
    if not isinstance(evidence_value, Mapping) \
            or set(evidence_value) != evidence_fields \
            or evidence_value.get("schema") != EVIDENCE_SCHEMA \
            or evidence_value.get("protocol_status") != PROTOCOL_STATUS \
            or evidence_value.get("calibrator") is not None \
            or evidence_value.get("contract_digest") != contract.digest() \
            or evidence_value.get("target") != target.to_dict():
        raise ValueError("stored oracle evidence identity differs")
    unsigned_evidence = {key: item for key, item in evidence_value.items()
                         if key != "evidence_digest"}
    if evidence_value["evidence_digest"] != _digest(unsigned_evidence):
        raise ValueError("stored oracle evidence digest differs")
    trials_value = evidence_value["trials"]
    if not isinstance(trials_value, list) or len(trials_value) != 2:
        raise ValueError("stored oracle evidence requires two trials")
    prompt = build_prompt(contract)
    schema = contrastive_output_schema()
    responses_by_trial: list[tuple[PairResponse, ...]] = []
    rebuilt_trials: list[Mapping[str, Any]] = []
    for trial, trial_value in enumerate(trials_value):
        trial_fields = {
            "trial", "presentation", "prompt_digest",
            "output_schema_digest", "payload", "receipt",
        }
        if not isinstance(trial_value, Mapping) \
                or set(trial_value) != trial_fields \
                or trial_value.get("trial") != trial:
            raise ValueError("stored oracle trial fields or order differ")
        presentation = _presentation(contract, trial)
        expected_presentation = _presentation_dict(presentation)
        if trial_value["presentation"] != expected_presentation:
            raise ValueError("stored oracle swapped presentation differs")
        payload = trial_value["payload"]
        receipt = trial_value["receipt"]
        if not isinstance(payload, Mapping) or not isinstance(receipt, Mapping):
            raise ValueError("stored oracle payload or receipt is invalid")
        responses = _parse_payload(payload)
        binding = _named_binding(prompt, schema, target, presentation)
        if trial_value["prompt_digest"] != binding["prompt_digest"] \
                or trial_value["output_schema_digest"] != \
                binding["output_schema_digest"]:
            raise ValueError("stored oracle prompt or schema binding differs")
        _validate_receipt(receipt, payload, binding, contract)
        responses_by_trial.append(responses)
        rebuilt_trials.append({
            "trial": trial,
            "presentation": expected_presentation,
            "prompt_digest": binding["prompt_digest"],
            "output_schema_digest": binding["output_schema_digest"],
            "payload": dict(payload),
            "receipt": dict(receipt),
        })
    _validate_fresh_trial_receipts(
        [trial["receipt"] for trial in rebuilt_trials])
    normalized, replayed_observation = _normalize_and_decode(
        contract, target, responses_by_trial)
    if evidence_value["normalized_votes"] != [list(item) for item in normalized]:
        raise ValueError("stored oracle normalized votes differ")
    if stored_observation.to_dict() != replayed_observation.to_dict():
        raise ValueError("stored oracle observation differs from decoded trials")
    evidence = ContrastiveOracleEvidence(
        contract.digest(), target,
        (rebuilt_trials[0], rebuilt_trials[1]), normalized)
    if evidence.to_dict() != dict(evidence_value):
        raise ValueError("stored oracle evidence does not reproduce")
    return ContrastiveOracleEvaluation(
        replayed_observation, evidence, contract.digest(), target)


def _paths_for(contract: ContrastiveOracleContract,
               exemplar_paths: Sequence[str] | None,
               foil_paths: Sequence[str] | None) -> dict[str, str]:
    selected = [image for pair in contract.pairs
                for image in (pair.exemplar, pair.foil)]
    if exemplar_paths is None and foil_paths is None:
        result = {image.content_digest: image.validate_path()
                  for image in selected}
        return result
    if exemplar_paths is None or foil_paths is None \
            or len(exemplar_paths) != 6 or len(foil_paths) != 6:
        raise ValueError("rebinding requires six exemplar and six foil paths")
    exemplar_bindings = tuple(
        ImageBinding.from_path(path) for path in exemplar_paths)
    foil_bindings = tuple(ImageBinding.from_path(path) for path in foil_paths)
    all_digests = [item.content_digest
                   for item in exemplar_bindings + foil_bindings]
    if len(set(all_digests)) != 12:
        raise ValueError("rebound reference pools are not byte-distinct")
    exemplar_pool = sorted(
        (item.to_dict() for item in exemplar_bindings),
        key=lambda item: item["content_digest"])
    foil_pool = sorted(
        (item.to_dict() for item in foil_bindings),
        key=lambda item: item["content_digest"])
    if _digest(exemplar_pool) != contract.exemplar_pool_digest \
            or _digest(foil_pool) != contract.foil_pool_digest:
        raise ValueError("rebound reference pools differ from frozen digests")
    result = {}
    for path in tuple(exemplar_paths) + tuple(foil_paths):
        identity = ImageBinding.from_path(path)
        result[identity.content_digest] = path
    if any(image.content_digest not in result for image in selected):
        raise ValueError("rebinding omits a frozen selected reference")
    for image in selected:
        image.validate_path(result[image.content_digest])
    return result


class CodexContrastiveOracle:
    def __init__(self, contract: ContrastiveOracleContract,
                 exemplar_paths: Sequence[str] | None = None,
                 foil_paths: Sequence[str] | None = None, *, minutes: int = 15,
                 executable: str = "codex", verbose: bool = False) -> None:
        self.contract = contract
        self._binding_error: Exception | None = None
        try:
            self._paths = _paths_for(contract, exemplar_paths, foil_paths)
        except Exception as exc:
            # Evaluation is the typed measurement boundary.  Preserve a bad
            # reference binding as Error instead of leaking an exception or
            # silently treating it as a negative observation.
            self._paths = {}
            self._binding_error = exc
        if isinstance(minutes, bool) or not isinstance(minutes, int) \
                or not 1 <= minutes <= 120:
            raise ValueError("minutes must be in [1, 120]")
        self.minutes = minutes
        self.executable = executable
        self.verbose = bool(verbose)

    def evaluate(self, target_png_path: str) -> ContrastiveOracleEvaluation:
        contract_digest = self.contract.digest()
        target: ImageBinding | None = None
        try:
            target = ImageBinding.from_path(target_png_path)
            if self._binding_error is not None:
                raise self._binding_error
            prompt = build_prompt(self.contract)
            schema = contrastive_output_schema()
            trial_records: list[Mapping[str, Any]] = []
            responses_by_trial: list[tuple[PairResponse, ...]] = []
            for trial in (0, 1):
                presentation = _presentation(self.contract, trial)
                # Revalidate all bytes immediately before every fresh turn.
                target.validate_path(target_png_path)
                paths = [target_png_path]
                names = ["target.png"]
                for pair in presentation:
                    paths.extend((
                        pair.left.validate_path(self._paths[pair.left.content_digest]),
                        pair.right.validate_path(self._paths[pair.right.content_digest]),
                    ))
                    names.extend((f"{pair.slot_id}_left.png",
                                  f"{pair.slot_id}_right.png"))
                result = codex_proposer.run_codex_named_images_structured(
                    prompt, paths, names, schema,
                    model=self.contract.model,
                    reasoning_effort=self.contract.reasoning_effort,
                    minutes=self.minutes, executable=self.executable,
                    verbose=self.verbose)
                payload = result.payload
                responses = _parse_payload(payload)
                receipt = _receipt_dict(result.receipt)
                binding = _named_binding(
                    prompt, schema, target, presentation)
                _validate_receipt(
                    receipt, payload, binding, self.contract)
                responses_by_trial.append(responses)
                trial_records.append({
                    "trial": trial,
                    "presentation": _presentation_dict(presentation),
                    "prompt_digest": binding["prompt_digest"],
                    "output_schema_digest": binding["output_schema_digest"],
                    "payload": dict(payload),
                    "receipt": receipt,
                })
            _validate_fresh_trial_receipts(
                [trial["receipt"] for trial in trial_records])
            # Catch changes after the second transport as binding failures.
            target.validate_path(target_png_path)
            for pair in self.contract.pairs:
                pair.exemplar.validate_path(self._paths[pair.exemplar.content_digest])
                pair.foil.validate_path(self._paths[pair.foil.content_digest])
            normalized, observation = _normalize_and_decode(
                self.contract, target, responses_by_trial)
            evidence = ContrastiveOracleEvidence(
                contract_digest, target,
                (trial_records[0], trial_records[1]), normalized)
            return ContrastiveOracleEvaluation(
                observation, evidence, contract_digest, target)
        except Exception as exc:
            return ContrastiveOracleEvaluation(G.Error(
                "contrastive-oracle-error", f"{type(exc).__name__}: {exc}",
                (contract_digest,)), None, contract_digest, target)

    def observable_contract(
            self, observable_id: str = OBSERVABLE_ID) -> G.ObservableContract:
        return G.ObservableContract(
            observable_id=observable_id,
            value_type=G.ValueType.BOOLEAN,
            unit=G.Unit.BOOLEAN,
            # This measures operational resemblance under one frozen bundle;
            # it does not assert direct truth of an unformalized prose kind.
            referent=(
                "panel.operational-resemblance-to-frozen-claim-reference-bundle"),
            reducer=G.Reducer.IDENTITY,
            evaluator=lambda context: self.evaluate(context).observation,
            indeterminate_modes=("contrastive-oracle-inconclusive",),
            source=G.ObservableSource.ORACLE,
            version="v2")


__all__ = [
    "CALIBRATOR", "CONTRACT_SCHEMA", "DECISION_POLICY", "EVIDENCE_SCHEMA",
    "EVALUATION_SCHEMA",
    "OBSERVABLE_ID", "PAIR_IDS", "PRESENTATION_POLICY", "PROTOCOL_STATUS",
    "SELECTION_POLICY", "CHOICES", "ImageBinding", "FrozenPair",
    "ContrastiveOracleContract", "ContrastiveOracleEvidence",
    "ContrastiveOracleEvaluation", "CodexContrastiveOracle",
    "contrastive_output_schema", "build_prompt", "replay_evaluation",
]
