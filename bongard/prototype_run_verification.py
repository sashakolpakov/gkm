"""Canonical persistence and cold replay for support-prototype episodes.

This is deliberately a separate outer archive from the HYBRID vision-run
format in :mod:`bongard.run_verification`.  A prototype run has one model call
(the support-only, finite-catalog proposal) and deterministic Python extraction
and evaluation everywhere else.  The verifier below never invokes Codex or
Lean.  It reconstructs the proposal receipt, reruns the neutral extractor from
the exact PNG bytes supplied by blob id, refits the support prototypes, replays the support
gate, recompiles the closed Python IR, and binds query evidence to the generic
model-free artifact chain.

The outer record is JSON-only.  Exact panel-byte preimages are supplied to the
verifier by blob id, just as they are supplied to the generic cold verifier.
Proposal-error and support-rejected records therefore remain reproducible from
the record plus their released support blobs; query blobs are required only for
a completed run.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

from bongard.artifacts import (
    ArtifactTamperError,
    BlobRef,
    ProposalFreeze,
    SupportCommitment,
    TruthEvidenceRecord,
    VerifiedRunArchive,
    canonical_digest,
    canonical_json,
    verify_archive_data,
    verify_proposal_freeze_data,
    verify_support_commitment_data,
)
from bongard.benchmark import (
    EpisodePlan,
    EpisodeResult,
    EpisodeStatus,
    PROTOCOL_VERSION,
    SUPPORT_PROTOTYPE_PREDICATE_MODE,
    SupportGatePolicy,
    SupportGateResult,
)
from bongard.evidence import Disposition
from bongard.legs.neutral_features import (
    extract_neutral_features,
    feature_group_catalog,
    project_neutral_feature_extraction,
)
from bongard.prototype_artifacts import (
    FeatureExtractionPreimage,
    PrototypeFreezePolicy,
    PrototypePreQueryFreeze,
    PrototypeQueryArtifact,
    PrototypeSupportReplayArtifact,
)
from bongard.prototype_calibration import PrototypeCalibrationRecord
from bongard.prototype_episode import (
    HeadlessPrototypeEpisode,
    PROTOTYPE_EPISODE_SCHEMA,
)
from bongard.proposer import (
    PROPOSAL_SCHEMA_VERSION,
    REJECTED_PROPOSAL_ATTEMPT_SCHEMA_VERSION,
    ProposalError,
    RejectedProposalAttempt,
    RejectedProposalError,
    RuleProposal,
    SupportPanelIdentity,
    parse_rule_proposal,
    pure_only_rule_proposal_schema,
    pure_proposer_prompt,
)
from bongard.release import OfficialReleaseDescriptor, ReleaseIdentityError
from bongard.synthesis import compile_prototype_proposal
from bongard.transport import (
    CODEX_RECEIPT_SCHEMA,
    STRUCTURED_INPUT_DIGEST_SCHEMA,
    CodexProposerFailure,
    CodexReceipt,
    semantic_panel_set_digest,
    validate_codex_receipt,
)


PROTOTYPE_OUTER_RUN_SCHEMA = "gkm.bongard-support-prototype-run.v1"
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_PREFIXED_HEX = re.compile(r"sha256:[0-9a-f]{64}\Z")
_BASE_PLAN_FIELDS = {
    "version",
    "task_id",
    "family",
    "split",
    "regime",
    "run_id",
    "verifier_id",
    "seed_digest",
    "corpus_digest",
    "task_manifest_digest",
    "support_commitment_digest",
    "latent_query_digest",
    "label_commitment_digest",
    "predicate_mode",
    "predicate_policy_digest",
}
_EPISODE_FIELDS = {
    "version",
    "task_id",
    "family",
    "split",
    "regime",
    "run_id",
    "plan_digest",
    "status",
    "score",
    "phases",
    "artifact_chain",
    "failure",
}
_SCORE_FIELDS = {
    "image_correct",
    "image_total",
    "image_accuracy",
    "puzzle_correct",
    "puzzle_accuracy",
    "determinate",
    "abstentions",
    "errors",
}
_OUTER_FIELDS = {
    "schema",
    "corpus_manifest_digest",
    "split_source_digest",
    "official_release",
    "calibration",
    "plan",
    "support_commitment",
    "episode",
    "prototype",
    "run_archive",
    "exposure",
    "record_digest",
}
_EPISODE_ARTIFACT_FIELDS = {
    "schema",
    "predicate_mode",
    "predicate_policy",
    "predicate_policy_digest",
    "proposal",
    "rejected_proposal_attempt",
    "pre_query_commitment",
    "observations",
}
_PROTOTYPE_FIELDS = _EPISODE_ARTIFACT_FIELDS | {
    "support_gate",
    "proposal_freeze",
}
_COMPLETE_PHASES = (
    "plan_committed",
    "support_released",
    "proposal_fixed",
    "support_gate_replayed",
    "proposal_frozen",
    "query_released",
    "predictions_committed",
    "labels_revealed",
    "cold_replay_verified",
)
_SUPPORT_REJECTED_PHASES = (
    "plan_committed",
    "support_released",
    "proposal_fixed",
    "support_gate_replayed",
    "proposal_frozen",
    "support_gate_rejected",
)
_PROPOSAL_ERROR_PHASES = (
    "plan_committed",
    "support_released",
    "proposal_failed",
)


class PrototypeRunVerificationError(ValueError):
    """A persisted prototype run fails deterministic reconstruction."""


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise PrototypeRunVerificationError(f"{label} must be a JSON object")
    return value


def _expect_fields(
    value: Mapping[str, Any], expected: set[str], label: str
) -> None:
    if set(value) != expected:
        raise PrototypeRunVerificationError(
            f"{label} fields differ from schema: "
            f"missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _hex(value: object, label: str, *, prefixed: bool = False) -> str:
    pattern = _PREFIXED_HEX if prefixed else _HEX
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        kind = "sha256: content address" if prefixed else "lowercase SHA-256"
        raise PrototypeRunVerificationError(f"{label} must be a {kind}")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise PrototypeRunVerificationError(
            f"{label} must be a canonical non-empty string"
        )
    return value


def _blob_from_data(value: object, label: str) -> BlobRef:
    data = _mapping(value, label)
    _expect_fields(data, {"blob_id", "sha256", "byte_count", "media_type"}, label)
    try:
        result = BlobRef(
            blob_id=data["blob_id"],
            sha256=data["sha256"],
            byte_count=data["byte_count"],
            media_type=data["media_type"],
        )
    except (TypeError, ValueError) as exc:
        raise PrototypeRunVerificationError(f"{label} is invalid: {exc}") from exc
    if result.to_data() != dict(data):
        raise PrototypeRunVerificationError(f"{label} changes during decoding")
    return result


def _receipt_from_data(value: object, label: str) -> CodexReceipt:
    data = _mapping(value, f"{label} receipt")
    try:
        validate_codex_receipt(data)
    except (CodexProposerFailure, TypeError, ValueError) as exc:
        raise PrototypeRunVerificationError(
            f"{label} has an invalid Codex receipt: {exc}"
        ) from exc
    converted = dict(data)
    converted["event_types"] = tuple(data["event_types"])
    converted["item_types"] = tuple(data["item_types"])
    try:
        receipt = CodexReceipt(**converted)
    except TypeError as exc:  # pragma: no cover - receipt validator owns fields.
        raise PrototypeRunVerificationError(
            f"{label} receipt cannot be reconstructed"
        ) from exc
    if receipt.to_dict() != dict(data):
        raise PrototypeRunVerificationError(f"{label} receipt round-trip drift")
    return receipt


def _payload_digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(dict(payload))).hexdigest()


def _text_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="strict")).hexdigest()


def _schema_digest(schema: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(dict(schema))).hexdigest()


def _verify_receipt_payload(
    receipt: CodexReceipt,
    payload: Mapping[str, Any],
    schema: Mapping[str, Any],
    prompt: str,
    *,
    identities: list[dict[str, object]],
    panel_set_digest: str,
) -> None:
    if receipt.schema != CODEX_RECEIPT_SCHEMA:
        raise PrototypeRunVerificationError("proposal receipt schema differs")
    if receipt.input_digest_schema != STRUCTURED_INPUT_DIGEST_SCHEMA:
        raise PrototypeRunVerificationError(
            "proposal receipt uses the wrong input-digest schema"
        )
    prompt_digest = _text_digest(prompt)
    schema_digest = _schema_digest(schema)
    view_digest = canonical_digest(identities)
    if receipt.prompt_digest != prompt_digest or receipt.task_digest != prompt_digest:
        raise PrototypeRunVerificationError(
            "proposal receipt does not bind the reconstructed PURE prompt"
        )
    if receipt.output_schema_digest != schema_digest:
        raise PrototypeRunVerificationError(
            "proposal receipt does not bind the PURE output schema"
        )
    if receipt.structured_output_digest != _payload_digest(payload):
        raise PrototypeRunVerificationError(
            "proposal receipt does not bind its structured payload"
        )
    if receipt.panel_view_digest != view_digest:
        raise PrototypeRunVerificationError(
            "proposal receipt does not bind the ordered 6+6 byte identities"
        )
    if receipt.panel_set_digest != panel_set_digest:
        raise PrototypeRunVerificationError(
            "proposal receipt semantic panel set differs"
        )
    envelope = {
        "schema": STRUCTURED_INPUT_DIGEST_SCHEMA,
        "task": prompt,
        "ordered_panel_identities": identities,
        "panel_view_digest": view_digest,
        "panel_set_digest": panel_set_digest,
        "prompt_digest": prompt_digest,
        "output_schema_digest": schema_digest,
    }
    if receipt.input_digest != canonical_digest(envelope):
        raise PrototypeRunVerificationError(
            "proposal receipt input envelope does not reproduce"
        )


def _normalize_blob_preimages(value: Mapping[str, bytes]) -> dict[str, bytes]:
    if not isinstance(value, Mapping) or any(
        not isinstance(blob_id, str) for blob_id in value
    ):
        raise PrototypeRunVerificationError(
            "blob_bytes_by_id must map blob ids to exact bytes"
        )
    result: dict[str, bytes] = {}
    for blob_id, payload in value.items():
        _text(blob_id, "blob byte id")
        if not isinstance(payload, bytes) or not payload:
            raise PrototypeRunVerificationError(
                f"blob byte preimage {blob_id} must be nonempty bytes"
            )
        result[blob_id] = payload
    return result


def _verify_blob_preimages(
    preimages: Mapping[str, bytes], expected: Mapping[str, BlobRef]
) -> None:
    if set(preimages) != set(expected):
        missing = sorted(set(expected) - set(preimages))
        extra = sorted(set(preimages) - set(expected))
        raise PrototypeRunVerificationError(
            "blob byte ids differ from the released support/query BlobRefs: "
            f"missing={missing}, extra={extra}"
        )
    for blob_id, blob in expected.items():
        payload = preimages[blob_id]
        try:
            blob.verify_bytes(payload)
        except ArtifactTamperError as exc:
            raise PrototypeRunVerificationError(
                f"blob preimage {blob_id} differs from its outer BlobRef"
            ) from exc


def _catalog_for_policy(policy: PrototypeFreezePolicy) -> dict[str, str]:
    canonical = {item.group_id: item.description for item in feature_group_catalog()}
    ids = tuple(item.feature_group_id for item in policy.allowed_feature_groups)
    if any(group_id not in canonical for group_id in ids):
        raise PrototypeRunVerificationError(
            "prototype policy contains a feature group outside the canonical catalog"
        )
    return {group_id: canonical[group_id] for group_id in ids}


def _verify_calibration_policy(
    calibration_value: object, policy: PrototypeFreezePolicy
) -> PrototypeCalibrationRecord:
    try:
        calibration = PrototypeCalibrationRecord.from_data(
            _mapping(calibration_value, "calibration")
        )
    except (TypeError, ValueError) as exc:
        raise PrototypeRunVerificationError(
            f"calibration record is invalid: {exc}"
        ) from exc
    content = calibration.content_data()
    catalog = _mapping(content["feature_catalog"], "calibration feature catalog")
    if catalog.get("digest") != policy.feature_catalog_digest:
        raise PrototypeRunVerificationError(
            "calibration feature catalog differs from prototype policy"
        )
    groups = {
        group["group_id"]: group
        for group in content["groups"]
        if isinstance(group, Mapping)
    }
    allowed = {
        item.feature_group_id: item for item in policy.allowed_feature_groups
    }
    if set(groups) != set(allowed):
        raise PrototypeRunVerificationError(
            "calibration and policy contain different feature groups"
        )
    for group_id, selected in allowed.items():
        calibrated = groups[group_id]
        if (
            calibrated.get("feature_space_digest") != selected.feature_space_digest
            or calibrated.get("selected_margin") != selected.decision_margin
        ):
            raise PrototypeRunVerificationError(
                f"calibration and policy differ for {group_id}"
            )
    return calibration


def _verify_plan_episode(
    plan_value: object,
    episode_value: object,
    *,
    support: SupportCommitment,
    policy: PrototypeFreezePolicy,
    corpus_manifest_digest: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any], str]:
    plan = _mapping(plan_value, "public prototype plan")
    episode = _mapping(episode_value, "outer prototype episode")
    _expect_fields(plan, _BASE_PLAN_FIELDS, "public prototype plan")
    _expect_fields(episode, _EPISODE_FIELDS, "outer prototype episode")
    if plan["version"] != PROTOCOL_VERSION or episode["version"] != PROTOCOL_VERSION:
        raise PrototypeRunVerificationError("benchmark protocol version differs")
    for name in ("task_id", "family", "split", "regime", "run_id"):
        if plan[name] != episode[name]:
            raise PrototypeRunVerificationError(
                f"public plan {name} differs from episode"
            )
    if plan["predicate_mode"] != SUPPORT_PROTOTYPE_PREDICATE_MODE:
        raise PrototypeRunVerificationError("public plan is not support-prototype mode")
    if plan["predicate_policy_digest"] != policy.digest():
        raise PrototypeRunVerificationError(
            "public plan predicate policy digest differs"
        )
    if plan["corpus_digest"] != corpus_manifest_digest.removeprefix("sha256:"):
        raise PrototypeRunVerificationError("public plan corpus digest differs")
    if support.run_id != plan["run_id"] or support.issued_by != plan["verifier_id"]:
        raise PrototypeRunVerificationError("support commitment identity differs from plan")
    if support.corpus_digest != plan["corpus_digest"]:
        raise PrototypeRunVerificationError("support commitment corpus differs from plan")
    if support.digest() != plan["support_commitment_digest"]:
        raise PrototypeRunVerificationError("support commitment digest differs from plan")
    if canonical_digest(dict(plan)) != episode["plan_digest"]:
        raise PrototypeRunVerificationError("episode plan digest does not reproduce")
    status = _text(episode["status"], "episode status")
    if status not in {
        EpisodeStatus.COMPLETE.value,
        EpisodeStatus.SUPPORT_REJECTED.value,
        EpisodeStatus.PROPOSAL_ERROR.value,
    }:
        raise PrototypeRunVerificationError(
            f"unsupported prototype episode status {status!r}"
        )
    return plan, episode, status


def _support_named_blobs(
    support: SupportCommitment,
) -> tuple[tuple[str, BlobRef], ...]:
    by_id = {item.panel.blob_id: item for item in support.support}
    expected = tuple(
        [(f"pos_{index}.png", f"support-positive-{index}", True) for index in range(6)]
        + [(f"neg_{index}.png", f"support-negative-{index}", False) for index in range(6)]
    )
    if set(by_id) != {blob_id for _name, blob_id, _positive in expected}:
        raise PrototypeRunVerificationError(
            "support BlobRefs are not the canonical six-positive/six-negative view"
        )
    result: list[tuple[str, BlobRef]] = []
    for name, blob_id, positive in expected:
        item = by_id[blob_id]
        if item.positive is not positive:
            raise PrototypeRunVerificationError(
                f"support BlobRef {blob_id} has the wrong side"
            )
        result.append((name, item.panel))
    return tuple(result)


def _semantic_digest_from_bytes(
    named: Sequence[tuple[str, BlobRef]], preimages: Mapping[str, bytes]
) -> str:
    with tempfile.TemporaryDirectory(prefix="bongard-prototype-cold-proposal-") as root:
        paths: list[str] = []
        for name, blob in named:
            path = Path(root) / name
            path.write_bytes(preimages[blob.blob_id])
            paths.append(str(path))
        try:
            return semantic_panel_set_digest(paths)
        except (CodexProposerFailure, OSError, ValueError) as exc:
            raise PrototypeRunVerificationError(
                f"cannot reproduce proposal semantic panel digest: {exc}"
            ) from exc


def _parse_pure_proposal(
    value: object,
    *,
    policy: PrototypeFreezePolicy,
    support: SupportCommitment,
    preimages: Mapping[str, bytes],
) -> RuleProposal:
    proposal_data = _mapping(value, "PURE proposal")
    expected_fields = {
        "schema",
        "positive_description",
        "panel_descriptions",
        "view",
        "observable_requests",
        "formula_template",
        "hybrid_claim",
        "confidence",
        "model_payload",
        "receipt",
    }
    _expect_fields(proposal_data, expected_fields, "PURE proposal")
    if proposal_data["schema"] != PROPOSAL_SCHEMA_VERSION:
        raise PrototypeRunVerificationError("PURE proposal schema differs")
    payload = _mapping(proposal_data["model_payload"], "PURE proposal payload")
    receipt = _receipt_from_data(proposal_data["receipt"], "PURE proposal")
    catalog = _catalog_for_policy(policy)
    try:
        proposal = parse_rule_proposal(
            payload,
            receipt=receipt,
            observable_catalog=catalog,
        )
    except (ProposalError, TypeError, ValueError) as exc:
        raise PrototypeRunVerificationError(
            f"archived PURE proposal cannot be parsed: {exc}"
        ) from exc
    if proposal.to_dict() != dict(proposal_data):
        raise PrototypeRunVerificationError(
            "PURE proposal fields do not reproduce from payload and receipt"
        )
    if proposal.is_hybrid or len(proposal.formula_atoms) != 1 or len(
        proposal.observable_requests
    ) != 1:
        raise PrototypeRunVerificationError(
            "prototype proposal is not one exact PURE catalog selection"
        )
    selected = proposal.formula_atoms[0]
    request = proposal.observable_requests[0]
    if request.observable_id != selected or request.arguments:
        raise PrototypeRunVerificationError(
            "PURE proposal selection/arguments differ from the closed catalog language"
        )
    named = _support_named_blobs(support)
    identities = [
        {
            "name": name,
            "byte_count": blob.byte_count,
            "content_digest": blob.sha256,
        }
        for name, blob in named
    ]
    _verify_receipt_payload(
        receipt,
        payload,
        pure_only_rule_proposal_schema(catalog),
        pure_proposer_prompt(catalog),
        identities=identities,
        panel_set_digest=_semantic_digest_from_bytes(named, preimages),
    )
    return proposal


def _fresh_preimage(
    archived: FeatureExtractionPreimage, group_id: str
) -> FeatureExtractionPreimage:
    fresh_full = extract_neutral_features(archived.panel_bytes)
    fresh_projected = project_neutral_feature_extraction(fresh_full, group_id)
    return FeatureExtractionPreimage.from_extraction(
        archived.panel_bytes, fresh_projected
    )


def _verify_rejected_support_extractions(
    support: SupportCommitment,
    preimages: Mapping[str, bytes],
) -> None:
    """Replay the pre-proposal neutral guard when no group was accepted."""

    for index, item in enumerate(support.support):
        payload = preimages[item.panel.blob_id]
        try:
            fresh = extract_neutral_features(payload)
        except (TypeError, ValueError) as exc:
            raise PrototypeRunVerificationError(
                f"proposal-error support extraction {index} cannot be rerun: {exc}"
            ) from exc
        if fresh.evidence.disposition is not Disposition.PRESENT:
            raise PrototypeRunVerificationError(
                "proposal-error record claims Codex was called after a non-present "
                f"support extraction at slot {index}"
            )


def _verify_fresh_preimage(
    archived: FeatureExtractionPreimage,
    *,
    group_id: str,
    expected_bytes: bytes,
    label: str,
) -> None:
    if archived.panel_bytes != expected_bytes:
        raise PrototypeRunVerificationError(
            f"{label} embedded extraction bytes differ from outer BlobRef preimage"
        )
    try:
        fresh = _fresh_preimage(archived, group_id)
    except (TypeError, ValueError) as exc:
        raise PrototypeRunVerificationError(
            f"{label} neutral extraction cannot be rerun: {exc}"
        ) from exc
    if fresh != archived:
        raise PrototypeRunVerificationError(
            f"{label} archived neutral extraction differs from fresh Python replay"
        )


def _verify_prequery(
    value: object,
    *,
    support: SupportCommitment,
    policy: PrototypeFreezePolicy,
    proposal: RuleProposal,
    preimages: Mapping[str, bytes],
) -> PrototypePreQueryFreeze:
    try:
        freeze = PrototypePreQueryFreeze.from_committed_data(
            _mapping(value, "prototype pre-query commitment"),
            support_commitment=support,
        )
    except (TypeError, ValueError) as exc:
        raise PrototypeRunVerificationError(
            f"prototype pre-query commitment is invalid: {exc}"
        ) from exc
    if freeze.policy != policy or freeze.policy_digest != policy.digest():
        raise PrototypeRunVerificationError(
            "pre-query freeze uses another prototype policy"
        )
    if freeze.selected_feature_group_id != proposal.formula_atoms[0]:
        raise PrototypeRunVerificationError(
            "PURE proposal selected group differs from pre-query freeze"
        )
    if freeze.semantic_proposal_digest != proposal.digest.removeprefix("sha256:"):
        raise PrototypeRunVerificationError(
            "pre-query freeze belongs to another PURE proposal"
        )
    by_digest = {
        item.panel.sha256: preimages[item.panel.blob_id]
        for item in support.support
    }
    for index, extraction in enumerate(
        freeze.positive_support + freeze.negative_support
    ):
        try:
            expected = by_digest[extraction.panel_digest]
        except KeyError as exc:
            raise PrototypeRunVerificationError(
                "pre-query support extraction is absent from outer support BlobRefs"
            ) from exc
        _verify_fresh_preimage(
            extraction,
            group_id=freeze.selected_feature_group_id,
            expected_bytes=expected,
            label=f"pre-query support extraction {index}",
        )
    # from_committed_data already refits centroids/formula from these now-fresh
    # packet preimages and checks the fixed six-plus-six side assignment.
    freeze.verify(support)
    return freeze


def _verify_compilation(
    *,
    proposal: RuleProposal,
    prequery: PrototypePreQueryFreeze,
    support: SupportCommitment,
    generic_freeze: ProposalFreeze,
    archive: VerifiedRunArchive | None,
) -> None:
    try:
        compiled = compile_prototype_proposal(
            proposal,
            prequery.feature_space,
            prequery.prototypes,
            prequery.positive_formula,
            issued_by=support.issued_by,
        )
    except (TypeError, ValueError) as exc:
        raise PrototypeRunVerificationError(
            f"prototype proposal cannot be canonically recompiled: {exc}"
        ) from exc
    if generic_freeze.formula != compiled.formula:
        raise PrototypeRunVerificationError(
            "generic proposal freeze formula differs from canonical prototype compilation"
        )
    if generic_freeze.proposer_digest != compiled.proposer_digest:
        raise PrototypeRunVerificationError(
            "generic proposal freeze proposer digest differs"
        )
    if generic_freeze.registry_digest != compiled.registry.digest():
        raise PrototypeRunVerificationError(
            "generic proposal freeze registry differs from Python compiler"
        )
    if generic_freeze.attachment_contract_digest != compiled.attachment_contract.digest():
        raise PrototypeRunVerificationError(
            "generic proposal freeze attachment contract differs from Python compiler"
        )
    if generic_freeze.support_commitment_digest != support.digest():
        raise PrototypeRunVerificationError(
            "generic proposal freeze support commitment differs"
        )
    if archive is not None:
        bundle = archive.bundle
        if bundle.freeze != generic_freeze:
            raise PrototypeRunVerificationError(
                "prototype proposal freeze differs from completed generic archive"
            )
        if bundle.attachment_contract != compiled.attachment_contract:
            raise PrototypeRunVerificationError(
                "generic archive attachment snapshot differs from Python compiler"
            )


def _verify_gate(
    value: object,
    *,
    support: SupportCommitment,
    proposal: RuleProposal,
    prequery: PrototypePreQueryFreeze,
    preimages: Mapping[str, bytes],
) -> tuple[str, tuple[str, ...]]:
    gate = _mapping(value, "prototype support gate")
    fields = {
        "version",
        "run_id",
        "proposal_digest",
        "support_commitment_digest",
        "policy",
        "ordered_entries",
        "counts",
        "result",
        "gate_digest",
    }
    _expect_fields(gate, fields, "prototype support gate")
    if gate["version"] != "support-replay-gate-artifact/v1":
        raise PrototypeRunVerificationError("support gate version differs")
    if gate["run_id"] != support.run_id:
        raise PrototypeRunVerificationError("support gate run differs")
    if gate["proposal_digest"] != proposal.digest.removeprefix("sha256:"):
        raise PrototypeRunVerificationError("support gate proposal differs")
    if gate["support_commitment_digest"] != support.digest():
        raise PrototypeRunVerificationError("support gate support commitment differs")
    if gate["policy"] != SupportGatePolicy.prototype().to_data():
        raise PrototypeRunVerificationError("support gate policy is not canonical prototype replay")
    entries = gate["ordered_entries"]
    if not isinstance(entries, list) or len(entries) != 12:
        raise PrototypeRunVerificationError("support gate is not twelve ordered entries")
    dispositions: list[Disposition] = []
    forward = 0
    reverse = 0
    artifact_digests: list[str] = []
    for index, (raw_entry, expected) in enumerate(
        zip(entries, support.support, strict=True)
    ):
        entry = _mapping(raw_entry, f"support gate entry {index}")
        _expect_fields(
            entry,
            {
                "slot_id",
                "panel",
                "positive",
                "evidence",
                "observer_artifact",
                "transport_attempted",
            },
            f"support gate entry {index}",
        )
        if (
            entry["slot_id"] != expected.panel.blob_id
            or entry["positive"] is not expected.positive
            or entry["transport_attempted"] is not True
            or _blob_from_data(entry["panel"], f"support gate panel {index}")
            != expected.panel
        ):
            raise PrototypeRunVerificationError(
                f"support gate entry {index} differs from outer support commitment"
            )
        observer_data = _mapping(
            entry["observer_artifact"], f"support replay artifact {index}"
        )
        if "positive" in observer_data:
            raise PrototypeRunVerificationError(
                f"support replay artifact {index} leaks a support-side field"
            )
        try:
            artifact = PrototypeSupportReplayArtifact.from_data(
                observer_data, freeze=prequery
            )
        except (TypeError, ValueError) as exc:
            raise PrototypeRunVerificationError(
                f"support replay artifact {index} is invalid: {exc}"
            ) from exc
        if artifact.panel_digest != expected.panel.sha256:
            raise PrototypeRunVerificationError(
                f"support replay artifact {index} names another panel"
            )
        _verify_fresh_preimage(
            artifact.extraction,
            group_id=prequery.selected_feature_group_id,
            expected_bytes=preimages[expected.panel.blob_id],
            label=f"support replay extraction {index}",
        )
        try:
            evidence = TruthEvidenceRecord.from_data(
                _mapping(entry["evidence"], f"support gate evidence {index}")
            )
        except (TypeError, ValueError) as exc:
            raise PrototypeRunVerificationError(
                f"support gate evidence {index} is invalid: {exc}"
            ) from exc
        reconstructed = TruthEvidenceRecord.from_evidence(
            artifact.evidence.to_evidence()
        )
        if evidence != reconstructed:
            raise PrototypeRunVerificationError(
                f"support gate evidence {index} differs from prototype replay"
            )
        disposition = artifact.evidence.disposition
        dispositions.append(disposition)
        forward += int(
            (expected.positive and disposition is Disposition.PRESENT)
            or (
                not expected.positive
                and disposition is Disposition.CERTIFIED_ABSENT
            )
        )
        reverse += int(
            (
                expected.positive
                and disposition is Disposition.CERTIFIED_ABSENT
            )
            or (not expected.positive and disposition is Disposition.PRESENT)
        )
        artifact_digests.append(artifact.digest())
    disposition_counts = Counter(dispositions)
    expected_counts = {
        "forward_matches": forward,
        "reverse_matches": reverse,
        "present": disposition_counts[Disposition.PRESENT],
        "nonmatch": disposition_counts[Disposition.CERTIFIED_ABSENT],
        "indeterminate": disposition_counts[Disposition.INDETERMINATE],
        "error": disposition_counts[Disposition.ERROR],
        "transport_attempts": 12,
    }
    if gate["counts"] != expected_counts:
        raise PrototypeRunVerificationError("support gate counts do not reproduce")
    if disposition_counts[Disposition.INDETERMINATE] or disposition_counts[
        Disposition.ERROR
    ]:
        result = SupportGateResult.OBSERVER_FAILURE.value
    elif forward == 12:
        result = SupportGateResult.ALIGNED.value
    elif reverse > forward:
        result = SupportGateResult.MISORIENTED.value
    else:
        result = SupportGateResult.UNSUPPORTED.value
    if gate["result"] != result:
        raise PrototypeRunVerificationError("support gate result does not reproduce")
    content = {key: gate[key] for key in fields if key != "gate_digest"}
    digest = canonical_digest(content)
    if gate["gate_digest"] != digest:
        raise PrototypeRunVerificationError("support gate digest does not reproduce")
    return result, tuple(artifact_digests)


def _expected_score(archive: VerifiedRunArchive) -> dict[str, object]:
    bundle = archive.bundle
    labels = {item.query_id: item.positive for item in bundle.labels.labels}
    predictions = {item.query_id: item for item in bundle.predictions.predictions}
    correct = sum(
        prediction.positive is not None
        and prediction.positive == labels[query_id]
        for query_id, prediction in predictions.items()
    )
    determinate = sum(item.positive is not None for item in predictions.values())
    errors = sum(
        item.disposition is Disposition.ERROR for item in predictions.values()
    )
    return {
        "image_correct": correct,
        "image_total": 2,
        "image_accuracy": correct / 2,
        "puzzle_correct": correct == 2,
        "puzzle_accuracy": float(correct == 2),
        "determinate": determinate,
        "abstentions": 2 - determinate,
        "errors": errors,
    }


def _verify_query_artifacts(
    value: object,
    *,
    archive: VerifiedRunArchive,
    prequery: PrototypePreQueryFreeze,
    preimages: Mapping[str, bytes],
) -> tuple[str, ...]:
    observations = _mapping(value, "prototype query observations")
    released = {item.query_id: item.panel for item in archive.bundle.release.queries}
    if set(observations) != set(released):
        raise PrototypeRunVerificationError(
            "prototype query artifacts do not cover exactly the released queries"
        )
    cold = {
        item.query_id: item for item in archive.bundle.cold_inputs.queries
    }
    digests: list[str] = []
    for public_id, blob in sorted(released.items()):
        try:
            artifact = PrototypeQueryArtifact.from_data(
                _mapping(observations[public_id], f"query artifact {public_id}"),
                freeze=prequery,
            )
        except (TypeError, ValueError) as exc:
            raise PrototypeRunVerificationError(
                f"query artifact {public_id} is invalid: {exc}"
            ) from exc
        if artifact.query_id != "query":
            raise PrototypeRunVerificationError(
                f"query artifact {public_id} does not preserve the neutral callback id"
            )
        if artifact.query_panel_digest != blob.sha256:
            raise PrototypeRunVerificationError(
                f"query artifact {public_id} names another released panel"
            )
        _verify_fresh_preimage(
            artifact.extraction,
            group_id=prequery.selected_feature_group_id,
            expected_bytes=preimages[blob.blob_id],
            label=f"query extraction {public_id}",
        )
        cold_query = cold[public_id]
        if len(cold_query.atom_inputs) != 1 or cold_query.atom_inputs[0].path != ():
            raise PrototypeRunVerificationError(
                f"query {public_id} cold evidence is not the one-atom prototype IR"
            )
        reconstructed = TruthEvidenceRecord.from_evidence(
            artifact.evidence.to_evidence()
        )
        if reconstructed != cold_query.atom_inputs[0].evidence:
            raise PrototypeRunVerificationError(
                f"query {public_id} prototype evidence differs from generic cold input"
            )
        digests.append(artifact.digest())
    return tuple(digests)


def _verify_rejected_attempt(
    value: object,
    *,
    policy: PrototypeFreezePolicy,
    support: SupportCommitment,
    preimages: Mapping[str, bytes],
) -> tuple[str, str]:
    data = _mapping(value, "rejected PURE proposal attempt")
    _expect_fields(
        data,
        {
            "schema",
            "proposal_schema",
            "model_payload",
            "receipt",
            "support_presentation",
            "parse_error",
            "attempt_digest",
        },
        "rejected PURE proposal attempt",
    )
    if (
        data["schema"] != REJECTED_PROPOSAL_ATTEMPT_SCHEMA_VERSION
        or data["proposal_schema"] != PROPOSAL_SCHEMA_VERSION
    ):
        raise PrototypeRunVerificationError("rejected PURE proposal schema differs")
    receipt = _receipt_from_data(data["receipt"], "rejected PURE proposal")
    raw_presentation = data["support_presentation"]
    if not isinstance(raw_presentation, list):
        raise PrototypeRunVerificationError(
            "rejected PURE support presentation must be a list"
        )
    presentation: list[SupportPanelIdentity] = []
    for raw in raw_presentation:
        item = _mapping(raw, "rejected support identity")
        _expect_fields(item, {"name", "byte_count", "content_digest"}, "rejected support identity")
        presentation.append(
            SupportPanelIdentity(
                item["name"], item["byte_count"], item["content_digest"]
            )
        )
    parse_error = _mapping(data["parse_error"], "rejected parse error")
    _expect_fields(parse_error, {"error_type", "reason"}, "rejected parse error")
    payload = _mapping(data["model_payload"], "rejected PURE payload")
    try:
        attempt = RejectedProposalAttempt(
            model_payload=payload,
            receipt=receipt,
            support_presentation=tuple(presentation),
            parse_error_type=parse_error["error_type"],
            parse_error_reason=parse_error["reason"],
        )
    except (TypeError, ValueError) as exc:
        raise PrototypeRunVerificationError(
            f"rejected PURE proposal attempt is invalid: {exc}"
        ) from exc
    if attempt.to_dict() != dict(data):
        raise PrototypeRunVerificationError(
            "rejected PURE proposal attempt does not round-trip"
        )
    catalog = _catalog_for_policy(policy)
    named = _support_named_blobs(support)
    expected_presentation = tuple(
        SupportPanelIdentity(name, blob.byte_count, blob.sha256)
        for name, blob in named
    )
    if tuple(presentation) != expected_presentation:
        raise PrototypeRunVerificationError(
            "rejected proposal support presentation differs from outer support"
        )
    _verify_receipt_payload(
        receipt,
        payload,
        pure_only_rule_proposal_schema(catalog),
        pure_proposer_prompt(catalog),
        identities=[item.to_dict() for item in presentation],
        panel_set_digest=_semantic_digest_from_bytes(named, preimages),
    )
    try:
        parsed = parse_rule_proposal(
            payload,
            receipt=receipt,
            observable_catalog=catalog,
        )
        if parsed.is_hybrid:
            raise ProposalError("PURE-only turn returned a HYBRID claim")
    except (ProposalError, TypeError, ValueError) as exc:
        if type(exc).__name__ != attempt.parse_error_type or (
            str(exc) or "proposal validation failed"
        ) != attempt.parse_error_reason:
            raise PrototypeRunVerificationError(
                "rejected PURE proposal parse error does not reproduce"
            ) from exc
    else:
        raise PrototypeRunVerificationError(
            "rejected PURE proposal now parses successfully"
        )
    return attempt.digest, receipt.receipt_digest


@dataclass(frozen=True)
class PrototypeRunVerification:
    run_id: str
    status: str
    record_digest: str
    plan_digest: str
    calibration_digest: str
    policy_digest: str
    proposal_digest: str | None
    proposal_receipt_digest: str | None
    pre_query_freeze_digest: str | None
    support_gate_digest: str | None
    archive_digest: str | None
    verified_blob_ids: tuple[str, ...]
    missing_blob_ids: tuple[str, ...]
    neutral_extraction_replays: int
    support_replay_artifact_digests: tuple[str, ...]
    query_artifact_digests: tuple[str, ...]
    unbound_claims: tuple[str, ...]
    archive: VerifiedRunArchive | None = field(default=None, repr=False, compare=False)

    def to_data(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "record_digest": self.record_digest,
            "plan_digest": self.plan_digest,
            "calibration_digest": self.calibration_digest,
            "policy_digest": self.policy_digest,
            "proposal_digest": self.proposal_digest,
            "proposal_receipt_digest": self.proposal_receipt_digest,
            "pre_query_freeze_digest": self.pre_query_freeze_digest,
            "support_gate_digest": self.support_gate_digest,
            "archive_digest": self.archive_digest,
            "verified_blob_ids": list(self.verified_blob_ids),
            "missing_blob_ids": list(self.missing_blob_ids),
            "neutral_extraction_replays": self.neutral_extraction_replays,
            "support_replay_artifact_digests": list(
                self.support_replay_artifact_digests
            ),
            "query_artifact_digests": list(self.query_artifact_digests),
            "unbound_claims": list(self.unbound_claims),
        }


def verify_prototype_run_data(
    record_value: Mapping[str, Any],
    *,
    blob_bytes_by_id: Mapping[str, bytes],
) -> PrototypeRunVerification:
    """Cold-verify one decoded prototype record without Codex or Lean."""

    record = _mapping(record_value, "prototype outer run")
    _expect_fields(record, _OUTER_FIELDS, "prototype outer run")
    if record["schema"] != PROTOTYPE_OUTER_RUN_SCHEMA:
        raise PrototypeRunVerificationError("prototype outer schema differs")
    record_digest = _hex(record["record_digest"], "record digest")
    content = {key: value for key, value in record.items() if key != "record_digest"}
    if canonical_digest(content) != record_digest:
        raise PrototypeRunVerificationError("prototype outer record digest mismatch")
    corpus_digest = _hex(
        record["corpus_manifest_digest"], "corpus manifest digest", prefixed=True
    )
    split_digest = record["split_source_digest"]
    if split_digest is not None:
        _hex(split_digest, "split source digest", prefixed=True)

    release_data = record["official_release"]
    if release_data is not None:
        try:
            release = OfficialReleaseDescriptor.from_dict(
                _mapping(release_data, "official release")
            )
        except (ReleaseIdentityError, TypeError, ValueError) as exc:
            raise PrototypeRunVerificationError(
                f"official release descriptor is invalid: {exc}"
            ) from exc
        if (
            release.corpus_manifest_sha256 != corpus_digest
            or release.split_sha256 != split_digest
        ):
            raise PrototypeRunVerificationError(
                "official release differs from outer corpus/split identity"
            )

    try:
        support = verify_support_commitment_data(record["support_commitment"])
    except (ArtifactTamperError, TypeError, ValueError) as exc:
        raise PrototypeRunVerificationError(
            f"outer support commitment is invalid: {exc}"
        ) from exc
    run_artifact = _mapping(record["prototype"], "prototype artifact")
    _expect_fields(
        run_artifact,
        _PROTOTYPE_FIELDS,
        "prototype artifact",
    )
    episode_artifact = run_artifact
    if (
        episode_artifact["schema"] != PROTOTYPE_EPISODE_SCHEMA
        or episode_artifact["predicate_mode"]
        != SUPPORT_PROTOTYPE_PREDICATE_MODE
    ):
        raise PrototypeRunVerificationError("prototype episode identity differs")
    try:
        policy = PrototypeFreezePolicy.from_data(
            _mapping(episode_artifact["predicate_policy"], "prototype policy")
        )
    except (TypeError, ValueError) as exc:
        raise PrototypeRunVerificationError(
            f"prototype policy is invalid: {exc}"
        ) from exc
    if episode_artifact["predicate_policy_digest"] != policy.digest():
        raise PrototypeRunVerificationError("prototype policy digest differs")
    calibration = _verify_calibration_policy(record["calibration"], policy)
    plan, episode, status = _verify_plan_episode(
        record["plan"],
        record["episode"],
        support=support,
        policy=policy,
        corpus_manifest_digest=corpus_digest,
    )
    if plan["split"] == "test" and release_data is None:
        raise PrototypeRunVerificationError(
            "test prototype run has no exact official release descriptor"
        )
    preimages = _normalize_blob_preimages(blob_bytes_by_id)
    support_blobs = {item.panel.blob_id: item.panel for item in support.support}

    archive: VerifiedRunArchive | None = None
    proposal_digest: str | None = None
    receipt_digest: str | None = None
    proposal_model: str | None = None
    prequery_digest: str | None = None
    gate_digest: str | None = None
    support_artifact_digests: tuple[str, ...] = ()
    query_artifact_digests: tuple[str, ...] = ()
    extraction_replays = 0

    if status == EpisodeStatus.PROPOSAL_ERROR.value:
        if tuple(episode["phases"]) != _PROPOSAL_ERROR_PHASES:
            raise PrototypeRunVerificationError(
                "proposal-error episode phases are not canonical"
            )
        if (
            episode_artifact["proposal"] is not None
            or episode_artifact["pre_query_commitment"] is not None
            or episode_artifact["observations"] != {}
            or run_artifact["support_gate"] is not None
            or run_artifact["proposal_freeze"] is not None
            or record["run_archive"] is not None
        ):
            raise PrototypeRunVerificationError(
                "proposal-error record contains post-proposal artifacts"
            )
        rejected = episode_artifact["rejected_proposal_attempt"]
        if rejected is None:
            raise PrototypeRunVerificationError(
                "proposal error has no replayable rejected structured attempt"
            )
        _verify_blob_preimages(preimages, support_blobs)
        _verify_rejected_support_extractions(support, preimages)
        extraction_replays += 12
        proposal_digest, receipt_digest = _verify_rejected_attempt(
            rejected,
            policy=policy,
            support=support,
            preimages=preimages,
        )
        proposal_model = _mapping(
            _mapping(rejected, "rejected PURE proposal attempt")["receipt"],
            "rejected PURE proposal receipt",
        )["requested_model"]
        rejected_parse_error = _mapping(
            _mapping(rejected, "rejected PURE proposal attempt")["parse_error"],
            "rejected PURE proposal parse error",
        )
        failure = _mapping(episode["failure"], "proposal-error failure")
        _expect_fields(
            failure,
            {"stage", "error_type", "reason"},
            "proposal-error failure",
        )
        if (
            failure["stage"] != "proposal"
            or failure["error_type"] != RejectedProposalError.__name__
            or failure["reason"] != rejected_parse_error["reason"]
        ):
            raise PrototypeRunVerificationError(
                "proposal-error failure does not reproduce the rejected attempt"
            )
        expected_score = {
            "image_correct": 0,
            "image_total": 2,
            "image_accuracy": 0.0,
            "puzzle_correct": False,
            "puzzle_accuracy": 0.0,
            "determinate": 0,
            "abstentions": 2,
            "errors": 2,
        }
        if episode["score"] != expected_score or episode["artifact_chain"] is not None:
            raise PrototypeRunVerificationError("proposal-error score/chain differs")
    else:
        if episode_artifact["rejected_proposal_attempt"] is not None:
            raise PrototypeRunVerificationError(
                "accepted proposal record also contains a rejected attempt"
            )
        if status == EpisodeStatus.COMPLETE.value:
            try:
                archive = verify_archive_data(
                    _mapping(record["run_archive"], "run archive")
                )
            except (ArtifactTamperError, TypeError, ValueError) as exc:
                raise PrototypeRunVerificationError(
                    f"generic run archive is invalid: {exc}"
                ) from exc
            if archive.bundle.support != support:
                raise PrototypeRunVerificationError(
                    "generic archive support differs from outer support"
                )
            _verify_blob_preimages(
                preimages,
                {
                    **support_blobs,
                    **{
                        item.panel.blob_id: item.panel
                        for item in archive.bundle.release.queries
                    },
                },
            )
        else:
            if record["run_archive"] is not None:
                raise PrototypeRunVerificationError(
                    "support-rejected run cannot contain a completed generic archive"
                )
            _verify_blob_preimages(preimages, support_blobs)
        proposal = _parse_pure_proposal(
            episode_artifact["proposal"],
            policy=policy,
            support=support,
            preimages=preimages,
        )
        proposal_digest = proposal.digest.removeprefix("sha256:")
        receipt_digest = proposal.receipt.receipt_digest
        proposal_model = proposal.receipt.requested_model
        prequery = _verify_prequery(
            episode_artifact["pre_query_commitment"],
            support=support,
            policy=policy,
            proposal=proposal,
            preimages=preimages,
        )
        prequery_digest = prequery.digest()
        extraction_replays += 12
        try:
            generic_freeze = verify_proposal_freeze_data(
                run_artifact["proposal_freeze"]
            )
        except (ArtifactTamperError, TypeError, ValueError) as exc:
            raise PrototypeRunVerificationError(
                f"standalone generic proposal freeze is invalid: {exc}"
            ) from exc
        gate_result, support_artifact_digests = _verify_gate(
            run_artifact["support_gate"],
            support=support,
            proposal=proposal,
            prequery=prequery,
            preimages=preimages,
        )
        extraction_replays += 12
        gate_digest = _mapping(
            run_artifact["support_gate"], "prototype support gate"
        )["gate_digest"]
        if generic_freeze.support_gate_digest != gate_digest:
            raise PrototypeRunVerificationError(
                "generic proposal freeze does not bind the replayed support gate"
            )
        _verify_compilation(
            proposal=proposal,
            prequery=prequery,
            support=support,
            generic_freeze=generic_freeze,
            archive=archive,
        )
        if status == EpisodeStatus.SUPPORT_REJECTED.value:
            if tuple(episode["phases"]) != _SUPPORT_REJECTED_PHASES:
                raise PrototypeRunVerificationError(
                    "support-rejected episode phases are not canonical"
                )
            if gate_result == SupportGateResult.ALIGNED.value:
                raise PrototypeRunVerificationError(
                    "support-rejected run contains an aligned support gate"
                )
            if episode_artifact["observations"] != {}:
                raise PrototypeRunVerificationError(
                    "support-rejected run contains query observations"
                )
            failure = _mapping(episode["failure"], "support-rejected failure")
            _expect_fields(
                failure,
                {"stage", "error_type", "reason"},
                "support-rejected failure",
            )
            if (
                failure["stage"] != "support_gate"
                or failure["error_type"] != "SupportGateRejected"
                or failure["reason"] != gate_result
            ):
                raise PrototypeRunVerificationError(
                    "support-rejected failure differs from replayed gate result"
                )
            expected_errors = (
                2 if gate_result == SupportGateResult.OBSERVER_FAILURE.value else 0
            )
            expected_score = {
                "image_correct": 0,
                "image_total": 2,
                "image_accuracy": 0.0,
                "puzzle_correct": False,
                "puzzle_accuracy": 0.0,
                "determinate": 0,
                "abstentions": 2,
                "errors": expected_errors,
            }
            if episode["score"] != expected_score or episode["artifact_chain"] is not None:
                raise PrototypeRunVerificationError(
                    "support-rejected score/chain differs from replayed gate"
                )
        else:
            assert archive is not None
            if tuple(episode["phases"]) != _COMPLETE_PHASES:
                raise PrototypeRunVerificationError(
                    "completed prototype episode phases are not canonical"
                )
            if gate_result != SupportGateResult.ALIGNED.value:
                raise PrototypeRunVerificationError(
                    "completed prototype episode lacks an aligned support gate"
                )
            if episode["failure"] is not None:
                raise PrototypeRunVerificationError(
                    "completed prototype episode contains a failure"
                )
            if episode["artifact_chain"] != archive.bundle.chain_data():
                raise PrototypeRunVerificationError(
                    "completed episode chain differs from generic archive"
                )
            if episode["score"] != _expected_score(archive):
                raise PrototypeRunVerificationError(
                    "completed episode score differs from generic cold replay"
                )
            query_artifact_digests = _verify_query_artifacts(
                episode_artifact["observations"],
                archive=archive,
                prequery=prequery,
                preimages=preimages,
            )
            extraction_replays += 2

    # Exposure is already included in the outer content address.  Cross-bind
    # the fields that affect this run; predecessor-ledger authenticity remains
    # external unless a caller supplies that ledger or an external anchor.
    exposure = record["exposure"]
    if exposure is not None:
        exposure_data = _mapping(exposure, "run exposure")
        for key, expected in (
            ("corpus_manifest_digest", corpus_digest),
            ("task_id", plan["task_id"]),
            ("plan_digest", episode["plan_digest"]),
        ):
            if exposure_data.get(key) != expected:
                raise PrototypeRunVerificationError(
                    f"run exposure {key} differs from prototype run"
                )
        if proposal_model is not None and exposure_data.get("model") != proposal_model:
            raise PrototypeRunVerificationError(
                "run exposure model differs from proposal receipt"
            )

    unbound = (
        "task-manifest membership and split assignment require the supplied complete corpus",
        "exposure-ledger predecessor authenticity requires the predecessor ledger or an external anchor",
        "the calibration task pixels are committed by the calibration record only through its source/task digests",
    )
    return PrototypeRunVerification(
        run_id=support.run_id,
        status=status,
        record_digest=record_digest,
        plan_digest=episode["plan_digest"],
        calibration_digest=calibration.digest(),
        policy_digest=policy.digest(),
        proposal_digest=proposal_digest,
        proposal_receipt_digest=receipt_digest,
        pre_query_freeze_digest=prequery_digest,
        support_gate_digest=gate_digest,
        archive_digest=archive.archive_digest if archive is not None else None,
        verified_blob_ids=tuple(sorted(preimages)),
        missing_blob_ids=(),
        neutral_extraction_replays=extraction_replays,
        support_replay_artifact_digests=support_artifact_digests,
        query_artifact_digests=query_artifact_digests,
        unbound_claims=unbound,
        archive=archive,
    )


def verify_prototype_run_bytes(
    payload: bytes | str,
    *,
    blob_bytes_by_id: Mapping[str, bytes],
) -> PrototypeRunVerification:
    """Reject duplicate keys/noncanonical JSON, then cold-verify the record."""

    raw = payload.encode("utf-8") if isinstance(payload, str) else payload

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PrototypeRunVerificationError(
                    f"duplicate JSON object key {key!r}"
                )
            result[key] = value
        return result

    try:
        decoded = json.loads(raw, object_pairs_hook=unique)
    except PrototypeRunVerificationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        raise PrototypeRunVerificationError(
            f"cannot decode prototype run JSON: {exc}"
        ) from exc
    if not isinstance(decoded, Mapping):
        raise PrototypeRunVerificationError("prototype run root must be an object")
    if canonical_json(decoded) != raw:
        raise PrototypeRunVerificationError(
            "prototype run bytes are not canonical JSON"
        )
    return verify_prototype_run_data(
        decoded,
        blob_bytes_by_id=blob_bytes_by_id,
    )


def build_prototype_run_record(
    *,
    corpus_manifest_digest: str,
    split_source_digest: str | None,
    official_release: OfficialReleaseDescriptor | Mapping[str, Any] | None,
    calibration: PrototypeCalibrationRecord,
    plan: EpisodePlan,
    result: EpisodeResult,
    episode: HeadlessPrototypeEpisode,
    exposure: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Assemble the exact outer shape consumed by the cold verifier.

    Panel bytes remain outside JSON and are supplied to the cold verifier by
    BlobRef id.  Existing paths are never serialized.
    """

    release_data: Mapping[str, Any] | None
    if isinstance(official_release, OfficialReleaseDescriptor):
        release_data = official_release.to_dict()
    else:
        release_data = official_release
    prototype_artifact = {
        **episode.artifact_data(),
        "support_gate": (
            result.support_gate.to_data() if result.support_gate is not None else None
        ),
        "proposal_freeze": (
            result.proposal_freeze.to_data()
            if result.proposal_freeze is not None
            else None
        ),
    }
    content: dict[str, Any] = {
        "schema": PROTOTYPE_OUTER_RUN_SCHEMA,
        "corpus_manifest_digest": corpus_manifest_digest,
        "split_source_digest": split_source_digest,
        "official_release": dict(release_data) if release_data is not None else None,
        "calibration": calibration.to_data(),
        "plan": plan.to_data(),
        "support_commitment": plan.support.to_data(),
        "episode": result.to_data(),
        "prototype": prototype_artifact,
        "run_archive": result.bundle.to_archive_data() if result.bundle else None,
        "exposure": dict(exposure) if exposure is not None else None,
    }
    return {**content, "record_digest": canonical_digest(content)}


__all__ = (
    "PROTOTYPE_OUTER_RUN_SCHEMA",
    "PrototypeRunVerification",
    "PrototypeRunVerificationError",
    "build_prototype_run_record",
    "verify_prototype_run_bytes",
    "verify_prototype_run_data",
)
