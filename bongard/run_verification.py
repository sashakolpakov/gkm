"""Cross-verify a completed outer run, its archive, and vision receipts.

``bongard.artifacts`` verifies the sealed support/freeze/query/prediction/label
chain.  That is necessary but not sufficient for a complete saved CLI run:
the outer episode summary and the separately archived Codex proposal and
observations must also reproduce the commitments inside that chain.

This module supplies that missing cross-layer verifier.  It deliberately
distinguishes two assurance levels:

* :func:`audit_completed_run_commitments` proves that the archive, receipt
  image-identity envelopes, and BlobRefs commit the same SHA-256 identities;
* :func:`verify_completed_run_data` additionally requires every referenced
  panel byte preimage and verifies it against its BlobRef.  BlobRef alone does
  not contain those bytes, so silently claiming byte replay would be false.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

from bongard.artifacts import (
    ArtifactTamperError,
    BlobRef,
    TruthEvidenceRecord,
    VerifiedRunArchive,
    canonical_digest,
    canonical_json,
    verify_archive_data,
)
from bongard.benchmark import (
    PROTOCOL_VERSION,
    SUPPORT_GATE_POLICY_VERSION,
    SupportGatePolicy,
)
from bongard.evidence import Disposition
from bongard.exposure import ExposureLedger, semantic_resolver_policy_digest
from bongard.historical_exposure import load_historical_exposure
from bongard.cohorts import classify_task
from bongard.proposer import (
    HYBRID_OBSERVATION_SCHEMA,
    HYBRID_OBSERVATION_SCHEMA_VERSION,
    HYBRID_ONLY_RULE_PROPOSAL_SCHEMA,
    HEADLESS_EPISODE_SCHEMA_VERSION,
    PROPOSAL_SCHEMA_VERSION,
    REJECTED_PROPOSAL_ATTEMPT_SCHEMA_VERSION,
    HybridObservation,
    ProposalError,
    RejectedProposalAttempt,
    RejectedProposalError,
    RuleProposal,
    SupportPanelIdentity,
    TRANSPORT_IDENTITY_FIELDS,
    codex_transport_identity,
    hybrid_observer_prompt,
    parse_hybrid_observation_or_error,
    parse_rule_proposal,
    proposer_prompt,
)
from bongard.synthesis import (
    SynthesisError,
    compile_hybrid_proposal,
    truth_from_hybrid_observation,
)
from bongard.release import OfficialReleaseDescriptor, ReleaseIdentityError
from bongard.transport import (
    CODEX_RECEIPT_SCHEMA,
    MAX_PANEL_PNG_BYTES,
    NAMED_IMAGE_INPUT_DIGEST_SCHEMA,
    STRUCTURED_INPUT_DIGEST_SCHEMA,
    CodexProposerFailure,
    CodexReceipt,
    semantic_panel_set_digest,
    validate_codex_receipt,
)


OUTER_RUN_SCHEMA = "gkm.bongard-episode-run.v4"
VISION_EPISODE_SCHEMA = HEADLESS_EPISODE_SCHEMA_VERSION
EXPOSURE_SCHEMA = "gkm.bongard-support-release-precommit.v2"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

_HEX = re.compile(r"[0-9a-f]{64}\Z")
_PREFIXED_HEX = re.compile(r"sha256:[0-9a-f]{64}\Z")
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
_OUTER_FIELDS = {
    "schema",
    "corpus_manifest_digest",
    "split_source_digest",
    "official_release",
    "plan",
    "episode",
    "vision",
    "run_archive",
    "exposure",
    "record_digest",
}
_PLAN_FIELDS = {
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
_PROPOSAL_FIELDS = {
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
_OBSERVATION_FIELDS = {
    "schema",
    "epistemic_status",
    "proposal_digest",
    "payload",
    "disposition",
    "provenance_digest",
    "receipt",
}
_REJECTED_PROPOSAL_ATTEMPT_FIELDS = {
    "schema",
    "proposal_schema",
    "model_payload",
    "receipt",
    "support_presentation",
    "parse_error",
    "attempt_digest",
}
_SUPPORT_PRESENTATION_FIELDS = {"name", "byte_count", "content_digest"}
_PARSE_ERROR_FIELDS = {"error_type", "reason"}
_EXPOSURE_FIELDS = {
    "schema",
    "corpus_manifest_digest",
    "task_id",
    "model",
    "plan_digest",
    "ledger_before_digest",
    "ledger_after_digest",
    "event_digest",
    "event",
    "ledger_before_event_count",
    "ledger_after_event_count",
    "ledger_input_supplied",
    "unseen_required",
    "semantic_unseen_required",
    "historical_seed_digest",
    "semantic_resolver_policy_digest",
    "expected_semantic_cohort",
    "classified_semantic_cohort",
    "semantic_unseen_receipt",
    "successor_filename",
    "external_anchor",
}


class RunVerificationError(ValueError):
    """A completed outer run does not reproduce its frozen commitments."""


class MissingBlobPreimagesError(RunVerificationError):
    """Strict verification was requested without every committed byte preimage."""

    def __init__(self, blob_ids: tuple[str, ...]):
        self.blob_ids = blob_ids
        rendered = ", ".join(blob_ids)
        super().__init__(
            "exact image-presentation verification requires byte preimages for "
            f"these BlobRefs: {rendered}; a BlobRef proves only byte-count/SHA-256 "
            "identity, not possession or decoding of the bytes"
        )


class MissingSupportPreimagesError(RunVerificationError):
    """Strict rejected-turn replay lacks one or more named support PNGs."""

    def __init__(self, names: tuple[str, ...]):
        self.names = names
        super().__init__(
            "exact rejected-proposal verification requires byte preimages for "
            f"these support images: {', '.join(names)}"
        )


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise RunVerificationError(f"{label} must be a JSON object")
    return value


def _expect_fields(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise RunVerificationError(
            f"{label} fields differ from schema: "
            f"missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _hex(value: object, label: str, *, prefixed: bool = False) -> str:
    pattern = _PREFIXED_HEX if prefixed else _HEX
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        kind = "sha256: content address" if prefixed else "lowercase SHA-256"
        raise RunVerificationError(f"{label} must be a {kind}")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise RunVerificationError(f"{label} must be a canonical non-empty string")
    return value


def _payload_digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(dict(payload))).hexdigest()


def _text_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="strict")).hexdigest()


def _schema_digest(schema: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(dict(schema))).hexdigest()


def _image_identity(name: str, blob: BlobRef) -> dict[str, object]:
    return {
        "name": name,
        "byte_count": blob.byte_count,
        "content_digest": blob.sha256,
    }


def _receipt_from_data(value: object, label: str) -> CodexReceipt:
    data = _mapping(value, f"{label} receipt")
    try:
        validate_codex_receipt(data)
    except (CodexProposerFailure, TypeError, ValueError) as exc:
        raise RunVerificationError(f"{label} has an invalid Codex receipt: {exc}") from exc
    converted = dict(data)
    converted["event_types"] = tuple(data["event_types"])
    converted["item_types"] = tuple(data["item_types"])
    try:
        return CodexReceipt(**converted)
    except TypeError as exc:  # pragma: no cover - validate_codex_receipt owns fields.
        raise RunVerificationError(f"{label} receipt cannot be reconstructed") from exc


def _verify_transport_identity(
    receipt: CodexReceipt,
    proposal_receipt: CodexReceipt,
    label: str,
) -> None:
    if codex_transport_identity(receipt) != codex_transport_identity(proposal_receipt):
        expected = dict(codex_transport_identity(proposal_receipt))
        actual = dict(codex_transport_identity(receipt))
        changed = [
            name for name in TRANSPORT_IDENTITY_FIELDS if actual[name] != expected[name]
        ]
        raise RunVerificationError(
            f"{label}: transport identity differs from proposer: {', '.join(changed)}"
        )


def _verify_receipt_payload(
    receipt: CodexReceipt,
    payload: Mapping[str, Any],
    schema: Mapping[str, Any],
    prompt: str,
    *,
    input_schema: str,
    identities: list[dict[str, object]],
    panel_set_digest: str,
    label: str,
) -> None:
    if receipt.schema != CODEX_RECEIPT_SCHEMA:
        raise RunVerificationError(f"{label} receipt schema differs")
    if receipt.input_digest_schema != input_schema:
        raise RunVerificationError(f"{label} receipt uses the wrong input-digest schema")
    expected_prompt = _text_digest(prompt)
    if receipt.prompt_digest != expected_prompt or receipt.task_digest != expected_prompt:
        raise RunVerificationError(f"{label} receipt does not bind the reconstructed prompt")
    expected_schema = _schema_digest(schema)
    if receipt.output_schema_digest != expected_schema:
        raise RunVerificationError(f"{label} receipt does not bind the output schema")
    if receipt.structured_output_digest != _payload_digest(payload):
        raise RunVerificationError(f"{label} receipt does not bind its structured payload")

    expected_view = canonical_digest(identities)
    if receipt.panel_view_digest != expected_view:
        raise RunVerificationError(
            f"{label} receipt image view differs from BlobRef byte identities"
        )
    if receipt.panel_set_digest != panel_set_digest:
        raise RunVerificationError(
            f"{label} receipt image set differs from BlobRef byte identities"
        )

    if input_schema == STRUCTURED_INPUT_DIGEST_SCHEMA:
        envelope: dict[str, object] = {
            "schema": input_schema,
            "task": prompt,
            "ordered_panel_identities": identities,
            "panel_view_digest": expected_view,
            "panel_set_digest": panel_set_digest,
            "prompt_digest": expected_prompt,
            "output_schema_digest": expected_schema,
        }
    else:
        envelope = {
            "schema": input_schema,
            "task": prompt,
            "ordered_image_identities": identities,
            "image_view_digest": expected_view,
            "image_set_digest": panel_set_digest,
            "prompt_digest": expected_prompt,
            "output_schema_digest": expected_schema,
        }
    if receipt.input_digest != canonical_digest(envelope):
        raise RunVerificationError(f"{label} receipt input envelope does not reproduce")


@dataclass(frozen=True)
class RejectedProposalAttemptVerification:
    """Replayed semantic rejection and the evidence that remains unavailable."""

    attempt_digest: str
    receipt_digest: str
    parse_error_type: str
    parse_error_reason: str
    verified_support_preimages: tuple[str, ...]
    missing_support_preimages: tuple[str, ...]
    limitations: tuple[str, ...]
    attempt: RejectedProposalAttempt = field(repr=False, compare=False)

    @property
    def support_byte_preimages_verified(self) -> bool:
        return not self.missing_support_preimages

    def to_data(self) -> dict[str, object]:
        return {
            "attempt_digest": self.attempt_digest,
            "receipt_digest": self.receipt_digest,
            "parse_error_type": self.parse_error_type,
            "parse_error_reason": self.parse_error_reason,
            "verified_support_preimages": list(self.verified_support_preimages),
            "missing_support_preimages": list(self.missing_support_preimages),
            "support_byte_preimages_verified": self.support_byte_preimages_verified,
            "limitations": list(self.limitations),
        }


def audit_rejected_proposal_attempt_data(
    value: object,
    *,
    support_bytes_by_name: Mapping[str, bytes] | None = None,
) -> RejectedProposalAttemptVerification:
    """Reproduce a semantically rejected proposer turn without admitting it.

    The raw structured payload is checked against its validated receipt and
    reparsed under the current frozen proposal language.  Exact support bytes
    are optional; when absent, the report names them rather than claiming
    byte-preimage or semantic-panel replay.
    """

    data = _mapping(value, "rejected proposal attempt")
    _expect_fields(
        data,
        _REJECTED_PROPOSAL_ATTEMPT_FIELDS,
        "rejected proposal attempt",
    )
    if data["schema"] != REJECTED_PROPOSAL_ATTEMPT_SCHEMA_VERSION:
        raise RunVerificationError("rejected proposal attempt schema differs")
    if data["proposal_schema"] != PROPOSAL_SCHEMA_VERSION:
        raise RunVerificationError("rejected attempt proposal schema differs")
    attempt_digest = _hex(
        data["attempt_digest"], "rejected attempt.attempt_digest", prefixed=True
    )
    attempt_content = {
        key: value for key, value in data.items() if key != "attempt_digest"
    }
    if "sha256:" + canonical_digest(attempt_content) != attempt_digest:
        raise RunVerificationError("rejected proposal attempt digest mismatch")

    payload = _mapping(data["model_payload"], "rejected proposal model_payload")
    receipt = _receipt_from_data(data["receipt"], "rejected proposal")
    presentation_value = data["support_presentation"]
    if not isinstance(presentation_value, list) or len(presentation_value) != 12:
        raise RunVerificationError(
            "rejected proposal support presentation must contain 12 identities"
        )
    presentation: list[SupportPanelIdentity] = []
    for index, item_value in enumerate(presentation_value):
        item = _mapping(item_value, f"support presentation[{index}]")
        _expect_fields(
            item,
            _SUPPORT_PRESENTATION_FIELDS,
            f"support presentation[{index}]",
        )
        try:
            presentation.append(
                SupportPanelIdentity(
                    name=item["name"],
                    byte_count=item["byte_count"],
                    content_digest=item["content_digest"],
                )
            )
        except (ProposalError, TypeError, ValueError) as exc:
            raise RunVerificationError(
                f"support presentation[{index}] is invalid: {exc}"
            ) from exc

    error = _mapping(data["parse_error"], "rejected proposal parse_error")
    _expect_fields(error, _PARSE_ERROR_FIELDS, "rejected proposal parse_error")
    error_type = _text(error["error_type"], "rejected parse_error.error_type")
    error_reason = _text(error["reason"], "rejected parse_error.reason")
    try:
        attempt = RejectedProposalAttempt(
            model_payload=payload,
            receipt=receipt,
            support_presentation=tuple(presentation),
            parse_error_type=error_type,
            parse_error_reason=error_reason,
        )
    except (ProposalError, TypeError, ValueError) as exc:
        raise RunVerificationError(f"rejected proposal attempt is invalid: {exc}") from exc
    if attempt.to_dict() != dict(data):
        raise RunVerificationError(
            "rejected proposal attempt does not reproduce from archived fields"
        )

    identities = [item.to_dict() for item in presentation]
    _verify_receipt_payload(
        receipt,
        payload,
        HYBRID_ONLY_RULE_PROPOSAL_SCHEMA,
        proposer_prompt({}),
        input_schema=STRUCTURED_INPUT_DIGEST_SCHEMA,
        identities=identities,
        panel_set_digest=receipt.panel_set_digest,
        label="rejected proposal",
    )

    try:
        parse_rule_proposal(payload, receipt=receipt, observable_catalog={})
    except ProposalError as exc:
        if type(exc).__name__ != error_type or (str(exc) or "proposal validation failed") \
                != error_reason:
            raise RunVerificationError(
                "rejected proposal parse error does not reproduce exactly"
            ) from exc
    except (TypeError, ValueError) as exc:
        raise RunVerificationError(
            "rejected proposal parser raised a non-protocol implementation error"
        ) from exc
    else:
        raise RunVerificationError(
            "rejected proposal payload is accepted by the declared proposal protocol"
        )

    expected_by_name = {item.name: item for item in presentation}
    if support_bytes_by_name is None:
        supplied: Mapping[str, bytes] = {}
    elif not isinstance(support_bytes_by_name, Mapping) or any(
        not isinstance(name, str) or not isinstance(payload_bytes, bytes)
        for name, payload_bytes in support_bytes_by_name.items()
    ):
        raise RunVerificationError(
            "support_bytes_by_name must map canonical support names to exact bytes"
        )
    else:
        supplied = support_bytes_by_name
    extras = set(supplied) - set(expected_by_name)
    if extras:
        raise RunVerificationError(
            f"support byte map contains unknown names: {sorted(extras)}"
        )
    verified_names: list[str] = []
    for name, panel_bytes in supplied.items():
        identity = expected_by_name[name]
        if len(panel_bytes) > MAX_PANEL_PNG_BYTES or not panel_bytes.startswith(
            PNG_SIGNATURE
        ):
            raise RunVerificationError(f"{name}: support byte preimage is not canonical PNG")
        if len(panel_bytes) != identity.byte_count or hashlib.sha256(
            panel_bytes
        ).hexdigest() != identity.content_digest:
            raise RunVerificationError(
                f"{name}: support byte preimage differs from archived identity"
            )
        verified_names.append(name)
    missing_names = tuple(sorted(set(expected_by_name) - set(supplied)))
    if not missing_names:
        with tempfile.TemporaryDirectory(prefix="bongard-rejection-verify-") as directory:
            root = Path(directory)
            paths: list[str] = []
            for item in presentation:
                path = root / item.name
                path.write_bytes(supplied[item.name])
                paths.append(str(path))
            try:
                semantic_digest = semantic_panel_set_digest(paths)
            except (CodexProposerFailure, OSError, ValueError) as exc:
                raise RunVerificationError(
                    "rejected proposal support bytes do not reproduce a canonical "
                    f"semantic panel set: {exc}"
                ) from exc
        if semantic_digest != receipt.panel_set_digest:
            raise RunVerificationError(
                "rejected proposal support bytes differ from receipt semantic panel set"
            )

    limitations = [
        "transport JSONL event-stream bytes are not embedded; only their receipt digest is bound",
        "the rejected turn has no RuleProposal, formula, support gate, freeze, query release, or run archive",
    ]
    if missing_names:
        limitations.append(
            "support byte preimages are absent or partial; semantic-panel decoding was not replayed"
        )
    return RejectedProposalAttemptVerification(
        attempt_digest=attempt_digest,
        receipt_digest=receipt.receipt_digest,
        parse_error_type=error_type,
        parse_error_reason=error_reason,
        verified_support_preimages=tuple(sorted(verified_names)),
        missing_support_preimages=missing_names,
        limitations=tuple(limitations),
        attempt=attempt,
    )


@dataclass(frozen=True)
class RejectedRunVerification:
    """Cross-layer audit of a proposal-error run with a captured model result."""

    run_id: str
    record_digest: str
    plan_digest: str
    attempt_digest: str
    receipt_digest: str
    official_release_digest: str | None
    verified_support_preimages: tuple[str, ...]
    missing_support_preimages: tuple[str, ...]
    unbound_outer_fields: tuple[str, ...]
    attempt: RejectedProposalAttempt = field(repr=False, compare=False)

    @property
    def support_byte_preimages_verified(self) -> bool:
        return not self.missing_support_preimages

    def to_data(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "record_digest": self.record_digest,
            "plan_digest": self.plan_digest,
            "attempt_digest": self.attempt_digest,
            "receipt_digest": self.receipt_digest,
            "official_release_digest": self.official_release_digest,
            "verified_support_preimages": list(self.verified_support_preimages),
            "missing_support_preimages": list(self.missing_support_preimages),
            "support_byte_preimages_verified": self.support_byte_preimages_verified,
            "unbound_outer_fields": list(self.unbound_outer_fields),
        }


def _verify_rejected_episode(value: object) -> Mapping[str, Any]:
    episode = _mapping(value, "outer rejected episode")
    _expect_fields(episode, _EPISODE_FIELDS, "outer rejected episode")
    if episode["version"] != PROTOCOL_VERSION:
        raise RunVerificationError("rejected episode protocol version differs")
    task_id = _text(episode["task_id"], "episode.task_id")
    family = episode["family"]
    if family not in {"ff", "bd", "hd"} or not task_id.startswith(f"{family}_"):
        raise RunVerificationError("rejected episode task/family identity is malformed")
    if episode["split"] not in {None, "train", "val", "test"}:
        raise RunVerificationError("rejected episode split is invalid")
    if episode["regime"] not in {None, "FF", "BA", "CM", "NV"}:
        raise RunVerificationError("rejected episode regime is invalid")
    _text(episode["run_id"], "episode.run_id")
    _hex(episode["plan_digest"], "episode.plan_digest")
    if episode["status"] != "proposal_error":
        raise RunVerificationError("rejected episode status must be proposal_error")
    if episode["artifact_chain"] is not None:
        raise RunVerificationError("rejected episode cannot claim an artifact chain")
    phases = episode["phases"]
    if not isinstance(phases, list) or tuple(phases) != (
        "plan_committed",
        "support_released",
        "proposal_failed",
    ):
        raise RunVerificationError("rejected episode phases are not canonical")
    score = _mapping(episode["score"], "rejected episode.score")
    _expect_fields(score, _SCORE_FIELDS, "rejected episode.score")
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
    if dict(score) != expected_score:
        raise RunVerificationError("rejected episode score is not canonical")
    failure = _mapping(episode["failure"], "rejected episode.failure")
    _expect_fields(failure, {"stage", "error_type", "reason"}, "episode.failure")
    if failure["stage"] != "proposal":
        raise RunVerificationError("rejected episode failure stage differs")
    _text(failure["error_type"], "episode.failure.error_type")
    _text(failure["reason"], "episode.failure.reason")
    return episode


def _verify_rejected_public_plan(
    value: object,
    *,
    episode: Mapping[str, Any],
    corpus_address: str,
) -> Mapping[str, Any]:
    plan = _mapping(value, "public rejected plan")
    _expect_fields(plan, _PLAN_FIELDS, "public rejected plan")
    if plan["version"] != PROTOCOL_VERSION:
        raise RunVerificationError("rejected public plan protocol version differs")
    for field_name in ("task_id", "family", "split", "regime", "run_id"):
        if plan[field_name] != episode[field_name]:
            raise RunVerificationError(
                f"rejected public plan {field_name} differs from episode"
            )
    _text(plan["verifier_id"], "plan.verifier_id")
    for field_name in (
        "seed_digest",
        "corpus_digest",
        "task_manifest_digest",
        "support_commitment_digest",
        "latent_query_digest",
        "label_commitment_digest",
    ):
        _hex(plan[field_name], f"plan.{field_name}")
    if plan["corpus_digest"] != corpus_address.removeprefix("sha256:"):
        raise RunVerificationError("rejected public plan corpus differs from outer run")
    if canonical_digest(dict(plan)) != episode["plan_digest"]:
        raise RunVerificationError(
            "rejected public plan canonical digest differs from episode.plan_digest"
        )
    return plan


def audit_rejected_run_commitments(
    record_value: Mapping[str, Any],
    *,
    support_bytes_by_name: Mapping[str, bytes] | None = None,
) -> RejectedRunVerification:
    """Audit a semantic proposal rejection without inventing a completed run."""

    record = _mapping(record_value, "outer rejected run record")
    _expect_fields(record, _OUTER_FIELDS, "outer rejected run record")
    if record["schema"] != OUTER_RUN_SCHEMA:
        raise RunVerificationError(
            f"unsupported outer rejected run schema {record['schema']!r}"
        )
    record_digest = _hex(record["record_digest"], "record.record_digest")
    content = {key: value for key, value in record.items() if key != "record_digest"}
    if canonical_digest(content) != record_digest:
        raise RunVerificationError("outer rejected run record digest mismatch")
    if record["run_archive"] is not None:
        raise RunVerificationError("proposal rejection must not contain a run archive")

    corpus_address = _hex(
        record["corpus_manifest_digest"],
        "record.corpus_manifest_digest",
        prefixed=True,
    )
    split_digest = record["split_source_digest"]
    if split_digest is not None:
        _hex(split_digest, "record.split_source_digest", prefixed=True)
    release_value = record["official_release"]
    official_release: OfficialReleaseDescriptor | None
    if release_value is None:
        official_release = None
    else:
        try:
            official_release = OfficialReleaseDescriptor.from_dict(
                _mapping(release_value, "official_release")
            )
        except (ReleaseIdentityError, TypeError, ValueError) as exc:
            raise RunVerificationError(
                f"official_release descriptor is invalid: {exc}"
            ) from exc
        if official_release.corpus_manifest_sha256 != corpus_address:
            raise RunVerificationError(
                "official_release corpus manifest differs from rejected run"
            )
        if official_release.split_sha256 != split_digest:
            raise RunVerificationError(
                "official_release split digest differs from rejected run"
            )

    episode = _verify_rejected_episode(record["episode"])
    plan = _verify_rejected_public_plan(
        record["plan"], episode=episode, corpus_address=corpus_address
    )
    exposure = _mapping(record["exposure"], "rejected run exposure")
    _verify_semantic_exposure(
        exposure,
        episode=episode,
        corpus_address=corpus_address,
    )
    if exposure.get("plan_digest") != canonical_digest(dict(plan)):
        raise RunVerificationError("rejected run exposure plan_digest differs")

    vision = _mapping(record["vision"], "rejected vision artifact")
    _expect_fields(
        vision,
        {
            "schema",
            "proposal",
            "rejected_proposal_attempt",
            "support_gate",
            "proposal_freeze",
            "observations",
        },
        "rejected vision artifact",
    )
    if vision["schema"] != VISION_EPISODE_SCHEMA:
        raise RunVerificationError("rejected vision episode schema differs")
    if vision["proposal"] is not None:
        raise RunVerificationError("rejected vision cannot contain an accepted proposal")
    if vision["support_gate"] is not None or vision["proposal_freeze"] is not None:
        raise RunVerificationError("rejected vision cannot contain a gate or freeze")
    if vision["observations"] != {}:
        raise RunVerificationError("rejected vision cannot contain query observations")
    if vision["rejected_proposal_attempt"] is None:
        raise RunVerificationError(
            "semantic proposal rejection lacks its rejected proposal attempt"
        )
    attempt_report = audit_rejected_proposal_attempt_data(
        vision["rejected_proposal_attempt"],
        support_bytes_by_name=support_bytes_by_name,
    )
    failure = _mapping(episode["failure"], "rejected episode.failure")
    if failure["error_type"] != RejectedProposalError.__name__ \
            or failure["reason"] != attempt_report.parse_error_reason:
        raise RunVerificationError(
            "episode failure does not reproduce the archived proposal rejection"
        )
    if exposure.get("model") != attempt_report.attempt.receipt.requested_model:
        raise RunVerificationError("exposure model differs from rejected receipt model")

    unbound = (
        "the public support_commitment_digest has no embedded commitment preimage on a pre-freeze failure",
        "task manifest bytes and split assignment require the supplied corpus",
        "seed_digest commits the private seed but the seed preimage is intentionally absent",
        "split source bytes are not embedded",
        "exposure-ledger predecessor authenticity needs the ledger or an external anchor",
        *attempt_report.limitations,
    )
    return RejectedRunVerification(
        run_id=episode["run_id"],
        record_digest=record_digest,
        plan_digest=canonical_digest(dict(plan)),
        attempt_digest=attempt_report.attempt_digest,
        receipt_digest=attempt_report.receipt_digest,
        official_release_digest=(
            official_release.digest if official_release is not None else None
        ),
        verified_support_preimages=attempt_report.verified_support_preimages,
        missing_support_preimages=attempt_report.missing_support_preimages,
        unbound_outer_fields=unbound,
        attempt=attempt_report.attempt,
    )


def _expected_score(archive: VerifiedRunArchive) -> dict[str, object]:
    bundle = archive.bundle
    predictions = {item.query_id: item for item in bundle.predictions.predictions}
    labels = {item.query_id: item.positive for item in bundle.labels.labels}
    correct = sum(
        item.positive is not None and item.positive == labels[item.query_id]
        for item in predictions.values()
    )
    determinate = sum(item.positive is not None for item in predictions.values())
    abstentions = 2 - determinate
    errors = sum(item.disposition.value == "error" for item in predictions.values())
    return {
        "image_correct": correct,
        "image_total": 2,
        "image_accuracy": correct / 2,
        "puzzle_correct": correct == 2,
        "puzzle_accuracy": float(correct == 2),
        "determinate": determinate,
        "abstentions": abstentions,
        "errors": errors,
    }


def _verify_episode(value: object, archive: VerifiedRunArchive) -> Mapping[str, Any]:
    episode = _mapping(value, "outer episode")
    _expect_fields(episode, _EPISODE_FIELDS, "outer episode")
    if episode["version"] != PROTOCOL_VERSION:
        raise RunVerificationError("outer episode protocol version differs")
    task_id = _text(episode["task_id"], "episode.task_id")
    family = episode["family"]
    if family not in {"ff", "bd", "hd"} or not task_id.startswith(f"{family}_"):
        raise RunVerificationError("episode task/family identity is malformed")
    if episode["split"] not in {None, "train", "val", "test"}:
        raise RunVerificationError("episode split is invalid")
    if episode["regime"] not in {None, "FF", "BA", "CM", "NV"}:
        raise RunVerificationError("episode regime is invalid")
    _hex(episode["plan_digest"], "episode.plan_digest")
    if episode["status"] != "complete" or episode["failure"] is not None:
        raise RunVerificationError("completed run must have complete status and null failure")
    if episode["run_id"] != archive.bundle.support.run_id:
        raise RunVerificationError("outer episode run_id differs from artifact bundle")
    phases = episode["phases"]
    if not isinstance(phases, list) or tuple(phases) != _COMPLETE_PHASES:
        raise RunVerificationError("completed episode phases are not canonical")
    chain = _mapping(episode["artifact_chain"], "episode.artifact_chain")
    if dict(chain) != archive.bundle.chain_data():
        raise RunVerificationError("outer episode artifact_chain differs from bundle")

    score = _mapping(episode["score"], "episode.score")
    _expect_fields(score, _SCORE_FIELDS, "episode.score")
    for name in (
        "image_correct",
        "image_total",
        "determinate",
        "abstentions",
        "errors",
    ):
        if isinstance(score[name], bool) or not isinstance(score[name], int):
            raise RunVerificationError(f"episode.score.{name} must be an integer")
    if not isinstance(score["puzzle_correct"], bool):
        raise RunVerificationError("episode.score.puzzle_correct must be Boolean")
    for name in ("image_accuracy", "puzzle_accuracy"):
        if not isinstance(score[name], float):
            raise RunVerificationError(f"episode.score.{name} must be a JSON float")
    if dict(score) != _expected_score(archive):
        raise RunVerificationError("outer episode score differs from verified bundle")
    return episode


def _verify_public_plan(
    value: object,
    *,
    episode: Mapping[str, Any],
    archive: VerifiedRunArchive,
    corpus_address: str,
) -> Mapping[str, Any]:
    plan = _mapping(value, "public episode plan")
    _expect_fields(plan, _PLAN_FIELDS, "public episode plan")
    if plan["version"] != PROTOCOL_VERSION:
        raise RunVerificationError("public plan protocol version differs")
    for field_name in ("task_id", "family", "split", "regime", "run_id"):
        if plan[field_name] != episode[field_name]:
            raise RunVerificationError(
                f"public plan {field_name} differs from outer episode"
            )
    verifier_id = _text(plan["verifier_id"], "plan.verifier_id")
    for field_name in (
        "seed_digest",
        "corpus_digest",
        "task_manifest_digest",
        "support_commitment_digest",
        "latent_query_digest",
        "label_commitment_digest",
    ):
        _hex(plan[field_name], f"plan.{field_name}")
    plan_digest = canonical_digest(dict(plan))
    if plan_digest != episode["plan_digest"]:
        raise RunVerificationError(
            "public plan canonical digest differs from episode.plan_digest"
        )

    bundle = archive.bundle
    if plan["run_id"] != bundle.support.run_id:
        raise RunVerificationError("public plan run differs from archive support")
    if verifier_id != bundle.support.issued_by:
        raise RunVerificationError("public plan verifier differs from support issuer")
    if plan["corpus_digest"] != corpus_address.removeprefix("sha256:") \
            or plan["corpus_digest"] != bundle.support.corpus_digest:
        raise RunVerificationError("public plan corpus differs from outer/archive corpus")
    if plan["support_commitment_digest"] != bundle.support.digest():
        raise RunVerificationError(
            "public plan support commitment differs from archive support"
        )
    latent_query_digest = canonical_digest(
        {
            "version": "latent-two-query-commitment/v1",
            "run_id": bundle.release.run_id,
            "queries": [item.to_data() for item in bundle.release.queries],
        }
    )
    if plan["latent_query_digest"] != latent_query_digest:
        raise RunVerificationError(
            "public plan latent query commitment differs from archive release"
        )
    label_seal_digest = canonical_digest(
        {
            "run_id": bundle.labels.run_id,
            "labels": [item.to_data() for item in bundle.labels.labels],
            "nonce": bundle.labels.verifier_nonce,
            "version": "latent-label-seal/v1",
        }
    )
    if plan["label_commitment_digest"] != label_seal_digest:
        raise RunVerificationError(
            "public plan label seal differs from archive label reveal"
        )
    return plan


def _support_presentation(
    archive: VerifiedRunArchive,
) -> tuple[list[dict[str, object]], tuple[tuple[str, BlobRef], ...]]:
    by_id = {item.panel.blob_id: item for item in archive.bundle.support.support}
    expected: list[tuple[str, str, bool]] = [
        (f"pos_{index}.png", f"support-positive-{index}", True)
        for index in range(6)
    ] + [
        (f"neg_{index}.png", f"support-negative-{index}", False)
        for index in range(6)
    ]
    if set(by_id) != {blob_id for _name, blob_id, _positive in expected}:
        raise RunVerificationError(
            "support BlobRefs do not form the canonical six-positive/six-negative view"
        )
    named: list[tuple[str, BlobRef]] = []
    for name, blob_id, positive in expected:
        item = by_id[blob_id]
        if item.positive is not positive:
            raise RunVerificationError(f"support BlobRef {blob_id!r} has the wrong side")
        named.append((name, item.panel))
    return [_image_identity(name, blob) for name, blob in named], tuple(named)


def _blob_from_gate_data(value: object, label: str) -> BlobRef:
    data = _mapping(value, label)
    _expect_fields(data, {"blob_id", "sha256", "byte_count", "media_type"}, label)
    try:
        return BlobRef(
            blob_id=str(data["blob_id"]),
            sha256=str(data["sha256"]),
            byte_count=data["byte_count"],
            media_type=str(data["media_type"]),
        )
    except (TypeError, ValueError) as exc:
        raise RunVerificationError(f"{label} is invalid: {exc}") from exc


def _verify_support_replay_gate(
    value: object,
    *,
    archived_freeze: object,
    archive: VerifiedRunArchive,
    proposal: RuleProposal,
    proposal_receipt: CodexReceipt,
    receipt_digests: list[str],
    thread_ids: set[str],
) -> tuple[tuple[str, HybridObservation], ...]:
    """Replay all twelve label-blind support observations from the archive."""

    freeze_data = _mapping(archived_freeze, "vision proposal_freeze")
    if dict(freeze_data) != archive.bundle.freeze.to_data():
        raise RunVerificationError(
            "vision proposal_freeze differs from the sealed run archive"
        )

    gate = _mapping(value, "vision support gate")
    gate_fields = {
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
    _expect_fields(gate, gate_fields, "vision support gate")
    if gate["version"] != "support-replay-gate-artifact/v1":
        raise RunVerificationError("support gate artifact version differs")
    if gate["run_id"] != archive.bundle.support.run_id:
        raise RunVerificationError("support gate run differs from archive")
    proposer_digest = proposal.digest.removeprefix("sha256:")
    if gate["proposal_digest"] != proposer_digest:
        raise RunVerificationError("support gate proposal digest differs")
    if gate["support_commitment_digest"] != archive.bundle.support.digest():
        raise RunVerificationError("support gate support commitment differs")
    expected_policy = SupportGatePolicy.empirical().to_data()
    policy = _mapping(gate["policy"], "support gate policy")
    if dict(policy) != expected_policy or policy.get("version") != SUPPORT_GATE_POLICY_VERSION:
        raise RunVerificationError("completed headless run has a noncanonical support gate policy")

    entries_value = gate["ordered_entries"]
    if not isinstance(entries_value, list) or len(entries_value) != 12:
        raise RunVerificationError("support gate must contain exactly twelve ordered entries")
    expected_support = archive.bundle.support.support
    if len(expected_support) != 12:
        raise RunVerificationError("archive support is not exactly six plus six")

    observations: list[tuple[str, HybridObservation]] = []
    dispositions: list[Disposition] = []
    forward = 0
    reverse = 0
    for index, (entry_value, expected) in enumerate(
        zip(entries_value, expected_support, strict=True)
    ):
        label = f"support-gate-{index}"
        entry = _mapping(entry_value, label)
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
            label,
        )
        if entry["transport_attempted"] is not True:
            raise RunVerificationError(
                f"{label}: completed gate lacks a fresh transport attempt"
            )
        if entry["slot_id"] != expected.panel.blob_id:
            raise RunVerificationError(f"{label}: support slot order differs")
        if not isinstance(entry["positive"], bool) or entry["positive"] is not expected.positive:
            raise RunVerificationError(f"{label}: verifier-side support label differs")
        if _blob_from_gate_data(entry["panel"], f"{label} panel") != expected.panel:
            raise RunVerificationError(f"{label}: panel BlobRef differs from support")

        observation_data = _mapping(entry["observer_artifact"], f"{label} observation")
        _expect_fields(observation_data, _OBSERVATION_FIELDS, f"{label} observation")
        if observation_data["schema"] != HYBRID_OBSERVATION_SCHEMA_VERSION:
            raise RunVerificationError(f"{label}: observation schema differs")
        payload = _mapping(observation_data["payload"], f"{label} payload")
        receipt = _receipt_from_data(observation_data["receipt"], label)
        _verify_transport_identity(receipt, proposal_receipt, label)
        if receipt.thread_id in thread_ids or receipt.receipt_digest in receipt_digests:
            raise RunVerificationError(f"{label}: Codex receipt identity is reused")
        thread_ids.add(receipt.thread_id)
        receipt_digests.append(receipt.receipt_digest)

        identities = [_image_identity("query.png", expected.panel)]
        named_set_digest = "sha256:" + canonical_digest(
            {"schema": NAMED_IMAGE_INPUT_DIGEST_SCHEMA, "images": identities}
        )
        _verify_receipt_payload(
            receipt,
            payload,
            HYBRID_OBSERVATION_SCHEMA,
            hybrid_observer_prompt(proposal),
            input_schema=NAMED_IMAGE_INPUT_DIGEST_SCHEMA,
            identities=identities,
            panel_set_digest=named_set_digest,
            label=label,
        )
        try:
            observation = parse_hybrid_observation_or_error(proposal, payload, receipt)
        except (ProposalError, TypeError, ValueError) as exc:
            raise RunVerificationError(
                f"{label}: observation cannot be parsed: {exc}"
            ) from exc
        if observation.to_dict() != dict(observation_data):
            raise RunVerificationError(
                f"{label}: observation does not reproduce from payload/receipt"
            )
        projected = truth_from_hybrid_observation(observation)
        evidence_data = _mapping(entry["evidence"], f"{label} evidence")
        try:
            archived_evidence = TruthEvidenceRecord.from_data(evidence_data)
        except (TypeError, ValueError) as exc:
            raise RunVerificationError(f"{label}: evidence is invalid: {exc}") from exc
        if archived_evidence.to_data() != TruthEvidenceRecord.from_evidence(projected).to_data():
            raise RunVerificationError(
                f"{label}: gate evidence differs from the archived observation"
            )
        disposition = projected.disposition
        dispositions.append(disposition)
        if (expected.positive and disposition is Disposition.PRESENT) or (
            not expected.positive and disposition is Disposition.CERTIFIED_ABSENT
        ):
            forward += 1
        if (expected.positive and disposition is Disposition.CERTIFIED_ABSENT) or (
            not expected.positive and disposition is Disposition.PRESENT
        ):
            reverse += 1
        observations.append((str(entry["slot_id"]), observation))

    disposition_counts = {item: dispositions.count(item) for item in Disposition}
    expected_counts = {
        "forward_matches": forward,
        "reverse_matches": reverse,
        "present": disposition_counts[Disposition.PRESENT],
        "nonmatch": disposition_counts[Disposition.CERTIFIED_ABSENT],
        "indeterminate": disposition_counts[Disposition.INDETERMINATE],
        "error": disposition_counts[Disposition.ERROR],
        "transport_attempts": 12,
    }
    counts = _mapping(gate["counts"], "support gate counts")
    if dict(counts) != expected_counts:
        raise RunVerificationError("support gate counts do not reproduce")
    if disposition_counts[Disposition.INDETERMINATE] or disposition_counts[Disposition.ERROR]:
        expected_result = "observer_failure"
    elif forward == 12:
        expected_result = "aligned"
    elif reverse > forward:
        expected_result = "misoriented"
    else:
        expected_result = "unsupported"
    if gate["result"] != expected_result:
        raise RunVerificationError("support gate result does not reproduce")
    if expected_result != "aligned":
        raise RunVerificationError(
            "completed run did not pass the strict six-present/six-nonmatch gate"
        )
    gate_content = {key: gate[key] for key in gate_fields if key != "gate_digest"}
    gate_digest = canonical_digest(gate_content)
    if gate["gate_digest"] != gate_digest:
        raise RunVerificationError("support gate digest does not reproduce")
    if archive.bundle.freeze.support_gate_digest != gate_digest:
        raise RunVerificationError("proposal freeze does not bind the support gate")
    return tuple(observations)


def _verify_canonical_hybrid_compilation(
    proposal: RuleProposal,
    *,
    plan: Mapping[str, Any],
    archive: VerifiedRunArchive,
) -> None:
    """Rebuild the only admissible HYBRID attachment in verifier-owned code.

    The proposal digest alone cannot establish that its prose was attached to
    the formula and executable-leg identity stored in the archive.  Recompiling
    here is side-effect free: construction hashes the canonical observer source
    and proposal-bound operational identity, but never invokes the observer.
    Exact snapshot equality therefore binds source and operational digests as
    well as the ordinary type/semantics fields.
    """

    verifier_id = _text(plan["verifier_id"], "plan.verifier_id")
    try:
        compiled = compile_hybrid_proposal(proposal, issued_by=verifier_id)
        expected_snapshot = compiled.registry.snapshot()
        expected_registry_digest = compiled.registry.digest()
    except (SynthesisError, TypeError, ValueError) as exc:
        raise RunVerificationError(
            f"visual proposal cannot be canonically compiled: {exc}"
        ) from exc

    bundle = archive.bundle
    frozen = bundle.freeze
    attached = bundle.attachment_contract

    if (
        attached.registry_snapshot.to_data() != expected_snapshot.to_data()
        or attached.registry_digest != expected_registry_digest
        or frozen.registry_digest != expected_registry_digest
    ):
        raise RunVerificationError(
            "frozen registry snapshot/digest differs from verifier-owned HYBRID "
            "compilation (including source or operational identity)"
        )
    if frozen.formula.to_data() != compiled.formula.to_data():
        raise RunVerificationError(
            "frozen formula differs from verifier-owned HYBRID compilation"
        )
    if (
        attached.to_data() != compiled.attachment_contract.to_data()
        or frozen.attachment_contract_digest
        != compiled.attachment_contract.digest()
    ):
        raise RunVerificationError(
            "frozen attachment contract differs from verifier-owned HYBRID compilation"
        )


def _query_presentation(
    archive: VerifiedRunArchive,
) -> tuple[tuple[str, BlobRef], ...]:
    queries = archive.bundle.release.queries
    expected_ids = ("query-0", "query-1")
    if tuple(item.query_id for item in queries) != expected_ids:
        raise RunVerificationError("query release does not use canonical public query IDs")
    expected_blobs = ("query-panel-0", "query-panel-1")
    if tuple(item.panel.blob_id for item in queries) != expected_blobs:
        raise RunVerificationError("query release does not use canonical query BlobRef IDs")
    return tuple((query.query_id, query.panel) for query in queries)


def _semantic_support_digest_from_bytes(
    named_support: tuple[tuple[str, BlobRef], ...],
    blob_bytes: Mapping[str, bytes],
) -> str:
    with tempfile.TemporaryDirectory(prefix="bongard-run-verify-") as directory:
        root = Path(directory)
        paths: list[str] = []
        for name, blob in named_support:
            path = root / name
            path.write_bytes(blob_bytes[blob.blob_id])
            paths.append(str(path))
        try:
            return semantic_panel_set_digest(paths)
        except (CodexProposerFailure, OSError, ValueError) as exc:
            raise RunVerificationError(
                f"support byte preimages do not reproduce a canonical semantic view: {exc}"
            ) from exc


def _verify_supplied_blob_bytes(
    blobs: tuple[BlobRef, ...], supplied: Mapping[str, bytes] | None,
) -> tuple[tuple[str, ...], tuple[str, ...], Mapping[str, bytes]]:
    expected = {blob.blob_id: blob for blob in blobs}
    if supplied is None:
        return (), tuple(sorted(expected)), {}
    if not isinstance(supplied, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, bytes)
        for key, value in supplied.items()
    ):
        raise RunVerificationError("blob_bytes_by_id must map blob IDs to exact bytes")
    extras = set(supplied) - set(expected)
    if extras:
        raise RunVerificationError(f"blob byte map contains unknown IDs: {sorted(extras)}")
    verified: list[str] = []
    for blob_id, payload in supplied.items():
        blob = expected[blob_id]
        if blob.media_type != "image/png" or len(payload) > MAX_PANEL_PNG_BYTES:
            raise RunVerificationError(f"{blob_id}: panel media type/size is not canonical")
        if not payload.startswith(PNG_SIGNATURE):
            raise RunVerificationError(f"{blob_id}: byte preimage is not a PNG")
        try:
            blob.verify_bytes(payload)
        except ArtifactTamperError as exc:
            raise RunVerificationError(f"{blob_id}: byte preimage differs from BlobRef") from exc
        verified.append(blob_id)
    missing = tuple(sorted(set(expected) - set(supplied)))
    return tuple(sorted(verified)), missing, supplied


def _verify_semantic_exposure(
    exposure: Mapping[str, Any],
    *,
    episode: Mapping[str, Any],
    corpus_address: str,
) -> None:
    """Reproduce the frozen semantic cohort/key interpretation.

    The predecessor ledger itself is external to the run, so this verifies the
    resolver, requested keys, cohort, and claimed predecessor binding. It does
    not authenticate the predecessor ledger's unavailable event history.
    """

    _expect_fields(exposure, _EXPOSURE_FIELDS, "run exposure")
    if exposure["schema"] != EXPOSURE_SCHEMA:
        raise RunVerificationError("run exposure schema differs")
    if exposure["corpus_manifest_digest"] != corpus_address:
        raise RunVerificationError("run exposure corpus differs")
    if exposure["task_id"] != episode["task_id"]:
        raise RunVerificationError("run exposure task differs")
    if exposure["plan_digest"] != episode["plan_digest"]:
        raise RunVerificationError("run exposure plan differs")
    if not isinstance(exposure["unseen_required"], bool) or not isinstance(
        exposure["semantic_unseen_required"], bool
    ):
        raise RunVerificationError("run exposure unseen policies must be Boolean")
    if exposure["semantic_unseen_required"] and not exposure["unseen_required"]:
        raise RunVerificationError("semantic unseen requires exact unseen")
    cohort = exposure["expected_semantic_cohort"]
    if cohort not in {None, "drill", "dev", "sealed"}:
        raise RunVerificationError("run exposure semantic cohort is invalid")
    receipt_value = exposure["semantic_unseen_receipt"]
    if exposure["semantic_unseen_required"]:
        historical = load_historical_exposure()
        policy_digest = semantic_resolver_policy_digest(historical)
        if exposure["historical_seed_digest"] != historical.seed_digest:
            raise RunVerificationError("run exposure historical seed differs")
        if exposure["semantic_resolver_policy_digest"] != policy_digest:
            raise RunVerificationError("run exposure semantic resolver differs")
        classified = classify_task(
            episode["task_id"],
            historical,
            split=episode["split"],
            regime=episode["regime"],
        )
        if exposure["classified_semantic_cohort"] != classified.semantic_cohort:
            raise RunVerificationError("run exposure classified cohort differs")
        if cohort is not None and (
            not classified.historically_clean
            or classified.semantic_cohort != cohort
        ):
            raise RunVerificationError(
                "run exposure task is outside its declared cohort"
            )
        receipt = _mapping(receipt_value, "semantic-unseen receipt")
        expected = ExposureLedger.create(corpus_address).assert_semantically_unseen(
            task_ids=(episode["task_id"],),
            historical_seed=historical,
            expected_historical_seed_digest=historical.seed_digest,
            expected_resolver_policy_digest=policy_digest,
        ).to_dict()
        if set(receipt) != set(expected):
            raise RunVerificationError("semantic-unseen receipt fields differ")
        for field in (
            "task_ids",
            "semantic_keys",
            "historical_seed_digest",
            "resolver_policy_digest",
        ):
            if receipt[field] != expected[field]:
                raise RunVerificationError(
                    f"semantic-unseen receipt {field} differs"
                )
        if receipt["ledger_digest"] != exposure["ledger_before_digest"]:
            raise RunVerificationError(
                "semantic-unseen receipt does not bind the predecessor ledger"
            )
    elif (
        receipt_value is not None
        or cohort is not None
        or exposure["classified_semantic_cohort"] is not None
        or exposure["historical_seed_digest"] is not None
        or exposure["semantic_resolver_policy_digest"] is not None
    ):
        raise RunVerificationError("unsolicited semantic-policy fields")


@dataclass(frozen=True)
class CompletedRunVerification:
    """Successful cross-layer bindings and explicitly unavailable evidence."""

    run_id: str
    record_digest: str
    archive_digest: str
    chain_digest: str
    plan_digest: str
    proposal_digest: str
    official_release_digest: str | None
    query_ids: tuple[str, str]
    receipt_digests: tuple[str, ...]
    verified_blob_ids: tuple[str, ...]
    missing_blob_preimages: tuple[str, ...]
    unbound_outer_fields: tuple[str, ...]
    archive: VerifiedRunArchive = field(repr=False, compare=False)
    proposal: RuleProposal = field(repr=False, compare=False)
    observations: tuple[tuple[str, HybridObservation], ...] = field(
        repr=False, compare=False
    )
    support_observations: tuple[tuple[str, HybridObservation], ...] = field(
        repr=False, compare=False
    )

    @property
    def byte_preimages_verified(self) -> bool:
        return not self.missing_blob_preimages

    def to_data(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "record_digest": self.record_digest,
            "archive_digest": self.archive_digest,
            "chain_digest": self.chain_digest,
            "plan_digest": self.plan_digest,
            "proposal_digest": self.proposal_digest,
            "official_release_digest": self.official_release_digest,
            "query_ids": list(self.query_ids),
            "receipt_digests": list(self.receipt_digests),
            "verified_blob_ids": list(self.verified_blob_ids),
            "missing_blob_preimages": list(self.missing_blob_preimages),
            "byte_preimages_verified": self.byte_preimages_verified,
            "unbound_outer_fields": list(self.unbound_outer_fields),
        }


def audit_completed_run_commitments(
    record_value: Mapping[str, Any],
    *,
    blob_bytes_by_id: Mapping[str, bytes] | None = None,
) -> CompletedRunVerification:
    """Verify every reproducible outer/archive/vision commitment.

    When ``blob_bytes_by_id`` is absent or partial, the returned report names
    every missing byte preimage.  It does *not* call that identity-only result
    byte replay; use :func:`verify_completed_run_data` for the strict boundary.
    """

    record = _mapping(record_value, "outer run record")
    _expect_fields(record, _OUTER_FIELDS, "outer run record")
    if record["schema"] != OUTER_RUN_SCHEMA:
        raise RunVerificationError(f"unsupported outer run schema {record['schema']!r}")
    record_digest = _hex(record["record_digest"], "record.record_digest")
    content = {key: value for key, value in record.items() if key != "record_digest"}
    if canonical_digest(content) != record_digest:
        raise RunVerificationError("outer run record digest mismatch")

    corpus_address = _hex(
        record["corpus_manifest_digest"], "record.corpus_manifest_digest", prefixed=True
    )
    split_digest = record["split_source_digest"]
    if split_digest is not None:
        _hex(split_digest, "record.split_source_digest", prefixed=True)
    release_value = record["official_release"]
    official_release: OfficialReleaseDescriptor | None
    if release_value is None:
        official_release = None
    else:
        release_data = _mapping(release_value, "official_release")
        try:
            official_release = OfficialReleaseDescriptor.from_dict(release_data)
        except (ReleaseIdentityError, TypeError, ValueError) as exc:
            raise RunVerificationError(
                f"official_release descriptor is invalid: {exc}"
            ) from exc
        if official_release.corpus_manifest_sha256 != corpus_address:
            raise RunVerificationError(
                "official_release corpus manifest differs from outer run"
            )
        if official_release.split_sha256 != split_digest:
            raise RunVerificationError(
                "official_release split digest differs from outer run"
            )
    archive_data = _mapping(record["run_archive"], "run_archive")
    try:
        archive = verify_archive_data(archive_data)
    except (ArtifactTamperError, TypeError, ValueError) as exc:
        raise RunVerificationError(f"run archive verification failed: {exc}") from exc
    if corpus_address.removeprefix("sha256:") != archive.bundle.support.corpus_digest:
        raise RunVerificationError(
            "outer corpus_manifest_digest differs from support.corpus_digest"
        )
    episode = _verify_episode(record["episode"], archive)
    plan = _verify_public_plan(
        record["plan"],
        episode=episode,
        archive=archive,
        corpus_address=corpus_address,
    )

    exposure = _mapping(record["exposure"], "run exposure")
    _verify_semantic_exposure(
        exposure,
        episode=episode,
        corpus_address=corpus_address,
    )
    for outer_name, exposure_name in (
        ("corpus_manifest_digest", "corpus_manifest_digest"),
        ("task_id", "task_id"),
        ("plan_digest", "plan_digest"),
    ):
        outer_value = record[outer_name] if outer_name in record else episode[outer_name]
        if exposure.get(exposure_name) != outer_value:
            raise RunVerificationError(
                f"run exposure {exposure_name} differs from the outer record"
            )
    if exposure.get("plan_digest") != canonical_digest(dict(plan)):
        raise RunVerificationError("run exposure plan_digest differs from public plan")

    vision = _mapping(record["vision"], "vision artifact")
    _expect_fields(
        vision,
        {
            "schema",
            "proposal",
            "rejected_proposal_attempt",
            "support_gate",
            "proposal_freeze",
            "observations",
        },
        "vision artifact",
    )
    if vision["schema"] != VISION_EPISODE_SCHEMA:
        raise RunVerificationError("vision episode schema differs")
    if vision["rejected_proposal_attempt"] is not None:
        raise RunVerificationError(
            "completed vision cannot contain a rejected proposal attempt"
        )

    proposal_data = _mapping(vision["proposal"], "vision proposal")
    _expect_fields(proposal_data, _PROPOSAL_FIELDS, "vision proposal")
    if proposal_data["schema"] != PROPOSAL_SCHEMA_VERSION:
        raise RunVerificationError("vision proposal schema differs")
    proposal_payload = _mapping(proposal_data["model_payload"], "proposal model_payload")
    proposal_receipt = _receipt_from_data(proposal_data["receipt"], "proposal")
    try:
        proposal = parse_rule_proposal(
            proposal_payload,
            receipt=proposal_receipt,
            observable_catalog={},
        )
    except (ProposalError, TypeError, ValueError) as exc:
        raise RunVerificationError(f"archived proposal cannot be parsed: {exc}") from exc
    if proposal.to_dict() != dict(proposal_data):
        raise RunVerificationError(
            "archived proposal fields do not reproduce from model_payload and receipt"
        )
    proposer_digest = proposal.digest.removeprefix("sha256:")
    if proposer_digest != archive.bundle.freeze.proposer_digest:
        raise RunVerificationError("parsed visual proposal differs from frozen proposer_digest")

    support_identities, named_support = _support_presentation(archive)
    support_panel_set = proposal_receipt.panel_set_digest
    if proposal_receipt.input_digest_schema != STRUCTURED_INPUT_DIGEST_SCHEMA:
        raise RunVerificationError("proposal receipt is not a labelled-support receipt")

    query_blobs = _query_presentation(archive)
    all_blobs = tuple(blob for _name, blob in named_support) + tuple(
        blob for _query_id, blob in query_blobs
    )
    verified_blob_ids, missing_blob_ids, supplied_bytes = _verify_supplied_blob_bytes(
        all_blobs, blob_bytes_by_id
    )
    support_blob_ids = {blob.blob_id for _name, blob in named_support}
    if support_blob_ids <= set(supplied_bytes):
        support_panel_set = _semantic_support_digest_from_bytes(
            named_support, supplied_bytes
        )
    _verify_receipt_payload(
        proposal_receipt,
        proposal_payload,
        HYBRID_ONLY_RULE_PROPOSAL_SCHEMA,
        proposer_prompt({}),
        input_schema=STRUCTURED_INPUT_DIGEST_SCHEMA,
        identities=support_identities,
        panel_set_digest=support_panel_set,
        label="proposal",
    )
    _verify_canonical_hybrid_compilation(
        proposal,
        plan=plan,
        archive=archive,
    )

    receipt_digests = [proposal_receipt.receipt_digest]
    thread_ids = {proposal_receipt.thread_id}
    support_observations = _verify_support_replay_gate(
        vision["support_gate"],
        archived_freeze=vision["proposal_freeze"],
        archive=archive,
        proposal=proposal,
        proposal_receipt=proposal_receipt,
        receipt_digests=receipt_digests,
        thread_ids=thread_ids,
    )

    exposure_model = exposure.get("model")
    if exposure_model != proposal_receipt.requested_model:
        raise RunVerificationError("exposure model differs from proposal receipt model")
    observation_values = _mapping(vision["observations"], "vision observations")
    released_ids = tuple(query_id for query_id, _blob in query_blobs)
    if set(observation_values) != set(released_ids):
        raise RunVerificationError(
            "vision observations must cover exactly the released public query IDs"
        )
    cold_by_id = {
        query.query_id: query for query in archive.bundle.cold_inputs.queries
    }
    observations: list[tuple[str, HybridObservation]] = []
    for query_id, query_blob in query_blobs:
        observation_data = _mapping(
            observation_values[query_id], f"vision observation {query_id}"
        )
        _expect_fields(
            observation_data, _OBSERVATION_FIELDS, f"vision observation {query_id}"
        )
        if observation_data["schema"] != HYBRID_OBSERVATION_SCHEMA_VERSION:
            raise RunVerificationError(f"{query_id}: observation schema differs")
        payload = _mapping(observation_data["payload"], f"{query_id} payload")
        receipt = _receipt_from_data(observation_data["receipt"], query_id)
        _verify_transport_identity(receipt, proposal_receipt, query_id)
        if receipt.thread_id in thread_ids or receipt.receipt_digest in receipt_digests:
            raise RunVerificationError(f"{query_id}: Codex receipt identity is reused")
        thread_ids.add(receipt.thread_id)
        receipt_digests.append(receipt.receipt_digest)

        identities = [_image_identity("query.png", query_blob)]
        named_set_digest = "sha256:" + canonical_digest(
            {"schema": NAMED_IMAGE_INPUT_DIGEST_SCHEMA, "images": identities}
        )
        _verify_receipt_payload(
            receipt,
            payload,
            HYBRID_OBSERVATION_SCHEMA,
            hybrid_observer_prompt(proposal),
            input_schema=NAMED_IMAGE_INPUT_DIGEST_SCHEMA,
            identities=identities,
            panel_set_digest=named_set_digest,
            label=query_id,
        )
        try:
            observation = parse_hybrid_observation_or_error(
                proposal, payload, receipt
            )
        except (ProposalError, TypeError, ValueError) as exc:
            raise RunVerificationError(f"{query_id}: observation cannot be parsed: {exc}") from exc
        if observation.to_dict() != dict(observation_data):
            raise RunVerificationError(
                f"{query_id}: archived observation does not reproduce from payload/receipt"
            )
        if observation.proposal_digest != proposal.digest:
            raise RunVerificationError(f"{query_id}: observation belongs to another proposal")

        cold = cold_by_id[query_id]
        if len(cold.atom_inputs) != 1 or cold.atom_inputs[0].path != ():
            raise RunVerificationError(
                f"{query_id}: HYBRID cold evidence must contain exactly the root atom"
            )
        projected = truth_from_hybrid_observation(observation)
        reconstructed = TruthEvidenceRecord.from_evidence(projected).to_data()
        if reconstructed != cold.atom_inputs[0].evidence.to_data():
            raise RunVerificationError(
                f"{query_id}: reconstructed vision Evidence/provenance differs from cold atom evidence"
            )
        observations.append((query_id, observation))

    unbound = (
        "corpus manifest bytes are not embedded; only its digest is tied to support",
        "task manifest bytes and split assignment require the supplied corpus",
        "seed_digest commits the private seed but the seed preimage is intentionally absent",
        "split source bytes are not embedded",
        "exposure-ledger predecessor authenticity needs the ledger or an external anchor",
    )
    return CompletedRunVerification(
        run_id=archive.bundle.support.run_id,
        record_digest=record_digest,
        archive_digest=archive.archive_digest,
        chain_digest=archive.replay_receipt.chain_digest,
        plan_digest=canonical_digest(dict(plan)),
        proposal_digest=proposer_digest,
        official_release_digest=(
            official_release.digest if official_release is not None else None
        ),
        query_ids=released_ids,  # type: ignore[arg-type]
        receipt_digests=tuple(receipt_digests),
        verified_blob_ids=verified_blob_ids,
        missing_blob_preimages=missing_blob_ids,
        unbound_outer_fields=unbound,
        archive=archive,
        proposal=proposal,
        observations=tuple(observations),
        support_observations=support_observations,
    )


def verify_completed_run_data(
    record: Mapping[str, Any],
    *,
    blob_bytes_by_id: Mapping[str, bytes] | None = None,
) -> CompletedRunVerification:
    """Strictly verify a completed run, including all panel byte preimages."""

    report = audit_completed_run_commitments(
        record, blob_bytes_by_id=blob_bytes_by_id
    )
    if report.missing_blob_preimages:
        raise MissingBlobPreimagesError(report.missing_blob_preimages)
    return report


def verify_rejected_run_data(
    record: Mapping[str, Any],
    *,
    support_bytes_by_name: Mapping[str, bytes] | None = None,
) -> RejectedRunVerification:
    """Strictly verify a semantic proposal rejection and all support PNGs."""

    report = audit_rejected_run_commitments(
        record, support_bytes_by_name=support_bytes_by_name
    )
    if report.missing_support_preimages:
        raise MissingSupportPreimagesError(report.missing_support_preimages)
    return report


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RunVerificationError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def verify_completed_run_bytes(
    payload: bytes | str,
    *,
    blob_bytes_by_id: Mapping[str, bytes] | None = None,
    require_blob_preimages: bool = True,
) -> CompletedRunVerification:
    """Reject non-canonical/duplicate-key JSON, then verify the completed run."""

    raw = payload.encode("utf-8") if isinstance(payload, str) else payload
    try:
        decoded = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except RunVerificationError:
        raise
    except (UnicodeError, json.JSONDecodeError, TypeError) as exc:
        raise RunVerificationError(f"cannot decode outer run record: {exc}") from exc
    if not isinstance(decoded, Mapping):
        raise RunVerificationError("outer run record root must be a JSON object")
    try:
        expected = canonical_json(decoded)
    except ValueError as exc:
        raise RunVerificationError(str(exc)) from exc
    if raw != expected:
        raise RunVerificationError("outer run record bytes are not canonical JSON")
    if require_blob_preimages:
        return verify_completed_run_data(decoded, blob_bytes_by_id=blob_bytes_by_id)
    return audit_completed_run_commitments(
        decoded, blob_bytes_by_id=blob_bytes_by_id
    )


def verify_rejected_run_bytes(
    payload: bytes | str,
    *,
    support_bytes_by_name: Mapping[str, bytes] | None = None,
    require_support_preimages: bool = True,
) -> RejectedRunVerification:
    """Reject non-canonical JSON, then replay a semantic proposal rejection."""

    raw = payload.encode("utf-8") if isinstance(payload, str) else payload
    try:
        decoded = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except RunVerificationError:
        raise
    except (UnicodeError, json.JSONDecodeError, TypeError) as exc:
        raise RunVerificationError(f"cannot decode outer rejected run: {exc}") from exc
    if not isinstance(decoded, Mapping):
        raise RunVerificationError("outer rejected run root must be a JSON object")
    try:
        expected = canonical_json(decoded)
    except ValueError as exc:
        raise RunVerificationError(str(exc)) from exc
    if raw != expected:
        raise RunVerificationError("outer rejected run bytes are not canonical JSON")
    if require_support_preimages:
        return verify_rejected_run_data(
            decoded, support_bytes_by_name=support_bytes_by_name
        )
    return audit_rejected_run_commitments(
        decoded, support_bytes_by_name=support_bytes_by_name
    )


__all__ = [
    "CompletedRunVerification",
    "EXPOSURE_SCHEMA",
    "MissingBlobPreimagesError",
    "MissingSupportPreimagesError",
    "OUTER_RUN_SCHEMA",
    "RejectedProposalAttemptVerification",
    "RejectedRunVerification",
    "RunVerificationError",
    "VISION_EPISODE_SCHEMA",
    "audit_completed_run_commitments",
    "audit_rejected_proposal_attempt_data",
    "audit_rejected_run_commitments",
    "verify_completed_run_bytes",
    "verify_completed_run_data",
    "verify_rejected_run_bytes",
    "verify_rejected_run_data",
]
