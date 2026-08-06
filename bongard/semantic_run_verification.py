"""Strict outer archive and cold replay for visual-semantic episodes.

The verifier in this module never invokes Codex or a proof assistant.  It
reconstructs the calibrated scorer family, visual-semantic policy, typed
proposal transport, Python compiler, pre-observation commitment, all panel
observations, support gate, generic proposal freeze, and the two-query artifact
chain.  Exact PNG preimages are supplied by BlobRef id and every verifier-owned
visual witness and registered Python atom is recomputed from those bytes.
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
    SupportGateArtifact,
    SupportGateEntry,
    SupportGatePolicy,
    SupportGateResult,
    VISUAL_SEMANTIC_PREDICATE_MODE,
)
from bongard.blind_soft_transport import (
    BlindSoftVerifierContext,
)
from bongard.evidence import Disposition
from bongard.proposer import SupportPanelIdentity
from bongard.release import OfficialReleaseDescriptor, ReleaseIdentityError
from bongard.semantic_calibration import SemanticCalibrationArtifact
from bongard.semantic_calibration_campaign import (
    SemanticCalibrationCampaignArtifact,
)
from bongard.semantic_commitment import SemanticPreObservationCommitment
from bongard.semantic_episode import (
    SEMANTIC_EPISODE_ARCHIVE_SCHEMA,
    VisualSemanticEpisode,
)
from bongard.semantic_observation import (
    VisualSemanticObservationArtifact,
    replay_semantic_atom_evidence,
)
from bongard.semantic_protocol import build_visual_semantic_policy
from bongard.semantic_synthesis import (
    CompiledVisualSemanticProposal,
    compile_visual_semantic_proposal,
)
from bongard.transport import (
    CodexReceipt,
    named_image_set_digest,
    semantic_panel_set_digest,
)
from bongard.typed_visual_transport import (
    RejectedTypedVisualProposalAttempt,
    TypedVisualProposalRejected,
    TypedVisualTransportResult,
)
from bongard.visual_predicate_catalog import DIRECT_VISUAL_ATOM_CATALOG


VISUAL_SEMANTIC_OUTER_RUN_SCHEMA = "gkm.bongard-visual-semantic-run.v2"
VISUAL_SEMANTIC_COMPACT_OUTER_RUN_SCHEMA = (
    "gkm.bongard-visual-semantic-campaign-referenced-run.v1"
)

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
_SEMANTIC_ARTIFACT_FIELDS = {
    "schema",
    "predicate_mode",
    "predicate_policy_digest",
    "pre_observation_commitment",
    "rejected_proposal_attempt",
    "support_observations",
    "query_observations",
    "python_predicate_authoritative",
    "optional_checker_may_affect_result",
    "support_gate",
    "proposal_freeze",
}
_OUTER_FIELDS = {
    "schema",
    "corpus_manifest_digest",
    "split_source_digest",
    "official_release",
    "calibration_campaign_digest",
    "calibration",
    "plan",
    "support_commitment",
    "episode",
    "visual_semantic",
    "run_archive",
    "exposure",
    "record_digest",
}
_COMPACT_OUTER_FIELDS = (
    _OUTER_FIELDS - {"calibration"}
) | {"calibration_digest"}
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


class VisualSemanticRunVerificationError(ValueError):
    """A persisted visual-semantic run fails deterministic reconstruction."""


@dataclass(frozen=True)
class VisualSemanticCalibrationCampaignAnchor:
    """Externally authenticated Stage-A campaign and its exact calibration.

    ``campaign_digest`` is the campaign's internal canonical content digest,
    not the persistence-file SHA-256.  This is the narrow authority accepted by
    the lower-level run verifier.  A public caller obtains it only after
    cold-replaying a complete campaign
    against a trusted corpus.  Keeping the full campaign external avoids
    copying it into every run while still making both identities mandatory.
    """

    campaign_digest: str
    calibration: SemanticCalibrationArtifact
    expected_codex_launcher_digest: str
    cloud_policy_cache_binding: str
    _calibration_digest: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.campaign_digest, str)
            or _HEX.fullmatch(self.campaign_digest) is None
        ):
            raise VisualSemanticRunVerificationError(
                "calibration campaign digest must be a lowercase SHA-256"
            )
        if not isinstance(self.calibration, SemanticCalibrationArtifact):
            raise TypeError(
                "campaign anchor calibration must be a SemanticCalibrationArtifact"
            )
        try:
            self.calibration.assert_untampered()
        except (TypeError, ValueError) as exc:
            raise VisualSemanticRunVerificationError(
                f"campaign anchor calibration is invalid: {exc}"
            ) from exc
        object.__setattr__(self, "_calibration_digest", self.calibration.digest)
        if (
            not isinstance(self.expected_codex_launcher_digest, str)
            or _HEX.fullmatch(self.expected_codex_launcher_digest) is None
        ):
            raise VisualSemanticRunVerificationError(
                "campaign anchor expected Codex launcher digest must be a "
                "lowercase SHA-256"
            )
        if self.cloud_policy_cache_binding != "absent" and (
            not isinstance(self.cloud_policy_cache_binding, str)
            or _PREFIXED_HEX.fullmatch(self.cloud_policy_cache_binding) is None
        ):
            raise VisualSemanticRunVerificationError(
                "campaign anchor cloud-policy cache binding must be absent or "
                "a sha256: content address"
            )

    @property
    def calibration_digest(self) -> str:
        return self._calibration_digest

    @classmethod
    def from_verified_campaign(
        cls, campaign: SemanticCalibrationCampaignArtifact
    ) -> "VisualSemanticCalibrationCampaignAnchor":
        if not isinstance(campaign, SemanticCalibrationCampaignArtifact):
            raise TypeError(
                "a full SemanticCalibrationCampaignArtifact is required; "
                "a bare calibration artifact is not accepted"
            )
        campaign.assert_untampered()
        execution_config = (
            campaign.score_batch.commitment_batch.proposal_archive.execution_config
        )
        return cls(
            campaign.digest,
            campaign.calibration,
            execution_config.expected_codex_launcher_digest,
            execution_config.cloud_policy_cache_binding,
        )


def _campaign_anchor(
    value: object,
) -> VisualSemanticCalibrationCampaignAnchor:
    if isinstance(value, VisualSemanticCalibrationCampaignAnchor):
        # Reconstructing also rechecks the possibly mutated calibration graph.
        return VisualSemanticCalibrationCampaignAnchor(
            value.campaign_digest,
            value.calibration,
            value.expected_codex_launcher_digest,
            value.cloud_policy_cache_binding,
        )
    if isinstance(value, SemanticCalibrationCampaignArtifact):
        return VisualSemanticCalibrationCampaignAnchor.from_verified_campaign(value)
    if isinstance(value, SemanticCalibrationArtifact):
        raise VisualSemanticRunVerificationError(
            "a bare SemanticCalibrationArtifact is not a campaign authority"
        )
    raise VisualSemanticRunVerificationError(
        "visual-semantic verification requires an externally verified full "
        "Stage-A campaign or an explicit campaign anchor"
    )


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise VisualSemanticRunVerificationError(f"{label} must be a JSON object")
    return value


def _expect_fields(
    value: Mapping[str, Any], expected: set[str], label: str
) -> None:
    if set(value) != expected:
        raise VisualSemanticRunVerificationError(
            f"{label} fields differ from schema: "
            f"missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _hex(value: object, label: str, *, prefixed: bool = False) -> str:
    pattern = _PREFIXED_HEX if prefixed else _HEX
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        kind = "sha256: content address" if prefixed else "lowercase SHA-256"
        raise VisualSemanticRunVerificationError(f"{label} must be a {kind}")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise VisualSemanticRunVerificationError(
            f"{label} must be canonical non-empty text"
        )
    return value


def _blob_from_data(value: object, label: str) -> BlobRef:
    data = _mapping(value, label)
    _expect_fields(data, {"blob_id", "sha256", "byte_count", "media_type"}, label)
    try:
        result = BlobRef(
            data["blob_id"],
            data["sha256"],
            data["byte_count"],
            data["media_type"],
        )
    except (TypeError, ValueError) as exc:
        raise VisualSemanticRunVerificationError(
            f"{label} is invalid: {exc}"
        ) from exc
    if result.to_data() != dict(data):
        raise VisualSemanticRunVerificationError(f"{label} round-trip drift")
    return result


def _normalise_preimages(value: Mapping[str, bytes]) -> dict[str, bytes]:
    if not isinstance(value, Mapping) or any(
        not isinstance(blob_id, str) for blob_id in value
    ):
        raise VisualSemanticRunVerificationError(
            "blob_bytes_by_id must map blob ids to exact bytes"
        )
    result: dict[str, bytes] = {}
    for blob_id, payload in value.items():
        _text(blob_id, "blob byte id")
        if not isinstance(payload, bytes) or not payload:
            raise VisualSemanticRunVerificationError(
                f"blob byte preimage {blob_id} must be nonempty bytes"
            )
        result[blob_id] = payload
    return result


def _verify_blob_preimages(
    preimages: Mapping[str, bytes], expected: Mapping[str, BlobRef]
) -> None:
    if set(preimages) != set(expected):
        raise VisualSemanticRunVerificationError(
            "blob byte ids differ from released support/query BlobRefs: "
            f"missing={sorted(set(expected) - set(preimages))}, "
            f"extra={sorted(set(preimages) - set(expected))}"
        )
    for blob_id, blob in expected.items():
        try:
            blob.verify_bytes(preimages[blob_id])
        except ArtifactTamperError as exc:
            raise VisualSemanticRunVerificationError(
                f"blob preimage {blob_id} differs from its BlobRef"
            ) from exc


def _support_named_blobs(
    support: SupportCommitment,
) -> tuple[tuple[str, BlobRef], ...]:
    by_id = {item.panel.blob_id: item for item in support.support}
    specification = tuple(
        [(f"pos_{index}.png", f"support-positive-{index}", True) for index in range(6)]
        + [(f"neg_{index}.png", f"support-negative-{index}", False) for index in range(6)]
    )
    if set(by_id) != {blob_id for _name, blob_id, _positive in specification}:
        raise VisualSemanticRunVerificationError(
            "support commitment is not the canonical 6+6 presentation"
        )
    result: list[tuple[str, BlobRef]] = []
    for name, blob_id, positive in specification:
        item = by_id[blob_id]
        if item.positive is not positive:
            raise VisualSemanticRunVerificationError(
                f"support panel {blob_id} has the wrong polarity"
            )
        result.append((name, item.panel))
    return tuple(result)


def _semantic_support_digest(
    named: Sequence[tuple[str, BlobRef]], preimages: Mapping[str, bytes]
) -> str:
    with tempfile.TemporaryDirectory(prefix="bongard-semantic-cold-proposal-") as root:
        paths: list[str] = []
        for name, blob in named:
            path = Path(root) / name
            path.write_bytes(preimages[blob.blob_id])
            paths.append(str(path))
        return semantic_panel_set_digest(paths)


def _verify_proposal_pixels(
    transport: TypedVisualTransportResult | RejectedTypedVisualProposalAttempt,
    *,
    support: SupportCommitment,
    preimages: Mapping[str, bytes],
) -> None:
    named = _support_named_blobs(support)
    expected = tuple(
        SupportPanelIdentity(name, blob.byte_count, blob.sha256)
        for name, blob in named
    )
    if transport.support_presentation != expected:
        raise VisualSemanticRunVerificationError(
            "typed proposal presentation differs from support BlobRefs"
        )
    if transport.receipt.panel_set_digest != _semantic_support_digest(
        named, preimages
    ):
        raise VisualSemanticRunVerificationError(
            "typed proposal receipt differs from exact support pixels"
        )


def _verify_successful_codex_environment(
    receipt: CodexReceipt,
    *,
    campaign_anchor: VisualSemanticCalibrationCampaignAnchor,
    label: str,
) -> None:
    """Bind retained successful model evidence to the Stage-A environment."""

    if not isinstance(receipt, CodexReceipt):
        raise VisualSemanticRunVerificationError(
            f"{label} lacks a successful Codex receipt"
        )
    if (
        receipt.codex_launcher_digest
        != campaign_anchor.expected_codex_launcher_digest
    ):
        raise VisualSemanticRunVerificationError(
            f"{label} Codex launcher differs from the Stage-A campaign"
        )
    if (
        receipt.cloud_config_bundle_cache_binding
        != campaign_anchor.cloud_policy_cache_binding
    ):
        raise VisualSemanticRunVerificationError(
            f"{label} cloud-policy cache differs from the Stage-A campaign"
        )


def _verify_scorer_pixels(
    artifact: VisualSemanticObservationArtifact,
    panel_bytes: bytes,
    *,
    campaign_anchor: VisualSemanticCalibrationCampaignAnchor,
) -> None:
    scorer = artifact.scorer_artifact
    if scorer is None:
        return
    if not isinstance(scorer.receipt, CodexReceipt):
        # The typed scorer artifact already proves that a non-Codex receipt is
        # an explicit failure outcome with no score.  It is retained for
        # attrition/error accounting, never promoted to successful evidence.
        if scorer.record.score is not None:
            raise VisualSemanticRunVerificationError(
                "blind scorer transport failure contains successful evidence"
            )
        return
    _verify_successful_codex_environment(
        scorer.receipt,
        campaign_anchor=campaign_anchor,
        label="blind scorer receipt",
    )
    with tempfile.TemporaryDirectory(prefix="bongard-semantic-cold-score-") as root:
        path = Path(root) / "query.png"
        path.write_bytes(panel_bytes)
        reproduced = named_image_set_digest((str(path),), ("query.png",))
    if scorer.receipt.panel_set_digest != reproduced:
        raise VisualSemanticRunVerificationError(
            "blind scorer receipt differs from exact panel pixels"
        )


def _expected_context(
    *,
    task_id: str,
    phase: str,
    ordinal: int,
    precommit: SemanticPreObservationCommitment,
) -> BlindSoftVerifierContext:
    transport = precommit.proposal_transport
    scorer_call_id = "score-" + canonical_digest(
        {
            "schema": "gkm.bongard-semantic-score-call-id.v1",
            "pre_observation_commitment_digest": precommit.digest,
            "phase": phase,
            "ordinal": ordinal,
        }
    )[:40]
    return BlindSoftVerifierContext(
        task_id=task_id,
        panel_id=f"{phase}-panel-{ordinal:02d}",
        proposer_call_id=transport.receipt.thread_id,
        proposer_receipt_digest=transport.receipt.receipt_digest,
        scorer_call_id=scorer_call_id,
        pre_observation_commitment_digest=precommit.digest,
    )


def _replay_registered_atoms(
    artifact: VisualSemanticObservationArtifact,
    compiled: CompiledVisualSemanticProposal,
) -> None:
    if replay_semantic_atom_evidence(artifact, compiled) != artifact.atom_evidence:
        raise VisualSemanticRunVerificationError(
            "semantic atom evidence differs from fresh Python evaluation"
        )


def _decode_observation(
    value: object,
    *,
    label: str,
    panel: BlobRef,
    panel_bytes: bytes,
    compiled: CompiledVisualSemanticProposal,
    calibration: SemanticCalibrationArtifact,
    precommit: SemanticPreObservationCommitment,
    task_id: str,
    phase: str,
    ordinal: int,
    campaign_anchor: VisualSemanticCalibrationCampaignAnchor,
) -> VisualSemanticObservationArtifact:
    try:
        artifact = VisualSemanticObservationArtifact.from_data(
            _mapping(value, label),
            compiled=compiled,
            protocol=calibration.protocol,
            panel_png=panel_bytes,
        )
    except (TypeError, ValueError) as exc:
        raise VisualSemanticRunVerificationError(
            f"{label} is invalid: {exc}"
        ) from exc
    if (
        artifact.panel_digest != panel.sha256
        or artifact.panel_byte_count != panel.byte_count
    ):
        raise VisualSemanticRunVerificationError(f"{label} names another panel")
    if artifact.pre_observation_commitment_digest != precommit.digest:
        raise VisualSemanticRunVerificationError(
            f"{label} belongs to another pre-observation commitment"
        )
    if artifact.scorer_artifact is not None and (
        artifact.scorer_artifact.context
        != _expected_context(
            task_id=task_id,
            phase=phase,
            ordinal=ordinal,
            precommit=precommit,
        )
    ):
        raise VisualSemanticRunVerificationError(
            f"{label} scorer context differs from the isolated call slot"
        )
    _verify_scorer_pixels(
        artifact,
        panel_bytes,
        campaign_anchor=campaign_anchor,
    )
    _replay_registered_atoms(artifact, compiled)
    return artifact


def _verify_plan_episode(
    plan_value: object,
    episode_value: object,
    *,
    support: SupportCommitment,
    policy_digest: str,
    corpus_manifest_digest: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any], str]:
    plan = _mapping(plan_value, "public visual-semantic plan")
    episode = _mapping(episode_value, "outer visual-semantic episode")
    _expect_fields(plan, _BASE_PLAN_FIELDS, "public visual-semantic plan")
    _expect_fields(episode, _EPISODE_FIELDS, "outer visual-semantic episode")
    if plan["version"] != PROTOCOL_VERSION or episode["version"] != PROTOCOL_VERSION:
        raise VisualSemanticRunVerificationError("benchmark protocol version differs")
    for name in ("task_id", "family", "split", "regime", "run_id"):
        if plan[name] != episode[name]:
            raise VisualSemanticRunVerificationError(
                f"public plan {name} differs from episode"
            )
    if plan["predicate_mode"] != VISUAL_SEMANTIC_PREDICATE_MODE:
        raise VisualSemanticRunVerificationError(
            "public plan is not visual-semantic mode"
        )
    if plan["predicate_policy_digest"] != policy_digest:
        raise VisualSemanticRunVerificationError(
            "public plan predicate policy digest differs"
        )
    for name in (
        "seed_digest",
        "corpus_digest",
        "task_manifest_digest",
        "support_commitment_digest",
        "latent_query_digest",
        "label_commitment_digest",
    ):
        _hex(plan[name], f"public plan {name}")
    if plan["corpus_digest"] != corpus_manifest_digest.removeprefix("sha256:"):
        raise VisualSemanticRunVerificationError("public plan corpus digest differs")
    if support.run_id != plan["run_id"] or support.issued_by != plan["verifier_id"]:
        raise VisualSemanticRunVerificationError(
            "support commitment identity differs from public plan"
        )
    if support.corpus_digest != plan["corpus_digest"]:
        raise VisualSemanticRunVerificationError(
            "support commitment corpus differs from public plan"
        )
    if support.digest() != plan["support_commitment_digest"]:
        raise VisualSemanticRunVerificationError(
            "support commitment digest differs from public plan"
        )
    if canonical_digest(dict(plan)) != episode["plan_digest"]:
        raise VisualSemanticRunVerificationError("episode plan digest does not reproduce")
    status = _text(episode["status"], "episode status")
    if status not in {
        EpisodeStatus.COMPLETE.value,
        EpisodeStatus.SUPPORT_REJECTED.value,
        EpisodeStatus.PROPOSAL_ERROR.value,
    }:
        raise VisualSemanticRunVerificationError(
            f"unsupported visual-semantic episode status {status!r}"
        )
    return plan, episode, status


def _verify_opened_plan_commitments(
    plan: Mapping[str, Any], archive: VerifiedRunArchive
) -> None:
    """Join pre-run query/label seals to their post-run openings.

    The generic artifact chain proves the release/prediction/reveal order, but
    its digest alone does not prove that the public episode plan committed to
    those same hidden queries and labels.  Reconstruct both plan preimages from
    the cold-verified archive so a self-rehashed outer record cannot substitute
    either commitment after the run.
    """

    bundle = archive.bundle
    latent_query_digest = canonical_digest(
        {
            "version": "latent-two-query-commitment/v1",
            "run_id": bundle.release.run_id,
            "queries": [item.to_data() for item in bundle.release.queries],
        }
    )
    if plan["latent_query_digest"] != latent_query_digest:
        raise VisualSemanticRunVerificationError(
            "public plan latent query commitment differs from archive release"
        )
    label_commitment_digest = canonical_digest(
        {
            "run_id": bundle.labels.run_id,
            "labels": [item.to_data() for item in bundle.labels.labels],
            "nonce": bundle.labels.verifier_nonce,
            "version": "latent-label-seal/v1",
        }
    )
    if plan["label_commitment_digest"] != label_commitment_digest:
        raise VisualSemanticRunVerificationError(
            "public plan label seal differs from archive label reveal"
        )


def _reconstruct_gate(
    value: object,
    *,
    support: SupportCommitment,
    precommit: SemanticPreObservationCommitment,
    observations: Sequence[VisualSemanticObservationArtifact],
) -> SupportGateArtifact:
    raw = _mapping(value, "visual-semantic support gate")
    if len(observations) != 12 or len(support.support) != 12:
        raise VisualSemanticRunVerificationError(
            "visual-semantic support replay requires twelve observations"
        )
    raw_entries = raw.get("ordered_entries")
    if not isinstance(raw_entries, list) or len(raw_entries) != 12:
        raise VisualSemanticRunVerificationError(
            "visual-semantic gate lacks twelve ordered entries"
        )
    entries: list[SupportGateEntry] = []
    dispositions: list[Disposition] = []
    forward = 0
    reverse = 0
    for index, (raw_entry, expected, observation) in enumerate(
        zip(raw_entries, support.support, observations, strict=True)
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
        try:
            evidence = TruthEvidenceRecord.from_data(
                _mapping(entry["evidence"], f"support gate evidence {index}")
            )
        except (TypeError, ValueError) as exc:
            raise VisualSemanticRunVerificationError(
                f"support gate evidence {index} is invalid: {exc}"
            ) from exc
        if (
            entry["slot_id"] != expected.panel.blob_id
            or entry["positive"] is not expected.positive
            or _blob_from_data(entry["panel"], f"support gate panel {index}")
            != expected.panel
            or canonical_json(entry["observer_artifact"])
            != canonical_json(observation.to_data())
            or evidence != observation.formula_evidence
            or entry["transport_attempted"] is not observation.transport_attempted
        ):
            raise VisualSemanticRunVerificationError(
                f"support gate entry {index} differs from replayed observation"
            )
        disposition = evidence.disposition
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
        entries.append(
            SupportGateEntry(
                slot_id=expected.panel.blob_id,
                panel=expected.panel,
                positive=expected.positive,
                evidence=evidence,
                observer_artifact=observation.to_data(),
                transport_attempted=observation.transport_attempted,
            )
        )
    counts = Counter(dispositions)
    if counts[Disposition.INDETERMINATE] or counts[Disposition.ERROR]:
        result = SupportGateResult.OBSERVER_FAILURE
    elif forward == 12:
        result = SupportGateResult.ALIGNED
    elif reverse > forward:
        result = SupportGateResult.MISORIENTED
    else:
        result = SupportGateResult.UNSUPPORTED
    gate = SupportGateArtifact(
        run_id=support.run_id,
        proposal_digest=precommit.digest,
        support_commitment_digest=support.digest(),
        policy=SupportGatePolicy.visual_semantic(),
        entries=tuple(entries),
        forward_matches=forward,
        reverse_matches=reverse,
        present_count=counts[Disposition.PRESENT],
        nonmatch_count=counts[Disposition.CERTIFIED_ABSENT],
        indeterminate_count=counts[Disposition.INDETERMINATE],
        error_count=counts[Disposition.ERROR],
        transport_attempt_count=sum(item.transport_attempted for item in observations),
        result=result,
    )
    if canonical_json(gate.to_data()) != canonical_json(dict(raw)):
        raise VisualSemanticRunVerificationError(
            "visual-semantic support gate does not reproduce"
        )
    return gate


def _verify_freeze(
    value: object,
    *,
    support: SupportCommitment,
    compiled: CompiledVisualSemanticProposal,
    precommit: SemanticPreObservationCommitment,
    gate: SupportGateArtifact,
    archive: VerifiedRunArchive | None,
) -> ProposalFreeze:
    try:
        freeze = verify_proposal_freeze_data(value)
    except (TypeError, ValueError) as exc:
        raise VisualSemanticRunVerificationError(
            f"visual-semantic proposal freeze is invalid: {exc}"
        ) from exc
    expected_proposal_id = "visual-semantic-" + compiled.proposal.digest[:16]
    checks = (
        (freeze.run_id, support.run_id, "run ID"),
        (freeze.proposal_id, expected_proposal_id, "proposal ID"),
        (freeze.support_commitment_digest, support.digest(), "support commitment"),
        (freeze.formula, compiled.formula, "compiled formula"),
        (freeze.proposer_digest, precommit.digest, "pre-observation commitment"),
        (freeze.registry_digest, compiled.registry.digest(), "registry"),
        (
            freeze.attachment_contract_digest,
            compiled.attachment_contract.digest(),
            "attachment contract",
        ),
        (freeze.support_gate_digest, gate.digest, "support gate"),
    )
    for actual, expected, label in checks:
        if actual != expected:
            raise VisualSemanticRunVerificationError(
                f"generic proposal freeze differs from {label}"
            )
    if archive is not None:
        if archive.bundle.freeze != freeze:
            raise VisualSemanticRunVerificationError(
                "generic archive freeze differs from standalone freeze"
            )
        if archive.bundle.attachment_contract.to_data() != (
            compiled.attachment_contract.to_data()
        ):
            raise VisualSemanticRunVerificationError(
                "generic archive attachment differs from Python compilation"
            )
    return freeze


def _expected_score(archive: VerifiedRunArchive) -> dict[str, object]:
    labels = {item.query_id: item.positive for item in archive.bundle.labels.labels}
    predictions = {
        item.query_id: item for item in archive.bundle.predictions.predictions
    }
    correct = sum(
        item.positive is not None and item.positive == labels[query_id]
        for query_id, item in predictions.items()
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


@dataclass(frozen=True)
class VisualSemanticRunVerification:
    run_id: str
    status: str
    record_digest: str
    plan_digest: str
    calibration_campaign_digest: str
    calibration_digest: str
    expected_codex_launcher_digest: str
    cloud_policy_cache_binding: str
    policy_digest: str
    proposal_transport_digest: str | None
    proposal_receipt_digest: str | None
    pre_observation_commitment_digest: str | None
    lowering_archive_digest: str | None
    support_gate_digest: str | None
    proposal_freeze_digest: str | None
    prediction_commitment_digest: str | None
    archive_digest: str | None
    verified_blob_ids: tuple[str, ...]
    support_observation_digests: tuple[str, ...]
    query_observation_digests: tuple[str, ...]
    registered_atom_replays: int
    optional_checker_required: bool = False
    archive: VerifiedRunArchive | None = field(default=None, repr=False, compare=False)

    def to_data(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "record_digest": self.record_digest,
            "plan_digest": self.plan_digest,
            "calibration_campaign_digest": self.calibration_campaign_digest,
            "calibration_digest": self.calibration_digest,
            "expected_codex_launcher_digest": (
                self.expected_codex_launcher_digest
            ),
            "cloud_policy_cache_binding": self.cloud_policy_cache_binding,
            "policy_digest": self.policy_digest,
            "proposal_transport_digest": self.proposal_transport_digest,
            "proposal_receipt_digest": self.proposal_receipt_digest,
            "pre_observation_commitment_digest": (
                self.pre_observation_commitment_digest
            ),
            "lowering_archive_digest": self.lowering_archive_digest,
            "support_gate_digest": self.support_gate_digest,
            "proposal_freeze_digest": self.proposal_freeze_digest,
            "prediction_commitment_digest": self.prediction_commitment_digest,
            "archive_digest": self.archive_digest,
            "verified_blob_ids": list(self.verified_blob_ids),
            "support_observation_digests": list(
                self.support_observation_digests
            ),
            "query_observation_digests": list(self.query_observation_digests),
            "registered_atom_replays": self.registered_atom_replays,
            "optional_checker_required": self.optional_checker_required,
        }


def _verify_visual_semantic_run_data(
    record_value: Mapping[str, Any],
    *,
    blob_bytes_by_id: Mapping[str, bytes],
    calibration_campaign: object | None,
    expected_task_manifest_digest: str | None,
    verified_campaign_anchor: VisualSemanticCalibrationCampaignAnchor | None = None,
) -> VisualSemanticRunVerification:
    if verified_campaign_anchor is None:
        campaign_anchor = _campaign_anchor(calibration_campaign)
    elif not isinstance(
        verified_campaign_anchor, VisualSemanticCalibrationCampaignAnchor
    ):
        raise TypeError(
            "verified_campaign_anchor must be a calibration campaign anchor"
        )
    else:
        # This private path is used only after the enclosing Stage-B artifact
        # has authenticated its one central full campaign.  Avoid rechecking
        # that same immutable calibration once per task.
        campaign_anchor = verified_campaign_anchor
    record = _mapping(record_value, "visual-semantic outer run")
    schema = record.get("schema")
    if schema == VISUAL_SEMANTIC_OUTER_RUN_SCHEMA:
        _expect_fields(record, _OUTER_FIELDS, "visual-semantic outer run")
        compact_calibration = False
    elif schema == VISUAL_SEMANTIC_COMPACT_OUTER_RUN_SCHEMA:
        _expect_fields(
            record,
            _COMPACT_OUTER_FIELDS,
            "campaign-referenced visual-semantic outer run",
        )
        compact_calibration = True
    else:
        raise VisualSemanticRunVerificationError(
            "visual-semantic outer schema differs"
        )
    record_digest = _hex(record["record_digest"], "record digest")
    content = {key: value for key, value in record.items() if key != "record_digest"}
    if canonical_digest(content) != record_digest:
        raise VisualSemanticRunVerificationError(
            "visual-semantic outer record digest mismatch"
        )
    campaign_digest = _hex(
        record["calibration_campaign_digest"],
        "calibration campaign digest",
    )
    if campaign_digest != campaign_anchor.campaign_digest:
        raise VisualSemanticRunVerificationError(
            "run calibration campaign digest differs from external campaign anchor"
        )
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
            raise VisualSemanticRunVerificationError(
                f"official release descriptor is invalid: {exc}"
            ) from exc
        if (
            release.corpus_manifest_sha256 != corpus_digest
            or release.split_sha256 != split_digest
        ):
            raise VisualSemanticRunVerificationError(
                "official release differs from outer corpus/split identity"
            )

    try:
        if compact_calibration:
            if (
                _hex(record["calibration_digest"], "calibration digest")
                != campaign_anchor.calibration_digest
            ):
                raise VisualSemanticRunVerificationError(
                    "run calibration reference differs from external campaign anchor"
                )
            calibration = campaign_anchor.calibration
        else:
            calibration = SemanticCalibrationArtifact.from_data(
                _mapping(record["calibration"], "semantic calibration")
            )
        policy = build_visual_semantic_policy(
            calibration.family,
            prospective_protocol=calibration.protocol,
        )
        support = verify_support_commitment_data(record["support_commitment"])
    except (TypeError, ValueError) as exc:
        raise VisualSemanticRunVerificationError(
            f"calibration, policy, or support commitment is invalid: {exc}"
        ) from exc
    if not compact_calibration and (
        calibration.digest != campaign_anchor.calibration_digest
        or canonical_json(calibration.to_data())
        != canonical_json(campaign_anchor.calibration.to_data())
    ):
        raise VisualSemanticRunVerificationError(
            "embedded semantic calibration differs from external campaign anchor"
        )
    if calibration.plan.corpus_manifest_digest != corpus_digest:
        raise VisualSemanticRunVerificationError(
            "calibration corpus manifest differs from benchmark corpus"
        )
    if calibration.plan.split_source_digest != split_digest:
        raise VisualSemanticRunVerificationError(
            "calibration split source differs from benchmark split"
        )

    semantic = _mapping(record["visual_semantic"], "visual-semantic artifact")
    _expect_fields(
        semantic, _SEMANTIC_ARTIFACT_FIELDS, "visual-semantic artifact"
    )
    if (
        semantic["schema"] != SEMANTIC_EPISODE_ARCHIVE_SCHEMA
        or semantic["predicate_mode"] != VISUAL_SEMANTIC_PREDICATE_MODE
        or semantic["predicate_policy_digest"] != policy.digest()
        or semantic["python_predicate_authoritative"] is not True
        or semantic["optional_checker_may_affect_result"] is not False
    ):
        raise VisualSemanticRunVerificationError(
            "visual-semantic episode identity or Python authority differs"
        )
    plan, episode, status = _verify_plan_episode(
        record["plan"],
        record["episode"],
        support=support,
        policy_digest=policy.digest(),
        corpus_manifest_digest=corpus_digest,
    )
    if expected_task_manifest_digest is not None:
        trusted_task_manifest_digest = _hex(
            expected_task_manifest_digest,
            "externally expected task manifest digest",
        )
        if plan["task_manifest_digest"] != trusted_task_manifest_digest:
            raise VisualSemanticRunVerificationError(
                "public plan task manifest digest differs from external authority"
            )
    if plan["split"] == "test" and release_data is None:
        raise VisualSemanticRunVerificationError(
            "test visual-semantic run lacks an official release descriptor"
        )
    preimages = _normalise_preimages(blob_bytes_by_id)
    support_blobs = {item.panel.blob_id: item.panel for item in support.support}

    proposal_transport_digest: str | None = None
    proposal_receipt_digest: str | None = None
    precommit_digest: str | None = None
    lowering_digest: str | None = None
    gate_digest: str | None = None
    freeze_digest: str | None = None
    prediction_digest: str | None = None
    support_digests: tuple[str, ...] = ()
    query_digests: tuple[str, ...] = ()
    atom_replays = 0
    archive: VerifiedRunArchive | None = None
    proposal_model: str | None = None

    if status == EpisodeStatus.PROPOSAL_ERROR.value:
        if tuple(episode["phases"]) != _PROPOSAL_ERROR_PHASES:
            raise VisualSemanticRunVerificationError(
                "proposal-error phases are not canonical"
            )
        if (
            semantic["pre_observation_commitment"] is not None
            or semantic["support_observations"] != []
            or semantic["query_observations"] != {}
            or semantic["support_gate"] is not None
            or semantic["proposal_freeze"] is not None
            or record["run_archive"] is not None
        ):
            raise VisualSemanticRunVerificationError(
                "proposal-error record contains post-proposal artifacts"
            )
        _verify_blob_preimages(preimages, support_blobs)
        try:
            rejected = RejectedTypedVisualProposalAttempt.from_data(
                _mapping(
                    semantic["rejected_proposal_attempt"],
                    "rejected typed proposal",
                ),
                catalog=DIRECT_VISUAL_ATOM_CATALOG,
                protocol=calibration.protocol,
            )
        except (TypeError, ValueError) as exc:
            raise VisualSemanticRunVerificationError(
                f"rejected typed proposal is invalid: {exc}"
            ) from exc
        _verify_proposal_pixels(rejected, support=support, preimages=preimages)
        _verify_successful_codex_environment(
            rejected.receipt,
            campaign_anchor=campaign_anchor,
            label="rejected typed proposer receipt",
        )
        proposal_transport_digest = rejected.digest
        proposal_receipt_digest = rejected.receipt.receipt_digest
        proposal_model = rejected.receipt.requested_model
        failure = _mapping(episode["failure"], "proposal-error failure")
        _expect_fields(failure, {"stage", "error_type", "reason"}, "proposal-error failure")
        if (
            failure["stage"] != "proposal"
            or failure["error_type"] != TypedVisualProposalRejected.__name__
            or failure["reason"] != rejected.parse_error_reason
        ):
            raise VisualSemanticRunVerificationError(
                "proposal-error failure differs from parser rejection"
            )
        if episode["artifact_chain"] is not None or episode["score"] != {
            "image_correct": 0,
            "image_total": 2,
            "image_accuracy": 0.0,
            "puzzle_correct": False,
            "puzzle_accuracy": 0.0,
            "determinate": 0,
            "abstentions": 2,
            "errors": 2,
        }:
            raise VisualSemanticRunVerificationError(
                "proposal-error score or chain differs"
            )
    else:
        if semantic["rejected_proposal_attempt"] is not None:
            raise VisualSemanticRunVerificationError(
                "accepted proposal record also contains a rejected attempt"
            )
        if status == EpisodeStatus.COMPLETE.value:
            try:
                archive = verify_archive_data(
                    _mapping(record["run_archive"], "generic run archive")
                )
            except (TypeError, ValueError) as exc:
                raise VisualSemanticRunVerificationError(
                    f"generic run archive is invalid: {exc}"
                ) from exc
            if archive.bundle.support != support:
                raise VisualSemanticRunVerificationError(
                    "generic archive support differs from outer support"
                )
            _verify_opened_plan_commitments(plan, archive)
            query_blobs = {
                item.panel.blob_id: item.panel
                for item in archive.bundle.release.queries
            }
            _verify_blob_preimages(preimages, {**support_blobs, **query_blobs})
        else:
            if record["run_archive"] is not None:
                raise VisualSemanticRunVerificationError(
                    "support-rejected run contains a completed archive"
                )
            _verify_blob_preimages(preimages, support_blobs)

        precommit_data = _mapping(
            semantic["pre_observation_commitment"],
            "pre-observation commitment",
        )
        transport = TypedVisualTransportResult.from_data(
            _mapping(precommit_data.get("proposal_transport"), "proposal transport"),
            catalog=DIRECT_VISUAL_ATOM_CATALOG,
            protocol=calibration.protocol,
        )
        _verify_proposal_pixels(transport, support=support, preimages=preimages)
        _verify_successful_codex_environment(
            transport.receipt,
            campaign_anchor=campaign_anchor,
            label="accepted typed proposer receipt",
        )
        compiled = compile_visual_semantic_proposal(
            transport.proposal,
            policy=policy,
            expected_policy_digest=policy.digest(),
            family=calibration.family,
            issued_by=support.issued_by,
        )
        precommit = SemanticPreObservationCommitment.verify_data(
            precommit_data,
            support=support,
            proposal_transport=transport,
            compiled=compiled,
            expected_digest=precommit_data.get("commitment_digest"),
        )
        proposal_transport_digest = transport.digest
        proposal_receipt_digest = transport.receipt.receipt_digest
        proposal_model = transport.receipt.requested_model
        precommit_digest = precommit.digest
        lowering_digest = compiled.lowering_archive.digest

        raw_support = semantic["support_observations"]
        if not isinstance(raw_support, list) or len(raw_support) != 12:
            raise VisualSemanticRunVerificationError(
                "accepted semantic proposal lacks twelve support observations"
            )
        support_observations: list[VisualSemanticObservationArtifact] = []
        for index, (raw_observation, item) in enumerate(
            zip(raw_support, support.support, strict=True)
        ):
            support_observations.append(
                _decode_observation(
                    raw_observation,
                    label=f"support observation {index}",
                    panel=item.panel,
                    panel_bytes=preimages[item.panel.blob_id],
                    compiled=compiled,
                    calibration=calibration,
                    precommit=precommit,
                    task_id=plan["task_id"],
                    phase="support",
                    ordinal=index,
                    campaign_anchor=campaign_anchor,
                )
            )
        support_digests = tuple(item.digest for item in support_observations)
        atom_replays += sum(len(item.atom_evidence) for item in support_observations)
        gate = _reconstruct_gate(
            semantic["support_gate"],
            support=support,
            precommit=precommit,
            observations=support_observations,
        )
        gate_digest = gate.digest
        freeze = _verify_freeze(
            semantic["proposal_freeze"],
            support=support,
            compiled=compiled,
            precommit=precommit,
            gate=gate,
            archive=archive,
        )
        freeze_digest = freeze.digest()

        if status == EpisodeStatus.SUPPORT_REJECTED.value:
            if tuple(episode["phases"]) != _SUPPORT_REJECTED_PHASES:
                raise VisualSemanticRunVerificationError(
                    "support-rejected phases are not canonical"
                )
            if gate.result is SupportGateResult.ALIGNED:
                raise VisualSemanticRunVerificationError(
                    "support-rejected record contains an aligned gate"
                )
            if semantic["query_observations"] != {}:
                raise VisualSemanticRunVerificationError(
                    "support-rejected record contains query observations"
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
                or failure["reason"] != gate.result.value
            ):
                raise VisualSemanticRunVerificationError(
                    "support-rejected failure differs from replayed gate"
                )
            expected_errors = (
                2 if gate.result is SupportGateResult.OBSERVER_FAILURE else 0
            )
            if episode["artifact_chain"] is not None or episode["score"] != {
                "image_correct": 0,
                "image_total": 2,
                "image_accuracy": 0.0,
                "puzzle_correct": False,
                "puzzle_accuracy": 0.0,
                "determinate": 0,
                "abstentions": 2,
                "errors": expected_errors,
            }:
                raise VisualSemanticRunVerificationError(
                    "support-rejected score or chain differs"
                )
        else:
            assert archive is not None
            if tuple(episode["phases"]) != _COMPLETE_PHASES:
                raise VisualSemanticRunVerificationError(
                    "completed visual-semantic phases are not canonical"
                )
            if gate.result is not SupportGateResult.ALIGNED:
                raise VisualSemanticRunVerificationError(
                    "completed visual-semantic run lacks aligned support"
                )
            if episode["failure"] is not None:
                raise VisualSemanticRunVerificationError(
                    "completed visual-semantic run contains a failure"
                )
            if episode["artifact_chain"] != archive.bundle.chain_data():
                raise VisualSemanticRunVerificationError(
                    "episode artifact chain differs from generic archive"
                )
            if episode["score"] != _expected_score(archive):
                raise VisualSemanticRunVerificationError(
                    "episode score differs from generic cold replay"
                )
            raw_queries = _mapping(
                semantic["query_observations"], "semantic query observations"
            )
            released = {
                item.query_id: item.panel for item in archive.bundle.release.queries
            }
            if set(raw_queries) != set(released):
                raise VisualSemanticRunVerificationError(
                    "query observations do not cover released queries"
                )
            cold = {
                item.query_id: item for item in archive.bundle.cold_inputs.queries
            }
            decoded_queries: list[VisualSemanticObservationArtifact] = []
            for ordinal, query_id in enumerate(sorted(released)):
                panel = released[query_id]
                observation = _decode_observation(
                    raw_queries[query_id],
                    label=f"query observation {query_id}",
                    panel=panel,
                    panel_bytes=preimages[panel.blob_id],
                    compiled=compiled,
                    calibration=calibration,
                    precommit=precommit,
                    task_id=plan["task_id"],
                    phase="query",
                    ordinal=ordinal,
                    campaign_anchor=campaign_anchor,
                )
                cold_query = cold[query_id]
                if (
                    cold_query.panel_digest != observation.panel_digest
                    or tuple(
                        (item.path, item.evidence)
                        for item in cold_query.atom_inputs
                    )
                    != tuple(
                        (item.path, item.evidence)
                        for item in observation.atom_evidence
                    )
                ):
                    raise VisualSemanticRunVerificationError(
                        f"query {query_id} evidence differs from generic cold inputs"
                    )
                decoded_queries.append(observation)
            query_digests = tuple(item.digest for item in decoded_queries)
            atom_replays += sum(len(item.atom_evidence) for item in decoded_queries)
            prediction_digest = archive.bundle.predictions.digest()

    exposure = record["exposure"]
    if exposure is not None:
        exposure_data = _mapping(exposure, "run exposure")
        for key, expected in (
            ("corpus_manifest_digest", corpus_digest),
            ("task_id", plan["task_id"]),
            ("plan_digest", episode["plan_digest"]),
        ):
            if exposure_data.get(key) != expected:
                raise VisualSemanticRunVerificationError(
                    f"run exposure {key} differs from visual-semantic run"
                )
        if proposal_model is not None and exposure_data.get("model") != proposal_model:
            raise VisualSemanticRunVerificationError(
                "run exposure model differs from proposal receipt"
            )

    return VisualSemanticRunVerification(
        run_id=support.run_id,
        status=status,
        record_digest=record_digest,
        plan_digest=episode["plan_digest"],
        calibration_campaign_digest=campaign_digest,
        calibration_digest=calibration.digest,
        expected_codex_launcher_digest=(
            campaign_anchor.expected_codex_launcher_digest
        ),
        cloud_policy_cache_binding=campaign_anchor.cloud_policy_cache_binding,
        policy_digest=policy.digest(),
        proposal_transport_digest=proposal_transport_digest,
        proposal_receipt_digest=proposal_receipt_digest,
        pre_observation_commitment_digest=precommit_digest,
        lowering_archive_digest=lowering_digest,
        support_gate_digest=gate_digest,
        proposal_freeze_digest=freeze_digest,
        prediction_commitment_digest=prediction_digest,
        archive_digest=archive.archive_digest if archive is not None else None,
        verified_blob_ids=tuple(sorted(preimages)),
        support_observation_digests=support_digests,
        query_observation_digests=query_digests,
        registered_atom_replays=atom_replays,
        archive=archive,
    )


def verify_visual_semantic_run_data(
    record_value: Mapping[str, Any],
    *,
    blob_bytes_by_id: Mapping[str, bytes],
    calibration_campaign: (
        SemanticCalibrationCampaignArtifact
        | VisualSemanticCalibrationCampaignAnchor
    ),
    expected_task_manifest_digest: str | None = None,
) -> VisualSemanticRunVerification:
    """Cold-verify internal bindings without Codex or Lean.

    This function treats ``blob_bytes_by_id`` as caller-supplied preimages and
    ``calibration_campaign`` as an already authenticated external authority.
    It proves archive consistency and exact Python replay, not that those bytes
    or the campaign belong to the official corpus.  When a caller has an
    authenticated task manifest, ``expected_task_manifest_digest`` joins that
    authority directly to the public plan.  The public CLI verifier separately
    cold-replays the full campaign, binds the trusted release and every official
    panel identity, then supplies that task-manifest digest here.
    """

    try:
        return _verify_visual_semantic_run_data(
            record_value,
            blob_bytes_by_id=blob_bytes_by_id,
            calibration_campaign=calibration_campaign,
            expected_task_manifest_digest=expected_task_manifest_digest,
        )
    except VisualSemanticRunVerificationError:
        raise
    except (ArtifactTamperError, OSError, TypeError, ValueError) as exc:
        raise VisualSemanticRunVerificationError(
            f"visual-semantic cold verification failed: {exc}"
        ) from exc


def _verify_visual_semantic_run_data_with_verified_anchor(
    record_value: Mapping[str, Any],
    *,
    blob_bytes_by_id: Mapping[str, bytes],
    campaign_anchor: VisualSemanticCalibrationCampaignAnchor,
    expected_task_manifest_digest: str | None = None,
) -> VisualSemanticRunVerification:
    """Replay one run after its enclosing batch verified one full campaign."""

    try:
        return _verify_visual_semantic_run_data(
            record_value,
            blob_bytes_by_id=blob_bytes_by_id,
            calibration_campaign=None,
            expected_task_manifest_digest=expected_task_manifest_digest,
            verified_campaign_anchor=campaign_anchor,
        )
    except VisualSemanticRunVerificationError:
        raise
    except (ArtifactTamperError, OSError, TypeError, ValueError) as exc:
        raise VisualSemanticRunVerificationError(
            f"visual-semantic cold verification failed: {exc}"
        ) from exc


def verify_visual_semantic_run_bytes(
    payload: bytes | str,
    *,
    blob_bytes_by_id: Mapping[str, bytes],
    calibration_campaign: (
        SemanticCalibrationCampaignArtifact
        | VisualSemanticCalibrationCampaignAnchor
    ),
    expected_task_manifest_digest: str | None = None,
) -> VisualSemanticRunVerification:
    """Reject duplicate keys/noncanonical JSON, then cold-verify the record."""

    raw = payload.encode("utf-8") if isinstance(payload, str) else payload

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise VisualSemanticRunVerificationError(
                    f"duplicate JSON object key {key!r}"
                )
            result[key] = value
        return result

    try:
        decoded = json.loads(raw, object_pairs_hook=unique)
    except VisualSemanticRunVerificationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        raise VisualSemanticRunVerificationError(
            f"cannot decode visual-semantic run JSON: {exc}"
        ) from exc
    if not isinstance(decoded, Mapping):
        raise VisualSemanticRunVerificationError(
            "visual-semantic run root must be an object"
        )
    if canonical_json(decoded) != raw:
        raise VisualSemanticRunVerificationError(
            "visual-semantic run bytes are not canonical JSON"
        )
    return verify_visual_semantic_run_data(
        decoded,
        blob_bytes_by_id=blob_bytes_by_id,
        calibration_campaign=calibration_campaign,
        expected_task_manifest_digest=expected_task_manifest_digest,
    )


def build_visual_semantic_run_record(
    *,
    corpus_manifest_digest: str,
    split_source_digest: str | None,
    official_release: OfficialReleaseDescriptor | Mapping[str, Any] | None,
    calibration_campaign: (
        SemanticCalibrationCampaignArtifact
        | VisualSemanticCalibrationCampaignAnchor
    ),
    plan: EpisodePlan,
    result: EpisodeResult,
    episode: VisualSemanticEpisode,
    exposure: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Assemble the exact outer record consumed by the cold verifier."""

    campaign_anchor = _campaign_anchor(calibration_campaign)
    return _build_visual_semantic_run_record_from_verified_anchor(
        corpus_manifest_digest=corpus_manifest_digest,
        split_source_digest=split_source_digest,
        official_release=official_release,
        campaign_anchor=campaign_anchor,
        plan=plan,
        result=result,
        episode=episode,
        exposure=exposure,
        compact_calibration=False,
    )


def _build_visual_semantic_run_record_from_verified_anchor(
    *,
    corpus_manifest_digest: str,
    split_source_digest: str | None,
    official_release: OfficialReleaseDescriptor | Mapping[str, Any] | None,
    campaign_anchor: VisualSemanticCalibrationCampaignAnchor,
    plan: EpisodePlan,
    result: EpisodeResult,
    episode: VisualSemanticEpisode,
    exposure: Mapping[str, Any] | None,
    compact_calibration: bool,
) -> dict[str, Any]:
    """Build after one enclosing trust boundary authenticated the campaign."""

    if not isinstance(campaign_anchor, VisualSemanticCalibrationCampaignAnchor):
        raise TypeError("campaign_anchor must be a verified campaign anchor")
    if type(compact_calibration) is not bool:
        raise TypeError("compact_calibration must be bool")
    calibration = campaign_anchor.calibration

    release_data: Mapping[str, Any] | None
    if isinstance(official_release, OfficialReleaseDescriptor):
        release_data = official_release.to_dict()
    else:
        release_data = official_release
    semantic_artifact = {
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
        "schema": (
            VISUAL_SEMANTIC_COMPACT_OUTER_RUN_SCHEMA
            if compact_calibration
            else VISUAL_SEMANTIC_OUTER_RUN_SCHEMA
        ),
        "corpus_manifest_digest": corpus_manifest_digest,
        "split_source_digest": split_source_digest,
        "official_release": dict(release_data) if release_data is not None else None,
        "calibration_campaign_digest": campaign_anchor.campaign_digest,
        "plan": plan.to_data(),
        "support_commitment": plan.support.to_data(),
        "episode": result.to_data(),
        "visual_semantic": semantic_artifact,
        "run_archive": result.bundle.to_archive_data() if result.bundle else None,
        "exposure": dict(exposure) if exposure is not None else None,
    }
    if compact_calibration:
        content["calibration_digest"] = campaign_anchor.calibration_digest
    else:
        content["calibration"] = calibration.to_data()
    return {**content, "record_digest": canonical_digest(content)}


__all__ = (
    "VISUAL_SEMANTIC_COMPACT_OUTER_RUN_SCHEMA",
    "VISUAL_SEMANTIC_OUTER_RUN_SCHEMA",
    "VisualSemanticCalibrationCampaignAnchor",
    "VisualSemanticRunVerification",
    "VisualSemanticRunVerificationError",
    "build_visual_semantic_run_record",
    "verify_visual_semantic_run_bytes",
    "verify_visual_semantic_run_data",
)
