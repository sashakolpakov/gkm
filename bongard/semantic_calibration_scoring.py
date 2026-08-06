"""Pre-family scoring boundary for visual-semantic calibration.

Calibration has to observe the ordinal soft scorer *before* a calibrated
``SoftScorerFamily`` exists.  The ordinary benchmark commitment therefore
cannot be reused: it deliberately binds that already-fitted family.  This
module supplies the missing earlier object in the causal chain::

    label-free plan + support-only proposal + panel witnesses
        -> calibration score commitment
        -> blind score artifact
        -> label join
        -> fitted family
        -> benchmark policy/commitment

The commitment contains no fitted family, calibration label, or scorer
output.  The later score artifact points back to its digest through the
existing ``pre_observation_commitment_digest`` field.  Python is the complete
reference implementation; no proof assistant or backend identity participates
in either content address.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import re
from typing import Any, Callable, Mapping

from bongard.artifacts import (
    SupportCommitment,
    canonical_digest,
    canonical_json,
    verify_support_commitment_data,
)
from bongard.blind_soft_transport import (
    BlindSoftScoreTransportArtifact,
    BlindSoftVerifierContext,
    VerifierWitnessSummary,
    canonical_witness_summaries,
    score_blind_soft_panel,
)
from bongard.canonical_cache import cached_content_data, cached_content_digest
from bongard.semantic_calibration import (
    CalibrationPanelSelection,
    SemanticCalibrationPlan,
)
from bongard.soft_predicates import SoftScorerProtocol
from bongard.transport import (
    CloudPolicyCacheSnapshot,
    run_codex_named_images_structured,
)
from bongard.typed_visual_transport import TypedVisualTransportResult
from bongard.visual_predicate_catalog import DIRECT_VISUAL_ATOM_CATALOG
from bongard.visual_witness_summaries import visual_witness_summaries
from bongard.visual_witness_bundle import (
    VisualWitnessBundle,
    extract_visual_witness_bundle,
    verify_visual_witness_bundle,
)


CALIBRATION_SCORE_COMMITMENT_SCHEMA = (
    "gkm.bongard-semantic-calibration-score-commitment.v2"
)
CALIBRATION_SCORE_ATTEMPT_SCHEMA = (
    "gkm.bongard-semantic-calibration-score-attempt.v1"
)
CALIBRATION_SCORE_REFERENCE_SEMANTICS = (
    "python-joint-visual-witness-plus-blind-ordinal-score/v2"
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")

PanelInput = str | Path | bytes
StructuredTransport = Callable[..., Any]


class SemanticCalibrationScoringError(ValueError):
    """A pre-family score commitment or its descendant is inconsistent."""


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise SemanticCalibrationScoringError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise SemanticCalibrationScoringError(f"{label} must be an object")
    return value


def _fields(
    value: Mapping[str, Any], expected: set[str], label: str
) -> Mapping[str, Any]:
    if set(value) != expected:
        raise SemanticCalibrationScoringError(
            f"{label} fields differ: missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )
    return value


def _panel_bytes(panel: PanelInput) -> bytes:
    if isinstance(panel, bytes):
        payload = panel
    elif isinstance(panel, (str, Path)):
        payload = Path(panel).read_bytes()
    else:
        raise TypeError("panel must be exact PNG bytes or a filesystem path")
    if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        raise SemanticCalibrationScoringError("calibration panel is not a PNG")
    return payload


def _validate_support_presentation(
    support: SupportCommitment,
    transport: TypedVisualTransportResult,
) -> None:
    """Bind the proposal's canonical pos/neg presentation to support bytes."""

    by_id = {item.panel.blob_id: item for item in support.support}
    if len(by_id) != 12 or len(transport.support_presentation) != 12:
        raise SemanticCalibrationScoringError(
            "calibration proposer requires exactly twelve support panels"
        )
    seen: set[str] = set()
    for presented in transport.support_presentation:
        stem = presented.name.removesuffix(".png")
        try:
            side, raw_index = stem.split("_", 1)
            index = int(raw_index)
        except (ValueError, TypeError) as exc:
            raise SemanticCalibrationScoringError(
                "typed proposer support presentation name is malformed"
            ) from exc
        if side not in {"pos", "neg"} or not 0 <= index < 6:
            raise SemanticCalibrationScoringError(
                "typed proposer support presentation is outside canonical 6+6 slots"
            )
        blob_id = f"support-{'positive' if side == 'pos' else 'negative'}-{index}"
        if blob_id in seen:
            raise SemanticCalibrationScoringError(
                "typed proposer repeats a support presentation slot"
            )
        seen.add(blob_id)
        try:
            committed = by_id[blob_id]
        except KeyError as exc:
            raise SemanticCalibrationScoringError(
                f"support commitment lacks proposer slot {blob_id!r}"
            ) from exc
        if committed.positive is not (side == "pos"):
            raise SemanticCalibrationScoringError(
                f"support polarity differs at proposer slot {blob_id!r}"
            )
        if (
            committed.panel.sha256 != presented.content_digest
            or committed.panel.byte_count != presented.byte_count
            or committed.panel.media_type != "image/png"
        ):
            raise SemanticCalibrationScoringError(
                f"typed proposer bytes differ from support at {blob_id!r}"
            )


@dataclass(frozen=True)
class SemanticCalibrationScoreCommitment:
    """Everything fixed after proposal/extraction and before a scorer call."""

    plan: SemanticCalibrationPlan
    selection: CalibrationPanelSelection
    support: SupportCommitment
    proposal_transport: TypedVisualTransportResult
    protocol: SoftScorerProtocol
    panel_byte_count: int
    witness_bundle: VisualWitnessBundle
    witness_summaries: tuple[VerifierWitnessSummary, ...]
    _sealed_digest: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._validate()
        object.__setattr__(self, "_sealed_digest", self.digest)

    def _validate(self) -> None:
        if not isinstance(self.plan, SemanticCalibrationPlan):
            raise TypeError("plan must be SemanticCalibrationPlan")
        self.plan.assert_untampered()
        if not isinstance(self.selection, CalibrationPanelSelection):
            raise TypeError("selection must be CalibrationPanelSelection")
        if self.plan.selection(self.selection.observation_id) != self.selection:
            raise SemanticCalibrationScoringError(
                "score selection differs from the frozen calibration plan"
            )
        if not isinstance(self.support, SupportCommitment):
            raise TypeError("support must be SupportCommitment")
        if verify_support_commitment_data(self.support.to_data()) != self.support:
            raise SemanticCalibrationScoringError(
                "support commitment is not canonically represented"
            )
        if not isinstance(self.proposal_transport, TypedVisualTransportResult):
            raise TypeError("proposal_transport must be TypedVisualTransportResult")
        if not isinstance(self.protocol, SoftScorerProtocol):
            raise TypeError("protocol must be SoftScorerProtocol")
        self.protocol.assert_untampered()
        protocol_digest = self.protocol.digest()
        if (
            self.plan.protocol_digest != protocol_digest
            or self.proposal_transport.scorer_protocol_digest != protocol_digest
        ):
            raise SemanticCalibrationScoringError(
                "plan, proposer, and scorer protocol identities differ"
            )
        if self.proposal_transport.catalog_digest != DIRECT_VISUAL_ATOM_CATALOG.digest:
            raise SemanticCalibrationScoringError(
                "calibration proposal uses a different direct atom catalog"
            )
        claim = self.proposal_transport.proposal.soft_claim
        if claim is None:
            raise SemanticCalibrationScoringError(
                "calibration scoring requires an emitted soft claim"
            )
        if claim.scorer_protocol_digest != protocol_digest:
            raise SemanticCalibrationScoringError(
                "calibration soft claim belongs to another scorer protocol"
            )
        replayed_transport = TypedVisualTransportResult.from_data(
            self.proposal_transport.to_data(),
            catalog=DIRECT_VISUAL_ATOM_CATALOG,
            protocol=self.protocol,
            expected_digest=self.proposal_transport.digest,
        )
        if replayed_transport != self.proposal_transport:
            raise SemanticCalibrationScoringError(
                "proposal transport is not canonically represented"
            )
        if (
            isinstance(self.panel_byte_count, bool)
            or not isinstance(self.panel_byte_count, int)
            or self.panel_byte_count <= 0
        ):
            raise SemanticCalibrationScoringError(
                "panel_byte_count must be positive"
            )
        verify_visual_witness_bundle(self.witness_bundle)
        if self.witness_bundle.panel_digest != self.selection.panel_digest:
            raise SemanticCalibrationScoringError(
                "witness bundle belongs to another calibration panel"
            )
        canonical = canonical_witness_summaries(self.witness_summaries)
        if canonical != self.witness_summaries:
            raise SemanticCalibrationScoringError(
                "calibration witness summaries are not canonical"
            )
        expected_summaries = canonical_witness_summaries(
            visual_witness_summaries(self.witness_bundle)
        )
        if expected_summaries != self.witness_summaries:
            raise SemanticCalibrationScoringError(
                "witness summaries do not reproduce from the frozen bundle"
            )
        _validate_support_presentation(self.support, self.proposal_transport)

    @classmethod
    def from_panel(
        cls,
        *,
        plan: SemanticCalibrationPlan,
        selection: CalibrationPanelSelection,
        support: SupportCommitment,
        proposal_transport: TypedVisualTransportResult,
        protocol: SoftScorerProtocol,
        panel: PanelInput,
    ) -> "SemanticCalibrationScoreCommitment":
        """Extract witnesses from the exact selected bytes and seal the score input."""

        payload = _panel_bytes(panel)
        panel_digest = hashlib.sha256(payload).hexdigest()
        if panel_digest != selection.panel_digest:
            raise SemanticCalibrationScoringError(
                "selected calibration panel bytes differ from the plan"
            )
        bundle = extract_visual_witness_bundle(payload)
        verify_visual_witness_bundle(bundle, expected_png_bytes=payload)
        summaries = canonical_witness_summaries(
            visual_witness_summaries(bundle, expected_png_bytes=payload)
        )
        return cls(
            plan=plan,
            selection=selection,
            support=support,
            proposal_transport=proposal_transport,
            protocol=protocol,
            panel_byte_count=len(payload),
            witness_bundle=bundle,
            witness_summaries=summaries,
        )

    def identity_data(self) -> dict[str, str]:
        claim = self.proposal_transport.proposal.soft_claim
        assert claim is not None
        return {
            "plan_digest": self.plan.digest,
            "selection_digest": self.selection.digest,
            "support_commitment_digest": self.support.digest(),
            "proposal_transport_digest": self.proposal_transport.digest,
            "proposer_receipt_digest": self.proposal_transport.receipt.receipt_digest,
            "soft_claim_digest": canonical_digest(claim.to_data()),
            "protocol_digest": self.protocol.digest(),
            "panel_digest": self.selection.panel_digest,
            "witness_bundle_digest": self.witness_bundle.digest(),
            "witness_summaries_digest": canonical_digest(
                [item.to_data() for item in self.witness_summaries]
            ),
        }

    def _canonical_anchor(self) -> tuple[object, ...]:
        return (
            tuple(sorted(self.identity_data().items())),
            self.panel_byte_count,
        )

    def _uncached_content_data(self) -> dict[str, object]:
        return {
            "schema": CALIBRATION_SCORE_COMMITMENT_SCHEMA,
            "reference_execution_semantics": (
                CALIBRATION_SCORE_REFERENCE_SEMANTICS
            ),
            "python_predicate_authoritative": True,
            "optional_checker_may_affect_result": False,
            "fitted_family_present": False,
            "calibration_label_state": "withheld",
            "identities": self.identity_data(),
            "plan": self.plan.to_data(),
            "selection": self.selection.to_data(),
            "support_commitment": self.support.to_data(),
            "proposal_transport": self.proposal_transport.to_data(),
            "prospective_protocol": self.protocol.to_data(),
            "panel": {
                "neutral_name": "query.png",
                "media_type": "image/png",
                "byte_count": self.panel_byte_count,
                "content_digest": self.selection.panel_digest,
            },
            "witness_bundle": self.witness_bundle.to_data(),
            "witness_summaries": [
                item.to_data() for item in self.witness_summaries
            ],
        }

    def content_data(self) -> dict[str, object]:
        return cached_content_data(
            self,
            self._canonical_anchor(),
            self._uncached_content_data,
        )

    @property
    def digest(self) -> str:
        return cached_content_digest(
            self,
            self._canonical_anchor(),
            self._uncached_content_data,
        )

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "commitment_digest": self.digest}

    def assert_untampered(self) -> None:
        # Re-run every semantic join before checking the sealed content address.
        self._validate()
        if self.digest != self._sealed_digest:
            raise SemanticCalibrationScoringError(
                "calibration score commitment changed after sealing"
            )

    @classmethod
    def from_data(
        cls,
        value: Mapping[str, Any],
        *,
        expected_digest: str | None = None,
        panel: PanelInput | None = None,
    ) -> "SemanticCalibrationScoreCommitment":
        data = _fields(
            _mapping(value, "calibration score commitment"),
            {
                "schema",
                "reference_execution_semantics",
                "python_predicate_authoritative",
                "optional_checker_may_affect_result",
                "fitted_family_present",
                "calibration_label_state",
                "identities",
                "plan",
                "selection",
                "support_commitment",
                "proposal_transport",
                "prospective_protocol",
                "panel",
                "witness_bundle",
                "witness_summaries",
                "commitment_digest",
            },
            "calibration score commitment",
        )
        if data["schema"] != CALIBRATION_SCORE_COMMITMENT_SCHEMA:
            raise SemanticCalibrationScoringError(
                "unsupported calibration score commitment schema"
            )
        if data["reference_execution_semantics"] != (
            CALIBRATION_SCORE_REFERENCE_SEMANTICS
        ):
            raise SemanticCalibrationScoringError(
                "calibration score reference semantics changed"
            )
        if (
            data["python_predicate_authoritative"] is not True
            or data["optional_checker_may_affect_result"] is not False
            or data["fitted_family_present"] is not False
            or data["calibration_label_state"] != "withheld"
        ):
            raise SemanticCalibrationScoringError(
                "calibration score authority or causal state changed"
            )
        plan = SemanticCalibrationPlan.from_data(
            _mapping(data["plan"], "calibration score plan")
        )
        selection = CalibrationPanelSelection.from_data(
            _mapping(data["selection"], "calibration score selection")
        )
        support = verify_support_commitment_data(data["support_commitment"])
        protocol = SoftScorerProtocol.from_data(
            _mapping(data["prospective_protocol"], "prospective protocol"),
            expected_digest=plan.protocol_digest,
        )
        proposal = TypedVisualTransportResult.from_data(
            _mapping(data["proposal_transport"], "proposal transport"),
            catalog=DIRECT_VISUAL_ATOM_CATALOG,
            protocol=protocol,
        )
        bundle = VisualWitnessBundle.from_data(
            _mapping(data["witness_bundle"], "witness bundle")
        )
        raw_summaries = data["witness_summaries"]
        if not isinstance(raw_summaries, list):
            raise SemanticCalibrationScoringError(
                "witness_summaries must be a list"
            )
        summaries = tuple(
            VerifierWitnessSummary.from_data(
                _mapping(item, "witness summary")
            )
            for item in raw_summaries
        )
        panel_identity = _fields(
            _mapping(data["panel"], "calibration panel identity"),
            {"neutral_name", "media_type", "byte_count", "content_digest"},
            "calibration panel identity",
        )
        if (
            panel_identity["neutral_name"] != "query.png"
            or panel_identity["media_type"] != "image/png"
            or panel_identity["content_digest"] != selection.panel_digest
        ):
            raise SemanticCalibrationScoringError(
                "calibration panel presentation identity changed"
            )
        result = cls(
            plan=plan,
            selection=selection,
            support=support,
            proposal_transport=proposal,
            protocol=protocol,
            panel_byte_count=panel_identity["byte_count"],
            witness_bundle=bundle,
            witness_summaries=summaries,
        )
        if data["identities"] != result.identity_data():
            raise SemanticCalibrationScoringError(
                "calibration score redundant identities differ"
            )
        archived = _digest(data["commitment_digest"], "commitment_digest")
        if archived != result.digest:
            raise SemanticCalibrationScoringError(
                "calibration score commitment digest differs"
            )
        if expected_digest is not None and result.digest != _digest(
            expected_digest, "expected commitment digest"
        ):
            raise SemanticCalibrationScoringError(
                "calibration score commitment differs from expected digest"
            )
        if canonical_json(result.to_data()) != canonical_json(dict(data)):
            raise SemanticCalibrationScoringError(
                "calibration score commitment is not canonical"
            )
        if panel is not None:
            payload = _panel_bytes(panel)
            if (
                len(payload) != result.panel_byte_count
                or hashlib.sha256(payload).hexdigest()
                != result.selection.panel_digest
            ):
                raise SemanticCalibrationScoringError(
                    "exact calibration panel bytes differ from commitment"
                )
            verify_visual_witness_bundle(
                result.witness_bundle, expected_png_bytes=payload
            )
        return result


@dataclass(frozen=True)
class SemanticCalibrationScoreAttempt:
    """A committed pre-family input and its one descendant scorer artifact."""

    commitment: SemanticCalibrationScoreCommitment
    score_artifact: BlindSoftScoreTransportArtifact
    _sealed_digest: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.commitment, SemanticCalibrationScoreCommitment):
            raise TypeError("commitment must be SemanticCalibrationScoreCommitment")
        self.commitment.assert_untampered()
        if not isinstance(self.score_artifact, BlindSoftScoreTransportArtifact):
            raise TypeError("score_artifact must be BlindSoftScoreTransportArtifact")
        self.score_artifact.assert_untampered()
        record = self.score_artifact.record
        expected = {
            "task_id": self.commitment.selection.task_id,
            "panel_id": self.commitment.selection.panel_id,
            "panel_digest": self.commitment.selection.panel_digest,
            "pre_observation_commitment_digest": self.commitment.digest,
            "scorer_protocol_digest": self.commitment.protocol.digest(),
            "proposer_receipt_digest": (
                self.commitment.proposal_transport.receipt.receipt_digest
            ),
            "witness_packet_digest": self.commitment.witness_bundle.digest(),
        }
        for name, wanted in expected.items():
            if getattr(record, name) != wanted:
                raise SemanticCalibrationScoringError(
                    f"calibration score record {name} differs from commitment"
                )
        claim = self.commitment.proposal_transport.proposal.soft_claim
        assert claim is not None
        if self.score_artifact.claim != claim:
            raise SemanticCalibrationScoringError(
                "calibration score artifact uses another soft claim"
            )
        if self.score_artifact.witness_summaries != (
            self.commitment.witness_summaries
        ):
            raise SemanticCalibrationScoringError(
                "calibration scorer summaries differ from commitment"
            )
        object.__setattr__(self, "_sealed_digest", self.digest)

    def _canonical_anchor(self) -> tuple[str, str]:
        return (self.commitment.digest, self.score_artifact.digest)

    def _uncached_content_data(self) -> dict[str, object]:
        return {
            "schema": CALIBRATION_SCORE_ATTEMPT_SCHEMA,
            "causal_order": "score_commitment_then_blind_scorer/v1",
            "commitment": self.commitment.to_data(),
            "commitment_digest": self.commitment.digest,
            "score_artifact": self.score_artifact.to_data(),
            "score_artifact_digest": self.score_artifact.digest,
        }

    def content_data(self) -> dict[str, object]:
        return cached_content_data(
            self,
            self._canonical_anchor(),
            self._uncached_content_data,
        )

    @property
    def digest(self) -> str:
        return cached_content_digest(
            self,
            self._canonical_anchor(),
            self._uncached_content_data,
        )

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "attempt_digest": self.digest}

    def assert_untampered(self) -> None:
        self.commitment.assert_untampered()
        self.score_artifact.assert_untampered()
        if self.digest != self._sealed_digest:
            raise SemanticCalibrationScoringError(
                "calibration score attempt changed after sealing"
            )

    @classmethod
    def from_data(
        cls,
        value: Mapping[str, Any],
        *,
        expected_digest: str | None = None,
        panel: PanelInput | None = None,
    ) -> "SemanticCalibrationScoreAttempt":
        data = _fields(
            _mapping(value, "calibration score attempt"),
            {
                "schema",
                "causal_order",
                "commitment",
                "commitment_digest",
                "score_artifact",
                "score_artifact_digest",
                "attempt_digest",
            },
            "calibration score attempt",
        )
        if data["schema"] != CALIBRATION_SCORE_ATTEMPT_SCHEMA or data[
            "causal_order"
        ] != "score_commitment_then_blind_scorer/v1":
            raise SemanticCalibrationScoringError(
                "unsupported calibration score attempt"
            )
        commitment = SemanticCalibrationScoreCommitment.from_data(
            _mapping(data["commitment"], "score attempt commitment"),
            expected_digest=_digest(
                data["commitment_digest"], "commitment_digest"
            ),
            panel=panel,
        )
        score = BlindSoftScoreTransportArtifact.from_data(
            _mapping(data["score_artifact"], "score attempt artifact"),
            expected_digest=_digest(
                data["score_artifact_digest"], "score_artifact_digest"
            ),
            expected_protocol_digest=commitment.protocol.digest(),
        )
        result = cls(commitment, score)
        archived = _digest(data["attempt_digest"], "attempt_digest")
        if result.digest != archived:
            raise SemanticCalibrationScoringError(
                "calibration score attempt digest differs"
            )
        if expected_digest is not None and result.digest != _digest(
            expected_digest, "expected attempt digest"
        ):
            raise SemanticCalibrationScoringError(
                "calibration score attempt differs from expected digest"
            )
        if canonical_json(result.to_data()) != canonical_json(dict(data)):
            raise SemanticCalibrationScoringError(
                "calibration score attempt is not canonical"
            )
        return result


def score_semantic_calibration_panel(
    panel: PanelInput,
    commitment: SemanticCalibrationScoreCommitment,
    *,
    minutes: int = 10,
    verbose: bool = False,
    executable: str = "codex",
    cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot | None = None,
    transport: StructuredTransport = run_codex_named_images_structured,
) -> SemanticCalibrationScoreAttempt:
    """Run exactly one blind score against an already-frozen calibration input."""

    if not isinstance(commitment, SemanticCalibrationScoreCommitment):
        raise TypeError("commitment must be SemanticCalibrationScoreCommitment")
    commitment.assert_untampered()
    payload = _panel_bytes(panel)
    if (
        len(payload) != commitment.panel_byte_count
        or hashlib.sha256(payload).hexdigest()
        != commitment.selection.panel_digest
    ):
        raise SemanticCalibrationScoringError(
            "calibration score panel bytes differ from commitment"
        )
    verify_visual_witness_bundle(
        commitment.witness_bundle, expected_png_bytes=payload
    )
    scorer_call_id = "cal-score-" + canonical_digest(
        {
            "schema": "gkm.bongard-semantic-calibration-score-call-id.v1",
            "commitment_digest": commitment.digest,
            "observation_id": commitment.selection.observation_id,
        }
    )[:40]
    context = BlindSoftVerifierContext(
        task_id=commitment.selection.task_id,
        panel_id=commitment.selection.panel_id,
        proposer_call_id=commitment.proposal_transport.receipt.thread_id,
        proposer_receipt_digest=(
            commitment.proposal_transport.receipt.receipt_digest
        ),
        scorer_call_id=scorer_call_id,
        pre_observation_commitment_digest=commitment.digest,
    )
    claim = commitment.proposal_transport.proposal.soft_claim
    assert claim is not None
    score = score_blind_soft_panel(
        panel,
        claim,
        protocol=commitment.protocol,
        witness_packet_digest=commitment.witness_bundle.digest(),
        witness_summaries=commitment.witness_summaries,
        context=context,
        minutes=minutes,
        verbose=verbose,
        executable=executable,
        cloud_policy_cache_snapshot=cloud_policy_cache_snapshot,
        transport=transport,
    )
    return SemanticCalibrationScoreAttempt(commitment, score)


__all__ = [
    "CALIBRATION_SCORE_ATTEMPT_SCHEMA",
    "CALIBRATION_SCORE_COMMITMENT_SCHEMA",
    "CALIBRATION_SCORE_REFERENCE_SEMANTICS",
    "SemanticCalibrationScoreAttempt",
    "SemanticCalibrationScoreCommitment",
    "SemanticCalibrationScoringError",
    "score_semantic_calibration_panel",
]
