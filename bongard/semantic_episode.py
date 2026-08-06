"""Sealed benchmark adapter for the typed visual-semantic pipeline.

One object acts as the support-only proposer, the twelve-call support replay
factory, and the two-call query observer factory expected by
``bongard.benchmark.run_episode``.  The proposer turn is compiled and wrapped
in a :class:`SemanticPreObservationCommitment` before any empirical replay.
Every later panel receives a fresh stateless observation session.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable, Mapping, Sequence

from bongard.artifacts import AtomPath, SupportCommitment, canonical_digest
from bongard.benchmark import (
    ObservationInput,
    ProposedRule,
    SupportGateMeasurement,
    SupportInput,
    VISUAL_SEMANTIC_PREDICATE_MODE,
)
from bongard.blind_soft_transport import BlindSoftVerifierContext
from bongard.evidence import Evidence
from bongard.semantic_commitment import SemanticPreObservationCommitment
from bongard.semantic_observation import (
    VisualSemanticObservationArtifact,
    observe_visual_semantic_panel,
)
from bongard.semantic_policy import VisualSemanticPolicy
from bongard.semantic_synthesis import (
    CompiledVisualSemanticProposal,
    compile_visual_semantic_proposal,
)
from bongard.soft_predicates import SoftScorerFamily, SoftScorerProtocol
from bongard.transport import (
    CodexReceipt,
    CloudPolicyCacheSnapshot,
    run_codex_named_images_structured,
    run_codex_structured,
)
from bongard.typed_visual_transport import (
    RejectedTypedVisualProposalAttempt,
    TypedVisualProposalRejected,
    TypedVisualTransportResult,
    propose_typed_visual,
)
from bongard.visual_predicate_catalog import DIRECT_VISUAL_ATOM_CATALOG


SEMANTIC_EPISODE_ARCHIVE_SCHEMA = "gkm.bongard-visual-semantic-episode-adapter.v1"
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CACHE_BINDING = re.compile(r"sha256:[0-9a-f]{64}\Z")

StructuredTransport = Callable[..., Any]


class SemanticEpisodeError(ValueError):
    """The sealed semantic adapter was used out of order or against drift."""


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise SemanticEpisodeError(f"invalid {label} {value!r}")
    return value


def _validate_execution_receipt(
    receipt: CodexReceipt,
    *,
    expected_launcher_digest: str | None,
    expected_cache_binding: str | None,
    role: str,
) -> None:
    """Reject a successful call from outside the calibrated environment."""

    if expected_launcher_digest is None:
        return
    if not isinstance(receipt, CodexReceipt):
        raise SemanticEpisodeError(f"{role} lacks a successful Codex receipt")
    if receipt.codex_launcher_digest != expected_launcher_digest:
        raise SemanticEpisodeError(
            f"{role} Codex launcher differs from the Stage-A environment"
        )
    if receipt.cloud_config_bundle_cache_binding != expected_cache_binding:
        raise SemanticEpisodeError(
            f"{role} cloud-policy cache differs from the Stage-A environment"
        )


def _validate_observation_input(
    query: ObservationInput,
    *,
    compiled: CompiledVisualSemanticProposal,
    precommit: SemanticPreObservationCommitment,
) -> None:
    if not isinstance(query, ObservationInput):
        raise TypeError("semantic observer requires ObservationInput")
    precommit.assert_untampered()
    if query.freeze.proposer_digest != precommit.digest:
        raise SemanticEpisodeError(
            "observation freeze belongs to another pre-observation commitment"
        )
    if query.freeze.formula != compiled.formula:
        raise SemanticEpisodeError("observation formula differs from compiled proposal")
    if query.freeze.registry_digest != compiled.registry.digest():
        raise SemanticEpisodeError("observation registry digest differs")
    if query.registry.digest() != compiled.registry.digest():
        raise SemanticEpisodeError("isolated observation registry differs")
    if query.freeze.attachment_contract_digest != (
        compiled.attachment_contract.digest()
    ):
        raise SemanticEpisodeError("observation attachment digest differs")
    compiled.attachment_contract.validate(query.freeze.formula, query.registry)


@dataclass
class _SemanticObservationSession:
    """One single-use, role-neutral panel callback."""

    phase: str
    ordinal: int
    task_id: str
    compiled: CompiledVisualSemanticProposal
    protocol: SoftScorerProtocol
    precommit: SemanticPreObservationCommitment
    scorer_minutes: int
    verbose: bool
    executable: str
    cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot | None
    expected_codex_launcher_digest: str | None
    expected_cloud_policy_cache_binding: str | None
    scorer_transport: StructuredTransport
    artifact: VisualSemanticObservationArtifact | None = None

    def __post_init__(self) -> None:
        if self.phase not in {"support", "query"}:
            raise SemanticEpisodeError("unknown semantic observation phase")
        if isinstance(self.ordinal, bool) or not isinstance(self.ordinal, int) or (
            self.ordinal < 0
        ):
            raise SemanticEpisodeError("observation ordinal must be non-negative")

    def _context(self) -> BlindSoftVerifierContext:
        transport = self.precommit.proposal_transport
        scorer_call_id = "score-" + canonical_digest(
            {
                "schema": "gkm.bongard-semantic-score-call-id.v1",
                "pre_observation_commitment_digest": self.precommit.digest,
                "phase": self.phase,
                "ordinal": self.ordinal,
            }
        )[:40]
        return BlindSoftVerifierContext(
            task_id=self.task_id,
            panel_id=f"{self.phase}-panel-{self.ordinal:02d}",
            proposer_call_id=transport.receipt.thread_id,
            proposer_receipt_digest=transport.receipt.receipt_digest,
            scorer_call_id=scorer_call_id,
            pre_observation_commitment_digest=self.precommit.digest,
        )

    def _run(self, query: ObservationInput) -> VisualSemanticObservationArtifact:
        if self.artifact is not None:
            raise SemanticEpisodeError("semantic observation session was reused")
        _validate_observation_input(
            query, compiled=self.compiled, precommit=self.precommit
        )
        artifact = observe_visual_semantic_panel(
            query.panel_path,
            self.compiled,
            protocol=self.protocol,
            context=self._context(),
            pre_observation_commitment_digest=self.precommit.digest,
            minutes=self.scorer_minutes,
            verbose=self.verbose,
            executable=self.executable,
            cloud_policy_cache_snapshot=self.cloud_policy_cache_snapshot,
            transport=self.scorer_transport,
        )
        if artifact.scorer_artifact is not None and isinstance(
            artifact.scorer_artifact.receipt, CodexReceipt
        ):
            _validate_execution_receipt(
                artifact.scorer_artifact.receipt,
                expected_launcher_digest=self.expected_codex_launcher_digest,
                expected_cache_binding=self.expected_cloud_policy_cache_binding,
                role=f"{self.phase} scorer",
            )
        if (
            artifact.panel_digest != query.panel.sha256
            or artifact.panel_byte_count != query.panel.byte_count
        ):
            raise SemanticEpisodeError(
                "semantic observation consumed bytes outside the neutral panel commitment"
            )
        self.artifact = artifact
        return artifact

    def observe_support(self, panel: ObservationInput) -> SupportGateMeasurement:
        if self.phase != "support":
            raise SemanticEpisodeError("query session cannot observe support")
        return self._run(panel).to_support_gate_measurement()

    def observe(
        self, query: ObservationInput
    ) -> Mapping[AtomPath, Evidence[bool]]:
        if self.phase != "query":
            raise SemanticEpisodeError("support session cannot observe a query")
        return self._run(query).evidence_by_path()


class VisualSemanticEpisode:
    """Headless typed proposer and fresh Python semantic observations."""

    requires_empirical_support_gate = True
    predicate_mode = VISUAL_SEMANTIC_PREDICATE_MODE

    def __init__(
        self,
        *,
        task_id: str,
        support_commitment: SupportCommitment,
        policy: VisualSemanticPolicy,
        family: SoftScorerFamily,
        protocol: SoftScorerProtocol,
        proposer_minutes: int = 15,
        scorer_minutes: int = 10,
        verbose: bool = False,
        executable: str = "codex",
        cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot | None = None,
        expected_codex_launcher_digest: str | None = None,
        expected_cloud_policy_cache_binding: str | None = None,
        proposer_transport: StructuredTransport = run_codex_structured,
        scorer_transport: StructuredTransport = run_codex_named_images_structured,
    ) -> None:
        self.task_id = _identifier(task_id, "task_id")
        if not isinstance(support_commitment, SupportCommitment):
            raise TypeError("support_commitment must be SupportCommitment")
        if not isinstance(policy, VisualSemanticPolicy):
            raise TypeError("policy must be VisualSemanticPolicy")
        if not isinstance(family, SoftScorerFamily):
            raise TypeError("family must be SoftScorerFamily")
        if not isinstance(protocol, SoftScorerProtocol):
            raise TypeError("protocol must be SoftScorerProtocol")
        protocol.assert_untampered()
        family.assert_untampered()
        family.verify_calibration()
        if protocol.digest() != family.protocol_digest:
            raise SemanticEpisodeError("protocol differs from fitted scorer family")
        if policy.soft_scorer_protocol_digest != protocol.digest():
            raise SemanticEpisodeError("policy differs from prospective protocol")
        if policy.soft_scorer_family_digest != family.digest():
            raise SemanticEpisodeError("policy differs from fitted scorer family")
        for name, value in (
            ("proposer_minutes", proposer_minutes),
            ("scorer_minutes", scorer_minutes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 120:
                raise SemanticEpisodeError(f"{name} must lie in [1, 120]")
        if not callable(proposer_transport) or not callable(scorer_transport):
            raise TypeError("semantic episode transports must be callable")
        if (expected_codex_launcher_digest is None) != (
            expected_cloud_policy_cache_binding is None
        ):
            raise SemanticEpisodeError(
                "calibrated launcher and cloud-policy identities must be supplied together"
            )
        if expected_codex_launcher_digest is not None:
            if _SHA256.fullmatch(expected_codex_launcher_digest) is None:
                raise SemanticEpisodeError(
                    "expected Codex launcher digest must be a lowercase SHA-256"
                )
            if expected_cloud_policy_cache_binding != "absent" and (
                not isinstance(expected_cloud_policy_cache_binding, str)
                or _CACHE_BINDING.fullmatch(
                    expected_cloud_policy_cache_binding
                )
                is None
            ):
                raise SemanticEpisodeError(
                    "expected cloud-policy cache binding is invalid"
                )
            if not isinstance(
                cloud_policy_cache_snapshot, CloudPolicyCacheSnapshot
            ):
                raise SemanticEpisodeError(
                    "calibrated execution requires one frozen cloud-policy cache snapshot"
                )
            if cloud_policy_cache_snapshot.binding != (
                expected_cloud_policy_cache_binding
            ):
                raise SemanticEpisodeError(
                    "current cloud-policy cache differs from the Stage-A environment"
                )

        self.support_commitment = support_commitment
        self.policy = policy
        self.family = family
        self.protocol = protocol
        self.predicate_policy_digest = policy.digest()
        self.proposer_minutes = proposer_minutes
        self.scorer_minutes = scorer_minutes
        self.verbose = bool(verbose)
        self.executable = executable
        self.cloud_policy_cache_snapshot = cloud_policy_cache_snapshot
        self.expected_codex_launcher_digest = expected_codex_launcher_digest
        self.expected_cloud_policy_cache_binding = (
            expected_cloud_policy_cache_binding
        )
        self.proposer_transport = proposer_transport
        self.scorer_transport = scorer_transport

        self.proposal_transport_result: TypedVisualTransportResult | None = None
        self.rejected_proposal_attempt: RejectedTypedVisualProposalAttempt | None = None
        self.compiled: CompiledVisualSemanticProposal | None = None
        self.pre_observation_commitment: SemanticPreObservationCommitment | None = None
        self.proposed_rule: ProposedRule | None = None
        self.support_sessions: list[_SemanticObservationSession] = []
        self.query_sessions: list[_SemanticObservationSession] = []
        self.query_artifacts: dict[str, VisualSemanticObservationArtifact] = {}

    def propose(self, support: SupportInput) -> ProposedRule:
        if self.proposed_rule is not None or self.proposal_transport_result is not None:
            raise SemanticEpisodeError("semantic proposer may be called exactly once")
        if not isinstance(support, SupportInput):
            raise TypeError("semantic proposer requires SupportInput")
        try:
            transport_result = propose_typed_visual(
                support.positive_paths,
                support.negative_paths,
                catalog=DIRECT_VISUAL_ATOM_CATALOG,
                protocol=self.protocol,
                minutes=self.proposer_minutes,
                verbose=self.verbose,
                executable=self.executable,
                cloud_policy_cache_snapshot=self.cloud_policy_cache_snapshot,
                transport=self.proposer_transport,
            )
        except TypedVisualProposalRejected as exc:
            self.rejected_proposal_attempt = exc.attempt
            _validate_execution_receipt(
                exc.attempt.receipt,
                expected_launcher_digest=self.expected_codex_launcher_digest,
                expected_cache_binding=self.expected_cloud_policy_cache_binding,
                role="typed proposer",
            )
            raise
        _validate_execution_receipt(
            transport_result.receipt,
            expected_launcher_digest=self.expected_codex_launcher_digest,
            expected_cache_binding=self.expected_cloud_policy_cache_binding,
            role="typed proposer",
        )
        compiled = compile_visual_semantic_proposal(
            transport_result.proposal,
            policy=self.policy,
            expected_policy_digest=self.predicate_policy_digest,
            family=self.family,
            issued_by=self.support_commitment.issued_by,
        )
        precommit = SemanticPreObservationCommitment(
            self.support_commitment, transport_result, compiled
        )
        proposed = ProposedRule(
            proposal_id="visual-semantic-" + transport_result.proposal.digest[:16],
            proposer_digest=precommit.digest,
            formula=compiled.formula,
            registry=compiled.registry,
            attachment_contract=compiled.attachment_contract,
        )
        self.proposal_transport_result = transport_result
        self.compiled = compiled
        self.pre_observation_commitment = precommit
        self.proposed_rule = proposed
        return proposed

    def _ready(self) -> tuple[CompiledVisualSemanticProposal, SemanticPreObservationCommitment]:
        if self.compiled is None or self.pre_observation_commitment is None:
            raise SemanticEpisodeError("proposal must be frozen before observations")
        self.pre_observation_commitment.assert_untampered()
        return self.compiled, self.pre_observation_commitment

    def _session(self, phase: str, ordinal: int) -> _SemanticObservationSession:
        compiled, precommit = self._ready()
        return _SemanticObservationSession(
            phase=phase,
            ordinal=ordinal,
            task_id=self.task_id,
            compiled=compiled,
            protocol=self.protocol,
            precommit=precommit,
            scorer_minutes=self.scorer_minutes,
            verbose=self.verbose,
            executable=self.executable,
            cloud_policy_cache_snapshot=self.cloud_policy_cache_snapshot,
            expected_codex_launcher_digest=(
                self.expected_codex_launcher_digest
            ),
            expected_cloud_policy_cache_binding=(
                self.expected_cloud_policy_cache_binding
            ),
            scorer_transport=self.scorer_transport,
        )

    def create_support_observer(self) -> _SemanticObservationSession:
        if len(self.support_sessions) >= 12:
            raise SemanticEpisodeError("semantic support factory exceeded twelve calls")
        session = self._session("support", len(self.support_sessions))
        self.support_sessions.append(session)
        return session

    def create_observer(self) -> _SemanticObservationSession:
        if len(self.query_sessions) >= 2:
            raise SemanticEpisodeError("semantic query factory exceeded two calls")
        session = self._session("query", len(self.query_sessions))
        self.query_sessions.append(session)
        return session

    def collect_observers(
        self,
        issued: Sequence[tuple[str, _SemanticObservationSession]],
    ) -> None:
        """Collect query artifacts only after both isolated calls completed."""

        if len(issued) != 2:
            raise SemanticEpisodeError("semantic episode must collect two query calls")
        artifacts: dict[str, VisualSemanticObservationArtifact] = {}
        for query_id, session in issued:
            if (
                not any(session is item for item in self.query_sessions)
                or session.phase != "query"
            ):
                raise SemanticEpisodeError("collector received an unknown query session")
            if session.artifact is None:
                raise SemanticEpisodeError("query session has no completed artifact")
            if query_id in artifacts:
                raise SemanticEpisodeError("query collector received a duplicate ID")
            artifacts[query_id] = session.artifact
        self.query_artifacts = artifacts

    def artifact_data(self) -> dict[str, object]:
        """Return the semantic sidecar; benchmark artifacts remain authoritative."""

        precommit = self.pre_observation_commitment
        return {
            "schema": SEMANTIC_EPISODE_ARCHIVE_SCHEMA,
            "predicate_mode": self.predicate_mode,
            "predicate_policy_digest": self.predicate_policy_digest,
            "pre_observation_commitment": (
                None if precommit is None else precommit.to_data()
            ),
            "rejected_proposal_attempt": (
                None
                if self.rejected_proposal_attempt is None
                else self.rejected_proposal_attempt.to_data()
            ),
            "support_observations": [
                None if item.artifact is None else item.artifact.to_data()
                for item in self.support_sessions
            ],
            "query_observations": {
                query_id: artifact.to_data()
                for query_id, artifact in sorted(self.query_artifacts.items())
            },
            "python_predicate_authoritative": True,
            "optional_checker_may_affect_result": False,
        }


__all__ = [
    "SEMANTIC_EPISODE_ARCHIVE_SCHEMA",
    "SemanticEpisodeError",
    "VisualSemanticEpisode",
]
