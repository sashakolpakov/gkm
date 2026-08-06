"""Verifier-owned headless Codex adapter for frozen support prototypes.

Codex receives only the labelled 6+6 support PNGs and selects exactly one
entry from a precommitted neutral feature catalog.  It never supplies feature
code, a threshold, weights, polarity, or a query observer.  This adapter owns
candidate-independent extraction, support-only fitting, canonical Python IR
compilation, fresh support replay, and held-out query evaluation.

The class implements the ordinary :mod:`bongard.benchmark` proposer, support
observer factory, and query observer boundaries.  The benchmark runner still
owns query release, joint prediction commitment, and label reveal.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable, Mapping

from bongard.artifacts import SupportCommitment, atom_paths
from bongard.benchmark import (
    ObservationInput,
    ProposedRule,
    SUPPORT_PROTOTYPE_PREDICATE_MODE,
    SupportGateMeasurement,
    SupportInput,
)
from bongard.evidence import Evidence
from bongard.legs.neutral_features import (
    NeutralFeatureExtraction,
    extract_neutral_features,
    feature_group_catalog,
    feature_group_catalog_digest,
    feature_space_for_group,
    project_neutral_feature_extraction,
    verify_neutral_feature_extraction,
)
from bongard.prototype_artifacts import (
    FeatureExtractionPreimage,
    PrototypeFreezePolicy,
    PrototypePreQueryFreeze,
    PrototypeQueryArtifact,
    PrototypeSupportReplayArtifact,
    canonical_digest,
)
from bongard.proposer import (
    DEFAULT_CODEX_MODEL,
    DEFAULT_REASONING_EFFORT,
    CloudPolicyCacheSnapshot,
    RejectedProposalAttempt,
    RejectedProposalError,
    RuleProposal,
    StructuredTransport,
    propose_pure_rule,
    run_codex_structured,
)
from bongard.support_prototypes import (
    PositivePrototypeFormula,
    SupportPrototypePlan,
    fit_support_prototypes,
    panel_side_assignment_digest,
)
from bongard.synthesis import CompiledProposal, compile_prototype_proposal


PROTOTYPE_EPISODE_SCHEMA = "bongard.headless-support-prototype-episode/v1"


class PrototypeEpisodeError(ValueError):
    """The verifier-owned prototype path violated its frozen contract."""


NeutralExtractor = Callable[[bytes], NeutralFeatureExtraction]
NeutralProjector = Callable[
    [NeutralFeatureExtraction, str | tuple[str, ...]], NeutralFeatureExtraction
]


class HeadlessPrototypeEpisode:
    """One Python-first, Lean-independent prototype episode adapter."""

    requires_empirical_support_gate = True
    predicate_mode = SUPPORT_PROTOTYPE_PREDICATE_MODE

    def __init__(
        self,
        *,
        support_commitment: SupportCommitment,
        policy: PrototypeFreezePolicy,
        model: str = DEFAULT_CODEX_MODEL,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        proposer_minutes: int = 15,
        verbose: bool = False,
        executable: str = "codex",
        proposer_transport: StructuredTransport = run_codex_structured,
        cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot | None = None,
        extractor: NeutralExtractor = extract_neutral_features,
        projector: NeutralProjector = project_neutral_feature_extraction,
    ) -> None:
        if not isinstance(support_commitment, SupportCommitment):
            raise TypeError("prototype episode requires a SupportCommitment")
        if not isinstance(policy, PrototypeFreezePolicy):
            raise TypeError("prototype episode requires a PrototypeFreezePolicy")
        if policy.feature_catalog_digest != feature_group_catalog_digest():
            raise PrototypeEpisodeError(
                "prototype policy names another neutral feature catalog"
            )
        canonical_catalog = {
            item.group_id: item.description for item in feature_group_catalog()
        }
        allowed_ids = tuple(
            item.feature_group_id for item in policy.allowed_feature_groups
        )
        if any(group_id not in canonical_catalog for group_id in allowed_ids):
            raise PrototypeEpisodeError(
                "prototype policy contains an unknown feature group"
            )
        for group_id in allowed_ids:
            policy.select(group_id, feature_space_for_group(group_id))
        if len(support_commitment.support) != 12:
            raise PrototypeEpisodeError(
                "prototype episode requires an exact 6+6 support commitment"
            )
        if sum(item.positive for item in support_commitment.support) != 6:
            raise PrototypeEpisodeError(
                "prototype support commitment is not six panels per side"
            )
        if not callable(extractor) or not callable(projector):
            raise TypeError("neutral extractor and projector must be callable")

        self.support_commitment = support_commitment
        self.policy = policy
        self.predicate_policy_digest = policy.digest()
        self.observable_catalog = {
            group_id: canonical_catalog[group_id] for group_id in allowed_ids
        }
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.proposer_minutes = proposer_minutes
        self.verbose = verbose
        self.executable = executable
        self.proposer_transport = proposer_transport
        self.cloud_policy_cache_snapshot = cloud_policy_cache_snapshot
        self.extractor = extractor
        self.projector = projector

        self._proposal_attempted = False
        self._proposal: RuleProposal | None = None
        self._rejected_proposal_attempt: RejectedProposalAttempt | None = None
        self._prequery: PrototypePreQueryFreeze | None = None
        self._compiled: CompiledProposal | None = None
        self._observations: dict[str, PrototypeQueryArtifact] = {}

    @property
    def proposal(self) -> RuleProposal | None:
        return self._proposal

    @property
    def prequery(self) -> PrototypePreQueryFreeze | None:
        return self._prequery

    def _extract(self, panel_bytes: bytes) -> NeutralFeatureExtraction:
        if not isinstance(panel_bytes, bytes) or not panel_bytes:
            raise PrototypeEpisodeError("neutral extraction requires exact PNG bytes")
        extraction = self.extractor(panel_bytes)
        if not isinstance(extraction, NeutralFeatureExtraction):
            raise TypeError("neutral extractor returned the wrong record type")
        verify_neutral_feature_extraction(extraction, panel_bytes)
        return extraction

    def _project(
        self,
        extraction: NeutralFeatureExtraction,
        group_id: str,
        panel_bytes: bytes,
    ) -> FeatureExtractionPreimage:
        projected = self.projector(extraction, group_id)
        if not isinstance(projected, NeutralFeatureExtraction):
            raise TypeError("neutral projector returned the wrong record type")
        verify_neutral_feature_extraction(projected, panel_bytes)
        return FeatureExtractionPreimage.from_extraction(panel_bytes, projected)

    @staticmethod
    def _read_support_side(paths: tuple[Path, ...]) -> tuple[bytes, ...]:
        payloads: list[bytes] = []
        for path in paths:
            if not isinstance(path, Path) or not path.is_file():
                raise PrototypeEpisodeError("benchmark support contains a missing PNG")
            payloads.append(path.read_bytes())
        return tuple(payloads)

    def propose(self, support: SupportInput) -> ProposedRule:
        """Extract support first, then make one closed-catalog Codex call."""

        if self._proposal_attempted:
            raise PrototypeEpisodeError(
                "one prototype episode permits exactly one proposer call"
            )
        if not isinstance(support, SupportInput):
            raise PrototypeEpisodeError("benchmark support input is malformed")
        self._proposal_attempted = True

        positive_bytes = self._read_support_side(support.positive_paths)
        negative_bytes = self._read_support_side(support.negative_paths)
        # All neutral numeric observables are computed before candidate
        # generation.  The later projection merely selects a frozen subset.
        positive_full = tuple(self._extract(payload) for payload in positive_bytes)
        negative_full = tuple(self._extract(payload) for payload in negative_bytes)
        for extraction in positive_full + negative_full:
            if extraction.evidence.disposition.value != "present":
                raise PrototypeEpisodeError(
                    "support extraction was not present; it cannot count as negative"
                )

        try:
            proposal = propose_pure_rule(
                support.positive_paths,
                support.negative_paths,
                observable_catalog=self.observable_catalog,
                model=self.model,
                reasoning_effort=self.reasoning_effort,
                minutes=self.proposer_minutes,
                verbose=self.verbose,
                executable=self.executable,
                cloud_policy_cache_snapshot=self.cloud_policy_cache_snapshot,
                transport=self.proposer_transport,
            )
        except RejectedProposalError as exc:
            self._rejected_proposal_attempt = exc.attempt
            raise
        if proposal.is_hybrid or len(proposal.formula_atoms) != 1:
            raise PrototypeEpisodeError(
                "prototype proposer must select exactly one PURE feature group"
            )
        selected_group = proposal.formula_atoms[0]
        feature_space = feature_space_for_group(selected_group)
        selected_policy = self.policy.select(selected_group, feature_space)

        # Candidate transport is not authorized to mutate support bytes.
        if positive_bytes != self._read_support_side(support.positive_paths) or (
            negative_bytes != self._read_support_side(support.negative_paths)
        ):
            raise PrototypeEpisodeError("support bytes changed during proposal")

        positive_preimages = tuple(
            self._project(extraction, selected_group, payload)
            for extraction, payload in zip(
                positive_full, positive_bytes, strict=True
            )
        )
        negative_preimages = tuple(
            self._project(extraction, selected_group, payload)
            for extraction, payload in zip(
                negative_full, negative_bytes, strict=True
            )
        )
        positive_packets = tuple(item.require_present() for item in positive_preimages)
        negative_packets = tuple(item.require_present() for item in negative_preimages)
        assignment_digest = panel_side_assignment_digest(
            tuple(item.panel_digest for item in positive_packets),
            tuple(item.panel_digest for item in negative_packets),
        )
        fit_plan = SupportPrototypePlan(
            feature_space.digest(),
            assignment_digest,
            self.policy.minimum_per_side,
        )
        prototypes = fit_support_prototypes(
            fit_plan,
            feature_space,
            positive_packets,
            negative_packets,
            expected_plan_digest=fit_plan.digest(),
        )
        predicate = PositivePrototypeFormula(
            claim=(
                "fixed positive-support prototype match for visual proposal "
                + proposal.digest
            ),
            feature_space_digest=feature_space.digest(),
            prototype_digest=prototypes.digest(),
            support_assignment_digest=assignment_digest,
            decision_margin=selected_policy.decision_margin,
        )
        prequery = PrototypePreQueryFreeze.create(
            support_commitment=self.support_commitment,
            policy=self.policy,
            selected_feature_group_id=selected_group,
            feature_space=feature_space,
            positive_support=positive_preimages,
            negative_support=negative_preimages,
            fit_plan=fit_plan,
            prototypes=prototypes,
            positive_formula=predicate,
            semantic_proposal_digest=proposal.digest.removeprefix("sha256:"),
        )
        compiled = compile_prototype_proposal(
            proposal,
            feature_space,
            prototypes,
            predicate,
            issued_by=self.support_commitment.issued_by,
        )
        if prequery.compiler_inputs().semantic_proposal_digest != compiled.proposer_digest:
            raise PrototypeEpisodeError(
                "prototype freeze belongs to another visual proposal"
            )

        self._proposal = proposal
        self._prequery = prequery
        self._compiled = compiled
        return ProposedRule(
            proposal_id="prototype-" + compiled.proposer_digest[:16],
            proposer_digest=compiled.proposer_digest,
            formula=compiled.formula,
            registry=compiled.registry,
            attachment_contract=compiled.attachment_contract,
        )

    def create_support_observer(self) -> "HeadlessPrototypeEpisode":
        if self._prequery is None or self._compiled is None:
            raise PrototypeEpisodeError("support replay cannot precede fitting")
        isolated = copy.deepcopy(self)
        if isolated is self:
            raise PrototypeEpisodeError("support observer retained object identity")
        isolated._observations = {}
        return isolated

    def _check_observation(self, observation: ObservationInput) -> bytes:
        if self._prequery is None or self._compiled is None:
            raise PrototypeEpisodeError("observation cannot precede prototype freeze")
        if not isinstance(observation, ObservationInput):
            raise PrototypeEpisodeError("observation input is malformed")
        if observation.query_id != "query" or observation.panel.blob_id != "query-panel":
            raise PrototypeEpisodeError("observation input is not neutral")
        if observation.panel_path.name != "query.png":
            raise PrototypeEpisodeError("neutral observation must be named query.png")
        if observation.freeze.proposer_digest != self._compiled.proposer_digest:
            raise PrototypeEpisodeError("observation freeze names another proposal")
        if observation.registry.digest() != self._compiled.registry.digest():
            raise PrototypeEpisodeError("observation registry differs from prototype")
        panel_bytes = observation.panel_path.read_bytes()
        observation.panel.verify_bytes(panel_bytes)
        return panel_bytes

    def _fresh_selected_preimage(
        self, observation: ObservationInput
    ) -> FeatureExtractionPreimage:
        if self._prequery is None:
            raise PrototypeEpisodeError("selected feature group is unavailable")
        panel_bytes = self._check_observation(observation)
        full = self._extract(panel_bytes)
        return self._project(
            full,
            self._prequery.selected_feature_group_id,
            panel_bytes,
        )

    def observe_support(self, panel: ObservationInput) -> SupportGateMeasurement:
        """Fresh deterministic replay with no support-side field."""

        if self._prequery is None:
            raise PrototypeEpisodeError("support replay cannot precede fitting")
        preimage = self._fresh_selected_preimage(panel)
        artifact = PrototypeSupportReplayArtifact.capture(
            freeze=self._prequery,
            extraction=preimage,
        )
        return SupportGateMeasurement(
            evidence=artifact.evidence.to_evidence(),
            observer_artifact=artifact.to_data(),
            transport_attempted=True,
        )

    def observe(
        self, query: ObservationInput
    ) -> Mapping[tuple[int, ...], Evidence[bool]]:
        """Extract and evaluate one isolated query after proposal freeze."""

        if self._prequery is None:
            raise PrototypeEpisodeError("query cannot precede prototype freeze")
        if query.query_id in self._observations:
            raise PrototypeEpisodeError("neutral query was already observed")
        preimage = self._fresh_selected_preimage(query)
        artifact = PrototypeQueryArtifact.capture(
            query_id=query.query_id,
            freeze=self._prequery,
            extraction=preimage,
        )
        paths = atom_paths(query.freeze.formula)
        if paths != ((),):
            raise PrototypeEpisodeError(
                "prototype episode formula must contain exactly one atom"
            )
        self._observations[query.query_id] = artifact
        return {(): artifact.evidence.to_evidence()}

    def artifact_data(self) -> dict[str, Any]:
        """Return policy, proposal, pre-query preimage, and query receipts."""

        return {
            "schema": PROTOTYPE_EPISODE_SCHEMA,
            "predicate_mode": self.predicate_mode,
            "predicate_policy": self.policy.to_data(),
            "predicate_policy_digest": self.predicate_policy_digest,
            "proposal": self._proposal.to_dict() if self._proposal else None,
            "rejected_proposal_attempt": (
                self._rejected_proposal_attempt.to_dict()
                if self._rejected_proposal_attempt is not None
                else None
            ),
            "pre_query_commitment": (
                self._prequery.committed_data() if self._prequery else None
            ),
            "observations": {
                query_id: artifact.to_data()
                for query_id, artifact in sorted(self._observations.items())
            },
        }

    @property
    def artifact_digest(self) -> str:
        return canonical_digest(self.artifact_data())


__all__ = [
    "HeadlessPrototypeEpisode",
    "PROTOTYPE_EPISODE_SCHEMA",
    "PrototypeEpisodeError",
]
