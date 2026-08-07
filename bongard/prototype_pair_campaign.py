"""Durable end-to-end coordinator for the prototype-pair engineering drill.

This module owns phase order, not visual semantics.  It persists the execution
precommit and exposure successor before releasing bytes, records every panel
release and model-attempt admission before transport, and delegates all visual
observation, calibration, predicate evaluation, version-space construction,
ranking, and query evaluation to their content-addressed authorities.

Calibration and support observations may execute concurrently only after their
releases and call claims have been persisted in deterministic schedule order.
Results are rejoined and archived in that same order.  Description, ranking,
and query turns are deliberately sequential.  Cold replay invokes no model.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import (
    RuntimeSourceSnapshotError,
    capture_loaded_source,
    verify_loaded_source,
)


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import importlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

from bongard.canonical import canonical_digest, canonical_json
from bongard.codex_no_tools_preflight import (
    CodexNoToolsAttestation,
    validate_codex_no_tools_attestation,
)
from bongard.official_panel_archive import OfficialPanelArchive, ReleasedOfficialPanel
from bongard.prototype_pair_cohort import PrototypePairCohortPlan
from bongard.prototype_pair_execution_precommit import (
    PHASE_ORDER,
    REQUIRED_RUNTIME_SOURCE_ROLES,
    PrototypePairExecutionPrecommit,
    PrototypePairExecutionPrecommitError,
    verify_prototype_pair_execution_precommit,
)
from bongard.prototype_scene_calibration import (
    PrototypeSceneCalibrationAssessment,
    PrototypeSceneCalibrationError,
    PrototypeSceneCalibrationFamily,
    PrototypeSceneCalibrationObservation,
    PrototypeSceneCalibrationPlan,
    PrototypeSceneEvaluationContext,
    assess_prototype_scene_calibration,
    calibration_algorithm_digest,
    create_prototype_scene_calibration_plan,
    fit_prototype_scene_calibration_family,
    verify_prototype_scene_calibration_family,
    verify_prototype_scene_calibration_plan,
)
from bongard.prototype_scene_headless_runner import (
    RUNNER_ID,
    PrototypeSceneCandidateFreeze,
    PrototypeSceneFreezeCommitReceipt,
    PrototypeSceneHeadlessArchive,
    PrototypeSceneHeadlessStatus,
    PrototypeSceneRankResponse,
    cold_replay_prototype_scene_headless_run,
    prototype_scene_rank_input_digest,
    prototype_scene_runner_source_digest,
    run_prototype_scene_headless,
)
from bongard.prototype_scene_codex_ranker import (
    PROTOTYPE_SCENE_CODEX_RANKER_PROTOCOL_ID,
    prototype_scene_codex_ranker_environment_digest,
    prototype_scene_codex_ranker_protocol_digest,
    prototype_scene_codex_ranker_transport_source_digest,
    verify_prototype_scene_codex_rank_response,
)
from bongard.prototype_object_scene_observer import (
    PROTOTYPE_SCENE_OBSERVER_PROTOCOL_ID,
    PrototypeReferenceCatalog,
    PrototypeRubricDescriptionArtifact,
    PrototypeSceneObserverArtifact,
    PrototypeSceneObserverStatus,
    build_prototype_reference_catalog,
    describe_prototype_references,
    observe_prototype_scene,
    prototype_rubric_description_protocol_digest,
    prototype_scene_observer_environment_digest,
    prototype_scene_observer_model_digest,
    prototype_scene_observer_source_digest,
    prototype_scene_scoring_protocol_digest,
    prototype_scene_transport_source_digest,
    seal_prototype_rubric_description_internal_error,
    seal_prototype_scene_internal_error,
    verify_prototype_reference_catalog,
    verify_prototype_rubric_description_artifact,
    verify_prototype_scene_observer_artifact,
)
from bongard.prototype_scene_predicates import (
    PrototypeScenePanelEvaluation,
    PrototypeScenePredicateLibrary,
)
from bongard.prototype_scene_runtime_adapter import (
    PrototypeSceneArtifactPurpose,
    PrototypeScenePhasedArtifactVerifier,
    PrototypeSceneRuntimeArtifactArchive,
    PrototypeSceneRuntimeArtifactInput,
    materialize_prototype_scene_calibration_observations,
    materialize_prototype_scene_panel,
    prototype_scene_evaluation_context_digest,
    prototype_scene_runtime_adapter_source_digest,
)
from bongard.prototype_scene_support_version_space import (
    PrototypeSceneSupportVersionSpace,
    build_prototype_scene_support_version_space,
)
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID
from bongard.release import OfficialReleaseDescriptor
from bongard.transport import CloudPolicyCacheSnapshot, CodexModelCatalogSnapshot


CAMPAIGN_SCHEMA = "gkm.bongard-prototype-pair-campaign.v3"
CAMPAIGN_ALGORITHM_ID = "bongard.prototype-pair/durable-campaign-v3"
CAMPAIGN_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_RAW_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}\Z")

_RUNTIME_SOURCE_MODULES = {
    "observer": "bongard.prototype_object_scene_observer",
    "observer-legacy-transport": "bongard.prototype_scene_observer",
    "object-observer-protocol": "bongard.prototype_object_observer_protocol",
    "object-hypotheses": "bongard.prototype_object_hypotheses",
    "object-profiles": "bongard.prototype_object_profiles",
    "visual-runtime": "bongard.prototype_visual_runtime",
    "visual-witnesses": "bongard.visual_witnesses",
    "contour-witnesses": "bongard.contour_witnesses",
    "visual-witness-bundle": "bongard.visual_witness_bundle",
    "calibration": "bongard.prototype_scene_calibration",
    "predicate": "bongard.prototype_scene_predicates",
    "version-space": "bongard.prototype_scene_support_version_space",
    "runner": "bongard.prototype_scene_headless_runner",
    "runtime-adapter": "bongard.prototype_scene_runtime_adapter",
    "ranker": "bongard.prototype_scene_codex_ranker",
    "campaign": "bongard.prototype_pair_campaign",
    "campaign-cli": "bongard.prototype_pair_campaign_cli",
    "campaign-store": "bongard.prototype_pair_campaign_store",
    "transport": "bongard.transport",
    "transport-preflight": "bongard.codex_no_tools_preflight",
    "official-panel-archive": "bongard.official_panel_archive",
    "precommit": "bongard.prototype_pair_execution_precommit",
    "canonical": "bongard.canonical",
    "exposure": "bongard.exposure",
    "release": "bongard.release",
    "cohort": "bongard.prototype_pair_cohort",
    "cohorts": "bongard.cohorts",
    "corpus": "bongard.corpus",
    "historical-exposure": "bongard.historical_exposure",
    "image-audit": "bongard.image_audit",
    "cluster-binomial": "bongard.cluster_binomial",
    "python-authority": "bongard.python_predicate_authority",
    "grounded-compat": "bongard.grounded_multimodal_predicates",
    "package-init": "bongard",
    "source-snapshot": "bongard.runtime_source_snapshot",
}

_TRIVIAL_RUNTIME_SOURCE_ROLES = frozenset(
    {"grounded-compat", "python-authority"}
)
_TRIVIAL_RUNTIME_SOURCE_SNAPSHOTS = {
    role: hashlib.sha256(
        Path(importlib.import_module(_RUNTIME_SOURCE_MODULES[role]).__file__).read_bytes()
    ).hexdigest()
    for role in _TRIVIAL_RUNTIME_SOURCE_ROLES
}


class PrototypePairCampaignError(RuntimeError):
    """A campaign phase, persistence edge, or replay invariant failed."""


class PrototypePairCampaignStatus(str, Enum):
    COMPLETE = "complete"
    DESCRIPTION_GAP = "description_gap"
    CALIBRATION_GAP = "calibration_gap"
    SUPPORT_LANGUAGE_GAP = "support_language_gap"
    SUPPORT_WITNESS_GAP = "support_witness_gap"
    RANKER_ERROR = "ranker_error"


def _address(value: object) -> str:
    return "sha256:" + canonical_digest(value)


def _require_address(value: object, label: str) -> str:
    if not isinstance(value, str) or _ADDRESS.fullmatch(value) is None:
        raise PrototypePairCampaignError(f"{label} must be a sha256: address")
    return value


def _require_raw_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or _RAW_SHA256.fullmatch(value) is None:
        raise PrototypePairCampaignError(f"{label} must be lowercase SHA-256")
    return value


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise PrototypePairCampaignError(f"{label} must be a bounded identifier")
    return value


def _module_source_sha256(module_name: str) -> str:
    module = importlib.import_module(module_name)
    source = getattr(module, "__file__", None)
    if not isinstance(source, str) or not source:
        raise PrototypePairCampaignError(
            f"runtime module {module_name!r} has no source file"
        )
    current = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    role = next(
        (
            candidate_role
            for candidate_role, candidate_module in _RUNTIME_SOURCE_MODULES.items()
            if candidate_module == module_name
        ),
        None,
    )
    if role == "cohort":
        loaded = getattr(module, "PLANNER_SOURCE_SHA256", None)
        if current != loaded:
            raise PrototypePairCampaignError(
                "cohort source differs from its imported planner seal"
            )
        return current
    if role in _TRIVIAL_RUNTIME_SOURCE_ROLES:
        if current != _TRIVIAL_RUNTIME_SOURCE_SNAPSHOTS[role]:
            raise PrototypePairCampaignError(
                f"runtime source for {module_name!r} changed after import"
            )
        return current
    try:
        loaded = verify_loaded_source(module_name)
    except RuntimeSourceSnapshotError as exc:
        raise PrototypePairCampaignError(str(exc)) from exc
    if current != loaded:
        raise PrototypePairCampaignError(
            f"runtime source for {module_name!r} differs from its loaded seal"
        )
    return loaded


def prototype_pair_campaign_runtime_source_digests() -> dict[str, str]:
    """Hash the curated campaign authority modules before precommit.

    This is an explicit security inventory, not a claim to recursively enumerate
    every transitive Python import.  Callers may bind additional source roles.
    """

    if not set(REQUIRED_RUNTIME_SOURCE_ROLES) <= set(_RUNTIME_SOURCE_MODULES):
        raise PrototypePairCampaignError(
            "campaign source inventory omits a precommit-required role"
        )
    result = {
        role: _module_source_sha256(module_name)
        for role, module_name in sorted(_RUNTIME_SOURCE_MODULES.items())
    }
    if result["campaign"] != CAMPAIGN_SOURCE_SHA256:
        raise PrototypePairCampaignError("campaign source changed while hashing")
    return result


def prototype_pair_campaign_algorithm_digest() -> str:
    return _address(
        {
            "schema": "gkm.bongard-prototype-pair-campaign-algorithm.v3",
            "algorithm_id": CAMPAIGN_ALGORITHM_ID,
            "source_sha256": CAMPAIGN_SOURCE_SHA256,
            "phase_order": list(PHASE_ORDER),
            "parallel_phases": [
                "twenty_eight_calibration_scenes_released_and_observed",
                "twelve_support_scenes_released_and_observed",
            ],
            "parallel_protocol": (
                "persist-releases-and-claims-in-schedule-order-before-workers;"
                "archive-results-in-schedule-order-after-join"
            ),
            "model_call_branch_totals": {
                "complete": 44,
                "support_gap": 41,
                "ranker_error": 42,
                "calibration_gap": 29,
                "description_gap": 1,
            },
            "query_release_requires_durable_candidate_freeze": True,
            "cold_replay_model_calls": 0,
            "transport_preflight": (
                "one-frozen-two-modality-no-tools-attestation-reused-by-all-"
                "observer-and-ranker-calls-and-revalidated-on-cold-replay"
            ),
            "terminal_journal_sealed_before_campaign_artifact": True,
            "model_calls_made_semantics": (
                "cumulative-unique-terminal-admissions;"
                "reused-terminal-makes-no-transport-call"
            ),
            "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
            "python_is_canonical_authority": True,
            "lean_required": False,
            "lean_affects_identity_or_decision": False,
        }
    )


@runtime_checkable
class CampaignClock(Protocol):
    def now(self, phase: str, subject_id: str, event: str) -> str: ...


@runtime_checkable
class CampaignStore(Protocol):
    """Narrow durable-store surface; implemented by campaign_store."""

    def persist_execution_precommit(
        self, precommit_bytes: bytes, expected_digest: str
    ) -> object: ...

    def verify_execution_precommit(
        self,
        receipt: object,
        expected_digest: str,
        expected_bytes: bytes | None = None,
    ) -> object: ...

    def authorize_release(self, **kwargs: object) -> object: ...

    def persist_canonical_object(
        self, kind: str, data: Mapping[str, Any], expected_record_digest: str
    ) -> object: ...

    def load_canonical_object(
        self, receipt: object, expected_record_digest: str
    ) -> Mapping[str, Any]: ...

    def claim_call(self, **kwargs: object) -> object: ...

    def finish_call(self, **kwargs: object) -> object: ...

    def load_call_journal(
        self,
        expected_key_digest: str,
        *,
        expected_authorization_digest: str | None = None,
    ) -> tuple[object, object | None]: ...

    def seal_call_journal(self, *args: object, **kwargs: object) -> object: ...

    def verify_call_journal_seal(self, *args: object, **kwargs: object) -> object: ...

    def persist_candidate_freeze(
        self, canonical_bytes: bytes, expected_record_digest: str | None = None
    ) -> PrototypeSceneFreezeCommitReceipt | Mapping[str, Any]: ...

    def load_candidate_freeze_commit(
        self, expected_commit_digest: str
    ) -> tuple[object, object, object]: ...


@dataclass(frozen=True, slots=True)
class PrototypePairCampaignConfiguration:
    actor: str
    parallel_workers: int
    observer_minutes: int
    observer_verbose: bool
    observer_executable: str
    ranker_minutes: int
    ranker_verbose: bool
    ranker_executable: str
    runtime_archive_source_id: str
    runtime_verifier_id: str

    def __post_init__(self) -> None:
        _identifier(self.actor, "campaign actor")
        _identifier(self.runtime_archive_source_id, "runtime archive source ID")
        _identifier(self.runtime_verifier_id, "runtime verifier ID")
        if (
            isinstance(self.parallel_workers, bool)
            or not isinstance(self.parallel_workers, int)
            or not 1 <= self.parallel_workers <= 28
            or isinstance(self.observer_minutes, bool)
            or not isinstance(self.observer_minutes, int)
            or not 1 <= self.observer_minutes <= 120
            or not isinstance(self.observer_verbose, bool)
            or not isinstance(self.observer_executable, str)
            or not self.observer_executable
            or isinstance(self.ranker_minutes, bool)
            or not isinstance(self.ranker_minutes, int)
            or not 1 <= self.ranker_minutes <= 120
            or not isinstance(self.ranker_verbose, bool)
            or not isinstance(self.ranker_executable, str)
            or not self.ranker_executable
        ):
            raise PrototypePairCampaignError("campaign configuration differs")

    def to_data(self) -> dict[str, object]:
        return {
            "actor": self.actor,
            "parallel_workers": self.parallel_workers,
            "observer_minutes": self.observer_minutes,
            "observer_verbose": self.observer_verbose,
            "observer_executable": self.observer_executable,
            "ranker_minutes": self.ranker_minutes,
            "ranker_verbose": self.ranker_verbose,
            "ranker_executable": self.ranker_executable,
            "runtime_archive_source_id": self.runtime_archive_source_id,
            "runtime_verifier_id": self.runtime_verifier_id,
        }

    @property
    def record_digest(self) -> str:
        return _address(
            {
                "schema": "gkm.bongard-prototype-pair-campaign-config.v1",
                **self.to_data(),
            }
        )

    @classmethod
    def from_data(
        cls, value: Mapping[str, Any]
    ) -> "PrototypePairCampaignConfiguration":
        expected = {
            "actor",
            "parallel_workers",
            "observer_minutes",
            "observer_verbose",
            "observer_executable",
            "ranker_minutes",
            "ranker_verbose",
            "ranker_executable",
            "runtime_archive_source_id",
            "runtime_verifier_id",
            "record_digest",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise PrototypePairCampaignError("campaign configuration fields differ")
        result = cls(
            actor=value["actor"],
            parallel_workers=value["parallel_workers"],
            observer_minutes=value["observer_minutes"],
            observer_verbose=value["observer_verbose"],
            observer_executable=value["observer_executable"],
            ranker_minutes=value["ranker_minutes"],
            ranker_verbose=value["ranker_verbose"],
            ranker_executable=value["ranker_executable"],
            runtime_archive_source_id=value["runtime_archive_source_id"],
            runtime_verifier_id=value["runtime_verifier_id"],
        )
        if result.record_digest != value["record_digest"]:
            raise PrototypePairCampaignError("campaign configuration digest differs")
        return result


@dataclass(frozen=True, slots=True)
class PrototypePairCampaignCallFailure:
    """Canonical terminal evidence when a claimed callback raises."""

    phase: str
    subject_id: str
    failure_type: str
    failure_message_digest: str
    record_digest: str

    def __post_init__(self) -> None:
        _identifier(self.phase, "failure phase")
        _identifier(self.subject_id, "failure subject")
        _identifier(self.failure_type, "failure type")
        _require_raw_sha(self.failure_message_digest, "failure message digest")
        if self.record_digest != _address(self.content_dict()):
            raise PrototypePairCampaignError("call failure digest differs")

    @classmethod
    def from_exception(
        cls, *, phase: str, subject_id: str, exception: Exception
    ) -> "PrototypePairCampaignCallFailure":
        failure_type = type(exception).__name__
        if _IDENTIFIER.fullmatch(failure_type) is None:
            failure_type = "UnclassifiedCampaignFailure"
        values: dict[str, str] = {
            "phase": phase,
            "subject_id": subject_id,
            "failure_type": failure_type,
            "failure_message_digest": hashlib.sha256(
                str(exception).encode("utf-8", errors="replace")
            ).hexdigest(),
        }
        provisional = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(provisional, name, value)
        return cls(**values, record_digest=_address(provisional.content_dict()))

    def content_dict(self) -> dict[str, object]:
        return {
            "schema": "gkm.bongard-prototype-pair-call-failure.v1",
            "phase": self.phase,
            "subject_id": self.subject_id,
            "failure_type": self.failure_type,
            "failure_message_digest": self.failure_message_digest,
            "message_archived": False,
            "terminal": True,
        }

    def to_data(self) -> dict[str, object]:
        return {**self.content_dict(), "record_digest": self.record_digest}

    @staticmethod
    def decode_campaign_artifact(
        value: Mapping[str, Any]
    ) -> "PrototypePairCampaignArtifact":
        expected = {
            "schema",
            "status",
            "precommit_digest",
            "cohort_plan_digest",
            "configuration",
            "exposure_successor_digest",
            "precommit_receipt",
            "release_authorization",
            "stored_objects",
            "call_terminals",
            "call_journal_seal_digest",
            "released_panels",
            "reference_catalog",
            "description_artifact",
            "calibration_plan",
            "calibration_artifacts",
            "calibration_runtime_archive_digest",
            "calibration_assessment",
            "calibration_family",
            "predicate_library",
            "support_artifacts",
            "support_panels",
            "support_version_space",
            "support_runtime_archive_digest",
            "query_artifacts",
            "query_runtime_archive_digest",
            "headless_archive",
            "call_failures",
            "phase_trace",
            "model_calls_made",
            "runtime_authority",
            "record_digest",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise PrototypePairCampaignError("campaign artifact fields differ")
        if value["schema"] != CAMPAIGN_SCHEMA:
            raise PrototypePairCampaignError("campaign schema differs")
        try:
            status = PrototypePairCampaignStatus(value["status"])
        except (TypeError, ValueError) as exc:
            raise PrototypePairCampaignError("campaign status differs") from exc
        for name in (
            "configuration",
            "precommit_receipt",
            "release_authorization",
            "reference_catalog",
            "description_artifact",
            "runtime_authority",
        ):
            if not isinstance(value[name], Mapping):
                raise PrototypePairCampaignError(f"campaign {name} is malformed")

        def rows(name: str) -> list[Any]:
            raw = value[name]
            if not isinstance(raw, list):
                raise PrototypePairCampaignError(f"campaign {name} must be a list")
            return raw

        def optional_mapping(name: str) -> Mapping[str, Any] | None:
            raw = value[name]
            if raw is None:
                return None
            if not isinstance(raw, Mapping):
                raise PrototypePairCampaignError(f"campaign {name} is malformed")
            return raw

        authority = value["runtime_authority"]
        expected_authority = {
            "campaign_source_sha256",
            "campaign_algorithm_digest",
            "predicate_authority_id",
            "python_is_canonical_authority",
            "lean_required",
            "lean_affects_identity_or_decision",
            "cold_replay_model_calls",
            "model_calls_made_semantics",
        }
        if set(authority) != expected_authority or {
            "predicate_authority_id": authority["predicate_authority_id"],
            "python_is_canonical_authority": authority[
                "python_is_canonical_authority"
            ],
            "lean_required": authority["lean_required"],
            "lean_affects_identity_or_decision": authority[
                "lean_affects_identity_or_decision"
            ],
            "cold_replay_model_calls": authority["cold_replay_model_calls"],
            "model_calls_made_semantics": authority[
                "model_calls_made_semantics"
            ],
        } != {
            "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
            "python_is_canonical_authority": True,
            "lean_required": False,
            "lean_affects_identity_or_decision": False,
            "cold_replay_model_calls": 0,
            "model_calls_made_semantics": (
                "cumulative-unique-terminal-admissions;"
                "reused-terminal-makes-no-transport-call"
            ),
        }:
            raise PrototypePairCampaignError("campaign runtime authority differs")
        calibration_plan_raw = optional_mapping("calibration_plan")
        assessment_raw = optional_mapping("calibration_assessment")
        family_raw = optional_mapping("calibration_family")
        library_raw = optional_mapping("predicate_library")
        version_raw = optional_mapping("support_version_space")
        headless_raw = optional_mapping("headless_archive")
        result = PrototypePairCampaignArtifact(
            status=status,
            precommit_digest=value["precommit_digest"],
            cohort_plan_digest=value["cohort_plan_digest"],
            configuration=PrototypePairCampaignConfiguration.from_data(
                value["configuration"]
            ),
            exposure_successor_digest=value["exposure_successor_digest"],
            precommit_receipt=dict(value["precommit_receipt"]),
            release_authorization=dict(value["release_authorization"]),
            stored_objects=tuple(
                PrototypePairStoredObject.from_data(item)
                for item in rows("stored_objects")
            ),
            call_terminals=tuple(
                dict(item) if isinstance(item, Mapping) else item
                for item in rows("call_terminals")
            ),
            call_journal_seal_digest=value["call_journal_seal_digest"],
            released_panels=tuple(
                ReleasedOfficialPanel.from_data(item)
                for item in rows("released_panels")
            ),
            reference_catalog=PrototypeReferenceCatalog.from_data(
                value["reference_catalog"]
            ),
            description_artifact=PrototypeRubricDescriptionArtifact.from_data(
                value["description_artifact"]
            ),
            calibration_plan=(
                None
                if calibration_plan_raw is None
                else PrototypeSceneCalibrationPlan.from_data(calibration_plan_raw)
            ),
            calibration_artifacts=tuple(
                PrototypeSceneObserverArtifact.from_data(item)
                for item in rows("calibration_artifacts")
            ),
            calibration_runtime_archive_digest=value[
                "calibration_runtime_archive_digest"
            ],
            calibration_assessment=(
                None
                if assessment_raw is None
                else PrototypeSceneCalibrationAssessment.from_data(assessment_raw)
            ),
            calibration_family=(
                None
                if family_raw is None
                else PrototypeSceneCalibrationFamily.from_data(family_raw)
            ),
            predicate_library=(
                None
                if library_raw is None
                else PrototypeScenePredicateLibrary.from_data(library_raw)
            ),
            support_artifacts=tuple(
                PrototypeSceneObserverArtifact.from_data(item)
                for item in rows("support_artifacts")
            ),
            support_panels=tuple(
                PrototypeScenePanelEvaluation.from_data(item)
                for item in rows("support_panels")
            ),
            support_version_space=(
                None
                if version_raw is None
                else PrototypeSceneSupportVersionSpace.from_data(version_raw)
            ),
            support_runtime_archive_digest=value[
                "support_runtime_archive_digest"
            ],
            query_artifacts=tuple(
                PrototypeSceneObserverArtifact.from_data(item)
                for item in rows("query_artifacts")
            ),
            query_runtime_archive_digest=value["query_runtime_archive_digest"],
            headless_archive=(
                None
                if headless_raw is None
                else PrototypeSceneHeadlessArchive.from_data(headless_raw)
            ),
            call_failures=tuple(
                PrototypePairCampaignCallFailure.from_data(item)
                for item in rows("call_failures")
            ),
            phase_trace=tuple(rows("phase_trace")),
            model_calls_made=value["model_calls_made"],
            campaign_source_sha256=authority["campaign_source_sha256"],
            campaign_algorithm_digest=authority["campaign_algorithm_digest"],
            record_digest=value["record_digest"],
        )
        if result.to_data() != dict(value):
            raise PrototypePairCampaignError("campaign artifact is not canonical")
        return result

    @classmethod
    def from_data(
        cls, value: Mapping[str, Any]
    ) -> "PrototypePairCampaignCallFailure":
        expected = {
            "schema",
            "phase",
            "subject_id",
            "failure_type",
            "failure_message_digest",
            "message_archived",
            "terminal",
            "record_digest",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise PrototypePairCampaignError("call failure fields differ")
        result = cls(
            phase=value["phase"],
            subject_id=value["subject_id"],
            failure_type=value["failure_type"],
            failure_message_digest=value["failure_message_digest"],
            record_digest=value["record_digest"],
        )
        if result.to_data() != dict(value):
            raise PrototypePairCampaignError("call failure is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class PrototypePairStoredObject:
    """Campaign-side binding to a store-owned immutable receipt."""

    kind: str
    object_identity_digest: str
    storage_receipt: Mapping[str, Any]
    storage_receipt_digest: str
    record_digest: str

    def __post_init__(self) -> None:
        _identifier(self.kind, "stored object kind")
        if not isinstance(self.object_identity_digest, str) or (
            _ADDRESS.fullmatch(self.object_identity_digest) is None
            and _RAW_SHA256.fullmatch(self.object_identity_digest) is None
        ):
            raise PrototypePairCampaignError("stored object identity is invalid")
        if not isinstance(self.storage_receipt, Mapping) or any(
            not isinstance(key, str) for key in self.storage_receipt
        ):
            raise PrototypePairCampaignError("storage receipt must be an object")
        canonical_receipt = dict(self.storage_receipt)
        object.__setattr__(self, "storage_receipt", canonical_receipt)
        if (
            self.storage_receipt_digest != _address(canonical_receipt)
            or self.record_digest != _address(self.content_dict())
        ):
            raise PrototypePairCampaignError("stored object receipt differs")

    @classmethod
    def seal(
        cls, *, kind: str, object_identity_digest: str, storage_receipt: object
    ) -> "PrototypePairStoredObject":
        receipt_data = _to_data(storage_receipt, "storage receipt")
        values: dict[str, object] = {
            "kind": kind,
            "object_identity_digest": object_identity_digest,
            "storage_receipt": receipt_data,
            "storage_receipt_digest": _address(receipt_data),
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,  # type: ignore[arg-type]
            record_digest=_address(provisional.content_dict()),
        )

    def content_dict(self) -> dict[str, object]:
        return {
            "schema": "gkm.bongard-prototype-pair-stored-object.v1",
            "kind": self.kind,
            "object_identity_digest": self.object_identity_digest,
            "storage_receipt": dict(self.storage_receipt),
            "storage_receipt_digest": self.storage_receipt_digest,
        }

    def to_data(self) -> dict[str, object]:
        return {**self.content_dict(), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "PrototypePairStoredObject":
        expected = {
            "schema",
            "kind",
            "object_identity_digest",
            "storage_receipt",
            "storage_receipt_digest",
            "record_digest",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise PrototypePairCampaignError("stored object fields differ")
        result = cls(
            kind=value["kind"],
            object_identity_digest=value["object_identity_digest"],
            storage_receipt=value["storage_receipt"],
            storage_receipt_digest=value["storage_receipt_digest"],
            record_digest=value["record_digest"],
        )
        if result.to_data() != dict(value):
            raise PrototypePairCampaignError("stored object is not canonical")
        return result


def _to_data(value: object, label: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        raw = dict(value)
    else:
        method = getattr(value, "to_data", None)
        if not callable(method):
            method = getattr(value, "to_dict", None)
        if not callable(method):
            raise PrototypePairCampaignError(f"{label} is not serializable")
        raw = method()
    if not isinstance(raw, Mapping) or any(not isinstance(key, str) for key in raw):
        raise PrototypePairCampaignError(f"{label} is not an object")
    # Round-trip through canonical JSON validation without changing caller data.
    canonical_json(dict(raw))
    return dict(raw)


@dataclass(frozen=True, slots=True)
class PrototypePairCampaignArtifact:
    status: PrototypePairCampaignStatus
    precommit_digest: str
    cohort_plan_digest: str
    configuration: PrototypePairCampaignConfiguration
    exposure_successor_digest: str
    precommit_receipt: Mapping[str, Any]
    release_authorization: Mapping[str, Any]
    stored_objects: tuple[PrototypePairStoredObject, ...]
    call_terminals: tuple[Mapping[str, Any], ...]
    call_journal_seal_digest: str
    released_panels: tuple[ReleasedOfficialPanel, ...]
    reference_catalog: PrototypeReferenceCatalog
    description_artifact: PrototypeRubricDescriptionArtifact
    calibration_plan: PrototypeSceneCalibrationPlan | None
    calibration_artifacts: tuple[PrototypeSceneObserverArtifact, ...]
    calibration_runtime_archive_digest: str | None
    calibration_assessment: PrototypeSceneCalibrationAssessment | None
    calibration_family: PrototypeSceneCalibrationFamily | None
    predicate_library: PrototypeScenePredicateLibrary | None
    support_artifacts: tuple[PrototypeSceneObserverArtifact, ...]
    support_panels: tuple[PrototypeScenePanelEvaluation, ...]
    support_version_space: PrototypeSceneSupportVersionSpace | None
    support_runtime_archive_digest: str | None
    query_artifacts: tuple[PrototypeSceneObserverArtifact, ...]
    query_runtime_archive_digest: str | None
    headless_archive: PrototypeSceneHeadlessArchive | None
    call_failures: tuple[PrototypePairCampaignCallFailure, ...]
    phase_trace: tuple[str, ...]
    model_calls_made: int
    campaign_source_sha256: str
    campaign_algorithm_digest: str
    record_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.status, PrototypePairCampaignStatus):
            raise TypeError("status must be PrototypePairCampaignStatus")
        for name in (
            "precommit_digest",
            "cohort_plan_digest",
            "exposure_successor_digest",
            "call_journal_seal_digest",
            "campaign_algorithm_digest",
            "record_digest",
        ):
            _require_address(getattr(self, name), name)
        _require_raw_sha(self.campaign_source_sha256, "campaign source SHA-256")
        for name in (
            "calibration_runtime_archive_digest",
            "support_runtime_archive_digest",
            "query_runtime_archive_digest",
        ):
            value = getattr(self, name)
            if value is not None:
                _require_address(value, name)
        if isinstance(self.model_calls_made, bool) or not isinstance(
            self.model_calls_made, int
        ):
            raise PrototypePairCampaignError("model call count must be an integer")
        if not isinstance(self.configuration, PrototypePairCampaignConfiguration):
            raise TypeError("configuration must be PrototypePairCampaignConfiguration")
        for name, value in (
            ("precommit receipt", self.precommit_receipt),
            ("release authorization", self.release_authorization),
        ):
            if not isinstance(value, Mapping) or any(
                not isinstance(key, str) for key in value
            ):
                raise PrototypePairCampaignError(f"{name} must be an object")
            canonical_json(dict(value))
        canonical_terminals = tuple(
            _to_data(value, "call terminal") for value in self.call_terminals
        )
        object.__setattr__(self, "call_terminals", canonical_terminals)
        if (
            len({item.panel_id for item in self.released_panels})
            != len(self.released_panels)
            or any(
                item.execution_precommit_digest != self.precommit_digest
                or item.exposure_successor_digest != self.exposure_successor_digest
                for item in self.released_panels
            )
        ):
            raise PrototypePairCampaignError("released panel inventory differs")
        if (
            self.campaign_source_sha256 != CAMPAIGN_SOURCE_SHA256
            or self.campaign_algorithm_digest
            != prototype_pair_campaign_algorithm_digest()
        ):
            raise PrototypePairCampaignError("campaign authority differs")
        self._validate_terminal_shape()
        if self.record_digest != _address(self.content_dict()):
            raise PrototypePairCampaignError("campaign artifact digest differs")

    def _validate_terminal_shape(self) -> None:
        status = self.status
        if self.description_artifact.plan_digest != self.cohort_plan_digest:
            raise PrototypePairCampaignError("description plan parent differs")
        common_success = self.description_artifact.status is (
            PrototypeSceneObserverStatus.SUCCESS
        )
        expected_trace: tuple[str, ...]
        if status is PrototypePairCampaignStatus.DESCRIPTION_GAP:
            valid = (
                not common_success
                and len(self.released_panels) == 6
                and self.calibration_plan is None
                and not self.calibration_artifacts
                and self.calibration_runtime_archive_digest is None
                and self.calibration_assessment is None
                and self.calibration_family is None
                and self.predicate_library is None
                and not self.support_artifacts
                and not self.support_panels
                and self.support_version_space is None
                and self.support_runtime_archive_digest is None
                and not self.query_artifacts
                and self.query_runtime_archive_digest is None
                and self.headless_archive is None
                and not self.call_failures
                and self.model_calls_made == 1
            )
            expected_trace = (*PHASE_ORDER[:4], PHASE_ORDER[-1])
        elif status is PrototypePairCampaignStatus.CALIBRATION_GAP:
            valid = (
                common_success
                and len(self.released_panels) == 34
                and self.calibration_plan is not None
                and len(self.calibration_artifacts) == 28
                and self.calibration_runtime_archive_digest is not None
                and self.calibration_assessment is not None
                and not self.calibration_assessment.all_four_bounds_accepted
                and self.calibration_family is None
                and self.predicate_library is None
                and not self.support_artifacts
                and not self.support_panels
                and self.support_version_space is None
                and self.support_runtime_archive_digest is None
                and not self.query_artifacts
                and self.query_runtime_archive_digest is None
                and self.headless_archive is None
                and not self.call_failures
                and self.model_calls_made == 29
            )
            expected_trace = (*PHASE_ORDER[:6], PHASE_ORDER[-1])
        elif status in {
            PrototypePairCampaignStatus.SUPPORT_LANGUAGE_GAP,
            PrototypePairCampaignStatus.SUPPORT_WITNESS_GAP,
        }:
            expected_headless = (
                PrototypeSceneHeadlessStatus.LANGUAGE_GAP
                if status is PrototypePairCampaignStatus.SUPPORT_LANGUAGE_GAP
                else PrototypeSceneHeadlessStatus.WITNESS_GAP
            )
            valid = (
                common_success
                and len(self.released_panels) == 46
                and self.calibration_plan is not None
                and len(self.calibration_artifacts) == 28
                and self.calibration_runtime_archive_digest is not None
                and self.calibration_assessment is not None
                and self.calibration_assessment.all_four_bounds_accepted
                and self.calibration_family is not None
                and self.predicate_library is not None
                and len(self.support_artifacts) == 12
                and len(self.support_panels) == 12
                and self.support_version_space is not None
                and self.support_runtime_archive_digest is not None
                and not self.query_artifacts
                and self.query_runtime_archive_digest is None
                and self.headless_archive is not None
                and self.headless_archive.status is expected_headless
                and not self.call_failures
                and self.model_calls_made == 41
            )
            expected_trace = (*PHASE_ORDER[:9], PHASE_ORDER[-1])
        elif status is PrototypePairCampaignStatus.RANKER_ERROR:
            valid = (
                common_success
                and len(self.released_panels) == 46
                and self.calibration_plan is not None
                and len(self.calibration_artifacts) == 28
                and self.calibration_runtime_archive_digest is not None
                and self.calibration_assessment is not None
                and self.calibration_assessment.all_four_bounds_accepted
                and self.calibration_family is not None
                and self.predicate_library is not None
                and len(self.support_artifacts) == 12
                and len(self.support_panels) == 12
                and self.support_version_space is not None
                and self.support_runtime_archive_digest is not None
                and not self.query_artifacts
                and self.query_runtime_archive_digest is None
                and self.headless_archive is None
                and len(self.call_failures) == 1
                and self.call_failures[0].phase
                == "headless_codex_candidate_ranked"
                and self.model_calls_made == 42
            )
            expected_trace = (*PHASE_ORDER[:10], PHASE_ORDER[-1])
        else:
            valid = (
                status is PrototypePairCampaignStatus.COMPLETE
                and common_success
                and len(self.released_panels) == 48
                and self.calibration_plan is not None
                and len(self.calibration_artifacts) == 28
                and self.calibration_runtime_archive_digest is not None
                and self.calibration_assessment is not None
                and self.calibration_assessment.all_four_bounds_accepted
                and self.calibration_family is not None
                and self.predicate_library is not None
                and len(self.support_artifacts) == 12
                and len(self.support_panels) == 12
                and self.support_version_space is not None
                and self.support_runtime_archive_digest is not None
                and len(self.query_artifacts) == 2
                and self.query_runtime_archive_digest is not None
                and self.headless_archive is not None
                and self.headless_archive.status
                is PrototypeSceneHeadlessStatus.COMPLETE
                and not self.call_failures
                and self.model_calls_made == 44
            )
            expected_trace = PHASE_ORDER
        if not valid or self.phase_trace != expected_trace:
            raise PrototypePairCampaignError("campaign terminal shape differs")
        if len(self.call_terminals) != self.model_calls_made:
            raise PrototypePairCampaignError(
                "every admitted model call must have one terminal outcome"
            )
        if self.support_version_space is not None and self.headless_archive is not None:
            if self.headless_archive.version_space != self.support_version_space:
                raise PrototypePairCampaignError("headless version space differs")

    def content_dict(self) -> dict[str, object]:
        return {
            "schema": CAMPAIGN_SCHEMA,
            "status": self.status.value,
            "precommit_digest": self.precommit_digest,
            "cohort_plan_digest": self.cohort_plan_digest,
            "configuration": {
                **self.configuration.to_data(),
                "record_digest": self.configuration.record_digest,
            },
            "exposure_successor_digest": self.exposure_successor_digest,
            "precommit_receipt": dict(self.precommit_receipt),
            "release_authorization": dict(self.release_authorization),
            "stored_objects": [item.to_data() for item in self.stored_objects],
            "call_terminals": [dict(item) for item in self.call_terminals],
            "call_journal_seal_digest": self.call_journal_seal_digest,
            "released_panels": [item.to_data() for item in self.released_panels],
            "reference_catalog": self.reference_catalog.to_data(),
            "description_artifact": self.description_artifact.to_data(),
            "calibration_plan": (
                None if self.calibration_plan is None else self.calibration_plan.to_data()
            ),
            "calibration_artifacts": [
                item.to_data() for item in self.calibration_artifacts
            ],
            "calibration_runtime_archive_digest": (
                self.calibration_runtime_archive_digest
            ),
            "calibration_assessment": (
                None
                if self.calibration_assessment is None
                else self.calibration_assessment.to_data()
            ),
            "calibration_family": (
                None
                if self.calibration_family is None
                else self.calibration_family.to_data()
            ),
            "predicate_library": (
                None
                if self.predicate_library is None
                else self.predicate_library.to_data()
            ),
            "support_artifacts": [item.to_data() for item in self.support_artifacts],
            "support_panels": [item.to_data() for item in self.support_panels],
            "support_version_space": (
                None
                if self.support_version_space is None
                else self.support_version_space.to_data()
            ),
            "support_runtime_archive_digest": self.support_runtime_archive_digest,
            "query_artifacts": [item.to_data() for item in self.query_artifacts],
            "query_runtime_archive_digest": self.query_runtime_archive_digest,
            "headless_archive": (
                None if self.headless_archive is None else self.headless_archive.to_data()
            ),
            "call_failures": [item.to_data() for item in self.call_failures],
            "phase_trace": list(self.phase_trace),
            "model_calls_made": self.model_calls_made,
            "runtime_authority": {
                "campaign_source_sha256": self.campaign_source_sha256,
                "campaign_algorithm_digest": self.campaign_algorithm_digest,
                "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
                "python_is_canonical_authority": True,
                "lean_required": False,
                "lean_affects_identity_or_decision": False,
                "cold_replay_model_calls": 0,
                "model_calls_made_semantics": (
                    "cumulative-unique-terminal-admissions;"
                    "reused-terminal-makes-no-transport-call"
                ),
            },
        }

    def to_data(self) -> dict[str, object]:
        return {**self.content_dict(), "record_digest": self.record_digest}

    @classmethod
    def from_data(
        cls, value: Mapping[str, Any]
    ) -> "PrototypePairCampaignArtifact":
        result = PrototypePairCampaignCallFailure.decode_campaign_artifact(value)
        if not isinstance(result, cls):
            raise PrototypePairCampaignError("campaign decoder returned wrong type")
        return result


@dataclass(slots=True)
class _CampaignState:
    stored_objects: list[PrototypePairStoredObject] = field(default_factory=list)
    call_terminals: list[Mapping[str, Any]] = field(default_factory=list)
    released_panels: list[ReleasedOfficialPanel] = field(default_factory=list)
    calibration_artifacts: list[PrototypeSceneObserverArtifact] = field(
        default_factory=list
    )
    support_artifacts: list[PrototypeSceneObserverArtifact] = field(
        default_factory=list
    )
    support_panels: list[PrototypeScenePanelEvaluation] = field(default_factory=list)
    query_artifacts: list[PrototypeSceneObserverArtifact] = field(
        default_factory=list
    )
    call_failures: list[PrototypePairCampaignCallFailure] = field(
        default_factory=list
    )
    model_calls_made: int = 0


@dataclass(frozen=True, slots=True)
class _CallTicket:
    claim: object
    fresh: bool
    terminal_outcome: object | None


@dataclass(frozen=True, slots=True)
class _PersistedObserverResult:
    """One completed observer turn durably terminalized off the archive order.

    Model calls may finish in any order.  The canonical campaign state is still
    assembled in the precommitted schedule order, but each completed result is
    written and its call journal entry terminalized immediately on the
    coordinator thread.  Thus a slow low-index turn cannot strand already
    completed higher-index evidence in volatile worker memory.
    """

    artifact: PrototypeSceneObserverArtifact
    stored_objects: tuple[PrototypePairStoredObject, ...]
    call_terminals: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, PrototypeSceneObserverArtifact):
            raise TypeError("persisted observer artifact has wrong type")
        if len(self.stored_objects) != 1 or len(self.call_terminals) != 1:
            raise PrototypePairCampaignError(
                "one fresh observer result must create one object and one terminal"
            )


def _record_json_bytes(value: object, label: str) -> bytes:
    return canonical_json(_to_data(value, label))


def _receipt_digest(value: object, label: str) -> str:
    data = _to_data(value, label)
    digest = data.get("record_digest")
    return _require_address(digest, f"{label} record digest")


def _persist_object(
    state: _CampaignState,
    store: CampaignStore,
    *,
    kind: str,
    data: Mapping[str, Any],
    expected_record_digest: str,
) -> tuple[PrototypePairStoredObject, object]:
    canonical = dict(data)
    canonical_json(canonical)
    receipt = store.persist_canonical_object(
        kind, canonical, expected_record_digest
    )
    loaded = store.load_canonical_object(receipt, expected_record_digest)
    if not isinstance(loaded, Mapping) or dict(loaded) != canonical:
        raise PrototypePairCampaignError(
            f"durable {kind} reload differs from persisted canonical object"
        )
    binding = PrototypePairStoredObject.seal(
        kind=kind,
        object_identity_digest=expected_record_digest,
        storage_receipt=receipt,
    )
    state.stored_objects.append(binding)
    return binding, receipt


def _persist_record(
    state: _CampaignState,
    store: CampaignStore,
    *,
    kind: str,
    value: object,
    expected_record_digest: str,
) -> tuple[PrototypePairStoredObject, object]:
    return _persist_object(
        state,
        store,
        kind=kind,
        data=_to_data(value, kind),
        expected_record_digest=expected_record_digest,
    )


def _claim_call(
    state: _CampaignState,
    store: CampaignStore,
    clock: CampaignClock,
    *,
    authorization: object,
    phase: str,
    subject_id: str,
    context_digest: str,
) -> _CallTicket:
    admission = store.claim_call(
        authorization=authorization,
        phase=phase,
        subject_id=subject_id,
        context_digest=context_digest,
        claimed_at=clock.now(phase, subject_id, "claimed"),
    )
    reason = getattr(admission, "reason", None)
    claim = getattr(admission, "claim", None)
    outcome = getattr(admission, "terminal_outcome", None)
    if reason == "preexisting_nonterminal_claim":
        raise PrototypePairCampaignError(
            "campaign call has a preexisting nonterminal claim; transport rerun is forbidden"
        )
    fresh = (
        getattr(admission, "model_eligible", None) is True
        and reason == "new_exclusive_claim"
        and outcome is None
    )
    reused = (
        getattr(admission, "model_eligible", None) is False
        and reason == "preexisting_terminal_outcome"
        and outcome is not None
    )
    if not fresh and not reused:
        raise PrototypePairCampaignError("campaign call admission is malformed")
    _receipt_digest(claim, "call claim")
    state.model_calls_made += 1
    if reused:
        state.call_terminals.append(_to_data(outcome, "call outcome"))
    return _CallTicket(claim=claim, fresh=fresh, terminal_outcome=outcome)


def _load_reused_call_result(
    state: _CampaignState,
    store: CampaignStore,
    ticket: _CallTicket,
    *,
    kind: str,
) -> Mapping[str, Any]:
    if ticket.fresh or ticket.terminal_outcome is None:
        raise PrototypePairCampaignError("fresh call has no reusable result")
    outcome = ticket.terminal_outcome
    result_digest = getattr(outcome, "result_digest", None)
    result_receipt = getattr(outcome, "result_receipt", None)
    if not isinstance(result_digest, str) or result_receipt is None:
        raise PrototypePairCampaignError("terminal call outcome is malformed")
    data = store.load_canonical_object(result_receipt, result_digest)
    state.stored_objects.append(
        PrototypePairStoredObject.seal(
            kind=kind,
            object_identity_digest=result_digest,
            storage_receipt=result_receipt,
        )
    )
    return data


def _verify_reused_observer_ticket(
    ticket: _CallTicket,
    artifact: PrototypeRubricDescriptionArtifact | PrototypeSceneObserverArtifact,
) -> None:
    if ticket.fresh or ticket.terminal_outcome is None:
        raise PrototypePairCampaignError("observer reuse ticket is not terminal")
    outcome = _to_data(ticket.terminal_outcome, "call outcome")
    if (
        outcome.get("result_digest") != artifact.artifact_digest
        or outcome.get("terminal_status") != _observer_terminal_status(artifact)
    ):
        raise PrototypePairCampaignError(
            "reused observer outcome status or result identity differs"
        )


def _finish_call(
    state: _CampaignState,
    store: CampaignStore,
    clock: CampaignClock,
    *,
    claim: object,
    phase: str,
    subject_id: str,
    terminal_status: str,
    result_digest: str,
    result_receipt: object,
) -> object:
    outcome = store.finish_call(
        claim=claim,
        terminal_status=terminal_status,
        result_digest=result_digest,
        result_receipt=result_receipt,
        finished_at=clock.now(phase, subject_id, "finished"),
    )
    outcome_data = _to_data(outcome, "call outcome")
    if outcome_data.get("result_digest") != result_digest:
        raise PrototypePairCampaignError("call outcome result digest differs")
    state.call_terminals.append(outcome_data)
    return outcome


def _observer_terminal_status(
    artifact: PrototypeRubricDescriptionArtifact | PrototypeSceneObserverArtifact,
) -> str:
    if artifact.status is PrototypeSceneObserverStatus.SUCCESS:
        return "success"
    if artifact.status is PrototypeSceneObserverStatus.PARSER_ERROR:
        return "parser_error"
    if artifact.status is PrototypeSceneObserverStatus.TRANSPORT_ERROR:
        return "transport_error"
    return "error"


def _terminalize_exception(
    state: _CampaignState,
    store: CampaignStore,
    clock: CampaignClock,
    *,
    claim: object,
    phase: str,
    subject_id: str,
    exception: Exception,
) -> PrototypePairCampaignCallFailure:
    failure = PrototypePairCampaignCallFailure.from_exception(
        phase=phase, subject_id=subject_id, exception=exception
    )
    _binding, receipt = _persist_record(
        state,
        store,
        kind="call_failure",
        value=failure,
        expected_record_digest=failure.record_digest,
    )
    _finish_call(
        state,
        store,
        clock,
        claim=claim,
        phase=phase,
        subject_id=subject_id,
        terminal_status="error",
        result_digest=failure.record_digest,
        result_receipt=receipt,
    )
    state.call_failures.append(failure)
    return failure


def _release_and_persist_panel(
    state: _CampaignState,
    store: CampaignStore,
    archive: OfficialPanelArchive,
    panel_id: str,
    *,
    precommit_digest: str,
    exposure_successor_digest: str,
) -> ReleasedOfficialPanel:
    released = ReleasedOfficialPanel.release(
        archive,
        panel_id,
        execution_precommit_digest=precommit_digest,
        exposure_successor_digest=exposure_successor_digest,
        expected_execution_precommit_digest=precommit_digest,
        expected_exposure_successor_digest=exposure_successor_digest,
    )
    _persist_record(
        state,
        store,
        kind="released_panel",
        value=released,
        expected_record_digest=released.record_digest,
    )
    state.released_panels.append(released)
    return released


def _released_by_id(
    values: Sequence[ReleasedOfficialPanel],
) -> dict[str, ReleasedOfficialPanel]:
    result = {item.panel_id: item for item in values}
    if len(result) != len(values):
        raise PrototypePairCampaignError("released panel inventory repeats an ID")
    return result


def _runtime_archive(
    *,
    phase: str,
    configuration: PrototypePairCampaignConfiguration,
    catalog: PrototypeReferenceCatalog,
    description: PrototypeRubricDescriptionArtifact,
    reference_panels: Mapping[str, ReleasedOfficialPanel],
    scene_panels: Sequence[ReleasedOfficialPanel],
    scene_artifacts: Sequence[PrototypeSceneObserverArtifact],
    scene_task_ids: Sequence[str],
    observation_context_digest: str,
    purpose: PrototypeSceneArtifactPurpose,
) -> PrototypeSceneRuntimeArtifactArchive:
    panels = tuple(scene_panels)
    artifacts = tuple(scene_artifacts)
    tasks = tuple(scene_task_ids)
    if not panels or len(panels) != len(artifacts) or len(panels) != len(tasks):
        raise PrototypePairCampaignError("runtime archive scene rows differ")
    catalog_bytes = _record_json_bytes(catalog, "reference catalog")
    description_bytes = _record_json_bytes(description, "description artifact")
    inputs: list[PrototypeSceneRuntimeArtifactInput] = []
    for panel, artifact, task_id in zip(panels, artifacts, tasks, strict=True):
        artifact_bytes = _record_json_bytes(artifact, "observer artifact")
        inputs.append(
            PrototypeSceneRuntimeArtifactInput(
                scene_task_id=task_id,
                panel_id=panel.panel_id,
                expected_observation_context_digest=observation_context_digest,
                exact_scene_png_bytes=panel.exact_png_bytes,
                expected_scene_sha256=panel.exact_png_digest.removeprefix("sha256:"),
                observer_artifact_json_bytes=artifact_bytes,
                expected_observer_artifact_json_sha256=hashlib.sha256(
                    artifact_bytes
                ).hexdigest(),
                expected_observer_artifact_digest=artifact.artifact_digest,
                purpose=purpose,
            )
        )
    reference_ids = tuple(item.source_panel_id for item in catalog.bindings)
    if set(reference_panels) != set(reference_ids):
        raise PrototypePairCampaignError("runtime reference inventory differs")
    archive = PrototypeSceneRuntimeArtifactArchive.seal_external(
        archive_source_id=f"{configuration.runtime_archive_source_id}-{phase}",
        verifier_id=f"{configuration.runtime_verifier_id}-{phase}",
        catalog_json_bytes=catalog_bytes,
        expected_catalog_json_sha256=hashlib.sha256(catalog_bytes).hexdigest(),
        expected_catalog_digest=catalog.catalog_digest,
        rubric_artifact_json_bytes=description_bytes,
        expected_rubric_artifact_json_sha256=hashlib.sha256(
            description_bytes
        ).hexdigest(),
        expected_rubric_artifact_digest=description.artifact_digest,
        prototype_reference_png_by_panel_id={
            panel_id: reference_panels[panel_id].exact_png_bytes
            for panel_id in reference_ids
        },
        expected_reference_sha256={
            panel_id: reference_panels[panel_id].exact_png_digest.removeprefix(
                "sha256:"
            )
            for panel_id in reference_ids
        },
        scenes=inputs,
        same_basic_renderer_population_valid=True,
        conditional_transport_assumption_accepted=True,
        observer_environment_valid=True,
    )
    return archive


def _persist_runtime_archive(
    state: _CampaignState,
    store: CampaignStore,
    archive: PrototypeSceneRuntimeArtifactArchive,
) -> str:
    data = {**archive.commitment_data(), "record_digest": archive.record_digest}
    _binding, receipt = _persist_object(
        state,
        store,
        kind="runtime_archive",
        data=data,
        expected_record_digest=archive.record_digest,
    )
    loaded = store.load_canonical_object(receipt, archive.record_digest)
    if loaded.get("record_digest") != archive.record_digest:
        raise PrototypePairCampaignError("runtime archive durable anchor differs")
    return _require_address(loaded["record_digest"], "runtime archive anchor")


def _verify_campaign_authorities(
    precommit: PrototypePairExecutionPrecommit,
    *,
    configuration: PrototypePairCampaignConfiguration,
    cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot | None,
    model_catalog_snapshot: CodexModelCatalogSnapshot,
    no_tools_attestation: CodexNoToolsAttestation,
    ranker: object,
    observed_codex_cli_version: str,
    observed_codex_launcher_sha256: str,
    observed_python_runtime_id: str,
    observed_python_runtime_identity_digest: str,
) -> None:
    identities = precommit.identities
    if identities.execution_configuration_digest != configuration.record_digest:
        raise PrototypePairCampaignError(
            "execution configuration differs from precommit"
        )
    current_sources = prototype_pair_campaign_runtime_source_digests()
    frozen_sources = dict(identities.runtime_source_digests)
    mismatches = {
        role: (frozen_sources.get(role), digest)
        for role, digest in current_sources.items()
        if frozen_sources.get(role) != digest
    }
    if mismatches:
        raise PrototypePairCampaignError(
            f"runtime source precommit differs from current Python authority: "
            f"{sorted(mismatches)}"
        )
    policy_binding = (
        "absent"
        if cloud_policy_cache_snapshot is None
        else cloud_policy_cache_snapshot.binding
    )
    if (
        not isinstance(model_catalog_snapshot, CodexModelCatalogSnapshot)
        or model_catalog_snapshot is not identities.codex_model_catalog_snapshot
        or not isinstance(no_tools_attestation, CodexNoToolsAttestation)
        or no_tools_attestation is not identities.codex_no_tools_attestation
    ):
        raise PrototypePairCampaignError(
            "Codex preflight objects differ from execution precommit"
        )
    try:
        validate_codex_no_tools_attestation(
            no_tools_attestation,
            expected_launcher_digest=identities.codex_launcher_sha256,
            expected_model_catalog_digest=model_catalog_snapshot.raw_digest,
            expected_cloud_policy_cache_binding=policy_binding,
        )
    except Exception as exc:
        raise PrototypePairCampaignError(
            "Codex no-tools preflight authority differs"
        ) from exc
    expected_observer = {
        "protocol_id": PROTOTYPE_SCENE_OBSERVER_PROTOCOL_ID,
        "description_protocol_digest": (
            prototype_rubric_description_protocol_digest()
        ),
        "scoring_protocol_digest": prototype_scene_scoring_protocol_digest(),
        "model_identity_digest": prototype_scene_observer_model_digest(
            identities.observer_model_id,
            identities.observer_reasoning_effort,
        ),
        "environment_digest": prototype_scene_observer_environment_digest(
            model=identities.observer_model_id,
            reasoning_effort=identities.observer_reasoning_effort,
            expected_launcher_digest=identities.codex_launcher_sha256,
            cloud_policy_cache_binding=policy_binding,
            model_catalog_digest=model_catalog_snapshot.raw_digest,
            no_tools_attestation_digest=no_tools_attestation.attestation_digest,
        ),
    }
    actual_observer = {
        "protocol_id": identities.observer_protocol_id,
        "description_protocol_digest": (
            identities.observer_description_protocol_digest
        ),
        "scoring_protocol_digest": identities.observer_scoring_protocol_digest,
        "model_identity_digest": identities.observer_model_identity_digest,
        "environment_digest": identities.observer_environment_digest,
    }
    if actual_observer != expected_observer:
        raise PrototypePairCampaignError("observer precommit authority differs")
    if (
        identities.calibration_algorithm_digest != calibration_algorithm_digest()
        or identities.cloud_policy_cache_binding != policy_binding
        or identities.runner_protocol_id != RUNNER_ID
        or identities.runner_algorithm_digest
        != prototype_scene_runner_source_digest()
        or identities.codex_cli_version != observed_codex_cli_version
        or identities.codex_launcher_sha256 != observed_codex_launcher_sha256
        or identities.python_runtime_id != observed_python_runtime_id
        or identities.python_runtime_identity_digest
        != observed_python_runtime_identity_digest
    ):
        raise PrototypePairCampaignError(
            "calibration, runner, CLI, or Python runtime precommit differs"
        )
    if (
        getattr(ranker, "model", None) != identities.ranker_model_id
        or getattr(ranker, "reasoning_effort", None)
        != identities.ranker_reasoning_effort
        or getattr(ranker, "minutes", None) != configuration.ranker_minutes
        or getattr(ranker, "verbose", None) is not configuration.ranker_verbose
        or getattr(ranker, "executable", None)
        != configuration.ranker_executable
    ):
        raise PrototypePairCampaignError("ranker request differs from precommit")
    ranker_identity = getattr(ranker, "model_identity_digest", None)
    expected_ranker_transport = prototype_scene_codex_ranker_transport_source_digest()
    if (
        ranker_identity != "sha256:" + identities.ranker_model_identity_digest
        or getattr(ranker, "expected_launcher_digest", None)
        != identities.codex_launcher_sha256
        or getattr(ranker, "expected_cloud_policy_cache_binding", None)
        != policy_binding
        or getattr(ranker, "expected_transport_source_digest", None)
        != expected_ranker_transport
        or getattr(ranker, "model_catalog_snapshot", None)
        is not model_catalog_snapshot
        or getattr(ranker, "no_tools_attestation", None)
        is not no_tools_attestation
        or getattr(ranker, "protocol_digest", None)
        != identities.ranker_protocol_digest
        or identities.ranker_protocol_id
        != PROTOTYPE_SCENE_CODEX_RANKER_PROTOCOL_ID
        or identities.ranker_protocol_digest
        != prototype_scene_codex_ranker_protocol_digest()
        or getattr(ranker, "environment_digest", None)
        != identities.ranker_environment_digest
    ):
        raise PrototypePairCampaignError(
            "ranker protocol, model, transport, or environment inputs differ"
        )
    if configuration.observer_executable == "":
        raise PrototypePairCampaignError("observer executable is empty")


def _make_campaign_artifact(
    state: _CampaignState,
    *,
    status: PrototypePairCampaignStatus,
    precommit: PrototypePairExecutionPrecommit,
    cohort_plan: PrototypePairCohortPlan,
    configuration: PrototypePairCampaignConfiguration,
    exposure_successor_digest: str,
    call_journal_seal_digest: str,
    precommit_receipt: object,
    release_authorization: object,
    reference_catalog: PrototypeReferenceCatalog,
    description_artifact: PrototypeRubricDescriptionArtifact,
    calibration_plan: PrototypeSceneCalibrationPlan | None = None,
    calibration_runtime_archive_digest: str | None = None,
    calibration_assessment: PrototypeSceneCalibrationAssessment | None = None,
    calibration_family: PrototypeSceneCalibrationFamily | None = None,
    predicate_library: PrototypeScenePredicateLibrary | None = None,
    support_version_space: PrototypeSceneSupportVersionSpace | None = None,
    support_runtime_archive_digest: str | None = None,
    query_runtime_archive_digest: str | None = None,
    headless_archive: PrototypeSceneHeadlessArchive | None = None,
    phase_trace: Sequence[str],
) -> PrototypePairCampaignArtifact:
    values: dict[str, object] = {
        "status": status,
        "precommit_digest": precommit.record_digest,
        "cohort_plan_digest": cohort_plan.record_digest,
        "configuration": configuration,
        "exposure_successor_digest": exposure_successor_digest,
        "call_journal_seal_digest": call_journal_seal_digest,
        "precommit_receipt": _to_data(precommit_receipt, "precommit receipt"),
        "release_authorization": _to_data(
            release_authorization, "release authorization"
        ),
        "stored_objects": tuple(state.stored_objects),
        "call_terminals": tuple(state.call_terminals),
        "released_panels": tuple(state.released_panels),
        "reference_catalog": reference_catalog,
        "description_artifact": description_artifact,
        "calibration_plan": calibration_plan,
        "calibration_artifacts": tuple(state.calibration_artifacts),
        "calibration_runtime_archive_digest": (
            calibration_runtime_archive_digest
        ),
        "calibration_assessment": calibration_assessment,
        "calibration_family": calibration_family,
        "predicate_library": predicate_library,
        "support_artifacts": tuple(state.support_artifacts),
        "support_panels": tuple(state.support_panels),
        "support_version_space": support_version_space,
        "support_runtime_archive_digest": support_runtime_archive_digest,
        "query_artifacts": tuple(state.query_artifacts),
        "query_runtime_archive_digest": query_runtime_archive_digest,
        "headless_archive": headless_archive,
        "call_failures": tuple(state.call_failures),
        "phase_trace": tuple(phase_trace),
        "model_calls_made": state.model_calls_made,
        "campaign_source_sha256": CAMPAIGN_SOURCE_SHA256,
        "campaign_algorithm_digest": prototype_pair_campaign_algorithm_digest(),
    }
    provisional = object.__new__(PrototypePairCampaignArtifact)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return PrototypePairCampaignArtifact(
        **values,  # type: ignore[arg-type]
        record_digest=_address(provisional.content_dict()),
    )


def _observe_scene(
    released: ReleasedOfficialPanel,
    *,
    scene_task_id: str,
    observation_context_digest: str,
    catalog: PrototypeReferenceCatalog,
    reference_png_by_panel_id: Mapping[str, bytes],
    description: PrototypeRubricDescriptionArtifact,
    precommit: PrototypePairExecutionPrecommit,
    configuration: PrototypePairCampaignConfiguration,
    cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot | None,
    model_catalog_snapshot: CodexModelCatalogSnapshot,
    no_tools_attestation: CodexNoToolsAttestation,
    scene_transport: Callable[..., object],
) -> PrototypeSceneObserverArtifact:
    identities = precommit.identities
    common = {
        "scene_task_id": scene_task_id,
        "scene_panel_id": released.panel_id,
        "observation_context_digest": observation_context_digest,
        "expected_scene_sha256": released.exact_png_digest.removeprefix("sha256:"),
        "catalog": catalog,
        "prototype_png_by_panel_id": reference_png_by_panel_id,
        "expected_catalog_digest": catalog.catalog_digest,
        "rubric_artifact": description,
        "expected_rubric_artifact_digest": description.artifact_digest,
        "model": identities.observer_model_id,
        "reasoning_effort": identities.observer_reasoning_effort,
        "expected_launcher_digest": identities.codex_launcher_sha256,
        "cloud_policy_cache_snapshot": cloud_policy_cache_snapshot,
        "model_catalog_snapshot": model_catalog_snapshot,
        "no_tools_attestation": no_tools_attestation,
    }
    try:
        return observe_prototype_scene(
            released.exact_png_bytes,
            **common,
            minutes=configuration.observer_minutes,
            verbose=configuration.observer_verbose,
            executable=configuration.observer_executable,
            transport=scene_transport,
        )
    except Exception as exc:
        return seal_prototype_scene_internal_error(
            released.exact_png_bytes,
            **common,
            exception=exc,
        )


def _persist_observer_result(
    state: _CampaignState,
    store: CampaignStore,
    clock: CampaignClock,
    *,
    claim: object,
    phase: str,
    subject_id: str,
    artifact: PrototypeRubricDescriptionArtifact | PrototypeSceneObserverArtifact,
    kind: str,
    precommit: PrototypePairExecutionPrecommit,
) -> None:
    _assert_observer_preflight_binding(artifact, precommit)
    _binding, receipt = _persist_record(
        state,
        store,
        kind=kind,
        value=artifact,
        expected_record_digest=artifact.artifact_digest,
    )
    _finish_call(
        state,
        store,
        clock,
        claim=claim,
        phase=phase,
        subject_id=subject_id,
        terminal_status=_observer_terminal_status(artifact),
        result_digest=artifact.artifact_digest,
        result_receipt=receipt,
    )


def _run_fresh_observer_batch(
    state: _CampaignState,
    store: CampaignStore,
    clock: CampaignClock,
    *,
    fresh_indices: Sequence[int],
    tickets: Sequence[_CallTicket],
    subject_ids: Sequence[str],
    turn: Callable[[int], PrototypeSceneObserverArtifact],
    phase: str,
    kind: str,
    precommit: PrototypePairExecutionPrecommit,
    max_workers: int,
) -> dict[int, PrototypeSceneObserverArtifact]:
    """Run fresh observer turns and persist each one as soon as it completes.

    Persistence happens only on this coordinator thread.  Per-result state
    deltas are staged by schedule index and merged after the batch so campaign
    serialization remains independent of worker completion order.
    """

    indices = tuple(fresh_indices)
    if (
        tuple(sorted(set(indices))) != indices
        or any(
            isinstance(index, bool) or not 0 <= index < len(tickets)
            for index in indices
        )
        or len(subject_ids) != len(tickets)
        or isinstance(max_workers, bool)
        or not isinstance(max_workers, int)
        or max_workers <= 0
    ):
        raise PrototypePairCampaignError("observer batch schedule is malformed")
    persisted: dict[int, _PersistedObserverResult] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures: dict[Future[PrototypeSceneObserverArtifact], int] = {
            executor.submit(turn, index): index for index in indices
        }
        for future in as_completed(futures):
            index = futures[future]
            artifact = future.result()
            detached = _CampaignState()
            _persist_observer_result(
                detached,
                store,
                clock,
                claim=tickets[index].claim,
                phase=phase,
                subject_id=subject_ids[index],
                artifact=artifact,
                kind=kind,
                precommit=precommit,
            )
            if (
                detached.released_panels
                or detached.calibration_artifacts
                or detached.support_artifacts
                or detached.support_panels
                or detached.query_artifacts
                or detached.call_failures
                or detached.model_calls_made != 0
            ):
                raise PrototypePairCampaignError(
                    "detached observer persistence mutated unrelated campaign state"
                )
            persisted[index] = _PersistedObserverResult(
                artifact=artifact,
                stored_objects=tuple(detached.stored_objects),
                call_terminals=tuple(detached.call_terminals),
            )
    if set(persisted) != set(indices):
        raise PrototypePairCampaignError("observer batch omitted a completed result")
    for index in indices:
        row = persisted[index]
        state.stored_objects.extend(row.stored_objects)
        state.call_terminals.extend(row.call_terminals)
    return {index: persisted[index].artifact for index in indices}


def _assert_observer_preflight_binding(
    artifact: PrototypeRubricDescriptionArtifact | PrototypeSceneObserverArtifact,
    precommit: PrototypePairExecutionPrecommit,
) -> None:
    identities = precommit.identities
    if (
        artifact.expected_launcher_digest != identities.codex_launcher_sha256
        or artifact.cloud_policy_cache_binding
        != identities.cloud_policy_cache_binding
        or artifact.model_catalog_digest
        != identities.codex_model_catalog_snapshot.raw_digest
        or artifact.no_tools_attestation_digest
        != identities.codex_no_tools_attestation.attestation_digest
        or artifact.environment_digest != identities.observer_environment_digest
    ):
        raise PrototypePairCampaignError(
            "observer result preflight binding differs from precommit"
        )


def run_prototype_pair_campaign(
    *,
    cohort_plan: PrototypePairCohortPlan,
    precommit: PrototypePairExecutionPrecommit,
    exposure_predecessor: object,
    release_descriptor: OfficialReleaseDescriptor,
    official_archive: OfficialPanelArchive,
    store: CampaignStore,
    clock: CampaignClock,
    configuration: PrototypePairCampaignConfiguration,
    cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot | None,
    model_catalog_snapshot: CodexModelCatalogSnapshot,
    no_tools_attestation: CodexNoToolsAttestation,
    description_transport: Callable[..., object],
    scene_transport: Callable[..., object],
    ranker: Callable[[tuple[str, ...], str], object],
    observed_codex_cli_version: str,
    observed_codex_launcher_sha256: str,
    observed_python_runtime_id: str,
    observed_python_runtime_identity_digest: str,
    expected_precommit_digest: str,
    expected_cohort_plan_digest: str,
    expected_identity_bundle_digest: str,
    expected_exposure_predecessor_digest: str,
) -> PrototypePairCampaignArtifact:
    """Execute the frozen 44-call campaign or a typed earlier terminal branch."""

    if not isinstance(cohort_plan, PrototypePairCohortPlan):
        raise TypeError("cohort_plan must be PrototypePairCohortPlan")
    if not isinstance(precommit, PrototypePairExecutionPrecommit):
        raise TypeError("precommit must be PrototypePairExecutionPrecommit")
    if not isinstance(release_descriptor, OfficialReleaseDescriptor):
        raise TypeError("release_descriptor must be OfficialReleaseDescriptor")
    if not isinstance(official_archive, OfficialPanelArchive):
        raise TypeError("official_archive must be OfficialPanelArchive")
    if not isinstance(configuration, PrototypePairCampaignConfiguration):
        raise TypeError("configuration must be PrototypePairCampaignConfiguration")
    if not isinstance(model_catalog_snapshot, CodexModelCatalogSnapshot):
        raise TypeError("model_catalog_snapshot must be CodexModelCatalogSnapshot")
    if not isinstance(no_tools_attestation, CodexNoToolsAttestation):
        raise TypeError("no_tools_attestation must be CodexNoToolsAttestation")
    if not callable(description_transport) or not callable(scene_transport):
        raise TypeError("observer transports must be callable")
    if not callable(ranker):
        raise TypeError("ranker must be callable")
    verify_prototype_pair_execution_precommit(
        precommit,
        cohort_plan=cohort_plan,
        identities=precommit.identities,
        expected_precommit_digest=expected_precommit_digest,
        expected_cohort_plan_digest=expected_cohort_plan_digest,
        expected_identity_bundle_digest=expected_identity_bundle_digest,
        expected_exposure_predecessor_digest=(
            expected_exposure_predecessor_digest
        ),
    )
    _verify_campaign_authorities(
        precommit,
        configuration=configuration,
        cloud_policy_cache_snapshot=cloud_policy_cache_snapshot,
        model_catalog_snapshot=model_catalog_snapshot,
        no_tools_attestation=no_tools_attestation,
        ranker=ranker,
        observed_codex_cli_version=observed_codex_cli_version,
        observed_codex_launcher_sha256=observed_codex_launcher_sha256,
        observed_python_runtime_id=observed_python_runtime_id,
        observed_python_runtime_identity_digest=(
            observed_python_runtime_identity_digest
        ),
    )
    if (
        release_descriptor.digest != cohort_plan.release_descriptor_digest
        or official_archive.release_descriptor_digest
        != cohort_plan.release_descriptor_digest
        or official_archive.archive_digest != release_descriptor.archive_sha256
    ):
        raise PrototypePairCampaignError("official release authority differs")

    state = _CampaignState()
    precommit_bytes = canonical_json(precommit.to_data()) + b"\n"
    precommit_receipt = store.persist_execution_precommit(
        precommit_bytes, expected_precommit_digest
    )
    if store.verify_execution_precommit(
        precommit_receipt, expected_precommit_digest, precommit_bytes
    ) != precommit_bytes:
        raise PrototypePairCampaignError("execution precommit durable reload differs")
    authorization = store.authorize_release(
        cohort_plan,
        exposure_predecessor,
        precommit_receipt,
        expected_plan_digest=expected_cohort_plan_digest,
        expected_execution_precommit_digest=expected_precommit_digest,
        expected_exposure_predecessor_digest=(
            expected_exposure_predecessor_digest
        ),
        actor=configuration.actor,
        observed_at=clock.now(
            "exposure_successor_persisted", cohort_plan.drill.task_id, "authorized"
        ),
    )
    authorization_data = _to_data(authorization, "release authorization")
    if (
        authorization_data.get("execution_precommit_digest")
        != precommit.record_digest
        or authorization_data.get("actor") != configuration.actor
    ):
        raise PrototypePairCampaignError(
            "release authorization configuration parent differs"
        )
    exposure_successor_digest = _require_address(
        authorization_data.get("exposure_successor_digest"),
        "exposure successor digest",
    )

    def seal_current_journal() -> str:
        terminal_keys = tuple(
            sorted(
                _require_address(
                    terminal.get("key_digest"), "campaign terminal key digest"
                )
                for terminal in state.call_terminals
            )
        )
        seal = store.seal_call_journal(
            authorization_data["record_digest"],
            expected_terminal_key_digests=terminal_keys,
            sealed_at=clock.now(
                "model_free_tamper_detecting_replay",
                cohort_plan.drill.task_id,
                "call-journal-sealed",
            ),
        )
        seal_data = _to_data(seal, "call journal seal")
        if (
            seal_data.get("authorization_digest")
            != authorization_data["record_digest"]
            or seal_data.get("terminal_key_count") != len(terminal_keys)
        ):
            raise PrototypePairCampaignError("call journal seal differs")
        return _require_address(
            seal_data.get("record_digest"), "call journal seal digest"
        )

    def finalize_campaign(
        candidate: PrototypePairCampaignArtifact,
    ) -> PrototypePairCampaignArtifact:
        receipt = store.persist_canonical_object(
            "campaign_artifact", candidate.to_data(), candidate.record_digest
        )
        loaded = store.load_canonical_object(receipt, candidate.record_digest)
        reloaded = PrototypePairCampaignArtifact.from_data(loaded)
        if reloaded != candidate:
            raise PrototypePairCampaignError("campaign artifact durable reload differs")
        return cold_replay_prototype_pair_campaign(
            reloaded,
            cohort_plan=cohort_plan,
            precommit=precommit,
            release_descriptor=release_descriptor,
            official_archive_path=official_archive.archive_path,
            store=store,
            expected_campaign_digest=candidate.record_digest,
            expected_precommit_digest=expected_precommit_digest,
            expected_cohort_plan_digest=expected_cohort_plan_digest,
            expected_identity_bundle_digest=expected_identity_bundle_digest,
            expected_exposure_predecessor_digest=(
                expected_exposure_predecessor_digest
            ),
        )
    _persist_record(
        state,
        store,
        kind="cohort_plan",
        value=cohort_plan,
        expected_record_digest=cohort_plan.record_digest,
    )
    _persist_record(
        state,
        store,
        kind="campaign_config",
        value={
            "schema": "gkm.bongard-prototype-pair-campaign-config.v1",
            **configuration.to_data(),
            "record_digest": configuration.record_digest,
        },
        expected_record_digest=configuration.record_digest,
    )

    reference_ids = tuple(
        panel_id
        for prototype in cohort_plan.prototypes
        for panel_id in prototype.panel_ids
    )
    reference_releases = tuple(
        _release_and_persist_panel(
            state,
            store,
            official_archive,
            panel_id,
            precommit_digest=precommit.record_digest,
            exposure_successor_digest=exposure_successor_digest,
        )
        for panel_id in reference_ids
    )
    reference_panels = _released_by_id(reference_releases)
    reference_png = {
        panel_id: reference_panels[panel_id].exact_png_bytes
        for panel_id in reference_ids
    }
    reference_sha = {
        panel_id: reference_panels[panel_id].exact_png_digest.removeprefix("sha256:")
        for panel_id in reference_ids
    }
    catalog = build_prototype_reference_catalog(
        cohort_plan,
        reference_png,
        expected_plan_digest=cohort_plan.record_digest,
        expected_reference_sha256=reference_sha,
    )
    _persist_record(
        state,
        store,
        kind="reference_catalog",
        value=catalog,
        expected_record_digest=catalog.catalog_digest,
    )

    description_phase = "prototype_description_observed"
    description_subject = "prototype_description"
    description_ticket = _claim_call(
        state,
        store,
        clock,
        authorization=authorization,
        phase=description_phase,
        subject_id=description_subject,
        context_digest="sha256:" + catalog.catalog_digest,
    )
    if description_ticket.fresh:
        try:
            description = describe_prototype_references(
                catalog,
                reference_png,
                expected_catalog_digest=catalog.catalog_digest,
                model=precommit.identities.observer_model_id,
                reasoning_effort=precommit.identities.observer_reasoning_effort,
                minutes=configuration.observer_minutes,
                verbose=configuration.observer_verbose,
                executable=configuration.observer_executable,
                cloud_policy_cache_snapshot=cloud_policy_cache_snapshot,
                model_catalog_snapshot=model_catalog_snapshot,
                no_tools_attestation=no_tools_attestation,
                expected_launcher_digest=precommit.identities.codex_launcher_sha256,
                transport=description_transport,
            )
        except Exception as exc:
            description = seal_prototype_rubric_description_internal_error(
                catalog,
                reference_png,
                expected_catalog_digest=catalog.catalog_digest,
                model=precommit.identities.observer_model_id,
                reasoning_effort=precommit.identities.observer_reasoning_effort,
                expected_launcher_digest=precommit.identities.codex_launcher_sha256,
                cloud_policy_cache_snapshot=cloud_policy_cache_snapshot,
                model_catalog_snapshot=model_catalog_snapshot,
                no_tools_attestation=no_tools_attestation,
                exception=exc,
            )
        _persist_observer_result(
            state,
            store,
            clock,
            claim=description_ticket.claim,
            phase=description_phase,
            subject_id=description_subject,
            artifact=description,
            kind="description_artifact",
            precommit=precommit,
        )
    else:
        description = PrototypeRubricDescriptionArtifact.from_data(
            _load_reused_call_result(
                state,
                store,
                description_ticket,
                kind="description_artifact",
            )
        )
        _verify_reused_observer_ticket(description_ticket, description)
        _assert_observer_preflight_binding(description, precommit)
    verify_prototype_rubric_description_artifact(
        description,
        catalog,
        reference_png,
        expected_catalog_digest=catalog.catalog_digest,
        expected_artifact_digest=description.artifact_digest,
    )
    if description.status is not PrototypeSceneObserverStatus.SUCCESS:
        candidate = _make_campaign_artifact(
            state,
            status=PrototypePairCampaignStatus.DESCRIPTION_GAP,
            precommit=precommit,
            cohort_plan=cohort_plan,
            configuration=configuration,
            exposure_successor_digest=exposure_successor_digest,
            call_journal_seal_digest=seal_current_journal(),
            precommit_receipt=precommit_receipt,
            release_authorization=authorization,
            reference_catalog=catalog,
            description_artifact=description,
            phase_trace=(*PHASE_ORDER[:4], PHASE_ORDER[-1]),
        )
        return finalize_campaign(candidate)

    description_address = "sha256:" + description.artifact_digest
    catalog_address = "sha256:" + catalog.catalog_digest
    scoring_protocol_address = (
        "sha256:" + precommit.identities.observer_scoring_protocol_digest
    )
    observer_model_address = (
        "sha256:" + precommit.identities.observer_model_identity_digest
    )
    observer_environment_address = (
        "sha256:" + precommit.identities.observer_environment_digest
    )
    calibration_plan = create_prototype_scene_calibration_plan(
        cohort_plan=cohort_plan,
        thresholds=precommit.identities.thresholds,
        description_catalog_digest=description_address,
        prototype_reference_digest=catalog_address,
        observer_protocol_id=precommit.identities.observer_protocol_id,
        observer_protocol_digest=scoring_protocol_address,
        model_id=precommit.identities.observer_model_id,
        model_identity_digest=observer_model_address,
        environment_digest=observer_environment_address,
        expected_cohort_plan_digest=cohort_plan.record_digest,
        expected_threshold_commitment=precommit.identities.threshold_commitment,
        expected_description_catalog_digest=description_address,
        expected_prototype_reference_digest=catalog_address,
        expected_observer_protocol_digest=scoring_protocol_address,
        expected_model_identity_digest=observer_model_address,
        expected_environment_digest=observer_environment_address,
    )
    verify_prototype_scene_calibration_plan(
        calibration_plan,
        cohort_plan=cohort_plan,
        expected_calibration_plan_digest=calibration_plan.record_digest,
        expected_cohort_plan_digest=cohort_plan.record_digest,
    )
    _persist_record(
        state,
        store,
        kind="calibration_plan",
        value=calibration_plan,
        expected_record_digest=calibration_plan.record_digest,
    )

    calibration_releases = tuple(
        _release_and_persist_panel(
            state,
            store,
            official_archive,
            scheduled.panel_id,
            precommit_digest=precommit.record_digest,
            exposure_successor_digest=exposure_successor_digest,
        )
        for scheduled in calibration_plan.scenes
    )
    calibration_phase = "twenty_eight_calibration_scenes_released_and_observed"
    calibration_tickets = tuple(
        _claim_call(
            state,
            store,
            clock,
            authorization=authorization,
            phase=calibration_phase,
            subject_id=scheduled.panel_id,
            context_digest=calibration_plan.record_digest,
        )
        for scheduled in calibration_plan.scenes
    )

    def calibration_turn(index: int) -> PrototypeSceneObserverArtifact:
        scheduled = calibration_plan.scenes[index]
        return _observe_scene(
            calibration_releases[index],
            scene_task_id=scheduled.task_id,
            observation_context_digest=calibration_plan.record_digest,
            catalog=catalog,
            reference_png_by_panel_id=reference_png,
            description=description,
            precommit=precommit,
            configuration=configuration,
            cloud_policy_cache_snapshot=cloud_policy_cache_snapshot,
            model_catalog_snapshot=model_catalog_snapshot,
            no_tools_attestation=no_tools_attestation,
            scene_transport=scene_transport,
        )

    fresh_indices = tuple(
        index for index, ticket in enumerate(calibration_tickets) if ticket.fresh
    )
    fresh_results = _run_fresh_observer_batch(
        state,
        store,
        clock,
        fresh_indices=fresh_indices,
        tickets=calibration_tickets,
        subject_ids=tuple(item.panel_id for item in calibration_plan.scenes),
        turn=calibration_turn,
        phase=calibration_phase,
        kind="observer_artifact",
        precommit=precommit,
        max_workers=configuration.parallel_workers,
    )
    for index, (scheduled, ticket) in enumerate(
        zip(calibration_plan.scenes, calibration_tickets, strict=True)
    ):
        subject_id = scheduled.panel_id
        if ticket.fresh:
            artifact = fresh_results[index]
        else:
            artifact = PrototypeSceneObserverArtifact.from_data(
                _load_reused_call_result(
                    state,
                    store,
                    ticket,
                    kind="observer_artifact",
                )
            )
            _verify_reused_observer_ticket(ticket, artifact)
            _assert_observer_preflight_binding(artifact, precommit)
        state.calibration_artifacts.append(artifact)

    calibration_archive = _runtime_archive(
        phase="calibration",
        configuration=configuration,
        catalog=catalog,
        description=description,
        reference_panels=reference_panels,
        scene_panels=calibration_releases,
        scene_artifacts=state.calibration_artifacts,
        scene_task_ids=tuple(item.task_id for item in calibration_plan.scenes),
        observation_context_digest=calibration_plan.record_digest,
        purpose=PrototypeSceneArtifactPurpose.CALIBRATION,
    )
    calibration_archive_digest = _persist_runtime_archive(
        state, store, calibration_archive
    )
    calibration_observations = materialize_prototype_scene_calibration_observations(
        calibration_archive,
        calibration_plan,
        expected_archive_digest=calibration_archive_digest,
    )
    assessment = assess_prototype_scene_calibration(
        calibration_plan,
        calibration_observations,
        expected_calibration_plan_digest=calibration_plan.record_digest,
    )
    _persist_record(
        state,
        store,
        kind="calibration_assessment",
        value=assessment,
        expected_record_digest=assessment.record_digest,
    )
    if not assessment.all_four_bounds_accepted:
        candidate = _make_campaign_artifact(
            state,
            status=PrototypePairCampaignStatus.CALIBRATION_GAP,
            precommit=precommit,
            cohort_plan=cohort_plan,
            configuration=configuration,
            exposure_successor_digest=exposure_successor_digest,
            call_journal_seal_digest=seal_current_journal(),
            precommit_receipt=precommit_receipt,
            release_authorization=authorization,
            reference_catalog=catalog,
            description_artifact=description,
            calibration_plan=calibration_plan,
            calibration_runtime_archive_digest=calibration_archive_digest,
            calibration_assessment=assessment,
            phase_trace=(*PHASE_ORDER[:6], PHASE_ORDER[-1]),
        )
        return finalize_campaign(candidate)

    family = fit_prototype_scene_calibration_family(
        calibration_plan,
        calibration_observations,
        expected_calibration_plan_digest=calibration_plan.record_digest,
    )
    verify_prototype_scene_calibration_family(
        family,
        calibration_plan=calibration_plan,
        cohort_plan=cohort_plan,
        observations=calibration_observations,
        expected_family_digest=family.record_digest,
        expected_calibration_plan_digest=calibration_plan.record_digest,
        expected_cohort_plan_digest=cohort_plan.record_digest,
    )
    _persist_record(
        state,
        store,
        kind="calibration_family",
        value=family,
        expected_record_digest=family.record_digest,
    )
    library = PrototypeScenePredicateLibrary.freeze(family)
    _persist_record(
        state,
        store,
        kind="predicate_library",
        value=library,
        expected_record_digest=library.record_digest,
    )
    context = PrototypeSceneEvaluationContext(
        cohort_plan_digest=family.cohort_plan_digest,
        description_catalog_digest=family.description_catalog_digest,
        prototype_reference_digest=family.prototype_reference_digest,
        observer_protocol_id=family.observer_protocol_id,
        observer_protocol_digest=family.observer_protocol_digest,
        model_id=family.model_id,
        model_identity_digest=family.model_identity_digest,
        environment_digest=family.environment_digest,
        same_basic_renderer_population_valid=True,
        conditional_transport_assumption_accepted=True,
        observer_environment_valid=True,
    )
    support_context_digest = prototype_scene_evaluation_context_digest(context)
    support_releases = tuple(
        _release_and_persist_panel(
            state,
            store,
            official_archive,
            role.source_panel_id,
            precommit_digest=precommit.record_digest,
            exposure_successor_digest=exposure_successor_digest,
        )
        for role in precommit.support_roles
    )
    support_phase = "twelve_support_scenes_released_and_observed"
    support_tickets = tuple(
        _claim_call(
            state,
            store,
            clock,
            authorization=authorization,
            phase=support_phase,
            subject_id=role.source_panel_id,
            context_digest=support_context_digest,
        )
        for role in precommit.support_roles
    )

    def support_turn(index: int) -> PrototypeSceneObserverArtifact:
        return _observe_scene(
            support_releases[index],
            scene_task_id=precommit.drill_task_id,
            observation_context_digest=support_context_digest,
            catalog=catalog,
            reference_png_by_panel_id=reference_png,
            description=description,
            precommit=precommit,
            configuration=configuration,
            cloud_policy_cache_snapshot=cloud_policy_cache_snapshot,
            model_catalog_snapshot=model_catalog_snapshot,
            no_tools_attestation=no_tools_attestation,
            scene_transport=scene_transport,
        )

    support_fresh = tuple(
        index for index, ticket in enumerate(support_tickets) if ticket.fresh
    )
    support_results = _run_fresh_observer_batch(
        state,
        store,
        clock,
        fresh_indices=support_fresh,
        tickets=support_tickets,
        subject_ids=tuple(item.source_panel_id for item in precommit.support_roles),
        turn=support_turn,
        phase=support_phase,
        kind="observer_artifact",
        precommit=precommit,
        max_workers=configuration.parallel_workers,
    )
    for index, (role, ticket) in enumerate(
        zip(precommit.support_roles, support_tickets, strict=True)
    ):
        if ticket.fresh:
            artifact = support_results[index]
        else:
            artifact = PrototypeSceneObserverArtifact.from_data(
                _load_reused_call_result(
                    state, store, ticket, kind="observer_artifact"
                )
            )
            _verify_reused_observer_ticket(ticket, artifact)
            _assert_observer_preflight_binding(artifact, precommit)
        state.support_artifacts.append(artifact)

    support_archive = _runtime_archive(
        phase="support",
        configuration=configuration,
        catalog=catalog,
        description=description,
        reference_panels=reference_panels,
        scene_panels=support_releases,
        scene_artifacts=state.support_artifacts,
        scene_task_ids=(precommit.drill_task_id,) * 12,
        observation_context_digest=support_context_digest,
        purpose=PrototypeSceneArtifactPurpose.RUNTIME_EVALUATION,
    )
    support_archive_digest = _persist_runtime_archive(state, store, support_archive)
    state.support_panels.extend(
        materialize_prototype_scene_panel(
            support_archive,
            family,
            role.source_panel_id,
            expected_archive_digest=support_archive_digest,
        )
        for role in precommit.support_roles
    )
    version = build_prototype_scene_support_version_space(
        library,
        family,
        state.support_panels[:6],
        state.support_panels[6:],
    )
    _persist_record(
        state,
        store,
        kind="support_version_space",
        value=version,
        expected_record_digest=version.record_digest,
    )
    phased_verifier = PrototypeScenePhasedArtifactVerifier.for_support(
        support_archive,
        expected_support_archive_digest=support_archive_digest,
        family=family,
        support_panels=state.support_panels,
    )

    class _RankerCallbackFailure(RuntimeError):
        pass

    rank_phase = "headless_codex_candidate_ranked"

    def ranked_response(
        survivor_ids: tuple[str, ...], rank_input_digest: str
    ) -> PrototypeSceneRankResponse:
        ticket = _claim_call(
            state,
            store,
            clock,
            authorization=authorization,
            phase=rank_phase,
            subject_id=precommit.drill_task_id,
            context_digest=rank_input_digest,
        )
        if ticket.fresh:
            try:
                raw = ranker(survivor_ids, rank_input_digest)
                response = (
                    raw
                    if isinstance(raw, PrototypeSceneRankResponse)
                    else PrototypeSceneRankResponse.from_data(raw)  # type: ignore[arg-type]
                )
                if (
                    response.ranker_protocol_id
                    != PROTOTYPE_SCENE_CODEX_RANKER_PROTOCOL_ID
                    or response.ranker_protocol_digest
                    != prototype_scene_codex_ranker_protocol_digest()
                    or response.model_id != precommit.identities.ranker_model_id
                    or response.model_identity_digest
                    != "sha256:"
                    + precommit.identities.ranker_model_identity_digest
                    or response.environment_digest
                    != precommit.identities.ranker_environment_digest
                ):
                    raise PrototypePairCampaignError(
                        "rank response authority differs from precommit"
                    )
                response.assert_matches(
                    expected_input_digest=rank_input_digest,
                    survivor_candidate_ids=survivor_ids,
                )
                verifier = getattr(ranker, "verify_response", None)
                if callable(verifier):
                    verifier(
                        response,
                        survivor_candidate_ids=survivor_ids,
                        rank_input_digest=rank_input_digest,
                        expected_response_digest=response.record_digest,
                    )
                _binding, receipt = _persist_record(
                    state,
                    store,
                    kind="rank_response",
                    value=response,
                    expected_record_digest=response.record_digest,
                )
                _finish_call(
                    state,
                    store,
                    clock,
                    claim=ticket.claim,
                    phase=rank_phase,
                    subject_id=precommit.drill_task_id,
                    terminal_status="success",
                    result_digest=response.record_digest,
                    result_receipt=receipt,
                )
                return response
            except Exception as exc:
                _terminalize_exception(
                    state,
                    store,
                    clock,
                    claim=ticket.claim,
                    phase=rank_phase,
                    subject_id=precommit.drill_task_id,
                    exception=exc,
                )
                raise _RankerCallbackFailure("ranker callback failed") from exc
        outcome = _to_data(ticket.terminal_outcome, "rank call outcome")
        result = _load_reused_call_result(
            state, store, ticket, kind=(
                "rank_response"
                if outcome.get("terminal_status") == "success"
                else "call_failure"
            )
        )
        if outcome.get("terminal_status") != "success":
            failure = PrototypePairCampaignCallFailure.from_data(result)
            if outcome.get("result_digest") != failure.record_digest:
                raise PrototypePairCampaignError("reused rank failure differs")
            state.call_failures.append(failure)
            raise _RankerCallbackFailure("reused terminal ranker failure")
        response = PrototypeSceneRankResponse.from_data(result)
        if outcome.get("result_digest") != response.record_digest:
            raise PrototypePairCampaignError("reused rank response differs")
        response.assert_matches(
            expected_input_digest=rank_input_digest,
            survivor_candidate_ids=survivor_ids,
        )
        verifier = getattr(ranker, "verify_response", None)
        if callable(verifier):
            verifier(
                response,
                survivor_candidate_ids=survivor_ids,
                rank_input_digest=rank_input_digest,
                expected_response_digest=response.record_digest,
            )
        return response

    durable_freeze: PrototypeSceneCandidateFreeze | None = None
    durable_commit: PrototypeSceneFreezeCommitReceipt | None = None
    freeze_commit_anchor: str | None = None

    def freeze_committer(freeze_bytes: bytes) -> PrototypeSceneFreezeCommitReceipt:
        nonlocal durable_freeze, durable_commit, freeze_commit_anchor
        try:
            freeze_data = json.loads(freeze_bytes.decode("utf-8", errors="strict"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise PrototypePairCampaignError("candidate freeze bytes are invalid") from exc
        freeze = PrototypeSceneCandidateFreeze.from_data(freeze_data)
        commit = store.persist_candidate_freeze(
            freeze_bytes, freeze.record_digest
        )
        commit = (
            commit
            if isinstance(commit, PrototypeSceneFreezeCommitReceipt)
            else PrototypeSceneFreezeCommitReceipt.from_data(commit)
        )
        reloaded, freeze_receipt, commit_receipt = (
            store.load_candidate_freeze_commit(commit.record_digest)
        )
        reloaded = (
            reloaded
            if isinstance(reloaded, PrototypeSceneFreezeCommitReceipt)
            else PrototypeSceneFreezeCommitReceipt.from_data(reloaded)
        )
        if reloaded != commit:
            raise PrototypePairCampaignError("candidate freeze commit reload differs")
        freeze_loaded = store.load_canonical_object(
            freeze_receipt, freeze.record_digest
        )
        commit_loaded = store.load_canonical_object(
            commit_receipt, commit.record_digest
        )
        durable_freeze = PrototypeSceneCandidateFreeze.from_data(freeze_loaded)
        durable_commit = PrototypeSceneFreezeCommitReceipt.from_data(commit_loaded)
        durable_commit.assert_matches(durable_freeze, freeze_bytes)
        state.stored_objects.extend(
            (
                PrototypePairStoredObject.seal(
                    kind="candidate_freeze",
                    object_identity_digest=durable_freeze.record_digest,
                    storage_receipt=freeze_receipt,
                ),
                PrototypePairStoredObject.seal(
                    kind="candidate_freeze_commit",
                    object_identity_digest=durable_commit.record_digest,
                    storage_receipt=commit_receipt,
                ),
            )
        )
        freeze_commit_anchor = durable_commit.record_digest
        return durable_commit

    query_archive_digest: str | None = None

    def query_source(
        freeze_data: Mapping[str, object],
    ) -> Mapping[str, PrototypeScenePanelEvaluation]:
        nonlocal query_archive_digest
        if durable_freeze is None or durable_commit is None or freeze_commit_anchor is None:
            raise PrototypePairCampaignError(
                "query source opened before durable candidate freeze reload"
            )
        if dict(freeze_data) != durable_freeze.to_data():
            raise PrototypePairCampaignError("query source freeze differs")
        query_releases = tuple(
            _release_and_persist_panel(
                state,
                store,
                official_archive,
                role.source_panel_id,
                precommit_digest=precommit.record_digest,
                exposure_successor_digest=exposure_successor_digest,
            )
            for role in precommit.query_roles
        )
        query_phase = "two_query_scenes_released_and_observed"
        tickets = tuple(
            _claim_call(
                state,
                store,
                clock,
                authorization=authorization,
                phase=query_phase,
                subject_id=role.source_panel_id,
                context_digest=_address(
                    {
                        "schema": (
                            "gkm.bongard-prototype-pair-query-call-context.v1"
                        ),
                        "evaluation_context_digest": support_context_digest,
                        "freeze_digest": durable_freeze.record_digest,
                        "freeze_commit_digest": durable_commit.record_digest,
                        "role_id": role.role_id,
                        "source_panel_id": role.source_panel_id,
                    }
                ),
            )
            for role in precommit.query_roles
        )
        for role, released, ticket in zip(
            precommit.query_roles, query_releases, tickets, strict=True
        ):
            if ticket.fresh:
                artifact = _observe_scene(
                    released,
                    scene_task_id=precommit.drill_task_id,
                    observation_context_digest=support_context_digest,
                    catalog=catalog,
                    reference_png_by_panel_id=reference_png,
                    description=description,
                    precommit=precommit,
                    configuration=configuration,
                    cloud_policy_cache_snapshot=cloud_policy_cache_snapshot,
                    model_catalog_snapshot=model_catalog_snapshot,
                    no_tools_attestation=no_tools_attestation,
                    scene_transport=scene_transport,
                )
                _persist_observer_result(
                    state,
                    store,
                    clock,
                    claim=ticket.claim,
                    phase=query_phase,
                    subject_id=role.source_panel_id,
                    artifact=artifact,
                    kind="observer_artifact",
                    precommit=precommit,
                )
            else:
                artifact = PrototypeSceneObserverArtifact.from_data(
                    _load_reused_call_result(
                        state, store, ticket, kind="observer_artifact"
                    )
                )
                _verify_reused_observer_ticket(ticket, artifact)
                _assert_observer_preflight_binding(artifact, precommit)
            state.query_artifacts.append(artifact)
        query_archive = _runtime_archive(
            phase="query",
            configuration=configuration,
            catalog=catalog,
            description=description,
            reference_panels=reference_panels,
            scene_panels=query_releases,
            scene_artifacts=state.query_artifacts,
            scene_task_ids=(precommit.drill_task_id,) * 2,
            observation_context_digest=support_context_digest,
            purpose=PrototypeSceneArtifactPurpose.RUNTIME_EVALUATION,
        )
        query_archive_digest = _persist_runtime_archive(
            state, store, query_archive
        )
        query_panels = tuple(
            materialize_prototype_scene_panel(
                query_archive,
                family,
                role.source_panel_id,
                expected_archive_digest=query_archive_digest,
            )
            for role in precommit.query_roles
        )
        phased_verifier.attach_query_archive_after_freeze(
            query_archive,
            expected_query_archive_digest=query_archive_digest,
            freeze=durable_freeze,
            freeze_commit=durable_commit,
            expected_freeze_commit_digest=freeze_commit_anchor,
        )
        by_side = {
            role.opaque_side_id: panel
            for role, panel in zip(precommit.query_roles, query_panels, strict=True)
        }
        if set(by_side) != {"side_0", "side_1"}:
            raise PrototypePairCampaignError("query opaque sides differ")
        return {"positive": by_side["side_0"], "negative": by_side["side_1"]}

    try:
        headless = run_prototype_scene_headless(
            family,
            library,
            state.support_panels[:6],
            state.support_panels[6:],
            artifact_verifier=phased_verifier,
            ranker=ranked_response,
            freeze_committer=freeze_committer,
            query_source=query_source,
        )
    except _RankerCallbackFailure:
        candidate = _make_campaign_artifact(
            state,
            status=PrototypePairCampaignStatus.RANKER_ERROR,
            precommit=precommit,
            cohort_plan=cohort_plan,
            configuration=configuration,
            exposure_successor_digest=exposure_successor_digest,
            call_journal_seal_digest=seal_current_journal(),
            precommit_receipt=precommit_receipt,
            release_authorization=authorization,
            reference_catalog=catalog,
            description_artifact=description,
            calibration_plan=calibration_plan,
            calibration_runtime_archive_digest=calibration_archive_digest,
            calibration_assessment=assessment,
            calibration_family=family,
            predicate_library=library,
            support_version_space=version,
            support_runtime_archive_digest=support_archive_digest,
            phase_trace=(*PHASE_ORDER[:10], PHASE_ORDER[-1]),
        )
        return finalize_campaign(candidate)
    if headless.version_space != version:
        raise PrototypePairCampaignError("runner version space differs")
    _persist_record(
        state,
        store,
        kind="headless_archive",
        value=headless,
        expected_record_digest=headless.record_digest,
    )
    if headless.status is not PrototypeSceneHeadlessStatus.COMPLETE:
        campaign_status = (
            PrototypePairCampaignStatus.SUPPORT_LANGUAGE_GAP
            if headless.status is PrototypeSceneHeadlessStatus.LANGUAGE_GAP
            else PrototypePairCampaignStatus.SUPPORT_WITNESS_GAP
        )
        candidate = _make_campaign_artifact(
            state,
            status=campaign_status,
            precommit=precommit,
            cohort_plan=cohort_plan,
            configuration=configuration,
            exposure_successor_digest=exposure_successor_digest,
            call_journal_seal_digest=seal_current_journal(),
            precommit_receipt=precommit_receipt,
            release_authorization=authorization,
            reference_catalog=catalog,
            description_artifact=description,
            calibration_plan=calibration_plan,
            calibration_runtime_archive_digest=calibration_archive_digest,
            calibration_assessment=assessment,
            calibration_family=family,
            predicate_library=library,
            support_version_space=version,
            support_runtime_archive_digest=support_archive_digest,
            headless_archive=headless,
            phase_trace=(*PHASE_ORDER[:9], PHASE_ORDER[-1]),
        )
        return finalize_campaign(candidate)
    if query_archive_digest is None or len(state.query_artifacts) != 2:
        raise PrototypePairCampaignError("complete runner omitted query archive")
    candidate = _make_campaign_artifact(
        state,
        status=PrototypePairCampaignStatus.COMPLETE,
        precommit=precommit,
        cohort_plan=cohort_plan,
        configuration=configuration,
        exposure_successor_digest=exposure_successor_digest,
        call_journal_seal_digest=seal_current_journal(),
        precommit_receipt=precommit_receipt,
        release_authorization=authorization,
        reference_catalog=catalog,
        description_artifact=description,
        calibration_plan=calibration_plan,
        calibration_runtime_archive_digest=calibration_archive_digest,
        calibration_assessment=assessment,
        calibration_family=family,
        predicate_library=library,
        support_version_space=version,
        support_runtime_archive_digest=support_archive_digest,
        query_runtime_archive_digest=query_archive_digest,
        headless_archive=headless,
        phase_trace=PHASE_ORDER,
    )
    return finalize_campaign(candidate)


def cold_replay_prototype_pair_campaign(
    artifact: PrototypePairCampaignArtifact | Mapping[str, Any],
    *,
    cohort_plan: PrototypePairCohortPlan,
    precommit: PrototypePairExecutionPrecommit,
    release_descriptor: OfficialReleaseDescriptor,
    official_archive_path: str | Path,
    store: CampaignStore,
    expected_campaign_digest: str,
    expected_precommit_digest: str,
    expected_cohort_plan_digest: str,
    expected_identity_bundle_digest: str,
    expected_exposure_predecessor_digest: str,
) -> PrototypePairCampaignArtifact:
    """Fresh, model-free reconstruction of every authoritative campaign edge."""

    expected_campaign = _require_address(
        expected_campaign_digest, "expected campaign digest"
    )
    restored = (
        artifact
        if isinstance(artifact, PrototypePairCampaignArtifact)
        else PrototypePairCampaignArtifact.from_data(artifact)
    )
    restored = PrototypePairCampaignArtifact.from_data(restored.to_data())
    if restored.record_digest != expected_campaign:
        raise PrototypePairCampaignError("campaign differs from external commitment")
    verify_prototype_pair_execution_precommit(
        precommit,
        cohort_plan=cohort_plan,
        identities=precommit.identities,
        expected_precommit_digest=expected_precommit_digest,
        expected_cohort_plan_digest=expected_cohort_plan_digest,
        expected_identity_bundle_digest=expected_identity_bundle_digest,
        expected_exposure_predecessor_digest=expected_exposure_predecessor_digest,
    )
    if (
        restored.precommit_digest != precommit.record_digest
        or restored.cohort_plan_digest != cohort_plan.record_digest
        or release_descriptor.digest != cohort_plan.release_descriptor_digest
        or restored.configuration.record_digest
        != precommit.identities.execution_configuration_digest
    ):
        raise PrototypePairCampaignError("campaign root commitments differ")
    frozen_sources = dict(precommit.identities.runtime_source_digests)
    current_sources = prototype_pair_campaign_runtime_source_digests()
    if any(frozen_sources.get(role) != digest for role, digest in current_sources.items()):
        raise PrototypePairCampaignError("cold runtime source authority differs")
    if (
        precommit.identities.calibration_algorithm_digest
        != calibration_algorithm_digest()
        or precommit.identities.runner_protocol_id != RUNNER_ID
        or precommit.identities.runner_algorithm_digest
        != prototype_scene_runner_source_digest()
        or precommit.identities.observer_description_protocol_digest
        != prototype_rubric_description_protocol_digest()
        or precommit.identities.observer_scoring_protocol_digest
        != prototype_scene_scoring_protocol_digest()
        or precommit.identities.observer_model_identity_digest
        != prototype_scene_observer_model_digest(
            precommit.identities.observer_model_id,
            precommit.identities.observer_reasoning_effort,
        )
        or precommit.identities.observer_environment_digest
        != prototype_scene_observer_environment_digest(
            model=precommit.identities.observer_model_id,
            reasoning_effort=precommit.identities.observer_reasoning_effort,
            expected_launcher_digest=precommit.identities.codex_launcher_sha256,
            cloud_policy_cache_binding=(
                precommit.identities.cloud_policy_cache_binding
            ),
            model_catalog_digest=(
                precommit.identities.codex_model_catalog_snapshot.raw_digest
            ),
            no_tools_attestation_digest=(
                precommit.identities.codex_no_tools_attestation.attestation_digest
            ),
        )
        or precommit.identities.ranker_protocol_id
        != PROTOTYPE_SCENE_CODEX_RANKER_PROTOCOL_ID
        or precommit.identities.ranker_protocol_digest
        != prototype_scene_codex_ranker_protocol_digest()
        or precommit.identities.ranker_environment_digest
        != prototype_scene_codex_ranker_environment_digest(
            model=precommit.identities.ranker_model_id,
            reasoning_effort=precommit.identities.ranker_reasoning_effort,
            expected_launcher_digest=(
                precommit.identities.codex_launcher_sha256
            ),
            expected_cloud_policy_cache_binding=(
                precommit.identities.cloud_policy_cache_binding
            ),
            expected_transport_source_digest=(
                prototype_scene_codex_ranker_transport_source_digest()
            ),
            model_catalog_snapshot=(
                precommit.identities.codex_model_catalog_snapshot
            ),
            no_tools_attestation=(
                precommit.identities.codex_no_tools_attestation
            ),
        )
    ):
        raise PrototypePairCampaignError("cold protocol authority differs")
    precommit_bytes = canonical_json(precommit.to_data()) + b"\n"
    if store.verify_execution_precommit(
        restored.precommit_receipt,
        expected_precommit_digest,
        precommit_bytes,
    ) != precommit_bytes:
        raise PrototypePairCampaignError("cold precommit bytes differ")
    authorization = restored.release_authorization
    if (
        authorization.get("plan_digest") != cohort_plan.record_digest
        or authorization.get("execution_precommit_digest")
        != precommit.record_digest
        or authorization.get("exposure_predecessor_digest")
        != expected_exposure_predecessor_digest
        or authorization.get("exposure_successor_digest")
        != restored.exposure_successor_digest
        or authorization.get("actor") != restored.configuration.actor
    ):
        raise PrototypePairCampaignError("release authorization parents differ")
    loader = getattr(store, "load_release_authorization", None)
    if callable(loader):
        loaded_authorization = loader(authorization["record_digest"])
        if _to_data(loaded_authorization, "release authorization") != dict(
            authorization
        ):
            raise PrototypePairCampaignError("release authorization reload differs")

    # Every campaign-side store receipt is dereferenced, not merely decoded.
    for stored in restored.stored_objects:
        loaded = store.load_canonical_object(
            stored.storage_receipt, stored.object_identity_digest
        )
        present = [
            key
            for key in ("record_digest", "artifact_digest", "catalog_digest")
            if loaded.get(key) == stored.object_identity_digest
        ]
        if len(present) != 1:
            raise PrototypePairCampaignError("stored object identity reload differs")

    # Rejoin every terminal outcome to its durable typed result and status.
    journal_loader = getattr(store, "load_call_journal", None)
    if not callable(journal_loader):
        raise PrototypePairCampaignError("store cannot cold-replay call journal")
    terminal_keys: set[str] = set()
    terminal_results: dict[tuple[object, object], object] = {}
    terminal_records: dict[tuple[object, object], Mapping[str, Any]] = {}
    for terminal in restored.call_terminals:
        result_digest = terminal.get("result_digest")
        result_receipt = terminal.get("result_receipt")
        phase = terminal.get("phase")
        status = terminal.get("terminal_status")
        key_digest = terminal.get("key_digest")
        if not isinstance(result_digest, str) or not isinstance(result_receipt, Mapping):
            raise PrototypePairCampaignError("call terminal result is malformed")
        if not isinstance(key_digest, str) or key_digest in terminal_keys:
            raise PrototypePairCampaignError("call terminal key repeats or is malformed")
        terminal_keys.add(key_digest)
        durable_claim, durable_outcome = journal_loader(
            key_digest,
            expected_authorization_digest=authorization["record_digest"],
        )
        if (
            durable_outcome is None
            or _to_data(durable_outcome, "durable call outcome")
            != dict(terminal)
            or getattr(durable_claim, "record_digest", None)
            != terminal.get("claim_digest")
        ):
            raise PrototypePairCampaignError("durable call journal differs")
        result_data = store.load_canonical_object(result_receipt, result_digest)
        if phase == "prototype_description_observed":
            result = PrototypeRubricDescriptionArtifact.from_data(result_data)
            expected_status = _observer_terminal_status(result)
            actual_digest = result.artifact_digest
        elif phase in {
            "twenty_eight_calibration_scenes_released_and_observed",
            "twelve_support_scenes_released_and_observed",
            "two_query_scenes_released_and_observed",
        }:
            result = PrototypeSceneObserverArtifact.from_data(result_data)
            expected_status = _observer_terminal_status(result)
            actual_digest = result.artifact_digest
        elif phase == "headless_codex_candidate_ranked" and status == "success":
            result = PrototypeSceneRankResponse.from_data(result_data)
            expected_status = "success"
            actual_digest = result.record_digest
        elif phase == "headless_codex_candidate_ranked" and status == "error":
            result = PrototypePairCampaignCallFailure.from_data(result_data)
            expected_status = "error"
            actual_digest = result.record_digest
        else:
            raise PrototypePairCampaignError("call terminal phase/status differs")
        if isinstance(
            result,
            (PrototypeRubricDescriptionArtifact, PrototypeSceneObserverArtifact),
        ) and (
            result.model_catalog_digest
            != precommit.identities.codex_model_catalog_snapshot.raw_digest
            or result.no_tools_attestation_digest
            != precommit.identities.codex_no_tools_attestation.attestation_digest
            or result.cloud_policy_cache_binding
            != precommit.identities.cloud_policy_cache_binding
            or result.expected_launcher_digest
            != precommit.identities.codex_launcher_sha256
        ):
            raise PrototypePairCampaignError(
                "observer terminal preflight binding differs from precommit"
            )
        if expected_status != status or actual_digest != result_digest:
            raise PrototypePairCampaignError("call terminal semantics differ")
        result_key = (phase, terminal.get("subject_id"))
        if result_key in terminal_results:
            raise PrototypePairCampaignError("call terminal subject repeats")
        terminal_results[result_key] = result
        terminal_records[result_key] = terminal

    journal_seal_verifier = getattr(store, "verify_call_journal_seal", None)
    if not callable(journal_seal_verifier):
        raise PrototypePairCampaignError("store cannot verify sealed call journal")
    journal_seal = journal_seal_verifier(
        authorization["record_digest"],
        expected_terminal_key_digests=tuple(sorted(terminal_keys)),
    )
    if (
        _to_data(journal_seal, "call journal seal").get("record_digest")
        != restored.call_journal_seal_digest
    ):
        raise PrototypePairCampaignError("campaign call journal seal differs")

    if terminal_results.get(
        ("prototype_description_observed", "prototype_description")
    ) != restored.description_artifact:
        raise PrototypePairCampaignError("description terminal result differs")
    description_terminal = terminal_records[
        ("prototype_description_observed", "prototype_description")
    ]
    if description_terminal.get("context_digest") != (
        "sha256:" + restored.reference_catalog.catalog_digest
    ):
        raise PrototypePairCampaignError("description claim context differs")
    for phase, artifacts in (
        (
            "twenty_eight_calibration_scenes_released_and_observed",
            restored.calibration_artifacts,
        ),
        (
            "twelve_support_scenes_released_and_observed",
            restored.support_artifacts,
        ),
        ("two_query_scenes_released_and_observed", restored.query_artifacts),
    ):
        for item in artifacts:
            if terminal_results.get((phase, item.scene_panel_id)) != item:
                raise PrototypePairCampaignError(
                    "observer terminal does not rejoin campaign artifact"
                )
    rank_terminal = terminal_results.get(
        ("headless_codex_candidate_ranked", precommit.drill_task_id)
    )
    if restored.status is PrototypePairCampaignStatus.RANKER_ERROR:
        if len(restored.call_failures) != 1 or rank_terminal != restored.call_failures[0]:
            raise PrototypePairCampaignError("rank failure terminal differs")
    elif restored.headless_archive is not None and (
        restored.headless_archive.rank_response is not None
        and rank_terminal != restored.headless_archive.rank_response
    ):
        raise PrototypePairCampaignError("rank response terminal differs")

    fresh_archive = OfficialPanelArchive.load(
        release_descriptor,
        official_archive_path,
        expected_release_descriptor_digest=cohort_plan.release_descriptor_digest,
    )
    for panel in restored.released_panels:
        panel.cold_verify(
            fresh_archive,
            expected_execution_precommit_digest=precommit.record_digest,
            expected_exposure_successor_digest=restored.exposure_successor_digest,
        )
    released = _released_by_id(restored.released_panels)
    reference_ids = tuple(
        panel_id
        for prototype in cohort_plan.prototypes
        for panel_id in prototype.panel_ids
    )
    references = {panel_id: released[panel_id] for panel_id in reference_ids}
    reference_png = {
        panel_id: references[panel_id].exact_png_bytes for panel_id in reference_ids
    }
    reference_sha = {
        panel_id: references[panel_id].exact_png_digest.removeprefix("sha256:")
        for panel_id in reference_ids
    }
    verify_prototype_reference_catalog(
        restored.reference_catalog,
        cohort_plan,
        reference_png,
        expected_plan_digest=cohort_plan.record_digest,
        expected_reference_sha256=reference_sha,
        expected_catalog_digest=restored.reference_catalog.catalog_digest,
    )
    verify_prototype_rubric_description_artifact(
        restored.description_artifact,
        restored.reference_catalog,
        reference_png,
        expected_catalog_digest=restored.reference_catalog.catalog_digest,
        expected_artifact_digest=restored.description_artifact.artifact_digest,
    )
    if restored.status is PrototypePairCampaignStatus.DESCRIPTION_GAP:
        return restored

    plan = restored.calibration_plan
    assessment = restored.calibration_assessment
    if plan is None or assessment is None:
        raise PrototypePairCampaignError("campaign calibration parents are absent")
    verify_prototype_scene_calibration_plan(
        plan,
        cohort_plan=cohort_plan,
        expected_calibration_plan_digest=plan.record_digest,
        expected_cohort_plan_digest=cohort_plan.record_digest,
    )
    for scheduled in plan.scenes:
        terminal = terminal_records.get(
            (
                "twenty_eight_calibration_scenes_released_and_observed",
                scheduled.panel_id,
            )
        )
        if terminal is None or terminal.get("context_digest") != plan.record_digest:
            raise PrototypePairCampaignError("calibration claim context differs")
    calibration_releases = tuple(released[item.panel_id] for item in plan.scenes)
    calibration_archive = _runtime_archive(
        phase="calibration",
        configuration=restored.configuration,
        catalog=restored.reference_catalog,
        description=restored.description_artifact,
        reference_panels=references,
        scene_panels=calibration_releases,
        scene_artifacts=restored.calibration_artifacts,
        scene_task_ids=tuple(item.task_id for item in plan.scenes),
        observation_context_digest=plan.record_digest,
        purpose=PrototypeSceneArtifactPurpose.CALIBRATION,
    )
    if calibration_archive.record_digest != restored.calibration_runtime_archive_digest:
        raise PrototypePairCampaignError("calibration archive replay differs")
    observations = materialize_prototype_scene_calibration_observations(
        calibration_archive,
        plan,
        expected_archive_digest=calibration_archive.record_digest,
    )
    replayed_assessment = assess_prototype_scene_calibration(
        plan, observations, expected_calibration_plan_digest=plan.record_digest
    )
    if replayed_assessment != assessment:
        raise PrototypePairCampaignError("calibration assessment replay differs")
    if restored.status is PrototypePairCampaignStatus.CALIBRATION_GAP:
        return restored

    family = restored.calibration_family
    library = restored.predicate_library
    version = restored.support_version_space
    if family is None or library is None or version is None:
        raise PrototypePairCampaignError("campaign support parents are absent")
    verify_prototype_scene_calibration_family(
        family,
        calibration_plan=plan,
        cohort_plan=cohort_plan,
        observations=observations,
        expected_family_digest=family.record_digest,
        expected_calibration_plan_digest=plan.record_digest,
        expected_cohort_plan_digest=cohort_plan.record_digest,
    )
    library.assert_matches_family(family)
    context = PrototypeSceneEvaluationContext(
        cohort_plan_digest=family.cohort_plan_digest,
        description_catalog_digest=family.description_catalog_digest,
        prototype_reference_digest=family.prototype_reference_digest,
        observer_protocol_id=family.observer_protocol_id,
        observer_protocol_digest=family.observer_protocol_digest,
        model_id=family.model_id,
        model_identity_digest=family.model_identity_digest,
        environment_digest=family.environment_digest,
        same_basic_renderer_population_valid=True,
        conditional_transport_assumption_accepted=True,
        observer_environment_valid=True,
    )
    context_digest = prototype_scene_evaluation_context_digest(context)
    support_releases = tuple(
        released[role.source_panel_id] for role in precommit.support_roles
    )
    for role in precommit.support_roles:
        terminal = terminal_records.get(
            ("twelve_support_scenes_released_and_observed", role.source_panel_id)
        )
        if terminal is None or terminal.get("context_digest") != context_digest:
            raise PrototypePairCampaignError("support claim context differs")
    support_archive = _runtime_archive(
        phase="support",
        configuration=restored.configuration,
        catalog=restored.reference_catalog,
        description=restored.description_artifact,
        reference_panels=references,
        scene_panels=support_releases,
        scene_artifacts=restored.support_artifacts,
        scene_task_ids=(precommit.drill_task_id,) * 12,
        observation_context_digest=context_digest,
        purpose=PrototypeSceneArtifactPurpose.RUNTIME_EVALUATION,
    )
    if support_archive.record_digest != restored.support_runtime_archive_digest:
        raise PrototypePairCampaignError("support archive replay differs")
    support_panels = tuple(
        materialize_prototype_scene_panel(
            support_archive,
            family,
            role.source_panel_id,
            expected_archive_digest=support_archive.record_digest,
        )
        for role in precommit.support_roles
    )
    if support_panels != restored.support_panels:
        raise PrototypePairCampaignError("support panel replay differs")
    replayed_version = build_prototype_scene_support_version_space(
        library, family, support_panels[:6], support_panels[6:]
    )
    if replayed_version != version:
        raise PrototypePairCampaignError("support version-space replay differs")
    if version.survivor_candidate_ids:
        rank_input_digest = prototype_scene_rank_input_digest(
            library_digest=library.record_digest,
            version_space_digest=version.record_digest,
            survivor_candidate_ids=version.survivor_candidate_ids,
        )
        rank_terminal_record = terminal_records.get(
            ("headless_codex_candidate_ranked", precommit.drill_task_id)
        )
        if (
            rank_terminal_record is None
            or rank_terminal_record.get("context_digest") != rank_input_digest
        ):
            raise PrototypePairCampaignError("rank claim context differs")
        rank_result = terminal_results.get(
            ("headless_codex_candidate_ranked", precommit.drill_task_id)
        )
        if isinstance(rank_result, PrototypeSceneRankResponse):
            verify_prototype_scene_codex_rank_response(
                rank_result,
                survivor_candidate_ids=version.survivor_candidate_ids,
                rank_input_digest=rank_input_digest,
                expected_response_digest=rank_result.record_digest,
                model=precommit.identities.ranker_model_id,
                reasoning_effort=precommit.identities.ranker_reasoning_effort,
                expected_launcher_digest=(
                    precommit.identities.codex_launcher_sha256
                ),
                expected_cloud_policy_cache_binding=(
                    precommit.identities.cloud_policy_cache_binding
                ),
                expected_transport_source_digest=(
                    prototype_scene_codex_ranker_transport_source_digest()
                ),
                model_catalog_snapshot=(
                    precommit.identities.codex_model_catalog_snapshot
                ),
                no_tools_attestation=(
                    precommit.identities.codex_no_tools_attestation
                ),
            )
        elif restored.status is not PrototypePairCampaignStatus.RANKER_ERROR:
            raise PrototypePairCampaignError(
                "successful rank terminal is absent or untyped"
            )
    if restored.status is PrototypePairCampaignStatus.RANKER_ERROR:
        return restored
    headless = restored.headless_archive
    if headless is None:
        raise PrototypePairCampaignError("campaign headless archive is absent")
    if headless.status is not PrototypeSceneHeadlessStatus.COMPLETE:
        verifier = PrototypeScenePhasedArtifactVerifier.for_support(
            support_archive,
            expected_support_archive_digest=support_archive.record_digest,
            family=family,
            support_panels=support_panels,
        )
    else:
        if headless.freeze is None or headless.freeze_commit is None:
            raise PrototypePairCampaignError("complete headless freeze is absent")
        query_releases = tuple(
            released[role.source_panel_id] for role in precommit.query_roles
        )
        query_archive = _runtime_archive(
            phase="query",
            configuration=restored.configuration,
            catalog=restored.reference_catalog,
            description=restored.description_artifact,
            reference_panels=references,
            scene_panels=query_releases,
            scene_artifacts=restored.query_artifacts,
            scene_task_ids=(precommit.drill_task_id,) * 2,
            observation_context_digest=context_digest,
            purpose=PrototypeSceneArtifactPurpose.RUNTIME_EVALUATION,
        )
        if query_archive.record_digest != restored.query_runtime_archive_digest:
            raise PrototypePairCampaignError("query archive replay differs")
        commit_loader = getattr(store, "load_candidate_freeze_commit", None)
        if not callable(commit_loader):
            raise PrototypePairCampaignError("store cannot reload freeze commit")
        commit, freeze_receipt, commit_receipt = commit_loader(
            headless.freeze_commit.record_digest
        )
        if commit != headless.freeze_commit:
            raise PrototypePairCampaignError("freeze commit replay differs")
        freeze_data = store.load_canonical_object(
            freeze_receipt, headless.freeze.record_digest
        )
        commit_data = store.load_canonical_object(
            commit_receipt, headless.freeze_commit.record_digest
        )
        freeze = PrototypeSceneCandidateFreeze.from_data(freeze_data)
        commit = PrototypeSceneFreezeCommitReceipt.from_data(commit_data)
        if freeze != headless.freeze or commit != headless.freeze_commit:
            raise PrototypePairCampaignError("durable freeze bytes replay differs")
        for role in precommit.query_roles:
            expected_query_context = _address(
                {
                    "schema": "gkm.bongard-prototype-pair-query-call-context.v1",
                    "evaluation_context_digest": context_digest,
                    "freeze_digest": freeze.record_digest,
                    "freeze_commit_digest": commit.record_digest,
                    "role_id": role.role_id,
                    "source_panel_id": role.source_panel_id,
                }
            )
            terminal = terminal_records.get(
                ("two_query_scenes_released_and_observed", role.source_panel_id)
            )
            if terminal is None or terminal.get("context_digest") != expected_query_context:
                raise PrototypePairCampaignError("query claim context differs")
        verifier = PrototypeScenePhasedArtifactVerifier.from_pinned_archives_for_cold_replay(
            support_archive,
            query_archive,
            expected_support_archive_digest=support_archive.record_digest,
            expected_query_archive_digest=query_archive.record_digest,
            family=family,
            support_panels=support_panels,
            freeze=freeze,
            freeze_commit=commit,
            expected_freeze_commit_digest=commit.record_digest,
        )
    replayed_headless = cold_replay_prototype_scene_headless_run(
        headless,
        expected_archive_digest=headless.record_digest,
        artifact_verifier=verifier,
    )
    if replayed_headless != headless:
        raise PrototypePairCampaignError("headless cold replay differs")
    return restored


__all__ = (
    "CAMPAIGN_ALGORITHM_ID",
    "CAMPAIGN_SCHEMA",
    "CAMPAIGN_SOURCE_SHA256",
    "CampaignClock",
    "CampaignStore",
    "PrototypePairCampaignArtifact",
    "PrototypePairCampaignCallFailure",
    "PrototypePairCampaignConfiguration",
    "PrototypePairCampaignError",
    "PrototypePairCampaignStatus",
    "PrototypePairStoredObject",
    "cold_replay_prototype_pair_campaign",
    "prototype_pair_campaign_algorithm_digest",
    "prototype_pair_campaign_runtime_source_digests",
    "run_prototype_pair_campaign",
)
