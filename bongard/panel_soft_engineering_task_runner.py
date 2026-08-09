"""Pure task runner for the uncalibrated panel-soft engineering diagnostic.

The runner owns no archive or model transport.  It receives exact support PNG
bytes and already receipted proposer/observer artifacts, verifies those bytes,
builds the deterministic positive-conjunction version space, applies one
explicit selection mode (receipted support-only ranking or named deterministic
baseline), and commits the selected two-orientation predicate pair.  Only after
the commit callback has returned and the exact canonical freeze bytes have been
reloaded does it invoke a query callback.  This runner proves callback order;
the enclosing campaign must authenticate external storage durability.  The
query callback must return both exact query PNGs and their observations under
the same complete vocabulary and observer contract.

This is explicitly an engineering diagnostic, not a scientific benchmark.
Same-model repeat consensus remains uncalibrated.  Python is the sole
executable predicate authority; Lean is absent, optional, and removable.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

import base64
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import re
from typing import Any, Callable, Mapping, Sequence

from bongard.canonical import canonical_digest, canonical_json
from bongard.object_bongard_batch import ObjectBongardTaskPlan
from bongard.panel_soft_observer import (
    PanelSoftObserverArtifact,
    aggregate_panel_soft_observer_artifacts,
    verify_panel_soft_observer_artifact,
    verify_panel_soft_observer_contract_identity,
)
from bongard.panel_soft_predicate import (
    PANEL_SOFT_ORIENTATIONS,
    PANEL_SOFT_PAIR_SELECTION_MODES,
    PanelSoftEngineeringPredicatePair,
    PanelSoftEngineeringQueryDecision,
    PanelSoftEngineeringQueryOutcome,
    PanelSoftEngineeringVersionSpace,
    PanelSoftObservationTable,
    PanelSoftOperationalConsensus,
    PanelSoftVocabulary,
)
from bongard.panel_soft_ranker import (
    PanelSoftRankArtifact,
    PanelSoftRankInput,
)
from bongard.panel_soft_proposer import (
    PanelSoftProposerArtifact,
    PanelSoftProposerStatus,
    verify_panel_soft_proposer_artifact,
)
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID


PANEL_SOFT_ENGINEERING_TASK_RUNNER_ID = (
    "bongard.panel-soft-engineering-task/support-rank-freeze-query-v3"
)
PANEL_SOFT_ENGINEERING_SUPPORT_GAP_SCHEMA = (
    "gkm.bongard-panel-soft-engineering-support-gap.v3"
)
PANEL_SOFT_ENGINEERING_TASK_FREEZE_SCHEMA = (
    "gkm.bongard-panel-soft-engineering-task-freeze.v3"
)
PANEL_SOFT_ENGINEERING_TASK_FREEZE_COMMIT_SCHEMA = (
    "gkm.bongard-panel-soft-engineering-task-freeze-commit.v3"
)
PANEL_SOFT_ENGINEERING_TASK_ARCHIVE_SCHEMA = (
    "gkm.bongard-panel-soft-engineering-task-archive.v3"
)
PANEL_SOFT_ENGINEERING_PROPOSER_TERMINAL_SCHEMA = (
    "gkm.bongard-panel-soft-engineering-proposer-terminal.v3"
)
PANEL_SOFT_ENGINEERING_QUERY_DENOMINATOR = 2
PANEL_SOFT_ENGINEERING_SUPPORT_PANEL_COUNT = 12

_RAW_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,511}\Z")


class PanelSoftEngineeringTaskRunnerError(RuntimeError):
    """A task parent, byte witness, freeze, query, or replay differs."""


class PanelSoftEngineeringTaskRunStatus(str, Enum):
    COMPLETE = "complete"
    SUPPORT_GAP = "support_gap"
    SUPPORT_ERROR = "support_error"


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "evaluation_kind": "engineering_diagnostic",
        "engineering_only": True,
        "uncalibrated": True,
        "scientific_evidence": False,
        "scientific_benchmark": False,
        "negation_allowed": False,
        "polarity_flip_allowed": False,
        "arbitrary_predicate_code_allowed": False,
        "support_only_codex_ranker_supported": True,
        "predicate_pair_selection_mode_explicit": True,
        "ranked_mode_requires_receipted_rank_artifact": True,
        "deterministic_baseline_is_explicit_mode_only": True,
        "silent_ranker_fallback_allowed": False,
        "default_rejects_unsealable_rank_artifact": True,
        "unverified_rank_artifact_override_is_engineering_only": True,
        "rank_override_authenticates_provenance_or_runtime": False,
        "campaign_external_runtime_and_journal_custody_verification_required": True,
        "ranker_callback_invocations_count_callbacks_only": True,
        "physical_rank_model_calls_require_external_journal_evidence": True,
        "python_predicate_pair_frozen_before_query_callback": True,
        "sealed_proposer_and_observer_artifacts_required": True,
        "lean_present": False,
        "lean_required": False,
        "lean_removable": True,
        "lean_affects_identity_selection_decision_scoring_or_replay": False,
    }


def panel_soft_engineering_task_runner_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def _raw_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _RAW_DIGEST.fullmatch(value) is None:
        raise PanelSoftEngineeringTaskRunnerError(
            f"{label} must be a raw lowercase SHA-256"
        )
    return value


def _address(value: object, label: str) -> str:
    if not isinstance(value, str) or _ADDRESS.fullmatch(value) is None:
        raise PanelSoftEngineeringTaskRunnerError(
            f"{label} must be a sha256: address"
        )
    return value


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise PanelSoftEngineeringTaskRunnerError(
            f"{label} must be a bounded identifier"
        )
    return value


def _content_address(value: object) -> str:
    return "sha256:" + canonical_digest(value)


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(not isinstance(key, str) for key in value)
        or set(value) != expected
    ):
        raise PanelSoftEngineeringTaskRunnerError(f"{label} fields differ")
    return value


def _png(value: object, label: str) -> bytes:
    if (
        not isinstance(value, bytes)
        or not value.startswith(b"\x89PNG\r\n\x1a\n")
        or not value
    ):
        raise PanelSoftEngineeringTaskRunnerError(f"{label} is not exact PNG bytes")
    return value


def _canonical_task(value: ObjectBongardTaskPlan) -> ObjectBongardTaskPlan:
    if not isinstance(value, ObjectBongardTaskPlan):
        raise TypeError("task_plan must be ObjectBongardTaskPlan")
    restored = ObjectBongardTaskPlan.from_data(value.to_data())
    if restored != value:
        raise PanelSoftEngineeringTaskRunnerError("task plan round trip differs")
    return restored


def _canonical_proposer(value: PanelSoftProposerArtifact) -> PanelSoftProposerArtifact:
    if not isinstance(value, PanelSoftProposerArtifact):
        raise TypeError("proposer_artifact must be PanelSoftProposerArtifact")
    restored = PanelSoftProposerArtifact.from_data(value.to_data())
    if restored != value:
        raise PanelSoftEngineeringTaskRunnerError("proposer artifact round trip differs")
    return restored


def _canonical_observer(value: PanelSoftObserverArtifact) -> PanelSoftObserverArtifact:
    if not isinstance(value, PanelSoftObserverArtifact):
        raise TypeError("panel observation must be PanelSoftObserverArtifact")
    restored = PanelSoftObserverArtifact.from_data(value.to_data())
    if restored != value:
        raise PanelSoftEngineeringTaskRunnerError("observer artifact round trip differs")
    return restored


def _selection_mode(value: object) -> str:
    if value not in PANEL_SOFT_PAIR_SELECTION_MODES:
        raise PanelSoftEngineeringTaskRunnerError(
            "predicate-pair selection mode differs"
        )
    return value  # type: ignore[return-value]


def _rank_override(selection_mode: str, value: object) -> bool:
    mode = _selection_mode(selection_mode)
    if type(value) is not bool:
        raise TypeError("allow_unverified_rank_artifact must be bool")
    if value and mode != "support_only_codex_ranker":
        raise PanelSoftEngineeringTaskRunnerError(
            "unverified rank-artifact override is valid only in ranked mode"
        )
    return value


def _canonical_rank_artifact(
    value: PanelSoftRankArtifact,
    version_space: PanelSoftEngineeringVersionSpace,
) -> PanelSoftRankArtifact:
    if not isinstance(value, PanelSoftRankArtifact):
        raise TypeError("ranker callback must return PanelSoftRankArtifact")
    restored = PanelSoftRankArtifact.from_data(value.to_data())
    expected_input = PanelSoftRankInput.freeze(version_space)
    if restored != value or restored.rank_input != expected_input:
        raise PanelSoftEngineeringTaskRunnerError(
            "rank artifact differs from the exact support version space"
        )
    return restored


def _support_gap_content(value: "PanelSoftEngineeringSupportGap") -> dict[str, object]:
    return {
        "schema": PANEL_SOFT_ENGINEERING_SUPPORT_GAP_SCHEMA,
        "version_space_digest": value.version_space_digest,
        "support_table_digest": value.support_table_digest,
        "missing_orientations": list(value.missing_orientations),
        "survivor_counts_by_orientation": {
            orientation: count
            for orientation, count in zip(
                PANEL_SOFT_ORIENTATIONS,
                value.survivor_counts_by_orientation,
                strict=True,
            )
        },
        "observer_error_cell_count": value.observer_error_cell_count,
        "observer_disagreement_cell_count": value.observer_disagreement_cell_count,
        "observer_indeterminate_cell_count": value.observer_indeterminate_cell_count,
        "gap_kind": "required-orientation-has-no-support-survivor",
        "failed_or_uncertain_observation_is_nonmatch": False,
        "query_callback_permitted": False,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class PanelSoftEngineeringSupportGap:
    version_space_digest: str
    support_table_digest: str
    missing_orientations: tuple[str, ...]
    survivor_counts_by_orientation: tuple[int, int]
    observer_error_cell_count: int
    observer_disagreement_cell_count: int
    observer_indeterminate_cell_count: int
    gap_digest: str

    def __post_init__(self) -> None:
        _raw_digest(self.version_space_digest, "gap version-space digest")
        _raw_digest(self.support_table_digest, "gap support-table digest")
        if (
            type(self.missing_orientations) is not tuple
            or not self.missing_orientations
            or self.missing_orientations
            != tuple(
                item for item in PANEL_SOFT_ORIENTATIONS if item in self.missing_orientations
            )
            or any(item not in PANEL_SOFT_ORIENTATIONS for item in self.missing_orientations)
            or type(self.survivor_counts_by_orientation) is not tuple
            or len(self.survivor_counts_by_orientation) != 2
            or any(type(item) is not int or item < 0 for item in self.survivor_counts_by_orientation)
            or any(
                type(item) is not int or item < 0
                for item in (
                    self.observer_error_cell_count,
                    self.observer_disagreement_cell_count,
                    self.observer_indeterminate_cell_count,
                )
            )
        ):
            raise PanelSoftEngineeringTaskRunnerError("support gap counters differ")
        _raw_digest(self.gap_digest, "support gap digest")
        if self.gap_digest != canonical_digest(_support_gap_content(self)):
            raise PanelSoftEngineeringTaskRunnerError("support gap digest differs")

    @classmethod
    def create(
        cls, version_space: PanelSoftEngineeringVersionSpace
    ) -> "PanelSoftEngineeringSupportGap":
        if not isinstance(version_space, PanelSoftEngineeringVersionSpace):
            raise TypeError("version_space must be PanelSoftEngineeringVersionSpace")
        counts = tuple(
            sum(item.orientation == orientation for item in version_space.survivor_formulas)
            for orientation in PANEL_SOFT_ORIENTATIONS
        )
        missing = tuple(
            orientation
            for orientation, count in zip(PANEL_SOFT_ORIENTATIONS, counts, strict=True)
            if count == 0
        )
        if not missing:
            raise PanelSoftEngineeringTaskRunnerError(
                "support gap requires a missing orientation survivor"
            )
        consensuses = tuple(
            item.operational_consensus for item in version_space.support_table.cells
        )
        values = {
            "version_space_digest": version_space.engineering_version_space_digest,
            "support_table_digest": version_space.support_table.table_digest,
            "missing_orientations": missing,
            "survivor_counts_by_orientation": counts,
            "observer_error_cell_count": sum(
                item is PanelSoftOperationalConsensus.ERROR for item in consensuses
            ),
            "observer_disagreement_cell_count": sum(
                item is PanelSoftOperationalConsensus.DISAGREEMENT for item in consensuses
            ),
            "observer_indeterminate_cell_count": sum(
                item is PanelSoftOperationalConsensus.REPEATED_INDETERMINATE
                for item in consensuses
            ),
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(**values, gap_digest=canonical_digest(_support_gap_content(provisional)))

    @property
    def has_observer_error(self) -> bool:
        return self.observer_error_cell_count > 0

    def to_data(self) -> dict[str, object]:
        return {**_support_gap_content(self), "gap_digest": self.gap_digest}

    @classmethod
    def from_data(cls, value: object) -> "PanelSoftEngineeringSupportGap":
        raw = _fields(
            value,
            {
                "schema", "version_space_digest", "support_table_digest",
                "missing_orientations", "survivor_counts_by_orientation",
                "observer_error_cell_count", "observer_disagreement_cell_count",
                "observer_indeterminate_cell_count", "gap_kind",
                "failed_or_uncertain_observation_is_nonmatch",
                "query_callback_permitted", *_authority_data(), "gap_digest",
            },
            "panel-soft support gap",
        )
        counts = raw["survivor_counts_by_orientation"]
        if (
            raw["schema"] != PANEL_SOFT_ENGINEERING_SUPPORT_GAP_SCHEMA
            or raw["gap_kind"] != "required-orientation-has-no-support-survivor"
            or raw["failed_or_uncertain_observation_is_nonmatch"] is not False
            or raw["query_callback_permitted"] is not False
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["missing_orientations"], list)
            or not isinstance(counts, Mapping)
            or list(counts) != list(PANEL_SOFT_ORIENTATIONS)
        ):
            raise PanelSoftEngineeringTaskRunnerError("support gap policy differs")
        result = cls(
            raw["version_space_digest"],
            raw["support_table_digest"],
            tuple(raw["missing_orientations"]),
            tuple(counts[item] for item in PANEL_SOFT_ORIENTATIONS),
            raw["observer_error_cell_count"],
            raw["observer_disagreement_cell_count"],
            raw["observer_indeterminate_cell_count"],
            raw["gap_digest"],
        )
        if result.to_data() != dict(raw):
            raise PanelSoftEngineeringTaskRunnerError("support gap is not canonical")
        return result


def _proposer_terminal_content(
    value: "PanelSoftEngineeringProposerTerminal",
) -> dict[str, object]:
    return {
        "schema": PANEL_SOFT_ENGINEERING_PROPOSER_TERMINAL_SCHEMA,
        "runner_id": PANEL_SOFT_ENGINEERING_TASK_RUNNER_ID,
        "runner_source_digest": value.runner_source_digest,
        "task_plan": value.task_plan.to_data(),
        "execution_precommit_digest": value.execution_precommit_digest,
        "selection_mode": value.selection_mode,
        "allow_unverified_rank_artifact": value.allow_unverified_rank_artifact,
        "rank_artifact_benchmark_sealable": value.rank_artifact_benchmark_sealable,
        "proposer_artifact": value.proposer_artifact.to_data(),
        "proposer_status": value.proposer_artifact.status.value,
        "proposer_failure_code": value.proposer_artifact.failure_code,
        "proposer_failure_type": value.proposer_artifact.failure_type,
        "support_png_base64_by_panel_id": {
            panel_id: encoded
            for panel_id, encoded in value.support_png_base64_by_panel_id
        },
        "status": PanelSoftEngineeringTaskRunStatus.SUPPORT_ERROR.value,
        "terminal_stage": "proposer",
        "support_observer_artifact_count": 0,
        "ranker_callback_invocations": 0,
        "freeze_commit_calls_made": 0,
        "freeze_reload_calls_made": 0,
        "query_source_calls_made": 0,
        "correct_count": 0,
        "determinate_count": 0,
        "abstain_count": 0,
        "error_count": 2,
        "query_denominator": PANEL_SOFT_ENGINEERING_QUERY_DENOMINATOR,
        "accuracy_ppm": 0,
        "coverage_ppm": 0,
        "query_pixels_released": False,
        "no_observation_table_or_version_space_fabricated": True,
        "exact_released_support_pngs_archived_for_cold_replay": True,
        "cold_replay_model_calls": 0,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class PanelSoftEngineeringProposerTerminal:
    """Typed fixed-denominator stop for a non-success proposer artifact."""

    runner_source_digest: str
    task_plan: ObjectBongardTaskPlan
    execution_precommit_digest: str
    selection_mode: str
    allow_unverified_rank_artifact: bool
    rank_artifact_benchmark_sealable: bool | None
    proposer_artifact: PanelSoftProposerArtifact
    support_png_base64_by_panel_id: tuple[tuple[str, str], ...]
    record_digest: str

    def __post_init__(self) -> None:
        _raw_digest(self.runner_source_digest, "proposer terminal runner digest")
        _address(self.execution_precommit_digest, "proposer terminal precommit")
        _raw_digest(self.record_digest, "proposer terminal digest")
        task = _canonical_task(self.task_plan)
        proposer = _canonical_proposer(self.proposer_artifact)
        _rank_override(
            self.selection_mode, self.allow_unverified_rank_artifact
        )
        if (
            self.rank_artifact_benchmark_sealable is not None
            or self.runner_source_digest
            != panel_soft_engineering_task_runner_source_digest()
            or task != self.task_plan
            or proposer != self.proposer_artifact
            or proposer.status is PanelSoftProposerStatus.SUCCESS
            or proposer.vocabulary is not None
            or proposer.support_panel_ids != _expected_support_ids(task)
            or tuple(item[0] for item in self.support_png_base64_by_panel_id)
            != _expected_support_ids(task)
            or self.record_digest != canonical_digest(_proposer_terminal_content(self))
        ):
            raise PanelSoftEngineeringTaskRunnerError(
                "proposer terminal content differs"
            )

    def to_data(self) -> dict[str, object]:
        return {**_proposer_terminal_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: object) -> "PanelSoftEngineeringProposerTerminal":
        raw = _fields(
            value,
            {
                "schema", "runner_id", "runner_source_digest", "task_plan",
                "execution_precommit_digest", "selection_mode",
                "allow_unverified_rank_artifact",
                "rank_artifact_benchmark_sealable", "proposer_artifact",
                "proposer_status", "proposer_failure_code",
                "proposer_failure_type", "support_png_base64_by_panel_id",
                "status", "terminal_stage", "support_observer_artifact_count",
                "ranker_callback_invocations",
                "freeze_commit_calls_made", "freeze_reload_calls_made",
                "query_source_calls_made", "correct_count", "determinate_count",
                "abstain_count", "error_count", "query_denominator",
                "accuracy_ppm", "coverage_ppm", "query_pixels_released",
                "no_observation_table_or_version_space_fabricated",
                "exact_released_support_pngs_archived_for_cold_replay",
                "cold_replay_model_calls", *_authority_data(), "record_digest",
            },
            "panel-soft proposer terminal",
        )
        if (
            raw["schema"] != PANEL_SOFT_ENGINEERING_PROPOSER_TERMINAL_SCHEMA
            or raw["runner_id"] != PANEL_SOFT_ENGINEERING_TASK_RUNNER_ID
            or raw["status"] != PanelSoftEngineeringTaskRunStatus.SUPPORT_ERROR.value
            or raw["terminal_stage"] != "proposer"
            or raw["support_observer_artifact_count"] != 0
            or type(raw["allow_unverified_rank_artifact"]) is not bool
            or raw["rank_artifact_benchmark_sealable"] is not None
            or raw["ranker_callback_invocations"] != 0
            or (raw["freeze_commit_calls_made"], raw["freeze_reload_calls_made"], raw["query_source_calls_made"]) != (0, 0, 0)
            or (raw["correct_count"], raw["determinate_count"], raw["abstain_count"], raw["error_count"]) != (0, 0, 0, 2)
            or raw["query_denominator"] != PANEL_SOFT_ENGINEERING_QUERY_DENOMINATOR
            or raw["accuracy_ppm"] != 0
            or raw["coverage_ppm"] != 0
            or raw["query_pixels_released"] is not False
            or raw["no_observation_table_or_version_space_fabricated"] is not True
            or raw["exact_released_support_pngs_archived_for_cold_replay"] is not True
            or raw["cold_replay_model_calls"] != 0
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["support_png_base64_by_panel_id"], Mapping)
        ):
            raise PanelSoftEngineeringTaskRunnerError(
                "proposer terminal policy differs"
            )
        task = ObjectBongardTaskPlan.from_data(raw["task_plan"])
        proposer = PanelSoftProposerArtifact.from_data(raw["proposer_artifact"])
        encoded = raw["support_png_base64_by_panel_id"]
        ids = _expected_support_ids(task)
        if (
            set(encoded) != set(ids)
            or raw["proposer_status"] != proposer.status.value
            or raw["proposer_failure_code"] != proposer.failure_code
            or raw["proposer_failure_type"] != proposer.failure_type
        ):
            raise PanelSoftEngineeringTaskRunnerError(
                "proposer terminal parent fields differ"
            )
        result = cls(
            raw["runner_source_digest"], task, raw["execution_precommit_digest"],
            raw["selection_mode"], raw["allow_unverified_rank_artifact"],
            raw["rank_artifact_benchmark_sealable"], proposer,
            tuple((item, encoded[item]) for item in ids),
            raw["record_digest"],
        )
        if result.to_data() != dict(raw):
            raise PanelSoftEngineeringTaskRunnerError(
                "proposer terminal is not canonical"
            )
        return result


def _freeze_content(value: "PanelSoftEngineeringTaskFreeze") -> dict[str, object]:
    pair = value.predicate_pair
    return {
        "schema": PANEL_SOFT_ENGINEERING_TASK_FREEZE_SCHEMA,
        "runner_id": PANEL_SOFT_ENGINEERING_TASK_RUNNER_ID,
        "runner_source_digest": value.runner_source_digest,
        "task_id": value.task_id,
        "task_plan_digest": value.task_plan_digest,
        "execution_precommit_digest": value.execution_precommit_digest,
        "proposer_artifact_digest": value.proposer_artifact_digest,
        "proposer_raw_evidence_digest": value.proposer_raw_evidence_digest,
        "vocabulary_digest": value.vocabulary_digest,
        "observer_contract_digest": value.observer_contract_digest,
        "support_panel_ids": list(value.support_panel_ids),
        "support_panel_png_digests": list(value.support_panel_png_digests),
        "support_observer_artifact_digests": list(
            value.support_observer_artifact_digests
        ),
        "support_table_digest": value.support_table_digest,
        "engineering_version_space_digest": value.engineering_version_space_digest,
        "selection_mode": value.selection_mode,
        "allow_unverified_rank_artifact": value.allow_unverified_rank_artifact,
        "rank_artifact_benchmark_sealable": value.rank_artifact_benchmark_sealable,
        "rank_artifact_digest": value.rank_artifact_digest,
        "rank_input_digest": value.rank_input_digest,
        "rank_receipt_digest": value.rank_receipt_digest,
        "predicate_pair": pair.to_data(),
        "predicate_pair_digest": pair.predicate_pair_digest,
        "version_space_digest": value.version_space_digest,
        "support_version_space_digest": value.support_version_space_digest,
        "selected_predicate_digest": value.selected_predicate_digest,
        "selected_predicate_is_two_orientation_pair": True,
        "sealed_query_panel_ids": list(value.sealed_query_panel_ids),
        "query_pixels_included": False,
        "query_observer_artifacts_included": False,
        "exact_support_pixels_verified_before_freeze": True,
        "complete_vocabulary_observed_twice_per_support_panel": True,
        "predicate_pair_frozen_before_query_source": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class PanelSoftEngineeringTaskFreeze:
    runner_source_digest: str
    task_id: str
    task_plan_digest: str
    execution_precommit_digest: str
    proposer_artifact_digest: str
    proposer_raw_evidence_digest: str
    vocabulary_digest: str
    observer_contract_digest: str
    support_panel_ids: tuple[str, ...]
    support_panel_png_digests: tuple[str, ...]
    support_observer_artifact_digests: tuple[str, ...]
    support_table_digest: str
    engineering_version_space_digest: str
    selection_mode: str
    allow_unverified_rank_artifact: bool
    rank_artifact_benchmark_sealable: bool | None
    rank_artifact_digest: str | None
    rank_input_digest: str | None
    rank_receipt_digest: str | None
    predicate_pair: PanelSoftEngineeringPredicatePair
    version_space_digest: str
    support_version_space_digest: str
    selected_predicate_digest: str
    sealed_query_panel_ids: tuple[str, str]
    record_digest: str

    def __post_init__(self) -> None:
        _raw_digest(self.runner_source_digest, "freeze runner source digest")
        _identifier(self.task_id, "freeze task ID")
        _address(self.task_plan_digest, "freeze task plan digest")
        _address(self.execution_precommit_digest, "freeze execution precommit digest")
        for name in (
            "proposer_artifact_digest", "proposer_raw_evidence_digest",
            "vocabulary_digest", "observer_contract_digest",
            "support_table_digest", "engineering_version_space_digest",
            "version_space_digest", "support_version_space_digest",
            "selected_predicate_digest",
        ):
            _raw_digest(getattr(self, name), name)
        mode = _selection_mode(self.selection_mode)
        _rank_override(mode, self.allow_unverified_rank_artifact)
        rank_digests = (
            self.rank_artifact_digest,
            self.rank_input_digest,
            self.rank_receipt_digest,
        )
        if mode == "support_only_codex_ranker":
            if type(self.rank_artifact_benchmark_sealable) is not bool:
                raise PanelSoftEngineeringTaskRunnerError(
                    "ranked freeze lacks a benchmark-sealability disposition"
                )
            if (
                not self.rank_artifact_benchmark_sealable
                and not self.allow_unverified_rank_artifact
            ):
                raise PanelSoftEngineeringTaskRunnerError(
                    "unsealable rank artifact lacks the engineering override"
                )
            if any(item is None for item in rank_digests):
                raise PanelSoftEngineeringTaskRunnerError(
                    "ranked freeze lacks a rank artifact commitment"
                )
            for label, item in zip(
                ("rank artifact digest", "rank input digest", "rank receipt digest"),
                rank_digests,
                strict=True,
            ):
                _raw_digest(item, label)
        elif (
            self.rank_artifact_benchmark_sealable is not None
            or any(item is not None for item in rank_digests)
        ):
            raise PanelSoftEngineeringTaskRunnerError(
                "baseline freeze contains a rank artifact commitment"
            )
        _address(self.record_digest, "freeze record digest")
        if not isinstance(self.predicate_pair, PanelSoftEngineeringPredicatePair):
            raise TypeError("freeze predicate pair has the wrong type")
        pair = PanelSoftEngineeringPredicatePair.from_data(
            self.predicate_pair.to_data()
        )
        table = pair.engineering_version_space.support_table
        if (
            self.runner_source_digest
            != panel_soft_engineering_task_runner_source_digest()
            or type(self.support_panel_ids) is not tuple
            or len(self.support_panel_ids) != PANEL_SOFT_ENGINEERING_SUPPORT_PANEL_COUNT
            or len(set(self.support_panel_ids)) != len(self.support_panel_ids)
            or any(_IDENTIFIER.fullmatch(item) is None for item in self.support_panel_ids)
            or type(self.support_panel_png_digests) is not tuple
            or len(self.support_panel_png_digests) != len(self.support_panel_ids)
            or type(self.support_observer_artifact_digests) is not tuple
            or len(self.support_observer_artifact_digests) != len(self.support_panel_ids)
            or len(set(self.support_observer_artifact_digests))
            != len(self.support_observer_artifact_digests)
            or any(_RAW_DIGEST.fullmatch(item) is None for item in self.support_panel_png_digests)
            or any(_RAW_DIGEST.fullmatch(item) is None for item in self.support_observer_artifact_digests)
            or pair != self.predicate_pair
            or pair.selection_mode != mode
            or self.vocabulary_digest != table.vocabulary.vocabulary_digest
            or self.observer_contract_digest != table.contract.contract_digest
            or self.support_panel_ids != table.panel_ids
            or self.support_panel_png_digests != table.panel_png_digests
            or self.support_table_digest != table.table_digest
            or self.engineering_version_space_digest
            != pair.engineering_version_space.engineering_version_space_digest
            or self.version_space_digest != self.engineering_version_space_digest
            or self.support_version_space_digest != self.engineering_version_space_digest
            or self.selected_predicate_digest != pair.predicate_pair_digest
            or type(self.sealed_query_panel_ids) is not tuple
            or len(self.sealed_query_panel_ids) != 2
            or len(set(self.sealed_query_panel_ids)) != 2
            or any(_IDENTIFIER.fullmatch(item) is None for item in self.sealed_query_panel_ids)
            or self.record_digest != _content_address(_freeze_content(self))
        ):
            raise PanelSoftEngineeringTaskRunnerError("task freeze content differs")

    @classmethod
    def seal(
        cls,
        *,
        task_plan: ObjectBongardTaskPlan,
        execution_precommit_digest: str,
        proposer_artifact: PanelSoftProposerArtifact,
        support_artifacts: Sequence[PanelSoftObserverArtifact],
        predicate_pair: PanelSoftEngineeringPredicatePair,
        rank_artifact: PanelSoftRankArtifact | None,
        allow_unverified_rank_artifact: bool,
    ) -> "PanelSoftEngineeringTaskFreeze":
        task = _canonical_task(task_plan)
        proposer = _canonical_proposer(proposer_artifact)
        artifacts = tuple(_canonical_observer(item) for item in support_artifacts)
        if not isinstance(predicate_pair, PanelSoftEngineeringPredicatePair):
            raise TypeError("predicate_pair must be PanelSoftEngineeringPredicatePair")
        pair = PanelSoftEngineeringPredicatePair.from_data(predicate_pair.to_data())
        _rank_override(pair.selection_mode, allow_unverified_rank_artifact)
        if pair != predicate_pair:
            raise PanelSoftEngineeringTaskRunnerError(
                "freeze predicate pair round trip differs"
            )
        space = pair.engineering_version_space
        table = space.support_table
        rank = (
            None
            if rank_artifact is None
            else _canonical_rank_artifact(rank_artifact, space)
        )
        if pair.selection_mode == "support_only_codex_ranker":
            if (
                rank is None
                or rank.selected_formula_digests
                != (pair.side0_formula_digest, pair.side1_formula_digest)
            ):
                raise PanelSoftEngineeringTaskRunnerError(
                    "ranked predicate pair differs from its rank artifact"
                )
            if (
                not rank.transport_provenance.benchmark_sealable
                and not allow_unverified_rank_artifact
            ):
                raise PanelSoftEngineeringTaskRunnerError(
                    "unsealable rank artifact lacks the engineering override"
                )
        elif rank is not None:
            raise PanelSoftEngineeringTaskRunnerError(
                "deterministic baseline cannot carry a rank artifact"
            )
        expected_support_ids = _expected_support_ids(task)
        query_ids = (task.side_0_query_panel_id, task.side_1_query_panel_id)
        panel_commitments = tuple(
            zip(table.panel_ids, table.panel_png_digests, strict=True)
        )
        if (
            proposer.status is not PanelSoftProposerStatus.SUCCESS
            or proposer.vocabulary is None
            or proposer.raw_proposer_evidence_digest is None
            or proposer.receipt is None
            or proposer.support_panel_ids != expected_support_ids
            or proposer.vocabulary != table.vocabulary
            or tuple(item.content_digest for item in proposer.presentation)
            != table.panel_png_digests
            or space.side0_panel_ids != task.side_0_support_panel_ids
            or space.side1_panel_ids != task.side_1_support_panel_ids
            or table.panel_ids != expected_support_ids
            or len(artifacts) != PANEL_SOFT_ENGINEERING_SUPPORT_PANEL_COUNT
            or tuple((item.panel_id, item.panel_png_digest) for item in artifacts)
            != panel_commitments
            or any(item.vocabulary != table.vocabulary for item in artifacts)
            or any(item.contract != table.contract for item in artifacts)
            or any(_runtime_tuple(item) != _runtime_tuple(proposer) for item in artifacts)
            or len(set(query_ids)) != 2
            or set(query_ids).intersection(expected_support_ids)
        ):
            raise PanelSoftEngineeringTaskRunnerError(
                "freeze proposer, task, table, artifact, or query lineage differs"
            )
        rebuilt_table = aggregate_panel_soft_observer_artifacts(
            artifacts,
            ordered_panel_commitments=panel_commitments,
            expected_vocabulary=proposer.vocabulary,
            expected_contract=table.contract,
        )
        if rebuilt_table != table:
            raise PanelSoftEngineeringTaskRunnerError(
                "freeze support artifacts do not reconstruct the predicate table"
            )
        _require_proposer_observer_call_distinctness(proposer, artifacts)
        values = {
            "runner_source_digest": panel_soft_engineering_task_runner_source_digest(),
            "task_id": task.task_id,
            "task_plan_digest": task.record_digest,
            "execution_precommit_digest": _address(
                execution_precommit_digest, "execution precommit digest"
            ),
            "proposer_artifact_digest": proposer.artifact_digest,
            "proposer_raw_evidence_digest": proposer.raw_proposer_evidence_digest,
            "vocabulary_digest": table.vocabulary.vocabulary_digest,
            "observer_contract_digest": table.contract.contract_digest,
            "support_panel_ids": table.panel_ids,
            "support_panel_png_digests": table.panel_png_digests,
            "support_observer_artifact_digests": tuple(
                item.artifact_digest for item in artifacts
            ),
            "support_table_digest": table.table_digest,
            "engineering_version_space_digest": (
                space.engineering_version_space_digest
            ),
            "selection_mode": pair.selection_mode,
            "allow_unverified_rank_artifact": allow_unverified_rank_artifact,
            "rank_artifact_benchmark_sealable": (
                None
                if rank is None
                else rank.transport_provenance.benchmark_sealable
            ),
            "rank_artifact_digest": (
                None if rank is None else rank.artifact_digest
            ),
            "rank_input_digest": (
                None if rank is None else rank.rank_input.rank_input_digest
            ),
            "rank_receipt_digest": (
                None if rank is None else rank.receipt.receipt_digest
            ),
            "predicate_pair": pair,
            "version_space_digest": (
                space.engineering_version_space_digest
            ),
            "support_version_space_digest": (
                space.engineering_version_space_digest
            ),
            "selected_predicate_digest": pair.predicate_pair_digest,
            "sealed_query_panel_ids": query_ids,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(**values, record_digest=_content_address(_freeze_content(provisional)))

    def to_data(self) -> dict[str, object]:
        return {**_freeze_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: object) -> "PanelSoftEngineeringTaskFreeze":
        raw = _fields(
            value,
            {
                "schema", "runner_id", "runner_source_digest", "task_id",
                "task_plan_digest", "execution_precommit_digest",
                "proposer_artifact_digest", "proposer_raw_evidence_digest",
                "vocabulary_digest", "observer_contract_digest",
                "support_panel_ids", "support_panel_png_digests",
                "support_observer_artifact_digests", "support_table_digest",
                "engineering_version_space_digest", "selection_mode",
                "allow_unverified_rank_artifact",
                "rank_artifact_benchmark_sealable",
                "rank_artifact_digest", "rank_input_digest",
                "rank_receipt_digest", "predicate_pair",
                "predicate_pair_digest", "version_space_digest",
                "support_version_space_digest",
                "selected_predicate_digest",
                "selected_predicate_is_two_orientation_pair",
                "sealed_query_panel_ids", "query_pixels_included",
                "query_observer_artifacts_included",
                "exact_support_pixels_verified_before_freeze",
                "complete_vocabulary_observed_twice_per_support_panel",
                "predicate_pair_frozen_before_query_source",
                *_authority_data(), "record_digest",
            },
            "panel-soft task freeze",
        )
        for name in (
            "support_panel_ids", "support_panel_png_digests",
            "support_observer_artifact_digests", "sealed_query_panel_ids",
        ):
            if not isinstance(raw[name], list):
                raise PanelSoftEngineeringTaskRunnerError(f"freeze {name} differs")
        pair = PanelSoftEngineeringPredicatePair.from_data(raw["predicate_pair"])
        if (
            raw["schema"] != PANEL_SOFT_ENGINEERING_TASK_FREEZE_SCHEMA
            or raw["runner_id"] != PANEL_SOFT_ENGINEERING_TASK_RUNNER_ID
            or raw["predicate_pair_digest"] != pair.predicate_pair_digest
            or raw["selected_predicate_is_two_orientation_pair"] is not True
            or raw["query_pixels_included"] is not False
            or raw["query_observer_artifacts_included"] is not False
            or raw["exact_support_pixels_verified_before_freeze"] is not True
            or raw["complete_vocabulary_observed_twice_per_support_panel"] is not True
            or raw["predicate_pair_frozen_before_query_source"] is not True
            or any(raw[key] != item for key, item in _authority_data().items())
        ):
            raise PanelSoftEngineeringTaskRunnerError("task freeze policy differs")
        result = cls(
            raw["runner_source_digest"], raw["task_id"], raw["task_plan_digest"],
            raw["execution_precommit_digest"], raw["proposer_artifact_digest"],
            raw["proposer_raw_evidence_digest"], raw["vocabulary_digest"],
            raw["observer_contract_digest"], tuple(raw["support_panel_ids"]),
            tuple(raw["support_panel_png_digests"]),
            tuple(raw["support_observer_artifact_digests"]),
            raw["support_table_digest"], raw["engineering_version_space_digest"],
            raw["selection_mode"], raw["allow_unverified_rank_artifact"],
            raw["rank_artifact_benchmark_sealable"],
            raw["rank_artifact_digest"],
            raw["rank_input_digest"], raw["rank_receipt_digest"], pair,
            raw["version_space_digest"], raw["support_version_space_digest"],
            raw["selected_predicate_digest"],
            tuple(raw["sealed_query_panel_ids"]), raw["record_digest"],
        )
        if result.to_data() != dict(raw):
            raise PanelSoftEngineeringTaskRunnerError("task freeze is not canonical")
        return result


def _commit_content(value: "PanelSoftEngineeringTaskFreezeCommit") -> dict[str, object]:
    return {
        "schema": PANEL_SOFT_ENGINEERING_TASK_FREEZE_COMMIT_SCHEMA,
        "task_id": value.task_id,
        "task_plan_digest": value.task_plan_digest,
        "execution_precommit_digest": value.execution_precommit_digest,
        "selection_mode": value.selection_mode,
        "allow_unverified_rank_artifact": value.allow_unverified_rank_artifact,
        "rank_artifact_benchmark_sealable": value.rank_artifact_benchmark_sealable,
        "predicate_pair_digest": value.predicate_pair_digest,
        "version_space_digest": value.version_space_digest,
        "support_version_space_digest": value.support_version_space_digest,
        "selected_predicate_digest": value.selected_predicate_digest,
        "task_freeze_digest": value.task_freeze_digest,
        "exact_freeze_payload_digest": value.exact_freeze_payload_digest,
        "task_freeze_store_receipt_digest": value.task_freeze_store_receipt_digest,
        "commit_callback_completed_before_query_source": True,
        "external_storage_durability_authenticated_by_runner": False,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class PanelSoftEngineeringTaskFreezeCommit:
    task_id: str
    task_plan_digest: str
    execution_precommit_digest: str
    selection_mode: str
    allow_unverified_rank_artifact: bool
    rank_artifact_benchmark_sealable: bool | None
    predicate_pair_digest: str
    version_space_digest: str
    support_version_space_digest: str
    selected_predicate_digest: str
    task_freeze_digest: str
    exact_freeze_payload_digest: str
    task_freeze_store_receipt_digest: str
    record_digest: str

    def __post_init__(self) -> None:
        _identifier(self.task_id, "commit task ID")
        _address(self.task_plan_digest, "commit task plan digest")
        _address(self.execution_precommit_digest, "commit precommit digest")
        mode = _selection_mode(self.selection_mode)
        _rank_override(mode, self.allow_unverified_rank_artifact)
        if self.rank_artifact_benchmark_sealable is not None and type(
            self.rank_artifact_benchmark_sealable
        ) is not bool:
            raise TypeError("rank_artifact_benchmark_sealable must be bool or None")
        for name in (
            "predicate_pair_digest", "version_space_digest",
            "support_version_space_digest", "selected_predicate_digest",
        ):
            _raw_digest(getattr(self, name), name)
        for name in (
            "task_freeze_digest", "exact_freeze_payload_digest",
            "task_freeze_store_receipt_digest", "record_digest",
        ):
            _address(getattr(self, name), name)
        if (
            self.version_space_digest != self.support_version_space_digest
            or self.predicate_pair_digest != self.selected_predicate_digest
            or (
                mode == "support_only_codex_ranker"
                and type(self.rank_artifact_benchmark_sealable) is not bool
            )
            or (
                mode == "deterministic_baseline"
                and self.rank_artifact_benchmark_sealable is not None
            )
            or (
                self.rank_artifact_benchmark_sealable is False
                and not self.allow_unverified_rank_artifact
            )
            or self.record_digest != _content_address(_commit_content(self))
        ):
            raise PanelSoftEngineeringTaskRunnerError("freeze commit differs")

    @classmethod
    def seal(
        cls,
        freeze: PanelSoftEngineeringTaskFreeze,
        exact_freeze_payload: bytes,
        *,
        task_freeze_store_receipt_digest: str,
    ) -> "PanelSoftEngineeringTaskFreezeCommit":
        if not isinstance(freeze, PanelSoftEngineeringTaskFreeze):
            raise TypeError("freeze must be PanelSoftEngineeringTaskFreeze")
        expected = canonical_json(freeze.to_data()) + b"\n"
        if exact_freeze_payload != expected:
            raise PanelSoftEngineeringTaskRunnerError(
                "freeze payload bytes are not exact canonical JSON"
            )
        values = {
            "task_id": freeze.task_id,
            "task_plan_digest": freeze.task_plan_digest,
            "execution_precommit_digest": freeze.execution_precommit_digest,
            "selection_mode": freeze.selection_mode,
            "allow_unverified_rank_artifact": (
                freeze.allow_unverified_rank_artifact
            ),
            "rank_artifact_benchmark_sealable": (
                freeze.rank_artifact_benchmark_sealable
            ),
            "predicate_pair_digest": freeze.predicate_pair.predicate_pair_digest,
            "version_space_digest": freeze.version_space_digest,
            "support_version_space_digest": freeze.support_version_space_digest,
            "selected_predicate_digest": freeze.selected_predicate_digest,
            "task_freeze_digest": freeze.record_digest,
            "exact_freeze_payload_digest": (
                "sha256:" + hashlib.sha256(expected).hexdigest()
            ),
            "task_freeze_store_receipt_digest": _address(
                task_freeze_store_receipt_digest, "task freeze store receipt digest"
            ),
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(**values, record_digest=_content_address(_commit_content(provisional)))

    def assert_matches(
        self,
        freeze: PanelSoftEngineeringTaskFreeze,
        exact_freeze_payload: bytes,
    ) -> None:
        replayed = type(self).seal(
            freeze,
            exact_freeze_payload,
            task_freeze_store_receipt_digest=self.task_freeze_store_receipt_digest,
        )
        if self != replayed:
            raise PanelSoftEngineeringTaskRunnerError("freeze commit replay differs")

    def to_data(self) -> dict[str, object]:
        return {**_commit_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: object) -> "PanelSoftEngineeringTaskFreezeCommit":
        raw = _fields(
            value,
            {
                "schema", "task_id", "task_plan_digest",
                "execution_precommit_digest", "selection_mode",
                "allow_unverified_rank_artifact",
                "rank_artifact_benchmark_sealable", "predicate_pair_digest",
                "version_space_digest", "support_version_space_digest",
                "selected_predicate_digest", "task_freeze_digest",
                "exact_freeze_payload_digest", "task_freeze_store_receipt_digest",
                "commit_callback_completed_before_query_source",
                "external_storage_durability_authenticated_by_runner",
                *_authority_data(),
                "record_digest",
            },
            "panel-soft freeze commit",
        )
        if (
            raw["schema"] != PANEL_SOFT_ENGINEERING_TASK_FREEZE_COMMIT_SCHEMA
            or raw["commit_callback_completed_before_query_source"] is not True
            or raw["external_storage_durability_authenticated_by_runner"] is not False
            or any(raw[key] != item for key, item in _authority_data().items())
        ):
            raise PanelSoftEngineeringTaskRunnerError("freeze commit policy differs")
        result = cls(
            raw["task_id"], raw["task_plan_digest"], raw["execution_precommit_digest"],
            raw["selection_mode"],
            raw["allow_unverified_rank_artifact"],
            raw["rank_artifact_benchmark_sealable"],
            raw["predicate_pair_digest"], raw["version_space_digest"],
            raw["support_version_space_digest"], raw["selected_predicate_digest"],
            raw["task_freeze_digest"],
            raw["exact_freeze_payload_digest"],
            raw["task_freeze_store_receipt_digest"], raw["record_digest"],
        )
        if result.to_data() != dict(raw):
            raise PanelSoftEngineeringTaskRunnerError("freeze commit is not canonical")
        return result


def _expected_support_ids(task: ObjectBongardTaskPlan) -> tuple[str, ...]:
    return (*task.side_0_support_panel_ids, *task.side_1_support_panel_ids)


def _canonical_png_map(
    values: Mapping[str, bytes], expected_ids: Sequence[str], *, label: str
) -> tuple[bytes, ...]:
    if (
        not isinstance(values, Mapping)
        or any(not isinstance(key, str) for key in values)
        or set(values) != set(expected_ids)
    ):
        raise PanelSoftEngineeringTaskRunnerError(f"{label} PNG inventory differs")
    return tuple(_png(values[item], f"{label} panel") for item in expected_ids)


def _runtime_tuple(value: object) -> tuple[object, ...]:
    return tuple(
        getattr(value, name)
        for name in (
            "model", "reasoning_effort", "expected_launcher_digest",
            "cloud_policy_cache_binding", "model_catalog_digest",
            "no_tools_attestation_digest",
        )
    )


def _require_proposer_observer_call_distinctness(
    proposer: PanelSoftProposerArtifact,
    artifacts: Sequence[PanelSoftObserverArtifact],
) -> None:
    """Keep the proposer turn distinct from every receipted observer repeat."""

    if proposer.receipt is None:
        raise PanelSoftEngineeringTaskRunnerError(
            "successful proposer artifact lacks a receipted call"
        )
    observer_receipts = tuple(
        repeat.receipt
        for artifact in artifacts
        for repeat in artifact.repeats
        if repeat.receipt is not None
    )
    if (
        proposer.receipt.receipt_digest
        in {item.receipt_digest for item in observer_receipts}
        or proposer.receipt.thread_id in {item.thread_id for item in observer_receipts}
    ):
        raise PanelSoftEngineeringTaskRunnerError(
            "proposer model-call identity is reused by an observer repeat"
        )


def _require_rank_call_distinctness(
    rank_artifact: PanelSoftRankArtifact | None,
    proposer: PanelSoftProposerArtifact,
    artifacts: Sequence[PanelSoftObserverArtifact],
) -> None:
    if rank_artifact is None:
        return
    if proposer.receipt is None:
        raise PanelSoftEngineeringTaskRunnerError(
            "rank call cannot be compared with a missing proposer receipt"
        )
    other_receipts = (
        proposer.receipt,
        *(
            repeat.receipt
            for artifact in artifacts
            for repeat in artifact.repeats
            if repeat.receipt is not None
        ),
    )
    if (
        rank_artifact.receipt.receipt_digest
        in {item.receipt_digest for item in other_receipts}
        or rank_artifact.receipt.thread_id
        in {item.thread_id for item in other_receipts}
    ):
        raise PanelSoftEngineeringTaskRunnerError(
            "ranker model-call identity is reused by proposer or observer"
        )


def _verified_support(
    task: ObjectBongardTaskPlan,
    proposer_artifact: PanelSoftProposerArtifact,
    support_png_by_panel_id: Mapping[str, bytes],
    support_artifacts: Sequence[PanelSoftObserverArtifact],
) -> tuple[
    PanelSoftProposerArtifact,
    tuple[bytes, ...],
    tuple[PanelSoftObserverArtifact, ...],
    PanelSoftObservationTable,
]:
    proposer = _canonical_proposer(proposer_artifact)
    expected_ids = _expected_support_ids(task)
    pngs = _canonical_png_map(
        support_png_by_panel_id, expected_ids, label="support"
    )
    if (
        proposer.status is not PanelSoftProposerStatus.SUCCESS
        or proposer.vocabulary is None
        or proposer.raw_proposer_evidence_digest is None
        or proposer.support_panel_ids != expected_ids
    ):
        raise PanelSoftEngineeringTaskRunnerError(
            "task runner requires a successful exact-plan proposer artifact"
        )
    verify_panel_soft_proposer_artifact(
        proposer,
        pngs,
        support_panel_ids=expected_ids,
        expected_artifact_digest=proposer.artifact_digest,
    )
    artifacts = tuple(_canonical_observer(item) for item in support_artifacts)
    if (
        len(artifacts) != PANEL_SOFT_ENGINEERING_SUPPORT_PANEL_COUNT
        or tuple(item.panel_id for item in artifacts) != expected_ids
        or any(item.vocabulary != proposer.vocabulary for item in artifacts)
        or any(_runtime_tuple(item) != _runtime_tuple(proposer) for item in artifacts)
    ):
        raise PanelSoftEngineeringTaskRunnerError(
            "support observer inventory, vocabulary, or runtime differs"
        )
    contract = artifacts[0].contract
    for artifact, panel, panel_id in zip(artifacts, pngs, expected_ids, strict=True):
        verify_panel_soft_observer_artifact(
            artifact,
            panel,
            panel_id=panel_id,
            vocabulary=proposer.vocabulary,
            expected_artifact_digest=artifact.artifact_digest,
            expected_contract_digest=contract.contract_digest,
        )
    table = aggregate_panel_soft_observer_artifacts(
        artifacts,
        ordered_panel_commitments=tuple(
            (panel_id, hashlib.sha256(panel).hexdigest())
            for panel_id, panel in zip(expected_ids, pngs, strict=True)
        ),
        expected_vocabulary=proposer.vocabulary,
        expected_contract=contract,
    )
    _require_proposer_observer_call_distinctness(proposer, artifacts)
    return proposer, pngs, artifacts, table


def _archive_content(value: "PanelSoftEngineeringTaskRunArchive") -> dict[str, object]:
    return {
        "schema": PANEL_SOFT_ENGINEERING_TASK_ARCHIVE_SCHEMA,
        "runner_id": PANEL_SOFT_ENGINEERING_TASK_RUNNER_ID,
        "runner_source_digest": value.runner_source_digest,
        "task_plan": value.task_plan.to_data(),
        "execution_precommit_digest": value.execution_precommit_digest,
        "proposer_artifact": value.proposer_artifact.to_data(),
        "support_png_base64_by_panel_id": {
            panel_id: encoded
            for panel_id, encoded in value.support_png_base64_by_panel_id
        },
        "support_artifacts": [item.to_data() for item in value.support_artifacts],
        "support_table": value.support_table.to_data(),
        "engineering_version_space": value.engineering_version_space.to_data(),
        "selection_mode": value.selection_mode,
        "allow_unverified_rank_artifact": value.allow_unverified_rank_artifact,
        "rank_artifact_benchmark_sealable": value.rank_artifact_benchmark_sealable,
        "rank_artifact": (
            None if value.rank_artifact is None else value.rank_artifact.to_data()
        ),
        "status": value.status.value,
        "support_gap": None if value.support_gap is None else value.support_gap.to_data(),
        "predicate_pair": None if value.predicate_pair is None else value.predicate_pair.to_data(),
        "freeze": None if value.freeze is None else value.freeze.to_data(),
        "freeze_commit": None if value.freeze_commit is None else value.freeze_commit.to_data(),
        "query_png_base64_by_side": {
            side: encoded for side, encoded in value.query_png_base64_by_side
        },
        "query_artifacts": [item.to_data() for item in value.query_artifacts],
        "query_decisions": [item.to_data() for item in value.query_decisions],
        "correct_count": value.correct_count,
        "determinate_count": value.determinate_count,
        "abstain_count": value.abstain_count,
        "error_count": value.error_count,
        "query_denominator": PANEL_SOFT_ENGINEERING_QUERY_DENOMINATOR,
        "accuracy_ppm": value.accuracy_ppm,
        "coverage_ppm": value.coverage_ppm,
        "ranker_callback_invocations": value.ranker_callback_invocations,
        "freeze_commit_calls_made": value.freeze_commit_calls_made,
        "freeze_reload_calls_made": value.freeze_reload_calls_made,
        "query_source_calls_made": value.query_source_calls_made,
        "query_source_called_only_after_exact_freeze_reload": True,
        "exact_released_pngs_archived_for_cold_replay": True,
        "cold_replay_model_calls": 0,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class PanelSoftEngineeringTaskRunArchive:
    runner_source_digest: str
    task_plan: ObjectBongardTaskPlan
    execution_precommit_digest: str
    proposer_artifact: PanelSoftProposerArtifact
    support_png_base64_by_panel_id: tuple[tuple[str, str], ...]
    support_artifacts: tuple[PanelSoftObserverArtifact, ...]
    support_table: PanelSoftObservationTable
    engineering_version_space: PanelSoftEngineeringVersionSpace
    selection_mode: str
    allow_unverified_rank_artifact: bool
    rank_artifact_benchmark_sealable: bool | None
    rank_artifact: PanelSoftRankArtifact | None
    status: PanelSoftEngineeringTaskRunStatus
    support_gap: PanelSoftEngineeringSupportGap | None
    predicate_pair: PanelSoftEngineeringPredicatePair | None
    freeze: PanelSoftEngineeringTaskFreeze | None
    freeze_commit: PanelSoftEngineeringTaskFreezeCommit | None
    query_png_base64_by_side: tuple[tuple[str, str], ...]
    query_artifacts: tuple[PanelSoftObserverArtifact, ...]
    query_decisions: tuple[PanelSoftEngineeringQueryDecision, ...]
    correct_count: int
    determinate_count: int
    abstain_count: int
    error_count: int
    accuracy_ppm: int
    coverage_ppm: int
    ranker_callback_invocations: int
    freeze_commit_calls_made: int
    freeze_reload_calls_made: int
    query_source_calls_made: int
    record_digest: str

    def __post_init__(self) -> None:
        _raw_digest(self.runner_source_digest, "archive runner source digest")
        _address(self.execution_precommit_digest, "archive precommit digest")
        _raw_digest(self.record_digest, "task archive digest")
        mode = _selection_mode(self.selection_mode)
        if type(self.allow_unverified_rank_artifact) is not bool:
            raise TypeError("allow_unverified_rank_artifact must be bool")
        rank = (
            None
            if self.rank_artifact is None
            else _canonical_rank_artifact(
                self.rank_artifact, self.engineering_version_space
            )
        )
        expected_sealable = (
            None if rank is None else rank.transport_provenance.benchmark_sealable
        )
        if (
            self.rank_artifact_benchmark_sealable != expected_sealable
            or (
                expected_sealable is False
                and not self.allow_unverified_rank_artifact
            )
        ):
            raise PanelSoftEngineeringTaskRunnerError(
                "archive rank sealability policy differs"
            )
        pair_rank_lineage = (
            self.predicate_pair is not None
            and self.predicate_pair.selection_mode == mode
            and (
                (mode == "deterministic_baseline" and rank is None)
                or (
                    mode == "support_only_codex_ranker"
                    and rank is not None
                    and rank.selected_formula_digests
                    == (
                        self.predicate_pair.side0_formula_digest,
                        self.predicate_pair.side1_formula_digest,
                    )
                )
            )
        )
        ranked_shape = (
            mode == "support_only_codex_ranker"
            and self.rank_artifact is not None
            and self.ranker_callback_invocations == 1
        )
        baseline_shape = (
            mode == "deterministic_baseline"
            and self.rank_artifact is None
            and self.ranker_callback_invocations == 0
        )
        complete_shape = (
            self.support_gap is None
            and self.predicate_pair is not None
            and self.freeze is not None
            and self.freeze_commit is not None
            and len(self.query_png_base64_by_side) == 2
            and len(self.query_artifacts) == 2
            and len(self.query_decisions) == 2
            and (
                self.freeze_commit_calls_made,
                self.freeze_reload_calls_made,
                self.query_source_calls_made,
            ) == (1, 1, 1)
            and (ranked_shape or baseline_shape)
            and pair_rank_lineage
        )
        gap_shape = (
            self.support_gap is not None
            and self.predicate_pair is None
            and self.freeze is None
            and self.freeze_commit is None
            and not self.query_png_base64_by_side
            and not self.query_artifacts
            and not self.query_decisions
            and self.rank_artifact is None
            and self.ranker_callback_invocations == 0
            and (
                self.freeze_commit_calls_made,
                self.freeze_reload_calls_made,
                self.query_source_calls_made,
            ) == (0, 0, 0)
        )
        if (
            not isinstance(self.status, PanelSoftEngineeringTaskRunStatus)
            or any(
                type(item) is not int or item < 0
                for item in (
                    self.correct_count, self.determinate_count, self.abstain_count,
                    self.error_count, self.accuracy_ppm, self.coverage_ppm,
                    self.ranker_callback_invocations,
                    self.freeze_commit_calls_made, self.freeze_reload_calls_made,
                    self.query_source_calls_made,
                )
            )
            or self.correct_count > PANEL_SOFT_ENGINEERING_QUERY_DENOMINATOR
            or self.correct_count > self.determinate_count
            or self.determinate_count + self.abstain_count + self.error_count
            != PANEL_SOFT_ENGINEERING_QUERY_DENOMINATOR
            or self.accuracy_ppm != self.correct_count * 500_000
            or self.coverage_ppm != self.determinate_count * 500_000
            or (
                self.status is PanelSoftEngineeringTaskRunStatus.COMPLETE
                and not complete_shape
            )
            or (
                self.status is not PanelSoftEngineeringTaskRunStatus.COMPLETE
                and not gap_shape
            )
            or self.record_digest != canonical_digest(_archive_content(self))
        ):
            raise PanelSoftEngineeringTaskRunnerError("task archive counters differ")

    def to_data(self) -> dict[str, object]:
        return {**_archive_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: object) -> "PanelSoftEngineeringTaskRunArchive":
        raw = _fields(
            value,
            {
                "schema", "runner_id", "runner_source_digest", "task_plan",
                "execution_precommit_digest", "proposer_artifact",
                "support_png_base64_by_panel_id", "support_artifacts",
                "support_table", "engineering_version_space", "selection_mode",
                "allow_unverified_rank_artifact",
                "rank_artifact_benchmark_sealable", "rank_artifact", "status",
                "support_gap", "predicate_pair", "freeze", "freeze_commit",
                "query_png_base64_by_side", "query_artifacts", "query_decisions",
                "correct_count", "determinate_count", "abstain_count", "error_count",
                "query_denominator", "accuracy_ppm", "coverage_ppm",
                "ranker_callback_invocations", "freeze_commit_calls_made", "freeze_reload_calls_made",
                "query_source_calls_made",
                "query_source_called_only_after_exact_freeze_reload",
                "exact_released_pngs_archived_for_cold_replay",
                "cold_replay_model_calls", *_authority_data(), "record_digest",
            },
            "panel-soft task archive",
        )
        if (
            raw["schema"] != PANEL_SOFT_ENGINEERING_TASK_ARCHIVE_SCHEMA
            or raw["runner_id"] != PANEL_SOFT_ENGINEERING_TASK_RUNNER_ID
            or raw["query_denominator"] != PANEL_SOFT_ENGINEERING_QUERY_DENOMINATOR
            or raw["query_source_called_only_after_exact_freeze_reload"] is not True
            or raw["exact_released_pngs_archived_for_cold_replay"] is not True
            or raw["cold_replay_model_calls"] != 0
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["support_png_base64_by_panel_id"], Mapping)
            or not isinstance(raw["query_png_base64_by_side"], Mapping)
            or not isinstance(raw["support_artifacts"], list)
            or not isinstance(raw["query_artifacts"], list)
            or not isinstance(raw["query_decisions"], list)
        ):
            raise PanelSoftEngineeringTaskRunnerError("task archive policy differs")
        task = ObjectBongardTaskPlan.from_data(raw["task_plan"])
        support_ids = _expected_support_ids(task)
        support_encoded = raw["support_png_base64_by_panel_id"]
        query_encoded = raw["query_png_base64_by_side"]
        if set(support_encoded) != set(support_ids) or set(query_encoded) not in (
            set(), {"side_0", "side_1"}
        ):
            raise PanelSoftEngineeringTaskRunnerError("archive PNG keys differ")
        try:
            status = PanelSoftEngineeringTaskRunStatus(raw["status"])
        except (TypeError, ValueError) as exc:
            raise PanelSoftEngineeringTaskRunnerError("archive status differs") from exc
        result = cls(
            raw["runner_source_digest"], task, raw["execution_precommit_digest"],
            PanelSoftProposerArtifact.from_data(raw["proposer_artifact"]),
            tuple((item, support_encoded[item]) for item in support_ids),
            tuple(PanelSoftObserverArtifact.from_data(item) for item in raw["support_artifacts"]),
            PanelSoftObservationTable.from_data(raw["support_table"]),
            PanelSoftEngineeringVersionSpace.from_data(raw["engineering_version_space"]),
            raw["selection_mode"],
            raw["allow_unverified_rank_artifact"],
            raw["rank_artifact_benchmark_sealable"],
            None if raw["rank_artifact"] is None else PanelSoftRankArtifact.from_data(raw["rank_artifact"]),
            status,
            None if raw["support_gap"] is None else PanelSoftEngineeringSupportGap.from_data(raw["support_gap"]),
            None if raw["predicate_pair"] is None else PanelSoftEngineeringPredicatePair.from_data(raw["predicate_pair"]),
            None if raw["freeze"] is None else PanelSoftEngineeringTaskFreeze.from_data(raw["freeze"]),
            None if raw["freeze_commit"] is None else PanelSoftEngineeringTaskFreezeCommit.from_data(raw["freeze_commit"]),
            tuple((side, query_encoded[side]) for side in ("side_0", "side_1") if side in query_encoded),
            tuple(PanelSoftObserverArtifact.from_data(item) for item in raw["query_artifacts"]),
            tuple(PanelSoftEngineeringQueryDecision.from_data(item) for item in raw["query_decisions"]),
            raw["correct_count"], raw["determinate_count"], raw["abstain_count"],
            raw["error_count"], raw["accuracy_ppm"], raw["coverage_ppm"],
            raw["ranker_callback_invocations"],
            raw["freeze_commit_calls_made"], raw["freeze_reload_calls_made"],
            raw["query_source_calls_made"], raw["record_digest"],
        )
        if result.to_data() != dict(raw):
            raise PanelSoftEngineeringTaskRunnerError("task archive is not canonical")
        return result


FreezeCommitter = Callable[
    [bytes], PanelSoftEngineeringTaskFreezeCommit | Mapping[str, Any]
]
FreezeReloader = Callable[[Mapping[str, Any]], bytes]
QuerySource = Callable[
    [Mapping[str, Any], Mapping[str, Any]],
    Mapping[str, tuple[bytes, PanelSoftObserverArtifact]],
]
PanelSoftRanker = Callable[
    [PanelSoftEngineeringVersionSpace], PanelSoftRankArtifact | Mapping[str, Any]
]


def _make_archive(**values: object) -> PanelSoftEngineeringTaskRunArchive:
    provisional = object.__new__(PanelSoftEngineeringTaskRunArchive)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return PanelSoftEngineeringTaskRunArchive(
        **values,  # type: ignore[arg-type]
        record_digest=canonical_digest(_archive_content(provisional)),
    )


def _encode_png_rows(
    ids: Sequence[str], pngs: Sequence[bytes]
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (panel_id, base64.b64encode(panel).decode("ascii"))
        for panel_id, panel in zip(ids, pngs, strict=True)
    )


def _decode_png_rows(
    rows: Sequence[tuple[str, str]], expected_ids: Sequence[str], *, label: str
) -> dict[str, bytes]:
    if tuple(item[0] for item in rows) != tuple(expected_ids):
        raise PanelSoftEngineeringTaskRunnerError(f"{label} archived PNG order differs")
    result: dict[str, bytes] = {}
    try:
        for panel_id, encoded in rows:
            result[panel_id] = _png(
                base64.b64decode(encoded, validate=True), f"{label} archived panel"
            )
    except (TypeError, ValueError) as exc:
        raise PanelSoftEngineeringTaskRunnerError(
            f"{label} archived PNG encoding differs"
        ) from exc
    return result


def _make_proposer_terminal(
    *,
    task: ObjectBongardTaskPlan,
    precommit: str,
    selection_mode: str,
    allow_unverified_rank_artifact: bool,
    proposer: PanelSoftProposerArtifact,
    support_pngs: Sequence[bytes],
) -> PanelSoftEngineeringProposerTerminal:
    values = {
        "runner_source_digest": panel_soft_engineering_task_runner_source_digest(),
        "task_plan": task,
        "execution_precommit_digest": precommit,
        "selection_mode": _selection_mode(selection_mode),
        "allow_unverified_rank_artifact": allow_unverified_rank_artifact,
        "rank_artifact_benchmark_sealable": None,
        "proposer_artifact": proposer,
        "support_png_base64_by_panel_id": _encode_png_rows(
            _expected_support_ids(task), support_pngs
        ),
    }
    provisional = object.__new__(PanelSoftEngineeringProposerTerminal)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return PanelSoftEngineeringProposerTerminal(
        **values,
        record_digest=canonical_digest(_proposer_terminal_content(provisional)),
    )


def _cold_replay_proposer_terminal(
    terminal: PanelSoftEngineeringProposerTerminal,
    *,
    expected_record_digest: str,
) -> PanelSoftEngineeringProposerTerminal:
    restored = PanelSoftEngineeringProposerTerminal.from_data(terminal.to_data())
    if restored.record_digest != _raw_digest(
        expected_record_digest, "expected proposer terminal digest"
    ):
        raise PanelSoftEngineeringTaskRunnerError(
            "proposer terminal differs from commitment"
        )
    task = _canonical_task(restored.task_plan)
    support_map = _decode_png_rows(
        restored.support_png_base64_by_panel_id,
        _expected_support_ids(task),
        label="support",
    )
    pngs = tuple(support_map[item] for item in _expected_support_ids(task))
    proposer = verify_panel_soft_proposer_artifact(
        restored.proposer_artifact,
        pngs,
        support_panel_ids=_expected_support_ids(task),
        expected_artifact_digest=restored.proposer_artifact.artifact_digest,
    )
    if (
        proposer.status is PanelSoftProposerStatus.SUCCESS
        or proposer.vocabulary is not None
    ):
        raise PanelSoftEngineeringTaskRunnerError(
            "proposer terminal fabricates a successful proposal"
        )
    return restored


def _query_metrics(
    decisions: Sequence[PanelSoftEngineeringQueryDecision],
) -> tuple[int, int, int, int]:
    outcomes = tuple(item.outcome for item in decisions)
    expected = (
        PanelSoftEngineeringQueryOutcome.SIDE0,
        PanelSoftEngineeringQueryOutcome.SIDE1,
    )
    return (
        sum(got is want for got, want in zip(outcomes, expected, strict=True)),
        sum(got in expected for got in outcomes),
        sum(got is PanelSoftEngineeringQueryOutcome.ABSTAIN for got in outcomes),
        sum(got is PanelSoftEngineeringQueryOutcome.ERROR for got in outcomes),
    )


def run_panel_soft_engineering_task(
    task_plan: ObjectBongardTaskPlan,
    proposer_artifact: PanelSoftProposerArtifact,
    support_png_by_panel_id: Mapping[str, bytes],
    support_artifacts: Sequence[PanelSoftObserverArtifact],
    *,
    execution_precommit_digest: str,
    selection_mode: str,
    ranker: PanelSoftRanker | None,
    allow_unverified_rank_artifact: bool = False,
    freeze_committer: FreezeCommitter,
    freeze_reloader: FreezeReloader,
    query_source: QuerySource,
) -> PanelSoftEngineeringTaskRunArchive | PanelSoftEngineeringProposerTerminal:
    """Run support synthesis, exact freeze custody, then two query decisions."""

    task = _canonical_task(task_plan)
    precommit = _address(execution_precommit_digest, "execution precommit digest")
    mode = _selection_mode(selection_mode)
    _rank_override(mode, allow_unverified_rank_artifact)
    proposer = _canonical_proposer(proposer_artifact)
    expected_support_ids = _expected_support_ids(task)
    support_pngs = _canonical_png_map(
        support_png_by_panel_id, expected_support_ids, label="support"
    )
    verify_panel_soft_proposer_artifact(
        proposer,
        support_pngs,
        support_panel_ids=expected_support_ids,
        expected_artifact_digest=proposer.artifact_digest,
    )
    if proposer.status is not PanelSoftProposerStatus.SUCCESS:
        if support_artifacts:
            raise PanelSoftEngineeringTaskRunnerError(
                "support observations exist after a failed proposer"
            )
        terminal = _make_proposer_terminal(
            task=task,
            precommit=precommit,
            selection_mode=mode,
            allow_unverified_rank_artifact=allow_unverified_rank_artifact,
            proposer=proposer,
            support_pngs=support_pngs,
        )
        return _cold_replay_proposer_terminal(
            terminal, expected_record_digest=terminal.record_digest
        )
    proposer, support_pngs, artifacts, table = _verified_support(
        task, proposer, support_png_by_panel_id, support_artifacts
    )
    space = PanelSoftEngineeringVersionSpace.create(
        table, task.side_0_support_panel_ids, task.side_1_support_panel_ids
    )
    common = {
        "runner_source_digest": panel_soft_engineering_task_runner_source_digest(),
        "task_plan": task,
        "execution_precommit_digest": precommit,
        "proposer_artifact": proposer,
        "support_png_base64_by_panel_id": _encode_png_rows(
            _expected_support_ids(task), support_pngs
        ),
        "support_artifacts": artifacts,
        "support_table": table,
        "engineering_version_space": space,
        "selection_mode": mode,
        "allow_unverified_rank_artifact": allow_unverified_rank_artifact,
    }
    survivor_counts = tuple(
        sum(item.orientation == orientation for item in space.survivor_formulas)
        for orientation in PANEL_SOFT_ORIENTATIONS
    )
    if any(count == 0 for count in survivor_counts):
        gap = PanelSoftEngineeringSupportGap.create(space)
        status = (
            PanelSoftEngineeringTaskRunStatus.SUPPORT_ERROR
            if gap.has_observer_error
            else PanelSoftEngineeringTaskRunStatus.SUPPORT_GAP
        )
        archive = _make_archive(
            **common, rank_artifact_benchmark_sealable=None,
            rank_artifact=None, ranker_callback_invocations=0,
            status=status, support_gap=gap, predicate_pair=None,
            freeze=None, freeze_commit=None, query_png_base64_by_side=(),
            query_artifacts=(), query_decisions=(), correct_count=0,
            determinate_count=0, abstain_count=0 if gap.has_observer_error else 2,
            error_count=2 if gap.has_observer_error else 0,
            accuracy_ppm=0, coverage_ppm=0, freeze_commit_calls_made=0,
            freeze_reload_calls_made=0, query_source_calls_made=0,
        )
        return cold_replay_panel_soft_engineering_task(
            archive, expected_record_digest=archive.record_digest
        )
    rank_artifact: PanelSoftRankArtifact | None
    if mode == "support_only_codex_ranker":
        if not callable(ranker):
            raise TypeError("ranked mode requires a ranker callback")
        try:
            raw_rank = ranker(space)
            rank_artifact = _canonical_rank_artifact(
                raw_rank
                if isinstance(raw_rank, PanelSoftRankArtifact)
                else PanelSoftRankArtifact.from_data(raw_rank),
                space,
            )
        except Exception as exc:
            raise PanelSoftEngineeringTaskRunnerError(
                "ranked selection failed; no baseline fallback was used"
            ) from exc
        if (
            not rank_artifact.transport_provenance.benchmark_sealable
            and not allow_unverified_rank_artifact
        ):
            raise PanelSoftEngineeringTaskRunnerError(
                "rank artifact is not benchmark-sealable; explicit engineering "
                "override required before freeze or query"
            )
        pair = PanelSoftEngineeringPredicatePair.create_ranked(
            space,
            side0_formula_digest=rank_artifact.selected_side0_formula_digest,
            side1_formula_digest=rank_artifact.selected_side1_formula_digest,
        )
        _require_rank_call_distinctness(rank_artifact, proposer, artifacts)
    else:
        if ranker is not None:
            raise PanelSoftEngineeringTaskRunnerError(
                "deterministic baseline mode cannot receive a ranker callback"
            )
        rank_artifact = None
        pair = PanelSoftEngineeringPredicatePair.create_deterministic_baseline(space)
    if not callable(freeze_committer) or not callable(freeze_reloader) or not callable(query_source):
        raise TypeError("complete task requires freeze and query callbacks")
    freeze = PanelSoftEngineeringTaskFreeze.seal(
        task_plan=task,
        execution_precommit_digest=precommit,
        proposer_artifact=proposer,
        support_artifacts=artifacts,
        predicate_pair=pair,
        rank_artifact=rank_artifact,
        allow_unverified_rank_artifact=allow_unverified_rank_artifact,
    )
    freeze_data = PanelSoftEngineeringTaskFreeze.from_data(freeze.to_data()).to_data()
    freeze_bytes = canonical_json(freeze_data) + b"\n"
    raw_commit = freeze_committer(freeze_bytes)
    commit = (
        raw_commit
        if isinstance(raw_commit, PanelSoftEngineeringTaskFreezeCommit)
        else PanelSoftEngineeringTaskFreezeCommit.from_data(raw_commit)
    )
    commit.assert_matches(freeze, freeze_bytes)
    reloaded = freeze_reloader(commit.to_data())
    if reloaded != freeze_bytes:
        raise PanelSoftEngineeringTaskRunnerError("freeze reload bytes differ")
    try:
        restored_freeze = PanelSoftEngineeringTaskFreeze.from_data(
            json.loads(reloaded.decode("utf-8", errors="strict"))
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PanelSoftEngineeringTaskRunnerError("freeze reload is not exact JSON") from exc
    if restored_freeze != freeze:
        raise PanelSoftEngineeringTaskRunnerError("reloaded freeze object differs")

    raw_queries = query_source(freeze_data, commit.to_data())
    if not isinstance(raw_queries, Mapping) or set(raw_queries) != {"side_0", "side_1"}:
        raise PanelSoftEngineeringTaskRunnerError(
            "query source must return exactly side_0 and side_1"
        )
    query_ids = (task.side_0_query_panel_id, task.side_1_query_panel_id)
    query_pngs: list[bytes] = []
    query_artifacts: list[PanelSoftObserverArtifact] = []
    for side, panel_id in zip(("side_0", "side_1"), query_ids, strict=True):
        row = raw_queries[side]
        if not isinstance(row, tuple) or len(row) != 2:
            raise PanelSoftEngineeringTaskRunnerError("query source row differs")
        panel = _png(row[0], "query panel")
        artifact = _canonical_observer(row[1])
        verify_panel_soft_observer_artifact(
            artifact, panel, panel_id=panel_id, vocabulary=table.vocabulary,
            expected_artifact_digest=artifact.artifact_digest,
            expected_contract_digest=table.contract.contract_digest,
        )
        verify_panel_soft_observer_contract_identity(artifacts[0], artifact)
        query_pngs.append(panel)
        query_artifacts.append(artifact)
    # The aggregator also enforces globally unique successful receipt/thread
    # identities across support and query observations.
    aggregate_panel_soft_observer_artifacts(
        (*artifacts, *query_artifacts),
        ordered_panel_commitments=tuple(
            (item.panel_id, item.panel_png_digest)
            for item in (*artifacts, *query_artifacts)
        ),
        expected_vocabulary=table.vocabulary,
        expected_contract=table.contract,
    )
    _require_proposer_observer_call_distinctness(
        proposer, (*artifacts, *query_artifacts)
    )
    _require_rank_call_distinctness(
        rank_artifact, proposer, (*artifacts, *query_artifacts)
    )
    decisions = tuple(
        PanelSoftEngineeringQueryDecision.create(pair, artifact.observation_table, panel_id)
        for artifact, panel_id in zip(query_artifacts, query_ids, strict=True)
    )
    correct, determinate, abstain, errors = _query_metrics(decisions)
    archive = _make_archive(
        **common,
        rank_artifact_benchmark_sealable=(
            None
            if rank_artifact is None
            else rank_artifact.transport_provenance.benchmark_sealable
        ),
        rank_artifact=rank_artifact,
        ranker_callback_invocations=1 if rank_artifact is not None else 0,
        status=PanelSoftEngineeringTaskRunStatus.COMPLETE,
        support_gap=None, predicate_pair=pair, freeze=freeze, freeze_commit=commit,
        query_png_base64_by_side=_encode_png_rows(("side_0", "side_1"), query_pngs),
        query_artifacts=tuple(query_artifacts), query_decisions=decisions,
        correct_count=correct, determinate_count=determinate,
        abstain_count=abstain, error_count=errors,
        accuracy_ppm=correct * 500_000, coverage_ppm=determinate * 500_000,
        freeze_commit_calls_made=1, freeze_reload_calls_made=1,
        query_source_calls_made=1,
    )
    return cold_replay_panel_soft_engineering_task(
        archive, expected_record_digest=archive.record_digest
    )


def cold_replay_panel_soft_engineering_task(
    archive: PanelSoftEngineeringTaskRunArchive | PanelSoftEngineeringProposerTerminal,
    *,
    expected_record_digest: str,
) -> PanelSoftEngineeringTaskRunArchive | PanelSoftEngineeringProposerTerminal:
    """Recompute the complete task archive from stored bytes with zero calls."""

    if isinstance(archive, PanelSoftEngineeringProposerTerminal):
        return _cold_replay_proposer_terminal(
            archive, expected_record_digest=expected_record_digest
        )
    if not isinstance(archive, PanelSoftEngineeringTaskRunArchive):
        raise TypeError("archive must be PanelSoftEngineeringTaskRunArchive")
    restored = PanelSoftEngineeringTaskRunArchive.from_data(archive.to_data())
    if restored.record_digest != _raw_digest(
        expected_record_digest, "expected task archive digest"
    ):
        raise PanelSoftEngineeringTaskRunnerError("task archive differs from commitment")
    if restored.runner_source_digest != panel_soft_engineering_task_runner_source_digest():
        raise PanelSoftEngineeringTaskRunnerError("task runner source differs on replay")
    task = _canonical_task(restored.task_plan)
    support_map = _decode_png_rows(
        restored.support_png_base64_by_panel_id,
        _expected_support_ids(task),
        label="support",
    )
    proposer, _support_pngs, artifacts, table = _verified_support(
        task, restored.proposer_artifact, support_map, restored.support_artifacts
    )
    space = PanelSoftEngineeringVersionSpace.create(
        table, task.side_0_support_panel_ids, task.side_1_support_panel_ids
    )
    if (
        proposer != restored.proposer_artifact
        or artifacts != restored.support_artifacts
        or table != restored.support_table
        or space != restored.engineering_version_space
    ):
        raise PanelSoftEngineeringTaskRunnerError("support replay differs")
    survivor_counts = tuple(
        sum(item.orientation == orientation for item in space.survivor_formulas)
        for orientation in PANEL_SOFT_ORIENTATIONS
    )
    if any(count == 0 for count in survivor_counts):
        expected_gap = PanelSoftEngineeringSupportGap.create(space)
        expected_status = (
            PanelSoftEngineeringTaskRunStatus.SUPPORT_ERROR
            if expected_gap.has_observer_error
            else PanelSoftEngineeringTaskRunStatus.SUPPORT_GAP
        )
        if (
            restored.status is not expected_status
            or restored.support_gap != expected_gap
            or any(
                item is not None
                for item in (
                    restored.rank_artifact, restored.predicate_pair,
                    restored.freeze, restored.freeze_commit
                )
            )
            or restored.query_png_base64_by_side
            or restored.query_artifacts
            or restored.query_decisions
            or (
                restored.correct_count, restored.determinate_count,
                restored.abstain_count, restored.error_count,
                restored.ranker_callback_invocations,
                restored.freeze_commit_calls_made, restored.freeze_reload_calls_made,
                restored.query_source_calls_made,
            )
            != (
                0, 0, 0 if expected_gap.has_observer_error else 2,
                2 if expected_gap.has_observer_error else 0, 0, 0, 0, 0,
            )
        ):
            raise PanelSoftEngineeringTaskRunnerError("support gap replay differs")
        return restored

    rank_artifact: PanelSoftRankArtifact | None
    if restored.selection_mode == "support_only_codex_ranker":
        if (
            restored.rank_artifact is None
            or restored.ranker_callback_invocations != 1
        ):
            raise PanelSoftEngineeringTaskRunnerError(
                "ranked replay lacks its one-call rank artifact"
            )
        rank_artifact = _canonical_rank_artifact(restored.rank_artifact, space)
        if (
            restored.rank_artifact_benchmark_sealable
            != rank_artifact.transport_provenance.benchmark_sealable
            or (
                not rank_artifact.transport_provenance.benchmark_sealable
                and not restored.allow_unverified_rank_artifact
            )
        ):
            raise PanelSoftEngineeringTaskRunnerError(
                "ranked replay sealability policy differs"
            )
        expected_pair = PanelSoftEngineeringPredicatePair.create_ranked(
            space,
            side0_formula_digest=rank_artifact.selected_side0_formula_digest,
            side1_formula_digest=rank_artifact.selected_side1_formula_digest,
        )
        _require_rank_call_distinctness(rank_artifact, proposer, artifacts)
    else:
        if (
            restored.rank_artifact is not None
            or restored.ranker_callback_invocations != 0
        ):
            raise PanelSoftEngineeringTaskRunnerError(
                "baseline replay contains a rank artifact"
            )
        rank_artifact = None
        expected_pair = PanelSoftEngineeringPredicatePair.create_deterministic_baseline(
            space
        )

    if (
        restored.status is not PanelSoftEngineeringTaskRunStatus.COMPLETE
        or restored.support_gap is not None
        or restored.predicate_pair != expected_pair
        or restored.freeze is None
        or restored.freeze_commit is None
        or len(restored.query_artifacts) != 2
        or len(restored.query_decisions) != 2
        or (restored.freeze_commit_calls_made, restored.freeze_reload_calls_made,
            restored.query_source_calls_made) != (1, 1, 1)
    ):
        raise PanelSoftEngineeringTaskRunnerError("complete replay phase inventory differs")
    expected_freeze = PanelSoftEngineeringTaskFreeze.seal(
        task_plan=task,
        execution_precommit_digest=restored.execution_precommit_digest,
        proposer_artifact=proposer,
        support_artifacts=artifacts,
        predicate_pair=expected_pair,
        rank_artifact=rank_artifact,
        allow_unverified_rank_artifact=(
            restored.allow_unverified_rank_artifact
        ),
    )
    freeze_bytes = canonical_json(expected_freeze.to_data()) + b"\n"
    restored.freeze_commit.assert_matches(expected_freeze, freeze_bytes)
    if restored.freeze != expected_freeze:
        raise PanelSoftEngineeringTaskRunnerError("task freeze replay differs")
    query_png_by_side = _decode_png_rows(
        restored.query_png_base64_by_side,
        ("side_0", "side_1"),
        label="query",
    )
    query_ids = (task.side_0_query_panel_id, task.side_1_query_panel_id)
    query_artifacts = tuple(
        _canonical_observer(item) for item in restored.query_artifacts
    )
    for side, panel_id, artifact in zip(
        ("side_0", "side_1"), query_ids, query_artifacts, strict=True
    ):
        verify_panel_soft_observer_artifact(
            artifact,
            query_png_by_side[side],
            panel_id=panel_id,
            vocabulary=table.vocabulary,
            expected_artifact_digest=artifact.artifact_digest,
            expected_contract_digest=table.contract.contract_digest,
        )
        verify_panel_soft_observer_contract_identity(artifacts[0], artifact)
    aggregate_panel_soft_observer_artifacts(
        (*artifacts, *query_artifacts),
        ordered_panel_commitments=tuple(
            (item.panel_id, item.panel_png_digest)
            for item in (*artifacts, *query_artifacts)
        ),
        expected_vocabulary=table.vocabulary,
        expected_contract=table.contract,
    )
    _require_proposer_observer_call_distinctness(
        proposer, (*artifacts, *query_artifacts)
    )
    _require_rank_call_distinctness(
        rank_artifact, proposer, (*artifacts, *query_artifacts)
    )
    decisions = tuple(
        PanelSoftEngineeringQueryDecision.create(
            expected_pair, artifact.observation_table, panel_id
        )
        for artifact, panel_id in zip(query_artifacts, query_ids, strict=True)
    )
    correct, determinate, abstain, errors = _query_metrics(decisions)
    if (
        decisions != restored.query_decisions
        or (correct, determinate, abstain, errors)
        != (
            restored.correct_count, restored.determinate_count,
            restored.abstain_count, restored.error_count,
        )
    ):
        raise PanelSoftEngineeringTaskRunnerError("query decision replay differs")
    return restored


__all__ = (
    "PANEL_SOFT_ENGINEERING_QUERY_DENOMINATOR",
    "PANEL_SOFT_ENGINEERING_SUPPORT_PANEL_COUNT",
    "PANEL_SOFT_ENGINEERING_TASK_RUNNER_ID",
    "PanelSoftEngineeringSupportGap",
    "PanelSoftEngineeringProposerTerminal",
    "PanelSoftEngineeringTaskFreeze",
    "PanelSoftEngineeringTaskFreezeCommit",
    "PanelSoftEngineeringTaskRunArchive",
    "PanelSoftEngineeringTaskRunStatus",
    "PanelSoftEngineeringTaskRunnerError",
    "PanelSoftRanker",
    "cold_replay_panel_soft_engineering_task",
    "panel_soft_engineering_task_runner_source_digest",
    "run_panel_soft_engineering_task",
)
