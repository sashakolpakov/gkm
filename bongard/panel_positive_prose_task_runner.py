"""Support admission, durable freeze, and query decisions for positive prose.

There is exactly one executable decision object: a frozen positive cue plus
the observer's fixed absolute interval thresholds.  Support observations gate
that object; proposer self-estimates never do.  The opposite outcome is only
the result of ``CERTIFIED_ABSENT`` on the same cue, never a foil, complement,
negative formula, or negation node.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from collections import Counter
from dataclasses import dataclass
from enum import Enum
import hashlib
from typing import Any, Mapping

from bongard.canonical import canonical_digest, canonical_json
from bongard.evidence import Disposition
from bongard.object_bongard_release_gate import ObjectBongardWriteOnceReceipt
from bongard.official_extracted_panel_archive import ReleasedOfficialExtractedPanel
from bongard.panel_feature_extracted_release_gate import (
    PanelFeatureExtractedExecutionPrecommit,
)
from bongard.panel_positive_prose_evidence_bundle import (
    PositiveProseEvidenceBundle,
    PositiveProseEvidencePhase,
    PositiveProseEvidenceRow,
    PositiveProsePanelRole,
    cold_replay_positive_prose_evidence_bundle,
)
from bongard.panel_positive_prose_observer import (
    PositiveProseCue,
    positive_prose_scale_digest,
)
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID


POSITIVE_PROSE_SUPPORT_ADMISSION_SCHEMA = (
    "gkm.bongard-positive-prose-support-admission.v1"
)
POSITIVE_PROSE_FROZEN_PREDICATE_SCHEMA = (
    "gkm.bongard-positive-prose-frozen-predicate.v1"
)
POSITIVE_PROSE_TASK_FREEZE_SCHEMA = "gkm.bongard-positive-prose-task-freeze.v1"
POSITIVE_PROSE_TASK_COMMIT_SCHEMA = "gkm.bongard-positive-prose-task-commit.v1"
POSITIVE_PROSE_QUERY_DECISION_SCHEMA = (
    "gkm.bongard-positive-prose-query-decision.v1"
)
POSITIVE_PROSE_TASK_RUNNER_ID = (
    "bongard.positive-prose/one-positive-cue-freeze-python-v1"
)

PRIMARY_PRESENT_REQUIRED = 5
CONTRAST_CERTIFIED_ABSENT_REQUIRED = 5
MAX_INDETERMINATE_PER_ROLE = 1
MAX_CONTRADICTIONS = 0
MAX_ERRORS = 0
PRESENT_LOWER_THRESHOLD = 3
CERTIFIED_ABSENT_UPPER_THRESHOLD = 1


class PositiveProseTaskRunnerError(RuntimeError):
    """A support admission, freeze, commit, release, or replay differs."""


class PositiveProseSupportStatus(str, Enum):
    SUPPORT_ADMISSIBLE = "support_admissible"
    SUPPORT_GAP = "support_gap"


class PositiveProseQueryOutcome(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    ABSTAIN = "abstain"
    ERROR = "error"


def panel_positive_prose_task_runner_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "implementation_language": "python",
        "python_is_canonical_authority": True,
        "engineering_only": True,
        "uncalibrated": True,
        "scientific_evidence": False,
        "benchmark_authoritative": False,
        "positive_only": True,
        "one_positive_cue_only": True,
        "one_positive_formula_only": False,
        "negative_formula_present": False,
        "foil_present": False,
        "negation_allowed": False,
        "complement_allowed": False,
        "polarity_flip_allowed": False,
        "arbitrary_predicate_code_allowed": False,
        "prose_is_inert_data": True,
        "lean_present": False,
        "lean_required": False,
        "lean_removable": True,
        "lean_affects_identity_selection_decision_or_replay": False,
    }


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(type(key) is not str for key in value)
        or set(value) != expected
    ):
        raise PositiveProseTaskRunnerError(f"{label} fields differ")
    return value


def _raw_digest(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise PositiveProseTaskRunnerError(f"{label} must be a raw SHA-256")
    return value


def _address(value: object, label: str) -> str:
    if (
        type(value) is not str
        or not value.startswith("sha256:")
        or len(value) != 71
        or any(char not in "0123456789abcdef" for char in value[7:])
    ):
        raise PositiveProseTaskRunnerError(f"{label} must be a sha256: address")
    return value


def _counts(rows: tuple[PositiveProseEvidenceRow, ...]) -> dict[str, int]:
    values = Counter(row.observer_artifact.observation.disposition.value for row in rows)
    return {item.value: values[item.value] for item in Disposition}


def _support_gap_reasons(
    primary: Mapping[str, int], contrast: Mapping[str, int]
) -> tuple[str, ...]:
    reasons: list[str] = []
    if primary[Disposition.PRESENT.value] < PRIMARY_PRESENT_REQUIRED:
        reasons.append("primary_present_below_five")
    if contrast[Disposition.CERTIFIED_ABSENT.value] < CONTRAST_CERTIFIED_ABSENT_REQUIRED:
        reasons.append("contrast_certified_absent_below_five")
    if primary[Disposition.CERTIFIED_ABSENT.value] > MAX_CONTRADICTIONS:
        reasons.append("primary_certified_absent_contradiction")
    if contrast[Disposition.PRESENT.value] > MAX_CONTRADICTIONS:
        reasons.append("contrast_present_contradiction")
    if primary[Disposition.ERROR.value] + contrast[Disposition.ERROR.value] > MAX_ERRORS:
        reasons.append("support_observer_error")
    if primary[Disposition.INDETERMINATE.value] > MAX_INDETERMINATE_PER_ROLE:
        reasons.append("primary_indeterminate_above_one")
    if contrast[Disposition.INDETERMINATE.value] > MAX_INDETERMINATE_PER_ROLE:
        reasons.append("contrast_indeterminate_above_one")
    return tuple(reasons)


def _admission_content(value: "PositiveProseSupportAdmission") -> dict[str, object]:
    return {
        "schema": POSITIVE_PROSE_SUPPORT_ADMISSION_SCHEMA,
        "runner_id": POSITIVE_PROSE_TASK_RUNNER_ID,
        "runner_source_digest": panel_positive_prose_task_runner_source_digest(),
        "evidence_bundle": value.evidence_bundle.to_data(),
        "evidence_bundle_address": value.evidence_bundle.artifact_address,
        "task_id": value.task_id,
        "task_plan_digest": value.evidence_bundle.task_plan.record_digest,
        "cue_digest": value.evidence_bundle.cue_digest,
        "primary_counts": dict(value.primary_counts),
        "contrast_counts": dict(value.contrast_counts),
        "status": value.status.value,
        "gap_reasons": list(value.gap_reasons),
        "support_rule": {
            "primary_present_required": PRIMARY_PRESENT_REQUIRED,
            "contrast_certified_absent_required": CONTRAST_CERTIFIED_ABSENT_REQUIRED,
            "maximum_indeterminate_per_role": MAX_INDETERMINATE_PER_ROLE,
            "maximum_contradictions": MAX_CONTRADICTIONS,
            "maximum_errors": MAX_ERRORS,
            "primary_contradiction": Disposition.CERTIFIED_ABSENT.value,
            "contrast_contradiction": Disposition.PRESENT.value,
        },
        "proposer_self_estimates_used_for_admission": False,
        "all_twelve_independent_observer_artifacts_used": True,
        "query_rows_present": False,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class PositiveProseSupportAdmission:
    """Total support gate with a typed gap rather than an exceptional rejection."""

    evidence_bundle: PositiveProseEvidenceBundle
    primary_counts: Mapping[str, int]
    contrast_counts: Mapping[str, int]
    status: PositiveProseSupportStatus
    gap_reasons: tuple[str, ...]
    record_digest: str

    def __post_init__(self) -> None:
        if type(self.evidence_bundle) is not PositiveProseEvidenceBundle:
            raise TypeError("admission needs exact evidence bundle")
        bundle = self.evidence_bundle
        if bundle.query_rows:
            raise PositiveProseTaskRunnerError("support admission cannot contain query rows")
        primary_rows = tuple(row for row in bundle.support_rows if row.role is PositiveProsePanelRole.PRIMARY_SUPPORT)
        contrast_rows = tuple(row for row in bundle.support_rows if row.role is PositiveProsePanelRole.CONTRAST_SUPPORT)
        expected_primary = _counts(primary_rows)
        expected_contrast = _counts(contrast_rows)
        reasons = _support_gap_reasons(expected_primary, expected_contrast)
        expected_status = (
            PositiveProseSupportStatus.SUPPORT_ADMISSIBLE
            if not reasons else PositiveProseSupportStatus.SUPPORT_GAP
        )
        if (
            dict(self.primary_counts) != expected_primary
            or dict(self.contrast_counts) != expected_contrast
            or type(self.status) is not PositiveProseSupportStatus
            or self.status is not expected_status
            or self.gap_reasons != reasons
            or self.record_digest != canonical_digest(_admission_content(self))
        ):
            raise PositiveProseTaskRunnerError("support admission counts or policy differ")

    @property
    def task_id(self) -> str:
        return self.evidence_bundle.task_plan.task_id

    @property
    def artifact_address(self) -> str:
        return "sha256:" + self.record_digest

    @classmethod
    def derive(
        cls,
        evidence_bundle: PositiveProseEvidenceBundle,
        *,
        expected_bundle_address: str,
    ) -> "PositiveProseSupportAdmission":
        bundle = cold_replay_positive_prose_evidence_bundle(
            evidence_bundle, expected_artifact_address=expected_bundle_address
        )
        primary = _counts(tuple(row for row in bundle.support_rows if row.role is PositiveProsePanelRole.PRIMARY_SUPPORT))
        contrast = _counts(tuple(row for row in bundle.support_rows if row.role is PositiveProsePanelRole.CONTRAST_SUPPORT))
        reasons = _support_gap_reasons(primary, contrast)
        values = {
            "evidence_bundle": bundle,
            "primary_counts": primary,
            "contrast_counts": contrast,
            "status": PositiveProseSupportStatus.SUPPORT_ADMISSIBLE if not reasons else PositiveProseSupportStatus.SUPPORT_GAP,
            "gap_reasons": reasons,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(**values, record_digest=canonical_digest(_admission_content(provisional)))

    def to_data(self) -> dict[str, object]:
        return {**_admission_content(self), "record_digest": self.record_digest, "artifact_address": self.artifact_address}

    @classmethod
    def from_data(cls, value: object) -> "PositiveProseSupportAdmission":
        raw = _fields(value, set(_admission_fields()), "positive prose support admission")
        if (
            raw["schema"] != POSITIVE_PROSE_SUPPORT_ADMISSION_SCHEMA
            or raw["runner_id"] != POSITIVE_PROSE_TASK_RUNNER_ID
            or raw["runner_source_digest"] != panel_positive_prose_task_runner_source_digest()
            or raw["support_rule"] != {
                "primary_present_required": 5,
                "contrast_certified_absent_required": 5,
                "maximum_indeterminate_per_role": 1,
                "maximum_contradictions": 0,
                "maximum_errors": 0,
                "primary_contradiction": Disposition.CERTIFIED_ABSENT.value,
                "contrast_contradiction": Disposition.PRESENT.value,
            }
            or raw["proposer_self_estimates_used_for_admission"] is not False
            or raw["all_twelve_independent_observer_artifacts_used"] is not True
            or raw["query_rows_present"] is not False
            or any(raw[key] != item for key, item in _authority_data().items())
        ):
            raise PositiveProseTaskRunnerError("support admission policy differs")
        bundle = PositiveProseEvidenceBundle.from_data(raw["evidence_bundle"])
        result = cls(
            bundle, dict(raw["primary_counts"]), dict(raw["contrast_counts"]),
            PositiveProseSupportStatus(raw["status"]), tuple(raw["gap_reasons"]),
            raw["record_digest"],
        )
        if (
            raw["evidence_bundle_address"] != bundle.artifact_address
            or raw["task_id"] != result.task_id
            or raw["task_plan_digest"] != bundle.task_plan.record_digest
            or raw["cue_digest"] != bundle.cue_digest
            or raw["artifact_address"] != result.artifact_address
            or result.to_data() != dict(raw)
        ):
            raise PositiveProseTaskRunnerError("support admission is not canonical")
        return result


def _admission_fields() -> tuple[str, ...]:
    return (
        "schema", "runner_id", "runner_source_digest", "evidence_bundle",
        "evidence_bundle_address", "task_id", "task_plan_digest", "cue_digest",
        "primary_counts", "contrast_counts", "status", "gap_reasons",
        "support_rule", "proposer_self_estimates_used_for_admission",
        "all_twelve_independent_observer_artifacts_used", "query_rows_present",
        *_authority_data(), "record_digest", "artifact_address",
    )


def _predicate_content(value: "PositiveProseFrozenPredicate") -> dict[str, object]:
    return {
        "schema": POSITIVE_PROSE_FROZEN_PREDICATE_SCHEMA,
        "cue": value.cue.to_data(),
        "cue_digest": value.cue.cue_digest,
        "scale_digest": value.scale_digest,
        "present_when_lower_at_least": value.present_when_lower_at_least,
        "certified_absent_when_upper_at_most": value.certified_absent_when_upper_at_most,
        "otherwise": Disposition.INDETERMINATE.value,
        "transport_or_parser_failure": Disposition.ERROR.value,
        "decision_mapping": {
            Disposition.PRESENT.value: PositiveProseQueryOutcome.POSITIVE.value,
            Disposition.CERTIFIED_ABSENT.value: PositiveProseQueryOutcome.NEGATIVE.value,
            Disposition.INDETERMINATE.value: PositiveProseQueryOutcome.ABSTAIN.value,
            Disposition.ERROR.value: PositiveProseQueryOutcome.ERROR.value,
        },
        "thresholds_selected_by_model": False,
        "cue_frozen_before_query_release": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class PositiveProseFrozenPredicate:
    cue: PositiveProseCue
    scale_digest: str
    present_when_lower_at_least: int
    certified_absent_when_upper_at_most: int
    predicate_digest: str

    def __post_init__(self) -> None:
        if type(self.cue) is not PositiveProseCue:
            raise TypeError("frozen predicate needs exact positive cue")
        if (
            self.scale_digest != positive_prose_scale_digest()
            or self.present_when_lower_at_least != PRESENT_LOWER_THRESHOLD
            or self.certified_absent_when_upper_at_most != CERTIFIED_ABSENT_UPPER_THRESHOLD
            or self.predicate_digest != canonical_digest(_predicate_content(self))
        ):
            raise PositiveProseTaskRunnerError("frozen cue or thresholds differ")

    @classmethod
    def freeze(cls, cue: PositiveProseCue) -> "PositiveProseFrozenPredicate":
        values = {
            "cue": cue,
            "scale_digest": positive_prose_scale_digest(),
            "present_when_lower_at_least": PRESENT_LOWER_THRESHOLD,
            "certified_absent_when_upper_at_most": CERTIFIED_ABSENT_UPPER_THRESHOLD,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(**values, predicate_digest=canonical_digest(_predicate_content(provisional)))

    def to_data(self) -> dict[str, object]:
        return {**_predicate_content(self), "predicate_digest": self.predicate_digest}

    @classmethod
    def from_data(cls, value: object) -> "PositiveProseFrozenPredicate":
        raw = _fields(
            value,
            {
                "schema", "cue", "cue_digest", "scale_digest",
                "present_when_lower_at_least",
                "certified_absent_when_upper_at_most", "otherwise",
                "transport_or_parser_failure", "decision_mapping",
                "thresholds_selected_by_model", "cue_frozen_before_query_release",
                *_authority_data(), "predicate_digest",
            },
            "positive prose frozen predicate",
        )
        if (
            raw["schema"] != POSITIVE_PROSE_FROZEN_PREDICATE_SCHEMA
            or raw["otherwise"] != Disposition.INDETERMINATE.value
            or raw["transport_or_parser_failure"] != Disposition.ERROR.value
            or raw["thresholds_selected_by_model"] is not False
            or raw["cue_frozen_before_query_release"] is not True
            or any(raw[key] != item for key, item in _authority_data().items())
        ):
            raise PositiveProseTaskRunnerError("frozen predicate policy differs")
        cue = PositiveProseCue.from_data(raw["cue"])
        result = cls(
            cue, raw["scale_digest"], raw["present_when_lower_at_least"],
            raw["certified_absent_when_upper_at_most"], raw["predicate_digest"],
        )
        if raw["cue_digest"] != cue.cue_digest or result.to_data() != dict(raw):
            raise PositiveProseTaskRunnerError("frozen predicate is not canonical")
        return result


def _verify_precommit(
    precommit: PanelFeatureExtractedExecutionPrecommit,
    admission: PositiveProseSupportAdmission,
) -> None:
    if type(precommit) is not PanelFeatureExtractedExecutionPrecommit:
        raise TypeError("positive prose freeze needs exact extracted precommit")
    task = admission.evidence_bundle.task_plan
    support = set(task.side_0_support_panel_ids + task.side_1_support_panel_ids)
    query = {task.side_0_query_panel_id, task.side_1_query_panel_id}
    if (
        task.task_id not in precommit.selected_task_ids
        or not support <= set(precommit.authorized_support_panel_ids)
        or not query <= set(precommit.sealed_query_panel_ids)
        or admission.evidence_bundle.execution_precommit_digest != precommit.record_digest
    ):
        raise PositiveProseTaskRunnerError("precommit does not bind task roles")


def _freeze_content(value: "PositiveProseTaskFreeze") -> dict[str, object]:
    bundle = value.support_admission.evidence_bundle
    return {
        "schema": POSITIVE_PROSE_TASK_FREEZE_SCHEMA,
        "runner_id": POSITIVE_PROSE_TASK_RUNNER_ID,
        "runner_source_digest": panel_positive_prose_task_runner_source_digest(),
        "support_admission": value.support_admission.to_data(),
        "support_admission_address": value.support_admission.artifact_address,
        "support_evidence_bundle_address": bundle.artifact_address,
        "task_plan": bundle.task_plan.to_data(),
        "task_id": value.task_id,
        "task_plan_digest": value.task_plan_digest,
        "authorization_digest": bundle.authorization_digest,
        "execution_precommit": value.execution_precommit.to_data(),
        "execution_precommit_digest": value.execution_precommit_digest,
        "selected_predicate": value.selected_predicate.to_data(),
        "selected_predicate_digest": value.selected_predicate_digest,
        "version_space_digest": value.version_space_digest,
        "support_version_space_digest": value.support_version_space_digest,
        "rank_response_digest": value.rank_response_digest,
        "selection_mode": "single_admitted_positive_cue_no_rank_call",
        "sealed_query_panel_ids": list(value.sealed_query_panel_ids),
        "query_bytes_included": False,
        "query_observations_included": False,
        "query_release_authorized_only_after_exact_durable_commit": True,
        "exact_proposer_and_twelve_support_journal_terminals_bound": True,
        "support_admissible_not_exact_separation": True,
        "model_calls_during_freeze": 0,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class PositiveProseTaskFreeze:
    support_admission: PositiveProseSupportAdmission
    execution_precommit: PanelFeatureExtractedExecutionPrecommit
    selected_predicate: PositiveProseFrozenPredicate
    version_space_digest: str
    rank_response_digest: str
    sealed_query_panel_ids: tuple[str, str]
    record_digest: str

    def __post_init__(self) -> None:
        if type(self.support_admission) is not PositiveProseSupportAdmission:
            raise TypeError("freeze needs exact support admission")
        _verify_precommit(self.execution_precommit, self.support_admission)
        admission = self.support_admission
        bundle = admission.evidence_bundle
        cue = bundle.support_rows[0].observer_artifact.request.cue
        expected_predicate = PositiveProseFrozenPredicate.freeze(cue)
        expected_space = canonical_digest(
            {
                "schema": "gkm.bongard-positive-prose-admitted-singleton-space.v1",
                "support_admission_address": admission.artifact_address,
                "predicate_digest": expected_predicate.predicate_digest,
                "survivor_count": 1,
            }
        )
        expected_rank = canonical_digest(
            {
                "schema": "gkm.bongard-positive-prose-no-rank-selection.v1",
                "version_space_digest": expected_space,
                "selected_predicate_digest": expected_predicate.predicate_digest,
                "model_call_made": False,
            }
        )
        if (
            admission.status is not PositiveProseSupportStatus.SUPPORT_ADMISSIBLE
            or admission.gap_reasons
            or not bundle.benchmark_sealable
            or self.selected_predicate != expected_predicate
            or self.version_space_digest != expected_space
            or self.rank_response_digest != expected_rank
            or self.sealed_query_panel_ids
            != (bundle.task_plan.side_0_query_panel_id, bundle.task_plan.side_1_query_panel_id)
            or self.record_digest != "sha256:" + canonical_digest(_freeze_content(self))
        ):
            raise PositiveProseTaskRunnerError("freeze admission, cue, or query seal differs")

    @property
    def task_id(self) -> str:
        return self.support_admission.task_id

    @property
    def task_plan_digest(self) -> str:
        return self.support_admission.evidence_bundle.task_plan.record_digest

    @property
    def execution_precommit_digest(self) -> str:
        return self.execution_precommit.record_digest

    @property
    def support_version_space_digest(self) -> str:
        return self.version_space_digest

    @property
    def selected_predicate_digest(self) -> str:
        return self.selected_predicate.predicate_digest

    @classmethod
    def seal(
        cls,
        support_admission: PositiveProseSupportAdmission,
        *,
        execution_precommit: PanelFeatureExtractedExecutionPrecommit,
    ) -> "PositiveProseTaskFreeze":
        if type(support_admission) is not PositiveProseSupportAdmission:
            raise TypeError("freeze needs exact support admission")
        # Re-run the complete receipt and journal replay at the temporal freeze edge.
        replayed_bundle = cold_replay_positive_prose_evidence_bundle(
            support_admission.evidence_bundle,
            expected_artifact_address=support_admission.evidence_bundle.artifact_address,
        )
        if replayed_bundle != support_admission.evidence_bundle:
            raise PositiveProseTaskRunnerError("support bundle replay differs at freeze")
        _verify_precommit(execution_precommit, support_admission)
        if support_admission.status is not PositiveProseSupportStatus.SUPPORT_ADMISSIBLE:
            raise PositiveProseTaskRunnerError("support gap cannot be frozen")
        cue = replayed_bundle.support_rows[0].observer_artifact.request.cue
        predicate = PositiveProseFrozenPredicate.freeze(cue)
        space = canonical_digest(
            {
                "schema": "gkm.bongard-positive-prose-admitted-singleton-space.v1",
                "support_admission_address": support_admission.artifact_address,
                "predicate_digest": predicate.predicate_digest,
                "survivor_count": 1,
            }
        )
        rank = canonical_digest(
            {
                "schema": "gkm.bongard-positive-prose-no-rank-selection.v1",
                "version_space_digest": space,
                "selected_predicate_digest": predicate.predicate_digest,
                "model_call_made": False,
            }
        )
        values = {
            "support_admission": support_admission,
            "execution_precommit": execution_precommit,
            "selected_predicate": predicate,
            "version_space_digest": space,
            "rank_response_digest": rank,
            "sealed_query_panel_ids": (
                replayed_bundle.task_plan.side_0_query_panel_id,
                replayed_bundle.task_plan.side_1_query_panel_id,
            ),
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(**values, record_digest="sha256:" + canonical_digest(_freeze_content(provisional)))

    def to_data(self) -> dict[str, object]:
        return {**_freeze_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: object) -> "PositiveProseTaskFreeze":
        raw = _fields(value, set(_freeze_fields()), "positive prose task freeze")
        if (
            raw["schema"] != POSITIVE_PROSE_TASK_FREEZE_SCHEMA
            or raw["runner_id"] != POSITIVE_PROSE_TASK_RUNNER_ID
            or raw["runner_source_digest"] != panel_positive_prose_task_runner_source_digest()
            or raw["selection_mode"] != "single_admitted_positive_cue_no_rank_call"
            or raw["query_bytes_included"] is not False
            or raw["query_observations_included"] is not False
            or raw["query_release_authorized_only_after_exact_durable_commit"] is not True
            or raw["exact_proposer_and_twelve_support_journal_terminals_bound"] is not True
            or raw["support_admissible_not_exact_separation"] is not True
            or raw["model_calls_during_freeze"] != 0
            or any(raw[key] != item for key, item in _authority_data().items())
        ):
            raise PositiveProseTaskRunnerError("task freeze policy differs")
        admission = PositiveProseSupportAdmission.from_data(raw["support_admission"])
        precommit = PanelFeatureExtractedExecutionPrecommit.from_data(raw["execution_precommit"])
        predicate = PositiveProseFrozenPredicate.from_data(raw["selected_predicate"])
        result = cls(
            admission, precommit, predicate, raw["version_space_digest"],
            raw["rank_response_digest"], tuple(raw["sealed_query_panel_ids"]),
            raw["record_digest"],
        )
        if (
            raw["support_admission_address"] != admission.artifact_address
            or raw["support_evidence_bundle_address"] != admission.evidence_bundle.artifact_address
            or raw["task_plan"] != admission.evidence_bundle.task_plan.to_data()
            or raw["task_id"] != result.task_id
            or raw["task_plan_digest"] != result.task_plan_digest
            or raw["authorization_digest"] != admission.evidence_bundle.authorization_digest
            or raw["execution_precommit_digest"] != precommit.record_digest
            or raw["selected_predicate_digest"] != predicate.predicate_digest
            or raw["support_version_space_digest"] != result.support_version_space_digest
            or result.to_data() != dict(raw)
        ):
            raise PositiveProseTaskRunnerError("task freeze is not canonical")
        return result


def _freeze_fields() -> tuple[str, ...]:
    return (
        "schema", "runner_id", "runner_source_digest", "support_admission",
        "support_admission_address", "support_evidence_bundle_address", "task_plan",
        "task_id", "task_plan_digest", "authorization_digest", "execution_precommit",
        "execution_precommit_digest", "selected_predicate",
        "selected_predicate_digest", "version_space_digest",
        "support_version_space_digest", "rank_response_digest", "selection_mode",
        "sealed_query_panel_ids", "query_bytes_included",
        "query_observations_included",
        "query_release_authorized_only_after_exact_durable_commit",
        "exact_proposer_and_twelve_support_journal_terminals_bound",
        "support_admissible_not_exact_separation", "model_calls_during_freeze",
        *_authority_data(), "record_digest",
    )


def verify_positive_prose_task_freeze(
    freeze: PositiveProseTaskFreeze, *, expected_record_digest: str
) -> PositiveProseTaskFreeze:
    if type(freeze) is not PositiveProseTaskFreeze:
        raise TypeError("freeze replay needs exact positive prose freeze")
    expected = _address(expected_record_digest, "expected task freeze digest")
    restored = PositiveProseTaskFreeze.from_data(freeze.to_data())
    cold_replay_positive_prose_evidence_bundle(
        restored.support_admission.evidence_bundle,
        expected_artifact_address=restored.support_admission.evidence_bundle.artifact_address,
    )
    if restored != freeze or restored.record_digest != expected:
        raise PositiveProseTaskRunnerError("task freeze cold replay differs")
    return restored


cold_replay_positive_prose_task_freeze = verify_positive_prose_task_freeze


def _commit_content(value: "PositiveProseTaskFreezeCommit") -> dict[str, object]:
    freeze = value.task_freeze
    return {
        "schema": POSITIVE_PROSE_TASK_COMMIT_SCHEMA,
        "runner_id": POSITIVE_PROSE_TASK_RUNNER_ID,
        "task_freeze": freeze.to_data(),
        "task_freeze_digest": freeze.record_digest,
        "task_freeze_store_receipt": value.task_freeze_store_receipt.to_data(),
        "task_freeze_store_receipt_digest": value.task_freeze_store_receipt.record_digest,
        "exact_freeze_payload_digest": value.task_freeze_store_receipt.payload_digest,
        "task_id": value.task_id,
        "task_plan_digest": value.task_plan_digest,
        "execution_precommit_digest": value.execution_precommit_digest,
        "version_space_digest": value.version_space_digest,
        "support_version_space_digest": value.support_version_space_digest,
        "rank_response_digest": value.rank_response_digest,
        "selected_predicate_digest": value.selected_predicate_digest,
        "durably_persisted_and_reloaded_before_query_release": True,
        "exact_canonical_freeze_bytes_bound": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class PositiveProseTaskFreezeCommit:
    task_freeze: PositiveProseTaskFreeze
    task_freeze_store_receipt: ObjectBongardWriteOnceReceipt
    record_digest: str

    def __post_init__(self) -> None:
        if type(self.task_freeze) is not PositiveProseTaskFreeze:
            raise TypeError("commit needs exact positive prose freeze")
        if type(self.task_freeze_store_receipt) is not ObjectBongardWriteOnceReceipt:
            raise TypeError("commit needs exact write-once receipt")
        payload = canonical_json(self.task_freeze.to_data()) + b"\n"
        receipt = self.task_freeze_store_receipt
        if (
            receipt.object_kind != "task-freeze"
            or receipt.object_digest != self.task_freeze.record_digest
            or receipt.payload_digest != "sha256:" + hashlib.sha256(payload).hexdigest()
            or receipt.size_bytes != len(payload)
            or self.record_digest != "sha256:" + canonical_digest(_commit_content(self))
        ):
            raise PositiveProseTaskRunnerError("commit does not bind exact durable freeze")

    @property
    def task_freeze_digest(self) -> str:
        return self.task_freeze.record_digest

    @property
    def exact_freeze_payload_digest(self) -> str:
        return self.task_freeze_store_receipt.payload_digest

    @property
    def task_freeze_store_receipt_digest(self) -> str:
        return self.task_freeze_store_receipt.record_digest

    @property
    def task_id(self) -> str:
        return self.task_freeze.task_id

    @property
    def task_plan_digest(self) -> str:
        return self.task_freeze.task_plan_digest

    @property
    def execution_precommit_digest(self) -> str:
        return self.task_freeze.execution_precommit_digest

    @property
    def version_space_digest(self) -> str:
        return self.task_freeze.version_space_digest

    @property
    def support_version_space_digest(self) -> str:
        return self.task_freeze.support_version_space_digest

    @property
    def rank_response_digest(self) -> str:
        return self.task_freeze.rank_response_digest

    @property
    def selected_predicate_digest(self) -> str:
        return self.task_freeze.selected_predicate_digest

    @classmethod
    def seal(
        cls,
        freeze: PositiveProseTaskFreeze,
        receipt: ObjectBongardWriteOnceReceipt,
    ) -> "PositiveProseTaskFreezeCommit":
        values = {"task_freeze": freeze, "task_freeze_store_receipt": receipt}
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(**values, record_digest="sha256:" + canonical_digest(_commit_content(provisional)))

    def to_data(self) -> dict[str, object]:
        return {**_commit_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: object) -> "PositiveProseTaskFreezeCommit":
        raw = _fields(
            value,
            {
                "schema", "runner_id", "task_freeze", "task_freeze_digest",
                "task_freeze_store_receipt", "task_freeze_store_receipt_digest",
                "exact_freeze_payload_digest", "task_id", "task_plan_digest",
                "execution_precommit_digest", "version_space_digest",
                "support_version_space_digest", "rank_response_digest",
                "selected_predicate_digest",
                "durably_persisted_and_reloaded_before_query_release",
                "exact_canonical_freeze_bytes_bound", *_authority_data(),
                "record_digest",
            },
            "positive prose task commit",
        )
        if (
            raw["schema"] != POSITIVE_PROSE_TASK_COMMIT_SCHEMA
            or raw["runner_id"] != POSITIVE_PROSE_TASK_RUNNER_ID
            or raw["durably_persisted_and_reloaded_before_query_release"] is not True
            or raw["exact_canonical_freeze_bytes_bound"] is not True
            or any(raw[key] != item for key, item in _authority_data().items())
        ):
            raise PositiveProseTaskRunnerError("task commit policy differs")
        freeze = PositiveProseTaskFreeze.from_data(raw["task_freeze"])
        receipt = ObjectBongardWriteOnceReceipt.from_data(raw["task_freeze_store_receipt"])
        result = cls(freeze, receipt, raw["record_digest"])
        if result.to_data() != dict(raw):
            raise PositiveProseTaskRunnerError("task commit is not canonical")
        return result


def verify_positive_prose_task_commit(
    commit: PositiveProseTaskFreezeCommit,
    *,
    expected_record_digest: str,
    task_commit_store_receipt: ObjectBongardWriteOnceReceipt | None = None,
) -> PositiveProseTaskFreezeCommit:
    if type(commit) is not PositiveProseTaskFreezeCommit:
        raise TypeError("commit replay needs exact positive prose commit")
    restored = PositiveProseTaskFreezeCommit.from_data(commit.to_data())
    verify_positive_prose_task_freeze(
        restored.task_freeze, expected_record_digest=restored.task_freeze.record_digest
    )
    if restored != commit or restored.record_digest != _address(expected_record_digest, "expected commit digest"):
        raise PositiveProseTaskRunnerError("task commit cold replay differs")
    if task_commit_store_receipt is not None:
        payload = canonical_json(restored.to_data()) + b"\n"
        receipt = task_commit_store_receipt
        if (
            type(receipt) is not ObjectBongardWriteOnceReceipt
            or receipt.object_kind != "task-decision-commit"
            or receipt.object_digest != restored.record_digest
            or receipt.payload_digest != "sha256:" + hashlib.sha256(payload).hexdigest()
            or receipt.size_bytes != len(payload)
        ):
            raise PositiveProseTaskRunnerError("commit receipt differs")
    return restored


cold_replay_positive_prose_task_commit = verify_positive_prose_task_commit


def _outcome(disposition: Disposition) -> PositiveProseQueryOutcome:
    return {
        Disposition.PRESENT: PositiveProseQueryOutcome.POSITIVE,
        Disposition.CERTIFIED_ABSENT: PositiveProseQueryOutcome.NEGATIVE,
        Disposition.INDETERMINATE: PositiveProseQueryOutcome.ABSTAIN,
        Disposition.ERROR: PositiveProseQueryOutcome.ERROR,
    }[disposition]


def _verify_released_query(
    freeze: PositiveProseTaskFreeze,
    bundle: PositiveProseEvidenceBundle,
    released: ReleasedOfficialExtractedPanel,
    receipt: ObjectBongardWriteOnceReceipt,
) -> tuple[int, PositiveProseEvidenceRow]:
    if type(released) is not ReleasedOfficialExtractedPanel:
        raise TypeError("query decision needs exact extracted released panel")
    if ReleasedOfficialExtractedPanel.from_data(released.to_data()) != released:
        raise PositiveProseTaskRunnerError("released query replay differs")
    if type(receipt) is not ObjectBongardWriteOnceReceipt:
        raise TypeError("query decision needs exact release-store receipt")
    payload = canonical_json(released.to_data()) + b"\n"
    try:
        ordinal = freeze.sealed_query_panel_ids.index(released.panel_id)
    except ValueError as exc:
        raise PositiveProseTaskRunnerError("released panel is not a sealed query") from exc
    row = bundle.query_rows[ordinal]
    if (
        receipt.object_kind != "released-extracted-query-panel"
        or receipt.object_digest != released.record_digest
        or receipt.payload_digest != "sha256:" + hashlib.sha256(payload).hexdigest()
        or receipt.size_bytes != len(payload)
        or released.execution_precommit_digest != freeze.execution_precommit_digest
        or row.phase is not PositiveProseEvidencePhase.QUERY
        or row.phase_index != ordinal
        or row.panel_id != released.panel_id
        or row.panel_png != released.exact_png_bytes
        or "sha256:" + row.panel_png_digest != released.exact_png_digest
    ):
        raise PositiveProseTaskRunnerError("released query, receipt, and evidence differ")
    return ordinal, row


def _decision_content(value: "PositiveProseQueryDecision") -> dict[str, object]:
    observation = value.query_row.observer_artifact.observation
    return {
        "schema": POSITIVE_PROSE_QUERY_DECISION_SCHEMA,
        "runner_id": POSITIVE_PROSE_TASK_RUNNER_ID,
        "task_freeze_digest": value.task_freeze_digest,
        "task_id": value.task_id,
        "selected_predicate": value.selected_predicate.to_data(),
        "selected_predicate_digest": value.selected_predicate.predicate_digest,
        "query_evidence_bundle": value.query_evidence_bundle.to_data(),
        "query_evidence_bundle_address": value.query_evidence_bundle.artifact_address,
        "released_query_panel": value.released_query_panel.to_data(),
        "released_query_panel_digest": value.released_query_panel.record_digest,
        "query_release_store_receipt": value.query_release_store_receipt.to_data(),
        "query_release_store_receipt_digest": value.query_release_store_receipt.record_digest,
        "query_ordinal": value.query_ordinal,
        "query_row": value.query_row.to_data(),
        "query_row_digest": value.query_row.record_digest,
        "query_panel_id": value.query_row.panel_id,
        "observation": observation.to_data(),
        "disposition": value.disposition.value,
        "outcome": value.outcome.value,
        "decision_mapping": "present-positive_certified-absent-negative_indeterminate-abstain_error-error",
        "query_truth_label_present": False,
        "negative_formula_evaluated": False,
        "model_calls_during_decision": 0,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class PositiveProseQueryDecision:
    task_freeze_digest: str
    task_id: str
    selected_predicate: PositiveProseFrozenPredicate
    query_evidence_bundle: PositiveProseEvidenceBundle
    released_query_panel: ReleasedOfficialExtractedPanel
    query_release_store_receipt: ObjectBongardWriteOnceReceipt
    query_ordinal: int
    query_row: PositiveProseEvidenceRow
    disposition: Disposition
    outcome: PositiveProseQueryOutcome
    decision_digest: str

    def __post_init__(self) -> None:
        _address(self.task_freeze_digest, "decision freeze digest")
        if type(self.query_ordinal) is not int or self.query_ordinal not in (0, 1):
            raise PositiveProseTaskRunnerError("query ordinal differs")
        bundle = self.query_evidence_bundle
        released = self.released_query_panel
        receipt = self.query_release_store_receipt
        payload = (
            canonical_json(released.to_data()) + b"\n"
            if type(released) is ReleasedOfficialExtractedPanel
            else b""
        )
        if (
            type(self.selected_predicate) is not PositiveProseFrozenPredicate
            or type(bundle) is not PositiveProseEvidenceBundle
            or len(bundle.query_rows) != 2
            or self.task_id != bundle.task_plan.task_id
            or type(self.query_row) is not PositiveProseEvidenceRow
            or self.query_row != bundle.query_rows[self.query_ordinal]
            or self.selected_predicate.cue.cue_digest != bundle.cue_digest
            or type(released) is not ReleasedOfficialExtractedPanel
            or type(receipt) is not ObjectBongardWriteOnceReceipt
            or receipt.object_kind != "released-extracted-query-panel"
            or receipt.object_digest != released.record_digest
            or receipt.payload_digest
            != "sha256:" + hashlib.sha256(payload).hexdigest()
            or receipt.size_bytes != len(payload)
            or released.execution_precommit_digest
            != bundle.execution_precommit_digest
            or released.panel_id != self.query_row.panel_id
            or released.exact_png_bytes != self.query_row.panel_png
            or released.exact_png_digest
            != "sha256:" + self.query_row.panel_png_digest
            or type(self.disposition) is not Disposition
            or type(self.outcome) is not PositiveProseQueryOutcome
            or self.disposition is not self.query_row.observer_artifact.observation.disposition
            or self.outcome is not _outcome(self.disposition)
            or self.decision_digest != canonical_digest(_decision_content(self))
        ):
            raise PositiveProseTaskRunnerError("query decision differs")

    @property
    def artifact_address(self) -> str:
        return "sha256:" + self.decision_digest

    @classmethod
    def create(
        cls,
        freeze: PositiveProseTaskFreeze,
        *,
        query_evidence_bundle: PositiveProseEvidenceBundle,
        released_query_panel: ReleasedOfficialExtractedPanel,
        query_release_store_receipt: ObjectBongardWriteOnceReceipt,
    ) -> "PositiveProseQueryDecision":
        if type(freeze) is not PositiveProseTaskFreeze:
            raise TypeError("query decision needs exact positive prose freeze")
        bundle = cold_replay_positive_prose_evidence_bundle(
            query_evidence_bundle,
            expected_artifact_address=query_evidence_bundle.artifact_address,
        )
        frozen_bundle = freeze.support_admission.evidence_bundle
        if (
            bundle.task_plan != frozen_bundle.task_plan
            or bundle.authorization_digest != frozen_bundle.authorization_digest
            or bundle.execution_precommit_digest != frozen_bundle.execution_precommit_digest
            or bundle.proposer_artifact != frozen_bundle.proposer_artifact
            or bundle.proposer_journal_terminal != frozen_bundle.proposer_journal_terminal
            or bundle.support_rows != frozen_bundle.support_rows
            or bundle.cue_digest != freeze.selected_predicate.cue.cue_digest
        ):
            raise PositiveProseTaskRunnerError("query bundle differs from frozen support custody")
        ordinal, row = _verify_released_query(
            freeze, bundle, released_query_panel, query_release_store_receipt
        )
        disposition = row.observer_artifact.observation.disposition
        values = {
            "task_freeze_digest": freeze.record_digest,
            "task_id": freeze.task_id,
            "selected_predicate": freeze.selected_predicate,
            "query_evidence_bundle": bundle,
            "released_query_panel": released_query_panel,
            "query_release_store_receipt": query_release_store_receipt,
            "query_ordinal": ordinal,
            "query_row": row,
            "disposition": disposition,
            "outcome": _outcome(disposition),
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(**values, decision_digest=canonical_digest(_decision_content(provisional)))

    def to_data(self) -> dict[str, object]:
        return {**_decision_content(self), "decision_digest": self.decision_digest, "artifact_address": self.artifact_address}

    @classmethod
    def from_data(cls, value: object) -> "PositiveProseQueryDecision":
        raw = _fields(
            value,
            {
                "schema", "runner_id", "task_freeze_digest", "task_id",
                "selected_predicate", "selected_predicate_digest",
                "query_evidence_bundle", "query_evidence_bundle_address",
                "released_query_panel", "released_query_panel_digest",
                "query_release_store_receipt", "query_release_store_receipt_digest",
                "query_ordinal", "query_row", "query_row_digest", "query_panel_id",
                "observation", "disposition", "outcome", "decision_mapping",
                "query_truth_label_present", "negative_formula_evaluated",
                "model_calls_during_decision", *_authority_data(),
                "decision_digest", "artifact_address",
            },
            "positive prose query decision",
        )
        if (
            raw["schema"] != POSITIVE_PROSE_QUERY_DECISION_SCHEMA
            or raw["runner_id"] != POSITIVE_PROSE_TASK_RUNNER_ID
            or raw["decision_mapping"] != "present-positive_certified-absent-negative_indeterminate-abstain_error-error"
            or raw["query_truth_label_present"] is not False
            or raw["negative_formula_evaluated"] is not False
            or raw["model_calls_during_decision"] != 0
            or any(raw[key] != item for key, item in _authority_data().items())
        ):
            raise PositiveProseTaskRunnerError("query decision policy differs")
        predicate = PositiveProseFrozenPredicate.from_data(raw["selected_predicate"])
        bundle = PositiveProseEvidenceBundle.from_data(raw["query_evidence_bundle"])
        released = ReleasedOfficialExtractedPanel.from_data(raw["released_query_panel"])
        receipt = ObjectBongardWriteOnceReceipt.from_data(raw["query_release_store_receipt"])
        row = PositiveProseEvidenceRow.from_data(raw["query_row"])
        result = cls(
            raw["task_freeze_digest"], raw["task_id"], predicate, bundle,
            released, receipt, raw["query_ordinal"], row,
            Disposition(raw["disposition"]), PositiveProseQueryOutcome(raw["outcome"]),
            raw["decision_digest"],
        )
        if (
            raw["selected_predicate_digest"] != predicate.predicate_digest
            or raw["query_evidence_bundle_address"] != bundle.artifact_address
            or raw["released_query_panel_digest"] != released.record_digest
            or raw["query_release_store_receipt_digest"] != receipt.record_digest
            or raw["query_row_digest"] != row.record_digest
            or raw["query_panel_id"] != row.panel_id
            or raw["observation"] != row.observer_artifact.observation.to_data()
            or raw["artifact_address"] != result.artifact_address
            or result.to_data() != dict(raw)
        ):
            raise PositiveProseTaskRunnerError("query decision is not canonical")
        return result


def cold_replay_positive_prose_query_decision(
    decision: PositiveProseQueryDecision,
    *,
    freeze: PositiveProseTaskFreeze,
    expected_artifact_address: str,
) -> PositiveProseQueryDecision:
    if type(decision) is not PositiveProseQueryDecision:
        raise TypeError("query replay needs exact positive prose decision")
    restored = PositiveProseQueryDecision.from_data(decision.to_data())
    replayed = PositiveProseQueryDecision.create(
        freeze,
        query_evidence_bundle=restored.query_evidence_bundle,
        released_query_panel=restored.released_query_panel,
        query_release_store_receipt=restored.query_release_store_receipt,
    )
    if (
        replayed != decision
        or replayed.artifact_address != _address(expected_artifact_address, "expected query decision address")
    ):
        raise PositiveProseTaskRunnerError("query decision cold replay differs")
    return replayed


__all__ = (
    "CERTIFIED_ABSENT_UPPER_THRESHOLD",
    "CONTRAST_CERTIFIED_ABSENT_REQUIRED",
    "PRESENT_LOWER_THRESHOLD",
    "PRIMARY_PRESENT_REQUIRED",
    "PositiveProseFrozenPredicate",
    "PositiveProseQueryDecision",
    "PositiveProseQueryOutcome",
    "PositiveProseSupportAdmission",
    "PositiveProseSupportStatus",
    "PositiveProseTaskFreeze",
    "PositiveProseTaskFreezeCommit",
    "PositiveProseTaskRunnerError",
    "cold_replay_positive_prose_query_decision",
    "cold_replay_positive_prose_task_commit",
    "cold_replay_positive_prose_task_freeze",
    "panel_positive_prose_task_runner_source_digest",
    "verify_positive_prose_task_commit",
    "verify_positive_prose_task_freeze",
)
