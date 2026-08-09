"""One-positive-formula freeze, durable commit, query decision, and replay.

The runner consumes only a task-bound closed-catalog inventory.  The declared
positive side is the only version space that gates execution; the opposite
space is retained inside the inventory as a diagnostic and is never asked to
produce a coherent negative formula.  A unique positive survivor is selected
without a model call.  Multiple survivors require both an exact receipted rank
artifact and its verified exactly-once text-journal terminal.  Query evaluation
executes one frozen Python ``AllOf``: match predicts the declared positive
side, nonmatch predicts the other side, indeterminate abstains, and error stays
error.

Lean is absent and removable.  No negation, complement, polarity flip,
negative formula, arbitrary code, or caller-selected digest is accepted.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
from enum import Enum
import hashlib
import re
from typing import Any, Mapping

from bongard.canonical import canonical_digest, canonical_json
from bongard.object_bongard_batch import ObjectBongardTaskPlan
from bongard.object_bongard_release_gate import (
    ObjectBongardExecutionPrecommit,
    ObjectBongardWriteOnceReceipt,
)
from bongard.object_bongard_turn_journal import (
    TEXT_MODALITY,
    TURN_CLAIM_SCHEMA,
    TURN_JOURNAL_MANIFEST_SCHEMA,
    TURN_JOURNAL_PROTOCOL_ID,
    TURN_OUTCOME_SCHEMA,
    TURN_RESULT_SCHEMA,
    ObjectBongardTextTurnJournalTransport,
    ObjectBongardTurnJournalSummary,
    _read_canonical as _read_turn_journal_record,
    object_bongard_turn_journal_source_digest,
    verify_object_bongard_turn_journal,
)
from bongard.official_extracted_panel_archive import ReleasedOfficialExtractedPanel
from bongard.official_panel_archive import ReleasedOfficialPanel
from bongard.panel_batched_typed_codex_observer import (
    TypedBatchedAxisCodexArtifact,
    complete_whole_panel_feature_axes,
    verify_typed_batched_axis_codex_artifact,
)
from bongard.panel_feature_closed_catalog_inventory import (
    ClosedCatalogSupportInventoryStatus,
)
from bongard.panel_feature_extracted_release_gate import (
    PanelFeatureExtractedExecutionPrecommit,
)
from bongard.panel_feature_evidence_bundle import (
    PanelFeatureEvidencePanel,
    PanelFeatureEvidencePhase,
)
from bongard.panel_hierarchical_feature_evidence_bundle import (
    HierarchicalFeatureEvidencePhase,
    HierarchicalPanelFeatureEvidenceRow,
)
from bongard.panel_hierarchical_visual_adapter import (
    HierarchicalPanelCodexArtifact,
    verify_hierarchical_panel_artifact,
)
from bongard.panel_feature_observation import (
    EngineeringFeatureDisposition,
    PanelFeatureObservationSet,
)
from bongard.panel_feature_predicate import (
    AllOf,
    EngineeringDisposition,
    EngineeringQueryOutcome,
    EngineeringSupportTable,
    evaluate_engineering_all_of,
)
from bongard.panel_feature_task_bound_inventory import (
    TaskBoundClosedCatalogInventory,
)
from bongard.panel_positive_formula_ranker import (
    POSITIVE_FORMULA_MAX_RANK_CANDIDATES,
    PositiveFormulaRankArtifact,
    positive_formula_ranker_output_schema,
    positive_formula_ranker_prompt,
)
from bongard.panel_soft_ontology import NativeOrientation
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID
from bongard.transport import CodexReceipt


PRIMARY_FORMULA_SUPPORT_PHASE_SCHEMA = (
    "gkm.bongard-primary-formula-support-phase.v1"
)
PRIMARY_FORMULA_TASK_GAP_SCHEMA = "gkm.bongard-primary-formula-task-gap.v1"
PRIMARY_FORMULA_RANK_TERMINAL_SCHEMA = (
    "gkm.bongard-primary-formula-rank-journal-terminal.v1"
)
PRIMARY_FORMULA_TASK_FREEZE_SCHEMA = (
    "gkm.bongard-primary-formula-task-freeze.v1"
)
PRIMARY_FORMULA_TASK_COMMIT_SCHEMA = (
    "gkm.bongard-primary-formula-task-freeze-commit.v1"
)
PRIMARY_FORMULA_QUERY_DECISION_SCHEMA = (
    "gkm.bongard-primary-formula-query-decision.v1"
)
PRIMARY_FORMULA_TASK_RUNNER_ID = (
    "bongard.panel-feature/one-positive-task-freeze-python-v1"
)
LEGACY_QUERY_EVIDENCE_KIND = "legacy_full_catalog_batched_panel_evidence_v2"
HIERARCHICAL_QUERY_EVIDENCE_KIND = (
    "hierarchical_full_catalog_panel_evidence_v1"
)

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")


class PrimaryFormulaTaskRunnerError(RuntimeError):
    """A task-bound phase, rank terminal, freeze, commit, or query differs."""


class PrimaryFormulaSupportStatus(str, Enum):
    PRIMARY_SUPPORT_GAP = "primary_support_gap"
    UNIQUE_PRIMARY_SURVIVOR = "unique_primary_survivor"
    RANK_REQUIRED = "rank_required"
    RANK_CAPACITY_GAP = "rank_capacity_gap"


class PrimaryFormulaGapKind(str, Enum):
    NO_PRIMARY_SUPPORT_SURVIVOR = "no_primary_support_survivor"
    PRIMARY_SURVIVOR_COUNT_EXCEEDS_RANK_CAPACITY = (
        "primary_survivor_count_exceeds_rank_capacity"
    )


Precommit = ObjectBongardExecutionPrecommit | PanelFeatureExtractedExecutionPrecommit
QueryEvidencePanel = PanelFeatureEvidencePanel | HierarchicalPanelFeatureEvidenceRow


def panel_feature_primary_task_runner_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(type(key) is not str for key in value)
        or set(value) != expected
    ):
        raise PrimaryFormulaTaskRunnerError(f"{label} fields differ")
    return value


def _digest(value: object, label: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise PrimaryFormulaTaskRunnerError(f"{label} must be a raw SHA-256")
    return value


def _address(value: object, label: str) -> str:
    if type(value) is not str or _ADDRESS.fullmatch(value) is None:
        raise PrimaryFormulaTaskRunnerError(f"{label} must be a sha256: address")
    return value


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
        "one_positive_formula_only": True,
        "negative_formula_present": False,
        "negation_allowed": False,
        "complement_allowed": False,
        "polarity_flip_allowed": False,
        "arbitrary_predicate_code_allowed": False,
        "primary_version_space_only_gate": True,
        "opposite_version_space_diagnostic_only": True,
        "lean_present": False,
        "lean_required": False,
        "lean_removable": True,
        "lean_affects_identity_selection_decision_or_replay": False,
    }


def _canonical_bound(value: object) -> TaskBoundClosedCatalogInventory:
    if type(value) is not TaskBoundClosedCatalogInventory:
        raise TypeError("primary runner needs exact TaskBoundClosedCatalogInventory")
    return value


def _canonical_precommit(value: object) -> tuple[str, Precommit]:
    if type(value) is ObjectBongardExecutionPrecommit:
        restored: Precommit = value
        kind = "object_bongard_execution_precommit_v1"
    elif type(value) is PanelFeatureExtractedExecutionPrecommit:
        restored = value
        kind = "panel_feature_extracted_execution_precommit_v1"
    else:
        raise TypeError("primary runner needs one exact known execution precommit")
    return kind, restored


def _precommit_from_data(kind: object, value: object) -> Precommit:
    if kind == "object_bongard_execution_precommit_v1":
        return ObjectBongardExecutionPrecommit.from_data(value)  # type: ignore[arg-type]
    if kind == "panel_feature_extracted_execution_precommit_v1":
        return PanelFeatureExtractedExecutionPrecommit.from_data(value)
    raise PrimaryFormulaTaskRunnerError("execution precommit kind differs")


def _verify_precommit_task(precommit: Precommit, task: ObjectBongardTaskPlan) -> None:
    support = set(task.side_0_support_panel_ids + task.side_1_support_panel_ids)
    query = {task.side_0_query_panel_id, task.side_1_query_panel_id}
    if (
        task.task_id not in precommit.selected_task_ids
        or not support <= set(precommit.authorized_support_panel_ids)
        or not query <= set(precommit.sealed_query_panel_ids)
        or support & query
    ):
        raise PrimaryFormulaTaskRunnerError(
            "execution precommit does not bind the task support/query partition"
        )


def _support_phase_values(
    bound: TaskBoundClosedCatalogInventory,
) -> tuple[PrimaryFormulaSupportStatus, PrimaryFormulaGapKind | None, int]:
    survivors = len(bound.inventory.primary_version_space.survivor_formula_digests)
    status, kind = classify_primary_formula_survivor_count(survivors)
    return status, kind, survivors


def classify_primary_formula_survivor_count(
    survivors: int,
) -> tuple[PrimaryFormulaSupportStatus, PrimaryFormulaGapKind | None]:
    """Total count classifier used before the bounded rank-input constructor."""

    if type(survivors) is not int or survivors < 0:
        raise PrimaryFormulaTaskRunnerError("primary survivor count is invalid")
    if survivors == 0:
        return (
            PrimaryFormulaSupportStatus.PRIMARY_SUPPORT_GAP,
            PrimaryFormulaGapKind.NO_PRIMARY_SUPPORT_SURVIVOR,
        )
    if survivors == 1:
        return PrimaryFormulaSupportStatus.UNIQUE_PRIMARY_SURVIVOR, None
    if survivors > POSITIVE_FORMULA_MAX_RANK_CANDIDATES:
        return (
            PrimaryFormulaSupportStatus.RANK_CAPACITY_GAP,
            PrimaryFormulaGapKind.PRIMARY_SURVIVOR_COUNT_EXCEEDS_RANK_CAPACITY,
        )
    return PrimaryFormulaSupportStatus.RANK_REQUIRED, None


def _gap_content(value: "PrimaryFormulaTaskGap") -> dict[str, object]:
    return {
        "schema": PRIMARY_FORMULA_TASK_GAP_SCHEMA,
        "kind": value.kind.value,
        "task_bound_inventory_address": value.task_bound_inventory_address,
        "primary_version_space_digest": value.primary_version_space_digest,
        "primary_survivor_count": value.primary_survivor_count,
        "rank_candidate_capacity": POSITIVE_FORMULA_MAX_RANK_CANDIDATES,
        "query_release_authorized": False,
        "typed_gap_not_exception": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class PrimaryFormulaTaskGap:
    kind: PrimaryFormulaGapKind
    task_bound_inventory_address: str
    primary_version_space_digest: str
    primary_survivor_count: int
    gap_digest: str

    def __post_init__(self) -> None:
        _address(self.task_bound_inventory_address, "gap inventory address")
        _digest(self.primary_version_space_digest, "gap version-space digest")
        if (
            type(self.kind) is not PrimaryFormulaGapKind
            or type(self.primary_survivor_count) is not int
            or self.primary_survivor_count < 0
            or (
                self.kind is PrimaryFormulaGapKind.NO_PRIMARY_SUPPORT_SURVIVOR
                and self.primary_survivor_count != 0
            )
            or (
                self.kind
                is PrimaryFormulaGapKind.PRIMARY_SURVIVOR_COUNT_EXCEEDS_RANK_CAPACITY
                and self.primary_survivor_count
                <= POSITIVE_FORMULA_MAX_RANK_CANDIDATES
            )
        ):
            raise PrimaryFormulaTaskRunnerError("primary formula gap differs")
        _digest(self.gap_digest, "primary formula gap digest")
        if self.gap_digest != canonical_digest(_gap_content(self)):
            raise PrimaryFormulaTaskRunnerError("primary formula gap digest differs")

    @classmethod
    def create(
        cls,
        bound: TaskBoundClosedCatalogInventory,
        kind: PrimaryFormulaGapKind,
    ) -> "PrimaryFormulaTaskGap":
        space = bound.inventory.primary_version_space
        values = {
            "kind": kind,
            "task_bound_inventory_address": bound.artifact_address,
            "primary_version_space_digest": space.version_space_digest,
            "primary_survivor_count": len(space.survivor_formula_digests),
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(**values, gap_digest=canonical_digest(_gap_content(provisional)))

    def to_data(self) -> dict[str, object]:
        return {**_gap_content(self), "gap_digest": self.gap_digest}

    @classmethod
    def from_data(cls, value: object) -> "PrimaryFormulaTaskGap":
        raw = _fields(
            value,
            {
                "schema",
                "kind",
                "task_bound_inventory_address",
                "primary_version_space_digest",
                "primary_survivor_count",
                "rank_candidate_capacity",
                "query_release_authorized",
                "typed_gap_not_exception",
                *_authority_data(),
                "gap_digest",
            },
            "primary formula task gap",
        )
        if (
            raw["schema"] != PRIMARY_FORMULA_TASK_GAP_SCHEMA
            or raw["rank_candidate_capacity"]
            != POSITIVE_FORMULA_MAX_RANK_CANDIDATES
            or raw["query_release_authorized"] is not False
            or raw["typed_gap_not_exception"] is not True
            or any(raw[name] != item for name, item in _authority_data().items())
        ):
            raise PrimaryFormulaTaskRunnerError("primary formula gap policy differs")
        result = cls(
            PrimaryFormulaGapKind(raw["kind"]),
            raw["task_bound_inventory_address"],
            raw["primary_version_space_digest"],
            raw["primary_survivor_count"],
            raw["gap_digest"],
        )
        if result.to_data() != dict(raw):
            raise PrimaryFormulaTaskRunnerError("primary formula gap is not canonical")
        return result


def _phase_content(value: "PrimaryFormulaSupportPhase") -> dict[str, object]:
    return {
        "schema": PRIMARY_FORMULA_SUPPORT_PHASE_SCHEMA,
        "runner_id": PRIMARY_FORMULA_TASK_RUNNER_ID,
        "task_bound_inventory": value.task_bound_inventory.to_data(),
        "task_bound_inventory_address": value.task_bound_inventory.artifact_address,
        "primary_version_space_digest": value.primary_version_space_digest,
        "primary_survivor_count": value.primary_survivor_count,
        "status": value.status.value,
        "gap": None if value.gap is None else value.gap.to_data(),
        "query_release_authorized": False,
        "model_calls": 0,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class PrimaryFormulaSupportPhase:
    """Typed support/rank-capacity phase; it never releases query pixels."""

    task_bound_inventory: TaskBoundClosedCatalogInventory
    primary_version_space_digest: str
    primary_survivor_count: int
    status: PrimaryFormulaSupportStatus
    gap: PrimaryFormulaTaskGap | None
    record_digest: str

    def __post_init__(self) -> None:
        bound = _canonical_bound(self.task_bound_inventory)
        status, kind, count = _support_phase_values(bound)
        expected_gap = None if kind is None else PrimaryFormulaTaskGap.create(bound, kind)
        if (
            self.primary_version_space_digest
            != bound.inventory.primary_version_space.version_space_digest
            or self.primary_survivor_count != count
            or self.status is not status
            or self.gap != expected_gap
        ):
            raise PrimaryFormulaTaskRunnerError("primary support phase differs")
        _digest(self.record_digest, "primary support phase digest")
        if self.record_digest != canonical_digest(_phase_content(self)):
            raise PrimaryFormulaTaskRunnerError("primary support phase digest differs")

    @property
    def artifact_address(self) -> str:
        return "sha256:" + self.record_digest

    @classmethod
    def create(
        cls, bound_inventory: TaskBoundClosedCatalogInventory
    ) -> "PrimaryFormulaSupportPhase":
        bound = _canonical_bound(bound_inventory)
        status, kind, count = _support_phase_values(bound)
        values = {
            "task_bound_inventory": bound,
            "primary_version_space_digest": (
                bound.inventory.primary_version_space.version_space_digest
            ),
            "primary_survivor_count": count,
            "status": status,
            "gap": None if kind is None else PrimaryFormulaTaskGap.create(bound, kind),
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(**values, record_digest=canonical_digest(_phase_content(provisional)))

    def to_data(self) -> dict[str, object]:
        return {**_phase_content(self), "record_digest": self.record_digest, "artifact_address": self.artifact_address}

    @classmethod
    def from_data(cls, value: object) -> "PrimaryFormulaSupportPhase":
        raw = _fields(
            value,
            {
                "schema",
                "runner_id",
                "task_bound_inventory",
                "task_bound_inventory_address",
                "primary_version_space_digest",
                "primary_survivor_count",
                "status",
                "gap",
                "query_release_authorized",
                "model_calls",
                *_authority_data(),
                "record_digest",
                "artifact_address",
            },
            "primary formula support phase",
        )
        if (
            raw["schema"] != PRIMARY_FORMULA_SUPPORT_PHASE_SCHEMA
            or raw["runner_id"] != PRIMARY_FORMULA_TASK_RUNNER_ID
            or raw["query_release_authorized"] is not False
            or raw["model_calls"] != 0
            or any(raw[name] != item for name, item in _authority_data().items())
        ):
            raise PrimaryFormulaTaskRunnerError("primary support phase policy differs")
        bound = TaskBoundClosedCatalogInventory.from_data(
            raw["task_bound_inventory"]
        )
        result = cls(
            bound,
            raw["primary_version_space_digest"],
            raw["primary_survivor_count"],
            PrimaryFormulaSupportStatus(raw["status"]),
            None if raw["gap"] is None else PrimaryFormulaTaskGap.from_data(raw["gap"]),
            raw["record_digest"],
        )
        if (
            raw["task_bound_inventory_address"] != bound.artifact_address
            or raw["artifact_address"] != result.artifact_address
            or result.to_data() != dict(raw)
        ):
            raise PrimaryFormulaTaskRunnerError("primary support phase is not canonical")
        return result


def _journal_summary_content(
    value: ObjectBongardTurnJournalSummary,
) -> dict[str, object]:
    return {
        "schema": "gkm.bongard-codex-turn-journal-summary.v1",
        "manifest_digest": value.manifest_digest,
        "turn_key": value.turn_key,
        "terminal_status": value.terminal_status,
        "claim_digest": value.claim_digest,
        "result_digest": value.result_digest,
        "outcome_digest": value.outcome_digest,
        "record_digest": value.record_digest,
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "lean_present": False,
        "lean_required": False,
        "lean_removable": True,
        "lean_affects_identity_or_replay": False,
    }


def _canonical_journal_summary(
    value: object,
) -> ObjectBongardTurnJournalSummary:
    if type(value) is not ObjectBongardTurnJournalSummary:
        raise TypeError("rank terminal needs exact ObjectBongardTurnJournalSummary")
    data = _journal_summary_content(value)
    body = dict(data)
    record_digest = body.pop("record_digest")
    if (
        value.terminal_status != "success"
        or value.claim_digest is None
        or value.result_digest is None
        or value.outcome_digest is None
        or any(
            type(item) is not str or _ADDRESS.fullmatch(item) is None
            for item in (
                value.manifest_digest,
                value.turn_key,
                value.claim_digest,
                value.result_digest,
                value.outcome_digest,
                value.record_digest,
            )
        )
        or record_digest != "sha256:" + canonical_digest(body)
        or value.to_data() != data
    ):
        raise PrimaryFormulaTaskRunnerError(
            "rank journal summary is not one canonical successful terminal"
        )
    return value


def _journal_summary_from_data(value: object) -> ObjectBongardTurnJournalSummary:
    raw = _fields(
        value,
        {
            "schema",
            "manifest_digest",
            "turn_key",
            "terminal_status",
            "claim_digest",
            "result_digest",
            "outcome_digest",
            "record_digest",
            "predicate_authority_id",
            "python_is_canonical_authority",
            "lean_present",
            "lean_required",
            "lean_removable",
            "lean_affects_identity_or_replay",
        },
        "rank journal summary",
    )
    result = ObjectBongardTurnJournalSummary(
        raw["manifest_digest"],
        raw["turn_key"],
        raw["terminal_status"],
        raw["claim_digest"],
        raw["result_digest"],
        raw["outcome_digest"],
        raw["record_digest"],
    )
    _canonical_journal_summary(result)
    if result.to_data() != dict(raw):
        raise PrimaryFormulaTaskRunnerError("rank journal summary differs")
    return result


def _rank_terminal_content(
    value: "PrimaryFormulaRankJournalTerminal",
) -> dict[str, object]:
    return {
        "schema": PRIMARY_FORMULA_RANK_TERMINAL_SCHEMA,
        "rank_artifact": value.rank_artifact.to_data(),
        "rank_artifact_address": value.rank_artifact.artifact_address,
        "journal_summary": value.journal_summary.to_data(),
        "journal_summary_digest": value.journal_summary.record_digest,
        "journal_manifest": dict(value.journal_manifest),
        "journal_claim": dict(value.journal_claim),
        "journal_result": dict(value.journal_result),
        "journal_outcome": dict(value.journal_outcome),
        "authorization_digest": value.authorization_digest,
        "execution_precommit_digest": value.execution_precommit_digest,
        "task_id": value.task_id,
        "turn_kind": value.turn_kind,
        "replayed_model_payload": dict(value.replayed_model_payload),
        "replayed_model_payload_digest": canonical_digest(
            value.replayed_model_payload
        ),
        "replayed_receipt": value.replayed_receipt.to_dict(),
        "replayed_receipt_digest": value.replayed_receipt.receipt_digest,
        "rank_artifact_and_journal_receipt_identical": True,
        "rank_artifact_and_journal_payload_identical": True,
        "successful_terminal_verified_before_embedding": True,
        "complete_canonical_journal_records_embedded": True,
        "offline_journal_digest_lineage_verified": True,
        "terminal_replay_model_calls": 0,
        **_authority_data(),
    }


def _canonical_journal_record(
    value: object,
    *,
    schema: str,
    fields: set[str],
    label: str,
) -> dict[str, Any]:
    raw = _fields(value, fields | {"record_digest"}, label)
    if raw["schema"] != schema:
        raise PrimaryFormulaTaskRunnerError(f"{label} schema differs")
    body = {key: item for key, item in raw.items() if key != "record_digest"}
    if raw["record_digest"] != "sha256:" + canonical_digest(body):
        raise PrimaryFormulaTaskRunnerError(f"{label} digest differs")
    return dict(raw)


def _verify_embedded_rank_journal(
    *,
    artifact: PositiveFormulaRankArtifact,
    summary: ObjectBongardTurnJournalSummary,
    manifest: object,
    claim: object,
    result: object,
    outcome: object,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest_row = _canonical_journal_record(
        manifest,
        schema=TURN_JOURNAL_MANIFEST_SCHEMA,
        fields={
            "schema",
            "protocol_id",
            "modality",
            "authorization_digest",
            "execution_precommit_digest",
            "task_id",
            "turn_kind",
            "prompt",
            "prompt_sha256",
            "output_schema",
            "output_schema_digest",
            "named_images",
            "runtime_binding",
            "journal_source_digest",
            "exclusive_claim_fsynced_before_transport",
            "complete_result_fsynced_before_terminal",
            "terminal_replay_calls_model",
            "nonterminal_claim_policy",
            "predicate_authority_id",
            "python_is_canonical_authority",
            "lean_present",
            "lean_required",
            "lean_removable",
            "lean_affects_identity_or_replay",
        },
        label="rank journal manifest",
    )
    claim_row = _canonical_journal_record(
        claim,
        schema=TURN_CLAIM_SCHEMA,
        fields={
            "schema",
            "turn_key",
            "manifest_digest",
            "authorization_digest",
            "execution_precommit_digest",
            "task_id",
            "turn_kind",
            "modality",
            "exclusive_create_and_fsync_before_transport",
        },
        label="rank journal claim",
    )
    result_row = _canonical_journal_record(
        result,
        schema=TURN_RESULT_SCHEMA,
        fields={
            "schema",
            "turn_key",
            "claim_digest",
            "manifest_digest",
            "status",
            "codex_structured_result",
            "payload_digest",
            "receipt_digest",
            "failure_code",
            "source_exception_type",
        },
        label="rank journal result",
    )
    outcome_row = _canonical_journal_record(
        outcome,
        schema=TURN_OUTCOME_SCHEMA,
        fields={
            "schema",
            "turn_key",
            "claim_digest",
            "manifest_digest",
            "terminal_status",
            "result_digest",
            "terminal",
            "result_persisted_and_fsynced_before_terminal",
        },
        label="rank journal outcome",
    )
    structured = result_row.get("codex_structured_result")
    expected_prompt = positive_formula_ranker_prompt(artifact.rank_input)
    expected_schema = positive_formula_ranker_output_schema(artifact.rank_input)
    expected_turn_key = "sha256:" + canonical_digest(
        {
            "schema": "gkm.bongard-codex-turn-key.v1",
            "authorization_digest": manifest_row["authorization_digest"],
            "execution_precommit_digest": manifest_row[
                "execution_precommit_digest"
            ],
            "task_id": manifest_row["task_id"],
            "turn_kind": manifest_row["turn_kind"],
            "modality": TEXT_MODALITY,
            "manifest_digest": manifest_row["record_digest"],
        }
    )
    journal_policy = {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "lean_present": False,
        "lean_required": False,
        "lean_removable": True,
        "lean_affects_identity_or_replay": False,
    }
    if (
        manifest_row["protocol_id"] != TURN_JOURNAL_PROTOCOL_ID
        or manifest_row["modality"] != TEXT_MODALITY
        or manifest_row["prompt"] != expected_prompt
        or manifest_row["prompt_sha256"]
        != hashlib.sha256(expected_prompt.encode("utf-8")).hexdigest()
        or manifest_row["output_schema"] != expected_schema
        or manifest_row["output_schema_digest"]
        != "sha256:" + canonical_digest(expected_schema)
        or manifest_row["named_images"] != []
        or manifest_row["journal_source_digest"]
        != object_bongard_turn_journal_source_digest()
        or manifest_row["exclusive_claim_fsynced_before_transport"] is not True
        or manifest_row["complete_result_fsynced_before_terminal"] is not True
        or manifest_row["terminal_replay_calls_model"] is not False
        or manifest_row["nonterminal_claim_policy"]
        != "refuse-without-transport"
        or any(manifest_row[name] != item for name, item in journal_policy.items())
        or claim_row["turn_key"] != expected_turn_key
        or claim_row["manifest_digest"] != manifest_row["record_digest"]
        or claim_row["authorization_digest"]
        != manifest_row["authorization_digest"]
        or claim_row["execution_precommit_digest"]
        != manifest_row["execution_precommit_digest"]
        or claim_row["task_id"] != manifest_row["task_id"]
        or claim_row["turn_kind"] != manifest_row["turn_kind"]
        or claim_row["modality"] != TEXT_MODALITY
        or claim_row["exclusive_create_and_fsync_before_transport"] is not True
        or result_row["turn_key"] != expected_turn_key
        or result_row["claim_digest"] != claim_row["record_digest"]
        or result_row["manifest_digest"] != manifest_row["record_digest"]
        or result_row["status"] != "success"
        or not isinstance(structured, Mapping)
        or set(structured) != {"payload", "receipt"}
        or structured["payload"] != artifact.model_payload
        or structured["receipt"] != artifact.receipt.to_dict()
        or result_row["payload_digest"]
        != "sha256:" + canonical_digest(artifact.model_payload)
        or result_row["receipt_digest"] != artifact.receipt.receipt_digest
        or result_row["failure_code"] is not None
        or result_row["source_exception_type"] is not None
        or outcome_row["turn_key"] != expected_turn_key
        or outcome_row["claim_digest"] != claim_row["record_digest"]
        or outcome_row["manifest_digest"] != manifest_row["record_digest"]
        or outcome_row["terminal_status"] != "success"
        or outcome_row["result_digest"] != result_row["record_digest"]
        or outcome_row["terminal"] is not True
        or outcome_row["result_persisted_and_fsynced_before_terminal"] is not True
        or summary.manifest_digest != manifest_row["record_digest"]
        or summary.turn_key != expected_turn_key
        or summary.claim_digest != claim_row["record_digest"]
        or summary.result_digest != result_row["record_digest"]
        or summary.outcome_digest != outcome_row["record_digest"]
    ):
        raise PrimaryFormulaTaskRunnerError(
            "embedded rank journal record lineage differs"
        )
    return manifest_row, claim_row, result_row, outcome_row


@dataclass(frozen=True, slots=True)
class PrimaryFormulaRankJournalTerminal:
    """Verified successful journal terminal exactly matching one rank artifact."""

    rank_artifact: PositiveFormulaRankArtifact
    journal_summary: ObjectBongardTurnJournalSummary
    journal_manifest: dict[str, Any]
    journal_claim: dict[str, Any]
    journal_result: dict[str, Any]
    journal_outcome: dict[str, Any]
    authorization_digest: str
    execution_precommit_digest: str
    task_id: str
    turn_kind: str
    replayed_model_payload: dict[str, Any]
    replayed_receipt: CodexReceipt
    terminal_digest: str

    def __post_init__(self) -> None:
        if type(self.rank_artifact) is not PositiveFormulaRankArtifact:
            raise TypeError("rank terminal needs exact PositiveFormulaRankArtifact")
        artifact = PositiveFormulaRankArtifact.from_data(self.rank_artifact.to_data())
        summary = _canonical_journal_summary(self.journal_summary)
        manifest, claim, result, outcome = _verify_embedded_rank_journal(
            artifact=artifact,
            summary=summary,
            manifest=self.journal_manifest,
            claim=self.journal_claim,
            result=self.journal_result,
            outcome=self.journal_outcome,
        )
        if (
            not artifact.benchmark_sealable
            or artifact.transport_provenance.kind
            != "production_exactly_once_journal"
            or type(self.replayed_model_payload) is not dict
            or self.replayed_model_payload != artifact.model_payload
            or type(self.replayed_receipt) is not CodexReceipt
            or self.replayed_receipt != artifact.receipt
            or summary.terminal_status != "success"
            or type(self.task_id) is not str
            or not self.task_id
            or type(self.turn_kind) is not str
            or not self.turn_kind
            or self.journal_manifest != manifest
            or self.journal_claim != claim
            or self.journal_result != result
            or self.journal_outcome != outcome
            or self.authorization_digest != manifest["authorization_digest"]
            or self.execution_precommit_digest
            != manifest["execution_precommit_digest"]
            or self.task_id != manifest["task_id"]
            or self.turn_kind != manifest["turn_kind"]
        ):
            raise PrimaryFormulaTaskRunnerError(
                "rank artifact and durable journal terminal differ"
            )
        _address(self.authorization_digest, "rank journal authorization digest")
        _address(
            self.execution_precommit_digest,
            "rank journal execution precommit digest",
        )
        _digest(self.terminal_digest, "rank journal terminal digest")
        if self.terminal_digest != canonical_digest(_rank_terminal_content(self)):
            raise PrimaryFormulaTaskRunnerError("rank journal terminal digest differs")

    @property
    def artifact_address(self) -> str:
        return "sha256:" + self.terminal_digest

    @classmethod
    def verify_and_embed(
        cls,
        journal: ObjectBongardTextTurnJournalTransport,
        rank_artifact: PositiveFormulaRankArtifact,
    ) -> "PrimaryFormulaRankJournalTerminal":
        if type(journal) is not ObjectBongardTextTurnJournalTransport:
            raise TypeError(
                "rank terminal requires exact ObjectBongardTextTurnJournalTransport"
            )
        if type(rank_artifact) is not PositiveFormulaRankArtifact:
            raise TypeError("rank terminal requires exact PositiveFormulaRankArtifact")
        artifact = PositiveFormulaRankArtifact.from_data(rank_artifact.to_data())
        summary = verify_object_bongard_turn_journal(journal)
        _canonical_journal_summary(summary)
        runtime = journal.runtime
        # Verification above proves this turn is already terminal.  This call
        # therefore replays the fsynced result and cannot invoke the underlying
        # transport.
        before_fresh = journal.fresh_call_count
        result = journal(
            journal.expected_prompt,
            journal.expected_output_schema,
            model=runtime.model,
            reasoning_effort=runtime.reasoning_effort,
            minutes=runtime.minutes,
            verbose=runtime.verbose,
            executable=runtime.executable,
            cloud_policy_cache_snapshot=runtime.cloud_policy_cache_snapshot,
            model_catalog_snapshot=runtime.model_catalog_snapshot,
            tool_surface_attestation=runtime.no_tools_attestation,
            expected_launcher_digest=runtime.expected_launcher_digest,
            expected_tool_surface_attestation_digest=(
                runtime.no_tools_attestation.attestation_digest
            ),
        )
        if (
            journal.fresh_call_count != before_fresh
            or result.payload != artifact.model_payload
            or result.receipt != artifact.receipt
        ):
            raise PrimaryFormulaTaskRunnerError(
                "rank journal terminal does not replay the exact rank result"
            )
        values = {
            "rank_artifact": artifact,
            "journal_summary": summary,
            "journal_manifest": _read_turn_journal_record(
                journal.manifest_path, "rank journal manifest"
            ),
            "journal_claim": _read_turn_journal_record(
                journal.claim_path, "rank journal claim"
            ),
            "journal_result": _read_turn_journal_record(
                journal.result_path, "rank journal result"
            ),
            "journal_outcome": _read_turn_journal_record(
                journal.outcome_path, "rank journal outcome"
            ),
            "authorization_digest": journal.authorization_digest,
            "execution_precommit_digest": journal.execution_precommit_digest,
            "task_id": journal.task_id,
            "turn_kind": journal.turn_kind,
            "replayed_model_payload": dict(result.payload),
            "replayed_receipt": result.receipt,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            terminal_digest=canonical_digest(_rank_terminal_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {
            **_rank_terminal_content(self),
            "terminal_digest": self.terminal_digest,
            "artifact_address": self.artifact_address,
        }

    @classmethod
    def from_data(cls, value: object) -> "PrimaryFormulaRankJournalTerminal":
        raw = _fields(
            value,
            {
                "schema",
                "rank_artifact",
                "rank_artifact_address",
                "journal_summary",
                "journal_summary_digest",
                "journal_manifest",
                "journal_claim",
                "journal_result",
                "journal_outcome",
                "authorization_digest",
                "execution_precommit_digest",
                "task_id",
                "turn_kind",
                "replayed_model_payload",
                "replayed_model_payload_digest",
                "replayed_receipt",
                "replayed_receipt_digest",
                "rank_artifact_and_journal_receipt_identical",
                "rank_artifact_and_journal_payload_identical",
                "successful_terminal_verified_before_embedding",
                "complete_canonical_journal_records_embedded",
                "offline_journal_digest_lineage_verified",
                "terminal_replay_model_calls",
                *_authority_data(),
                "terminal_digest",
                "artifact_address",
            },
            "primary formula rank journal terminal",
        )
        if (
            raw["schema"] != PRIMARY_FORMULA_RANK_TERMINAL_SCHEMA
            or raw["rank_artifact_and_journal_receipt_identical"] is not True
            or raw["rank_artifact_and_journal_payload_identical"] is not True
            or raw["successful_terminal_verified_before_embedding"] is not True
            or raw["complete_canonical_journal_records_embedded"] is not True
            or raw["offline_journal_digest_lineage_verified"] is not True
            or raw["terminal_replay_model_calls"] != 0
            or any(raw[name] != item for name, item in _authority_data().items())
            or type(raw["replayed_model_payload"]) is not dict
        ):
            raise PrimaryFormulaTaskRunnerError("rank journal terminal policy differs")
        artifact = PositiveFormulaRankArtifact.from_data(raw["rank_artifact"])
        summary = _journal_summary_from_data(raw["journal_summary"])
        receipt_raw = artifact.receipt.to_dict()
        if raw["replayed_receipt"] != receipt_raw:
            raise PrimaryFormulaTaskRunnerError("rank journal receipt differs")
        result = cls(
            artifact,
            summary,
            dict(raw["journal_manifest"]),
            dict(raw["journal_claim"]),
            dict(raw["journal_result"]),
            dict(raw["journal_outcome"]),
            raw["authorization_digest"],
            raw["execution_precommit_digest"],
            raw["task_id"],
            raw["turn_kind"],
            dict(raw["replayed_model_payload"]),
            artifact.receipt,
            raw["terminal_digest"],
        )
        if (
            raw["rank_artifact_address"] != artifact.artifact_address
            or raw["journal_summary_digest"] != summary.record_digest
            or raw["replayed_model_payload_digest"]
            != canonical_digest(result.replayed_model_payload)
            or raw["replayed_receipt_digest"] != result.replayed_receipt.receipt_digest
            or raw["artifact_address"] != result.artifact_address
            or result.to_data() != dict(raw)
        ):
            raise PrimaryFormulaTaskRunnerError(
                "rank journal terminal is not canonical"
            )
        return result


def _unique_rank_response_digest(
    phase: PrimaryFormulaSupportPhase, formula: AllOf
) -> str:
    return canonical_digest(
        {
            "schema": "gkm.bongard-primary-formula-unique-selection.v1",
            "task_bound_inventory_address": (
                phase.task_bound_inventory.artifact_address
            ),
            "primary_version_space_digest": phase.primary_version_space_digest,
            "selected_formula_digest": formula.formula_digest,
            "selection_mode": "unique_primary_support_survivor",
            "model_call_made": False,
        }
    )


def _freeze_content(value: "PrimaryFormulaTaskFreeze") -> dict[str, object]:
    bound = value.support_phase.task_bound_inventory
    return {
        "schema": PRIMARY_FORMULA_TASK_FREEZE_SCHEMA,
        "runner_id": PRIMARY_FORMULA_TASK_RUNNER_ID,
        "runner_source_digest": panel_feature_primary_task_runner_source_digest(),
        "support_phase": value.support_phase.to_data(),
        "support_phase_address": value.support_phase.artifact_address,
        "task_bound_inventory_address": bound.artifact_address,
        "task_plan": bound.task_plan.to_data(),
        "task_id": value.task_id,
        "task_plan_digest": value.task_plan_digest,
        "execution_precommit_kind": value.execution_precommit_kind,
        "execution_precommit": value.execution_precommit.to_data(),
        "execution_precommit_digest": value.execution_precommit_digest,
        "primary_orientation": NativeOrientation.SIDE0_POSITIVE.value,
        "primary_version_space_digest": value.version_space_digest,
        "version_space_digest": value.version_space_digest,
        "support_version_space_digest": value.support_version_space_digest,
        "selected_formula": value.selected_formula.to_data(),
        "selected_formula_digest": value.selected_formula.formula_digest,
        "selected_predicate_digest": value.selected_predicate_digest,
        "selection_mode": value.selection_mode,
        "rank_artifact": (
            None if value.rank_artifact is None else value.rank_artifact.to_data()
        ),
        "rank_artifact_digest": (
            None if value.rank_artifact is None else value.rank_artifact.artifact_digest
        ),
        "rank_journal_terminal": (
            None
            if value.rank_journal_terminal is None
            else value.rank_journal_terminal.to_data()
        ),
        "rank_journal_terminal_address": (
            None
            if value.rank_journal_terminal is None
            else value.rank_journal_terminal.artifact_address
        ),
        "rank_response_digest": value.rank_response_digest,
        "sealed_query_panel_ids": list(value.sealed_query_panel_ids),
        "observer_contract_digest": bound.observer_contract_digest,
        "measurement_protocol_digest": bound.measurement_protocol_digest,
        "query_bytes_included": False,
        "query_observations_included": False,
        "query_release_authorized_only_after_exact_durable_commit": True,
        "all_primary_survivors_retained": True,
        "caller_selected_digest_accepted": False,
        "unique_survivor_rank_model_calls": 0,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class PrimaryFormulaTaskFreeze:
    """Full one-positive Python predicate frozen before query release."""

    support_phase: PrimaryFormulaSupportPhase
    execution_precommit_kind: str
    execution_precommit: Precommit
    selected_formula: AllOf
    selection_mode: str
    rank_artifact: PositiveFormulaRankArtifact | None
    rank_journal_terminal: PrimaryFormulaRankJournalTerminal | None
    rank_response_digest: str
    sealed_query_panel_ids: tuple[str, str]
    record_digest: str

    def __post_init__(self) -> None:
        if type(self.support_phase) is not PrimaryFormulaSupportPhase:
            raise TypeError("freeze needs exact PrimaryFormulaSupportPhase")
        phase = self.support_phase
        kind, precommit = _canonical_precommit(self.execution_precommit)
        bound = phase.task_bound_inventory
        task = bound.task_plan
        _verify_precommit_task(precommit, task)
        space = bound.inventory.primary_version_space
        if (
            self.execution_precommit_kind != kind
            or type(self.selected_formula) is not AllOf
            or self.selected_formula.formula_digest
            not in space.survivor_formula_digests
            or self.selected_formula.native_orientation
            is not NativeOrientation.SIDE0_POSITIVE
            or self.selected_formula.vocabulary_digest
            != bound.inventory.vocabulary.vocabulary_digest
            or self.sealed_query_panel_ids != bound.sealed_query_panel_ids
            or phase.status
            not in {
                PrimaryFormulaSupportStatus.UNIQUE_PRIMARY_SURVIVOR,
                PrimaryFormulaSupportStatus.RANK_REQUIRED,
            }
            or bound.inventory.status
            is not ClosedCatalogSupportInventoryStatus.PRIMARY_VERSION_SPACE_NONEMPTY
        ):
            raise PrimaryFormulaTaskRunnerError(
                "freeze task, phase, precommit, or selected positive formula differs"
            )
        if phase.status is PrimaryFormulaSupportStatus.UNIQUE_PRIMARY_SURVIVOR:
            expected_formula = space.survivor_formulas[0]
            expected_response = _unique_rank_response_digest(phase, expected_formula)
            valid_selection = (
                self.selected_formula == expected_formula
                and self.selection_mode == "unique_primary_support_survivor"
                and self.rank_artifact is None
                and self.rank_journal_terminal is None
                and self.rank_response_digest == expected_response
            )
        else:
            if (
                type(self.rank_artifact) is not PositiveFormulaRankArtifact
                or type(self.rank_journal_terminal)
                is not PrimaryFormulaRankJournalTerminal
            ):
                valid_selection = False
            else:
                artifact = PositiveFormulaRankArtifact.from_data(
                    self.rank_artifact.to_data()
                )
                terminal = PrimaryFormulaRankJournalTerminal.from_data(
                    self.rank_journal_terminal.to_data()
                )
                try:
                    selected = artifact.resolve_selected_all_of(
                        space,
                        source_survivor_inventory_address=bound.artifact_address,
                    )
                except Exception:
                    valid_selection = False
                else:
                    valid_selection = (
                        terminal.rank_artifact == artifact
                        and terminal.execution_precommit_digest
                        == precommit.record_digest
                        and terminal.task_id == task.task_id
                        and terminal.replayed_receipt == artifact.receipt
                        and terminal.replayed_model_payload == artifact.model_payload
                        and self.selected_formula == selected
                        and self.selection_mode
                        == "verified_rank_with_durable_journal_terminal"
                        and self.rank_response_digest == artifact.artifact_digest
                    )
        _digest(self.rank_response_digest, "freeze rank response digest")
        _address(self.record_digest, "primary formula task freeze digest")
        if (
            not valid_selection
            or self.record_digest
            != "sha256:" + canonical_digest(_freeze_content(self))
        ):
            raise PrimaryFormulaTaskRunnerError(
                "primary formula freeze selection or content differs"
            )

    @property
    def task_id(self) -> str:
        return self.support_phase.task_bound_inventory.task_plan.task_id

    @property
    def task_plan_digest(self) -> str:
        return self.support_phase.task_bound_inventory.task_plan.record_digest

    @property
    def execution_precommit_digest(self) -> str:
        return self.execution_precommit.record_digest

    @property
    def version_space_digest(self) -> str:
        return self.support_phase.primary_version_space_digest

    @property
    def support_version_space_digest(self) -> str:
        return self.version_space_digest

    @property
    def selected_predicate_digest(self) -> str:
        return self.selected_formula.formula_digest

    @classmethod
    def seal(
        cls,
        *,
        support_phase: PrimaryFormulaSupportPhase,
        execution_precommit: Precommit,
        rank_artifact: PositiveFormulaRankArtifact | None = None,
        rank_journal: ObjectBongardTextTurnJournalTransport | None = None,
    ) -> "PrimaryFormulaTaskFreeze":
        if type(support_phase) is not PrimaryFormulaSupportPhase:
            raise TypeError("freeze needs exact PrimaryFormulaSupportPhase")
        phase = support_phase
        kind, precommit = _canonical_precommit(execution_precommit)
        _verify_precommit_task(precommit, phase.task_bound_inventory.task_plan)
        space = phase.task_bound_inventory.inventory.primary_version_space
        if phase.status is PrimaryFormulaSupportStatus.UNIQUE_PRIMARY_SURVIVOR:
            if rank_artifact is not None or rank_journal is not None:
                raise PrimaryFormulaTaskRunnerError(
                    "unique primary survivor forbids an unnecessary rank turn"
                )
            formula = space.survivor_formulas[0]
            terminal = None
            response = _unique_rank_response_digest(phase, formula)
            mode = "unique_primary_support_survivor"
            artifact = None
        elif phase.status is PrimaryFormulaSupportStatus.RANK_REQUIRED:
            if (
                type(rank_artifact) is not PositiveFormulaRankArtifact
                or type(rank_journal) is not ObjectBongardTextTurnJournalTransport
            ):
                raise PrimaryFormulaTaskRunnerError(
                    "multiple survivors require an exact rank artifact and durable journal"
                )
            artifact = PositiveFormulaRankArtifact.from_data(rank_artifact.to_data())
            terminal = PrimaryFormulaRankJournalTerminal.verify_and_embed(
                rank_journal, artifact
            )
            formula = artifact.resolve_selected_all_of(
                space,
                source_survivor_inventory_address=(
                    phase.task_bound_inventory.artifact_address
                ),
            )
            response = artifact.artifact_digest
            mode = "verified_rank_with_durable_journal_terminal"
        else:
            if phase.gap is not None:
                raise PrimaryFormulaTaskRunnerError(
                    f"cannot freeze typed closed phase: {phase.gap.kind.value}"
                )
            raise PrimaryFormulaTaskRunnerError("primary support phase is not freezeable")
        values = {
            "support_phase": phase,
            "execution_precommit_kind": kind,
            "execution_precommit": precommit,
            "selected_formula": formula,
            "selection_mode": mode,
            "rank_artifact": artifact,
            "rank_journal_terminal": terminal,
            "rank_response_digest": response,
            "sealed_query_panel_ids": phase.task_bound_inventory.sealed_query_panel_ids,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            record_digest="sha256:" + canonical_digest(_freeze_content(provisional)),
        )

    def resolve_selected_all_of(self) -> AllOf:
        space = self.support_phase.task_bound_inventory.inventory.primary_version_space
        matches = tuple(
            item
            for item in space.survivor_formulas
            if item.formula_digest == self.selected_formula.formula_digest
        )
        if len(matches) != 1 or matches[0] != self.selected_formula:
            raise PrimaryFormulaTaskRunnerError(
                "frozen formula is not one exact primary survivor"
            )
        return matches[0]

    def to_data(self) -> dict[str, object]:
        return {**_freeze_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: object) -> "PrimaryFormulaTaskFreeze":
        raw = _fields(
            value,
            {
                "schema",
                "runner_id",
                "runner_source_digest",
                "support_phase",
                "support_phase_address",
                "task_bound_inventory_address",
                "task_plan",
                "task_id",
                "task_plan_digest",
                "execution_precommit_kind",
                "execution_precommit",
                "execution_precommit_digest",
                "primary_orientation",
                "primary_version_space_digest",
                "version_space_digest",
                "support_version_space_digest",
                "selected_formula",
                "selected_formula_digest",
                "selected_predicate_digest",
                "selection_mode",
                "rank_artifact",
                "rank_artifact_digest",
                "rank_journal_terminal",
                "rank_journal_terminal_address",
                "rank_response_digest",
                "sealed_query_panel_ids",
                "observer_contract_digest",
                "measurement_protocol_digest",
                "query_bytes_included",
                "query_observations_included",
                "query_release_authorized_only_after_exact_durable_commit",
                "all_primary_survivors_retained",
                "caller_selected_digest_accepted",
                "unique_survivor_rank_model_calls",
                *_authority_data(),
                "record_digest",
            },
            "primary formula task freeze",
        )
        if (
            raw["schema"] != PRIMARY_FORMULA_TASK_FREEZE_SCHEMA
            or raw["runner_id"] != PRIMARY_FORMULA_TASK_RUNNER_ID
            or raw["runner_source_digest"]
            != panel_feature_primary_task_runner_source_digest()
            or raw["primary_orientation"]
            != NativeOrientation.SIDE0_POSITIVE.value
            or raw["query_bytes_included"] is not False
            or raw["query_observations_included"] is not False
            or raw["query_release_authorized_only_after_exact_durable_commit"]
            is not True
            or raw["all_primary_survivors_retained"] is not True
            or raw["caller_selected_digest_accepted"] is not False
            or raw["unique_survivor_rank_model_calls"] != 0
            or any(raw[name] != item for name, item in _authority_data().items())
            or type(raw["sealed_query_panel_ids"]) is not list
        ):
            raise PrimaryFormulaTaskRunnerError("primary formula freeze policy differs")
        phase = PrimaryFormulaSupportPhase.from_data(raw["support_phase"])
        precommit = _precommit_from_data(
            raw["execution_precommit_kind"], raw["execution_precommit"]
        )
        formula = AllOf.from_data(raw["selected_formula"])
        artifact = (
            None
            if raw["rank_artifact"] is None
            else PositiveFormulaRankArtifact.from_data(raw["rank_artifact"])
        )
        terminal = (
            None
            if raw["rank_journal_terminal"] is None
            else PrimaryFormulaRankJournalTerminal.from_data(
                raw["rank_journal_terminal"]
            )
        )
        result = cls(
            phase,
            raw["execution_precommit_kind"],
            precommit,
            formula,
            raw["selection_mode"],
            artifact,
            terminal,
            raw["rank_response_digest"],
            tuple(raw["sealed_query_panel_ids"]),
            raw["record_digest"],
        )
        bound = phase.task_bound_inventory
        expected_rank_digest = None if artifact is None else artifact.artifact_digest
        expected_terminal_address = (
            None if terminal is None else terminal.artifact_address
        )
        if (
            raw["support_phase_address"] != phase.artifact_address
            or raw["task_bound_inventory_address"] != bound.artifact_address
            or raw["task_plan"] != bound.task_plan.to_data()
            or raw["task_id"] != result.task_id
            or raw["task_plan_digest"] != result.task_plan_digest
            or raw["execution_precommit_digest"]
            != result.execution_precommit_digest
            or raw["primary_version_space_digest"] != result.version_space_digest
            or raw["version_space_digest"] != result.version_space_digest
            or raw["support_version_space_digest"]
            != result.support_version_space_digest
            or raw["selected_formula_digest"] != formula.formula_digest
            or raw["selected_predicate_digest"]
            != result.selected_predicate_digest
            or raw["rank_artifact_digest"] != expected_rank_digest
            or raw["rank_journal_terminal_address"]
            != expected_terminal_address
            or raw["observer_contract_digest"] != bound.observer_contract_digest
            or raw["measurement_protocol_digest"]
            != bound.measurement_protocol_digest
            or result.to_data() != dict(raw)
        ):
            raise PrimaryFormulaTaskRunnerError(
                "primary formula freeze is not canonical"
            )
        return result


def verify_primary_formula_task_freeze(
    freeze: PrimaryFormulaTaskFreeze,
    *,
    expected_record_digest: str,
) -> PrimaryFormulaTaskFreeze:
    """Zero-call exact replay and selected-survivor resolution."""

    if type(freeze) is not PrimaryFormulaTaskFreeze:
        raise TypeError("freeze verification needs exact PrimaryFormulaTaskFreeze")
    expected = _address(expected_record_digest, "expected freeze digest")
    restored = PrimaryFormulaTaskFreeze.from_data(freeze.to_data())
    if restored.record_digest != expected or restored != freeze:
        raise PrimaryFormulaTaskRunnerError("primary formula freeze replay differs")
    restored.resolve_selected_all_of()
    return restored


cold_replay_primary_formula_task_freeze = verify_primary_formula_task_freeze


def _canonical_store_receipt(value: object) -> ObjectBongardWriteOnceReceipt:
    if type(value) is not ObjectBongardWriteOnceReceipt:
        raise TypeError("primary runner needs exact ObjectBongardWriteOnceReceipt")
    return value


def _commit_content(value: "PrimaryFormulaTaskFreezeCommit") -> dict[str, object]:
    freeze = value.task_freeze
    receipt = value.task_freeze_store_receipt
    return {
        "schema": PRIMARY_FORMULA_TASK_COMMIT_SCHEMA,
        "runner_id": PRIMARY_FORMULA_TASK_RUNNER_ID,
        "task_freeze": freeze.to_data(),
        "task_freeze_digest": freeze.record_digest,
        "task_freeze_store_receipt": receipt.to_data(),
        "task_freeze_store_receipt_digest": receipt.record_digest,
        "exact_freeze_payload_digest": receipt.payload_digest,
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
class PrimaryFormulaTaskFreezeCommit:
    """Decision commit binding the exact durably stored freeze bytes."""

    task_freeze: PrimaryFormulaTaskFreeze
    task_freeze_store_receipt: ObjectBongardWriteOnceReceipt
    record_digest: str

    def __post_init__(self) -> None:
        if type(self.task_freeze) is not PrimaryFormulaTaskFreeze:
            raise TypeError("commit needs exact PrimaryFormulaTaskFreeze")
        freeze = self.task_freeze
        receipt = _canonical_store_receipt(self.task_freeze_store_receipt)
        payload = canonical_json(freeze.to_data()) + b"\n"
        if (
            receipt.object_kind != "task-freeze"
            or receipt.object_digest != freeze.record_digest
            or receipt.payload_digest
            != "sha256:" + hashlib.sha256(payload).hexdigest()
            or receipt.size_bytes != len(payload)
        ):
            raise PrimaryFormulaTaskRunnerError(
                "freeze store receipt does not bind exact canonical freeze bytes"
            )
        _address(self.record_digest, "primary formula commit digest")
        if self.record_digest != "sha256:" + canonical_digest(_commit_content(self)):
            raise PrimaryFormulaTaskRunnerError("primary formula commit differs")

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

    @property
    def task_freeze_digest(self) -> str:
        return self.task_freeze.record_digest

    @property
    def exact_freeze_payload_digest(self) -> str:
        return self.task_freeze_store_receipt.payload_digest

    @property
    def task_freeze_store_receipt_digest(self) -> str:
        return self.task_freeze_store_receipt.record_digest

    @classmethod
    def seal(
        cls,
        freeze: PrimaryFormulaTaskFreeze,
        freeze_receipt: ObjectBongardWriteOnceReceipt,
    ) -> "PrimaryFormulaTaskFreezeCommit":
        if type(freeze) is not PrimaryFormulaTaskFreeze:
            raise TypeError("commit needs exact PrimaryFormulaTaskFreeze")
        frozen = freeze
        receipt = _canonical_store_receipt(freeze_receipt)
        provisional = object.__new__(cls)
        object.__setattr__(provisional, "task_freeze", frozen)
        object.__setattr__(provisional, "task_freeze_store_receipt", receipt)
        return cls(
            frozen,
            receipt,
            "sha256:" + canonical_digest(_commit_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {**_commit_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: object) -> "PrimaryFormulaTaskFreezeCommit":
        raw = _fields(
            value,
            {
                "schema",
                "runner_id",
                "task_freeze",
                "task_freeze_digest",
                "task_freeze_store_receipt",
                "task_freeze_store_receipt_digest",
                "exact_freeze_payload_digest",
                "task_id",
                "task_plan_digest",
                "execution_precommit_digest",
                "version_space_digest",
                "support_version_space_digest",
                "rank_response_digest",
                "selected_predicate_digest",
                "durably_persisted_and_reloaded_before_query_release",
                "exact_canonical_freeze_bytes_bound",
                *_authority_data(),
                "record_digest",
            },
            "primary formula task freeze commit",
        )
        if (
            raw["schema"] != PRIMARY_FORMULA_TASK_COMMIT_SCHEMA
            or raw["runner_id"] != PRIMARY_FORMULA_TASK_RUNNER_ID
            or raw["durably_persisted_and_reloaded_before_query_release"] is not True
            or raw["exact_canonical_freeze_bytes_bound"] is not True
            or any(raw[name] != item for name, item in _authority_data().items())
        ):
            raise PrimaryFormulaTaskRunnerError("primary formula commit policy differs")
        freeze = PrimaryFormulaTaskFreeze.from_data(raw["task_freeze"])
        receipt = ObjectBongardWriteOnceReceipt.from_data(
            raw["task_freeze_store_receipt"]
        )
        result = cls(freeze, receipt, raw["record_digest"])
        if (
            raw["task_freeze_digest"] != result.task_freeze_digest
            or raw["task_freeze_store_receipt_digest"]
            != result.task_freeze_store_receipt_digest
            or raw["exact_freeze_payload_digest"]
            != result.exact_freeze_payload_digest
            or raw["task_id"] != result.task_id
            or raw["task_plan_digest"] != result.task_plan_digest
            or raw["execution_precommit_digest"]
            != result.execution_precommit_digest
            or raw["version_space_digest"] != result.version_space_digest
            or raw["support_version_space_digest"]
            != result.support_version_space_digest
            or raw["rank_response_digest"] != result.rank_response_digest
            or raw["selected_predicate_digest"]
            != result.selected_predicate_digest
            or result.to_data() != dict(raw)
        ):
            raise PrimaryFormulaTaskRunnerError(
                "primary formula task commit is not canonical"
            )
        return result


def verify_primary_formula_task_commit(
    commit: PrimaryFormulaTaskFreezeCommit,
    *,
    expected_record_digest: str,
    task_commit_store_receipt: ObjectBongardWriteOnceReceipt | None = None,
) -> PrimaryFormulaTaskFreezeCommit:
    """Zero-call commit replay, optionally including its own durable receipt."""

    if type(commit) is not PrimaryFormulaTaskFreezeCommit:
        raise TypeError("commit verification needs exact PrimaryFormulaTaskFreezeCommit")
    expected = _address(expected_record_digest, "expected commit digest")
    restored = PrimaryFormulaTaskFreezeCommit.from_data(commit.to_data())
    if restored.record_digest != expected or restored != commit:
        raise PrimaryFormulaTaskRunnerError("primary formula commit replay differs")
    if task_commit_store_receipt is not None:
        receipt = _canonical_store_receipt(task_commit_store_receipt)
        payload = canonical_json(restored.to_data()) + b"\n"
        if (
            receipt.object_kind != "task-decision-commit"
            or receipt.object_digest != restored.record_digest
            or receipt.payload_digest
            != "sha256:" + hashlib.sha256(payload).hexdigest()
            or receipt.size_bytes != len(payload)
        ):
            raise PrimaryFormulaTaskRunnerError(
                "task commit receipt does not bind exact canonical commit bytes"
            )
    return restored


cold_replay_primary_formula_task_commit = verify_primary_formula_task_commit


ReleasedQueryPanel = ReleasedOfficialPanel | ReleasedOfficialExtractedPanel


def _canonical_released_query_panel(
    value: object,
) -> tuple[str, ReleasedQueryPanel]:
    if type(value) is ReleasedOfficialPanel:
        restored: ReleasedQueryPanel = ReleasedOfficialPanel.from_data(value.to_data())
        kind = "official_zip_released_panel"
    elif type(value) is ReleasedOfficialExtractedPanel:
        restored = ReleasedOfficialExtractedPanel.from_data(value.to_data())
        kind = "official_extracted_released_panel"
    else:
        raise TypeError("query decision needs one exact known released panel class")
    if restored != value:
        raise PrimaryFormulaTaskRunnerError("released query panel replay differs")
    return kind, restored


def _released_query_panel_from_data(kind: object, value: object) -> ReleasedQueryPanel:
    if kind == "official_zip_released_panel":
        return ReleasedOfficialPanel.from_data(value)  # type: ignore[arg-type]
    if kind == "official_extracted_released_panel":
        return ReleasedOfficialExtractedPanel.from_data(value)
    raise PrimaryFormulaTaskRunnerError("released query panel kind differs")


def _canonical_query_evidence(
    value: object,
) -> tuple[str, QueryEvidencePanel]:
    expected_axes = tuple(
        item.axis_digest for item in complete_whole_panel_feature_axes()
    )
    if type(value) is PanelFeatureEvidencePanel:
        restored: QueryEvidencePanel = PanelFeatureEvidencePanel.from_data(
            value.to_data()
        )
        artifact = restored.batched_axis_artifact
        if (
            restored != value
            or restored.phase is not PanelFeatureEvidencePhase.QUERY
            or restored.owner_artifact is not None
            or restored.axis_artifacts
            or type(artifact) is not TypedBatchedAxisCodexArtifact
            or restored.observation_set != artifact.observation_set
            or tuple(
                item.axis.axis_digest
                for item in restored.observation_set.axis_observations
            )
            != expected_axes
            or tuple(item.axis_digest for item in artifact.request.axes)
            != expected_axes
        ):
            raise PrimaryFormulaTaskRunnerError(
                "query evidence is not one exact full-catalog batched artifact"
            )
        verify_typed_batched_axis_codex_artifact(
            artifact,
            restored.panel_png,
            expected_artifact_digest=artifact.artifact_digest,
        )
        return LEGACY_QUERY_EVIDENCE_KIND, restored
    if type(value) is HierarchicalPanelFeatureEvidenceRow:
        restored = HierarchicalPanelFeatureEvidenceRow.from_data(value.to_data())
        artifact = restored.artifact
        if (
            restored != value
            or restored.phase is not HierarchicalFeatureEvidencePhase.QUERY
            or type(artifact) is not HierarchicalPanelCodexArtifact
            or tuple(item.axis_digest for item in artifact.request.axes)
            != expected_axes
            or tuple(
                item.axis.axis_digest
                for item in artifact.observation_set.axis_observations
            )
            != expected_axes
        ):
            raise PrimaryFormulaTaskRunnerError(
                "query evidence is not one exact full-catalog hierarchical artifact"
            )
        verify_hierarchical_panel_artifact(
            artifact,
            restored.panel_png,
            expected_artifact_digest=artifact.artifact_digest,
        )
        return HIERARCHICAL_QUERY_EVIDENCE_KIND, restored
    raise TypeError(
        "query decision needs one exact known full-receipt evidence row class"
    )


def _query_evidence_from_data(
    kind: object,
    value: object,
) -> QueryEvidencePanel:
    try:
        if kind == LEGACY_QUERY_EVIDENCE_KIND:
            return PanelFeatureEvidencePanel.from_data(value)  # type: ignore[arg-type]
        if kind == HIERARCHICAL_QUERY_EVIDENCE_KIND:
            return HierarchicalPanelFeatureEvidenceRow.from_data(value)
    except (TypeError, ValueError) as exc:
        raise PrimaryFormulaTaskRunnerError(
            "query evidence does not match its exact class tag"
        ) from exc
    raise PrimaryFormulaTaskRunnerError("query evidence kind differs")


def _query_evidence_parts(
    value: QueryEvidencePanel,
) -> tuple[object, int, str, bytes, str, PanelFeatureObservationSet]:
    if type(value) is PanelFeatureEvidencePanel:
        return (
            value.phase,
            value.phase_index,
            value.panel_id,
            value.panel_png,
            value.panel_png_digest,
            value.observation_set,
        )
    if type(value) is HierarchicalPanelFeatureEvidenceRow:
        return (
            value.phase,
            value.phase_index,
            value.panel_id,
            value.panel_png,
            value.panel_png_digest,
            value.artifact.observation_set,
        )
    raise TypeError("query evidence row class differs")


def _verify_query_custody(
    *,
    freeze: PrimaryFormulaTaskFreeze | None,
    released: ReleasedQueryPanel,
    release_receipt: ObjectBongardWriteOnceReceipt,
    evidence: QueryEvidencePanel,
) -> int:
    (
        evidence_phase,
        evidence_phase_index,
        evidence_panel_id,
        evidence_panel_png,
        evidence_panel_png_digest,
        evidence_observation,
    ) = _query_evidence_parts(evidence)
    phase_is_query = (
        type(evidence) is PanelFeatureEvidencePanel
        and evidence_phase is PanelFeatureEvidencePhase.QUERY
    ) or (
        type(evidence) is HierarchicalPanelFeatureEvidenceRow
        and evidence_phase is HierarchicalFeatureEvidencePhase.QUERY
    )
    payload = canonical_json(released.to_data()) + b"\n"
    expected_object_kind = (
        "released-query-panel"
        if type(released) is ReleasedOfficialPanel
        else "released-extracted-query-panel"
    )
    if (
        release_receipt.object_kind != expected_object_kind
        or release_receipt.object_digest != released.record_digest
        or release_receipt.payload_digest
        != "sha256:" + hashlib.sha256(payload).hexdigest()
        or release_receipt.size_bytes != len(payload)
        or evidence_panel_id != released.panel_id
        or evidence_panel_png != released.exact_png_bytes
        or "sha256:" + evidence_panel_png_digest != released.exact_png_digest
        or evidence_observation.panel_digest != evidence_panel_png_digest
        or not phase_is_query
    ):
        raise PrimaryFormulaTaskRunnerError(
            "released query, durable receipt, and observer evidence differ"
        )
    if freeze is None:
        return evidence_phase_index
    if released.execution_precommit_digest != freeze.execution_precommit_digest:
        raise PrimaryFormulaTaskRunnerError(
            "released query belongs to a different execution precommit"
        )
    try:
        ordinal = freeze.sealed_query_panel_ids.index(released.panel_id)
    except ValueError as exc:
        raise PrimaryFormulaTaskRunnerError(
            "released panel is not a sealed query for the frozen task"
        ) from exc
    if evidence_phase_index != ordinal:
        raise PrimaryFormulaTaskRunnerError(
            "query evidence phase index is swapped across sealed query identities"
        )
    return ordinal


def _engineering_disposition(
    value: EngineeringFeatureDisposition,
) -> EngineeringDisposition:
    if type(value) is not EngineeringFeatureDisposition:
        raise TypeError("query observation disposition type differs")
    return {
        EngineeringFeatureDisposition.MATCH: EngineeringDisposition.MATCH,
        EngineeringFeatureDisposition.NONMATCH: EngineeringDisposition.NONMATCH,
        EngineeringFeatureDisposition.INDETERMINATE: (
            EngineeringDisposition.INDETERMINATE
        ),
        EngineeringFeatureDisposition.ERROR: EngineeringDisposition.ERROR,
    }[value]


def _query_table(
    freeze: PrimaryFormulaTaskFreeze,
    observation: PanelFeatureObservationSet,
) -> EngineeringSupportTable:
    bound = freeze.support_phase.task_bound_inventory
    if type(observation) is not PanelFeatureObservationSet:
        raise TypeError("query decision needs exact PanelFeatureObservationSet")
    restored = PanelFeatureObservationSet.from_data(observation.to_data())
    expected_axes = complete_whole_panel_feature_axes()
    if (
        restored != observation
        or tuple(item.axis for item in restored.axis_observations) != expected_axes
        or restored.observer_contract_digest != bound.observer_contract_digest
        or restored.measurement_protocol_digest
        != bound.measurement_protocol_digest
        or restored.panel_digest
        in {item[1] for item in bound.support_panel_bindings}
    ):
        raise PrimaryFormulaTaskRunnerError(
            "query observation catalog, protocol, or panel custody differs"
        )
    vocabulary = bound.inventory.vocabulary
    values = {
        (restored.panel_digest, spec.spec_digest): _engineering_disposition(
            restored.evaluate(spec)
        )
        for spec in vocabulary.specs
    }
    return EngineeringSupportTable.create(
        vocabulary, (restored.panel_digest,), values
    )


def _query_outcome(
    disposition: EngineeringDisposition,
    orientation: NativeOrientation,
) -> EngineeringQueryOutcome:
    if disposition is EngineeringDisposition.ERROR:
        return EngineeringQueryOutcome.ERROR
    if disposition is EngineeringDisposition.INDETERMINATE:
        return EngineeringQueryOutcome.ABSTAIN
    if disposition is EngineeringDisposition.MATCH:
        return (
            EngineeringQueryOutcome.SIDE0
            if orientation is NativeOrientation.SIDE0_POSITIVE
            else EngineeringQueryOutcome.SIDE1
        )
    return (
        EngineeringQueryOutcome.SIDE1
        if orientation is NativeOrientation.SIDE0_POSITIVE
        else EngineeringQueryOutcome.SIDE0
    )


def _decision_content(value: "PrimaryFormulaQueryDecision") -> dict[str, object]:
    return {
        "schema": PRIMARY_FORMULA_QUERY_DECISION_SCHEMA,
        "runner_id": PRIMARY_FORMULA_TASK_RUNNER_ID,
        "task_freeze_digest": value.task_freeze_digest,
        "task_id": value.task_id,
        "task_bound_inventory_address": value.task_bound_inventory_address,
        "primary_version_space_digest": value.primary_version_space_digest,
        "selected_formula": value.selected_formula.to_data(),
        "selected_formula_digest": value.selected_formula.formula_digest,
        "released_query_kind": value.released_query_kind,
        "released_query_panel": value.released_query_panel.to_data(),
        "released_query_panel_digest": value.released_query_panel.record_digest,
        "query_release_store_receipt": value.query_release_store_receipt.to_data(),
        "query_release_store_receipt_digest": (
            value.query_release_store_receipt.record_digest
        ),
        "query_evidence_kind": value.query_evidence_kind,
        "query_evidence_panel": value.query_evidence_panel.to_data(),
        "query_evidence_panel_digest": value.query_evidence_panel.record_digest,
        "query_ordinal": value.query_ordinal,
        "query_panel_id": value.query_panel_id,
        "query_observation": value.query_observation.to_data(),
        "query_observation_set_digest": (
            value.query_observation.observation_set_digest
        ),
        "query_table": value.query_table.to_data(),
        "query_table_digest": value.query_table.table_digest,
        "formula_disposition": value.formula_disposition.value,
        "outcome": value.outcome.value,
        "decision_rule": (
            "one-positive-all-of-match-positive-nonmatch-other-"
            "indeterminate-abstain-error-error"
        ),
        "query_truth_label_present": False,
        "negative_formula_evaluated": False,
        "model_calls_during_decision": 0,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class PrimaryFormulaQueryDecision:
    """One-formula deterministic decision over one exact query observation."""

    task_freeze_digest: str
    task_id: str
    task_bound_inventory_address: str
    primary_version_space_digest: str
    selected_formula: AllOf
    released_query_kind: str
    released_query_panel: ReleasedQueryPanel
    query_release_store_receipt: ObjectBongardWriteOnceReceipt
    query_evidence_kind: str
    query_evidence_panel: QueryEvidencePanel
    query_ordinal: int
    query_table: EngineeringSupportTable
    formula_disposition: EngineeringDisposition
    outcome: EngineeringQueryOutcome
    decision_digest: str

    def __post_init__(self) -> None:
        _address(self.task_freeze_digest, "query task freeze digest")
        _address(
            self.task_bound_inventory_address,
            "query task-bound inventory address",
        )
        _digest(self.primary_version_space_digest, "query version-space digest")
        released_kind, released = _canonical_released_query_panel(
            self.released_query_panel
        )
        receipt = _canonical_store_receipt(self.query_release_store_receipt)
        evidence_kind, evidence = _canonical_query_evidence(
            self.query_evidence_panel
        )
        ordinal = _verify_query_custody(
            freeze=None,
            released=released,
            release_receipt=receipt,
            evidence=evidence,
        )
        if (
            type(self.task_id) is not str
            or not self.task_id
            or self.released_query_kind != released_kind
            or self.query_evidence_kind != evidence_kind
            or self.query_ordinal != ordinal
            or self.query_ordinal not in (0, 1)
            or type(self.selected_formula) is not AllOf
            or type(self.query_table) is not EngineeringSupportTable
            or type(self.formula_disposition) is not EngineeringDisposition
            or type(self.outcome) is not EngineeringQueryOutcome
            or self.query_table.panel_digests
            != (self.query_observation.panel_digest,)
            or self.query_table.vocabulary.vocabulary_digest
            != self.selected_formula.vocabulary_digest
        ):
            raise PrimaryFormulaTaskRunnerError("query decision inputs differ")
        expected_disposition = evaluate_engineering_all_of(
            self.selected_formula,
            self.query_table,
            self.query_observation.panel_digest,
        )
        expected_outcome = _query_outcome(
            expected_disposition, self.selected_formula.native_orientation
        )
        if (
            self.formula_disposition is not expected_disposition
            or self.outcome is not expected_outcome
        ):
            raise PrimaryFormulaTaskRunnerError("query decision replay differs")
        _digest(self.decision_digest, "primary formula query decision digest")
        if self.decision_digest != canonical_digest(_decision_content(self)):
            raise PrimaryFormulaTaskRunnerError("query decision digest differs")

    @property
    def artifact_address(self) -> str:
        return "sha256:" + self.decision_digest

    @property
    def query_panel_id(self) -> str:
        return self.released_query_panel.panel_id

    @property
    def query_observation(self) -> PanelFeatureObservationSet:
        return _query_evidence_parts(self.query_evidence_panel)[5]

    @classmethod
    def create(
        cls,
        freeze: PrimaryFormulaTaskFreeze,
        *,
        released_query_panel: ReleasedQueryPanel,
        query_release_store_receipt: ObjectBongardWriteOnceReceipt,
        query_evidence_panel: QueryEvidencePanel,
    ) -> "PrimaryFormulaQueryDecision":
        if type(freeze) is not PrimaryFormulaTaskFreeze:
            raise TypeError("query decision needs exact PrimaryFormulaTaskFreeze")
        frozen = freeze
        released_kind, released = _canonical_released_query_panel(
            released_query_panel
        )
        receipt = _canonical_store_receipt(query_release_store_receipt)
        evidence_kind, evidence = _canonical_query_evidence(query_evidence_panel)
        ordinal = _verify_query_custody(
            freeze=frozen,
            released=released,
            release_receipt=receipt,
            evidence=evidence,
        )
        observation = _query_evidence_parts(evidence)[5]
        table = _query_table(frozen, observation)
        formula = frozen.resolve_selected_all_of()
        disposition = evaluate_engineering_all_of(
            formula, table, observation.panel_digest
        )
        values = {
            "task_freeze_digest": frozen.record_digest,
            "task_id": frozen.task_id,
            "task_bound_inventory_address": (
                frozen.support_phase.task_bound_inventory.artifact_address
            ),
            "primary_version_space_digest": frozen.version_space_digest,
            "selected_formula": formula,
            "released_query_kind": released_kind,
            "released_query_panel": released,
            "query_release_store_receipt": receipt,
            "query_evidence_kind": evidence_kind,
            "query_evidence_panel": evidence,
            "query_ordinal": ordinal,
            "query_table": table,
            "formula_disposition": disposition,
            "outcome": _query_outcome(disposition, formula.native_orientation),
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            decision_digest=canonical_digest(_decision_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {
            **_decision_content(self),
            "decision_digest": self.decision_digest,
            "artifact_address": self.artifact_address,
        }

    @classmethod
    def from_data(cls, value: object) -> "PrimaryFormulaQueryDecision":
        raw = _fields(
            value,
            {
                "schema",
                "runner_id",
                "task_freeze_digest",
                "task_id",
                "task_bound_inventory_address",
                "primary_version_space_digest",
                "selected_formula",
                "selected_formula_digest",
                "released_query_kind",
                "released_query_panel",
                "released_query_panel_digest",
                "query_release_store_receipt",
                "query_release_store_receipt_digest",
                "query_evidence_kind",
                "query_evidence_panel",
                "query_evidence_panel_digest",
                "query_ordinal",
                "query_panel_id",
                "query_observation",
                "query_observation_set_digest",
                "query_table",
                "query_table_digest",
                "formula_disposition",
                "outcome",
                "decision_rule",
                "query_truth_label_present",
                "negative_formula_evaluated",
                "model_calls_during_decision",
                *_authority_data(),
                "decision_digest",
                "artifact_address",
            },
            "primary formula query decision",
        )
        if (
            raw["schema"] != PRIMARY_FORMULA_QUERY_DECISION_SCHEMA
            or raw["runner_id"] != PRIMARY_FORMULA_TASK_RUNNER_ID
            or raw["decision_rule"]
            != "one-positive-all-of-match-positive-nonmatch-other-indeterminate-abstain-error-error"
            or raw["query_truth_label_present"] is not False
            or raw["negative_formula_evaluated"] is not False
            or raw["model_calls_during_decision"] != 0
            or any(raw[name] != item for name, item in _authority_data().items())
        ):
            raise PrimaryFormulaTaskRunnerError("query decision policy differs")
        formula = AllOf.from_data(raw["selected_formula"])
        released = _released_query_panel_from_data(
            raw["released_query_kind"], raw["released_query_panel"]
        )
        receipt = ObjectBongardWriteOnceReceipt.from_data(
            raw["query_release_store_receipt"]
        )
        evidence = _query_evidence_from_data(
            raw["query_evidence_kind"], raw["query_evidence_panel"]
        )
        observation = PanelFeatureObservationSet.from_data(raw["query_observation"])
        table = EngineeringSupportTable.from_data(raw["query_table"])
        result = cls(
            raw["task_freeze_digest"],
            raw["task_id"],
            raw["task_bound_inventory_address"],
            raw["primary_version_space_digest"],
            formula,
            raw["released_query_kind"],
            released,
            receipt,
            raw["query_evidence_kind"],
            evidence,
            raw["query_ordinal"],
            table,
            EngineeringDisposition(raw["formula_disposition"]),
            EngineeringQueryOutcome(raw["outcome"]),
            raw["decision_digest"],
        )
        if (
            raw["selected_formula_digest"] != formula.formula_digest
            or raw["released_query_panel_digest"] != released.record_digest
            or raw["query_release_store_receipt_digest"] != receipt.record_digest
            or raw["query_evidence_kind"]
            not in (
                LEGACY_QUERY_EVIDENCE_KIND,
                HIERARCHICAL_QUERY_EVIDENCE_KIND,
            )
            or raw["query_evidence_panel_digest"] != evidence.record_digest
            or raw["query_panel_id"] != result.query_panel_id
            or raw["query_observation"] != result.query_observation.to_data()
            or raw["query_observation_set_digest"]
            != observation.observation_set_digest
            or raw["query_table_digest"] != table.table_digest
            or raw["artifact_address"] != result.artifact_address
            or result.to_data() != dict(raw)
        ):
            raise PrimaryFormulaTaskRunnerError("query decision is not canonical")
        return result


def cold_replay_primary_formula_query_decision(
    decision: PrimaryFormulaQueryDecision,
    *,
    freeze: PrimaryFormulaTaskFreeze,
    expected_artifact_address: str,
) -> PrimaryFormulaQueryDecision:
    """Rebuild a query decision from embedded observations with zero calls."""

    if type(decision) is not PrimaryFormulaQueryDecision:
        raise TypeError("query replay needs exact PrimaryFormulaQueryDecision")
    if type(freeze) is not PrimaryFormulaTaskFreeze:
        raise TypeError("query replay needs exact PrimaryFormulaTaskFreeze")
    expected = _address(expected_artifact_address, "expected query decision address")
    restored = PrimaryFormulaQueryDecision.from_data(decision.to_data())
    replayed = PrimaryFormulaQueryDecision.create(
        freeze,
        released_query_panel=restored.released_query_panel,
        query_release_store_receipt=restored.query_release_store_receipt,
        query_evidence_panel=restored.query_evidence_panel,
    )
    if replayed != decision or replayed.artifact_address != expected:
        raise PrimaryFormulaTaskRunnerError("query decision cold replay differs")
    return replayed


__all__ = (
    "HIERARCHICAL_QUERY_EVIDENCE_KIND",
    "LEGACY_QUERY_EVIDENCE_KIND",
    "PRIMARY_FORMULA_TASK_RUNNER_ID",
    "POSITIVE_FORMULA_MAX_RANK_CANDIDATES",
    "PrimaryFormulaGapKind",
    "PrimaryFormulaQueryDecision",
    "PrimaryFormulaRankJournalTerminal",
    "PrimaryFormulaSupportPhase",
    "PrimaryFormulaSupportStatus",
    "PrimaryFormulaTaskFreeze",
    "PrimaryFormulaTaskFreezeCommit",
    "PrimaryFormulaTaskGap",
    "PrimaryFormulaTaskRunnerError",
    "classify_primary_formula_survivor_count",
    "cold_replay_primary_formula_query_decision",
    "cold_replay_primary_formula_task_commit",
    "cold_replay_primary_formula_task_freeze",
    "panel_feature_primary_task_runner_source_digest",
    "verify_primary_formula_task_commit",
    "verify_primary_formula_task_freeze",
)
