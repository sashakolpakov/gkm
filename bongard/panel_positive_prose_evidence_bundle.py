"""Exact task and journal custody for the one-positive prose lane.

The bundle is deliberately a receipt archive, not a classifier.  It binds one
support-only proposer turn, all twelve independent support observer turns, and
either no query turns or both preregistered query turns.  Dataset roles live
only in this Python custody layer; every observer still sees one ``panel.png``
and the single frozen positive cue.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
import hashlib
from typing import Any, Mapping, Sequence

from bongard.canonical import canonical_digest
from bongard.object_bongard_batch import ObjectBongardTaskPlan
from bongard.object_bongard_turn_journal import (
    NAMED_IMAGE_MODALITY,
    TURN_CLAIM_SCHEMA,
    TURN_JOURNAL_MANIFEST_SCHEMA,
    TURN_JOURNAL_PROTOCOL_ID,
    TURN_OUTCOME_SCHEMA,
    TURN_RESULT_SCHEMA,
    ObjectBongardNamedImageTurnJournalTransport,
    ObjectBongardTurnJournalSummary,
    _read_canonical as _read_journal_record,
    object_bongard_turn_journal_source_digest,
    verify_object_bongard_turn_journal,
)
from bongard.panel_positive_prose_observer import (
    PositiveProsePanelArtifact,
    PositiveProsePanelRequest,
    positive_prose_panel_output_schema,
    positive_prose_panel_prompt,
    verify_positive_prose_panel_artifact,
)
from bongard.panel_support_positive_proposer import (
    SUPPORT_POSITIVE_PRESENTATION_NAMES,
    SupportPositiveProposerArtifact,
    support_positive_proposer_output_schema,
    support_positive_proposer_prompt,
    verify_support_positive_proposer_artifact,
)
from bongard.panel_typed_codex_observer import _exact_png
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID


POSITIVE_PROSE_JOURNAL_TERMINAL_SCHEMA = (
    "gkm.bongard-positive-prose-journal-terminal.v1"
)
POSITIVE_PROSE_EVIDENCE_ROW_SCHEMA = "gkm.bongard-positive-prose-evidence-row.v1"
POSITIVE_PROSE_EVIDENCE_BUNDLE_SCHEMA = (
    "gkm.bongard-positive-prose-evidence-bundle.v1"
)
POSITIVE_PROSE_EVIDENCE_PROTOCOL_ID = (
    "bongard.positive-prose/exact-task-journal-artifacts-v1"
)
PROPOSER_TURN_KIND = "positive_prose_proposer"


class PositiveProseEvidenceError(ValueError):
    """An evidence row, journal terminal, role, or replay differs."""


class PositiveProseEvidencePhase(str, Enum):
    SUPPORT = "support"
    QUERY = "query"


class PositiveProsePanelRole(str, Enum):
    PRIMARY_SUPPORT = "primary_support"
    CONTRAST_SUPPORT = "contrast_support"
    PRIMARY_QUERY = "primary_query"
    CONTRAST_QUERY = "contrast_query"


def panel_positive_prose_evidence_bundle_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "positive_only": True,
        "prose_is_inert_data": True,
        "foil_present": False,
        "negative_formula_present": False,
        "negation_allowed": False,
        "lean_present": False,
        "lean_required": False,
        "lean_removable": True,
    }


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(type(key) is not str for key in value)
        or set(value) != expected
    ):
        raise PositiveProseEvidenceError(f"{label} fields differ")
    return value


def _address(value: object, label: str) -> str:
    if (
        type(value) is not str
        or not value.startswith("sha256:")
        or len(value) != 71
        or any(char not in "0123456789abcdef" for char in value[7:])
    ):
        raise PositiveProseEvidenceError(f"{label} must be a sha256: address")
    return value


def _raw_digest(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise PositiveProseEvidenceError(f"{label} must be a raw SHA-256")
    return value


_MANIFEST_FIELDS = {
    "schema", "protocol_id", "modality", "authorization_digest",
    "execution_precommit_digest", "task_id", "turn_kind", "prompt",
    "prompt_sha256", "output_schema", "output_schema_digest", "named_images",
    "runtime_binding", "journal_source_digest",
    "exclusive_claim_fsynced_before_transport",
    "complete_result_fsynced_before_terminal", "terminal_replay_calls_model",
    "nonterminal_claim_policy", "predicate_authority_id",
    "python_is_canonical_authority", "lean_present", "lean_required",
    "lean_removable", "lean_affects_identity_or_replay",
}
_CLAIM_FIELDS = {
    "schema", "turn_key", "manifest_digest", "authorization_digest",
    "execution_precommit_digest", "task_id", "turn_kind", "modality",
    "exclusive_create_and_fsync_before_transport",
}
_RESULT_FIELDS = {
    "schema", "turn_key", "claim_digest", "manifest_digest", "status",
    "codex_structured_result", "payload_digest", "receipt_digest",
    "failure_code", "source_exception_type",
}
_OUTCOME_FIELDS = {
    "schema", "turn_key", "claim_digest", "manifest_digest",
    "terminal_status", "result_digest", "terminal",
    "result_persisted_and_fsynced_before_terminal",
}


def _canonical_record(
    value: object, *, schema: str, fields: set[str], label: str
) -> dict[str, Any]:
    raw = _fields(value, fields | {"record_digest"}, label)
    body = {key: item for key, item in raw.items() if key != "record_digest"}
    if raw["schema"] != schema or raw["record_digest"] != "sha256:" + canonical_digest(body):
        raise PositiveProseEvidenceError(f"{label} digest differs")
    return dict(raw)


def _summary_from_data(value: object) -> ObjectBongardTurnJournalSummary:
    raw = _fields(
        value,
        {
            "schema", "manifest_digest", "turn_key", "terminal_status",
            "claim_digest", "result_digest", "outcome_digest", "record_digest",
            "predicate_authority_id", "python_is_canonical_authority",
            "lean_present", "lean_required", "lean_removable",
            "lean_affects_identity_or_replay",
        },
        "positive prose journal summary",
    )
    result = ObjectBongardTurnJournalSummary(
        raw["manifest_digest"], raw["turn_key"], raw["terminal_status"],
        raw["claim_digest"], raw["result_digest"], raw["outcome_digest"],
        raw["record_digest"],
    )
    if result.to_data() != dict(raw):
        raise PositiveProseEvidenceError("journal summary is not canonical")
    return result


def _terminal_content(value: "PositiveProseJournalTerminal") -> dict[str, object]:
    return {
        "schema": POSITIVE_PROSE_JOURNAL_TERMINAL_SCHEMA,
        "artifact_kind": value.artifact_kind,
        "artifact_digest": value.artifact_digest,
        "journal_summary": value.journal_summary.to_data(),
        "journal_manifest": dict(value.journal_manifest),
        "journal_claim": dict(value.journal_claim),
        "journal_result": dict(value.journal_result),
        "journal_outcome": dict(value.journal_outcome),
        "authorization_digest": value.authorization_digest,
        "execution_precommit_digest": value.execution_precommit_digest,
        "task_id": value.task_id,
        "turn_kind": value.turn_kind,
        "complete_canonical_journal_records_embedded": True,
        "terminal_replay_model_calls": 0,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class PositiveProseJournalTerminal:
    """Full fsynced named-image journal records for exactly one artifact."""

    artifact_kind: str
    artifact_digest: str
    journal_summary: ObjectBongardTurnJournalSummary
    journal_manifest: dict[str, Any]
    journal_claim: dict[str, Any]
    journal_result: dict[str, Any]
    journal_outcome: dict[str, Any]
    authorization_digest: str
    execution_precommit_digest: str
    task_id: str
    turn_kind: str
    terminal_digest: str

    def __post_init__(self) -> None:
        if self.artifact_kind not in {"proposer", "panel_observer"}:
            raise PositiveProseEvidenceError("journal artifact kind differs")
        _raw_digest(self.artifact_digest, "journal artifact digest")
        _raw_digest(self.terminal_digest, "journal terminal digest")
        _address(self.authorization_digest, "journal authorization digest")
        _address(self.execution_precommit_digest, "journal precommit digest")
        summary = self.journal_summary
        if type(summary) is not ObjectBongardTurnJournalSummary:
            raise TypeError("journal terminal needs exact typed summary")
        manifest = _canonical_record(
            self.journal_manifest, schema=TURN_JOURNAL_MANIFEST_SCHEMA,
            fields=_MANIFEST_FIELDS, label="positive prose journal manifest",
        )
        claim = _canonical_record(
            self.journal_claim, schema=TURN_CLAIM_SCHEMA,
            fields=_CLAIM_FIELDS, label="positive prose journal claim",
        )
        result = _canonical_record(
            self.journal_result, schema=TURN_RESULT_SCHEMA,
            fields=_RESULT_FIELDS, label="positive prose journal result",
        )
        outcome = _canonical_record(
            self.journal_outcome, schema=TURN_OUTCOME_SCHEMA,
            fields=_OUTCOME_FIELDS, label="positive prose journal outcome",
        )
        expected_turn_key = "sha256:" + canonical_digest(
            {
                "schema": "gkm.bongard-codex-turn-key.v1",
                "authorization_digest": manifest["authorization_digest"],
                "execution_precommit_digest": manifest["execution_precommit_digest"],
                "task_id": manifest["task_id"],
                "turn_kind": manifest["turn_kind"],
                "modality": NAMED_IMAGE_MODALITY,
                "manifest_digest": manifest["record_digest"],
            }
        )
        policy = {
            "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
            "python_is_canonical_authority": True,
            "lean_present": False,
            "lean_required": False,
            "lean_removable": True,
            "lean_affects_identity_or_replay": False,
        }
        if (
            summary.terminal_status not in {"success", "failure"}
            or summary.record_digest
            != "sha256:" + canonical_digest(
                {
                    key: item
                    for key, item in summary.to_data().items()
                    if key != "record_digest"
                }
            )
            or manifest["protocol_id"] != TURN_JOURNAL_PROTOCOL_ID
            or manifest["modality"] != NAMED_IMAGE_MODALITY
            or manifest["journal_source_digest"]
            != object_bongard_turn_journal_source_digest()
            or manifest["exclusive_claim_fsynced_before_transport"] is not True
            or manifest["complete_result_fsynced_before_terminal"] is not True
            or manifest["terminal_replay_calls_model"] is not False
            or manifest["nonterminal_claim_policy"] != "refuse-without-transport"
            or any(manifest[key] != item for key, item in policy.items())
            or claim["turn_key"] != expected_turn_key
            or claim["manifest_digest"] != manifest["record_digest"]
            or claim["authorization_digest"] != manifest["authorization_digest"]
            or claim["execution_precommit_digest"]
            != manifest["execution_precommit_digest"]
            or claim["task_id"] != manifest["task_id"]
            or claim["turn_kind"] != manifest["turn_kind"]
            or claim["modality"] != NAMED_IMAGE_MODALITY
            or claim["exclusive_create_and_fsync_before_transport"] is not True
            or result["turn_key"] != expected_turn_key
            or result["claim_digest"] != claim["record_digest"]
            or result["manifest_digest"] != manifest["record_digest"]
            or outcome["turn_key"] != expected_turn_key
            or outcome["claim_digest"] != claim["record_digest"]
            or outcome["manifest_digest"] != manifest["record_digest"]
            or outcome["terminal_status"] != result["status"]
            or outcome["result_digest"] != result["record_digest"]
            or outcome["terminal"] is not True
            or outcome["result_persisted_and_fsynced_before_terminal"] is not True
            or summary.manifest_digest != manifest["record_digest"]
            or summary.turn_key != expected_turn_key
            or summary.claim_digest != claim["record_digest"]
            or summary.result_digest != result["record_digest"]
            or summary.outcome_digest != outcome["record_digest"]
            or summary.terminal_status != result["status"]
            or self.authorization_digest != manifest["authorization_digest"]
            or self.execution_precommit_digest != manifest["execution_precommit_digest"]
            or self.task_id != manifest["task_id"]
            or self.turn_kind != manifest["turn_kind"]
            or self.journal_manifest != manifest
            or self.journal_claim != claim
            or self.journal_result != result
            or self.journal_outcome != outcome
            or self.terminal_digest != canonical_digest(_terminal_content(self))
        ):
            raise PositiveProseEvidenceError("journal terminal lineage differs")

    @property
    def artifact_address(self) -> str:
        return "sha256:" + self.terminal_digest

    @classmethod
    def verify_and_embed(
        cls,
        journal: ObjectBongardNamedImageTurnJournalTransport,
        *,
        artifact_kind: str,
        artifact_digest: str,
    ) -> "PositiveProseJournalTerminal":
        if type(journal) is not ObjectBongardNamedImageTurnJournalTransport:
            raise TypeError("terminal needs exact named-image journal")
        summary = verify_object_bongard_turn_journal(journal)
        if summary.terminal_status not in {"success", "failure"}:
            raise PositiveProseEvidenceError("journal is not durably terminal")
        values = {
            "artifact_kind": artifact_kind,
            "artifact_digest": artifact_digest,
            "journal_summary": summary,
            "journal_manifest": _read_journal_record(journal.manifest_path, "manifest"),
            "journal_claim": _read_journal_record(journal.claim_path, "claim"),
            "journal_result": _read_journal_record(journal.result_path, "result"),
            "journal_outcome": _read_journal_record(journal.outcome_path, "outcome"),
            "authorization_digest": journal.authorization_digest,
            "execution_precommit_digest": journal.execution_precommit_digest,
            "task_id": journal.task_id,
            "turn_kind": journal.turn_kind,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values, terminal_digest=canonical_digest(_terminal_content(provisional))
        )

    def to_data(self) -> dict[str, object]:
        return {
            **_terminal_content(self),
            "terminal_digest": self.terminal_digest,
            "artifact_address": self.artifact_address,
        }

    @classmethod
    def from_data(cls, value: object) -> "PositiveProseJournalTerminal":
        raw = _fields(
            value,
            {
                "schema", "artifact_kind", "artifact_digest", "journal_summary",
                "journal_manifest", "journal_claim", "journal_result",
                "journal_outcome", "authorization_digest",
                "execution_precommit_digest", "task_id", "turn_kind",
                "complete_canonical_journal_records_embedded",
                "terminal_replay_model_calls", *_authority_data(),
                "terminal_digest", "artifact_address",
            },
            "positive prose journal terminal",
        )
        if (
            raw["schema"] != POSITIVE_PROSE_JOURNAL_TERMINAL_SCHEMA
            or raw["complete_canonical_journal_records_embedded"] is not True
            or raw["terminal_replay_model_calls"] != 0
            or any(raw[key] != item for key, item in _authority_data().items())
        ):
            raise PositiveProseEvidenceError("journal terminal policy differs")
        result = cls(
            raw["artifact_kind"], raw["artifact_digest"],
            _summary_from_data(raw["journal_summary"]),
            dict(raw["journal_manifest"]), dict(raw["journal_claim"]),
            dict(raw["journal_result"]), dict(raw["journal_outcome"]),
            raw["authorization_digest"], raw["execution_precommit_digest"],
            raw["task_id"], raw["turn_kind"], raw["terminal_digest"],
        )
        if raw["artifact_address"] != result.artifact_address or result.to_data() != dict(raw):
            raise PositiveProseEvidenceError("journal terminal is not canonical")
        return result


def _row_content(value: "PositiveProseEvidenceRow") -> dict[str, object]:
    return {
        "schema": POSITIVE_PROSE_EVIDENCE_ROW_SCHEMA,
        "phase": value.phase.value,
        "phase_index": value.phase_index,
        "role": value.role.value,
        "panel_id": value.panel_id,
        "panel_png_base64": base64.b64encode(value.panel_png).decode("ascii"),
        "panel_png_digest": value.panel_png_digest,
        "panel_png_byte_count": len(value.panel_png),
        "observer_artifact": value.observer_artifact.to_data(),
        "observer_artifact_digest": value.observer_artifact.artifact_digest,
        "journal_terminal": value.journal_terminal.to_data(),
        "journal_terminal_digest": value.journal_terminal.terminal_digest,
        "role_or_panel_id_model_visible": False,
    }


@dataclass(frozen=True, slots=True)
class PositiveProseEvidenceRow:
    phase: PositiveProseEvidencePhase
    phase_index: int
    role: PositiveProsePanelRole
    panel_id: str
    panel_png: bytes
    observer_artifact: PositiveProsePanelArtifact
    journal_terminal: PositiveProseJournalTerminal
    record_digest: str

    def __post_init__(self) -> None:
        if type(self.phase) is not PositiveProseEvidencePhase:
            raise TypeError("evidence phase differs")
        if type(self.role) is not PositiveProsePanelRole:
            raise TypeError("evidence role differs")
        maximum = 12 if self.phase is PositiveProseEvidencePhase.SUPPORT else 2
        if type(self.phase_index) is not int or self.phase_index not in range(maximum):
            raise PositiveProseEvidenceError("evidence phase index differs")
        panel = _exact_png(self.panel_png, "positive prose evidence panel")
        if panel != self.panel_png or type(self.panel_id) is not str or not self.panel_id:
            raise PositiveProseEvidenceError("evidence panel differs")
        if type(self.observer_artifact) is not PositiveProsePanelArtifact:
            raise TypeError("row needs exact positive prose artifact")
        if type(self.journal_terminal) is not PositiveProseJournalTerminal:
            raise TypeError("row needs exact journal terminal")
        context = self.observer_artifact.request.context
        if (
            context.panel_id != self.panel_id
            or context.panel_png_digest != self.panel_png_digest
            or context.panel_png_byte_count != len(panel)
            or self.journal_terminal.artifact_kind != "panel_observer"
            or self.journal_terminal.artifact_digest
            != self.observer_artifact.artifact_digest
            or self.record_digest != canonical_digest(_row_content(self))
        ):
            raise PositiveProseEvidenceError("row panel, artifact, or digest differs")

    @property
    def panel_png_digest(self) -> str:
        return hashlib.sha256(self.panel_png).hexdigest()

    @classmethod
    def create(
        cls,
        *,
        phase: PositiveProseEvidencePhase,
        phase_index: int,
        role: PositiveProsePanelRole,
        panel_id: str,
        panel_png: bytes,
        observer_artifact: PositiveProsePanelArtifact,
        journal_terminal: PositiveProseJournalTerminal,
    ) -> "PositiveProseEvidenceRow":
        values = {
            "phase": phase, "phase_index": phase_index, "role": role,
            "panel_id": panel_id, "panel_png": _exact_png(panel_png),
            "observer_artifact": observer_artifact,
            "journal_terminal": journal_terminal,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(**values, record_digest=canonical_digest(_row_content(provisional)))

    def to_data(self) -> dict[str, object]:
        return {**_row_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: object) -> "PositiveProseEvidenceRow":
        raw = _fields(
            value,
            {
                "schema", "phase", "phase_index", "role", "panel_id",
                "panel_png_base64", "panel_png_digest", "panel_png_byte_count",
                "observer_artifact", "observer_artifact_digest",
                "journal_terminal", "journal_terminal_digest",
                "role_or_panel_id_model_visible", "record_digest",
            },
            "positive prose evidence row",
        )
        if (
            raw["schema"] != POSITIVE_PROSE_EVIDENCE_ROW_SCHEMA
            or raw["role_or_panel_id_model_visible"] is not False
        ):
            raise PositiveProseEvidenceError("evidence row policy differs")
        try:
            panel = base64.b64decode(raw["panel_png_base64"], validate=True)
        except Exception as exc:
            raise PositiveProseEvidenceError("evidence PNG is malformed") from exc
        artifact = PositiveProsePanelArtifact.from_data(raw["observer_artifact"])
        terminal = PositiveProseJournalTerminal.from_data(raw["journal_terminal"])
        result = cls(
            PositiveProseEvidencePhase(raw["phase"]), raw["phase_index"],
            PositiveProsePanelRole(raw["role"]), raw["panel_id"], panel,
            artifact, terminal, raw["record_digest"],
        )
        if (
            raw["panel_png_digest"] != result.panel_png_digest
            or raw["panel_png_byte_count"] != len(panel)
            or raw["observer_artifact_digest"] != artifact.artifact_digest
            or raw["journal_terminal_digest"] != terminal.terminal_digest
            or result.to_data() != dict(raw)
        ):
            raise PositiveProseEvidenceError("evidence row is not canonical")
        return result


def _expected_rows(task: ObjectBongardTaskPlan, query_count: int):
    support_ids = task.side_0_support_panel_ids + task.side_1_support_panel_ids
    result = [
        (
            PositiveProseEvidencePhase.SUPPORT,
            index,
            PositiveProsePanelRole.PRIMARY_SUPPORT
            if index < 6 else PositiveProsePanelRole.CONTRAST_SUPPORT,
            panel_id,
        )
        for index, panel_id in enumerate(support_ids)
    ]
    if query_count:
        result.extend(
            (
                PositiveProseEvidencePhase.QUERY, index,
                PositiveProsePanelRole.PRIMARY_QUERY
                if index == 0 else PositiveProsePanelRole.CONTRAST_QUERY,
                panel_id,
            )
            for index, panel_id in enumerate(
                (task.side_0_query_panel_id, task.side_1_query_panel_id)
            )
        )
    return tuple(result)


def _bundle_content(value: "PositiveProseEvidenceBundle") -> dict[str, object]:
    query_count = len(value.query_rows)
    receipts = [value.proposer_artifact.codex_receipt.receipt_digest]
    receipts.extend(
        row.observer_artifact.receipt.receipt_digest
        for row in value.rows
        if row.observer_artifact.receipt is not None
    )
    return {
        "schema": POSITIVE_PROSE_EVIDENCE_BUNDLE_SCHEMA,
        "protocol_id": POSITIVE_PROSE_EVIDENCE_PROTOCOL_ID,
        "protocol_source_digest": panel_positive_prose_evidence_bundle_source_digest(),
        "task_plan": value.task_plan.to_data(),
        "task_plan_digest": value.task_plan.record_digest,
        "authorization_digest": value.authorization_digest,
        "execution_precommit_digest": value.execution_precommit_digest,
        "proposer_artifact": value.proposer_artifact.to_data(),
        "proposer_artifact_digest": value.proposer_artifact.artifact_digest,
        "proposer_journal_terminal": value.proposer_journal_terminal.to_data(),
        "proposer_journal_terminal_digest": value.proposer_journal_terminal.terminal_digest,
        "rows": [item.to_data() for item in value.rows],
        "row_order": "task-plan-primary-support-then-contrast-support-then-two-query",
        "support_panel_count": len(value.support_rows),
        "query_panel_count": query_count,
        "query_phase_complete": query_count == 2,
        "query_phase_absent_or_complete": True,
        "cue_digest": value.cue_digest,
        "shared_runtime_digest": value.proposer_artifact.runtime.runtime_digest,
        "physical_model_call_count": 1 + len(value.rows),
        "physical_receipt_digests": receipts,
        "complete_journal_terminal_count": 1 + len(value.rows),
        "exact_task_panel_ids_bytes_and_roles_bound": True,
        "support_and_query_roles_model_visible": False,
        "proposer_self_estimates_are_support_admission": False,
        "cold_replay_model_call_count": 0,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class PositiveProseEvidenceBundle:
    task_plan: ObjectBongardTaskPlan
    authorization_digest: str
    execution_precommit_digest: str
    proposer_artifact: SupportPositiveProposerArtifact
    proposer_journal_terminal: PositiveProseJournalTerminal
    rows: tuple[PositiveProseEvidenceRow, ...]
    record_digest: str

    def __post_init__(self) -> None:
        if type(self.task_plan) is not ObjectBongardTaskPlan:
            raise TypeError("bundle needs exact task plan")
        _address(self.authorization_digest, "bundle authorization digest")
        _address(self.execution_precommit_digest, "bundle precommit digest")
        if type(self.proposer_artifact) is not SupportPositiveProposerArtifact:
            raise TypeError("bundle needs exact proposer artifact")
        if type(self.proposer_journal_terminal) is not PositiveProseJournalTerminal:
            raise TypeError("bundle needs exact proposer journal terminal")
        if type(self.rows) is not tuple or any(
            type(item) is not PositiveProseEvidenceRow for item in self.rows
        ):
            raise TypeError("bundle rows must be exact")
        self._verify(cold_replay=False)
        if self.record_digest != canonical_digest(_bundle_content(self)):
            raise PositiveProseEvidenceError("bundle digest differs")

    @property
    def artifact_address(self) -> str:
        return "sha256:" + self.record_digest

    @property
    def support_rows(self) -> tuple[PositiveProseEvidenceRow, ...]:
        return tuple(item for item in self.rows if item.phase is PositiveProseEvidencePhase.SUPPORT)

    @property
    def query_rows(self) -> tuple[PositiveProseEvidenceRow, ...]:
        return tuple(item for item in self.rows if item.phase is PositiveProseEvidencePhase.QUERY)

    @property
    def cue_digest(self) -> str:
        if self.proposer_artifact.rubric is None:
            raise PositiveProseEvidenceError("bundle proposer has no admitted cue")
        return self.support_rows[0].observer_artifact.request.cue.cue_digest

    @property
    def benchmark_sealable(self) -> bool:
        return self.proposer_artifact.benchmark_sealable and all(
            item.observer_artifact.benchmark_sealable for item in self.rows
        )

    def _verify(self, *, cold_replay: bool) -> None:
        support = self.support_rows
        query = self.query_rows
        if len(support) != 12 or len(query) not in {0, 2}:
            raise PositiveProseEvidenceError("bundle needs twelve support and zero or two query rows")
        observed = tuple((row.phase, row.phase_index, row.role, row.panel_id) for row in self.rows)
        if observed != _expected_rows(self.task_plan, len(query)):
            raise PositiveProseEvidenceError("bundle panel IDs, order, or roles differ")
        if len({row.panel_id for row in self.rows}) != len(self.rows):
            raise PositiveProseEvidenceError("bundle panel IDs are duplicated")
        proposer = self.proposer_artifact
        if proposer.rubric is None or proposer.proposal_gap is not None:
            raise PositiveProseEvidenceError("bundle proposer did not admit one cue")
        proposer_terminal = self.proposer_journal_terminal
        if (
            proposer_terminal.artifact_kind != "proposer"
            or proposer_terminal.artifact_digest != proposer.artifact_digest
            or proposer_terminal.authorization_digest != self.authorization_digest
            or proposer_terminal.execution_precommit_digest != self.execution_precommit_digest
            or proposer_terminal.task_id != self.task_plan.task_id
            or proposer_terminal.turn_kind != PROPOSER_TURN_KIND
        ):
            raise PositiveProseEvidenceError("proposer journal ownership differs")
        self._verify_call(
            proposer_terminal,
            prompt=support_positive_proposer_prompt(proposer.request),
            schema=support_positive_proposer_output_schema(proposer.request),
            images=tuple(zip(SUPPORT_POSITIVE_PRESENTATION_NAMES, (row.panel_png for row in support), strict=True)),
            payload=proposer.model_payload,
            receipt=proposer.codex_receipt.to_dict(),
            runtime=proposer.runtime,
        )
        proposer_provenance = proposer.transport_provenance
        if (
            proposer_provenance.kind != "production_exactly_once_journal"
            or (
                proposer_provenance.journal_manifest_digest,
                proposer_provenance.journal_turn_key,
                proposer_provenance.journal_claim_digest,
                proposer_provenance.journal_result_digest,
                proposer_provenance.journal_outcome_digest,
                proposer_provenance.journal_terminal_record_digest,
            )
            != (
                proposer_terminal.journal_summary.manifest_digest,
                proposer_terminal.journal_summary.turn_key,
                proposer_terminal.journal_summary.claim_digest,
                proposer_terminal.journal_summary.result_digest,
                proposer_terminal.journal_summary.outcome_digest,
                proposer_terminal.journal_summary.record_digest,
            )
        ):
            raise PositiveProseEvidenceError("proposer journal provenance differs")
        if tuple(
            (len(row.panel_png), row.panel_png_digest) for row in support
        ) != tuple((item.byte_count, item.content_digest) for item in proposer.request.presentation):
            raise PositiveProseEvidenceError("proposer support bytes or side order differ")
        receipt_digests = [proposer.codex_receipt.receipt_digest]
        for row in self.rows:
            artifact = row.observer_artifact
            terminal = row.journal_terminal
            expected_request = PositiveProsePanelRequest.build_from_proposer(
                artifact.request.context, proposer,
                expected_artifact_digest=proposer.artifact_digest,
            )
            turn = f"positive_prose_{row.phase.value}_{row.phase_index:02d}"
            if (
                artifact.request != expected_request
                or artifact.request.context.runtime != proposer.runtime
                or terminal.authorization_digest != self.authorization_digest
                or terminal.execution_precommit_digest != self.execution_precommit_digest
                or terminal.task_id != self.task_plan.task_id
                or terminal.turn_kind != turn
            ):
                raise PositiveProseEvidenceError("observer request or journal ownership differs")
            self._verify_call(
                terminal,
                prompt=positive_prose_panel_prompt(artifact.request),
                schema=positive_prose_panel_output_schema(artifact.request),
                images=(("panel.png", row.panel_png),),
                payload=artifact.model_payload,
                receipt=None if artifact.receipt is None else artifact.receipt.to_dict(),
                runtime=artifact.request.context.runtime,
            )
            provenance = artifact.transport_provenance
            if (
                provenance.kind != "production_exactly_once_journal"
                or provenance.journal_terminal_status
                != terminal.journal_summary.terminal_status
                or (
                    provenance.journal_manifest_digest,
                    provenance.journal_turn_key,
                    provenance.journal_claim_digest,
                    provenance.journal_result_digest,
                    provenance.journal_outcome_digest,
                    provenance.journal_terminal_record_digest,
                )
                != (
                    terminal.journal_summary.manifest_digest,
                    terminal.journal_summary.turn_key,
                    terminal.journal_summary.claim_digest,
                    terminal.journal_summary.result_digest,
                    terminal.journal_summary.outcome_digest,
                    terminal.journal_summary.record_digest,
                )
            ):
                raise PositiveProseEvidenceError("observer journal provenance differs")
            if artifact.receipt is not None:
                receipt_digests.append(artifact.receipt.receipt_digest)
        if len(receipt_digests) != len(set(receipt_digests)):
            raise PositiveProseEvidenceError("physical receipt is reused across calls")
        if cold_replay:
            verify_support_positive_proposer_artifact(
                proposer,
                tuple(row.panel_png for row in support[:6]),
                tuple(row.panel_png for row in support[6:]),
                expected_artifact_digest=proposer.artifact_digest,
                proposer_journal_terminal=proposer_terminal.journal_summary,
            )
            jobs = []
            with ThreadPoolExecutor(max_workers=8) as executor:
                for row in self.rows:
                    jobs.append(
                        executor.submit(
                            verify_positive_prose_panel_artifact,
                            row.observer_artifact,
                            row.panel_png,
                            expected_artifact_digest=row.observer_artifact.artifact_digest,
                            source_proposer_artifact=proposer,
                            expected_source_proposer_artifact_digest=proposer.artifact_digest,
                            query_journal_terminal=row.journal_terminal.journal_summary,
                            expected_request_digest=row.observer_artifact.request.request_digest,
                        )
                    )
                for future in as_completed(jobs):
                    try:
                        future.result()
                    except Exception as exc:
                        raise PositiveProseEvidenceError("observer cold replay failed") from exc

    @staticmethod
    def _verify_call(
        terminal: PositiveProseJournalTerminal,
        *,
        prompt: str,
        schema: Mapping[str, Any],
        images: Sequence[tuple[str, bytes]],
        payload: Mapping[str, Any] | None,
        receipt: Mapping[str, Any] | None,
        runtime: object,
    ) -> None:
        manifest = terminal.journal_manifest
        result = terminal.journal_result
        expected_images = [
            {"name": name, "byte_count": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            for name, data in images
        ]
        structured = result["codex_structured_result"]
        runtime_binding = manifest["runtime_binding"]
        if (
            manifest["prompt"] != prompt
            or manifest["prompt_sha256"] != hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            or manifest["output_schema"] != dict(schema)
            or manifest["output_schema_digest"] != "sha256:" + canonical_digest(schema)
            or manifest["named_images"] != expected_images
            or not isinstance(runtime_binding, Mapping)
            or runtime_binding.get("model") != runtime.model
            or runtime_binding.get("reasoning_effort") != runtime.reasoning_effort
            or runtime_binding.get("cloud_policy_cache_binding")
            != runtime.cloud_policy_cache_binding
            or runtime_binding.get("model_catalog_raw_digest")
            != runtime.model_catalog_digest
            or runtime_binding.get("expected_launcher_digest")
            != runtime.expected_launcher_digest
            or runtime_binding.get("no_tools_attestation_digest")
            != runtime.no_tools_attestation_digest
        ):
            raise PositiveProseEvidenceError("journal prompt, schema, or exact images differ")
        if result["status"] == "success":
            if (
                not isinstance(structured, Mapping)
                or set(structured) != {"payload", "receipt"}
                or structured["payload"] != payload
                or structured["receipt"] != receipt
                or result["payload_digest"] != "sha256:" + canonical_digest(payload)
                or receipt is None
                or result["receipt_digest"] != receipt["receipt_digest"]
                or result["failure_code"] is not None
                or result["source_exception_type"] is not None
            ):
                raise PositiveProseEvidenceError("successful journal result differs from artifact")
        elif (
            result["status"] != "failure"
            or payload is not None
            or receipt is not None
            or structured is not None
            or result["payload_digest"] is not None
            or result["receipt_digest"] is not None
            or type(result["failure_code"]) is not str
            or type(result["source_exception_type"]) is not str
        ):
            raise PositiveProseEvidenceError("failed journal result differs from artifact")

    @classmethod
    def create(
        cls,
        *,
        task_plan: ObjectBongardTaskPlan,
        authorization_digest: str,
        execution_precommit_digest: str,
        proposer_artifact: SupportPositiveProposerArtifact,
        proposer_journal_terminal: PositiveProseJournalTerminal,
        rows: Sequence[PositiveProseEvidenceRow],
    ) -> "PositiveProseEvidenceBundle":
        values = {
            "task_plan": task_plan,
            "authorization_digest": authorization_digest,
            "execution_precommit_digest": execution_precommit_digest,
            "proposer_artifact": proposer_artifact,
            "proposer_journal_terminal": proposer_journal_terminal,
            "rows": tuple(rows),
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(**values, record_digest=canonical_digest(_bundle_content(provisional)))

    def to_data(self) -> dict[str, object]:
        return {**_bundle_content(self), "record_digest": self.record_digest, "artifact_address": self.artifact_address}

    @classmethod
    def from_data(cls, value: object) -> "PositiveProseEvidenceBundle":
        expected = set(_bundle_content_fields()) | {"record_digest", "artifact_address"}
        raw = _fields(value, expected, "positive prose evidence bundle")
        if (
            raw["schema"] != POSITIVE_PROSE_EVIDENCE_BUNDLE_SCHEMA
            or raw["protocol_id"] != POSITIVE_PROSE_EVIDENCE_PROTOCOL_ID
            or raw["protocol_source_digest"] != panel_positive_prose_evidence_bundle_source_digest()
            or raw["row_order"] != "task-plan-primary-support-then-contrast-support-then-two-query"
            or raw["query_phase_absent_or_complete"] is not True
            or raw["exact_task_panel_ids_bytes_and_roles_bound"] is not True
            or raw["support_and_query_roles_model_visible"] is not False
            or raw["proposer_self_estimates_are_support_admission"] is not False
            or raw["cold_replay_model_call_count"] != 0
            or any(raw[key] != item for key, item in _authority_data().items())
            or type(raw["rows"]) is not list
        ):
            raise PositiveProseEvidenceError("evidence bundle policy differs")
        result = cls(
            ObjectBongardTaskPlan.from_data(raw["task_plan"]),
            raw["authorization_digest"], raw["execution_precommit_digest"],
            SupportPositiveProposerArtifact.from_data(raw["proposer_artifact"]),
            PositiveProseJournalTerminal.from_data(raw["proposer_journal_terminal"]),
            tuple(PositiveProseEvidenceRow.from_data(item) for item in raw["rows"]),
            raw["record_digest"],
        )
        if result.to_data() != dict(raw):
            raise PositiveProseEvidenceError("evidence bundle is not canonical")
        return result


def _bundle_content_fields() -> tuple[str, ...]:
    # One provisional object is enough to keep from_data's exact-field gate
    # explicit without duplicating the long policy dictionary elsewhere.
    return (
        "schema", "protocol_id", "protocol_source_digest", "task_plan",
        "task_plan_digest", "authorization_digest", "execution_precommit_digest",
        "proposer_artifact", "proposer_artifact_digest",
        "proposer_journal_terminal", "proposer_journal_terminal_digest", "rows",
        "row_order", "support_panel_count", "query_panel_count",
        "query_phase_complete", "query_phase_absent_or_complete", "cue_digest",
        "shared_runtime_digest", "physical_model_call_count",
        "physical_receipt_digests", "complete_journal_terminal_count",
        "exact_task_panel_ids_bytes_and_roles_bound",
        "support_and_query_roles_model_visible",
        "proposer_self_estimates_are_support_admission",
        "cold_replay_model_call_count", *_authority_data(),
    )


def cold_replay_positive_prose_evidence_bundle(
    bundle: PositiveProseEvidenceBundle,
    *,
    expected_artifact_address: str,
) -> PositiveProseEvidenceBundle:
    if type(bundle) is not PositiveProseEvidenceBundle:
        raise TypeError("cold replay needs exact positive prose bundle")
    expected = _address(expected_artifact_address, "expected bundle address")
    restored = PositiveProseEvidenceBundle.from_data(bundle.to_data())
    if restored.artifact_address != expected:
        raise PositiveProseEvidenceError("bundle differs from external commitment")
    restored._verify(cold_replay=True)
    return restored


__all__ = (
    "POSITIVE_PROSE_EVIDENCE_PROTOCOL_ID",
    "PROPOSER_TURN_KIND",
    "PositiveProseEvidenceBundle",
    "PositiveProseEvidenceError",
    "PositiveProseEvidencePhase",
    "PositiveProseEvidenceRow",
    "PositiveProseJournalTerminal",
    "PositiveProsePanelRole",
    "cold_replay_positive_prose_evidence_bundle",
    "panel_positive_prose_evidence_bundle_source_digest",
)
