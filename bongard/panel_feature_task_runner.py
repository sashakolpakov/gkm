"""Pure-Python task boundary for typed panel-feature Bongard predicates.

The core consumes an exact :class:`ObjectBongardTaskPlan`, twelve support PNG
byte strings in semantic side-0-then-side-1 order, a receipted canonical
``PanelFeatureProposerResult``, and one complete ``PanelFeatureObservationSet``
per support panel.  It builds the two native positive-only engineering version
spaces without compiling prose or accepting arbitrary predicate code.

The deployed observer catalog is the batch adapter's exact, canonically
ordered, complete whole-panel axis tuple.  It is preregistered by Python and
never selected from proposer nominations.  Nominated owner-local features are
retained in the vocabulary, but evaluate deterministically as indeterminate
because their axes are outside this deployment catalog.

Zero survivors remain a typed support gap.  A unique survivor in each native
orientation needs no rank call.  Any nonempty multi-survivor space can proceed
only through an exact benchmark-sealable ``PanelFeatureRankArtifact`` bound to
the proposer, table, partitions, and both spaces; otherwise it remains a
selection gap.  The two explicit selected formula digests are sealed in a
content-addressed task freeze.  Query pixels remain sealed until that exact
freeze and its decision commit have both been durably persisted and reloaded.
Pixel release and candidate-neutral observation are separate calls: neither
callback receives the frozen predicate.  A content-addressed archive then
supports model-free replay with no callbacks.

This module is deliberately engineering-only and uncalibrated.  Python is the
executable predicate authority.  Lean is neither imported nor required and
does not affect identity, selection, decisions, or replay.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from enum import Enum
import hashlib
import re
from typing import Any, Callable, Mapping, Sequence

from bongard.canonical import canonical_digest, canonical_json
from bongard.object_bongard_batch import ObjectBongardTaskPlan
from bongard.object_bongard_release_gate import (
    ObjectBongardTaskCommitProtocol,
    ObjectBongardTaskFreezeProtocol,
    ObjectBongardWriteOnceReceipt,
    PreparedObjectBongardRelease,
    persist_object_bongard_task_commit,
    persist_object_bongard_task_freeze,
    release_object_bongard_query_panel,
    release_object_bongard_support_panel,
    verify_prepared_object_bongard_release,
)
from bongard.official_panel_archive import (
    RELEASED_PANEL_SCHEMA,
    OfficialPanelArchive,
    ReleasedOfficialPanel,
)
from bongard.official_extracted_panel_archive import (
    RELEASED_EXTRACTED_PANEL_SCHEMA,
    ReleasedOfficialExtractedPanel,
)
from bongard.panel_feature_observation import (
    EngineeringFeatureDisposition,
    FeatureAxis,
    PanelFeatureObservationSet,
)
from bongard.panel_batched_typed_codex_observer import (
    complete_whole_panel_feature_axes,
)
from bongard.panel_feature_predicate import (
    AllOf,
    EngineeringDisposition,
    EngineeringFeatureVersionSpace,
    EngineeringQueryOutcome,
    EngineeringSupportTable,
    FeatureVocabulary,
    evaluate_engineering_all_of,
)
from bongard.panel_feature_proposer import (
    PANEL_FEATURE_OBSERVER_VOCABULARY_SCHEMA,
    PANEL_FEATURE_PRESENTATION_NAMES,
    PANEL_FEATURE_PROPOSER_NOMINATION_GAP_SCHEMA,
    PANEL_FEATURE_PROPOSER_NOMINATION_SCHEMA,
    PANEL_FEATURE_PROPOSER_PROTOCOL_ID,
    PANEL_FEATURE_PROPOSER_RESULT_SCHEMA,
    PanelFeatureEstimateVector,
    PanelFeatureNomination,
    PanelFeatureNominationGap,
    PanelFeatureNominationGapCode,
    PanelFeatureObserverVocabulary,
    PanelFeatureProposerResult,
    panel_feature_proposer_contract_digest,
)
from bongard.panel_feature_ranker import (
    PanelFeatureRankArtifact,
    PanelFeatureRankInput,
)
from bongard.panel_soft_ontology import (
    LanguageGapArtifact,
    NativeFeatureProposal,
    NativeOrientation,
    PanelFeatureSpec,
    feature_catalog_digest,
)
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID


PANEL_FEATURE_TASK_RUNNER_ID = (
    "bongard.panel-feature-task/verified-support-rank-durable-freeze-python-v4"
)
PANEL_FEATURE_SUPPORT_DERIVATION_RUNNER_ID = (
    "bongard.panel-feature-task/complete-whole-panel-batch-durable-freeze-python-v3"
)
PANEL_FEATURE_TASK_ARCHIVE_SCHEMA = "gkm.bongard-panel-feature-task-archive.v4"
PANEL_FEATURE_TASK_SUPPORT_GAP_SCHEMA = (
    "gkm.bongard-panel-feature-task-support-gap.v3"
)
PANEL_FEATURE_TASK_SELECTION_GAP_SCHEMA = (
    "gkm.bongard-panel-feature-task-selection-gap.v2"
)
PANEL_FEATURE_TASK_FREEZE_SCHEMA = "gkm.bongard-panel-feature-task-freeze.v3"
PANEL_FEATURE_TASK_FREEZE_COMMIT_SCHEMA = (
    "gkm.bongard-panel-feature-task-freeze-commit.v3"
)
PANEL_FEATURE_SUPPORT_DERIVATION_SCHEMA = (
    "gkm.bongard-panel-feature-support-derivation.v1"
)
PANEL_FEATURE_SELECTED_PREDICATE_SCHEMA = (
    "gkm.bongard-panel-feature-explicit-selected-predicate.v1"
)
PANEL_FEATURE_SELECTED_PAIR_SCHEMA = (
    "gkm.bongard-panel-feature-explicit-selected-pair.v1"
)
PANEL_FEATURE_QUERY_DECISION_SCHEMA = (
    "gkm.bongard-panel-feature-explicit-query-decision.v1"
)
PANEL_FEATURE_SUPPORT_PANEL_COUNT = 12
PANEL_FEATURE_QUERY_PANEL_COUNT = 2

_RAW_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_ORIENTATIONS = (
    NativeOrientation.SIDE0_POSITIVE,
    NativeOrientation.SIDE1_POSITIVE,
)
_SIDES = ("side_0", "side_1")


class PanelFeatureTaskRunnerError(RuntimeError):
    """A task, provenance edge, observation matrix, freeze, or replay differs."""


class PanelFeatureTaskRunStatus(str, Enum):
    COMPLETE = "complete"
    SUPPORT_GAP = "support_gap"
    SELECTION_GAP = "selection_gap"


class PanelFeatureSupportDerivationStatus(str, Enum):
    UNIQUE_PAIR = "unique_pair"
    SUPPORT_GAP = "support_gap"
    SELECTION_GAP = "selection_gap"


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "implementation_language": "python",
        "engineering_only": True,
        "uncalibrated": True,
        "scientific_evidence": False,
        "benchmark_authoritative": False,
        "positive_only": True,
        "negation_allowed": False,
        "polarity_flip_allowed": False,
        "arbitrary_predicate_code_allowed": False,
        "lean_present": False,
        "lean_required": False,
        "lean_affects_identity_selection_decision_or_replay": False,
    }


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(type(key) is not str for key in value)
        or set(value) != expected
    ):
        raise PanelFeatureTaskRunnerError(f"{label} fields differ")
    return value


def _raw_digest(value: object, label: str) -> str:
    if type(value) is not str or _RAW_DIGEST.fullmatch(value) is None:
        raise PanelFeatureTaskRunnerError(
            f"{label} must be a raw lowercase SHA-256"
        )
    return value


def _address(value: object, label: str) -> str:
    if type(value) is not str or _ADDRESS.fullmatch(value) is None:
        raise PanelFeatureTaskRunnerError(f"{label} must be a sha256: address")
    return value


def _png(value: object, label: str) -> bytes:
    if (
        type(value) is not bytes
        or not value.startswith(b"\x89PNG\r\n\x1a\n")
        or len(value) <= 8
    ):
        raise PanelFeatureTaskRunnerError(f"{label} must be exact nonempty PNG bytes")
    return value


def _task(value: object) -> ObjectBongardTaskPlan:
    if type(value) is not ObjectBongardTaskPlan:
        raise TypeError("task_plan must be exact ObjectBongardTaskPlan")
    restored = ObjectBongardTaskPlan.from_data(value.to_data())
    if restored != value:
        raise PanelFeatureTaskRunnerError("task plan canonical reload differs")
    return restored


def _support_ids(task: ObjectBongardTaskPlan) -> tuple[str, ...]:
    return (*task.side_0_support_panel_ids, *task.side_1_support_panel_ids)


def _deployment_axis_catalog_data() -> list[dict[str, object]]:
    return [item.to_data() for item in complete_whole_panel_feature_axes()]


def _deployment_axis_catalog_digest() -> str:
    return canonical_digest(
        {
            "schema": "gkm.bongard-panel-feature-deployment-axis-catalog.v1",
            "axes": _deployment_axis_catalog_data(),
            "complete_whole_panel_catalog": True,
            "caller_or_candidate_selected": False,
        }
    )


def _support_pngs(value: Sequence[bytes]) -> tuple[bytes, ...]:
    if isinstance(value, (bytes, bytearray, str, Mapping)):
        raise TypeError("support PNGs must be an ordered sequence of PNG bytes")
    try:
        result = tuple(value)
    except TypeError as exc:
        raise TypeError("support_pngs must be an ordered sequence") from exc
    if len(result) != PANEL_FEATURE_SUPPORT_PANEL_COUNT:
        raise PanelFeatureTaskRunnerError(
            "support PNG sequence must contain exact side0-six then side1-six"
        )
    return tuple(_png(item, f"support panel {index}") for index, item in enumerate(result))


AuthenticatedReleasedPanel = ReleasedOfficialPanel | ReleasedOfficialExtractedPanel


def _canonical_released_panel(value: object) -> AuthenticatedReleasedPanel:
    if type(value) is ReleasedOfficialPanel:
        restored: AuthenticatedReleasedPanel = ReleasedOfficialPanel.from_data(
            value.to_data()
        )
    elif type(value) is ReleasedOfficialExtractedPanel:
        restored = ReleasedOfficialExtractedPanel.from_data(value.to_data())
    else:
        raise TypeError(
            "released panel must be an exact ZIP-backed or manifest-backed official record"
        )
    if restored != value:
        raise PanelFeatureTaskRunnerError("released panel canonical reload differs")
    return restored


def _released_panel_from_data(value: object) -> AuthenticatedReleasedPanel:
    if not isinstance(value, Mapping):
        raise PanelFeatureTaskRunnerError("released panel record differs")
    schema = value.get("schema")
    if schema == RELEASED_PANEL_SCHEMA:
        return ReleasedOfficialPanel.from_data(value)
    if schema == RELEASED_EXTRACTED_PANEL_SCHEMA:
        return ReleasedOfficialExtractedPanel.from_data(value)
    raise PanelFeatureTaskRunnerError("released panel authority schema differs")


def _released_panel_authority_identity(
    value: AuthenticatedReleasedPanel,
) -> tuple[str, ...]:
    receipt = value.release_receipt
    if type(value) is ReleasedOfficialPanel:
        return (
            "official-zip",
            receipt.release_descriptor_digest,
            receipt.archive_digest,
            receipt.central_directory_digest,
        )
    return (
        "official-extracted-manifest",
        receipt.release_descriptor_digest,
        receipt.corpus_manifest_digest,
        receipt.extracted_archive_digest,
    )


def _canonical_store_receipt(value: object) -> ObjectBongardWriteOnceReceipt:
    if type(value) is not ObjectBongardWriteOnceReceipt:
        raise TypeError("release-store receipt must be exact ObjectBongardWriteOnceReceipt")
    restored = ObjectBongardWriteOnceReceipt.from_data(value.to_data())
    if restored != value:
        raise PanelFeatureTaskRunnerError("release-store receipt canonical reload differs")
    return restored


def _released_rows(
    rows: Sequence[tuple[AuthenticatedReleasedPanel, ObjectBongardWriteOnceReceipt]],
    identifiers: Sequence[str],
    *,
    expected_execution_precommit_digest: str,
    expected_exposure_successor_digest: str,
    object_kind: str,
    label: str,
) -> tuple[
    tuple[AuthenticatedReleasedPanel, ...],
    tuple[ObjectBongardWriteOnceReceipt, ...],
    tuple[bytes, ...],
]:
    if isinstance(rows, (bytes, bytearray, str, Mapping)):
        raise TypeError(f"{label} release rows must be an ordered sequence")
    try:
        values = tuple(rows)
    except TypeError as exc:
        raise TypeError(f"{label} release rows must be an ordered sequence") from exc
    if len(values) != len(identifiers):
        raise PanelFeatureTaskRunnerError(f"{label} release count differs")
    panels: list[AuthenticatedReleasedPanel] = []
    receipts: list[ObjectBongardWriteOnceReceipt] = []
    for index, (row, expected_id) in enumerate(zip(values, identifiers, strict=True)):
        if type(row) is not tuple or len(row) != 2:
            raise PanelFeatureTaskRunnerError(f"{label} release row {index} differs")
        panel = _canonical_released_panel(row[0])
        receipt = _canonical_store_receipt(row[1])
        expected_payload = canonical_json(panel.to_data()) + b"\n"
        if (
            panel.panel_id != expected_id
            or panel.execution_precommit_digest
            != _address(
                expected_execution_precommit_digest,
                "expected execution precommit digest",
            )
            or panel.exposure_successor_digest
            != _address(
                expected_exposure_successor_digest,
                "expected exposure successor digest",
            )
            or panel.exact_png_digest
            != "sha256:" + hashlib.sha256(panel.exact_png_bytes).hexdigest()
            or panel.release_receipt.sha256 != panel.exact_png_digest
            or receipt.object_kind != object_kind
            or receipt.object_digest != panel.record_digest
            or receipt.payload_digest
            != "sha256:" + hashlib.sha256(expected_payload).hexdigest()
            or receipt.size_bytes != len(expected_payload)
        ):
            raise PanelFeatureTaskRunnerError(
                f"{label} release identity, pixels, authority, or durable receipt differs"
            )
        panels.append(panel)
        receipts.append(receipt)
    archive_identities = {_released_panel_authority_identity(item) for item in panels}
    if len(archive_identities) != 1:
        raise PanelFeatureTaskRunnerError(
            f"{label} releases do not share one authenticated official archive"
        )
    return tuple(panels), tuple(receipts), tuple(
        item.exact_png_bytes for item in panels
    )


def _presentation_digest(pngs: Sequence[bytes]) -> str:
    return canonical_digest(
        {
            "schema": "gkm.bongard-panel-feature-proposer-presentation.v1",
            "images": [
                {"name": name, "sha256": hashlib.sha256(panel).hexdigest()}
                for name, panel in zip(
                    PANEL_FEATURE_PRESENTATION_NAMES, pngs, strict=True
                )
            ],
        }
    )


def engineering_disposition_from_observation(
    value: EngineeringFeatureDisposition,
) -> EngineeringDisposition:
    """Map the observer enum into the predicate enum without truthiness tricks."""

    if type(value) is not EngineeringFeatureDisposition:
        raise TypeError(
            "observation disposition must be exact EngineeringFeatureDisposition"
        )
    mapping = {
        EngineeringFeatureDisposition.MATCH: EngineeringDisposition.MATCH,
        EngineeringFeatureDisposition.NONMATCH: EngineeringDisposition.NONMATCH,
        EngineeringFeatureDisposition.INDETERMINATE: (
            EngineeringDisposition.INDETERMINATE
        ),
        EngineeringFeatureDisposition.ERROR: EngineeringDisposition.ERROR,
    }
    if set(mapping) != set(EngineeringFeatureDisposition):  # pragma: no cover
        raise RuntimeError("observation disposition mapping is incomplete")
    return mapping[value]


def _nomination_from_data(value: object) -> PanelFeatureNomination:
    raw = _fields(
        value,
        {
            "schema",
            "source_block",
            "raw_slot",
            "proposal",
            "estimates_in_presentation_order",
            "native_support_count",
            "native_unclear_count",
            "contrast_support_count",
            "contrast_does_not_support_count",
            "contrast_unclear_count",
            "support_margin",
            "admission_rule",
            "narration_executable",
        },
        "panel-feature nomination",
    )
    estimates = raw["estimates_in_presentation_order"]
    if (
        raw["schema"] != PANEL_FEATURE_PROPOSER_NOMINATION_SCHEMA
        or raw["admission_rule"]
        != (
            "native-support-at-least-five-native-unclear-at-most-one-"
            "contrast-does-not-support-at-least-five-contrast-support-at-most-one-"
            "contrast-unclear-at-most-one-margin-at-least-three"
        )
        or raw["narration_executable"] is not False
        or type(estimates) is not list
    ):
        raise PanelFeatureTaskRunnerError("panel-feature nomination policy differs")
    try:
        result = PanelFeatureNomination(
            raw["source_block"],
            raw["raw_slot"],
            NativeFeatureProposal.from_data(raw["proposal"]),
            PanelFeatureEstimateVector(tuple(estimates)),
            raw["native_support_count"],
            raw["native_unclear_count"],
            raw["contrast_support_count"],
            raw["contrast_does_not_support_count"],
            raw["contrast_unclear_count"],
            raw["support_margin"],
        )
    except (TypeError, ValueError) as exc:
        raise PanelFeatureTaskRunnerError("panel-feature nomination differs") from exc
    if result.to_data() != dict(raw):
        raise PanelFeatureTaskRunnerError("panel-feature nomination is not canonical")
    return result


def _nomination_gap_from_data(value: object) -> PanelFeatureNominationGap:
    raw = _fields(
        value,
        {
            "schema",
            "native_orientation",
            "raw_slot",
            "code",
            "candidate_payload_digest",
        },
        "panel-feature nomination gap",
    )
    if raw["schema"] != PANEL_FEATURE_PROPOSER_NOMINATION_GAP_SCHEMA:
        raise PanelFeatureTaskRunnerError("nomination-gap schema differs")
    try:
        result = PanelFeatureNominationGap(
            NativeOrientation(raw["native_orientation"]),
            raw["raw_slot"],
            PanelFeatureNominationGapCode(raw["code"]),
            raw["candidate_payload_digest"],
        )
    except (TypeError, ValueError) as exc:
        raise PanelFeatureTaskRunnerError("nomination-gap value differs") from exc
    if result.to_data() != dict(raw):
        raise PanelFeatureTaskRunnerError("nomination gap is not canonical")
    return result


def _observer_vocabulary_from_data(
    value: object,
) -> PanelFeatureObserverVocabulary | None:
    if value is None:
        return None
    raw = _fields(
        value,
        {
            "schema",
            "catalog_digest",
            "specs",
            "spec_order",
            "provenance_included",
            "narration_included",
        },
        "panel-feature observer vocabulary",
    )
    if (
        raw["schema"] != PANEL_FEATURE_OBSERVER_VOCABULARY_SCHEMA
        or raw["catalog_digest"] != feature_catalog_digest()
        or raw["spec_order"] != "spec-digest-ascending"
        or raw["provenance_included"] is not False
        or raw["narration_included"] is not False
        or type(raw["specs"]) is not list
    ):
        raise PanelFeatureTaskRunnerError("observer vocabulary policy differs")
    result = PanelFeatureObserverVocabulary(
        tuple(PanelFeatureSpec.from_data(item) for item in raw["specs"])
    )
    if result.to_data() != dict(raw):
        raise PanelFeatureTaskRunnerError("observer vocabulary is not canonical")
    return result


def _proposer_result_from_data(value: object) -> PanelFeatureProposerResult:
    raw = _fields(
        value,
        {
            "schema",
            "protocol_id",
            "contract_digest",
            "payload_digest",
            "receipt_digest",
            "nominations",
            "language_gaps",
            "nomination_gaps",
            "observer_vocabulary",
            "typed_feature_specs_only",
            "narration_executable",
            "global_spec_deduplication",
        },
        "panel-feature proposer result",
    )
    if (
        raw["schema"] != PANEL_FEATURE_PROPOSER_RESULT_SCHEMA
        or raw["protocol_id"] != PANEL_FEATURE_PROPOSER_PROTOCOL_ID
        or raw["contract_digest"] != panel_feature_proposer_contract_digest()
        or raw["typed_feature_specs_only"] is not True
        or raw["narration_executable"] is not False
        or raw["global_spec_deduplication"] is not True
        or any(
            type(raw[name]) is not list
            for name in ("nominations", "language_gaps", "nomination_gaps")
        )
    ):
        raise PanelFeatureTaskRunnerError("panel-feature proposer policy differs")
    try:
        result = PanelFeatureProposerResult(
            raw["payload_digest"],
            raw["receipt_digest"],
            tuple(_nomination_from_data(item) for item in raw["nominations"]),
            tuple(LanguageGapArtifact.from_data(item) for item in raw["language_gaps"]),
            tuple(
                _nomination_gap_from_data(item) for item in raw["nomination_gaps"]
            ),
            _observer_vocabulary_from_data(raw["observer_vocabulary"]),
        )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, PanelFeatureTaskRunnerError):
            raise
        raise PanelFeatureTaskRunnerError("panel-feature proposer result differs") from exc
    order = {orientation: index for index, orientation in enumerate(_ORIENTATIONS)}
    expected_nominations = tuple(
        sorted(
            result.nominations,
            key=lambda item: (order[item.native_orientation], item.raw_slot),
        )
    )
    if (
        result.nominations != expected_nominations
        or result.language_gaps
        != tuple(sorted(result.language_gaps, key=lambda item: item.gap_digest))
        or result.nomination_gaps != tuple(sorted(result.nomination_gaps))
        or result.to_data() != dict(raw)
    ):
        raise PanelFeatureTaskRunnerError("proposer result is not canonical")
    return result


def _canonical_proposer(value: object) -> PanelFeatureProposerResult:
    if type(value) is not PanelFeatureProposerResult:
        raise TypeError("proposer_result must be exact PanelFeatureProposerResult")
    restored = _proposer_result_from_data(value.to_data())
    if restored != value:
        raise PanelFeatureTaskRunnerError("proposer result canonical reload differs")
    return restored


def _derive_vocabulary_and_verify_provenance(
    task: ObjectBongardTaskPlan,
    pngs: tuple[bytes, ...],
    proposer_result: PanelFeatureProposerResult,
) -> tuple[PanelFeatureProposerResult, FeatureVocabulary]:
    proposer = _canonical_proposer(proposer_result)
    if proposer.observer_vocabulary is None or not proposer.nominations:
        raise PanelFeatureTaskRunnerError(
            "task runner requires a nonempty two-orientation proposer vocabulary"
        )
    expected_presentation = _presentation_digest(pngs)
    task_context = _address(task.record_digest, "task record digest").split(":", 1)[1]
    contract = panel_feature_proposer_contract_digest()
    expected_block_orientation = {
        "block_a": NativeOrientation.SIDE0_POSITIVE,
        "block_b": NativeOrientation.SIDE1_POSITIVE,
    }
    side_specs: dict[NativeOrientation, list[PanelFeatureSpec]] = {
        orientation: [] for orientation in _ORIENTATIONS
    }
    for nomination in proposer.nominations:
        provenance = nomination.proposal.provenance
        expected_orientation = expected_block_orientation.get(nomination.source_block)
        if (
            expected_orientation is None
            or nomination.native_orientation is not expected_orientation
            or provenance.native_orientation is not expected_orientation
            or provenance.proposer_contract_digest != contract
            or provenance.proposer_receipt_digest != proposer.receipt_digest
            or provenance.support_set_digest != expected_presentation
            or provenance.task_context_digest != task_context
        ):
            raise PanelFeatureTaskRunnerError(
                "proposer provenance does not bind task, support presentation, and orientation"
            )
        side_specs[expected_orientation].append(nomination.spec)
    if any(not side_specs[item] for item in _ORIENTATIONS):
        raise PanelFeatureTaskRunnerError(
            "proposer nominations do not cover both native orientations"
        )
    vocabulary = FeatureVocabulary.create(
        side0_specs=side_specs[NativeOrientation.SIDE0_POSITIVE],
        side1_specs=side_specs[NativeOrientation.SIDE1_POSITIVE],
    )
    if vocabulary.specs != proposer.observer_vocabulary.specs:
        raise PanelFeatureTaskRunnerError(
            "predicate vocabulary differs from proposer observer vocabulary"
        )
    return proposer, FeatureVocabulary.from_data(vocabulary.to_data())


def _canonical_observation(value: object) -> PanelFeatureObservationSet:
    if type(value) is not PanelFeatureObservationSet:
        raise TypeError("panel observation must be exact PanelFeatureObservationSet")
    try:
        restored = PanelFeatureObservationSet.from_data(value.to_data())
    except (TypeError, ValueError) as exc:
        raise PanelFeatureTaskRunnerError(
            "panel observation canonical reload failed"
        ) from exc
    if restored != value:
        raise PanelFeatureTaskRunnerError("panel observation canonical reload differs")
    return restored


def _deployment_axis_digests() -> tuple[str, ...]:
    return tuple(item.axis_digest for item in complete_whole_panel_feature_axes())


def _verify_observation_batch(
    observations: Sequence[PanelFeatureObservationSet],
    pngs: Sequence[bytes],
    *,
    expected_contract_digest: str | None = None,
    expected_protocol_digest: str | None = None,
    label: str,
) -> tuple[tuple[PanelFeatureObservationSet, ...], str, str]:
    if isinstance(observations, (bytes, str, Mapping)):
        raise TypeError(f"{label} observations must be an ordered sequence")
    values = tuple(_canonical_observation(item) for item in observations)
    if len(values) != len(pngs):
        raise PanelFeatureTaskRunnerError(f"{label} observation count differs")
    required_axes = _deployment_axis_digests()
    for index, (observation, panel) in enumerate(zip(values, pngs, strict=True)):
        if observation.panel_digest != hashlib.sha256(panel).hexdigest():
            raise PanelFeatureTaskRunnerError(
                f"{label} observation {index} is bound to different PNG bytes"
            )
        actual_axes = tuple(
            item.axis.axis_digest for item in observation.axis_observations
        )
        if actual_axes != required_axes:
            raise PanelFeatureTaskRunnerError(
                f"{label} observation {index} is not the exact fixed complete "
                "whole-panel deployment catalog"
            )
    contract = values[0].observer_contract_digest
    protocol = values[0].measurement_protocol_digest
    if any(
        item.observer_contract_digest != contract
        or item.measurement_protocol_digest != protocol
        for item in values
    ):
        raise PanelFeatureTaskRunnerError(
            f"{label} observations do not share one observer contract and protocol"
        )
    if expected_contract_digest is not None and contract != expected_contract_digest:
        raise PanelFeatureTaskRunnerError(f"{label} observer contract differs")
    if expected_protocol_digest is not None and protocol != expected_protocol_digest:
        raise PanelFeatureTaskRunnerError(f"{label} measurement protocol differs")
    return values, contract, protocol


def _table_for_observations(
    vocabulary: FeatureVocabulary,
    observations: Sequence[PanelFeatureObservationSet],
) -> EngineeringSupportTable:
    panel_digests = tuple(item.panel_digest for item in observations)
    if len(panel_digests) != len(set(panel_digests)):
        raise PanelFeatureTaskRunnerError(
            "support panel content digests must be unique for an exact panel/spec table"
        )
    values = {
        (observation.panel_digest, spec.spec_digest): (
            engineering_disposition_from_observation(observation.evaluate(spec))
        )
        for observation in observations
        for spec in vocabulary.specs
    }
    return EngineeringSupportTable.create(vocabulary, panel_digests, values)


def _derive_support(
    task_plan: ObjectBongardTaskPlan,
    support_pngs: Sequence[bytes],
    proposer_result: PanelFeatureProposerResult,
    support_observations: Sequence[PanelFeatureObservationSet],
) -> tuple[
    ObjectBongardTaskPlan,
    tuple[bytes, ...],
    PanelFeatureProposerResult,
    tuple[PanelFeatureObservationSet, ...],
    FeatureVocabulary,
    EngineeringSupportTable,
    EngineeringFeatureVersionSpace,
    EngineeringFeatureVersionSpace,
    str,
    str,
]:
    task = _task(task_plan)
    pngs = _support_pngs(support_pngs)
    proposer, vocabulary = _derive_vocabulary_and_verify_provenance(
        task, pngs, proposer_result
    )
    observations, contract, protocol = _verify_observation_batch(
        support_observations,
        pngs,
        label="support",
    )
    table = _table_for_observations(vocabulary, observations)
    side0_panels = tuple(item.panel_digest for item in observations[:6])
    side1_panels = tuple(item.panel_digest for item in observations[6:])
    side0_space = EngineeringFeatureVersionSpace.create(
        table,
        NativeOrientation.SIDE0_POSITIVE,
        side0_panels,
        side1_panels,
    )
    side1_space = EngineeringFeatureVersionSpace.create(
        table,
        NativeOrientation.SIDE1_POSITIVE,
        side0_panels,
        side1_panels,
    )
    return (
        task,
        pngs,
        proposer,
        observations,
        vocabulary,
        table,
        side0_space,
        side1_space,
        contract,
        protocol,
    )


def _support_gap_content(value: "PanelFeatureTaskSupportGap") -> dict[str, object]:
    return {
        "schema": PANEL_FEATURE_TASK_SUPPORT_GAP_SCHEMA,
        "support_table_digest": value.support_table_digest,
        "side0_version_space_digest": value.side0_version_space_digest,
        "side1_version_space_digest": value.side1_version_space_digest,
        "missing_orientations": [item.value for item in value.missing_orientations],
        "survivor_counts_by_orientation": {
            orientation.value: count
            for orientation, count in zip(
                _ORIENTATIONS, value.survivor_counts_by_orientation, strict=True
            )
        },
        "error_cell_count": value.error_cell_count,
        "indeterminate_cell_count": value.indeterminate_cell_count,
        "gap_kind": "required-native-positive-version-space-is-empty",
        "failed_or_uncertain_observation_counts_as_nonmatch": False,
        "freeze_callback_permitted": False,
        "query_release_permitted": False,
        "query_observation_permitted": False,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class PanelFeatureTaskSupportGap:
    support_table_digest: str
    side0_version_space_digest: str
    side1_version_space_digest: str
    missing_orientations: tuple[NativeOrientation, ...]
    survivor_counts_by_orientation: tuple[int, int]
    error_cell_count: int
    indeterminate_cell_count: int
    gap_digest: str

    def __post_init__(self) -> None:
        for label, value in (
            ("support table", self.support_table_digest),
            ("side0 version space", self.side0_version_space_digest),
            ("side1 version space", self.side1_version_space_digest),
            ("support gap", self.gap_digest),
        ):
            _raw_digest(value, f"{label} digest")
        if (
            type(self.missing_orientations) is not tuple
            or not self.missing_orientations
            or self.missing_orientations
            != tuple(item for item in _ORIENTATIONS if item in self.missing_orientations)
            or type(self.survivor_counts_by_orientation) is not tuple
            or len(self.survivor_counts_by_orientation) != 2
            or any(
                type(item) is not int or item < 0
                for item in self.survivor_counts_by_orientation
            )
            or any(
                type(item) is not int or item < 0
                for item in (self.error_cell_count, self.indeterminate_cell_count)
            )
            or self.gap_digest != canonical_digest(_support_gap_content(self))
        ):
            raise PanelFeatureTaskRunnerError("support gap identity differs")

    @classmethod
    def create(
        cls,
        side0_space: EngineeringFeatureVersionSpace,
        side1_space: EngineeringFeatureVersionSpace,
    ) -> "PanelFeatureTaskSupportGap":
        if (
            type(side0_space) is not EngineeringFeatureVersionSpace
            or type(side1_space) is not EngineeringFeatureVersionSpace
        ):
            raise TypeError("support gap requires exact engineering version spaces")
        if side0_space.support_table != side1_space.support_table:
            raise PanelFeatureTaskRunnerError("support-gap version spaces use different tables")
        counts = (
            len(side0_space.survivor_formula_digests),
            len(side1_space.survivor_formula_digests),
        )
        missing = tuple(
            orientation
            for orientation, count in zip(_ORIENTATIONS, counts, strict=True)
            if count == 0
        )
        if not missing:
            raise PanelFeatureTaskRunnerError("support gap requires an empty orientation")
        table = side0_space.support_table
        values = {
            "support_table_digest": table.table_digest,
            "side0_version_space_digest": side0_space.version_space_digest,
            "side1_version_space_digest": side1_space.version_space_digest,
            "missing_orientations": missing,
            "survivor_counts_by_orientation": counts,
            "error_cell_count": sum(
                item.disposition is EngineeringDisposition.ERROR for item in table.cells
            ),
            "indeterminate_cell_count": sum(
                item.disposition is EngineeringDisposition.INDETERMINATE
                for item in table.cells
            ),
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            gap_digest=canonical_digest(_support_gap_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {**_support_gap_content(self), "gap_digest": self.gap_digest}

    @classmethod
    def from_data(cls, value: object) -> "PanelFeatureTaskSupportGap":
        raw = _fields(
            value,
            {
                "schema",
                "support_table_digest",
                "side0_version_space_digest",
                "side1_version_space_digest",
                "missing_orientations",
                "survivor_counts_by_orientation",
                "error_cell_count",
                "indeterminate_cell_count",
                "gap_kind",
                "failed_or_uncertain_observation_counts_as_nonmatch",
                "freeze_callback_permitted",
                "query_release_permitted",
                "query_observation_permitted",
                *_authority_data(),
                "gap_digest",
            },
            "panel-feature support gap",
        )
        counts = raw["survivor_counts_by_orientation"]
        if (
            raw["schema"] != PANEL_FEATURE_TASK_SUPPORT_GAP_SCHEMA
            or raw["gap_kind"]
            != "required-native-positive-version-space-is-empty"
            or raw["failed_or_uncertain_observation_counts_as_nonmatch"] is not False
            or raw["freeze_callback_permitted"] is not False
            or raw["query_release_permitted"] is not False
            or raw["query_observation_permitted"] is not False
            or any(raw[key] != item for key, item in _authority_data().items())
            or type(raw["missing_orientations"]) is not list
            or not isinstance(counts, Mapping)
            or list(counts) != [item.value for item in _ORIENTATIONS]
        ):
            raise PanelFeatureTaskRunnerError("support gap policy differs")
        try:
            result = cls(
                raw["support_table_digest"],
                raw["side0_version_space_digest"],
                raw["side1_version_space_digest"],
                tuple(NativeOrientation(item) for item in raw["missing_orientations"]),
                tuple(counts[item.value] for item in _ORIENTATIONS),
                raw["error_cell_count"],
                raw["indeterminate_cell_count"],
                raw["gap_digest"],
            )
        except (TypeError, ValueError) as exc:
            raise PanelFeatureTaskRunnerError("support gap value differs") from exc
        if result.to_data() != dict(raw):
            raise PanelFeatureTaskRunnerError("support gap is not canonical")
        return result


def _selection_gap_content(
    value: "PanelFeatureTaskSelectionGap",
) -> dict[str, object]:
    return {
        "schema": PANEL_FEATURE_TASK_SELECTION_GAP_SCHEMA,
        "support_table_digest": value.support_table_digest,
        "side0_version_space_digest": value.side0_version_space_digest,
        "side1_version_space_digest": value.side1_version_space_digest,
        "survivor_counts_by_orientation": {
            orientation.value: count
            for orientation, count in zip(
                _ORIENTATIONS, value.survivor_counts_by_orientation, strict=True
            )
        },
        "gap_kind": "multiple-support-consistent-formulas-require-external-selection",
        "implicit_digest_order_selection_allowed": False,
        "freeze_callback_permitted": False,
        "query_release_permitted": False,
        "query_observation_permitted": False,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class PanelFeatureTaskSelectionGap:
    """A nonempty version space that no authenticated selector has resolved."""

    support_table_digest: str
    side0_version_space_digest: str
    side1_version_space_digest: str
    survivor_counts_by_orientation: tuple[int, int]
    gap_digest: str

    def __post_init__(self) -> None:
        for label, value in (
            ("support table", self.support_table_digest),
            ("side0 version space", self.side0_version_space_digest),
            ("side1 version space", self.side1_version_space_digest),
            ("selection gap", self.gap_digest),
        ):
            _raw_digest(value, f"{label} digest")
        if (
            type(self.survivor_counts_by_orientation) is not tuple
            or len(self.survivor_counts_by_orientation) != 2
            or any(type(item) is not int or item < 1 for item in self.survivor_counts_by_orientation)
            or self.survivor_counts_by_orientation == (1, 1)
            or self.gap_digest != canonical_digest(_selection_gap_content(self))
        ):
            raise PanelFeatureTaskRunnerError("selection gap identity differs")

    @classmethod
    def create(
        cls,
        side0_space: EngineeringFeatureVersionSpace,
        side1_space: EngineeringFeatureVersionSpace,
    ) -> "PanelFeatureTaskSelectionGap":
        if (
            type(side0_space) is not EngineeringFeatureVersionSpace
            or type(side1_space) is not EngineeringFeatureVersionSpace
            or side0_space.support_table != side1_space.support_table
        ):
            raise TypeError("selection gap requires two version spaces over one table")
        counts = (
            len(side0_space.survivor_formula_digests),
            len(side1_space.survivor_formula_digests),
        )
        if 0 in counts or counts == (1, 1):
            raise PanelFeatureTaskRunnerError(
                "selection gap requires nonempty unresolved version spaces"
            )
        values = {
            "support_table_digest": side0_space.support_table.table_digest,
            "side0_version_space_digest": side0_space.version_space_digest,
            "side1_version_space_digest": side1_space.version_space_digest,
            "survivor_counts_by_orientation": counts,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            gap_digest=canonical_digest(_selection_gap_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {**_selection_gap_content(self), "gap_digest": self.gap_digest}

    @classmethod
    def from_data(cls, value: object) -> "PanelFeatureTaskSelectionGap":
        raw = _fields(
            value,
            {
                "schema",
                "support_table_digest",
                "side0_version_space_digest",
                "side1_version_space_digest",
                "survivor_counts_by_orientation",
                "gap_kind",
                "implicit_digest_order_selection_allowed",
                "freeze_callback_permitted",
                "query_release_permitted",
                "query_observation_permitted",
                *_authority_data(),
                "gap_digest",
            },
            "panel-feature selection gap",
        )
        counts = raw["survivor_counts_by_orientation"]
        if (
            raw["schema"] != PANEL_FEATURE_TASK_SELECTION_GAP_SCHEMA
            or raw["gap_kind"]
            != "multiple-support-consistent-formulas-require-external-selection"
            or raw["implicit_digest_order_selection_allowed"] is not False
            or raw["freeze_callback_permitted"] is not False
            or raw["query_release_permitted"] is not False
            or raw["query_observation_permitted"] is not False
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(counts, Mapping)
            or list(counts) != [item.value for item in _ORIENTATIONS]
        ):
            raise PanelFeatureTaskRunnerError("selection gap policy differs")
        result = cls(
            raw["support_table_digest"],
            raw["side0_version_space_digest"],
            raw["side1_version_space_digest"],
            tuple(counts[item.value] for item in _ORIENTATIONS),
            raw["gap_digest"],
        )
        if result.to_data() != dict(raw):
            raise PanelFeatureTaskRunnerError("selection gap is not canonical")
        return result


def _encode_png_rows(
    identifiers: Sequence[str], pngs: Sequence[bytes]
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (identifier, base64.b64encode(panel).decode("ascii"))
        for identifier, panel in zip(identifiers, pngs, strict=True)
    )


def _decode_png_rows(
    rows: Sequence[tuple[str, str]],
    identifiers: Sequence[str],
    *,
    label: str,
) -> tuple[bytes, ...]:
    if tuple(item[0] for item in rows) != tuple(identifiers):
        raise PanelFeatureTaskRunnerError(f"{label} PNG identities or order differ")
    result: list[bytes] = []
    for index, row in enumerate(rows):
        if type(row) is not tuple or len(row) != 2 or type(row[1]) is not str:
            raise PanelFeatureTaskRunnerError(f"{label} PNG archive row differs")
        try:
            decoded = base64.b64decode(row[1], validate=True)
        except (ValueError, TypeError) as exc:
            raise PanelFeatureTaskRunnerError(
                f"{label} PNG base64 differs"
            ) from exc
        result.append(_png(decoded, f"{label} panel {index}"))
    return tuple(result)


def _support_derivation_phase(
    side0_space: EngineeringFeatureVersionSpace,
    side1_space: EngineeringFeatureVersionSpace,
) -> tuple[
    PanelFeatureSupportDerivationStatus,
    PanelFeatureTaskSupportGap | None,
    PanelFeatureTaskSelectionGap | None,
]:
    counts = (
        len(side0_space.survivor_formula_digests),
        len(side1_space.survivor_formula_digests),
    )
    if 0 in counts:
        return (
            PanelFeatureSupportDerivationStatus.SUPPORT_GAP,
            PanelFeatureTaskSupportGap.create(side0_space, side1_space),
            None,
        )
    if counts != (1, 1):
        return (
            PanelFeatureSupportDerivationStatus.SELECTION_GAP,
            None,
            PanelFeatureTaskSelectionGap.create(side0_space, side1_space),
        )
    return PanelFeatureSupportDerivationStatus.UNIQUE_PAIR, None, None


def _support_derivation_content(
    value: "PanelFeatureSupportDerivation",
) -> dict[str, object]:
    return {
        "schema": PANEL_FEATURE_SUPPORT_DERIVATION_SCHEMA,
        "runner_id": PANEL_FEATURE_SUPPORT_DERIVATION_RUNNER_ID,
        "task_plan": value.task_plan.to_data(),
        "support_png_base64_by_panel_id": {
            panel_id: encoded
            for panel_id, encoded in value.support_png_base64_by_panel_id
        },
        "proposer_result": value.proposer_result.to_data(),
        "proposer_result_digest": value.proposer_result.result_digest,
        "support_observations": [
            item.to_data() for item in value.support_observations
        ],
        "deployment_observer_axes": _deployment_axis_catalog_data(),
        "deployment_observer_axis_catalog_digest": (
            _deployment_axis_catalog_digest()
        ),
        "vocabulary": value.vocabulary.to_data(),
        "support_table": value.support_table.to_data(),
        "side0_version_space": value.side0_version_space.to_data(),
        "side1_version_space": value.side1_version_space.to_data(),
        "observer_contract_digest": value.observer_contract_digest,
        "measurement_protocol_digest": value.measurement_protocol_digest,
        "status": value.status.value,
        "support_gap": (
            None if value.support_gap is None else value.support_gap.to_data()
        ),
        "selection_gap": (
            None if value.selection_gap is None else value.selection_gap.to_data()
        ),
        "support_only": True,
        "query_pixels_included": False,
        "query_release_capability": False,
        "freeze_created": False,
        "predicate_pair_created": False,
        "callbacks_accepted": False,
        "cold_replay_model_calls": 0,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class PanelFeatureSupportDerivation:
    """Mechanical support-only result with no query or freeze capability."""

    task_plan: ObjectBongardTaskPlan
    support_png_base64_by_panel_id: tuple[tuple[str, str], ...]
    proposer_result: PanelFeatureProposerResult
    support_observations: tuple[PanelFeatureObservationSet, ...]
    vocabulary: FeatureVocabulary
    support_table: EngineeringSupportTable
    side0_version_space: EngineeringFeatureVersionSpace
    side1_version_space: EngineeringFeatureVersionSpace
    observer_contract_digest: str
    measurement_protocol_digest: str
    status: PanelFeatureSupportDerivationStatus
    support_gap: PanelFeatureTaskSupportGap | None
    selection_gap: PanelFeatureTaskSelectionGap | None
    record_digest: str

    def __post_init__(self) -> None:
        if type(self.task_plan) is not ObjectBongardTaskPlan:
            raise TypeError("support derivation needs ObjectBongardTaskPlan")
        if (
            type(self.support_png_base64_by_panel_id) is not tuple
            or type(self.support_observations) is not tuple
        ):
            raise TypeError("support derivation needs exact tuple inputs")
        if type(self.proposer_result) is not PanelFeatureProposerResult:
            raise TypeError("support derivation needs PanelFeatureProposerResult")
        if type(self.status) is not PanelFeatureSupportDerivationStatus:
            raise TypeError("support derivation status differs")
        _raw_digest(self.record_digest, "support derivation record digest")
        pngs = _decode_png_rows(
            self.support_png_base64_by_panel_id,
            _support_ids(self.task_plan),
            label="support derivation",
        )
        (
            task,
            canonical_pngs,
            proposer,
            observations,
            vocabulary,
            table,
            side0_space,
            side1_space,
            contract,
            protocol,
        ) = _derive_support(
            self.task_plan,
            pngs,
            self.proposer_result,
            self.support_observations,
        )
        status, support_gap, selection_gap = _support_derivation_phase(
            side0_space, side1_space
        )
        if (
            task != self.task_plan
            or canonical_pngs != pngs
            or proposer != self.proposer_result
            or observations != self.support_observations
            or vocabulary != self.vocabulary
            or table != self.support_table
            or side0_space != self.side0_version_space
            or side1_space != self.side1_version_space
            or contract != self.observer_contract_digest
            or protocol != self.measurement_protocol_digest
            or status is not self.status
            or support_gap != self.support_gap
            or selection_gap != self.selection_gap
            or self.record_digest != canonical_digest(_support_derivation_content(self))
        ):
            raise PanelFeatureTaskRunnerError(
                "support derivation content differs from canonical replay"
            )

    @property
    def artifact_address(self) -> str:
        return "sha256:" + self.record_digest

    @classmethod
    def create(
        cls,
        task_plan: ObjectBongardTaskPlan,
        support_pngs: Sequence[bytes],
        proposer_result: PanelFeatureProposerResult,
        support_observations: Sequence[PanelFeatureObservationSet],
    ) -> "PanelFeatureSupportDerivation":
        (
            task,
            pngs,
            proposer,
            observations,
            vocabulary,
            table,
            side0_space,
            side1_space,
            contract,
            protocol,
        ) = _derive_support(
            task_plan, support_pngs, proposer_result, support_observations
        )
        status, support_gap, selection_gap = _support_derivation_phase(
            side0_space, side1_space
        )
        values = {
            "task_plan": task,
            "support_png_base64_by_panel_id": _encode_png_rows(
                _support_ids(task), pngs
            ),
            "proposer_result": proposer,
            "support_observations": observations,
            "vocabulary": vocabulary,
            "support_table": table,
            "side0_version_space": side0_space,
            "side1_version_space": side1_space,
            "observer_contract_digest": contract,
            "measurement_protocol_digest": protocol,
            "status": status,
            "support_gap": support_gap,
            "selection_gap": selection_gap,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            record_digest=canonical_digest(_support_derivation_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {
            **_support_derivation_content(self),
            "record_digest": self.record_digest,
            "artifact_address": self.artifact_address,
        }

    @classmethod
    def from_data(cls, value: object) -> "PanelFeatureSupportDerivation":
        raw = _fields(
            value,
            {
                "schema",
                "runner_id",
                "task_plan",
                "support_png_base64_by_panel_id",
                "proposer_result",
                "proposer_result_digest",
                "support_observations",
                "deployment_observer_axes",
                "deployment_observer_axis_catalog_digest",
                "vocabulary",
                "support_table",
                "side0_version_space",
                "side1_version_space",
                "observer_contract_digest",
                "measurement_protocol_digest",
                "status",
                "support_gap",
                "selection_gap",
                "support_only",
                "query_pixels_included",
                "query_release_capability",
                "freeze_created",
                "predicate_pair_created",
                "callbacks_accepted",
                "cold_replay_model_calls",
                *_authority_data(),
                "record_digest",
                "artifact_address",
            },
            "panel-feature support derivation",
        )
        if (
            raw["schema"] != PANEL_FEATURE_SUPPORT_DERIVATION_SCHEMA
            or raw["runner_id"] != PANEL_FEATURE_SUPPORT_DERIVATION_RUNNER_ID
            or raw["deployment_observer_axes"]
            != _deployment_axis_catalog_data()
            or raw["deployment_observer_axis_catalog_digest"]
            != _deployment_axis_catalog_digest()
            or raw["support_only"] is not True
            or raw["query_pixels_included"] is not False
            or raw["query_release_capability"] is not False
            or raw["freeze_created"] is not False
            or raw["predicate_pair_created"] is not False
            or raw["callbacks_accepted"] is not False
            or raw["cold_replay_model_calls"] != 0
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["support_png_base64_by_panel_id"], Mapping)
            or type(raw["support_observations"]) is not list
        ):
            raise PanelFeatureTaskRunnerError("support derivation policy differs")
        task = ObjectBongardTaskPlan.from_data(raw["task_plan"])
        encoded = raw["support_png_base64_by_panel_id"]
        if set(encoded) != set(_support_ids(task)):
            raise PanelFeatureTaskRunnerError(
                "support derivation panel identities differ"
            )
        proposer = _proposer_result_from_data(raw["proposer_result"])
        if raw["proposer_result_digest"] != proposer.result_digest:
            raise PanelFeatureTaskRunnerError(
                "support derivation proposer digest differs"
            )
        try:
            result = cls(
                task_plan=task,
                support_png_base64_by_panel_id=tuple(
                    (panel_id, encoded[panel_id]) for panel_id in _support_ids(task)
                ),
                proposer_result=proposer,
                support_observations=tuple(
                    PanelFeatureObservationSet.from_data(item)
                    for item in raw["support_observations"]
                ),
                vocabulary=FeatureVocabulary.from_data(raw["vocabulary"]),
                support_table=EngineeringSupportTable.from_data(
                    raw["support_table"]
                ),
                side0_version_space=EngineeringFeatureVersionSpace.from_data(
                    raw["side0_version_space"]
                ),
                side1_version_space=EngineeringFeatureVersionSpace.from_data(
                    raw["side1_version_space"]
                ),
                observer_contract_digest=raw["observer_contract_digest"],
                measurement_protocol_digest=raw["measurement_protocol_digest"],
                status=PanelFeatureSupportDerivationStatus(raw["status"]),
                support_gap=(
                    None
                    if raw["support_gap"] is None
                    else PanelFeatureTaskSupportGap.from_data(raw["support_gap"])
                ),
                selection_gap=(
                    None
                    if raw["selection_gap"] is None
                    else PanelFeatureTaskSelectionGap.from_data(
                        raw["selection_gap"]
                    )
                ),
                record_digest=raw["record_digest"],
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, PanelFeatureTaskRunnerError):
                raise
            raise PanelFeatureTaskRunnerError(
                "support derivation value differs"
            ) from exc
        if (
            raw["artifact_address"] != result.artifact_address
            or result.to_data() != dict(raw)
        ):
            raise PanelFeatureTaskRunnerError(
                "support derivation is not canonical"
            )
        return result


def derive_panel_feature_support(
    task_plan: ObjectBongardTaskPlan,
    support_pngs: Sequence[bytes],
    proposer_result: PanelFeatureProposerResult,
    support_observations: Sequence[PanelFeatureObservationSet],
) -> PanelFeatureSupportDerivation:
    """Derive support state without accepting any query or persistence hook."""

    return PanelFeatureSupportDerivation.create(
        task_plan, support_pngs, proposer_result, support_observations
    )


def _selected_predicate_content(
    value: "PanelFeatureSelectedPredicate",
) -> dict[str, object]:
    return {
        "schema": PANEL_FEATURE_SELECTED_PREDICATE_SCHEMA,
        "version_space": value.version_space.to_data(),
        "selected_formula_digest": value.selected_formula_digest,
        "selection_mode": value.selection_mode,
        "rank_artifact_digest": value.rank_artifact_digest,
        "selected_formula_must_be_verified_survivor": True,
        "implicit_minimum_or_digest_selection_used": False,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class PanelFeatureSelectedPredicate:
    """One explicitly selected support survivor, never an implicit minimum."""

    version_space: EngineeringFeatureVersionSpace
    selected_formula_digest: str
    selection_mode: str
    rank_artifact_digest: str | None
    predicate_digest: str

    def __post_init__(self) -> None:
        if type(self.version_space) is not EngineeringFeatureVersionSpace:
            raise TypeError("selected predicate needs EngineeringFeatureVersionSpace")
        _raw_digest(self.selected_formula_digest, "selected formula digest")
        if self.selected_formula_digest not in self.version_space.survivor_formula_digests:
            raise PanelFeatureTaskRunnerError(
                "selected formula is not a verified support survivor"
            )
        counts = len(self.version_space.survivor_formula_digests)
        if self.selection_mode == "unique_support_survivor":
            if counts != 1 or self.rank_artifact_digest is not None:
                raise PanelFeatureTaskRunnerError(
                    "unique selection mode differs from support space"
                )
        elif self.selection_mode == "verified_support_rank_artifact":
            if self.rank_artifact_digest is None:
                raise PanelFeatureTaskRunnerError(
                    "ranked selection needs its exact artifact digest"
                )
            _raw_digest(self.rank_artifact_digest, "rank artifact digest")
        else:
            raise PanelFeatureTaskRunnerError("selected predicate mode differs")
        _raw_digest(self.predicate_digest, "selected predicate digest")
        if self.predicate_digest != canonical_digest(
            _selected_predicate_content(self)
        ):
            raise PanelFeatureTaskRunnerError("selected predicate digest differs")

    @property
    def formula(self) -> AllOf:
        return next(
            item
            for item in self.version_space.survivor_formulas
            if item.formula_digest == self.selected_formula_digest
        )

    @classmethod
    def create(
        cls,
        version_space: EngineeringFeatureVersionSpace,
        selected_formula_digest: str,
        *,
        rank_artifact_digest: str | None,
    ) -> "PanelFeatureSelectedPredicate":
        mode = (
            "unique_support_survivor"
            if rank_artifact_digest is None
            else "verified_support_rank_artifact"
        )
        values = {
            "version_space": version_space,
            "selected_formula_digest": selected_formula_digest,
            "selection_mode": mode,
            "rank_artifact_digest": rank_artifact_digest,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            predicate_digest=canonical_digest(
                _selected_predicate_content(provisional)
            ),
        )

    def to_data(self) -> dict[str, object]:
        return {
            **_selected_predicate_content(self),
            "predicate_digest": self.predicate_digest,
        }

    @classmethod
    def from_data(cls, value: object) -> "PanelFeatureSelectedPredicate":
        raw = _fields(
            value,
            {
                "schema",
                "version_space",
                "selected_formula_digest",
                "selection_mode",
                "rank_artifact_digest",
                "selected_formula_must_be_verified_survivor",
                "implicit_minimum_or_digest_selection_used",
                *_authority_data(),
                "predicate_digest",
            },
            "explicit selected predicate",
        )
        if (
            raw["schema"] != PANEL_FEATURE_SELECTED_PREDICATE_SCHEMA
            or raw["selected_formula_must_be_verified_survivor"] is not True
            or raw["implicit_minimum_or_digest_selection_used"] is not False
            or any(raw[key] != item for key, item in _authority_data().items())
        ):
            raise PanelFeatureTaskRunnerError("selected predicate policy differs")
        result = cls(
            EngineeringFeatureVersionSpace.from_data(raw["version_space"]),
            raw["selected_formula_digest"],
            raw["selection_mode"],
            raw["rank_artifact_digest"],
            raw["predicate_digest"],
        )
        if result.to_data() != dict(raw):
            raise PanelFeatureTaskRunnerError("selected predicate is not canonical")
        return result


def _selected_pair_content(
    value: "PanelFeatureSelectedPredicatePair",
) -> dict[str, object]:
    return {
        "schema": PANEL_FEATURE_SELECTED_PAIR_SCHEMA,
        "side0_predicate": value.side0_predicate.to_data(),
        "side1_predicate": value.side1_predicate.to_data(),
        "selection_mode": value.selection_mode,
        "rank_artifact_digest": value.rank_artifact_digest,
        "selected_formula_digests": list(value.selected_formula_digests),
        "one_explicit_survivor_per_native_orientation": True,
        "implicit_minimum_or_digest_selection_used": False,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class PanelFeatureSelectedPredicatePair:
    side0_predicate: PanelFeatureSelectedPredicate
    side1_predicate: PanelFeatureSelectedPredicate
    selection_mode: str
    rank_artifact_digest: str | None
    pair_digest: str

    def __post_init__(self) -> None:
        if (
            type(self.side0_predicate) is not PanelFeatureSelectedPredicate
            or type(self.side1_predicate) is not PanelFeatureSelectedPredicate
        ):
            raise TypeError("selected pair predicates differ")
        left = self.side0_predicate.version_space
        right = self.side1_predicate.version_space
        survivor_counts = (
            len(left.survivor_formula_digests),
            len(right.survivor_formula_digests),
        )
        if (
            left.native_orientation is not NativeOrientation.SIDE0_POSITIVE
            or right.native_orientation is not NativeOrientation.SIDE1_POSITIVE
            or left.support_table != right.support_table
            or left.side0_panel_digests != right.side0_panel_digests
            or left.side1_panel_digests != right.side1_panel_digests
            or self.side0_predicate.selection_mode != self.selection_mode
            or self.side1_predicate.selection_mode != self.selection_mode
            or self.side0_predicate.rank_artifact_digest
            != self.rank_artifact_digest
            or self.side1_predicate.rank_artifact_digest
            != self.rank_artifact_digest
            or (
                self.selection_mode == "verified_support_rank_artifact"
                and survivor_counts == (1, 1)
            )
        ):
            raise PanelFeatureTaskRunnerError("selected predicate pair custody differs")
        _raw_digest(self.pair_digest, "selected predicate pair digest")
        if self.pair_digest != canonical_digest(_selected_pair_content(self)):
            raise PanelFeatureTaskRunnerError("selected predicate pair digest differs")

    @property
    def selected_formula_digests(self) -> tuple[str, str]:
        return (
            self.side0_predicate.selected_formula_digest,
            self.side1_predicate.selected_formula_digest,
        )

    @property
    def vocabulary(self) -> FeatureVocabulary:
        return self.side0_predicate.version_space.support_table.vocabulary

    @classmethod
    def create(
        cls,
        side0_space: EngineeringFeatureVersionSpace,
        side1_space: EngineeringFeatureVersionSpace,
        selected_formula_digests: tuple[str, str],
        *,
        rank_artifact_digest: str | None,
    ) -> "PanelFeatureSelectedPredicatePair":
        if type(selected_formula_digests) is not tuple or len(selected_formula_digests) != 2:
            raise TypeError("selected formula digests must be an exact pair")
        side0 = PanelFeatureSelectedPredicate.create(
            side0_space,
            selected_formula_digests[0],
            rank_artifact_digest=rank_artifact_digest,
        )
        side1 = PanelFeatureSelectedPredicate.create(
            side1_space,
            selected_formula_digests[1],
            rank_artifact_digest=rank_artifact_digest,
        )
        values = {
            "side0_predicate": side0,
            "side1_predicate": side1,
            "selection_mode": side0.selection_mode,
            "rank_artifact_digest": rank_artifact_digest,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            pair_digest=canonical_digest(_selected_pair_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {**_selected_pair_content(self), "pair_digest": self.pair_digest}

    @classmethod
    def from_data(cls, value: object) -> "PanelFeatureSelectedPredicatePair":
        raw = _fields(
            value,
            {
                "schema",
                "side0_predicate",
                "side1_predicate",
                "selection_mode",
                "rank_artifact_digest",
                "selected_formula_digests",
                "one_explicit_survivor_per_native_orientation",
                "implicit_minimum_or_digest_selection_used",
                *_authority_data(),
                "pair_digest",
            },
            "explicit selected predicate pair",
        )
        if (
            raw["schema"] != PANEL_FEATURE_SELECTED_PAIR_SCHEMA
            or raw["one_explicit_survivor_per_native_orientation"] is not True
            or raw["implicit_minimum_or_digest_selection_used"] is not False
            or type(raw["selected_formula_digests"]) is not list
            or any(raw[key] != item for key, item in _authority_data().items())
        ):
            raise PanelFeatureTaskRunnerError("selected pair policy differs")
        result = cls(
            PanelFeatureSelectedPredicate.from_data(raw["side0_predicate"]),
            PanelFeatureSelectedPredicate.from_data(raw["side1_predicate"]),
            raw["selection_mode"],
            raw["rank_artifact_digest"],
            raw["pair_digest"],
        )
        if (
            raw["selected_formula_digests"] != list(result.selected_formula_digests)
            or result.to_data() != dict(raw)
        ):
            raise PanelFeatureTaskRunnerError("selected pair is not canonical")
        return result


def _query_outcome(
    side0: EngineeringDisposition, side1: EngineeringDisposition
) -> EngineeringQueryOutcome:
    if EngineeringDisposition.ERROR in (side0, side1):
        return EngineeringQueryOutcome.ERROR
    if (
        side0 is EngineeringDisposition.MATCH
        and side1 is EngineeringDisposition.NONMATCH
    ):
        return EngineeringQueryOutcome.SIDE0
    if (
        side1 is EngineeringDisposition.MATCH
        and side0 is EngineeringDisposition.NONMATCH
    ):
        return EngineeringQueryOutcome.SIDE1
    return EngineeringQueryOutcome.ABSTAIN


def _query_decision_content(value: "PanelFeatureQueryDecision") -> dict[str, object]:
    return {
        "schema": PANEL_FEATURE_QUERY_DECISION_SCHEMA,
        "predicate_pair": value.predicate_pair.to_data(),
        "query_table": value.query_table.to_data(),
        "panel_digest": value.panel_digest,
        "side0_disposition": value.side0_disposition.value,
        "side1_disposition": value.side1_disposition.value,
        "outcome": value.outcome.value,
        "decision_rule": "explicit-native-positive-and-other-native-negative",
        "nonmatch_alone_predicts_opposite": False,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class PanelFeatureQueryDecision:
    predicate_pair: PanelFeatureSelectedPredicatePair
    query_table: EngineeringSupportTable
    panel_digest: str
    side0_disposition: EngineeringDisposition
    side1_disposition: EngineeringDisposition
    outcome: EngineeringQueryOutcome
    decision_digest: str

    def __post_init__(self) -> None:
        if (
            type(self.predicate_pair) is not PanelFeatureSelectedPredicatePair
            or type(self.query_table) is not EngineeringSupportTable
        ):
            raise TypeError("query decision inputs differ")
        _raw_digest(self.panel_digest, "query panel digest")
        if (
            self.query_table.panel_digests != (self.panel_digest,)
            or self.query_table.vocabulary.vocabulary_digest
            != self.predicate_pair.vocabulary.vocabulary_digest
        ):
            raise PanelFeatureTaskRunnerError("query table custody differs")
        side0 = evaluate_engineering_all_of(
            self.predicate_pair.side0_predicate.formula,
            self.query_table,
            self.panel_digest,
        )
        side1 = evaluate_engineering_all_of(
            self.predicate_pair.side1_predicate.formula,
            self.query_table,
            self.panel_digest,
        )
        if (
            self.side0_disposition,
            self.side1_disposition,
            self.outcome,
        ) != (side0, side1, _query_outcome(side0, side1)):
            raise PanelFeatureTaskRunnerError("query decision replay differs")
        _raw_digest(self.decision_digest, "query decision digest")
        if self.decision_digest != canonical_digest(_query_decision_content(self)):
            raise PanelFeatureTaskRunnerError("query decision digest differs")

    @classmethod
    def create(
        cls,
        predicate_pair: PanelFeatureSelectedPredicatePair,
        query_table: EngineeringSupportTable,
        panel_digest: str,
    ) -> "PanelFeatureQueryDecision":
        side0 = evaluate_engineering_all_of(
            predicate_pair.side0_predicate.formula, query_table, panel_digest
        )
        side1 = evaluate_engineering_all_of(
            predicate_pair.side1_predicate.formula, query_table, panel_digest
        )
        values = {
            "predicate_pair": predicate_pair,
            "query_table": query_table,
            "panel_digest": panel_digest,
            "side0_disposition": side0,
            "side1_disposition": side1,
            "outcome": _query_outcome(side0, side1),
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            decision_digest=canonical_digest(_query_decision_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {**_query_decision_content(self), "decision_digest": self.decision_digest}

    @classmethod
    def from_data(cls, value: object) -> "PanelFeatureQueryDecision":
        raw = _fields(
            value,
            {
                "schema",
                "predicate_pair",
                "query_table",
                "panel_digest",
                "side0_disposition",
                "side1_disposition",
                "outcome",
                "decision_rule",
                "nonmatch_alone_predicts_opposite",
                *_authority_data(),
                "decision_digest",
            },
            "explicit query decision",
        )
        if (
            raw["schema"] != PANEL_FEATURE_QUERY_DECISION_SCHEMA
            or raw["decision_rule"]
            != "explicit-native-positive-and-other-native-negative"
            or raw["nonmatch_alone_predicts_opposite"] is not False
            or any(raw[key] != item for key, item in _authority_data().items())
        ):
            raise PanelFeatureTaskRunnerError("query decision policy differs")
        try:
            result = cls(
                PanelFeatureSelectedPredicatePair.from_data(raw["predicate_pair"]),
                EngineeringSupportTable.from_data(raw["query_table"]),
                raw["panel_digest"],
                EngineeringDisposition(raw["side0_disposition"]),
                EngineeringDisposition(raw["side1_disposition"]),
                EngineeringQueryOutcome(raw["outcome"]),
                raw["decision_digest"],
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, PanelFeatureTaskRunnerError):
                raise
            raise PanelFeatureTaskRunnerError("query decision value differs") from exc
        if result.to_data() != dict(raw):
            raise PanelFeatureTaskRunnerError("query decision is not canonical")
        return result


def _combined_version_space_digest(
    side0_space: EngineeringFeatureVersionSpace,
    side1_space: EngineeringFeatureVersionSpace,
) -> str:
    return canonical_digest(
        {
            "schema": "gkm.bongard-panel-feature-combined-version-space.v1",
            "side0_version_space_digest": side0_space.version_space_digest,
            "side1_version_space_digest": side1_space.version_space_digest,
            "selection_policy": "external-only-no-implicit-digest-order",
        }
    )


def _selection_response_digest(pair: PanelFeatureSelectedPredicatePair) -> str:
    if pair.rank_artifact_digest is not None:
        return pair.rank_artifact_digest
    return canonical_digest(
        {
            "schema": "gkm.bongard-panel-feature-unique-selection-response.v1",
            "selected_formula_digests": list(pair.selected_formula_digests),
            "selection_mode": pair.selection_mode,
            "model_call_made": False,
        }
    )


def _freeze_content(value: "PanelFeatureTaskFreeze") -> dict[str, object]:
    return {
        "schema": PANEL_FEATURE_TASK_FREEZE_SCHEMA,
        "runner_id": PANEL_FEATURE_TASK_RUNNER_ID,
        "task_id": value.task_id,
        "task_plan_digest": value.task_plan_digest,
        "execution_precommit_digest": value.execution_precommit_digest,
        "proposer_result_digest": value.proposer_result_digest,
        "support_table_digest": value.support_table_digest,
        "side0_version_space": value.side0_version_space.to_data(),
        "side1_version_space": value.side1_version_space.to_data(),
        "version_space_digest": value.version_space_digest,
        "support_version_space_digest": value.support_version_space_digest,
        "rank_response_digest": value.rank_response_digest,
        "rank_artifact": (
            None if value.rank_artifact is None else value.rank_artifact.to_data()
        ),
        "rank_artifact_digest": value.rank_artifact_digest,
        "selection_mode": value.predicate_pair.selection_mode,
        "selected_formula_digests": list(
            value.predicate_pair.selected_formula_digests
        ),
        "predicate_pair": value.predicate_pair.to_data(),
        "selected_predicate_digest": value.selected_predicate_digest,
        "sealed_query_panel_ids": list(value.sealed_query_panel_ids),
        "deployment_observer_axis_catalog_digest": (
            _deployment_axis_catalog_digest()
        ),
        "query_bytes_included": False,
        "query_observations_included": False,
        "implicit_survivor_selection_used": False,
        "one_explicit_selected_survivor_per_orientation": True,
        "all_version_space_survivors_retained": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class PanelFeatureTaskFreeze:
    """Full Python predicate IR implementing the official task-freeze protocol."""

    task_id: str
    task_plan_digest: str
    execution_precommit_digest: str
    proposer_result_digest: str
    support_table_digest: str
    side0_version_space: EngineeringFeatureVersionSpace
    side1_version_space: EngineeringFeatureVersionSpace
    version_space_digest: str
    support_version_space_digest: str
    rank_response_digest: str
    rank_artifact: PanelFeatureRankArtifact | None
    rank_artifact_digest: str | None
    predicate_pair: PanelFeatureSelectedPredicatePair
    selected_predicate_digest: str
    sealed_query_panel_ids: tuple[str, str]
    record_digest: str

    def __post_init__(self) -> None:
        if type(self.task_id) is not str or not self.task_id:
            raise PanelFeatureTaskRunnerError("freeze task ID differs")
        for label, value in (
            ("task plan", self.task_plan_digest),
            ("execution precommit", self.execution_precommit_digest),
            ("task freeze", self.record_digest),
        ):
            _address(value, f"{label} digest")
        for label, value in (
            ("proposer result", self.proposer_result_digest),
            ("support table", self.support_table_digest),
            ("version space", self.version_space_digest),
            ("support version space", self.support_version_space_digest),
            ("rank response", self.rank_response_digest),
            ("selected predicate", self.selected_predicate_digest),
        ):
            _raw_digest(value, f"{label} digest")
        if (
            type(self.side0_version_space) is not EngineeringFeatureVersionSpace
            or type(self.side1_version_space) is not EngineeringFeatureVersionSpace
            or type(self.predicate_pair) is not PanelFeatureSelectedPredicatePair
        ):
            raise TypeError("freeze version spaces or selected pair differ")
        if (
            self.rank_artifact is not None
            and type(self.rank_artifact) is not PanelFeatureRankArtifact
        ):
            raise TypeError(
                "freeze rank artifact must be exact PanelFeatureRankArtifact"
            )
        survivor_counts = (
            len(self.side0_version_space.survivor_formula_digests),
            len(self.side1_version_space.survivor_formula_digests),
        )
        verified_rank: PanelFeatureRankArtifact | None = None
        if self.rank_artifact is not None:
            try:
                verified_rank = _canonical_rank_artifact(
                    self.rank_artifact,
                    proposer=self.rank_artifact.rank_input.proposer_result,
                    side0_space=self.side0_version_space,
                    side1_space=self.side1_version_space,
                )
            except Exception as exc:
                raise PanelFeatureTaskRunnerError(
                    "freeze rank artifact does not bind the exact support spaces"
                ) from exc
        rank_shape = (
            self.rank_artifact is None
            and self.rank_artifact_digest is None
            and survivor_counts == (1, 1)
            and self.predicate_pair.selection_mode == "unique_support_survivor"
        ) or (
            verified_rank is not None
            and survivor_counts != (1, 1)
            and self.rank_artifact_digest == verified_rank.artifact_digest
            and self.proposer_result_digest
            == verified_rank.rank_input.proposer_result.result_digest
            and self.predicate_pair.selection_mode
            == "verified_support_rank_artifact"
            and self.predicate_pair.selected_formula_digests
            == verified_rank.selected_formula_digests
        )
        if (
            type(self.side0_version_space) is not EngineeringFeatureVersionSpace
            or type(self.side1_version_space) is not EngineeringFeatureVersionSpace
            or type(self.predicate_pair) is not PanelFeatureSelectedPredicatePair
            or self.side0_version_space.support_table
            != self.side1_version_space.support_table
            or self.support_table_digest
            != self.side0_version_space.support_table.table_digest
            or not self.side0_version_space.survivor_formula_digests
            or not self.side1_version_space.survivor_formula_digests
            or not rank_shape
            or self.predicate_pair.side0_predicate.version_space
            != self.side0_version_space
            or self.predicate_pair.side1_predicate.version_space
            != self.side1_version_space
            or self.rank_artifact_digest != self.predicate_pair.rank_artifact_digest
            or self.rank_response_digest != _selection_response_digest(
                self.predicate_pair
            )
            or self.selected_predicate_digest != self.predicate_pair.pair_digest
            or self.version_space_digest
            != _combined_version_space_digest(
                self.side0_version_space, self.side1_version_space
            )
            or self.support_version_space_digest != self.version_space_digest
            or type(self.sealed_query_panel_ids) is not tuple
            or len(self.sealed_query_panel_ids) != 2
            or len(set(self.sealed_query_panel_ids)) != 2
            or self.record_digest != "sha256:" + canonical_digest(_freeze_content(self))
        ):
            raise PanelFeatureTaskRunnerError("task freeze content differs")

    @classmethod
    def seal(
        cls,
        *,
        task: ObjectBongardTaskPlan,
        execution_precommit_digest: str,
        proposer: PanelFeatureProposerResult,
        side0_space: EngineeringFeatureVersionSpace,
        side1_space: EngineeringFeatureVersionSpace,
        rank_artifact: PanelFeatureRankArtifact | None,
    ) -> "PanelFeatureTaskFreeze":
        if (
            type(task) is not ObjectBongardTaskPlan
            or type(proposer) is not PanelFeatureProposerResult
            or type(side0_space) is not EngineeringFeatureVersionSpace
            or type(side1_space) is not EngineeringFeatureVersionSpace
        ):
            raise TypeError("freeze task, proposer, or version spaces differ")
        survivor_counts = (
            len(side0_space.survivor_formula_digests),
            len(side1_space.survivor_formula_digests),
        )
        if 0 in survivor_counts:
            raise PanelFeatureTaskRunnerError(
                "cannot freeze an empty support version space"
            )
        verified_rank: PanelFeatureRankArtifact | None
        if survivor_counts == (1, 1):
            if rank_artifact is not None:
                raise PanelFeatureTaskRunnerError(
                    "unique support pair cannot freeze an unnecessary rank artifact"
                )
            verified_rank = None
            selected_formula_digests = (
                side0_space.survivor_formula_digests[0],
                side1_space.survivor_formula_digests[0],
            )
        else:
            verified_rank = _canonical_rank_artifact(
                rank_artifact,
                proposer=proposer,
                side0_space=side0_space,
                side1_space=side1_space,
            )
            selected_formula_digests = verified_rank.selected_formula_digests
        rank_artifact_digest = (
            None if verified_rank is None else verified_rank.artifact_digest
        )
        pair = PanelFeatureSelectedPredicatePair.create(
            side0_space,
            side1_space,
            selected_formula_digests,
            rank_artifact_digest=rank_artifact_digest,
        )
        combined = _combined_version_space_digest(side0_space, side1_space)
        values = {
            "task_id": task.task_id,
            "task_plan_digest": task.record_digest,
            "execution_precommit_digest": _address(
                execution_precommit_digest, "execution precommit digest"
            ),
            "proposer_result_digest": proposer.result_digest,
            "support_table_digest": side0_space.support_table.table_digest,
            "side0_version_space": side0_space,
            "side1_version_space": side1_space,
            "version_space_digest": combined,
            "support_version_space_digest": combined,
            "rank_response_digest": _selection_response_digest(pair),
            "rank_artifact": verified_rank,
            "rank_artifact_digest": rank_artifact_digest,
            "predicate_pair": pair,
            "selected_predicate_digest": pair.pair_digest,
            "sealed_query_panel_ids": (
                task.side_0_query_panel_id,
                task.side_1_query_panel_id,
            ),
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            record_digest="sha256:" + canonical_digest(_freeze_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {**_freeze_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: object) -> "PanelFeatureTaskFreeze":
        raw = _fields(
            value,
            {
                "schema",
                "runner_id",
                "task_id",
                "task_plan_digest",
                "execution_precommit_digest",
                "proposer_result_digest",
                "support_table_digest",
                "side0_version_space",
                "side1_version_space",
                "version_space_digest",
                "support_version_space_digest",
                "rank_response_digest",
                "rank_artifact",
                "rank_artifact_digest",
                "selection_mode",
                "selected_formula_digests",
                "predicate_pair",
                "selected_predicate_digest",
                "sealed_query_panel_ids",
                "deployment_observer_axis_catalog_digest",
                "query_bytes_included",
                "query_observations_included",
                "implicit_survivor_selection_used",
                "one_explicit_selected_survivor_per_orientation",
                "all_version_space_survivors_retained",
                *_authority_data(),
                "record_digest",
            },
            "panel-feature task freeze",
        )
        if (
            raw["schema"] != PANEL_FEATURE_TASK_FREEZE_SCHEMA
            or raw["runner_id"] != PANEL_FEATURE_TASK_RUNNER_ID
            or raw["deployment_observer_axis_catalog_digest"]
            != _deployment_axis_catalog_digest()
            or raw["query_bytes_included"] is not False
            or raw["query_observations_included"] is not False
            or raw["implicit_survivor_selection_used"] is not False
            or raw["one_explicit_selected_survivor_per_orientation"] is not True
            or raw["all_version_space_survivors_retained"] is not True
            or type(raw["selected_formula_digests"]) is not list
            or any(raw[key] != item for key, item in _authority_data().items())
            or type(raw["sealed_query_panel_ids"]) is not list
        ):
            raise PanelFeatureTaskRunnerError("task freeze policy differs")
        result = cls(
            raw["task_id"],
            raw["task_plan_digest"],
            raw["execution_precommit_digest"],
            raw["proposer_result_digest"],
            raw["support_table_digest"],
            EngineeringFeatureVersionSpace.from_data(raw["side0_version_space"]),
            EngineeringFeatureVersionSpace.from_data(raw["side1_version_space"]),
            raw["version_space_digest"],
            raw["support_version_space_digest"],
            raw["rank_response_digest"],
            (
                None
                if raw["rank_artifact"] is None
                else PanelFeatureRankArtifact.from_data(raw["rank_artifact"])
            ),
            raw["rank_artifact_digest"],
            PanelFeatureSelectedPredicatePair.from_data(raw["predicate_pair"]),
            raw["selected_predicate_digest"],
            tuple(raw["sealed_query_panel_ids"]),
            raw["record_digest"],
        )
        if (
            raw["selection_mode"] != result.predicate_pair.selection_mode
            or raw["selected_formula_digests"]
            != list(result.predicate_pair.selected_formula_digests)
            or result.to_data() != dict(raw)
        ):
            raise PanelFeatureTaskRunnerError("task freeze is not canonical")
        return result


def _freeze_commit_content(
    value: "PanelFeatureTaskFreezeCommit",
) -> dict[str, object]:
    return {
        "schema": PANEL_FEATURE_TASK_FREEZE_COMMIT_SCHEMA,
        "task_id": value.task_id,
        "task_plan_digest": value.task_plan_digest,
        "execution_precommit_digest": value.execution_precommit_digest,
        "version_space_digest": value.version_space_digest,
        "support_version_space_digest": value.support_version_space_digest,
        "rank_response_digest": value.rank_response_digest,
        "selected_predicate_digest": value.selected_predicate_digest,
        "task_freeze_digest": value.task_freeze_digest,
        "exact_freeze_payload_digest": value.exact_freeze_payload_digest,
        "task_freeze_store_receipt_digest": value.task_freeze_store_receipt_digest,
        "durably_persisted_and_reloaded_before_query_release": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class PanelFeatureTaskFreezeCommit:
    """Durable decision commit implementing the official release protocol."""

    task_id: str
    task_plan_digest: str
    execution_precommit_digest: str
    version_space_digest: str
    support_version_space_digest: str
    rank_response_digest: str
    selected_predicate_digest: str
    task_freeze_digest: str
    exact_freeze_payload_digest: str
    task_freeze_store_receipt_digest: str
    record_digest: str

    def __post_init__(self) -> None:
        if type(self.task_id) is not str or not self.task_id:
            raise PanelFeatureTaskRunnerError("freeze-commit task ID differs")
        for label, value in (
            ("task plan", self.task_plan_digest),
            ("execution precommit", self.execution_precommit_digest),
            ("task freeze", self.task_freeze_digest),
            ("exact freeze payload", self.exact_freeze_payload_digest),
            ("task freeze store receipt", self.task_freeze_store_receipt_digest),
            ("freeze commit", self.record_digest),
        ):
            _address(value, f"{label} digest")
        for label, value in (
            ("version space", self.version_space_digest),
            ("support version space", self.support_version_space_digest),
            ("rank response", self.rank_response_digest),
            ("selected predicate", self.selected_predicate_digest),
        ):
            _raw_digest(value, f"{label} digest")
        if (
            self.version_space_digest != self.support_version_space_digest
            or self.record_digest
            != "sha256:" + canonical_digest(_freeze_commit_content(self))
        ):
            raise PanelFeatureTaskRunnerError("freeze commit content differs")

    @classmethod
    def seal(
        cls,
        freeze: PanelFeatureTaskFreeze,
        freeze_receipt: ObjectBongardWriteOnceReceipt,
    ) -> "PanelFeatureTaskFreezeCommit":
        if type(freeze) is not PanelFeatureTaskFreeze:
            raise TypeError("commit freeze must be exact PanelFeatureTaskFreeze")
        receipt = _canonical_store_receipt(freeze_receipt)
        payload = canonical_json(freeze.to_data()) + b"\n"
        if (
            receipt.object_kind != "task-freeze"
            or receipt.object_digest != freeze.record_digest
            or receipt.payload_digest
            != "sha256:" + hashlib.sha256(payload).hexdigest()
            or receipt.size_bytes != len(payload)
        ):
            raise PanelFeatureTaskRunnerError(
                "freeze store receipt does not bind exact canonical freeze bytes"
            )
        values = {
            "task_id": freeze.task_id,
            "task_plan_digest": freeze.task_plan_digest,
            "execution_precommit_digest": freeze.execution_precommit_digest,
            "version_space_digest": freeze.version_space_digest,
            "support_version_space_digest": freeze.support_version_space_digest,
            "rank_response_digest": freeze.rank_response_digest,
            "selected_predicate_digest": freeze.selected_predicate_digest,
            "task_freeze_digest": freeze.record_digest,
            "exact_freeze_payload_digest": receipt.payload_digest,
            "task_freeze_store_receipt_digest": receipt.record_digest,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            record_digest="sha256:"
            + canonical_digest(_freeze_commit_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {**_freeze_commit_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: object) -> "PanelFeatureTaskFreezeCommit":
        raw = _fields(
            value,
            {
                "schema",
                "task_id",
                "task_plan_digest",
                "execution_precommit_digest",
                "version_space_digest",
                "support_version_space_digest",
                "rank_response_digest",
                "selected_predicate_digest",
                "task_freeze_digest",
                "exact_freeze_payload_digest",
                "task_freeze_store_receipt_digest",
                "durably_persisted_and_reloaded_before_query_release",
                *_authority_data(),
                "record_digest",
            },
            "panel-feature task freeze commit",
        )
        if (
            raw["schema"] != PANEL_FEATURE_TASK_FREEZE_COMMIT_SCHEMA
            or raw["durably_persisted_and_reloaded_before_query_release"] is not True
            or any(raw[key] != item for key, item in _authority_data().items())
        ):
            raise PanelFeatureTaskRunnerError("freeze commit policy differs")
        result = cls(
            raw["task_id"],
            raw["task_plan_digest"],
            raw["execution_precommit_digest"],
            raw["version_space_digest"],
            raw["support_version_space_digest"],
            raw["rank_response_digest"],
            raw["selected_predicate_digest"],
            raw["task_freeze_digest"],
            raw["exact_freeze_payload_digest"],
            raw["task_freeze_store_receipt_digest"],
            raw["record_digest"],
        )
        if result.to_data() != dict(raw):
            raise PanelFeatureTaskRunnerError("freeze commit is not canonical")
        return result


def _verify_durable_freeze(
    freeze: PanelFeatureTaskFreeze,
    commit: PanelFeatureTaskFreezeCommit,
    freeze_receipt: ObjectBongardWriteOnceReceipt,
    commit_receipt: ObjectBongardWriteOnceReceipt,
) -> tuple[
    PanelFeatureTaskFreeze,
    PanelFeatureTaskFreezeCommit,
    ObjectBongardWriteOnceReceipt,
    ObjectBongardWriteOnceReceipt,
]:
    frozen = PanelFeatureTaskFreeze.from_data(freeze.to_data())
    committed = PanelFeatureTaskFreezeCommit.from_data(commit.to_data())
    freeze_store = _canonical_store_receipt(freeze_receipt)
    commit_store = _canonical_store_receipt(commit_receipt)
    expected_commit = PanelFeatureTaskFreezeCommit.seal(frozen, freeze_store)
    commit_payload = canonical_json(committed.to_data()) + b"\n"
    if (
        committed != expected_commit
        or commit_store.object_kind != "task-decision-commit"
        or commit_store.object_digest != committed.record_digest
        or commit_store.payload_digest
        != "sha256:" + hashlib.sha256(commit_payload).hexdigest()
        or commit_store.size_bytes != len(commit_payload)
    ):
        raise PanelFeatureTaskRunnerError(
            "durable freeze commit or exact persisted bytes differ"
        )
    return frozen, committed, freeze_store, commit_store


def _archive_content(value: "PanelFeatureTaskArchive") -> dict[str, object]:
    return {
        "schema": PANEL_FEATURE_TASK_ARCHIVE_SCHEMA,
        "runner_id": PANEL_FEATURE_TASK_RUNNER_ID,
        "deployment_observer_axes": _deployment_axis_catalog_data(),
        "deployment_observer_axis_catalog_digest": (
            _deployment_axis_catalog_digest()
        ),
        "deployment_observer_axis_subset_permitted": False,
        "deployment_observer_catalog_caller_or_candidate_selected": False,
        "task_plan": value.task_plan.to_data(),
        "execution_precommit_digest": value.execution_precommit_digest,
        "exposure_successor_digest": value.exposure_successor_digest,
        "support_released_panels": [
            item.to_data() for item in value.support_released_panels
        ],
        "support_release_store_receipts": [
            item.to_data() for item in value.support_release_store_receipts
        ],
        "support_png_base64_by_panel_id": {
            panel_id: encoded
            for panel_id, encoded in value.support_png_base64_by_panel_id
        },
        "proposer_result": value.proposer_result.to_data(),
        "proposer_result_digest": value.proposer_result.result_digest,
        "support_observations": [item.to_data() for item in value.support_observations],
        "vocabulary": value.vocabulary.to_data(),
        "support_table": value.support_table.to_data(),
        "side0_version_space": value.side0_version_space.to_data(),
        "side1_version_space": value.side1_version_space.to_data(),
        "status": value.status.value,
        "support_gap": None if value.support_gap is None else value.support_gap.to_data(),
        "selection_gap": (
            None if value.selection_gap is None else value.selection_gap.to_data()
        ),
        "rank_artifact": (
            None if value.rank_artifact is None else value.rank_artifact.to_data()
        ),
        "rank_artifact_digest": (
            None
            if value.rank_artifact is None
            else value.rank_artifact.artifact_digest
        ),
        "predicate_pair": (
            None if value.predicate_pair is None else value.predicate_pair.to_data()
        ),
        "task_freeze": (
            None if value.task_freeze is None else value.task_freeze.to_data()
        ),
        "task_freeze_store_receipt": (
            None
            if value.task_freeze_store_receipt is None
            else value.task_freeze_store_receipt.to_data()
        ),
        "task_freeze_commit": (
            None
            if value.task_freeze_commit is None
            else value.task_freeze_commit.to_data()
        ),
        "task_freeze_commit_store_receipt": (
            None
            if value.task_freeze_commit_store_receipt is None
            else value.task_freeze_commit_store_receipt.to_data()
        ),
        "sealed_query_panel_ids": {
            "side_0": value.task_plan.side_0_query_panel_id,
            "side_1": value.task_plan.side_1_query_panel_id,
        },
        "query_png_base64_by_side": {
            side: encoded for side, encoded in value.query_png_base64_by_side
        },
        "query_released_panels": [
            item.to_data() for item in value.query_released_panels
        ],
        "query_release_store_receipts": [
            item.to_data() for item in value.query_release_store_receipts
        ],
        "query_observations": [item.to_data() for item in value.query_observations],
        "query_decisions": [item.to_data() for item in value.query_decisions],
        "freeze_persist_reload_invocations": value.freeze_persist_reload_invocations,
        "query_release_invocations": value.query_release_invocations,
        "query_observer_invocations": value.query_observer_invocations,
        "query_release_called_only_after_exact_task_freeze_commit_reload": True,
        "query_release_and_observation_are_separate_calls": True,
        "query_observer_receives_predicate_or_formula": False,
        "exact_png_bytes_archived_for_cold_replay": True,
        "cold_replay_callback_invocations": 0,
        "cold_replay_model_calls": 0,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class PanelFeatureTaskArchive:
    task_plan: ObjectBongardTaskPlan
    execution_precommit_digest: str
    exposure_successor_digest: str
    support_released_panels: tuple[AuthenticatedReleasedPanel, ...]
    support_release_store_receipts: tuple[ObjectBongardWriteOnceReceipt, ...]
    support_png_base64_by_panel_id: tuple[tuple[str, str], ...]
    proposer_result: PanelFeatureProposerResult
    support_observations: tuple[PanelFeatureObservationSet, ...]
    vocabulary: FeatureVocabulary
    support_table: EngineeringSupportTable
    side0_version_space: EngineeringFeatureVersionSpace
    side1_version_space: EngineeringFeatureVersionSpace
    status: PanelFeatureTaskRunStatus
    support_gap: PanelFeatureTaskSupportGap | None
    selection_gap: PanelFeatureTaskSelectionGap | None
    rank_artifact: PanelFeatureRankArtifact | None
    predicate_pair: PanelFeatureSelectedPredicatePair | None
    task_freeze: PanelFeatureTaskFreeze | None
    task_freeze_store_receipt: ObjectBongardWriteOnceReceipt | None
    task_freeze_commit: PanelFeatureTaskFreezeCommit | None
    task_freeze_commit_store_receipt: ObjectBongardWriteOnceReceipt | None
    query_png_base64_by_side: tuple[tuple[str, str], ...]
    query_released_panels: tuple[AuthenticatedReleasedPanel, ...]
    query_release_store_receipts: tuple[ObjectBongardWriteOnceReceipt, ...]
    query_observations: tuple[PanelFeatureObservationSet, ...]
    query_decisions: tuple[PanelFeatureQueryDecision, ...]
    freeze_persist_reload_invocations: int
    query_release_invocations: int
    query_observer_invocations: int
    record_digest: str

    def __post_init__(self) -> None:
        task = _task(self.task_plan)
        _raw_digest(self.record_digest, "task archive digest")
        if task != self.task_plan:
            raise PanelFeatureTaskRunnerError("archive task differs")
        _address(self.execution_precommit_digest, "archive execution precommit digest")
        _address(self.exposure_successor_digest, "archive exposure successor digest")
        if (
            type(self.support_released_panels) is not tuple
            or len(self.support_released_panels) != PANEL_FEATURE_SUPPORT_PANEL_COUNT
            or any(
                type(item) not in {ReleasedOfficialPanel, ReleasedOfficialExtractedPanel}
                for item in self.support_released_panels
            )
            or type(self.support_release_store_receipts) is not tuple
            or len(self.support_release_store_receipts) != PANEL_FEATURE_SUPPORT_PANEL_COUNT
            or any(
                type(item) is not ObjectBongardWriteOnceReceipt
                for item in self.support_release_store_receipts
            )
            or type(self.support_png_base64_by_panel_id) is not tuple
            or tuple(item[0] for item in self.support_png_base64_by_panel_id)
            != _support_ids(task)
            or type(self.support_observations) is not tuple
            or len(self.support_observations) != PANEL_FEATURE_SUPPORT_PANEL_COUNT
            or any(
                type(item) is not PanelFeatureObservationSet
                for item in self.support_observations
            )
            or type(self.vocabulary) is not FeatureVocabulary
            or type(self.support_table) is not EngineeringSupportTable
            or type(self.side0_version_space) is not EngineeringFeatureVersionSpace
            or type(self.side1_version_space) is not EngineeringFeatureVersionSpace
            or type(self.status) is not PanelFeatureTaskRunStatus
            or type(self.query_png_base64_by_side) is not tuple
            or type(self.query_released_panels) is not tuple
            or type(self.query_release_store_receipts) is not tuple
            or type(self.query_observations) is not tuple
            or type(self.query_decisions) is not tuple
            or any(
                type(item) is not PanelFeatureObservationSet
                for item in self.query_observations
            )
            or any(type(item) is not PanelFeatureQueryDecision for item in self.query_decisions)
            or type(self.freeze_persist_reload_invocations) is not int
            or type(self.query_release_invocations) is not int
            or type(self.query_observer_invocations) is not int
        ):
            raise PanelFeatureTaskRunnerError("task archive field types differ")
        _canonical_proposer(self.proposer_result)
        support_panels, support_receipts, support_pngs = _released_rows(
            tuple(
                zip(
                    self.support_released_panels,
                    self.support_release_store_receipts,
                    strict=True,
                )
            ),
            _support_ids(task),
            expected_execution_precommit_digest=self.execution_precommit_digest,
            expected_exposure_successor_digest=self.exposure_successor_digest,
            object_kind="released-support-panel",
            label="archived support",
        )
        encoded_support = _decode_png_rows(
            self.support_png_base64_by_panel_id,
            _support_ids(task),
            label="archived support",
        )
        if (
            support_panels != self.support_released_panels
            or support_receipts != self.support_release_store_receipts
            or support_pngs != encoded_support
        ):
            raise PanelFeatureTaskRunnerError("archived support release custody differs")
        if self.query_released_panels or self.query_release_store_receipts:
            query_panels, query_receipts, query_pngs = _released_rows(
                tuple(
                    zip(
                        self.query_released_panels,
                        self.query_release_store_receipts,
                        strict=True,
                    )
                ),
                (task.side_0_query_panel_id, task.side_1_query_panel_id),
                expected_execution_precommit_digest=self.execution_precommit_digest,
                expected_exposure_successor_digest=self.exposure_successor_digest,
                object_kind="released-query-panel",
                label="archived query",
            )
            if (
                query_panels != self.query_released_panels
                or query_receipts != self.query_release_store_receipts
                or query_pngs
                != _decode_png_rows(
                    self.query_png_base64_by_side, _SIDES, label="archived query"
                )
            ):
                raise PanelFeatureTaskRunnerError("archived query release custody differs")
        complete_shape = (
            self.support_gap is None
            and self.selection_gap is None
            and type(self.predicate_pair) is PanelFeatureSelectedPredicatePair
            and (
                (
                    self.rank_artifact is None
                    and self.predicate_pair.rank_artifact_digest is None
                    and self.predicate_pair.selection_mode
                    == "unique_support_survivor"
                )
                or (
                    type(self.rank_artifact) is PanelFeatureRankArtifact
                    and self.predicate_pair.rank_artifact_digest
                    == self.rank_artifact.artifact_digest
                    and self.predicate_pair.selection_mode
                    == "verified_support_rank_artifact"
                )
            )
            and type(self.task_freeze) is PanelFeatureTaskFreeze
            and self.task_freeze.predicate_pair == self.predicate_pair
            and self.task_freeze.rank_artifact == self.rank_artifact
            and type(self.task_freeze_store_receipt) is ObjectBongardWriteOnceReceipt
            and type(self.task_freeze_commit) is PanelFeatureTaskFreezeCommit
            and type(self.task_freeze_commit_store_receipt) is ObjectBongardWriteOnceReceipt
            and tuple(item[0] for item in self.query_png_base64_by_side) == _SIDES
            and len(self.query_released_panels) == PANEL_FEATURE_QUERY_PANEL_COUNT
            and len(self.query_release_store_receipts) == PANEL_FEATURE_QUERY_PANEL_COUNT
            and len(self.query_observations) == PANEL_FEATURE_QUERY_PANEL_COUNT
            and len(self.query_decisions) == PANEL_FEATURE_QUERY_PANEL_COUNT
            and (
                self.freeze_persist_reload_invocations,
                self.query_release_invocations,
                self.query_observer_invocations,
            )
            == (1, 1, 2)
        )
        gap_shape = (
            type(self.support_gap) is PanelFeatureTaskSupportGap
            and self.selection_gap is None
            and self.rank_artifact is None
            and self.predicate_pair is None
            and self.task_freeze is None
            and self.task_freeze_store_receipt is None
            and self.task_freeze_commit is None
            and self.task_freeze_commit_store_receipt is None
            and not self.query_png_base64_by_side
            and not self.query_released_panels
            and not self.query_release_store_receipts
            and not self.query_observations
            and not self.query_decisions
            and (
                self.freeze_persist_reload_invocations,
                self.query_release_invocations,
                self.query_observer_invocations,
            )
            == (0, 0, 0)
        )
        selection_gap_shape = (
            self.support_gap is None
            and type(self.selection_gap) is PanelFeatureTaskSelectionGap
            and self.rank_artifact is None
            and self.predicate_pair is None
            and self.task_freeze is None
            and self.task_freeze_store_receipt is None
            and self.task_freeze_commit is None
            and self.task_freeze_commit_store_receipt is None
            and not self.query_png_base64_by_side
            and not self.query_released_panels
            and not self.query_release_store_receipts
            and not self.query_observations
            and not self.query_decisions
            and (
                self.freeze_persist_reload_invocations,
                self.query_release_invocations,
                self.query_observer_invocations,
            )
            == (0, 0, 0)
        )
        if (
            (
                self.status is PanelFeatureTaskRunStatus.COMPLETE
                and not complete_shape
            )
            or (
                self.status is PanelFeatureTaskRunStatus.SUPPORT_GAP
                and not gap_shape
            )
            or (
                self.status is PanelFeatureTaskRunStatus.SELECTION_GAP
                and not selection_gap_shape
            )
            or self.record_digest != canonical_digest(_archive_content(self))
        ):
            raise PanelFeatureTaskRunnerError("task archive phase shape or digest differs")

    @property
    def archive_address(self) -> str:
        return "sha256:" + self.record_digest

    def to_data(self) -> dict[str, object]:
        return {
            **_archive_content(self),
            "record_digest": self.record_digest,
            "archive_address": self.archive_address,
        }

    @classmethod
    def from_data(cls, value: object) -> "PanelFeatureTaskArchive":
        raw = _fields(
            value,
            {
                "schema",
                "runner_id",
                "deployment_observer_axes",
                "deployment_observer_axis_catalog_digest",
                "deployment_observer_axis_subset_permitted",
                "deployment_observer_catalog_caller_or_candidate_selected",
                "task_plan",
                "execution_precommit_digest",
                "exposure_successor_digest",
                "support_released_panels",
                "support_release_store_receipts",
                "support_png_base64_by_panel_id",
                "proposer_result",
                "proposer_result_digest",
                "support_observations",
                "vocabulary",
                "support_table",
                "side0_version_space",
                "side1_version_space",
                "status",
                "support_gap",
                "selection_gap",
                "rank_artifact",
                "rank_artifact_digest",
                "predicate_pair",
                "task_freeze",
                "task_freeze_store_receipt",
                "task_freeze_commit",
                "task_freeze_commit_store_receipt",
                "sealed_query_panel_ids",
                "query_png_base64_by_side",
                "query_released_panels",
                "query_release_store_receipts",
                "query_observations",
                "query_decisions",
                "freeze_persist_reload_invocations",
                "query_release_invocations",
                "query_observer_invocations",
                "query_release_called_only_after_exact_task_freeze_commit_reload",
                "query_release_and_observation_are_separate_calls",
                "query_observer_receives_predicate_or_formula",
                "exact_png_bytes_archived_for_cold_replay",
                "cold_replay_callback_invocations",
                "cold_replay_model_calls",
                *_authority_data(),
                "record_digest",
                "archive_address",
            },
            "panel-feature task archive",
        )
        if (
            raw["schema"] != PANEL_FEATURE_TASK_ARCHIVE_SCHEMA
            or raw["runner_id"] != PANEL_FEATURE_TASK_RUNNER_ID
            or raw["deployment_observer_axes"]
            != _deployment_axis_catalog_data()
            or raw["deployment_observer_axis_catalog_digest"]
            != _deployment_axis_catalog_digest()
            or raw["deployment_observer_axis_subset_permitted"] is not False
            or raw[
                "deployment_observer_catalog_caller_or_candidate_selected"
            ] is not False
            or raw[
                "query_release_called_only_after_exact_task_freeze_commit_reload"
            ] is not True
            or raw["query_release_and_observation_are_separate_calls"] is not True
            or raw["query_observer_receives_predicate_or_formula"] is not False
            or raw["exact_png_bytes_archived_for_cold_replay"] is not True
            or raw["cold_replay_callback_invocations"] != 0
            or raw["cold_replay_model_calls"] != 0
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["support_png_base64_by_panel_id"], Mapping)
            or not isinstance(raw["query_png_base64_by_side"], Mapping)
            or type(raw["support_released_panels"]) is not list
            or type(raw["support_release_store_receipts"]) is not list
            or type(raw["query_released_panels"]) is not list
            or type(raw["query_release_store_receipts"]) is not list
            or type(raw["support_observations"]) is not list
            or type(raw["query_observations"]) is not list
            or type(raw["query_decisions"]) is not list
        ):
            raise PanelFeatureTaskRunnerError("task archive policy differs")
        task = ObjectBongardTaskPlan.from_data(raw["task_plan"])
        support_ids = _support_ids(task)
        support_encoded = raw["support_png_base64_by_panel_id"]
        query_encoded = raw["query_png_base64_by_side"]
        sealed_queries = raw["sealed_query_panel_ids"]
        if (
            set(support_encoded) != set(support_ids)
            or set(query_encoded) not in (set(), set(_SIDES))
            or not isinstance(sealed_queries, Mapping)
            or dict(sealed_queries)
            != {
                "side_0": task.side_0_query_panel_id,
                "side_1": task.side_1_query_panel_id,
            }
        ):
            raise PanelFeatureTaskRunnerError("archive panel identities differ")
        proposer = _proposer_result_from_data(raw["proposer_result"])
        if raw["proposer_result_digest"] != proposer.result_digest:
            raise PanelFeatureTaskRunnerError("archive proposer result digest differs")
        try:
            status = PanelFeatureTaskRunStatus(raw["status"])
        except (TypeError, ValueError) as exc:
            raise PanelFeatureTaskRunnerError("archive status differs") from exc
        result = cls(
            task,
            raw["execution_precommit_digest"],
            raw["exposure_successor_digest"],
            tuple(
                _released_panel_from_data(item)
                for item in raw["support_released_panels"]
            ),
            tuple(
                ObjectBongardWriteOnceReceipt.from_data(item)
                for item in raw["support_release_store_receipts"]
            ),
            tuple((item, support_encoded[item]) for item in support_ids),
            proposer,
            tuple(
                PanelFeatureObservationSet.from_data(item)
                for item in raw["support_observations"]
            ),
            FeatureVocabulary.from_data(raw["vocabulary"]),
            EngineeringSupportTable.from_data(raw["support_table"]),
            EngineeringFeatureVersionSpace.from_data(raw["side0_version_space"]),
            EngineeringFeatureVersionSpace.from_data(raw["side1_version_space"]),
            status,
            (
                None
                if raw["support_gap"] is None
                else PanelFeatureTaskSupportGap.from_data(raw["support_gap"])
            ),
            (
                None
                if raw["selection_gap"] is None
                else PanelFeatureTaskSelectionGap.from_data(raw["selection_gap"])
            ),
            (
                None
                if raw["rank_artifact"] is None
                else PanelFeatureRankArtifact.from_data(raw["rank_artifact"])
            ),
            (
                None
                if raw["predicate_pair"] is None
                else PanelFeatureSelectedPredicatePair.from_data(
                    raw["predicate_pair"]
                )
            ),
            (
                None
                if raw["task_freeze"] is None
                else PanelFeatureTaskFreeze.from_data(raw["task_freeze"])
            ),
            (
                None
                if raw["task_freeze_store_receipt"] is None
                else ObjectBongardWriteOnceReceipt.from_data(
                    raw["task_freeze_store_receipt"]
                )
            ),
            (
                None
                if raw["task_freeze_commit"] is None
                else PanelFeatureTaskFreezeCommit.from_data(
                    raw["task_freeze_commit"]
                )
            ),
            (
                None
                if raw["task_freeze_commit_store_receipt"] is None
                else ObjectBongardWriteOnceReceipt.from_data(
                    raw["task_freeze_commit_store_receipt"]
                )
            ),
            tuple((side, query_encoded[side]) for side in _SIDES if side in query_encoded),
            tuple(
                _released_panel_from_data(item)
                for item in raw["query_released_panels"]
            ),
            tuple(
                ObjectBongardWriteOnceReceipt.from_data(item)
                for item in raw["query_release_store_receipts"]
            ),
            tuple(
                PanelFeatureObservationSet.from_data(item)
                for item in raw["query_observations"]
            ),
            tuple(
                PanelFeatureQueryDecision.from_data(item)
                for item in raw["query_decisions"]
            ),
            raw["freeze_persist_reload_invocations"],
            raw["query_release_invocations"],
            raw["query_observer_invocations"],
            raw["record_digest"],
        )
        if (
            raw["rank_artifact_digest"]
            != (
                None
                if result.rank_artifact is None
                else result.rank_artifact.artifact_digest
            )
            or raw["archive_address"] != result.archive_address
            or result.to_data() != dict(raw)
        ):
            raise PanelFeatureTaskRunnerError("task archive is not canonical")
        return result


def _make_archive(**values: object) -> PanelFeatureTaskArchive:
    provisional = object.__new__(PanelFeatureTaskArchive)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return PanelFeatureTaskArchive(
        **values,  # type: ignore[arg-type]
        record_digest=canonical_digest(_archive_content(provisional)),
    )


ReleasedPanelRow = tuple[AuthenticatedReleasedPanel, ObjectBongardWriteOnceReceipt]
FreezePersistReload = Callable[
    [PanelFeatureTaskFreeze],
    tuple[
        PanelFeatureTaskFreezeCommit,
        ObjectBongardWriteOnceReceipt,
        ObjectBongardWriteOnceReceipt,
    ],
]
QueryReleaseCallback = Callable[[], Mapping[str, ReleasedPanelRow]]
PanelObserverCallback = Callable[
    [str, bytes, tuple[FeatureAxis, ...]], PanelFeatureObservationSet
]
SupportProposerCallback = Callable[
    [tuple[bytes, ...], str], PanelFeatureProposerResult
]
SupportRankCallback = Callable[
    [
        EngineeringFeatureVersionSpace,
        EngineeringFeatureVersionSpace,
        PanelFeatureProposerResult,
    ],
    PanelFeatureRankArtifact,
]
SupportObserverCallback = PanelObserverCallback
QueryObserverCallback = PanelObserverCallback


def _canonical_rank_artifact(
    value: object,
    *,
    proposer: PanelFeatureProposerResult,
    side0_space: EngineeringFeatureVersionSpace,
    side1_space: EngineeringFeatureVersionSpace,
) -> PanelFeatureRankArtifact:
    if type(value) is not PanelFeatureRankArtifact:
        raise PanelFeatureTaskRunnerError(
            "rank resolution returned no exact PanelFeatureRankArtifact"
        )
    try:
        restored = PanelFeatureRankArtifact.from_data(value.to_data())
        expected_input = PanelFeatureRankInput.freeze(
            side0_space, side1_space, proposer
        )
    except Exception as exc:
        raise PanelFeatureTaskRunnerError(
            "rank artifact canonical replay failed"
        ) from exc
    if (
        restored != value
        or restored.rank_input != expected_input
        or not restored.transport_provenance.benchmark_sealable
        or restored.selected_side0_formula_digest
        not in side0_space.survivor_formula_digests
        or restored.selected_side1_formula_digest
        not in side1_space.survivor_formula_digests
    ):
        raise PanelFeatureTaskRunnerError(
            "rank artifact is unsealable or bound to different support inputs"
        )
    return restored


def _query_release_rows(
    callback_result: object,
    task: ObjectBongardTaskPlan,
    *,
    execution_precommit_digest: str,
    exposure_successor_digest: str,
) -> tuple[
    tuple[AuthenticatedReleasedPanel, ...],
    tuple[ObjectBongardWriteOnceReceipt, ...],
    tuple[bytes, ...],
]:
    query_ids = (task.side_0_query_panel_id, task.side_1_query_panel_id)
    if (
        not isinstance(callback_result, Mapping)
        or any(type(key) is not str for key in callback_result)
        or set(callback_result) != set(query_ids)
    ):
        raise PanelFeatureTaskRunnerError(
            "query release must return the exact two sealed query panel IDs"
        )
    return _released_rows(
        tuple(callback_result[item] for item in query_ids),
        query_ids,
        expected_execution_precommit_digest=execution_precommit_digest,
        expected_exposure_successor_digest=exposure_successor_digest,
        object_kind="released-query-panel",
        label="query",
    )


def _observe_unlabelled_panels(
    pngs: Sequence[bytes],
    observer: PanelObserverCallback,
) -> tuple[PanelFeatureObservationSet, ...]:
    """Observe panels with opaque names and the preregistered batch catalog."""

    if not callable(observer):
        raise TypeError("panel observer must be callable")
    axes = complete_whole_panel_feature_axes()
    order = tuple(
        sorted(
            range(len(pngs)),
            key=lambda index: (hashlib.sha256(pngs[index]).hexdigest(), index),
        )
    )
    by_index: dict[int, PanelFeatureObservationSet] = {}
    for ordinal, index in enumerate(order):
        token = f"panel_{ordinal:03d}"
        by_index[index] = observer(token, pngs[index], axes)
    return tuple(by_index[index] for index in range(len(pngs)))


def run_panel_feature_task(
    task_plan: ObjectBongardTaskPlan,
    support_releases: Sequence[ReleasedPanelRow],
    proposer_result: PanelFeatureProposerResult,
    support_observations: Sequence[PanelFeatureObservationSet],
    *,
    execution_precommit_digest: str,
    exposure_successor_digest: str,
    rank_artifact: PanelFeatureRankArtifact | None = None,
    rank_callback: SupportRankCallback | None = None,
    freeze_persist_reload: FreezePersistReload | None,
    query_release_callback: QueryReleaseCallback | None,
    query_observation_callback: QueryObserverCallback | None,
) -> PanelFeatureTaskArchive:
    """Engineering/test lane over already authenticated released-panel records.

    Pixel release and observation are deliberately split.  The zero-argument
    release callback cannot receive a formula, and the observer receives only
    an opaque token, exact bytes, and the fixed complete whole-panel deployment
    catalog.  This
    injected-callback lane remains non-benchmark-authoritative; use
    :func:`run_panel_feature_task_with_official_release` for the existing
    release-gate-backed production path.
    """

    task = _task(task_plan)
    support_panels, support_receipts, support_pngs = _released_rows(
        support_releases,
        _support_ids(task),
        expected_execution_precommit_digest=execution_precommit_digest,
        expected_exposure_successor_digest=exposure_successor_digest,
        object_kind="released-support-panel",
        label="support",
    )

    (
        task,
        pngs,
        proposer,
        observations,
        vocabulary,
        table,
        side0_space,
        side1_space,
        contract,
        protocol,
    ) = _derive_support(
        task, support_pngs, proposer_result, support_observations
    )
    common = {
        "task_plan": task,
        "execution_precommit_digest": execution_precommit_digest,
        "exposure_successor_digest": exposure_successor_digest,
        "support_released_panels": support_panels,
        "support_release_store_receipts": support_receipts,
        "support_png_base64_by_panel_id": _encode_png_rows(
            _support_ids(task), pngs
        ),
        "proposer_result": proposer,
        "support_observations": observations,
        "vocabulary": vocabulary,
        "support_table": table,
        "side0_version_space": side0_space,
        "side1_version_space": side1_space,
    }
    if not side0_space.survivor_formulas or not side1_space.survivor_formulas:
        gap = PanelFeatureTaskSupportGap.create(side0_space, side1_space)
        archive = _make_archive(
            **common,
            status=PanelFeatureTaskRunStatus.SUPPORT_GAP,
            support_gap=gap,
            selection_gap=None,
            rank_artifact=None,
            predicate_pair=None,
            task_freeze=None,
            task_freeze_store_receipt=None,
            task_freeze_commit=None,
            task_freeze_commit_store_receipt=None,
            query_png_base64_by_side=(),
            query_released_panels=(),
            query_release_store_receipts=(),
            query_observations=(),
            query_decisions=(),
            freeze_persist_reload_invocations=0,
            query_release_invocations=0,
            query_observer_invocations=0,
        )
        return cold_replay_panel_feature_task(
            archive, expected_archive_address=archive.archive_address
        )
    survivor_counts = (
        len(side0_space.survivor_formula_digests),
        len(side1_space.survivor_formula_digests),
    )
    verified_rank: PanelFeatureRankArtifact | None = None
    if survivor_counts != (1, 1):
        try:
            if rank_artifact is not None and rank_callback is not None:
                raise PanelFeatureTaskRunnerError(
                    "rank artifact and callback are mutually exclusive"
                )
            candidate = (
                rank_callback(side0_space, side1_space, proposer)
                if rank_callback is not None and callable(rank_callback)
                else rank_artifact
            )
            verified_rank = _canonical_rank_artifact(
                candidate,
                proposer=proposer,
                side0_space=side0_space,
                side1_space=side1_space,
            )
        except Exception:
            gap = PanelFeatureTaskSelectionGap.create(side0_space, side1_space)
            archive = _make_archive(
                **common,
                status=PanelFeatureTaskRunStatus.SELECTION_GAP,
                support_gap=None,
                selection_gap=gap,
                rank_artifact=None,
                predicate_pair=None,
                task_freeze=None,
                task_freeze_store_receipt=None,
                task_freeze_commit=None,
                task_freeze_commit_store_receipt=None,
                query_png_base64_by_side=(),
                query_released_panels=(),
                query_release_store_receipts=(),
                query_observations=(),
                query_decisions=(),
                freeze_persist_reload_invocations=0,
                query_release_invocations=0,
                query_observer_invocations=0,
            )
            return cold_replay_panel_feature_task(
                archive, expected_archive_address=archive.archive_address
            )
    if (
        not callable(freeze_persist_reload)
        or not callable(query_release_callback)
        or not callable(query_observation_callback)
    ):
        raise TypeError(
            "a selected support pair requires freeze, query release, and query observer callbacks"
        )
    freeze = PanelFeatureTaskFreeze.seal(
        task=task,
        execution_precommit_digest=execution_precommit_digest,
        proposer=proposer,
        side0_space=side0_space,
        side1_space=side1_space,
        rank_artifact=verified_rank,
    )
    durable = freeze_persist_reload(freeze)
    if type(durable) is not tuple or len(durable) != 3:
        raise PanelFeatureTaskRunnerError("freeze persistence result differs")
    reloaded_freeze, commit, freeze_receipt, commit_receipt = _verify_durable_freeze(
        freeze, durable[0], durable[1], durable[2]
    )
    if reloaded_freeze != freeze:
        raise PanelFeatureTaskRunnerError("durably reloaded task freeze differs")
    # This is the first point at which query pixels may be obtained.
    query_panels, query_receipts, query_pngs = _query_release_rows(
        query_release_callback(),
        task,
        execution_precommit_digest=execution_precommit_digest,
        exposure_successor_digest=exposure_successor_digest,
    )
    raw_query_observations = _observe_unlabelled_panels(
        query_pngs, query_observation_callback
    )
    query_observations, _, _ = _verify_observation_batch(
        raw_query_observations,
        query_pngs,
        expected_contract_digest=contract,
        expected_protocol_digest=protocol,
        label="query",
    )
    reloaded_pair = reloaded_freeze.predicate_pair
    decisions: list[PanelFeatureQueryDecision] = []
    for observation in query_observations:
        query_table = _table_for_observations(vocabulary, (observation,))
        decisions.append(
            PanelFeatureQueryDecision.create(
                reloaded_pair, query_table, observation.panel_digest
            )
        )
    archive = _make_archive(
        **common,
        status=PanelFeatureTaskRunStatus.COMPLETE,
        support_gap=None,
        selection_gap=None,
        rank_artifact=verified_rank,
        predicate_pair=reloaded_pair,
        task_freeze=reloaded_freeze,
        task_freeze_store_receipt=freeze_receipt,
        task_freeze_commit=commit,
        task_freeze_commit_store_receipt=commit_receipt,
        query_png_base64_by_side=_encode_png_rows(_SIDES, query_pngs),
        query_released_panels=query_panels,
        query_release_store_receipts=query_receipts,
        query_observations=query_observations,
        query_decisions=tuple(decisions),
        freeze_persist_reload_invocations=1,
        query_release_invocations=1,
        query_observer_invocations=2,
    )
    return cold_replay_panel_feature_task(
        archive, expected_archive_address=archive.archive_address
    )


def run_panel_feature_task_with_support_callbacks(
    task_plan: ObjectBongardTaskPlan,
    support_releases: Sequence[ReleasedPanelRow],
    *,
    proposer_callback: SupportProposerCallback,
    observation_callback: SupportObserverCallback,
    rank_callback: SupportRankCallback | None = None,
    execution_precommit_digest: str,
    exposure_successor_digest: str,
    freeze_persist_reload: FreezePersistReload | None,
    query_release_callback: QueryReleaseCallback | None,
    query_observation_callback: QueryObserverCallback | None,
) -> PanelFeatureTaskArchive:
    """Run one proposer call, then twelve independent support batch calls.

    The proposer still receives the required two contrastive blocks.  The
    observer does not: its calls are reordered by pixel digest and contain no
    side, block, position, candidate spec, formula, or orientation.  Every
    observer receives the exact preregistered complete whole-panel tuple.
    Query release and its two batch calls remain forbidden until after the
    selected predicate freeze and commit have been durably reloaded.
    """

    task = _task(task_plan)
    _, _, pngs = _released_rows(
        support_releases,
        _support_ids(task),
        expected_execution_precommit_digest=execution_precommit_digest,
        expected_exposure_successor_digest=exposure_successor_digest,
        object_kind="released-support-panel",
        label="support",
    )
    if not callable(proposer_callback) or not callable(observation_callback):
        raise TypeError("support proposer and observation callbacks must be callable")
    proposer = proposer_callback(pngs, task.record_digest.split(":", 1)[1])
    canonical_proposer, _vocabulary = _derive_vocabulary_and_verify_provenance(
        task, pngs, proposer
    )
    observations = _observe_unlabelled_panels(pngs, observation_callback)
    return run_panel_feature_task(
        task,
        support_releases,
        canonical_proposer,
        observations,
        execution_precommit_digest=execution_precommit_digest,
        exposure_successor_digest=exposure_successor_digest,
        rank_callback=rank_callback,
        freeze_persist_reload=freeze_persist_reload,
        query_release_callback=query_release_callback,
        query_observation_callback=query_observation_callback,
    )


def run_panel_feature_task_with_official_release(
    task_plan: ObjectBongardTaskPlan,
    *,
    prepared: PreparedObjectBongardRelease,
    archive: OfficialPanelArchive,
    proposer_callback: SupportProposerCallback,
    observation_callback: SupportObserverCallback,
    query_observation_callback: QueryObserverCallback,
    rank_callback: SupportRankCallback | None = None,
) -> PanelFeatureTaskArchive:
    """Release pixels through the repository's trusted official gate directly."""

    task = _task(task_plan)
    if type(prepared) is not PreparedObjectBongardRelease:
        raise TypeError("prepared must be exact PreparedObjectBongardRelease")
    if type(archive) is not OfficialPanelArchive:
        raise TypeError("archive must be exact OfficialPanelArchive")
    prepared = verify_prepared_object_bongard_release(prepared)
    matches = tuple(item for item in prepared.plan.tasks if item.task_id == task.task_id)
    if len(matches) != 1 or matches[0] != task:
        raise PanelFeatureTaskRunnerError("task differs from prepared official release")
    support_releases = tuple(
        release_object_bongard_support_panel(
            prepared=prepared, archive=archive, panel_id=panel_id
        )
        for panel_id in _support_ids(task)
    )
    durable_state: dict[str, object] = {}

    def persist_freeze(
        freeze: PanelFeatureTaskFreeze,
    ) -> tuple[
        PanelFeatureTaskFreezeCommit,
        ObjectBongardWriteOnceReceipt,
        ObjectBongardWriteOnceReceipt,
    ]:
        freeze_receipt = persist_object_bongard_task_freeze(
            store=prepared.store, freeze=freeze
        )
        prepared.store.verify(freeze_receipt, expected_data=freeze.to_data())
        commit = PanelFeatureTaskFreezeCommit.seal(freeze, freeze_receipt)
        commit_receipt = persist_object_bongard_task_commit(
            store=prepared.store, commit=commit
        )
        prepared.store.verify(commit_receipt, expected_data=commit.to_data())
        durable_state.update(
            freeze=freeze,
            commit=commit,
            freeze_receipt=freeze_receipt,
            commit_receipt=commit_receipt,
        )
        return commit, freeze_receipt, commit_receipt

    def release_queries() -> Mapping[str, ReleasedPanelRow]:
        if set(durable_state) != {
            "freeze",
            "commit",
            "freeze_receipt",
            "commit_receipt",
        }:
            raise PanelFeatureTaskRunnerError(
                "query release attempted before exact durable freeze and commit"
            )
        return {
            panel_id: release_object_bongard_query_panel(
                prepared=prepared,
                archive=archive,
                panel_id=panel_id,
                task_freeze=durable_state["freeze"],
                task_commit=durable_state["commit"],
                task_freeze_receipt=durable_state["freeze_receipt"],
                task_commit_receipt=durable_state["commit_receipt"],
            )
            for panel_id in (
                task.side_0_query_panel_id,
                task.side_1_query_panel_id,
            )
        }

    return run_panel_feature_task_with_support_callbacks(
        task,
        support_releases,
        proposer_callback=proposer_callback,
        observation_callback=observation_callback,
        rank_callback=rank_callback,
        execution_precommit_digest=prepared.precommit.record_digest,
        exposure_successor_digest=prepared.successor.digest,
        freeze_persist_reload=persist_freeze,
        query_release_callback=release_queries,
        query_observation_callback=query_observation_callback,
    )


def cold_replay_panel_feature_task(
    archive: PanelFeatureTaskArchive,
    *,
    expected_archive_address: str,
) -> PanelFeatureTaskArchive:
    """Recompute a task solely from archived bytes and typed records.

    The function has no callback parameters and performs zero model calls.
    """

    if type(archive) is not PanelFeatureTaskArchive:
        raise TypeError("archive must be exact PanelFeatureTaskArchive")
    restored = PanelFeatureTaskArchive.from_data(archive.to_data())
    if restored.archive_address != _address(
        expected_archive_address, "expected archive address"
    ):
        raise PanelFeatureTaskRunnerError("task archive differs from commitment")
    support_pngs = _decode_png_rows(
        restored.support_png_base64_by_panel_id,
        _support_ids(restored.task_plan),
        label="support",
    )
    (
        task,
        _pngs,
        proposer,
        observations,
        vocabulary,
        table,
        side0_space,
        side1_space,
        contract,
        protocol,
    ) = _derive_support(
        restored.task_plan,
        support_pngs,
        restored.proposer_result,
        restored.support_observations,
    )
    if (
        task != restored.task_plan
        or proposer != restored.proposer_result
        or observations != restored.support_observations
        or vocabulary != restored.vocabulary
        or table != restored.support_table
        or side0_space != restored.side0_version_space
        or side1_space != restored.side1_version_space
    ):
        raise PanelFeatureTaskRunnerError("support cold replay differs")
    has_gap = not side0_space.survivor_formulas or not side1_space.survivor_formulas
    if has_gap:
        gap = PanelFeatureTaskSupportGap.create(side0_space, side1_space)
        if (
            restored.status is not PanelFeatureTaskRunStatus.SUPPORT_GAP
            or restored.support_gap != gap
            or restored.selection_gap is not None
            or restored.rank_artifact is not None
            or restored.predicate_pair is not None
            or restored.task_freeze is not None
            or restored.task_freeze_store_receipt is not None
            or restored.task_freeze_commit is not None
            or restored.task_freeze_commit_store_receipt is not None
            or restored.query_png_base64_by_side
            or restored.query_released_panels
            or restored.query_release_store_receipts
            or restored.query_observations
            or restored.query_decisions
            or (
                restored.freeze_persist_reload_invocations,
                restored.query_release_invocations,
                restored.query_observer_invocations,
            )
            != (0, 0, 0)
        ):
            raise PanelFeatureTaskRunnerError("support-gap cold replay differs")
        return restored
    counts = (
        len(side0_space.survivor_formula_digests),
        len(side1_space.survivor_formula_digests),
    )
    verified_rank: PanelFeatureRankArtifact | None = None
    if counts != (1, 1) and restored.rank_artifact is None:
        gap = PanelFeatureTaskSelectionGap.create(side0_space, side1_space)
        if (
            restored.status is not PanelFeatureTaskRunStatus.SELECTION_GAP
            or restored.support_gap is not None
            or restored.selection_gap != gap
            or restored.rank_artifact is not None
            or restored.predicate_pair is not None
            or restored.task_freeze is not None
            or restored.task_freeze_store_receipt is not None
            or restored.task_freeze_commit is not None
            or restored.task_freeze_commit_store_receipt is not None
            or restored.query_png_base64_by_side
            or restored.query_released_panels
            or restored.query_release_store_receipts
            or restored.query_observations
            or restored.query_decisions
            or (
                restored.freeze_persist_reload_invocations,
                restored.query_release_invocations,
                restored.query_observer_invocations,
            )
            != (0, 0, 0)
        ):
            raise PanelFeatureTaskRunnerError("selection-gap cold replay differs")
        return restored
    if counts != (1, 1):
        verified_rank = _canonical_rank_artifact(
            restored.rank_artifact,
            proposer=proposer,
            side0_space=side0_space,
            side1_space=side1_space,
        )
        selected_formula_digests = verified_rank.selected_formula_digests
    else:
        if restored.rank_artifact is not None:
            raise PanelFeatureTaskRunnerError(
                "unique support pair cannot archive an unnecessary rank artifact"
            )
        selected_formula_digests = (
            side0_space.survivor_formula_digests[0],
            side1_space.survivor_formula_digests[0],
        )
    expected_pair = PanelFeatureSelectedPredicatePair.create(
        side0_space,
        side1_space,
        selected_formula_digests,
        rank_artifact_digest=(
            None if verified_rank is None else verified_rank.artifact_digest
        ),
    )
    if (
        restored.status is not PanelFeatureTaskRunStatus.COMPLETE
        or restored.support_gap is not None
        or restored.selection_gap is not None
        or restored.predicate_pair != expected_pair
        or type(restored.task_freeze) is not PanelFeatureTaskFreeze
        or type(restored.task_freeze_store_receipt)
        is not ObjectBongardWriteOnceReceipt
        or type(restored.task_freeze_commit) is not PanelFeatureTaskFreezeCommit
        or type(restored.task_freeze_commit_store_receipt)
        is not ObjectBongardWriteOnceReceipt
        or (
            restored.freeze_persist_reload_invocations,
            restored.query_release_invocations,
            restored.query_observer_invocations,
        )
        != (1, 1, 2)
    ):
        raise PanelFeatureTaskRunnerError("complete cold-replay phase differs")
    assert restored.task_freeze is not None
    assert restored.task_freeze_store_receipt is not None
    assert restored.task_freeze_commit is not None
    assert restored.task_freeze_commit_store_receipt is not None
    expected_freeze = PanelFeatureTaskFreeze.seal(
        task=task,
        execution_precommit_digest=restored.execution_precommit_digest,
        proposer=proposer,
        side0_space=side0_space,
        side1_space=side1_space,
        rank_artifact=verified_rank,
    )
    frozen, committed, freeze_receipt, commit_receipt = _verify_durable_freeze(
        restored.task_freeze,
        restored.task_freeze_commit,
        restored.task_freeze_store_receipt,
        restored.task_freeze_commit_store_receipt,
    )
    if (
        frozen != expected_freeze
        or frozen.predicate_pair != expected_pair
        or committed != restored.task_freeze_commit
        or freeze_receipt != restored.task_freeze_store_receipt
        or commit_receipt != restored.task_freeze_commit_store_receipt
        or not isinstance(frozen, ObjectBongardTaskFreezeProtocol)
        or not isinstance(committed, ObjectBongardTaskCommitProtocol)
    ):
        raise PanelFeatureTaskRunnerError("task freeze cold replay differs")
    query_panels, query_receipts, query_pngs = _released_rows(
        tuple(
            zip(
                restored.query_released_panels,
                restored.query_release_store_receipts,
                strict=True,
            )
        ),
        (task.side_0_query_panel_id, task.side_1_query_panel_id),
        expected_execution_precommit_digest=restored.execution_precommit_digest,
        expected_exposure_successor_digest=restored.exposure_successor_digest,
        object_kind="released-query-panel",
        label="query",
    )
    if (
        query_panels != restored.query_released_panels
        or query_receipts != restored.query_release_store_receipts
        or query_pngs
        != _decode_png_rows(
            restored.query_png_base64_by_side, _SIDES, label="query"
        )
    ):
        raise PanelFeatureTaskRunnerError("query release cold replay differs")
    query_observations, _, _ = _verify_observation_batch(
        restored.query_observations,
        query_pngs,
        expected_contract_digest=contract,
        expected_protocol_digest=protocol,
        label="query",
    )
    decisions = tuple(
        PanelFeatureQueryDecision.create(
            expected_pair,
            _table_for_observations(vocabulary, (observation,)),
            observation.panel_digest,
        )
        for observation in query_observations
    )
    if decisions != restored.query_decisions:
        raise PanelFeatureTaskRunnerError("query cold replay differs")
    return restored


__all__ = (
    "AuthenticatedReleasedPanel",
    "PANEL_FEATURE_QUERY_PANEL_COUNT",
    "PANEL_FEATURE_SUPPORT_PANEL_COUNT",
    "PANEL_FEATURE_TASK_RUNNER_ID",
    "PANEL_FEATURE_SUPPORT_DERIVATION_SCHEMA",
    "PanelFeatureSupportDerivation",
    "PanelFeatureSupportDerivationStatus",
    "PanelFeatureQueryDecision",
    "PanelFeatureSelectedPredicate",
    "PanelFeatureSelectedPredicatePair",
    "PanelFeatureTaskArchive",
    "PanelFeatureTaskFreeze",
    "PanelFeatureTaskFreezeCommit",
    "PanelFeatureTaskRunStatus",
    "PanelFeatureTaskRunnerError",
    "PanelFeatureTaskSelectionGap",
    "PanelFeatureTaskSupportGap",
    "FreezePersistReload",
    "PanelObserverCallback",
    "QueryObserverCallback",
    "QueryReleaseCallback",
    "ReleasedPanelRow",
    "SupportObserverCallback",
    "SupportProposerCallback",
    "SupportRankCallback",
    "cold_replay_panel_feature_task",
    "derive_panel_feature_support",
    "engineering_disposition_from_observation",
    "run_panel_feature_task",
    "run_panel_feature_task_with_official_release",
    "run_panel_feature_task_with_support_callbacks",
)
