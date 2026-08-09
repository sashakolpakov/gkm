"""Pure-Python task boundary for typed panel-feature Bongard predicates.

The core consumes an exact :class:`ObjectBongardTaskPlan`, twelve support PNG
byte strings in semantic side-0-then-side-1 order, a receipted canonical
``PanelFeatureProposerResult``, and one complete ``PanelFeatureObservationSet``
per support panel.  It builds the two native positive-only engineering version
spaces without compiling prose or accepting arbitrary predicate code.

If both orientations have a survivor, the selected pair is serialized to the
canonical JSON bytes and passed to one persistence-and-reload callback.  Query
pixels cannot be requested until that callback returns the exact same bytes
and those bytes have been canonically decoded into the exact frozen pair.  A
content-addressed archive then supports model-free replay with no callbacks.

This module is deliberately engineering-only and uncalibrated.  Python is the
executable predicate authority.  Lean is neither imported nor required and
does not affect identity, selection, decisions, or replay.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import re
from typing import Any, Callable, Mapping, Sequence

from bongard.canonical import canonical_digest, canonical_json
from bongard.object_bongard_batch import ObjectBongardTaskPlan
from bongard.panel_feature_observation import (
    EngineeringFeatureDisposition,
    FeatureAxis,
    PanelFeatureObservationSet,
)
from bongard.panel_feature_predicate import (
    EngineeringDisposition,
    EngineeringFeatureVersionSpace,
    EngineeringQueryDecision,
    EngineeringSupportTable,
    FeatureVocabulary,
    FrozenEngineeringFeaturePredicatePair,
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
from bongard.panel_soft_ontology import (
    LanguageGapArtifact,
    NativeFeatureProposal,
    NativeOrientation,
    PanelFeatureSpec,
    feature_catalog_digest,
)
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID


PANEL_FEATURE_TASK_RUNNER_ID = (
    "bongard.panel-feature-task/support-freeze-query-python-v1"
)
PANEL_FEATURE_TASK_ARCHIVE_SCHEMA = "gkm.bongard-panel-feature-task-archive.v1"
PANEL_FEATURE_TASK_SUPPORT_GAP_SCHEMA = (
    "gkm.bongard-panel-feature-task-support-gap.v1"
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


def _support_pngs(value: Sequence[bytes]) -> tuple[bytes, ...]:
    if isinstance(value, (bytes, bytearray, str, Mapping)):
        raise TypeError("support_pngs must be an ordered sequence of PNG bytes")
    try:
        result = tuple(value)
    except TypeError as exc:
        raise TypeError("support_pngs must be an ordered sequence") from exc
    if len(result) != PANEL_FEATURE_SUPPORT_PANEL_COUNT:
        raise PanelFeatureTaskRunnerError(
            "support PNG sequence must contain exact side0-six then side1-six"
        )
    return tuple(_png(item, f"support panel {index}") for index, item in enumerate(result))


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
    restored = PanelFeatureObservationSet.from_data(value.to_data())
    if restored != value:
        raise PanelFeatureTaskRunnerError("panel observation canonical reload differs")
    return restored


def _required_axis_digests(vocabulary: FeatureVocabulary) -> tuple[str, ...]:
    return tuple(
        sorted({FeatureAxis.for_spec(spec).axis_digest for spec in vocabulary.specs})
    )


def _verify_observation_batch(
    observations: Sequence[PanelFeatureObservationSet],
    pngs: Sequence[bytes],
    vocabulary: FeatureVocabulary,
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
    required_axes = _required_axis_digests(vocabulary)
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
                f"{label} observation {index} is not the exact complete vocabulary-axis table"
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
        vocabulary,
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
        "query_callback_permitted": False,
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
                "query_callback_permitted",
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
            or raw["query_callback_permitted"] is not False
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


def _archive_content(value: "PanelFeatureTaskArchive") -> dict[str, object]:
    return {
        "schema": PANEL_FEATURE_TASK_ARCHIVE_SCHEMA,
        "runner_id": PANEL_FEATURE_TASK_RUNNER_ID,
        "task_plan": value.task_plan.to_data(),
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
        "predicate_pair": (
            None if value.predicate_pair is None else value.predicate_pair.to_data()
        ),
        "frozen_pair_canonical_base64": value.frozen_pair_canonical_base64,
        "sealed_query_panel_ids": {
            "side_0": value.task_plan.side_0_query_panel_id,
            "side_1": value.task_plan.side_1_query_panel_id,
        },
        "query_png_base64_by_side": {
            side: encoded for side, encoded in value.query_png_base64_by_side
        },
        "query_observations": [item.to_data() for item in value.query_observations],
        "query_decisions": [item.to_data() for item in value.query_decisions],
        "persist_reload_callback_invocations": (
            value.persist_reload_callback_invocations
        ),
        "query_callback_invocations": value.query_callback_invocations,
        "query_callback_called_only_after_exact_frozen_pair_reload": True,
        "exact_png_bytes_archived_for_cold_replay": True,
        "cold_replay_callback_invocations": 0,
        "cold_replay_model_calls": 0,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class PanelFeatureTaskArchive:
    task_plan: ObjectBongardTaskPlan
    support_png_base64_by_panel_id: tuple[tuple[str, str], ...]
    proposer_result: PanelFeatureProposerResult
    support_observations: tuple[PanelFeatureObservationSet, ...]
    vocabulary: FeatureVocabulary
    support_table: EngineeringSupportTable
    side0_version_space: EngineeringFeatureVersionSpace
    side1_version_space: EngineeringFeatureVersionSpace
    status: PanelFeatureTaskRunStatus
    support_gap: PanelFeatureTaskSupportGap | None
    predicate_pair: FrozenEngineeringFeaturePredicatePair | None
    frozen_pair_canonical_base64: str | None
    query_png_base64_by_side: tuple[tuple[str, str], ...]
    query_observations: tuple[PanelFeatureObservationSet, ...]
    query_decisions: tuple[EngineeringQueryDecision, ...]
    persist_reload_callback_invocations: int
    query_callback_invocations: int
    record_digest: str

    def __post_init__(self) -> None:
        task = _task(self.task_plan)
        _raw_digest(self.record_digest, "task archive digest")
        if task != self.task_plan:
            raise PanelFeatureTaskRunnerError("archive task differs")
        if (
            type(self.support_png_base64_by_panel_id) is not tuple
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
            or type(self.query_observations) is not tuple
            or type(self.query_decisions) is not tuple
            or any(
                type(item) is not PanelFeatureObservationSet
                for item in self.query_observations
            )
            or any(type(item) is not EngineeringQueryDecision for item in self.query_decisions)
            or type(self.persist_reload_callback_invocations) is not int
            or type(self.query_callback_invocations) is not int
        ):
            raise PanelFeatureTaskRunnerError("task archive field types differ")
        _canonical_proposer(self.proposer_result)
        complete_shape = (
            self.support_gap is None
            and type(self.predicate_pair) is FrozenEngineeringFeaturePredicatePair
            and type(self.frozen_pair_canonical_base64) is str
            and tuple(item[0] for item in self.query_png_base64_by_side) == _SIDES
            and len(self.query_observations) == PANEL_FEATURE_QUERY_PANEL_COUNT
            and len(self.query_decisions) == PANEL_FEATURE_QUERY_PANEL_COUNT
            and (
                self.persist_reload_callback_invocations,
                self.query_callback_invocations,
            )
            == (1, 1)
        )
        gap_shape = (
            type(self.support_gap) is PanelFeatureTaskSupportGap
            and self.predicate_pair is None
            and self.frozen_pair_canonical_base64 is None
            and not self.query_png_base64_by_side
            and not self.query_observations
            and not self.query_decisions
            and (
                self.persist_reload_callback_invocations,
                self.query_callback_invocations,
            )
            == (0, 0)
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
                "task_plan",
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
                "predicate_pair",
                "frozen_pair_canonical_base64",
                "sealed_query_panel_ids",
                "query_png_base64_by_side",
                "query_observations",
                "query_decisions",
                "persist_reload_callback_invocations",
                "query_callback_invocations",
                "query_callback_called_only_after_exact_frozen_pair_reload",
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
            or raw["query_callback_called_only_after_exact_frozen_pair_reload"]
            is not True
            or raw["exact_png_bytes_archived_for_cold_replay"] is not True
            or raw["cold_replay_callback_invocations"] != 0
            or raw["cold_replay_model_calls"] != 0
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["support_png_base64_by_panel_id"], Mapping)
            or not isinstance(raw["query_png_base64_by_side"], Mapping)
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
                if raw["predicate_pair"] is None
                else FrozenEngineeringFeaturePredicatePair.from_data(
                    raw["predicate_pair"]
                )
            ),
            raw["frozen_pair_canonical_base64"],
            tuple((side, query_encoded[side]) for side in _SIDES if side in query_encoded),
            tuple(
                PanelFeatureObservationSet.from_data(item)
                for item in raw["query_observations"]
            ),
            tuple(
                EngineeringQueryDecision.from_data(item)
                for item in raw["query_decisions"]
            ),
            raw["persist_reload_callback_invocations"],
            raw["query_callback_invocations"],
            raw["record_digest"],
        )
        if raw["archive_address"] != result.archive_address or result.to_data() != dict(raw):
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


PersistAndReload = Callable[[bytes], bytes]
QueryCallback = Callable[
    [FrozenEngineeringFeaturePredicatePair],
    Mapping[str, tuple[bytes, PanelFeatureObservationSet]],
]
SupportProposerCallback = Callable[
    [tuple[bytes, ...], str], PanelFeatureProposerResult
]
SupportObserverCallback = Callable[
    [str, bytes, tuple[PanelFeatureSpec, ...]], PanelFeatureObservationSet
]


def _reload_frozen_pair(
    pair: FrozenEngineeringFeaturePredicatePair,
    persist_and_reload: PersistAndReload,
) -> tuple[FrozenEngineeringFeaturePredicatePair, bytes]:
    frozen = canonical_json(pair.to_data())
    reloaded = persist_and_reload(frozen)
    if type(reloaded) is not bytes or reloaded != frozen:
        raise PanelFeatureTaskRunnerError("persisted frozen-pair bytes changed on reload")
    try:
        raw = json.loads(reloaded.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PanelFeatureTaskRunnerError("reloaded frozen pair is not strict JSON") from exc
    if type(raw) is not dict or canonical_json(raw) != reloaded:
        raise PanelFeatureTaskRunnerError("reloaded frozen pair is not canonical JSON")
    restored = FrozenEngineeringFeaturePredicatePair.from_data(raw)
    if restored != pair or restored.pair_digest != pair.pair_digest:
        raise PanelFeatureTaskRunnerError("reloaded frozen pair differs")
    return restored, frozen


def _query_rows(
    callback_result: object,
    task: ObjectBongardTaskPlan,
) -> tuple[tuple[bytes, PanelFeatureObservationSet], ...]:
    query_ids = (task.side_0_query_panel_id, task.side_1_query_panel_id)
    if (
        not isinstance(callback_result, Mapping)
        or any(type(key) is not str for key in callback_result)
        or set(callback_result) != set(query_ids)
    ):
        raise PanelFeatureTaskRunnerError(
            "query callback must return rows for the exact two sealed query panel IDs"
        )
    result: list[tuple[bytes, PanelFeatureObservationSet]] = []
    for side, panel_id in zip(_SIDES, query_ids, strict=True):
        row = callback_result[panel_id]
        if type(row) is not tuple or len(row) != 2:
            raise PanelFeatureTaskRunnerError("query callback row differs")
        result.append((_png(row[0], f"{side} query panel"), _canonical_observation(row[1])))
    return tuple(result)


def run_panel_feature_task(
    task_plan: ObjectBongardTaskPlan,
    support_pngs: Sequence[bytes],
    proposer_result: PanelFeatureProposerResult,
    support_observations: Sequence[PanelFeatureObservationSet],
    *,
    persist_and_reload: PersistAndReload | None,
    query_callback: QueryCallback | None,
) -> PanelFeatureTaskArchive:
    """Build support predicates, freeze/reload, then release both sealed queries.

    ``support_pngs`` and ``support_observations`` are ordered exactly as the
    task plan's six side-0 supports followed by its six side-1 supports.  A
    support gap returns before either callback is inspected or invoked.
    """

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
    common = {
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
    }
    if not side0_space.survivor_formulas or not side1_space.survivor_formulas:
        gap = PanelFeatureTaskSupportGap.create(side0_space, side1_space)
        archive = _make_archive(
            **common,
            status=PanelFeatureTaskRunStatus.SUPPORT_GAP,
            support_gap=gap,
            predicate_pair=None,
            frozen_pair_canonical_base64=None,
            query_png_base64_by_side=(),
            query_observations=(),
            query_decisions=(),
            persist_reload_callback_invocations=0,
            query_callback_invocations=0,
        )
        return cold_replay_panel_feature_task(
            archive, expected_archive_address=archive.archive_address
        )
    if not callable(persist_and_reload) or not callable(query_callback):
        raise TypeError("surviving support predicates require freeze and query callbacks")
    pair = FrozenEngineeringFeaturePredicatePair.create(side0_space, side1_space)
    reloaded_pair, frozen_bytes = _reload_frozen_pair(pair, persist_and_reload)
    # This is the first point at which query pixels may be obtained.
    rows = _query_rows(query_callback(reloaded_pair), task)
    query_pngs = tuple(item[0] for item in rows)
    query_observations, _, _ = _verify_observation_batch(
        tuple(item[1] for item in rows),
        query_pngs,
        vocabulary,
        expected_contract_digest=contract,
        expected_protocol_digest=protocol,
        label="query",
    )
    decisions: list[EngineeringQueryDecision] = []
    for observation in query_observations:
        query_table = _table_for_observations(vocabulary, (observation,))
        decisions.append(
            EngineeringQueryDecision.create(
                reloaded_pair, query_table, observation.panel_digest
            )
        )
    archive = _make_archive(
        **common,
        status=PanelFeatureTaskRunStatus.COMPLETE,
        support_gap=None,
        predicate_pair=reloaded_pair,
        frozen_pair_canonical_base64=base64.b64encode(frozen_bytes).decode("ascii"),
        query_png_base64_by_side=_encode_png_rows(_SIDES, query_pngs),
        query_observations=query_observations,
        query_decisions=tuple(decisions),
        persist_reload_callback_invocations=1,
        query_callback_invocations=1,
    )
    return cold_replay_panel_feature_task(
        archive, expected_archive_address=archive.archive_address
    )


def run_panel_feature_task_with_support_callbacks(
    task_plan: ObjectBongardTaskPlan,
    support_pngs: Sequence[bytes],
    *,
    proposer_callback: SupportProposerCallback,
    observation_callback: SupportObserverCallback,
    persist_and_reload: PersistAndReload | None,
    query_callback: QueryCallback | None,
) -> PanelFeatureTaskArchive:
    """Live support adapter with neutral names and no early query access.

    The proposer receives ``(twelve_pngs, raw_task_record_digest)``.  After its
    provenance is checked, the observer is called twelve times with only the
    neutral presentation name, exact PNG bytes, and globally deduplicated spec
    tuple.  Receipts remain inside the returned canonical proposer/observation
    artifacts.  The query callback is passed onward to :func:`run_panel_feature_task`
    and therefore cannot run before exact frozen-pair persistence and reload.
    """

    task = _task(task_plan)
    pngs = _support_pngs(support_pngs)
    if not callable(proposer_callback) or not callable(observation_callback):
        raise TypeError("support proposer and observation callbacks must be callable")
    proposer = proposer_callback(pngs, task.record_digest.split(":", 1)[1])
    canonical_proposer, vocabulary = _derive_vocabulary_and_verify_provenance(
        task, pngs, proposer
    )
    observations = tuple(
        observation_callback(name, panel, vocabulary.specs)
        for name, panel in zip(PANEL_FEATURE_PRESENTATION_NAMES, pngs, strict=True)
    )
    return run_panel_feature_task(
        task,
        pngs,
        canonical_proposer,
        observations,
        persist_and_reload=persist_and_reload,
        query_callback=query_callback,
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
            or restored.predicate_pair is not None
            or restored.frozen_pair_canonical_base64 is not None
            or restored.query_png_base64_by_side
            or restored.query_observations
            or restored.query_decisions
            or (
                restored.persist_reload_callback_invocations,
                restored.query_callback_invocations,
            )
            != (0, 0)
        ):
            raise PanelFeatureTaskRunnerError("support-gap cold replay differs")
        return restored
    expected_pair = FrozenEngineeringFeaturePredicatePair.create(
        side0_space, side1_space
    )
    if (
        restored.status is not PanelFeatureTaskRunStatus.COMPLETE
        or restored.support_gap is not None
        or restored.predicate_pair != expected_pair
        or type(restored.frozen_pair_canonical_base64) is not str
        or (
            restored.persist_reload_callback_invocations,
            restored.query_callback_invocations,
        )
        != (1, 1)
    ):
        raise PanelFeatureTaskRunnerError("complete cold-replay phase differs")
    try:
        frozen = base64.b64decode(
            restored.frozen_pair_canonical_base64, validate=True
        )
        raw_pair = json.loads(frozen.decode("utf-8", errors="strict"))
    except (ValueError, TypeError, UnicodeError, json.JSONDecodeError) as exc:
        raise PanelFeatureTaskRunnerError("archived frozen-pair bytes differ") from exc
    if (
        type(raw_pair) is not dict
        or canonical_json(raw_pair) != frozen
        or FrozenEngineeringFeaturePredicatePair.from_data(raw_pair) != expected_pair
        or frozen != canonical_json(expected_pair.to_data())
    ):
        raise PanelFeatureTaskRunnerError("frozen-pair cold replay differs")
    query_pngs = _decode_png_rows(
        restored.query_png_base64_by_side, _SIDES, label="query"
    )
    query_observations, _, _ = _verify_observation_batch(
        restored.query_observations,
        query_pngs,
        vocabulary,
        expected_contract_digest=contract,
        expected_protocol_digest=protocol,
        label="query",
    )
    decisions = tuple(
        EngineeringQueryDecision.create(
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
    "PANEL_FEATURE_QUERY_PANEL_COUNT",
    "PANEL_FEATURE_SUPPORT_PANEL_COUNT",
    "PANEL_FEATURE_TASK_RUNNER_ID",
    "PanelFeatureTaskArchive",
    "PanelFeatureTaskRunStatus",
    "PanelFeatureTaskRunnerError",
    "PanelFeatureTaskSupportGap",
    "PersistAndReload",
    "QueryCallback",
    "SupportObserverCallback",
    "SupportProposerCallback",
    "cold_replay_panel_feature_task",
    "engineering_disposition_from_observation",
    "run_panel_feature_task",
    "run_panel_feature_task_with_support_callbacks",
)
