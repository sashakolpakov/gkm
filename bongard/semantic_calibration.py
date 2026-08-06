"""Prospective, metadata-only calibration for dynamic soft visual claims.

The calibration plan is frozen after a :class:`SoftScorerProtocol` exists and
before any development side labels are joined.  It commits only non-test split
membership, opaque task/panel identities, exact panel byte digests, and
dependence clusters.  It contains no affirmative label or panel polarity.

A label can enter the chain only through :func:`join_calibration_label`, which
first validates a sealed :class:`BlindSoftScoreTransportArtifact` and then
creates a receipt whose parents are the exact score artifact, score record,
and scorer receipt.  This is a content-addressed causal ordering guarantee,
not a wall-clock timestamp or signature.  An outer runner remains responsible
for authenticating the split and label-reveal receipts it supplies.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
import re
from typing import Any, Mapping, Sequence

from bongard.artifacts import canonical_digest
from bongard.blind_soft_transport import BlindSoftScoreTransportArtifact
from bongard.corpus import SplitIndex
from bongard.soft_predicates import (
    SoftFamilyDevelopmentUnit,
    SoftPredicateIntegrityError,
    SoftScorerFamily,
    SoftScorerProtocol,
)
from bongard.transport import CodexReceipt


SEMANTIC_CALIBRATION_PLAN_SCHEMA = "gkm.bongard-semantic-calibration-plan.v2"
CALIBRATION_PANEL_SELECTION_SCHEMA = (
    "gkm.bongard-semantic-calibration-panel-selection.v1"
)
CALIBRATION_LABEL_JOIN_RECEIPT_SCHEMA = (
    "gkm.bongard-semantic-calibration-label-join-receipt.v1"
)
SEMANTIC_CALIBRATION_MEASUREMENT_SCHEMA = (
    "gkm.bongard-semantic-calibration-measurement.v1"
)
SEMANTIC_CALIBRATION_ARTIFACT_SCHEMA = (
    "gkm.bongard-semantic-calibration-artifact.v1"
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\Z")
_ALLOWED_DEVELOPMENT_SPLITS = frozenset({"train", "val"})


class SemanticCalibrationError(ValueError):
    """A prospective calibration commitment or causal join is invalid."""


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise SemanticCalibrationError(f"{name} must be a lowercase sha256")
    return value


def _address(value: object, name: str) -> str:
    if not isinstance(value, str) or _ADDRESS.fullmatch(value) is None:
        raise SemanticCalibrationError(
            f"{name} must be a sha256: content address"
        )
    return value


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise SemanticCalibrationError(f"invalid {name} {value!r}")
    return value


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise SemanticCalibrationError(f"{name} must be an object")
    return value


def _fields(
    value: Mapping[str, Any], expected: set[str], name: str
) -> Mapping[str, Any]:
    actual = set(value)
    if actual != expected:
        raise SemanticCalibrationError(
            f"{name} fields differ: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return value


def _list(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise SemanticCalibrationError(f"{name} must be a list")
    return value


def _expected_digest(actual: str, expected: str | None, name: str) -> None:
    if expected is not None and actual != _digest(expected, f"expected {name}"):
        raise SoftPredicateIntegrityError(f"{name} digest mismatch")


@dataclass(frozen=True, order=True)
class CalibrationPanelSelection:
    """One label-free, exact development-panel commitment."""

    observation_id: str
    task_id: str
    panel_id: str
    panel_digest: str
    split: str
    dependence_cluster_id: str

    def __post_init__(self) -> None:
        _identifier(self.observation_id, "observation_id")
        _identifier(self.task_id, "task_id")
        _identifier(self.panel_id, "neutral panel_id")
        _digest(self.panel_digest, "panel_digest")
        if self.split not in _ALLOWED_DEVELOPMENT_SPLITS:
            raise SemanticCalibrationError(
                "calibration selection split must be train or val, never test"
            )
        _identifier(self.dependence_cluster_id, "dependence_cluster_id")

    def to_data(self) -> dict[str, object]:
        # There is deliberately no side, polarity, label, or source path.
        return {
            "schema": CALIBRATION_PANEL_SELECTION_SCHEMA,
            "observation_id": self.observation_id,
            "task_id": self.task_id,
            "panel_id": self.panel_id,
            "panel_digest": self.panel_digest,
            "split": self.split,
            "dependence_cluster_id": self.dependence_cluster_id,
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self.to_data())

    @classmethod
    def from_data(
        cls, value: Mapping[str, Any]
    ) -> "CalibrationPanelSelection":
        data = _fields(
            _mapping(value, "calibration panel selection"),
            {
                "schema",
                "observation_id",
                "task_id",
                "panel_id",
                "panel_digest",
                "split",
                "dependence_cluster_id",
            },
            "calibration panel selection",
        )
        if data["schema"] != CALIBRATION_PANEL_SELECTION_SCHEMA:
            raise SemanticCalibrationError(
                "unsupported calibration-panel-selection schema"
            )
        result = cls(
            observation_id=data["observation_id"],
            task_id=data["task_id"],
            panel_id=data["panel_id"],
            panel_digest=data["panel_digest"],
            split=data["split"],
            dependence_cluster_id=data["dependence_cluster_id"],
        )
        if result.to_data() != dict(data):
            raise SoftPredicateIntegrityError(
                "calibration panel selection is not canonical"
            )
        return result


@dataclass(frozen=True)
class SemanticCalibrationPlan:
    """Label-free development sampling plan bound to one scorer protocol.

    V2 deliberately admits at most one panel per task and distinguishes the
    trusted full-corpus manifest from the selected development submanifest.
    This makes the calibration estimand task-weighted and rejects
    pseudo-replication rather than trying to repair it after scoring.
    Dependence clusters still cover shared proposal/rubric/annotation batches
    across distinct tasks.
    """

    protocol_digest: str
    corpus_manifest_digest: str
    development_manifest_digest: str
    split_source_digest: str
    split_manifest_digest: str
    label_reveal_protocol_digest: str
    selections: tuple[CalibrationPanelSelection, ...]
    _sealed_digest: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _digest(self.protocol_digest, "protocol_digest")
        _address(self.corpus_manifest_digest, "corpus_manifest_digest")
        _address(
            self.development_manifest_digest,
            "development_manifest_digest",
        )
        _address(self.split_source_digest, "split_source_digest")
        _digest(self.split_manifest_digest, "split_manifest_digest")
        _digest(
            self.label_reveal_protocol_digest,
            "label_reveal_protocol_digest",
        )
        if not isinstance(self.selections, tuple) or not self.selections:
            raise SemanticCalibrationError(
                "calibration plan needs a non-empty immutable selection"
            )
        if any(
            not isinstance(item, CalibrationPanelSelection)
            for item in self.selections
        ):
            raise TypeError("calibration plan contains a malformed selection")
        if tuple(
            sorted(self.selections, key=lambda item: item.observation_id)
        ) != self.selections:
            raise SemanticCalibrationError(
                "calibration selections must be sorted by observation_id"
            )
        for name in (
            "observation_id",
            "task_id",
            "panel_id",
            "panel_digest",
        ):
            values = tuple(getattr(item, name) for item in self.selections)
            if len(values) != len(set(values)):
                raise SemanticCalibrationError(
                    f"calibration plan repeats {name}"
                )
        object.__setattr__(self, "_sealed_digest", self.digest)

    @classmethod
    def create(
        cls,
        protocol: SoftScorerProtocol,
        split_index: SplitIndex,
        selections: Sequence[CalibrationPanelSelection],
        *,
        corpus_manifest_digest: str,
        development_manifest_digest: str,
        label_reveal_protocol_digest: str,
    ) -> "SemanticCalibrationPlan":
        """Validate official split membership and freeze a label-free plan."""

        if not isinstance(protocol, SoftScorerProtocol):
            raise TypeError("protocol must be a SoftScorerProtocol")
        protocol.assert_untampered()
        if not isinstance(split_index, SplitIndex):
            raise TypeError("split_index must be a SplitIndex")
        if not split_index.groups or split_index.source_digest is None:
            raise SemanticCalibrationError(
                "calibration requires an authenticated non-empty split index"
            )
        _address(split_index.source_digest, "split source_digest")
        if isinstance(selections, (str, bytes)) or not isinstance(
            selections, Sequence
        ):
            raise TypeError("selections must be a sequence")
        checked: list[CalibrationPanelSelection] = []
        for item in selections:
            if not isinstance(item, CalibrationPanelSelection):
                raise TypeError(
                    "selections must contain CalibrationPanelSelection values"
                )
            assignment = split_index.assignment(item.task_id)
            if assignment.split == "test":
                raise SemanticCalibrationError(
                    f"official test task {item.task_id!r} cannot enter calibration"
                )
            if assignment.split not in _ALLOWED_DEVELOPMENT_SPLITS:
                raise SemanticCalibrationError(
                    f"task {item.task_id!r} lacks train/val split membership"
                )
            if item.split != assignment.split:
                raise SemanticCalibrationError(
                    f"selection split for {item.task_id!r} differs from split index"
                )
            checked.append(item)
        return cls(
            protocol_digest=protocol.digest(),
            corpus_manifest_digest=_address(
                corpus_manifest_digest, "corpus_manifest_digest"
            ),
            development_manifest_digest=_address(
                development_manifest_digest,
                "development_manifest_digest",
            ),
            split_source_digest=split_index.source_digest,
            split_manifest_digest=canonical_digest(
                split_index.to_manifest_dict()
            ),
            label_reveal_protocol_digest=_digest(
                label_reveal_protocol_digest,
                "label_reveal_protocol_digest",
            ),
            selections=tuple(
                sorted(checked, key=lambda item: item.observation_id)
            ),
        )

    def selection(self, observation_id: str) -> CalibrationPanelSelection:
        _identifier(observation_id, "observation_id")
        for item in self.selections:
            if item.observation_id == observation_id:
                return item
        raise KeyError(f"observation is absent from calibration plan: {observation_id}")

    def content_data(self) -> dict[str, object]:
        return {
            "schema": SEMANTIC_CALIBRATION_PLAN_SCHEMA,
            "protocol_digest": self.protocol_digest,
            "corpus_manifest_digest": self.corpus_manifest_digest,
            "development_manifest_digest": self.development_manifest_digest,
            "split_source_digest": self.split_source_digest,
            "split_manifest_digest": self.split_manifest_digest,
            "label_reveal_protocol_digest": self.label_reveal_protocol_digest,
            "selections": [item.to_data() for item in self.selections],
            "label_state": "withheld",
            "allowed_splits": sorted(_ALLOWED_DEVELOPMENT_SPLITS),
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self.content_data())

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "plan_digest": self.digest}

    def assert_untampered(self) -> None:
        if self.digest != self._sealed_digest:
            raise SoftPredicateIntegrityError(
                "semantic calibration plan changed after sealing"
            )

    @classmethod
    def from_data(
        cls,
        value: Mapping[str, Any],
        *,
        expected_digest: str | None = None,
    ) -> "SemanticCalibrationPlan":
        data = _fields(
            _mapping(value, "semantic calibration plan"),
            {
                "schema",
                "protocol_digest",
                "corpus_manifest_digest",
                "development_manifest_digest",
                "split_source_digest",
                "split_manifest_digest",
                "label_reveal_protocol_digest",
                "selections",
                "label_state",
                "allowed_splits",
                "plan_digest",
            },
            "semantic calibration plan",
        )
        if data["schema"] != SEMANTIC_CALIBRATION_PLAN_SCHEMA:
            raise SemanticCalibrationError(
                "unsupported semantic-calibration-plan schema"
            )
        if data["label_state"] != "withheld":
            raise SemanticCalibrationError(
                "calibration plan contains a revealed label state"
            )
        if data["allowed_splits"] != sorted(_ALLOWED_DEVELOPMENT_SPLITS):
            raise SemanticCalibrationError(
                "calibration plan allowed splits changed"
            )
        raw_selections = _list(data["selections"], "plan selections")
        result = cls(
            protocol_digest=data["protocol_digest"],
            corpus_manifest_digest=data["corpus_manifest_digest"],
            development_manifest_digest=data["development_manifest_digest"],
            split_source_digest=data["split_source_digest"],
            split_manifest_digest=data["split_manifest_digest"],
            label_reveal_protocol_digest=data[
                "label_reveal_protocol_digest"
            ],
            selections=tuple(
                CalibrationPanelSelection.from_data(
                    _mapping(item, "plan selection")
                )
                for item in raw_selections
            ),
        )
        archived_digest = _digest(data["plan_digest"], "plan_digest")
        if archived_digest != result.digest:
            raise SoftPredicateIntegrityError(
                "semantic calibration plan digest mismatch"
            )
        _expected_digest(result.digest, expected_digest, "calibration plan")
        if result.to_data() != dict(data):
            raise SoftPredicateIntegrityError(
                "semantic calibration plan is not canonical"
            )
        return result


@dataclass(frozen=True)
class CalibrationLabelJoinReceipt:
    """Content-addressed join whose score parents necessarily already exist."""

    plan_digest: str
    selection_digest: str
    score_artifact_digest: str
    score_record_digest: str
    scorer_receipt_digest: str
    label_reveal_protocol_digest: str
    label_reveal_receipt_digest: str
    affirmative_label: bool

    def __post_init__(self) -> None:
        for name in (
            "plan_digest",
            "selection_digest",
            "score_artifact_digest",
            "score_record_digest",
            "scorer_receipt_digest",
            "label_reveal_protocol_digest",
            "label_reveal_receipt_digest",
        ):
            _digest(getattr(self, name), name)
        if type(self.affirmative_label) is not bool:
            raise TypeError("affirmative_label must be literal bool")

    def content_data(self) -> dict[str, object]:
        return {
            "schema": CALIBRATION_LABEL_JOIN_RECEIPT_SCHEMA,
            "causal_order": "sealed_score_artifact_and_receipt_then_label_join/v1",
            "plan_digest": self.plan_digest,
            "selection_digest": self.selection_digest,
            "score_parents": {
                "artifact_digest": self.score_artifact_digest,
                "record_digest": self.score_record_digest,
                "scorer_receipt_digest": self.scorer_receipt_digest,
            },
            "label_reveal": {
                "protocol_digest": self.label_reveal_protocol_digest,
                "receipt_digest": self.label_reveal_receipt_digest,
                "affirmative_label": self.affirmative_label,
            },
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self.content_data())

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "join_receipt_digest": self.digest}

    @classmethod
    def from_data(
        cls,
        value: Mapping[str, Any],
        *,
        expected_digest: str | None = None,
    ) -> "CalibrationLabelJoinReceipt":
        data = _fields(
            _mapping(value, "calibration label-join receipt"),
            {
                "schema",
                "causal_order",
                "plan_digest",
                "selection_digest",
                "score_parents",
                "label_reveal",
                "join_receipt_digest",
            },
            "calibration label-join receipt",
        )
        if data["schema"] != CALIBRATION_LABEL_JOIN_RECEIPT_SCHEMA:
            raise SemanticCalibrationError(
                "unsupported calibration-label-join schema"
            )
        if data["causal_order"] != (
            "sealed_score_artifact_and_receipt_then_label_join/v1"
        ):
            raise SemanticCalibrationError("label-join causal order changed")
        parents = _fields(
            _mapping(data["score_parents"], "label-join score parents"),
            {"artifact_digest", "record_digest", "scorer_receipt_digest"},
            "label-join score parents",
        )
        reveal = _fields(
            _mapping(data["label_reveal"], "label reveal"),
            {"protocol_digest", "receipt_digest", "affirmative_label"},
            "label reveal",
        )
        result = cls(
            plan_digest=data["plan_digest"],
            selection_digest=data["selection_digest"],
            score_artifact_digest=parents["artifact_digest"],
            score_record_digest=parents["record_digest"],
            scorer_receipt_digest=parents["scorer_receipt_digest"],
            label_reveal_protocol_digest=reveal["protocol_digest"],
            label_reveal_receipt_digest=reveal["receipt_digest"],
            affirmative_label=reveal["affirmative_label"],
        )
        archived_digest = _digest(
            data["join_receipt_digest"], "join_receipt_digest"
        )
        if archived_digest != result.digest:
            raise SoftPredicateIntegrityError(
                "calibration label-join receipt digest mismatch"
            )
        _expected_digest(result.digest, expected_digest, "label-join receipt")
        if result.to_data() != dict(data):
            raise SoftPredicateIntegrityError(
                "calibration label-join receipt is not canonical"
            )
        return result


def _score_bin(score: float, edges: tuple[float, ...]) -> int:
    index = bisect.bisect_right(edges, float(score)) - 1
    return min(index, len(edges) - 2)


def _reject_measurement_overlaps(
    measurements: Sequence["SemanticCalibrationMeasurement"],
) -> None:
    """Reject repeated task, panel, model-call, or receipt identities."""

    identity_extractors: Mapping[str, tuple[str, ...]] = {
        "task_id": tuple(
            item.development_unit.task_id for item in measurements
        ),
        "panel_id": tuple(item.selection.panel_id for item in measurements),
        "panel_digest": tuple(
            item.development_unit.panel_digest for item in measurements
        ),
        "proposer_call_id": tuple(
            item.development_unit.proposer_call_id for item in measurements
        ),
        "scorer_call_id": tuple(
            item.development_unit.scorer_call_id for item in measurements
        ),
        "score_artifact_digest": tuple(
            item.score_artifact_digest for item in measurements
        ),
        "score_record_digest": tuple(
            item.development_unit.score_record_digest for item in measurements
        ),
        "label_reveal_receipt_digest": tuple(
            item.label_reveal_receipt_digest for item in measurements
        ),
    }
    for name, values in identity_extractors.items():
        if len(values) != len(set(values)):
            raise SemanticCalibrationError(
                f"calibration measurements overlap on {name}"
            )


@dataclass(frozen=True)
class SemanticCalibrationMeasurement:
    """One admitted score artifact after a verifier label reveal."""

    plan_digest: str
    selection: CalibrationPanelSelection
    score_artifact_digest: str
    label_reveal_receipt_digest: str
    join_receipt: CalibrationLabelJoinReceipt
    development_unit: SoftFamilyDevelopmentUnit
    _sealed_digest: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _digest(self.plan_digest, "plan_digest")
        if not isinstance(self.selection, CalibrationPanelSelection):
            raise TypeError("selection must be a CalibrationPanelSelection")
        _digest(self.score_artifact_digest, "score_artifact_digest")
        _digest(
            self.label_reveal_receipt_digest,
            "label_reveal_receipt_digest",
        )
        if not isinstance(self.join_receipt, CalibrationLabelJoinReceipt):
            raise TypeError("join_receipt must be CalibrationLabelJoinReceipt")
        if not isinstance(self.development_unit, SoftFamilyDevelopmentUnit):
            raise TypeError("development_unit must be SoftFamilyDevelopmentUnit")
        expected = {
            "plan_digest": self.plan_digest,
            "selection_digest": self.selection.digest,
            "score_artifact_digest": self.score_artifact_digest,
            "score_record_digest": self.development_unit.score_record_digest,
            "label_reveal_receipt_digest": self.label_reveal_receipt_digest,
            "affirmative_label": self.development_unit.affirmative_label,
        }
        for name, value in expected.items():
            if getattr(self.join_receipt, name) != value:
                raise SemanticCalibrationError(
                    f"label-join receipt {name} differs from measurement"
                )
        unit_expected = {
            "observation_id": self.selection.observation_id,
            "task_id": self.selection.task_id,
            "panel_digest": self.selection.panel_digest,
            "dependence_cluster_id": self.selection.dependence_cluster_id,
            "annotation_receipt_digest": self.join_receipt.digest,
        }
        for name, value in unit_expected.items():
            if getattr(self.development_unit, name) != value:
                raise SemanticCalibrationError(
                    f"development unit {name} differs from frozen selection"
                )
        object.__setattr__(self, "_sealed_digest", self.digest)

    def content_data(self) -> dict[str, object]:
        return {
            "schema": SEMANTIC_CALIBRATION_MEASUREMENT_SCHEMA,
            "plan_digest": self.plan_digest,
            "selection": self.selection.to_data(),
            "selection_digest": self.selection.digest,
            "score_artifact_digest": self.score_artifact_digest,
            "label_reveal_receipt_digest": self.label_reveal_receipt_digest,
            "join_receipt": self.join_receipt.to_data(),
            "development_unit": self.development_unit.to_data(),
            "development_unit_digest": self.development_unit.digest(),
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self.content_data())

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "measurement_digest": self.digest}

    def assert_untampered(self) -> None:
        if self.digest != self._sealed_digest:
            raise SoftPredicateIntegrityError(
                "semantic calibration measurement changed after sealing"
            )

    @classmethod
    def from_data(
        cls,
        value: Mapping[str, Any],
        *,
        expected_digest: str | None = None,
    ) -> "SemanticCalibrationMeasurement":
        data = _fields(
            _mapping(value, "semantic calibration measurement"),
            {
                "schema",
                "plan_digest",
                "selection",
                "selection_digest",
                "score_artifact_digest",
                "label_reveal_receipt_digest",
                "join_receipt",
                "development_unit",
                "development_unit_digest",
                "measurement_digest",
            },
            "semantic calibration measurement",
        )
        if data["schema"] != SEMANTIC_CALIBRATION_MEASUREMENT_SCHEMA:
            raise SemanticCalibrationError(
                "unsupported semantic-calibration-measurement schema"
            )
        selection = CalibrationPanelSelection.from_data(
            _mapping(data["selection"], "measurement selection")
        )
        if data["selection_digest"] != selection.digest:
            raise SoftPredicateIntegrityError(
                "measurement selection digest mismatch"
            )
        join_receipt = CalibrationLabelJoinReceipt.from_data(
            _mapping(data["join_receipt"], "measurement join receipt")
        )
        development_unit = SoftFamilyDevelopmentUnit.from_data(
            _mapping(data["development_unit"], "measurement development unit")
        )
        if data["development_unit_digest"] != development_unit.digest():
            raise SoftPredicateIntegrityError(
                "measurement development-unit digest mismatch"
            )
        result = cls(
            plan_digest=data["plan_digest"],
            selection=selection,
            score_artifact_digest=data["score_artifact_digest"],
            label_reveal_receipt_digest=data[
                "label_reveal_receipt_digest"
            ],
            join_receipt=join_receipt,
            development_unit=development_unit,
        )
        archived_digest = _digest(
            data["measurement_digest"], "measurement_digest"
        )
        if archived_digest != result.digest:
            raise SoftPredicateIntegrityError(
                "semantic calibration measurement digest mismatch"
            )
        _expected_digest(result.digest, expected_digest, "calibration measurement")
        if result.to_data() != dict(data):
            raise SoftPredicateIntegrityError(
                "semantic calibration measurement is not canonical"
            )
        return result


def join_calibration_label(
    plan: SemanticCalibrationPlan,
    protocol: SoftScorerProtocol,
    observation_id: str,
    score_artifact: BlindSoftScoreTransportArtifact,
    affirmative_label: bool,
    *,
    label_reveal_receipt_digest: str,
) -> SemanticCalibrationMeasurement:
    """Join a revealed label only after validating the sealed score parents."""

    if not isinstance(plan, SemanticCalibrationPlan):
        raise TypeError("plan must be a SemanticCalibrationPlan")
    plan.assert_untampered()
    if not isinstance(protocol, SoftScorerProtocol):
        raise TypeError("protocol must be a SoftScorerProtocol")
    protocol.assert_untampered()
    if protocol.digest() != plan.protocol_digest:
        raise SemanticCalibrationError(
            "protocol differs from the prospective calibration plan"
        )
    if type(affirmative_label) is not bool:
        raise TypeError("affirmative_label must be literal bool")
    reveal_digest = _digest(
        label_reveal_receipt_digest, "label_reveal_receipt_digest"
    )
    selection = plan.selection(observation_id)
    if not isinstance(score_artifact, BlindSoftScoreTransportArtifact):
        raise TypeError("score_artifact must be BlindSoftScoreTransportArtifact")
    score_artifact.assert_untampered()
    record = score_artifact.record
    if record.outcome != "present" or record.score is None:
        raise SemanticCalibrationError(
            "failed scorer record cannot become a calibration measurement"
        )
    if not isinstance(score_artifact.receipt, CodexReceipt):
        raise SemanticCalibrationError(
            "present calibration score lacks an admitted Codex receipt"
        )
    if (
        score_artifact.protocol_digest != plan.protocol_digest
        or record.scorer_protocol_digest != plan.protocol_digest
    ):
        raise SemanticCalibrationError(
            "score artifact belongs to a different scorer protocol"
        )
    expected_identity = {
        "task_id": selection.task_id,
        "panel_id": selection.panel_id,
        "panel_digest": selection.panel_digest,
    }
    for name, value in expected_identity.items():
        if getattr(record, name) != value:
            raise SemanticCalibrationError(
                f"score record {name} differs from planned panel identity"
            )
    if record.scorer_receipt_digest != score_artifact.receipt.receipt_digest:
        raise SemanticCalibrationError(
            "score record is not bound to the retained scorer receipt"
        )
    artifact_digest = score_artifact.digest
    record_digest = record.digest()
    join_receipt = CalibrationLabelJoinReceipt(
        plan_digest=plan.digest,
        selection_digest=selection.digest,
        score_artifact_digest=artifact_digest,
        score_record_digest=record_digest,
        scorer_receipt_digest=record.scorer_receipt_digest,
        label_reveal_protocol_digest=plan.label_reveal_protocol_digest,
        label_reveal_receipt_digest=reveal_digest,
        affirmative_label=affirmative_label,
    )
    unit = SoftFamilyDevelopmentUnit(
        observation_id=selection.observation_id,
        task_id=selection.task_id,
        panel_digest=selection.panel_digest,
        claim_digest=record.claim_digest,
        scorer_protocol_digest=plan.protocol_digest,
        proposer_call_id=record.proposer_call_id,
        scorer_call_id=record.scorer_call_id,
        dependence_cluster_id=selection.dependence_cluster_id,
        score_record_digest=record_digest,
        annotation_receipt_digest=join_receipt.digest,
        score=float(record.score),
        affirmative_label=affirmative_label,
        score_bin_index=_score_bin(float(record.score), protocol.score_bin_edges),
    )
    return SemanticCalibrationMeasurement(
        plan_digest=plan.digest,
        selection=selection,
        score_artifact_digest=artifact_digest,
        label_reveal_receipt_digest=reveal_digest,
        join_receipt=join_receipt,
        development_unit=unit,
    )


@dataclass(frozen=True)
class SemanticCalibrationArtifact:
    """Exact accepted development manifest and its fitted scorer family."""

    plan: SemanticCalibrationPlan
    protocol: SoftScorerProtocol
    measurements: tuple[SemanticCalibrationMeasurement, ...]
    family: SoftScorerFamily
    _sealed_digest: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.plan, SemanticCalibrationPlan):
            raise TypeError("plan must be SemanticCalibrationPlan")
        self.plan.assert_untampered()
        if not isinstance(self.protocol, SoftScorerProtocol):
            raise TypeError("protocol must be SoftScorerProtocol")
        self.protocol.assert_untampered()
        if self.protocol.digest() != self.plan.protocol_digest:
            raise SemanticCalibrationError(
                "calibration artifact protocol differs from plan"
            )
        if not isinstance(self.measurements, tuple) or any(
            not isinstance(item, SemanticCalibrationMeasurement)
            for item in self.measurements
        ):
            raise TypeError("measurements must be an immutable measurement tuple")
        if tuple(
            sorted(
                self.measurements,
                key=lambda item: item.selection.observation_id,
            )
        ) != self.measurements:
            raise SemanticCalibrationError(
                "measurements must be sorted by observation_id"
            )
        for item in self.measurements:
            item.assert_untampered()
            if item.plan_digest != self.plan.digest:
                raise SemanticCalibrationError(
                    "measurement belongs to a different calibration plan"
                )
            if item.selection != self.plan.selection(
                item.selection.observation_id
            ):
                raise SemanticCalibrationError(
                    "measurement selection differs from calibration plan"
                )
            if (
                item.development_unit.scorer_protocol_digest
                != self.plan.protocol_digest
            ):
                raise SemanticCalibrationError(
                    "measurement belongs to a different scorer protocol"
                )
        selected_ids = tuple(item.observation_id for item in self.plan.selections)
        measured_ids = tuple(
            item.selection.observation_id for item in self.measurements
        )
        if measured_ids != selected_ids:
            raise SemanticCalibrationError(
                "completed measurements differ from the exact planned selection"
            )
        _reject_measurement_overlaps(self.measurements)
        clusters_by_bin: dict[int, set[str]] = {
            index: set()
            for index in range(len(self.protocol.score_bin_edges) - 1)
        }
        for item in self.measurements:
            unit = item.development_unit
            clusters_by_bin.setdefault(unit.score_bin_index, set()).add(
                unit.dependence_cluster_id
            )
        underpopulated = tuple(
            index
            for index, clusters in sorted(clusters_by_bin.items())
            if len(clusters) < self.protocol.minimum_clusters_per_bin
        )
        if underpopulated:
            raise SemanticCalibrationError(
                "calibration score bins are underpopulated: "
                + ", ".join(str(index) for index in underpopulated)
            )
        if not isinstance(self.family, SoftScorerFamily):
            raise TypeError("family must be a SoftScorerFamily")
        self.family.assert_untampered()
        expected_units = tuple(
            item.development_unit for item in self.measurements
        )
        if (
            self.family.protocol_digest != self.plan.protocol_digest
            or self.family.development_units != expected_units
        ):
            raise SemanticCalibrationError(
                "fitted family differs from the exact accepted manifest"
            )
        reproduced = SoftScorerFamily.fit(
            self.protocol,
            expected_units,
            expected_protocol_digest=self.plan.protocol_digest,
        )
        if reproduced.to_data() != self.family.to_data():
            raise SoftPredicateIntegrityError(
                "fitted family does not reproduce accepted measurements"
            )
        object.__setattr__(self, "_sealed_digest", self.digest)

    def content_data(self) -> dict[str, object]:
        return {
            "schema": SEMANTIC_CALIBRATION_ARTIFACT_SCHEMA,
            "plan": self.plan.to_data(),
            "plan_digest": self.plan.digest,
            "protocol": self.protocol.to_data(),
            "protocol_digest": self.plan.protocol_digest,
            "measurements": [item.to_data() for item in self.measurements],
            "accepted_units": [
                item.development_unit.to_data() for item in self.measurements
            ],
            "family": self.family.to_data(),
            "family_digest": self.family.digest(),
            "development_manifest_digest": (
                self.family.development_manifest_digest
            ),
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self.content_data())

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "calibration_artifact_digest": self.digest}

    def assert_untampered(self) -> None:
        self.plan.assert_untampered()
        self.protocol.assert_untampered()
        self.family.assert_untampered()
        for item in self.measurements:
            item.assert_untampered()
        if self.digest != self._sealed_digest:
            raise SoftPredicateIntegrityError(
                "semantic calibration artifact changed after sealing"
            )

    @classmethod
    def from_data(
        cls,
        value: Mapping[str, Any],
        *,
        expected_digest: str | None = None,
    ) -> "SemanticCalibrationArtifact":
        data = _fields(
            _mapping(value, "semantic calibration artifact"),
            {
                "schema",
                "plan",
                "plan_digest",
                "protocol",
                "protocol_digest",
                "measurements",
                "accepted_units",
                "family",
                "family_digest",
                "development_manifest_digest",
                "calibration_artifact_digest",
            },
            "semantic calibration artifact",
        )
        if data["schema"] != SEMANTIC_CALIBRATION_ARTIFACT_SCHEMA:
            raise SemanticCalibrationError(
                "unsupported semantic-calibration-artifact schema"
            )
        plan = SemanticCalibrationPlan.from_data(
            _mapping(data["plan"], "calibration artifact plan")
        )
        if data["plan_digest"] != plan.digest:
            raise SoftPredicateIntegrityError(
                "calibration artifact plan digest mismatch"
            )
        protocol_digest = _digest(
            data["protocol_digest"], "protocol_digest"
        )
        protocol = SoftScorerProtocol.from_data(
            _mapping(data["protocol"], "calibration artifact protocol"),
            expected_digest=protocol_digest,
        )
        measurements = tuple(
            SemanticCalibrationMeasurement.from_data(
                _mapping(item, "calibration artifact measurement")
            )
            for item in _list(data["measurements"], "artifact measurements")
        )
        archived_units = _list(data["accepted_units"], "accepted_units")
        units = tuple(
            SoftFamilyDevelopmentUnit.from_data(
                _mapping(item, "accepted development unit")
            )
            for item in archived_units
        )
        if units != tuple(item.development_unit for item in measurements):
            raise SoftPredicateIntegrityError(
                "accepted units differ from calibration measurements"
            )
        family_digest = _digest(data["family_digest"], "family_digest")
        family = SoftScorerFamily.from_data(
            _mapping(data["family"], "calibration artifact family"),
            expected_digest=family_digest,
        )
        if data["development_manifest_digest"] != (
            family.development_manifest_digest
        ):
            raise SoftPredicateIntegrityError(
                "calibration development-manifest digest mismatch"
            )
        result = cls(
            plan=plan,
            protocol=protocol,
            measurements=measurements,
            family=family,
        )
        archived_digest = _digest(
            data["calibration_artifact_digest"],
            "calibration_artifact_digest",
        )
        if archived_digest != result.digest:
            raise SoftPredicateIntegrityError(
                "semantic calibration artifact digest mismatch"
            )
        _expected_digest(result.digest, expected_digest, "calibration artifact")
        if result.to_data() != dict(data):
            raise SoftPredicateIntegrityError(
                "semantic calibration artifact is not canonical"
            )
        return result


def fit_semantic_calibration(
    plan: SemanticCalibrationPlan,
    protocol: SoftScorerProtocol,
    measurements: Sequence[SemanticCalibrationMeasurement],
) -> SemanticCalibrationArtifact:
    """Validate the exact completed plan, then fit its scorer family."""

    if isinstance(measurements, (str, bytes)) or not isinstance(
        measurements, Sequence
    ):
        raise TypeError("measurements must be a sequence")
    ordered = tuple(
        sorted(measurements, key=lambda item: item.selection.observation_id)
    )
    # Constructing the family first may raise the lower-level sparse-bin error;
    # preflight with the artifact-independent checks needed for a clear public
    # underpopulation error.
    if not isinstance(plan, SemanticCalibrationPlan):
        raise TypeError("plan must be SemanticCalibrationPlan")
    if not isinstance(protocol, SoftScorerProtocol):
        raise TypeError("protocol must be SoftScorerProtocol")
    plan.assert_untampered()
    protocol.assert_untampered()
    if protocol.digest() != plan.protocol_digest:
        raise SemanticCalibrationError("protocol differs from calibration plan")
    if any(
        not isinstance(item, SemanticCalibrationMeasurement) for item in ordered
    ):
        raise TypeError("measurements contains a malformed item")
    selected_ids = tuple(item.observation_id for item in plan.selections)
    measured_ids = tuple(item.selection.observation_id for item in ordered)
    if measured_ids != selected_ids:
        raise SemanticCalibrationError(
            "completed measurements differ from the exact planned selection"
        )
    _reject_measurement_overlaps(ordered)
    clusters_by_bin = {
        index: set() for index in range(len(protocol.score_bin_edges) - 1)
    }
    for item in ordered:
        clusters_by_bin[item.development_unit.score_bin_index].add(
            item.development_unit.dependence_cluster_id
        )
    sparse = tuple(
        index
        for index, clusters in sorted(clusters_by_bin.items())
        if len(clusters) < protocol.minimum_clusters_per_bin
    )
    if sparse:
        raise SemanticCalibrationError(
            "calibration score bins are underpopulated: "
            + ", ".join(str(index) for index in sparse)
        )
    units = tuple(item.development_unit for item in ordered)
    family = SoftScorerFamily.fit(
        protocol,
        units,
        expected_protocol_digest=plan.protocol_digest,
    )
    return SemanticCalibrationArtifact(
        plan=plan,
        protocol=protocol,
        measurements=ordered,
        family=family,
    )


__all__ = [
    "CALIBRATION_LABEL_JOIN_RECEIPT_SCHEMA",
    "CALIBRATION_PANEL_SELECTION_SCHEMA",
    "SEMANTIC_CALIBRATION_ARTIFACT_SCHEMA",
    "SEMANTIC_CALIBRATION_MEASUREMENT_SCHEMA",
    "SEMANTIC_CALIBRATION_PLAN_SCHEMA",
    "CalibrationLabelJoinReceipt",
    "CalibrationPanelSelection",
    "SemanticCalibrationArtifact",
    "SemanticCalibrationError",
    "SemanticCalibrationMeasurement",
    "SemanticCalibrationPlan",
    "fit_semantic_calibration",
    "join_calibration_label",
]
