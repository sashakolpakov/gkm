"""Component-wise observer for one frozen positive prose conjunction.

This is an isolated successor to :mod:`panel_positive_prose_observer`.  A
headless support proposer supplies exactly two affirmative components.  The
vision model scores those components separately on a fixed absolute scale;
Python alone projects the two intervals to dispositions and conjunction
status.  The observer is intentionally raw-panel-only and uncalibrated: it
does not claim to solve crop selection, contextual batching, or latent action
segmentation.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

from bongard import panel_positive_prose_observer as _v1
from bongard import prototype_scene_observer as _scene_runtime
from bongard.canonical import canonical_digest, canonical_json
from bongard.codex_no_tools_preflight import CodexNoToolsAttestation
from bongard.evidence import Disposition
from bongard.object_bongard_turn_journal import (
    ObjectBongardTurnJournalSummary,
    object_bongard_turn_journal_source_digest,
)
from bongard.panel_positive_prose_observer import (
    PositiveProsePanelContext,
    PositiveProseTransportProvenance,
)
from bongard.panel_support_positive_proposer import (
    PositiveConjunctionRubric,
    SupportPositiveProposerArtifact,
    SupportPositiveProposerError,
    panel_support_positive_proposer_source_digest,
    verify_support_positive_proposer_artifact,
)
from bongard.panel_typed_codex_observer import _bind_runtime, _exact_png, _receipt_from_data
from bongard.prototype_scene_observer import (
    PrototypeImageIdentity,
    PrototypeSceneObserverStatus,
)
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID
from bongard.transport import (
    NAMED_IMAGE_INPUT_DIGEST_SCHEMA,
    CloudPolicyCacheSnapshot,
    CodexModelCatalogSnapshot,
    CodexReceipt,
    run_codex_named_images_structured,
    validate_codex_named_image_receipt,
    validate_codex_strict_output_schema,
)


COMPONENTWISE_CUE_SCHEMA = "gkm.bongard-componentwise-positive-prose-cue.v1"
COMPONENTWISE_TERMINAL_SCHEMA = (
    "gkm.bongard-componentwise-positive-proposer-terminal.v1"
)
COMPONENTWISE_REQUEST_SCHEMA = (
    "gkm.bongard-componentwise-positive-prose-panel-request.v1"
)
COMPONENTWISE_OBSERVATION_SCHEMA = (
    "gkm.bongard-componentwise-positive-prose-observation.v1"
)
COMPONENTWISE_ARTIFACT_SCHEMA = (
    "gkm.bongard-componentwise-positive-prose-panel-artifact.v1"
)
COMPONENTWISE_PROTOCOL_ID = (
    "bongard.componentwise-positive-prose-observer/one-panel-absolute-v1"
)
POSITIVE_ORIENTATION = "side0_positive"

COMPONENT_SCORE_ANCHORS: tuple[tuple[int, str], ...] = (
    (0, "The complete panel clearly does not exhibit this component."),
    (
        1,
        "A decisive visible contradiction rules this component out; score 1 is "
        "certified absence, not uncertainty.",
    ),
    (2, "Evidence for this component is ambiguous, mixed, or insufficient."),
    (3, "The complete panel exhibits this component."),
    (4, "The complete panel clearly and completely exhibits this component."),
)

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_CODE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")


class ComponentwisePositiveProseError(ValueError):
    """A component-wise request, artifact, custody record, or replay is invalid."""


class ComponentwiseMatchStatus(str, Enum):
    MATCH = "match"
    NONMATCH = "nonmatch"
    INDETERMINATE = "indeterminate"
    ERROR = "error"


ComponentDisposition = Disposition


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(type(key) is not str for key in value)
        or set(value) != expected
    ):
        raise ComponentwisePositiveProseError(f"{label} fields differ")
    return value


def _digest(value: object, label: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise ComponentwisePositiveProseError(f"{label} must be a raw SHA-256")
    return value


def _address(value: object, label: str) -> str:
    if type(value) is not str or _ADDRESS.fullmatch(value) is None:
        raise ComponentwisePositiveProseError(f"{label} must be a sha256: address")
    return value


def _canonical_payload(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ComponentwisePositiveProseError("model payload must be an object")
    try:
        result = json.loads(canonical_json(dict(value)).decode("utf-8"))
    except Exception as exc:
        raise ComponentwisePositiveProseError(
            "model payload is not canonical JSON"
        ) from exc
    if type(result) is not dict:
        raise ComponentwisePositiveProseError("model payload must be an object")
    return result


def panel_componentwise_positive_prose_observer_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


@dataclass(frozen=True, slots=True)
class ProposerTerminalBinding:
    proposer_artifact_digest: str
    manifest_digest: str
    turn_key: str
    claim_digest: str
    result_digest: str
    outcome_digest: str
    record_digest: str
    binding_digest: str

    def __post_init__(self) -> None:
        _digest(self.proposer_artifact_digest, "proposer artifact digest")
        for label, value in (
            ("manifest digest", self.manifest_digest),
            ("turn key", self.turn_key),
            ("claim digest", self.claim_digest),
            ("result digest", self.result_digest),
            ("outcome digest", self.outcome_digest),
            ("record digest", self.record_digest),
        ):
            _address(value, label)
        _digest(self.binding_digest, "proposer terminal binding digest")
        if self.binding_digest != canonical_digest(self.content_data()):
            raise ComponentwisePositiveProseError(
                "proposer terminal binding digest differs"
            )

    @classmethod
    def from_summary(
        cls,
        summary: ObjectBongardTurnJournalSummary,
        *,
        proposer_artifact_digest: str,
    ) -> "ProposerTerminalBinding":
        if (
            type(summary) is not ObjectBongardTurnJournalSummary
            or summary.terminal_status != "success"
            or any(
                type(value) is not str
                for value in (
                    summary.claim_digest,
                    summary.result_digest,
                    summary.outcome_digest,
                )
            )
        ):
            raise ComponentwisePositiveProseError(
                "external proposer terminal must be a durable success"
            )
        values = {
            "proposer_artifact_digest": _digest(
                proposer_artifact_digest, "proposer artifact digest"
            ),
            "manifest_digest": summary.manifest_digest,
            "turn_key": summary.turn_key,
            "claim_digest": summary.claim_digest,
            "result_digest": summary.result_digest,
            "outcome_digest": summary.outcome_digest,
            "record_digest": summary.record_digest,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values, binding_digest=canonical_digest(provisional.content_data())
        )

    def content_data(self) -> dict[str, object]:
        return {
            "schema": COMPONENTWISE_TERMINAL_SCHEMA,
            "proposer_artifact_digest": self.proposer_artifact_digest,
            "terminal_status": "success",
            "manifest_digest": self.manifest_digest,
            "turn_key": self.turn_key,
            "claim_digest": self.claim_digest,
            "result_digest": self.result_digest,
            "outcome_digest": self.outcome_digest,
            "record_digest": self.record_digest,
            "external_typed_terminal_required": True,
        }

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "binding_digest": self.binding_digest}

    @classmethod
    def from_data(cls, value: object) -> "ProposerTerminalBinding":
        raw = _fields(
            value,
            {
                "schema", "proposer_artifact_digest", "terminal_status",
                "manifest_digest", "turn_key", "claim_digest", "result_digest",
                "outcome_digest", "record_digest",
                "external_typed_terminal_required", "binding_digest",
            },
            "proposer terminal binding",
        )
        if (
            raw["schema"] != COMPONENTWISE_TERMINAL_SCHEMA
            or raw["terminal_status"] != "success"
            or raw["external_typed_terminal_required"] is not True
        ):
            raise ComponentwisePositiveProseError(
                "proposer terminal binding policy differs"
            )
        result = cls(
            raw["proposer_artifact_digest"], raw["manifest_digest"],
            raw["turn_key"], raw["claim_digest"], raw["result_digest"],
            raw["outcome_digest"], raw["record_digest"], raw["binding_digest"],
        )
        if result.to_data() != dict(raw):
            raise ComponentwisePositiveProseError(
                "proposer terminal binding is not canonical"
            )
        return result

    def matches(self, summary: ObjectBongardTurnJournalSummary) -> bool:
        return (
            type(summary) is ObjectBongardTurnJournalSummary
            and summary.terminal_status == "success"
            and (
                summary.manifest_digest,
                summary.turn_key,
                summary.claim_digest,
                summary.result_digest,
                summary.outcome_digest,
                summary.record_digest,
            )
            == (
                self.manifest_digest,
                self.turn_key,
                self.claim_digest,
                self.result_digest,
                self.outcome_digest,
                self.record_digest,
            )
        )


def _restore_admitted_proposer(
    artifact: SupportPositiveProposerArtifact,
    group_a_pngs: Sequence[bytes],
    group_b_pngs: Sequence[bytes],
    *,
    expected_artifact_digest: str,
    proposer_journal_terminal: ObjectBongardTurnJournalSummary,
) -> tuple[SupportPositiveProposerArtifact, ProposerTerminalBinding]:
    if type(artifact) is not SupportPositiveProposerArtifact:
        raise TypeError("component observer needs exact proposer artifact")
    try:
        restored = verify_support_positive_proposer_artifact(
            artifact,
            group_a_pngs,
            group_b_pngs,
            expected_artifact_digest=expected_artifact_digest,
            proposer_journal_terminal=proposer_journal_terminal,
        )
    except SupportPositiveProposerError as exc:
        raise ComponentwisePositiveProseError(
            "source proposer pixels, receipt, or external terminal failed replay"
        ) from exc
    if (
        restored.rubric is None
        or restored.proposal_gap is not None
        or restored.benchmark_sealable is not True
    ):
        raise ComponentwisePositiveProseError(
            "component observer requires an admitted journal-sealed proposer"
        )
    binding = ProposerTerminalBinding.from_summary(
        proposer_journal_terminal,
        proposer_artifact_digest=restored.artifact_digest,
    )
    return restored, binding


@dataclass(frozen=True, slots=True)
class ComponentwisePositiveCue:
    cue_text: str
    component_1: str
    component_2: str
    source_proposer_artifact_digest: str
    source_proposer_request_digest: str
    source_rubric_digest: str
    proposer_terminal: ProposerTerminalBinding
    cue_digest: str

    def __post_init__(self) -> None:
        try:
            rubric = PositiveConjunctionRubric(
                self.cue_text, self.component_1, self.component_2
            )
        except SupportPositiveProposerError as exc:
            raise ComponentwisePositiveProseError(
                "component cue prose differs"
            ) from exc
        for label, value in (
            ("source proposer artifact digest", self.source_proposer_artifact_digest),
            ("source proposer request digest", self.source_proposer_request_digest),
            ("source rubric digest", self.source_rubric_digest),
            ("cue digest", self.cue_digest),
        ):
            _digest(value, label)
        if (
            type(self.proposer_terminal) is not ProposerTerminalBinding
            or rubric.rubric_digest != self.source_rubric_digest
            or self.proposer_terminal.proposer_artifact_digest
            != self.source_proposer_artifact_digest
            or self.cue_digest != canonical_digest(self.content_data())
        ):
            raise ComponentwisePositiveProseError("component cue binding differs")

    @classmethod
    def from_proposer(
        cls,
        artifact: SupportPositiveProposerArtifact,
        terminal: ProposerTerminalBinding,
    ) -> "ComponentwisePositiveCue":
        rubric = artifact.rubric
        if type(rubric) is not PositiveConjunctionRubric:
            raise ComponentwisePositiveProseError("source rubric was not admitted")
        values = {
            "cue_text": rubric.cue_text,
            "component_1": rubric.component_1,
            "component_2": rubric.component_2,
            "source_proposer_artifact_digest": artifact.artifact_digest,
            "source_proposer_request_digest": artifact.request_digest,
            "source_rubric_digest": rubric.rubric_digest,
            "proposer_terminal": terminal,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(**values, cue_digest=canonical_digest(provisional.content_data()))

    def component_digest(self, index: int) -> str:
        if index not in (1, 2):
            raise ComponentwisePositiveProseError("component index differs")
        return canonical_digest(
            {
                "schema": "gkm.bongard-positive-prose-component.v1",
                "cue_digest": self.cue_digest,
                "index": index,
                "text": self.component_1 if index == 1 else self.component_2,
            }
        )

    def content_data(self) -> dict[str, object]:
        return {
            "schema": COMPONENTWISE_CUE_SCHEMA,
            "cue_text": self.cue_text,
            "component_1": self.component_1,
            "component_2": self.component_2,
            "source_kind": "admitted_journal_sealed_support_positive_proposer",
            "source_proposer_artifact_digest": self.source_proposer_artifact_digest,
            "source_proposer_request_digest": self.source_proposer_request_digest,
            "source_rubric_digest": self.source_rubric_digest,
            "proposer_terminal": self.proposer_terminal.to_data(),
            "positive_orientation": POSITIVE_ORIENTATION,
            "prose_is_inert": True,
            "component_count": 2,
            "foil_or_complement_present": False,
            "model_threshold_or_polarity_present": False,
        }

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "cue_digest": self.cue_digest}

    @classmethod
    def from_data(cls, value: object) -> "ComponentwisePositiveCue":
        raw = _fields(
            value,
            {
                "schema", "cue_text", "component_1", "component_2", "source_kind",
                "source_proposer_artifact_digest", "source_proposer_request_digest",
                "source_rubric_digest", "proposer_terminal", "positive_orientation",
                "prose_is_inert", "component_count", "foil_or_complement_present",
                "model_threshold_or_polarity_present", "cue_digest",
            },
            "component cue",
        )
        if (
            raw["schema"] != COMPONENTWISE_CUE_SCHEMA
            or raw["source_kind"]
            != "admitted_journal_sealed_support_positive_proposer"
            or raw["positive_orientation"] != POSITIVE_ORIENTATION
            or raw["prose_is_inert"] is not True
            or raw["component_count"] != 2
            or raw["foil_or_complement_present"] is not False
            or raw["model_threshold_or_polarity_present"] is not False
        ):
            raise ComponentwisePositiveProseError("component cue policy differs")
        result = cls(
            raw["cue_text"], raw["component_1"], raw["component_2"],
            raw["source_proposer_artifact_digest"],
            raw["source_proposer_request_digest"], raw["source_rubric_digest"],
            ProposerTerminalBinding.from_data(raw["proposer_terminal"]),
            raw["cue_digest"],
        )
        if result.to_data() != dict(raw):
            raise ComponentwisePositiveProseError("component cue is not canonical")
        return result


def _request_content(value: "ComponentwisePositiveProsePanelRequest") -> dict[str, object]:
    return {
        "schema": COMPONENTWISE_REQUEST_SCHEMA,
        "context": value.context.to_data(),
        "cue": value.cue.to_data(),
        "positive_orientation": POSITIVE_ORIENTATION,
        "model_visible_image_names": ["panel.png"],
        "model_returns_two_absolute_intervals_only": True,
        "raw_panel_only": True,
        "candidate_independent_transformed_view_supported": False,
        "crop_or_context_adapter_present": False,
        "observer_calibrated": False,
    }


@dataclass(frozen=True, slots=True)
class ComponentwisePositiveProsePanelRequest:
    context: PositiveProsePanelContext
    cue: ComponentwisePositiveCue
    request_digest: str

    def __post_init__(self) -> None:
        if type(self.context) is not PositiveProsePanelContext:
            raise TypeError("component request needs exact panel context")
        if type(self.cue) is not ComponentwisePositiveCue:
            raise TypeError("component request needs exact cue")
        _digest(self.request_digest, "component request digest")
        if self.request_digest != canonical_digest(_request_content(self)):
            raise ComponentwisePositiveProseError("component request digest differs")

    @classmethod
    def build_from_proposer(
        cls,
        context: PositiveProsePanelContext,
        proposer_artifact: SupportPositiveProposerArtifact,
        group_a_pngs: Sequence[bytes],
        group_b_pngs: Sequence[bytes],
        *,
        expected_artifact_digest: str,
        proposer_journal_terminal: ObjectBongardTurnJournalSummary,
    ) -> "ComponentwisePositiveProsePanelRequest":
        if type(context) is not PositiveProsePanelContext:
            raise TypeError("component request needs exact panel context")
        proposer, terminal = _restore_admitted_proposer(
            proposer_artifact,
            group_a_pngs,
            group_b_pngs,
            expected_artifact_digest=expected_artifact_digest,
            proposer_journal_terminal=proposer_journal_terminal,
        )
        if proposer.runtime != context.runtime:
            raise ComponentwisePositiveProseError(
                "proposer and component observer runtimes differ"
            )
        cue = ComponentwisePositiveCue.from_proposer(proposer, terminal)
        provisional = object.__new__(cls)
        object.__setattr__(provisional, "context", context)
        object.__setattr__(provisional, "cue", cue)
        return cls(context, cue, canonical_digest(_request_content(provisional)))

    def to_data(self) -> dict[str, object]:
        return {**_request_content(self), "request_digest": self.request_digest}

    @classmethod
    def from_data(cls, value: object) -> "ComponentwisePositiveProsePanelRequest":
        raw = _fields(
            value,
            {
                "schema", "context", "cue", "positive_orientation",
                "model_visible_image_names", "model_returns_two_absolute_intervals_only",
                "raw_panel_only", "candidate_independent_transformed_view_supported",
                "crop_or_context_adapter_present", "observer_calibrated",
                "request_digest",
            },
            "component request",
        )
        if (
            raw["schema"] != COMPONENTWISE_REQUEST_SCHEMA
            or raw["positive_orientation"] != POSITIVE_ORIENTATION
            or raw["model_visible_image_names"] != ["panel.png"]
            or raw["model_returns_two_absolute_intervals_only"] is not True
            or raw["raw_panel_only"] is not True
            or raw["candidate_independent_transformed_view_supported"] is not False
            or raw["crop_or_context_adapter_present"] is not False
            or raw["observer_calibrated"] is not False
        ):
            raise ComponentwisePositiveProseError("component request policy differs")
        result = cls(
            PositiveProsePanelContext.from_data(raw["context"]),
            ComponentwisePositiveCue.from_data(raw["cue"]),
            raw["request_digest"],
        )
        if result.to_data() != dict(raw):
            raise ComponentwisePositiveProseError("component request is not canonical")
        return result


@dataclass(frozen=True, order=True, slots=True)
class ComponentScoreInterval:
    lower: int
    upper: int

    def __post_init__(self) -> None:
        if (
            type(self.lower) is not int
            or type(self.upper) is not int
            or not 0 <= self.lower <= self.upper <= 4
        ):
            raise ComponentwisePositiveProseError(
                "component score interval must lie in 0..4"
            )

    def to_data(self) -> dict[str, int]:
        return {"lower": self.lower, "upper": self.upper}

    @classmethod
    def from_data(cls, value: object) -> "ComponentScoreInterval":
        raw = _fields(value, {"lower", "upper"}, "component score interval")
        return cls(raw["lower"], raw["upper"])


def classify_component_interval(interval: ComponentScoreInterval) -> ComponentDisposition:
    if type(interval) is not ComponentScoreInterval:
        raise TypeError("component classifier needs exact score interval")
    if interval.lower >= 3:
        return ComponentDisposition.PRESENT
    if interval.upper <= 1:
        return ComponentDisposition.CERTIFIED_ABSENT
    return ComponentDisposition.INDETERMINATE


def combine_component_dispositions(
    component_1: ComponentDisposition,
    component_2: ComponentDisposition,
) -> tuple[ComponentDisposition, ComponentwiseMatchStatus]:
    if not isinstance(component_1, ComponentDisposition) or not isinstance(
        component_2, ComponentDisposition
    ):
        raise TypeError("conjunction projection needs exact dispositions")
    if ComponentDisposition.ERROR in (component_1, component_2):
        return ComponentDisposition.ERROR, ComponentwiseMatchStatus.ERROR
    if ComponentDisposition.CERTIFIED_ABSENT in (component_1, component_2):
        return ComponentDisposition.CERTIFIED_ABSENT, ComponentwiseMatchStatus.NONMATCH
    if component_1 is component_2 is ComponentDisposition.PRESENT:
        return ComponentDisposition.PRESENT, ComponentwiseMatchStatus.MATCH
    return ComponentDisposition.INDETERMINATE, ComponentwiseMatchStatus.INDETERMINATE


def componentwise_scale_digest() -> str:
    return canonical_digest(
        {
            "schema": "gkm.bongard-componentwise-positive-prose-scale.v1",
            "anchors": [list(item) for item in COMPONENT_SCORE_ANCHORS],
            "interval_semantics": "inclusive-narrowest-honest-range",
            "present_when_lower_at_least": 3,
            "certified_absent_when_upper_at_most": 1,
            "score_1_is_decisive_visible_contradiction": True,
            "python_conjunction_precedence": [
                "error", "certified_absent", "both_present", "indeterminate"
            ],
            "model_selects_threshold_or_polarity": False,
        }
    )


def componentwise_positive_prose_output_schema(
    request: ComponentwisePositiveProsePanelRequest | None = None,
) -> dict[str, object]:
    if request is not None and type(request) is not ComponentwisePositiveProsePanelRequest:
        raise TypeError("component output schema request has wrong type")
    score = {"type": "integer", "enum": [0, 1, 2, 3, 4]}
    names = (
        "component_1_lower", "component_1_upper",
        "component_2_lower", "component_2_upper",
    )
    schema: dict[str, object] = {
        "type": "object",
        "properties": {name: dict(score) for name in names},
        "required": list(names),
        "additionalProperties": False,
    }
    validate_codex_strict_output_schema(schema)
    return schema


def componentwise_positive_prose_prompt(
    request: ComponentwisePositiveProsePanelRequest,
) -> str:
    if type(request) is not ComponentwisePositiveProsePanelRequest:
        raise TypeError("component prompt request has wrong type")
    anchors = "\n".join(
        f"{level}: {meaning}" for level, meaning in COMPONENT_SCORE_ANCHORS
    )
    return (
        "Inspect exactly one complete drawing named panel.png. Score each of the "
        "two frozen affirmative components independently. Do not compensate for "
        "one failed component with evidence for the other. Do not invent another "
        "description, comparison, threshold, or polarity. When a component refers "
        "to one figure, its claimed parts must belong to one coherent figure.\n\n"
        f"FULL POSITIVE CONJUNCTION (context only)\n{request.cue.cue_text}\n\n"
        f"COMPONENT 1\n{request.cue.component_1}\n\n"
        f"COMPONENT 2\n{request.cue.component_2}\n\n"
        "Use this fixed scale separately for each component:\n"
        f"{anchors}\n\n"
        "For each component return the narrowest honest inclusive lower and upper "
        "scores. Score 1 requires a decisive visible contradiction. Score 2 is "
        "required for ambiguity or insufficient evidence."
    )


def componentwise_protocol_digest() -> str:
    return canonical_digest(
        {
            "schema": "gkm.bongard-componentwise-positive-prose-protocol.v1",
            "protocol_id": COMPONENTWISE_PROTOCOL_ID,
            "source_digest": panel_componentwise_positive_prose_observer_source_digest(),
            "v1_custody_dependency_source_digest": (
                _v1.panel_positive_prose_observer_source_digest()
            ),
            "proposer_source_digest": panel_support_positive_proposer_source_digest(),
            "runtime_source_digest": _scene_runtime.prototype_scene_observer_source_digest(),
            "transport_source_digest": _scene_runtime.prototype_scene_transport_source_digest(),
            "journal_source_digest": object_bongard_turn_journal_source_digest(),
            "scale_digest": componentwise_scale_digest(),
            "python_predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
            "lean_present": False,
            "raw_panel_only": True,
            "observer_calibrated": False,
        }
    )


def _observation_content(
    value: "ComponentwisePositiveProseObservation",
) -> dict[str, object]:
    return {
        "schema": COMPONENTWISE_OBSERVATION_SCHEMA,
        "cue_digest": value.cue_digest,
        "component_1_digest": value.component_1_digest,
        "component_2_digest": value.component_2_digest,
        "component_1_interval": (
            None if value.component_1_interval is None else value.component_1_interval.to_data()
        ),
        "component_2_interval": (
            None if value.component_2_interval is None else value.component_2_interval.to_data()
        ),
        "component_1_disposition": value.component_1_disposition.value,
        "component_2_disposition": value.component_2_disposition.value,
        "conjunction_disposition": value.conjunction_disposition.value,
        "match_status": value.match_status.value,
        "error_code": value.error_code,
        "error_type": value.error_type,
        "scale_digest": componentwise_scale_digest(),
    }


@dataclass(frozen=True, slots=True)
class ComponentwisePositiveProseObservation:
    cue_digest: str
    component_1_digest: str
    component_2_digest: str
    component_1_interval: ComponentScoreInterval | None
    component_2_interval: ComponentScoreInterval | None
    component_1_disposition: ComponentDisposition
    component_2_disposition: ComponentDisposition
    conjunction_disposition: ComponentDisposition
    match_status: ComponentwiseMatchStatus
    error_code: str | None
    error_type: str | None
    observation_digest: str

    def __post_init__(self) -> None:
        for label, value in (
            ("cue digest", self.cue_digest),
            ("component 1 digest", self.component_1_digest),
            ("component 2 digest", self.component_2_digest),
            ("observation digest", self.observation_digest),
        ):
            _digest(value, label)
        if not all(
            isinstance(item, ComponentDisposition)
            for item in (
                self.component_1_disposition,
                self.component_2_disposition,
                self.conjunction_disposition,
            )
        ) or not isinstance(self.match_status, ComponentwiseMatchStatus):
            raise ComponentwisePositiveProseError("observation disposition differs")
        projected = combine_component_dispositions(
            self.component_1_disposition, self.component_2_disposition
        )
        if projected != (self.conjunction_disposition, self.match_status):
            raise ComponentwisePositiveProseError("Python conjunction projection differs")
        if self.conjunction_disposition is ComponentDisposition.ERROR:
            if (
                self.component_1_interval is not None
                or self.component_2_interval is not None
                or self.component_1_disposition is not ComponentDisposition.ERROR
                or self.component_2_disposition is not ComponentDisposition.ERROR
                or type(self.error_code) is not str
                or _CODE.fullmatch(self.error_code) is None
                or type(self.error_type) is not str
                or _CODE.fullmatch(self.error_type) is None
            ):
                raise ComponentwisePositiveProseError("error observation differs")
        elif (
            type(self.component_1_interval) is not ComponentScoreInterval
            or type(self.component_2_interval) is not ComponentScoreInterval
            or self.component_1_disposition
            is not classify_component_interval(self.component_1_interval)
            or self.component_2_disposition
            is not classify_component_interval(self.component_2_interval)
            or self.error_code is not None
            or self.error_type is not None
        ):
            raise ComponentwisePositiveProseError("scored observation differs")
        if self.observation_digest != canonical_digest(_observation_content(self)):
            raise ComponentwisePositiveProseError("observation digest differs")

    @classmethod
    def from_intervals(
        cls,
        cue: ComponentwisePositiveCue,
        first: ComponentScoreInterval,
        second: ComponentScoreInterval,
    ) -> "ComponentwisePositiveProseObservation":
        d1 = classify_component_interval(first)
        d2 = classify_component_interval(second)
        conjunction, status = combine_component_dispositions(d1, d2)
        values = {
            "cue_digest": cue.cue_digest,
            "component_1_digest": cue.component_digest(1),
            "component_2_digest": cue.component_digest(2),
            "component_1_interval": first,
            "component_2_interval": second,
            "component_1_disposition": d1,
            "component_2_disposition": d2,
            "conjunction_disposition": conjunction,
            "match_status": status,
            "error_code": None,
            "error_type": None,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(**values, observation_digest=canonical_digest(_observation_content(provisional)))

    @classmethod
    def error(
        cls,
        cue: ComponentwisePositiveCue,
        error_code: str,
        error_type: str,
    ) -> "ComponentwisePositiveProseObservation":
        values = {
            "cue_digest": cue.cue_digest,
            "component_1_digest": cue.component_digest(1),
            "component_2_digest": cue.component_digest(2),
            "component_1_interval": None,
            "component_2_interval": None,
            "component_1_disposition": ComponentDisposition.ERROR,
            "component_2_disposition": ComponentDisposition.ERROR,
            "conjunction_disposition": ComponentDisposition.ERROR,
            "match_status": ComponentwiseMatchStatus.ERROR,
            "error_code": error_code,
            "error_type": error_type,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(**values, observation_digest=canonical_digest(_observation_content(provisional)))

    def to_data(self) -> dict[str, object]:
        return {**_observation_content(self), "observation_digest": self.observation_digest}

    @classmethod
    def from_data(cls, value: object) -> "ComponentwisePositiveProseObservation":
        raw = _fields(
            value,
            {
                "schema", "cue_digest", "component_1_digest", "component_2_digest",
                "component_1_interval", "component_2_interval",
                "component_1_disposition", "component_2_disposition",
                "conjunction_disposition", "match_status", "error_code", "error_type",
                "scale_digest", "observation_digest",
            },
            "component observation",
        )
        if (
            raw["schema"] != COMPONENTWISE_OBSERVATION_SCHEMA
            or raw["scale_digest"] != componentwise_scale_digest()
        ):
            raise ComponentwisePositiveProseError("component observation policy differs")
        try:
            result = cls(
                raw["cue_digest"], raw["component_1_digest"], raw["component_2_digest"],
                None if raw["component_1_interval"] is None else ComponentScoreInterval.from_data(raw["component_1_interval"]),
                None if raw["component_2_interval"] is None else ComponentScoreInterval.from_data(raw["component_2_interval"]),
                ComponentDisposition(raw["component_1_disposition"]),
                ComponentDisposition(raw["component_2_disposition"]),
                ComponentDisposition(raw["conjunction_disposition"]),
                ComponentwiseMatchStatus(raw["match_status"]),
                raw["error_code"], raw["error_type"], raw["observation_digest"],
            )
        except ComponentwisePositiveProseError:
            raise
        except Exception as exc:
            raise ComponentwisePositiveProseError(
                "component observation enum differs"
            ) from exc
        if result.to_data() != dict(raw):
            raise ComponentwisePositiveProseError("component observation is not canonical")
        return result


def _parse_payload(
    value: object, cue: ComponentwisePositiveCue
) -> ComponentwisePositiveProseObservation:
    raw = _fields(
        value,
        {
            "component_1_lower", "component_1_upper",
            "component_2_lower", "component_2_upper",
        },
        "component model payload",
    )
    return ComponentwisePositiveProseObservation.from_intervals(
        cue,
        ComponentScoreInterval(raw["component_1_lower"], raw["component_1_upper"]),
        ComponentScoreInterval(raw["component_2_lower"], raw["component_2_upper"]),
    )


def _receipt_data(value: CodexReceipt | None) -> object:
    return None if value is None else value.to_dict()


def _artifact_authority() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "prose_is_inert_data": True,
        "separate_component_intervals_required": True,
        "negative_foil_or_complement_present": False,
        "model_threshold_or_polarity_present": False,
        "negation_rescue_allowed": False,
        "lean_present": False,
        "lean_required": False,
        "lean_removable": True,
        "raw_panel_only": True,
        "candidate_independent_transformed_view_supported": False,
        "crop_or_context_adapter_present": False,
        "observer_calibrated": False,
        "latent_action_segmentation_solved": False,
    }


def _artifact_content(value: "ComponentwisePositiveProsePanelArtifact") -> dict[str, object]:
    return {
        "schema": COMPONENTWISE_ARTIFACT_SCHEMA,
        "request": value.request.to_data(),
        "request_digest": value.request.request_digest,
        "source_digest": value.source_digest,
        "protocol_digest": value.protocol_digest,
        "transport_provenance": value.transport_provenance.to_data(),
        "prompt_digest": value.prompt_digest,
        "output_schema_digest": value.output_schema_digest,
        "presentation": [item.to_data() for item in value.presentation],
        "physical_call_count": value.physical_call_count,
        "status": value.status.value,
        "model_payload": value.model_payload,
        "receipt": _receipt_data(value.receipt),
        "observation": value.observation.to_data(),
        "failure_code": value.failure_code,
        "failure_type": value.failure_type,
        "benchmark_sealable": value.benchmark_sealable,
        "model_visible_image_names": ["panel.png"],
        **_artifact_authority(),
    }


@dataclass(frozen=True, slots=True)
class ComponentwisePositiveProsePanelArtifact:
    request: ComponentwisePositiveProsePanelRequest
    source_digest: str
    protocol_digest: str
    transport_provenance: PositiveProseTransportProvenance
    prompt_digest: str
    output_schema_digest: str
    presentation: tuple[PrototypeImageIdentity, ...]
    physical_call_count: int
    status: PrototypeSceneObserverStatus
    model_payload: Mapping[str, Any] | None
    receipt: CodexReceipt | None
    observation: ComponentwisePositiveProseObservation
    failure_code: str | None
    failure_type: str | None
    artifact_digest: str

    @property
    def benchmark_sealable(self) -> bool:
        return (
            self.status is PrototypeSceneObserverStatus.SUCCESS
            and self.transport_provenance.benchmark_sealable
        )

    def __post_init__(self) -> None:
        if type(self.request) is not ComponentwisePositiveProsePanelRequest:
            raise TypeError("component artifact needs exact request")
        for label, value in (
            ("source digest", self.source_digest),
            ("protocol digest", self.protocol_digest),
            ("prompt digest", self.prompt_digest),
            ("output schema digest", self.output_schema_digest),
            ("artifact digest", self.artifact_digest),
        ):
            _digest(value, label)
        if type(self.transport_provenance) is not PositiveProseTransportProvenance:
            raise TypeError("component artifact provenance differs")
        prompt = componentwise_positive_prose_prompt(self.request)
        schema = componentwise_positive_prose_output_schema(self.request)
        if (
            self.source_digest != panel_componentwise_positive_prose_observer_source_digest()
            or self.protocol_digest != componentwise_protocol_digest()
            or self.prompt_digest != hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            or self.output_schema_digest != canonical_digest(schema)
            or type(self.presentation) is not tuple
            or len(self.presentation) != 1
            or self.presentation[0].name != "panel.png"
            or self.presentation[0].content_digest != self.request.context.panel_png_digest
            or self.presentation[0].byte_count != self.request.context.panel_png_byte_count
            or self.physical_call_count != 1
            or not isinstance(self.status, PrototypeSceneObserverStatus)
            or type(self.observation) is not ComponentwisePositiveProseObservation
            or self.observation.cue_digest != self.request.cue.cue_digest
        ):
            raise ComponentwisePositiveProseError("component artifact binding differs")
        if self.model_payload is not None:
            object.__setattr__(self, "model_payload", _canonical_payload(self.model_payload))
        if self.status is PrototypeSceneObserverStatus.SUCCESS:
            if (
                self.model_payload is None
                or self.receipt is None
                or self.failure_code is not None
                or self.failure_type is not None
                or self.observation != _parse_payload(self.model_payload, self.request.cue)
            ):
                raise ComponentwisePositiveProseError("successful component artifact differs")
        elif self.status in {
            PrototypeSceneObserverStatus.PARSER_ERROR,
            PrototypeSceneObserverStatus.TRANSPORT_ERROR,
        }:
            if (
                self.observation.conjunction_disposition is not ComponentDisposition.ERROR
                or type(self.failure_code) is not str
                or _CODE.fullmatch(self.failure_code) is None
                or type(self.failure_type) is not str
                or _CODE.fullmatch(self.failure_type) is None
                or self.observation.error_code != self.failure_code
                or self.observation.error_type != self.failure_type
            ):
                raise ComponentwisePositiveProseError("failed artifact lacks typed error")
            if self.status is PrototypeSceneObserverStatus.PARSER_ERROR and (
                self.model_payload is None or self.receipt is None
            ):
                raise ComponentwisePositiveProseError("parser error lacks payload receipt")
            if self.status is PrototypeSceneObserverStatus.TRANSPORT_ERROR and (
                self.model_payload is not None or self.receipt is not None
            ):
                raise ComponentwisePositiveProseError("transport error contains payload")
        else:
            raise ComponentwisePositiveProseError("component artifact status differs")
        if self.receipt is not None:
            runtime = self.request.context.runtime
            view = [item.to_data() for item in self.presentation]
            expected_set = "sha256:" + canonical_digest(
                {"schema": NAMED_IMAGE_INPUT_DIGEST_SCHEMA, "images": view}
            )
            if (
                self.receipt.prompt_digest != self.prompt_digest
                or self.receipt.output_schema_digest != self.output_schema_digest
                or self.receipt.structured_output_digest
                != canonical_digest(dict(self.model_payload or {}))
                or self.receipt.panel_view_digest != canonical_digest(view)
                or self.receipt.panel_set_digest != expected_set
                or self.receipt.requested_model != runtime.model
                or self.receipt.requested_reasoning_effort != runtime.reasoning_effort
                or self.receipt.codex_launcher_digest != runtime.expected_launcher_digest
                or self.receipt.cloud_config_bundle_cache_binding
                != runtime.cloud_policy_cache_binding
                or self.receipt.model_catalog_digest != runtime.model_catalog_digest
                or self.receipt.tool_surface_attestation_digest
                != runtime.no_tools_attestation_digest
            ):
                raise ComponentwisePositiveProseError("component receipt binding differs")
        if self.artifact_digest != canonical_digest(_artifact_content(self)):
            raise ComponentwisePositiveProseError("component artifact digest differs")

    def to_data(self) -> dict[str, object]:
        return {**_artifact_content(self), "artifact_digest": self.artifact_digest}

    @classmethod
    def from_data(cls, value: object) -> "ComponentwisePositiveProsePanelArtifact":
        expected = {
            "schema", "request", "request_digest", "source_digest", "protocol_digest",
            "transport_provenance", "prompt_digest", "output_schema_digest",
            "presentation", "physical_call_count", "status", "model_payload", "receipt",
            "observation", "failure_code", "failure_type", "benchmark_sealable",
            "model_visible_image_names", "artifact_digest", *_artifact_authority(),
        }
        raw = _fields(value, expected, "component artifact")
        if (
            raw["schema"] != COMPONENTWISE_ARTIFACT_SCHEMA
            or raw["model_visible_image_names"] != ["panel.png"]
            or type(raw["presentation"]) is not list
            or any(raw[key] != item for key, item in _artifact_authority().items())
        ):
            raise ComponentwisePositiveProseError("component artifact policy differs")
        try:
            status = PrototypeSceneObserverStatus(raw["status"])
        except Exception as exc:
            raise ComponentwisePositiveProseError("component artifact status unknown") from exc
        result = cls(
            ComponentwisePositiveProsePanelRequest.from_data(raw["request"]),
            raw["source_digest"], raw["protocol_digest"],
            PositiveProseTransportProvenance.from_data(raw["transport_provenance"]),
            raw["prompt_digest"], raw["output_schema_digest"],
            tuple(PrototypeImageIdentity.from_data(item) for item in raw["presentation"]),
            raw["physical_call_count"], status, raw["model_payload"],
            None if raw["receipt"] is None else _receipt_from_data(raw["receipt"]),
            ComponentwisePositiveProseObservation.from_data(raw["observation"]),
            raw["failure_code"], raw["failure_type"], raw["artifact_digest"],
        )
        if (
            result.request.request_digest != raw["request_digest"]
            or result.benchmark_sealable is not raw["benchmark_sealable"]
            or result.to_data() != dict(raw)
        ):
            raise ComponentwisePositiveProseError("component artifact replay differs")
        return result


def _seal_artifact(
    *,
    request: ComponentwisePositiveProsePanelRequest,
    provenance: PositiveProseTransportProvenance,
    presentation: tuple[PrototypeImageIdentity, ...],
    status: PrototypeSceneObserverStatus,
    payload: Mapping[str, Any] | None,
    receipt: CodexReceipt | None,
    observation: ComponentwisePositiveProseObservation,
    failure_code: str | None,
    failure_type: str | None,
) -> ComponentwisePositiveProsePanelArtifact:
    values = {
        "request": request,
        "source_digest": panel_componentwise_positive_prose_observer_source_digest(),
        "protocol_digest": componentwise_protocol_digest(),
        "transport_provenance": provenance,
        "prompt_digest": hashlib.sha256(
            componentwise_positive_prose_prompt(request).encode("utf-8")
        ).hexdigest(),
        "output_schema_digest": canonical_digest(
            componentwise_positive_prose_output_schema(request)
        ),
        "presentation": presentation,
        "physical_call_count": 1,
        "status": status,
        "model_payload": None if payload is None else _canonical_payload(payload),
        "receipt": receipt,
        "observation": observation,
        "failure_code": failure_code,
        "failure_type": failure_type,
    }
    provisional = object.__new__(ComponentwisePositiveProsePanelArtifact)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ComponentwisePositiveProsePanelArtifact(
        **values, artifact_digest=canonical_digest(_artifact_content(provisional))
    )


def _verify_lineage(
    request: ComponentwisePositiveProsePanelRequest,
    proposer_artifact: SupportPositiveProposerArtifact,
    group_a_pngs: Sequence[bytes],
    group_b_pngs: Sequence[bytes],
    *,
    expected_artifact_digest: str,
    proposer_journal_terminal: ObjectBongardTurnJournalSummary,
) -> None:
    proposer, terminal = _restore_admitted_proposer(
        proposer_artifact,
        group_a_pngs,
        group_b_pngs,
        expected_artifact_digest=expected_artifact_digest,
        proposer_journal_terminal=proposer_journal_terminal,
    )
    if (
        proposer.runtime != request.context.runtime
        or request.cue != ComponentwisePositiveCue.from_proposer(proposer, terminal)
        or not request.cue.proposer_terminal.matches(proposer_journal_terminal)
    ):
        raise ComponentwisePositiveProseError(
            "component request differs from exact proposer lineage"
        )


def observe_componentwise_positive_prose_panel(
    panel_png: bytes,
    *,
    request: ComponentwisePositiveProsePanelRequest,
    source_proposer_artifact: SupportPositiveProposerArtifact,
    group_a_pngs: Sequence[bytes],
    group_b_pngs: Sequence[bytes],
    expected_source_proposer_artifact_digest: str,
    proposer_journal_terminal: ObjectBongardTurnJournalSummary,
    minutes: int = 15,
    verbose: bool = False,
    executable: str = "codex",
    cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot | None,
    expected_launcher_digest: str,
    model_catalog_snapshot: CodexModelCatalogSnapshot,
    no_tools_attestation: CodexNoToolsAttestation,
    transport=run_codex_named_images_structured,
) -> ComponentwisePositiveProsePanelArtifact:
    """Score both frozen components in one physical raw-panel vision call."""

    panel = _exact_png(panel_png)
    if type(request) is not ComponentwisePositiveProsePanelRequest:
        raise TypeError("component observer needs exact request")
    _verify_lineage(
        request,
        source_proposer_artifact,
        group_a_pngs,
        group_b_pngs,
        expected_artifact_digest=expected_source_proposer_artifact_digest,
        proposer_journal_terminal=proposer_journal_terminal,
    )
    if not callable(transport):
        raise TypeError("component transport must be callable")
    context = request.context
    runtime = _bind_runtime(
        model=context.runtime.model,
        reasoning_effort=context.runtime.reasoning_effort,
        expected_launcher_digest=expected_launcher_digest,
        cloud_policy_cache_snapshot=cloud_policy_cache_snapshot,
        model_catalog_snapshot=model_catalog_snapshot,
        no_tools_attestation=no_tools_attestation,
    )
    if (
        runtime != context.runtime
        or hashlib.sha256(panel).hexdigest() != context.panel_png_digest
        or len(panel) != context.panel_png_byte_count
    ):
        raise ComponentwisePositiveProseError("request belongs to another panel/runtime")
    prompt = componentwise_positive_prose_prompt(request)
    schema = componentwise_positive_prose_output_schema(request)
    presentation_bytes = (("panel.png", panel),)
    presentation = _scene_runtime._image_identities(presentation_bytes)
    _scene_runtime._assert_model_visible_boundary(
        prompt,
        schema,
        ("panel.png",),
        hidden_values=(
            context.panel_id,
            context.panel_png_digest,
            context.context_digest,
            request.cue.cue_digest,
            request.request_digest,
            request.cue.proposer_terminal.binding_digest,
            POSITIVE_ORIENTATION,
        ),
        allowed_visual_words=("side",),
    )
    try:
        payload, receipt = _scene_runtime._stage_and_call(
            presentation_bytes,
            prompt=prompt,
            schema=schema,
            model=runtime.model,
            reasoning_effort=runtime.reasoning_effort,
            minutes=minutes,
            verbose=verbose,
            executable=executable,
            cloud_policy_cache_snapshot=cloud_policy_cache_snapshot,
            expected_launcher_digest=expected_launcher_digest,
            model_catalog_snapshot=model_catalog_snapshot,
            no_tools_attestation=no_tools_attestation,
            transport=transport,
        )
    except Exception as exc:
        error_type = _scene_runtime._exception_type(exc)
        try:
            provenance = _v1._transport_provenance(transport)
        except Exception:
            provenance = PositiveProseTransportProvenance.create(
                "production_direct"
                if transport is run_codex_named_images_structured
                else "injected_unverified"
            )
        return _seal_artifact(
            request=request,
            provenance=provenance,
            presentation=presentation,
            status=PrototypeSceneObserverStatus.TRANSPORT_ERROR,
            payload=None,
            receipt=None,
            observation=ComponentwisePositiveProseObservation.error(
                request.cue, "component_observer_transport_failed", error_type
            ),
            failure_code="component_observer_transport_failed",
            failure_type=error_type,
        )
    provenance = _v1._transport_provenance(transport)
    try:
        observation = _parse_payload(payload, request.cue)
    except Exception as exc:
        error_type = _scene_runtime._exception_type(exc)
        return _seal_artifact(
            request=request,
            provenance=provenance,
            presentation=presentation,
            status=PrototypeSceneObserverStatus.PARSER_ERROR,
            payload=payload,
            receipt=receipt,
            observation=ComponentwisePositiveProseObservation.error(
                request.cue, "component_observer_payload_rejected", error_type
            ),
            failure_code="component_observer_payload_rejected",
            failure_type=error_type,
        )
    return _seal_artifact(
        request=request,
        provenance=provenance,
        presentation=presentation,
        status=PrototypeSceneObserverStatus.SUCCESS,
        payload=payload,
        receipt=receipt,
        observation=observation,
        failure_code=None,
        failure_type=None,
    )


def verify_componentwise_positive_prose_panel_artifact(
    artifact: ComponentwisePositiveProsePanelArtifact,
    panel_png: bytes,
    *,
    expected_artifact_digest: str,
    source_proposer_artifact: SupportPositiveProposerArtifact,
    group_a_pngs: Sequence[bytes],
    group_b_pngs: Sequence[bytes],
    expected_source_proposer_artifact_digest: str,
    proposer_journal_terminal: ObjectBongardTurnJournalSummary,
    query_journal_terminal: ObjectBongardTurnJournalSummary | None = None,
    expected_request_digest: str | None = None,
) -> ComponentwisePositiveProsePanelArtifact:
    """Zero-model-call replay of pixels, proposer/query custody, and projection."""

    if type(artifact) is not ComponentwisePositiveProsePanelArtifact:
        raise TypeError("component replay needs exact artifact")
    restored = ComponentwisePositiveProsePanelArtifact.from_data(artifact.to_data())
    if restored.artifact_digest != _digest(
        expected_artifact_digest, "expected component artifact digest"
    ):
        raise ComponentwisePositiveProseError("component artifact commitment differs")
    try:
        _v1._verify_external_journal_terminal(
            restored.transport_provenance, query_journal_terminal
        )
    except Exception as exc:
        raise ComponentwisePositiveProseError(
            "external query journal terminal differs"
        ) from exc
    if expected_request_digest is not None and restored.request.request_digest != _digest(
        expected_request_digest, "expected component request digest"
    ):
        raise ComponentwisePositiveProseError("component request commitment differs")
    _verify_lineage(
        restored.request,
        source_proposer_artifact,
        group_a_pngs,
        group_b_pngs,
        expected_artifact_digest=expected_source_proposer_artifact_digest,
        proposer_journal_terminal=proposer_journal_terminal,
    )
    panel = _exact_png(panel_png)
    context = restored.request.context
    if (
        hashlib.sha256(panel).hexdigest() != context.panel_png_digest
        or len(panel) != context.panel_png_byte_count
    ):
        raise ComponentwisePositiveProseError("component replay panel differs")
    if restored.receipt is not None:
        assert restored.model_payload is not None
        prompt = componentwise_positive_prose_prompt(restored.request)
        schema = componentwise_positive_prose_output_schema(restored.request)
        with tempfile.TemporaryDirectory(prefix="bongard-componentwise-replay-") as raw:
            target = Path(raw) / "panel.png"
            target.write_bytes(panel)
            validate_codex_named_image_receipt(
                restored.receipt,
                prompt,
                (str(target.resolve()),),
                ("panel.png",),
                schema,
                dict(restored.model_payload),
            )
            if target.read_bytes() != panel:
                raise ComponentwisePositiveProseError("component replay panel changed")
    return restored


__all__ = (
    "COMPONENTWISE_PROTOCOL_ID",
    "COMPONENT_SCORE_ANCHORS",
    "ComponentDisposition",
    "ComponentScoreInterval",
    "ComponentwiseMatchStatus",
    "ComponentwisePositiveCue",
    "ComponentwisePositiveProseError",
    "ComponentwisePositiveProseObservation",
    "ComponentwisePositiveProsePanelArtifact",
    "ComponentwisePositiveProsePanelRequest",
    "ProposerTerminalBinding",
    "classify_component_interval",
    "combine_component_dispositions",
    "componentwise_positive_prose_output_schema",
    "componentwise_positive_prose_prompt",
    "componentwise_protocol_digest",
    "componentwise_scale_digest",
    "observe_componentwise_positive_prose_panel",
    "panel_componentwise_positive_prose_observer_source_digest",
    "verify_componentwise_positive_prose_panel_artifact",
)
