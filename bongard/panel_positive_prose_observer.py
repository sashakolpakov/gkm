"""One-sided, one-panel observer for a frozen positive prose cue.

The cue is inert data and may describe a conjunction.  No foil, complement,
negative-class description, candidate choice, threshold choice, polarity
choice, executable prose, or Lean input exists on this boundary.  A vision
model returns only an inclusive interval on the fixed 0..4 absolute-match
scale; Python projects it to the canonical four dispositions.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

from bongard import prototype_scene_observer as _scene_runtime
from bongard.canonical import canonical_digest, canonical_json
from bongard.codex_no_tools_preflight import CodexNoToolsAttestation
from bongard.evidence import Disposition
from bongard.object_bongard_turn_journal import (
    ObjectBongardNamedImageTurnJournalTransport,
    ObjectBongardTurnJournalSummary,
    object_bongard_turn_journal_source_digest,
)
from bongard.panel_typed_codex_observer import (
    TypedCodexRuntimeBinding,
    _bind_runtime,
    _exact_png,
    _receipt_from_data,
)
from bongard.panel_support_positive_proposer import (
    PositiveConjunctionRubric,
    SupportPositiveProposerArtifact,
    SupportPositiveProposerError,
)
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


POSITIVE_PROSE_CUE_SCHEMA = "gkm.bongard-positive-prose-cue.v1"
POSITIVE_PROSE_CONTEXT_SCHEMA = "gkm.bongard-positive-prose-panel-context.v1"
POSITIVE_PROSE_REQUEST_SCHEMA = "gkm.bongard-positive-prose-panel-request.v1"
POSITIVE_PROSE_OBSERVATION_SCHEMA = "gkm.bongard-positive-prose-observation.v1"
POSITIVE_PROSE_TRANSPORT_SCHEMA = "gkm.bongard-positive-prose-transport.v1"
POSITIVE_PROSE_ARTIFACT_SCHEMA = "gkm.bongard-positive-prose-panel-artifact.v1"
POSITIVE_PROSE_PROTOCOL_ID = "bongard.positive-prose-observer/one-panel-absolute-v1"
POSITIVE_ORIENTATION = "side0_positive"
MAX_POSITIVE_CUE_BYTES = 2048

POSITIVE_PROSE_SCORE_ANCHORS: tuple[tuple[int, str], ...] = (
    (0, "The complete panel clearly mismatches the positive cue."),
    (
        1,
        "Visible evidence decisively rules out a complete match because at least "
        "one required part of the positive cue fails.",
    ),
    (
        2,
        "The evidence is ambiguous, mixed, insufficient, or resolves only part "
        "of a conjunction.",
    ),
    (3, "The complete panel matches the positive cue."),
    (4, "The complete panel clearly and completely matches the positive cue."),
)

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_PANEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,511}\Z")
_CODE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_TRANSPORT_KINDS = frozenset(
    {"production_direct", "production_exactly_once_journal", "injected_unverified"}
)


class PositiveProseObserverError(ValueError):
    """A positive cue, observation, custody record, or replay is invalid."""


PositiveProseDisposition = Disposition


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(type(key) is not str for key in value)
        or set(value) != expected
    ):
        raise PositiveProseObserverError(f"{label} fields differ")
    return value


def _digest(value: object, label: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise PositiveProseObserverError(f"{label} must be a raw SHA-256")
    return value


def _address(value: object, label: str) -> str:
    if type(value) is not str or _ADDRESS.fullmatch(value) is None:
        raise PositiveProseObserverError(f"{label} must be a sha256: address")
    return value


def _panel_id(value: object) -> str:
    if type(value) is not str or _PANEL_ID.fullmatch(value) is None:
        raise PositiveProseObserverError("panel ID is invalid")
    return value


def _canonical_payload(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise PositiveProseObserverError("model payload must be an object")
    try:
        result = json.loads(canonical_json(dict(value)).decode("utf-8"))
    except Exception as exc:
        raise PositiveProseObserverError("model payload is not canonical JSON") from exc
    if type(result) is not dict:
        raise PositiveProseObserverError("model payload must be an object")
    return result


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "positive_orientation": POSITIVE_ORIENTATION,
        "prose_is_inert_data": True,
        "inert_prose_may_describe_a_conjunction": True,
        "admitted_support_positive_proposer_artifact_required": True,
        "bare_cue_or_string_allowed": False,
        "source_proposal_gap_allowed": False,
        "foil_field_present": False,
        "complement_field_present": False,
        "negative_class_description_field_present": False,
        "candidate_selection_allowed": False,
        "threshold_selection_allowed": False,
        "polarity_selection_allowed": False,
        "negation_allowed": False,
        "lean_present": False,
        "lean_required": False,
        "lean_removable": True,
        "lean_defines_identity_decision_or_replay": False,
    }


def panel_positive_prose_observer_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def _restore_admitted_proposer(
    artifact: SupportPositiveProposerArtifact,
    *,
    expected_artifact_digest: str,
) -> SupportPositiveProposerArtifact:
    """Verify the frozen proposer envelope without admitting support pixels here."""

    if type(artifact) is not SupportPositiveProposerArtifact:
        raise TypeError("positive prose request needs SupportPositiveProposerArtifact")
    expected = _digest(
        expected_artifact_digest, "expected support positive proposer artifact digest"
    )
    try:
        restored = SupportPositiveProposerArtifact.from_data(artifact.to_data())
    except SupportPositiveProposerError as exc:
        raise PositiveProseObserverError(
            "positive prose source proposer artifact failed replay"
        ) from exc
    if restored.artifact_digest != expected:
        raise PositiveProseObserverError(
            "positive prose source proposer differs from commitment"
        )
    if restored.rubric is None or restored.proposal_gap is not None:
        raise PositiveProseObserverError(
            "positive prose source proposer produced a proposal gap"
        )
    return restored


@dataclass(frozen=True, slots=True)
class PositiveProseCue:
    text: str
    source_proposer_artifact_digest: str
    source_proposer_request_digest: str
    source_rubric_digest: str
    source_proposer_benchmark_sealable: bool
    cue_digest: str

    def __post_init__(self) -> None:
        if (
            type(self.text) is not str
            or not self.text
            or self.text != self.text.strip()
            or len(self.text.encode("utf-8")) > MAX_POSITIVE_CUE_BYTES
            or any(ord(char) < 32 and char not in "\n\t" for char in self.text)
        ):
            raise PositiveProseObserverError("positive cue text is not bounded prose")
        _digest(
            self.source_proposer_artifact_digest,
            "positive cue source proposer artifact digest",
        )
        _digest(
            self.source_proposer_request_digest,
            "positive cue source proposer request digest",
        )
        _digest(self.source_rubric_digest, "positive cue source rubric digest")
        if type(self.source_proposer_benchmark_sealable) is not bool:
            raise PositiveProseObserverError(
                "positive cue source proposer sealability differs"
            )
        _digest(self.cue_digest, "positive cue digest")
        if self.cue_digest != canonical_digest(self.content_data()):
            raise PositiveProseObserverError("positive cue digest differs")

    @classmethod
    def _from_verified_proposer(
        cls, artifact: SupportPositiveProposerArtifact
    ) -> "PositiveProseCue":
        if type(artifact) is not SupportPositiveProposerArtifact:
            raise TypeError("positive cue needs exact verified proposer artifact")
        rubric = artifact.rubric
        if type(rubric) is not PositiveConjunctionRubric or artifact.proposal_gap is not None:
            raise PositiveProseObserverError(
                "positive cue unavailable because proposer rubric was not admitted"
            )
        values = {
            "text": rubric.cue_text,
            "source_proposer_artifact_digest": artifact.artifact_digest,
            "source_proposer_request_digest": artifact.request_digest,
            "source_rubric_digest": rubric.rubric_digest,
            "source_proposer_benchmark_sealable": artifact.benchmark_sealable,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(**values, cue_digest=canonical_digest(provisional.content_data()))

    def content_data(self) -> dict[str, object]:
        return {
            "schema": POSITIVE_PROSE_CUE_SCHEMA,
            "text": self.text,
            "source_kind": "admitted_support_positive_proposer_artifact",
            "source_proposer_artifact_digest": self.source_proposer_artifact_digest,
            "source_proposer_request_digest": self.source_proposer_request_digest,
            "source_rubric_digest": self.source_rubric_digest,
            "source_proposer_benchmark_sealable": (
                self.source_proposer_benchmark_sealable
            ),
            "source_rubric_admitted": True,
            "source_proposal_gap_present": False,
            "cue_role": "positive",
            "positive_orientation": POSITIVE_ORIENTATION,
            "prose_is_executable": False,
            "conjunction_allowed": True,
            "foil_field_present": False,
            "complement_field_present": False,
            "negative_class_description_field_present": False,
        }

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "cue_digest": self.cue_digest}

    @classmethod
    def from_data(cls, value: object) -> "PositiveProseCue":
        raw = _fields(
            value,
            {
                "schema", "text", "source_kind",
                "source_proposer_artifact_digest",
                "source_proposer_request_digest", "source_rubric_digest",
                "source_proposer_benchmark_sealable", "source_rubric_admitted",
                "source_proposal_gap_present", "cue_role", "positive_orientation",
                "prose_is_executable", "conjunction_allowed",
                "foil_field_present", "complement_field_present",
                "negative_class_description_field_present", "cue_digest",
            },
            "positive prose cue",
        )
        if (
            raw["schema"] != POSITIVE_PROSE_CUE_SCHEMA
            or raw["source_kind"]
            != "admitted_support_positive_proposer_artifact"
            or raw["source_rubric_admitted"] is not True
            or raw["source_proposal_gap_present"] is not False
            or raw["cue_role"] != "positive"
            or raw["positive_orientation"] != POSITIVE_ORIENTATION
            or raw["prose_is_executable"] is not False
            or raw["conjunction_allowed"] is not True
            or raw["foil_field_present"] is not False
            or raw["complement_field_present"] is not False
            or raw["negative_class_description_field_present"] is not False
        ):
            raise PositiveProseObserverError("positive cue policy differs")
        result = cls(
            raw["text"], raw["source_proposer_artifact_digest"],
            raw["source_proposer_request_digest"], raw["source_rubric_digest"],
            raw["source_proposer_benchmark_sealable"], raw["cue_digest"],
        )
        if result.to_data() != dict(raw):
            raise PositiveProseObserverError("positive cue is not canonical")
        return result


@dataclass(frozen=True, order=True, slots=True)
class PositiveProseScoreInterval:
    lower: int
    upper: int

    def __post_init__(self) -> None:
        if (
            type(self.lower) is not int
            or type(self.upper) is not int
            or not 0 <= self.lower <= self.upper <= 4
        ):
            raise PositiveProseObserverError("score interval must lie in 0..4")

    def to_data(self) -> dict[str, int]:
        return {"lower": self.lower, "upper": self.upper}

    @classmethod
    def from_data(cls, value: object) -> "PositiveProseScoreInterval":
        raw = _fields(value, {"lower", "upper"}, "positive score interval")
        return cls(raw["lower"], raw["upper"])


def classify_positive_prose_interval(
    interval: PositiveProseScoreInterval,
) -> PositiveProseDisposition:
    if type(interval) is not PositiveProseScoreInterval:
        raise TypeError("interval must be PositiveProseScoreInterval")
    if interval.lower >= 3:
        return PositiveProseDisposition.PRESENT
    if interval.upper <= 1:
        return PositiveProseDisposition.CERTIFIED_ABSENT
    return PositiveProseDisposition.INDETERMINATE


def positive_prose_scale_digest() -> str:
    return canonical_digest(
        {
            "schema": "gkm.bongard-positive-prose-absolute-scale.v1",
            "anchors": [list(item) for item in POSITIVE_PROSE_SCORE_ANCHORS],
            "interval_semantics": "inclusive-narrowest-honest-range",
            "present_when_lower_at_least": 3,
            "certified_absent_when_upper_at_most": 1,
            "otherwise": PositiveProseDisposition.INDETERMINATE.value,
            "transport_or_parser_failure": PositiveProseDisposition.ERROR.value,
            "measurement_or_transport_failure_is_absence": False,
            "uncertain_visual_fit_is_absence": False,
        }
    )


def _context_content(value: "PositiveProsePanelContext") -> dict[str, object]:
    return {
        "schema": POSITIVE_PROSE_CONTEXT_SCHEMA,
        "panel_id": value.panel_id,
        "panel_png_digest": value.panel_png_digest,
        "panel_png_byte_count": value.panel_png_byte_count,
        "runtime": value.runtime.to_data(),
        "model_visible_image_names": ["panel.png"],
        "task_phase_side_class_or_query_role_model_visible": False,
    }


@dataclass(frozen=True, slots=True)
class PositiveProsePanelContext:
    panel_id: str
    panel_png_digest: str
    panel_png_byte_count: int
    runtime: TypedCodexRuntimeBinding
    context_digest: str

    def __post_init__(self) -> None:
        _panel_id(self.panel_id)
        _digest(self.panel_png_digest, "panel PNG digest")
        if type(self.panel_png_byte_count) is not int or self.panel_png_byte_count <= 0:
            raise PositiveProseObserverError("panel byte count differs")
        if type(self.runtime) is not TypedCodexRuntimeBinding:
            raise TypeError("positive prose context needs typed runtime")
        _digest(self.context_digest, "positive prose context digest")
        if self.context_digest != canonical_digest(_context_content(self)):
            raise PositiveProseObserverError("positive prose context digest differs")

    @classmethod
    def build(
        cls,
        panel_png: bytes,
        *,
        panel_id: str,
        model: str,
        reasoning_effort: str,
        expected_launcher_digest: str,
        cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot | None,
        model_catalog_snapshot: CodexModelCatalogSnapshot,
        no_tools_attestation: CodexNoToolsAttestation,
    ) -> "PositiveProsePanelContext":
        panel = _exact_png(panel_png)
        runtime = _bind_runtime(
            model=model,
            reasoning_effort=reasoning_effort,
            expected_launcher_digest=expected_launcher_digest,
            cloud_policy_cache_snapshot=cloud_policy_cache_snapshot,
            model_catalog_snapshot=model_catalog_snapshot,
            no_tools_attestation=no_tools_attestation,
        )
        values = {
            "panel_id": _panel_id(panel_id),
            "panel_png_digest": hashlib.sha256(panel).hexdigest(),
            "panel_png_byte_count": len(panel),
            "runtime": runtime,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(**values, context_digest=canonical_digest(_context_content(provisional)))

    def to_data(self) -> dict[str, object]:
        return {**_context_content(self), "context_digest": self.context_digest}

    @classmethod
    def from_data(cls, value: object) -> "PositiveProsePanelContext":
        raw = _fields(
            value,
            {
                "schema", "panel_id", "panel_png_digest", "panel_png_byte_count",
                "runtime", "model_visible_image_names",
                "task_phase_side_class_or_query_role_model_visible", "context_digest",
            },
            "positive prose panel context",
        )
        if (
            raw["schema"] != POSITIVE_PROSE_CONTEXT_SCHEMA
            or raw["model_visible_image_names"] != ["panel.png"]
            or raw["task_phase_side_class_or_query_role_model_visible"] is not False
        ):
            raise PositiveProseObserverError("positive prose context policy differs")
        result = cls(
            raw["panel_id"], raw["panel_png_digest"], raw["panel_png_byte_count"],
            TypedCodexRuntimeBinding.from_data(raw["runtime"]), raw["context_digest"],
        )
        if result.to_data() != dict(raw):
            raise PositiveProseObserverError("positive prose context is not canonical")
        return result


def _request_content(value: "PositiveProsePanelRequest") -> dict[str, object]:
    return {
        "schema": POSITIVE_PROSE_REQUEST_SCHEMA,
        "context": value.context.to_data(),
        "cue": value.cue.to_data(),
        "positive_orientation": POSITIVE_ORIENTATION,
        "model_visible_image_names": ["panel.png"],
        "model_returns_only_absolute_match_interval": True,
        "foil_field_present": False,
        "complement_field_present": False,
        "negative_class_description_field_present": False,
    }


@dataclass(frozen=True, slots=True)
class PositiveProsePanelRequest:
    context: PositiveProsePanelContext
    cue: PositiveProseCue
    request_digest: str

    def __post_init__(self) -> None:
        if type(self.context) is not PositiveProsePanelContext:
            raise TypeError("positive prose request needs exact context")
        if type(self.cue) is not PositiveProseCue:
            raise TypeError("positive prose request needs exact cue")
        _digest(self.request_digest, "positive prose request digest")
        if self.request_digest != canonical_digest(_request_content(self)):
            raise PositiveProseObserverError("positive prose request digest differs")

    @classmethod
    def build_from_proposer(
        cls,
        context: PositiveProsePanelContext,
        proposer_artifact: SupportPositiveProposerArtifact,
        *,
        expected_artifact_digest: str,
    ) -> "PositiveProsePanelRequest":
        if type(context) is not PositiveProsePanelContext:
            raise TypeError("request needs exact positive prose context")
        restored = _restore_admitted_proposer(
            proposer_artifact,
            expected_artifact_digest=expected_artifact_digest,
        )
        if restored.runtime != context.runtime:
            raise PositiveProseObserverError(
                "positive prose proposer and query observer runtimes differ"
            )
        cue = PositiveProseCue._from_verified_proposer(restored)
        provisional = object.__new__(cls)
        object.__setattr__(provisional, "context", context)
        object.__setattr__(provisional, "cue", cue)
        return cls(context, cue, canonical_digest(_request_content(provisional)))

    def to_data(self) -> dict[str, object]:
        return {**_request_content(self), "request_digest": self.request_digest}

    @classmethod
    def from_data(cls, value: object) -> "PositiveProsePanelRequest":
        raw = _fields(
            value,
            {
                "schema", "context", "cue", "positive_orientation",
                "model_visible_image_names", "model_returns_only_absolute_match_interval",
                "foil_field_present", "complement_field_present",
                "negative_class_description_field_present", "request_digest",
            },
            "positive prose panel request",
        )
        if (
            raw["schema"] != POSITIVE_PROSE_REQUEST_SCHEMA
            or raw["positive_orientation"] != POSITIVE_ORIENTATION
            or raw["model_visible_image_names"] != ["panel.png"]
            or raw["model_returns_only_absolute_match_interval"] is not True
            or raw["foil_field_present"] is not False
            or raw["complement_field_present"] is not False
            or raw["negative_class_description_field_present"] is not False
        ):
            raise PositiveProseObserverError("positive prose request policy differs")
        result = cls(
            PositiveProsePanelContext.from_data(raw["context"]),
            PositiveProseCue.from_data(raw["cue"]),
            raw["request_digest"],
        )
        if result.to_data() != dict(raw):
            raise PositiveProseObserverError("positive prose request is not canonical")
        return result


def _verify_request_proposer_lineage(
    request: PositiveProsePanelRequest,
    proposer: SupportPositiveProposerArtifact,
) -> None:
    if type(request) is not PositiveProsePanelRequest:
        raise TypeError("positive prose lineage needs exact request")
    if type(proposer) is not SupportPositiveProposerArtifact:
        raise TypeError("positive prose lineage needs exact proposer artifact")
    rubric = proposer.rubric
    if rubric is None or proposer.proposal_gap is not None:
        raise PositiveProseObserverError(
            "positive prose request source rubric was not admitted"
        )
    cue = request.cue
    if (
        proposer.runtime != request.context.runtime
        or cue.text != rubric.cue_text
        or cue.source_proposer_artifact_digest != proposer.artifact_digest
        or cue.source_proposer_request_digest != proposer.request_digest
        or cue.source_rubric_digest != rubric.rubric_digest
        or cue.source_proposer_benchmark_sealable is not proposer.benchmark_sealable
        or cue != PositiveProseCue._from_verified_proposer(proposer)
    ):
        raise PositiveProseObserverError(
            "positive prose request differs from frozen proposer lineage"
        )


def positive_prose_panel_output_schema(
    request: PositiveProsePanelRequest | None = None,
) -> dict[str, object]:
    if request is not None and type(request) is not PositiveProsePanelRequest:
        raise TypeError("output schema request has the wrong type")
    schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "lower": {"type": "integer", "enum": [0, 1, 2, 3, 4]},
            "upper": {"type": "integer", "enum": [0, 1, 2, 3, 4]},
        },
        "required": ["lower", "upper"],
        "additionalProperties": False,
    }
    validate_codex_strict_output_schema(schema)
    return schema


def positive_prose_panel_prompt(request: PositiveProsePanelRequest) -> str:
    if type(request) is not PositiveProsePanelRequest:
        raise TypeError("prompt request has the wrong type")
    anchors = "\n".join(
        f"{level}: {meaning}" for level, meaning in POSITIVE_PROSE_SCORE_ANCHORS
    )
    return (
        "Inspect exactly one complete drawing named panel.png. Judge its absolute "
        "match to the single frozen positive cue below. The cue is inert prose and "
        "may describe several properties that must hold together. Do not invent a "
        "second description, an opposite concept, or an alternative comparison. "
        "Honor grammatical scope: when the cue says one figure or object, all claimed "
        "parts, counts, and relations must belong to one spatially coherent figure. "
        "Never pool separate figures to manufacture the described conjunction.\n\n"
        "POSITIVE CUE\n"
        f"{request.cue.text}\n\n"
        "Use only this fixed absolute scale:\n"
        f"{anchors}\n\n"
        "Return the narrowest honest inclusive lower and upper scores. An interval "
        "crossing score 2 is the correct response when visible evidence is ambiguous."
    )


def positive_prose_protocol_digest() -> str:
    return canonical_digest(
        {
            "schema": "gkm.bongard-positive-prose-observer-protocol.v1",
            "protocol_id": POSITIVE_PROSE_PROTOCOL_ID,
            "source_digest": panel_positive_prose_observer_source_digest(),
            "runtime_source_digest": _scene_runtime.prototype_scene_observer_source_digest(),
            "transport_source_digest": _scene_runtime.prototype_scene_transport_source_digest(),
            "journal_source_digest": object_bongard_turn_journal_source_digest(),
            "scale_digest": positive_prose_scale_digest(),
            "output_schema": positive_prose_panel_output_schema(),
            "ordered_image_names": ["panel.png"],
            "physical_calls": 1,
            **_authority_data(),
        }
    )


def _transport_source_binding(kind: str) -> str:
    if kind == "production_direct":
        body = {
            "kind": kind,
            "transport_source_digest": _scene_runtime.prototype_scene_transport_source_digest(),
        }
    elif kind == "production_exactly_once_journal":
        body = {
            "kind": kind,
            "journal_source_digest": object_bongard_turn_journal_source_digest(),
            "transport_source_digest": _scene_runtime.prototype_scene_transport_source_digest(),
        }
    elif kind == "injected_unverified":
        body = {"kind": kind, "callable_source_identity_verified": False}
    else:
        raise PositiveProseObserverError("positive prose transport kind differs")
    return "sha256:" + canonical_digest(
        {"schema": "gkm.bongard-positive-prose-transport-source.v1", **body}
    )


@dataclass(frozen=True, slots=True)
class PositiveProseTransportProvenance:
    kind: str
    source_binding: str
    production_transport_chain_verified: bool
    benchmark_sealable: bool
    journal_terminal_status: str | None = None
    journal_manifest_digest: str | None = None
    journal_turn_key: str | None = None
    journal_claim_digest: str | None = None
    journal_result_digest: str | None = None
    journal_outcome_digest: str | None = None
    journal_terminal_record_digest: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in _TRANSPORT_KINDS:
            raise PositiveProseObserverError("positive prose transport kind differs")
        _address(self.source_binding, "positive prose transport source binding")
        journal = self.kind == "production_exactly_once_journal"
        production = self.kind != "injected_unverified"
        expected_benchmark = journal and self.journal_terminal_status == "success"
        if (
            self.source_binding != _transport_source_binding(self.kind)
            or self.production_transport_chain_verified is not production
            or self.benchmark_sealable is not expected_benchmark
        ):
            raise PositiveProseObserverError("positive prose transport provenance differs")
        journal_values = (
            self.journal_manifest_digest,
            self.journal_turn_key,
            self.journal_claim_digest,
            self.journal_result_digest,
            self.journal_outcome_digest,
            self.journal_terminal_record_digest,
        )
        if journal:
            if self.journal_terminal_status not in {"success", "failure"} or any(
                type(item) is not str or _ADDRESS.fullmatch(item) is None
                for item in journal_values
            ):
                raise PositiveProseObserverError(
                    "positive prose journal terminal provenance differs"
                )
        elif self.journal_terminal_status is not None or any(
            item is not None for item in journal_values
        ):
            raise PositiveProseObserverError(
                "non-journal positive prose transport names a journal"
            )

    @classmethod
    def create(
        cls,
        kind: str,
        *,
        journal_summary: ObjectBongardTurnJournalSummary | None = None,
    ) -> "PositiveProseTransportProvenance":
        if kind == "production_exactly_once_journal":
            if (
                type(journal_summary) is not ObjectBongardTurnJournalSummary
                or journal_summary.terminal_status not in {"success", "failure"}
            ):
                raise PositiveProseObserverError(
                    "positive prose journal is not durably terminal"
                )
            return cls(
                kind=kind,
                source_binding=_transport_source_binding(kind),
                production_transport_chain_verified=True,
                benchmark_sealable=journal_summary.terminal_status == "success",
                journal_terminal_status=journal_summary.terminal_status,
                journal_manifest_digest=journal_summary.manifest_digest,
                journal_turn_key=journal_summary.turn_key,
                journal_claim_digest=journal_summary.claim_digest,
                journal_result_digest=journal_summary.result_digest,
                journal_outcome_digest=journal_summary.outcome_digest,
                journal_terminal_record_digest=journal_summary.record_digest,
            )
        if journal_summary is not None:
            raise PositiveProseObserverError(
                "non-journal positive prose transport received journal custody"
            )
        return cls(
            kind,
            _transport_source_binding(kind),
            kind == "production_direct",
            False,
        )

    def to_data(self) -> dict[str, object]:
        return {
            "schema": POSITIVE_PROSE_TRANSPORT_SCHEMA,
            "kind": self.kind,
            "source_binding": self.source_binding,
            "production_transport_chain_verified": self.production_transport_chain_verified,
            "benchmark_sealable": self.benchmark_sealable,
            "journal_terminal_status": self.journal_terminal_status,
            "journal_manifest_digest": self.journal_manifest_digest,
            "journal_turn_key": self.journal_turn_key,
            "journal_claim_digest": self.journal_claim_digest,
            "journal_result_digest": self.journal_result_digest,
            "journal_outcome_digest": self.journal_outcome_digest,
            "journal_terminal_record_digest": self.journal_terminal_record_digest,
            "physical_model_call_cold_authenticated": False,
            "transport_history_authenticated_by_artifact_alone": False,
            "benchmark_requires_external_typed_journal_terminal": True,
        }

    @classmethod
    def from_data(cls, value: object) -> "PositiveProseTransportProvenance":
        raw = _fields(
            value,
            {
                "schema", "kind", "source_binding",
                "production_transport_chain_verified", "benchmark_sealable",
                "journal_terminal_status", "journal_manifest_digest",
                "journal_turn_key", "journal_claim_digest",
                "journal_result_digest", "journal_outcome_digest",
                "journal_terminal_record_digest",
                "physical_model_call_cold_authenticated",
                "transport_history_authenticated_by_artifact_alone",
                "benchmark_requires_external_typed_journal_terminal",
            },
            "positive prose transport provenance",
        )
        if (
            raw["schema"] != POSITIVE_PROSE_TRANSPORT_SCHEMA
            or raw["physical_model_call_cold_authenticated"] is not False
            or raw["transport_history_authenticated_by_artifact_alone"] is not False
            or raw["benchmark_requires_external_typed_journal_terminal"] is not True
        ):
            raise PositiveProseObserverError("positive prose transport policy differs")
        result = cls(
            kind=raw["kind"],
            source_binding=raw["source_binding"],
            production_transport_chain_verified=raw[
                "production_transport_chain_verified"
            ],
            benchmark_sealable=raw["benchmark_sealable"],
            journal_terminal_status=raw["journal_terminal_status"],
            journal_manifest_digest=raw["journal_manifest_digest"],
            journal_turn_key=raw["journal_turn_key"],
            journal_claim_digest=raw["journal_claim_digest"],
            journal_result_digest=raw["journal_result_digest"],
            journal_outcome_digest=raw["journal_outcome_digest"],
            journal_terminal_record_digest=raw["journal_terminal_record_digest"],
        )
        if result.to_data() != dict(raw):
            raise PositiveProseObserverError("transport provenance is not canonical")
        return result


def _transport_provenance(transport: object) -> PositiveProseTransportProvenance:
    if transport is run_codex_named_images_structured:
        return PositiveProseTransportProvenance.create("production_direct")
    if (
        type(transport) is ObjectBongardNamedImageTurnJournalTransport
        and getattr(transport, "_underlying_transport", None)
        is run_codex_named_images_structured
        and transport.runtime.transport_source_digest
        == _scene_runtime.prototype_scene_transport_source_digest()
    ):
        return PositiveProseTransportProvenance.create(
            "production_exactly_once_journal", journal_summary=transport.verify()
        )
    return PositiveProseTransportProvenance.create("injected_unverified")


def _verify_external_journal_terminal(
    provenance: PositiveProseTransportProvenance,
    summary: ObjectBongardTurnJournalSummary | None,
) -> None:
    if provenance.kind != "production_exactly_once_journal":
        if summary is not None:
            raise PositiveProseObserverError(
                "non-journal artifact received external journal custody"
            )
        return
    if (
        type(summary) is not ObjectBongardTurnJournalSummary
        or summary.terminal_status != provenance.journal_terminal_status
        or (
            summary.manifest_digest,
            summary.turn_key,
            summary.claim_digest,
            summary.result_digest,
            summary.outcome_digest,
            summary.record_digest,
        )
        != (
            provenance.journal_manifest_digest,
            provenance.journal_turn_key,
            provenance.journal_claim_digest,
            provenance.journal_result_digest,
            provenance.journal_outcome_digest,
            provenance.journal_terminal_record_digest,
        )
    ):
        raise PositiveProseObserverError(
            "external positive prose journal terminal differs from artifact custody"
        )


def _observation_content(value: "PositiveProseObservation") -> dict[str, object]:
    return {
        "schema": POSITIVE_PROSE_OBSERVATION_SCHEMA,
        "cue_digest": value.cue_digest,
        "disposition": value.disposition.value,
        "interval": None if value.interval is None else value.interval.to_data(),
        "error_code": value.error_code,
        "error_type": value.error_type,
        "scale_digest": positive_prose_scale_digest(),
    }


@dataclass(frozen=True, slots=True)
class PositiveProseObservation:
    cue_digest: str
    disposition: PositiveProseDisposition
    interval: PositiveProseScoreInterval | None
    error_code: str | None
    error_type: str | None
    observation_digest: str

    def __post_init__(self) -> None:
        _digest(self.cue_digest, "observation cue digest")
        if not isinstance(self.disposition, PositiveProseDisposition):
            raise TypeError("positive prose disposition differs")
        if self.disposition is PositiveProseDisposition.ERROR:
            if (
                self.interval is not None
                or type(self.error_code) is not str
                or _CODE.fullmatch(self.error_code) is None
                or type(self.error_type) is not str
                or _CODE.fullmatch(self.error_type) is None
            ):
                raise PositiveProseObserverError("error observation differs")
        elif (
            type(self.interval) is not PositiveProseScoreInterval
            or self.error_code is not None
            or self.error_type is not None
            or classify_positive_prose_interval(self.interval) is not self.disposition
        ):
            raise PositiveProseObserverError("scored observation differs")
        _digest(self.observation_digest, "observation digest")
        if self.observation_digest != canonical_digest(_observation_content(self)):
            raise PositiveProseObserverError("observation digest differs")

    @classmethod
    def from_interval(
        cls, cue_digest: str, interval: PositiveProseScoreInterval
    ) -> "PositiveProseObservation":
        values = {
            "cue_digest": cue_digest,
            "disposition": classify_positive_prose_interval(interval),
            "interval": interval,
            "error_code": None,
            "error_type": None,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(**values, observation_digest=canonical_digest(_observation_content(provisional)))

    @classmethod
    def error(
        cls, cue_digest: str, error_code: str, error_type: str
    ) -> "PositiveProseObservation":
        values = {
            "cue_digest": cue_digest,
            "disposition": PositiveProseDisposition.ERROR,
            "interval": None,
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
    def from_data(cls, value: object) -> "PositiveProseObservation":
        raw = _fields(
            value,
            {
                "schema", "cue_digest", "disposition", "interval", "error_code",
                "error_type", "scale_digest", "observation_digest",
            },
            "positive prose observation",
        )
        if raw["schema"] != POSITIVE_PROSE_OBSERVATION_SCHEMA or raw[
            "scale_digest"
        ] != positive_prose_scale_digest():
            raise PositiveProseObserverError("observation policy differs")
        try:
            disposition = PositiveProseDisposition(raw["disposition"])
        except Exception as exc:
            raise PositiveProseObserverError("observation disposition is unknown") from exc
        result = cls(
            raw["cue_digest"], disposition,
            None if raw["interval"] is None else PositiveProseScoreInterval.from_data(raw["interval"]),
            raw["error_code"], raw["error_type"], raw["observation_digest"],
        )
        if result.to_data() != dict(raw):
            raise PositiveProseObserverError("observation is not canonical")
        return result


def _parse_payload(value: object, cue_digest: str) -> PositiveProseObservation:
    raw = _fields(value, {"lower", "upper"}, "positive prose payload")
    return PositiveProseObservation.from_interval(
        cue_digest, PositiveProseScoreInterval(raw["lower"], raw["upper"])
    )


def _receipt_data(value: CodexReceipt | None) -> object:
    return None if value is None else value.to_dict()


def _optional_receipt_from_data(value: object) -> CodexReceipt | None:
    return None if value is None else _receipt_from_data(value)


def _artifact_content(value: "PositiveProsePanelArtifact") -> dict[str, object]:
    return {
        "schema": POSITIVE_PROSE_ARTIFACT_SCHEMA,
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
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class PositiveProsePanelArtifact:
    request: PositiveProsePanelRequest
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
    observation: PositiveProseObservation
    failure_code: str | None
    failure_type: str | None
    artifact_digest: str

    @property
    def benchmark_sealable(self) -> bool:
        return (
            self.status is PrototypeSceneObserverStatus.SUCCESS
            and self.transport_provenance.benchmark_sealable
            and self.request.cue.source_proposer_benchmark_sealable
        )

    def __post_init__(self) -> None:
        if type(self.request) is not PositiveProsePanelRequest:
            raise TypeError("artifact needs exact positive prose request")
        for label, item in (
            ("source digest", self.source_digest),
            ("protocol digest", self.protocol_digest),
            ("prompt digest", self.prompt_digest),
            ("output schema digest", self.output_schema_digest),
            ("artifact digest", self.artifact_digest),
        ):
            _digest(item, label)
        if type(self.transport_provenance) is not PositiveProseTransportProvenance:
            raise TypeError("artifact transport provenance differs")
        prompt = positive_prose_panel_prompt(self.request)
        schema = positive_prose_panel_output_schema(self.request)
        if (
            self.source_digest != panel_positive_prose_observer_source_digest()
            or self.protocol_digest != positive_prose_protocol_digest()
            or self.prompt_digest != hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            or self.output_schema_digest != canonical_digest(schema)
            or type(self.presentation) is not tuple
            or len(self.presentation) != 1
            or self.presentation[0].name != "panel.png"
            or self.presentation[0].content_digest != self.request.context.panel_png_digest
            or self.presentation[0].byte_count != self.request.context.panel_png_byte_count
            or self.physical_call_count != 1
            or not isinstance(self.status, PrototypeSceneObserverStatus)
            or type(self.observation) is not PositiveProseObservation
            or self.observation.cue_digest != self.request.cue.cue_digest
        ):
            raise PositiveProseObserverError("artifact binding differs")
        if self.model_payload is not None:
            object.__setattr__(self, "model_payload", _canonical_payload(self.model_payload))
        if self.status is PrototypeSceneObserverStatus.SUCCESS:
            if (
                self.model_payload is None
                or self.receipt is None
                or self.failure_code is not None
                or self.failure_type is not None
                or self.observation != _parse_payload(
                    self.model_payload, self.request.cue.cue_digest
                )
            ):
                raise PositiveProseObserverError("successful artifact differs")
        elif self.status in {
            PrototypeSceneObserverStatus.PARSER_ERROR,
            PrototypeSceneObserverStatus.TRANSPORT_ERROR,
        }:
            if (
                self.observation.disposition is not PositiveProseDisposition.ERROR
                or type(self.failure_code) is not str
                or _CODE.fullmatch(self.failure_code) is None
                or type(self.failure_type) is not str
                or _CODE.fullmatch(self.failure_type) is None
                or self.observation.error_code != self.failure_code
                or self.observation.error_type != self.failure_type
            ):
                raise PositiveProseObserverError("failed artifact lacks typed error")
            if self.status is PrototypeSceneObserverStatus.PARSER_ERROR and (
                self.model_payload is None or self.receipt is None
            ):
                raise PositiveProseObserverError("parser error lacks receipted payload")
            if self.status is PrototypeSceneObserverStatus.TRANSPORT_ERROR and (
                self.model_payload is not None or self.receipt is not None
            ):
                raise PositiveProseObserverError("transport error contains payload")
        else:
            raise PositiveProseObserverError("artifact status is unsupported")
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
                raise PositiveProseObserverError("artifact receipt binding differs")
        if self.artifact_digest != canonical_digest(_artifact_content(self)):
            raise PositiveProseObserverError("artifact digest differs")

    def to_data(self) -> dict[str, object]:
        return {**_artifact_content(self), "artifact_digest": self.artifact_digest}

    @classmethod
    def from_data(cls, value: object) -> "PositiveProsePanelArtifact":
        expected = set(_artifact_content_fields()) | {"artifact_digest"}
        raw = _fields(value, expected, "positive prose artifact")
        if (
            raw["schema"] != POSITIVE_PROSE_ARTIFACT_SCHEMA
            or raw["model_visible_image_names"] != ["panel.png"]
            or any(raw[key] != item for key, item in _authority_data().items())
            or type(raw["presentation"]) is not list
        ):
            raise PositiveProseObserverError("artifact policy differs")
        try:
            status = PrototypeSceneObserverStatus(raw["status"])
        except Exception as exc:
            raise PositiveProseObserverError("artifact status is unknown") from exc
        result = cls(
            PositiveProsePanelRequest.from_data(raw["request"]),
            raw["source_digest"], raw["protocol_digest"],
            PositiveProseTransportProvenance.from_data(raw["transport_provenance"]),
            raw["prompt_digest"], raw["output_schema_digest"],
            tuple(PrototypeImageIdentity.from_data(item) for item in raw["presentation"]),
            raw["physical_call_count"], status, raw["model_payload"],
            _optional_receipt_from_data(raw["receipt"]),
            PositiveProseObservation.from_data(raw["observation"]),
            raw["failure_code"], raw["failure_type"], raw["artifact_digest"],
        )
        if result.request.request_digest != raw["request_digest"]:
            raise PositiveProseObserverError("artifact request digest differs")
        if result.benchmark_sealable is not raw["benchmark_sealable"]:
            raise PositiveProseObserverError("artifact sealability differs")
        if result.to_data() != dict(raw):
            raise PositiveProseObserverError("artifact is not canonical")
        return result


def _artifact_content_fields() -> tuple[str, ...]:
    return (
        "schema", "request", "request_digest", "source_digest", "protocol_digest",
        "transport_provenance", "prompt_digest", "output_schema_digest",
        "presentation", "physical_call_count", "status", "model_payload", "receipt",
        "observation", "failure_code", "failure_type", "benchmark_sealable",
        "model_visible_image_names", *_authority_data(),
    )


def _seal_artifact(
    *,
    request: PositiveProsePanelRequest,
    provenance: PositiveProseTransportProvenance,
    presentation: tuple[PrototypeImageIdentity, ...],
    status: PrototypeSceneObserverStatus,
    payload: Mapping[str, Any] | None,
    receipt: CodexReceipt | None,
    observation: PositiveProseObservation,
    failure_code: str | None,
    failure_type: str | None,
) -> PositiveProsePanelArtifact:
    values = {
        "request": request,
        "source_digest": panel_positive_prose_observer_source_digest(),
        "protocol_digest": positive_prose_protocol_digest(),
        "transport_provenance": provenance,
        "prompt_digest": hashlib.sha256(
            positive_prose_panel_prompt(request).encode("utf-8")
        ).hexdigest(),
        "output_schema_digest": canonical_digest(
            positive_prose_panel_output_schema(request)
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
    provisional = object.__new__(PositiveProsePanelArtifact)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return PositiveProsePanelArtifact(
        **values, artifact_digest=canonical_digest(_artifact_content(provisional))
    )


def observe_positive_prose_panel(
    panel_png: bytes,
    *,
    request: PositiveProsePanelRequest,
    source_proposer_artifact: SupportPositiveProposerArtifact,
    expected_source_proposer_artifact_digest: str,
    minutes: int = 15,
    verbose: bool = False,
    executable: str = "codex",
    cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot | None,
    expected_launcher_digest: str,
    model_catalog_snapshot: CodexModelCatalogSnapshot,
    no_tools_attestation: CodexNoToolsAttestation,
    transport=run_codex_named_images_structured,
) -> PositiveProsePanelArtifact:
    """Observe one panel using one frozen positive cue and one physical call."""

    panel = _exact_png(panel_png)
    if type(request) is not PositiveProsePanelRequest:
        raise TypeError("observer needs exact PositiveProsePanelRequest")
    proposer = _restore_admitted_proposer(
        source_proposer_artifact,
        expected_artifact_digest=expected_source_proposer_artifact_digest,
    )
    _verify_request_proposer_lineage(request, proposer)
    if not callable(transport):
        raise TypeError("positive prose transport must be callable")
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
        raise PositiveProseObserverError("request belongs to another panel or runtime")
    prompt = positive_prose_panel_prompt(request)
    schema = positive_prose_panel_output_schema(request)
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
            POSITIVE_ORIENTATION,
        ),
        # "side" is permitted only as ordinary visual language inside the
        # frozen cue (for example, "four straight sides").  No dataset-side
        # identifier is model-visible.
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
            provenance = _transport_provenance(transport)
        except Exception:
            provenance = (
                PositiveProseTransportProvenance.create("production_direct")
                if transport is run_codex_named_images_structured
                else PositiveProseTransportProvenance.create("injected_unverified")
            )
        return _seal_artifact(
            request=request, provenance=provenance, presentation=presentation,
            status=PrototypeSceneObserverStatus.TRANSPORT_ERROR,
            payload=None, receipt=None,
            observation=PositiveProseObservation.error(
                request.cue.cue_digest, "observer_transport_failed", error_type
            ),
            failure_code="observer_transport_failed", failure_type=error_type,
        )
    provenance = _transport_provenance(transport)
    try:
        observation = _parse_payload(payload, request.cue.cue_digest)
    except Exception as exc:
        error_type = _scene_runtime._exception_type(exc)
        return _seal_artifact(
            request=request, provenance=provenance, presentation=presentation,
            status=PrototypeSceneObserverStatus.PARSER_ERROR,
            payload=payload, receipt=receipt,
            observation=PositiveProseObservation.error(
                request.cue.cue_digest, "observer_payload_rejected", error_type
            ),
            failure_code="observer_payload_rejected", failure_type=error_type,
        )
    return _seal_artifact(
        request=request, provenance=provenance, presentation=presentation,
        status=PrototypeSceneObserverStatus.SUCCESS, payload=payload, receipt=receipt,
        observation=observation, failure_code=None, failure_type=None,
    )


def verify_positive_prose_panel_artifact(
    artifact: PositiveProsePanelArtifact,
    panel_png: bytes,
    *,
    expected_artifact_digest: str,
    source_proposer_artifact: SupportPositiveProposerArtifact,
    expected_source_proposer_artifact_digest: str,
    query_journal_terminal: ObjectBongardTurnJournalSummary | None = None,
    expected_request_digest: str | None = None,
) -> PositiveProsePanelArtifact:
    """Cold replay query pixels, frozen proposer lineage, receipt, and projection."""

    if type(artifact) is not PositiveProsePanelArtifact:
        raise TypeError("cold replay needs PositiveProsePanelArtifact")
    restored = PositiveProsePanelArtifact.from_data(artifact.to_data())
    if restored.artifact_digest != _digest(
        expected_artifact_digest, "expected positive prose artifact digest"
    ):
        raise PositiveProseObserverError("artifact differs from commitment")
    _verify_external_journal_terminal(
        restored.transport_provenance, query_journal_terminal
    )
    if expected_request_digest is not None and restored.request.request_digest != _digest(
        expected_request_digest, "expected positive prose request digest"
    ):
        raise PositiveProseObserverError("request differs from commitment")
    proposer = _restore_admitted_proposer(
        source_proposer_artifact,
        expected_artifact_digest=expected_source_proposer_artifact_digest,
    )
    _verify_request_proposer_lineage(restored.request, proposer)
    panel = _exact_png(panel_png)
    context = restored.request.context
    if (
        hashlib.sha256(panel).hexdigest() != context.panel_png_digest
        or len(panel) != context.panel_png_byte_count
    ):
        raise PositiveProseObserverError("cold replay panel differs")
    if restored.receipt is not None:
        assert restored.model_payload is not None
        prompt = positive_prose_panel_prompt(restored.request)
        schema = positive_prose_panel_output_schema(restored.request)
        with tempfile.TemporaryDirectory(prefix="bongard-positive-prose-replay-") as raw:
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
                raise PositiveProseObserverError("cold replay panel changed")
    return restored


__all__ = (
    "MAX_POSITIVE_CUE_BYTES",
    "POSITIVE_ORIENTATION",
    "POSITIVE_PROSE_PROTOCOL_ID",
    "POSITIVE_PROSE_SCORE_ANCHORS",
    "PositiveProseCue",
    "PositiveProseDisposition",
    "PositiveProseObservation",
    "PositiveProseObserverError",
    "PositiveProsePanelArtifact",
    "PositiveProsePanelContext",
    "PositiveProsePanelRequest",
    "PositiveProseScoreInterval",
    "PositiveProseTransportProvenance",
    "classify_positive_prose_interval",
    "observe_positive_prose_panel",
    "panel_positive_prose_observer_source_digest",
    "positive_prose_panel_output_schema",
    "positive_prose_panel_prompt",
    "positive_prose_protocol_digest",
    "positive_prose_scale_digest",
    "verify_positive_prose_panel_artifact",
)
