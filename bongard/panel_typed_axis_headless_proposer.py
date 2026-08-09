"""Support-only headless Codex nominations for the fixed typed-axis slate.

The model sees exactly six ``primary`` and six ``contrast`` support PNGs under
neutral ordinal names.  It returns one closed-domain value or typed gap for
each of the eight fixed axes, plus one bounded affirmative description.  The
description is inert prose.  Nominations are hints only: they have no
candidate-selection authority and this module never derives, filters, ranks,
or serializes a formula inventory.

Benchmark-sealable execution requires the shared exactly-once ObjectBongard
named-image journal.  An injected callable remains useful for unit tests, but
is recorded as unverified and unsealable.  Cold replay validates source,
runtime, prompt, schema, pixels, payload, receipt, and an externally supplied
journal terminal without invoking a model.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
import hashlib
import re
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence, TypeAlias

from bongard import prototype_scene_observer as _scene_runtime
from bongard.canonical import canonical_digest, canonical_json
from bongard.object_bongard_turn_journal import (
    ObjectBongardNamedImageTurnJournalTransport,
    ObjectBongardTurnCallFailed,
    ObjectBongardTurnJournalSummary,
    ObjectBongardTurnRuntime,
    object_bongard_turn_journal_source_digest,
)
from bongard.panel_probe_transport import (
    call_panel_probe,
    panel_probe_transport_source_digest,
)
from bongard.panel_typed_axis_slate_v2 import (
    AXES,
    AXIS_DOMAINS,
    AXIS_COUNT,
    Axis,
    AxisNomination,
    TypedNominationSlate,
    TypedSupportMatrix,
    typed_axis_slate_algorithm_digest,
)
from bongard.panel_typed_codex_observer import (
    TypedCodexRuntimeBinding,
    _bind_runtime,
    _canonical_payload,
    _digest,
    _exact_png,
    _receipt_from_data,
    _validate_receipt_binding,
)
from bongard.prototype_scene_observer import PrototypeImageIdentity
from bongard.transport import (
    CloudPolicyCacheSnapshot,
    CodexModelCatalogSnapshot,
    CodexReceipt,
    run_codex_named_images_structured,
    validate_codex_named_image_receipt,
    validate_codex_strict_output_schema,
)


HEADLESS_TYPED_AXIS_PROTOCOL_ID = (
    "bongard.panel-typed-axis-headless-proposer/twelve-support-hints-v1"
)
HEADLESS_TYPED_AXIS_RUNTIME_SCHEMA = (
    "gkm.bongard-typed-axis-headless-runtime.v1"
)
HEADLESS_TYPED_AXIS_REQUEST_SCHEMA = (
    "gkm.bongard-typed-axis-headless-request.v1"
)
HEADLESS_TYPED_AXIS_OUTCOME_SCHEMA = (
    "gkm.bongard-typed-axis-headless-outcome.v1"
)
HEADLESS_TYPED_AXIS_ARTIFACT_SCHEMA = (
    "gkm.bongard-typed-axis-headless-artifact.v1"
)
HEADLESS_TYPED_AXIS_ATTEMPT_ERROR_SCHEMA = (
    "gkm.bongard-typed-axis-headless-attempt-error.v1"
)
HEADLESS_TYPED_AXIS_TRANSPORT_PROVENANCE_SCHEMA = (
    "gkm.bongard-typed-axis-headless-transport-provenance.v1"
)
SUPPORT_ROLE_SIZE = 6
SUPPORT_IMAGE_COUNT = 12
PRIMARY_PRESENTATION_NAMES = tuple(
    f"primary_{index:02d}.png" for index in range(SUPPORT_ROLE_SIZE)
)
CONTRAST_PRESENTATION_NAMES = tuple(
    f"contrast_{index:02d}.png" for index in range(SUPPORT_ROLE_SIZE)
)
HEADLESS_TYPED_AXIS_PRESENTATION_NAMES = (
    PRIMARY_PRESENTATION_NAMES + CONTRAST_PRESENTATION_NAMES
)

GAP_VALUE_TOKEN = "gap"
NO_GAP_REASON_TOKEN = "none"
TYPED_GAP_REASON_CODES = (
    "ambiguous_visible_evidence",
    "outside_closed_domain",
)
MAX_POSITIVE_DESCRIPTION_UTF8_BYTES = 768
MAX_PROMPT_UTF8_BYTES = 32 * 1024
MAX_RESPONSE_UTF8_BYTES = 8 * 1024

_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_RAW_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_EXCEPTION_TYPE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]{0,255}\Z")
_FORBIDDEN_DESCRIPTION = re.compile(
    r"(?:```|`|[{};]|[<>]=?|[\u2264\u2265]|%|"
    r"\b(?:not|no|without|lack|lacks|lacking|absence|absent|negative|"
    r"negation|opposite|foil|complement|contrast|threshold|cutoff|score|"
    r"probability|confidence|predicate|formula|candidate|query|lean|"
    r"class|dataset|task|phase|side|group|primary|support|panel|image|drawing|"
    r"def|lambda|import|exec|eval|return|function|python|javascript|sql|regex)\b)",
    re.IGNORECASE,
)


class HeadlessTypedAxisProposerError(ValueError):
    """A support presentation, output, custody record, or replay is invalid."""


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(type(key) is not str for key in value)
        or set(value) != expected
    ):
        raise HeadlessTypedAxisProposerError(f"{label} fields differ")
    return value


def _raw_digest(value: object, label: str) -> str:
    if type(value) is not str or _RAW_DIGEST.fullmatch(value) is None:
        raise HeadlessTypedAxisProposerError(
            f"{label} must be 64 lowercase hexadecimal characters"
        )
    return value


def _address(value: object, label: str) -> str:
    if type(value) is not str or _ADDRESS.fullmatch(value) is None:
        raise HeadlessTypedAxisProposerError(f"{label} must be a sha256: address")
    return value


def panel_typed_axis_headless_proposer_source_digest() -> str:
    """Return the authenticated source bytes loaded for this adapter."""

    return verify_loaded_source(
        __name__, expected_source_sha256=_LOADED_SOURCE_SHA256
    )


def _positive_description(value: object) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise HeadlessTypedAxisProposerError(
            "positive description must be nonempty trimmed prose"
        )
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise HeadlessTypedAxisProposerError(
            "positive description must be UTF-8"
        ) from exc
    if (
        len(encoded) > MAX_POSITIVE_DESCRIPTION_UTF8_BYTES
        or any(ord(character) < 32 for character in value)
    ):
        raise HeadlessTypedAxisProposerError(
            "positive description exceeds its inert prose bound"
        )
    if _FORBIDDEN_DESCRIPTION.search(value) is not None:
        raise HeadlessTypedAxisProposerError(
            "positive description contains forbidden policy, negation, code, or identifiers"
        )
    return value


def _exception_type(exception: Exception) -> str:
    value = f"{type(exception).__module__}.{type(exception).__qualname__}"
    return value if _EXCEPTION_TYPE.fullmatch(value) is not None else "builtins.Exception"


def _exception_detail_digest(exception: Exception) -> str:
    return canonical_digest(
        {
            "schema": "gkm.bongard-typed-axis-attempt-exception.v1",
            "source_exception_type": _exception_type(exception),
            "message": str(exception),
        }
    )


def _freeze_role(value: Sequence[bytes], label: str) -> tuple[bytes, ...]:
    if isinstance(value, (str, bytes, bytearray)) or len(value) != SUPPORT_ROLE_SIZE:
        raise HeadlessTypedAxisProposerError(f"{label} must contain exactly six PNGs")
    return tuple(
        _exact_png(item, f"{label} PNG {index}")
        for index, item in enumerate(value)
    )


def _encode_axis_value(axis: Axis, value: int | str) -> str:
    if value not in AXIS_DOMAINS[axis] or (
        isinstance(value, bool) and any(type(item) is int for item in AXIS_DOMAINS[axis])
    ):
        raise HeadlessTypedAxisProposerError(
            f"value lies outside closed {axis.value} domain"
        )
    return f"count_{value}" if type(value) is int else value


def _encoded_axis_domain(axis: Axis) -> tuple[str, ...]:
    return tuple(_encode_axis_value(axis, value) for value in AXIS_DOMAINS[axis])


def _decode_axis_value(axis: Axis, value: object) -> int | str:
    if type(value) is not str:
        raise HeadlessTypedAxisProposerError(
            f"{axis.value} nomination must use its schema-safe string token"
        )
    encoded = _encoded_axis_domain(axis)
    try:
        index = encoded.index(value)
    except ValueError as exc:
        raise HeadlessTypedAxisProposerError(
            f"{axis.value} nomination lies outside its closed domain"
        ) from exc
    return AXIS_DOMAINS[axis][index]


@dataclass(frozen=True, slots=True)
class HeadlessTypedAxisRuntimeBinding:
    """Serializable identity of the complete ObjectBongard turn runtime."""

    typed_runtime: TypedCodexRuntimeBinding
    minutes: int
    verbose: bool
    executable: str
    cloud_policy_cache_snapshot_present: bool
    model_catalog_canonical_digest: str
    transport_source_digest: str

    def __post_init__(self) -> None:
        if type(self.typed_runtime) is not TypedCodexRuntimeBinding:
            raise TypeError("headless runtime needs an exact typed runtime")
        if type(self.minutes) is not int or not 1 <= self.minutes <= 120:
            raise HeadlessTypedAxisProposerError("runtime minutes must lie in 1..120")
        if type(self.verbose) is not bool:
            raise HeadlessTypedAxisProposerError("runtime verbosity differs")
        if type(self.executable) is not str or not self.executable:
            raise HeadlessTypedAxisProposerError("runtime executable differs")
        if type(self.cloud_policy_cache_snapshot_present) is not bool:
            raise HeadlessTypedAxisProposerError("runtime policy snapshot flag differs")
        _raw_digest(
            self.model_catalog_canonical_digest,
            "runtime model catalog canonical digest",
        )
        _raw_digest(self.transport_source_digest, "runtime transport source digest")
        if self.cloud_policy_cache_snapshot_present is (
            self.typed_runtime.cloud_policy_cache_binding == "absent"
        ):
            raise HeadlessTypedAxisProposerError(
                "runtime policy snapshot presence differs from its binding"
            )

    @classmethod
    def from_runtime(
        cls, runtime: ObjectBongardTurnRuntime
    ) -> "HeadlessTypedAxisRuntimeBinding":
        if type(runtime) is not ObjectBongardTurnRuntime:
            raise TypeError("runtime must be an exact ObjectBongardTurnRuntime")
        typed = _bind_runtime(
            model=runtime.model,
            reasoning_effort=runtime.reasoning_effort,
            expected_launcher_digest=runtime.expected_launcher_digest,
            cloud_policy_cache_snapshot=runtime.cloud_policy_cache_snapshot,
            model_catalog_snapshot=runtime.model_catalog_snapshot,
            no_tools_attestation=runtime.no_tools_attestation,
        )
        result = cls(
            typed,
            runtime.minutes,
            runtime.verbose,
            runtime.executable,
            runtime.cloud_policy_cache_snapshot is not None,
            runtime.model_catalog_snapshot.canonical_digest,
            runtime.transport_source_digest,
        )
        expected = {
            **typed.to_data(),
            "minutes": runtime.minutes,
            "verbose": runtime.verbose,
            "executable": runtime.executable,
            "cloud_policy_cache_snapshot_present": (
                runtime.cloud_policy_cache_snapshot is not None
            ),
            "model_catalog_canonical_digest": (
                runtime.model_catalog_snapshot.canonical_digest
            ),
            "transport_source_digest": runtime.transport_source_digest,
        }
        actual = {
            **result.typed_runtime.to_data(),
            "minutes": result.minutes,
            "verbose": result.verbose,
            "executable": result.executable,
            "cloud_policy_cache_snapshot_present": (
                result.cloud_policy_cache_snapshot_present
            ),
            "model_catalog_canonical_digest": result.model_catalog_canonical_digest,
            "transport_source_digest": result.transport_source_digest,
        }
        if actual != expected:  # pragma: no cover - constructor policy guard
            raise HeadlessTypedAxisProposerError("runtime binding projection differs")
        return result

    @property
    def runtime_digest(self) -> str:
        return canonical_digest(self.to_data())

    def to_data(self) -> dict[str, object]:
        return {
            "schema": HEADLESS_TYPED_AXIS_RUNTIME_SCHEMA,
            "typed_runtime": self.typed_runtime.to_data(),
            "minutes": self.minutes,
            "verbose": self.verbose,
            "executable": self.executable,
            "cloud_policy_cache_snapshot_present": (
                self.cloud_policy_cache_snapshot_present
            ),
            "model_catalog_canonical_digest": self.model_catalog_canonical_digest,
            "transport_source_digest": self.transport_source_digest,
        }

    @classmethod
    def from_data(cls, value: object) -> "HeadlessTypedAxisRuntimeBinding":
        raw = _fields(
            value,
            {
                "schema",
                "typed_runtime",
                "minutes",
                "verbose",
                "executable",
                "cloud_policy_cache_snapshot_present",
                "model_catalog_canonical_digest",
                "transport_source_digest",
            },
            "headless typed-axis runtime",
        )
        if raw["schema"] != HEADLESS_TYPED_AXIS_RUNTIME_SCHEMA:
            raise HeadlessTypedAxisProposerError("headless runtime schema differs")
        result = cls(
            TypedCodexRuntimeBinding.from_data(raw["typed_runtime"]),
            raw["minutes"],
            raw["verbose"],
            raw["executable"],
            raw["cloud_policy_cache_snapshot_present"],
            raw["model_catalog_canonical_digest"],
            raw["transport_source_digest"],
        )
        if canonical_json(result.to_data()) != canonical_json(dict(raw)):
            raise HeadlessTypedAxisProposerError("headless runtime is not canonical")
        return result

    def matches(self, runtime: ObjectBongardTurnRuntime) -> bool:
        return type(runtime) is ObjectBongardTurnRuntime and self == type(self).from_runtime(
            runtime
        )


@dataclass(frozen=True, slots=True)
class HeadlessTypedAxisProposerRequest:
    """Frozen matrix/pixel/runtime request; row identifiers are never model-visible."""

    runtime: HeadlessTypedAxisRuntimeBinding
    support_matrix_address: str
    typed_axis_algorithm_digest: str
    presentation: tuple[PrototypeImageIdentity, ...]

    def __post_init__(self) -> None:
        if type(self.runtime) is not HeadlessTypedAxisRuntimeBinding:
            raise TypeError("request needs an exact headless runtime")
        _address(self.support_matrix_address, "support matrix address")
        if self.typed_axis_algorithm_digest != typed_axis_slate_algorithm_digest():
            raise HeadlessTypedAxisProposerError(
                "request closed typed-axis algorithm differs"
            )
        if (
            type(self.presentation) is not tuple
            or len(self.presentation) != SUPPORT_IMAGE_COUNT
            or any(type(item) is not PrototypeImageIdentity for item in self.presentation)
            or tuple(item.name for item in self.presentation)
            != HEADLESS_TYPED_AXIS_PRESENTATION_NAMES
        ):
            raise HeadlessTypedAxisProposerError(
                "request must bind the exact neutral 6+6 presentation"
            )

    @classmethod
    def build(
        cls,
        primary_pngs: Sequence[bytes],
        contrast_pngs: Sequence[bytes],
        *,
        matrix: TypedSupportMatrix,
        runtime: ObjectBongardTurnRuntime,
    ) -> "HeadlessTypedAxisProposerRequest":
        if type(matrix) is not TypedSupportMatrix:
            raise TypeError("request needs an exact typed support matrix")
        primary = _freeze_role(primary_pngs, "primary support")
        contrast = _freeze_role(contrast_pngs, "contrast support")
        presentation = tuple(
            PrototypeImageIdentity(name, len(raw), hashlib.sha256(raw).hexdigest())
            for name, raw in zip(
                HEADLESS_TYPED_AXIS_PRESENTATION_NAMES,
                (*primary, *contrast),
                strict=True,
            )
        )
        return cls(
            HeadlessTypedAxisRuntimeBinding.from_runtime(runtime),
            matrix.matrix_address,
            typed_axis_slate_algorithm_digest(),
            presentation,
        )

    @property
    def request_digest(self) -> str:
        return canonical_digest(self.to_data())

    def to_data(self) -> dict[str, object]:
        return {
            "schema": HEADLESS_TYPED_AXIS_REQUEST_SCHEMA,
            "protocol_id": HEADLESS_TYPED_AXIS_PROTOCOL_ID,
            "runtime": self.runtime.to_data(),
            "support_matrix_address": self.support_matrix_address,
            "typed_axis_algorithm_digest": self.typed_axis_algorithm_digest,
            "axis_order": [axis.value for axis in AXES],
            "closed_domains": {
                axis.value: list(AXIS_DOMAINS[axis]) for axis in AXES
            },
            "presentation": [item.to_data() for item in self.presentation],
            "support_role_sizes": [SUPPORT_ROLE_SIZE, SUPPORT_ROLE_SIZE],
            "model_visible_image_names": list(
                HEADLESS_TYPED_AXIS_PRESENTATION_NAMES
            ),
            "model_visible_roles": ["primary", "contrast"],
            "dataset_task_side_row_ids_model_visible": False,
            "query_image_count": 0,
            "candidate_or_formula_material_model_visible": False,
            "model_call_count": 1,
            "candidate_selection_authority": False,
        }

    @classmethod
    def from_data(cls, value: object) -> "HeadlessTypedAxisProposerRequest":
        raw = _fields(
            value,
            {
                "schema",
                "protocol_id",
                "runtime",
                "support_matrix_address",
                "typed_axis_algorithm_digest",
                "axis_order",
                "closed_domains",
                "presentation",
                "support_role_sizes",
                "model_visible_image_names",
                "model_visible_roles",
                "dataset_task_side_row_ids_model_visible",
                "query_image_count",
                "candidate_or_formula_material_model_visible",
                "model_call_count",
                "candidate_selection_authority",
            },
            "headless typed-axis request",
        )
        if (
            raw["schema"] != HEADLESS_TYPED_AXIS_REQUEST_SCHEMA
            or raw["protocol_id"] != HEADLESS_TYPED_AXIS_PROTOCOL_ID
            or raw["axis_order"] != [axis.value for axis in AXES]
            or raw["closed_domains"]
            != {axis.value: list(AXIS_DOMAINS[axis]) for axis in AXES}
            or type(raw["presentation"]) is not list
            or raw["support_role_sizes"] != [SUPPORT_ROLE_SIZE, SUPPORT_ROLE_SIZE]
            or raw["model_visible_image_names"]
            != list(HEADLESS_TYPED_AXIS_PRESENTATION_NAMES)
            or raw["model_visible_roles"] != ["primary", "contrast"]
            or raw["dataset_task_side_row_ids_model_visible"] is not False
            or raw["query_image_count"] != 0
            or raw["candidate_or_formula_material_model_visible"] is not False
            or raw["model_call_count"] != 1
            or raw["candidate_selection_authority"] is not False
        ):
            raise HeadlessTypedAxisProposerError("headless request policy differs")
        result = cls(
            HeadlessTypedAxisRuntimeBinding.from_data(raw["runtime"]),
            raw["support_matrix_address"],
            raw["typed_axis_algorithm_digest"],
            tuple(
                PrototypeImageIdentity.from_data(item)
                for item in raw["presentation"]
            ),
        )
        if canonical_json(result.to_data()) != canonical_json(dict(raw)):
            raise HeadlessTypedAxisProposerError("headless request is not canonical")
        return result


def headless_typed_axis_proposer_prompt(
    request: HeadlessTypedAxisProposerRequest,
) -> str:
    """Return the fixed support-only visual nomination prompt."""

    if type(request) is not HeadlessTypedAxisProposerRequest:
        raise TypeError("prompt needs an exact headless typed-axis request")
    domain_text = "\n".join(
        f"- {axis.value}: {', '.join(_encoded_axis_domain(axis))}"
        for axis in AXES
    )
    prompt = (
        "Inspect exactly twelve complete support drawings. The files "
        f"{', '.join(PRIMARY_PRESENTATION_NAMES)} are the primary supports. "
        f"The files {', '.join(CONTRAST_PRESENTATION_NAMES)} are contrast "
        "supports and may be heterogeneous. For each fixed visual axis below, "
        "return at most one affirmative value that visibly characterizes the "
        "primary supports and is useful in comparison. If the primary supports "
        "do not justify one closed value, return status gap, value gap, and one "
        "of the two non-none gap reason codes. Never invent a value. Integer "
        "counts are encoded as count_N strings.\n"
        f"{domain_text}\n"
        "For every nominated axis return status nominated and gap_reason_code "
        "none. For every gap return value gap. Also return one short affirmative "
        "positive_description of visible primary character, such as a bird-like "
        "silhouette or oblique angles. Keep that description observational and "
        "inert. Do not describe a shared contrast rule, an absence, an opposite, "
        "a decision rule, a cutoff, a polarity, executable code, or any hidden "
        "identifier. Do not derive or name compound hypotheses. Return only the "
        "strict JSON fields. Python validates and decodes the output; your choices "
        "are optional hints and cannot select or remove any downstream option."
    )
    if len(prompt.encode("utf-8")) > MAX_PROMPT_UTF8_BYTES:
        raise HeadlessTypedAxisProposerError("headless prompt exceeds capacity")
    return prompt


def _axis_output_schema(axis: Axis) -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["nominated", "gap"]},
            "value": {
                "type": "string",
                "enum": [*_encoded_axis_domain(axis), GAP_VALUE_TOKEN],
            },
            "gap_reason_code": {
                "type": "string",
                "enum": [NO_GAP_REASON_TOKEN, *TYPED_GAP_REASON_CODES],
            },
        },
        "required": ["status", "value", "gap_reason_code"],
        "additionalProperties": False,
    }


def headless_typed_axis_proposer_output_schema(
    request: HeadlessTypedAxisProposerRequest,
) -> dict[str, object]:
    """Return the strict all-string output schema used by the model turn."""

    if type(request) is not HeadlessTypedAxisProposerRequest:
        raise TypeError("schema needs an exact headless typed-axis request")
    properties: dict[str, object] = {
        axis.value: _axis_output_schema(axis) for axis in AXES
    }
    properties["positive_description"] = {"type": "string"}
    schema: dict[str, object] = {
        "type": "object",
        "properties": properties,
        "required": [axis.value for axis in AXES] + ["positive_description"],
        "additionalProperties": False,
    }
    validate_codex_strict_output_schema(schema)
    return schema


def _parse_nomination(axis: Axis, value: object) -> AxisNomination:
    raw = _fields(
        value,
        {"status", "value", "gap_reason_code"},
        f"{axis.value} nomination payload",
    )
    status = raw["status"]
    token = raw["value"]
    reason = raw["gap_reason_code"]
    if status == "nominated":
        if token == GAP_VALUE_TOKEN or reason != NO_GAP_REASON_TOKEN:
            raise HeadlessTypedAxisProposerError(
                f"nominated {axis.value} has gap fields"
            )
        return AxisNomination.nominate(axis, _decode_axis_value(axis, token))
    if status == "gap":
        if token != GAP_VALUE_TOKEN or reason not in TYPED_GAP_REASON_CODES:
            raise HeadlessTypedAxisProposerError(
                f"gap {axis.value} has nomination fields"
            )
        return AxisNomination.gap(axis, reason)
    raise HeadlessTypedAxisProposerError(f"{axis.value} nomination status differs")


@dataclass(frozen=True, slots=True)
class HeadlessTypedAxisOutcome:
    """Typed nomination hints plus separately bounded inert prose."""

    nomination_slate: TypedNominationSlate
    positive_description: str

    def __post_init__(self) -> None:
        if type(self.nomination_slate) is not TypedNominationSlate:
            raise TypeError("headless outcome needs an exact nomination slate")
        _positive_description(self.positive_description)

    @property
    def outcome_digest(self) -> str:
        return canonical_digest(self.to_data())

    def to_data(self) -> dict[str, object]:
        return {
            "schema": HEADLESS_TYPED_AXIS_OUTCOME_SCHEMA,
            "nomination_slate": self.nomination_slate.to_data(),
            "positive_description": self.positive_description,
            "description_is_bounded_inert_prose": True,
            "candidate_selection_authority": False,
            "formula_or_threshold_produced": False,
            "polarity_or_negative_concept_produced": False,
            "query_material_seen": False,
            "lean_material_produced": False,
        }

    @classmethod
    def from_data(cls, value: object) -> "HeadlessTypedAxisOutcome":
        raw = _fields(
            value,
            {
                "schema",
                "nomination_slate",
                "positive_description",
                "description_is_bounded_inert_prose",
                "candidate_selection_authority",
                "formula_or_threshold_produced",
                "polarity_or_negative_concept_produced",
                "query_material_seen",
                "lean_material_produced",
            },
            "headless typed-axis outcome",
        )
        if (
            raw["schema"] != HEADLESS_TYPED_AXIS_OUTCOME_SCHEMA
            or raw["description_is_bounded_inert_prose"] is not True
            or raw["candidate_selection_authority"] is not False
            or any(
                raw[key] is not False
                for key in (
                    "formula_or_threshold_produced",
                    "polarity_or_negative_concept_produced",
                    "query_material_seen",
                    "lean_material_produced",
                )
            )
        ):
            raise HeadlessTypedAxisProposerError("headless outcome policy differs")
        result = cls(
            TypedNominationSlate.from_data(raw["nomination_slate"]),
            raw["positive_description"],
        )
        if canonical_json(result.to_data()) != canonical_json(dict(raw)):
            raise HeadlessTypedAxisProposerError("headless outcome is not canonical")
        return result


def _parse_outcome(
    payload: Mapping[str, Any], support_matrix_address: str
) -> HeadlessTypedAxisOutcome:
    raw = _fields(
        payload,
        {axis.value for axis in AXES} | {"positive_description"},
        "headless typed-axis payload",
    )
    slate = TypedNominationSlate(
        support_matrix_address,
        tuple(_parse_nomination(axis, raw[axis.value]) for axis in AXES),
    )
    return HeadlessTypedAxisOutcome(slate, _positive_description(raw["positive_description"]))


def _transport_source_binding(kind: str) -> str:
    if kind == "production_exactly_once_journal":
        content: dict[str, object] = {
            "schema": "gkm.bongard-typed-axis-headless-transport-source.v1",
            "kind": kind,
            "neutral_probe_transport_source_digest": (
                panel_probe_transport_source_digest()
            ),
            "journal_source_digest": object_bongard_turn_journal_source_digest(),
            "underlying_named_image_transport_source_digest": (
                _scene_runtime.prototype_scene_transport_source_digest()
            ),
        }
    elif kind == "injected_unverified":
        content = {
            "schema": "gkm.bongard-typed-axis-headless-transport-source.v1",
            "kind": kind,
            "callable_source_identity_verified": False,
            "neutral_probe_transport_source_digest": (
                panel_probe_transport_source_digest()
            ),
        }
    else:
        raise HeadlessTypedAxisProposerError("headless transport kind differs")
    return "sha256:" + canonical_digest(content)


@dataclass(frozen=True, slots=True)
class HeadlessTypedAxisTransportProvenance:
    """Durable journal lineage, or an explicit unsealable injection marker."""

    kind: str
    source_binding: str
    production_transport_chain_verified: bool
    attempt_custody_authenticated: bool
    journal_terminal_status: str | None = None
    journal_manifest_digest: str | None = None
    journal_turn_key: str | None = None
    journal_claim_digest: str | None = None
    journal_result_digest: str | None = None
    journal_outcome_digest: str | None = None
    journal_terminal_record_digest: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {
            "production_exactly_once_journal",
            "injected_unverified",
        }:
            raise HeadlessTypedAxisProposerError("headless transport kind differs")
        if self.source_binding != _transport_source_binding(self.kind):
            raise HeadlessTypedAxisProposerError(
                "headless transport source binding differs"
            )
        journal = self.kind == "production_exactly_once_journal"
        if (
            self.production_transport_chain_verified is not journal
            or self.attempt_custody_authenticated is not journal
        ):
            raise HeadlessTypedAxisProposerError(
                "headless attempt-custody policy differs"
            )
        values = (
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
                for item in values
            ):
                raise HeadlessTypedAxisProposerError(
                    "headless journal terminal provenance differs"
                )
        elif self.journal_terminal_status is not None or any(
            item is not None for item in values
        ):
            raise HeadlessTypedAxisProposerError(
                "injected transport cannot name journal custody"
            )

    @classmethod
    def create(
        cls,
        kind: str,
        *,
        journal_summary: ObjectBongardTurnJournalSummary | None = None,
    ) -> "HeadlessTypedAxisTransportProvenance":
        if kind == "production_exactly_once_journal":
            if (
                type(journal_summary) is not ObjectBongardTurnJournalSummary
                or journal_summary.terminal_status not in {"success", "failure"}
            ):
                raise HeadlessTypedAxisProposerError(
                    "headless journal is not a durable terminal attempt"
                )
            return cls(
                kind,
                _transport_source_binding(kind),
                True,
                True,
                journal_summary.terminal_status,
                journal_summary.manifest_digest,
                journal_summary.turn_key,
                journal_summary.claim_digest,
                journal_summary.result_digest,
                journal_summary.outcome_digest,
                journal_summary.record_digest,
            )
        if journal_summary is not None:
            raise HeadlessTypedAxisProposerError(
                "injected transport received external journal custody"
            )
        return cls(kind, _transport_source_binding(kind), False, False, None)

    def to_data(self) -> dict[str, object]:
        return {
            "schema": HEADLESS_TYPED_AXIS_TRANSPORT_PROVENANCE_SCHEMA,
            "kind": self.kind,
            "source_binding": self.source_binding,
            "production_transport_chain_verified": (
                self.production_transport_chain_verified
            ),
            "attempt_custody_authenticated": self.attempt_custody_authenticated,
            "journal_terminal_status": self.journal_terminal_status,
            "journal_manifest_digest": self.journal_manifest_digest,
            "journal_turn_key": self.journal_turn_key,
            "journal_claim_digest": self.journal_claim_digest,
            "journal_result_digest": self.journal_result_digest,
            "journal_outcome_digest": self.journal_outcome_digest,
            "journal_terminal_record_digest": self.journal_terminal_record_digest,
            "physical_model_call_cold_authenticated": False,
            "benchmark_requires_external_journal_terminal": True,
            "injected_transport_is_unsealable": True,
        }

    @classmethod
    def from_data(cls, value: object) -> "HeadlessTypedAxisTransportProvenance":
        raw = _fields(
            value,
            {
                "schema",
                "kind",
                "source_binding",
                "production_transport_chain_verified",
                "attempt_custody_authenticated",
                "journal_terminal_status",
                "journal_manifest_digest",
                "journal_turn_key",
                "journal_claim_digest",
                "journal_result_digest",
                "journal_outcome_digest",
                "journal_terminal_record_digest",
                "physical_model_call_cold_authenticated",
                "benchmark_requires_external_journal_terminal",
                "injected_transport_is_unsealable",
            },
            "headless typed-axis transport provenance",
        )
        if (
            raw["schema"] != HEADLESS_TYPED_AXIS_TRANSPORT_PROVENANCE_SCHEMA
            or raw["physical_model_call_cold_authenticated"] is not False
            or raw["benchmark_requires_external_journal_terminal"] is not True
            or raw["injected_transport_is_unsealable"] is not True
        ):
            raise HeadlessTypedAxisProposerError(
                "headless transport provenance policy differs"
            )
        result = cls(
            raw["kind"],
            raw["source_binding"],
            raw["production_transport_chain_verified"],
            raw["attempt_custody_authenticated"],
            raw["journal_terminal_status"],
            raw["journal_manifest_digest"],
            raw["journal_turn_key"],
            raw["journal_claim_digest"],
            raw["journal_result_digest"],
            raw["journal_outcome_digest"],
            raw["journal_terminal_record_digest"],
        )
        if canonical_json(result.to_data()) != canonical_json(dict(raw)):
            raise HeadlessTypedAxisProposerError(
                "headless transport provenance is not canonical"
            )
        return result


def _transport_provenance(
    transport: object,
    runtime: ObjectBongardTurnRuntime,
) -> HeadlessTypedAxisTransportProvenance:
    if (
        type(transport) is ObjectBongardNamedImageTurnJournalTransport
        and transport.runtime == runtime
        and getattr(transport, "_underlying_transport", None)
        is run_codex_named_images_structured
        and runtime.transport_source_digest
        == _scene_runtime.prototype_scene_transport_source_digest()
    ):
        return HeadlessTypedAxisTransportProvenance.create(
            "production_exactly_once_journal",
            journal_summary=transport.verify(),
        )
    return HeadlessTypedAxisTransportProvenance.create("injected_unverified")


def _verify_external_journal_terminal(
    provenance: HeadlessTypedAxisTransportProvenance,
    summary: ObjectBongardTurnJournalSummary | None,
) -> None:
    if provenance.kind != "production_exactly_once_journal":
        if summary is not None:
            raise HeadlessTypedAxisProposerError(
                "injected headless artifact received external journal custody"
            )
        return
    if (
        type(summary) is not ObjectBongardTurnJournalSummary
        or (
            summary.terminal_status,
            summary.manifest_digest,
            summary.turn_key,
            summary.claim_digest,
            summary.result_digest,
            summary.outcome_digest,
            summary.record_digest,
        )
        != (
            provenance.journal_terminal_status,
            provenance.journal_manifest_digest,
            provenance.journal_turn_key,
            provenance.journal_claim_digest,
            provenance.journal_result_digest,
            provenance.journal_outcome_digest,
            provenance.journal_terminal_record_digest,
        )
    ):
        raise HeadlessTypedAxisProposerError(
            "external headless proposer journal terminal differs"
        )


def _contract_digest(request: HeadlessTypedAxisProposerRequest) -> str:
    prompt = headless_typed_axis_proposer_prompt(request)
    schema = headless_typed_axis_proposer_output_schema(request)
    return canonical_digest(
        {
            "schema": "gkm.bongard-typed-axis-headless-contract.v1",
            "protocol_id": HEADLESS_TYPED_AXIS_PROTOCOL_ID,
            "proposer_source_sha256": (
                panel_typed_axis_headless_proposer_source_digest()
            ),
            "neutral_probe_transport_source_sha256": (
                panel_probe_transport_source_digest()
            ),
            "journal_source_sha256": object_bongard_turn_journal_source_digest(),
            "underlying_transport_source_sha256": (
                _scene_runtime.prototype_scene_transport_source_digest()
            ),
            "request_digest": request.request_digest,
            "runtime_digest": request.runtime.runtime_digest,
            "typed_axis_algorithm_digest": request.typed_axis_algorithm_digest,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "output_schema_digest": canonical_digest(schema),
            "presentation_names": list(HEADLESS_TYPED_AXIS_PRESENTATION_NAMES),
            "primary_support_count": SUPPORT_ROLE_SIZE,
            "contrast_support_count": SUPPORT_ROLE_SIZE,
            "query_image_count": 0,
            "count_tokens_are_strings": True,
            "one_value_or_typed_gap_per_axis": True,
            "candidate_selection_authority": False,
            "inventory_derivation_present": False,
            "positive_description_enters_inventory": False,
            "proposer_or_artifact_digest_enters_inventory": False,
            "positive_description_enters_candidate_rank_prompt": False,
            "proposer_or_artifact_digest_enters_candidate_rank_prompt": False,
            "nomination_hints_enter_candidate_rank_prompt": False,
            "formula_threshold_polarity_negative_concept_or_lean_present": False,
        }
    )


@dataclass(frozen=True, slots=True)
class HeadlessTypedAxisProposerArtifact:
    """Receipted support-only output with no downstream inventory authority."""

    request: HeadlessTypedAxisProposerRequest
    transport_provenance: HeadlessTypedAxisTransportProvenance
    request_digest: str
    proposer_source_digest: str
    contract_digest: str
    prompt_digest: str
    output_schema_digest: str
    payload_digest: str
    model_payload: Mapping[str, Any]
    codex_receipt: CodexReceipt
    outcome: HeadlessTypedAxisOutcome

    def __post_init__(self) -> None:
        if type(self.request) is not HeadlessTypedAxisProposerRequest:
            raise TypeError("artifact needs its exact headless request")
        if type(self.transport_provenance) is not HeadlessTypedAxisTransportProvenance:
            raise TypeError("artifact needs exact headless transport provenance")
        if (
            self.transport_provenance.kind
            == "production_exactly_once_journal"
            and self.transport_provenance.journal_terminal_status != "success"
        ):
            raise HeadlessTypedAxisProposerError(
                "successful artifact needs a successful physical journal terminal"
            )
        if type(self.outcome) is not HeadlessTypedAxisOutcome:
            raise TypeError("artifact needs an exact headless outcome")
        for label, value in (
            ("request digest", self.request_digest),
            ("proposer source digest", self.proposer_source_digest),
            ("contract digest", self.contract_digest),
            ("prompt digest", self.prompt_digest),
            ("output schema digest", self.output_schema_digest),
            ("payload digest", self.payload_digest),
        ):
            _raw_digest(value, label)
        payload = _canonical_payload(self.model_payload, "headless typed-axis payload")
        object.__setattr__(self, "model_payload", payload)
        prompt = headless_typed_axis_proposer_prompt(self.request)
        schema = headless_typed_axis_proposer_output_schema(self.request)
        if (
            self.request_digest != self.request.request_digest
            or self.proposer_source_digest
            != panel_typed_axis_headless_proposer_source_digest()
            or self.contract_digest != _contract_digest(self.request)
            or self.prompt_digest
            != hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            or self.output_schema_digest != canonical_digest(schema)
            or self.payload_digest != canonical_digest(payload)
        ):
            raise HeadlessTypedAxisProposerError(
                "headless frozen source/runtime/prompt/schema envelope differs"
            )
        parsed = _parse_outcome(payload, self.request.support_matrix_address)
        if parsed != self.outcome:
            raise HeadlessTypedAxisProposerError(
                "headless outcome differs from raw payload projection"
            )
        try:
            _validate_receipt_binding(
                self.codex_receipt,
                runtime=self.request.runtime.typed_runtime,
                prompt_digest=self.prompt_digest,
                output_schema_digest=self.output_schema_digest,
                payload_digest=self.payload_digest,
                presentation=self.request.presentation,
            )
        except Exception as exc:
            raise HeadlessTypedAxisProposerError(
                "headless receipt binding differs"
            ) from exc

    @property
    def benchmark_sealable(self) -> bool:
        return (
            self.transport_provenance.attempt_custody_authenticated
            and self.transport_provenance.journal_terminal_status == "success"
        )

    @property
    def artifact_digest(self) -> str:
        return canonical_digest(self.content_data())

    @property
    def attempt_digest(self) -> str:
        return self.artifact_digest

    def content_data(self) -> dict[str, object]:
        return {
            "schema": HEADLESS_TYPED_AXIS_ARTIFACT_SCHEMA,
            "protocol_id": HEADLESS_TYPED_AXIS_PROTOCOL_ID,
            "request": self.request.to_data(),
            "transport_provenance": self.transport_provenance.to_data(),
            "benchmark_sealable": self.benchmark_sealable,
            "attempt_status": "success",
            "runner_must_bind_attempt": True,
            "attempt_error_is_axis_gap": False,
            "attempt_error_is_negative_evidence": False,
            "request_digest": self.request_digest,
            "proposer_source_digest": self.proposer_source_digest,
            "contract_digest": self.contract_digest,
            "prompt_digest": self.prompt_digest,
            "output_schema_digest": self.output_schema_digest,
            "payload_digest": self.payload_digest,
            "model_payload": dict(self.model_payload),
            "codex_receipt": self.codex_receipt.to_dict(),
            "codex_receipt_digest": self.codex_receipt.receipt_digest,
            "outcome": self.outcome.to_data(),
            "outcome_digest": self.outcome.outcome_digest,
            "support_image_count": SUPPORT_IMAGE_COUNT,
            "query_image_count": 0,
            "model_call_count": 1,
            "model_visible_image_names": list(
                HEADLESS_TYPED_AXIS_PRESENTATION_NAMES
            ),
            "dataset_task_side_row_ids_model_visible": False,
            "count_values_schema_encoded_as_strings": True,
            "candidate_selection_authority": False,
            "inventory_derivation_or_filtering_performed": False,
            "nomination_hints_embedded_in_inventory": False,
            "positive_description_embedded_in_inventory": False,
            "proposer_or_artifact_digest_embedded_in_inventory": False,
            "positive_description_enters_candidate_rank_prompt": False,
            "proposer_or_artifact_digest_enters_candidate_rank_prompt": False,
            "nomination_hints_enter_candidate_rank_prompt": False,
            "candidate_rank_prompt_excludes_all_proposer_material": True,
            "negation_or_negative_concept_present": False,
            "formula_or_threshold_present": False,
            "lean_present": False,
        }

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "artifact_digest": self.artifact_digest}

    @classmethod
    def from_data(cls, value: object) -> "HeadlessTypedAxisProposerArtifact":
        raw = _fields(
            value,
            {
                "schema",
                "protocol_id",
                "request",
                "transport_provenance",
                "benchmark_sealable",
                "attempt_status",
                "runner_must_bind_attempt",
                "attempt_error_is_axis_gap",
                "attempt_error_is_negative_evidence",
                "request_digest",
                "proposer_source_digest",
                "contract_digest",
                "prompt_digest",
                "output_schema_digest",
                "payload_digest",
                "model_payload",
                "codex_receipt",
                "codex_receipt_digest",
                "outcome",
                "outcome_digest",
                "support_image_count",
                "query_image_count",
                "model_call_count",
                "model_visible_image_names",
                "dataset_task_side_row_ids_model_visible",
                "count_values_schema_encoded_as_strings",
                "candidate_selection_authority",
                "inventory_derivation_or_filtering_performed",
                "nomination_hints_embedded_in_inventory",
                "positive_description_embedded_in_inventory",
                "proposer_or_artifact_digest_embedded_in_inventory",
                "positive_description_enters_candidate_rank_prompt",
                "proposer_or_artifact_digest_enters_candidate_rank_prompt",
                "nomination_hints_enter_candidate_rank_prompt",
                "candidate_rank_prompt_excludes_all_proposer_material",
                "negation_or_negative_concept_present",
                "formula_or_threshold_present",
                "lean_present",
                "artifact_digest",
            },
            "headless typed-axis artifact",
        )
        false_fields = (
            "dataset_task_side_row_ids_model_visible",
            "candidate_selection_authority",
            "inventory_derivation_or_filtering_performed",
            "nomination_hints_embedded_in_inventory",
            "positive_description_embedded_in_inventory",
            "proposer_or_artifact_digest_embedded_in_inventory",
            "positive_description_enters_candidate_rank_prompt",
            "proposer_or_artifact_digest_enters_candidate_rank_prompt",
            "nomination_hints_enter_candidate_rank_prompt",
            "negation_or_negative_concept_present",
            "formula_or_threshold_present",
            "lean_present",
        )
        if (
            raw["schema"] != HEADLESS_TYPED_AXIS_ARTIFACT_SCHEMA
            or raw["protocol_id"] != HEADLESS_TYPED_AXIS_PROTOCOL_ID
            or raw["attempt_status"] != "success"
            or raw["runner_must_bind_attempt"] is not True
            or raw["attempt_error_is_axis_gap"] is not False
            or raw["attempt_error_is_negative_evidence"] is not False
            or raw["support_image_count"] != SUPPORT_IMAGE_COUNT
            or raw["query_image_count"] != 0
            or raw["model_call_count"] != 1
            or raw["model_visible_image_names"]
            != list(HEADLESS_TYPED_AXIS_PRESENTATION_NAMES)
            or raw["count_values_schema_encoded_as_strings"] is not True
            or raw["candidate_rank_prompt_excludes_all_proposer_material"] is not True
            or any(raw[field] is not False for field in false_fields)
        ):
            raise HeadlessTypedAxisProposerError("headless artifact policy differs")
        result = cls(
            HeadlessTypedAxisProposerRequest.from_data(raw["request"]),
            HeadlessTypedAxisTransportProvenance.from_data(
                raw["transport_provenance"]
            ),
            raw["request_digest"],
            raw["proposer_source_digest"],
            raw["contract_digest"],
            raw["prompt_digest"],
            raw["output_schema_digest"],
            raw["payload_digest"],
            _canonical_payload(raw["model_payload"], "archived headless payload"),
            _receipt_from_data(raw["codex_receipt"]),
            HeadlessTypedAxisOutcome.from_data(raw["outcome"]),
        )
        if (
            raw["benchmark_sealable"] is not result.benchmark_sealable
            or raw["codex_receipt_digest"] != result.codex_receipt.receipt_digest
            or raw["outcome_digest"] != result.outcome.outcome_digest
            or raw["artifact_digest"] != result.artifact_digest
            or canonical_json(result.to_data()) != canonical_json(dict(raw))
        ):
            raise HeadlessTypedAxisProposerError(
                "headless artifact digest or derived field differs"
            )
        return result


@dataclass(frozen=True, slots=True)
class HeadlessTypedAxisAttemptErrorArtifact:
    """Mandatory non-evidential record of one failed proposer attempt.

    A payload-contract rejection preserves the raw receipted payload and
    deterministically replays the same parser rejection.  A physical turn
    failure has no payload or receipt and is instead bound to the journal's
    durable failure terminal.  Neither case is an axis gap, absence witness,
    negative example, or candidate-selection input.
    """

    request: HeadlessTypedAxisProposerRequest
    transport_provenance: HeadlessTypedAxisTransportProvenance
    request_digest: str
    proposer_source_digest: str
    contract_digest: str
    prompt_digest: str
    output_schema_digest: str
    failure_stage: str
    failure_code: str
    source_exception_type: str
    failure_detail_digest: str
    model_payload: Mapping[str, Any] | None
    payload_digest: str | None
    codex_receipt: CodexReceipt | None

    def __post_init__(self) -> None:
        if type(self.request) is not HeadlessTypedAxisProposerRequest:
            raise TypeError("attempt error needs its exact headless request")
        if type(self.transport_provenance) is not HeadlessTypedAxisTransportProvenance:
            raise TypeError("attempt error needs exact transport provenance")
        for label, value in (
            ("request digest", self.request_digest),
            ("proposer source digest", self.proposer_source_digest),
            ("contract digest", self.contract_digest),
            ("prompt digest", self.prompt_digest),
            ("output schema digest", self.output_schema_digest),
            ("failure detail digest", self.failure_detail_digest),
        ):
            _raw_digest(value, label)
        if (
            type(self.source_exception_type) is not str
            or _EXCEPTION_TYPE.fullmatch(self.source_exception_type) is None
        ):
            raise HeadlessTypedAxisProposerError(
                "attempt error exception type differs"
            )
        prompt = headless_typed_axis_proposer_prompt(self.request)
        schema = headless_typed_axis_proposer_output_schema(self.request)
        if (
            self.request_digest != self.request.request_digest
            or self.proposer_source_digest
            != panel_typed_axis_headless_proposer_source_digest()
            or self.contract_digest != _contract_digest(self.request)
            or self.prompt_digest
            != hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            or self.output_schema_digest != canonical_digest(schema)
        ):
            raise HeadlessTypedAxisProposerError(
                "attempt error source/runtime/prompt/schema envelope differs"
            )

        production_status = self.transport_provenance.journal_terminal_status
        if self.failure_stage == "payload_contract":
            if self.failure_code != "payload_contract_rejected":
                raise HeadlessTypedAxisProposerError(
                    "payload rejection failure code differs"
                )
            if (
                self.model_payload is None
                or type(self.codex_receipt) is not CodexReceipt
                or self.payload_digest is None
                or (
                    self.transport_provenance.kind
                    == "production_exactly_once_journal"
                    and production_status != "success"
                )
            ):
                raise HeadlessTypedAxisProposerError(
                    "payload rejection must preserve a successful receipted turn"
                )
            payload = _canonical_payload(
                self.model_payload, "rejected headless typed-axis payload"
            )
            object.__setattr__(self, "model_payload", payload)
            _raw_digest(self.payload_digest, "rejected payload digest")
            if self.payload_digest != canonical_digest(payload):
                raise HeadlessTypedAxisProposerError(
                    "rejected payload digest differs"
                )
            try:
                _validate_receipt_binding(
                    self.codex_receipt,
                    runtime=self.request.runtime.typed_runtime,
                    prompt_digest=self.prompt_digest,
                    output_schema_digest=self.output_schema_digest,
                    payload_digest=self.payload_digest,
                    presentation=self.request.presentation,
                )
            except Exception as exc:
                raise HeadlessTypedAxisProposerError(
                    "rejected payload receipt binding differs"
                ) from exc
            try:
                _parse_outcome(payload, self.request.support_matrix_address)
            except Exception as replayed:
                if (
                    self.source_exception_type != _exception_type(replayed)
                    or self.failure_detail_digest
                    != _exception_detail_digest(replayed)
                ):
                    raise HeadlessTypedAxisProposerError(
                        "payload rejection parser witness differs"
                    ) from replayed
            else:
                raise HeadlessTypedAxisProposerError(
                    "valid payload cannot be archived as a parser rejection"
                )
        elif self.failure_stage == "physical_turn":
            if (
                self.failure_code != "physical_turn_failed"
                or self.model_payload is not None
                or self.payload_digest is not None
                or self.codex_receipt is not None
                or (
                    self.transport_provenance.kind
                    == "production_exactly_once_journal"
                    and production_status != "failure"
                )
            ):
                raise HeadlessTypedAxisProposerError(
                    "physical failure artifact fields differ"
                )
            if self.transport_provenance.kind == "production_exactly_once_journal":
                expected = ObjectBongardTurnCallFailed(
                    turn_key=self.transport_provenance.journal_turn_key,  # type: ignore[arg-type]
                    failure_digest=self.transport_provenance.journal_result_digest,  # type: ignore[arg-type]
                )
                if (
                    self.source_exception_type != _exception_type(expected)
                    or self.failure_detail_digest
                    != _exception_detail_digest(expected)
                ):
                    raise HeadlessTypedAxisProposerError(
                        "physical journal failure witness differs"
                    )
        else:
            raise HeadlessTypedAxisProposerError("attempt failure stage differs")

    @property
    def benchmark_sealable(self) -> bool:
        return False

    @property
    def attempt_custody_authenticated(self) -> bool:
        return self.transport_provenance.attempt_custody_authenticated

    @property
    def attempt_digest(self) -> str:
        return canonical_digest(self.content_data())

    def content_data(self) -> dict[str, object]:
        payload_present = self.model_payload is not None
        receipt_present = self.codex_receipt is not None
        return {
            "schema": HEADLESS_TYPED_AXIS_ATTEMPT_ERROR_SCHEMA,
            "protocol_id": HEADLESS_TYPED_AXIS_PROTOCOL_ID,
            "attempt_status": "error",
            "request": self.request.to_data(),
            "transport_provenance": self.transport_provenance.to_data(),
            "benchmark_sealable": self.benchmark_sealable,
            "attempt_custody_authenticated": self.attempt_custody_authenticated,
            "request_digest": self.request_digest,
            "proposer_source_digest": self.proposer_source_digest,
            "contract_digest": self.contract_digest,
            "prompt_digest": self.prompt_digest,
            "output_schema_digest": self.output_schema_digest,
            "failure_stage": self.failure_stage,
            "failure_code": self.failure_code,
            "source_exception_type": self.source_exception_type,
            "failure_detail_digest": self.failure_detail_digest,
            "model_payload": (
                None if self.model_payload is None else dict(self.model_payload)
            ),
            "payload_digest": self.payload_digest,
            "codex_receipt": (
                None if self.codex_receipt is None else self.codex_receipt.to_dict()
            ),
            "codex_receipt_digest": (
                None
                if self.codex_receipt is None
                else self.codex_receipt.receipt_digest
            ),
            "raw_receipted_payload_preserved": payload_present and receipt_present,
            "runner_must_bind_attempt": True,
            "attempt_error_is_axis_gap": False,
            "attempt_error_is_negative_evidence": False,
            "attempt_error_can_nominate_or_rank": False,
            "candidate_selection_authority": False,
            "support_image_count": SUPPORT_IMAGE_COUNT,
            "query_image_count": 0,
            "inventory_derivation_or_filtering_performed": False,
            "positive_description_enters_candidate_rank_prompt": False,
            "proposer_or_artifact_digest_enters_candidate_rank_prompt": False,
            "nomination_hints_enter_candidate_rank_prompt": False,
            "formula_threshold_polarity_negative_concept_or_lean_present": False,
        }

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "attempt_digest": self.attempt_digest}

    @classmethod
    def from_data(cls, value: object) -> "HeadlessTypedAxisAttemptErrorArtifact":
        raw = _fields(
            value,
            {
                "schema",
                "protocol_id",
                "attempt_status",
                "request",
                "transport_provenance",
                "benchmark_sealable",
                "attempt_custody_authenticated",
                "request_digest",
                "proposer_source_digest",
                "contract_digest",
                "prompt_digest",
                "output_schema_digest",
                "failure_stage",
                "failure_code",
                "source_exception_type",
                "failure_detail_digest",
                "model_payload",
                "payload_digest",
                "codex_receipt",
                "codex_receipt_digest",
                "raw_receipted_payload_preserved",
                "runner_must_bind_attempt",
                "attempt_error_is_axis_gap",
                "attempt_error_is_negative_evidence",
                "attempt_error_can_nominate_or_rank",
                "candidate_selection_authority",
                "support_image_count",
                "query_image_count",
                "inventory_derivation_or_filtering_performed",
                "positive_description_enters_candidate_rank_prompt",
                "proposer_or_artifact_digest_enters_candidate_rank_prompt",
                "nomination_hints_enter_candidate_rank_prompt",
                "formula_threshold_polarity_negative_concept_or_lean_present",
                "attempt_digest",
            },
            "headless typed-axis attempt error",
        )
        false_fields = (
            "attempt_error_is_axis_gap",
            "attempt_error_is_negative_evidence",
            "attempt_error_can_nominate_or_rank",
            "candidate_selection_authority",
            "inventory_derivation_or_filtering_performed",
            "positive_description_enters_candidate_rank_prompt",
            "proposer_or_artifact_digest_enters_candidate_rank_prompt",
            "nomination_hints_enter_candidate_rank_prompt",
            "formula_threshold_polarity_negative_concept_or_lean_present",
        )
        if (
            raw["schema"] != HEADLESS_TYPED_AXIS_ATTEMPT_ERROR_SCHEMA
            or raw["protocol_id"] != HEADLESS_TYPED_AXIS_PROTOCOL_ID
            or raw["attempt_status"] != "error"
            or raw["runner_must_bind_attempt"] is not True
            or raw["benchmark_sealable"] is not False
            or raw["support_image_count"] != SUPPORT_IMAGE_COUNT
            or raw["query_image_count"] != 0
            or any(raw[field] is not False for field in false_fields)
        ):
            raise HeadlessTypedAxisProposerError(
                "headless attempt error policy differs"
            )
        receipt = (
            None
            if raw["codex_receipt"] is None
            else _receipt_from_data(raw["codex_receipt"])
        )
        result = cls(
            HeadlessTypedAxisProposerRequest.from_data(raw["request"]),
            HeadlessTypedAxisTransportProvenance.from_data(
                raw["transport_provenance"]
            ),
            raw["request_digest"],
            raw["proposer_source_digest"],
            raw["contract_digest"],
            raw["prompt_digest"],
            raw["output_schema_digest"],
            raw["failure_stage"],
            raw["failure_code"],
            raw["source_exception_type"],
            raw["failure_detail_digest"],
            (
                None
                if raw["model_payload"] is None
                else _canonical_payload(
                    raw["model_payload"], "archived rejected headless payload"
                )
            ),
            raw["payload_digest"],
            receipt,
        )
        if (
            raw["benchmark_sealable"] is not result.benchmark_sealable
            or raw["attempt_custody_authenticated"]
            is not result.attempt_custody_authenticated
            or raw["codex_receipt_digest"]
            != (None if receipt is None else receipt.receipt_digest)
            or raw["raw_receipted_payload_preserved"]
            is not (result.model_payload is not None and result.codex_receipt is not None)
            or raw["attempt_digest"] != result.attempt_digest
            or canonical_json(result.to_data()) != canonical_json(dict(raw))
        ):
            raise HeadlessTypedAxisProposerError(
                "headless attempt error digest or derived field differs"
            )
        return result


HeadlessTypedAxisProposerResult: TypeAlias = (
    HeadlessTypedAxisProposerArtifact | HeadlessTypedAxisAttemptErrorArtifact
)


def headless_typed_axis_candidate_rank_prompt_material(
    outcome: HeadlessTypedAxisOutcome,
) -> tuple[object, ...]:
    """Return no rank-prompt material from any proposer output."""

    if type(outcome) is not HeadlessTypedAxisOutcome:
        raise TypeError("rank prompt guard needs an exact headless outcome")
    return ()


def build_headless_typed_axis_turn_journal(
    journal_directory: str | Path,
    primary_pngs: Sequence[bytes],
    contrast_pngs: Sequence[bytes],
    *,
    matrix: TypedSupportMatrix,
    request: HeadlessTypedAxisProposerRequest,
    runtime: ObjectBongardTurnRuntime,
    authorization_digest: str,
    execution_precommit_digest: str,
    task_id: str,
    underlying_transport=run_codex_named_images_structured,
) -> ObjectBongardNamedImageTurnJournalTransport:
    """Construct the exact external journal required for a sealable turn."""

    primary = _freeze_role(primary_pngs, "primary support")
    contrast = _freeze_role(contrast_pngs, "contrast support")
    rebuilt = HeadlessTypedAxisProposerRequest.build(
        primary, contrast, matrix=matrix, runtime=runtime
    )
    if rebuilt != request:
        raise HeadlessTypedAxisProposerError(
            "journal request belongs to other pixels, matrix, or runtime"
        )
    return ObjectBongardNamedImageTurnJournalTransport(
        journal_directory,
        authorization_digest=authorization_digest,
        execution_precommit_digest=execution_precommit_digest,
        task_id=task_id,
        turn_kind="typed_axis_support_narrator",
        expected_prompt=headless_typed_axis_proposer_prompt(request),
        expected_images=tuple(
            zip(
                HEADLESS_TYPED_AXIS_PRESENTATION_NAMES,
                (*primary, *contrast),
                strict=True,
            )
        ),
        expected_output_schema=headless_typed_axis_proposer_output_schema(request),
        runtime=runtime,
        underlying_transport=underlying_transport,
    )


def _attempt_error_artifact(
    request: HeadlessTypedAxisProposerRequest,
    provenance: HeadlessTypedAxisTransportProvenance,
    exception: Exception,
    *,
    failure_stage: str,
    model_payload: Mapping[str, Any] | None = None,
    codex_receipt: CodexReceipt | None = None,
) -> HeadlessTypedAxisAttemptErrorArtifact:
    prompt = headless_typed_axis_proposer_prompt(request)
    schema = headless_typed_axis_proposer_output_schema(request)
    payload = (
        None
        if model_payload is None
        else _canonical_payload(model_payload, "rejected headless typed-axis payload")
    )
    return HeadlessTypedAxisAttemptErrorArtifact(
        request,
        provenance,
        request.request_digest,
        panel_typed_axis_headless_proposer_source_digest(),
        _contract_digest(request),
        hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        canonical_digest(schema),
        failure_stage,
        (
            "payload_contract_rejected"
            if failure_stage == "payload_contract"
            else "physical_turn_failed"
        ),
        _exception_type(exception),
        _exception_detail_digest(exception),
        payload,
        None if payload is None else canonical_digest(payload),
        codex_receipt,
    )


def run_headless_typed_axis_proposer(
    primary_pngs: Sequence[bytes],
    contrast_pngs: Sequence[bytes],
    *,
    matrix: TypedSupportMatrix,
    request: HeadlessTypedAxisProposerRequest,
    runtime: ObjectBongardTurnRuntime,
    transport: object,
) -> HeadlessTypedAxisProposerResult:
    """Return a closed success-or-error result for exactly one admitted attempt."""

    if type(matrix) is not TypedSupportMatrix:
        raise TypeError("headless proposer needs an exact support matrix")
    if type(request) is not HeadlessTypedAxisProposerRequest:
        raise TypeError("headless proposer needs its exact request")
    if type(runtime) is not ObjectBongardTurnRuntime:
        raise TypeError("headless proposer needs an exact ObjectBongard runtime")
    if not callable(transport):
        raise TypeError("headless proposer transport must be callable")
    primary = _freeze_role(primary_pngs, "primary support")
    contrast = _freeze_role(contrast_pngs, "contrast support")
    rebuilt = HeadlessTypedAxisProposerRequest.build(
        primary, contrast, matrix=matrix, runtime=runtime
    )
    if rebuilt != request or not request.runtime.matches(runtime):
        raise HeadlessTypedAxisProposerError(
            "headless request belongs to other pixels, matrix, or runtime"
        )
    prompt = headless_typed_axis_proposer_prompt(request)
    schema = headless_typed_axis_proposer_output_schema(request)
    presentation = tuple(
        zip(
            HEADLESS_TYPED_AXIS_PRESENTATION_NAMES,
            (*primary, *contrast),
            strict=True,
        )
    )
    try:
        payload, receipt = call_panel_probe(
            presentation,
            prompt=prompt,
            schema=schema,
            journal=transport,  # type: ignore[arg-type]
            runtime=runtime,
        )
    except Exception as exc:
        provenance = _transport_provenance(transport, runtime)
        return _attempt_error_artifact(
            request,
            provenance,
            exc,
            failure_stage="physical_turn",
        )

    frozen = _canonical_payload(payload, "headless typed-axis payload")
    provenance = _transport_provenance(transport, runtime)
    try:
        frozen = _canonical_payload(payload, "headless typed-axis payload")
        if len(canonical_json(frozen)) > MAX_RESPONSE_UTF8_BYTES:
            raise HeadlessTypedAxisProposerError(
                "headless typed-axis payload exceeds capacity"
            )
        outcome = _parse_outcome(frozen, matrix.matrix_address)
    except Exception as exc:
        return _attempt_error_artifact(
            request,
            provenance,
            exc,
            failure_stage="payload_contract",
            model_payload=frozen,
            codex_receipt=receipt,
        )
    return HeadlessTypedAxisProposerArtifact(
        request,
        provenance,
        request.request_digest,
        panel_typed_axis_headless_proposer_source_digest(),
        _contract_digest(request),
        hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        canonical_digest(schema),
        canonical_digest(frozen),
        frozen,
        receipt,
        outcome,
    )


def verify_headless_typed_axis_proposer_artifact(
    artifact: HeadlessTypedAxisProposerArtifact,
    primary_pngs: Sequence[bytes],
    contrast_pngs: Sequence[bytes],
    *,
    matrix: TypedSupportMatrix,
    expected_artifact_digest: str,
    proposer_journal_terminal: ObjectBongardTurnJournalSummary | None = None,
) -> HeadlessTypedAxisProposerArtifact:
    """Cold replay all custody and parsing with exactly zero model calls."""

    if type(artifact) is not HeadlessTypedAxisProposerArtifact:
        raise TypeError("cold replay needs an exact headless proposer artifact")
    if type(matrix) is not TypedSupportMatrix:
        raise TypeError("cold replay needs an exact support matrix")
    expected = _digest(expected_artifact_digest, "expected headless artifact digest")
    restored = HeadlessTypedAxisProposerArtifact.from_data(artifact.to_data())
    if restored.artifact_digest != expected:
        raise HeadlessTypedAxisProposerError(
            "headless artifact differs from expected commitment"
        )
    _verify_external_journal_terminal(
        restored.transport_provenance, proposer_journal_terminal
    )
    if matrix.matrix_address != restored.request.support_matrix_address:
        raise HeadlessTypedAxisProposerError("cold replay support matrix differs")
    primary = _freeze_role(primary_pngs, "primary support")
    contrast = _freeze_role(contrast_pngs, "contrast support")
    presentation = tuple(
        PrototypeImageIdentity(name, len(raw), hashlib.sha256(raw).hexdigest())
        for name, raw in zip(
            HEADLESS_TYPED_AXIS_PRESENTATION_NAMES,
            (*primary, *contrast),
            strict=True,
        )
    )
    rebuilt = HeadlessTypedAxisProposerRequest(
        restored.request.runtime,
        matrix.matrix_address,
        typed_axis_slate_algorithm_digest(),
        presentation,
    )
    if rebuilt != restored.request:
        raise HeadlessTypedAxisProposerError("cold replay support pixels differ")
    prompt = headless_typed_axis_proposer_prompt(rebuilt)
    schema = headless_typed_axis_proposer_output_schema(rebuilt)
    with tempfile.TemporaryDirectory(prefix="bongard-typed-axis-headless-replay-") as raw:
        directory = Path(raw)
        paths: list[str] = []
        for name, data in zip(
            HEADLESS_TYPED_AXIS_PRESENTATION_NAMES,
            (*primary, *contrast),
            strict=True,
        ):
            target = directory / name
            target.write_bytes(data)
            paths.append(str(target.resolve()))
        try:
            validate_codex_named_image_receipt(
                restored.codex_receipt,
                prompt,
                tuple(paths),
                HEADLESS_TYPED_AXIS_PRESENTATION_NAMES,
                schema,
                dict(restored.model_payload),
            )
        except Exception as exc:
            raise HeadlessTypedAxisProposerError(
                "headless receipt cold replay failed"
            ) from exc
        for path, expected_bytes in zip(
            paths, (*primary, *contrast), strict=True
        ):
            if Path(path).read_bytes() != expected_bytes:
                raise HeadlessTypedAxisProposerError(
                    "headless cold replay pixels changed"
                )
    if _parse_outcome(
        restored.model_payload, matrix.matrix_address
    ) != restored.outcome:
        raise HeadlessTypedAxisProposerError(
            "headless outcome cold replay differs"
        )
    return restored


def verify_headless_typed_axis_attempt_error_artifact(
    artifact: HeadlessTypedAxisAttemptErrorArtifact,
    primary_pngs: Sequence[bytes],
    contrast_pngs: Sequence[bytes],
    *,
    matrix: TypedSupportMatrix,
    expected_attempt_digest: str,
    proposer_journal_terminal: ObjectBongardTurnJournalSummary | None = None,
) -> HeadlessTypedAxisAttemptErrorArtifact:
    """Cold replay a mandatory non-evidential error attempt with zero calls."""

    if type(artifact) is not HeadlessTypedAxisAttemptErrorArtifact:
        raise TypeError("cold replay needs an exact headless attempt error")
    if type(matrix) is not TypedSupportMatrix:
        raise TypeError("cold replay needs an exact support matrix")
    expected = _digest(expected_attempt_digest, "expected headless attempt digest")
    restored = HeadlessTypedAxisAttemptErrorArtifact.from_data(artifact.to_data())
    if restored.attempt_digest != expected:
        raise HeadlessTypedAxisProposerError(
            "headless error attempt differs from expected commitment"
        )
    _verify_external_journal_terminal(
        restored.transport_provenance, proposer_journal_terminal
    )
    if matrix.matrix_address != restored.request.support_matrix_address:
        raise HeadlessTypedAxisProposerError("cold replay support matrix differs")
    primary = _freeze_role(primary_pngs, "primary support")
    contrast = _freeze_role(contrast_pngs, "contrast support")
    presentation = tuple(
        PrototypeImageIdentity(name, len(raw), hashlib.sha256(raw).hexdigest())
        for name, raw in zip(
            HEADLESS_TYPED_AXIS_PRESENTATION_NAMES,
            (*primary, *contrast),
            strict=True,
        )
    )
    rebuilt = HeadlessTypedAxisProposerRequest(
        restored.request.runtime,
        matrix.matrix_address,
        typed_axis_slate_algorithm_digest(),
        presentation,
    )
    if rebuilt != restored.request:
        raise HeadlessTypedAxisProposerError("cold replay support pixels differ")

    if restored.failure_stage == "payload_contract":
        if restored.codex_receipt is None or restored.model_payload is None:
            raise HeadlessTypedAxisProposerError(
                "receipted payload rejection lost its raw envelope"
            )
        prompt = headless_typed_axis_proposer_prompt(rebuilt)
        schema = headless_typed_axis_proposer_output_schema(rebuilt)
        with tempfile.TemporaryDirectory(
            prefix="bongard-typed-axis-headless-error-replay-"
        ) as raw:
            directory = Path(raw)
            paths: list[str] = []
            for name, data in zip(
                HEADLESS_TYPED_AXIS_PRESENTATION_NAMES,
                (*primary, *contrast),
                strict=True,
            ):
                target = directory / name
                target.write_bytes(data)
                paths.append(str(target.resolve()))
            try:
                validate_codex_named_image_receipt(
                    restored.codex_receipt,
                    prompt,
                    tuple(paths),
                    HEADLESS_TYPED_AXIS_PRESENTATION_NAMES,
                    schema,
                    dict(restored.model_payload),
                )
            except Exception as exc:
                raise HeadlessTypedAxisProposerError(
                    "rejected payload receipt cold replay failed"
                ) from exc
            for path, expected_bytes in zip(
                paths, (*primary, *contrast), strict=True
            ):
                if Path(path).read_bytes() != expected_bytes:
                    raise HeadlessTypedAxisProposerError(
                        "headless error cold replay pixels changed"
                    )
    return restored


def verify_headless_typed_axis_proposer_result(
    result: HeadlessTypedAxisProposerResult,
    primary_pngs: Sequence[bytes],
    contrast_pngs: Sequence[bytes],
    *,
    matrix: TypedSupportMatrix,
    expected_attempt_digest: str,
    proposer_journal_terminal: ObjectBongardTurnJournalSummary | None = None,
) -> HeadlessTypedAxisProposerResult:
    """Verify the closed public success-or-error result union."""

    if type(result) is HeadlessTypedAxisProposerArtifact:
        return verify_headless_typed_axis_proposer_artifact(
            result,
            primary_pngs,
            contrast_pngs,
            matrix=matrix,
            expected_artifact_digest=expected_attempt_digest,
            proposer_journal_terminal=proposer_journal_terminal,
        )
    if type(result) is HeadlessTypedAxisAttemptErrorArtifact:
        return verify_headless_typed_axis_attempt_error_artifact(
            result,
            primary_pngs,
            contrast_pngs,
            matrix=matrix,
            expected_attempt_digest=expected_attempt_digest,
            proposer_journal_terminal=proposer_journal_terminal,
        )
    raise TypeError("result must be the closed headless proposer result union")


def headless_typed_axis_attempt_binding(
    result: HeadlessTypedAxisProposerResult,
) -> dict[str, object]:
    """Return the mandatory runner binding for either closed result variant."""

    if type(result) is HeadlessTypedAxisProposerArtifact:
        status = "success"
        attempt_digest = result.attempt_digest
        request = result.request
        provenance = result.transport_provenance
        benchmark_sealable = result.benchmark_sealable
        attempt_custody_authenticated = (
            provenance.attempt_custody_authenticated
        )
    elif type(result) is HeadlessTypedAxisAttemptErrorArtifact:
        status = "error"
        attempt_digest = result.attempt_digest
        request = result.request
        provenance = result.transport_provenance
        benchmark_sealable = False
        attempt_custody_authenticated = result.attempt_custody_authenticated
    else:
        raise TypeError("attempt binding needs the closed proposer result union")
    return {
        "schema": "gkm.bongard-typed-axis-headless-attempt-binding.v1",
        "attempt_status": status,
        "attempt_digest": attempt_digest,
        "request_digest": request.request_digest,
        "support_matrix_address": request.support_matrix_address,
        "journal_terminal_record_digest": (
            provenance.journal_terminal_record_digest
        ),
        "attempt_custody_authenticated": attempt_custody_authenticated,
        "benchmark_sealable": benchmark_sealable,
        "runner_must_bind_attempt": True,
        "omission_or_reroll_allowed": False,
        "error_is_axis_gap_or_negative_evidence": False,
    }


__all__ = (
    "CONTRAST_PRESENTATION_NAMES",
    "GAP_VALUE_TOKEN",
    "HEADLESS_TYPED_AXIS_PRESENTATION_NAMES",
    "HeadlessTypedAxisOutcome",
    "HeadlessTypedAxisAttemptErrorArtifact",
    "HeadlessTypedAxisProposerArtifact",
    "HeadlessTypedAxisProposerError",
    "HeadlessTypedAxisProposerRequest",
    "HeadlessTypedAxisProposerResult",
    "HeadlessTypedAxisRuntimeBinding",
    "HeadlessTypedAxisTransportProvenance",
    "PRIMARY_PRESENTATION_NAMES",
    "TYPED_GAP_REASON_CODES",
    "build_headless_typed_axis_turn_journal",
    "headless_typed_axis_proposer_output_schema",
    "headless_typed_axis_proposer_prompt",
    "headless_typed_axis_candidate_rank_prompt_material",
    "headless_typed_axis_attempt_binding",
    "panel_typed_axis_headless_proposer_source_digest",
    "run_headless_typed_axis_proposer",
    "verify_headless_typed_axis_attempt_error_artifact",
    "verify_headless_typed_axis_proposer_artifact",
    "verify_headless_typed_axis_proposer_result",
)
