"""Two-phase, prototype-conditioned whole-scene visual observer.

The description turn sees only six neutrally named prototype PNGs selected by
an authenticated :class:`PrototypePairCohortPlan`.  It produces a bounded
empirical measurement rubric for each opaque group.  The scoring turn sees one
neutral scene, those same six references, and the frozen rubrics.  Rubric prose
conditions the visual measurement, but is never parsed as code or used as
executable predicate authority; the resulting finite score cells are the
Python-authoritative observation record.

Every model-visible byte is receipt-attested.  Any transport or parser failure
produces an exhaustive pair of ERROR values and can never mean absence.  Cold
verification reconstructs prompts, schemas, names, and exact byte snapshots
without invoking a model or resolving corpus paths.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass, field
from enum import Enum
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import tempfile
from typing import Any, Callable, Mapping, Sequence
import zlib

from PIL import Image

import bongard.transport as _transport_module
from bongard.canonical import canonical_digest, canonical_json
from bongard.codex_no_tools_preflight import (
    CodexNoToolsAttestation,
    validate_codex_no_tools_attestation,
)
from bongard.prototype_pair_cohort import (
    OPAQUE_TAG_IDS,
    PYTHON_AUTHORITY_ID,
    PrototypePairCohortPlan,
)
from bongard.transport import (
    CODEX_ISOLATION_POLICY,
    CODEX_RECEIPT_SCHEMA,
    NAMED_IMAGE_INPUT_DIGEST_SCHEMA,
    CloudPolicyCacheSnapshot,
    CodexModelCatalogSnapshot,
    CodexProposerFailure,
    CodexReceipt,
    CodexStructuredResult,
    run_codex_named_images_structured,
    validate_codex_named_image_receipt,
    validate_codex_receipt,
    validate_codex_strict_output_schema,
)


PROTOTYPE_REFERENCE_CATALOG_SCHEMA = (
    "gkm.bongard-prototype-reference-catalog.v1"
)
PROTOTYPE_RUBRIC_DESCRIPTION_ARTIFACT_SCHEMA = (
    "gkm.bongard-prototype-rubric-description-artifact.v2"
)
PROTOTYPE_SCENE_OBSERVER_ARTIFACT_SCHEMA = (
    "gkm.bongard-prototype-scene-observer-artifact.v2"
)
PROTOTYPE_SCENE_OBSERVER_PROTOCOL_ID = (
    "bongard.prototype-scene-observer/two-phase-neutral-prototypes-v1"
)
PROTOTYPE_GROUP_IDS = ("group_0", "group_1")
PPM_SCALE = 1_000_000

_RAW_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_CODE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_SCENE_TASK_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,255}\Z")
_SCENE_PANEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./-]{0,511}\Z")
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_MAX_PNG_BYTES = 4_000_000
_MAX_PNG_PIXELS = 16_777_216
_MAX_PROSE_BYTES = 768
_PROSE_SHAPE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ,.\'-]{0,767}\Z")
_FORBIDDEN_WORD = re.compile(
    r"\b(?:task|side|label|candidate|formula|query|path|system|developer|"
    r"assistant|user|tool|prompt|instruction|ignore|override|bypass|role|"
    r"function|code|python|lean|theorem|predicate|schema|answer|output|"
    r"positive|negative|class)s?\b",
    re.IGNORECASE,
)
_FORBIDDEN_TEXT_SYNTAX = re.compile(
    r"(?:https?://|file://|[\\/{}\[\]();:=<>`$]|<\||\|>|\.\.)",
    re.IGNORECASE,
)
_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class PrototypeSceneObserverError(ValueError):
    """A catalog, artifact, payload, or byte commitment is invalid."""


class PrototypeScenePayloadError(PrototypeSceneObserverError):
    """A receipted model payload violates the frozen finite grammar."""


class PrototypeSceneObserverStatus(str, Enum):
    SUCCESS = "success"
    PARSER_ERROR = "parser_error"
    TRANSPORT_ERROR = "transport_error"
    INTERNAL_ERROR = "internal_error"
    PREREQUISITE_ERROR = "prerequisite_error"


class PrototypeRubricState(str, Enum):
    DEFINED = "defined"
    ERROR = "error"


class PrototypeSceneScoreState(str, Enum):
    SCORED = "scored"
    INDETERMINATE = "indeterminate"
    ERROR = "error"


def _require_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _RAW_SHA256.fullmatch(value) is None:
        raise PrototypeSceneObserverError(
            f"{label} must be 64 lowercase hexadecimal characters"
        )
    return value


def _require_address(value: object, label: str) -> str:
    if not isinstance(value, str) or _ADDRESS.fullmatch(value) is None:
        raise PrototypeSceneObserverError(f"{label} must be a sha256: address")
    return value


def _require_code(value: object, label: str) -> str:
    if not isinstance(value, str) or _CODE.fullmatch(value) is None:
        raise PrototypeSceneObserverError(f"{label} must be a bounded code")
    return value


def _exact_fields(
    value: object, expected: set[str] | frozenset[str], label: str
) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(not isinstance(key, str) for key in value)
        or set(value) != set(expected)
    ):
        raise PrototypeSceneObserverError(f"{label} fields differ from schema")
    return value


def _json_list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise PrototypeSceneObserverError(f"{label} must be a JSON list")
    return value


def _canonical_payload(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise PrototypeSceneObserverError("model payload must be an object")
    try:
        decoded = json.loads(canonical_json(dict(value)).decode("utf-8"))
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise PrototypeSceneObserverError(
            "model payload is not finite canonical JSON"
        ) from exc
    if not isinstance(decoded, dict):
        raise PrototypeSceneObserverError("model payload must be an object")
    return decoded


def _validate_prose(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or value != value.strip()
        or not value
        or len(value.encode("utf-8", errors="strict")) > _MAX_PROSE_BYTES
        or _PROSE_SHAPE.fullmatch(value) is None
        or _FORBIDDEN_WORD.search(value) is not None
        or _FORBIDDEN_TEXT_SYNTAX.search(value) is not None
    ):
        raise PrototypeScenePayloadError(
            f"{label} violates the bounded neutral-prose boundary"
        )
    return value


def _validate_exact_png(data: object, label: str) -> bytes:
    """Validate one bounded, single-frame PNG including chunk CRC and EOF."""

    if not isinstance(data, bytes):
        raise TypeError(f"{label} must be bytes")
    if not 0 < len(data) <= _MAX_PNG_BYTES or not data.startswith(_PNG_SIGNATURE):
        raise PrototypeSceneObserverError(f"{label} is not a bounded exact PNG")
    cursor = len(_PNG_SIGNATURE)
    chunks: list[tuple[bytes, bytes]] = []
    saw_iend = False
    while cursor < len(data):
        if cursor + 12 > len(data):
            raise PrototypeSceneObserverError(f"{label} has a truncated PNG chunk")
        length = int.from_bytes(data[cursor : cursor + 4], "big")
        kind = data[cursor + 4 : cursor + 8]
        end = cursor + 12 + length
        if end > len(data) or not re.fullmatch(rb"[A-Za-z]{4}", kind):
            raise PrototypeSceneObserverError(f"{label} has invalid PNG framing")
        payload = data[cursor + 8 : cursor + 8 + length]
        expected_crc = int.from_bytes(data[cursor + 8 + length : end], "big")
        if (zlib.crc32(kind + payload) & 0xFFFFFFFF) != expected_crc:
            raise PrototypeSceneObserverError(f"{label} has a PNG CRC mismatch")
        if saw_iend:
            raise PrototypeSceneObserverError(f"{label} has bytes after PNG IEND")
        chunks.append((kind, payload))
        if kind == b"IEND":
            if length != 0 or end != len(data):
                raise PrototypeSceneObserverError(f"{label} has a nonterminal IEND")
            saw_iend = True
        cursor = end
    if (
        not saw_iend
        or not chunks
        or chunks[0][0] != b"IHDR"
        or len(chunks[0][1]) != 13
        or sum(kind == b"IHDR" for kind, _ in chunks) != 1
        or not any(kind == b"IDAT" for kind, _ in chunks)
    ):
        raise PrototypeSceneObserverError(f"{label} lacks required PNG chunks")
    width = int.from_bytes(chunks[0][1][0:4], "big")
    height = int.from_bytes(chunks[0][1][4:8], "big")
    if width < 1 or height < 1 or width * height > _MAX_PNG_PIXELS:
        raise PrototypeSceneObserverError(f"{label} dimensions exceed the guard")
    try:
        with Image.open(BytesIO(data)) as image:
            if image.format != "PNG" or getattr(image, "n_frames", 1) != 1:
                raise PrototypeSceneObserverError(f"{label} is not one PNG frame")
            image.verify()
    except PrototypeSceneObserverError:
        raise
    except Exception as exc:
        raise PrototypeSceneObserverError(
            f"{label} PNG decoding failed: {type(exc).__name__}"
        ) from exc
    return data


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "lean_present": False,
        "lean_required": False,
        "lean_removable": True,
        "lean_affects_decision": False,
        "optional_secondary_checker_detachable": True,
    }


def _validate_authority(value: Mapping[str, Any]) -> None:
    if dict(value) != _authority_data():
        raise PrototypeSceneObserverError("runtime authority differs from policy")


def prototype_scene_observer_source_digest() -> str:
    return _SOURCE_SHA256


def prototype_scene_transport_source_digest() -> str:
    source = getattr(_transport_module, "__file__", None)
    if not isinstance(source, str) or not source:
        raise PrototypeSceneObserverError("transport source location is unavailable")
    return hashlib.sha256(Path(source).read_bytes()).hexdigest()


@dataclass(frozen=True, order=True, slots=True)
class PrototypeImageIdentity:
    name: str
    byte_count: int
    content_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or re.fullmatch(
            r"[a-z][a-z0-9_-]{0,63}\.png", self.name
        ) is None:
            raise PrototypeSceneObserverError("neutral image name is invalid")
        if (
            isinstance(self.byte_count, bool)
            or not isinstance(self.byte_count, int)
            or not 0 < self.byte_count <= _MAX_PNG_BYTES
        ):
            raise PrototypeSceneObserverError("image byte count is invalid")
        _require_digest(self.content_digest, "image content digest")

    def to_data(self) -> dict[str, object]:
        return {
            "name": self.name,
            "byte_count": self.byte_count,
            "content_digest": self.content_digest,
        }

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "PrototypeImageIdentity":
        raw = _exact_fields(
            value, {"name", "byte_count", "content_digest"}, "image identity"
        )
        result = cls(**dict(raw))
        if result.to_data() != dict(raw):
            raise PrototypeSceneObserverError("image identity is not canonical")
        return result


@dataclass(frozen=True, order=True, slots=True)
class PrototypeReferenceBinding:
    tag_id: str
    group_id: str
    reference_index: int
    source_panel_id: str
    prototype_binding_digest: str
    name: str
    byte_count: int
    content_digest: str

    def __post_init__(self) -> None:
        if self.tag_id not in OPAQUE_TAG_IDS:
            raise PrototypeSceneObserverError("reference tag is not frozen")
        tag_index = OPAQUE_TAG_IDS.index(self.tag_id)
        if self.group_id != PROTOTYPE_GROUP_IDS[tag_index]:
            raise PrototypeSceneObserverError("reference neutral group differs")
        if (
            isinstance(self.reference_index, bool)
            or not isinstance(self.reference_index, int)
            or not 0 <= self.reference_index < 3
            or self.name
            != f"{self.group_id}_ref_{self.reference_index}.png"
        ):
            raise PrototypeSceneObserverError("reference position differs")
        if not isinstance(self.source_panel_id, str) or not self.source_panel_id:
            raise PrototypeSceneObserverError("reference panel identity is empty")
        _require_address(self.prototype_binding_digest, "prototype binding digest")
        PrototypeImageIdentity(
            self.name, self.byte_count, self.content_digest
        )

    def to_data(self) -> dict[str, object]:
        return {
            "tag_id": self.tag_id,
            "group_id": self.group_id,
            "reference_index": self.reference_index,
            "source_panel_id": self.source_panel_id,
            "prototype_binding_digest": self.prototype_binding_digest,
            "name": self.name,
            "byte_count": self.byte_count,
            "content_digest": self.content_digest,
        }

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "PrototypeReferenceBinding":
        raw = _exact_fields(
            value,
            {
                "tag_id",
                "group_id",
                "reference_index",
                "source_panel_id",
                "prototype_binding_digest",
                "name",
                "byte_count",
                "content_digest",
            },
            "prototype reference binding",
        )
        result = cls(**dict(raw))
        if result.to_data() != dict(raw):
            raise PrototypeSceneObserverError(
                "prototype reference binding is not canonical"
            )
        return result


def _reference_catalog_preimage(
    catalog: "PrototypeReferenceCatalog",
) -> dict[str, object]:
    return {
        "schema": PROTOTYPE_REFERENCE_CATALOG_SCHEMA,
        "plan_digest": catalog.plan_digest,
        "planner_source_digest": catalog.planner_source_digest,
        "planner_algorithm_digest": catalog.planner_algorithm_digest,
        "prototype_binding_catalog_digest": (
            catalog.prototype_binding_catalog_digest
        ),
        "source_digest": catalog.source_digest,
        "bindings": [item.to_data() for item in catalog.bindings],
        "runtime_authority": _authority_data(),
    }


@dataclass(frozen=True, slots=True)
class PrototypeReferenceCatalog:
    plan_digest: str
    planner_source_digest: str
    planner_algorithm_digest: str
    prototype_binding_catalog_digest: str
    source_digest: str
    bindings: tuple[PrototypeReferenceBinding, ...]
    catalog_digest: str
    _sealed_digest: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _require_address(self.plan_digest, "prototype cohort plan digest")
        _require_digest(self.planner_source_digest, "planner source digest")
        _require_address(self.planner_algorithm_digest, "planner algorithm digest")
        for name in (
            "prototype_binding_catalog_digest",
            "source_digest",
            "catalog_digest",
        ):
            _require_digest(getattr(self, name), name)
        expected_positions = tuple(
            (tag_id, group_id, index, f"{group_id}_ref_{index}.png")
            for tag_id, group_id in zip(
                OPAQUE_TAG_IDS, PROTOTYPE_GROUP_IDS, strict=True
            )
            for index in range(3)
        )
        observed = tuple(
            (item.tag_id, item.group_id, item.reference_index, item.name)
            for item in self.bindings
        )
        if observed != expected_positions or len({x.source_panel_id for x in self.bindings}) != 6:
            raise PrototypeSceneObserverError(
                "reference catalog is not the exact ordered two-by-three grid"
            )
        if self.source_digest != prototype_scene_observer_source_digest():
            raise PrototypeSceneObserverError("observer source digest differs")
        computed = canonical_digest(_reference_catalog_preimage(self))
        if self.catalog_digest != computed:
            raise PrototypeSceneObserverError("reference catalog digest differs")
        object.__setattr__(self, "_sealed_digest", computed)

    @property
    def presentation(self) -> tuple[PrototypeImageIdentity, ...]:
        return tuple(
            PrototypeImageIdentity(x.name, x.byte_count, x.content_digest)
            for x in self.bindings
        )

    def to_data(self) -> dict[str, object]:
        return {
            **_reference_catalog_preimage(self),
            "catalog_digest": self.catalog_digest,
        }

    @classmethod
    def from_data(
        cls,
        value: Mapping[str, Any],
        *,
        expected_catalog_digest: str | None = None,
    ) -> "PrototypeReferenceCatalog":
        raw = _exact_fields(
            value,
            {
                "schema",
                "plan_digest",
                "planner_source_digest",
                "planner_algorithm_digest",
                "prototype_binding_catalog_digest",
                "source_digest",
                "bindings",
                "runtime_authority",
                "catalog_digest",
            },
            "prototype reference catalog",
        )
        if raw["schema"] != PROTOTYPE_REFERENCE_CATALOG_SCHEMA:
            raise PrototypeSceneObserverError("unsupported reference catalog")
        authority = raw["runtime_authority"]
        if not isinstance(authority, Mapping):
            raise PrototypeSceneObserverError("catalog authority is invalid")
        _validate_authority(authority)
        bindings = _json_list(raw["bindings"], "reference bindings")
        if any(not isinstance(item, Mapping) for item in bindings):
            raise PrototypeSceneObserverError("reference binding is invalid")
        result = cls(
            plan_digest=raw["plan_digest"],
            planner_source_digest=raw["planner_source_digest"],
            planner_algorithm_digest=raw["planner_algorithm_digest"],
            prototype_binding_catalog_digest=raw[
                "prototype_binding_catalog_digest"
            ],
            source_digest=raw["source_digest"],
            bindings=tuple(
                PrototypeReferenceBinding.from_data(item) for item in bindings
            ),
            catalog_digest=raw["catalog_digest"],
        )
        if expected_catalog_digest is not None and result.catalog_digest != (
            _require_digest(expected_catalog_digest, "expected catalog digest")
        ):
            raise PrototypeSceneObserverError(
                "catalog differs from external digest commitment"
            )
        if result.to_data() != dict(raw):
            raise PrototypeSceneObserverError(
                "reference catalog is not canonically represented"
            )
        return result

    def assert_untampered(self) -> None:
        computed = canonical_digest(_reference_catalog_preimage(self))
        if computed != self.catalog_digest or computed != self._sealed_digest:
            raise PrototypeSceneObserverError("reference catalog changed after sealing")


def _expected_reference_rows(
    plan: PrototypePairCohortPlan,
) -> tuple[tuple[str, str, int, str, str], ...]:
    if not isinstance(plan, PrototypePairCohortPlan):
        raise TypeError("plan must be PrototypePairCohortPlan")
    rows: list[tuple[str, str, int, str, str]] = []
    for prototype, tag_id, group_id in zip(
        plan.prototypes, OPAQUE_TAG_IDS, PROTOTYPE_GROUP_IDS, strict=True
    ):
        if prototype.tag_id != tag_id or len(prototype.panel_ids) != 3:
            raise PrototypeSceneObserverError("plan prototype order differs")
        binding_digest = prototype.to_data()["record_digest"]
        if not isinstance(binding_digest, str):
            raise PrototypeSceneObserverError("prototype binding digest is invalid")
        for index, panel_id in enumerate(prototype.panel_ids):
            rows.append((tag_id, group_id, index, panel_id, binding_digest))
    return tuple(rows)


def build_prototype_reference_catalog(
    plan: PrototypePairCohortPlan,
    prototype_png_by_panel_id: Mapping[str, bytes],
    *,
    expected_plan_digest: str,
    expected_reference_sha256: Mapping[str, str],
) -> PrototypeReferenceCatalog:
    """Freeze the exact six caller-supplied prototype bytes in plan order."""

    expected_plan = _require_address(expected_plan_digest, "expected plan digest")
    if plan.record_digest != expected_plan:
        raise PrototypeSceneObserverError("plan differs from external commitment")
    rows = _expected_reference_rows(plan)
    panel_ids = tuple(row[3] for row in rows)
    for label, mapping in (
        ("prototype byte mapping", prototype_png_by_panel_id),
        ("reference SHA-256 mapping", expected_reference_sha256),
    ):
        if (
            not isinstance(mapping, Mapping)
            or any(not isinstance(key, str) for key in mapping)
            or set(mapping) != set(panel_ids)
        ):
            raise PrototypeSceneObserverError(f"{label} keys differ from plan")
    bindings: list[PrototypeReferenceBinding] = []
    for tag_id, group_id, index, panel_id, binding_digest in rows:
        data = _validate_exact_png(
            prototype_png_by_panel_id[panel_id], f"prototype {group_id}/{index}"
        )
        committed = _require_digest(
            expected_reference_sha256[panel_id], "reference SHA-256 commitment"
        )
        observed = hashlib.sha256(data).hexdigest()
        if observed != committed:
            raise PrototypeSceneObserverError(
                "prototype bytes differ from external SHA-256 commitment"
            )
        bindings.append(
            PrototypeReferenceBinding(
                tag_id=tag_id,
                group_id=group_id,
                reference_index=index,
                source_panel_id=panel_id,
                prototype_binding_digest=binding_digest,
                name=f"{group_id}_ref_{index}.png",
                byte_count=len(data),
                content_digest=observed,
            )
        )
    prototype_binding_catalog_digest = canonical_digest(
        {
            "schema": "gkm.bongard-prototype-binding-catalog.v1",
            "bindings": [item.to_data() for item in plan.prototypes],
        }
    )
    values: dict[str, object] = {
        "plan_digest": plan.record_digest,
        "planner_source_digest": plan.planner_source_sha256,
        "planner_algorithm_digest": plan.planner_algorithm_digest,
        "prototype_binding_catalog_digest": prototype_binding_catalog_digest,
        "source_digest": prototype_scene_observer_source_digest(),
        "bindings": tuple(bindings),
    }
    provisional = object.__new__(PrototypeReferenceCatalog)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return PrototypeReferenceCatalog(
        **values,  # type: ignore[arg-type]
        catalog_digest=canonical_digest(_reference_catalog_preimage(provisional)),
    )


def verify_prototype_reference_catalog(
    catalog: PrototypeReferenceCatalog,
    plan: PrototypePairCohortPlan,
    prototype_png_by_panel_id: Mapping[str, bytes],
    *,
    expected_plan_digest: str,
    expected_reference_sha256: Mapping[str, str],
    expected_catalog_digest: str,
) -> PrototypeReferenceCatalog:
    if not isinstance(catalog, PrototypeReferenceCatalog):
        raise TypeError("catalog must be PrototypeReferenceCatalog")
    catalog.assert_untampered()
    rebuilt = build_prototype_reference_catalog(
        plan,
        prototype_png_by_panel_id,
        expected_plan_digest=expected_plan_digest,
        expected_reference_sha256=expected_reference_sha256,
    )
    if catalog != rebuilt or catalog.catalog_digest != _require_digest(
        expected_catalog_digest, "expected catalog digest"
    ):
        raise PrototypeSceneObserverError(
            "reference catalog differs from cold reconstruction"
        )
    if PrototypeReferenceCatalog.from_data(
        catalog.to_data(), expected_catalog_digest=expected_catalog_digest
    ) != catalog:
        raise PrototypeSceneObserverError("reference catalog round trip differs")
    return catalog


@dataclass(frozen=True, order=True, slots=True)
class PrototypeRubric:
    tag_id: str
    group_id: str
    state: PrototypeRubricState
    prose: str | None
    reason_code: str | None
    error_type: str | None

    def __post_init__(self) -> None:
        if self.tag_id not in OPAQUE_TAG_IDS:
            raise PrototypeSceneObserverError("rubric tag is not frozen")
        if self.group_id != PROTOTYPE_GROUP_IDS[OPAQUE_TAG_IDS.index(self.tag_id)]:
            raise PrototypeSceneObserverError("rubric group differs")
        if not isinstance(self.state, PrototypeRubricState):
            raise TypeError("rubric state must be PrototypeRubricState")
        if self.state is PrototypeRubricState.DEFINED:
            _validate_prose(self.prose, "rubric prose")
            if self.reason_code is not None or self.error_type is not None:
                raise PrototypeSceneObserverError("defined rubric has failure fields")
        else:
            if self.prose is not None:
                raise PrototypeSceneObserverError("error rubric carries prose")
            _require_code(self.reason_code, "rubric reason code")
            _require_code(self.error_type, "rubric error type")

    @classmethod
    def defined(cls, tag_id: str, group_id: str, prose: str) -> "PrototypeRubric":
        return cls(tag_id, group_id, PrototypeRubricState.DEFINED, prose, None, None)

    @classmethod
    def error(
        cls, tag_id: str, group_id: str, reason_code: str, error_type: str
    ) -> "PrototypeRubric":
        return cls(
            tag_id,
            group_id,
            PrototypeRubricState.ERROR,
            None,
            reason_code,
            error_type,
        )

    def to_data(self) -> dict[str, object]:
        return {
            "tag_id": self.tag_id,
            "group_id": self.group_id,
            "state": self.state.value,
            "prose": self.prose,
            "reason_code": self.reason_code,
            "error_type": self.error_type,
        }

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "PrototypeRubric":
        raw = _exact_fields(
            value,
            {"tag_id", "group_id", "state", "prose", "reason_code", "error_type"},
            "prototype rubric",
        )
        try:
            state = PrototypeRubricState(raw["state"])
        except (TypeError, ValueError) as exc:
            raise PrototypeSceneObserverError("unknown rubric state") from exc
        result = cls(
            raw["tag_id"],
            raw["group_id"],
            state,
            raw["prose"],
            raw["reason_code"],
            raw["error_type"],
        )
        if result.to_data() != dict(raw):
            raise PrototypeSceneObserverError("rubric is not canonical")
        return result


@dataclass(frozen=True, order=True, slots=True)
class PrototypeSceneScore:
    tag_id: str
    group_id: str
    state: PrototypeSceneScoreState
    lower_ppm: int | None
    upper_ppm: int | None
    reason_code: str | None
    error_type: str | None

    def __post_init__(self) -> None:
        if self.tag_id not in OPAQUE_TAG_IDS:
            raise PrototypeSceneObserverError("score tag is not frozen")
        if self.group_id != PROTOTYPE_GROUP_IDS[OPAQUE_TAG_IDS.index(self.tag_id)]:
            raise PrototypeSceneObserverError("score group differs")
        if not isinstance(self.state, PrototypeSceneScoreState):
            raise TypeError("score state must be PrototypeSceneScoreState")
        if self.state is PrototypeSceneScoreState.SCORED:
            if (
                isinstance(self.lower_ppm, bool)
                or isinstance(self.upper_ppm, bool)
                or not isinstance(self.lower_ppm, int)
                or not isinstance(self.upper_ppm, int)
                or not 0 <= self.lower_ppm <= self.upper_ppm <= PPM_SCALE
                or self.reason_code is not None
                or self.error_type is not None
            ):
                raise PrototypeSceneObserverError(
                    "scored cell requires one canonical PPM interval"
                )
        elif self.state is PrototypeSceneScoreState.INDETERMINATE:
            if (
                self.lower_ppm is not None
                or self.upper_ppm is not None
                or self.error_type is not None
            ):
                raise PrototypeSceneObserverError(
                    "indeterminate cell carries an interval or error type"
                )
            _require_code(self.reason_code, "indeterminate reason code")
        else:
            if self.lower_ppm is not None or self.upper_ppm is not None:
                raise PrototypeSceneObserverError("error cell carries an interval")
            _require_code(self.reason_code, "error reason code")
            _require_code(self.error_type, "error type")

    @classmethod
    def scored(
        cls, tag_id: str, group_id: str, lower_ppm: int, upper_ppm: int
    ) -> "PrototypeSceneScore":
        return cls(
            tag_id,
            group_id,
            PrototypeSceneScoreState.SCORED,
            lower_ppm,
            upper_ppm,
            None,
            None,
        )

    @classmethod
    def indeterminate(
        cls, tag_id: str, group_id: str, reason_code: str
    ) -> "PrototypeSceneScore":
        return cls(
            tag_id,
            group_id,
            PrototypeSceneScoreState.INDETERMINATE,
            None,
            None,
            reason_code,
            None,
        )

    @classmethod
    def error(
        cls, tag_id: str, group_id: str, reason_code: str, error_type: str
    ) -> "PrototypeSceneScore":
        return cls(
            tag_id,
            group_id,
            PrototypeSceneScoreState.ERROR,
            None,
            None,
            reason_code,
            error_type,
        )

    def to_data(self) -> dict[str, object]:
        return {
            "tag_id": self.tag_id,
            "group_id": self.group_id,
            "state": self.state.value,
            "lower_ppm": self.lower_ppm,
            "upper_ppm": self.upper_ppm,
            "reason_code": self.reason_code,
            "error_type": self.error_type,
        }

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "PrototypeSceneScore":
        raw = _exact_fields(
            value,
            {
                "tag_id",
                "group_id",
                "state",
                "lower_ppm",
                "upper_ppm",
                "reason_code",
                "error_type",
            },
            "prototype scene score",
        )
        try:
            state = PrototypeSceneScoreState(raw["state"])
        except (TypeError, ValueError) as exc:
            raise PrototypeSceneObserverError("unknown scene score state") from exc
        result = cls(
            raw["tag_id"],
            raw["group_id"],
            state,
            raw["lower_ppm"],
            raw["upper_ppm"],
            raw["reason_code"],
            raw["error_type"],
        )
        if result.to_data() != dict(raw):
            raise PrototypeSceneObserverError("scene score is not canonical")
        return result


def prototype_rubric_description_prompt() -> str:
    return (
        "You are an empirical visual observer. Inspect six reference images "
        "arranged as two neutral groups, with three examples in each group. "
        "Each reference contains a recurring object or configuration. For each "
        "declared group, summarize the recurring visible object appearance "
        "shared by its three examples, ignoring incidental pose, scale, stroke "
        "style, and location. Use one concise neutral prose sentence about "
        "visible shape, geometry, texture, or organization. This sentence will "
        "later be used only as a fixed visual rubric for detecting whether a "
        "new scene contains at least one similar object. Do not give commands, "
        "locations, source names, hidden roles, or executable text. Return both "
        "groups in the declared order. In each rubric use only ordinary words, "
        "spaces, commas, periods, apostrophes, and hyphens."
    )


def prototype_rubric_description_output_schema() -> dict[str, object]:
    rubric_properties: dict[str, object] = {
        "group_id": {"type": "string", "enum": list(PROTOTYPE_GROUP_IDS)},
        "rubric": {"type": "string"},
    }
    return {
        "type": "object",
        "properties": {
            "rubrics": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": rubric_properties,
                    "required": list(rubric_properties),
                    "additionalProperties": False,
                },
            }
        },
        "required": ["rubrics"],
        "additionalProperties": False,
    }


def prototype_scene_observer_output_schema() -> dict[str, object]:
    nullable_integer = {"anyOf": [{"type": "integer"}, {"type": "null"}]}
    nullable_string = {"anyOf": [{"type": "string"}, {"type": "null"}]}
    cell_properties: dict[str, object] = {
        "group_id": {"type": "string", "enum": list(PROTOTYPE_GROUP_IDS)},
        "state": {"type": "string", "enum": ["scored", "indeterminate"]},
        "lower_ppm": nullable_integer,
        "upper_ppm": nullable_integer,
        "reason_code": nullable_string,
    }
    return {
        "type": "object",
        "properties": {
            "description": {"type": "string"},
            "cells": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": cell_properties,
                    "required": list(cell_properties),
                    "additionalProperties": False,
                },
            },
        },
        "required": ["description", "cells"],
        "additionalProperties": False,
    }


def prototype_scene_observer_prompt(rubrics: Sequence[PrototypeRubric]) -> str:
    values = tuple(rubrics)
    if len(values) != 2 or tuple(item.group_id for item in values) != (
        PROTOTYPE_GROUP_IDS
    ):
        raise PrototypeSceneObserverError("scoring prompt rubrics differ")
    if any(item.state is not PrototypeRubricState.DEFINED for item in values):
        return (
            "Reference descriptions are unavailable. No empirical visual "
            "scoring call is authorized for this scene."
        )
    rendered = "\n".join(
        f"- {item.group_id}: {item.prose}" for item in values
    )
    return (
        "You are an empirical visual observer. Inspect scene.png and the six "
        "neutral reference images. The frozen neutral descriptions below were "
        "written from the three references in each group. For each group, judge "
        "whether scene.png contains at least one visible object or configuration "
        "matching that recurring appearance. Score presence of a matching "
        "object, not global similarity of the whole scene. Unrelated additional "
        "objects must not lower a presence score. For both groups in the "
        "declared order, return an integer uncertainty interval from 0 to "
        "1000000 inclusive, where 0 means definitely absent and 1000000 means "
        "definitely present. Use indeterminate only when the visual evidence "
        "cannot genuinely be resolved; express ordinary uncertainty with an "
        "interval. For a scored cell, lower_ppm and upper_ppm must be integers "
        "with lower_ppm no greater than upper_ppm and reason_code must be null. "
        "For an indeterminate cell, both interval values must be null and "
        "reason_code must be a short identifier. Also return one concise "
        "neutral prose sentence describing the visible scene. In that sentence "
        "use only ordinary words, spaces, commas, periods, apostrophes, and "
        "hyphens. Do not give commands, locations, source names, hidden roles, "
        "or executable text.\n\nFrozen group descriptions:\n"
        f"{rendered}"
    )


def prototype_scene_observer_model_digest(
    model: str, reasoning_effort: str
) -> str:
    if not isinstance(model, str) or not model:
        raise PrototypeSceneObserverError("model must be nonempty")
    if not isinstance(reasoning_effort, str) or not reasoning_effort:
        raise PrototypeSceneObserverError("reasoning effort must be nonempty")
    return canonical_digest(
        {
            "schema": "gkm.bongard-prototype-scene-model-request.v1",
            "model": model,
            "reasoning_effort": reasoning_effort,
        }
    )


def prototype_rubric_description_protocol_digest() -> str:
    prompt = prototype_rubric_description_prompt()
    schema = prototype_rubric_description_output_schema()
    validate_codex_strict_output_schema(schema)
    return canonical_digest(
        {
            "schema": "gkm.bongard-prototype-description-protocol.v1",
            "protocol_id": PROTOTYPE_SCENE_OBSERVER_PROTOCOL_ID,
            "phase": "reference-description",
            "source_digest": prototype_scene_observer_source_digest(),
            "transport_source_digest": prototype_scene_transport_source_digest(),
            "receipt_schema": CODEX_RECEIPT_SCHEMA,
            "isolation_policy": CODEX_ISOLATION_POLICY,
            "prompt_digest": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "output_schema_digest": canonical_digest(schema),
            "ordered_names": [
                f"{group_id}_ref_{index}.png"
                for group_id in PROTOTYPE_GROUP_IDS
                for index in range(3)
            ],
            "rubric_order": list(PROTOTYPE_GROUP_IDS),
            "receipt_domain": NAMED_IMAGE_INPUT_DIGEST_SCHEMA,
            "prose_role": "frozen-empirical-measurement-rubric",
            "prose_conditions_observation": True,
            "prose_is_never_executable": True,
            "failure_semantics": "both-rubrics-error-never-absence",
            **_authority_data(),
        }
    )


def prototype_scene_scoring_protocol_digest() -> str:
    schema = prototype_scene_observer_output_schema()
    validate_codex_strict_output_schema(schema)
    return canonical_digest(
        {
            "schema": "gkm.bongard-prototype-scoring-protocol.v1",
            "protocol_id": PROTOTYPE_SCENE_OBSERVER_PROTOCOL_ID,
            "phase": "whole-scene-scoring",
            "source_digest": prototype_scene_observer_source_digest(),
            "transport_source_digest": prototype_scene_transport_source_digest(),
            "receipt_schema": CODEX_RECEIPT_SCHEMA,
            "isolation_policy": CODEX_ISOLATION_POLICY,
            "prompt_family": "frozen-two-rubric-neutral-scene-v1",
            "output_schema_digest": canonical_digest(schema),
            "ordered_names": [
                "scene.png",
                *[
                    f"{group_id}_ref_{index}.png"
                    for group_id in PROTOTYPE_GROUP_IDS
                    for index in range(3)
                ],
            ],
            "score_order": list(PROTOTYPE_GROUP_IDS),
            "receipt_domain": NAMED_IMAGE_INPUT_DIGEST_SCHEMA,
            "failure_semantics": "both-score-cells-error-never-absence",
            "prose_role": "frozen-empirical-measurement-rubric",
            "prose_conditions_observation": True,
            "prose_is_never_executable": True,
            **_authority_data(),
        }
    )


def _receipt_from_data(value: object) -> CodexReceipt:
    expected = set(CodexReceipt.__dataclass_fields__)
    raw = _exact_fields(value, expected, "archived receipt")
    try:
        validate_codex_receipt(raw)
        if not isinstance(raw["event_types"], list) or not isinstance(
            raw["item_types"], list
        ):
            raise PrototypeSceneObserverError(
                "receipt event summaries must be JSON lists"
            )
        result = CodexReceipt(
            **{
                **dict(raw),
                "event_types": tuple(raw["event_types"]),
                "item_types": tuple(raw["item_types"]),
            }
        )
    except (CodexProposerFailure, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, PrototypeSceneObserverError):
            raise
        raise PrototypeSceneObserverError("archived receipt is invalid") from exc
    if result.to_dict() != dict(raw):
        raise PrototypeSceneObserverError("receipt is not canonical")
    return result


def _policy_cache_binding(
    snapshot: CloudPolicyCacheSnapshot | None,
) -> str:
    binding = "absent" if snapshot is None else snapshot.binding
    if binding != "absent" and (
        not isinstance(binding, str) or _ADDRESS.fullmatch(binding) is None
    ):
        raise PrototypeSceneObserverError("policy cache binding is invalid")
    return binding


def prototype_scene_observer_environment_digest(
    *,
    model: str,
    reasoning_effort: str,
    expected_launcher_digest: str | None,
    cloud_policy_cache_binding: str,
    model_catalog_digest: str,
    no_tools_attestation_digest: str,
) -> str:
    prototype_scene_observer_model_digest(model, reasoning_effort)
    if expected_launcher_digest is not None:
        _require_digest(expected_launcher_digest, "expected launcher digest")
    if cloud_policy_cache_binding != "absent":
        _require_address(cloud_policy_cache_binding, "policy cache binding")
    _require_digest(model_catalog_digest, "model catalog digest")
    _require_digest(no_tools_attestation_digest, "no-tools attestation digest")
    return canonical_digest(
        {
            "schema": "gkm.bongard-prototype-observer-precommitted-environment.v1",
            "model": model,
            "reasoning_effort": reasoning_effort,
            "expected_launcher_digest": expected_launcher_digest,
            "cloud_policy_cache_binding": cloud_policy_cache_binding,
            "model_catalog_digest": model_catalog_digest,
            "no_tools_attestation_digest": no_tools_attestation_digest,
            "observer_source_digest": prototype_scene_observer_source_digest(),
            "transport_source_digest": prototype_scene_transport_source_digest(),
            "receipt_schema": CODEX_RECEIPT_SCHEMA,
            "input_digest_schema": NAMED_IMAGE_INPUT_DIGEST_SCHEMA,
            "isolation_policy": CODEX_ISOLATION_POLICY,
            "transport_entrypoint": "run_codex_named_images_structured",
        }
    )


def _validate_no_tools_runtime(
    *,
    model_catalog_snapshot: CodexModelCatalogSnapshot,
    no_tools_attestation: CodexNoToolsAttestation,
    expected_launcher_digest: str | None,
    cloud_policy_cache_binding: str,
) -> tuple[str, str]:
    if not isinstance(model_catalog_snapshot, CodexModelCatalogSnapshot):
        raise PrototypeSceneObserverError(
            "exact Codex model catalog snapshot is required"
        )
    if expected_launcher_digest is None:
        raise PrototypeSceneObserverError(
            "Codex launcher commitment is required for visual observation"
        )
    try:
        validated = validate_codex_no_tools_attestation(
            no_tools_attestation,
            expected_launcher_digest=expected_launcher_digest,
            expected_model_catalog_digest=model_catalog_snapshot.raw_digest,
            expected_cloud_policy_cache_binding=cloud_policy_cache_binding,
        )
    except (CodexProposerFailure, TypeError, ValueError) as exc:
        raise PrototypeSceneObserverError(
            "Codex no-tools runtime differs from its frozen attestation"
        ) from exc
    return model_catalog_snapshot.raw_digest, validated.attestation_digest


def _failure_digest(
    phase: str,
    status: PrototypeSceneObserverStatus,
    failure_code: str | None,
    failure_type: str | None,
    payload: Mapping[str, Any] | None,
) -> str | None:
    if status is PrototypeSceneObserverStatus.SUCCESS:
        if failure_code is not None or failure_type is not None:
            raise PrototypeSceneObserverError("success carries failure fields")
        return None
    code = _require_code(failure_code, "failure code")
    kind = _require_code(failure_type, "failure type")
    return canonical_digest(
        {
            "schema": "gkm.bongard-prototype-observer-failure.v1",
            "phase": phase,
            "status": status.value,
            "failure_code": code,
            "failure_type": kind,
            "model_payload": payload,
        }
    )


def _exception_type(exception: BaseException) -> str:
    value = type(exception).__name__
    return value if _CODE.fullmatch(value) is not None else "UnclassifiedInternalError"


def _rubric_error_pair(
    reason_code: str, error_type: str
) -> tuple[PrototypeRubric, ...]:
    return tuple(
        PrototypeRubric.error(tag_id, group_id, reason_code, error_type)
        for tag_id, group_id in zip(
            OPAQUE_TAG_IDS, PROTOTYPE_GROUP_IDS, strict=True
        )
    )


def _score_error_pair(
    reason_code: str, error_type: str
) -> tuple[PrototypeSceneScore, ...]:
    return tuple(
        PrototypeSceneScore.error(tag_id, group_id, reason_code, error_type)
        for tag_id, group_id in zip(
            OPAQUE_TAG_IDS, PROTOTYPE_GROUP_IDS, strict=True
        )
    )


def _parse_rubric_payload(
    payload: Mapping[str, Any],
) -> tuple[PrototypeRubric, ...]:
    raw = _exact_fields(payload, {"rubrics"}, "rubric description payload")
    values = _json_list(raw["rubrics"], "rubric descriptions")
    if len(values) != 2:
        raise PrototypeScenePayloadError(
            "description payload does not exhaust both groups"
        )
    rubrics: list[PrototypeRubric] = []
    for index, (item, tag_id, group_id) in enumerate(
        zip(values, OPAQUE_TAG_IDS, PROTOTYPE_GROUP_IDS, strict=True)
    ):
        row = _exact_fields(
            item, {"group_id", "rubric"}, f"rubric description {index}"
        )
        if row["group_id"] != group_id:
            raise PrototypeScenePayloadError(
                "rubric descriptions differ from canonical group order"
            )
        rubrics.append(
            PrototypeRubric.defined(
                tag_id, group_id, _validate_prose(row["rubric"], "rubric")
            )
        )
    return tuple(rubrics)


def _parse_scene_payload(
    payload: Mapping[str, Any],
) -> tuple[str, tuple[PrototypeSceneScore, ...]]:
    raw = _exact_fields(payload, {"description", "cells"}, "scene payload")
    description = _validate_prose(raw["description"], "scene description")
    values = _json_list(raw["cells"], "scene cells")
    if len(values) != 2:
        raise PrototypeScenePayloadError("scene payload does not exhaust both groups")
    scores: list[PrototypeSceneScore] = []
    for index, (item, tag_id, group_id) in enumerate(
        zip(values, OPAQUE_TAG_IDS, PROTOTYPE_GROUP_IDS, strict=True)
    ):
        row = _exact_fields(
            item,
            {"group_id", "state", "lower_ppm", "upper_ppm", "reason_code"},
            f"scene score cell {index}",
        )
        if row["group_id"] != group_id:
            raise PrototypeScenePayloadError(
                "scene score cells differ from canonical group order"
            )
        if row["state"] == "scored":
            if row["reason_code"] is not None:
                raise PrototypeScenePayloadError(
                    "scored scene cell carries a reason code"
                )
            try:
                scores.append(
                    PrototypeSceneScore.scored(
                        tag_id,
                        group_id,
                        row["lower_ppm"],
                        row["upper_ppm"],
                    )
                )
            except (PrototypeSceneObserverError, TypeError) as exc:
                raise PrototypeScenePayloadError(
                    "scored scene cell has an invalid interval"
                ) from exc
        elif row["state"] == "indeterminate":
            if row["lower_ppm"] is not None or row["upper_ppm"] is not None:
                raise PrototypeScenePayloadError(
                    "indeterminate scene cell carries an interval"
                )
            try:
                scores.append(
                    PrototypeSceneScore.indeterminate(
                        tag_id, group_id, row["reason_code"]
                    )
                )
            except (PrototypeSceneObserverError, TypeError) as exc:
                raise PrototypeScenePayloadError(
                    "indeterminate scene cell has no bounded reason code"
                ) from exc
        else:
            raise PrototypeScenePayloadError("unknown scene cell state")
    return description, tuple(scores)


def _description_artifact_preimage(
    artifact: "PrototypeRubricDescriptionArtifact",
) -> dict[str, object]:
    return {
        "schema": PROTOTYPE_RUBRIC_DESCRIPTION_ARTIFACT_SCHEMA,
        "status": artifact.status.value,
        "plan_digest": artifact.plan_digest,
        "catalog_digest": artifact.catalog_digest,
        "presentation": [item.to_data() for item in artifact.presentation],
        "prompt_digest": artifact.prompt_digest,
        "output_schema_digest": artifact.output_schema_digest,
        "protocol_digest": artifact.protocol_digest,
        "source_digest": artifact.source_digest,
        "transport_source_digest": artifact.transport_source_digest,
        "model": artifact.model,
        "reasoning_effort": artifact.reasoning_effort,
        "model_digest": artifact.model_digest,
        "expected_launcher_digest": artifact.expected_launcher_digest,
        "cloud_policy_cache_binding": artifact.cloud_policy_cache_binding,
        "model_catalog_digest": artifact.model_catalog_digest,
        "no_tools_attestation_digest": artifact.no_tools_attestation_digest,
        "environment_digest": artifact.environment_digest,
        "model_payload": artifact.model_payload,
        "receipt": None if artifact.receipt is None else artifact.receipt.to_dict(),
        "failure_code": artifact.failure_code,
        "failure_type": artifact.failure_type,
        "failure_digest": artifact.failure_digest,
        "rubrics": [item.to_data() for item in artifact.rubrics],
        "runtime_authority": _authority_data(),
    }


@dataclass(frozen=True, slots=True)
class PrototypeRubricDescriptionArtifact:
    status: PrototypeSceneObserverStatus
    plan_digest: str
    catalog_digest: str
    presentation: tuple[PrototypeImageIdentity, ...]
    prompt_digest: str
    output_schema_digest: str
    protocol_digest: str
    source_digest: str
    transport_source_digest: str
    model: str
    reasoning_effort: str
    model_digest: str
    expected_launcher_digest: str | None
    cloud_policy_cache_binding: str
    model_catalog_digest: str
    no_tools_attestation_digest: str
    environment_digest: str
    model_payload: Mapping[str, Any] | None
    receipt: CodexReceipt | None
    failure_code: str | None
    failure_type: str | None
    failure_digest: str | None
    rubrics: tuple[PrototypeRubric, ...]
    artifact_digest: str
    _sealed_digest: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.status, PrototypeSceneObserverStatus):
            raise TypeError("description status must be PrototypeSceneObserverStatus")
        if self.status is PrototypeSceneObserverStatus.PREREQUISITE_ERROR:
            raise PrototypeSceneObserverError(
                "reference description has no prerequisite-error state"
            )
        _require_address(self.plan_digest, "description plan digest")
        for name in (
            "catalog_digest",
            "prompt_digest",
            "output_schema_digest",
            "protocol_digest",
            "source_digest",
            "transport_source_digest",
            "model_digest",
            "environment_digest",
            "artifact_digest",
        ):
            _require_digest(getattr(self, name), name)
        if self.source_digest != prototype_scene_observer_source_digest():
            raise PrototypeSceneObserverError("description source digest differs")
        if self.transport_source_digest != prototype_scene_transport_source_digest():
            raise PrototypeSceneObserverError("description transport source differs")
        if self.model_digest != prototype_scene_observer_model_digest(
            self.model, self.reasoning_effort
        ):
            raise PrototypeSceneObserverError("description model digest differs")
        if self.expected_launcher_digest is not None:
            _require_digest(
                self.expected_launcher_digest, "description launcher commitment"
            )
        if self.cloud_policy_cache_binding != "absent":
            _require_address(
                self.cloud_policy_cache_binding, "description policy cache binding"
            )
        _require_digest(self.model_catalog_digest, "description model catalog digest")
        _require_digest(
            self.no_tools_attestation_digest,
            "description no-tools attestation digest",
        )
        if self.environment_digest != prototype_scene_observer_environment_digest(
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            expected_launcher_digest=self.expected_launcher_digest,
            cloud_policy_cache_binding=self.cloud_policy_cache_binding,
            model_catalog_digest=self.model_catalog_digest,
            no_tools_attestation_digest=self.no_tools_attestation_digest,
        ):
            raise PrototypeSceneObserverError(
                "description precommitted environment digest differs"
            )
        expected_names = tuple(
            f"{group_id}_ref_{index}.png"
            for group_id in PROTOTYPE_GROUP_IDS
            for index in range(3)
        )
        if tuple(item.name for item in self.presentation) != expected_names:
            raise PrototypeSceneObserverError(
                "description presentation order differs"
            )
        prompt = prototype_rubric_description_prompt()
        schema = prototype_rubric_description_output_schema()
        if self.prompt_digest != hashlib.sha256(prompt.encode("utf-8")).hexdigest():
            raise PrototypeSceneObserverError("description prompt digest differs")
        if self.output_schema_digest != canonical_digest(schema):
            raise PrototypeSceneObserverError("description schema digest differs")
        if self.protocol_digest != prototype_rubric_description_protocol_digest():
            raise PrototypeSceneObserverError("description protocol digest differs")
        if tuple((item.tag_id, item.group_id) for item in self.rubrics) != tuple(
            zip(OPAQUE_TAG_IDS, PROTOTYPE_GROUP_IDS, strict=True)
        ):
            raise PrototypeSceneObserverError("description rubric order differs")
        payload = (
            None
            if self.model_payload is None
            else _canonical_payload(self.model_payload)
        )
        object.__setattr__(self, "model_payload", payload)
        expected_failure = _failure_digest(
            "reference-description",
            self.status,
            self.failure_code,
            self.failure_type,
            payload,
        )
        if self.failure_digest != expected_failure:
            raise PrototypeSceneObserverError("description failure digest differs")
        if self.status is PrototypeSceneObserverStatus.SUCCESS:
            if self.receipt is None or payload is None:
                raise PrototypeSceneObserverError(
                    "successful description lacks payload or receipt"
                )
            if _parse_rubric_payload(payload) != self.rubrics:
                raise PrototypeSceneObserverError(
                    "successful description payload differs from rubrics"
                )
        elif self.status is PrototypeSceneObserverStatus.PARSER_ERROR:
            if self.receipt is None or payload is None:
                raise PrototypeSceneObserverError(
                    "parser error must retain payload and receipt"
                )
            try:
                _parse_rubric_payload(payload)
            except PrototypeSceneObserverError:
                pass
            else:
                raise PrototypeSceneObserverError(
                    "parser-error description payload is admissible"
                )
            if self.rubrics != _rubric_error_pair(
                "observer_payload_rejected", "PrototypeScenePayloadError"
            ):
                raise PrototypeSceneObserverError(
                    "description parser failure is not exhaustive ERROR"
                )
        elif self.status is PrototypeSceneObserverStatus.TRANSPORT_ERROR:
            if self.receipt is not None or payload is not None:
                raise PrototypeSceneObserverError(
                    "description transport error claims output evidence"
                )
            if self.rubrics != _rubric_error_pair(
                "observer_transport_failed", "PrototypeSceneTransportFailure"
            ):
                raise PrototypeSceneObserverError(
                    "description transport failure is not exhaustive ERROR"
                )
        elif self.status is PrototypeSceneObserverStatus.INTERNAL_ERROR:
            if self.receipt is not None or payload is not None:
                raise PrototypeSceneObserverError(
                    "description internal error claims output evidence"
                )
            if (
                self.failure_code != "observer_internal_error"
                or self.rubrics
                != _rubric_error_pair(
                    "observer_internal_error", "PrototypeSceneInternalError"
                )
            ):
                raise PrototypeSceneObserverError(
                    "description internal failure is not exhaustive ERROR"
                )
        else:  # pragma: no cover - constructor rejects prerequisite above.
            raise PrototypeSceneObserverError("description status is not exhaustive")
        if self.receipt is not None:
            try:
                validate_codex_receipt(self.receipt.to_dict())
            except (CodexProposerFailure, TypeError, ValueError) as exc:
                raise PrototypeSceneObserverError("description receipt is invalid") from exc
            if (
                self.receipt.input_digest_schema != NAMED_IMAGE_INPUT_DIGEST_SCHEMA
                or self.receipt.requested_model != self.model
                or self.receipt.requested_reasoning_effort != self.reasoning_effort
                or self.receipt.prompt_digest != self.prompt_digest
                or self.receipt.output_schema_digest != self.output_schema_digest
                or self.receipt.structured_output_digest != canonical_digest(payload)
                or self.receipt.panel_view_digest
                != canonical_digest([item.to_data() for item in self.presentation])
                or (
                    self.expected_launcher_digest is not None
                    and self.receipt.codex_launcher_digest
                    != self.expected_launcher_digest
                )
                or self.receipt.cloud_config_bundle_cache_binding
                != self.cloud_policy_cache_binding
                or self.receipt.model_catalog_digest
                != self.model_catalog_digest
                or self.receipt.tool_surface_attestation_digest
                != self.no_tools_attestation_digest
            ):
                raise PrototypeSceneObserverError(
                    "description receipt bindings differ"
                )
        computed = canonical_digest(_description_artifact_preimage(self))
        if self.artifact_digest != computed:
            raise PrototypeSceneObserverError("description artifact digest differs")
        object.__setattr__(self, "_sealed_digest", computed)

    def to_data(self) -> dict[str, object]:
        return {
            **_description_artifact_preimage(self),
            "artifact_digest": self.artifact_digest,
        }

    @classmethod
    def from_data(
        cls,
        value: Mapping[str, Any],
        *,
        expected_artifact_digest: str | None = None,
    ) -> "PrototypeRubricDescriptionArtifact":
        raw = _exact_fields(
            value,
            {
                "schema",
                "status",
                "plan_digest",
                "catalog_digest",
                "presentation",
                "prompt_digest",
                "output_schema_digest",
                "protocol_digest",
                "source_digest",
                "transport_source_digest",
                "model",
                "reasoning_effort",
                "model_digest",
                "expected_launcher_digest",
                "cloud_policy_cache_binding",
                "model_catalog_digest",
                "no_tools_attestation_digest",
                "environment_digest",
                "model_payload",
                "receipt",
                "failure_code",
                "failure_type",
                "failure_digest",
                "rubrics",
                "runtime_authority",
                "artifact_digest",
            },
            "prototype rubric description artifact",
        )
        if raw["schema"] != PROTOTYPE_RUBRIC_DESCRIPTION_ARTIFACT_SCHEMA:
            raise PrototypeSceneObserverError("unsupported description artifact")
        authority = raw["runtime_authority"]
        if not isinstance(authority, Mapping):
            raise PrototypeSceneObserverError("description authority is invalid")
        _validate_authority(authority)
        raw_presentation = _json_list(raw["presentation"], "presentation")
        raw_rubrics = _json_list(raw["rubrics"], "rubrics")
        if any(not isinstance(item, Mapping) for item in (*raw_presentation, *raw_rubrics)):
            raise PrototypeSceneObserverError("description child record is invalid")
        raw_receipt = raw["receipt"]
        result = cls(
            status=PrototypeSceneObserverStatus(raw["status"]),
            plan_digest=raw["plan_digest"],
            catalog_digest=raw["catalog_digest"],
            presentation=tuple(
                PrototypeImageIdentity.from_data(item) for item in raw_presentation
            ),
            prompt_digest=raw["prompt_digest"],
            output_schema_digest=raw["output_schema_digest"],
            protocol_digest=raw["protocol_digest"],
            source_digest=raw["source_digest"],
            transport_source_digest=raw["transport_source_digest"],
            model=raw["model"],
            reasoning_effort=raw["reasoning_effort"],
            model_digest=raw["model_digest"],
            expected_launcher_digest=raw["expected_launcher_digest"],
            cloud_policy_cache_binding=raw["cloud_policy_cache_binding"],
            model_catalog_digest=raw["model_catalog_digest"],
            no_tools_attestation_digest=raw["no_tools_attestation_digest"],
            environment_digest=raw["environment_digest"],
            model_payload=(
                None
                if raw["model_payload"] is None
                else _canonical_payload(raw["model_payload"])
            ),
            receipt=None if raw_receipt is None else _receipt_from_data(raw_receipt),
            failure_code=raw["failure_code"],
            failure_type=raw["failure_type"],
            failure_digest=raw["failure_digest"],
            rubrics=tuple(PrototypeRubric.from_data(item) for item in raw_rubrics),
            artifact_digest=raw["artifact_digest"],
        )
        if expected_artifact_digest is not None and result.artifact_digest != (
            _require_digest(
                expected_artifact_digest, "expected description artifact digest"
            )
        ):
            raise PrototypeSceneObserverError(
                "description differs from external artifact commitment"
            )
        if result.to_data() != dict(raw):
            raise PrototypeSceneObserverError(
                "description artifact is not canonically represented"
            )
        return result

    def assert_untampered(self) -> None:
        computed = canonical_digest(_description_artifact_preimage(self))
        if computed != self.artifact_digest or computed != self._sealed_digest:
            raise PrototypeSceneObserverError(
                "description artifact changed after sealing"
            )


def _scene_artifact_preimage(
    artifact: "PrototypeSceneObserverArtifact",
) -> dict[str, object]:
    return {
        "schema": PROTOTYPE_SCENE_OBSERVER_ARTIFACT_SCHEMA,
        "status": artifact.status.value,
        "plan_digest": artifact.plan_digest,
        "observation_context_digest": artifact.observation_context_digest,
        "catalog_digest": artifact.catalog_digest,
        "rubric_description_digest": artifact.rubric_description_digest,
        "scene_task_id": artifact.scene_task_id,
        "scene_panel_id": artifact.scene_panel_id,
        "scene_digest": artifact.scene_digest,
        "presentation": [item.to_data() for item in artifact.presentation],
        "prompt_digest": artifact.prompt_digest,
        "output_schema_digest": artifact.output_schema_digest,
        "protocol_digest": artifact.protocol_digest,
        "source_digest": artifact.source_digest,
        "transport_source_digest": artifact.transport_source_digest,
        "model": artifact.model,
        "reasoning_effort": artifact.reasoning_effort,
        "model_digest": artifact.model_digest,
        "expected_launcher_digest": artifact.expected_launcher_digest,
        "cloud_policy_cache_binding": artifact.cloud_policy_cache_binding,
        "model_catalog_digest": artifact.model_catalog_digest,
        "no_tools_attestation_digest": artifact.no_tools_attestation_digest,
        "environment_digest": artifact.environment_digest,
        "model_payload": artifact.model_payload,
        "receipt": None if artifact.receipt is None else artifact.receipt.to_dict(),
        "failure_code": artifact.failure_code,
        "failure_type": artifact.failure_type,
        "failure_digest": artifact.failure_digest,
        "description": artifact.description,
        "scores": [item.to_data() for item in artifact.scores],
        "runtime_authority": _authority_data(),
    }


@dataclass(frozen=True, slots=True)
class PrototypeSceneObserverArtifact:
    status: PrototypeSceneObserverStatus
    plan_digest: str
    observation_context_digest: str
    catalog_digest: str
    rubric_description_digest: str
    scene_task_id: str
    scene_panel_id: str
    scene_digest: str
    presentation: tuple[PrototypeImageIdentity, ...]
    prompt_digest: str
    output_schema_digest: str
    protocol_digest: str
    source_digest: str
    transport_source_digest: str
    model: str
    reasoning_effort: str
    model_digest: str
    expected_launcher_digest: str | None
    cloud_policy_cache_binding: str
    model_catalog_digest: str
    no_tools_attestation_digest: str
    environment_digest: str
    model_payload: Mapping[str, Any] | None
    receipt: CodexReceipt | None
    failure_code: str | None
    failure_type: str | None
    failure_digest: str | None
    description: str
    scores: tuple[PrototypeSceneScore, ...]
    artifact_digest: str
    _sealed_digest: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.status, PrototypeSceneObserverStatus):
            raise TypeError("scene status must be PrototypeSceneObserverStatus")
        _require_address(self.plan_digest, "scene cohort plan digest")
        _require_address(
            self.observation_context_digest, "observation context digest"
        )
        for name in (
            "catalog_digest",
            "rubric_description_digest",
            "scene_digest",
            "prompt_digest",
            "output_schema_digest",
            "protocol_digest",
            "source_digest",
            "transport_source_digest",
            "model_digest",
            "environment_digest",
            "artifact_digest",
        ):
            _require_digest(getattr(self, name), name)
        if (
            not isinstance(self.scene_task_id, str)
            or _SCENE_TASK_ID.fullmatch(self.scene_task_id) is None
            or not isinstance(self.scene_panel_id, str)
            or _SCENE_PANEL_ID.fullmatch(self.scene_panel_id) is None
            or not self.scene_panel_id.startswith(f"bd/{self.scene_task_id}/")
            or not self.scene_panel_id.endswith(".png")
        ):
            raise PrototypeSceneObserverError(
                "scheduled scene task/panel identity is invalid"
            )
        if self.source_digest != prototype_scene_observer_source_digest():
            raise PrototypeSceneObserverError("scene source digest differs")
        if self.transport_source_digest != prototype_scene_transport_source_digest():
            raise PrototypeSceneObserverError("scene transport source differs")
        if self.model_digest != prototype_scene_observer_model_digest(
            self.model, self.reasoning_effort
        ):
            raise PrototypeSceneObserverError("scene model digest differs")
        if self.expected_launcher_digest is not None:
            _require_digest(self.expected_launcher_digest, "scene launcher commitment")
        if self.cloud_policy_cache_binding != "absent":
            _require_address(
                self.cloud_policy_cache_binding, "scene policy cache binding"
            )
        _require_digest(self.model_catalog_digest, "scene model catalog digest")
        _require_digest(
            self.no_tools_attestation_digest,
            "scene no-tools attestation digest",
        )
        if self.environment_digest != prototype_scene_observer_environment_digest(
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            expected_launcher_digest=self.expected_launcher_digest,
            cloud_policy_cache_binding=self.cloud_policy_cache_binding,
            model_catalog_digest=self.model_catalog_digest,
            no_tools_attestation_digest=self.no_tools_attestation_digest,
        ):
            raise PrototypeSceneObserverError(
                "scene precommitted environment digest differs"
            )
        expected_names = (
            "scene.png",
            *[
                f"{group_id}_ref_{index}.png"
                for group_id in PROTOTYPE_GROUP_IDS
                for index in range(3)
            ],
        )
        if tuple(item.name for item in self.presentation) != expected_names:
            raise PrototypeSceneObserverError("scene presentation order differs")
        _validate_prose(self.description, "scene audit description")
        if tuple((item.tag_id, item.group_id) for item in self.scores) != tuple(
            zip(OPAQUE_TAG_IDS, PROTOTYPE_GROUP_IDS, strict=True)
        ):
            raise PrototypeSceneObserverError("scene scores are not exhaustive")
        schema = prototype_scene_observer_output_schema()
        if self.output_schema_digest != canonical_digest(schema):
            raise PrototypeSceneObserverError("scene schema digest differs")
        if self.protocol_digest != prototype_scene_scoring_protocol_digest():
            raise PrototypeSceneObserverError("scene protocol digest differs")
        payload = (
            None
            if self.model_payload is None
            else _canonical_payload(self.model_payload)
        )
        object.__setattr__(self, "model_payload", payload)
        expected_failure = _failure_digest(
            "whole-scene-scoring",
            self.status,
            self.failure_code,
            self.failure_type,
            payload,
        )
        if self.failure_digest != expected_failure:
            raise PrototypeSceneObserverError("scene failure digest differs")
        if self.status is PrototypeSceneObserverStatus.SUCCESS:
            if self.receipt is None or payload is None:
                raise PrototypeSceneObserverError(
                    "successful scene observation lacks payload or receipt"
                )
            description, scores = _parse_scene_payload(payload)
            if description != self.description or scores != self.scores:
                raise PrototypeSceneObserverError(
                    "successful scene payload differs from sealed values"
                )
        elif self.status is PrototypeSceneObserverStatus.PARSER_ERROR:
            if self.receipt is None or payload is None:
                raise PrototypeSceneObserverError(
                    "scene parser error must retain payload and receipt"
                )
            try:
                _parse_scene_payload(payload)
            except PrototypeSceneObserverError:
                pass
            else:
                raise PrototypeSceneObserverError(
                    "scene parser-error payload is admissible"
                )
            if self.scores != _score_error_pair(
                "observer_payload_rejected", "PrototypeScenePayloadError"
            ):
                raise PrototypeSceneObserverError(
                    "scene parser failure is not exhaustive ERROR"
                )
        elif self.status is PrototypeSceneObserverStatus.TRANSPORT_ERROR:
            if self.receipt is not None or payload is not None:
                raise PrototypeSceneObserverError(
                    "scene transport error claims output evidence"
                )
            if self.scores != _score_error_pair(
                "observer_transport_failed", "PrototypeSceneTransportFailure"
            ):
                raise PrototypeSceneObserverError(
                    "scene transport failure is not exhaustive ERROR"
                )
        elif self.status is PrototypeSceneObserverStatus.INTERNAL_ERROR:
            if self.receipt is not None or payload is not None:
                raise PrototypeSceneObserverError(
                    "scene internal error claims output evidence"
                )
            if (
                self.failure_code != "observer_internal_error"
                or self.scores
                != _score_error_pair(
                    "observer_internal_error", "PrototypeSceneInternalError"
                )
            ):
                raise PrototypeSceneObserverError(
                    "scene internal failure is not exhaustive ERROR"
                )
        else:
            if self.receipt is not None or payload is not None:
                raise PrototypeSceneObserverError(
                    "scene prerequisite error claims model execution"
                )
            if self.scores != _score_error_pair(
                "rubric_prerequisite_failed",
                "PrototypeRubricPrerequisiteFailure",
            ):
                raise PrototypeSceneObserverError(
                    "scene prerequisite failure is not exhaustive ERROR"
                )
        if self.receipt is not None:
            try:
                validate_codex_receipt(self.receipt.to_dict())
            except (CodexProposerFailure, TypeError, ValueError) as exc:
                raise PrototypeSceneObserverError("scene receipt is invalid") from exc
            if (
                self.receipt.input_digest_schema != NAMED_IMAGE_INPUT_DIGEST_SCHEMA
                or self.receipt.requested_model != self.model
                or self.receipt.requested_reasoning_effort != self.reasoning_effort
                or self.receipt.prompt_digest != self.prompt_digest
                or self.receipt.output_schema_digest != self.output_schema_digest
                or self.receipt.structured_output_digest != canonical_digest(payload)
                or self.receipt.panel_view_digest
                != canonical_digest([item.to_data() for item in self.presentation])
                or (
                    self.expected_launcher_digest is not None
                    and self.receipt.codex_launcher_digest
                    != self.expected_launcher_digest
                )
                or self.receipt.cloud_config_bundle_cache_binding
                != self.cloud_policy_cache_binding
                or self.receipt.model_catalog_digest
                != self.model_catalog_digest
                or self.receipt.tool_surface_attestation_digest
                != self.no_tools_attestation_digest
            ):
                raise PrototypeSceneObserverError("scene receipt bindings differ")
        computed = canonical_digest(_scene_artifact_preimage(self))
        if self.artifact_digest != computed:
            raise PrototypeSceneObserverError("scene artifact digest differs")
        object.__setattr__(self, "_sealed_digest", computed)

    def to_data(self) -> dict[str, object]:
        return {
            **_scene_artifact_preimage(self),
            "artifact_digest": self.artifact_digest,
        }

    @classmethod
    def from_data(
        cls,
        value: Mapping[str, Any],
        *,
        expected_artifact_digest: str | None = None,
    ) -> "PrototypeSceneObserverArtifact":
        raw = _exact_fields(
            value,
            {
                "schema",
                "status",
                "plan_digest",
                "observation_context_digest",
                "catalog_digest",
                "rubric_description_digest",
                "scene_task_id",
                "scene_panel_id",
                "scene_digest",
                "presentation",
                "prompt_digest",
                "output_schema_digest",
                "protocol_digest",
                "source_digest",
                "transport_source_digest",
                "model",
                "reasoning_effort",
                "model_digest",
                "expected_launcher_digest",
                "cloud_policy_cache_binding",
                "model_catalog_digest",
                "no_tools_attestation_digest",
                "environment_digest",
                "model_payload",
                "receipt",
                "failure_code",
                "failure_type",
                "failure_digest",
                "description",
                "scores",
                "runtime_authority",
                "artifact_digest",
            },
            "prototype scene observer artifact",
        )
        if raw["schema"] != PROTOTYPE_SCENE_OBSERVER_ARTIFACT_SCHEMA:
            raise PrototypeSceneObserverError("unsupported scene artifact")
        authority = raw["runtime_authority"]
        if not isinstance(authority, Mapping):
            raise PrototypeSceneObserverError("scene authority is invalid")
        _validate_authority(authority)
        raw_presentation = _json_list(raw["presentation"], "scene presentation")
        raw_scores = _json_list(raw["scores"], "scene scores")
        if any(not isinstance(item, Mapping) for item in (*raw_presentation, *raw_scores)):
            raise PrototypeSceneObserverError("scene child record is invalid")
        raw_receipt = raw["receipt"]
        result = cls(
            status=PrototypeSceneObserverStatus(raw["status"]),
            plan_digest=raw["plan_digest"],
            observation_context_digest=raw["observation_context_digest"],
            catalog_digest=raw["catalog_digest"],
            rubric_description_digest=raw["rubric_description_digest"],
            scene_task_id=raw["scene_task_id"],
            scene_panel_id=raw["scene_panel_id"],
            scene_digest=raw["scene_digest"],
            presentation=tuple(
                PrototypeImageIdentity.from_data(item) for item in raw_presentation
            ),
            prompt_digest=raw["prompt_digest"],
            output_schema_digest=raw["output_schema_digest"],
            protocol_digest=raw["protocol_digest"],
            source_digest=raw["source_digest"],
            transport_source_digest=raw["transport_source_digest"],
            model=raw["model"],
            reasoning_effort=raw["reasoning_effort"],
            model_digest=raw["model_digest"],
            expected_launcher_digest=raw["expected_launcher_digest"],
            cloud_policy_cache_binding=raw["cloud_policy_cache_binding"],
            model_catalog_digest=raw["model_catalog_digest"],
            no_tools_attestation_digest=raw["no_tools_attestation_digest"],
            environment_digest=raw["environment_digest"],
            model_payload=(
                None
                if raw["model_payload"] is None
                else _canonical_payload(raw["model_payload"])
            ),
            receipt=None if raw_receipt is None else _receipt_from_data(raw_receipt),
            failure_code=raw["failure_code"],
            failure_type=raw["failure_type"],
            failure_digest=raw["failure_digest"],
            description=raw["description"],
            scores=tuple(PrototypeSceneScore.from_data(item) for item in raw_scores),
            artifact_digest=raw["artifact_digest"],
        )
        if expected_artifact_digest is not None and result.artifact_digest != (
            _require_digest(expected_artifact_digest, "expected scene artifact digest")
        ):
            raise PrototypeSceneObserverError(
                "scene differs from external artifact commitment"
            )
        if result.to_data() != dict(raw):
            raise PrototypeSceneObserverError(
                "scene artifact is not canonically represented"
            )
        return result

    def assert_untampered(self) -> None:
        computed = canonical_digest(_scene_artifact_preimage(self))
        if computed != self.artifact_digest or computed != self._sealed_digest:
            raise PrototypeSceneObserverError("scene artifact changed after sealing")

    def to_calibration_observation_data(
        self, *, calibration_plan_digest: str
    ) -> Mapping[str, Any]:
        """Adapt this already-frozen scene turn to calibration's strict record.

        The requested calibration identity must have been bound as
        ``observation_context_digest`` before scene bytes were staged.  Parser
        and transport outcomes still represent one observer call and keep two
        typed failure scores.  A prerequisite failure represents zero calls and
        is deliberately inadmissible.
        """

        self.assert_untampered()
        expected = _require_address(
            calibration_plan_digest, "calibration plan digest"
        )
        if self.observation_context_digest != expected:
            raise PrototypeSceneObserverError(
                "scene did not precommit the requested calibration plan"
            )
        if self.status is PrototypeSceneObserverStatus.PREREQUISITE_ERROR:
            raise PrototypeSceneObserverError(
                "prerequisite failure made zero observer calls"
            )
        if self.expected_launcher_digest is None:
            raise PrototypeSceneObserverError(
                "calibration requires an external launcher byte commitment"
            )
        from bongard.prototype_scene_calibration import (
            OBSERVER_ADAPTER_PROTOCOL_ID,
            PrototypeSceneCalibrationObservation,
            PrototypeSceneScoreStatus,
            PrototypeSceneTagScore,
        )

        adapted: list[PrototypeSceneTagScore] = []
        for score in self.scores:
            if self.status is PrototypeSceneObserverStatus.PARSER_ERROR:
                status = PrototypeSceneScoreStatus.PARSER_ERROR
            elif self.status is PrototypeSceneObserverStatus.TRANSPORT_ERROR:
                status = PrototypeSceneScoreStatus.TRANSPORT_ERROR
            elif score.state is PrototypeSceneScoreState.SCORED:
                status = PrototypeSceneScoreStatus.SCORE
            elif score.state is PrototypeSceneScoreState.INDETERMINATE:
                status = PrototypeSceneScoreStatus.INDETERMINATE
            else:
                status = PrototypeSceneScoreStatus.ERROR
            if status is PrototypeSceneScoreStatus.SCORE:
                adapted.append(
                    PrototypeSceneTagScore(
                        tag_id=score.tag_id,
                        status=status,
                        lower_ppm=score.lower_ppm,
                        upper_ppm=score.upper_ppm,
                        reason_code="scored",
                        error_type=None,
                    )
                )
            else:
                reason_code = score.reason_code or "observer_indeterminate"
                error_type = score.error_type or "PrototypeSceneIndeterminate"
                adapted.append(
                    PrototypeSceneTagScore(
                        tag_id=score.tag_id,
                        status=status,
                        lower_ppm=None,
                        upper_ppm=None,
                        reason_code=reason_code,
                        error_type=error_type,
                    )
                )
        observation = PrototypeSceneCalibrationObservation(
            calibration_plan_digest=expected,
            cohort_plan_digest=self.plan_digest,
            task_id=self.scene_task_id,
            panel_id=self.scene_panel_id,
            observer_artifact_digest="sha256:" + self.artifact_digest,
            observer_artifact_schema=PROTOTYPE_SCENE_OBSERVER_ARTIFACT_SCHEMA,
            description_catalog_digest=(
                "sha256:" + self.rubric_description_digest
            ),
            prototype_reference_digest="sha256:" + self.catalog_digest,
            observer_protocol_id=PROTOTYPE_SCENE_OBSERVER_PROTOCOL_ID,
            observer_protocol_digest="sha256:" + self.protocol_digest,
            model_id=self.model,
            model_identity_digest="sha256:" + self.model_digest,
            environment_digest="sha256:" + self.environment_digest,
            observer_call_count=1,
            scores=tuple(adapted),  # type: ignore[arg-type]
            adapter_protocol_id=OBSERVER_ADAPTER_PROTOCOL_ID,
        )
        return observation.to_data()


NamedImageTransport = Callable[..., CodexStructuredResult]


def _reference_presentation(
    catalog: PrototypeReferenceCatalog,
    prototype_png_by_panel_id: Mapping[str, bytes],
) -> tuple[tuple[str, bytes], ...]:
    catalog.assert_untampered()
    panel_ids = tuple(item.source_panel_id for item in catalog.bindings)
    if (
        not isinstance(prototype_png_by_panel_id, Mapping)
        or any(not isinstance(key, str) for key in prototype_png_by_panel_id)
        or set(prototype_png_by_panel_id) != set(panel_ids)
    ):
        raise PrototypeSceneObserverError(
            "prototype bytes differ from exact catalog key set"
        )
    values: list[tuple[str, bytes]] = []
    for binding in catalog.bindings:
        data = _validate_exact_png(
            prototype_png_by_panel_id[binding.source_panel_id], binding.name
        )
        if len(data) != binding.byte_count or hashlib.sha256(data).hexdigest() != (
            binding.content_digest
        ):
            raise PrototypeSceneObserverError(
                "prototype bytes differ from catalog identity"
            )
        values.append((binding.name, data))
    return tuple(values)


def _image_identities(
    presentation: Sequence[tuple[str, bytes]],
) -> tuple[PrototypeImageIdentity, ...]:
    return tuple(
        PrototypeImageIdentity(name, len(data), hashlib.sha256(data).hexdigest())
        for name, data in presentation
    )


def _assert_model_visible_boundary(
    prompt: str,
    schema: Mapping[str, Any],
    names: Sequence[str],
    *,
    hidden_values: Sequence[str],
) -> None:
    envelope = prompt + "\n" + json.dumps(schema, sort_keys=True) + "\n" + "\n".join(names)
    for word in ("task", "side", "label", "path", "candidate", "formula", "query"):
        if re.search(rf"\b{word}s?\b", envelope, re.IGNORECASE):
            raise PrototypeSceneObserverError(
                "model-visible envelope contains experimental vocabulary"
            )
    if any(value and value in envelope for value in hidden_values):
        raise PrototypeSceneObserverError(
            "model-visible envelope discloses an internal identity"
        )


def _stage_and_call(
    presentation: Sequence[tuple[str, bytes]],
    *,
    prompt: str,
    schema: Mapping[str, Any],
    model: str,
    reasoning_effort: str,
    minutes: int,
    verbose: bool,
    executable: str,
    cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot | None,
    expected_launcher_digest: str | None,
    model_catalog_snapshot: CodexModelCatalogSnapshot,
    no_tools_attestation: CodexNoToolsAttestation,
    transport: NamedImageTransport,
) -> tuple[dict[str, Any], CodexReceipt]:
    with tempfile.TemporaryDirectory(prefix="bongard-prototype-observer-") as raw:
        directory = Path(raw)
        paths: list[str] = []
        for name, data in presentation:
            target = directory / name
            target.write_bytes(data)
            paths.append(str(target.resolve()))
        names = tuple(name for name, _ in presentation)
        result = transport(
            prompt,
            tuple(paths),
            names,
            schema,
            model=model,
            reasoning_effort=reasoning_effort,
            minutes=minutes,
            verbose=verbose,
            executable=executable,
            cloud_policy_cache_snapshot=cloud_policy_cache_snapshot,
            model_catalog_snapshot=model_catalog_snapshot,
            expected_launcher_digest=expected_launcher_digest,
            tool_surface_attestation=no_tools_attestation,
            expected_tool_surface_attestation_digest=(
                no_tools_attestation.attestation_digest
            ),
        )
        if not isinstance(result, CodexStructuredResult):
            raise PrototypeSceneObserverError(
                "named-image transport returned the wrong result type"
            )
        payload = _canonical_payload(result.payload)
        if not isinstance(result.receipt, CodexReceipt):
            raise PrototypeSceneObserverError(
                "named-image transport returned no CodexReceipt"
            )
        receipt = result.receipt
        validate_codex_named_image_receipt(
            receipt, prompt, tuple(paths), names, schema, payload
        )
        if receipt.requested_model != model or (
            receipt.requested_reasoning_effort != reasoning_effort
        ):
            raise PrototypeSceneObserverError("receipt model request differs")
        if expected_launcher_digest is not None and (
            receipt.codex_launcher_digest != expected_launcher_digest
        ):
            raise PrototypeSceneObserverError("receipt launcher differs")
        if cloud_policy_cache_snapshot is not None and (
            receipt.cloud_config_bundle_cache_binding
            != cloud_policy_cache_snapshot.binding
        ):
            raise PrototypeSceneObserverError("receipt policy cache differs")
        if (
            receipt.model_catalog_digest != model_catalog_snapshot.raw_digest
            or receipt.tool_surface_attestation_digest
            != no_tools_attestation.attestation_digest
        ):
            raise PrototypeSceneObserverError(
                "receipt no-tools preflight binding differs"
            )
        for path, (_, expected) in zip(paths, presentation, strict=True):
            if Path(path).read_bytes() != expected:
                raise PrototypeSceneObserverError(
                    "named-image presentation changed during execution"
                )
        return payload, receipt


def _build_description_artifact(
    *,
    status: PrototypeSceneObserverStatus,
    catalog: PrototypeReferenceCatalog,
    identities: tuple[PrototypeImageIdentity, ...],
    model: str,
    reasoning_effort: str,
    expected_launcher_digest: str | None,
    cloud_policy_cache_binding: str,
    model_catalog_digest: str,
    no_tools_attestation_digest: str,
    payload: Mapping[str, Any] | None,
    receipt: CodexReceipt | None,
    failure_code: str | None,
    failure_type: str | None,
    rubrics: tuple[PrototypeRubric, ...],
) -> PrototypeRubricDescriptionArtifact:
    prompt = prototype_rubric_description_prompt()
    schema = prototype_rubric_description_output_schema()
    canonical_payload = None if payload is None else _canonical_payload(payload)
    failure_digest = _failure_digest(
        "reference-description",
        status,
        failure_code,
        failure_type,
        canonical_payload,
    )
    environment_digest = prototype_scene_observer_environment_digest(
        model=model,
        reasoning_effort=reasoning_effort,
        expected_launcher_digest=expected_launcher_digest,
        cloud_policy_cache_binding=cloud_policy_cache_binding,
        model_catalog_digest=model_catalog_digest,
        no_tools_attestation_digest=no_tools_attestation_digest,
    )
    values: dict[str, object] = {
        "status": status,
        "plan_digest": catalog.plan_digest,
        "catalog_digest": catalog.catalog_digest,
        "presentation": identities,
        "prompt_digest": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "output_schema_digest": canonical_digest(schema),
        "protocol_digest": prototype_rubric_description_protocol_digest(),
        "source_digest": prototype_scene_observer_source_digest(),
        "transport_source_digest": prototype_scene_transport_source_digest(),
        "model": model,
        "reasoning_effort": reasoning_effort,
        "model_digest": prototype_scene_observer_model_digest(
            model, reasoning_effort
        ),
        "expected_launcher_digest": expected_launcher_digest,
        "cloud_policy_cache_binding": cloud_policy_cache_binding,
        "model_catalog_digest": model_catalog_digest,
        "no_tools_attestation_digest": no_tools_attestation_digest,
        "environment_digest": environment_digest,
        "model_payload": canonical_payload,
        "receipt": receipt,
        "failure_code": failure_code,
        "failure_type": failure_type,
        "failure_digest": failure_digest,
        "rubrics": rubrics,
    }
    provisional = object.__new__(PrototypeRubricDescriptionArtifact)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return PrototypeRubricDescriptionArtifact(
        **values,  # type: ignore[arg-type]
        artifact_digest=canonical_digest(_description_artifact_preimage(provisional)),
    )


def seal_prototype_rubric_description_internal_error(
    catalog: PrototypeReferenceCatalog,
    prototype_png_by_panel_id: Mapping[str, bytes],
    *,
    expected_catalog_digest: str,
    model: str,
    reasoning_effort: str,
    expected_launcher_digest: str | None,
    cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot | None,
    model_catalog_snapshot: CodexModelCatalogSnapshot,
    no_tools_attestation: CodexNoToolsAttestation,
    exception: Exception,
) -> PrototypeRubricDescriptionArtifact:
    """Seal an unexpected local callback failure as exhaustive typed ERROR."""

    if not isinstance(exception, Exception):
        raise TypeError("internal description failure must be Exception")
    if not isinstance(catalog, PrototypeReferenceCatalog):
        raise TypeError("catalog must be PrototypeReferenceCatalog")
    catalog.assert_untampered()
    if catalog.catalog_digest != _require_digest(
        expected_catalog_digest, "expected catalog digest"
    ):
        raise PrototypeSceneObserverError("catalog differs from commitment")
    presentation = _reference_presentation(catalog, prototype_png_by_panel_id)
    identities = _image_identities(presentation)
    if identities != catalog.presentation:
        raise PrototypeSceneObserverError(
            "reference presentation differs from frozen catalog"
        )
    prototype_scene_observer_model_digest(model, reasoning_effort)
    policy_binding = _policy_cache_binding(cloud_policy_cache_snapshot)
    model_catalog_digest, no_tools_attestation_digest = _validate_no_tools_runtime(
        model_catalog_snapshot=model_catalog_snapshot,
        no_tools_attestation=no_tools_attestation,
        expected_launcher_digest=expected_launcher_digest,
        cloud_policy_cache_binding=policy_binding,
    )
    return _build_description_artifact(
        status=PrototypeSceneObserverStatus.INTERNAL_ERROR,
        catalog=catalog,
        identities=identities,
        model=model,
        reasoning_effort=reasoning_effort,
        expected_launcher_digest=expected_launcher_digest,
        cloud_policy_cache_binding=policy_binding,
        model_catalog_digest=model_catalog_digest,
        no_tools_attestation_digest=no_tools_attestation_digest,
        payload=None,
        receipt=None,
        failure_code="observer_internal_error",
        failure_type=_exception_type(exception),
        rubrics=_rubric_error_pair(
            "observer_internal_error", "PrototypeSceneInternalError"
        ),
    )


def describe_prototype_references(
    catalog: PrototypeReferenceCatalog,
    prototype_png_by_panel_id: Mapping[str, bytes],
    *,
    expected_catalog_digest: str,
    model: str,
    reasoning_effort: str = "medium",
    minutes: int = 15,
    verbose: bool = False,
    executable: str = "codex",
    cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot | None = None,
    expected_launcher_digest: str | None = None,
    model_catalog_snapshot: CodexModelCatalogSnapshot,
    no_tools_attestation: CodexNoToolsAttestation,
    transport: NamedImageTransport = run_codex_named_images_structured,
) -> PrototypeRubricDescriptionArtifact:
    """Run the six-reference prose phase through an injected named-image call."""

    if not isinstance(catalog, PrototypeReferenceCatalog):
        raise TypeError("catalog must be PrototypeReferenceCatalog")
    if catalog.catalog_digest != _require_digest(
        expected_catalog_digest, "expected catalog digest"
    ):
        raise PrototypeSceneObserverError(
            "reference catalog differs from external commitment"
        )
    if not callable(transport):
        raise TypeError("transport must be callable")
    prototype_scene_observer_model_digest(model, reasoning_effort)
    policy_binding = _policy_cache_binding(cloud_policy_cache_snapshot)
    model_catalog_digest, no_tools_attestation_digest = _validate_no_tools_runtime(
        model_catalog_snapshot=model_catalog_snapshot,
        no_tools_attestation=no_tools_attestation,
        expected_launcher_digest=expected_launcher_digest,
        cloud_policy_cache_binding=policy_binding,
    )
    presentation = _reference_presentation(
        catalog, prototype_png_by_panel_id
    )
    identities = _image_identities(presentation)
    if identities != catalog.presentation:
        raise PrototypeSceneObserverError(
            "reference presentation differs from frozen catalog"
        )
    prompt = prototype_rubric_description_prompt()
    schema = prototype_rubric_description_output_schema()
    validate_codex_strict_output_schema(schema)
    _assert_model_visible_boundary(
        prompt,
        schema,
        tuple(name for name, _ in presentation),
        hidden_values=(
            catalog.plan_digest,
            *OPAQUE_TAG_IDS,
            *(item.source_panel_id for item in catalog.bindings),
        ),
    )
    try:
        payload, receipt = _stage_and_call(
            presentation,
            prompt=prompt,
            schema=schema,
            model=model,
            reasoning_effort=reasoning_effort,
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
        failure_type = type(exc).__name__
        if _CODE.fullmatch(failure_type) is None:
            failure_type = "UnclassifiedTransportFailure"
        return _build_description_artifact(
            status=PrototypeSceneObserverStatus.TRANSPORT_ERROR,
            catalog=catalog,
            identities=identities,
            model=model,
            reasoning_effort=reasoning_effort,
            expected_launcher_digest=expected_launcher_digest,
            cloud_policy_cache_binding=policy_binding,
            model_catalog_digest=model_catalog_digest,
            no_tools_attestation_digest=no_tools_attestation_digest,
            payload=None,
            receipt=None,
            failure_code="transport_failed",
            failure_type=failure_type,
            rubrics=_rubric_error_pair(
                "observer_transport_failed", "PrototypeSceneTransportFailure"
            ),
        )
    try:
        rubrics = _parse_rubric_payload(payload)
        return _build_description_artifact(
            status=PrototypeSceneObserverStatus.SUCCESS,
            catalog=catalog,
            identities=identities,
            model=model,
            reasoning_effort=reasoning_effort,
            expected_launcher_digest=expected_launcher_digest,
            cloud_policy_cache_binding=policy_binding,
            model_catalog_digest=model_catalog_digest,
            no_tools_attestation_digest=no_tools_attestation_digest,
            payload=payload,
            receipt=receipt,
            failure_code=None,
            failure_type=None,
            rubrics=rubrics,
        )
    except (PrototypeSceneObserverError, TypeError, ValueError):
        return _build_description_artifact(
            status=PrototypeSceneObserverStatus.PARSER_ERROR,
            catalog=catalog,
            identities=identities,
            model=model,
            reasoning_effort=reasoning_effort,
            expected_launcher_digest=expected_launcher_digest,
            cloud_policy_cache_binding=policy_binding,
            model_catalog_digest=model_catalog_digest,
            no_tools_attestation_digest=no_tools_attestation_digest,
            payload=payload,
            receipt=receipt,
            failure_code="payload_rejected",
            failure_type="PrototypeScenePayloadError",
            rubrics=_rubric_error_pair(
                "observer_payload_rejected", "PrototypeScenePayloadError"
            ),
        )


def _validate_description_parent(
    artifact: PrototypeRubricDescriptionArtifact,
    catalog: PrototypeReferenceCatalog,
    identities: tuple[PrototypeImageIdentity, ...],
    expected_artifact_digest: str,
) -> None:
    artifact.assert_untampered()
    if artifact.artifact_digest != _require_digest(
        expected_artifact_digest, "expected description artifact digest"
    ):
        raise PrototypeSceneObserverError(
            "description differs from external artifact commitment"
        )
    if (
        artifact.plan_digest != catalog.plan_digest
        or artifact.catalog_digest != catalog.catalog_digest
        or artifact.presentation != identities
    ):
        raise PrototypeSceneObserverError(
            "description artifact binds another reference catalog"
        )


def verify_prototype_rubric_description_artifact(
    artifact: PrototypeRubricDescriptionArtifact,
    catalog: PrototypeReferenceCatalog,
    prototype_png_by_panel_id: Mapping[str, bytes],
    *,
    expected_catalog_digest: str,
    expected_artifact_digest: str,
) -> PrototypeRubricDescriptionArtifact:
    """Cold-verify reference bytes, prompt, payload, receipt, and archive."""

    if not isinstance(artifact, PrototypeRubricDescriptionArtifact):
        raise TypeError("artifact must be PrototypeRubricDescriptionArtifact")
    if catalog.catalog_digest != _require_digest(
        expected_catalog_digest, "expected catalog digest"
    ):
        raise PrototypeSceneObserverError("catalog differs from commitment")
    presentation = _reference_presentation(catalog, prototype_png_by_panel_id)
    identities = _image_identities(presentation)
    _validate_description_parent(
        artifact, catalog, identities, expected_artifact_digest
    )
    prompt = prototype_rubric_description_prompt()
    schema = prototype_rubric_description_output_schema()
    if artifact.receipt is not None:
        if artifact.model_payload is None:
            raise PrototypeSceneObserverError(
                "receipted description lacks its payload"
            )
        with tempfile.TemporaryDirectory(
            prefix="bongard-prototype-description-replay-"
        ) as raw:
            directory = Path(raw)
            paths: list[str] = []
            for name, data in presentation:
                target = directory / name
                target.write_bytes(data)
                paths.append(str(target.resolve()))
            try:
                validate_codex_named_image_receipt(
                    artifact.receipt,
                    prompt,
                    tuple(paths),
                    tuple(name for name, _ in presentation),
                    schema,
                    artifact.model_payload,
                )
            except (CodexProposerFailure, TypeError, ValueError) as exc:
                raise PrototypeSceneObserverError(
                    "cold description receipt validation failed"
                ) from exc
    if PrototypeRubricDescriptionArtifact.from_data(
        artifact.to_data(), expected_artifact_digest=expected_artifact_digest
    ) != artifact:
        raise PrototypeSceneObserverError("description cold round trip differs")
    return artifact


def _build_scene_artifact(
    *,
    status: PrototypeSceneObserverStatus,
    catalog: PrototypeReferenceCatalog,
    rubric_artifact: PrototypeRubricDescriptionArtifact,
    observation_context_digest: str,
    scene_task_id: str,
    scene_panel_id: str,
    exact_scene_png_bytes: bytes,
    identities: tuple[PrototypeImageIdentity, ...],
    model: str,
    reasoning_effort: str,
    expected_launcher_digest: str | None,
    cloud_policy_cache_binding: str,
    model_catalog_digest: str,
    no_tools_attestation_digest: str,
    payload: Mapping[str, Any] | None,
    receipt: CodexReceipt | None,
    failure_code: str | None,
    failure_type: str | None,
    description: str,
    scores: tuple[PrototypeSceneScore, ...],
) -> PrototypeSceneObserverArtifact:
    prompt = prototype_scene_observer_prompt(rubric_artifact.rubrics)
    schema = prototype_scene_observer_output_schema()
    canonical_payload = None if payload is None else _canonical_payload(payload)
    failure_digest = _failure_digest(
        "whole-scene-scoring",
        status,
        failure_code,
        failure_type,
        canonical_payload,
    )
    environment_digest = prototype_scene_observer_environment_digest(
        model=model,
        reasoning_effort=reasoning_effort,
        expected_launcher_digest=expected_launcher_digest,
        cloud_policy_cache_binding=cloud_policy_cache_binding,
        model_catalog_digest=model_catalog_digest,
        no_tools_attestation_digest=no_tools_attestation_digest,
    )
    values: dict[str, object] = {
        "status": status,
        "plan_digest": catalog.plan_digest,
        "observation_context_digest": observation_context_digest,
        "catalog_digest": catalog.catalog_digest,
        "rubric_description_digest": rubric_artifact.artifact_digest,
        "scene_task_id": scene_task_id,
        "scene_panel_id": scene_panel_id,
        "scene_digest": hashlib.sha256(exact_scene_png_bytes).hexdigest(),
        "presentation": identities,
        "prompt_digest": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "output_schema_digest": canonical_digest(schema),
        "protocol_digest": prototype_scene_scoring_protocol_digest(),
        "source_digest": prototype_scene_observer_source_digest(),
        "transport_source_digest": prototype_scene_transport_source_digest(),
        "model": model,
        "reasoning_effort": reasoning_effort,
        "model_digest": prototype_scene_observer_model_digest(
            model, reasoning_effort
        ),
        "expected_launcher_digest": expected_launcher_digest,
        "cloud_policy_cache_binding": cloud_policy_cache_binding,
        "model_catalog_digest": model_catalog_digest,
        "no_tools_attestation_digest": no_tools_attestation_digest,
        "environment_digest": environment_digest,
        "model_payload": canonical_payload,
        "receipt": receipt,
        "failure_code": failure_code,
        "failure_type": failure_type,
        "failure_digest": failure_digest,
        "description": description,
        "scores": scores,
    }
    provisional = object.__new__(PrototypeSceneObserverArtifact)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return PrototypeSceneObserverArtifact(
        **values,  # type: ignore[arg-type]
        artifact_digest=canonical_digest(_scene_artifact_preimage(provisional)),
    )


def seal_prototype_scene_internal_error(
    exact_scene_png_bytes: bytes,
    *,
    scene_task_id: str,
    scene_panel_id: str,
    observation_context_digest: str,
    expected_scene_sha256: str,
    catalog: PrototypeReferenceCatalog,
    prototype_png_by_panel_id: Mapping[str, bytes],
    expected_catalog_digest: str,
    rubric_artifact: PrototypeRubricDescriptionArtifact,
    expected_rubric_artifact_digest: str,
    model: str,
    reasoning_effort: str,
    expected_launcher_digest: str | None,
    cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot | None,
    model_catalog_snapshot: CodexModelCatalogSnapshot,
    no_tools_attestation: CodexNoToolsAttestation,
    exception: Exception,
) -> PrototypeSceneObserverArtifact:
    """Seal an unexpected local scene failure without fabricating absence."""

    if not isinstance(exception, Exception):
        raise TypeError("internal scene failure must be Exception")
    context = _require_address(
        observation_context_digest, "observation context digest"
    )
    scene = _validate_exact_png(exact_scene_png_bytes, "scene")
    if hashlib.sha256(scene).hexdigest() != _require_digest(
        expected_scene_sha256, "expected scene SHA-256"
    ):
        raise PrototypeSceneObserverError(
            "scene differs from external SHA-256 commitment"
        )
    if (
        not isinstance(scene_task_id, str)
        or _SCENE_TASK_ID.fullmatch(scene_task_id) is None
        or not isinstance(scene_panel_id, str)
        or _SCENE_PANEL_ID.fullmatch(scene_panel_id) is None
        or not scene_panel_id.startswith(f"bd/{scene_task_id}/")
        or not scene_panel_id.endswith(".png")
    ):
        raise PrototypeSceneObserverError("scheduled scene identity is invalid")
    refs = _reference_presentation(catalog, prototype_png_by_panel_id)
    ref_identities = _image_identities(refs)
    _validate_description_parent(
        rubric_artifact,
        catalog,
        ref_identities,
        expected_rubric_artifact_digest,
    )
    if catalog.catalog_digest != _require_digest(
        expected_catalog_digest, "expected catalog digest"
    ):
        raise PrototypeSceneObserverError("catalog differs from commitment")
    verify_prototype_rubric_description_artifact(
        rubric_artifact,
        catalog,
        prototype_png_by_panel_id,
        expected_catalog_digest=expected_catalog_digest,
        expected_artifact_digest=expected_rubric_artifact_digest,
    )
    prototype_scene_observer_model_digest(model, reasoning_effort)
    policy_binding = _policy_cache_binding(cloud_policy_cache_snapshot)
    model_catalog_digest, no_tools_attestation_digest = _validate_no_tools_runtime(
        model_catalog_snapshot=model_catalog_snapshot,
        no_tools_attestation=no_tools_attestation,
        expected_launcher_digest=expected_launcher_digest,
        cloud_policy_cache_binding=policy_binding,
    )
    identities = _image_identities((("scene.png", scene), *refs))
    return _build_scene_artifact(
        status=PrototypeSceneObserverStatus.INTERNAL_ERROR,
        catalog=catalog,
        rubric_artifact=rubric_artifact,
        observation_context_digest=context,
        scene_task_id=scene_task_id,
        scene_panel_id=scene_panel_id,
        exact_scene_png_bytes=scene,
        identities=identities,
        model=model,
        reasoning_effort=reasoning_effort,
        expected_launcher_digest=expected_launcher_digest,
        cloud_policy_cache_binding=policy_binding,
        model_catalog_digest=model_catalog_digest,
        no_tools_attestation_digest=no_tools_attestation_digest,
        payload=None,
        receipt=None,
        failure_code="observer_internal_error",
        failure_type=_exception_type(exception),
        description="The scene measurement ended with an internal failure.",
        scores=_score_error_pair(
            "observer_internal_error", "PrototypeSceneInternalError"
        ),
    )


def observe_prototype_scene(
    exact_scene_png_bytes: bytes,
    *,
    scene_task_id: str,
    scene_panel_id: str,
    observation_context_digest: str,
    expected_scene_sha256: str,
    catalog: PrototypeReferenceCatalog,
    prototype_png_by_panel_id: Mapping[str, bytes],
    expected_catalog_digest: str,
    rubric_artifact: PrototypeRubricDescriptionArtifact,
    expected_rubric_artifact_digest: str,
    model: str,
    reasoning_effort: str = "medium",
    minutes: int = 15,
    verbose: bool = False,
    executable: str = "codex",
    cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot | None = None,
    expected_launcher_digest: str | None = None,
    model_catalog_snapshot: CodexModelCatalogSnapshot,
    no_tools_attestation: CodexNoToolsAttestation,
    transport: NamedImageTransport = run_codex_named_images_structured,
) -> PrototypeSceneObserverArtifact:
    """Score one exact whole scene against two frozen prototype rubrics."""

    if not callable(transport):
        raise TypeError("transport must be callable")
    context = _require_address(
        observation_context_digest, "observation context digest"
    )
    scene = _validate_exact_png(exact_scene_png_bytes, "scene")
    if hashlib.sha256(scene).hexdigest() != _require_digest(
        expected_scene_sha256, "expected scene SHA-256"
    ):
        raise PrototypeSceneObserverError(
            "scene differs from external SHA-256 commitment"
        )
    if (
        not isinstance(scene_task_id, str)
        or _SCENE_TASK_ID.fullmatch(scene_task_id) is None
        or not isinstance(scene_panel_id, str)
        or _SCENE_PANEL_ID.fullmatch(scene_panel_id) is None
        or not scene_panel_id.startswith(f"bd/{scene_task_id}/")
        or not scene_panel_id.endswith(".png")
    ):
        raise PrototypeSceneObserverError("scheduled scene identity is invalid")
    refs = _reference_presentation(catalog, prototype_png_by_panel_id)
    ref_identities = _image_identities(refs)
    _validate_description_parent(
        rubric_artifact,
        catalog,
        ref_identities,
        expected_rubric_artifact_digest,
    )
    if catalog.catalog_digest != _require_digest(
        expected_catalog_digest, "expected catalog digest"
    ):
        raise PrototypeSceneObserverError("catalog differs from commitment")
    verify_prototype_rubric_description_artifact(
        rubric_artifact,
        catalog,
        prototype_png_by_panel_id,
        expected_catalog_digest=expected_catalog_digest,
        expected_artifact_digest=expected_rubric_artifact_digest,
    )
    prototype_scene_observer_model_digest(model, reasoning_effort)
    policy_binding = _policy_cache_binding(cloud_policy_cache_snapshot)
    model_catalog_digest, no_tools_attestation_digest = _validate_no_tools_runtime(
        model_catalog_snapshot=model_catalog_snapshot,
        no_tools_attestation=no_tools_attestation,
        expected_launcher_digest=expected_launcher_digest,
        cloud_policy_cache_binding=policy_binding,
    )
    presentation = (("scene.png", scene), *refs)
    identities = _image_identities(presentation)
    prompt = prototype_scene_observer_prompt(rubric_artifact.rubrics)
    schema = prototype_scene_observer_output_schema()
    validate_codex_strict_output_schema(schema)
    _assert_model_visible_boundary(
        prompt,
        schema,
        tuple(name for name, _ in presentation),
        hidden_values=(
            catalog.plan_digest,
            context,
            scene_task_id,
            scene_panel_id,
            *OPAQUE_TAG_IDS,
            *(item.source_panel_id for item in catalog.bindings),
        ),
    )
    if rubric_artifact.status is not PrototypeSceneObserverStatus.SUCCESS:
        return _build_scene_artifact(
            status=PrototypeSceneObserverStatus.PREREQUISITE_ERROR,
            catalog=catalog,
            rubric_artifact=rubric_artifact,
            observation_context_digest=context,
            scene_task_id=scene_task_id,
            scene_panel_id=scene_panel_id,
            exact_scene_png_bytes=scene,
            identities=identities,
            model=model,
            reasoning_effort=reasoning_effort,
            expected_launcher_digest=expected_launcher_digest,
            cloud_policy_cache_binding=policy_binding,
            model_catalog_digest=model_catalog_digest,
            no_tools_attestation_digest=no_tools_attestation_digest,
            payload=None,
            receipt=None,
            failure_code="rubric_prerequisite_failed",
            failure_type="PrototypeRubricPrerequisiteFailure",
            description=(
                "Scene observation unavailable because reference descriptions failed."
            ),
            scores=_score_error_pair(
                "rubric_prerequisite_failed",
                "PrototypeRubricPrerequisiteFailure",
            ),
        )
    try:
        payload, receipt = _stage_and_call(
            presentation,
            prompt=prompt,
            schema=schema,
            model=model,
            reasoning_effort=reasoning_effort,
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
        failure_type = type(exc).__name__
        if _CODE.fullmatch(failure_type) is None:
            failure_type = "UnclassifiedTransportFailure"
        return _build_scene_artifact(
            status=PrototypeSceneObserverStatus.TRANSPORT_ERROR,
            catalog=catalog,
            rubric_artifact=rubric_artifact,
            observation_context_digest=context,
            scene_task_id=scene_task_id,
            scene_panel_id=scene_panel_id,
            exact_scene_png_bytes=scene,
            identities=identities,
            model=model,
            reasoning_effort=reasoning_effort,
            expected_launcher_digest=expected_launcher_digest,
            cloud_policy_cache_binding=policy_binding,
            model_catalog_digest=model_catalog_digest,
            no_tools_attestation_digest=no_tools_attestation_digest,
            payload=None,
            receipt=None,
            failure_code="transport_failed",
            failure_type=failure_type,
            description="Scene observation transport failed before values.",
            scores=_score_error_pair(
                "observer_transport_failed", "PrototypeSceneTransportFailure"
            ),
        )
    try:
        description, scores = _parse_scene_payload(payload)
        return _build_scene_artifact(
            status=PrototypeSceneObserverStatus.SUCCESS,
            catalog=catalog,
            rubric_artifact=rubric_artifact,
            observation_context_digest=context,
            scene_task_id=scene_task_id,
            scene_panel_id=scene_panel_id,
            exact_scene_png_bytes=scene,
            identities=identities,
            model=model,
            reasoning_effort=reasoning_effort,
            expected_launcher_digest=expected_launcher_digest,
            cloud_policy_cache_binding=policy_binding,
            model_catalog_digest=model_catalog_digest,
            no_tools_attestation_digest=no_tools_attestation_digest,
            payload=payload,
            receipt=receipt,
            failure_code=None,
            failure_type=None,
            description=description,
            scores=scores,
        )
    except (PrototypeSceneObserverError, TypeError, ValueError):
        return _build_scene_artifact(
            status=PrototypeSceneObserverStatus.PARSER_ERROR,
            catalog=catalog,
            rubric_artifact=rubric_artifact,
            observation_context_digest=context,
            scene_task_id=scene_task_id,
            scene_panel_id=scene_panel_id,
            exact_scene_png_bytes=scene,
            identities=identities,
            model=model,
            reasoning_effort=reasoning_effort,
            expected_launcher_digest=expected_launcher_digest,
            cloud_policy_cache_binding=policy_binding,
            model_catalog_digest=model_catalog_digest,
            no_tools_attestation_digest=no_tools_attestation_digest,
            payload=payload,
            receipt=receipt,
            failure_code="payload_rejected",
            failure_type="PrototypeScenePayloadError",
            description="Scene observation response failed finite value validation.",
            scores=_score_error_pair(
                "observer_payload_rejected", "PrototypeScenePayloadError"
            ),
        )


def verify_prototype_scene_observer_artifact(
    artifact: PrototypeSceneObserverArtifact,
    exact_scene_png_bytes: bytes,
    *,
    expected_scene_task_id: str,
    expected_scene_panel_id: str,
    expected_observation_context_digest: str,
    expected_scene_sha256: str,
    catalog: PrototypeReferenceCatalog,
    prototype_png_by_panel_id: Mapping[str, bytes],
    expected_catalog_digest: str,
    rubric_artifact: PrototypeRubricDescriptionArtifact,
    expected_rubric_artifact_digest: str,
    expected_artifact_digest: str,
) -> PrototypeSceneObserverArtifact:
    """Cold-verify the complete seven-image scoring envelope and archive."""

    if not isinstance(artifact, PrototypeSceneObserverArtifact):
        raise TypeError("artifact must be PrototypeSceneObserverArtifact")
    artifact.assert_untampered()
    if artifact.artifact_digest != _require_digest(
        expected_artifact_digest, "expected scene artifact digest"
    ):
        raise PrototypeSceneObserverError(
            "scene differs from external artifact commitment"
        )
    context = _require_address(
        expected_observation_context_digest,
        "expected observation context digest",
    )
    scene = _validate_exact_png(exact_scene_png_bytes, "scene")
    scene_digest = hashlib.sha256(scene).hexdigest()
    if scene_digest != _require_digest(
        expected_scene_sha256, "expected scene SHA-256"
    ):
        raise PrototypeSceneObserverError("scene differs from byte commitment")
    if catalog.catalog_digest != _require_digest(
        expected_catalog_digest, "expected catalog digest"
    ):
        raise PrototypeSceneObserverError("catalog differs from commitment")
    verify_prototype_rubric_description_artifact(
        rubric_artifact,
        catalog,
        prototype_png_by_panel_id,
        expected_catalog_digest=expected_catalog_digest,
        expected_artifact_digest=expected_rubric_artifact_digest,
    )
    refs = _reference_presentation(catalog, prototype_png_by_panel_id)
    ref_identities = _image_identities(refs)
    _validate_description_parent(
        rubric_artifact,
        catalog,
        ref_identities,
        expected_rubric_artifact_digest,
    )
    presentation = (("scene.png", scene), *refs)
    identities = _image_identities(presentation)
    if (
        artifact.plan_digest != catalog.plan_digest
        or artifact.observation_context_digest != context
        or artifact.catalog_digest != catalog.catalog_digest
        or artifact.rubric_description_digest != rubric_artifact.artifact_digest
        or artifact.scene_task_id != expected_scene_task_id
        or artifact.scene_panel_id != expected_scene_panel_id
        or artifact.scene_digest != scene_digest
        or artifact.presentation != identities
    ):
        raise PrototypeSceneObserverError(
            "scene artifact differs from cold parent reconstruction"
        )
    prompt = prototype_scene_observer_prompt(rubric_artifact.rubrics)
    schema = prototype_scene_observer_output_schema()
    if artifact.prompt_digest != hashlib.sha256(prompt.encode("utf-8")).hexdigest():
        raise PrototypeSceneObserverError("cold scene prompt digest differs")
    if artifact.receipt is not None:
        if artifact.model_payload is None:
            raise PrototypeSceneObserverError("receipted scene lacks its payload")
        with tempfile.TemporaryDirectory(
            prefix="bongard-prototype-scene-replay-"
        ) as raw:
            directory = Path(raw)
            paths: list[str] = []
            for name, data in presentation:
                target = directory / name
                target.write_bytes(data)
                paths.append(str(target.resolve()))
            try:
                validate_codex_named_image_receipt(
                    artifact.receipt,
                    prompt,
                    tuple(paths),
                    tuple(name for name, _ in presentation),
                    schema,
                    artifact.model_payload,
                )
            except (CodexProposerFailure, TypeError, ValueError) as exc:
                raise PrototypeSceneObserverError(
                    "cold scene receipt validation failed"
                ) from exc
    if PrototypeSceneObserverArtifact.from_data(
        artifact.to_data(), expected_artifact_digest=expected_artifact_digest
    ) != artifact:
        raise PrototypeSceneObserverError("scene cold round trip differs")
    return artifact


observe_prototype_whole_scene = observe_prototype_scene


__all__ = [
    "PPM_SCALE",
    "PROTOTYPE_GROUP_IDS",
    "PROTOTYPE_REFERENCE_CATALOG_SCHEMA",
    "PROTOTYPE_RUBRIC_DESCRIPTION_ARTIFACT_SCHEMA",
    "PROTOTYPE_SCENE_OBSERVER_ARTIFACT_SCHEMA",
    "PROTOTYPE_SCENE_OBSERVER_PROTOCOL_ID",
    "NamedImageTransport",
    "PrototypeImageIdentity",
    "PrototypeReferenceBinding",
    "PrototypeReferenceCatalog",
    "PrototypeRubric",
    "PrototypeRubricDescriptionArtifact",
    "PrototypeRubricState",
    "PrototypeSceneObserverArtifact",
    "PrototypeSceneObserverError",
    "PrototypeSceneObserverStatus",
    "PrototypeScenePayloadError",
    "PrototypeSceneScore",
    "PrototypeSceneScoreState",
    "build_prototype_reference_catalog",
    "describe_prototype_references",
    "observe_prototype_scene",
    "observe_prototype_whole_scene",
    "prototype_rubric_description_output_schema",
    "prototype_rubric_description_prompt",
    "prototype_rubric_description_protocol_digest",
    "prototype_scene_observer_environment_digest",
    "prototype_scene_observer_model_digest",
    "prototype_scene_observer_output_schema",
    "prototype_scene_observer_prompt",
    "prototype_scene_observer_source_digest",
    "prototype_scene_scoring_protocol_digest",
    "prototype_scene_transport_source_digest",
    "seal_prototype_rubric_description_internal_error",
    "seal_prototype_scene_internal_error",
    "verify_prototype_reference_catalog",
    "verify_prototype_rubric_description_artifact",
    "verify_prototype_scene_observer_artifact",
]
