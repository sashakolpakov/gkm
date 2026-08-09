"""Predicate-gated engineering release of the remaining historical panels.

The 28-scene historical plan was already exposed by an earlier campaign.  The
remaining sixteen panels are therefore an engineering ablation source, never a
blind benchmark.  This module restores causal discipline for the current
pipeline without making a false blindness claim:

* metadata planning derives ``all plan ordinals - exact support ordinals``;
* no released-panel record is opened before an exact Python predicate and a
  caller-supplied durable-freeze commitment have both been verified;
* geometry/observer consumers receive neutral aliases and pixels only; and
* historical labels are interpreted only after every prediction digest exists.

No model or Lean implementation participates in any identity or decision.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Callable, Mapping

from bongard.canonical import canonical_digest, canonical_json
from bongard.evidence import Disposition
from bongard.object_bongard_panel_rubric_calibration import (
    DEFAULT_OBJECT_BONGARD_PANEL_RUBRIC_CALIBRATION_SOURCE,
    PANEL_RUBRIC_CALIBRATION_SELECTED_ORDINALS,
)
from bongard.object_scene_anchor_python_predicate import (
    ObjectSceneAnchorPythonPredicate,
)
from bongard.object_scene_anchor_version_space import ObjectSceneAnchorOrientation
from bongard.official_panel_archive import ReleasedOfficialPanel
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID
from bongard.transport import MAX_PANEL_PNG_BYTES


OBJECT_SCENE_ANCHOR_EXPOSED_QUERY_PLAN_SCHEMA = (
    "gkm.object-scene-anchor-exposed-query-plan.v1"
)
OBJECT_SCENE_ANCHOR_EXPOSED_QUERY_RELEASE_SCHEMA = (
    "gkm.object-scene-anchor-exposed-query-release.v1"
)
OBJECT_SCENE_ANCHOR_EXPOSED_QUERY_PREDICTION_SCHEMA = (
    "gkm.object-scene-anchor-exposed-query-prediction.v1"
)
OBJECT_SCENE_ANCHOR_EXPOSED_QUERY_SCORE_SCHEMA = (
    "gkm.object-scene-anchor-exposed-query-score.v1"
)
OBJECT_SCENE_ANCHOR_DURABLE_PREDICATE_FREEZE_SCHEMA = (
    "gkm.object-scene-anchor-durable-predicate-freeze-commitment.v1"
)
OBJECT_SCENE_ANCHOR_RELEASED_RECORD_LOCATOR_SCHEMA = (
    "gkm.object-scene-anchor-released-record-locator.v1"
)
OBJECT_SCENE_ANCHOR_EXPOSED_QUERY_GATE_ID = (
    "bongard.object-scene-anchor-exposed-query-gate/predicate-before-pixels-v1"
)

HISTORICAL_PLAN_FILE_SHA256 = (
    "0447225b24bd440299f7976d29332d9bcd98a5f3d3d10b4fd453eb1ca634dc2c"
)
HISTORICAL_PLAN_RECORD_DIGEST = (
    "sha256:d5643e8efc0fefaddeecd1fe90c2d47dfe25fc49aa401a06a6ba68615560102d"
)
HISTORICAL_RUNTIME_ARCHIVE_FILE_SHA256 = (
    "e84685ccf4d69787f2f093de116824dc0bb53ac08e8abe19cdd04338cfea70ba"
)
HISTORICAL_RUNTIME_ARCHIVE_RECORD_DIGEST = (
    "sha256:4c66c94695182de03c38ef879ac80cecd717a2a1ca07af4dd2e1356851c2883a"
)

# This is an audit assertion, not the selection algorithm.  The builder first
# computes the ordered set difference and then proves that the authenticated
# historical plan currently yields this expected inventory.
EXPECTED_EXPOSED_QUERY_ORDINALS = (
    2,
    6,
    8,
    9,
    10,
    11,
    12,
    13,
    15,
    16,
    22,
    23,
    24,
    25,
    26,
    27,
)
EXPOSED_QUERY_PANEL_COUNT = 16

_RAW_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_QUERY_ALIAS = re.compile(r"query_[0-9]{3}\Z")
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_MAX_JSON_BYTES = 16 * 1024 * 1024


class ObjectSceneAnchorExposedQueryGateError(ValueError):
    """A metadata, custody, release, prediction, or scoring invariant failed."""


def _raw_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _RAW_DIGEST.fullmatch(value) is None:
        raise ObjectSceneAnchorExposedQueryGateError(
            f"{label} must be a raw lowercase SHA-256"
        )
    return value


def _address(value: object, label: str) -> str:
    if not isinstance(value, str) or _ADDRESS.fullmatch(value) is None:
        raise ObjectSceneAnchorExposedQueryGateError(
            f"{label} must be a sha256: address"
        )
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ObjectSceneAnchorExposedQueryGateError(
            f"{label} must be an integer at least {minimum}"
        )
    return value


def _fields(
    value: object, expected: set[str] | frozenset[str], label: str
) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(not isinstance(key, str) for key in value)
        or set(value) != set(expected)
    ):
        raise ObjectSceneAnchorExposedQueryGateError(f"{label} fields differ")
    return value


def _alias(value: object) -> str:
    if not isinstance(value, str) or _QUERY_ALIAS.fullmatch(value) is None:
        raise ObjectSceneAnchorExposedQueryGateError(
            "query alias must be neutral query_NNN"
        )
    return value


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "model_calls_permitted": False,
        "official_test_pixels_consumed": False,
        "historical_pixels_previously_exposed": True,
        "blindness_claimed": False,
        "engineering_ablation_only": True,
    }


def object_scene_anchor_exposed_query_gate_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def _read_exact_json(path: Path, expected_sha256: str) -> dict[str, object]:
    """Read one explicitly addressed metadata or released-record file."""

    _raw_digest(expected_sha256, "expected JSON file digest")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ObjectSceneAnchorExposedQueryGateError(
            f"cannot open addressed JSON {path}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not 0 < before.st_size <= _MAX_JSON_BYTES
        ):
            raise ObjectSceneAnchorExposedQueryGateError(
                "addressed JSON is not a bounded private regular file"
            )
        payload = b""
        while len(payload) <= _MAX_JSON_BYTES:
            chunk = os.read(
                descriptor,
                min(1024 * 1024, _MAX_JSON_BYTES + 1 - len(payload)),
            )
            if not chunk:
                break
            payload += chunk
        after = os.fstat(descriptor)
        if (
            (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            or len(payload) != before.st_size
            or hashlib.sha256(payload).hexdigest() != expected_sha256
        ):
            raise ObjectSceneAnchorExposedQueryGateError(
                "addressed JSON bytes differ from commitment"
            )
    finally:
        os.close(descriptor)
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ObjectSceneAnchorExposedQueryGateError(
            "addressed JSON is malformed"
        ) from exc
    if not isinstance(decoded, dict) or canonical_json(decoded) + b"\n" != payload:
        raise ObjectSceneAnchorExposedQueryGateError(
            "addressed JSON is not canonical JSON plus one newline"
        )
    return decoded


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorHistoricalSceneMetadata:
    ordinal: int
    task_id: str
    panel_id: str
    plan_scene_record_digest: str
    png_byte_count: int
    png_sha256: str
    tag_0_state: str = field(repr=False)
    tag_1_state: str = field(repr=False)

    def __post_init__(self) -> None:
        _integer(self.ordinal, "historical scene ordinal")
        _address(self.plan_scene_record_digest, "plan scene record digest")
        _integer(self.png_byte_count, "historical PNG byte count", minimum=1)
        _raw_digest(self.png_sha256, "historical PNG digest")
        if (
            not isinstance(self.task_id, str)
            or not self.task_id
            or not isinstance(self.panel_id, str)
            or not self.panel_id
            or self.tag_0_state not in ("present", "absent")
            or self.tag_1_state not in ("present", "absent")
            or {self.tag_0_state, self.tag_1_state} != {"present", "absent"}
        ):
            raise ObjectSceneAnchorExposedQueryGateError(
                "historical scene metadata differs"
            )


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorHistoricalMetadata:
    """Runtime-only plan authority; label states never enter the public plan."""

    plan_file_sha256: str
    plan_record_digest: str
    runtime_archive_file_sha256: str
    runtime_archive_record_digest: str
    scenes: tuple[ObjectSceneAnchorHistoricalSceneMetadata, ...]

    def __post_init__(self) -> None:
        _raw_digest(self.plan_file_sha256, "historical plan file digest")
        _address(self.plan_record_digest, "historical plan record digest")
        _raw_digest(
            self.runtime_archive_file_sha256, "runtime archive file digest"
        )
        _address(
            self.runtime_archive_record_digest, "runtime archive record digest"
        )
        if (
            type(self.scenes) is not tuple
            or len(self.scenes) != 28
            or any(
                type(item) is not ObjectSceneAnchorHistoricalSceneMetadata
                for item in self.scenes
            )
            or tuple(item.ordinal for item in self.scenes) != tuple(range(28))
            or len({item.panel_id for item in self.scenes}) != 28
            or len({item.png_sha256 for item in self.scenes}) != 28
        ):
            raise ObjectSceneAnchorExposedQueryGateError(
                "historical metadata must contain exact ordered 28-scene plan"
            )

    @property
    def by_ordinal(self) -> dict[int, ObjectSceneAnchorHistoricalSceneMetadata]:
        return {item.ordinal: item for item in self.scenes}


def build_object_scene_anchor_historical_metadata(
    *,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    runtime_archive: Mapping[str, Any],
    runtime_archive_file_sha256: str,
) -> ObjectSceneAnchorHistoricalMetadata:
    """Verify and join metadata only; no released-panel record is accepted."""

    _raw_digest(plan_file_sha256, "historical plan file digest")
    _raw_digest(runtime_archive_file_sha256, "runtime archive file digest")
    if (
        hashlib.sha256(canonical_json(dict(plan)) + b"\n").hexdigest()
        != plan_file_sha256
        or hashlib.sha256(
            canonical_json(dict(runtime_archive)) + b"\n"
        ).hexdigest()
        != runtime_archive_file_sha256
    ):
        raise ObjectSceneAnchorExposedQueryGateError(
            "historical metadata file bytes differ from supplied identities"
        )
    raw_scenes = plan.get("scenes") if isinstance(plan, Mapping) else None
    runtime_scenes = (
        runtime_archive.get("scenes") if isinstance(runtime_archive, Mapping) else None
    )
    if (
        plan.get("schema") != "gkm.bongard-prototype-scene-calibration-plan.v1"
        or not isinstance(raw_scenes, list)
        or runtime_archive.get("schema")
        != "gkm.bongard-prototype-scene-runtime-archive.v1"
        or not isinstance(runtime_scenes, list)
    ):
        raise ObjectSceneAnchorExposedQueryGateError(
            "historical plan/runtime metadata schema differs"
        )
    plan_record_digest = _address(plan.get("record_digest"), "plan record digest")
    runtime_record_digest = _address(
        runtime_archive.get("record_digest"), "runtime archive record digest"
    )
    if plan_record_digest != "sha256:" + canonical_digest(
        {key: item for key, item in plan.items() if key != "record_digest"}
    ):
        raise ObjectSceneAnchorExposedQueryGateError(
            "historical plan record digest differs"
        )
    if runtime_record_digest != "sha256:" + canonical_digest(
        {key: item for key, item in runtime_archive.items() if key != "record_digest"}
    ):
        raise ObjectSceneAnchorExposedQueryGateError(
            "historical runtime archive digest differs"
        )
    runtime_by_panel = {
        item.get("panel_id"): item
        for item in runtime_scenes
        if isinstance(item, Mapping) and isinstance(item.get("panel_id"), str)
    }
    if len(runtime_by_panel) != 28:
        raise ObjectSceneAnchorExposedQueryGateError(
            "historical runtime archive scene inventory differs"
        )
    scenes: list[ObjectSceneAnchorHistoricalSceneMetadata] = []
    for expected_ordinal, scene in enumerate(raw_scenes):
        if not isinstance(scene, Mapping) or scene.get("ordinal") != expected_ordinal:
            raise ObjectSceneAnchorExposedQueryGateError(
                "historical plan ordinals are not exact and ordered"
            )
        states = scene.get("expected_tag_states")
        runtime = runtime_by_panel.get(scene.get("panel_id"))
        if (
            not isinstance(states, list)
            or len(states) != 2
            or tuple(item.get("tag_id") for item in states if isinstance(item, Mapping))
            != ("opaque_visual_tag_0", "opaque_visual_tag_1")
            or not isinstance(runtime, Mapping)
            or runtime.get("scene_task_id") != scene.get("task_id")
            or runtime.get("observation_context_digest") != plan_record_digest
        ):
            raise ObjectSceneAnchorExposedQueryGateError(
                "historical plan/runtime scene join differs"
            )
        scenes.append(
            ObjectSceneAnchorHistoricalSceneMetadata(
                ordinal=expected_ordinal,
                task_id=scene.get("task_id"),
                panel_id=scene.get("panel_id"),
                plan_scene_record_digest=scene.get("record_digest"),
                png_byte_count=runtime.get("scene_png_byte_count"),
                png_sha256=runtime.get("scene_png_sha256"),
                tag_0_state=states[0].get("state"),
                tag_1_state=states[1].get("state"),
            )
        )
    return ObjectSceneAnchorHistoricalMetadata(
        plan_file_sha256=plan_file_sha256,
        plan_record_digest=plan_record_digest,
        runtime_archive_file_sha256=runtime_archive_file_sha256,
        runtime_archive_record_digest=runtime_record_digest,
        scenes=tuple(scenes),
    )


def load_object_scene_anchor_historical_metadata(
    directory: str | os.PathLike[str] = (
        DEFAULT_OBJECT_BONGARD_PANEL_RUBRIC_CALIBRATION_SOURCE
    ),
) -> ObjectSceneAnchorHistoricalMetadata:
    """Load only the small plan/runtime metadata files, never panel records."""

    root = Path(directory)
    plan = _read_exact_json(
        root / "calibration_plan" / f"{HISTORICAL_PLAN_FILE_SHA256}.json",
        HISTORICAL_PLAN_FILE_SHA256,
    )
    runtime = _read_exact_json(
        root
        / "runtime_archive"
        / f"{HISTORICAL_RUNTIME_ARCHIVE_FILE_SHA256}.json",
        HISTORICAL_RUNTIME_ARCHIVE_FILE_SHA256,
    )
    metadata = build_object_scene_anchor_historical_metadata(
        plan=plan,
        plan_file_sha256=HISTORICAL_PLAN_FILE_SHA256,
        runtime_archive=runtime,
        runtime_archive_file_sha256=HISTORICAL_RUNTIME_ARCHIVE_FILE_SHA256,
    )
    if (
        metadata.plan_record_digest != HISTORICAL_PLAN_RECORD_DIGEST
        or metadata.runtime_archive_record_digest
        != HISTORICAL_RUNTIME_ARCHIVE_RECORD_DIGEST
    ):
        raise ObjectSceneAnchorExposedQueryGateError(
            "pinned historical metadata identity differs"
        )
    return metadata


def _plan_item_content(
    value: "ObjectSceneAnchorExposedQueryPlanItem",
) -> dict[str, object]:
    return {
        "query_alias": value.query_alias,
        "ordinal": value.ordinal,
        "task_id": value.task_id,
        "panel_id": value.panel_id,
        "plan_scene_record_digest": value.plan_scene_record_digest,
        "png_byte_count": value.png_byte_count,
        "png_sha256": value.png_sha256,
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorExposedQueryPlanItem:
    query_alias: str
    ordinal: int
    task_id: str
    panel_id: str
    plan_scene_record_digest: str
    png_byte_count: int
    png_sha256: str
    item_digest: str

    def __post_init__(self) -> None:
        _alias(self.query_alias)
        _integer(self.ordinal, "query ordinal")
        _address(self.plan_scene_record_digest, "plan scene record digest")
        _integer(self.png_byte_count, "query PNG byte count", minimum=1)
        _raw_digest(self.png_sha256, "query PNG digest")
        _raw_digest(self.item_digest, "query plan item digest")
        if (
            not isinstance(self.task_id, str)
            or not self.task_id
            or not isinstance(self.panel_id, str)
            or not self.panel_id
            or self.item_digest != canonical_digest(_plan_item_content(self))
        ):
            raise ObjectSceneAnchorExposedQueryGateError(
                "query plan item identity differs"
            )

    def to_data(self) -> dict[str, object]:
        return {**_plan_item_content(self), "item_digest": self.item_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorExposedQueryPlanItem":
        raw = _fields(
            value,
            {
                "query_alias",
                "ordinal",
                "task_id",
                "panel_id",
                "plan_scene_record_digest",
                "png_byte_count",
                "png_sha256",
                "item_digest",
            },
            "query plan item",
        )
        result = cls(
            raw["query_alias"],
            raw["ordinal"],
            raw["task_id"],
            raw["panel_id"],
            raw["plan_scene_record_digest"],
            raw["png_byte_count"],
            raw["png_sha256"],
            raw["item_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorExposedQueryGateError(
                "query plan item is not canonical"
            )
        return result


def _plan_content(value: "ObjectSceneAnchorExposedQueryPlan") -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ANCHOR_EXPOSED_QUERY_PLAN_SCHEMA,
        "gate_id": OBJECT_SCENE_ANCHOR_EXPOSED_QUERY_GATE_ID,
        "gate_source_digest": object_scene_anchor_exposed_query_gate_source_digest(),
        "historical_plan_file_sha256": value.historical_plan_file_sha256,
        "historical_plan_record_digest": value.historical_plan_record_digest,
        "historical_runtime_archive_file_sha256": (
            value.historical_runtime_archive_file_sha256
        ),
        "historical_runtime_archive_record_digest": (
            value.historical_runtime_archive_record_digest
        ),
        "all_plan_ordinals": list(value.all_plan_ordinals),
        "support_ordinals": list(value.support_ordinals),
        "derived_query_ordinals": list(value.derived_query_ordinals),
        "derivation": "ordered-all-plan-ordinals-minus-exact-support-ordinal-set",
        "expected_query_ordinal_assertion": list(EXPECTED_EXPOSED_QUERY_ORDINALS),
        "items": [item.to_data() for item in value.items],
        "query_count": value.query_count,
        "labels_in_plan": False,
        "released_record_locators_in_plan": False,
        "pixels_in_plan": False,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorExposedQueryPlan:
    historical_plan_file_sha256: str
    historical_plan_record_digest: str
    historical_runtime_archive_file_sha256: str
    historical_runtime_archive_record_digest: str
    all_plan_ordinals: tuple[int, ...]
    support_ordinals: tuple[int, ...]
    derived_query_ordinals: tuple[int, ...]
    items: tuple[ObjectSceneAnchorExposedQueryPlanItem, ...]
    query_count: int
    plan_digest: str

    def __post_init__(self) -> None:
        _raw_digest(self.historical_plan_file_sha256, "plan file digest")
        _address(self.historical_plan_record_digest, "plan record digest")
        _raw_digest(
            self.historical_runtime_archive_file_sha256,
            "runtime archive file digest",
        )
        _address(
            self.historical_runtime_archive_record_digest,
            "runtime archive record digest",
        )
        _integer(self.query_count, "query count")
        if (
            type(self.all_plan_ordinals) is not tuple
            or type(self.support_ordinals) is not tuple
            or type(self.derived_query_ordinals) is not tuple
            or type(self.items) is not tuple
            or self.all_plan_ordinals != tuple(range(28))
            or self.support_ordinals
            != tuple(sorted(PANEL_RUBRIC_CALIBRATION_SELECTED_ORDINALS))
        ):
            raise ObjectSceneAnchorExposedQueryGateError(
                "query plan parent ordinal inventory differs"
            )
        derived = tuple(
            item
            for item in self.all_plan_ordinals
            if item not in frozenset(self.support_ordinals)
        )
        aliases = tuple(f"query_{index:03d}" for index in range(len(derived)))
        if (
            derived != self.derived_query_ordinals
            or derived != EXPECTED_EXPOSED_QUERY_ORDINALS
            or self.query_count != EXPOSED_QUERY_PANEL_COUNT
            or len(self.items) != self.query_count
            or tuple(item.ordinal for item in self.items) != derived
            or tuple(item.query_alias for item in self.items) != aliases
            or len({item.panel_id for item in self.items}) != self.query_count
            or any(
                type(item) is not ObjectSceneAnchorExposedQueryPlanItem
                for item in self.items
            )
        ):
            raise ObjectSceneAnchorExposedQueryGateError(
                "query plan is not the exact authenticated set difference"
            )
        _raw_digest(self.plan_digest, "query plan digest")
        if self.plan_digest != canonical_digest(_plan_content(self)):
            raise ObjectSceneAnchorExposedQueryGateError(
                "query plan digest differs"
            )

    @property
    def by_alias(self) -> dict[str, ObjectSceneAnchorExposedQueryPlanItem]:
        return {item.query_alias: item for item in self.items}

    def to_data(self) -> dict[str, object]:
        return {**_plan_content(self), "plan_digest": self.plan_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorExposedQueryPlan":
        raw = _fields(
            value,
            {
                "schema",
                "gate_id",
                "gate_source_digest",
                "historical_plan_file_sha256",
                "historical_plan_record_digest",
                "historical_runtime_archive_file_sha256",
                "historical_runtime_archive_record_digest",
                "all_plan_ordinals",
                "support_ordinals",
                "derived_query_ordinals",
                "derivation",
                "expected_query_ordinal_assertion",
                "items",
                "query_count",
                "labels_in_plan",
                "released_record_locators_in_plan",
                "pixels_in_plan",
                *_authority_data(),
                "plan_digest",
            },
            "exposed query plan",
        )
        if (
            raw["schema"] != OBJECT_SCENE_ANCHOR_EXPOSED_QUERY_PLAN_SCHEMA
            or raw["gate_id"] != OBJECT_SCENE_ANCHOR_EXPOSED_QUERY_GATE_ID
            or raw["gate_source_digest"]
            != object_scene_anchor_exposed_query_gate_source_digest()
            or raw["derivation"]
            != "ordered-all-plan-ordinals-minus-exact-support-ordinal-set"
            or raw["expected_query_ordinal_assertion"]
            != list(EXPECTED_EXPOSED_QUERY_ORDINALS)
            or raw["labels_in_plan"] is not False
            or raw["released_record_locators_in_plan"] is not False
            or raw["pixels_in_plan"] is not False
            or any(raw[key] != item for key, item in _authority_data().items())
            or any(
                not isinstance(raw[key], list)
                for key in (
                    "all_plan_ordinals",
                    "support_ordinals",
                    "derived_query_ordinals",
                    "items",
                )
            )
        ):
            raise ObjectSceneAnchorExposedQueryGateError(
                "exposed query plan policy differs"
            )
        result = cls(
            raw["historical_plan_file_sha256"],
            raw["historical_plan_record_digest"],
            raw["historical_runtime_archive_file_sha256"],
            raw["historical_runtime_archive_record_digest"],
            tuple(raw["all_plan_ordinals"]),
            tuple(raw["support_ordinals"]),
            tuple(raw["derived_query_ordinals"]),
            tuple(
                ObjectSceneAnchorExposedQueryPlanItem.from_data(item)
                for item in raw["items"]
            ),
            raw["query_count"],
            raw["plan_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorExposedQueryGateError(
                "exposed query plan is not canonical"
            )
        return result


def build_object_scene_anchor_exposed_query_plan(
    metadata: ObjectSceneAnchorHistoricalMetadata,
) -> ObjectSceneAnchorExposedQueryPlan:
    if type(metadata) is not ObjectSceneAnchorHistoricalMetadata:
        raise TypeError("metadata must be exact historical metadata")
    all_ordinals = tuple(item.ordinal for item in metadata.scenes)
    support = tuple(sorted(PANEL_RUBRIC_CALIBRATION_SELECTED_ORDINALS))
    support_set = frozenset(support)
    derived = tuple(item for item in all_ordinals if item not in support_set)
    # The actual selector is the set difference above.  The named constant is
    # checked only as a pinned audit expectation.
    if derived != EXPECTED_EXPOSED_QUERY_ORDINALS:
        raise ObjectSceneAnchorExposedQueryGateError(
            "authenticated set difference differs from expected remaining sixteen"
        )
    by_ordinal = metadata.by_ordinal
    items: list[ObjectSceneAnchorExposedQueryPlanItem] = []
    for index, ordinal in enumerate(derived):
        scene = by_ordinal[ordinal]
        values = {
            "query_alias": f"query_{index:03d}",
            "ordinal": ordinal,
            "task_id": scene.task_id,
            "panel_id": scene.panel_id,
            "plan_scene_record_digest": scene.plan_scene_record_digest,
            "png_byte_count": scene.png_byte_count,
            "png_sha256": scene.png_sha256,
        }
        provisional = object.__new__(ObjectSceneAnchorExposedQueryPlanItem)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        items.append(
            ObjectSceneAnchorExposedQueryPlanItem(
                **values,
                item_digest=canonical_digest(_plan_item_content(provisional)),
            )
        )
    values = {
        "historical_plan_file_sha256": metadata.plan_file_sha256,
        "historical_plan_record_digest": metadata.plan_record_digest,
        "historical_runtime_archive_file_sha256": (
            metadata.runtime_archive_file_sha256
        ),
        "historical_runtime_archive_record_digest": (
            metadata.runtime_archive_record_digest
        ),
        "all_plan_ordinals": all_ordinals,
        "support_ordinals": support,
        "derived_query_ordinals": derived,
        "items": tuple(items),
        "query_count": len(items),
    }
    provisional = object.__new__(ObjectSceneAnchorExposedQueryPlan)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectSceneAnchorExposedQueryPlan(
        **values,
        plan_digest=canonical_digest(_plan_content(provisional)),
    )


def verify_object_scene_anchor_exposed_query_plan(
    plan: ObjectSceneAnchorExposedQueryPlan,
    metadata: ObjectSceneAnchorHistoricalMetadata,
) -> ObjectSceneAnchorExposedQueryPlan:
    if type(plan) is not ObjectSceneAnchorExposedQueryPlan:
        raise TypeError("plan must be exact exposed query plan")
    restored = ObjectSceneAnchorExposedQueryPlan.from_data(plan.to_data())
    replayed = build_object_scene_anchor_exposed_query_plan(metadata)
    if restored != replayed:
        raise ObjectSceneAnchorExposedQueryGateError(
            "exposed query plan differs from metadata-only replay"
        )
    return restored


def _durable_content(
    value: "ObjectSceneAnchorPredicateDurableFreezeCommitment",
) -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ANCHOR_DURABLE_PREDICATE_FREEZE_SCHEMA,
        "gate_id": OBJECT_SCENE_ANCHOR_EXPOSED_QUERY_GATE_ID,
        "predicate_digest": value.predicate_digest,
        "predicate_payload_sha256": value.predicate_payload_sha256,
        "predicate_payload_byte_count": value.predicate_payload_byte_count,
        "persistence_receipt_digest": value.persistence_receipt_digest,
        "persisted_and_reloaded_before_query_release": True,
        "caller_supplied_commitment": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorPredicateDurableFreezeCommitment:
    """Caller-owned evidence that exact predicate bytes were durably reloaded."""

    predicate_digest: str
    predicate_payload_sha256: str
    predicate_payload_byte_count: int
    persistence_receipt_digest: str
    commitment_digest: str

    def __post_init__(self) -> None:
        _raw_digest(self.predicate_digest, "predicate digest")
        _raw_digest(self.predicate_payload_sha256, "predicate payload digest")
        _integer(
            self.predicate_payload_byte_count,
            "predicate payload byte count",
            minimum=1,
        )
        _raw_digest(self.persistence_receipt_digest, "persistence receipt digest")
        _raw_digest(self.commitment_digest, "durable freeze commitment digest")
        if self.commitment_digest != canonical_digest(_durable_content(self)):
            raise ObjectSceneAnchorExposedQueryGateError(
                "durable predicate freeze commitment differs"
            )

    def to_data(self) -> dict[str, object]:
        return {**_durable_content(self), "commitment_digest": self.commitment_digest}

    @classmethod
    def from_data(
        cls, value: object
    ) -> "ObjectSceneAnchorPredicateDurableFreezeCommitment":
        raw = _fields(
            value,
            {
                "schema",
                "gate_id",
                "predicate_digest",
                "predicate_payload_sha256",
                "predicate_payload_byte_count",
                "persistence_receipt_digest",
                "persisted_and_reloaded_before_query_release",
                "caller_supplied_commitment",
                *_authority_data(),
                "commitment_digest",
            },
            "durable predicate freeze commitment",
        )
        if (
            raw["schema"] != OBJECT_SCENE_ANCHOR_DURABLE_PREDICATE_FREEZE_SCHEMA
            or raw["gate_id"] != OBJECT_SCENE_ANCHOR_EXPOSED_QUERY_GATE_ID
            or raw["persisted_and_reloaded_before_query_release"] is not True
            or raw["caller_supplied_commitment"] is not True
            or any(raw[key] != item for key, item in _authority_data().items())
        ):
            raise ObjectSceneAnchorExposedQueryGateError(
                "durable predicate freeze policy differs"
            )
        result = cls(
            raw["predicate_digest"],
            raw["predicate_payload_sha256"],
            raw["predicate_payload_byte_count"],
            raw["persistence_receipt_digest"],
            raw["commitment_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorExposedQueryGateError(
                "durable predicate freeze commitment is not canonical"
            )
        return result


def bind_caller_durable_object_scene_anchor_python_predicate(
    predicate: ObjectSceneAnchorPythonPredicate,
    *,
    persisted_predicate_payload_sha256: str,
    persisted_predicate_payload_byte_count: int,
    persistence_receipt_digest: str,
) -> ObjectSceneAnchorPredicateDurableFreezeCommitment:
    """Bind caller-supplied persistence evidence to exact canonical bytes.

    This function does not perform persistence and makes no such claim.  The
    caller must supply the digest/count obtained from its durable write+reload
    protocol and its independent persistence receipt.
    """

    if type(predicate) is not ObjectSceneAnchorPythonPredicate:
        raise TypeError("predicate must be exact ObjectSceneAnchorPythonPredicate")
    frozen = ObjectSceneAnchorPythonPredicate.from_data(predicate.to_data())
    payload = canonical_json(frozen.to_data())
    if (
        _raw_digest(
            persisted_predicate_payload_sha256,
            "persisted predicate payload digest",
        )
        != hashlib.sha256(payload).hexdigest()
        or _integer(
            persisted_predicate_payload_byte_count,
            "persisted predicate payload byte count",
            minimum=1,
        )
        != len(payload)
    ):
        raise ObjectSceneAnchorExposedQueryGateError(
            "caller durable payload differs from exact Python predicate bytes"
        )
    values = {
        "predicate_digest": frozen.predicate_digest,
        "predicate_payload_sha256": persisted_predicate_payload_sha256,
        "predicate_payload_byte_count": persisted_predicate_payload_byte_count,
        "persistence_receipt_digest": _raw_digest(
            persistence_receipt_digest, "persistence receipt digest"
        ),
    }
    provisional = object.__new__(
        ObjectSceneAnchorPredicateDurableFreezeCommitment
    )
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectSceneAnchorPredicateDurableFreezeCommitment(
        **values,
        commitment_digest=canonical_digest(_durable_content(provisional)),
    )


def _locator_content(
    value: "ObjectSceneAnchorQueryReleasedRecordLocator",
) -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ANCHOR_RELEASED_RECORD_LOCATOR_SCHEMA,
        "query_alias": value.query_alias,
        "released_record_file_sha256": value.released_record_file_sha256,
        "released_record_digest": value.released_record_digest,
        "metadata_only": True,
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorQueryReleasedRecordLocator:
    query_alias: str
    released_record_file_sha256: str
    released_record_digest: str
    locator_digest: str

    def __post_init__(self) -> None:
        _alias(self.query_alias)
        _raw_digest(
            self.released_record_file_sha256, "released record file digest"
        )
        _address(self.released_record_digest, "released record digest")
        _raw_digest(self.locator_digest, "released record locator digest")
        if self.locator_digest != canonical_digest(_locator_content(self)):
            raise ObjectSceneAnchorExposedQueryGateError(
                "released record locator digest differs"
            )

    @classmethod
    def create(
        cls,
        query_alias: str,
        *,
        released_record_file_sha256: str,
        released_record_digest: str,
    ) -> "ObjectSceneAnchorQueryReleasedRecordLocator":
        values = {
            "query_alias": query_alias,
            "released_record_file_sha256": released_record_file_sha256,
            "released_record_digest": released_record_digest,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            locator_digest=canonical_digest(_locator_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {**_locator_content(self), "locator_digest": self.locator_digest}

    @classmethod
    def from_data(
        cls, value: object
    ) -> "ObjectSceneAnchorQueryReleasedRecordLocator":
        raw = _fields(
            value,
            {
                "schema",
                "query_alias",
                "released_record_file_sha256",
                "released_record_digest",
                "metadata_only",
                "locator_digest",
            },
            "released record locator",
        )
        if (
            raw["schema"] != OBJECT_SCENE_ANCHOR_RELEASED_RECORD_LOCATOR_SCHEMA
            or raw["metadata_only"] is not True
        ):
            raise ObjectSceneAnchorExposedQueryGateError(
                "released record locator policy differs"
            )
        result = cls(
            raw["query_alias"],
            raw["released_record_file_sha256"],
            raw["released_record_digest"],
            raw["locator_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorExposedQueryGateError(
                "released record locator is not canonical"
            )
        return result


def _release_item_content(
    value: "ObjectSceneAnchorExposedQueryReleaseItem",
) -> dict[str, object]:
    return {
        "query_alias": value.query_alias,
        "plan_item_digest": value.plan_item_digest,
        "locator_digest": value.locator_digest,
        "released_record_file_sha256": value.released_record_file_sha256,
        "released_record_digest": value.released_record_digest,
        "released_execution_precommit_digest": (
            value.released_execution_precommit_digest
        ),
        "released_exposure_successor_digest": (
            value.released_exposure_successor_digest
        ),
        "png_byte_count": value.png_byte_count,
        "png_sha256": value.png_sha256,
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorExposedQueryReleaseItem:
    query_alias: str
    plan_item_digest: str
    locator_digest: str
    released_record_file_sha256: str
    released_record_digest: str
    released_execution_precommit_digest: str
    released_exposure_successor_digest: str
    png_byte_count: int
    png_sha256: str
    item_digest: str

    def __post_init__(self) -> None:
        _alias(self.query_alias)
        _raw_digest(self.plan_item_digest, "query plan item digest")
        _raw_digest(self.locator_digest, "released record locator digest")
        _raw_digest(
            self.released_record_file_sha256, "released record file digest"
        )
        _address(self.released_record_digest, "released record digest")
        _address(
            self.released_execution_precommit_digest,
            "released execution precommit digest",
        )
        _address(
            self.released_exposure_successor_digest,
            "released exposure successor digest",
        )
        _integer(self.png_byte_count, "released PNG byte count", minimum=1)
        _raw_digest(self.png_sha256, "released PNG digest")
        _raw_digest(self.item_digest, "released query item digest")
        if self.item_digest != canonical_digest(_release_item_content(self)):
            raise ObjectSceneAnchorExposedQueryGateError(
                "released query item digest differs"
            )

    def to_data(self) -> dict[str, object]:
        return {**_release_item_content(self), "item_digest": self.item_digest}

    @classmethod
    def from_data(
        cls, value: object
    ) -> "ObjectSceneAnchorExposedQueryReleaseItem":
        raw = _fields(
            value,
            {
                "query_alias",
                "plan_item_digest",
                "locator_digest",
                "released_record_file_sha256",
                "released_record_digest",
                "released_execution_precommit_digest",
                "released_exposure_successor_digest",
                "png_byte_count",
                "png_sha256",
                "item_digest",
            },
            "released query item",
        )
        result = cls(
            raw["query_alias"],
            raw["plan_item_digest"],
            raw["locator_digest"],
            raw["released_record_file_sha256"],
            raw["released_record_digest"],
            raw["released_execution_precommit_digest"],
            raw["released_exposure_successor_digest"],
            raw["png_byte_count"],
            raw["png_sha256"],
            raw["item_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorExposedQueryGateError(
                "released query item is not canonical"
            )
        return result


def _release_content(
    value: "ObjectSceneAnchorExposedQueryRelease",
) -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ANCHOR_EXPOSED_QUERY_RELEASE_SCHEMA,
        "gate_id": OBJECT_SCENE_ANCHOR_EXPOSED_QUERY_GATE_ID,
        "gate_source_digest": object_scene_anchor_exposed_query_gate_source_digest(),
        "query_plan_digest": value.query_plan_digest,
        "predicate_digest": value.predicate_digest,
        "durable_freeze_commitment_digest": (
            value.durable_freeze_commitment_digest
        ),
        "predicate_verified_before_record_loader_calls": True,
        "durable_freeze_verified_before_record_loader_calls": True,
        "exact_loader_call_count": value.exact_loader_call_count,
        "items": [item.to_data() for item in value.items],
        "query_aliases": list(value.query_aliases),
        "labels_revealed": False,
        "neutral_geometry_input_policy": "query-alias-and-exact-PNG-only",
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorExposedQueryRelease:
    query_plan_digest: str
    predicate_digest: str
    durable_freeze_commitment_digest: str
    exact_loader_call_count: int
    items: tuple[ObjectSceneAnchorExposedQueryReleaseItem, ...]
    query_aliases: tuple[str, ...]
    release_digest: str

    def __post_init__(self) -> None:
        for label, item in (
            ("query plan digest", self.query_plan_digest),
            ("predicate digest", self.predicate_digest),
            ("durable freeze commitment digest", self.durable_freeze_commitment_digest),
            ("release digest", self.release_digest),
        ):
            _raw_digest(item, label)
        _integer(self.exact_loader_call_count, "record loader call count")
        expected_aliases = tuple(
            f"query_{index:03d}" for index in range(EXPOSED_QUERY_PANEL_COUNT)
        )
        if (
            type(self.items) is not tuple
            or type(self.query_aliases) is not tuple
            or self.query_aliases != expected_aliases
            or tuple(item.query_alias for item in self.items) != expected_aliases
            or any(
                type(item) is not ObjectSceneAnchorExposedQueryReleaseItem
                for item in self.items
            )
            or self.exact_loader_call_count != EXPOSED_QUERY_PANEL_COUNT
            or len(self.items) != EXPOSED_QUERY_PANEL_COUNT
            or len({item.released_record_file_sha256 for item in self.items})
            != EXPOSED_QUERY_PANEL_COUNT
            or len({item.released_record_digest for item in self.items})
            != EXPOSED_QUERY_PANEL_COUNT
        ):
            raise ObjectSceneAnchorExposedQueryGateError(
                "released query inventory differs"
            )
        if self.release_digest != canonical_digest(_release_content(self)):
            raise ObjectSceneAnchorExposedQueryGateError(
                "released query digest differs"
            )

    def to_data(self) -> dict[str, object]:
        return {**_release_content(self), "release_digest": self.release_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorExposedQueryRelease":
        raw = _fields(
            value,
            {
                "schema",
                "gate_id",
                "gate_source_digest",
                "query_plan_digest",
                "predicate_digest",
                "durable_freeze_commitment_digest",
                "predicate_verified_before_record_loader_calls",
                "durable_freeze_verified_before_record_loader_calls",
                "exact_loader_call_count",
                "items",
                "query_aliases",
                "labels_revealed",
                "neutral_geometry_input_policy",
                *_authority_data(),
                "release_digest",
            },
            "exposed query release",
        )
        if (
            raw["schema"] != OBJECT_SCENE_ANCHOR_EXPOSED_QUERY_RELEASE_SCHEMA
            or raw["gate_id"] != OBJECT_SCENE_ANCHOR_EXPOSED_QUERY_GATE_ID
            or raw["gate_source_digest"]
            != object_scene_anchor_exposed_query_gate_source_digest()
            or raw["predicate_verified_before_record_loader_calls"] is not True
            or raw["durable_freeze_verified_before_record_loader_calls"] is not True
            or raw["labels_revealed"] is not False
            or raw["neutral_geometry_input_policy"]
            != "query-alias-and-exact-PNG-only"
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["items"], list)
            or not isinstance(raw["query_aliases"], list)
        ):
            raise ObjectSceneAnchorExposedQueryGateError(
                "exposed query release policy differs"
            )
        result = cls(
            raw["query_plan_digest"],
            raw["predicate_digest"],
            raw["durable_freeze_commitment_digest"],
            raw["exact_loader_call_count"],
            tuple(
                ObjectSceneAnchorExposedQueryReleaseItem.from_data(item)
                for item in raw["items"]
            ),
            tuple(raw["query_aliases"]),
            raw["release_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorExposedQueryGateError(
                "exposed query release is not canonical"
            )
        return result


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorNeutralQueryPanelInput:
    """The only object passed onward to geometry/observation preparation."""

    query_alias: str
    exact_png_bytes: bytes = field(repr=False)
    png_sha256: str
    release_item_digest: str

    def __post_init__(self) -> None:
        _alias(self.query_alias)
        _raw_digest(self.png_sha256, "neutral query PNG digest")
        _raw_digest(self.release_item_digest, "released query item digest")
        if (
            not isinstance(self.exact_png_bytes, bytes)
            or not self.exact_png_bytes.startswith(_PNG_SIGNATURE)
            or not 0 < len(self.exact_png_bytes) <= MAX_PANEL_PNG_BYTES
            or hashlib.sha256(self.exact_png_bytes).hexdigest() != self.png_sha256
        ):
            raise ObjectSceneAnchorExposedQueryGateError(
                "neutral query input bytes differ"
            )


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorExposedQueryRuntimeBundle:
    release: ObjectSceneAnchorExposedQueryRelease
    released_panels: tuple[ReleasedOfficialPanel, ...] = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.release) is not ObjectSceneAnchorExposedQueryRelease:
            raise TypeError("release must be exact exposed query release")
        if (
            type(self.released_panels) is not tuple
            or len(self.released_panels) != EXPOSED_QUERY_PANEL_COUNT
            or any(
                type(item) is not ReleasedOfficialPanel
                for item in self.released_panels
            )
        ):
            raise ObjectSceneAnchorExposedQueryGateError(
                "released runtime panel inventory differs"
            )
        for receipt, panel in zip(
            self.release.items, self.released_panels, strict=True
        ):
            if (
                panel.record_digest != receipt.released_record_digest
                or panel.execution_precommit_digest
                != receipt.released_execution_precommit_digest
                or panel.exposure_successor_digest
                != receipt.released_exposure_successor_digest
                or len(panel.exact_png_bytes) != receipt.png_byte_count
                or panel.exact_png_digest != "sha256:" + receipt.png_sha256
            ):
                raise ObjectSceneAnchorExposedQueryGateError(
                    "released runtime panel differs from freeze"
                )

    @property
    def neutral_panel_inputs(
        self,
    ) -> tuple[ObjectSceneAnchorNeutralQueryPanelInput, ...]:
        return tuple(
            ObjectSceneAnchorNeutralQueryPanelInput(
                query_alias=item.query_alias,
                exact_png_bytes=panel.exact_png_bytes,
                png_sha256=item.png_sha256,
                release_item_digest=item.item_digest,
            )
            for item, panel in zip(
                self.release.items, self.released_panels, strict=True
            )
        )


ReleasedRecordLoader = Callable[
    [ObjectSceneAnchorQueryReleasedRecordLocator], Mapping[str, Any]
]


def _verified_release_gate(
    plan: ObjectSceneAnchorExposedQueryPlan,
    predicate: ObjectSceneAnchorPythonPredicate,
    durable_freeze: ObjectSceneAnchorPredicateDurableFreezeCommitment,
    expected_durable_freeze_commitment_digest: str,
) -> tuple[
    ObjectSceneAnchorExposedQueryPlan,
    ObjectSceneAnchorPythonPredicate,
    ObjectSceneAnchorPredicateDurableFreezeCommitment,
]:
    frozen_plan = ObjectSceneAnchorExposedQueryPlan.from_data(plan.to_data())
    frozen_predicate = ObjectSceneAnchorPythonPredicate.from_data(predicate.to_data())
    frozen_durable = ObjectSceneAnchorPredicateDurableFreezeCommitment.from_data(
        durable_freeze.to_data()
    )
    payload = canonical_json(frozen_predicate.to_data())
    if (
        frozen_durable.commitment_digest
        != _raw_digest(
            expected_durable_freeze_commitment_digest,
            "expected durable freeze commitment digest",
        )
        or frozen_durable.predicate_digest != frozen_predicate.predicate_digest
        or frozen_durable.predicate_payload_sha256
        != hashlib.sha256(payload).hexdigest()
        or frozen_durable.predicate_payload_byte_count != len(payload)
    ):
        raise ObjectSceneAnchorExposedQueryGateError(
            "predicate or durable freeze gate differs"
        )
    return frozen_plan, frozen_predicate, frozen_durable


def release_object_scene_anchor_exposed_queries(
    *,
    plan: ObjectSceneAnchorExposedQueryPlan,
    predicate: ObjectSceneAnchorPythonPredicate,
    durable_freeze: ObjectSceneAnchorPredicateDurableFreezeCommitment,
    expected_durable_freeze_commitment_digest: str,
    locators: tuple[ObjectSceneAnchorQueryReleasedRecordLocator, ...],
    load_released_record: ReleasedRecordLoader,
) -> ObjectSceneAnchorExposedQueryRuntimeBundle:
    """Verify predicate custody, then call the exact released-record loader."""

    if type(plan) is not ObjectSceneAnchorExposedQueryPlan:
        raise TypeError("plan must be exact exposed query plan")
    if type(predicate) is not ObjectSceneAnchorPythonPredicate:
        raise TypeError("predicate must be exact ObjectSceneAnchorPythonPredicate")
    if type(durable_freeze) is not ObjectSceneAnchorPredicateDurableFreezeCommitment:
        raise TypeError("durable_freeze must be exact durable commitment")
    if type(locators) is not tuple or any(
        type(item) is not ObjectSceneAnchorQueryReleasedRecordLocator
        for item in locators
    ):
        raise TypeError("locators must be an exact typed tuple")
    if not callable(load_released_record):
        raise TypeError("load_released_record must be callable")

    # This entire gate executes before the first loader invocation.
    frozen_plan, frozen_predicate, frozen_durable = _verified_release_gate(
        plan,
        predicate,
        durable_freeze,
        expected_durable_freeze_commitment_digest,
    )
    expected_aliases = tuple(item.query_alias for item in frozen_plan.items)
    if (
        len(locators) != EXPOSED_QUERY_PANEL_COUNT
        or tuple(item.query_alias for item in locators) != expected_aliases
        or len({item.released_record_file_sha256 for item in locators})
        != EXPOSED_QUERY_PANEL_COUNT
        or len({item.released_record_digest for item in locators})
        != EXPOSED_QUERY_PANEL_COUNT
    ):
        raise ObjectSceneAnchorExposedQueryGateError(
            "released record locator inventory differs before pixel release"
        )

    released_panels: list[ReleasedOfficialPanel] = []
    release_items: list[ObjectSceneAnchorExposedQueryReleaseItem] = []
    for plan_item, locator in zip(frozen_plan.items, locators, strict=True):
        raw = load_released_record(locator)
        if not isinstance(raw, Mapping):
            raise ObjectSceneAnchorExposedQueryGateError(
                "released record loader returned non-mapping"
            )
        if hashlib.sha256(canonical_json(dict(raw)) + b"\n").hexdigest() != (
            locator.released_record_file_sha256
        ):
            raise ObjectSceneAnchorExposedQueryGateError(
                "released record loader bytes differ from exact locator"
            )
        released = ReleasedOfficialPanel.from_data(raw)
        if (
            released.panel_id != plan_item.panel_id
            or released.record_digest != locator.released_record_digest
            or len(released.exact_png_bytes) != plan_item.png_byte_count
            or released.exact_png_digest != "sha256:" + plan_item.png_sha256
        ):
            raise ObjectSceneAnchorExposedQueryGateError(
                "released record differs from exact neutral query plan item"
            )
        values = {
            "query_alias": plan_item.query_alias,
            "plan_item_digest": plan_item.item_digest,
            "locator_digest": locator.locator_digest,
            "released_record_file_sha256": locator.released_record_file_sha256,
            "released_record_digest": released.record_digest,
            "released_execution_precommit_digest": (
                released.execution_precommit_digest
            ),
            "released_exposure_successor_digest": (
                released.exposure_successor_digest
            ),
            "png_byte_count": len(released.exact_png_bytes),
            "png_sha256": released.exact_png_digest.removeprefix("sha256:"),
        }
        provisional = object.__new__(ObjectSceneAnchorExposedQueryReleaseItem)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        release_items.append(
            ObjectSceneAnchorExposedQueryReleaseItem(
                **values,
                item_digest=canonical_digest(_release_item_content(provisional)),
            )
        )
        released_panels.append(released)
    values = {
        "query_plan_digest": frozen_plan.plan_digest,
        "predicate_digest": frozen_predicate.predicate_digest,
        "durable_freeze_commitment_digest": frozen_durable.commitment_digest,
        "exact_loader_call_count": len(released_panels),
        "items": tuple(release_items),
        "query_aliases": expected_aliases,
    }
    provisional = object.__new__(ObjectSceneAnchorExposedQueryRelease)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    release = ObjectSceneAnchorExposedQueryRelease(
        **values,
        release_digest=canonical_digest(_release_content(provisional)),
    )
    return ObjectSceneAnchorExposedQueryRuntimeBundle(
        release=release,
        released_panels=tuple(released_panels),
    )


def released_record_directory_loader(
    directory: str | os.PathLike[str],
) -> ReleasedRecordLoader:
    """Create a loader that opens only each explicitly supplied locator path."""

    root = Path(directory)

    def load(
        locator: ObjectSceneAnchorQueryReleasedRecordLocator,
    ) -> Mapping[str, Any]:
        if type(locator) is not ObjectSceneAnchorQueryReleasedRecordLocator:
            raise TypeError("locator must be exact released record locator")
        return _read_exact_json(
            root / f"{locator.released_record_file_sha256}.json",
            locator.released_record_file_sha256,
        )

    return load


def verify_object_scene_anchor_exposed_query_release(
    bundle: ObjectSceneAnchorExposedQueryRuntimeBundle,
    *,
    plan: ObjectSceneAnchorExposedQueryPlan,
    predicate: ObjectSceneAnchorPythonPredicate,
    durable_freeze: ObjectSceneAnchorPredicateDurableFreezeCommitment,
    expected_durable_freeze_commitment_digest: str,
    locators: tuple[ObjectSceneAnchorQueryReleasedRecordLocator, ...],
    load_released_record: ReleasedRecordLoader,
) -> ObjectSceneAnchorExposedQueryRuntimeBundle:
    """Cold-replay the exact predicate gate and all sixteen record loads."""

    if type(bundle) is not ObjectSceneAnchorExposedQueryRuntimeBundle:
        raise TypeError("bundle must be exact exposed query runtime bundle")
    restored = ObjectSceneAnchorExposedQueryRelease.from_data(
        bundle.release.to_data()
    )
    replayed = release_object_scene_anchor_exposed_queries(
        plan=plan,
        predicate=predicate,
        durable_freeze=durable_freeze,
        expected_durable_freeze_commitment_digest=(
            expected_durable_freeze_commitment_digest
        ),
        locators=locators,
        load_released_record=load_released_record,
    )
    if (
        replayed.release != restored
        or replayed.released_panels != bundle.released_panels
    ):
        raise ObjectSceneAnchorExposedQueryGateError(
            "exposed query release differs from cold replay"
        )
    return ObjectSceneAnchorExposedQueryRuntimeBundle(
        release=restored,
        released_panels=bundle.released_panels,
    )


def _prediction_content(
    value: "ObjectSceneAnchorQueryPrediction",
) -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ANCHOR_EXPOSED_QUERY_PREDICTION_SCHEMA,
        "query_alias": value.query_alias,
        "query_release_digest": value.query_release_digest,
        "predicate_digest": value.predicate_digest,
        "disposition": value.disposition.value,
        "created_without_expected_label": True,
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorQueryPrediction:
    query_alias: str
    query_release_digest: str
    predicate_digest: str
    disposition: Disposition
    prediction_digest: str

    def __post_init__(self) -> None:
        _alias(self.query_alias)
        _raw_digest(self.query_release_digest, "query release digest")
        _raw_digest(self.predicate_digest, "prediction predicate digest")
        if type(self.disposition) is not Disposition:
            raise TypeError("prediction disposition must be exact Disposition")
        _raw_digest(self.prediction_digest, "prediction digest")
        if self.prediction_digest != canonical_digest(_prediction_content(self)):
            raise ObjectSceneAnchorExposedQueryGateError(
                "query prediction digest differs"
            )

    @classmethod
    def create(
        cls,
        *,
        query_alias: str,
        query_release_digest: str,
        predicate_digest: str,
        disposition: Disposition,
    ) -> "ObjectSceneAnchorQueryPrediction":
        values = {
            "query_alias": query_alias,
            "query_release_digest": query_release_digest,
            "predicate_digest": predicate_digest,
            "disposition": disposition,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            prediction_digest=canonical_digest(_prediction_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {
            **_prediction_content(self),
            "prediction_digest": self.prediction_digest,
        }

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorQueryPrediction":
        raw = _fields(
            value,
            {
                "schema",
                "query_alias",
                "query_release_digest",
                "predicate_digest",
                "disposition",
                "created_without_expected_label",
                "prediction_digest",
            },
            "query prediction",
        )
        if (
            raw["schema"] != OBJECT_SCENE_ANCHOR_EXPOSED_QUERY_PREDICTION_SCHEMA
            or raw["created_without_expected_label"] is not True
        ):
            raise ObjectSceneAnchorExposedQueryGateError(
                "query prediction policy differs"
            )
        try:
            disposition = Disposition(raw["disposition"])
        except (TypeError, ValueError) as exc:
            raise ObjectSceneAnchorExposedQueryGateError(
                "query prediction disposition differs"
            ) from exc
        result = cls(
            raw["query_alias"],
            raw["query_release_digest"],
            raw["predicate_digest"],
            disposition,
            raw["prediction_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorExposedQueryGateError(
                "query prediction is not canonical"
            )
        return result


def _score_row_content(
    value: "ObjectSceneAnchorQueryScoreRow",
) -> dict[str, object]:
    return {
        "query_alias": value.query_alias,
        "prediction_digest": value.prediction_digest,
        "predicted_disposition": value.predicted_disposition.value,
        "expected_disposition": value.expected_disposition.value,
        "correct": value.correct,
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorQueryScoreRow:
    query_alias: str
    prediction_digest: str
    predicted_disposition: Disposition
    expected_disposition: Disposition
    correct: bool
    row_digest: str

    def __post_init__(self) -> None:
        _alias(self.query_alias)
        _raw_digest(self.prediction_digest, "score prediction digest")
        if (
            type(self.predicted_disposition) is not Disposition
            or type(self.expected_disposition) is not Disposition
            or self.expected_disposition
            not in (Disposition.PRESENT, Disposition.CERTIFIED_ABSENT)
            or type(self.correct) is not bool
            or self.correct
            is not (self.predicted_disposition is self.expected_disposition)
        ):
            raise ObjectSceneAnchorExposedQueryGateError(
                "query score row disposition differs"
            )
        _raw_digest(self.row_digest, "query score row digest")
        if self.row_digest != canonical_digest(_score_row_content(self)):
            raise ObjectSceneAnchorExposedQueryGateError(
                "query score row digest differs"
            )

    def to_data(self) -> dict[str, object]:
        return {**_score_row_content(self), "row_digest": self.row_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorQueryScoreRow":
        raw = _fields(
            value,
            {
                "query_alias",
                "prediction_digest",
                "predicted_disposition",
                "expected_disposition",
                "correct",
                "row_digest",
            },
            "query score row",
        )
        try:
            predicted = Disposition(raw["predicted_disposition"])
            expected = Disposition(raw["expected_disposition"])
        except (TypeError, ValueError) as exc:
            raise ObjectSceneAnchorExposedQueryGateError(
                "query score row disposition differs"
            ) from exc
        result = cls(
            raw["query_alias"],
            raw["prediction_digest"],
            predicted,
            expected,
            raw["correct"],
            raw["row_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorExposedQueryGateError(
                "query score row is not canonical"
            )
        return result


def _score_content(value: "ObjectSceneAnchorExposedQueryScore") -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ANCHOR_EXPOSED_QUERY_SCORE_SCHEMA,
        "gate_id": OBJECT_SCENE_ANCHOR_EXPOSED_QUERY_GATE_ID,
        "gate_source_digest": object_scene_anchor_exposed_query_gate_source_digest(),
        "query_plan_digest": value.query_plan_digest,
        "query_release_digest": value.query_release_digest,
        "predicate_digest": value.predicate_digest,
        "prediction_digests": list(value.prediction_digests),
        "prediction_set_digest": value.prediction_set_digest,
        "rows": [item.to_data() for item in value.rows],
        "query_count": value.query_count,
        "determinate_count": value.determinate_count,
        "correct_count": value.correct_count,
        "accuracy_ppm": value.accuracy_ppm,
        "all_prediction_digests_validated_before_label_access": True,
        "labels_created_only_in_this_post_prediction_artifact": True,
        "indeterminate_or_error_predictions_are_incorrect": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorExposedQueryScore:
    query_plan_digest: str
    query_release_digest: str
    predicate_digest: str
    prediction_digests: tuple[str, ...]
    prediction_set_digest: str
    rows: tuple[ObjectSceneAnchorQueryScoreRow, ...]
    query_count: int
    determinate_count: int
    correct_count: int
    accuracy_ppm: int
    score_digest: str

    def __post_init__(self) -> None:
        for label, item in (
            ("query plan digest", self.query_plan_digest),
            ("query release digest", self.query_release_digest),
            ("predicate digest", self.predicate_digest),
            ("prediction set digest", self.prediction_set_digest),
            ("query score digest", self.score_digest),
        ):
            _raw_digest(item, label)
        for item in self.prediction_digests:
            _raw_digest(item, "prediction digest")
        for label, item in (
            ("query count", self.query_count),
            ("determinate count", self.determinate_count),
            ("correct count", self.correct_count),
            ("accuracy ppm", self.accuracy_ppm),
        ):
            _integer(item, label)
        expected_aliases = tuple(
            f"query_{index:03d}" for index in range(EXPOSED_QUERY_PANEL_COUNT)
        )
        if (
            type(self.prediction_digests) is not tuple
            or type(self.rows) is not tuple
            or self.query_count != EXPOSED_QUERY_PANEL_COUNT
            or len(self.prediction_digests) != self.query_count
            or len(set(self.prediction_digests)) != self.query_count
            or len(self.rows) != self.query_count
            or any(
                type(item) is not ObjectSceneAnchorQueryScoreRow
                for item in self.rows
            )
            or tuple(item.query_alias for item in self.rows) != expected_aliases
            or tuple(item.prediction_digest for item in self.rows)
            != self.prediction_digests
            or self.determinate_count
            != sum(
                item.predicted_disposition
                in (Disposition.PRESENT, Disposition.CERTIFIED_ABSENT)
                for item in self.rows
            )
            or self.correct_count != sum(item.correct for item in self.rows)
            or self.accuracy_ppm
            != (self.correct_count * 1_000_000) // self.query_count
            or self.prediction_set_digest
            != canonical_digest(
                {
                    "schema": "gkm.object-scene-anchor-query-prediction-set.v1",
                    "query_release_digest": self.query_release_digest,
                    "predicate_digest": self.predicate_digest,
                    "prediction_digests": list(self.prediction_digests),
                    "complete_exact_set": True,
                }
            )
        ):
            raise ObjectSceneAnchorExposedQueryGateError(
                "exposed query score inventory differs"
            )
        if self.score_digest != canonical_digest(_score_content(self)):
            raise ObjectSceneAnchorExposedQueryGateError(
                "exposed query score digest differs"
            )

    def to_data(self) -> dict[str, object]:
        return {**_score_content(self), "score_digest": self.score_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorExposedQueryScore":
        raw = _fields(
            value,
            {
                "schema",
                "gate_id",
                "gate_source_digest",
                "query_plan_digest",
                "query_release_digest",
                "predicate_digest",
                "prediction_digests",
                "prediction_set_digest",
                "rows",
                "query_count",
                "determinate_count",
                "correct_count",
                "accuracy_ppm",
                "all_prediction_digests_validated_before_label_access",
                "labels_created_only_in_this_post_prediction_artifact",
                "indeterminate_or_error_predictions_are_incorrect",
                *_authority_data(),
                "score_digest",
            },
            "exposed query score",
        )
        if (
            raw["schema"] != OBJECT_SCENE_ANCHOR_EXPOSED_QUERY_SCORE_SCHEMA
            or raw["gate_id"] != OBJECT_SCENE_ANCHOR_EXPOSED_QUERY_GATE_ID
            or raw["gate_source_digest"]
            != object_scene_anchor_exposed_query_gate_source_digest()
            or raw["all_prediction_digests_validated_before_label_access"] is not True
            or raw["labels_created_only_in_this_post_prediction_artifact"] is not True
            or raw["indeterminate_or_error_predictions_are_incorrect"] is not True
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["prediction_digests"], list)
            or not isinstance(raw["rows"], list)
        ):
            raise ObjectSceneAnchorExposedQueryGateError(
                "exposed query score policy differs"
            )
        result = cls(
            raw["query_plan_digest"],
            raw["query_release_digest"],
            raw["predicate_digest"],
            tuple(raw["prediction_digests"]),
            raw["prediction_set_digest"],
            tuple(
                ObjectSceneAnchorQueryScoreRow.from_data(item)
                for item in raw["rows"]
            ),
            raw["query_count"],
            raw["determinate_count"],
            raw["correct_count"],
            raw["accuracy_ppm"],
            raw["score_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorExposedQueryGateError(
                "exposed query score is not canonical"
            )
        return result


def reveal_and_score_object_scene_anchor_exposed_queries(
    *,
    metadata: ObjectSceneAnchorHistoricalMetadata,
    plan: ObjectSceneAnchorExposedQueryPlan,
    release: ObjectSceneAnchorExposedQueryRelease,
    predicate: ObjectSceneAnchorPythonPredicate,
    predictions: tuple[ObjectSceneAnchorQueryPrediction, ...],
) -> ObjectSceneAnchorExposedQueryScore:
    """Validate every prediction digest, then and only then interpret labels."""

    if type(metadata) is not ObjectSceneAnchorHistoricalMetadata:
        raise TypeError("metadata must be exact historical metadata")
    if type(plan) is not ObjectSceneAnchorExposedQueryPlan:
        raise TypeError("plan must be exact exposed query plan")
    if type(release) is not ObjectSceneAnchorExposedQueryRelease:
        raise TypeError("release must be exact exposed query release")
    if type(predicate) is not ObjectSceneAnchorPythonPredicate:
        raise TypeError("predicate must be exact ObjectSceneAnchorPythonPredicate")
    if type(predictions) is not tuple or any(
        type(item) is not ObjectSceneAnchorQueryPrediction for item in predictions
    ):
        raise TypeError("predictions must be an exact typed tuple")

    frozen_plan = verify_object_scene_anchor_exposed_query_plan(plan, metadata)
    frozen_release = ObjectSceneAnchorExposedQueryRelease.from_data(
        release.to_data()
    )
    frozen_predicate = ObjectSceneAnchorPythonPredicate.from_data(predicate.to_data())
    frozen_predictions = tuple(
        ObjectSceneAnchorQueryPrediction.from_data(item.to_data())
        for item in predictions
    )
    aliases = tuple(item.query_alias for item in frozen_plan.items)
    if (
        frozen_release.query_plan_digest != frozen_plan.plan_digest
        or frozen_release.predicate_digest != frozen_predicate.predicate_digest
        or frozen_release.query_aliases != aliases
        or len(frozen_predictions) != EXPOSED_QUERY_PANEL_COUNT
        or tuple(item.query_alias for item in frozen_predictions) != aliases
        or any(
            item.query_release_digest != frozen_release.release_digest
            or item.predicate_digest != frozen_predicate.predicate_digest
            for item in frozen_predictions
        )
        or len({item.prediction_digest for item in frozen_predictions})
        != EXPOSED_QUERY_PANEL_COUNT
    ):
        raise ObjectSceneAnchorExposedQueryGateError(
            "complete prediction digest set is absent; labels remain sealed"
        )

    # No expected tag state is accessed above this point.
    metadata_by_ordinal = metadata.by_ordinal
    orientation = frozen_predicate.candidate.orientation
    if orientation not in (
        ObjectSceneAnchorOrientation.SIDE0_POSITIVE,
        ObjectSceneAnchorOrientation.SIDE1_POSITIVE,
    ):
        raise ObjectSceneAnchorExposedQueryGateError(
            "predicate orientation cannot select historical label"
        )
    rows: list[ObjectSceneAnchorQueryScoreRow] = []
    for plan_item, prediction in zip(
        frozen_plan.items, frozen_predictions, strict=True
    ):
        historical = metadata_by_ordinal[plan_item.ordinal]
        tag_state = (
            historical.tag_0_state
            if orientation is ObjectSceneAnchorOrientation.SIDE0_POSITIVE
            else historical.tag_1_state
        )
        expected = (
            Disposition.PRESENT
            if tag_state == "present"
            else Disposition.CERTIFIED_ABSENT
        )
        values = {
            "query_alias": plan_item.query_alias,
            "prediction_digest": prediction.prediction_digest,
            "predicted_disposition": prediction.disposition,
            "expected_disposition": expected,
            "correct": prediction.disposition is expected,
        }
        provisional = object.__new__(ObjectSceneAnchorQueryScoreRow)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        rows.append(
            ObjectSceneAnchorQueryScoreRow(
                **values,
                row_digest=canonical_digest(_score_row_content(provisional)),
            )
        )
    prediction_digests = tuple(item.prediction_digest for item in frozen_predictions)
    prediction_set_digest = canonical_digest(
        {
            "schema": "gkm.object-scene-anchor-query-prediction-set.v1",
            "query_release_digest": frozen_release.release_digest,
            "predicate_digest": frozen_predicate.predicate_digest,
            "prediction_digests": list(prediction_digests),
            "complete_exact_set": True,
        }
    )
    frozen_rows = tuple(rows)
    values = {
        "query_plan_digest": frozen_plan.plan_digest,
        "query_release_digest": frozen_release.release_digest,
        "predicate_digest": frozen_predicate.predicate_digest,
        "prediction_digests": prediction_digests,
        "prediction_set_digest": prediction_set_digest,
        "rows": frozen_rows,
        "query_count": len(frozen_rows),
        "determinate_count": sum(
            item.predicted_disposition
            in (Disposition.PRESENT, Disposition.CERTIFIED_ABSENT)
            for item in frozen_rows
        ),
        "correct_count": sum(item.correct for item in frozen_rows),
        "accuracy_ppm": (
            sum(item.correct for item in frozen_rows) * 1_000_000
        )
        // len(frozen_rows),
    }
    provisional = object.__new__(ObjectSceneAnchorExposedQueryScore)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectSceneAnchorExposedQueryScore(
        **values,
        score_digest=canonical_digest(_score_content(provisional)),
    )


def verify_object_scene_anchor_exposed_query_score(
    score: ObjectSceneAnchorExposedQueryScore,
    *,
    metadata: ObjectSceneAnchorHistoricalMetadata,
    plan: ObjectSceneAnchorExposedQueryPlan,
    release: ObjectSceneAnchorExposedQueryRelease,
    predicate: ObjectSceneAnchorPythonPredicate,
    predictions: tuple[ObjectSceneAnchorQueryPrediction, ...],
) -> ObjectSceneAnchorExposedQueryScore:
    if type(score) is not ObjectSceneAnchorExposedQueryScore:
        raise TypeError("score must be exact exposed query score")
    restored = ObjectSceneAnchorExposedQueryScore.from_data(score.to_data())
    replayed = reveal_and_score_object_scene_anchor_exposed_queries(
        metadata=metadata,
        plan=plan,
        release=release,
        predicate=predicate,
        predictions=predictions,
    )
    if restored != replayed:
        raise ObjectSceneAnchorExposedQueryGateError(
            "exposed query score differs from model-free cold replay"
        )
    return restored


__all__ = (
    "EXPECTED_EXPOSED_QUERY_ORDINALS",
    "EXPOSED_QUERY_PANEL_COUNT",
    "HISTORICAL_PLAN_FILE_SHA256",
    "HISTORICAL_PLAN_RECORD_DIGEST",
    "HISTORICAL_RUNTIME_ARCHIVE_FILE_SHA256",
    "HISTORICAL_RUNTIME_ARCHIVE_RECORD_DIGEST",
    "OBJECT_SCENE_ANCHOR_DURABLE_PREDICATE_FREEZE_SCHEMA",
    "OBJECT_SCENE_ANCHOR_EXPOSED_QUERY_GATE_ID",
    "OBJECT_SCENE_ANCHOR_EXPOSED_QUERY_PLAN_SCHEMA",
    "OBJECT_SCENE_ANCHOR_EXPOSED_QUERY_PREDICTION_SCHEMA",
    "OBJECT_SCENE_ANCHOR_EXPOSED_QUERY_RELEASE_SCHEMA",
    "OBJECT_SCENE_ANCHOR_EXPOSED_QUERY_SCORE_SCHEMA",
    "OBJECT_SCENE_ANCHOR_RELEASED_RECORD_LOCATOR_SCHEMA",
    "ObjectSceneAnchorExposedQueryGateError",
    "ObjectSceneAnchorExposedQueryPlan",
    "ObjectSceneAnchorExposedQueryPlanItem",
    "ObjectSceneAnchorExposedQueryRelease",
    "ObjectSceneAnchorExposedQueryReleaseItem",
    "ObjectSceneAnchorExposedQueryRuntimeBundle",
    "ObjectSceneAnchorExposedQueryScore",
    "ObjectSceneAnchorHistoricalMetadata",
    "ObjectSceneAnchorHistoricalSceneMetadata",
    "ObjectSceneAnchorNeutralQueryPanelInput",
    "ObjectSceneAnchorPredicateDurableFreezeCommitment",
    "ObjectSceneAnchorQueryPrediction",
    "ObjectSceneAnchorQueryReleasedRecordLocator",
    "ObjectSceneAnchorQueryScoreRow",
    "ReleasedRecordLoader",
    "bind_caller_durable_object_scene_anchor_python_predicate",
    "build_object_scene_anchor_exposed_query_plan",
    "build_object_scene_anchor_historical_metadata",
    "load_object_scene_anchor_historical_metadata",
    "object_scene_anchor_exposed_query_gate_source_digest",
    "release_object_scene_anchor_exposed_queries",
    "released_record_directory_loader",
    "reveal_and_score_object_scene_anchor_exposed_queries",
    "verify_object_scene_anchor_exposed_query_plan",
    "verify_object_scene_anchor_exposed_query_release",
    "verify_object_scene_anchor_exposed_query_score",
)
