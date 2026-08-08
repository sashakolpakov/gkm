"""Sealed 24-call calibration for whole-panel soft-rubric predicates.

The calibration boundary is deliberately smaller than the retired object-atlas
pipeline.  Twelve already exposed PNGs are loaded from exact released-panel
records, two nominated prose contrasts are frozen, and each contrast is judged
once on each complete panel.  Only after all 24 artifacts have been durably
written and reloaded are the historical support sides introduced.

Python owns observation projection, candidate identity, support admission,
rank selection, and cold replay.  This module imports no atlas, hypothesis,
lineage, ranker, or Lean implementation.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Callable, Mapping, Sequence

from bongard import prototype_scene_observer as _scene_runtime
from bongard.canonical import canonical_digest, canonical_json
from bongard.evidence import Disposition
from bongard.object_bongard_panel_rubric_observer import (
    ObjectBongardPanelRubricArtifact,
    object_bongard_panel_rubric_output_schema,
    object_bongard_panel_rubric_prompt,
    observe_object_bongard_panel_rubric,
    verify_object_bongard_panel_rubric_artifact,
)
from bongard.object_bongard_panel_rubric_slate import (
    ObjectBongardPanelRubricSlateSelection,
    select_object_bongard_panel_rubric_slate,
)
from bongard.object_bongard_panel_rubric_version_space import (
    ObjectBongardPanelRubricSupportVersionSpace,
    PANEL_RUBRIC_SUPPORT_PANELS_PER_SIDE,
    build_object_bongard_panel_rubric_support_version_space,
)
from bongard.object_bongard_rubric_language import ObjectBongardRubricSpec
from bongard.object_bongard_semantics import (
    ObjectBongardSemanticArtifact,
    verify_object_bongard_semantic_artifact,
)
from bongard.object_bongard_turn_journal import (
    ObjectBongardNamedImageTurnJournalTransport,
    ObjectBongardTurnJournalSummary,
    ObjectBongardTurnRuntime,
    verify_object_bongard_turn_journal,
)
from bongard.official_panel_archive import ReleasedOfficialPanel
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID
from bongard.transport import CodexStructuredResult, run_codex_named_images_structured


PANEL_RUBRIC_CALIBRATION_SOURCE_SCHEMA = (
    "gkm.bongard-panel-rubric-calibration-source.v1"
)
PANEL_RUBRIC_CALIBRATION_PLAN_SCHEMA = (
    "gkm.bongard-panel-rubric-calibration-plan.v1"
)
PANEL_RUBRIC_CALIBRATION_FAILURE_EVIDENCE_SCHEMA = (
    "gkm.bongard-panel-rubric-calibration-failure-evidence.v1"
)
PANEL_RUBRIC_CALIBRATION_LIVE_RUN_SCHEMA = (
    "gkm.bongard-panel-rubric-calibration-live-run.v1"
)
PANEL_RUBRIC_CALIBRATION_BATCH_SCHEMA = (
    "gkm.bongard-panel-rubric-calibration-observation-batch.v1"
)
PANEL_RUBRIC_CALIBRATION_FREEZE_SCHEMA = (
    "gkm.bongard-panel-rubric-calibration-durable-freeze.v1"
)
PANEL_RUBRIC_CALIBRATION_ASSESSMENT_SCHEMA = (
    "gkm.bongard-panel-rubric-calibration-assessment.v1"
)
PANEL_RUBRIC_CALIBRATION_ALGORITHM_ID = (
    "bongard.panel-rubric-calibration/two-ranks-whole-panel-bounded-support-v1"
)

DEFAULT_OBJECT_BONGARD_PANEL_RUBRIC_CALIBRATION_SOURCE = Path(
    "downloads/ShapeBongard_V2_full/"
    "prototype_pair_python_campaign_20260807_object_v1/objects"
)
PANEL_RUBRIC_CALIBRATION_GROUP_0_ORDINALS = (0, 1, 3, 4, 5, 7)
PANEL_RUBRIC_CALIBRATION_GROUP_1_ORDINALS = (14, 17, 18, 19, 20, 21)
PANEL_RUBRIC_CALIBRATION_SELECTED_ORDINALS = (
    PANEL_RUBRIC_CALIBRATION_GROUP_0_ORDINALS
    + PANEL_RUBRIC_CALIBRATION_GROUP_1_ORDINALS
)
PANEL_RUBRIC_CALIBRATION_SPEC_COUNT = 2
PANEL_RUBRIC_CALIBRATION_PANEL_COUNT = 12
PANEL_RUBRIC_CALIBRATION_JOB_COUNT = 24

_PLAN_FILE_SHA256 = (
    "0447225b24bd440299f7976d29332d9bcd98a5f3d3d10b4fd453eb1ca634dc2c"
)
_PLAN_RECORD_DIGEST = (
    "sha256:d5643e8efc0fefaddeecd1fe90c2d47dfe25fc49aa401a06a6ba68615560102d"
)

# ordinal, neutral group index, panel ID, released-record file SHA-256,
# released-record content address, exact PNG SHA-256.  The loader opens only
# these records.  It does not scan a cohort or touch an official test split.
_PINNED_RELEASES = (
    (0, 0, "bd/bd_two_mismatch_sectors8-thin_seven_lines2_0000/1/0.png", "d0348829244ca07a387b9621f664fdbe6fc639ca43018caaefd43592a10928e6", "sha256:62c612838ecc986ceeb8e78b63d81b513d0ed0586c2697134c901ca7bf070d68", "71308a64182c5c193b46a7f3bd285194e5d71e42f44501b1b8180bbc8269c5d2"),
    (1, 0, "bd/bd_two_mismatch_sectors8-thin_seven_lines1_0000/1/2.png", "48cb46ab86d09ff1fad6f5f79da72b6fb9fb7ad72ee81c6af17479b049a6c36d", "sha256:7204fb38d6d97f3b7ab96cb713cbc8a7527d650905badbe1d2830b54999b9537", "328bb589fbefa3bacbd0118fcf45985d8973a2d72d626a9212a047f9e5dbe809"),
    (3, 0, "bd/bd_two_mismatch_sectors8-advanced_lamp5_0000/1/0.png", "36ccb57ff6c3d8f76b64f03c05ad4d98aed9bc20217cd1cd945f085ce8b6e45b", "sha256:79d9c48d8779027f17e895121f57e676786836465d642b360584dce6c05df159", "7422d3b033451f92486091739e3ad593b97889a0bd9746938f82b1461c73c29a"),
    (4, 0, "bd/bd_two_mismatch_sectors8-three_mismatch_triangles4_0000/1/2.png", "ff529a338fd2ba4cf037aeb92474df314b85e27e19d96dd59cbd9b470ae7e39e", "sha256:d14c18c14e3b7c38d2c61ec4859cb515bec3b25cae20824af41186e6b280ec5b", "a818d3f3d730ddfeed399075ed1d3bf23f91c876c2bdc5bae046fcd8a7c85bf8"),
    (5, 0, "bd/bd_open_line_arc5-two_mismatch_sectors8_0000/1/6.png", "b044be705b4d65d8cb16af956f14d5b9f6fe9014e7a653d04ca15385dc1c8cbe", "sha256:cf22b0feac983d8170939babe0c0058322858ac63785336c976729c4d3033d87", "6ee727aeded6b2c20c0ef1d9f34b9faed2fe22e7872019101ecc565eca7b8919"),
    (7, 0, "bd/bd_thin_parallel_bridge-two_mismatch_sectors8_0000/1/2.png", "bf4a4430fe1e5beac31468cf78c82c5d34eba615aa4d262af3d26acbc77ffb73", "sha256:7d416534b9255e4ad8c097502ce29634716c97891385caecc7b74946c4e73501", "0ea165d64ac9935d73237f78a14f948a7d6a84e32e64127dcf4e5475f8ccf64c"),
    (14, 1, "bd/bd_acute_nonequi_triangle-exist_triangle_three_lines4_0000/1/2.png", "660a8dd1c80a692d68ae35e10f9b939d3ca536f7a83c9704c9d694c21690b957", "sha256:7130787849334e9f2c45b6f29ae4920241cb42749e44d9b6e6d9245252ecab14", "14847a726f5e80fdd68ff82515007ddee47108ebc0c7c84e2dc1e94e08a8fd34"),
    (17, 1, "bd/bd_two_mismatch_triangles6-exist_triangle_three_lines4_0000/1/6.png", "af7f676535cca1bf15051ec23d9ba6de0b7158ec2f5fd2acf69fe6ecab8a81eb", "sha256:dd835e1855a2c07210f416aec1cbdac091545e4a321d9e6024db131f6f6e99fd", "1f35c3c4bf1e138cc25beec5615cee59baddab98a69a244147e121dd1ac3605d"),
    (18, 1, "bd/bd_thin_symm_band-exist_triangle_three_lines4_0000/1/1.png", "18739494e35a4fbefcc4468996399c46ec304cd0abf4b97beef06c8a4b81cef3", "sha256:18f0b91009dcbd06e5a3788f630b87d06feb35f661086f3963400d63333ecc9a", "95ac8cd158f36268e1e96e863d49fb5a8c0f38a8ab05955d4667020f67e701bf"),
    (19, 1, "bd/bd_dagger_sector1-exist_triangle_three_lines4_0000/1/5.png", "cf3235e51c7d84cd280ab7a74df9a5cb40e7c19d605f9dffe30913363cce0fd0", "sha256:3b98dadc3955b264e7af1ebb3ebc27a879d2f1d9937b4063799ecada073c4394", "d2c77806e01ffff998969901529344e34786fd408aaed3712920d8932cacb8ab"),
    (20, 1, "bd/bd_thin_regular_lamp-exist_triangle_three_lines4_0000/1/2.png", "6174740819ba35114a070aadfcd0770e63f87df171e42fb5a926e7a358c64915", "sha256:273cafea58880aee46dac8592e0a3a39c3bc005b98aece5c69c55f49306d4989", "d27b9a673ce1768729f3a533321f05c4a2addd1619011df85103d5d43f55a933"),
    (21, 1, "bd/bd_exist_triangle_three_lines4-thin_seven_lines3_0000/1/5.png", "15eb3f56bdb656a59f9d64cc30995176d03d2b017f74aa9844ea9be383956901", "sha256:f42d0b372be5993525fee462644534c996c4adfddedd215d692bec1b006cbc27", "a348487f86bf2f1c306789ef3f386934d333a9fcf7337c3888d6c9762b834859"),
)

_RAW_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_EXCEPTION_TYPE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]{0,255}\Z")
_MAX_JSON_BYTES = 16 * 1024 * 1024


class ObjectBongardPanelRubricCalibrationError(ValueError):
    """A source, nomination, observation batch, or replay differs."""


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "lean_present": False,
        "lean_required": False,
        "lean_removable": True,
        "lean_defines_identity_or_decision": False,
        "lean_required_for_replay": False,
        "lean_removal_changes_decision": False,
    }


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(not isinstance(key, str) for key in value)
        or set(value) != expected
    ):
        raise ObjectBongardPanelRubricCalibrationError(f"{label} fields differ")
    return value


def _raw_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _RAW_DIGEST.fullmatch(value) is None:
        raise ObjectBongardPanelRubricCalibrationError(
            f"{label} must be a raw lowercase SHA-256"
        )
    return value


def _address(value: object, label: str) -> str:
    if not isinstance(value, str) or _ADDRESS.fullmatch(value) is None:
        raise ObjectBongardPanelRubricCalibrationError(
            f"{label} must be a sha256: address"
        )
    return value


def _raw_or_address(value: object, label: str) -> str:
    if not isinstance(value, str) or (
        _RAW_DIGEST.fullmatch(value) is None and _ADDRESS.fullmatch(value) is None
    ):
        raise ObjectBongardPanelRubricCalibrationError(
            f"{label} must be a raw or addressed SHA-256"
        )
    return value


def object_bongard_panel_rubric_calibration_source_digest() -> str:
    return verify_loaded_source(
        __name__, expected_source_sha256=_LOADED_SOURCE_SHA256
    )


def _read_exact_json(path: Path, expected_file_sha256: str) -> dict[str, Any]:
    digest = _raw_digest(expected_file_sha256, "file SHA-256")
    if path.name != f"{digest}.json":
        raise ObjectBongardPanelRubricCalibrationError(
            "record filename differs from its commitment"
        )
    try:
        before = path.lstat()
    except OSError as exc:
        raise ObjectBongardPanelRubricCalibrationError(
            "pinned record is unavailable"
        ) from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or not 0 < before.st_size <= _MAX_JSON_BYTES
    ):
        raise ObjectBongardPanelRubricCalibrationError(
            "pinned record is not a bounded regular file"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ObjectBongardPanelRubricCalibrationError(
            "cannot open pinned record"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino, opened.st_size) != (
            before.st_dev,
            before.st_ino,
            before.st_size,
        ):
            raise ObjectBongardPanelRubricCalibrationError(
                "pinned record changed while opening"
            )
        payload = bytearray()
        while len(payload) < opened.st_size:
            chunk = os.read(descriptor, min(65536, opened.st_size - len(payload)))
            if not chunk:
                raise ObjectBongardPanelRubricCalibrationError(
                    "pinned record was truncated"
                )
            payload.extend(chunk)
        after = os.fstat(descriptor)
        if (after.st_dev, after.st_ino, after.st_size) != (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
        ):
            raise ObjectBongardPanelRubricCalibrationError(
                "pinned record changed while reading"
            )
    finally:
        os.close(descriptor)
    frozen = bytes(payload)
    if hashlib.sha256(frozen).hexdigest() != digest:
        raise ObjectBongardPanelRubricCalibrationError(
            "pinned record bytes differ from their commitment"
        )
    try:
        decoded = json.loads(frozen.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ObjectBongardPanelRubricCalibrationError(
            "pinned record is malformed JSON"
        ) from exc
    if not isinstance(decoded, dict) or canonical_json(decoded) + b"\n" != frozen:
        raise ObjectBongardPanelRubricCalibrationError(
            "pinned record is not canonical JSON plus one newline"
        )
    return decoded


def _verify_record_digest(
    value: Mapping[str, Any], *, field: str, addressed: bool, label: str
) -> None:
    expected = canonical_digest(
        {key: item for key, item in value.items() if key != field}
    )
    if value.get(field) != ("sha256:" + expected if addressed else expected):
        raise ObjectBongardPanelRubricCalibrationError(
            f"{label} canonical record digest differs"
        )


def _panel_content(value: "ObjectBongardPanelRubricCalibrationPanel") -> dict[str, object]:
    return {
        "ordinal": value.ordinal,
        "neutral_group_index_commitment": value.group_index,
        "task_id": value.task_id,
        "panel_id": value.panel_id,
        "released_file_sha256": value.released_file_sha256,
        "released_record_digest": value.released_record_digest,
        "png_sha256": value.png_sha256,
        "whole_panel_bytes_only": True,
        "atlas_geometry_present": False,
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardPanelRubricCalibrationPanel:
    ordinal: int
    group_index: int
    task_id: str
    panel_id: str
    released_file_sha256: str
    released_record_digest: str
    png_sha256: str
    exact_png_bytes: bytes
    panel_binding_digest: str

    def __post_init__(self) -> None:
        rows = {row[0]: row for row in _PINNED_RELEASES}
        pinned = rows.get(self.ordinal)
        if (
            pinned is None
            or self.group_index != pinned[1]
            or self.panel_id != pinned[2]
            or self.released_file_sha256 != pinned[3]
            or self.released_record_digest != pinned[4]
            or self.png_sha256 != pinned[5]
            or self.task_id != self.panel_id.split("/")[1]
        ):
            raise ObjectBongardPanelRubricCalibrationError(
                "calibration panel differs from the pinned release"
            )
        _raw_digest(self.released_file_sha256, "released file digest")
        _address(self.released_record_digest, "released record digest")
        _raw_digest(self.png_sha256, "PNG digest")
        _raw_digest(self.panel_binding_digest, "panel binding digest")
        if (
            not isinstance(self.exact_png_bytes, bytes)
            or not self.exact_png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
            or hashlib.sha256(self.exact_png_bytes).hexdigest() != self.png_sha256
            or self.panel_binding_digest != canonical_digest(_panel_content(self))
        ):
            raise ObjectBongardPanelRubricCalibrationError(
                "calibration panel bytes or binding differ"
            )

    def commitment_data(self) -> dict[str, object]:
        return {**_panel_content(self), "panel_binding_digest": self.panel_binding_digest}


def _source_content(value: "ObjectBongardPanelRubricCalibrationSource") -> dict[str, object]:
    return {
        "schema": PANEL_RUBRIC_CALIBRATION_SOURCE_SCHEMA,
        "algorithm_id": PANEL_RUBRIC_CALIBRATION_ALGORITHM_ID,
        "implementation_source_sha256": (
            object_bongard_panel_rubric_calibration_source_digest()
        ),
        "historical_plan_file_sha256": value.historical_plan_file_sha256,
        "historical_plan_record_digest": value.historical_plan_record_digest,
        "selected_ordinals": list(PANEL_RUBRIC_CALIBRATION_SELECTED_ORDINALS),
        "panels": [item.commitment_data() for item in value.panels],
        "selection_policy": "exactly-six-per-neutral-group-from-prior-exposure",
        "official_test_pixels_opened": False,
        "fresh_broad_cohort_pixels_opened": False,
        "support_roles_model_visible": False,
        "whole_panel_only": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardPanelRubricCalibrationSource:
    historical_plan_file_sha256: str
    historical_plan_record_digest: str
    panels: tuple[ObjectBongardPanelRubricCalibrationPanel, ...]
    source_digest: str

    def __post_init__(self) -> None:
        if (
            self.historical_plan_file_sha256 != _PLAN_FILE_SHA256
            or self.historical_plan_record_digest != _PLAN_RECORD_DIGEST
            or not isinstance(self.panels, tuple)
            or len(self.panels) != PANEL_RUBRIC_CALIBRATION_PANEL_COUNT
            or tuple(item.ordinal for item in self.panels)
            != PANEL_RUBRIC_CALIBRATION_SELECTED_ORDINALS
            or tuple(item.group_index for item in self.panels) != (0,) * 6 + (1,) * 6
            or len({item.panel_id for item in self.panels}) != len(self.panels)
        ):
            raise ObjectBongardPanelRubricCalibrationError(
                "calibration source inventory differs"
            )
        _raw_digest(self.source_digest, "calibration source digest")
        if self.source_digest != canonical_digest(_source_content(self)):
            raise ObjectBongardPanelRubricCalibrationError(
                "calibration source digest differs"
            )

    @property
    def group_0_panels(self) -> tuple[ObjectBongardPanelRubricCalibrationPanel, ...]:
        return tuple(item for item in self.panels if item.group_index == 0)

    @property
    def group_1_panels(self) -> tuple[ObjectBongardPanelRubricCalibrationPanel, ...]:
        return tuple(item for item in self.panels if item.group_index == 1)

    def panel_by_id(self, panel_id: str) -> ObjectBongardPanelRubricCalibrationPanel:
        rows = tuple(item for item in self.panels if item.panel_id == panel_id)
        if len(rows) != 1:
            raise ObjectBongardPanelRubricCalibrationError(
                "panel is outside the exact calibration source"
            )
        return rows[0]

    def to_data(self) -> dict[str, object]:
        return {**_source_content(self), "source_digest": self.source_digest}


def load_object_bongard_panel_rubric_calibration_source(
    directory: str | os.PathLike[str] = (
        DEFAULT_OBJECT_BONGARD_PANEL_RUBRIC_CALIBRATION_SOURCE
    ),
) -> ObjectBongardPanelRubricCalibrationSource:
    """Load only the exact 12 already exposed released-panel records."""

    root = Path(directory)
    plan = _read_exact_json(
        root / "calibration_plan" / f"{_PLAN_FILE_SHA256}.json",
        _PLAN_FILE_SHA256,
    )
    _verify_record_digest(
        plan,
        field="record_digest",
        addressed=True,
        label="historical calibration plan",
    )
    if (
        plan.get("schema") != "gkm.bongard-prototype-scene-calibration-plan.v1"
        or plan.get("record_digest") != _PLAN_RECORD_DIGEST
        or not isinstance(plan.get("scenes"), list)
        or len(plan["scenes"]) != 28
        or [item.get("ordinal") for item in plan["scenes"]] != list(range(28))
    ):
        raise ObjectBongardPanelRubricCalibrationError(
            "historical calibration plan identity differs"
        )
    scenes = {item["ordinal"]: item for item in plan["scenes"]}
    panels: list[ObjectBongardPanelRubricCalibrationPanel] = []
    for ordinal, group_index, panel_id, file_sha, record_digest, png_sha in _PINNED_RELEASES:
        scene = scenes[ordinal]
        expected_states = ("present", "absent") if group_index == 0 else ("absent", "present")
        states = scene.get("expected_tag_states")
        task_id = panel_id.split("/")[1]
        if (
            scene.get("task_id") != task_id
            or scene.get("panel_id") != panel_id
            or not isinstance(states, list)
            or tuple(item.get("state") for item in states) != expected_states
            or tuple(item.get("tag_id") for item in states)
            != ("opaque_visual_tag_0", "opaque_visual_tag_1")
        ):
            raise ObjectBongardPanelRubricCalibrationError(
                "historical selected panel or sealed support direction differs"
            )
        raw = _read_exact_json(
            root / "released_panel" / f"{file_sha}.json", file_sha
        )
        released = ReleasedOfficialPanel.from_data(raw)
        if (
            released.panel_id != panel_id
            or released.record_digest != record_digest
            or released.exact_png_digest != "sha256:" + png_sha
        ):
            raise ObjectBongardPanelRubricCalibrationError(
                "released-panel identity differs"
            )
        values = {
            "ordinal": ordinal,
            "group_index": group_index,
            "task_id": task_id,
            "panel_id": panel_id,
            "released_file_sha256": file_sha,
            "released_record_digest": record_digest,
            "png_sha256": png_sha,
            "exact_png_bytes": released.exact_png_bytes,
        }
        provisional = object.__new__(ObjectBongardPanelRubricCalibrationPanel)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        panels.append(
            ObjectBongardPanelRubricCalibrationPanel(
                **values,  # type: ignore[arg-type]
                panel_binding_digest=canonical_digest(_panel_content(provisional)),
            )
        )
    values = {
        "historical_plan_file_sha256": _PLAN_FILE_SHA256,
        "historical_plan_record_digest": _PLAN_RECORD_DIGEST,
        "panels": tuple(panels),
    }
    provisional = object.__new__(ObjectBongardPanelRubricCalibrationSource)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectBongardPanelRubricCalibrationSource(
        **values,
        source_digest=canonical_digest(_source_content(provisional)),
    )


def _plan_content(value: "ObjectBongardPanelRubricCalibrationPlan") -> dict[str, object]:
    return {
        "schema": PANEL_RUBRIC_CALIBRATION_PLAN_SCHEMA,
        "algorithm_id": PANEL_RUBRIC_CALIBRATION_ALGORITHM_ID,
        "source": value.source.to_data(),
        "nomination_artifact": value.nomination_artifact.to_data(),
        "nomination_authorization_digest": value.nomination_authorization_digest,
        "nomination_execution_precommit_digest": (
            value.nomination_execution_precommit_digest
        ),
        "nomination_cold_replay_digest": value.nomination_cold_replay_digest,
        "nomination_result_digest": value.nomination_result_digest,
        "nomination_source_digest": value.nomination_source_digest,
        "rubric_specs": [item.to_data() for item in value.rubric_specs],
        "job_order": "candidate-rank-then-source-ordinal",
        "job_count": PANEL_RUBRIC_CALIBRATION_JOB_COUNT,
        "one_complete_panel_per_job": True,
        "labels_visible_to_observer": False,
        "labels_introduced_only_after_durable_batch_reload": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardPanelRubricCalibrationPlan:
    source: ObjectBongardPanelRubricCalibrationSource
    nomination_artifact: ObjectBongardSemanticArtifact
    nomination_authorization_digest: str
    nomination_execution_precommit_digest: str
    nomination_cold_replay_digest: str
    nomination_result_digest: str
    nomination_source_digest: str
    rubric_specs: tuple[ObjectBongardRubricSpec, ObjectBongardRubricSpec]
    plan_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.source, ObjectBongardPanelRubricCalibrationSource):
            raise TypeError("source must be a panel calibration source")
        for name in (
            "nomination_authorization_digest",
            "nomination_execution_precommit_digest",
            "nomination_cold_replay_digest",
            "nomination_result_digest",
        ):
            _address(getattr(self, name), name)
        _raw_digest(self.nomination_source_digest, "nomination source digest")
        if not isinstance(self.nomination_artifact, ObjectBongardSemanticArtifact):
            raise TypeError("nomination artifact must be typed")
        artifact = ObjectBongardSemanticArtifact.from_data(
            self.nomination_artifact.to_data(),
            expected_artifact_digest=self.nomination_artifact.artifact_digest,
        )
        expected_groups = (
            tuple(sorted(item.panel_id for item in self.source.group_0_panels)),
            tuple(sorted(item.panel_id for item in self.source.group_1_panels)),
        )
        if artifact != self.nomination_artifact or artifact.group_panel_ids != expected_groups:
            raise ObjectBongardPanelRubricCalibrationError(
                "nomination artifact groups differ from the pinned panels"
            )
        verify_object_bongard_semantic_artifact(
            artifact,
            support_png_by_panel_id={
                item.panel_id: item.exact_png_bytes for item in self.source.panels
            },
            expected_task_id=self.source.panels[0].task_id,
            expected_observation_context_digest=(
                self.nomination_execution_precommit_digest
            ),
            expected_artifact_digest=artifact.artifact_digest,
        )
        expected_specs = tuple(
            ObjectBongardRubricSpec.from_semantic_artifact(
                artifact,
                expected_artifact_digest=artifact.artifact_digest,
                candidate_rank=rank,
            )
            for rank in (0, 1)
        )
        if self.rubric_specs != expected_specs:
            raise ObjectBongardPanelRubricCalibrationError(
                "calibration plan does not contain the exact two nominated ranks"
            )
        _raw_digest(self.plan_digest, "panel calibration plan digest")
        if self.plan_digest != canonical_digest(_plan_content(self)):
            raise ObjectBongardPanelRubricCalibrationError(
                "panel calibration plan digest differs"
            )

    def to_data(self) -> dict[str, object]:
        return {**_plan_content(self), "plan_digest": self.plan_digest}


def bind_object_bongard_panel_rubric_calibration_nomination(
    source: ObjectBongardPanelRubricCalibrationSource,
    nomination: object,
) -> ObjectBongardPanelRubricCalibrationPlan:
    """Bind a command-verified nomination without importing its command module.

    The accepted nomination wrapper is structural on purpose: frozen historical
    nomination verifiers and the current command verifier can both supply the
    same seven exact fields without making this Python predicate path depend on
    a mutable launcher implementation.
    """

    if not isinstance(source, ObjectBongardPanelRubricCalibrationSource):
        raise TypeError("source must be a panel calibration source")
    required = (
        "artifact",
        "authorization_digest",
        "execution_precommit_digest",
        "cold_replay_digest",
        "result_digest",
        "source_digest",
        "accepted",
    )
    if any(not hasattr(nomination, name) for name in required):
        raise TypeError("nomination must be a verified nomination wrapper")
    artifact = getattr(nomination, "artifact")
    if getattr(nomination, "accepted") is not True:
        raise ObjectBongardPanelRubricCalibrationError(
            "rejected nomination cannot enter calibration"
        )
    if not isinstance(artifact, ObjectBongardSemanticArtifact):
        raise TypeError("verified nomination artifact has the wrong type")
    frozen = ObjectBongardSemanticArtifact.from_data(
        artifact.to_data(), expected_artifact_digest=artifact.artifact_digest
    )
    specs = tuple(
        ObjectBongardRubricSpec.from_semantic_artifact(
            frozen,
            expected_artifact_digest=frozen.artifact_digest,
            candidate_rank=rank,
        )
        for rank in (0, 1)
    )
    values = {
        "source": source,
        "nomination_artifact": frozen,
        "nomination_authorization_digest": getattr(
            nomination, "authorization_digest"
        ),
        "nomination_execution_precommit_digest": getattr(
            nomination, "execution_precommit_digest"
        ),
        "nomination_cold_replay_digest": getattr(
            nomination, "cold_replay_digest"
        ),
        "nomination_result_digest": getattr(nomination, "result_digest"),
        "nomination_source_digest": getattr(nomination, "source_digest"),
        "rubric_specs": specs,
    }
    provisional = object.__new__(ObjectBongardPanelRubricCalibrationPlan)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectBongardPanelRubricCalibrationPlan(
        **values,  # type: ignore[arg-type]
        plan_digest=canonical_digest(_plan_content(provisional)),
    )


def _journal_summary_from_data(value: object) -> ObjectBongardTurnJournalSummary:
    authority = {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "lean_present": False,
        "lean_required": False,
        "lean_removable": True,
        "lean_affects_identity_or_replay": False,
    }
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
            *authority,
        },
        "turn journal summary",
    )
    if (
        raw["schema"] != "gkm.bongard-codex-turn-journal-summary.v1"
        or raw["terminal_status"] not in {"success", "failure", "unclaimed"}
        or any(raw[key] != item for key, item in authority.items())
        or raw["record_digest"]
        != "sha256:"
        + canonical_digest(
            {key: item for key, item in raw.items() if key != "record_digest"}
        )
    ):
        raise ObjectBongardPanelRubricCalibrationError(
            "turn journal summary policy or digest differs"
        )
    for name in (
        "manifest_digest",
        "turn_key",
        "claim_digest",
        "result_digest",
        "outcome_digest",
        "record_digest",
    ):
        item = raw[name]
        if item is not None:
            _raw_or_address(item, f"journal summary {name}")
    result = ObjectBongardTurnJournalSummary(
        raw["manifest_digest"],
        raw["turn_key"],
        raw["terminal_status"],
        raw["claim_digest"],
        raw["result_digest"],
        raw["outcome_digest"],
        raw["record_digest"],
    )
    if result.to_data() != dict(raw):
        raise ObjectBongardPanelRubricCalibrationError(
            "turn journal summary is not canonical"
        )
    return result


def _failure_evidence_content(
    value: "ObjectBongardPanelRubricFailureEvidence",
) -> dict[str, object]:
    return {
        "schema": PANEL_RUBRIC_CALIBRATION_FAILURE_EVIDENCE_SCHEMA,
        "panel_binding_digest": value.panel_binding_digest,
        "rubric_spec_digest": value.rubric_spec_digest,
        "exception_type": value.exception_type,
        "diagnostic_code": value.diagnostic_code,
        "message_prefix_sha256": value.message_prefix_sha256,
        "message_prefix_byte_count": value.message_prefix_byte_count,
        "message_truncated": value.message_truncated,
        "raw_message_persisted": False,
        "secret_values_persisted": False,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardPanelRubricFailureEvidence:
    """Bounded, secret-free diagnostic for one failed physical turn."""

    panel_binding_digest: str
    rubric_spec_digest: str
    exception_type: str
    diagnostic_code: str
    message_prefix_sha256: str | None
    message_prefix_byte_count: int
    message_truncated: bool
    evidence_digest: str

    def __post_init__(self) -> None:
        _raw_digest(self.panel_binding_digest, "failure panel binding digest")
        _raw_digest(self.rubric_spec_digest, "failure rubric spec digest")
        if (
            not isinstance(self.exception_type, str)
            or _EXCEPTION_TYPE.fullmatch(self.exception_type) is None
            or self.diagnostic_code
            not in {
                "cloud_policy_cache_expired",
                "cloud_policy_cache_rejected",
                "model_catalog_rejected",
                "no_tools_attestation_rejected",
                "launcher_binding_rejected",
                "transport_timeout",
                "transport_unavailable",
                "unclassified_transport_failure",
                "journal_envelope_failure_unclassified",
            }
            or isinstance(self.message_prefix_byte_count, bool)
            or not isinstance(self.message_prefix_byte_count, int)
            or not 0 <= self.message_prefix_byte_count <= 4096
            or not isinstance(self.message_truncated, bool)
        ):
            raise ObjectBongardPanelRubricCalibrationError(
                "failure evidence classification differs"
            )
        if self.message_prefix_sha256 is None:
            if self.message_prefix_byte_count != 0 or self.message_truncated:
                raise ObjectBongardPanelRubricCalibrationError(
                    "message-free failure evidence has message metadata"
                )
        else:
            _raw_digest(self.message_prefix_sha256, "failure message digest")
        _address(self.evidence_digest, "failure evidence digest")
        if self.evidence_digest != "sha256:" + canonical_digest(
            _failure_evidence_content(self)
        ):
            raise ObjectBongardPanelRubricCalibrationError(
                "failure evidence digest differs"
            )

    def to_data(self) -> dict[str, object]:
        return {
            **_failure_evidence_content(self),
            "evidence_digest": self.evidence_digest,
        }

    @classmethod
    def from_data(cls, value: object) -> "ObjectBongardPanelRubricFailureEvidence":
        raw = _fields(
            value,
            {
                "schema",
                "panel_binding_digest",
                "rubric_spec_digest",
                "exception_type",
                "diagnostic_code",
                "message_prefix_sha256",
                "message_prefix_byte_count",
                "message_truncated",
                "raw_message_persisted",
                "secret_values_persisted",
                *_authority_data(),
                "evidence_digest",
            },
            "failure evidence",
        )
        if (
            raw["schema"] != PANEL_RUBRIC_CALIBRATION_FAILURE_EVIDENCE_SCHEMA
            or raw["raw_message_persisted"] is not False
            or raw["secret_values_persisted"] is not False
            or any(raw[key] != item for key, item in _authority_data().items())
        ):
            raise ObjectBongardPanelRubricCalibrationError(
                "failure evidence policy differs"
            )
        result = cls(
            raw["panel_binding_digest"],
            raw["rubric_spec_digest"],
            raw["exception_type"],
            raw["diagnostic_code"],
            raw["message_prefix_sha256"],
            raw["message_prefix_byte_count"],
            raw["message_truncated"],
            raw["evidence_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardPanelRubricCalibrationError(
                "failure evidence is not canonical"
            )
        return result


def _classify_failure_message(message: str) -> str:
    lowered = message.lower()
    if "cloud" in lowered and "cache" in lowered and (
        "expired" in lowered or "stale" in lowered
    ):
        return "cloud_policy_cache_expired"
    if "cloud" in lowered and "cache" in lowered:
        return "cloud_policy_cache_rejected"
    if "model" in lowered and "catalog" in lowered:
        return "model_catalog_rejected"
    if "attestation" in lowered or "tool surface" in lowered:
        return "no_tools_attestation_rejected"
    if "launcher" in lowered:
        return "launcher_binding_rejected"
    if "timeout" in lowered or "timed out" in lowered:
        return "transport_timeout"
    if any(word in lowered for word in ("network", "connection", "dns", "socket")):
        return "transport_unavailable"
    return "unclassified_transport_failure"


def _seal_failure_evidence(
    panel: ObjectBongardPanelRubricCalibrationPanel,
    spec: ObjectBongardRubricSpec,
    exception: Exception,
) -> ObjectBongardPanelRubricFailureEvidence:
    exception_type = f"{type(exception).__module__}.{type(exception).__qualname__}"
    if _EXCEPTION_TYPE.fullmatch(exception_type) is None:
        exception_type = "builtins.Exception"
    try:
        raw = str(exception).encode("utf-8", errors="replace")
    except Exception:
        raw = b""
    prefix = raw[:4096]
    values = {
        "panel_binding_digest": panel.panel_binding_digest,
        "rubric_spec_digest": spec.spec_digest,
        "exception_type": exception_type,
        "diagnostic_code": _classify_failure_message(
            prefix.decode("utf-8", errors="replace")
        ),
        "message_prefix_sha256": (
            None if not prefix else hashlib.sha256(prefix).hexdigest()
        ),
        "message_prefix_byte_count": len(prefix),
        "message_truncated": len(raw) > len(prefix),
    }
    provisional = object.__new__(ObjectBongardPanelRubricFailureEvidence)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectBongardPanelRubricFailureEvidence(
        **values,
        evidence_digest="sha256:" + canonical_digest(
            _failure_evidence_content(provisional)
        ),
    )


def _fallback_failure_evidence(
    panel: ObjectBongardPanelRubricCalibrationPanel,
    spec: ObjectBongardRubricSpec,
    exception_type: str,
) -> ObjectBongardPanelRubricFailureEvidence:
    if _EXCEPTION_TYPE.fullmatch(exception_type) is None:
        exception_type = "builtins.Exception"
    values = {
        "panel_binding_digest": panel.panel_binding_digest,
        "rubric_spec_digest": spec.spec_digest,
        "exception_type": exception_type,
        "diagnostic_code": "journal_envelope_failure_unclassified",
        "message_prefix_sha256": None,
        "message_prefix_byte_count": 0,
        "message_truncated": False,
    }
    provisional = object.__new__(ObjectBongardPanelRubricFailureEvidence)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectBongardPanelRubricFailureEvidence(
        **values,
        evidence_digest="sha256:" + canonical_digest(
            _failure_evidence_content(provisional)
        ),
    )


def _write_once_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json(dict(value)) + b"\n"
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        try:
            info = path.lstat()
        except OSError as exc:
            raise ObjectBongardPanelRubricCalibrationError(
                "existing durable record cannot be inspected"
            ) from exc
        if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise ObjectBongardPanelRubricCalibrationError(
                "existing durable record is not a regular file"
            )
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
            os, "O_NOFOLLOW", 0
        )
        descriptor = os.open(path, flags)
        try:
            existing = bytearray()
            while len(existing) < info.st_size:
                chunk = os.read(descriptor, min(65536, info.st_size - len(existing)))
                if not chunk:
                    break
                existing.extend(chunk)
        finally:
            os.close(descriptor)
        if bytes(existing) != payload:
            raise ObjectBongardPanelRubricCalibrationError(
                "durable record differs from an existing commitment"
            )
        return
    try:
        written = 0
        while written < len(payload):
            written += os.write(descriptor, payload[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


class ObjectBongardPanelRubricJournalDispatcher:
    """One exactly-once whole-panel journal plus sanitized failure evidence."""

    def __init__(
        self,
        journal_root: str | os.PathLike[str],
        *,
        panel: ObjectBongardPanelRubricCalibrationPanel,
        rubric_spec: ObjectBongardRubricSpec,
        authorization_digest: str,
        execution_precommit_digest: str,
        runtime: ObjectBongardTurnRuntime,
        underlying_transport: Callable[..., CodexStructuredResult] = (
            run_codex_named_images_structured
        ),
    ) -> None:
        if not isinstance(panel, ObjectBongardPanelRubricCalibrationPanel):
            raise TypeError("panel must be a typed calibration panel")
        if not isinstance(rubric_spec, ObjectBongardRubricSpec):
            raise TypeError("rubric_spec must be typed")
        if not isinstance(runtime, ObjectBongardTurnRuntime):
            raise TypeError("runtime must be typed")
        if runtime.transport_source_digest != (
            _scene_runtime.prototype_scene_transport_source_digest()
        ):
            raise ObjectBongardPanelRubricCalibrationError(
                "runtime transport source differs from the panel observer"
            )
        _raw_or_address(authorization_digest, "authorization digest")
        _raw_or_address(execution_precommit_digest, "execution precommit digest")
        base = (
            Path(journal_root)
            / f"rank_{rubric_spec.candidate_rank}"
            / f"ordinal_{panel.ordinal:03d}"
        )
        self.panel = panel
        self.rubric_spec = rubric_spec
        self.runtime = runtime
        self.failure_evidence_path = base / "failure_evidence.json"

        def diagnostic_transport(*args: Any, **kwargs: Any) -> CodexStructuredResult:
            try:
                return underlying_transport(*args, **kwargs)
            except Exception as exc:
                evidence = _seal_failure_evidence(panel, rubric_spec, exc)
                _write_once_json(self.failure_evidence_path, evidence.to_data())
                raise

        self.journal = ObjectBongardNamedImageTurnJournalTransport(
            base / "turn",
            authorization_digest=authorization_digest,
            execution_precommit_digest=execution_precommit_digest,
            task_id=panel.task_id,
            turn_kind=(
                f"panel_rubric_rank_{rubric_spec.candidate_rank}_"
                f"ordinal_{panel.ordinal:03d}"
            ),
            expected_prompt=object_bongard_panel_rubric_prompt(rubric_spec),
            expected_images=(("panel.png", panel.exact_png_bytes),),
            expected_output_schema=object_bongard_panel_rubric_output_schema(),
            runtime=runtime,
            underlying_transport=diagnostic_transport,
        )

    @property
    def fresh_call_count(self) -> int:
        return self.journal.fresh_call_count

    @property
    def reused_call_count(self) -> int:
        return self.journal.reused_call_count

    def __call__(
        self,
        task: str,
        image_png_paths: Sequence[str],
        image_names: Sequence[str],
        output_schema: Mapping[str, Any],
        **kwargs: Any,
    ) -> CodexStructuredResult:
        return self.journal(
            task, image_png_paths, image_names, output_schema, **kwargs
        )

    def verify(
        self,
    ) -> tuple[ObjectBongardTurnJournalSummary, ObjectBongardPanelRubricFailureEvidence | None]:
        summary = verify_object_bongard_turn_journal(self.journal)
        evidence: ObjectBongardPanelRubricFailureEvidence | None = None
        if self.failure_evidence_path.exists():
            try:
                raw = json.loads(self.failure_evidence_path.read_text("utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise ObjectBongardPanelRubricCalibrationError(
                    "failure evidence record cannot be replayed"
                ) from exc
            evidence = ObjectBongardPanelRubricFailureEvidence.from_data(raw)
        if summary.terminal_status == "failure" and evidence is None:
            result_path = self.journal.directory / "result.json"
            try:
                result = json.loads(result_path.read_text("utf-8"))
                source_type = result["source_exception_type"]
            except Exception:
                source_type = "builtins.Exception"
            evidence = _fallback_failure_evidence(
                self.panel, self.rubric_spec, source_type
            )
            _write_once_json(self.failure_evidence_path, evidence.to_data())
        if (summary.terminal_status == "success") != (evidence is None):
            raise ObjectBongardPanelRubricCalibrationError(
                "journal status and failure evidence differ"
            )
        return summary, evidence


def _live_run_content(
    value: "ObjectBongardPanelRubricCalibrationLiveRun",
) -> dict[str, object]:
    return {
        "schema": PANEL_RUBRIC_CALIBRATION_LIVE_RUN_SCHEMA,
        "plan_digest": value.plan_digest,
        "panel_binding_digest": value.panel_binding_digest,
        "rubric_spec_digest": value.rubric_spec_digest,
        "observer_artifact": value.artifact.to_data(),
        "journal_summary": value.journal_summary.to_data(),
        "failure_evidence": (
            None if value.failure_evidence is None else value.failure_evidence.to_data()
        ),
        "fresh_call_count": value.fresh_call_count,
        "reused_call_count": value.reused_call_count,
        "support_side_visible_to_observer": False,
        "one_complete_panel_call": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardPanelRubricCalibrationLiveRun:
    plan_digest: str
    panel_binding_digest: str
    rubric_spec_digest: str
    artifact: ObjectBongardPanelRubricArtifact
    journal_summary: ObjectBongardTurnJournalSummary
    failure_evidence: ObjectBongardPanelRubricFailureEvidence | None
    fresh_call_count: int
    reused_call_count: int
    run_digest: str

    def __post_init__(self) -> None:
        for name in (
            "plan_digest",
            "panel_binding_digest",
            "rubric_spec_digest",
            "run_digest",
        ):
            _raw_digest(getattr(self, name), name)
        if (
            not isinstance(self.artifact, ObjectBongardPanelRubricArtifact)
            or self.artifact.rubric_spec_digest != self.rubric_spec_digest
            or not isinstance(self.journal_summary, ObjectBongardTurnJournalSummary)
            or self.journal_summary.terminal_status not in {"success", "failure"}
            or isinstance(self.fresh_call_count, bool)
            or not isinstance(self.fresh_call_count, int)
            or isinstance(self.reused_call_count, bool)
            or not isinstance(self.reused_call_count, int)
            or min(self.fresh_call_count, self.reused_call_count) < 0
            or self.fresh_call_count + self.reused_call_count != 1
        ):
            raise ObjectBongardPanelRubricCalibrationError(
                "live calibration run binding differs"
            )
        if self.journal_summary.terminal_status == "failure":
            if (
                not isinstance(
                    self.failure_evidence,
                    ObjectBongardPanelRubricFailureEvidence,
                )
                or self.failure_evidence.panel_binding_digest
                != self.panel_binding_digest
                or self.failure_evidence.rubric_spec_digest
                != self.rubric_spec_digest
                or self.artifact.observation.disposition is not Disposition.ERROR
            ):
                raise ObjectBongardPanelRubricCalibrationError(
                    "failed run lacks bounded diagnostic evidence"
                )
        elif self.failure_evidence is not None:
            raise ObjectBongardPanelRubricCalibrationError(
                "successful journal contains failure evidence"
            )
        if self.run_digest != canonical_digest(_live_run_content(self)):
            raise ObjectBongardPanelRubricCalibrationError(
                "live calibration run digest differs"
            )

    def to_data(self) -> dict[str, object]:
        return {**_live_run_content(self), "run_digest": self.run_digest}

    @classmethod
    def from_data(
        cls, value: object
    ) -> "ObjectBongardPanelRubricCalibrationLiveRun":
        raw = _fields(
            value,
            {
                "schema",
                "plan_digest",
                "panel_binding_digest",
                "rubric_spec_digest",
                "observer_artifact",
                "journal_summary",
                "failure_evidence",
                "fresh_call_count",
                "reused_call_count",
                "support_side_visible_to_observer",
                "one_complete_panel_call",
                *_authority_data(),
                "run_digest",
            },
            "live calibration run",
        )
        if (
            raw["schema"] != PANEL_RUBRIC_CALIBRATION_LIVE_RUN_SCHEMA
            or raw["support_side_visible_to_observer"] is not False
            or raw["one_complete_panel_call"] is not True
            or any(raw[key] != item for key, item in _authority_data().items())
        ):
            raise ObjectBongardPanelRubricCalibrationError(
                "live calibration run policy differs"
            )
        result = cls(
            raw["plan_digest"],
            raw["panel_binding_digest"],
            raw["rubric_spec_digest"],
            ObjectBongardPanelRubricArtifact.from_data(raw["observer_artifact"]),
            _journal_summary_from_data(raw["journal_summary"]),
            (
                None
                if raw["failure_evidence"] is None
                else ObjectBongardPanelRubricFailureEvidence.from_data(
                    raw["failure_evidence"]
                )
            ),
            raw["fresh_call_count"],
            raw["reused_call_count"],
            raw["run_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardPanelRubricCalibrationError(
                "live calibration run is not canonical"
            )
        return result


PanelRubricJournalDispatcherFactory = Callable[
    ..., ObjectBongardPanelRubricJournalDispatcher
]


def run_object_bongard_panel_rubric_calibration_observation(
    plan: ObjectBongardPanelRubricCalibrationPlan,
    panel: ObjectBongardPanelRubricCalibrationPanel,
    rubric_spec: ObjectBongardRubricSpec,
    *,
    runtime: ObjectBongardTurnRuntime,
    journal_root: str | os.PathLike[str],
    authorization_digest: str,
    execution_precommit_digest: str,
    underlying_transport: Callable[..., CodexStructuredResult] = (
        run_codex_named_images_structured
    ),
    dispatcher_factory: PanelRubricJournalDispatcherFactory = (
        ObjectBongardPanelRubricJournalDispatcher
    ),
) -> ObjectBongardPanelRubricCalibrationLiveRun:
    """Run one blind, one-image calibration turn through its own journal."""

    if not isinstance(plan, ObjectBongardPanelRubricCalibrationPlan):
        raise TypeError("plan must be typed")
    if panel not in plan.source.panels or rubric_spec not in plan.rubric_specs:
        raise ObjectBongardPanelRubricCalibrationError(
            "calibration job lies outside the frozen 24-job inventory"
        )
    if not isinstance(runtime, ObjectBongardTurnRuntime):
        raise TypeError("runtime must be typed")
    context = "sha256:" + canonical_digest(
        {
            "schema": "gkm.bongard-panel-rubric-calibration-context.v1",
            "plan_digest": plan.plan_digest,
            "panel_binding_digest": panel.panel_binding_digest,
            "rubric_spec_digest": rubric_spec.spec_digest,
            "authorization_digest": _raw_or_address(
                authorization_digest, "authorization digest"
            ),
            "execution_precommit_digest": _raw_or_address(
                execution_precommit_digest, "execution precommit digest"
            ),
            "support_side_visible_to_observer": False,
        }
    )
    dispatcher = dispatcher_factory(
        journal_root,
        panel=panel,
        rubric_spec=rubric_spec,
        authorization_digest=authorization_digest,
        execution_precommit_digest=execution_precommit_digest,
        runtime=runtime,
        underlying_transport=underlying_transport,
    )
    if not isinstance(dispatcher, ObjectBongardPanelRubricJournalDispatcher):
        raise TypeError("dispatcher factory returned the wrong type")
    artifact = observe_object_bongard_panel_rubric(
        panel.exact_png_bytes,
        panel_id=panel.panel_id,
        rubric_spec=rubric_spec,
        expected_panel_sha256=panel.png_sha256,
        expected_rubric_spec_digest=rubric_spec.spec_digest,
        model=runtime.model,
        reasoning_effort=runtime.reasoning_effort,
        minutes=runtime.minutes,
        verbose=runtime.verbose,
        executable=runtime.executable,
        cloud_policy_cache_snapshot=runtime.cloud_policy_cache_snapshot,
        expected_launcher_digest=runtime.expected_launcher_digest,
        model_catalog_snapshot=runtime.model_catalog_snapshot,
        no_tools_attestation=runtime.no_tools_attestation,
        transport=dispatcher,
        observation_context_digest=context,
    )
    if ObjectBongardPanelRubricArtifact.from_data(artifact.to_data()) != artifact:
        raise ObjectBongardPanelRubricCalibrationError(
            "live panel artifact canonical reload differs"
        )
    summary, failure_evidence = dispatcher.verify()
    values = {
        "plan_digest": plan.plan_digest,
        "panel_binding_digest": panel.panel_binding_digest,
        "rubric_spec_digest": rubric_spec.spec_digest,
        "artifact": artifact,
        "journal_summary": summary,
        "failure_evidence": failure_evidence,
        "fresh_call_count": dispatcher.fresh_call_count,
        "reused_call_count": dispatcher.reused_call_count,
    }
    provisional = object.__new__(ObjectBongardPanelRubricCalibrationLiveRun)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectBongardPanelRubricCalibrationLiveRun(
        **values,
        run_digest=canonical_digest(_live_run_content(provisional)),
    )


def _batch_content(
    value: "ObjectBongardPanelRubricCalibrationObservationBatch",
) -> dict[str, object]:
    return {
        "schema": PANEL_RUBRIC_CALIBRATION_BATCH_SCHEMA,
        "algorithm_id": PANEL_RUBRIC_CALIBRATION_ALGORITHM_ID,
        "plan_digest": value.plan_digest,
        "source_digest": value.source_digest,
        "authorization_digest": value.authorization_digest,
        "execution_precommit_digest": value.execution_precommit_digest,
        "runtime_identity_digest": value.runtime_identity_digest,
        "parallel_workers": value.parallel_workers,
        "runs": [item.to_data() for item in value.runs],
        "run_order": "candidate-rank-then-source-ordinal",
        "expected_run_count": PANEL_RUBRIC_CALIBRATION_JOB_COUNT,
        "all_artifacts_exist_before_support_labels": True,
        "support_labels_present_in_run_records": False,
        "selection_performed": False,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardPanelRubricCalibrationObservationBatch:
    plan_digest: str
    source_digest: str
    authorization_digest: str
    execution_precommit_digest: str
    runtime_identity_digest: str
    parallel_workers: int
    runs: tuple[ObjectBongardPanelRubricCalibrationLiveRun, ...]
    batch_digest: str

    def __post_init__(self) -> None:
        for name in (
            "plan_digest",
            "source_digest",
            "runtime_identity_digest",
            "batch_digest",
        ):
            _raw_digest(getattr(self, name), name)
        _raw_or_address(self.authorization_digest, "batch authorization digest")
        _raw_or_address(
            self.execution_precommit_digest, "batch execution precommit digest"
        )
        blocks = (
            tuple(
                self.runs[index : index + PANEL_RUBRIC_CALIBRATION_PANEL_COUNT]
                for index in range(0, len(self.runs), PANEL_RUBRIC_CALIBRATION_PANEL_COUNT)
            )
            if isinstance(self.runs, tuple)
            else ()
        )
        if (
            isinstance(self.parallel_workers, bool)
            or not isinstance(self.parallel_workers, int)
            or not 1 <= self.parallel_workers <= 32
            or not isinstance(self.runs, tuple)
            or len(self.runs) != PANEL_RUBRIC_CALIBRATION_JOB_COUNT
            or any(
                not isinstance(item, ObjectBongardPanelRubricCalibrationLiveRun)
                for item in self.runs
            )
            or len(
                {
                    (item.panel_binding_digest, item.rubric_spec_digest)
                    for item in self.runs
                }
            )
            != PANEL_RUBRIC_CALIBRATION_JOB_COUNT
            or len({item.panel_binding_digest for item in self.runs})
            != PANEL_RUBRIC_CALIBRATION_PANEL_COUNT
            or len(blocks) != PANEL_RUBRIC_CALIBRATION_SPEC_COUNT
            or any(
                len({item.rubric_spec_digest for item in block}) != 1
                for block in blocks
            )
            or len({block[0].rubric_spec_digest for block in blocks}) != 2
            or any(
                item.artifact.runtime_identity_digest
                != self.runtime_identity_digest
                for item in self.runs
            )
            or self.batch_digest != canonical_digest(_batch_content(self))
        ):
            raise ObjectBongardPanelRubricCalibrationError(
                "24-run calibration observation batch differs"
            )

    @property
    def fresh_call_count(self) -> int:
        return sum(item.fresh_call_count for item in self.runs)

    @property
    def reused_call_count(self) -> int:
        return sum(item.reused_call_count for item in self.runs)

    def artifacts_by_spec_digest(
        self,
    ) -> dict[str, tuple[ObjectBongardPanelRubricArtifact, ...]]:
        result: dict[str, list[ObjectBongardPanelRubricArtifact]] = {}
        for run in self.runs:
            result.setdefault(run.rubric_spec_digest, []).append(run.artifact)
        return {key: tuple(items) for key, items in result.items()}

    def to_data(self) -> dict[str, object]:
        return {**_batch_content(self), "batch_digest": self.batch_digest}

    @classmethod
    def from_data(
        cls, value: object
    ) -> "ObjectBongardPanelRubricCalibrationObservationBatch":
        raw = _fields(
            value,
            {
                "schema",
                "algorithm_id",
                "plan_digest",
                "source_digest",
                "authorization_digest",
                "execution_precommit_digest",
                "runtime_identity_digest",
                "parallel_workers",
                "runs",
                "run_order",
                "expected_run_count",
                "all_artifacts_exist_before_support_labels",
                "support_labels_present_in_run_records",
                "selection_performed",
                *_authority_data(),
                "batch_digest",
            },
            "panel calibration batch",
        )
        if (
            raw["schema"] != PANEL_RUBRIC_CALIBRATION_BATCH_SCHEMA
            or raw["algorithm_id"] != PANEL_RUBRIC_CALIBRATION_ALGORITHM_ID
            or raw["run_order"] != "candidate-rank-then-source-ordinal"
            or raw["expected_run_count"] != PANEL_RUBRIC_CALIBRATION_JOB_COUNT
            or raw["all_artifacts_exist_before_support_labels"] is not True
            or raw["support_labels_present_in_run_records"] is not False
            or raw["selection_performed"] is not False
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["runs"], list)
        ):
            raise ObjectBongardPanelRubricCalibrationError(
                "panel calibration batch policy differs"
            )
        result = cls(
            raw["plan_digest"],
            raw["source_digest"],
            raw["authorization_digest"],
            raw["execution_precommit_digest"],
            raw["runtime_identity_digest"],
            raw["parallel_workers"],
            tuple(
                ObjectBongardPanelRubricCalibrationLiveRun.from_data(item)
                for item in raw["runs"]
            ),
            raw["batch_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardPanelRubricCalibrationError(
                "panel calibration batch is not canonical"
            )
        return result


def run_object_bongard_panel_rubric_calibration_observations(
    plan: ObjectBongardPanelRubricCalibrationPlan,
    *,
    runtime: ObjectBongardTurnRuntime,
    journal_root: str | os.PathLike[str],
    authorization_digest: str,
    execution_precommit_digest: str,
    parallel_workers: int = 4,
    underlying_transport: Callable[..., CodexStructuredResult] = (
        run_codex_named_images_structured
    ),
    dispatcher_factory: PanelRubricJournalDispatcherFactory = (
        ObjectBongardPanelRubricJournalDispatcher
    ),
) -> ObjectBongardPanelRubricCalibrationObservationBatch:
    """Complete all 24 blind jobs before returning any label-capable object."""

    if not isinstance(plan, ObjectBongardPanelRubricCalibrationPlan):
        raise TypeError("plan must be typed")
    if (
        isinstance(parallel_workers, bool)
        or not isinstance(parallel_workers, int)
        or not 1 <= parallel_workers <= 32
    ):
        raise ObjectBongardPanelRubricCalibrationError(
            "parallel_workers must lie in 1..32"
        )
    jobs = tuple(
        (panel, spec)
        for spec in plan.rubric_specs
        for panel in plan.source.panels
    )
    if len(jobs) != PANEL_RUBRIC_CALIBRATION_JOB_COUNT:
        raise ObjectBongardPanelRubricCalibrationError(
            "frozen calibration job count differs"
        )

    def run_job(
        job: tuple[
            ObjectBongardPanelRubricCalibrationPanel, ObjectBongardRubricSpec
        ],
    ) -> ObjectBongardPanelRubricCalibrationLiveRun:
        panel, spec = job
        return run_object_bongard_panel_rubric_calibration_observation(
            plan,
            panel,
            spec,
            runtime=runtime,
            journal_root=journal_root,
            authorization_digest=authorization_digest,
            execution_precommit_digest=execution_precommit_digest,
            underlying_transport=underlying_transport,
            dispatcher_factory=dispatcher_factory,
        )

    with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        runs = tuple(executor.map(run_job, jobs))
    values = {
        "plan_digest": plan.plan_digest,
        "source_digest": plan.source.source_digest,
        "authorization_digest": authorization_digest,
        "execution_precommit_digest": execution_precommit_digest,
        "runtime_identity_digest": runs[0].artifact.runtime_identity_digest,
        "parallel_workers": parallel_workers,
        "runs": runs,
    }
    provisional = object.__new__(
        ObjectBongardPanelRubricCalibrationObservationBatch
    )
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectBongardPanelRubricCalibrationObservationBatch(
        **values,
        batch_digest=canonical_digest(_batch_content(provisional)),
    )


def _freeze_content(
    value: "ObjectBongardPanelRubricCalibrationDurableFreeze",
) -> dict[str, object]:
    return {
        "schema": PANEL_RUBRIC_CALIBRATION_FREEZE_SCHEMA,
        "algorithm_id": PANEL_RUBRIC_CALIBRATION_ALGORITHM_ID,
        "batch": value.batch.to_data(),
        "batch_file_sha256": value.batch_file_sha256,
        "batch_file_byte_count": value.batch_file_byte_count,
        "exact_canonical_bytes_fsynced": True,
        "exact_canonical_bytes_reloaded": True,
        "all_24_artifacts_frozen_before_support_labels": True,
        "selection_performed_before_freeze": False,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardPanelRubricCalibrationDurableFreeze:
    batch: ObjectBongardPanelRubricCalibrationObservationBatch
    batch_file_sha256: str
    batch_file_byte_count: int
    freeze_digest: str

    def __post_init__(self) -> None:
        if not isinstance(
            self.batch, ObjectBongardPanelRubricCalibrationObservationBatch
        ):
            raise TypeError("freeze batch must be typed")
        _raw_digest(self.batch_file_sha256, "frozen batch file digest")
        if (
            isinstance(self.batch_file_byte_count, bool)
            or not isinstance(self.batch_file_byte_count, int)
            or self.batch_file_byte_count <= 0
            or self.batch_file_byte_count
            != len(canonical_json(self.batch.to_data()) + b"\n")
            or self.batch_file_sha256
            != hashlib.sha256(canonical_json(self.batch.to_data()) + b"\n").hexdigest()
        ):
            raise ObjectBongardPanelRubricCalibrationError(
                "durable batch byte commitment differs"
            )
        _address(self.freeze_digest, "durable freeze digest")
        if self.freeze_digest != "sha256:" + canonical_digest(
            _freeze_content(self)
        ):
            raise ObjectBongardPanelRubricCalibrationError(
                "durable freeze digest differs"
            )

    def to_data(self) -> dict[str, object]:
        return {**_freeze_content(self), "freeze_digest": self.freeze_digest}

    @classmethod
    def from_data(
        cls, value: object
    ) -> "ObjectBongardPanelRubricCalibrationDurableFreeze":
        raw = _fields(
            value,
            {
                "schema",
                "algorithm_id",
                "batch",
                "batch_file_sha256",
                "batch_file_byte_count",
                "exact_canonical_bytes_fsynced",
                "exact_canonical_bytes_reloaded",
                "all_24_artifacts_frozen_before_support_labels",
                "selection_performed_before_freeze",
                *_authority_data(),
                "freeze_digest",
            },
            "durable panel calibration freeze",
        )
        if (
            raw["schema"] != PANEL_RUBRIC_CALIBRATION_FREEZE_SCHEMA
            or raw["algorithm_id"] != PANEL_RUBRIC_CALIBRATION_ALGORITHM_ID
            or raw["exact_canonical_bytes_fsynced"] is not True
            or raw["exact_canonical_bytes_reloaded"] is not True
            or raw["all_24_artifacts_frozen_before_support_labels"] is not True
            or raw["selection_performed_before_freeze"] is not False
            or any(raw[key] != item for key, item in _authority_data().items())
        ):
            raise ObjectBongardPanelRubricCalibrationError(
                "durable panel calibration freeze policy differs"
            )
        result = cls(
            ObjectBongardPanelRubricCalibrationObservationBatch.from_data(
                raw["batch"]
            ),
            raw["batch_file_sha256"],
            raw["batch_file_byte_count"],
            raw["freeze_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardPanelRubricCalibrationError(
                "durable panel calibration freeze is not canonical"
            )
        return result


def persist_and_reload_object_bongard_panel_rubric_calibration_batch(
    batch: ObjectBongardPanelRubricCalibrationObservationBatch,
    path: str | os.PathLike[str],
) -> ObjectBongardPanelRubricCalibrationDurableFreeze:
    """Fsync and exact-reload the full 24-artifact batch before assessment."""

    if not isinstance(batch, ObjectBongardPanelRubricCalibrationObservationBatch):
        raise TypeError("batch must be typed")
    target = Path(path)
    _write_once_json(target, batch.to_data())
    try:
        info = target.lstat()
        payload = target.read_bytes()
    except OSError as exc:
        raise ObjectBongardPanelRubricCalibrationError(
            "durable batch cannot be reloaded"
        ) from exc
    expected = canonical_json(batch.to_data()) + b"\n"
    if (
        not stat.S_ISREG(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or payload != expected
    ):
        raise ObjectBongardPanelRubricCalibrationError(
            "durable batch bytes differ after reload"
        )
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ObjectBongardPanelRubricCalibrationError(
            "durable batch is malformed"
        ) from exc
    restored = ObjectBongardPanelRubricCalibrationObservationBatch.from_data(
        decoded
    )
    if restored != batch:
        raise ObjectBongardPanelRubricCalibrationError(
            "durable batch typed reload differs"
        )
    values = {
        "batch": restored,
        "batch_file_sha256": hashlib.sha256(payload).hexdigest(),
        "batch_file_byte_count": len(payload),
    }
    provisional = object.__new__(
        ObjectBongardPanelRubricCalibrationDurableFreeze
    )
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectBongardPanelRubricCalibrationDurableFreeze(
        **values,
        freeze_digest="sha256:" + canonical_digest(_freeze_content(provisional)),
    )


def _disposition_counts(values: Sequence[Disposition]) -> dict[str, int]:
    frozen = tuple(values)
    if len(frozen) != PANEL_RUBRIC_SUPPORT_PANELS_PER_SIDE:
        raise ObjectBongardPanelRubricCalibrationError(
            "disposition counts require one exact six-panel side"
        )
    return {item.value: frozen.count(item) for item in Disposition}


def _assessment_content(
    value: "ObjectBongardPanelRubricCalibrationAssessment",
) -> dict[str, object]:
    support_counts = []
    for rank, space in enumerate(value.version_spaces):
        support_counts.append(
            {
                "candidate_rank": rank,
                "target_side": _disposition_counts(space.row[:6]),
                "foil_side": _disposition_counts(space.row[6:]),
                "bounded_admissible": bool(space.survivor_candidate_digests),
                "strict_exact_support": bool(
                    space.strict_survivor_candidate_digests
                ),
                "support_acceptance_tier": space.support_acceptance_tier.value,
            }
        )
    return {
        "schema": PANEL_RUBRIC_CALIBRATION_ASSESSMENT_SCHEMA,
        "algorithm_id": PANEL_RUBRIC_CALIBRATION_ALGORITHM_ID,
        "implementation_source_sha256": (
            object_bongard_panel_rubric_calibration_source_digest()
        ),
        "plan_digest": value.plan_digest,
        "source_digest": value.source_digest,
        "frozen_batch_digest": value.frozen_batch_digest,
        "freeze_digest": value.freeze_digest,
        "runtime_identity_digest": value.runtime_identity_digest,
        "version_spaces": [item.to_data() for item in value.version_spaces],
        "support_disposition_counts": support_counts,
        "slate_selection": value.slate_selection.to_data(),
        "accepted": value.slate_selection.selected_candidate_digest is not None,
        "selected_candidate_rank": (
            None
            if value.slate_selection.selected_rubric_spec is None
            else value.slate_selection.selected_rubric_spec.candidate_rank
        ),
        "selection_rule": "rank-zero-if-bounded-admissible-else-rank-one",
        "strict_six_plus_six_changes_selection": False,
        "any_contradiction_or_error_rejects_candidate": True,
        "maximum_indeterminate_per_side": 1,
        "minimum_expected_definite_per_side": 5,
        "certified_absent_observation_meaning": "foil_preferred",
        "literal_visual_cue_absence_claimed": False,
        "labels_introduced_after_all_24_artifacts_and_durable_reload": True,
        "model_calls_during_assessment": 0,
        "ranker_used": False,
        "query_pixels_used": False,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardPanelRubricCalibrationAssessment:
    plan_digest: str
    source_digest: str
    frozen_batch_digest: str
    freeze_digest: str
    runtime_identity_digest: str
    version_spaces: tuple[
        ObjectBongardPanelRubricSupportVersionSpace,
        ObjectBongardPanelRubricSupportVersionSpace,
    ]
    slate_selection: ObjectBongardPanelRubricSlateSelection
    assessment_digest: str

    def __post_init__(self) -> None:
        for name in (
            "plan_digest",
            "source_digest",
            "frozen_batch_digest",
            "runtime_identity_digest",
            "assessment_digest",
        ):
            _raw_digest(getattr(self, name), name)
        _address(self.freeze_digest, "assessment freeze digest")
        if (
            not isinstance(self.version_spaces, tuple)
            or len(self.version_spaces) != 2
            or any(
                not isinstance(item, ObjectBongardPanelRubricSupportVersionSpace)
                for item in self.version_spaces
            )
            or not isinstance(
                self.slate_selection, ObjectBongardPanelRubricSlateSelection
            )
        ):
            raise ObjectBongardPanelRubricCalibrationError(
                "calibration assessment inventory differs"
            )
        expected = select_object_bongard_panel_rubric_slate(
            self.slate_selection.rubric_specs, self.version_spaces
        )
        if self.slate_selection != expected:
            raise ObjectBongardPanelRubricCalibrationError(
                "calibration assessment slate selection differs"
            )
        if self.assessment_digest != canonical_digest(_assessment_content(self)):
            raise ObjectBongardPanelRubricCalibrationError(
                "calibration assessment digest differs"
            )

    def to_data(self) -> dict[str, object]:
        return {
            **_assessment_content(self),
            "assessment_digest": self.assessment_digest,
        }

    @classmethod
    def from_data(
        cls, value: object
    ) -> "ObjectBongardPanelRubricCalibrationAssessment":
        raw = _fields(
            value,
            {
                "schema",
                "algorithm_id",
                "implementation_source_sha256",
                "plan_digest",
                "source_digest",
                "frozen_batch_digest",
                "freeze_digest",
                "runtime_identity_digest",
                "version_spaces",
                "support_disposition_counts",
                "slate_selection",
                "accepted",
                "selected_candidate_rank",
                "selection_rule",
                "strict_six_plus_six_changes_selection",
                "any_contradiction_or_error_rejects_candidate",
                "maximum_indeterminate_per_side",
                "minimum_expected_definite_per_side",
                "certified_absent_observation_meaning",
                "literal_visual_cue_absence_claimed",
                "labels_introduced_after_all_24_artifacts_and_durable_reload",
                "model_calls_during_assessment",
                "ranker_used",
                "query_pixels_used",
                *_authority_data(),
                "assessment_digest",
            },
            "panel calibration assessment",
        )
        if (
            raw["schema"] != PANEL_RUBRIC_CALIBRATION_ASSESSMENT_SCHEMA
            or raw["algorithm_id"] != PANEL_RUBRIC_CALIBRATION_ALGORITHM_ID
            or raw["implementation_source_sha256"]
            != object_bongard_panel_rubric_calibration_source_digest()
            or raw["selection_rule"]
            != "rank-zero-if-bounded-admissible-else-rank-one"
            or raw["strict_six_plus_six_changes_selection"] is not False
            or raw["any_contradiction_or_error_rejects_candidate"] is not True
            or raw["maximum_indeterminate_per_side"] != 1
            or raw["minimum_expected_definite_per_side"] != 5
            or raw["certified_absent_observation_meaning"] != "foil_preferred"
            or raw["literal_visual_cue_absence_claimed"] is not False
            or raw["labels_introduced_after_all_24_artifacts_and_durable_reload"]
            is not True
            or raw["model_calls_during_assessment"] != 0
            or raw["ranker_used"] is not False
            or raw["query_pixels_used"] is not False
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["version_spaces"], list)
            or not isinstance(raw["support_disposition_counts"], list)
        ):
            raise ObjectBongardPanelRubricCalibrationError(
                "panel calibration assessment policy differs"
            )
        spaces = tuple(
            ObjectBongardPanelRubricSupportVersionSpace.from_data(item)
            for item in raw["version_spaces"]
        )
        selection = ObjectBongardPanelRubricSlateSelection.from_data(
            raw["slate_selection"]
        )
        result = cls(
            raw["plan_digest"],
            raw["source_digest"],
            raw["frozen_batch_digest"],
            raw["freeze_digest"],
            raw["runtime_identity_digest"],
            spaces,  # type: ignore[arg-type]
            selection,
            raw["assessment_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardPanelRubricCalibrationError(
                "panel calibration assessment is not canonical"
            )
        return result


def _validate_frozen_batch_against_plan(
    plan: ObjectBongardPanelRubricCalibrationPlan,
    frozen: ObjectBongardPanelRubricCalibrationDurableFreeze,
) -> None:
    batch = frozen.batch
    expected = tuple(
        (panel.panel_binding_digest, spec.spec_digest)
        for spec in plan.rubric_specs
        for panel in plan.source.panels
    )
    actual = tuple(
        (item.panel_binding_digest, item.rubric_spec_digest)
        for item in batch.runs
    )
    if (
        batch.plan_digest != plan.plan_digest
        or batch.source_digest != plan.source.source_digest
        or actual != expected
    ):
        raise ObjectBongardPanelRubricCalibrationError(
            "frozen 24-run batch differs from the calibration plan"
        )
    for run in batch.runs:
        panel = next(
            item
            for item in plan.source.panels
            if item.panel_binding_digest == run.panel_binding_digest
        )
        spec = next(
            item
            for item in plan.rubric_specs
            if item.spec_digest == run.rubric_spec_digest
        )
        expected_context = "sha256:" + canonical_digest(
            {
                "schema": "gkm.bongard-panel-rubric-calibration-context.v1",
                "plan_digest": plan.plan_digest,
                "panel_binding_digest": panel.panel_binding_digest,
                "rubric_spec_digest": spec.spec_digest,
                "authorization_digest": batch.authorization_digest,
                "execution_precommit_digest": batch.execution_precommit_digest,
                "support_side_visible_to_observer": False,
            }
        )
        if (
            run.artifact.panel_id != panel.panel_id
            or run.artifact.panel_digest != panel.png_sha256
            or run.artifact.rubric_spec != spec
            or run.artifact.observation_context_digest != expected_context
        ):
            raise ObjectBongardPanelRubricCalibrationError(
                "frozen artifact differs from its blind job commitment"
            )


def assess_object_bongard_panel_rubric_calibration(
    plan: ObjectBongardPanelRubricCalibrationPlan,
    frozen: ObjectBongardPanelRubricCalibrationDurableFreeze,
) -> ObjectBongardPanelRubricCalibrationAssessment:
    """Introduce support sides only after the exact batch freeze/reload."""

    if not isinstance(plan, ObjectBongardPanelRubricCalibrationPlan):
        raise TypeError("plan must be typed")
    if not isinstance(frozen, ObjectBongardPanelRubricCalibrationDurableFreeze):
        raise TypeError("assessment requires a durable freeze, not a live batch")
    _validate_frozen_batch_against_plan(plan, frozen)
    by_spec = frozen.batch.artifacts_by_spec_digest()
    spaces: list[ObjectBongardPanelRubricSupportVersionSpace] = []
    for spec in plan.rubric_specs:
        artifacts = by_spec[spec.spec_digest]
        by_panel = {item.panel_id: item for item in artifacts}
        spaces.append(
            build_object_bongard_panel_rubric_support_version_space(
                spec,
                tuple(by_panel[item.panel_id] for item in plan.source.group_0_panels),
                tuple(by_panel[item.panel_id] for item in plan.source.group_1_panels),
            )
        )
    version_spaces = tuple(spaces)
    selection = select_object_bongard_panel_rubric_slate(
        plan.rubric_specs, version_spaces
    )
    values = {
        "plan_digest": plan.plan_digest,
        "source_digest": plan.source.source_digest,
        "frozen_batch_digest": frozen.batch.batch_digest,
        "freeze_digest": frozen.freeze_digest,
        "runtime_identity_digest": frozen.batch.runtime_identity_digest,
        "version_spaces": version_spaces,
        "slate_selection": selection,
    }
    provisional = object.__new__(ObjectBongardPanelRubricCalibrationAssessment)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectBongardPanelRubricCalibrationAssessment(
        **values,  # type: ignore[arg-type]
        assessment_digest=canonical_digest(_assessment_content(provisional)),
    )


def cold_verify_object_bongard_panel_rubric_calibration(
    assessment: ObjectBongardPanelRubricCalibrationAssessment,
    plan: ObjectBongardPanelRubricCalibrationPlan,
    frozen: ObjectBongardPanelRubricCalibrationDurableFreeze,
) -> ObjectBongardPanelRubricCalibrationAssessment:
    """Replay exact pixels, receipts, projection, support, and selection offline."""

    if not isinstance(assessment, ObjectBongardPanelRubricCalibrationAssessment):
        raise TypeError("assessment must be typed")
    decoded_assessment = ObjectBongardPanelRubricCalibrationAssessment.from_data(
        assessment.to_data()
    )
    decoded_freeze = ObjectBongardPanelRubricCalibrationDurableFreeze.from_data(
        frozen.to_data()
    )
    _validate_frozen_batch_against_plan(plan, decoded_freeze)
    for run in decoded_freeze.batch.runs:
        panel = next(
            item
            for item in plan.source.panels
            if item.panel_binding_digest == run.panel_binding_digest
        )
        spec = next(
            item
            for item in plan.rubric_specs
            if item.spec_digest == run.rubric_spec_digest
        )
        verify_object_bongard_panel_rubric_artifact(
            run.artifact,
            panel.exact_png_bytes,
            panel_id=panel.panel_id,
            rubric_spec=spec,
            expected_artifact_digest=run.artifact.artifact_digest,
            expected_runtime_identity_digest=(
                decoded_freeze.batch.runtime_identity_digest
            ),
        )
    replayed = assess_object_bongard_panel_rubric_calibration(
        plan, decoded_freeze
    )
    if decoded_assessment != replayed:
        raise ObjectBongardPanelRubricCalibrationError(
            "cold panel calibration replay differs"
        )
    return decoded_assessment


__all__ = (
    "DEFAULT_OBJECT_BONGARD_PANEL_RUBRIC_CALIBRATION_SOURCE",
    "ObjectBongardPanelRubricCalibrationAssessment",
    "ObjectBongardPanelRubricCalibrationDurableFreeze",
    "ObjectBongardPanelRubricCalibrationError",
    "ObjectBongardPanelRubricCalibrationLiveRun",
    "ObjectBongardPanelRubricCalibrationObservationBatch",
    "ObjectBongardPanelRubricCalibrationPanel",
    "ObjectBongardPanelRubricCalibrationPlan",
    "ObjectBongardPanelRubricCalibrationSource",
    "ObjectBongardPanelRubricFailureEvidence",
    "ObjectBongardPanelRubricJournalDispatcher",
    "PANEL_RUBRIC_CALIBRATION_ALGORITHM_ID",
    "PANEL_RUBRIC_CALIBRATION_GROUP_0_ORDINALS",
    "PANEL_RUBRIC_CALIBRATION_GROUP_1_ORDINALS",
    "PANEL_RUBRIC_CALIBRATION_JOB_COUNT",
    "PANEL_RUBRIC_CALIBRATION_SELECTED_ORDINALS",
    "assess_object_bongard_panel_rubric_calibration",
    "bind_object_bongard_panel_rubric_calibration_nomination",
    "cold_verify_object_bongard_panel_rubric_calibration",
    "load_object_bongard_panel_rubric_calibration_source",
    "object_bongard_panel_rubric_calibration_source_digest",
    "persist_and_reload_object_bongard_panel_rubric_calibration_batch",
    "run_object_bongard_panel_rubric_calibration_observation",
    "run_object_bongard_panel_rubric_calibration_observations",
)
