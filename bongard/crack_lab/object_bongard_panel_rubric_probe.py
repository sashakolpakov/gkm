"""Diagnostic-only whole-panel probe for the rejected v10 rubric calibration.

This command is deliberately *not* a calibration authorization.  It reuses the
authenticated runtime objects frozen by the rejected v10 calibration, but that
old authorization did not authorize these twelve new whole-panel calls.  The
command therefore writes ``diagnostic_unsealed`` into every aggregate record
and cannot open query, broad-cohort, or official-test pixels.

The model-visible boundary contains exactly ``panel.png`` and the frozen rank-0
rubric.  Group membership is consumed only after all twelve artifacts have
been persisted and cold-replayed without model access.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

import argparse
import base64
import binascii
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable, Mapping, Sequence

from bongard.canonical import canonical_digest, canonical_json
from bongard.object_bongard_panel_rubric_observer import (
    ObjectBongardPanelRubricArtifact,
    PanelRubricDisposition,
    object_bongard_panel_rubric_observer_source_digest,
    object_bongard_panel_rubric_output_schema,
    object_bongard_panel_rubric_prompt,
    object_bongard_panel_rubric_protocol_digest,
    observe_object_bongard_panel_rubric,
    verify_object_bongard_panel_rubric_artifact,
)
from bongard.object_bongard_rubric_calibration import (
    DEFAULT_OBJECT_RUBRIC_CALIBRATION_SOURCE,
)
from bongard.object_bongard_rubric_calibration_command import (
    ObjectBongardRubricCalibrationExecutionPrecommit,
)
from bongard.object_bongard_rubric_language import (
    ObjectBongardRubricSpec,
    object_bongard_rubric_language_source_digest,
)
from bongard.object_bongard_semantics import (
    ObjectBongardSemanticArtifact,
    verify_object_bongard_semantic_artifact,
)
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID
from bongard.transport import run_codex_named_images_structured


PROBE_MANIFEST_SCHEMA = "gkm.bongard-panel-rubric-probe-manifest.v1"
PROBE_REPLAY_SCHEMA = "gkm.bongard-panel-rubric-probe-artifact-replay.v1"
PROBE_RESULT_SCHEMA = "gkm.bongard-panel-rubric-probe-result.v1"
PROBE_STATUS = "diagnostic_unsealed"
PROBE_PANEL_COUNT = 12
PROBE_MAX_WORKERS = 4
MANIFEST_FILENAME = "manifest.json"
RESULT_FILENAME = "result.json"
ARTIFACT_DIRECTORY = "artifacts"
REPLAY_DIRECTORY = "replays"

DEFAULT_V10_NOMINATION_ROOT = Path(
    "downloads/ShapeBongard_V2_full/"
    "object_rubric_nomination_20260808_all_support_v10"
)
DEFAULT_REJECTED_V10_CALIBRATION_ROOT = Path(
    "downloads/ShapeBongard_V2_full/"
    "object_rubric_calibration_20260808_all_support_v10"
)

V10_NOMINATION_ARTIFACT_DIGEST = (
    "c765cdfaba7315ce04265e2151490a86f25d042347eac5cba8a7fc1282dc7c29"
)
V10_NOMINATION_AUTHORIZATION_DIGEST = (
    "sha256:65d2c58cb09bd3e7aeecde0093a50047ccb1676af105559758b589e5cdd368fe"
)
V10_NOMINATION_PRECOMMIT_DIGEST = (
    "sha256:caaa7aea85d3c35838c0abfbc052743f7fe05a7e52ff817c2a3a1c2e2ba992bd"
)
V10_NOMINATION_REPLAY_DIGEST = (
    "sha256:b1c20a920e12f4d2e85f42a3cee06d7565e308f52378e5edfb6bc4ee7c9ed6c4"
)
V10_NOMINATION_RESULT_DIGEST = (
    "sha256:2e0bcd7e0792641265806ccde66bac1af7f791746cf02051454f57ebf7fac4cf"
)
REJECTED_V10_CALIBRATION_AUTHORIZATION_DIGEST = (
    "sha256:096c890ec7b2a2fa7b943e2698527c4d30aa6246246085c433e17fd6f5be5cb5"
)
REJECTED_V10_CALIBRATION_PRECOMMIT_DIGEST = (
    "sha256:97c84bcc1542bf438f4c1ec0720047540432267c8446995076b9c78cfd318bc2"
)
V10_NOMINATION_SOURCE_DIGEST = (
    "78c0228d4326dc5e9335fd506e9dce23ec08d2ce4fef6d9a53653b8ab4cbefbe"
)
REJECTED_V10_CALIBRATION_SOURCE_DIGEST = (
    "dcc811caf38c29a25f9fb0bcd64a672872efa264ea610a0758eaafc96c7fccf6"
)
V10_CONTEXT_TASK_ID = "bd_two_mismatch_sectors8-thin_seven_lines2_0000"

# Exact canonical bytes of the five accepted nomination parents and the two
# rejected-calibration runtime parents.  These are file SHA-256 values, not the
# records' internal content addresses.
_V10_NOMINATION_PARENT_FILES = (
    ("authorization.json", "4ad41a8dd69b30661ea47394a3582cdacd90fcfd934d4755c6b7c2e277a6a586", "authorization_digest", V10_NOMINATION_AUTHORIZATION_DIGEST),
    ("execution_precommit.json", "26dea77e646ee138c2e96ff8bf87f0cf7919e76e4d1f3b9a4da81d305717691a", "precommit_digest", V10_NOMINATION_PRECOMMIT_DIGEST),
    ("semantic_artifact.json", "2906a76c6dd971301ce99671720476c0f34c62206f2dfaeaf1025cb03bf923c8", "artifact_digest", V10_NOMINATION_ARTIFACT_DIGEST),
    ("cold_replay.json", "07f2a56448ad4e8eabb4462d5b9602462407596bd30ed7ff3b942c478e4ff3d1", "replay_digest", V10_NOMINATION_REPLAY_DIGEST),
    ("result.json", "e3a3312eb30a71fc0e531937706edd86d54699f76850eefea064fd40d3f60bf5", "result_digest", V10_NOMINATION_RESULT_DIGEST),
)
_V10_CALIBRATION_PARENT_FILES = (
    ("authorization.json", "df0378173d1d3bf18a2b996858e9703138a6a6d5e17b79ad2c48e7b0ddd93398", "authorization_digest", REJECTED_V10_CALIBRATION_AUTHORIZATION_DIGEST),
    ("execution_precommit.json", "d6ebcd80b3950d51511e811d65f6a5fba072edb1b76e91f9775443c847daae58", "precommit_digest", REJECTED_V10_CALIBRATION_PRECOMMIT_DIGEST),
)

# ordinal, neutral group index, panel ID, released-record file SHA-256,
# released-record content address, exact PNG SHA-256.  Only these twelve
# already-exposed records are opened; no directory scan is performed.
_PINNED_V10_RELEASES = (
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

_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")


class ObjectBongardPanelRubricProbeError(RuntimeError):
    """The diagnostic boundary, persisted evidence, or replay differs."""


@dataclass(frozen=True, slots=True)
class _PinnedV10Nomination:
    artifact: ObjectBongardSemanticArtifact
    authorization_digest: str
    execution_precommit_digest: str
    cold_replay_digest: str
    result_digest: str
    source_digest: str
    accepted: bool


@dataclass(frozen=True, slots=True)
class _PinnedProbePanel:
    ordinal: int
    group_index: int
    panel_id: str
    released_file_sha256: str
    released_record_digest: str
    png_sha256: str
    exact_png_bytes: bytes


@dataclass(frozen=True, slots=True)
class _PinnedV10Source:
    panels: tuple[_PinnedProbePanel, ...]
    source_digest: str
    rejected_calibration_source_digest: str
    parent_file_sha256: Mapping[str, str]

    @property
    def group_a_panels(self) -> tuple[_PinnedProbePanel, ...]:
        return tuple(item for item in self.panels if item.group_index == 0)

    @property
    def group_b_panels(self) -> tuple[_PinnedProbePanel, ...]:
        return tuple(item for item in self.panels if item.group_index == 1)


@dataclass(frozen=True, slots=True)
class _ProbeInputs:
    nomination: _PinnedV10Nomination
    source: _PinnedV10Source
    precommit: ObjectBongardRubricCalibrationExecutionPrecommit
    rubric_spec: ObjectBongardRubricSpec


@dataclass(frozen=True, slots=True)
class _BlindPanelJob:
    probe_index: int
    panel_id: str
    panel_sha256: str
    exact_png_bytes: bytes
    observation_context_digest: str


@dataclass(frozen=True, slots=True)
class VerifiedObjectBongardPanelRubricProbe:
    output_root: Path
    manifest_digest: str
    result_digest: str
    exact_survivor: bool
    group_counts: Mapping[str, Mapping[str, int]]

    def summary_data(self) -> dict[str, object]:
        return {
            "schema": "gkm.bongard-panel-rubric-probe-summary.v1",
            "status": PROBE_STATUS,
            "output_root": str(self.output_root),
            "manifest_digest": self.manifest_digest,
            "result_digest": self.result_digest,
            "exact_survivor": self.exact_survivor,
            "group_counts": {
                group: dict(counts) for group, counts in self.group_counts.items()
            },
            "old_calibration_authorization_authorizes_probe_jobs": False,
        }


Transport = Callable[..., Any]


def object_bongard_panel_rubric_probe_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "lean_present": False,
        "lean_required": False,
        "lean_removable": True,
        "lean_defines_identity_or_decision": False,
        "lean_required_for_replay": False,
        "negation_allowed": False,
        "polarity_flip_allowed": False,
        "threshold_selection_allowed": False,
        "model_selection_allowed": False,
    }


def _record(body: Mapping[str, Any]) -> dict[str, Any]:
    frozen = json.loads(canonical_json(dict(body)).decode("utf-8"))
    if not isinstance(frozen, dict):
        raise ObjectBongardPanelRubricProbeError("record body is not an object")
    return {**frozen, "record_digest": "sha256:" + canonical_digest(frozen)}


def _verify_record(value: object, *, schema: str, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ObjectBongardPanelRubricProbeError(f"{label} is not an object")
    raw = dict(value)
    digest = raw.pop("record_digest", None)
    if raw.get("schema") != schema or digest != "sha256:" + canonical_digest(raw):
        raise ObjectBongardPanelRubricProbeError(f"{label} digest or schema differs")
    return {**raw, "record_digest": digest}


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_once(path: Path, value: Mapping[str, Any], label: str) -> str:
    payload = canonical_json(dict(value)) + b"\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise ObjectBongardPanelRubricProbeError(f"{label} already exists") from exc
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)
    if path.read_bytes() != payload:
        raise ObjectBongardPanelRubricProbeError(f"persisted {label} changed")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ObjectBongardPanelRubricProbeError(f"cannot read {label}") from exc
    if not isinstance(value, dict) or payload != canonical_json(value) + b"\n":
        raise ObjectBongardPanelRubricProbeError(f"{label} is not canonical JSON")
    return value


def _fresh_output_root(value: str | os.PathLike[str]) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    parent = candidate.parent.resolve(strict=True)
    root = parent / candidate.name
    if not candidate.name or root.exists() or root.is_symlink():
        raise ObjectBongardPanelRubricProbeError("output root must be fresh")
    root.mkdir(mode=0o700)
    (root / ARTIFACT_DIRECTORY).mkdir(mode=0o700)
    (root / REPLAY_DIRECTORY).mkdir(mode=0o700)
    _fsync_directory(root)
    _fsync_directory(parent)
    return root


def _existing_output_root(value: str | os.PathLike[str]) -> Path:
    root = Path(value).expanduser().resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise ObjectBongardPanelRubricProbeError("probe root is not a directory")
    return root


def _read_pinned_json(
    path: Path, *, expected_file_sha256: str, label: str
) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("utf-8", errors="strict"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ObjectBongardPanelRubricProbeError(f"cannot read pinned {label}") from exc
    if (
        hashlib.sha256(payload).hexdigest() != expected_file_sha256
        or not isinstance(value, dict)
        or payload != canonical_json(value) + b"\n"
    ):
        raise ObjectBongardPanelRubricProbeError(
            f"pinned {label} file bytes differ"
        )
    return value


def _verify_pinned_content_digest(
    value: Mapping[str, Any],
    *,
    digest_field: str,
    expected_digest: str,
    label: str,
) -> None:
    body = dict(value)
    observed = body.pop(digest_field, None)
    computed = canonical_digest(body)
    if (
        observed != expected_digest
        or expected_digest.removeprefix("sha256:") != computed
    ):
        raise ObjectBongardPanelRubricProbeError(
            f"pinned {label} content digest differs"
        )


def _load_parent_records(
    root_value: str | os.PathLike[str],
    commitments: Sequence[tuple[str, str, str, str]],
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    root = Path(root_value).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ObjectBongardPanelRubricProbeError(f"{label} root is not a directory")
    result: dict[str, dict[str, Any]] = {}
    for filename, file_sha256, digest_field, digest in commitments:
        record = _read_pinned_json(
            root / filename,
            expected_file_sha256=file_sha256,
            label=f"{label} {filename}",
        )
        _verify_pinned_content_digest(
            record,
            digest_field=digest_field,
            expected_digest=digest,
            label=f"{label} {filename}",
        )
        result[filename] = record
    return result


def _load_pinned_release_panels(
    source_directory: str | os.PathLike[str],
) -> tuple[_PinnedProbePanel, ...]:
    root = Path(source_directory).expanduser().resolve(strict=True)
    release_root = root / "released_panel"
    if not release_root.is_dir():
        raise ObjectBongardPanelRubricProbeError(
            "pinned released-panel directory is unavailable"
        )
    panels: list[_PinnedProbePanel] = []
    for (
        ordinal,
        group_index,
        panel_id,
        file_sha256,
        record_digest,
        png_sha256,
    ) in _PINNED_V10_RELEASES:
        record = _read_pinned_json(
            release_root / f"{file_sha256}.json",
            expected_file_sha256=file_sha256,
            label=f"released panel ordinal {ordinal}",
        )
        _verify_pinned_content_digest(
            record,
            digest_field="record_digest",
            expected_digest=record_digest,
            label=f"released panel ordinal {ordinal}",
        )
        encoded = record.get("exact_png_base64")
        try:
            png = base64.b64decode(encoded, validate=True)
        except (TypeError, ValueError, binascii.Error) as exc:
            raise ObjectBongardPanelRubricProbeError(
                f"released panel ordinal {ordinal} PNG is malformed"
            ) from exc
        if (
            record.get("schema") != "gkm.bongard-released-panel.v1"
            or record.get("panel_id") != panel_id
            or record.get("exact_png_digest") != "sha256:" + png_sha256
            or hashlib.sha256(png).hexdigest() != png_sha256
            or not png.startswith(b"\x89PNG\r\n\x1a\n")
        ):
            raise ObjectBongardPanelRubricProbeError(
                f"released panel ordinal {ordinal} identity differs"
            )
        panels.append(
            _PinnedProbePanel(
                ordinal,
                group_index,
                panel_id,
                file_sha256,
                record_digest,
                png_sha256,
                png,
            )
        )
    if len(panels) != PROBE_PANEL_COUNT:
        raise ObjectBongardPanelRubricProbeError("pinned panel count differs")
    return tuple(panels)


def _load_probe_inputs(
    *,
    nomination_root: str | os.PathLike[str],
    rejected_calibration_root: str | os.PathLike[str],
    source_directory: str | os.PathLike[str],
) -> _ProbeInputs:
    nomination_records = _load_parent_records(
        nomination_root,
        _V10_NOMINATION_PARENT_FILES,
        label="v10 nomination",
    )
    authorization = nomination_records["authorization.json"]
    nomination_precommit = nomination_records["execution_precommit.json"]
    cold_replay = nomination_records["cold_replay.json"]
    nomination_result = nomination_records["result.json"]
    if (
        authorization.get("schema")
        != "gkm.bongard-object-rubric-nomination-authorization.v4"
        or nomination_precommit.get("schema")
        != "gkm.bongard-object-rubric-nomination-precommit.v4"
        or cold_replay.get("schema")
        != "gkm.bongard-object-rubric-nomination-cold-replay.v4"
        or nomination_result.get("schema")
        != "gkm.bongard-object-rubric-nomination-result.v4"
        or authorization.get("source_digest") != V10_NOMINATION_SOURCE_DIGEST
        or nomination_precommit.get("source_digest")
        != V10_NOMINATION_SOURCE_DIGEST
        or cold_replay.get("source_digest") != V10_NOMINATION_SOURCE_DIGEST
        or nomination_result.get("source_digest") != V10_NOMINATION_SOURCE_DIGEST
        or nomination_precommit.get("authorization_digest")
        != V10_NOMINATION_AUTHORIZATION_DIGEST
        or cold_replay.get("authorization_digest")
        != V10_NOMINATION_AUTHORIZATION_DIGEST
        or nomination_result.get("authorization_digest")
        != V10_NOMINATION_AUTHORIZATION_DIGEST
        or cold_replay.get("execution_precommit_digest")
        != V10_NOMINATION_PRECOMMIT_DIGEST
        or nomination_result.get("execution_precommit_digest")
        != V10_NOMINATION_PRECOMMIT_DIGEST
        or nomination_result.get("cold_replay_digest")
        != V10_NOMINATION_REPLAY_DIGEST
        or cold_replay.get("semantic_artifact_digest")
        != V10_NOMINATION_ARTIFACT_DIGEST
        or nomination_result.get("semantic_artifact_digest")
        != V10_NOMINATION_ARTIFACT_DIGEST
        or nomination_result.get("accepted") is not True
        or authorization.get("groups") != nomination_precommit.get("groups")
        or authorization.get("source_digests")
        != nomination_precommit.get("source_digests")
        or authorization.get("support_panels_per_group") != 6
    ):
        raise ObjectBongardPanelRubricProbeError(
            "pinned v10 nomination parent linkage differs"
        )

    panels = _load_pinned_release_panels(source_directory)
    expected_groups = (
        tuple(sorted(item.panel_id for item in panels if item.group_index == 0)),
        tuple(sorted(item.panel_id for item in panels if item.group_index == 1)),
    )
    expected_group_rows = tuple(
        {
            "group_id": f"group_{group_index}",
            "panel_binding_digests": authorization["groups"][group_index][
                "panel_binding_digests"
            ],
            "panel_ids": list(expected_groups[group_index]),
            "png_sha256": [
                next(item.png_sha256 for item in panels if item.panel_id == panel_id)
                for panel_id in expected_groups[group_index]
            ],
        }
        for group_index in (0, 1)
    )
    if tuple(authorization.get("groups", ())) != expected_group_rows:
        raise ObjectBongardPanelRubricProbeError(
            "pinned release records differ from the v10 group commitment"
        )

    artifact = ObjectBongardSemanticArtifact.from_data(
        nomination_records["semantic_artifact.json"],
        expected_artifact_digest=V10_NOMINATION_ARTIFACT_DIGEST,
    )
    if artifact.group_panel_ids != expected_groups:
        raise ObjectBongardPanelRubricProbeError(
            "semantic artifact group presentation differs"
        )
    support_png = {item.panel_id: item.exact_png_bytes for item in panels}
    verify_object_bongard_semantic_artifact(
        artifact,
        support_png_by_panel_id=support_png,
        expected_task_id=V10_CONTEXT_TASK_ID,
        expected_observation_context_digest=V10_NOMINATION_PRECOMMIT_DIGEST,
        expected_artifact_digest=V10_NOMINATION_ARTIFACT_DIGEST,
    )
    nomination = _PinnedV10Nomination(
        artifact,
        V10_NOMINATION_AUTHORIZATION_DIGEST,
        V10_NOMINATION_PRECOMMIT_DIGEST,
        V10_NOMINATION_REPLAY_DIGEST,
        V10_NOMINATION_RESULT_DIGEST,
        V10_NOMINATION_SOURCE_DIGEST,
        True,
    )

    calibration_records = _load_parent_records(
        rejected_calibration_root,
        _V10_CALIBRATION_PARENT_FILES,
        label="rejected v10 calibration",
    )
    calibration_authorization = calibration_records["authorization.json"]
    calibration_precommit = calibration_records["execution_precommit.json"]
    if (
        calibration_authorization.get("schema")
        != "gkm.bongard-object-rubric-calibration-authorization.v4"
        or calibration_precommit.get("schema")
        != "gkm.bongard-object-rubric-calibration-execution-precommit.v4"
        or calibration_authorization.get("source_digest")
        != REJECTED_V10_CALIBRATION_SOURCE_DIGEST
        or calibration_precommit.get("source_digest")
        != REJECTED_V10_CALIBRATION_SOURCE_DIGEST
        or calibration_precommit.get("authorization_digest")
        != REJECTED_V10_CALIBRATION_AUTHORIZATION_DIGEST
        or calibration_authorization.get("nomination_binding")
        != calibration_precommit.get("nomination_binding")
        or calibration_authorization.get("source_digests")
        != calibration_precommit.get("source_digests")
    ):
        raise ObjectBongardPanelRubricProbeError(
            "rejected v10 calibration parent linkage differs"
        )
    precommit = ObjectBongardRubricCalibrationExecutionPrecommit.from_data(
        calibration_precommit
    )
    if (
        precommit.precommit_digest != REJECTED_V10_CALIBRATION_PRECOMMIT_DIGEST
        or precommit.authorization_digest
        != REJECTED_V10_CALIBRATION_AUTHORIZATION_DIGEST
        or precommit.nomination_binding.artifact_digest
        != nomination.artifact.artifact_digest
        or precommit.nomination_binding.authorization_digest
        != nomination.authorization_digest
        or precommit.nomination_binding.execution_precommit_digest
        != nomination.execution_precommit_digest
        or precommit.nomination_binding.cold_replay_digest
        != nomination.cold_replay_digest
        or precommit.nomination_binding.command_result_digest
        != nomination.result_digest
    ):
        raise ObjectBongardPanelRubricProbeError(
            "runtime precommit is not the rejected v10 calibration precommit"
        )
    parent_files = {
        f"nomination/{filename}": file_sha256
        for filename, file_sha256, _field, _digest in _V10_NOMINATION_PARENT_FILES
    }
    parent_files.update(
        {
            f"rejected_calibration/{filename}": file_sha256
            for filename, file_sha256, _field, _digest in _V10_CALIBRATION_PARENT_FILES
        }
    )
    parent_files.update(
        {
            f"released_panel/{item.panel_id}": item.released_file_sha256
            for item in panels
        }
    )
    source = _PinnedV10Source(
        panels,
        V10_NOMINATION_SOURCE_DIGEST,
        REJECTED_V10_CALIBRATION_SOURCE_DIGEST,
        parent_files,
    )
    rubric_spec = ObjectBongardRubricSpec.from_semantic_artifact(
        nomination.artifact,
        expected_artifact_digest=nomination.artifact.artifact_digest,
        candidate_rank=0,
    )
    return _ProbeInputs(nomination, source, precommit, rubric_spec)


def _runtime_binding(inputs: _ProbeInputs) -> dict[str, object]:
    return json.loads(canonical_json(inputs.precommit.runtime.binding).decode("utf-8"))


def _blind_jobs(inputs: _ProbeInputs) -> tuple[_BlindPanelJob, ...]:
    runtime_digest = canonical_digest(
        {"schema": "gkm.bongard-panel-rubric-probe-runtime-binding.v1", **_runtime_binding(inputs)}
    )
    jobs: list[_BlindPanelJob] = []
    for index, panel in enumerate(inputs.source.panels):
        context = "sha256:" + canonical_digest(
            {
                "schema": "gkm.bongard-panel-rubric-probe-context.v1",
                "status": PROBE_STATUS,
                "probe_index": index,
                "panel_id": panel.panel_id,
                "panel_sha256": panel.png_sha256,
                "rubric_spec_digest": inputs.rubric_spec.spec_digest,
                "runtime_binding_digest": runtime_digest,
                "nomination_artifact_digest": inputs.nomination.artifact.artifact_digest,
                "rejected_calibration_precommit_digest": inputs.precommit.precommit_digest,
            }
        )
        jobs.append(
            _BlindPanelJob(
                index,
                panel.panel_id,
                panel.png_sha256,
                panel.exact_png_bytes,
                context,
            )
        )
    return tuple(jobs)


def _manifest(inputs: _ProbeInputs, *, parallel_workers: int) -> dict[str, Any]:
    if isinstance(parallel_workers, bool) or not 1 <= parallel_workers <= PROBE_MAX_WORKERS:
        raise ObjectBongardPanelRubricProbeError("parallel workers must lie in 1..4")
    jobs = _blind_jobs(inputs)
    prompt = object_bongard_panel_rubric_prompt(inputs.rubric_spec)
    schema = object_bongard_panel_rubric_output_schema()
    runtime = _runtime_binding(inputs)
    body = {
        "schema": PROBE_MANIFEST_SCHEMA,
        "status": PROBE_STATUS,
        "purpose": "rank-0-whole-panel-observer-diagnostic-on-already-exposed-calibration-only",
        "authorization": {
            "rejected_calibration_authorization_digest": inputs.precommit.authorization_digest,
            "rejected_calibration_precommit_digest": inputs.precommit.precommit_digest,
            "old_calibration_authorization_authorizes_probe_jobs": False,
            "new_probe_authorization_present": False,
            "benchmark_or_calibration_claim_authorized": False,
        },
        "pixel_scope": {
            "exact_already_exposed_calibration_panel_count": len(jobs),
            "query_pixels_opened": False,
            "broad_cohort_pixels_opened": False,
            "official_test_pixels_opened": False,
            "other_pixels_authorized": False,
        },
        "call_policy": {
            "rank": 0,
            "one_complete_panel_per_call": True,
            "calls_per_panel": 1,
            "physical_call_count": len(jobs),
            "parallel_workers": parallel_workers,
            "maximum_parallel_workers": PROBE_MAX_WORKERS,
            "model_visible_image_names": ["panel.png"],
            "labels_or_roles_visible_to_observer": False,
            "labels_used_for_disposition_aggregation_only_after_all_artifacts_persisted_and_replayed": True,
        },
        "nomination": {
            "artifact_digest": inputs.nomination.artifact.artifact_digest,
            "authorization_digest": inputs.nomination.authorization_digest,
            "execution_precommit_digest": inputs.nomination.execution_precommit_digest,
            "cold_replay_digest": inputs.nomination.cold_replay_digest,
            "result_digest": inputs.nomination.result_digest,
        },
        "frozen_nomination_source_digest": inputs.source.source_digest,
        "frozen_rejected_calibration_source_digest": (
            inputs.source.rejected_calibration_source_digest
        ),
        "rubric_spec": inputs.rubric_spec.to_data(),
        "rubric_spec_digest": inputs.rubric_spec.spec_digest,
        "prompt_digest": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "output_schema_digest": canonical_digest(schema),
        "runtime_binding": runtime,
        "runtime_binding_digest": canonical_digest(
            {"schema": "gkm.bongard-panel-rubric-probe-runtime-binding.v1", **runtime}
        ),
        "runtime_objects_reused_from_rejected_v10_precommit": True,
        "source_identities": {
            "probe_source_sha256": object_bongard_panel_rubric_probe_source_digest(),
            "panel_observer_source_sha256": object_bongard_panel_rubric_observer_source_digest(),
            "panel_observer_protocol_sha256": object_bongard_panel_rubric_protocol_digest(),
            "rubric_spec_authority_source_sha256": (
                object_bongard_rubric_language_source_digest()
            ),
            "semantic_artifact_source_sha256": inputs.nomination.artifact.source_digest,
            "runtime_transport_source_sha256": inputs.precommit.runtime.transport_source_digest,
            "pinned_parent_file_sha256": dict(inputs.source.parent_file_sha256),
        },
        "panels": [
            {
                "probe_index": job.probe_index,
                "source_ordinal": panel.ordinal,
                "panel_id": job.panel_id,
                "panel_sha256": job.panel_sha256,
                "released_record_file_sha256": panel.released_file_sha256,
                "released_record_digest": panel.released_record_digest,
                "observation_context_digest": job.observation_context_digest,
            }
            for job, panel in zip(jobs, inputs.source.panels, strict=True)
        ],
        **_authority_data(),
    }
    return _record(body)


def _artifact_filename(index: int) -> str:
    return f"{index:03d}.json"


def _artifact_replay_record(
    *,
    manifest_digest: str,
    job: _BlindPanelJob,
    artifact: ObjectBongardPanelRubricArtifact,
    artifact_file_sha256: str,
) -> dict[str, Any]:
    return _record(
        {
            "schema": PROBE_REPLAY_SCHEMA,
            "status": PROBE_STATUS,
            "manifest_digest": manifest_digest,
            "probe_index": job.probe_index,
            "panel_id": job.panel_id,
            "panel_sha256": job.panel_sha256,
            "artifact_file_sha256": artifact_file_sha256,
            "artifact_digest": artifact.artifact_digest,
            "runtime_identity_digest": artifact.runtime_identity_digest,
            "observation_digest": artifact.observation.observation_digest,
            "disposition": artifact.observation.disposition.value,
            "physical_call_count": artifact.physical_call_count,
            "cold_replay_model_calls": 0,
            "cold_replay_verified": True,
            "old_calibration_authorization_authorizes_probe_job": False,
        }
    )


def _run_blind_job(
    *,
    root: Path,
    manifest_digest: str,
    inputs: _ProbeInputs,
    job: _BlindPanelJob,
    transport: Transport,
) -> ObjectBongardPanelRubricArtifact:
    runtime = inputs.precommit.runtime
    artifact = observe_object_bongard_panel_rubric(
        job.exact_png_bytes,
        panel_id=job.panel_id,
        rubric_spec=inputs.rubric_spec,
        expected_panel_sha256=job.panel_sha256,
        expected_rubric_spec_digest=inputs.rubric_spec.spec_digest,
        model=runtime.model,
        reasoning_effort=runtime.reasoning_effort,
        minutes=runtime.minutes,
        verbose=runtime.verbose,
        executable=runtime.executable,
        cloud_policy_cache_snapshot=runtime.cloud_policy_cache_snapshot,
        expected_launcher_digest=runtime.expected_launcher_digest,
        model_catalog_snapshot=runtime.model_catalog_snapshot,
        no_tools_attestation=runtime.no_tools_attestation,
        transport=transport,
        observation_context_digest=job.observation_context_digest,
    )
    artifact_path = root / ARTIFACT_DIRECTORY / _artifact_filename(job.probe_index)
    file_sha256 = _write_once(
        artifact_path, artifact.to_data(), f"artifact {job.probe_index}"
    )
    restored = ObjectBongardPanelRubricArtifact.from_data(
        _read_json(artifact_path, f"artifact {job.probe_index}")
    )
    replayed = verify_object_bongard_panel_rubric_artifact(
        restored,
        job.exact_png_bytes,
        panel_id=job.panel_id,
        rubric_spec=inputs.rubric_spec,
        expected_artifact_digest=artifact.artifact_digest,
        expected_runtime_identity_digest=artifact.runtime_identity_digest,
    )
    replay = _artifact_replay_record(
        manifest_digest=manifest_digest,
        job=job,
        artifact=replayed,
        artifact_file_sha256=file_sha256,
    )
    _write_once(
        root / REPLAY_DIRECTORY / _artifact_filename(job.probe_index),
        replay,
        f"artifact replay {job.probe_index}",
    )
    return replayed


_DISPOSITIONS = tuple(item.value for item in PanelRubricDisposition)


def _group_counts(
    source: _PinnedV10Source,
    artifacts_by_panel: Mapping[str, ObjectBongardPanelRubricArtifact],
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for name, group_index in (("group_a", 0), ("group_b", 1)):
        panels = tuple(
            item for item in source.panels if item.group_index == group_index
        )
        result[name] = {
            disposition: sum(
                artifacts_by_panel[item.panel_id].observation.disposition.value
                == disposition
                for item in panels
            )
            for disposition in _DISPOSITIONS
        }
    return result


def _result_record(
    *,
    manifest_digest: str,
    inputs: _ProbeInputs,
    jobs: tuple[_BlindPanelJob, ...],
    artifacts: tuple[ObjectBongardPanelRubricArtifact, ...],
) -> dict[str, Any]:
    by_panel = {item.panel_id: item for item in artifacts}
    if len(by_panel) != PROBE_PANEL_COUNT or set(by_panel) != {
        item.panel_id for item in inputs.source.panels
    }:
        raise ObjectBongardPanelRubricProbeError("completed artifact inventory differs")
    counts = _group_counts(inputs.source, by_panel)
    exact_survivor = (
        counts["group_a"][PanelRubricDisposition.PRESENT.value] == 6
        and counts["group_a"][PanelRubricDisposition.CERTIFIED_ABSENT.value] == 0
        and counts["group_a"][PanelRubricDisposition.INDETERMINATE.value] == 0
        and counts["group_a"][PanelRubricDisposition.ERROR.value] == 0
        and counts["group_b"][PanelRubricDisposition.CERTIFIED_ABSENT.value] == 6
        and counts["group_b"][PanelRubricDisposition.PRESENT.value] == 0
        and counts["group_b"][PanelRubricDisposition.INDETERMINATE.value] == 0
        and counts["group_b"][PanelRubricDisposition.ERROR.value] == 0
    )
    return _record(
        {
            "schema": PROBE_RESULT_SCHEMA,
            "status": PROBE_STATUS,
            "manifest_digest": manifest_digest,
            "rank": 0,
            "rubric_spec_digest": inputs.rubric_spec.spec_digest,
            "physical_call_count": sum(item.physical_call_count for item in artifacts),
            "persisted_artifact_count": len(artifacts),
            "model_free_cold_replay_count": len(artifacts),
            "cold_replay_model_calls": 0,
            "group_counts": counts,
            "exact_survivor_rule": "group-a-six-present-and-group-b-six-certified-absence",
            "exact_survivor": exact_survivor,
            "labels_used_for_group_counts_after_all_calls_completed": True,
            "observer_received_group_labels_or_roles": False,
            "old_calibration_authorization_authorizes_probe_jobs": False,
            "benchmark_or_calibration_claim_authorized": False,
            "artifacts": [
                {
                    "probe_index": job.probe_index,
                    "panel_id": job.panel_id,
                    "artifact_digest": artifact.artifact_digest,
                    "runtime_identity_digest": artifact.runtime_identity_digest,
                    "observation_digest": artifact.observation.observation_digest,
                    "disposition": artifact.observation.disposition.value,
                }
                for job, artifact in zip(jobs, artifacts, strict=True)
            ],
            **_authority_data(),
        }
    )


def _verification(
    root: Path,
    manifest: Mapping[str, Any],
    result: Mapping[str, Any],
) -> VerifiedObjectBongardPanelRubricProbe:
    counts = result.get("group_counts")
    if not isinstance(counts, Mapping):
        raise ObjectBongardPanelRubricProbeError("result group counts are malformed")
    return VerifiedObjectBongardPanelRubricProbe(
        root,
        manifest["record_digest"],
        result["record_digest"],
        result["exact_survivor"],
        {
            group: dict(value)
            for group, value in counts.items()
            if isinstance(group, str) and isinstance(value, Mapping)
        },
    )


def _run_loaded_probe(
    output_root: str | os.PathLike[str],
    inputs: _ProbeInputs,
    *,
    parallel_workers: int,
    transport: Transport,
) -> VerifiedObjectBongardPanelRubricProbe:
    root = _fresh_output_root(output_root)
    manifest = _manifest(inputs, parallel_workers=parallel_workers)
    _write_once(root / MANIFEST_FILENAME, manifest, "pre-call manifest")
    manifest_digest = manifest["record_digest"]
    jobs = _blind_jobs(inputs)

    completed: dict[int, ObjectBongardPanelRubricArtifact] = {}
    with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        futures = {
            executor.submit(
                _run_blind_job,
                root=root,
                manifest_digest=manifest_digest,
                inputs=inputs,
                job=job,
                transport=transport,
            ): job.probe_index
            for job in jobs
        }
        for future in as_completed(futures):
            completed[futures[future]] = future.result()
    artifacts = tuple(completed[index] for index in range(PROBE_PANEL_COUNT))

    # This is the first point at which group membership affects a computation.
    result = _result_record(
        manifest_digest=manifest_digest,
        inputs=inputs,
        jobs=jobs,
        artifacts=artifacts,
    )
    _write_once(root / RESULT_FILENAME, result, "probe result")
    return _verification(root, manifest, result)


def run_object_bongard_panel_rubric_probe(
    output_root: str | os.PathLike[str],
    *,
    nomination_root: str | os.PathLike[str] = DEFAULT_V10_NOMINATION_ROOT,
    rejected_calibration_root: str | os.PathLike[str] = (
        DEFAULT_REJECTED_V10_CALIBRATION_ROOT
    ),
    source_directory: str | os.PathLike[str] = DEFAULT_OBJECT_RUBRIC_CALIBRATION_SOURCE,
    parallel_workers: int = PROBE_MAX_WORKERS,
    transport: Transport = run_codex_named_images_structured,
) -> VerifiedObjectBongardPanelRubricProbe:
    """Run twelve new diagnostic calls after cold-verifying every v10 parent."""

    inputs = _load_probe_inputs(
        nomination_root=nomination_root,
        rejected_calibration_root=rejected_calibration_root,
        source_directory=source_directory,
    )
    return _run_loaded_probe(
        output_root,
        inputs,
        parallel_workers=parallel_workers,
        transport=transport,
    )


def _verify_loaded_probe(
    output_root: str | os.PathLike[str], inputs: _ProbeInputs
) -> VerifiedObjectBongardPanelRubricProbe:
    root = _existing_output_root(output_root)
    if {item.name for item in root.iterdir()} != {
        MANIFEST_FILENAME,
        RESULT_FILENAME,
        ARTIFACT_DIRECTORY,
        REPLAY_DIRECTORY,
    }:
        raise ObjectBongardPanelRubricProbeError("probe root inventory differs")
    stored_manifest = _verify_record(
        _read_json(root / MANIFEST_FILENAME, "probe manifest"),
        schema=PROBE_MANIFEST_SCHEMA,
        label="probe manifest",
    )
    call_policy = stored_manifest.get("call_policy")
    if not isinstance(call_policy, Mapping):
        raise ObjectBongardPanelRubricProbeError("manifest call policy is malformed")
    workers = call_policy.get("parallel_workers")
    expected_manifest = _manifest(inputs, parallel_workers=workers)
    if stored_manifest != expected_manifest:
        raise ObjectBongardPanelRubricProbeError("pre-call manifest differs on replay")
    jobs = _blind_jobs(inputs)
    expected_files = {_artifact_filename(index) for index in range(PROBE_PANEL_COUNT)}
    artifact_root = root / ARTIFACT_DIRECTORY
    replay_root = root / REPLAY_DIRECTORY
    if (
        not artifact_root.is_dir()
        or not replay_root.is_dir()
        or {item.name for item in artifact_root.iterdir()} != expected_files
        or {item.name for item in replay_root.iterdir()} != expected_files
    ):
        raise ObjectBongardPanelRubricProbeError("artifact/replay inventory differs")
    artifacts: list[ObjectBongardPanelRubricArtifact] = []
    for job in jobs:
        artifact_path = artifact_root / _artifact_filename(job.probe_index)
        artifact_bytes = artifact_path.read_bytes()
        artifact = ObjectBongardPanelRubricArtifact.from_data(
            _read_json(artifact_path, f"artifact {job.probe_index}")
        )
        replayed = verify_object_bongard_panel_rubric_artifact(
            artifact,
            job.exact_png_bytes,
            panel_id=job.panel_id,
            rubric_spec=inputs.rubric_spec,
            expected_artifact_digest=artifact.artifact_digest,
            expected_runtime_identity_digest=artifact.runtime_identity_digest,
        )
        if replayed.observation_context_digest != job.observation_context_digest:
            raise ObjectBongardPanelRubricProbeError("artifact context differs")
        expected_replay = _artifact_replay_record(
            manifest_digest=stored_manifest["record_digest"],
            job=job,
            artifact=replayed,
            artifact_file_sha256=hashlib.sha256(artifact_bytes).hexdigest(),
        )
        stored_replay = _verify_record(
            _read_json(
                replay_root / _artifact_filename(job.probe_index),
                f"artifact replay {job.probe_index}",
            ),
            schema=PROBE_REPLAY_SCHEMA,
            label=f"artifact replay {job.probe_index}",
        )
        if stored_replay != expected_replay:
            raise ObjectBongardPanelRubricProbeError("artifact replay differs")
        artifacts.append(replayed)
    expected_result = _result_record(
        manifest_digest=stored_manifest["record_digest"],
        inputs=inputs,
        jobs=jobs,
        artifacts=tuple(artifacts),
    )
    stored_result = _verify_record(
        _read_json(root / RESULT_FILENAME, "probe result"),
        schema=PROBE_RESULT_SCHEMA,
        label="probe result",
    )
    if stored_result != expected_result:
        raise ObjectBongardPanelRubricProbeError("probe result differs on replay")
    return _verification(root, stored_manifest, stored_result)


def verify_object_bongard_panel_rubric_probe(
    output_root: str | os.PathLike[str],
    *,
    nomination_root: str | os.PathLike[str] = DEFAULT_V10_NOMINATION_ROOT,
    rejected_calibration_root: str | os.PathLike[str] = (
        DEFAULT_REJECTED_V10_CALIBRATION_ROOT
    ),
    source_directory: str | os.PathLike[str] = DEFAULT_OBJECT_RUBRIC_CALIBRATION_SOURCE,
) -> VerifiedObjectBongardPanelRubricProbe:
    """Cold-replay a completed diagnostic directory without model access."""

    inputs = _load_probe_inputs(
        nomination_root=nomination_root,
        rejected_calibration_root=rejected_calibration_root,
        source_directory=source_directory,
    )
    return _verify_loaded_probe(output_root, inputs)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or cold-verify the unsealed v10 rank-0 whole-panel diagnostic"
    )
    commands = parser.add_subparsers(dest="operation", required=True)
    for name in ("launch", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--output-root", required=True, type=Path)
        command.add_argument(
            "--nomination-root", type=Path, default=DEFAULT_V10_NOMINATION_ROOT
        )
        command.add_argument(
            "--rejected-calibration-root",
            type=Path,
            default=DEFAULT_REJECTED_V10_CALIBRATION_ROOT,
        )
        command.add_argument(
            "--source-directory",
            type=Path,
            default=DEFAULT_OBJECT_RUBRIC_CALIBRATION_SOURCE,
        )
    commands.choices["launch"].add_argument(
        "--parallel-workers", type=int, default=PROBE_MAX_WORKERS
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(None if argv is None else list(argv))
    try:
        common = {
            "nomination_root": args.nomination_root,
            "rejected_calibration_root": args.rejected_calibration_root,
            "source_directory": args.source_directory,
        }
        if args.operation == "launch":
            verified = run_object_bongard_panel_rubric_probe(
                args.output_root,
                parallel_workers=args.parallel_workers,
                **common,
            )
        else:
            verified = verify_object_bongard_panel_rubric_probe(
                args.output_root, **common
            )
    except Exception as exc:
        print(
            canonical_json(
                {
                    "schema": "gkm.bongard-panel-rubric-probe-error.v1",
                    "status": PROBE_STATUS,
                    "error_type": type(exc).__name__,
                    "message": str(exc)[:2000],
                }
            ).decode("utf-8"),
            file=sys.stderr,
        )
        return 1
    print(canonical_json(verified.summary_data()).decode("utf-8"))
    return 0 if verified.exact_survivor else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = (
    "DEFAULT_REJECTED_V10_CALIBRATION_ROOT",
    "DEFAULT_V10_NOMINATION_ROOT",
    "ObjectBongardPanelRubricProbeError",
    "PROBE_MAX_WORKERS",
    "PROBE_PANEL_COUNT",
    "PROBE_STATUS",
    "VerifiedObjectBongardPanelRubricProbe",
    "main",
    "object_bongard_panel_rubric_probe_source_digest",
    "run_object_bongard_panel_rubric_probe",
    "verify_object_bongard_panel_rubric_probe",
)
