"""Production launch and cold-replay boundary for prose-rubric calibration.

The calibration driver owns the visual and predicate semantics.  This module
owns the causal filesystem boundary around it:

1. cold-verify and embed the one-turn neutral semantic nomination predecessor;
2. bind its two ranked positive cue pairs to two canonical signed comparisons;
3. freeze and persist the complete 24-job / 30-sheet calibration inventory;
4. capture and durably persist the exact Codex runtime precommit;
5. only then admit the journaled contrastive vision calls; and
6. replay pixels, nomination, journals, predicates, and assessment without a
   model transport.

The implementation is Python-authoritative.  It neither imports Lean nor
gives a proof checker any role in record identity, execution, or replay.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

import argparse
import base64
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Callable, Mapping, Sequence

from bongard.canonical import canonical_digest, canonical_json
from bongard.codex_no_tools_preflight import (
    CodexNoToolsAttestation,
    attest_codex_no_tools,
)
from bongard.object_bongard_rubric_calibration import (
    DEFAULT_OBJECT_RUBRIC_CALIBRATION_SOURCE,
    OBJECT_RUBRIC_CALIBRATION_ALGORITHM_ID,
    ObjectBongardRubricCalibrationAssessment,
    ObjectBongardRubricCalibrationSource,
    ObjectBongardRubricObservationBatch,
    assess_object_bongard_rubric_calibration,
    _bind_object_bongard_rubric_calibration_nomination_content,
    cold_verify_object_bongard_rubric_calibration,
    load_object_bongard_rubric_calibration_source,
    object_bongard_rubric_calibration_source_digest,
    run_object_bongard_rubric_calibration_observation,
    run_object_bongard_rubric_calibration_observations,
)
from bongard.object_bongard_rubric_observer import (
    ObjectBongardRubricObserverArtifact,
    object_bongard_rubric_observer_catalog_digest,
    object_bongard_rubric_observer_output_schema,
    object_bongard_rubric_observer_prompt,
    object_bongard_rubric_observer_protocol_digest,
    object_bongard_rubric_observer_source_digest,
    object_bongard_rubric_ordinal_scale_digest,
)
from bongard.object_bongard_rubric_version_space import (
    object_bongard_rubric_version_space_algorithm_digest,
)
from bongard.object_bongard_rubric_slate import (
    object_bongard_rubric_slate_algorithm_digest,
    object_bongard_rubric_slate_source_digest,
)
from bongard.object_bongard_rubric_nomination_command import (
    VerifiedObjectBongardRubricNomination,
    cold_verify_object_bongard_rubric_nomination,
    copy_verified_object_bongard_rubric_nomination,
    object_bongard_rubric_nomination_command_source_digest,
)
from bongard.object_bongard_semantics import (
    object_bongard_semantics_protocol_digest,
    object_bongard_semantics_source_digest,
)
from bongard.object_bongard_turn_journal import (
    ObjectBongardTurnRuntime,
    object_bongard_turn_journal_source_digest,
)
from bongard.prototype_object_hypotheses import (
    object_hypothesis_extractor_artifact_digest,
    object_hypothesis_extractor_source_digest,
    render_object_hypothesis_atlas,
)
from bongard.prototype_object_lineages import (
    object_lineage_artifact_digest,
    object_lineage_source_digest,
)
from bongard.prototype_object_scene_observer import (
    prototype_scene_observer_source_digest,
    prototype_scene_transport_source_digest,
)
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID
from bongard.transport import (
    DEFAULT_CODEX_MODEL,
    DEFAULT_REASONING_EFFORT,
    PINNED_CODEX_CLI_VERSION,
    CloudPolicyCacheSnapshot,
    CodexModelCatalogSnapshot,
    CodexStructuredResult,
    codex_cli_authenticated_fingerprint,
    run_codex_named_images_structured,
    snapshot_cloud_policy_cache,
    snapshot_pinned_model_catalog,
)


OBJECT_RUBRIC_CALIBRATION_AUTHORIZATION_SCHEMA = (
    "gkm.bongard-object-rubric-calibration-authorization.v4"
)
OBJECT_RUBRIC_CALIBRATION_PRECOMMIT_SCHEMA = (
    "gkm.bongard-object-rubric-calibration-execution-precommit.v4"
)
OBJECT_RUBRIC_CALIBRATION_INVENTORY_SCHEMA = (
    "gkm.bongard-object-rubric-calibration-observation-inventory.v4"
)
OBJECT_RUBRIC_CALIBRATION_REPLAY_SCHEMA = (
    "gkm.bongard-object-rubric-calibration-disk-replay.v4"
)
OBJECT_RUBRIC_CALIBRATION_COMMAND_ID = (
    "bongard.object-rubric-calibration-command/seal-run-reload-v4"
)

CALIBRATION_MODEL = "gpt-5.6-sol"
CALIBRATION_REASONING_EFFORT = "medium"
CALIBRATION_JOB_COUNT = 24
CALIBRATION_SHEET_JOURNAL_COUNT = 30
CALIBRATION_PARALLEL_WORKERS = 4
CALIBRATION_MINUTES = 15
DEFAULT_CODEX_EXECUTABLE = "codex"
DEFAULT_CALIBRATION_CODEX_LAUNCHER_SHA256 = (
    "19c4f144c5226a9f17c58e6f0fa854843b0f77a6eb420f40e2745a12f10f5d37"
)

HISTORICAL_RELEASE_AUTHORIZATION_RECORD_DIGEST = (
    "sha256:28c4d00c687edee9448b637d9b6b0a749b0d9ea4fed6b578c6e7e60c2858ea7c"
)
HISTORICAL_RELEASE_AUTHORIZATION_FILE_SHA256 = (
    "f4d63556a2ba62324314e307cbe16a79831edd3653b45e205978afbfbf57b724"
)
HISTORICAL_EXECUTION_PRECOMMIT_RECORD_DIGEST = (
    "sha256:bbe51288494a6c1184da1fc521b9c0c9ce6efdea8f3226fac99c9d11112aefdb"
)
HISTORICAL_EXECUTION_PRECOMMIT_FILE_SHA256 = (
    "88bb217a06da66dab353a9b8f8c80c1ff745bc440a15a374ffd3d3479049dd31"
)
HISTORICAL_COHORT_PLAN_DIGEST = (
    "sha256:a3fd9037cb11e86e892045d7b961cb477ad2d225885cfaac25338f34bb6189e4"
)
HISTORICAL_EXPOSURE_SUCCESSOR_DIGEST = (
    "sha256:73f4f6ad2cdb5413456b4298722cc26cd8de9e733e80e7b178d97b87d11fd276"
)

CALIBRATION_ACCEPTANCE_RULE = (
    "pass-iff-at-least-one-of-four-fixed-rank-major-object-then-scene-"
    "candidates-is-an-exact-six-positive-present-six-negative-certified-"
    "absent-support-survivor/v3"
)

AUTHORIZATION_FILENAME = "authorization.json"
PRECOMMIT_FILENAME = "execution_precommit.json"
INVENTORY_FILENAME = "observation_inventory.json"
ASSESSMENT_FILENAME = "assessment.json"
REPLAY_FILENAME = "cold_replay.json"
JOURNAL_DIRECTORY = "journals"
OBSERVER_ARTIFACT_DIRECTORY = "observer_artifacts"
NOMINATION_DIRECTORY = "semantic_nomination"

_RAW_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_TASK_ID = re.compile(r"(?:bd|ff|hd)_[A-Za-z0-9_.-]+\Z")
_PANEL_ID = re.compile(r"(?:bd|ff|hd)/[A-Za-z0-9_./-]+\.png\Z")
_SHEET_NAME = re.compile(r"sheet_[0-9]{3}\.png\Z")
_MAX_RECORD_BYTES = 512 * 1024 * 1024

RuntimeAttester = Callable[..., CodexNoToolsAttestation]
LauncherFingerprinter = Callable[..., Mapping[str, str]]
ObservationRunner = Callable[..., ObjectBongardRubricObservationBatch]
NamedImageTransport = Callable[..., CodexStructuredResult]


class ObjectBongardRubricCalibrationCommandError(RuntimeError):
    """The production calibration boundary or disk replay is invalid."""


def object_bongard_rubric_calibration_command_source_digest() -> str:
    return verify_loaded_source(
        __name__, expected_source_sha256=_LOADED_SOURCE_SHA256
    )


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


@dataclass(frozen=True, slots=True)
class ObjectBongardRubricCalibrationNominationBinding:
    """Immutable content addresses for the verified nomination predecessor."""

    artifact_digest: str
    authorization_digest: str
    execution_precommit_digest: str
    cold_replay_digest: str
    command_result_digest: str

    def __post_init__(self) -> None:
        _raw_digest(self.artifact_digest, "nomination artifact digest")
        for name in (
            "authorization_digest",
            "execution_precommit_digest",
            "cold_replay_digest",
            "command_result_digest",
        ):
            _address(getattr(self, name), f"nomination {name}")

    def to_data(self) -> dict[str, str]:
        return {
            "artifact_digest": self.artifact_digest,
            "authorization_digest": self.authorization_digest,
            "execution_precommit_digest": self.execution_precommit_digest,
            "cold_replay_digest": self.cold_replay_digest,
            "command_result_digest": self.command_result_digest,
        }

    @classmethod
    def from_data(
        cls, value: object
    ) -> "ObjectBongardRubricCalibrationNominationBinding":
        raw = _exact_fields(
            value,
            {
                "artifact_digest",
                "authorization_digest",
                "execution_precommit_digest",
                "cold_replay_digest",
                "command_result_digest",
            },
            "semantic nomination binding",
        )
        result = cls(**raw)  # type: ignore[arg-type]
        if result.to_data() != dict(raw):
            raise ObjectBongardRubricCalibrationCommandError(
                "semantic nomination binding is not canonical"
            )
        return result


def _nomination_binding(
    source: ObjectBongardRubricCalibrationSource,
) -> ObjectBongardRubricCalibrationNominationBinding:
    artifact = source.nomination_artifact
    if (
        artifact is None
        or source.nomination_authorization_digest is None
        or source.nomination_precommit_digest is None
        or source.nomination_replay_digest is None
        or source.nomination_result_digest is None
    ):
        raise ObjectBongardRubricCalibrationCommandError(
            "calibration command requires one sealed semantic nomination"
        )
    return ObjectBongardRubricCalibrationNominationBinding(
        artifact.artifact_digest,
        source.nomination_authorization_digest,
        source.nomination_precommit_digest,
        source.nomination_replay_digest,
        source.nomination_result_digest,
    )


def _bind_verified_nomination(
    source: ObjectBongardRubricCalibrationSource,
    nomination: VerifiedObjectBongardRubricNomination,
) -> ObjectBongardRubricCalibrationSource:
    if not isinstance(nomination, VerifiedObjectBongardRubricNomination):
        raise TypeError("nomination must be a cold-verified typed predecessor")
    if not nomination.accepted:
        raise ObjectBongardRubricCalibrationCommandError(
            "semantic nomination predecessor was not accepted"
        )
    if nomination.source_digest != source.source_digest:
        raise ObjectBongardRubricCalibrationCommandError(
            "semantic nomination predecessor belongs to another source"
        )
    return _bind_object_bongard_rubric_calibration_nomination_content(
        source,
        nomination.artifact,
        nomination_authorization_digest=nomination.authorization_digest,
        nomination_precommit_digest=nomination.execution_precommit_digest,
        nomination_replay_digest=nomination.cold_replay_digest,
        nomination_result_digest=nomination.result_digest,
    )


def _raw_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _RAW_DIGEST.fullmatch(value) is None:
        raise ObjectBongardRubricCalibrationCommandError(
            f"{label} must be a lowercase raw SHA-256"
        )
    return value


def _address(value: object, label: str) -> str:
    if not isinstance(value, str) or _ADDRESS.fullmatch(value) is None:
        raise ObjectBongardRubricCalibrationCommandError(
            f"{label} must be a sha256: address"
        )
    return value


def _bounded_text(value: object, label: str, *, maximum: int = 1024) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
        or len(value.encode("utf-8", errors="strict")) > maximum
    ):
        raise ObjectBongardRubricCalibrationCommandError(
            f"{label} must be bounded exact text"
        )
    return value


def _exact_fields(
    value: object, expected: set[str], label: str
) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(not isinstance(key, str) for key in value)
        or set(value) != expected
    ):
        raise ObjectBongardRubricCalibrationCommandError(
            f"{label} fields differ"
        )
    return value


def _canonical_clone(value: object, label: str) -> Any:
    try:
        return json.loads(canonical_json(value).decode("utf-8", errors="strict"))
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise ObjectBongardRubricCalibrationCommandError(
            f"{label} is not finite canonical JSON"
        ) from exc


def _ensure_output_root(value: str | os.PathLike[str]) -> Path:
    requested = Path(value).expanduser()
    try:
        absolute = requested.absolute()
        missing: list[Path] = []
        cursor = absolute
        while not cursor.exists():
            missing.append(cursor)
            parent = cursor.parent
            if parent == cursor:
                raise ObjectBongardRubricCalibrationCommandError(
                    "cannot locate an existing output-root ancestor"
                )
            cursor = parent
        ancestor = cursor.resolve(strict=True)
        if cursor != ancestor or not ancestor.is_dir():
            raise ObjectBongardRubricCalibrationCommandError(
                "output-root ancestor must be one canonical directory"
            )
        for path in reversed(missing):
            os.mkdir(path, 0o700)
            _fsync_directory(path)
            _fsync_directory(path.parent)
        resolved = requested.resolve(strict=True)
        info = resolved.lstat()
    except OSError as exc:
        raise ObjectBongardRubricCalibrationCommandError(
            "cannot create or authenticate calibration output root"
        ) from exc
    if (
        requested.absolute() != resolved
        or not stat.S_ISDIR(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
    ):
        raise ObjectBongardRubricCalibrationCommandError(
            "calibration output root must be one canonical directory"
        )
    return resolved


def _ensure_directory(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or "/" in name or name in {".", ".."}:
        raise ObjectBongardRubricCalibrationCommandError(
            "output subdirectory name is invalid"
        )
    path = root / name
    try:
        created = False
        try:
            os.mkdir(path, 0o700)
            created = True
        except FileExistsError:
            pass
        info = path.lstat()
    except OSError as exc:
        raise ObjectBongardRubricCalibrationCommandError(
            "cannot create calibration output subdirectory"
        ) from exc
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise ObjectBongardRubricCalibrationCommandError(
            "calibration output subdirectory is not a real directory"
        )
    if created:
        _fsync_directory(path)
        _fsync_directory(root)
    return path


def _ensure_child_directory(parent: Path, name: str) -> Path:
    if (
        not isinstance(name, str)
        or not name
        or "/" in name
        or name in {".", ".."}
    ):
        raise ObjectBongardRubricCalibrationCommandError(
            "journal directory component is invalid"
        )
    path = parent / name
    try:
        created = False
        try:
            os.mkdir(path, 0o700)
            created = True
        except FileExistsError:
            pass
        info = path.lstat()
    except OSError as exc:
        raise ObjectBongardRubricCalibrationCommandError(
            "cannot create authorized journal directory"
        ) from exc
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise ObjectBongardRubricCalibrationCommandError(
            "authorized journal path contains a non-directory"
        )
    if created:
        _fsync_directory(path)
        _fsync_directory(parent)
    return path


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_canonical_record(path: Path, label: str) -> dict[str, Any]:
    if not hasattr(os, "O_NOFOLLOW"):
        raise ObjectBongardRubricCalibrationCommandError(
            "platform lacks no-follow record access"
        )
    try:
        before = path.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or before.st_nlink != 1
            or not 0 < before.st_size <= _MAX_RECORD_BYTES
        ):
            raise ObjectBongardRubricCalibrationCommandError(
                f"{label} is not bounded singly-linked regular data"
            )
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
        )
    except OSError as exc:
        raise ObjectBongardRubricCalibrationCommandError(
            f"cannot open {label}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        blocks: list[bytes] = []
        total = 0
        while True:
            block = os.read(descriptor, min(1024 * 1024, _MAX_RECORD_BYTES + 1 - total))
            if not block:
                break
            blocks.append(block)
            total += len(block)
            if total > _MAX_RECORD_BYTES:
                raise ObjectBongardRubricCalibrationCommandError(
                    f"{label} exceeds its byte bound"
                )
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda info: (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )
    if identity(before) != identity(opened) or identity(opened) != identity(after):
        raise ObjectBongardRubricCalibrationCommandError(
            f"{label} changed while being read"
        )
    payload = b"".join(blocks)
    try:
        decoded = json.loads(payload.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ObjectBongardRubricCalibrationCommandError(
            f"{label} is not canonical UTF-8 JSON"
        ) from exc
    if not isinstance(decoded, dict) or canonical_json(decoded) + b"\n" != payload:
        raise ObjectBongardRubricCalibrationCommandError(
            f"{label} bytes are not canonical"
        )
    return decoded


def _write_once(path: Path, value: Mapping[str, Any], label: str) -> tuple[Path, str]:
    frozen = _canonical_clone(value, label)
    if not isinstance(frozen, dict):
        raise ObjectBongardRubricCalibrationCommandError(f"{label} must be an object")
    payload = canonical_json(frozen) + b"\n"
    if path.exists():
        if _read_canonical_record(path, label) != frozen:
            raise ObjectBongardRubricCalibrationCommandError(
                f"existing {label} differs from the authorized record"
            )
        return path, hashlib.sha256(payload).hexdigest()
    if not hasattr(os, "O_NOFOLLOW"):
        raise ObjectBongardRubricCalibrationCommandError(
            "platform lacks safe exclusive record persistence"
        )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        if _read_canonical_record(path, label) != frozen:
            raise ObjectBongardRubricCalibrationCommandError(
                f"racing {label} differs from the authorized record"
            )
        return path, hashlib.sha256(payload).hexdigest()
    except OSError as exc:
        raise ObjectBongardRubricCalibrationCommandError(
            f"cannot exclusively persist {label}"
        ) from exc
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise ObjectBongardRubricCalibrationCommandError(
                    f"could not completely persist {label}"
                )
            offset += written
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)
    if _read_canonical_record(path, label) != frozen:
        raise ObjectBongardRubricCalibrationCommandError(
            f"persisted {label} failed exact reload"
        )
    return path, hashlib.sha256(payload).hexdigest()


def _source_digest_inventory() -> tuple[tuple[str, str], ...]:
    values = {
        "calibration_command_source_sha256": (
            object_bongard_rubric_calibration_command_source_digest()
        ),
        "calibration_driver_source_sha256": (
            object_bongard_rubric_calibration_source_digest()
        ),
        "nomination_command_source_sha256": (
            object_bongard_rubric_nomination_command_source_digest()
        ),
        "hypothesis_extractor_artifact_digest": (
            object_hypothesis_extractor_artifact_digest()
        ),
        "hypothesis_extractor_source_sha256": (
            object_hypothesis_extractor_source_digest()
        ),
        "lineage_extractor_artifact_digest": object_lineage_artifact_digest(),
        "lineage_extractor_source_sha256": object_lineage_source_digest(),
        "prototype_object_observer_source_sha256": (
            prototype_scene_observer_source_digest()
        ),
        "semantic_nomination_protocol_digest": (
            object_bongard_semantics_protocol_digest()
        ),
        "semantic_nomination_source_sha256": (
            object_bongard_semantics_source_digest()
        ),
        "rubric_observer_catalog_digest": (
            object_bongard_rubric_observer_catalog_digest()
        ),
        "rubric_observer_protocol_digest": (
            object_bongard_rubric_observer_protocol_digest()
        ),
        "rubric_observer_source_sha256": (
            object_bongard_rubric_observer_source_digest()
        ),
        "rubric_ordinal_scale_digest": (
            object_bongard_rubric_ordinal_scale_digest()
        ),
        "rubric_version_space_algorithm_digest": (
            object_bongard_rubric_version_space_algorithm_digest()
        ),
        "rubric_slate_algorithm_digest": (
            object_bongard_rubric_slate_algorithm_digest()
        ),
        "rubric_slate_source_sha256": object_bongard_rubric_slate_source_digest(),
        "transport_source_sha256": prototype_scene_transport_source_digest(),
        "turn_journal_source_sha256": object_bongard_turn_journal_source_digest(),
    }
    result = tuple(sorted(values.items()))
    for name, digest in result:
        _bounded_text(name, "source digest role")
        _raw_digest(digest, name)
    return result


@dataclass(frozen=True, slots=True)
class CalibrationSheetCommitment:
    sheet_index: int
    sheet_name: str
    prompt_sha256: str
    scene_png_sha256: str
    atlas_png_sha256: str
    output_schema_digest: str
    sheet_digest: str

    def __post_init__(self) -> None:
        if type(self.sheet_index) is not int or not 0 <= self.sheet_index < 32:
            raise ObjectBongardRubricCalibrationCommandError(
                "sheet index is outside the frozen atlas bound"
            )
        if not isinstance(self.sheet_name, str) or _SHEET_NAME.fullmatch(self.sheet_name) is None:
            raise ObjectBongardRubricCalibrationCommandError("sheet name is invalid")
        for name in (
            "prompt_sha256",
            "scene_png_sha256",
            "atlas_png_sha256",
            "output_schema_digest",
            "sheet_digest",
        ):
            _raw_digest(getattr(self, name), name)
        if self.sheet_digest != canonical_digest(self.content_data()):
            raise ObjectBongardRubricCalibrationCommandError(
                "sheet commitment digest differs"
            )

    def content_data(self) -> dict[str, object]:
        return {
            "sheet_index": self.sheet_index,
            "sheet_name": self.sheet_name,
            "prompt_sha256": self.prompt_sha256,
            "scene_png_sha256": self.scene_png_sha256,
            "atlas_png_sha256": self.atlas_png_sha256,
            "output_schema_digest": self.output_schema_digest,
        }

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "sheet_digest": self.sheet_digest}

    @classmethod
    def from_data(cls, value: object) -> "CalibrationSheetCommitment":
        raw = _exact_fields(
            value,
            {
                "sheet_index",
                "sheet_name",
                "prompt_sha256",
                "scene_png_sha256",
                "atlas_png_sha256",
                "output_schema_digest",
                "sheet_digest",
            },
            "sheet commitment",
        )
        result = cls(**raw)  # type: ignore[arg-type]
        if result.to_data() != dict(raw):
            raise ObjectBongardRubricCalibrationCommandError(
                "sheet commitment is not canonical"
            )
        return result


@dataclass(frozen=True, slots=True)
class CalibrationObservationJobCommitment:
    job_index: int
    rubric_spec_index: int
    panel_ordinal: int
    task_id: str
    panel_id: str
    panel_binding_digest: str
    rubric_spec_digest: str
    hypothesis_packet_digest: str
    lineage_packet_digest: str
    sheets: tuple[CalibrationSheetCommitment, ...]
    job_digest: str

    def __post_init__(self) -> None:
        if type(self.job_index) is not int or not 0 <= self.job_index < CALIBRATION_JOB_COUNT:
            raise ObjectBongardRubricCalibrationCommandError("job index is invalid")
        if self.rubric_spec_index not in (0, 1):
            raise ObjectBongardRubricCalibrationCommandError(
                "rubric spec index must be canonical rank zero or one"
            )
        if type(self.panel_ordinal) is not int or self.panel_ordinal < 0:
            raise ObjectBongardRubricCalibrationCommandError("panel ordinal is invalid")
        if not isinstance(self.task_id, str) or _TASK_ID.fullmatch(self.task_id) is None:
            raise ObjectBongardRubricCalibrationCommandError("task ID is invalid")
        if not isinstance(self.panel_id, str) or _PANEL_ID.fullmatch(self.panel_id) is None:
            raise ObjectBongardRubricCalibrationCommandError("panel ID is invalid")
        for name in (
            "panel_binding_digest",
            "rubric_spec_digest",
            "hypothesis_packet_digest",
            "lineage_packet_digest",
            "job_digest",
        ):
            _raw_digest(getattr(self, name), name)
        if (
            not isinstance(self.sheets, tuple)
            or not self.sheets
            or tuple(item.sheet_index for item in self.sheets) != tuple(range(len(self.sheets)))
            or self.job_digest != canonical_digest(self.content_data())
        ):
            raise ObjectBongardRubricCalibrationCommandError(
                "job sheet inventory or digest differs"
            )

    def content_data(self) -> dict[str, object]:
        return {
            "job_index": self.job_index,
            "rubric_spec_index": self.rubric_spec_index,
            "panel_ordinal": self.panel_ordinal,
            "task_id": self.task_id,
            "panel_id": self.panel_id,
            "panel_binding_digest": self.panel_binding_digest,
            "rubric_spec_digest": self.rubric_spec_digest,
            "hypothesis_packet_digest": self.hypothesis_packet_digest,
            "lineage_packet_digest": self.lineage_packet_digest,
            "sheets": [item.to_data() for item in self.sheets],
        }

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "job_digest": self.job_digest}

    @classmethod
    def from_data(cls, value: object) -> "CalibrationObservationJobCommitment":
        raw = _exact_fields(
            value,
            {
                "job_index",
                "rubric_spec_index",
                "panel_ordinal",
                "task_id",
                "panel_id",
                "panel_binding_digest",
                "rubric_spec_digest",
                "hypothesis_packet_digest",
                "lineage_packet_digest",
                "sheets",
                "job_digest",
            },
            "observation job commitment",
        )
        if not isinstance(raw["sheets"], list):
            raise ObjectBongardRubricCalibrationCommandError(
                "observation job sheets must be a list"
            )
        result = cls(
            raw["job_index"],
            raw["rubric_spec_index"],
            raw["panel_ordinal"],
            raw["task_id"],
            raw["panel_id"],
            raw["panel_binding_digest"],
            raw["rubric_spec_digest"],
            raw["hypothesis_packet_digest"],
            raw["lineage_packet_digest"],
            tuple(CalibrationSheetCommitment.from_data(item) for item in raw["sheets"]),
            raw["job_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardRubricCalibrationCommandError(
                "observation job commitment is not canonical"
            )
        return result


def _historical_authority_data(
    source_directory: str | os.PathLike[str],
    source: ObjectBongardRubricCalibrationSource,
) -> dict[str, object]:
    """Authenticate the prior release records that authorized these pixels."""

    object_root = Path(source_directory).expanduser().resolve(strict=True)
    campaign_root = object_root.parent
    authorization_path = (
        campaign_root
        / "authorizations"
        / (HISTORICAL_RELEASE_AUTHORIZATION_RECORD_DIGEST.removeprefix("sha256:") + ".json")
    )
    precommit_path = (
        object_root
        / "execution_precommit"
        / (HISTORICAL_EXECUTION_PRECOMMIT_FILE_SHA256 + ".json")
    )
    authorization = _read_canonical_record(
        authorization_path, "historical release authorization"
    )
    precommit = _read_canonical_record(
        precommit_path, "historical execution precommit"
    )
    authorization_file_sha256 = hashlib.sha256(
        canonical_json(authorization) + b"\n"
    ).hexdigest()
    precommit_file_sha256 = hashlib.sha256(
        canonical_json(precommit) + b"\n"
    ).hexdigest()
    selected_task_ids = authorization.get("selected_task_ids")
    exposure_successor_receipt = authorization.get("exposure_successor_receipt")
    execution_precommit_receipt = authorization.get("execution_precommit_receipt")
    cohort = precommit.get("cohort")
    if (
        authorization_file_sha256
        != HISTORICAL_RELEASE_AUTHORIZATION_FILE_SHA256
        or authorization.get("schema")
        != "gkm.bongard-prototype-pair-release-authorization.v1"
        or authorization.get("record_digest")
        != HISTORICAL_RELEASE_AUTHORIZATION_RECORD_DIGEST
        or authorization.get("execution_precommit_digest")
        != HISTORICAL_EXECUTION_PRECOMMIT_RECORD_DIGEST
        or authorization.get("exposure_successor_digest")
        != HISTORICAL_EXPOSURE_SUCCESSOR_DIGEST
        or authorization.get("plan_digest") != HISTORICAL_COHORT_PLAN_DIGEST
        or authorization.get("phase") != "prototype_pair_selected_task_release"
        or authorization.get("actor") != "prototype-pair-campaign-cli"
        or authorization.get("purpose")
        != (
            "release exactly the 31 preselected exact-unused TRAIN task "
            "identities for prototype-conditioned targeted engineering"
        )
        or authorization.get("exactly_one_successor_event") is not True
        or authorization.get("predecessor_exact_unused_verified") is not True
        or authorization.get("successor_persisted_and_reloaded_before_authorization")
        is not True
        or not isinstance(selected_task_ids, list)
        or any(not isinstance(item, str) for item in selected_task_ids)
        or len(selected_task_ids) != 31
        or len(set(selected_task_ids)) != 31
        or authorization.get("selected_task_count") != len(selected_task_ids)
        or not {item.task_id for item in source.panels} <= set(selected_task_ids)
        or not isinstance(exposure_successor_receipt, Mapping)
        or exposure_successor_receipt.get("object_record_digest")
        != HISTORICAL_EXPOSURE_SUCCESSOR_DIGEST
        or exposure_successor_receipt.get("canonical_bytes_digest")
        != "sha256:1bcde18e387539f13c4006b4a147e61c75feacb86bb031f10a6e8ba3412fe48d"
        or not isinstance(execution_precommit_receipt, Mapping)
        or execution_precommit_receipt.get("object_record_digest")
        != HISTORICAL_EXECUTION_PRECOMMIT_RECORD_DIGEST
        or execution_precommit_receipt.get("canonical_bytes_digest")
        != "sha256:" + HISTORICAL_EXECUTION_PRECOMMIT_FILE_SHA256
        or precommit_file_sha256 != HISTORICAL_EXECUTION_PRECOMMIT_FILE_SHA256
        or precommit.get("schema")
        != "gkm.bongard-prototype-pair-execution-precommit.v4"
        or precommit.get("record_digest")
        != HISTORICAL_EXECUTION_PRECOMMIT_RECORD_DIGEST
        or not isinstance(cohort, Mapping)
        or cohort.get("plan_digest") != HISTORICAL_COHORT_PLAN_DIGEST
    ):
        raise ObjectBongardRubricCalibrationCommandError(
            "historical release authorization/precommit binding differs"
        )
    for panel in source.panels:
        released_path = (
            object_root
            / "released_panel"
            / f"{panel.released_file_sha256}.json"
        )
        released = _read_canonical_record(
            released_path, "historical released panel"
        )
        released_file_sha256 = hashlib.sha256(
            canonical_json(released) + b"\n"
        ).hexdigest()
        if (
            released_file_sha256 != panel.released_file_sha256
            or released.get("schema") != "gkm.bongard-released-panel.v1"
            or released.get("panel_id") != panel.panel_id
            or released.get("record_digest") != panel.released_record_digest
            or released.get("exact_png_digest") != "sha256:" + panel.png_sha256
            or released.get("execution_precommit_digest")
            != HISTORICAL_EXECUTION_PRECOMMIT_RECORD_DIGEST
            or released.get("exposure_successor_digest")
            != HISTORICAL_EXPOSURE_SUCCESSOR_DIGEST
            or released.get("released_after_durable_exposure") is not True
        ):
            raise ObjectBongardRubricCalibrationCommandError(
                "released calibration panel differs from historical authority"
            )
    return {
        "historical_release_authorization_record_digest": (
            HISTORICAL_RELEASE_AUTHORIZATION_RECORD_DIGEST
        ),
        "historical_release_authorization_file_sha256": (
            authorization_file_sha256
        ),
        "historical_execution_precommit_record_digest": (
            HISTORICAL_EXECUTION_PRECOMMIT_RECORD_DIGEST
        ),
        "historical_execution_precommit_file_sha256": precommit_file_sha256,
        "historical_cohort_plan_digest": HISTORICAL_COHORT_PLAN_DIGEST,
        "historical_exposure_successor_digest": (
            HISTORICAL_EXPOSURE_SUCCESSOR_DIGEST
        ),
        "released_panel_parent_binding_count": len(source.panels),
        "historical_release_reused": True,
        "new_release_authorization_created": False,
    }


def _build_job_inventory(
    source: ObjectBongardRubricCalibrationSource,
) -> tuple[CalibrationObservationJobCommitment, ...]:
    schema_digest = canonical_digest(object_bongard_rubric_observer_output_schema())
    jobs: list[CalibrationObservationJobCommitment] = []
    for spec_index, spec in enumerate(source.rubric_specs):
        for panel in source.panels:
            rendered = dict(
                render_object_hypothesis_atlas(
                    panel.hypothesis_packet, panel.exact_png_bytes
                )
            )
            sheets: list[CalibrationSheetCommitment] = []
            for sheet in panel.hypothesis_packet.atlas_sheets:
                content = {
                    "sheet_index": sheet.sheet_index,
                    "sheet_name": sheet.name,
                    "prompt_sha256": hashlib.sha256(
                        object_bongard_rubric_observer_prompt(spec, sheet).encode(
                            "utf-8", errors="strict"
                        )
                    ).hexdigest(),
                    "scene_png_sha256": panel.png_sha256,
                    "atlas_png_sha256": hashlib.sha256(
                        rendered[sheet.name]
                    ).hexdigest(),
                    "output_schema_digest": schema_digest,
                }
                sheets.append(
                    CalibrationSheetCommitment(
                        **content,
                        sheet_digest=canonical_digest(content),
                    )
                )
            job_content = {
                "job_index": len(jobs),
                "rubric_spec_index": spec_index,
                "panel_ordinal": panel.ordinal,
                "task_id": panel.task_id,
                "panel_id": panel.panel_id,
                "panel_binding_digest": panel.panel_binding_digest,
                "rubric_spec_digest": spec.spec_digest,
                "hypothesis_packet_digest": panel.hypothesis_packet.digest(),
                "lineage_packet_digest": panel.lineage_packet.digest(),
                "sheets": [item.to_data() for item in sheets],
            }
            jobs.append(
                CalibrationObservationJobCommitment(
                    job_index=job_content["job_index"],  # type: ignore[arg-type]
                    rubric_spec_index=spec_index,
                    panel_ordinal=panel.ordinal,
                    task_id=panel.task_id,
                    panel_id=panel.panel_id,
                    panel_binding_digest=panel.panel_binding_digest,
                    rubric_spec_digest=spec.spec_digest,
                    hypothesis_packet_digest=panel.hypothesis_packet.digest(),
                    lineage_packet_digest=panel.lineage_packet.digest(),
                    sheets=tuple(sheets),
                    job_digest=canonical_digest(job_content),
                )
            )
    if (
        len(jobs) != CALIBRATION_JOB_COUNT
        or sum(len(item.sheets) for item in jobs)
        != CALIBRATION_SHEET_JOURNAL_COUNT
        or tuple(item.rubric_spec_index for item in jobs)
        != (0,) * 12 + (1,) * 12
        or tuple(item.panel_ordinal for item in jobs)
        != tuple(item.ordinal for item in source.panels) * 2
        or len({item.panel_binding_digest for item in jobs}) != 12
    ):
        raise ObjectBongardRubricCalibrationCommandError(
            "calibration job/sheet inventory is not exactly 24/30"
        )
    return tuple(jobs)


def _authorization_content(
    value: "ObjectBongardRubricCalibrationAuthorization",
) -> dict[str, object]:
    return {
        "schema": OBJECT_RUBRIC_CALIBRATION_AUTHORIZATION_SCHEMA,
        "command_id": OBJECT_RUBRIC_CALIBRATION_COMMAND_ID,
        "calibration_algorithm_id": OBJECT_RUBRIC_CALIBRATION_ALGORITHM_ID,
        "source_digest": value.source_digest,
        "nomination_binding": value.nomination_binding.to_data(),
        "source_digests": [
            {"role": role, "sha256": digest}
            for role, digest in value.source_digests
        ],
        "historical_authority": dict(value.historical_authority),
        "jobs": [item.to_data() for item in value.jobs],
        "job_inventory_digest": value.job_inventory_digest,
        "observation_job_count": CALIBRATION_JOB_COUNT,
        "sheet_journal_count": CALIBRATION_SHEET_JOURNAL_COUNT,
        "runtime_policy": {
            "model": CALIBRATION_MODEL,
            "reasoning_effort": CALIBRATION_REASONING_EFFORT,
            "minutes": value.minutes,
            "verbose": value.verbose,
            "executable": value.executable,
            "expected_launcher_sha256": value.expected_launcher_sha256,
            "parallel_workers": CALIBRATION_PARALLEL_WORKERS,
        },
        "acceptance_rule": CALIBRATION_ACCEPTANCE_RULE,
        "threshold_tuning_authorized": False,
        "preferred_candidate_selection_authorized": False,
        "query_pixels_authorized": False,
        "fresh_broad_cohort_pixels_authorized": False,
        "fresh_release_authorization_authorized": False,
        "labels_visible_to_observer": False,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardRubricCalibrationAuthorization:
    source_digest: str
    nomination_binding: ObjectBongardRubricCalibrationNominationBinding
    source_digests: tuple[tuple[str, str], ...]
    historical_authority: Mapping[str, object]
    jobs: tuple[CalibrationObservationJobCommitment, ...]
    job_inventory_digest: str
    minutes: int
    verbose: bool
    executable: str
    expected_launcher_sha256: str
    authorization_digest: str

    def __post_init__(self) -> None:
        _raw_digest(self.source_digest, "calibration source digest")
        if not isinstance(
            self.nomination_binding,
            ObjectBongardRubricCalibrationNominationBinding,
        ):
            raise TypeError("nomination_binding has the wrong type")
        if (
            not isinstance(self.source_digests, tuple)
            or self.source_digests != tuple(sorted(self.source_digests))
            or len({role for role, _digest in self.source_digests})
            != len(self.source_digests)
        ):
            raise ObjectBongardRubricCalibrationCommandError(
                "authorization source digest inventory differs"
            )
        for role, digest in self.source_digests:
            _bounded_text(role, "source digest role")
            _raw_digest(digest, role)
        expected_historical = {
            "historical_release_authorization_record_digest": (
                HISTORICAL_RELEASE_AUTHORIZATION_RECORD_DIGEST
            ),
            "historical_release_authorization_file_sha256": (
                HISTORICAL_RELEASE_AUTHORIZATION_FILE_SHA256
            ),
            "historical_execution_precommit_record_digest": (
                HISTORICAL_EXECUTION_PRECOMMIT_RECORD_DIGEST
            ),
            "historical_execution_precommit_file_sha256": (
                HISTORICAL_EXECUTION_PRECOMMIT_FILE_SHA256
            ),
            "historical_cohort_plan_digest": HISTORICAL_COHORT_PLAN_DIGEST,
            "historical_exposure_successor_digest": (
                HISTORICAL_EXPOSURE_SUCCESSOR_DIGEST
            ),
            "released_panel_parent_binding_count": 12,
            "historical_release_reused": True,
            "new_release_authorization_created": False,
        }
        if dict(self.historical_authority) != expected_historical:
            raise ObjectBongardRubricCalibrationCommandError(
                "historical authority inventory differs"
            )
        if (
            not isinstance(self.jobs, tuple)
            or len(self.jobs) != CALIBRATION_JOB_COUNT
            or tuple(item.job_index for item in self.jobs)
            != tuple(range(CALIBRATION_JOB_COUNT))
            or sum(len(item.sheets) for item in self.jobs)
            != CALIBRATION_SHEET_JOURNAL_COUNT
            or len(
                {(item.panel_binding_digest, item.rubric_spec_digest) for item in self.jobs}
            )
            != CALIBRATION_JOB_COUNT
        ):
            raise ObjectBongardRubricCalibrationCommandError(
                "authorization must contain exactly 24 jobs and 30 sheets"
            )
        _raw_digest(self.job_inventory_digest, "job inventory digest")
        if self.job_inventory_digest != canonical_digest(
            [item.to_data() for item in self.jobs]
        ):
            raise ObjectBongardRubricCalibrationCommandError(
                "authorization job inventory digest differs"
            )
        if type(self.minutes) is not int or not 1 <= self.minutes <= 120:
            raise ObjectBongardRubricCalibrationCommandError(
                "authorization minutes must lie in 1..120"
            )
        if not isinstance(self.verbose, bool):
            raise ObjectBongardRubricCalibrationCommandError(
                "authorization verbosity is invalid"
            )
        _bounded_text(self.executable, "Codex executable")
        _raw_digest(self.expected_launcher_sha256, "expected launcher digest")
        _address(self.authorization_digest, "authorization digest")
        expected = "sha256:" + canonical_digest(_authorization_content(self))
        if self.authorization_digest != expected:
            raise ObjectBongardRubricCalibrationCommandError(
                "authorization digest differs"
            )

    def to_data(self) -> dict[str, object]:
        return {
            **_authorization_content(self),
            "authorization_digest": self.authorization_digest,
        }

    @classmethod
    def from_data(
        cls, value: object
    ) -> "ObjectBongardRubricCalibrationAuthorization":
        expected = {
            "schema",
            "command_id",
            "calibration_algorithm_id",
            "source_digest",
            "nomination_binding",
            "source_digests",
            "historical_authority",
            "jobs",
            "job_inventory_digest",
            "observation_job_count",
            "sheet_journal_count",
            "runtime_policy",
            "acceptance_rule",
            "threshold_tuning_authorized",
            "preferred_candidate_selection_authorized",
            "query_pixels_authorized",
            "fresh_broad_cohort_pixels_authorized",
            "fresh_release_authorization_authorized",
            "labels_visible_to_observer",
            *_authority_data(),
            "authorization_digest",
        }
        raw = _exact_fields(value, expected, "calibration authorization")
        if (
            raw["schema"] != OBJECT_RUBRIC_CALIBRATION_AUTHORIZATION_SCHEMA
            or raw["command_id"] != OBJECT_RUBRIC_CALIBRATION_COMMAND_ID
            or raw["calibration_algorithm_id"]
            != OBJECT_RUBRIC_CALIBRATION_ALGORITHM_ID
            or raw["observation_job_count"] != CALIBRATION_JOB_COUNT
            or raw["sheet_journal_count"] != CALIBRATION_SHEET_JOURNAL_COUNT
            or raw["acceptance_rule"] != CALIBRATION_ACCEPTANCE_RULE
            or raw["threshold_tuning_authorized"] is not False
            or raw["preferred_candidate_selection_authorized"] is not False
            or raw["query_pixels_authorized"] is not False
            or raw["fresh_broad_cohort_pixels_authorized"] is not False
            or raw["fresh_release_authorization_authorized"] is not False
            or raw["labels_visible_to_observer"] is not False
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["source_digests"], list)
            or not isinstance(raw["nomination_binding"], Mapping)
            or not isinstance(raw["historical_authority"], Mapping)
            or not isinstance(raw["jobs"], list)
        ):
            raise ObjectBongardRubricCalibrationCommandError(
                "calibration authorization policy differs"
            )
        runtime = _exact_fields(
            raw["runtime_policy"],
            {
                "model",
                "reasoning_effort",
                "minutes",
                "verbose",
                "executable",
                "expected_launcher_sha256",
                "parallel_workers",
            },
            "authorization runtime policy",
        )
        if (
            runtime["model"] != CALIBRATION_MODEL
            or runtime["reasoning_effort"] != CALIBRATION_REASONING_EFFORT
            or runtime["parallel_workers"] != CALIBRATION_PARALLEL_WORKERS
        ):
            raise ObjectBongardRubricCalibrationCommandError(
                "authorization runtime selectors differ"
            )
        source_digests: list[tuple[str, str]] = []
        for row in raw["source_digests"]:
            item = _exact_fields(row, {"role", "sha256"}, "source digest row")
            source_digests.append((item["role"], item["sha256"]))
        result = cls(
            raw["source_digest"],
            ObjectBongardRubricCalibrationNominationBinding.from_data(
                raw["nomination_binding"]
            ),
            tuple(source_digests),
            _canonical_clone(raw["historical_authority"], "historical authority"),
            tuple(CalibrationObservationJobCommitment.from_data(item) for item in raw["jobs"]),
            raw["job_inventory_digest"],
            runtime["minutes"],
            runtime["verbose"],
            runtime["executable"],
            runtime["expected_launcher_sha256"],
            raw["authorization_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardRubricCalibrationCommandError(
                "calibration authorization is not canonical"
            )
        return result


def prepare_object_bongard_rubric_calibration_authorization(
    source: ObjectBongardRubricCalibrationSource,
    *,
    source_directory: str | os.PathLike[str] = (
        DEFAULT_OBJECT_RUBRIC_CALIBRATION_SOURCE
    ),
    minutes: int = CALIBRATION_MINUTES,
    verbose: bool = False,
    executable: str = DEFAULT_CODEX_EXECUTABLE,
    expected_launcher_sha256: str = DEFAULT_CALIBRATION_CODEX_LAUNCHER_SHA256,
) -> ObjectBongardRubricCalibrationAuthorization:
    if not isinstance(source, ObjectBongardRubricCalibrationSource):
        raise TypeError("source must be ObjectBongardRubricCalibrationSource")
    jobs = _build_job_inventory(source)
    values = {
        "source_digest": source.source_digest,
        "nomination_binding": _nomination_binding(source),
        "source_digests": _source_digest_inventory(),
        "historical_authority": _historical_authority_data(
            source_directory, source
        ),
        "jobs": jobs,
        "job_inventory_digest": canonical_digest([item.to_data() for item in jobs]),
        "minutes": minutes,
        "verbose": verbose,
        "executable": executable,
        "expected_launcher_sha256": expected_launcher_sha256,
    }
    provisional = object.__new__(ObjectBongardRubricCalibrationAuthorization)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectBongardRubricCalibrationAuthorization(
        **values,
        authorization_digest="sha256:" + canonical_digest(
            _authorization_content(provisional)
        ),
    )


def verify_object_bongard_rubric_calibration_authorization(
    authorization: ObjectBongardRubricCalibrationAuthorization,
    source: ObjectBongardRubricCalibrationSource,
    *,
    source_directory: str | os.PathLike[str] = (
        DEFAULT_OBJECT_RUBRIC_CALIBRATION_SOURCE
    ),
) -> ObjectBongardRubricCalibrationAuthorization:
    if not isinstance(authorization, ObjectBongardRubricCalibrationAuthorization):
        raise TypeError("authorization must be the typed authorization")
    replayed = prepare_object_bongard_rubric_calibration_authorization(
        source,
        source_directory=source_directory,
        minutes=authorization.minutes,
        verbose=authorization.verbose,
        executable=authorization.executable,
        expected_launcher_sha256=authorization.expected_launcher_sha256,
    )
    if replayed != authorization:
        raise ObjectBongardRubricCalibrationCommandError(
            "calibration authorization differs from exact source replay"
        )
    return authorization


def _encode_exact_bytes(value: bytes, label: str) -> str:
    if not isinstance(value, bytes):
        raise ObjectBongardRubricCalibrationCommandError(
            f"{label} must contain exact bytes"
        )
    return base64.b64encode(value).decode("ascii")


def _decode_exact_bytes(value: object, label: str) -> bytes:
    if not isinstance(value, str):
        raise ObjectBongardRubricCalibrationCommandError(
            f"{label} base64 is invalid"
        )
    try:
        decoded = base64.b64decode(value.encode("ascii"), validate=True)
    except (UnicodeError, ValueError) as exc:
        raise ObjectBongardRubricCalibrationCommandError(
            f"{label} base64 is invalid"
        ) from exc
    if _encode_exact_bytes(decoded, label) != value:
        raise ObjectBongardRubricCalibrationCommandError(
            f"{label} base64 is not canonical"
        )
    return decoded


def _precommit_content(
    value: "ObjectBongardRubricCalibrationExecutionPrecommit",
) -> dict[str, object]:
    runtime = value.runtime
    cache = runtime.cloud_policy_cache_snapshot
    if not isinstance(cache, CloudPolicyCacheSnapshot):
        raise ObjectBongardRubricCalibrationCommandError(
            "precommit requires an exact cloud-policy snapshot"
        )
    return {
        "schema": OBJECT_RUBRIC_CALIBRATION_PRECOMMIT_SCHEMA,
        "command_id": OBJECT_RUBRIC_CALIBRATION_COMMAND_ID,
        "authorization_digest": value.authorization_digest,
        "source_digest": value.source_digest,
        "nomination_binding": value.nomination_binding.to_data(),
        "source_digests": [
            {"role": role, "sha256": digest}
            for role, digest in value.source_digests
        ],
        "job_inventory_digest": value.job_inventory_digest,
        "observation_job_count": CALIBRATION_JOB_COUNT,
        "sheet_journal_count": CALIBRATION_SHEET_JOURNAL_COUNT,
        "parallel_workers": CALIBRATION_PARALLEL_WORKERS,
        "runtime_binding": runtime.binding,
        "cloud_policy_cache_snapshot_base64": (
            None if cache.data is None else _encode_exact_bytes(cache.data, "policy cache")
        ),
        "model_catalog_snapshot_base64": _encode_exact_bytes(
            runtime.model_catalog_snapshot.data, "model catalog"
        ),
        "no_tools_attestation": runtime.no_tools_attestation.to_dict(),
        "launcher_fingerprint": dict(value.launcher_fingerprint),
        "acceptance_rule": CALIBRATION_ACCEPTANCE_RULE,
        "threshold_tuning_authorized": False,
        "preferred_candidate_selection_authorized": False,
        "query_pixels_authorized": False,
        "fresh_broad_cohort_pixels_authorized": False,
        "fresh_release_authorization_authorized": False,
        "labels_visible_to_observer": False,
        "authorization_and_precommit_required_before_inference": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardRubricCalibrationExecutionPrecommit:
    authorization_digest: str
    source_digest: str
    nomination_binding: ObjectBongardRubricCalibrationNominationBinding
    source_digests: tuple[tuple[str, str], ...]
    job_inventory_digest: str
    runtime: ObjectBongardTurnRuntime
    launcher_fingerprint: Mapping[str, str]
    precommit_digest: str

    def __post_init__(self) -> None:
        _address(self.authorization_digest, "authorization digest")
        _raw_digest(self.source_digest, "precommit source digest")
        if not isinstance(
            self.nomination_binding,
            ObjectBongardRubricCalibrationNominationBinding,
        ):
            raise TypeError("nomination_binding has the wrong type")
        _raw_digest(self.job_inventory_digest, "precommit job inventory digest")
        if (
            not isinstance(self.source_digests, tuple)
            or self.source_digests != tuple(sorted(self.source_digests))
        ):
            raise ObjectBongardRubricCalibrationCommandError(
                "precommit source digest inventory differs"
            )
        for role, digest in self.source_digests:
            _bounded_text(role, "source digest role")
            _raw_digest(digest, role)
        if not isinstance(self.runtime, ObjectBongardTurnRuntime):
            raise TypeError("runtime must be ObjectBongardTurnRuntime")
        if (
            self.runtime.model != CALIBRATION_MODEL
            or self.runtime.reasoning_effort != CALIBRATION_REASONING_EFFORT
            or not isinstance(
                self.runtime.cloud_policy_cache_snapshot,
                CloudPolicyCacheSnapshot,
            )
        ):
            raise ObjectBongardRubricCalibrationCommandError(
                "precommit runtime is outside the calibration policy"
            )
        fingerprint = _exact_fields(
            self.launcher_fingerprint,
            {"version", "launcher_digest"},
            "launcher fingerprint",
        )
        if (
            fingerprint["version"] != PINNED_CODEX_CLI_VERSION
            or fingerprint["launcher_digest"]
            != self.runtime.expected_launcher_digest
        ):
            raise ObjectBongardRubricCalibrationCommandError(
                "authenticated launcher fingerprint differs"
            )
        _address(self.precommit_digest, "execution precommit digest")
        if self.precommit_digest != "sha256:" + canonical_digest(
            _precommit_content(self)
        ):
            raise ObjectBongardRubricCalibrationCommandError(
                "execution precommit digest differs"
            )

    def to_data(self) -> dict[str, object]:
        return {
            **_precommit_content(self),
            "precommit_digest": self.precommit_digest,
        }

    @classmethod
    def from_data(
        cls, value: object
    ) -> "ObjectBongardRubricCalibrationExecutionPrecommit":
        expected = {
            "schema",
            "command_id",
            "authorization_digest",
            "source_digest",
            "nomination_binding",
            "source_digests",
            "job_inventory_digest",
            "observation_job_count",
            "sheet_journal_count",
            "parallel_workers",
            "runtime_binding",
            "cloud_policy_cache_snapshot_base64",
            "model_catalog_snapshot_base64",
            "no_tools_attestation",
            "launcher_fingerprint",
            "acceptance_rule",
            "threshold_tuning_authorized",
            "preferred_candidate_selection_authorized",
            "query_pixels_authorized",
            "fresh_broad_cohort_pixels_authorized",
            "fresh_release_authorization_authorized",
            "labels_visible_to_observer",
            "authorization_and_precommit_required_before_inference",
            *_authority_data(),
            "precommit_digest",
        }
        raw = _exact_fields(value, expected, "calibration execution precommit")
        if (
            raw["schema"] != OBJECT_RUBRIC_CALIBRATION_PRECOMMIT_SCHEMA
            or raw["command_id"] != OBJECT_RUBRIC_CALIBRATION_COMMAND_ID
            or raw["observation_job_count"] != CALIBRATION_JOB_COUNT
            or raw["sheet_journal_count"] != CALIBRATION_SHEET_JOURNAL_COUNT
            or raw["parallel_workers"] != CALIBRATION_PARALLEL_WORKERS
            or raw["acceptance_rule"] != CALIBRATION_ACCEPTANCE_RULE
            or raw["threshold_tuning_authorized"] is not False
            or raw["preferred_candidate_selection_authorized"] is not False
            or raw["query_pixels_authorized"] is not False
            or raw["fresh_broad_cohort_pixels_authorized"] is not False
            or raw["fresh_release_authorization_authorized"] is not False
            or raw["labels_visible_to_observer"] is not False
            or raw["authorization_and_precommit_required_before_inference"]
            is not True
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["source_digests"], list)
            or not isinstance(raw["nomination_binding"], Mapping)
            or not isinstance(raw["runtime_binding"], Mapping)
            or not isinstance(raw["no_tools_attestation"], Mapping)
            or not isinstance(raw["launcher_fingerprint"], Mapping)
        ):
            raise ObjectBongardRubricCalibrationCommandError(
                "calibration execution precommit policy differs"
            )
        source_digests: list[tuple[str, str]] = []
        for row in raw["source_digests"]:
            item = _exact_fields(row, {"role", "sha256"}, "source digest row")
            source_digests.append((item["role"], item["sha256"]))
        cache_value = raw["cloud_policy_cache_snapshot_base64"]
        cache = CloudPolicyCacheSnapshot(
            None
            if cache_value is None
            else _decode_exact_bytes(cache_value, "policy cache")
        )
        catalog = CodexModelCatalogSnapshot(
            _decode_exact_bytes(raw["model_catalog_snapshot_base64"], "model catalog")
        )
        attestation = CodexNoToolsAttestation.from_mapping(
            raw["no_tools_attestation"]
        )
        binding = _exact_fields(
            raw["runtime_binding"],
            {
                "model",
                "reasoning_effort",
                "minutes",
                "verbose",
                "executable",
                "cloud_policy_cache_snapshot_present",
                "cloud_policy_cache_binding",
                "model_catalog_raw_digest",
                "model_catalog_canonical_digest",
                "expected_launcher_digest",
                "no_tools_attestation_digest",
                "transport_source_digest",
            },
            "runtime binding",
        )
        runtime = ObjectBongardTurnRuntime(
            model=binding["model"],
            reasoning_effort=binding["reasoning_effort"],
            minutes=binding["minutes"],
            verbose=binding["verbose"],
            executable=binding["executable"],
            cloud_policy_cache_snapshot=cache,
            model_catalog_snapshot=catalog,
            expected_launcher_digest=binding["expected_launcher_digest"],
            no_tools_attestation=attestation,
            transport_source_digest=binding["transport_source_digest"],
        )
        if runtime.binding != dict(binding):
            raise ObjectBongardRubricCalibrationCommandError(
                "serialized runtime binding differs from exact snapshots"
            )
        result = cls(
            raw["authorization_digest"],
            raw["source_digest"],
            ObjectBongardRubricCalibrationNominationBinding.from_data(
                raw["nomination_binding"]
            ),
            tuple(source_digests),
            raw["job_inventory_digest"],
            runtime,
            _canonical_clone(raw["launcher_fingerprint"], "launcher fingerprint"),
            raw["precommit_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardRubricCalibrationCommandError(
                "calibration execution precommit is not canonical"
            )
        return result


def prepare_object_bongard_rubric_calibration_execution_precommit(
    authorization: ObjectBongardRubricCalibrationAuthorization,
    source: ObjectBongardRubricCalibrationSource,
    *,
    cloud_policy_cache_snapshotter: Callable[[], CloudPolicyCacheSnapshot] = (
        snapshot_cloud_policy_cache
    ),
    model_catalog_snapshotter: Callable[[], CodexModelCatalogSnapshot] = (
        snapshot_pinned_model_catalog
    ),
    launcher_fingerprinter: LauncherFingerprinter = (
        codex_cli_authenticated_fingerprint
    ),
    runtime_attester: RuntimeAttester = attest_codex_no_tools,
) -> ObjectBongardRubricCalibrationExecutionPrecommit:
    if not isinstance(authorization, ObjectBongardRubricCalibrationAuthorization):
        raise TypeError("authorization must be the typed authorization")
    if source.source_digest != authorization.source_digest:
        raise ObjectBongardRubricCalibrationCommandError(
            "precommit source differs from authorization"
        )
    if authorization.source_digests != _source_digest_inventory():
        raise ObjectBongardRubricCalibrationCommandError(
            "authoritative Python sources changed after authorization"
        )
    cache = cloud_policy_cache_snapshotter()
    catalog = model_catalog_snapshotter()
    if not isinstance(cache, CloudPolicyCacheSnapshot) or not isinstance(
        catalog, CodexModelCatalogSnapshot
    ):
        raise ObjectBongardRubricCalibrationCommandError(
            "runtime snapshotter returned an invalid object"
        )
    fingerprint = launcher_fingerprinter(
        authorization.executable,
        expected_launcher_digest=authorization.expected_launcher_sha256,
    )
    attestation = runtime_attester(
        executable=authorization.executable,
        expected_launcher_digest=authorization.expected_launcher_sha256,
        model_catalog_snapshot=catalog,
        cloud_policy_cache_snapshot=cache,
    )
    if not isinstance(attestation, CodexNoToolsAttestation):
        raise ObjectBongardRubricCalibrationCommandError(
            "runtime attester returned an invalid object"
        )
    runtime = ObjectBongardTurnRuntime(
        model=CALIBRATION_MODEL,
        reasoning_effort=CALIBRATION_REASONING_EFFORT,
        minutes=authorization.minutes,
        verbose=authorization.verbose,
        executable=authorization.executable,
        cloud_policy_cache_snapshot=cache,
        model_catalog_snapshot=catalog,
        expected_launcher_digest=authorization.expected_launcher_sha256,
        no_tools_attestation=attestation,
        transport_source_digest=prototype_scene_transport_source_digest(),
    )
    values = {
        "authorization_digest": authorization.authorization_digest,
        "source_digest": authorization.source_digest,
        "nomination_binding": authorization.nomination_binding,
        "source_digests": authorization.source_digests,
        "job_inventory_digest": authorization.job_inventory_digest,
        "runtime": runtime,
        "launcher_fingerprint": _canonical_clone(
            fingerprint, "launcher fingerprint"
        ),
    }
    provisional = object.__new__(
        ObjectBongardRubricCalibrationExecutionPrecommit
    )
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectBongardRubricCalibrationExecutionPrecommit(
        **values,
        precommit_digest="sha256:" + canonical_digest(
            _precommit_content(provisional)
        ),
    )


def verify_object_bongard_rubric_calibration_execution_precommit(
    precommit: ObjectBongardRubricCalibrationExecutionPrecommit,
    authorization: ObjectBongardRubricCalibrationAuthorization,
    source: ObjectBongardRubricCalibrationSource,
) -> ObjectBongardRubricCalibrationExecutionPrecommit:
    if not isinstance(precommit, ObjectBongardRubricCalibrationExecutionPrecommit):
        raise TypeError("precommit must be the typed execution precommit")
    if (
        precommit.authorization_digest != authorization.authorization_digest
        or precommit.source_digest != source.source_digest
        or precommit.nomination_binding != _nomination_binding(source)
        or precommit.nomination_binding != authorization.nomination_binding
        or precommit.source_digests != authorization.source_digests
        or precommit.source_digests != _source_digest_inventory()
        or precommit.job_inventory_digest != authorization.job_inventory_digest
        or precommit.runtime.model != CALIBRATION_MODEL
        or precommit.runtime.reasoning_effort != CALIBRATION_REASONING_EFFORT
        or precommit.runtime.minutes != authorization.minutes
        or precommit.runtime.verbose != authorization.verbose
        or precommit.runtime.executable != authorization.executable
        or precommit.runtime.expected_launcher_digest
        != authorization.expected_launcher_sha256
        or precommit.runtime.transport_source_digest
        != prototype_scene_transport_source_digest()
    ):
        raise ObjectBongardRubricCalibrationCommandError(
            "execution precommit differs from authorization or current sources"
        )
    return precommit


@dataclass(frozen=True, slots=True)
class ObserverArtifactFileCommitment:
    run_index: int
    panel_binding_digest: str
    rubric_spec_digest: str
    panel_id: str
    observer_artifact_digest: str
    file_sha256: str

    def __post_init__(self) -> None:
        if type(self.run_index) is not int or not 0 <= self.run_index < CALIBRATION_JOB_COUNT:
            raise ObjectBongardRubricCalibrationCommandError(
                "observer artifact run index is invalid"
            )
        for name in (
            "panel_binding_digest",
            "rubric_spec_digest",
            "observer_artifact_digest",
            "file_sha256",
        ):
            _raw_digest(getattr(self, name), name)
        if not isinstance(self.panel_id, str) or _PANEL_ID.fullmatch(self.panel_id) is None:
            raise ObjectBongardRubricCalibrationCommandError(
                "observer artifact panel ID is invalid"
            )

    def to_data(self) -> dict[str, object]:
        return {
            "run_index": self.run_index,
            "panel_binding_digest": self.panel_binding_digest,
            "rubric_spec_digest": self.rubric_spec_digest,
            "panel_id": self.panel_id,
            "observer_artifact_digest": self.observer_artifact_digest,
            "file_sha256": self.file_sha256,
        }

    @classmethod
    def from_data(cls, value: object) -> "ObserverArtifactFileCommitment":
        raw = _exact_fields(
            value,
            {
                "run_index",
                "panel_binding_digest",
                "rubric_spec_digest",
                "panel_id",
                "observer_artifact_digest",
                "file_sha256",
            },
            "observer artifact file commitment",
        )
        result = cls(**raw)  # type: ignore[arg-type]
        if result.to_data() != dict(raw):
            raise ObjectBongardRubricCalibrationCommandError(
                "observer artifact file commitment is not canonical"
            )
        return result


def _inventory_content(
    value: "ObjectBongardRubricCalibrationObservationInventory",
) -> dict[str, object]:
    return {
        "schema": OBJECT_RUBRIC_CALIBRATION_INVENTORY_SCHEMA,
        "command_id": OBJECT_RUBRIC_CALIBRATION_COMMAND_ID,
        "authorization_digest": value.authorization_digest,
        "execution_precommit_digest": value.execution_precommit_digest,
        "source_digest": value.source_digest,
        "nomination_binding": value.nomination_binding.to_data(),
        "job_inventory_digest": value.job_inventory_digest,
        "observation_job_count": CALIBRATION_JOB_COUNT,
        "sheet_journal_count": CALIBRATION_SHEET_JOURNAL_COUNT,
        "parallel_workers": CALIBRATION_PARALLEL_WORKERS,
        "observer_artifact_files": [item.to_data() for item in value.artifact_files],
        "observation_batch": value.batch.to_data(),
        "fresh_model_call_count": value.fresh_model_call_count,
        "reused_model_call_count": value.reused_model_call_count,
        "labels_visible_to_observer": False,
        "query_pixels_used": False,
        "fresh_broad_cohort_pixels_used": False,
        "historical_released_pixels_only": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardRubricCalibrationObservationInventory:
    authorization_digest: str
    execution_precommit_digest: str
    source_digest: str
    nomination_binding: ObjectBongardRubricCalibrationNominationBinding
    job_inventory_digest: str
    artifact_files: tuple[ObserverArtifactFileCommitment, ...]
    batch: ObjectBongardRubricObservationBatch
    fresh_model_call_count: int
    reused_model_call_count: int
    inventory_digest: str

    def __post_init__(self) -> None:
        _address(self.authorization_digest, "inventory authorization digest")
        _address(
            self.execution_precommit_digest,
            "inventory execution precommit digest",
        )
        _raw_digest(self.source_digest, "inventory source digest")
        if not isinstance(
            self.nomination_binding,
            ObjectBongardRubricCalibrationNominationBinding,
        ):
            raise TypeError("nomination_binding has the wrong type")
        _raw_digest(self.job_inventory_digest, "inventory job digest")
        if (
            not isinstance(self.artifact_files, tuple)
            or len(self.artifact_files) != CALIBRATION_JOB_COUNT
            or tuple(item.run_index for item in self.artifact_files)
            != tuple(range(CALIBRATION_JOB_COUNT))
            or len({item.observer_artifact_digest for item in self.artifact_files})
            != CALIBRATION_JOB_COUNT
            or not isinstance(self.batch, ObjectBongardRubricObservationBatch)
            or self.batch.source_digest != self.source_digest
            or len(self.batch.runs) != CALIBRATION_JOB_COUNT
        ):
            raise ObjectBongardRubricCalibrationCommandError(
                "observation inventory batch/artifact cardinality differs"
            )
        for file_row, run in zip(self.artifact_files, self.batch.runs, strict=True):
            if (
                file_row.panel_binding_digest != run.panel_binding_digest
                or file_row.rubric_spec_digest != run.rubric_spec_digest
                or file_row.panel_id != run.artifact.panel_id
                or file_row.observer_artifact_digest
                != run.artifact.artifact_digest
            ):
                raise ObjectBongardRubricCalibrationCommandError(
                    "observer artifact file inventory differs from batch"
                )
        for name in ("fresh_model_call_count", "reused_model_call_count"):
            item = getattr(self, name)
            if type(item) is not int or item < 0:
                raise ObjectBongardRubricCalibrationCommandError(
                    f"{name} must be a nonnegative integer"
                )
        if (
            self.fresh_model_call_count
            != sum(item.fresh_call_count for item in self.batch.runs)
            or self.reused_model_call_count
            != sum(item.reused_call_count for item in self.batch.runs)
            or self.fresh_model_call_count + self.reused_model_call_count
            != CALIBRATION_SHEET_JOURNAL_COUNT
        ):
            raise ObjectBongardRubricCalibrationCommandError(
                "observation inventory call counts differ from 30 journals"
            )
        _address(self.inventory_digest, "observation inventory digest")
        if self.inventory_digest != "sha256:" + canonical_digest(
            _inventory_content(self)
        ):
            raise ObjectBongardRubricCalibrationCommandError(
                "observation inventory digest differs"
            )

    def to_data(self) -> dict[str, object]:
        return {
            **_inventory_content(self),
            "inventory_digest": self.inventory_digest,
        }

    @classmethod
    def from_data(
        cls, value: object
    ) -> "ObjectBongardRubricCalibrationObservationInventory":
        expected = {
            "schema",
            "command_id",
            "authorization_digest",
            "execution_precommit_digest",
            "source_digest",
            "nomination_binding",
            "job_inventory_digest",
            "observation_job_count",
            "sheet_journal_count",
            "parallel_workers",
            "observer_artifact_files",
            "observation_batch",
            "fresh_model_call_count",
            "reused_model_call_count",
            "labels_visible_to_observer",
            "query_pixels_used",
            "fresh_broad_cohort_pixels_used",
            "historical_released_pixels_only",
            *_authority_data(),
            "inventory_digest",
        }
        raw = _exact_fields(value, expected, "calibration observation inventory")
        if (
            raw["schema"] != OBJECT_RUBRIC_CALIBRATION_INVENTORY_SCHEMA
            or raw["command_id"] != OBJECT_RUBRIC_CALIBRATION_COMMAND_ID
            or raw["observation_job_count"] != CALIBRATION_JOB_COUNT
            or raw["sheet_journal_count"] != CALIBRATION_SHEET_JOURNAL_COUNT
            or raw["parallel_workers"] != CALIBRATION_PARALLEL_WORKERS
            or raw["labels_visible_to_observer"] is not False
            or raw["query_pixels_used"] is not False
            or raw["fresh_broad_cohort_pixels_used"] is not False
            or raw["historical_released_pixels_only"] is not True
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["nomination_binding"], Mapping)
            or not isinstance(raw["observer_artifact_files"], list)
        ):
            raise ObjectBongardRubricCalibrationCommandError(
                "calibration observation inventory policy differs"
            )
        result = cls(
            raw["authorization_digest"],
            raw["execution_precommit_digest"],
            raw["source_digest"],
            ObjectBongardRubricCalibrationNominationBinding.from_data(
                raw["nomination_binding"]
            ),
            raw["job_inventory_digest"],
            tuple(
                ObserverArtifactFileCommitment.from_data(item)
                for item in raw["observer_artifact_files"]
            ),
            ObjectBongardRubricObservationBatch.from_data(
                raw["observation_batch"]
            ),
            raw["fresh_model_call_count"],
            raw["reused_model_call_count"],
            raw["inventory_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardRubricCalibrationCommandError(
                "calibration observation inventory is not canonical"
            )
        return result


def _replay_content(
    value: "ObjectBongardRubricCalibrationDiskReplay",
) -> dict[str, object]:
    return {
        "schema": OBJECT_RUBRIC_CALIBRATION_REPLAY_SCHEMA,
        "command_id": OBJECT_RUBRIC_CALIBRATION_COMMAND_ID,
        "authorization_digest": value.authorization_digest,
        "execution_precommit_digest": value.execution_precommit_digest,
        "observation_inventory_digest": value.observation_inventory_digest,
        "assessment_digest": value.assessment_digest,
        "source_digest": value.source_digest,
        "nomination_binding": value.nomination_binding.to_data(),
        "verified_observation_job_count": CALIBRATION_JOB_COUNT,
        "verified_sheet_journal_count": CALIBRATION_SHEET_JOURNAL_COUNT,
        "survivor_counts_in_frozen_spec_order": list(value.survivor_counts),
        "slate_selection_digest": value.slate_selection_digest,
        "selected_candidate_digest": value.selected_candidate_digest,
        "acceptance_rule": CALIBRATION_ACCEPTANCE_RULE,
        "accepted": value.accepted,
        "threshold_tuning_performed": False,
        "preferred_candidate_selected": False,
        "fresh_broad_release_prepared": False,
        "model_calls_during_replay": 0,
        "query_pixels_used": False,
        "fresh_broad_cohort_pixels_used": False,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardRubricCalibrationDiskReplay:
    authorization_digest: str
    execution_precommit_digest: str
    observation_inventory_digest: str
    assessment_digest: str
    source_digest: str
    nomination_binding: ObjectBongardRubricCalibrationNominationBinding
    survivor_counts: tuple[int, ...]
    slate_selection_digest: str
    selected_candidate_digest: str | None
    accepted: bool
    replay_digest: str

    def __post_init__(self) -> None:
        for name in (
            "authorization_digest",
            "execution_precommit_digest",
            "observation_inventory_digest",
        ):
            _address(getattr(self, name), name)
        _raw_digest(self.assessment_digest, "assessment digest")
        _raw_digest(self.source_digest, "replay source digest")
        if not isinstance(
            self.nomination_binding,
            ObjectBongardRubricCalibrationNominationBinding,
        ):
            raise TypeError("nomination_binding has the wrong type")
        _raw_digest(self.slate_selection_digest, "slate selection digest")
        if self.selected_candidate_digest is not None:
            _raw_digest(self.selected_candidate_digest, "selected candidate digest")
        if (
            not isinstance(self.survivor_counts, tuple)
            or len(self.survivor_counts) != 2
            or any(type(item) is not int or item < 0 for item in self.survivor_counts)
            or not isinstance(self.accepted, bool)
            or self.accepted is not any(item >= 1 for item in self.survivor_counts)
            or self.accepted is not (self.selected_candidate_digest is not None)
        ):
            raise ObjectBongardRubricCalibrationCommandError(
                "calibration acceptance result differs from the frozen rule"
            )
        _address(self.replay_digest, "disk replay digest")
        if self.replay_digest != "sha256:" + canonical_digest(
            _replay_content(self)
        ):
            raise ObjectBongardRubricCalibrationCommandError(
                "disk replay digest differs"
            )

    def to_data(self) -> dict[str, object]:
        return {**_replay_content(self), "replay_digest": self.replay_digest}

    @classmethod
    def from_data(
        cls, value: object
    ) -> "ObjectBongardRubricCalibrationDiskReplay":
        expected = {
            "schema",
            "command_id",
            "authorization_digest",
            "execution_precommit_digest",
            "observation_inventory_digest",
            "assessment_digest",
            "source_digest",
            "nomination_binding",
            "verified_observation_job_count",
            "verified_sheet_journal_count",
            "survivor_counts_in_frozen_spec_order",
            "slate_selection_digest",
            "selected_candidate_digest",
            "acceptance_rule",
            "accepted",
            "threshold_tuning_performed",
            "preferred_candidate_selected",
            "fresh_broad_release_prepared",
            "model_calls_during_replay",
            "query_pixels_used",
            "fresh_broad_cohort_pixels_used",
            *_authority_data(),
            "replay_digest",
        }
        raw = _exact_fields(value, expected, "calibration disk replay")
        if (
            raw["schema"] != OBJECT_RUBRIC_CALIBRATION_REPLAY_SCHEMA
            or raw["command_id"] != OBJECT_RUBRIC_CALIBRATION_COMMAND_ID
            or raw["verified_observation_job_count"] != CALIBRATION_JOB_COUNT
            or raw["verified_sheet_journal_count"]
            != CALIBRATION_SHEET_JOURNAL_COUNT
            or raw["acceptance_rule"] != CALIBRATION_ACCEPTANCE_RULE
            or raw["threshold_tuning_performed"] is not False
            or raw["preferred_candidate_selected"] is not False
            or raw["fresh_broad_release_prepared"] is not False
            or raw["model_calls_during_replay"] != 0
            or raw["query_pixels_used"] is not False
            or raw["fresh_broad_cohort_pixels_used"] is not False
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["nomination_binding"], Mapping)
            or not isinstance(raw["survivor_counts_in_frozen_spec_order"], list)
        ):
            raise ObjectBongardRubricCalibrationCommandError(
                "calibration disk replay policy differs"
            )
        counts = tuple(raw["survivor_counts_in_frozen_spec_order"])
        if len(counts) != 2:
            raise ObjectBongardRubricCalibrationCommandError(
                "disk replay survivor count inventory differs"
            )
        result = cls(
            raw["authorization_digest"],
            raw["execution_precommit_digest"],
            raw["observation_inventory_digest"],
            raw["assessment_digest"],
            raw["source_digest"],
            ObjectBongardRubricCalibrationNominationBinding.from_data(
                raw["nomination_binding"]
            ),
            counts,  # type: ignore[arg-type]
            raw["slate_selection_digest"],
            raw["selected_candidate_digest"],
            raw["accepted"],
            raw["replay_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardRubricCalibrationCommandError(
                "calibration disk replay is not canonical"
            )
        return result


def _existing_output_root(value: str | os.PathLike[str]) -> Path:
    requested = Path(value).expanduser()
    try:
        resolved = requested.resolve(strict=True)
        info = resolved.lstat()
    except OSError as exc:
        raise ObjectBongardRubricCalibrationCommandError(
            "calibration output root is unavailable"
        ) from exc
    if (
        requested.absolute() != resolved
        or not stat.S_ISDIR(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
    ):
        raise ObjectBongardRubricCalibrationCommandError(
            "calibration output root must be one canonical directory"
        )
    return resolved


def _bind_embedded_nomination(
    root: Path,
    *,
    source_directory: str | os.PathLike[str],
) -> ObjectBongardRubricCalibrationSource:
    base = load_object_bongard_rubric_calibration_source(source_directory)
    nomination_root = _existing_output_root(root / NOMINATION_DIRECTORY)
    nomination = cold_verify_object_bongard_rubric_nomination(
        nomination_root,
        source_root=source_directory,
    )
    return _bind_verified_nomination(base, nomination)


def _copy_and_bind_nomination(
    root: Path,
    nomination_root: str | os.PathLike[str],
    *,
    source_directory: str | os.PathLike[str],
) -> ObjectBongardRubricCalibrationSource:
    base = load_object_bongard_rubric_calibration_source(source_directory)
    external_root = _existing_output_root(nomination_root)
    destination = root / NOMINATION_DIRECTORY
    if destination == external_root or destination.is_relative_to(external_root):
        raise ObjectBongardRubricCalibrationCommandError(
            "semantic nomination source cannot contain its embedded destination"
        )
    external = cold_verify_object_bongard_rubric_nomination(
        external_root,
        source_root=source_directory,
    )
    if destination.exists():
        embedded = cold_verify_object_bongard_rubric_nomination(
            _existing_output_root(destination),
            source_root=source_directory,
        )
        if embedded != external:
            raise ObjectBongardRubricCalibrationCommandError(
                "embedded semantic nomination differs from requested predecessor"
            )
    else:
        embedded = copy_verified_object_bongard_rubric_nomination(
            external_root,
            destination,
            source_root=source_directory,
        )
    return _bind_verified_nomination(base, embedded)


def persist_object_bongard_rubric_calibration_authorization(
    authorization: ObjectBongardRubricCalibrationAuthorization,
    output_root: str | os.PathLike[str],
) -> Path:
    if not isinstance(authorization, ObjectBongardRubricCalibrationAuthorization):
        raise TypeError("authorization must be the typed authorization")
    root = _ensure_output_root(output_root)
    path, _file_digest = _write_once(
        root / AUTHORIZATION_FILENAME,
        authorization.to_data(),
        "calibration authorization",
    )
    if load_object_bongard_rubric_calibration_authorization(root) != authorization:
        raise ObjectBongardRubricCalibrationCommandError(
            "persisted authorization failed typed reload"
        )
    return path


def load_object_bongard_rubric_calibration_authorization(
    output_root: str | os.PathLike[str],
) -> ObjectBongardRubricCalibrationAuthorization:
    root = _existing_output_root(output_root)
    return ObjectBongardRubricCalibrationAuthorization.from_data(
        _read_canonical_record(
            root / AUTHORIZATION_FILENAME, "calibration authorization"
        )
    )


def persist_object_bongard_rubric_calibration_execution_precommit(
    precommit: ObjectBongardRubricCalibrationExecutionPrecommit,
    output_root: str | os.PathLike[str],
) -> Path:
    if not isinstance(precommit, ObjectBongardRubricCalibrationExecutionPrecommit):
        raise TypeError("precommit must be the typed execution precommit")
    root = _ensure_output_root(output_root)
    path, _file_digest = _write_once(
        root / PRECOMMIT_FILENAME,
        precommit.to_data(),
        "calibration execution precommit",
    )
    if load_object_bongard_rubric_calibration_execution_precommit(root) != precommit:
        raise ObjectBongardRubricCalibrationCommandError(
            "persisted execution precommit failed typed reload"
        )
    return path


def load_object_bongard_rubric_calibration_execution_precommit(
    output_root: str | os.PathLike[str],
) -> ObjectBongardRubricCalibrationExecutionPrecommit:
    root = _existing_output_root(output_root)
    return ObjectBongardRubricCalibrationExecutionPrecommit.from_data(
        _read_canonical_record(
            root / PRECOMMIT_FILENAME, "calibration execution precommit"
        )
    )


def _new_observation_inventory(
    authorization: ObjectBongardRubricCalibrationAuthorization,
    precommit: ObjectBongardRubricCalibrationExecutionPrecommit,
    batch: ObjectBongardRubricObservationBatch,
    artifact_files: tuple[ObserverArtifactFileCommitment, ...],
) -> ObjectBongardRubricCalibrationObservationInventory:
    if precommit.nomination_binding != authorization.nomination_binding:
        raise ObjectBongardRubricCalibrationCommandError(
            "observation inventory nomination differs from the active seal"
        )
    values = {
        "authorization_digest": authorization.authorization_digest,
        "execution_precommit_digest": precommit.precommit_digest,
        "source_digest": authorization.source_digest,
        "nomination_binding": authorization.nomination_binding,
        "job_inventory_digest": authorization.job_inventory_digest,
        "artifact_files": artifact_files,
        "batch": batch,
        "fresh_model_call_count": sum(item.fresh_call_count for item in batch.runs),
        "reused_model_call_count": sum(item.reused_call_count for item in batch.runs),
    }
    provisional = object.__new__(
        ObjectBongardRubricCalibrationObservationInventory
    )
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectBongardRubricCalibrationObservationInventory(
        **values,
        inventory_digest="sha256:" + canonical_digest(
            _inventory_content(provisional)
        ),
    )


def persist_object_bongard_rubric_calibration_observation_inventory(
    authorization: ObjectBongardRubricCalibrationAuthorization,
    precommit: ObjectBongardRubricCalibrationExecutionPrecommit,
    batch: ObjectBongardRubricObservationBatch,
    output_root: str | os.PathLike[str],
) -> ObjectBongardRubricCalibrationObservationInventory:
    if not isinstance(batch, ObjectBongardRubricObservationBatch):
        raise TypeError("batch must be ObjectBongardRubricObservationBatch")
    if (
        precommit.authorization_digest != authorization.authorization_digest
        or precommit.nomination_binding != authorization.nomination_binding
        or batch.source_digest != authorization.source_digest
    ):
        raise ObjectBongardRubricCalibrationCommandError(
            "observation batch differs from authorization/precommit"
        )
    root = _ensure_output_root(output_root)
    artifact_root = _ensure_directory(root, OBSERVER_ARTIFACT_DIRECTORY)
    rows: list[ObserverArtifactFileCommitment] = []
    for index, run in enumerate(batch.runs):
        artifact = run.artifact
        path, file_sha256 = _write_once(
            artifact_root / f"{artifact.artifact_digest}.json",
            artifact.to_data(),
            "rubric observer artifact",
        )
        if path.name != f"{artifact.artifact_digest}.json":
            raise ObjectBongardRubricCalibrationCommandError(
                "observer artifact filename differs from content identity"
            )
        rows.append(
            ObserverArtifactFileCommitment(
                index,
                run.panel_binding_digest,
                run.rubric_spec_digest,
                artifact.panel_id,
                artifact.artifact_digest,
                file_sha256,
            )
        )
    inventory = _new_observation_inventory(
        authorization, precommit, batch, tuple(rows)
    )
    _write_once(
        root / INVENTORY_FILENAME,
        inventory.to_data(),
        "calibration observation inventory",
    )
    reloaded = load_object_bongard_rubric_calibration_observation_inventory(root)
    if reloaded != inventory:
        raise ObjectBongardRubricCalibrationCommandError(
            "persisted observation inventory failed typed reload"
        )
    return reloaded


def load_object_bongard_rubric_calibration_observation_inventory(
    output_root: str | os.PathLike[str],
) -> ObjectBongardRubricCalibrationObservationInventory:
    root = _existing_output_root(output_root)
    return ObjectBongardRubricCalibrationObservationInventory.from_data(
        _read_canonical_record(
            root / INVENTORY_FILENAME, "calibration observation inventory"
        )
    )


def persist_object_bongard_rubric_calibration_assessment(
    assessment: ObjectBongardRubricCalibrationAssessment,
    output_root: str | os.PathLike[str],
) -> Path:
    if not isinstance(assessment, ObjectBongardRubricCalibrationAssessment):
        raise TypeError("assessment must be the typed calibration assessment")
    root = _ensure_output_root(output_root)
    path, _file_digest = _write_once(
        root / ASSESSMENT_FILENAME,
        assessment.to_data(),
        "calibration assessment",
    )
    if load_object_bongard_rubric_calibration_assessment(root) != assessment:
        raise ObjectBongardRubricCalibrationCommandError(
            "persisted calibration assessment failed typed reload"
        )
    return path


def load_object_bongard_rubric_calibration_assessment(
    output_root: str | os.PathLike[str],
) -> ObjectBongardRubricCalibrationAssessment:
    root = _existing_output_root(output_root)
    return ObjectBongardRubricCalibrationAssessment.from_data(
        _read_canonical_record(
            root / ASSESSMENT_FILENAME, "calibration assessment"
        )
    )


def persist_object_bongard_rubric_calibration_disk_replay(
    replay: ObjectBongardRubricCalibrationDiskReplay,
    output_root: str | os.PathLike[str],
) -> Path:
    if not isinstance(replay, ObjectBongardRubricCalibrationDiskReplay):
        raise TypeError("replay must be the typed calibration disk replay")
    root = _ensure_output_root(output_root)
    path, _file_digest = _write_once(
        root / REPLAY_FILENAME,
        replay.to_data(),
        "calibration disk replay",
    )
    if load_object_bongard_rubric_calibration_disk_replay(root) != replay:
        raise ObjectBongardRubricCalibrationCommandError(
            "persisted disk replay failed typed reload"
        )
    return path


def load_object_bongard_rubric_calibration_disk_replay(
    output_root: str | os.PathLike[str],
) -> ObjectBongardRubricCalibrationDiskReplay:
    root = _existing_output_root(output_root)
    return ObjectBongardRubricCalibrationDiskReplay.from_data(
        _read_canonical_record(root / REPLAY_FILENAME, "calibration disk replay")
    )


def _verify_observer_artifact_files(
    root: Path,
    inventory: ObjectBongardRubricCalibrationObservationInventory,
) -> None:
    artifact_root = root / OBSERVER_ARTIFACT_DIRECTORY
    try:
        info = artifact_root.lstat()
    except OSError as exc:
        raise ObjectBongardRubricCalibrationCommandError(
            "observer artifact directory is unavailable"
        ) from exc
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise ObjectBongardRubricCalibrationCommandError(
            "observer artifact directory is invalid"
        )
    expected_names = {
        f"{item.observer_artifact_digest}.json" for item in inventory.artifact_files
    }
    actual_names: set[str] = set()
    for path in artifact_root.iterdir():
        item_info = path.lstat()
        if not stat.S_ISREG(item_info.st_mode) or stat.S_ISLNK(item_info.st_mode):
            raise ObjectBongardRubricCalibrationCommandError(
                "observer artifact directory contains a non-file entry"
            )
        actual_names.add(path.name)
    if actual_names != expected_names:
        raise ObjectBongardRubricCalibrationCommandError(
            "observer artifact file inventory differs"
        )
    for file_row, run in zip(
        inventory.artifact_files, inventory.batch.runs, strict=True
    ):
        path = artifact_root / f"{file_row.observer_artifact_digest}.json"
        raw = _read_canonical_record(path, "rubric observer artifact")
        file_sha256 = hashlib.sha256(canonical_json(raw) + b"\n").hexdigest()
        decoded = ObjectBongardRubricObserverArtifact.from_data(raw)
        if file_sha256 != file_row.file_sha256 or decoded != run.artifact:
            raise ObjectBongardRubricCalibrationCommandError(
                "persisted observer artifact differs from inventory"
            )


def _walk_journal_manifests(journal_root: Path) -> set[str]:
    try:
        root_info = journal_root.lstat()
    except OSError as exc:
        raise ObjectBongardRubricCalibrationCommandError(
            "journal root is unavailable"
        ) from exc
    if not stat.S_ISDIR(root_info.st_mode) or stat.S_ISLNK(root_info.st_mode):
        raise ObjectBongardRubricCalibrationCommandError("journal root is invalid")
    manifests: set[str] = set()
    for directory, directory_names, filenames in os.walk(
        journal_root, topdown=True, followlinks=False
    ):
        base = Path(directory)
        for name in directory_names:
            info = (base / name).lstat()
            if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
                raise ObjectBongardRubricCalibrationCommandError(
                    "journal tree contains a linked/non-directory branch"
                )
        for name in filenames:
            path = base / name
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
                raise ObjectBongardRubricCalibrationCommandError(
                    "journal tree contains a linked/non-file record"
                )
            if name == "manifest.json":
                manifests.add(path.relative_to(journal_root).as_posix())
    return manifests


def _precreate_authorized_journal_tree(
    root: Path,
    authorization: ObjectBongardRubricCalibrationAuthorization,
) -> Path:
    """Durably create every ancestor of all 15 claims before inference."""

    journal_root = _ensure_directory(root, JOURNAL_DIRECTORY)
    leaves: set[Path] = set()
    for job in authorization.jobs:
        task_root = _ensure_child_directory(journal_root, job.task_id)
        spec_root = _ensure_child_directory(task_root, job.rubric_spec_digest)
        for sheet in job.sheets:
            leaves.add(
                _ensure_child_directory(
                    spec_root, f"sheet_{sheet.sheet_index:03d}"
                )
            )
    if len(leaves) != CALIBRATION_SHEET_JOURNAL_COUNT:
        raise ObjectBongardRubricCalibrationCommandError(
            "precreated journal leaf inventory is not exactly 15"
        )
    # Re-fsync the complete bottom-up tree after construction.  The helper
    # already fsyncs each new child and parent; this final pass makes the
    # causal gate explicit even on a fully pre-existing resume tree.
    for leaf in sorted(leaves, key=lambda item: item.as_posix()):
        _fsync_directory(leaf)
        _fsync_directory(leaf.parent)
        _fsync_directory(leaf.parent.parent)
    _fsync_directory(journal_root)
    _fsync_directory(root)
    return journal_root


def _verify_journals_from_disk(
    root: Path,
    source: ObjectBongardRubricCalibrationSource,
    authorization: ObjectBongardRubricCalibrationAuthorization,
    precommit: ObjectBongardRubricCalibrationExecutionPrecommit,
    inventory: ObjectBongardRubricCalibrationObservationInventory,
) -> None:
    journal_root = root / JOURNAL_DIRECTORY
    expected_manifests = {
        (
            Path(job.task_id)
            / job.rubric_spec_digest
            / f"sheet_{sheet.sheet_index:03d}"
            / "manifest.json"
        ).as_posix()
        for job in authorization.jobs
        for sheet in job.sheets
    }
    if len(expected_manifests) != CALIBRATION_SHEET_JOURNAL_COUNT:
        raise ObjectBongardRubricCalibrationCommandError(
            "authorized journal path inventory is not exactly 15"
        )
    actual_manifests = _walk_journal_manifests(journal_root)
    if actual_manifests != expected_manifests:
        raise ObjectBongardRubricCalibrationCommandError(
            "persisted journal manifest inventory differs from authorization"
        )

    def forbidden_transport(*_args: Any, **_kwargs: Any) -> CodexStructuredResult:
        raise AssertionError("cold journal replay attempted a model call")

    for job, run in zip(authorization.jobs, inventory.batch.runs, strict=True):
        matches = tuple(
            item for item in source.panels if item.ordinal == job.panel_ordinal
        )
        if len(matches) != 1:
            raise ObjectBongardRubricCalibrationCommandError(
                "authorized panel ordinal is absent during journal replay"
            )
        panel = matches[0]
        spec = source.rubric_specs[job.rubric_spec_index]
        reconstructed = run_object_bongard_rubric_calibration_observation(
            panel,
            spec,
            runtime=precommit.runtime,
            journal_root=journal_root,
            authorization_digest=authorization.authorization_digest,
            execution_precommit_digest=precommit.precommit_digest,
            underlying_transport=forbidden_transport,
        )
        if (
            reconstructed.artifact != run.artifact
            or reconstructed.journal_summaries != run.journal_summaries
            or reconstructed.fresh_call_count != 0
            or reconstructed.reused_call_count != len(job.sheets)
        ):
            raise ObjectBongardRubricCalibrationCommandError(
                "journal-reconstructed observation differs from inventory"
            )


def _new_disk_replay(
    authorization: ObjectBongardRubricCalibrationAuthorization,
    precommit: ObjectBongardRubricCalibrationExecutionPrecommit,
    inventory: ObjectBongardRubricCalibrationObservationInventory,
    assessment: ObjectBongardRubricCalibrationAssessment,
) -> ObjectBongardRubricCalibrationDiskReplay:
    active_nomination = authorization.nomination_binding
    if (
        precommit.nomination_binding != active_nomination
        or inventory.nomination_binding != active_nomination
    ):
        raise ObjectBongardRubricCalibrationCommandError(
            "disk replay nomination differs across the sealed records"
        )
    counts = tuple(
        len(item.survivor_candidate_digests)
        for item in assessment.spec_assessments
    )
    if len(counts) != 2:
        raise ObjectBongardRubricCalibrationCommandError(
            "assessment does not contain the two frozen ranked rubric specs"
        )
    selection = assessment.slate_selection
    if selection is None:
        raise ObjectBongardRubricCalibrationCommandError(
            "assessment lacks its deterministic four-candidate slate selection"
        )
    values = {
        "authorization_digest": authorization.authorization_digest,
        "execution_precommit_digest": precommit.precommit_digest,
        "observation_inventory_digest": inventory.inventory_digest,
        "assessment_digest": assessment.assessment_digest,
        "source_digest": authorization.source_digest,
        "nomination_binding": authorization.nomination_binding,
        "survivor_counts": counts,
        "slate_selection_digest": selection.selection_digest,
        "selected_candidate_digest": selection.selected_candidate_digest,
        "accepted": selection.selected_candidate_digest is not None,
    }
    provisional = object.__new__(ObjectBongardRubricCalibrationDiskReplay)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectBongardRubricCalibrationDiskReplay(
        **values,
        replay_digest="sha256:" + canonical_digest(_replay_content(provisional)),
    )


def cold_replay_object_bongard_rubric_calibration_directory(
    output_root: str | os.PathLike[str],
    *,
    source_directory: str | os.PathLike[str] = (
        DEFAULT_OBJECT_RUBRIC_CALIBRATION_SOURCE
    ),
    require_replay_record: bool = True,
) -> ObjectBongardRubricCalibrationDiskReplay:
    """Reload and replay the complete calibration directory without a model."""

    if not isinstance(require_replay_record, bool):
        raise TypeError("require_replay_record must be bool")
    root = _existing_output_root(output_root)
    source = _bind_embedded_nomination(
        root,
        source_directory=source_directory,
    )
    authorization = load_object_bongard_rubric_calibration_authorization(root)
    verify_object_bongard_rubric_calibration_authorization(
        authorization, source, source_directory=source_directory
    )
    precommit = load_object_bongard_rubric_calibration_execution_precommit(root)
    verify_object_bongard_rubric_calibration_execution_precommit(
        precommit, authorization, source
    )
    inventory = load_object_bongard_rubric_calibration_observation_inventory(root)
    if (
        inventory.authorization_digest != authorization.authorization_digest
        or inventory.execution_precommit_digest != precommit.precommit_digest
        or inventory.source_digest != source.source_digest
        or inventory.nomination_binding != _nomination_binding(source)
        or inventory.nomination_binding != authorization.nomination_binding
        or inventory.nomination_binding != precommit.nomination_binding
        or inventory.job_inventory_digest != authorization.job_inventory_digest
    ):
        raise ObjectBongardRubricCalibrationCommandError(
            "observation inventory differs from authorization/precommit/source"
        )
    _verify_observer_artifact_files(root, inventory)
    _verify_journals_from_disk(
        root, source, authorization, precommit, inventory
    )
    assessment = load_object_bongard_rubric_calibration_assessment(root)
    cold_verify_object_bongard_rubric_calibration(
        assessment, source, inventory.batch
    )
    replay = _new_disk_replay(
        authorization, precommit, inventory, assessment
    )
    if require_replay_record:
        stored = load_object_bongard_rubric_calibration_disk_replay(root)
        if stored != replay:
            raise ObjectBongardRubricCalibrationCommandError(
                "persisted cold replay record differs from model-free replay"
            )
    return replay


@dataclass(frozen=True, slots=True)
class ObjectBongardRubricCalibrationCommandResult:
    output_root: Path
    authorization: ObjectBongardRubricCalibrationAuthorization
    precommit: ObjectBongardRubricCalibrationExecutionPrecommit
    inventory: ObjectBongardRubricCalibrationObservationInventory
    assessment: ObjectBongardRubricCalibrationAssessment
    replay: ObjectBongardRubricCalibrationDiskReplay

    def __post_init__(self) -> None:
        if (
            not isinstance(self.output_root, Path)
            or not self.output_root.is_absolute()
            or self.authorization.authorization_digest
            != self.precommit.authorization_digest
            or self.precommit.precommit_digest
            != self.inventory.execution_precommit_digest
            or self.inventory.inventory_digest
            != self.replay.observation_inventory_digest
            or self.assessment.assessment_digest != self.replay.assessment_digest
            or self.authorization.source_digest != self.replay.source_digest
            or self.authorization.nomination_binding
            != self.precommit.nomination_binding
            or self.authorization.nomination_binding
            != self.inventory.nomination_binding
            or self.authorization.nomination_binding
            != self.replay.nomination_binding
        ):
            raise ObjectBongardRubricCalibrationCommandError(
                "calibration command result chain differs"
            )

    @property
    def accepted(self) -> bool:
        return self.replay.accepted

    def summary_data(self) -> dict[str, object]:
        return {
            "schema": "gkm.bongard-object-rubric-calibration-command-summary.v3",
            "authorization_digest": self.authorization.authorization_digest,
            "execution_precommit_digest": self.precommit.precommit_digest,
            "observation_inventory_digest": self.inventory.inventory_digest,
            "assessment_digest": self.assessment.assessment_digest,
            "cold_replay_digest": self.replay.replay_digest,
            "nomination_binding": self.authorization.nomination_binding.to_data(),
            "observation_job_count": CALIBRATION_JOB_COUNT,
            "sheet_journal_count": CALIBRATION_SHEET_JOURNAL_COUNT,
            "fresh_model_call_count": self.inventory.fresh_model_call_count,
            "reused_model_call_count": self.inventory.reused_model_call_count,
            "survivor_counts_in_frozen_spec_order": list(
                self.replay.survivor_counts
            ),
            "acceptance_rule": CALIBRATION_ACCEPTANCE_RULE,
            "accepted": self.accepted,
            "threshold_tuning_performed": False,
            "preferred_candidate_selected": False,
            "fresh_broad_release_prepared": False,
            **_authority_data(),
        }


def _load_command_result(
    root: Path,
    replay: ObjectBongardRubricCalibrationDiskReplay,
) -> ObjectBongardRubricCalibrationCommandResult:
    return ObjectBongardRubricCalibrationCommandResult(
        root,
        load_object_bongard_rubric_calibration_authorization(root),
        load_object_bongard_rubric_calibration_execution_precommit(root),
        load_object_bongard_rubric_calibration_observation_inventory(root),
        load_object_bongard_rubric_calibration_assessment(root),
        replay,
    )


def run_object_bongard_rubric_calibration_command(
    output_root: str | os.PathLike[str],
    *,
    nomination_root: str | os.PathLike[str],
    source_directory: str | os.PathLike[str] = (
        DEFAULT_OBJECT_RUBRIC_CALIBRATION_SOURCE
    ),
    minutes: int = CALIBRATION_MINUTES,
    verbose: bool = False,
    executable: str = DEFAULT_CODEX_EXECUTABLE,
    expected_launcher_sha256: str = DEFAULT_CALIBRATION_CODEX_LAUNCHER_SHA256,
    cloud_policy_cache_snapshotter: Callable[[], CloudPolicyCacheSnapshot] = (
        snapshot_cloud_policy_cache
    ),
    model_catalog_snapshotter: Callable[[], CodexModelCatalogSnapshot] = (
        snapshot_pinned_model_catalog
    ),
    launcher_fingerprinter: LauncherFingerprinter = (
        codex_cli_authenticated_fingerprint
    ),
    runtime_attester: RuntimeAttester = attest_codex_no_tools,
    observation_runner: ObservationRunner = (
        run_object_bongard_rubric_calibration_observations
    ),
    underlying_transport: NamedImageTransport = (
        run_codex_named_images_structured
    ),
) -> ObjectBongardRubricCalibrationCommandResult:
    """Launch or resume the exact calibration and finish with disk replay.

    The supplied nomination directory is cold-verified and embedded before
    calibration authorization.  A resume must name the same predecessor.
    Existing canonical records are reused.  A completed observation inventory
    causes zero inference calls.  Partial work is resumed through the
    calibration driver's exclusive per-sheet journals; stranded claims remain
    fail-closed rather than being rerolled.
    """

    if DEFAULT_CODEX_MODEL != CALIBRATION_MODEL or DEFAULT_REASONING_EFFORT != CALIBRATION_REASONING_EFFORT:
        raise ObjectBongardRubricCalibrationCommandError(
            "transport defaults differ from the frozen calibration model"
        )
    root = _ensure_output_root(output_root)
    source = _copy_and_bind_nomination(
        root,
        nomination_root,
        source_directory=source_directory,
    )
    expected_authorization = prepare_object_bongard_rubric_calibration_authorization(
        source,
        source_directory=source_directory,
        minutes=minutes,
        verbose=verbose,
        executable=executable,
        expected_launcher_sha256=expected_launcher_sha256,
    )
    persist_object_bongard_rubric_calibration_authorization(
        expected_authorization, root
    )
    authorization = load_object_bongard_rubric_calibration_authorization(root)
    verify_object_bongard_rubric_calibration_authorization(
        authorization, source, source_directory=source_directory
    )

    precommit_path = root / PRECOMMIT_FILENAME
    if precommit_path.exists():
        precommit = load_object_bongard_rubric_calibration_execution_precommit(root)
    else:
        precommit = prepare_object_bongard_rubric_calibration_execution_precommit(
            authorization,
            source,
            cloud_policy_cache_snapshotter=cloud_policy_cache_snapshotter,
            model_catalog_snapshotter=model_catalog_snapshotter,
            launcher_fingerprinter=launcher_fingerprinter,
            runtime_attester=runtime_attester,
        )
        persist_object_bongard_rubric_calibration_execution_precommit(
            precommit, root
        )
    precommit = load_object_bongard_rubric_calibration_execution_precommit(root)
    verify_object_bongard_rubric_calibration_execution_precommit(
        precommit, authorization, source
    )

    # This is the causal inference gate.  Both records above have been fsynced
    # and parsed back into their strict typed forms before a transport can be
    # passed to the observation runner.
    inventory_path = root / INVENTORY_FILENAME
    if inventory_path.exists():
        inventory = load_object_bongard_rubric_calibration_observation_inventory(root)
    else:
        journal_root = _precreate_authorized_journal_tree(root, authorization)
        batch = observation_runner(
            source,
            runtime=precommit.runtime,
            journal_root=journal_root,
            authorization_digest=authorization.authorization_digest,
            execution_precommit_digest=precommit.precommit_digest,
            parallel_workers=CALIBRATION_PARALLEL_WORKERS,
            underlying_transport=underlying_transport,
        )
        inventory = persist_object_bongard_rubric_calibration_observation_inventory(
            authorization, precommit, batch, root
        )
    if (
        inventory.authorization_digest != authorization.authorization_digest
        or inventory.execution_precommit_digest != precommit.precommit_digest
        or inventory.source_digest != source.source_digest
        or inventory.nomination_binding != _nomination_binding(source)
        or inventory.nomination_binding != authorization.nomination_binding
        or inventory.nomination_binding != precommit.nomination_binding
        or inventory.job_inventory_digest != authorization.job_inventory_digest
    ):
        raise ObjectBongardRubricCalibrationCommandError(
            "loaded observation inventory differs from the active seal"
        )

    assessment_path = root / ASSESSMENT_FILENAME
    if assessment_path.exists():
        assessment = load_object_bongard_rubric_calibration_assessment(root)
    else:
        assessment = assess_object_bongard_rubric_calibration(
            source, inventory.batch
        )
        persist_object_bongard_rubric_calibration_assessment(assessment, root)
    if assessment.source_digest != source.source_digest:
        raise ObjectBongardRubricCalibrationCommandError(
            "loaded assessment differs from the active calibration source"
        )

    replay_path = root / REPLAY_FILENAME
    replay = cold_replay_object_bongard_rubric_calibration_directory(
        root,
        source_directory=source_directory,
        require_replay_record=replay_path.exists(),
    )
    if not replay_path.exists():
        persist_object_bongard_rubric_calibration_disk_replay(replay, root)
    stored_replay = load_object_bongard_rubric_calibration_disk_replay(root)
    if stored_replay != replay:
        raise ObjectBongardRubricCalibrationCommandError(
            "persisted replay differs after exact reload"
        )
    return _load_command_result(root, stored_replay)


def verify_object_bongard_rubric_calibration_command_directory(
    output_root: str | os.PathLike[str],
    *,
    source_directory: str | os.PathLike[str] = (
        DEFAULT_OBJECT_RUBRIC_CALIBRATION_SOURCE
    ),
) -> ObjectBongardRubricCalibrationCommandResult:
    root = _existing_output_root(output_root)
    replay = cold_replay_object_bongard_rubric_calibration_directory(
        root,
        source_directory=source_directory,
        require_replay_record=True,
    )
    return _load_command_result(root, replay)


def _add_common_cli_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--source-directory",
        type=Path,
        default=DEFAULT_OBJECT_RUBRIC_CALIBRATION_SOURCE,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Launch or cold-verify the sealed vision-nominated rubric calibration"
        )
    )
    subparsers = parser.add_subparsers(dest="operation", required=True)
    launch = subparsers.add_parser("launch", help="launch or resume calibration")
    _add_common_cli_arguments(launch)
    launch.add_argument("--nomination-root", required=True, type=Path)
    launch.add_argument("--minutes", type=int, default=CALIBRATION_MINUTES)
    launch.add_argument("--verbose", action="store_true")
    launch.add_argument("--executable", default=DEFAULT_CODEX_EXECUTABLE)
    launch.add_argument(
        "--expected-launcher-sha256",
        default=DEFAULT_CALIBRATION_CODEX_LAUNCHER_SHA256,
    )
    verify = subparsers.add_parser("verify", help="perform disk-only cold replay")
    _add_common_cli_arguments(verify)
    arguments = parser.parse_args(None if argv is None else list(argv))
    try:
        if arguments.operation == "launch":
            result = run_object_bongard_rubric_calibration_command(
                arguments.output_root,
                nomination_root=arguments.nomination_root,
                source_directory=arguments.source_directory,
                minutes=arguments.minutes,
                verbose=arguments.verbose,
                executable=arguments.executable,
                expected_launcher_sha256=arguments.expected_launcher_sha256,
            )
        else:
            result = verify_object_bongard_rubric_calibration_command_directory(
                arguments.output_root,
                source_directory=arguments.source_directory,
            )
    except Exception as exc:
        print(
            canonical_json(
                {
                    "schema": "gkm.bongard-object-rubric-calibration-command-error.v1",
                    "error_type": type(exc).__name__,
                    "message": str(exc)[:2000],
                }
            ).decode("utf-8"),
            file=sys.stderr,
        )
        return 1
    print(canonical_json(result.summary_data()).decode("utf-8"))
    return 0 if result.accepted else 2


__all__ = (
    "CALIBRATION_ACCEPTANCE_RULE",
    "CALIBRATION_JOB_COUNT",
    "CALIBRATION_MODEL",
    "CALIBRATION_PARALLEL_WORKERS",
    "CALIBRATION_REASONING_EFFORT",
    "CALIBRATION_SHEET_JOURNAL_COUNT",
    "DEFAULT_CALIBRATION_CODEX_LAUNCHER_SHA256",
    "HISTORICAL_EXECUTION_PRECOMMIT_RECORD_DIGEST",
    "HISTORICAL_RELEASE_AUTHORIZATION_RECORD_DIGEST",
    "NOMINATION_DIRECTORY",
    "CalibrationObservationJobCommitment",
    "CalibrationSheetCommitment",
    "ObjectBongardRubricCalibrationAuthorization",
    "ObjectBongardRubricCalibrationCommandError",
    "ObjectBongardRubricCalibrationCommandResult",
    "ObjectBongardRubricCalibrationDiskReplay",
    "ObjectBongardRubricCalibrationExecutionPrecommit",
    "ObjectBongardRubricCalibrationNominationBinding",
    "ObjectBongardRubricCalibrationObservationInventory",
    "ObserverArtifactFileCommitment",
    "cold_replay_object_bongard_rubric_calibration_directory",
    "load_object_bongard_rubric_calibration_assessment",
    "load_object_bongard_rubric_calibration_authorization",
    "load_object_bongard_rubric_calibration_disk_replay",
    "load_object_bongard_rubric_calibration_execution_precommit",
    "load_object_bongard_rubric_calibration_observation_inventory",
    "main",
    "object_bongard_rubric_calibration_command_source_digest",
    "persist_object_bongard_rubric_calibration_assessment",
    "persist_object_bongard_rubric_calibration_authorization",
    "persist_object_bongard_rubric_calibration_disk_replay",
    "persist_object_bongard_rubric_calibration_execution_precommit",
    "persist_object_bongard_rubric_calibration_observation_inventory",
    "prepare_object_bongard_rubric_calibration_authorization",
    "prepare_object_bongard_rubric_calibration_execution_precommit",
    "run_object_bongard_rubric_calibration_command",
    "verify_object_bongard_rubric_calibration_authorization",
    "verify_object_bongard_rubric_calibration_command_directory",
    "verify_object_bongard_rubric_calibration_execution_precommit",
)


if __name__ == "__main__":
    raise SystemExit(main())
