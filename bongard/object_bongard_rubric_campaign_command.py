"""Launch and cold-verify the preregistered broad object-rubric campaign.

This is the production filesystem boundary missing from the campaign core.
It seals the exact calibration result, cohort metadata, exposure timestamp,
runtime snapshots, no-tools attestation, concurrency, and call budget before
the official archive can release a support panel.  Resume reconstructs that
same seal; it never invents a new timestamp or ambient runtime identity.

The command is Python-authoritative.  Lean is absent, removable, and has no
role in artifact identity, selection, scoring, or replay.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

import argparse
import base64
from dataclasses import dataclass
from datetime import UTC, datetime
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
from bongard.object_bongard_release_gate import (
    ObjectBongardReleaseStore,
    ObjectBongardWriteOnceReceipt,
)
from bongard.object_bongard_rubric_calibration_command import (
    ObjectBongardRubricCalibrationCommandResult,
    verify_object_bongard_rubric_calibration_command_directory,
)
from bongard.object_bongard_rubric_campaign import (
    CAMPAIGN_ID,
    ObjectBongardRubricCampaignArchive,
    ObjectBongardRubricCampaignMetadata,
    ObjectBongardRubricCampaignRuntime,
    PersistedObjectBongardRubricCampaign,
    cold_replay_object_bongard_rubric_campaign,
    object_bongard_rubric_campaign_source_bindings,
    prepare_object_bongard_rubric_campaign,
    run_object_bongard_rubric_campaign,
    verify_object_bongard_rubric_campaign_metadata,
)
from bongard.object_bongard_rubric_ranker import (
    object_bongard_rubric_ranker_transport_source_digest,
)
from bongard.object_bongard_turn_journal import ObjectBongardTurnRuntime
from bongard.official_panel_archive import OfficialPanelArchive
from bongard.prototype_scene_observer import prototype_scene_transport_source_digest
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID
from bongard.transport import (
    PINNED_CODEX_CLI_VERSION,
    CloudPolicyCacheSnapshot,
    CodexModelCatalogSnapshot,
    CodexStructuredResult,
    codex_cli_authenticated_fingerprint,
    run_codex_named_images_structured,
    run_codex_text_structured,
    snapshot_cloud_policy_cache,
    snapshot_pinned_model_catalog,
)


COMMAND_ID = "bongard.object-rubric-campaign-command/seal-run-replay-v1"
AUTHORIZATION_SCHEMA = "gkm.bongard-object-rubric-campaign-authorization.v1"
RUNTIME_PRECOMMIT_SCHEMA = (
    "gkm.bongard-object-rubric-campaign-runtime-precommit.v1"
)
RESULT_SCHEMA = "gkm.bongard-object-rubric-campaign-command-result.v1"
REPLAY_SCHEMA = "gkm.bongard-object-rubric-campaign-command-replay.v1"

MODEL = "gpt-5.6-sol"
REASONING_EFFORT = "medium"
DEFAULT_MINUTES = 15
DEFAULT_PARALLEL_WORKERS = 4
# Exact worst case: 12 tasks * (1 semantic + 14 panels * 32 sheets + 1 rank).
DEFAULT_MAX_PHYSICAL_MODEL_CALLS = 5_400
DEFAULT_CODEX_EXECUTABLE = "codex"
DEFAULT_CODEX_LAUNCHER_SHA256 = (
    "19c4f144c5226a9f17c58e6f0fa854843b0f77a6eb420f40e2745a12f10f5d37"
)

_REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PREREGISTRATION = (
    _REPOSITORY_ROOT / "bongard/data/object_bongard_rubric_train_20260808.prereg.json"
)
DEFAULT_PLAN = (
    _REPOSITORY_ROOT / "bongard/data/object_bongard_rubric_train_20260808.plan.json"
)
DEFAULT_DESCRIPTOR = (
    _REPOSITORY_ROOT / "bongard/data/shape_bongard_v2_release_v1.json"
)
DEFAULT_SPLIT = (
    _REPOSITORY_ROOT
    / "downloads/ShapeBongard_V2_full/ShapeBongard_V2/ShapeBongard_V2_split.json"
)
DEFAULT_PREDECESSOR = (
    _REPOSITORY_ROOT
    / "downloads/ShapeBongard_V2_full/prototype_pair_python_campaign_20260807_object_v1/objects/exposure_successor/1bcde18e387539f13c4006b4a147e61c75feacb86bb031f10a6e8ba3412fe48d.json"
)
DEFAULT_ARCHIVE = _REPOSITORY_ROOT / "downloads/ShapeBongard_V2.zip"
EXPECTED_PREREGISTRATION_DIGEST = (
    "sha256:b4e29960a9524f5785139a3ddf462d5ddec784d52eb0f2678cb1674820dd8107"
)

AUTHORIZATION_FILENAME = "authorization.json"
RUNTIME_PRECOMMIT_FILENAME = "runtime_precommit.json"
RESULT_FILENAME = "result.json"
REPLAY_FILENAME = "cold_replay.json"
RELEASE_STORE_DIRECTORY = "release-store"
JOURNAL_DIRECTORY = "journals"

_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_RAW_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_MAX_RECORD_BYTES = 512 * 1024 * 1024

CalibrationVerifier = Callable[..., ObjectBongardRubricCalibrationCommandResult]
RuntimeAttester = Callable[..., CodexNoToolsAttestation]
LauncherFingerprinter = Callable[..., Mapping[str, str]]
NamedImageTransport = Callable[..., CodexStructuredResult]
TextTransport = Callable[..., CodexStructuredResult]


class ObjectBongardRubricCampaignCommandError(RuntimeError):
    """The launch seal, campaign execution, or disk replay failed closed."""


def object_bongard_rubric_campaign_command_source_digest() -> str:
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
        "lean_defines_artifact_identity_or_decision": False,
        "lean_required_for_replay": False,
    }


def _address(value: object) -> str:
    return "sha256:" + canonical_digest(value)


def _require_address(value: object, label: str) -> str:
    if not isinstance(value, str) or _ADDRESS.fullmatch(value) is None:
        raise ObjectBongardRubricCampaignCommandError(
            f"{label} must be a sha256: address"
        )
    return value


def _require_raw_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _RAW_DIGEST.fullmatch(value) is None:
        raise ObjectBongardRubricCampaignCommandError(
            f"{label} must be a raw lowercase SHA-256"
        )
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_output_root(value: str | os.PathLike[str]) -> Path:
    requested = Path(value).expanduser().absolute()
    missing: list[Path] = []
    cursor = requested
    while not cursor.exists():
        missing.append(cursor)
        if cursor.parent == cursor:
            raise ObjectBongardRubricCampaignCommandError(
                "cannot locate an existing output-root ancestor"
            )
        cursor = cursor.parent
    if cursor.resolve(strict=True) != cursor or not cursor.is_dir():
        raise ObjectBongardRubricCampaignCommandError(
            "output-root ancestor is not one canonical directory"
        )
    for path in reversed(missing):
        os.mkdir(path, 0o700)
        _fsync_directory(path)
        _fsync_directory(path.parent)
    resolved = requested.resolve(strict=True)
    info = resolved.lstat()
    if resolved != requested or not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise ObjectBongardRubricCampaignCommandError(
            "output root must be one canonical real directory"
        )
    return resolved


def _ensure_subdirectory(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or "/" in name or name in {".", ".."}:
        raise ObjectBongardRubricCampaignCommandError(
            "output subdirectory name is invalid"
        )
    path = root / name
    try:
        os.mkdir(path, 0o700)
        _fsync_directory(path)
        _fsync_directory(root)
    except FileExistsError:
        pass
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise ObjectBongardRubricCampaignCommandError(
            "output subdirectory is not a real directory"
        )
    return path


def _existing_output_root(value: str | os.PathLike[str]) -> Path:
    requested = Path(value).expanduser().absolute()
    try:
        resolved = requested.resolve(strict=True)
        info = requested.lstat()
    except OSError as exc:
        raise ObjectBongardRubricCampaignCommandError(
            "output root is unavailable"
        ) from exc
    if (
        resolved != requested
        or not stat.S_ISDIR(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
    ):
        raise ObjectBongardRubricCampaignCommandError(
            "output root must be one canonical real directory"
        )
    return resolved


def _existing_subdirectory(root: Path, name: str) -> Path:
    path = root / name
    try:
        info = path.lstat()
    except OSError as exc:
        raise ObjectBongardRubricCampaignCommandError(
            f"required output subdirectory {name!r} is unavailable"
        ) from exc
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise ObjectBongardRubricCampaignCommandError(
            f"required output subdirectory {name!r} is not a real directory"
        )
    return path


def _stable_read(path: Path, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        before = path.lstat()
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ObjectBongardRubricCampaignCommandError(f"cannot open {label}") from exc
    identity = lambda info: (
        info.st_dev, info.st_ino, info.st_mode, info.st_nlink,
        info.st_size, info.st_mtime_ns, info.st_ctime_ns,
    )
    try:
        opened = os.fstat(descriptor)
        if (
            identity(before) != identity(opened)
            or not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or not 0 < opened.st_size <= _MAX_RECORD_BYTES
        ):
            raise ObjectBongardRubricCampaignCommandError(
                f"{label} is not one stable bounded file"
            )
        blocks: list[bytes] = []
        total = 0
        while block := os.read(descriptor, min(1024 * 1024, _MAX_RECORD_BYTES + 1 - total)):
            blocks.append(block)
            total += len(block)
            if total > _MAX_RECORD_BYTES:
                raise ObjectBongardRubricCampaignCommandError(
                    f"{label} exceeds its byte bound"
                )
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if total != opened.st_size or identity(opened) != identity(after):
        raise ObjectBongardRubricCampaignCommandError(f"{label} changed while read")
    return b"".join(blocks)


def _read_record(path: Path, label: str) -> dict[str, Any]:
    payload = _stable_read(path, label)
    try:
        value = json.loads(payload.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ObjectBongardRubricCampaignCommandError(
            f"{label} is not UTF-8 JSON"
        ) from exc
    if not isinstance(value, dict) or canonical_json(value) + b"\n" != payload:
        raise ObjectBongardRubricCampaignCommandError(
            f"{label} bytes are not canonical"
        )
    return value


def _write_once(path: Path, value: Mapping[str, Any], label: str) -> None:
    payload = canonical_json(dict(value)) + b"\n"
    if path.exists():
        if _stable_read(path, label) != payload:
            raise ObjectBongardRubricCampaignCommandError(
                f"existing {label} differs"
            )
        return
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        if _stable_read(path, label) != payload:
            raise ObjectBongardRubricCampaignCommandError(
                f"racing {label} differs"
            )
        return
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise ObjectBongardRubricCampaignCommandError(
                    f"short write for {label}"
                )
            offset += written
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)
    if _stable_read(path, label) != payload:
        raise ObjectBongardRubricCampaignCommandError(
            f"persisted {label} differs after reload"
        )


def _encode_bytes(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _decode_bytes(value: object, label: str) -> bytes:
    if not isinstance(value, str):
        raise ObjectBongardRubricCampaignCommandError(f"{label} is not base64")
    try:
        result = base64.b64decode(value.encode("ascii"), validate=True)
    except (UnicodeError, ValueError) as exc:
        raise ObjectBongardRubricCampaignCommandError(f"{label} is not base64") from exc
    if _encode_bytes(result) != value:
        raise ObjectBongardRubricCampaignCommandError(
            f"{label} base64 is not canonical"
        )
    return result


def _authorization_content(
    value: "ObjectBongardRubricCampaignAuthorization",
) -> dict[str, object]:
    return {
        "schema": AUTHORIZATION_SCHEMA,
        "command_id": COMMAND_ID,
        "preregistration_digest": value.preregistration_digest,
        "batch_plan_digest": value.batch_plan_digest,
        "release_descriptor_digest": value.release_descriptor_digest,
        "exposure_predecessor_digest": value.exposure_predecessor_digest,
        "calibration_assessment_digest": value.calibration_assessment_digest,
        "calibration_replay_digest": value.calibration_replay_digest,
        "calibration_observation_inventory_digest": (
            value.calibration_observation_inventory_digest
        ),
        "campaign_source_bindings": [
            list(item) for item in value.campaign_source_bindings
        ],
        "command_source_digest": value.command_source_digest,
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "minutes": value.minutes,
        "parallel_workers": value.parallel_workers,
        "max_physical_model_calls": value.max_physical_model_calls,
        "executable": value.executable,
        "expected_launcher_sha256": value.expected_launcher_sha256,
        "exposure_observed_at": value.exposure_observed_at,
        "calibration_accepted_before_fresh_support_release": True,
        "official_test_authorized": False,
        "targeted_engineering_only": True,
        "fixed_score_denominator": 24,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardRubricCampaignAuthorization:
    preregistration_digest: str
    batch_plan_digest: str
    release_descriptor_digest: str
    exposure_predecessor_digest: str
    calibration_assessment_digest: str
    calibration_replay_digest: str
    calibration_observation_inventory_digest: str
    campaign_source_bindings: tuple[tuple[str, str], ...]
    command_source_digest: str
    minutes: int
    parallel_workers: int
    max_physical_model_calls: int
    executable: str
    expected_launcher_sha256: str
    exposure_observed_at: str
    authorization_digest: str

    def __post_init__(self) -> None:
        for name in (
            "preregistration_digest", "batch_plan_digest",
            "release_descriptor_digest", "exposure_predecessor_digest",
            "calibration_replay_digest",
            "calibration_observation_inventory_digest",
            "authorization_digest",
        ):
            _require_address(getattr(self, name), name)
        for name in (
            "calibration_assessment_digest", "command_source_digest",
            "expected_launcher_sha256",
        ):
            _require_raw_digest(getattr(self, name), name)
        if (
            self.campaign_source_bindings
            != tuple(sorted(self.campaign_source_bindings))
            or dict(self.campaign_source_bindings)
            != object_bongard_rubric_campaign_source_bindings()
            or self.command_source_digest
            != object_bongard_rubric_campaign_command_source_digest()
            or isinstance(self.minutes, bool)
            or not isinstance(self.minutes, int)
            or not 1 <= self.minutes <= 120
            or isinstance(self.parallel_workers, bool)
            or not isinstance(self.parallel_workers, int)
            or not 1 <= self.parallel_workers <= 12
            or isinstance(self.max_physical_model_calls, bool)
            or not isinstance(self.max_physical_model_calls, int)
            or not 1 <= self.max_physical_model_calls <= 100_000
            or not isinstance(self.executable, str)
            or not self.executable
            or not isinstance(self.exposure_observed_at, str)
            or not self.exposure_observed_at.endswith("Z")
            or self.authorization_digest != _address(_authorization_content(self))
        ):
            raise ObjectBongardRubricCampaignCommandError(
                "campaign authorization content differs"
            )

    def to_data(self) -> dict[str, object]:
        return {
            **_authorization_content(self),
            "authorization_digest": self.authorization_digest,
        }

    @classmethod
    def seal(
        cls,
        metadata: ObjectBongardRubricCampaignMetadata,
        calibration: ObjectBongardRubricCalibrationCommandResult,
        *,
        minutes: int,
        parallel_workers: int,
        max_physical_model_calls: int,
        executable: str,
        expected_launcher_sha256: str,
        exposure_observed_at: str,
    ) -> "ObjectBongardRubricCampaignAuthorization":
        if not calibration.accepted:
            raise ObjectBongardRubricCampaignCommandError(
                "broad campaign requires an accepted calibration replay"
            )
        values: dict[str, object] = {
            "preregistration_digest": metadata.preregistration_digest,
            "batch_plan_digest": metadata.plan.record_digest,
            "release_descriptor_digest": metadata.descriptor.digest,
            "exposure_predecessor_digest": metadata.predecessor.digest,
            "calibration_assessment_digest": calibration.assessment.assessment_digest,
            "calibration_replay_digest": calibration.replay.replay_digest,
            "calibration_observation_inventory_digest": (
                calibration.inventory.inventory_digest
            ),
            "campaign_source_bindings": tuple(
                sorted(object_bongard_rubric_campaign_source_bindings().items())
            ),
            "command_source_digest": (
                object_bongard_rubric_campaign_command_source_digest()
            ),
            "minutes": minutes,
            "parallel_workers": parallel_workers,
            "max_physical_model_calls": max_physical_model_calls,
            "executable": executable,
            "expected_launcher_sha256": expected_launcher_sha256,
            "exposure_observed_at": exposure_observed_at,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,  # type: ignore[arg-type]
            authorization_digest=_address(_authorization_content(provisional)),
        )

    @classmethod
    def from_data(
        cls, value: Mapping[str, Any]
    ) -> "ObjectBongardRubricCampaignAuthorization":
        required = {
            "schema", "command_id", "preregistration_digest",
            "batch_plan_digest", "release_descriptor_digest",
            "exposure_predecessor_digest", "calibration_assessment_digest",
            "calibration_replay_digest", "calibration_observation_inventory_digest",
            "campaign_source_bindings", "command_source_digest", "model",
            "reasoning_effort", "minutes", "parallel_workers",
            "max_physical_model_calls", "executable",
            "expected_launcher_sha256", "exposure_observed_at",
            "calibration_accepted_before_fresh_support_release",
            "official_test_authorized", "targeted_engineering_only",
            "fixed_score_denominator", *_authority_data(),
            "authorization_digest",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != required
            or value["schema"] != AUTHORIZATION_SCHEMA
            or value["command_id"] != COMMAND_ID
            or value["model"] != MODEL
            or value["reasoning_effort"] != REASONING_EFFORT
            or value["calibration_accepted_before_fresh_support_release"] is not True
            or value["official_test_authorized"] is not False
            or value["targeted_engineering_only"] is not True
            or value["fixed_score_denominator"] != 24
            or any(value[key] != item for key, item in _authority_data().items())
            or not isinstance(value["campaign_source_bindings"], list)
        ):
            raise ObjectBongardRubricCampaignCommandError(
                "campaign authorization fields differ"
            )
        result = cls(
            preregistration_digest=value["preregistration_digest"],
            batch_plan_digest=value["batch_plan_digest"],
            release_descriptor_digest=value["release_descriptor_digest"],
            exposure_predecessor_digest=value["exposure_predecessor_digest"],
            calibration_assessment_digest=value["calibration_assessment_digest"],
            calibration_replay_digest=value["calibration_replay_digest"],
            calibration_observation_inventory_digest=value[
                "calibration_observation_inventory_digest"
            ],
            campaign_source_bindings=tuple(
                tuple(item) for item in value["campaign_source_bindings"]
            ),
            command_source_digest=value["command_source_digest"],
            minutes=value["minutes"],
            parallel_workers=value["parallel_workers"],
            max_physical_model_calls=value["max_physical_model_calls"],
            executable=value["executable"],
            expected_launcher_sha256=value["expected_launcher_sha256"],
            exposure_observed_at=value["exposure_observed_at"],
            authorization_digest=value["authorization_digest"],
        )
        if result.to_data() != dict(value):
            raise ObjectBongardRubricCampaignCommandError(
                "campaign authorization is not canonical"
            )
        return result


def _runtime_precommit_content(
    value: "ObjectBongardRubricCampaignRuntimePrecommit",
) -> dict[str, object]:
    cache = value.runtime.visual.cloud_policy_cache_snapshot
    if not isinstance(cache, CloudPolicyCacheSnapshot):
        raise ObjectBongardRubricCampaignCommandError(
            "runtime precommit requires a policy-cache snapshot"
        )
    return {
        "schema": RUNTIME_PRECOMMIT_SCHEMA,
        "command_id": COMMAND_ID,
        "authorization_digest": value.authorization_digest,
        "runtime_binding": value.runtime.binding,
        "runtime_binding_digest": value.runtime.binding_digest,
        "cloud_policy_cache_snapshot_base64": (
            None if cache.data is None else _encode_bytes(cache.data)
        ),
        "model_catalog_snapshot_base64": _encode_bytes(
            value.runtime.visual.model_catalog_snapshot.data
        ),
        "no_tools_attestation": value.runtime.visual.no_tools_attestation.to_dict(),
        "launcher_fingerprint": dict(value.launcher_fingerprint),
        "both_text_and_named_image_modalities_attested": True,
        "runtime_precommit_persisted_before_official_archive_panel_release": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardRubricCampaignRuntimePrecommit:
    authorization_digest: str
    runtime: ObjectBongardRubricCampaignRuntime
    launcher_fingerprint: Mapping[str, str]
    precommit_digest: str

    def __post_init__(self) -> None:
        _require_address(self.authorization_digest, "authorization digest")
        _require_address(self.precommit_digest, "runtime precommit digest")
        if not isinstance(self.runtime, ObjectBongardRubricCampaignRuntime):
            raise TypeError("runtime has the wrong type")
        fingerprint = dict(self.launcher_fingerprint)
        if (
            set(fingerprint) != {"version", "launcher_digest"}
            or fingerprint["version"] != PINNED_CODEX_CLI_VERSION
            or fingerprint["launcher_digest"]
            != self.runtime.visual.expected_launcher_digest
            or self.runtime.visual.model_catalog_snapshot
            != self.runtime.rank.model_catalog_snapshot
            or self.runtime.visual.no_tools_attestation
            != self.runtime.rank.no_tools_attestation
            or self.runtime.visual.cloud_policy_cache_snapshot
            != self.runtime.rank.cloud_policy_cache_snapshot
            or self.precommit_digest != _address(_runtime_precommit_content(self))
        ):
            raise ObjectBongardRubricCampaignCommandError(
                "campaign runtime precommit differs"
            )

    def to_data(self) -> dict[str, object]:
        return {
            **_runtime_precommit_content(self),
            "precommit_digest": self.precommit_digest,
        }

    @classmethod
    def seal(
        cls,
        authorization: ObjectBongardRubricCampaignAuthorization,
        *,
        cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot,
        model_catalog_snapshot: CodexModelCatalogSnapshot,
        no_tools_attestation: CodexNoToolsAttestation,
        launcher_fingerprint: Mapping[str, str],
    ) -> "ObjectBongardRubricCampaignRuntimePrecommit":
        common = {
            "model": MODEL,
            "reasoning_effort": REASONING_EFFORT,
            "minutes": authorization.minutes,
            "verbose": False,
            "executable": authorization.executable,
            "cloud_policy_cache_snapshot": cloud_policy_cache_snapshot,
            "model_catalog_snapshot": model_catalog_snapshot,
            "expected_launcher_digest": authorization.expected_launcher_sha256,
            "no_tools_attestation": no_tools_attestation,
        }
        visual = ObjectBongardTurnRuntime(
            **common,
            transport_source_digest=prototype_scene_transport_source_digest(),
        )
        rank = ObjectBongardTurnRuntime(
            **common,
            transport_source_digest=(
                object_bongard_rubric_ranker_transport_source_digest()
            ),
        )
        runtime = ObjectBongardRubricCampaignRuntime(
            visual=visual,
            rank=rank,
            max_workers=authorization.parallel_workers,
            max_physical_model_calls=authorization.max_physical_model_calls,
        )
        values = {
            "authorization_digest": authorization.authorization_digest,
            "runtime": runtime,
            "launcher_fingerprint": dict(launcher_fingerprint),
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            precommit_digest=_address(_runtime_precommit_content(provisional)),
        )

    @classmethod
    def from_data(
        cls, value: Mapping[str, Any]
    ) -> "ObjectBongardRubricCampaignRuntimePrecommit":
        required = {
            "schema", "command_id", "authorization_digest", "runtime_binding",
            "runtime_binding_digest", "cloud_policy_cache_snapshot_base64",
            "model_catalog_snapshot_base64", "no_tools_attestation",
            "launcher_fingerprint", "both_text_and_named_image_modalities_attested",
            "runtime_precommit_persisted_before_official_archive_panel_release",
            *_authority_data(), "precommit_digest",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != required
            or value["schema"] != RUNTIME_PRECOMMIT_SCHEMA
            or value["command_id"] != COMMAND_ID
            or value["both_text_and_named_image_modalities_attested"] is not True
            or value["runtime_precommit_persisted_before_official_archive_panel_release"]
            is not True
            or any(value[key] != item for key, item in _authority_data().items())
            or not isinstance(value["runtime_binding"], Mapping)
            or not isinstance(value["no_tools_attestation"], Mapping)
            or not isinstance(value["launcher_fingerprint"], Mapping)
        ):
            raise ObjectBongardRubricCampaignCommandError(
                "runtime precommit fields differ"
            )
        cache_value = value["cloud_policy_cache_snapshot_base64"]
        cache = CloudPolicyCacheSnapshot(
            None if cache_value is None else _decode_bytes(cache_value, "policy cache")
        )
        catalog = CodexModelCatalogSnapshot(
            _decode_bytes(value["model_catalog_snapshot_base64"], "model catalog")
        )
        attestation = CodexNoToolsAttestation.from_mapping(
            value["no_tools_attestation"]
        )
        binding = value["runtime_binding"]
        visual_binding = binding.get("visual")
        rank_binding = binding.get("rank")
        if not isinstance(visual_binding, Mapping) or not isinstance(rank_binding, Mapping):
            raise ObjectBongardRubricCampaignCommandError(
                "runtime turn bindings are malformed"
            )
        def turn(row: Mapping[str, Any]) -> ObjectBongardTurnRuntime:
            return ObjectBongardTurnRuntime(
                model=row["model"],
                reasoning_effort=row["reasoning_effort"],
                minutes=row["minutes"],
                verbose=row["verbose"],
                executable=row["executable"],
                cloud_policy_cache_snapshot=cache,
                model_catalog_snapshot=catalog,
                expected_launcher_digest=row["expected_launcher_digest"],
                no_tools_attestation=attestation,
                transport_source_digest=row["transport_source_digest"],
            )
        runtime = ObjectBongardRubricCampaignRuntime(
            visual=turn(visual_binding),
            rank=turn(rank_binding),
            max_workers=binding["max_workers"],
            max_physical_model_calls=binding["max_physical_model_calls"],
        )
        if runtime.binding != dict(binding) or runtime.binding_digest != value["runtime_binding_digest"]:
            raise ObjectBongardRubricCampaignCommandError(
                "runtime snapshot reconstruction differs"
            )
        result = cls(
            authorization_digest=value["authorization_digest"],
            runtime=runtime,
            launcher_fingerprint=dict(value["launcher_fingerprint"]),
            precommit_digest=value["precommit_digest"],
        )
        if result.to_data() != dict(value):
            raise ObjectBongardRubricCampaignCommandError(
                "runtime precommit is not canonical"
            )
        return result


def _result_content(value: "ObjectBongardRubricCampaignCommandResult") -> dict[str, object]:
    campaign = value.campaign
    return {
        "schema": RESULT_SCHEMA,
        "command_id": COMMAND_ID,
        "authorization_digest": value.authorization_digest,
        "runtime_precommit_digest": value.runtime_precommit_digest,
        "campaign_record_digest": campaign.archive.record_digest,
        "campaign_store_receipt": campaign.store_receipt.to_data(),
        "task_count": len(campaign.archive.task_executions),
        "complete_task_count": campaign.archive.complete_task_count,
        "gap_task_count": campaign.archive.gap_task_count,
        "correct_count": campaign.archive.correct_count,
        "abstention_count": campaign.archive.abstention_count,
        "fixed_score_denominator": campaign.archive.fixed_score_denominator,
        "accuracy_ppm": campaign.archive.accuracy_ppm,
        "physical_model_calls": campaign.archive.physical_model_calls,
        "official_benchmark_result": False,
        "targeted_engineering_result": True,
        "calibration_gate_passed": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardRubricCampaignCommandResult:
    authorization_digest: str
    runtime_precommit_digest: str
    campaign: PersistedObjectBongardRubricCampaign
    result_digest: str

    def __post_init__(self) -> None:
        _require_address(self.authorization_digest, "authorization digest")
        _require_address(self.runtime_precommit_digest, "runtime precommit digest")
        _require_address(self.result_digest, "command result digest")
        if (
            not isinstance(self.campaign, PersistedObjectBongardRubricCampaign)
            or self.campaign.archive.fixed_score_denominator != 24
            or len(self.campaign.archive.task_executions) != 12
            or self.result_digest != _address(_result_content(self))
        ):
            raise ObjectBongardRubricCampaignCommandError(
                "campaign command result differs"
            )

    def to_data(self) -> dict[str, object]:
        return {**_result_content(self), "result_digest": self.result_digest}

    @classmethod
    def seal(
        cls,
        authorization: ObjectBongardRubricCampaignAuthorization,
        precommit: ObjectBongardRubricCampaignRuntimePrecommit,
        campaign: PersistedObjectBongardRubricCampaign,
    ) -> "ObjectBongardRubricCampaignCommandResult":
        values = {
            "authorization_digest": authorization.authorization_digest,
            "runtime_precommit_digest": precommit.precommit_digest,
            "campaign": campaign,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            result_digest=_address(_result_content(provisional)),
        )


def _replay_content(value: "ObjectBongardRubricCampaignCommandReplay") -> dict[str, object]:
    return {
        "schema": REPLAY_SCHEMA,
        "command_id": COMMAND_ID,
        "authorization_digest": value.authorization_digest,
        "runtime_precommit_digest": value.runtime_precommit_digest,
        "command_result_digest": value.command_result_digest,
        "campaign_record_digest": value.campaign_record_digest,
        "fixed_score_denominator": value.fixed_score_denominator,
        "correct_count": value.correct_count,
        "abstention_count": value.abstention_count,
        "accuracy_ppm": value.accuracy_ppm,
        "model_calls": 0,
        "model_free": True,
        "tamper_detecting": True,
        "official_pngs_artifacts_journals_receipts_freezes_and_commits_replayed": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardRubricCampaignCommandReplay:
    authorization_digest: str
    runtime_precommit_digest: str
    command_result_digest: str
    campaign_record_digest: str
    fixed_score_denominator: int
    correct_count: int
    abstention_count: int
    accuracy_ppm: int
    replay_digest: str

    def __post_init__(self) -> None:
        for name in (
            "authorization_digest", "runtime_precommit_digest",
            "command_result_digest", "campaign_record_digest", "replay_digest",
        ):
            _require_address(getattr(self, name), name)
        if (
            self.fixed_score_denominator != 24
            or isinstance(self.correct_count, bool)
            or not isinstance(self.correct_count, int)
            or not 0 <= self.correct_count <= 24
            or isinstance(self.abstention_count, bool)
            or not isinstance(self.abstention_count, int)
            or not 0 <= self.abstention_count <= 24
            or self.accuracy_ppm != self.correct_count * 1_000_000 // 24
            or self.replay_digest != _address(_replay_content(self))
        ):
            raise ObjectBongardRubricCampaignCommandError(
                "campaign command replay differs"
            )

    def to_data(self) -> dict[str, object]:
        return {**_replay_content(self), "replay_digest": self.replay_digest}

    @classmethod
    def seal(
        cls,
        result: ObjectBongardRubricCampaignCommandResult,
        campaign: ObjectBongardRubricCampaignArchive,
    ) -> "ObjectBongardRubricCampaignCommandReplay":
        if campaign != result.campaign.archive:
            raise ObjectBongardRubricCampaignCommandError(
                "cold campaign differs from command result"
            )
        values = {
            "authorization_digest": result.authorization_digest,
            "runtime_precommit_digest": result.runtime_precommit_digest,
            "command_result_digest": result.result_digest,
            "campaign_record_digest": campaign.record_digest,
            "fixed_score_denominator": campaign.fixed_score_denominator,
            "correct_count": campaign.correct_count,
            "abstention_count": campaign.abstention_count,
            "accuracy_ppm": campaign.accuracy_ppm,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            replay_digest=_address(_replay_content(provisional)),
        )

    @classmethod
    def from_data(
        cls, value: Mapping[str, Any]
    ) -> "ObjectBongardRubricCampaignCommandReplay":
        required = {
            "schema", "command_id", "authorization_digest",
            "runtime_precommit_digest", "command_result_digest",
            "campaign_record_digest", "fixed_score_denominator", "correct_count",
            "abstention_count", "accuracy_ppm", "model_calls", "model_free",
            "tamper_detecting",
            "official_pngs_artifacts_journals_receipts_freezes_and_commits_replayed",
            *_authority_data(), "replay_digest",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != required
            or value["schema"] != REPLAY_SCHEMA
            or value["command_id"] != COMMAND_ID
            or value["model_calls"] != 0
            or value["model_free"] is not True
            or value["tamper_detecting"] is not True
            or value[
                "official_pngs_artifacts_journals_receipts_freezes_and_commits_replayed"
            ] is not True
            or any(value[key] != item for key, item in _authority_data().items())
        ):
            raise ObjectBongardRubricCampaignCommandError(
                "campaign command replay fields differ"
            )
        result = cls(
            authorization_digest=value["authorization_digest"],
            runtime_precommit_digest=value["runtime_precommit_digest"],
            command_result_digest=value["command_result_digest"],
            campaign_record_digest=value["campaign_record_digest"],
            fixed_score_denominator=value["fixed_score_denominator"],
            correct_count=value["correct_count"],
            abstention_count=value["abstention_count"],
            accuracy_ppm=value["accuracy_ppm"],
            replay_digest=value["replay_digest"],
        )
        if result.to_data() != dict(value):
            raise ObjectBongardRubricCampaignCommandError(
                "campaign command replay is not canonical"
            )
        return result


@dataclass(frozen=True, slots=True)
class ObjectBongardRubricCampaignCommandExecution:
    output_root: Path
    authorization: ObjectBongardRubricCampaignAuthorization
    runtime_precommit: ObjectBongardRubricCampaignRuntimePrecommit
    result: ObjectBongardRubricCampaignCommandResult
    replay: ObjectBongardRubricCampaignCommandReplay

    def __post_init__(self) -> None:
        if (
            not isinstance(self.output_root, Path)
            or not self.output_root.is_absolute()
            or self.authorization.authorization_digest
            != self.runtime_precommit.authorization_digest
            or self.authorization.authorization_digest
            != self.result.authorization_digest
            or self.runtime_precommit.precommit_digest
            != self.result.runtime_precommit_digest
            or self.replay.authorization_digest
            != self.authorization.authorization_digest
            or self.replay.runtime_precommit_digest
            != self.runtime_precommit.precommit_digest
            or self.result.result_digest != self.replay.command_result_digest
            or self.result.campaign.archive.record_digest
            != self.replay.campaign_record_digest
        ):
            raise ObjectBongardRubricCampaignCommandError(
                "campaign command execution chain differs"
            )

    def summary_data(self) -> dict[str, object]:
        campaign = self.result.campaign.archive
        return {
            "schema": "gkm.bongard-object-rubric-campaign-command-summary.v1",
            "authorization_digest": self.authorization.authorization_digest,
            "runtime_precommit_digest": self.runtime_precommit.precommit_digest,
            "campaign_record_digest": campaign.record_digest,
            "command_result_digest": self.result.result_digest,
            "cold_replay_digest": self.replay.replay_digest,
            "task_count": len(campaign.task_executions),
            "complete_task_count": campaign.complete_task_count,
            "gap_task_count": campaign.gap_task_count,
            "correct_count": campaign.correct_count,
            "abstention_count": campaign.abstention_count,
            "fixed_score_denominator": campaign.fixed_score_denominator,
            "accuracy_ppm": campaign.accuracy_ppm,
            "physical_model_calls": campaign.physical_model_calls,
            "model_free_cold_replay": True,
            "calibration_gate_passed": True,
            "official_benchmark_result": False,
            "targeted_engineering_result": True,
            **_authority_data(),
        }


def _load_metadata(
    *,
    preregistration_path: str | Path,
    plan_path: str | Path,
    descriptor_path: str | Path,
    split_path: str | Path,
    predecessor_path: str | Path,
) -> ObjectBongardRubricCampaignMetadata:
    return verify_object_bongard_rubric_campaign_metadata(
        preregistration_path=preregistration_path,
        expected_preregistration_digest=EXPECTED_PREREGISTRATION_DIGEST,
        plan_path=plan_path,
        descriptor_path=descriptor_path,
        split_path=split_path,
        predecessor_path=predecessor_path,
    )


def _load_authorization(path: Path) -> ObjectBongardRubricCampaignAuthorization:
    return ObjectBongardRubricCampaignAuthorization.from_data(
        _read_record(path, "campaign authorization")
    )


def _load_runtime_precommit(
    path: Path,
) -> ObjectBongardRubricCampaignRuntimePrecommit:
    return ObjectBongardRubricCampaignRuntimePrecommit.from_data(
        _read_record(path, "campaign runtime precommit")
    )


def _load_campaign(
    store: ObjectBongardReleaseStore,
    receipt: ObjectBongardWriteOnceReceipt,
) -> PersistedObjectBongardRubricCampaign:
    path = store.root / receipt.relative_path
    raw = _read_record(path, "durable campaign archive")
    archive = ObjectBongardRubricCampaignArchive.from_data(raw)
    store.verify(receipt, expected_data=archive.to_data())
    return PersistedObjectBongardRubricCampaign(archive, receipt)


def _load_result(
    path: Path,
    store: ObjectBongardReleaseStore,
) -> ObjectBongardRubricCampaignCommandResult:
    raw = _read_record(path, "campaign command result")
    receipt_raw = raw.get("campaign_store_receipt")
    if not isinstance(receipt_raw, Mapping):
        raise ObjectBongardRubricCampaignCommandError(
            "campaign command result lacks a store receipt"
        )
    receipt = ObjectBongardWriteOnceReceipt.from_data(receipt_raw)
    campaign = _load_campaign(store, receipt)
    result = ObjectBongardRubricCampaignCommandResult(
        authorization_digest=raw.get("authorization_digest"),
        runtime_precommit_digest=raw.get("runtime_precommit_digest"),
        campaign=campaign,
        result_digest=raw.get("result_digest"),
    )
    if result.to_data() != raw:
        raise ObjectBongardRubricCampaignCommandError(
            "campaign command result is not canonical"
        )
    return result


def _assert_authorization_matches(
    authorization: ObjectBongardRubricCampaignAuthorization,
    metadata: ObjectBongardRubricCampaignMetadata,
    calibration: ObjectBongardRubricCalibrationCommandResult,
    *,
    minutes: int,
    parallel_workers: int,
    max_physical_model_calls: int,
    executable: str,
    expected_launcher_sha256: str,
) -> None:
    expected = ObjectBongardRubricCampaignAuthorization.seal(
        metadata,
        calibration,
        minutes=minutes,
        parallel_workers=parallel_workers,
        max_physical_model_calls=max_physical_model_calls,
        executable=executable,
        expected_launcher_sha256=expected_launcher_sha256,
        exposure_observed_at=authorization.exposure_observed_at,
    )
    if expected != authorization:
        raise ObjectBongardRubricCampaignCommandError(
            "loaded launch authorization differs from current exact inputs"
        )


def _assert_campaign_launch_gate(
    result: ObjectBongardRubricCampaignCommandResult,
    authorization: ObjectBongardRubricCampaignAuthorization,
    precommit: ObjectBongardRubricCampaignRuntimePrecommit,
) -> None:
    archive = result.campaign.archive
    configuration = dict(archive.execution_precommit.configuration)
    expected: dict[str, object] = {
        "campaign_id": CAMPAIGN_ID,
        "preregistration_digest": authorization.preregistration_digest,
        "runtime_binding_digest": precommit.runtime.binding_digest,
        "max_workers": precommit.runtime.max_workers,
        "max_physical_model_calls": precommit.runtime.max_physical_model_calls,
        "headless": True,
        "pure_python_predicates": True,
        "lean_required": False,
        "fixed_query_denominator": 24,
        "launch_authorization_digest": authorization.authorization_digest,
        "campaign_runtime_precommit_digest": precommit.precommit_digest,
        "calibration_assessment_digest": (
            authorization.calibration_assessment_digest
        ),
        "calibration_replay_digest": authorization.calibration_replay_digest,
        "calibration_observation_inventory_digest": (
            authorization.calibration_observation_inventory_digest
        ),
    }
    if configuration != expected or archive.runtime_binding_digest != (
        precommit.runtime.binding_digest
    ):
        raise ObjectBongardRubricCampaignCommandError(
            "durable campaign is not bound to the accepted calibration and launch seal"
        )


def _prepare_or_load_runtime_precommit(
    root: Path,
    authorization: ObjectBongardRubricCampaignAuthorization,
    *,
    cloud_policy_cache_snapshotter: Callable[[], CloudPolicyCacheSnapshot],
    model_catalog_snapshotter: Callable[[], CodexModelCatalogSnapshot],
    launcher_fingerprinter: LauncherFingerprinter,
    runtime_attester: RuntimeAttester,
) -> ObjectBongardRubricCampaignRuntimePrecommit:
    path = root / RUNTIME_PRECOMMIT_FILENAME
    if path.exists():
        result = _load_runtime_precommit(path)
    else:
        cache = cloud_policy_cache_snapshotter()
        catalog = model_catalog_snapshotter()
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
        result = ObjectBongardRubricCampaignRuntimePrecommit.seal(
            authorization,
            cloud_policy_cache_snapshot=cache,
            model_catalog_snapshot=catalog,
            no_tools_attestation=attestation,
            launcher_fingerprint=fingerprint,
        )
        _write_once(path, result.to_data(), "campaign runtime precommit")
        result = _load_runtime_precommit(path)
    _assert_runtime_precommit_matches_authorization(result, authorization)
    return result


def _assert_runtime_precommit_matches_authorization(
    precommit: ObjectBongardRubricCampaignRuntimePrecommit,
    authorization: ObjectBongardRubricCampaignAuthorization,
) -> None:
    expected_turn = (
        MODEL,
        REASONING_EFFORT,
        authorization.minutes,
        False,
        authorization.executable,
        authorization.expected_launcher_sha256,
    )
    actual_turns = tuple(
        (
            turn.model,
            turn.reasoning_effort,
            turn.minutes,
            turn.verbose,
            turn.executable,
            turn.expected_launcher_digest,
        )
        for turn in (precommit.runtime.visual, precommit.runtime.rank)
    )
    if (
        precommit.authorization_digest != authorization.authorization_digest
        or precommit.runtime.max_workers != authorization.parallel_workers
        or precommit.runtime.max_physical_model_calls
        != authorization.max_physical_model_calls
        or actual_turns != (expected_turn, expected_turn)
    ):
        raise ObjectBongardRubricCampaignCommandError(
            "runtime precommit differs from launch authorization"
        )


def _verify_calibration(
    calibration_root: str | Path,
    verifier: CalibrationVerifier,
) -> ObjectBongardRubricCalibrationCommandResult:
    result = verifier(calibration_root)
    if (
        not isinstance(result, ObjectBongardRubricCalibrationCommandResult)
        or not result.accepted
    ):
        raise ObjectBongardRubricCampaignCommandError(
            "broad launch is closed until calibration cold replay is accepted"
        )
    return result


def run_object_bongard_rubric_campaign_command(
    output_root: str | os.PathLike[str],
    *,
    calibration_root: str | os.PathLike[str],
    preregistration_path: str | Path = DEFAULT_PREREGISTRATION,
    plan_path: str | Path = DEFAULT_PLAN,
    descriptor_path: str | Path = DEFAULT_DESCRIPTOR,
    split_path: str | Path = DEFAULT_SPLIT,
    predecessor_path: str | Path = DEFAULT_PREDECESSOR,
    archive_path: str | Path = DEFAULT_ARCHIVE,
    minutes: int = DEFAULT_MINUTES,
    parallel_workers: int = DEFAULT_PARALLEL_WORKERS,
    max_physical_model_calls: int = DEFAULT_MAX_PHYSICAL_MODEL_CALLS,
    executable: str = DEFAULT_CODEX_EXECUTABLE,
    expected_launcher_sha256: str = DEFAULT_CODEX_LAUNCHER_SHA256,
    calibration_verifier: CalibrationVerifier = (
        verify_object_bongard_rubric_calibration_command_directory
    ),
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
    visual_transport: NamedImageTransport = run_codex_named_images_structured,
    rank_transport: TextTransport = run_codex_text_structured,
) -> ObjectBongardRubricCampaignCommandExecution:
    """Launch or resume the calibrated broad campaign, then cold-replay it."""

    root = _ensure_output_root(output_root)
    calibration = _verify_calibration(calibration_root, calibration_verifier)
    metadata = _load_metadata(
        preregistration_path=preregistration_path,
        plan_path=plan_path,
        descriptor_path=descriptor_path,
        split_path=split_path,
        predecessor_path=predecessor_path,
    )
    authorization_path = root / AUTHORIZATION_FILENAME
    if authorization_path.exists():
        authorization = _load_authorization(authorization_path)
    else:
        observed_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace(
            "+00:00", "Z"
        )
        authorization = ObjectBongardRubricCampaignAuthorization.seal(
            metadata,
            calibration,
            minutes=minutes,
            parallel_workers=parallel_workers,
            max_physical_model_calls=max_physical_model_calls,
            executable=executable,
            expected_launcher_sha256=expected_launcher_sha256,
            exposure_observed_at=observed_at,
        )
        _write_once(
            authorization_path, authorization.to_data(), "campaign authorization"
        )
        authorization = _load_authorization(authorization_path)
    _assert_authorization_matches(
        authorization,
        metadata,
        calibration,
        minutes=minutes,
        parallel_workers=parallel_workers,
        max_physical_model_calls=max_physical_model_calls,
        executable=executable,
        expected_launcher_sha256=expected_launcher_sha256,
    )
    runtime_precommit = _prepare_or_load_runtime_precommit(
        root,
        authorization,
        cloud_policy_cache_snapshotter=cloud_policy_cache_snapshotter,
        model_catalog_snapshotter=model_catalog_snapshotter,
        launcher_fingerprinter=launcher_fingerprinter,
        runtime_attester=runtime_attester,
    )

    # No panel member can be read before both immutable records above exist.
    archive = OfficialPanelArchive.load(
        metadata.descriptor,
        archive_path,
        expected_release_descriptor_digest=metadata.descriptor.digest,
    )
    store_root = _ensure_subdirectory(root, RELEASE_STORE_DIRECTORY)
    journal_root = _ensure_subdirectory(root, JOURNAL_DIRECTORY)
    store = ObjectBongardReleaseStore(store_root)
    result_path = root / RESULT_FILENAME
    if result_path.exists():
        result = _load_result(result_path, store)
    else:
        prepared = prepare_object_bongard_rubric_campaign(
            metadata=metadata,
            archive=archive,
            store=store,
            runtime=runtime_precommit.runtime,
            exposure_observed_at=authorization.exposure_observed_at,
            launch_gate_bindings={
                "launch_authorization_digest": (
                    authorization.authorization_digest
                ),
                "campaign_runtime_precommit_digest": (
                    runtime_precommit.precommit_digest
                ),
                "calibration_assessment_digest": (
                    authorization.calibration_assessment_digest
                ),
                "calibration_replay_digest": (
                    authorization.calibration_replay_digest
                ),
                "calibration_observation_inventory_digest": (
                    authorization.calibration_observation_inventory_digest
                ),
            },
        )
        campaign = run_object_bongard_rubric_campaign(
            prepared=prepared,
            archive=archive,
            runtime=runtime_precommit.runtime,
            journals_root=journal_root,
            visual_transport=visual_transport,
            rank_transport=rank_transport,
        )
        result = ObjectBongardRubricCampaignCommandResult.seal(
            authorization, runtime_precommit, campaign
        )
        _write_once(result_path, result.to_data(), "campaign command result")
        result = _load_result(result_path, store)
    if (
        result.authorization_digest != authorization.authorization_digest
        or result.runtime_precommit_digest != runtime_precommit.precommit_digest
    ):
        raise ObjectBongardRubricCampaignCommandError(
            "campaign result differs from launch seal"
        )
    _assert_campaign_launch_gate(result, authorization, runtime_precommit)
    replayed_campaign = cold_replay_object_bongard_rubric_campaign(
        result.campaign.archive,
        expected_campaign_digest=result.campaign.archive.record_digest,
        campaign_store_receipt=result.campaign.store_receipt,
        store=store,
        archive=archive,
        runtime=runtime_precommit.runtime,
        journals_root=journal_root,
    )
    replay = ObjectBongardRubricCampaignCommandReplay.seal(
        result, replayed_campaign
    )
    replay_path = root / REPLAY_FILENAME
    _write_once(replay_path, replay.to_data(), "campaign command cold replay")
    stored_replay = ObjectBongardRubricCampaignCommandReplay.from_data(
        _read_record(replay_path, "campaign command cold replay")
    )
    if stored_replay != replay:
        raise ObjectBongardRubricCampaignCommandError(
            "persisted campaign replay differs"
        )
    return ObjectBongardRubricCampaignCommandExecution(
        root, authorization, runtime_precommit, result, stored_replay
    )


def verify_object_bongard_rubric_campaign_command_directory(
    output_root: str | os.PathLike[str],
    *,
    calibration_root: str | os.PathLike[str],
    preregistration_path: str | Path = DEFAULT_PREREGISTRATION,
    plan_path: str | Path = DEFAULT_PLAN,
    descriptor_path: str | Path = DEFAULT_DESCRIPTOR,
    split_path: str | Path = DEFAULT_SPLIT,
    predecessor_path: str | Path = DEFAULT_PREDECESSOR,
    archive_path: str | Path = DEFAULT_ARCHIVE,
    calibration_verifier: CalibrationVerifier = (
        verify_object_bongard_rubric_calibration_command_directory
    ),
) -> ObjectBongardRubricCampaignCommandExecution:
    """Perform a strictly model-free replay of a finished command directory."""

    root = _existing_output_root(output_root)
    calibration = _verify_calibration(calibration_root, calibration_verifier)
    metadata = _load_metadata(
        preregistration_path=preregistration_path,
        plan_path=plan_path,
        descriptor_path=descriptor_path,
        split_path=split_path,
        predecessor_path=predecessor_path,
    )
    authorization = _load_authorization(root / AUTHORIZATION_FILENAME)
    _assert_authorization_matches(
        authorization,
        metadata,
        calibration,
        minutes=authorization.minutes,
        parallel_workers=authorization.parallel_workers,
        max_physical_model_calls=authorization.max_physical_model_calls,
        executable=authorization.executable,
        expected_launcher_sha256=authorization.expected_launcher_sha256,
    )
    runtime_precommit = _load_runtime_precommit(
        root / RUNTIME_PRECOMMIT_FILENAME
    )
    _assert_runtime_precommit_matches_authorization(
        runtime_precommit, authorization
    )

    archive = OfficialPanelArchive.load(
        metadata.descriptor,
        archive_path,
        expected_release_descriptor_digest=metadata.descriptor.digest,
    )
    store = ObjectBongardReleaseStore(
        _existing_subdirectory(root, RELEASE_STORE_DIRECTORY)
    )
    journal_root = _existing_subdirectory(root, JOURNAL_DIRECTORY)
    result = _load_result(root / RESULT_FILENAME, store)
    if (
        result.authorization_digest != authorization.authorization_digest
        or result.runtime_precommit_digest != runtime_precommit.precommit_digest
    ):
        raise ObjectBongardRubricCampaignCommandError(
            "campaign result differs from launch seal"
        )
    _assert_campaign_launch_gate(result, authorization, runtime_precommit)
    replayed_campaign = cold_replay_object_bongard_rubric_campaign(
        result.campaign.archive,
        expected_campaign_digest=result.campaign.archive.record_digest,
        campaign_store_receipt=result.campaign.store_receipt,
        store=store,
        archive=archive,
        runtime=runtime_precommit.runtime,
        journals_root=journal_root,
    )
    expected_replay = ObjectBongardRubricCampaignCommandReplay.seal(
        result, replayed_campaign
    )
    stored_replay = ObjectBongardRubricCampaignCommandReplay.from_data(
        _read_record(root / REPLAY_FILENAME, "campaign command cold replay")
    )
    if stored_replay != expected_replay:
        raise ObjectBongardRubricCampaignCommandError(
            "persisted campaign cold replay differs from disk replay"
        )
    return ObjectBongardRubricCampaignCommandExecution(
        root, authorization, runtime_precommit, result, stored_replay
    )


def _add_common_cli_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--calibration-root", required=True, type=Path)
    parser.add_argument(
        "--preregistration-path", type=Path, default=DEFAULT_PREREGISTRATION
    )
    parser.add_argument("--plan-path", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--descriptor-path", type=Path, default=DEFAULT_DESCRIPTOR)
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--predecessor-path", type=Path, default=DEFAULT_PREDECESSOR)
    parser.add_argument("--archive-path", type=Path, default=DEFAULT_ARCHIVE)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Launch or cold-verify the calibration-gated preregistered "
            "Bongard prose-rubric campaign"
        )
    )
    subparsers = parser.add_subparsers(dest="operation", required=True)
    launch = subparsers.add_parser("launch", help="launch or resume campaign")
    _add_common_cli_arguments(launch)
    launch.add_argument("--minutes", type=int, default=DEFAULT_MINUTES)
    launch.add_argument(
        "--parallel-workers", type=int, default=DEFAULT_PARALLEL_WORKERS
    )
    launch.add_argument(
        "--max-physical-model-calls",
        type=int,
        default=DEFAULT_MAX_PHYSICAL_MODEL_CALLS,
    )
    launch.add_argument("--executable", default=DEFAULT_CODEX_EXECUTABLE)
    launch.add_argument(
        "--expected-launcher-sha256",
        default=DEFAULT_CODEX_LAUNCHER_SHA256,
    )
    verify = subparsers.add_parser(
        "verify", help="perform model-free disk cold replay"
    )
    _add_common_cli_arguments(verify)
    arguments = parser.parse_args(None if argv is None else list(argv))
    common = {
        "calibration_root": arguments.calibration_root,
        "preregistration_path": arguments.preregistration_path,
        "plan_path": arguments.plan_path,
        "descriptor_path": arguments.descriptor_path,
        "split_path": arguments.split_path,
        "predecessor_path": arguments.predecessor_path,
        "archive_path": arguments.archive_path,
    }
    try:
        if arguments.operation == "launch":
            result = run_object_bongard_rubric_campaign_command(
                arguments.output_root,
                **common,
                minutes=arguments.minutes,
                parallel_workers=arguments.parallel_workers,
                max_physical_model_calls=arguments.max_physical_model_calls,
                executable=arguments.executable,
                expected_launcher_sha256=arguments.expected_launcher_sha256,
            )
        else:
            result = verify_object_bongard_rubric_campaign_command_directory(
                arguments.output_root,
                **common,
            )
    except Exception as exc:
        print(
            canonical_json(
                {
                    "schema": "gkm.bongard-object-rubric-campaign-command-error.v1",
                    "error_type": type(exc).__name__,
                    "message": str(exc)[:2000],
                }
            ).decode("utf-8"),
            file=sys.stderr,
        )
        return 1
    print(canonical_json(result.summary_data()).decode("utf-8"))
    return 0


__all__ = (
    "DEFAULT_ARCHIVE",
    "DEFAULT_DESCRIPTOR",
    "DEFAULT_MAX_PHYSICAL_MODEL_CALLS",
    "DEFAULT_PARALLEL_WORKERS",
    "DEFAULT_PLAN",
    "DEFAULT_PREREGISTRATION",
    "DEFAULT_PREDECESSOR",
    "DEFAULT_SPLIT",
    "ObjectBongardRubricCampaignAuthorization",
    "ObjectBongardRubricCampaignCommandError",
    "ObjectBongardRubricCampaignCommandExecution",
    "ObjectBongardRubricCampaignCommandReplay",
    "ObjectBongardRubricCampaignCommandResult",
    "ObjectBongardRubricCampaignRuntimePrecommit",
    "main",
    "object_bongard_rubric_campaign_command_source_digest",
    "run_object_bongard_rubric_campaign_command",
    "verify_object_bongard_rubric_campaign_command_directory",
)


if __name__ == "__main__":
    raise SystemExit(main())
