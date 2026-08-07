"""Authenticated launcher for the preregistered prototype-pair campaign.

The first stage of this module is deliberately metadata-only.  It cold-replays
the checked cohort plan from the release descriptor, split-derived task
inventory, historical seed, and exact exposure predecessor without opening an
official panel archive, creating a campaign store, or invoking Codex.  Runtime
identities and the execution precommit are frozen next.  The campaign
coordinator remains the sole owner of phase-zero precommit persistence and of
all subsequent panel release and model-call ordering.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import threading
from typing import Any, Callable, Mapping, Sequence

from bongard.canonical import canonical_digest, canonical_json
from bongard.codex_no_tools_preflight import (
    CodexNoToolsAttestation,
    attest_codex_no_tools,
)
from bongard.exposure import ExposureLedger, semantic_resolver_policy_digest
from bongard.historical_exposure import (
    HistoricalExposureSeed,
    load_historical_exposure,
)
from bongard.official_panel_archive import OfficialPanelArchive
from bongard.prototype_pair_campaign import (
    PrototypePairCampaignConfiguration,
    prototype_pair_campaign_runtime_source_digests,
)
from bongard.prototype_pair_campaign_store import PrototypePairCampaignStore
from bongard.prototype_pair_cohort import (
    ALGORITHM_ID as COHORT_ALGORITHM_ID,
    OFFICIAL_BASIC_GENERATOR_SHA256,
    OFFICIAL_BASIC_SAMPLER_SHA256,
    OPAQUE_TAG_IDS,
    PrototypePairCohortPlan,
    task_id_inventory_digest,
    verify_prototype_pair_cohort_plan,
)
from bongard.prototype_pair_execution_precommit import (
    PrototypePairExecutionIdentities,
    PrototypePairExecutionPrecommit,
    prepare_prototype_pair_execution_precommit,
    verify_prototype_pair_execution_precommit,
)
from bongard.prototype_scene_calibration import (
    PrototypeSceneTagThreshold,
    calibration_algorithm_digest,
    threshold_commitment,
)
from bongard.prototype_scene_codex_ranker import (
    PROTOTYPE_SCENE_CODEX_RANKER_PROTOCOL_ID,
    PrototypeSceneCodexRanker,
    prototype_scene_codex_ranker_environment_digest,
    prototype_scene_codex_ranker_model_identity_digest,
    prototype_scene_codex_ranker_protocol_digest,
    prototype_scene_codex_ranker_transport_source_digest,
)
from bongard.prototype_scene_headless_runner import (
    RUNNER_ID,
    prototype_scene_runner_source_digest,
)
from bongard.prototype_object_scene_observer import (
    PROTOTYPE_SCENE_OBSERVER_PROTOCOL_ID,
    prototype_rubric_description_protocol_digest,
    prototype_scene_observer_environment_digest,
    prototype_scene_observer_model_digest,
    prototype_scene_scoring_protocol_digest,
)
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID
from bongard.release import OfficialReleaseDescriptor
from bongard.transport import (
    CodexModelCatalogSnapshot,
    CloudPolicyCacheSnapshot,
    codex_cli_authenticated_fingerprint,
    run_codex_named_images_structured,
    snapshot_pinned_model_catalog,
    snapshot_cloud_policy_cache,
)


PREREGISTRATION_SCHEMA = (
    "gkm.bongard-prototype-pair-targeted-engineering-preregistration.v1"
)
PREREGISTRATION_SCOPE = "exact-unused-train-semantics-reused-targeted-engineering"
DEFAULT_ABSENT_UPPER_PPM = 250_000
DEFAULT_PRESENT_LOWER_PPM = 750_000
DEFAULT_OBSERVER_MINUTES = 15
DEFAULT_RANKER_MINUTES = 15
DEFAULT_PARALLEL_WORKERS = 8
DEFAULT_ACTOR = "prototype-pair-campaign-cli"
DEFAULT_RUNTIME_ARCHIVE_SOURCE_ID = "official-shapebongard-v2-zip"
DEFAULT_RUNTIME_VERIFIER_ID = "prototype-scene-runtime-adapter-v1"
DEFAULT_CAMPAIGN_MODEL = "gpt-5.6-sol"
DEFAULT_CAMPAIGN_REASONING_EFFORT = "medium"

_RAW_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_MAX_METADATA_BYTES = 64 * 1024 * 1024
_MAX_SPLIT_BYTES = 16 * 1024 * 1024
_MAX_PYTHON_EXECUTABLE_BYTES = 512 * 1024 * 1024
_OFFICIAL_SPLIT_GROUPS = frozenset(
    {"train", "val", "test_ff", "test_bd", "test_hd_comb", "test_hd_novel"}
)


class PrototypePairCampaignCliError(RuntimeError):
    """A launcher input, runtime identity, or dispatch invariant failed."""


def _require_raw_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _RAW_SHA256.fullmatch(value) is None:
        raise PrototypePairCampaignCliError(f"{label} must be lowercase SHA-256")
    return value


def _require_address(value: object, label: str) -> str:
    if not isinstance(value, str) or _ADDRESS.fullmatch(value) is None:
        raise PrototypePairCampaignCliError(f"{label} must be a sha256: address")
    return value


def _file_identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _stable_read_regular(
    value: str | Path,
    *,
    label: str,
    byte_limit: int = _MAX_METADATA_BYTES,
) -> tuple[Path, bytes]:
    path = Path(os.path.abspath(os.path.expanduser(os.fspath(value))))
    if not hasattr(os, "O_NOFOLLOW"):
        raise PrototypePairCampaignCliError(
            f"platform cannot safely open {label} without following symlinks"
        )
    try:
        before = path.lstat()
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
        )
    except OSError as exc:
        raise PrototypePairCampaignCliError(f"cannot open {label}: {path}") from exc
    try:
        opened = os.fstat(descriptor)
        identity = _file_identity(opened)
        if (
            not stat.S_ISREG(opened.st_mode)
            or _file_identity(before) != identity
            or opened.st_size <= 0
            or opened.st_size > byte_limit
        ):
            raise PrototypePairCampaignCliError(
                f"{label} is not one stable bounded regular file"
            )
        chunks: list[bytes] = []
        total = 0
        while True:
            block = os.read(descriptor, min(65_536, byte_limit + 1 - total))
            if not block:
                break
            chunks.append(block)
            total += len(block)
            if total > byte_limit:
                raise PrototypePairCampaignCliError(f"{label} exceeds its byte limit")
        if total != opened.st_size or _file_identity(os.fstat(descriptor)) != identity:
            raise PrototypePairCampaignCliError(f"{label} changed while being read")
    finally:
        os.close(descriptor)
    try:
        after = path.lstat()
    except OSError as exc:
        raise PrototypePairCampaignCliError(f"{label} disappeared after read") from exc
    if _file_identity(after) != identity:
        raise PrototypePairCampaignCliError(f"{label} path changed while being read")
    return path, b"".join(chunks)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise PrototypePairCampaignCliError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> object:
    raise PrototypePairCampaignCliError(f"non-finite JSON value: {value}")


def _strict_json_object(payload: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        raw = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except PrototypePairCampaignCliError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PrototypePairCampaignCliError(f"{label} is not strict JSON") from exc
    if not isinstance(raw, Mapping) or any(not isinstance(key, str) for key in raw):
        raise PrototypePairCampaignCliError(f"{label} root must be an object")
    return raw


def _exact_object(
    value: object, fields: set[str], *, label: str
) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(not isinstance(key, str) for key in value)
        or set(value) != fields
    ):
        raise PrototypePairCampaignCliError(f"{label} fields differ")
    return value


@dataclass(frozen=True, slots=True)
class PrototypePairPreregistrationPins:
    record_digest: str
    selection_seed: str
    seed_commitment: str
    namespace: str
    plan_digest: str
    release_descriptor_digest: str
    corpus_manifest_digest: str
    split_source_digest: str
    task_inventory_digest: str
    historical_seed_digest: str
    exposure_predecessor_digest: str
    planner_source_sha256: str
    planner_algorithm_digest: str


def _load_preregistration(
    path: str | Path, *, expected_record_digest: str
) -> PrototypePairPreregistrationPins:
    _source, payload = _stable_read_regular(path, label="preregistration")
    raw = _strict_json_object(payload, label="preregistration")
    expected_fields = {
        "schema",
        "created_at",
        "scope",
        "seed",
        "source",
        "planner",
        "selection",
        "statistics",
        "execution",
        "authority",
        "claims",
        "record_digest",
    }
    _exact_object(raw, expected_fields, label="preregistration")
    body = {key: value for key, value in raw.items() if key != "record_digest"}
    external = _require_raw_sha256(
        expected_record_digest, "expected preregistration digest"
    )
    if (
        raw["record_digest"] != external
        or canonical_digest(body) != external
        or raw["schema"] != PREREGISTRATION_SCHEMA
        or raw["scope"] != PREREGISTRATION_SCOPE
    ):
        raise PrototypePairCampaignCliError(
            "preregistration differs from its external commitment"
        )
    seed = _exact_object(
        raw["seed"],
        {"value", "provenance", "namespace", "commitment"},
        label="preregistration seed",
    )
    source = _exact_object(
        raw["source"],
        {
            "release_descriptor_digest",
            "corpus_manifest_digest",
            "split_source_digest",
            "task_inventory_digest",
            "historical_seed_digest",
            "exposure_predecessor_digest",
        },
        label="preregistration source",
    )
    planner = _exact_object(
        raw["planner"],
        {"algorithm_id", "source_sha256", "algorithm_digest"},
        label="preregistration planner",
    )
    selection = _exact_object(
        raw["selection"],
        {
            "candidate_count",
            "selected_task_count",
            "drill_task_id",
            "drill_shape_families",
            "plan_digest",
        },
        label="preregistration selection",
    )
    execution = _exact_object(
        raw["execution"],
        {
            "metadata_only_selection",
            "panel_bytes_opened_before_preregistration",
            "action_program_json_authorized",
            "thresholds_must_be_frozen_before_calibration",
            "formula_must_be_frozen_before_query_pixels",
            "cold_replay_must_be_model_free",
            "official_test_authorized",
        },
        label="preregistration execution",
    )
    authority = _exact_object(
        raw["authority"],
        {
            "predicate_authority_id",
            "python_is_canonical_authority",
            "lean_required",
            "lean_defines_artifact_identity",
            "lean_affects_selection_or_decision",
            "optional_secondary_checker_detachable",
        },
        label="preregistration authority",
    )
    claims = _exact_object(
        raw["claims"],
        {
            "targeted_engineering_only",
            "semantics_reused",
            "benchmark_claim_authorized",
            "unseen_claim_authorized",
        },
        label="preregistration claims",
    )
    if execution != {
        "metadata_only_selection": True,
        "panel_bytes_opened_before_preregistration": False,
        "action_program_json_authorized": False,
        "thresholds_must_be_frozen_before_calibration": True,
        "formula_must_be_frozen_before_query_pixels": True,
        "cold_replay_must_be_model_free": True,
        "official_test_authorized": False,
    }:
        raise PrototypePairCampaignCliError("preregistration execution policy differs")
    if authority != {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "lean_required": False,
        "lean_defines_artifact_identity": False,
        "lean_affects_selection_or_decision": False,
        "optional_secondary_checker_detachable": True,
    }:
        raise PrototypePairCampaignCliError("preregistration authority differs")
    if claims != {
        "targeted_engineering_only": True,
        "semantics_reused": True,
        "benchmark_claim_authorized": False,
        "unseen_claim_authorized": False,
    }:
        raise PrototypePairCampaignCliError("preregistration claim scope differs")
    if (
        not isinstance(seed["value"], str)
        or not seed["value"]
        or not isinstance(seed["namespace"], str)
        or not seed["namespace"]
        or not isinstance(seed["provenance"], str)
        or not seed["provenance"]
        or planner["algorithm_id"] != COHORT_ALGORITHM_ID
    ):
        raise PrototypePairCampaignCliError("preregistration seed/planner differs")
    for key in (
        "release_descriptor_digest",
        "corpus_manifest_digest",
        "split_source_digest",
        "task_inventory_digest",
        "historical_seed_digest",
        "exposure_predecessor_digest",
    ):
        _require_address(source[key], f"preregistration {key}")
    return PrototypePairPreregistrationPins(
        record_digest=external,
        selection_seed=seed["value"],
        seed_commitment=_require_address(seed["commitment"], "seed commitment"),
        namespace=seed["namespace"],
        plan_digest=_require_address(selection["plan_digest"], "plan digest"),
        release_descriptor_digest=source["release_descriptor_digest"],
        corpus_manifest_digest=source["corpus_manifest_digest"],
        split_source_digest=source["split_source_digest"],
        task_inventory_digest=source["task_inventory_digest"],
        historical_seed_digest=source["historical_seed_digest"],
        exposure_predecessor_digest=source["exposure_predecessor_digest"],
        planner_source_sha256=_require_raw_sha256(
            planner["source_sha256"], "planner source digest"
        ),
        planner_algorithm_digest=_require_address(
            planner["algorithm_digest"], "planner algorithm digest"
        ),
    )


def _task_inventory_from_split(
    split_bytes: bytes,
    *,
    expected_task_inventory_digest: str,
    release_task_inventory_digest: str,
) -> tuple[str, ...]:
    raw = _strict_json_object(split_bytes, label="official split")
    if set(raw) != _OFFICIAL_SPLIT_GROUPS:
        raise PrototypePairCampaignCliError(
            "official split must contain the exact six released groups"
        )
    groups: list[tuple[str, ...]] = []
    for name in sorted(_OFFICIAL_SPLIT_GROUPS):
        values = raw[name]
        if (
            not isinstance(values, list)
            or any(
                not isinstance(item, str) or not item or item != item.strip()
                for item in values
            )
            or len(values) != len(set(values))
        ):
            raise PrototypePairCampaignCliError(
                f"official split group {name!r} has invalid task IDs"
            )
        groups.append(tuple(values))
    flattened = tuple(item for group in groups for item in group)
    task_ids = tuple(sorted(set(flattened)))
    if not task_ids or len(task_ids) != len(flattened):
        raise PrototypePairCampaignCliError(
            "official split groups do not form one disjoint exhaustive inventory"
        )
    digest = task_id_inventory_digest(task_ids)
    if (
        digest
        != _require_address(
            expected_task_inventory_digest,
            "preregistered task inventory digest",
        )
        or digest
        != _require_address(
            release_task_inventory_digest,
            "release task inventory digest",
        )
    ):
        raise PrototypePairCampaignCliError(
            "split-derived task inventory differs from release/preregistration"
        )
    return task_ids


@dataclass(frozen=True, slots=True)
class VerifiedPrototypePairCampaignMetadata:
    pins: PrototypePairPreregistrationPins
    cohort_plan: PrototypePairCohortPlan
    release_descriptor: OfficialReleaseDescriptor
    split_bytes: bytes
    task_ids: tuple[str, ...]
    historical_seed: HistoricalExposureSeed
    exposure_predecessor: ExposureLedger


def verify_prototype_pair_campaign_metadata(
    *,
    preregistration_path: str | Path,
    expected_preregistration_digest: str,
    cohort_plan_path: str | Path,
    release_descriptor_path: str | Path,
    split_path: str | Path,
    historical_seed_path: str | Path,
    exposure_predecessor_path: str | Path,
) -> VerifiedPrototypePairCampaignMetadata:
    """Cold-replay the checked plan without archive, store, pixel, or model access."""

    pins = _load_preregistration(
        preregistration_path,
        expected_record_digest=expected_preregistration_digest,
    )
    _plan_path, plan_payload = _stable_read_regular(
        cohort_plan_path, label="cohort plan"
    )
    plan_raw = _strict_json_object(plan_payload, label="cohort plan")
    if canonical_json(plan_raw) + b"\n" != plan_payload:
        raise PrototypePairCampaignCliError("cohort plan bytes are not canonical")
    plan = PrototypePairCohortPlan.from_data(plan_raw)
    if (
        plan.record_digest != pins.plan_digest
        or plan.namespace != pins.namespace
        or plan.planner_source_sha256 != pins.planner_source_sha256
        or plan.planner_algorithm_digest != pins.planner_algorithm_digest
    ):
        raise PrototypePairCampaignCliError("cohort plan differs from preregistration")

    _release_path, release_payload = _stable_read_regular(
        release_descriptor_path, label="release descriptor"
    )
    release_raw = _strict_json_object(release_payload, label="release descriptor")
    if canonical_json(release_raw) + b"\n" != release_payload:
        raise PrototypePairCampaignCliError(
            "release descriptor bytes are not canonical"
        )
    release = OfficialReleaseDescriptor.from_dict(release_raw)
    if release.digest != pins.release_descriptor_digest:
        raise PrototypePairCampaignCliError(
            "release descriptor differs from preregistration"
        )

    split_source, split_bytes = _stable_read_regular(
        split_path, label="official split", byte_limit=_MAX_SPLIT_BYTES
    )
    if (
        split_source.name != release.split_filename
        or len(split_bytes) != release.split_size_bytes
        or "sha256:" + hashlib.sha256(split_bytes).hexdigest()
        != release.split_sha256
    ):
        raise PrototypePairCampaignCliError("official split differs from release")
    task_ids = _task_inventory_from_split(
        split_bytes,
        expected_task_inventory_digest=pins.task_inventory_digest,
        release_task_inventory_digest=release.task_ids_sha256,
    )

    historical_source, historical_payload = _stable_read_regular(
        historical_seed_path, label="historical seed"
    )
    _strict_json_object(historical_payload, label="historical seed")
    historical = load_historical_exposure(historical_source, verify_evidence=False)
    _historical_source_after, historical_payload_after = _stable_read_regular(
        historical_source, label="historical seed"
    )
    if historical_payload_after != historical_payload:
        raise PrototypePairCampaignCliError(
            "historical seed changed during verification"
        )
    if historical.seed_digest != pins.historical_seed_digest:
        raise PrototypePairCampaignCliError(
            "historical seed differs from preregistration"
        )

    _ledger_path, ledger_payload = _stable_read_regular(
        exposure_predecessor_path, label="exposure predecessor"
    )
    ledger_raw = _strict_json_object(ledger_payload, label="exposure predecessor")
    exposure = ExposureLedger.from_dict(ledger_raw)
    if (
        exposure.digest != pins.exposure_predecessor_digest
        or exposure.to_json().encode("utf-8") != ledger_payload
    ):
        raise PrototypePairCampaignCliError(
            "exposure predecessor differs from exact preregistration"
        )

    verified = verify_prototype_pair_cohort_plan(
        plan_raw,
        expected_plan_digest=pins.plan_digest,
        release_descriptor=release,
        split_bytes=split_bytes,
        task_ids=task_ids,
        exposure_predecessor=exposure,
        historical_seed=historical,
        selection_seed=pins.selection_seed,
        expected_seed_commitment=pins.seed_commitment,
        expected_release_descriptor_digest=pins.release_descriptor_digest,
        expected_corpus_manifest_digest=pins.corpus_manifest_digest,
        expected_split_source_digest=pins.split_source_digest,
        expected_task_inventory_digest=pins.task_inventory_digest,
        expected_exposure_predecessor_digest=pins.exposure_predecessor_digest,
        expected_historical_seed_digest=pins.historical_seed_digest,
        expected_resolver_policy_digest=semantic_resolver_policy_digest(historical),
        expected_basic_sampler_sha256=OFFICIAL_BASIC_SAMPLER_SHA256,
        expected_basic_generator_sha256=OFFICIAL_BASIC_GENERATOR_SHA256,
    )
    return VerifiedPrototypePairCampaignMetadata(
        pins=pins,
        cohort_plan=verified,
        release_descriptor=release,
        split_bytes=split_bytes,
        task_ids=task_ids,
        historical_seed=historical,
        exposure_predecessor=exposure,
    )


@dataclass(frozen=True, slots=True)
class PythonRuntimeIdentity:
    runtime_id: str
    identity_digest: str
    executable_sha256: str


def snapshot_python_runtime_identity() -> PythonRuntimeIdentity:
    """Bind interpreter metadata and the resolved executable's exact bytes."""

    try:
        executable = Path(sys.executable).resolve(strict=True)
    except OSError as exc:
        raise PrototypePairCampaignCliError(
            "cannot resolve the active Python executable"
        ) from exc
    _path, payload = _stable_read_regular(
        executable,
        label="Python executable",
        byte_limit=_MAX_PYTHON_EXECUTABLE_BYTES,
    )
    executable_sha256 = hashlib.sha256(payload).hexdigest()
    implementation = getattr(sys.implementation, "name", "unknown")
    cache_tag = getattr(sys.implementation, "cache_tag", None)
    version = tuple(sys.version_info[:5])
    runtime_id = f"{implementation}-{version[0]}.{version[1]}.{version[2]}"
    identity = {
        "schema": "gkm.bongard-python-runtime-identity.v1",
        "runtime_id": runtime_id,
        "implementation": implementation,
        "version_info": list(version),
        "version_text": sys.version,
        "hexversion": sys.hexversion,
        "cache_tag": cache_tag,
        "abi_flags": getattr(sys, "abiflags", ""),
        "byteorder": sys.byteorder,
        "executable_sha256": executable_sha256,
        "executable_size_bytes": len(payload),
    }
    return PythonRuntimeIdentity(
        runtime_id=runtime_id,
        identity_digest=canonical_digest(identity),
        executable_sha256=executable_sha256,
    )


class UtcCampaignClock:
    """Thread-safe UTC clock with strictly increasing microsecond timestamps."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last: datetime | None = None

    def now(self, phase: str, subject_id: str, event: str) -> str:
        if not all(
            isinstance(item, str) and item for item in (phase, subject_id, event)
        ):
            raise PrototypePairCampaignCliError("clock event identity is empty")
        with self._lock:
            current = datetime.now(UTC)
            if self._last is not None and current <= self._last:
                current = self._last + timedelta(microseconds=1)
            self._last = current
        return current.isoformat(timespec="microseconds").replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class PreparedPrototypePairCampaignLaunch:
    metadata: VerifiedPrototypePairCampaignMetadata
    identities: PrototypePairExecutionIdentities
    precommit: PrototypePairExecutionPrecommit
    policy_snapshot: CloudPolicyCacheSnapshot
    model_catalog_snapshot: CodexModelCatalogSnapshot
    no_tools_attestation: CodexNoToolsAttestation
    codex_cli_version: str
    codex_launcher_sha256: str
    python_runtime: PythonRuntimeIdentity
    store: PrototypePairCampaignStore
    official_archive: OfficialPanelArchive
    ranker: PrototypeSceneCodexRanker
    configuration: PrototypePairCampaignConfiguration
    clock: UtcCampaignClock


def prepare_prototype_pair_campaign_launch(
    *,
    preregistration_path: str | Path,
    expected_preregistration_digest: str,
    cohort_plan_path: str | Path,
    release_descriptor_path: str | Path,
    split_path: str | Path,
    historical_seed_path: str | Path,
    exposure_predecessor_path: str | Path,
    official_archive_path: str | Path,
    store_root: str | Path,
    expected_codex_launcher_sha256: str,
    model: str = DEFAULT_CAMPAIGN_MODEL,
    reasoning_effort: str = DEFAULT_CAMPAIGN_REASONING_EFFORT,
    absent_upper_ppm: int = DEFAULT_ABSENT_UPPER_PPM,
    present_lower_ppm: int = DEFAULT_PRESENT_LOWER_PPM,
    actor: str = DEFAULT_ACTOR,
    parallel_workers: int = DEFAULT_PARALLEL_WORKERS,
    observer_minutes: int = DEFAULT_OBSERVER_MINUTES,
    ranker_minutes: int = DEFAULT_RANKER_MINUTES,
    verbose: bool = False,
    codex_executable: str = "codex",
    runtime_archive_source_id: str = DEFAULT_RUNTIME_ARCHIVE_SOURCE_ID,
    runtime_verifier_id: str = DEFAULT_RUNTIME_VERIFIER_ID,
) -> PreparedPrototypePairCampaignLaunch:
    """Freeze runtime identity and precommit, then build metadata-only handles."""

    metadata = verify_prototype_pair_campaign_metadata(
        preregistration_path=preregistration_path,
        expected_preregistration_digest=expected_preregistration_digest,
        cohort_plan_path=cohort_plan_path,
        release_descriptor_path=release_descriptor_path,
        split_path=split_path,
        historical_seed_path=historical_seed_path,
        exposure_predecessor_path=exposure_predecessor_path,
    )
    launcher_pin = _require_raw_sha256(
        expected_codex_launcher_sha256, "expected Codex launcher digest"
    )
    policy_snapshot = snapshot_cloud_policy_cache()
    fingerprint = codex_cli_authenticated_fingerprint(
        codex_executable, expected_launcher_digest=launcher_pin
    )
    if (
        not isinstance(fingerprint, Mapping)
        or set(fingerprint) != {"version", "launcher_digest"}
        or fingerprint["launcher_digest"] != launcher_pin
        or not isinstance(fingerprint["version"], str)
        or not fingerprint["version"]
    ):
        raise PrototypePairCampaignCliError(
            "authenticated Codex fingerprint differs from commitment"
        )
    model_catalog_snapshot = snapshot_pinned_model_catalog()
    no_tools_attestation = attest_codex_no_tools(
        executable=codex_executable,
        expected_launcher_digest=launcher_pin,
        model_catalog_snapshot=model_catalog_snapshot,
        cloud_policy_cache_snapshot=policy_snapshot,
    )
    python_runtime = snapshot_python_runtime_identity()
    configuration = PrototypePairCampaignConfiguration(
        actor=actor,
        parallel_workers=parallel_workers,
        observer_minutes=observer_minutes,
        observer_verbose=verbose,
        observer_executable=codex_executable,
        ranker_minutes=ranker_minutes,
        ranker_verbose=verbose,
        ranker_executable=codex_executable,
        runtime_archive_source_id=runtime_archive_source_id,
        runtime_verifier_id=runtime_verifier_id,
    )
    thresholds = tuple(
        PrototypeSceneTagThreshold(tag_id, absent_upper_ppm, present_lower_ppm)
        for tag_id in OPAQUE_TAG_IDS
    )
    threshold_digest = threshold_commitment(thresholds)
    policy_binding = policy_snapshot.binding
    runtime_sources = prototype_pair_campaign_runtime_source_digests()
    ranker_model_digest = prototype_scene_codex_ranker_model_identity_digest(
        model, reasoning_effort
    ).removeprefix("sha256:")
    ranker_environment_digest = prototype_scene_codex_ranker_environment_digest(
        model=model,
        reasoning_effort=reasoning_effort,
        expected_launcher_digest=launcher_pin,
        expected_cloud_policy_cache_binding=policy_binding,
        expected_transport_source_digest=(
            prototype_scene_codex_ranker_transport_source_digest()
        ),
        model_catalog_snapshot=model_catalog_snapshot,
        no_tools_attestation=no_tools_attestation,
    )
    identities = PrototypePairExecutionIdentities.create(
        exposure_predecessor_digest=metadata.exposure_predecessor.digest,
        execution_configuration_digest=configuration.record_digest,
        thresholds=thresholds,
        threshold_commitment=threshold_digest,
        calibration_algorithm_digest=calibration_algorithm_digest(),
        observer_protocol_id=PROTOTYPE_SCENE_OBSERVER_PROTOCOL_ID,
        observer_description_protocol_digest=(
            prototype_rubric_description_protocol_digest()
        ),
        observer_scoring_protocol_digest=prototype_scene_scoring_protocol_digest(),
        observer_environment_digest=prototype_scene_observer_environment_digest(
            model=model,
            reasoning_effort=reasoning_effort,
            expected_launcher_digest=launcher_pin,
            cloud_policy_cache_binding=policy_binding,
            model_catalog_digest=model_catalog_snapshot.raw_digest,
            no_tools_attestation_digest=no_tools_attestation.attestation_digest,
        ),
        observer_model_id=model,
        observer_reasoning_effort=reasoning_effort,
        observer_model_identity_digest=prototype_scene_observer_model_digest(
            model, reasoning_effort
        ),
        ranker_model_id=model,
        ranker_reasoning_effort=reasoning_effort,
        ranker_model_identity_digest=ranker_model_digest,
        ranker_protocol_id=PROTOTYPE_SCENE_CODEX_RANKER_PROTOCOL_ID,
        ranker_protocol_digest=prototype_scene_codex_ranker_protocol_digest(),
        ranker_environment_digest=ranker_environment_digest,
        runner_protocol_id=RUNNER_ID,
        runner_algorithm_digest=prototype_scene_runner_source_digest(),
        codex_cli_version=fingerprint["version"],
        codex_launcher_sha256=launcher_pin,
        cloud_policy_cache_binding=policy_binding,
        codex_model_catalog_snapshot=model_catalog_snapshot,
        codex_no_tools_attestation=no_tools_attestation,
        python_runtime_id=python_runtime.runtime_id,
        python_runtime_identity_digest=python_runtime.identity_digest,
        runtime_source_digests=runtime_sources,
    )
    precommit = prepare_prototype_pair_execution_precommit(
        cohort_plan=metadata.cohort_plan,
        identities=identities,
        expected_cohort_plan_digest=metadata.cohort_plan.record_digest,
        expected_identity_bundle_digest=identities.record_digest,
        expected_exposure_predecessor_digest=metadata.exposure_predecessor.digest,
    )
    precommit = verify_prototype_pair_execution_precommit(
        precommit.to_data(),
        cohort_plan=metadata.cohort_plan.to_data(),
        identities=identities.to_data(),
        expected_precommit_digest=precommit.record_digest,
        expected_cohort_plan_digest=metadata.cohort_plan.record_digest,
        expected_identity_bundle_digest=identities.record_digest,
        expected_exposure_predecessor_digest=metadata.exposure_predecessor.digest,
    )

    # The cold verification above deliberately reconstructs every embedded
    # authority from canonical bytes.  From this point onward the reconstructed
    # precommit is the authority, so all live adapters must retain its exact
    # frozen objects rather than the equivalent pre-serialization instances.
    identities = precommit.identities
    model_catalog_snapshot = identities.codex_model_catalog_snapshot
    no_tools_attestation = identities.codex_no_tools_attestation

    # These constructors inspect only persistence layout and ZIP metadata.
    # The campaign coordinator persists phase zero before it invokes
    # OfficialPanelArchive.read_panel or ReleasedOfficialPanel.release.
    store = PrototypePairCampaignStore.open(store_root)
    official_archive = OfficialPanelArchive.load(
        metadata.release_descriptor,
        official_archive_path,
        expected_release_descriptor_digest=metadata.pins.release_descriptor_digest,
    )
    ranker = PrototypeSceneCodexRanker(
        model=model,
        expected_launcher_digest=launcher_pin,
        cloud_policy_cache_snapshot=policy_snapshot,
        expected_cloud_policy_cache_binding=policy_binding,
        expected_transport_source_digest=(
            prototype_scene_codex_ranker_transport_source_digest()
        ),
        model_catalog_snapshot=model_catalog_snapshot,
        no_tools_attestation=no_tools_attestation,
        reasoning_effort=reasoning_effort,
        minutes=configuration.ranker_minutes,
        verbose=configuration.ranker_verbose,
        executable=configuration.ranker_executable,
    )
    return PreparedPrototypePairCampaignLaunch(
        metadata=metadata,
        identities=identities,
        precommit=precommit,
        policy_snapshot=policy_snapshot,
        model_catalog_snapshot=model_catalog_snapshot,
        no_tools_attestation=no_tools_attestation,
        codex_cli_version=fingerprint["version"],
        codex_launcher_sha256=launcher_pin,
        python_runtime=python_runtime,
        store=store,
        official_archive=official_archive,
        ranker=ranker,
        configuration=configuration,
        clock=UtcCampaignClock(),
    )


def _campaign_entrypoint() -> Callable[..., object]:
    from bongard import prototype_pair_campaign as campaign_module

    entrypoint = getattr(campaign_module, "run_prototype_pair_campaign", None)
    if not callable(entrypoint):
        raise PrototypePairCampaignCliError(
            "run_prototype_pair_campaign is unavailable"
        )
    return entrypoint


def dispatch_prepared_prototype_pair_campaign(
    prepared: PreparedPrototypePairCampaignLaunch,
) -> object:
    """Invoke the coordinator exactly once; it owns phase-zero persistence."""

    if not isinstance(prepared, PreparedPrototypePairCampaignLaunch):
        raise TypeError("prepared must be PreparedPrototypePairCampaignLaunch")
    metadata = prepared.metadata
    return _campaign_entrypoint()(
        cohort_plan=metadata.cohort_plan,
        precommit=prepared.precommit,
        exposure_predecessor=metadata.exposure_predecessor,
        release_descriptor=metadata.release_descriptor,
        official_archive=prepared.official_archive,
        store=prepared.store,
        clock=prepared.clock,
        configuration=prepared.configuration,
        cloud_policy_cache_snapshot=prepared.policy_snapshot,
        model_catalog_snapshot=prepared.model_catalog_snapshot,
        no_tools_attestation=prepared.no_tools_attestation,
        description_transport=run_codex_named_images_structured,
        scene_transport=run_codex_named_images_structured,
        ranker=prepared.ranker,
        observed_codex_cli_version=prepared.codex_cli_version,
        observed_codex_launcher_sha256=prepared.codex_launcher_sha256,
        observed_python_runtime_id=prepared.python_runtime.runtime_id,
        observed_python_runtime_identity_digest=(
            prepared.python_runtime.identity_digest
        ),
        expected_precommit_digest=prepared.precommit.record_digest,
        expected_cohort_plan_digest=metadata.cohort_plan.record_digest,
        expected_identity_bundle_digest=prepared.identities.record_digest,
        expected_exposure_predecessor_digest=metadata.exposure_predecessor.digest,
    )


def run_prototype_pair_campaign_from_paths(**kwargs: Any) -> object:
    """Prepare from explicit paths and dispatch one official campaign."""

    return dispatch_prepared_prototype_pair_campaign(
        prepare_prototype_pair_campaign_launch(**kwargs)
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the preregistered prototype-pair engineering campaign"
    )
    for flag in (
        "preregistration",
        "cohort-plan",
        "release-descriptor",
        "split",
        "historical-seed",
        "exposure-predecessor",
        "official-archive",
        "store-root",
    ):
        parser.add_argument(f"--{flag}", required=True)
    parser.add_argument("--expected-preregistration-digest", required=True)
    parser.add_argument("--codex-launcher-sha256", required=True)
    parser.add_argument("--codex-executable", default="codex")
    parser.add_argument("--model", default=DEFAULT_CAMPAIGN_MODEL)
    parser.add_argument(
        "--reasoning-effort", default=DEFAULT_CAMPAIGN_REASONING_EFFORT
    )
    parser.add_argument(
        "--absent-upper-ppm", type=int, default=DEFAULT_ABSENT_UPPER_PPM
    )
    parser.add_argument(
        "--present-lower-ppm", type=int, default=DEFAULT_PRESENT_LOWER_PPM
    )
    parser.add_argument("--actor", default=DEFAULT_ACTOR)
    parser.add_argument(
        "--parallel-workers", type=int, default=DEFAULT_PARALLEL_WORKERS
    )
    parser.add_argument(
        "--observer-minutes", type=int, default=DEFAULT_OBSERVER_MINUTES
    )
    parser.add_argument("--ranker-minutes", type=int, default=DEFAULT_RANKER_MINUTES)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--runtime-archive-source-id", default=DEFAULT_RUNTIME_ARCHIVE_SOURCE_ID
    )
    parser.add_argument("--runtime-verifier-id", default=DEFAULT_RUNTIME_VERIFIER_ID)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_prototype_pair_campaign_from_paths(
        preregistration_path=args.preregistration,
        expected_preregistration_digest=args.expected_preregistration_digest,
        cohort_plan_path=args.cohort_plan,
        release_descriptor_path=args.release_descriptor,
        split_path=args.split,
        historical_seed_path=args.historical_seed,
        exposure_predecessor_path=args.exposure_predecessor,
        official_archive_path=args.official_archive,
        store_root=args.store_root,
        expected_codex_launcher_sha256=args.codex_launcher_sha256,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        absent_upper_ppm=args.absent_upper_ppm,
        present_lower_ppm=args.present_lower_ppm,
        actor=args.actor,
        parallel_workers=args.parallel_workers,
        observer_minutes=args.observer_minutes,
        ranker_minutes=args.ranker_minutes,
        verbose=args.verbose,
        codex_executable=args.codex_executable,
        runtime_archive_source_id=args.runtime_archive_source_id,
        runtime_verifier_id=args.runtime_verifier_id,
    )
    status = getattr(getattr(result, "status", None), "value", None)
    digest = getattr(result, "record_digest", None)
    if not isinstance(status, str) or not isinstance(digest, str):
        raise PrototypePairCampaignCliError(
            "campaign entrypoint returned no canonical status/digest"
        )
    sys.stdout.buffer.write(
        canonical_json({"campaign_digest": digest, "status": status}) + b"\n"
    )
    return 0


__all__ = (
    "DEFAULT_ABSENT_UPPER_PPM",
    "DEFAULT_CAMPAIGN_MODEL",
    "DEFAULT_CAMPAIGN_REASONING_EFFORT",
    "DEFAULT_PRESENT_LOWER_PPM",
    "PreparedPrototypePairCampaignLaunch",
    "PrototypePairCampaignCliError",
    "PythonRuntimeIdentity",
    "UtcCampaignClock",
    "VerifiedPrototypePairCampaignMetadata",
    "dispatch_prepared_prototype_pair_campaign",
    "main",
    "prepare_prototype_pair_campaign_launch",
    "run_prototype_pair_campaign_from_paths",
    "snapshot_python_runtime_identity",
    "verify_prototype_pair_campaign_metadata",
)


if __name__ == "__main__":  # pragma: no cover - exercised through the real CLI
    raise SystemExit(main())
