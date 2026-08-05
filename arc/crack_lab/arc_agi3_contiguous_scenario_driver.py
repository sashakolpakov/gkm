#!/usr/bin/env python3
"""Fail-closed production S01--S12 scenario receipt driver.

This driver deliberately distinguishes executable unit/conformance tests from
production-path observations.  ``run`` emits one immutable typed receipt for
every scenario.  A scenario without an in-tree production observer is
``BLOCKED``; no caller-provided JSON, pytest outcome, or boolean can turn it
into ``PASS``.  ``verify`` reopens every receipt and every control/runtime
binding without running or mutating the campaign.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


SCHEMA = 1
KIND = "arc_agi3_contiguous_scenario_driver"
SCENARIO_KIND = "arc_agi3_contiguous_scenario_receipt"
MAX_FILE_BYTES = 16 * 1024 * 1024
OBSERVATION_KIND = "arc_agi3_contiguous_production_observation"
OBSERVATION_ID_RE = re.compile(r"[a-z][a-z0-9_]{0,127}")
EVIDENCE_KIND_RE = re.compile(r"[a-z][a-z0-9_.:-]{0,127}")


class ScenarioDriverError(RuntimeError):
    """Scenario execution or immutable reverification failed closed."""


@dataclass(frozen=True)
class ScenarioDefinition:
    scenario_id: str
    owner: str
    required_observations: tuple[str, ...]


@dataclass(frozen=True)
class ScenarioExecutionContext:
    """Control-bound inputs available to one in-tree observer."""

    repository: Path
    runtime_manifest_path: Path
    runtime_manifest_sha256: str
    output_root: Path


@dataclass(frozen=True)
class ScenarioEvidence:
    """One immutable machine-evidence object reopened by ``verify``."""

    observation_id: str
    kind: str
    path: str
    sha256: str
    bytes: int


@dataclass(frozen=True)
class ProductionObservation:
    """Typed result of a genuine, in-tree production observer."""

    schema: int
    kind: str
    scenario_id: str
    owner: str
    status: str
    machine_observed: bool
    required_observations: tuple[str, ...]
    evidence: tuple[ScenarioEvidence, ...]
    evidence_sha256: str


class ProductionObserver(Protocol):
    """Execute and independently reopen one scenario's real evidence."""

    def observe(
        self,
        definition: ScenarioDefinition,
        context: ScenarioExecutionContext,
    ) -> ProductionObservation:
        ...

    def verify(
        self,
        definition: ScenarioDefinition,
        observation: ProductionObservation,
        evidence: Mapping[str, bytes],
    ) -> None:
        ...


SCENARIOS = (
    ScenarioDefinition(
        "S01",
        "arc_agi3_contiguous_s01_v1",
        (
            "fresh_zero_solver",
            "live_pinned_model_turn",
            "container_created_source",
            "independent_exact_replay",
            "typed_token_usage",
        ),
    ),
    ScenarioDefinition(
        "S02",
        "arc_agi3_contiguous_s02_v1",
        (
            "external_auth_exchange",
            "controller_egress_readiness",
            "controller_egress_allow_deny_probe",
            "credential_and_canary_containment",
            "rotation_and_unwritable_state_inverse",
        ),
    ),
    ScenarioDefinition(
        "S03",
        "arc_agi3_contiguous_s03_v1",
        (
            "effective_oci_isolation",
            "networkless_named_volume_arena",
            "attack_matrix",
            "bounded_terminal_streams",
            "descendant_free_teardown",
        ),
    ),
    ScenarioDefinition(
        "S04",
        "arc_agi3_contiguous_s04_v1",
        (
            "live_protocol_lifecycle",
            "method_phase_cardinality_firewall",
            "token_redaction",
            "lost_response_exactly_once",
            "binding_forgery_inverse",
        ),
    ),
    ScenarioDefinition(
        "S05",
        "arc_agi3_contiguous_s05_v1",
        (
            "six_live_lanes",
            "disjoint_process_and_thread_identities",
            "cross_lane_substitution_inverse",
            "sibling_survives_teardown",
        ),
    ),
    ScenarioDefinition(
        "S06",
        "arc_agi3_contiguous_s06_v1",
        (
            "unlimited_and_finite_budget_paths",
            "monotone_effort_escalation",
            "auxiliary_dispatch_thresholds",
            "soft_deadline_drain_without_interrupt",
            "typed_provider_settlement",
        ),
    ),
    ScenarioDefinition(
        "S07",
        "arc_agi3_contiguous_s07_v1",
        (
            "real_daemon_sigkill_matrix",
            "fresh_supervisor_recovery",
            "exact_once_promotion_or_quarantine",
            "zero_leaked_processes_and_containers",
        ),
    ),
    ScenarioDefinition(
        "S08",
        "arc_agi3_contiguous_s08_v1",
        (
            "acknowledged_thread_rebinding",
            "matching_wip_recovery",
            "mismatch_and_taint_quarantine",
            "coherence_reset_new_thread",
        ),
    ),
    ScenarioDefinition(
        "S09",
        "arc_agi3_contiguous_s09_v2",
        (
            "production_candidate_origin",
            "exact_k_to_k_plus_one",
            "pre_debrief_source_boundary",
            "context_specific_action7_exact_or_reconstruct",
            "reward_boundary_absorbing_no_action7",
            "fresh_replay_from_sealed_reward",
            "negative_candidate_matrix",
        ),
    ),
    ScenarioDefinition(
        "S10",
        "arc_agi3_contiguous_s10_v1",
        (
            "promotion_fault_matrix",
            "atomic_pointer_recovery",
            "acknowledgement_loss_exact_once",
            "byte_identical_old_or_new_version",
        ),
    ),
    ScenarioDefinition(
        "S11",
        "arc_agi3_contiguous_s11_v1",
        (
            "complete_immutable_evidence",
            "controller_state_inventory_and_scan",
            "mutation_and_deletion_matrix",
            "terminal_retention_recovery",
        ),
    ),
    ScenarioDefinition(
        "S12",
        "arc_agi3_contiguous_s12_v1",
        (
            "authoritative_25_game_183_boundary_inventory",
            "schema_v2_exact_release",
            "independent_release_reverification",
            "scorecard_input_binding",
        ),
    ),
)


# Production observers are intentionally in-tree implementations, never
# dynamically loaded names or caller-supplied status documents.  Empty means
# launch remains closed.  The driver nevertheless implements the complete
# typed PASS path so adding a future observer cannot bypass immutable evidence
# reopening or silently turn a missing scenario into authority.
PRODUCTION_OBSERVERS: Mapping[str, ProductionObserver] = {}


def _production_observer_registry() -> Mapping[str, ProductionObserver]:
    expected = {item.scenario_id for item in SCENARIOS}
    if (
        not isinstance(PRODUCTION_OBSERVERS, Mapping)
        or not set(PRODUCTION_OBSERVERS).issubset(expected)
        or any(
            not callable(getattr(observer, "observe", None))
            or not callable(getattr(observer, "verify", None))
            for observer in PRODUCTION_OBSERVERS.values()
        )
    ):
        raise ScenarioDriverError(
            "production observer registry is malformed"
        )
    return PRODUCTION_OBSERVERS


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _reject_symlinked_components(path: Path, *, label: str) -> None:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        if os.path.lexists(current) and current.is_symlink():
            raise ScenarioDriverError(
                f"{label} contains a symlinked component"
            )


def _path_pointer(path: Path, *, label: str) -> tuple[tuple[Any, ...], ...]:
    """Bind every absolute path component without following a symlink."""

    absolute = Path(os.path.abspath(path))
    if absolute != path or not absolute.is_absolute():
        raise ScenarioDriverError(f"{label} path is not canonical")
    current = Path(absolute.anchor)
    result: list[tuple[Any, ...]] = []
    for component in absolute.parts[1:]:
        current /= component
        metadata = current.stat(follow_symlinks=False)
        if stat.S_ISLNK(metadata.st_mode):
            raise ScenarioDriverError(
                f"{label} contains a symlinked component"
            )
        result.append((
            str(current),
            metadata.st_dev,
            metadata.st_ino,
            stat.S_IFMT(metadata.st_mode),
            stat.S_IMODE(metadata.st_mode),
        ))
    return tuple(result)


def _directory_identity(path: Path, *, label: str) -> tuple[int, int]:
    _reject_symlinked_components(path, label=label)
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ScenarioDriverError(f"{label} is unavailable") from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise ScenarioDriverError(
            f"{label} is not a private host-owned directory"
        )
    return metadata.st_dev, metadata.st_ino


def _read_regular(
    path: Path,
    *,
    label: str,
    required_mode: int | None = None,
    require_owner: bool = False,
) -> bytes:
    selected = Path(path)
    if not selected.is_absolute() or selected.is_symlink():
        raise ScenarioDriverError(f"{label} path is not canonical")
    try:
        pointer_before = _path_pointer(selected, label=label)
        parent_descriptor = os.open(
            selected.parent,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
    except (OSError, ScenarioDriverError) as exc:
        raise ScenarioDriverError(
            f"{label} is unavailable or aliased"
        ) from exc
    descriptor: int | None = None
    try:
        parent_before = os.fstat(parent_descriptor)
        if not stat.S_ISDIR(parent_before.st_mode):
            raise ScenarioDriverError(
                f"{label} parent is not a directory"
            )
        descriptor = os.open(
            selected.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
        before = os.fstat(descriptor)
        linked_before = os.stat(
            selected.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        stable_file_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_uid",
            "st_gid",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size > MAX_FILE_BYTES
            or (
                required_mode is not None
                and stat.S_IMODE(before.st_mode) != required_mode
            )
            or (require_owner and before.st_uid != os.getuid())
            or any(
                getattr(before, field) != getattr(linked_before, field)
                for field in stable_file_fields
            )
        ):
            raise ScenarioDriverError(
                f"{label} is not a bounded unaliased file"
            )
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            if not block:
                raise ScenarioDriverError(
                    f"{label} changed during observation"
                )
            chunks.append(block)
            remaining -= len(block)
        if os.read(descriptor, 1):
            raise ScenarioDriverError(
                f"{label} grew during observation"
            )
        after = os.fstat(descriptor)
        linked_after = os.stat(
            selected.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        parent_after = os.fstat(parent_descriptor)
        parent_linked_after = selected.parent.stat(
            follow_symlinks=False
        )
        stable_parent_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_uid",
            "st_gid",
        )
        if (
            any(
                getattr(before, field) != getattr(after, field)
                or getattr(after, field) != getattr(linked_after, field)
                for field in stable_file_fields
            )
            or any(
                getattr(parent_before, field)
                != getattr(parent_after, field)
                or getattr(parent_after, field)
                != getattr(parent_linked_after, field)
                for field in stable_parent_fields
            )
            or _path_pointer(selected, label=label) != pointer_before
        ):
            raise ScenarioDriverError(
                f"{label} pointer or metadata changed during observation"
            )
        return b"".join(chunks)
    except ScenarioDriverError:
        raise
    except OSError as exc:
        raise ScenarioDriverError(
            f"{label} changed during observation"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_descriptor)


def _write_new(path: Path, body: Mapping[str, Any]) -> str:
    payload = canonical_json(dict(body))
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise ScenarioDriverError(
                    "scenario receipt write made no progress"
                )
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return sha256(payload)


def _evidence_projection(
    evidence: Sequence[ScenarioEvidence],
) -> list[dict[str, Any]]:
    return [asdict(item) for item in evidence]


def _observation_projection(
    observation: ProductionObservation,
) -> dict[str, Any]:
    return {
        "schema": observation.schema,
        "kind": observation.kind,
        "scenario_id": observation.scenario_id,
        "owner": observation.owner,
        "status": observation.status,
        "machine_observed": observation.machine_observed,
        "required_observations": list(
            observation.required_observations
        ),
        "evidence": _evidence_projection(observation.evidence),
        "evidence_sha256": observation.evidence_sha256,
    }


def _observation_from_mapping(value: object) -> ProductionObservation:
    fields = {
        "schema",
        "kind",
        "scenario_id",
        "owner",
        "status",
        "machine_observed",
        "required_observations",
        "evidence",
        "evidence_sha256",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ScenarioDriverError(
            "production observation schema is not exact"
        )
    required = value["required_observations"]
    rows = value["evidence"]
    if not isinstance(required, list) or not isinstance(rows, list):
        raise ScenarioDriverError(
            "production observation collections are malformed"
        )
    evidence_fields = {"observation_id", "kind", "path", "sha256", "bytes"}
    evidence: list[ScenarioEvidence] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != evidence_fields:
            raise ScenarioDriverError(
                "production evidence reference schema is not exact"
            )
        try:
            evidence.append(ScenarioEvidence(**row))
        except TypeError as exc:
            raise ScenarioDriverError(
                "production evidence reference cannot be typed"
            ) from exc
    try:
        return ProductionObservation(
            schema=value["schema"],
            kind=value["kind"],
            scenario_id=value["scenario_id"],
            owner=value["owner"],
            status=value["status"],
            machine_observed=value["machine_observed"],
            required_observations=tuple(required),
            evidence=tuple(evidence),
            evidence_sha256=value["evidence_sha256"],
        )
    except TypeError as exc:
        raise ScenarioDriverError(
            "production observation cannot be typed"
        ) from exc


def _validate_production_observation(
    definition: ScenarioDefinition,
    observation: ProductionObservation,
    *,
    observer: ProductionObserver,
    repository: Path,
) -> ProductionObservation:
    """Reopen all evidence and invoke the scenario-specific validator."""

    if (
        not isinstance(observation, ProductionObservation)
        or observation.schema != SCHEMA
        or isinstance(observation.schema, bool)
        or observation.kind != OBSERVATION_KIND
        or observation.scenario_id != definition.scenario_id
        or observation.owner != definition.owner
        or observation.status != "PASS"
        or observation.machine_observed is not True
        or observation.required_observations
        != definition.required_observations
        or not isinstance(observation.evidence, tuple)
        or len(observation.evidence)
        != len(definition.required_observations)
        or not isinstance(observation.evidence_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", observation.evidence_sha256)
        is None
    ):
        raise ScenarioDriverError(
            f"{definition.scenario_id} production observation is malformed"
        )
    if observation.evidence_sha256 != sha256(
        canonical_json(_evidence_projection(observation.evidence))
    ):
        raise ScenarioDriverError(
            f"{definition.scenario_id} evidence inventory digest differs"
        )
    expected_ids = set(definition.required_observations)
    observed_ids: set[str] = set()
    observed_paths: set[str] = set()
    reopened: dict[str, bytes] = {}
    repository = Path(os.path.abspath(repository))
    for item in observation.evidence:
        if (
            not isinstance(item, ScenarioEvidence)
            or OBSERVATION_ID_RE.fullmatch(item.observation_id) is None
            or EVIDENCE_KIND_RE.fullmatch(item.kind) is None
            or item.observation_id not in expected_ids
            or item.observation_id in observed_ids
            or not isinstance(item.path, str)
            or not isinstance(item.sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", item.sha256) is None
            or not isinstance(item.bytes, int)
            or isinstance(item.bytes, bool)
            or not 0 < item.bytes <= MAX_FILE_BYTES
        ):
            raise ScenarioDriverError(
                f"{definition.scenario_id} evidence reference is malformed"
            )
        path = Path(item.path)
        if (
            not path.is_absolute()
            or Path(os.path.abspath(path)) != path
            or item.path in observed_paths
        ):
            raise ScenarioDriverError(
                f"{definition.scenario_id} evidence path is not unique and canonical"
            )
        try:
            path.relative_to(repository)
        except ValueError:
            pass
        else:
            raise ScenarioDriverError(
                f"{definition.scenario_id} evidence is inside the control tree"
            )
        raw = _read_regular(
            path,
            label=(
                f"{definition.scenario_id} {item.observation_id} evidence"
            ),
            required_mode=0o400,
            require_owner=True,
        )
        metadata = path.lstat()
        if (
            metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) != 0o400
            or len(raw) != item.bytes
            or sha256(raw) != item.sha256
        ):
            raise ScenarioDriverError(
                f"{definition.scenario_id} evidence bytes or ownership differ"
            )
        observed_ids.add(item.observation_id)
        observed_paths.add(item.path)
        reopened[item.observation_id] = raw
    if observed_ids != expected_ids:
        raise ScenarioDriverError(
            f"{definition.scenario_id} evidence coverage is incomplete"
        )
    try:
        observer.verify(definition, observation, reopened)
    except ScenarioDriverError:
        raise
    except Exception as exc:
        raise ScenarioDriverError(
            f"{definition.scenario_id} scenario-specific verification failed"
        ) from exc
    for item in observation.evidence:
        path = Path(item.path)
        raw = _read_regular(
            path,
            label=(
                f"{definition.scenario_id} {item.observation_id} "
                "post-verification evidence"
            ),
            required_mode=0o400,
            require_owner=True,
        )
        metadata = path.lstat()
        if (
            metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) != 0o400
            or raw != reopened[item.observation_id]
            or len(raw) != item.bytes
            or sha256(raw) != item.sha256
        ):
            raise ScenarioDriverError(
                f"{definition.scenario_id} evidence changed during "
                "scenario-specific verification"
            )
    return observation


def _control_contract(repository: Path) -> str:
    if not repository.is_absolute():
        raise ScenarioDriverError("repository path is not absolute")
    try:
        import arc_agi3_contiguous_conformance as conformance
    except ImportError as exc:
        raise ScenarioDriverError(
            "canonical conformance control is unavailable"
        ) from exc
    return conformance.control_contract_sha256(repository)


def _blocked_receipt(
    definition: ScenarioDefinition,
    *,
    control_contract_sha256: str,
    runtime_manifest_path: Path,
    runtime_manifest_sha256: str,
) -> dict[str, Any]:
    observation = {
        "kind": "driver_capability_scan",
        "machine_observed": True,
        "observer_available": False,
        "reason": "production_observer_not_implemented",
        "missing_observations":
            list(definition.required_observations),
    }
    return {
        "schema": SCHEMA,
        "kind": SCENARIO_KIND,
        "scenario_id": definition.scenario_id,
        "owner": definition.owner,
        "status": "BLOCKED",
        "launch_authority": False,
        "control_contract_sha256": control_contract_sha256,
        "runtime_manifest_path": str(runtime_manifest_path),
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "required_observations":
            list(definition.required_observations),
        "observation": observation,
        "observation_sha256": sha256(canonical_json(observation)),
    }


def _pass_receipt(
    definition: ScenarioDefinition,
    *,
    control_contract_sha256: str,
    runtime_manifest_path: Path,
    runtime_manifest_sha256: str,
    observation: ProductionObservation,
) -> dict[str, Any]:
    projection = _observation_projection(observation)
    return {
        "schema": SCHEMA,
        "kind": SCENARIO_KIND,
        "scenario_id": definition.scenario_id,
        "owner": definition.owner,
        "status": "PASS",
        # A single scenario never authorizes launch.  Only the exact ordered
        # aggregate of all twelve independently verified receipts may do so.
        "launch_authority": False,
        "control_contract_sha256": control_contract_sha256,
        "runtime_manifest_path": str(runtime_manifest_path),
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "required_observations":
            list(definition.required_observations),
        "observation": projection,
        "observation_sha256": sha256(canonical_json(projection)),
    }


def run(
    *,
    repository: Path,
    runtime_manifest_path: Path,
    runtime_manifest_sha256: str,
    output_root: Path,
) -> dict[str, Any]:
    if (
        not output_root.is_absolute()
        or output_root.exists()
        or output_root.is_symlink()
    ):
        raise ScenarioDriverError(
            "scenario output root must be exclusively new"
        )
    runtime_bytes = _read_regular(
        runtime_manifest_path, label="Python runtime manifest"
    )
    if sha256(runtime_bytes) != runtime_manifest_sha256:
        raise ScenarioDriverError("runtime manifest digest differs")
    control_digest = _control_contract(repository)
    output_root.mkdir(mode=0o700)
    scenario_root = output_root / "scenarios"
    scenario_root.mkdir(mode=0o700)
    rows: list[dict[str, Any]] = []
    blockers: list[dict[str, str]] = []
    observers = _production_observer_registry()
    context = ScenarioExecutionContext(
        repository=repository,
        runtime_manifest_path=runtime_manifest_path,
        runtime_manifest_sha256=runtime_manifest_sha256,
        output_root=output_root,
    )
    for definition in SCENARIOS:
        observer = observers.get(definition.scenario_id)
        if observer is None:
            body = _blocked_receipt(
                definition,
                control_contract_sha256=control_digest,
                runtime_manifest_path=runtime_manifest_path,
                runtime_manifest_sha256=runtime_manifest_sha256,
            )
            blockers.append({
                "scenario_id": definition.scenario_id,
                "reason": "production_observer_not_implemented",
            })
        else:
            try:
                observation = observer.observe(definition, context)
            except ScenarioDriverError:
                raise
            except Exception as exc:
                raise ScenarioDriverError(
                    f"{definition.scenario_id} production observer failed"
                ) from exc
            observation = _validate_production_observation(
                definition,
                observation,
                observer=observer,
                repository=repository,
            )
            body = _pass_receipt(
                definition,
                control_contract_sha256=control_digest,
                runtime_manifest_path=runtime_manifest_path,
                runtime_manifest_sha256=runtime_manifest_sha256,
                observation=observation,
            )
        path = scenario_root / f"{definition.scenario_id}.json"
        digest = _write_new(path, body)
        rows.append(
            {
                "scenario_id": definition.scenario_id,
                "owner": definition.owner,
                "path": str(path),
                "sha256": digest,
                "status": body["status"],
            }
        )
    all_pass = all(row["status"] == "PASS" for row in rows)
    aggregate = {
        "schema": SCHEMA,
        "kind": KIND,
        "mode": "run",
        "status": "PASS" if all_pass else "BLOCKED",
        "launch_authority": all_pass,
        "control_contract_sha256": control_digest,
        "runtime_manifest_path": str(runtime_manifest_path),
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "scenario_receipts": rows,
        "scenario_receipts_sha256": sha256(canonical_json(rows)),
        "blockers": blockers,
    }
    aggregate_path = output_root / "scenario_driver_receipt.json"
    aggregate_sha256 = _write_new(aggregate_path, aggregate)
    return {
        **aggregate,
        "receipt_path": str(aggregate_path),
        "receipt_sha256": aggregate_sha256,
    }


def verify(receipt_path: Path, *, repository: Path) -> dict[str, Any]:
    receipt_path = Path(receipt_path)
    if (
        receipt_path.name != "scenario_driver_receipt.json"
        or not receipt_path.is_absolute()
    ):
        raise ScenarioDriverError(
            "scenario driver receipt has the wrong canonical path"
        )
    output_root = receipt_path.parent
    scenario_root = output_root / "scenarios"
    output_identity = _directory_identity(
        output_root, label="scenario output root"
    )
    scenario_identity = _directory_identity(
        scenario_root, label="scenario receipt root"
    )
    raw = _read_regular(
        receipt_path,
        label="scenario driver receipt",
        required_mode=0o400,
        require_owner=True,
    )
    try:
        aggregate = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ScenarioDriverError(
            "scenario driver receipt is malformed"
        ) from exc
    if not isinstance(aggregate, dict):
        raise ScenarioDriverError(
            "scenario driver receipt is malformed"
        )
    runtime_path = Path(str(aggregate.get("runtime_manifest_path", "")))
    runtime_raw = _read_regular(
        runtime_path, label="reopened Python runtime manifest"
    )
    rows = aggregate.get("scenario_receipts")
    expected_ids = [item.scenario_id for item in SCENARIOS]
    observers = _production_observer_registry()
    expected_statuses = [
        (
            "PASS"
            if definition.scenario_id in observers
            else "BLOCKED"
        )
        for definition in SCENARIOS
    ]
    expected_blockers = [
        {
            "scenario_id": definition.scenario_id,
            "reason": "production_observer_not_implemented",
        }
        for definition in SCENARIOS
        if definition.scenario_id not in observers
    ]
    all_pass = expected_statuses == ["PASS"] * len(SCENARIOS)
    aggregate_keys = {
        "schema",
        "kind",
        "mode",
        "status",
        "launch_authority",
        "control_contract_sha256",
        "runtime_manifest_path",
        "runtime_manifest_sha256",
        "scenario_receipts",
        "scenario_receipts_sha256",
        "blockers",
    }
    if (
        not isinstance(aggregate, dict)
        or set(aggregate) != aggregate_keys
        or raw != canonical_json(aggregate)
        or aggregate.get("schema") != SCHEMA
        or aggregate.get("kind") != KIND
        or aggregate.get("mode") != "run"
        or aggregate.get("status")
        != ("PASS" if all_pass else "BLOCKED")
        or aggregate.get("launch_authority") is not all_pass
        or aggregate.get("control_contract_sha256")
        != _control_contract(repository)
        or aggregate.get("runtime_manifest_sha256")
        != sha256(runtime_raw)
        or not isinstance(rows, list)
        or any(not isinstance(row, dict) for row in rows)
        or [row.get("scenario_id") for row in rows]
        != expected_ids
        or [row.get("status") for row in rows]
        != expected_statuses
        or aggregate.get("scenario_receipts_sha256")
        != sha256(canonical_json(rows))
        or aggregate.get("blockers") != expected_blockers
    ):
        raise ScenarioDriverError(
            "scenario aggregate binding differs"
        )
    expected_names = {
        f"{definition.scenario_id}.json" for definition in SCENARIOS
    }
    if set(os.listdir(scenario_root)) != expected_names:
        raise ScenarioDriverError(
            "scenario receipt root inventory differs"
        )
    receipt_keys = {
        "schema",
        "kind",
        "scenario_id",
        "owner",
        "status",
        "launch_authority",
        "control_contract_sha256",
        "runtime_manifest_path",
        "runtime_manifest_sha256",
        "required_observations",
        "observation",
        "observation_sha256",
    }
    verified_receipts: dict[Path, bytes] = {}
    for definition, row, expected_status in zip(
        SCENARIOS, rows, expected_statuses, strict=True
    ):
        path = scenario_root / f"{definition.scenario_id}.json"
        body_raw = _read_regular(
            path,
            label=f"{definition.scenario_id} receipt",
            required_mode=0o400,
            require_owner=True,
        )
        verified_receipts[path] = body_raw
        try:
            body = json.loads(body_raw)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ScenarioDriverError(
                f"{definition.scenario_id} receipt is malformed"
            ) from exc
        if (
            not isinstance(row, dict)
            or row != {
                "scenario_id": definition.scenario_id,
                "owner": definition.owner,
                "path": str(path),
                "sha256": sha256(body_raw),
                "status": expected_status,
            }
            or not isinstance(body, dict)
            or set(body) != receipt_keys
            or body_raw != canonical_json(body)
            or body.get("schema") != SCHEMA
            or body.get("kind") != SCENARIO_KIND
            or body.get("scenario_id") != definition.scenario_id
            or body.get("owner") != definition.owner
            or body.get("status") != expected_status
            or body.get("launch_authority") is not False
            or body.get("control_contract_sha256")
            != aggregate["control_contract_sha256"]
            or body.get("runtime_manifest_path") != str(runtime_path)
            or body.get("runtime_manifest_sha256")
            != aggregate["runtime_manifest_sha256"]
            or body.get("required_observations")
            != list(definition.required_observations)
        ):
            raise ScenarioDriverError(
                f"{definition.scenario_id} receipt differs"
            )
        if expected_status == "BLOCKED":
            expected = _blocked_receipt(
                definition,
                control_contract_sha256=(
                    aggregate["control_contract_sha256"]
                ),
                runtime_manifest_path=runtime_path,
                runtime_manifest_sha256=(
                    aggregate["runtime_manifest_sha256"]
                ),
            )
            if body != expected:
                raise ScenarioDriverError(
                    f"{definition.scenario_id} BLOCKED receipt differs"
                )
            continue
        observation_value = body.get("observation")
        if body.get("observation_sha256") != sha256(
            canonical_json(observation_value)
        ):
            raise ScenarioDriverError(
                f"{definition.scenario_id} observation digest differs"
            )
        observation = _observation_from_mapping(observation_value)
        observer = observers.get(definition.scenario_id)
        assert observer is not None
        observation = _validate_production_observation(
            definition,
            observation,
            observer=observer,
            repository=repository,
        )
        expected = _pass_receipt(
            definition,
            control_contract_sha256=(
                aggregate["control_contract_sha256"]
            ),
            runtime_manifest_path=runtime_path,
            runtime_manifest_sha256=(
                aggregate["runtime_manifest_sha256"]
            ),
            observation=observation,
        )
        if body != expected:
            raise ScenarioDriverError(
                f"{definition.scenario_id} PASS receipt differs"
            )
    if _read_regular(
        receipt_path,
        label="reopened scenario driver receipt",
        required_mode=0o400,
        require_owner=True,
    ) != raw or any(
        _read_regular(
            path,
            label=f"reopened {path.stem} scenario receipt",
            required_mode=0o400,
            require_owner=True,
        )
        != expected_raw
        for path, expected_raw in verified_receipts.items()
    ):
        raise ScenarioDriverError(
            "scenario receipt bytes changed during verification"
        )
    if (
        _directory_identity(
            output_root, label="scenario output root"
        )
        != output_identity
        or _directory_identity(
            scenario_root, label="scenario receipt root"
        )
        != scenario_identity
    ):
        raise ScenarioDriverError(
            "scenario receipt directory identity changed during verify"
        )
    return {
        "schema": SCHEMA,
        "kind": KIND,
        "mode": "verify",
        "status": "PASS" if all_pass else "BLOCKED",
        "launch_authority": all_pass,
        "receipt_path": str(receipt_path),
        "receipt_sha256": sha256(raw),
        "control_contract_sha256":
            aggregate["control_contract_sha256"],
        "runtime_manifest_path": str(runtime_path),
        "runtime_manifest_sha256":
            aggregate["runtime_manifest_sha256"],
        "scenario_ids": expected_ids,
        "scenario_statuses": [
            row["status"] for row in rows
        ],
        "scenario_receipts": rows,
        "scenario_receipts_sha256":
            aggregate["scenario_receipts_sha256"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", allow_abbrev=False)
    run_parser.add_argument("--repository", required=True)
    run_parser.add_argument("--runtime-manifest", required=True)
    run_parser.add_argument(
        "--runtime-manifest-sha256", required=True
    )
    run_parser.add_argument("--output-root", required=True)
    verify_parser = subparsers.add_parser(
        "verify", allow_abbrev=False
    )
    verify_parser.add_argument("--repository", required=True)
    verify_parser.add_argument("--receipt", required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        result = run(
            repository=Path(args.repository),
            runtime_manifest_path=Path(args.runtime_manifest),
            runtime_manifest_sha256=args.runtime_manifest_sha256,
            output_root=Path(args.output_root),
        )
    else:
        result = verify(
            Path(args.receipt),
            repository=Path(args.repository),
        )
    sys.stdout.buffer.write(canonical_json(result))
    return 0 if result["status"] == "PASS" else 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ScenarioDriverError, OSError) as error:
        print(f"scenario driver failed: {error}", file=sys.stderr)
        raise SystemExit(70)
