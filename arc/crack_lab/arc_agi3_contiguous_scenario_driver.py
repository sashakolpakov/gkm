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
import stat
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


SCHEMA = 1
KIND = "arc_agi3_contiguous_scenario_driver"
SCENARIO_KIND = "arc_agi3_contiguous_scenario_receipt"
MAX_FILE_BYTES = 16 * 1024 * 1024


class ScenarioDriverError(RuntimeError):
    """Scenario execution or immutable reverification failed closed."""


@dataclass(frozen=True)
class ScenarioDefinition:
    scenario_id: str
    owner: str
    required_observations: tuple[str, ...]


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


# Production observers are intentionally in-tree callables, never dynamically
# loaded names or caller-supplied status documents.  Empty means launch remains
# closed while the driver/receipt/reverification machinery itself is usable.
PRODUCTION_OBSERVERS: Mapping[
    str, Callable[[ScenarioDefinition], Mapping[str, Any]]
] = {}


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


def _read_regular(path: Path, *, label: str) -> bytes:
    if not path.is_absolute() or path.is_symlink():
        raise ScenarioDriverError(f"{label} path is not canonical")
    _reject_symlinked_components(path, label=label)
    before = path.lstat()
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size > MAX_FILE_BYTES
    ):
        raise ScenarioDriverError(
            f"{label} is not a bounded unaliased file"
        )
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        current = os.fstat(descriptor)
        if (
            current.st_dev,
            current.st_ino,
            current.st_size,
        ) != (
            before.st_dev,
            before.st_ino,
            before.st_size,
        ):
            raise ScenarioDriverError(
                f"{label} changed during observation"
            )
        result = bytearray()
        while len(result) <= MAX_FILE_BYTES:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            result.extend(block)
        if len(result) > MAX_FILE_BYTES:
            raise ScenarioDriverError(f"{label} exceeds its bound")
        return bytes(result)
    finally:
        os.close(descriptor)


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
    for definition in SCENARIOS:
        observer = PRODUCTION_OBSERVERS.get(definition.scenario_id)
        if observer is not None:
            raise ScenarioDriverError(
                "production observer execution is not admitted until its "
                "typed result validator is implemented"
            )
        body = _blocked_receipt(
            definition,
            control_contract_sha256=control_digest,
            runtime_manifest_path=runtime_manifest_path,
            runtime_manifest_sha256=runtime_manifest_sha256,
        )
        path = scenario_root / f"{definition.scenario_id}.json"
        digest = _write_new(path, body)
        rows.append(
            {
                "scenario_id": definition.scenario_id,
                "owner": definition.owner,
                "path": str(path),
                "sha256": digest,
                "status": "BLOCKED",
            }
        )
    aggregate = {
        "schema": SCHEMA,
        "kind": KIND,
        "mode": "run",
        "status": "BLOCKED",
        "launch_authority": False,
        "control_contract_sha256": control_digest,
        "runtime_manifest_path": str(runtime_manifest_path),
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "scenario_receipts": rows,
        "scenario_receipts_sha256": sha256(canonical_json(rows)),
        "blockers": [
            {
                "scenario_id": definition.scenario_id,
                "reason": "production_observer_not_implemented",
            }
            for definition in SCENARIOS
        ],
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
        receipt_path, label="scenario driver receipt"
    )
    try:
        aggregate = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ScenarioDriverError(
            "scenario driver receipt is malformed"
        ) from exc
    runtime_path = Path(str(aggregate.get("runtime_manifest_path", "")))
    runtime_raw = _read_regular(
        runtime_path, label="reopened Python runtime manifest"
    )
    rows = aggregate.get("scenario_receipts")
    expected_ids = [item.scenario_id for item in SCENARIOS]
    expected_blockers = [
        {
            "scenario_id": definition.scenario_id,
            "reason": "production_observer_not_implemented",
        }
        for definition in SCENARIOS
    ]
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
        or aggregate.get("status") != "BLOCKED"
        or aggregate.get("launch_authority") is not False
        or aggregate.get("control_contract_sha256")
        != _control_contract(repository)
        or aggregate.get("runtime_manifest_sha256")
        != sha256(runtime_raw)
        or not isinstance(rows, list)
        or [row.get("scenario_id") for row in rows]
        != expected_ids
        or aggregate.get("scenario_receipts_sha256")
        != sha256(canonical_json(rows))
        or aggregate.get("blockers") != expected_blockers
    ):
        raise ScenarioDriverError(
            "scenario aggregate binding differs"
        )
    for definition, row in zip(SCENARIOS, rows, strict=True):
        path = scenario_root / f"{definition.scenario_id}.json"
        body_raw = _read_regular(
            path, label=f"{definition.scenario_id} receipt"
        )
        try:
            body = json.loads(body_raw)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ScenarioDriverError(
                f"{definition.scenario_id} receipt is malformed"
            ) from exc
        expected = _blocked_receipt(
            definition,
            control_contract_sha256=
                aggregate["control_contract_sha256"],
            runtime_manifest_path=runtime_path,
            runtime_manifest_sha256=
                aggregate["runtime_manifest_sha256"],
        )
        if (
            row
            != {
                "scenario_id": definition.scenario_id,
                "owner": definition.owner,
                "path": str(path),
                "sha256": sha256(body_raw),
                "status": "BLOCKED",
            }
            or body != expected
            or body_raw != canonical_json(expected)
        ):
            raise ScenarioDriverError(
                f"{definition.scenario_id} receipt differs"
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
        "status": "BLOCKED",
        "launch_authority": False,
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
