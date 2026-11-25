#!/usr/bin/env python3
"""Sealed two-game pilot gate for the ARC-AGI-3 contiguous campaign.

The pilot lineage is deliberately separate from the canonical 25-game
campaign.  It executes the predeclared ft09 -> lp85 manifest through a supplied
production-stack executor, from four empty roots per game, and emits one
authenticated gate receipt.  It cannot override the production inventory or
publish canonical promotions.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import arc_agi3_contiguous_scheduler as Scheduler
import arc_agi3_contiguous_supervisor as Supervisor


SCHEMA = 1
MAX_CONTROL_BYTES = 4 * 1024 * 1024
PILOT_MANIFEST = (("ft09", 6), ("lp85", 8))
PILOT_AUDITS = (
    "replay",
    "action_protocol",
    "taint",
    "exact_boundary",
    "hashes",
    "manifest",
    "usage",
    "containment",
    "terminal_retention",
    "journal_replay",
)
PILOT_AUDIT_REQUIRED_CHECKS = {
    "replay": (
        "all_levels_replayed_from_zero",
        "parent_boundary_replayed",
        "source_replay_from_zero",
        "exact_target_reached",
    ),
    "action_protocol": (
        "all_actions_match_public_schema",
        "bare_action6_rejected_before_environment_call",
        "malformed_explicit_actions_make_zero_environment_calls",
        "coordinate_action6_is_in_frame",
        "protocol_violation_has_zero_result_authority",
        "safe_step_normalization_exercised",
    ),
    "taint": (
        "source_scan_clean",
        "environment_scan_clean",
        "transcript_scan_clean",
        "controller_canaries_absent",
        "hidden_game_and_environment_access_absent",
    ),
    "exact_boundary": (
        "every_level_has_exact_pre_debrief_boundary",
        "checkpoint_action_count_within_cap",
        "winning_source_snapshot_present",
    ),
    "hashes": (
        "all_references_reopened",
        "all_hashes_match",
        "stale_manifests_absent",
        "unsealed_evidence_absent",
    ),
    "manifest": (
        "schema_v2_exact",
        "predecessor_chain_exact",
        "source_tree_bound",
        "checkpoint_bound",
        "unexpected_entries_absent",
    ),
    "usage": (
        "all_attempts_accounted",
        "double_settlement_absent",
        "infrastructure_failures_have_zero_complexity",
        "protocol_failures_have_zero_result_authority",
        "limit_policy_uniform",
    ),
    "containment": (
        "all_host_children_accounted",
        "live_host_children_absent",
        "attempt_containers_absent",
        "controller_state_isolated",
        "sandbox_failure_rematerialized_without_campaign_mutation",
    ),
    "terminal_retention": (
        "copy_before_purge",
        "retained_inventory_exact",
        "post_retention_replay_passed",
        "stale_archive_entries_absent",
    ),
    "journal_replay": (
        "hash_chain_exact",
        "restart_idempotent",
        "pending_decisions_resolved",
        "promotion_acknowledgements_recovered",
        "state_projection_exact",
    ),
}
EMPTY_ENTRIES_SHA256 = hashlib.sha256(b"[]").hexdigest()
PILOT_MANIFEST_SHA256 = Scheduler.sha256_json([
    {
        "pilot_sequence": index,
        "game": game,
        "authoritative_target": target,
    }
    for index, (game, target) in enumerate(PILOT_MANIFEST, start=1)
])
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PRODUCTION_OUTCOME_NAME = "production_outcome.json"
PILOT_RUN_NAME = "pilot_run.json"


class PilotContractError(RuntimeError):
    """A frozen pilot execution or evidence contract failed."""


@dataclass(frozen=True)
class PilotExecution:
    schema: int
    pilot_sequence: int
    game: str
    authoritative_target: int
    run_root: Path
    artifact_root: Path
    wip_root: Path
    controller_state_root: Path
    scheduler_root: Path
    empty_root_genesis_receipt_path: Path
    empty_root_genesis_receipt_sha256: str
    previous_run_receipt_sha256: str | None
    pilot_manifest_sha256: str
    image_digest: str
    control_contract_sha256: str
    production_stack_attestation_path: Path
    production_stack_attestation_sha256: str


class ProductionPilotExecutor(Protocol):
    production_stack_attestation_path: Path

    def execute_game(self, execution: PilotExecution) -> Path:
        """Return one sealed pilot-run receipt after complete audited solve."""


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise PilotContractError(
            "pilot evidence is not canonical JSON"
        ) from exc


def _strict_json(raw: bytes, *, label: str) -> dict[str, Any]:
    if not raw or len(raw) > MAX_CONTROL_BYTES:
        raise PilotContractError(
            f"{label} is empty or exceeds its byte bound"
        )

    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise PilotContractError(
                    f"{label} has a duplicate JSON key"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=unique,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(token)
            ),
        )
    except PilotContractError:
        raise
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise PilotContractError(
            f"{label} is not strict JSON"
        ) from exc
    if (
        not isinstance(value, dict)
        or _canonical_json(value) + b"\n" != raw
    ):
        raise PilotContractError(
            f"{label} is not a canonical JSON object"
        )
    return value


def _read_control(path: Path, *, label: str) -> bytes:
    try:
        raw = Supervisor._post_incident_meta_read(
            Path(path),
            label=label,
            maximum=MAX_CONTROL_BYTES,
            allow_empty=False,
        )
    except (OSError, Supervisor.SupervisorContractError) as exc:
        raise PilotContractError(
            f"{label} is not owner-held immutable evidence"
        ) from exc
    return raw


def _read_key(path: Path) -> bytes:
    raw = _read_control(path, label="pilot authentication key")
    if len(raw) != 32:
        raise PilotContractError(
            "pilot authentication key must contain exactly 32 bytes"
        )
    return raw


def _authentication(
    key: bytes, body: Mapping[str, object]
) -> str:
    return hmac.new(
        key,
        b"arc-agi3-contiguous-pilot-v1\0"
        + _canonical_json(dict(body)),
        hashlib.sha256,
    ).hexdigest()


def _authenticated_value(
    value: Mapping[str, object],
    *,
    key: bytes,
    label: str,
) -> dict[str, object]:
    selected = dict(value)
    observed_receipt = selected.pop("receipt_sha256", None)
    observed_authentication = selected.pop(
        "host_authentication_sha256", None
    )
    receipt = hashlib.sha256(_canonical_json(selected)).hexdigest()
    authentication = _authentication(key, selected)
    if (
        not isinstance(observed_receipt, str)
        or not isinstance(observed_authentication, str)
        or not hmac.compare_digest(observed_receipt, receipt)
        or not hmac.compare_digest(
            observed_authentication, authentication
        )
    ):
        raise PilotContractError(
            f"{label} is not content-addressed and host-authenticated"
        )
    return selected


def _write_authenticated(
    path: Path,
    body: Mapping[str, object],
    *,
    key: bytes,
    label: str,
) -> tuple[dict[str, object], str]:
    selected = dict(body)
    value = {
        **selected,
        "receipt_sha256": hashlib.sha256(
            _canonical_json(selected)
        ).hexdigest(),
        "host_authentication_sha256":
            _authentication(key, selected),
    }
    try:
        _path, file_sha256 = Supervisor._post_incident_meta_write(
            path, value, label=label
        )
    except Supervisor.SupervisorContractError as exc:
        raise PilotContractError(
            f"{label} could not be published"
        ) from exc
    return value, file_sha256


def _evidence_reference(
    value: object,
    *,
    label: str,
    expected_statuses: frozenset[str] = frozenset({"PASS"}),
) -> tuple[Path, str, dict[str, Any]]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"path", "sha256", "status"}
        or value.get("status") not in expected_statuses
        or not isinstance(value.get("path"), str)
        or not Path(str(value["path"])).is_absolute()
        or not isinstance(value.get("sha256"), str)
        or SHA256_RE.fullmatch(str(value["sha256"])) is None
    ):
        raise PilotContractError(
            f"{label} evidence reference is malformed"
        )
    path = Path(str(value["path"]))
    raw = _read_control(path, label=label)
    if hashlib.sha256(raw).hexdigest() != value["sha256"]:
        raise PilotContractError(
            f"{label} evidence changed"
        )
    receipt = _strict_json(raw, label=label)
    if receipt.get("status") not in expected_statuses:
        raise PilotContractError(
            f"{label} evidence is not exact PASS"
        )
    return path, str(value["sha256"]), receipt


def _control_reference(
    value: object,
    *,
    label: str,
    expected_kind: str,
    expected_status: str | None,
) -> tuple[Path, str, dict[str, Any]]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"path", "sha256", "status"}
        or value.get("status") != "CONTROL"
        or not isinstance(value.get("path"), str)
        or not Path(str(value["path"])).is_absolute()
        or not isinstance(value.get("sha256"), str)
        or SHA256_RE.fullmatch(str(value["sha256"])) is None
    ):
        raise PilotContractError(f"{label} reference is malformed")
    path = Path(str(value["path"]))
    raw = _read_control(path, label=label)
    if hashlib.sha256(raw).hexdigest() != value["sha256"]:
        raise PilotContractError(f"{label} changed")
    receipt = _strict_json(raw, label=label)
    if (
        receipt.get("kind") != expected_kind
        or (
            expected_status is not None
            and receipt.get("status") != expected_status
        )
    ):
        raise PilotContractError(f"{label} is not the expected control")
    return path, str(value["sha256"]), receipt


def _exact_reference_path(
    reference: Mapping[str, object],
    *,
    expected_path: Path,
    label: str,
) -> None:
    if Path(str(reference.get("path"))).resolve() != expected_path.resolve():
        raise PilotContractError(f"{label} path binding changed")


def verify_production_stack_attestation(
    path: Path,
    *,
    expected_sha256: str,
    expected_image_digest: str,
    expected_control_contract_sha256: str,
) -> dict[str, Any]:
    """Reopen the real launch-authority/runtime/executor stack evidence."""

    raw = _read_control(
        Path(path), label="pilot production stack attestation"
    )
    if Path(path).name != "production_stack_attestation.json":
        raise PilotContractError(
            "pilot production stack attestation path is noncanonical"
        )
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise PilotContractError(
            "pilot production stack attestation changed"
        )
    value = _strict_json(
        raw, label="pilot production stack attestation"
    )
    expected_fields = {
        "schema",
        "kind",
        "status",
        "image_digest",
        "control_contract_sha256",
        "scheduler_policy_sha256",
        "meta_protocol_sha256",
        "prelaunch_conformance",
        "runtime_manifest",
        "pilot_executor_source",
        "pilot_executor_control_path",
        "attempt_launch_attestation",
        "backend_launch_attestation",
        "controller_launch_attestation",
        "guardian_start_receipt",
        "substrate_preflight_receipt",
        "host_child_ledger_audit",
        "backend_contract_sha256",
        "input_bundle_contract_sha256",
        "admission_contract_sha256",
        "gkm_legs_source_sha256",
        "perception_seed_sha256",
        "production_stack_scope",
        "full_campaign_launch_authority",
        "production_entry_reachable",
        "synthetic_evidence",
    }
    if (
        set(value) != expected_fields
        or value.get("schema") != SCHEMA
        or value.get("kind")
        != (
            "arc_agi3_contiguous_pilot_"
            "production_stack_attestation"
        )
        or value.get("status") != "PASS"
        or value.get("image_digest") != expected_image_digest
        or value.get("control_contract_sha256")
        != expected_control_contract_sha256
        or value.get("scheduler_policy_sha256")
        != Scheduler.SCHEDULER_POLICY_SHA256
        or value.get("meta_protocol_sha256")
        != Supervisor.POST_INCIDENT_META_PROTOCOL_SHA256
        or any(
            SHA256_RE.fullmatch(str(value.get(name))) is None
            for name in (
                "backend_contract_sha256",
                "input_bundle_contract_sha256",
                "admission_contract_sha256",
                "gkm_legs_source_sha256",
                "perception_seed_sha256",
            )
        )
        or value.get("production_stack_scope")
        != "pilot_only_ft09_then_lp85"
        or not isinstance(value.get("pilot_executor_control_path"), str)
        or value.get("pilot_executor_control_path") == ""
        or value.get("full_campaign_launch_authority") is not False
        or value.get("production_entry_reachable") is not True
        or value.get("synthetic_evidence") is not False
    ):
        raise PilotContractError(
            "pilot production stack is not exact launch authority"
        )
    conformance_path, _conformance_sha, conformance = (
        _evidence_reference(
            value["prelaunch_conformance"],
            label="pilot prelaunch conformance",
        )
    )
    runtime_path, _runtime_sha, runtime = _control_reference(
        value["runtime_manifest"],
        label="pilot runtime manifest",
        expected_kind="arc_agi3_python_runtime_manifest",
        expected_status=None,
    )
    source_reference = value["pilot_executor_source"]
    if (
        not isinstance(source_reference, Mapping)
        or set(source_reference) != {"path", "sha256", "status"}
        or source_reference.get("status") != "CONTROL"
        or not isinstance(source_reference.get("path"), str)
        or not Path(str(source_reference["path"])).is_absolute()
        or SHA256_RE.fullmatch(
            str(source_reference.get("sha256"))
        )
        is None
    ):
        raise PilotContractError(
            "pilot production executor source reference is malformed"
        )
    source_path = Path(str(source_reference["path"]))
    source_raw = _read_control(
        source_path, label="pilot production executor source"
    )
    if hashlib.sha256(source_raw).hexdigest() != source_reference["sha256"]:
        raise PilotContractError("pilot production executor source changed")
    attempt_reference = value["attempt_launch_attestation"]
    if (
        not isinstance(attempt_reference, Mapping)
        or set(attempt_reference) != {"path", "sha256", "status"}
        or attempt_reference.get("status") != "CONTROL"
        or not isinstance(attempt_reference.get("path"), str)
        or not Path(str(attempt_reference["path"])).is_absolute()
        or SHA256_RE.fullmatch(
            str(attempt_reference.get("sha256"))
        )
        is None
    ):
        raise PilotContractError(
            "pilot attempt launch attestation reference is malformed"
        )
    attempt_path = Path(str(attempt_reference["path"]))
    try:
        import arc_agi3_container_backend as Container

        attempt_attestation = Container._load_launch_attestation(
            attempt_path,
            expected_sha256=attempt_reference["sha256"],
        )
    except Exception as exc:
        raise PilotContractError(
            "pilot attempt launch attestation is not genuine"
        ) from exc
    backend_path, _backend_sha, backend_launch = _control_reference(
        value["backend_launch_attestation"],
        label="pilot backend launch attestation",
        expected_kind="contiguous_backend_launch",
        expected_status=None,
    )
    controller_path, controller_sha, controller_launch = (
        _control_reference(
            value["controller_launch_attestation"],
            label="pilot controller launch attestation",
            expected_kind="arc_agi3_controller_launch",
            expected_status=None,
        )
    )
    guardian_path, guardian_sha, guardian = _control_reference(
        value["guardian_start_receipt"],
        label="pilot guardian start receipt",
        expected_kind="arc_agi3_controller_guardian_start",
        expected_status=None,
    )
    substrate_path, _substrate_sha, substrate = _control_reference(
        value["substrate_preflight_receipt"],
        label="pilot substrate preflight receipt",
        expected_kind="contiguous_substrate_preflight",
        expected_status="PASS",
    )
    ledger_path, _ledger_sha, ledger = _control_reference(
        value["host_child_ledger_audit"],
        label="pilot host-child ledger audit",
        expected_kind="arc_agi3_managed_host_child_ledger_audit",
        expected_status=None,
    )
    try:
        validated_conformance = Supervisor.Conformance.validate_result(
            conformance
        )
    except Exception as exc:
        raise PilotContractError(
            "pilot prelaunch conformance is not a current real PASS"
        ) from exc
    import gkm_legs

    control_files = validated_conformance.get(
        "control_contract_files_sha256"
    )
    status_counts = ledger.get("status_counts")
    state_probe = guardian.get("state_root_write_probe")
    backend_document = backend_launch.get("launch")
    if (
        conformance.get("kind")
        != "arc_agi3_contiguous_conformance"
        or conformance.get("status") != "PASS"
        or conformance.get("launch_authority") is not False
        or conformance.get("container_image_digest") is not None
        or conformance.get("control_contract_sha256")
        != expected_control_contract_sha256
        or runtime.get("kind")
        != "arc_agi3_python_runtime_manifest"
        or source_path.suffix != ".py"
        or not isinstance(control_files, Mapping)
        or control_files.get("arc/crack_lab/gkm_legs.py")
        != value.get("gkm_legs_source_sha256")
        or control_files.get(
            value.get("pilot_executor_control_path")
        )
        != source_reference["sha256"]
        or attempt_attestation.image.manifest_digest
        != expected_image_digest
        or conformance.get("suite_runtime_manifest_path")
        != str(runtime_path)
        or conformance.get("suite_runtime_manifest_sha256")
        != value["runtime_manifest"]["sha256"]
        or hashlib.sha256(
            gkm_legs.PERCEPTION_SEED.encode("utf-8")
        ).hexdigest()
        != value.get("perception_seed_sha256")
        or not isinstance(backend_document, Mapping)
        or backend_document.get("substrate_preflight_receipt_path")
        != str(substrate_path)
        or backend_document.get("substrate_preflight_receipt_sha256")
        != value["substrate_preflight_receipt"]["sha256"]
        or backend_document.get("controller_launch_receipt_path")
        != str(controller_path)
        or backend_document.get("controller_launch_receipt_sha256")
        != controller_sha
        or backend_document.get(
            "controller_guardian_start_receipt_path"
        )
        != str(guardian_path)
        or backend_document.get(
            "controller_guardian_start_receipt_sha256"
        )
        != guardian_sha
        or controller_launch.get("credentials_in_argv_or_env") is not False
        or controller_launch.get("bridge_or_arena_mounts") != 0
        or controller_launch.get(
            "egress_live_probe_before_controller_create"
        )
        is not True
        or not isinstance(state_probe, Mapping)
        or state_probe.get("kind")
        != "controller_state_root_write_probe"
        or state_probe.get("status") != "PASS"
        or state_probe.get("probe_absent_after_fsync") is not True
        or substrate.get("state_root_write_probe_status") != "PASS"
        or substrate.get("state_root") != controller_launch.get("state_root")
        or substrate.get("guardian_start_receipt_path")
        != str(guardian_path)
        or substrate.get("guardian_start_receipt_sha256")
        != guardian_sha
        or substrate.get("controller_launch_receipt_sha256")
        != controller_sha
        or any(
            substrate.get(name) is not False
            for name in (
                "proposer_container_started",
                "bridge_connected",
                "thread_started",
                "turn_started",
            )
        )
        or any(
            substrate.get(name) is not True
            for name in (
                "controller_inspect_absent",
                "controller_identity_query_empty",
                "controller_no_descendants",
                "egress_proxy_inspect_absent",
                "egress_proxy_identity_query_empty",
                "egress_proxy_no_descendants",
            )
        )
        or not isinstance(status_counts, Mapping)
        or status_counts.get("PENDING") != 0
        or status_counts.get("ACTIVE") != 0
        or ledger.get("all_receipts_authenticated") is not True
        or ledger.get("external_absence_proof_required_count") != 0
        or ledger.get("all_children_accounted_for") is not True
        or not conformance_path.is_absolute()
        or not runtime_path.is_absolute()
        or not backend_path.is_absolute()
        or not controller_path.is_absolute()
        or not ledger_path.is_absolute()
    ):
        raise PilotContractError(
            "pilot production stack referenced nonproduction evidence"
        )
    return {
        **value,
        "path": str(Path(path).resolve()),
        "file_sha256": expected_sha256,
    }


def _content_addressed_receipt(
    path: Path,
    *,
    label: str,
    expected_kind: str,
    expected_status: str,
    run_root: Path,
) -> tuple[dict[str, Any], str]:
    resolved = Path(path).resolve()
    if run_root != resolved and run_root not in resolved.parents:
        raise PilotContractError(f"{label} escaped its pilot run root")
    raw = _read_control(resolved, label=label)
    value = _strict_json(raw, label=label)
    selected = dict(value)
    observed = selected.pop("receipt_sha256", None)
    if (
        value.get("schema") != SCHEMA
        or value.get("kind") != expected_kind
        or value.get("status") != expected_status
        or SHA256_RE.fullmatch(str(observed)) is None
        or not hmac.compare_digest(
            str(observed),
            hashlib.sha256(_canonical_json(selected)).hexdigest(),
        )
    ):
        raise PilotContractError(
            f"{label} is not an exact content-addressed receipt"
        )
    return value, hashlib.sha256(raw).hexdigest()


def _verify_pilot_operator_terminal(
    reference: object,
    *,
    run_root: Path,
    pilot_sequence: int,
    game: str,
    target: int,
    image_digest: str,
    control_contract_sha256: str,
    production_stack_attestation_sha256: str,
) -> tuple[dict[str, Any], str]:
    path, reference_sha, _receipt = _evidence_reference(
        reference,
        label="pilot operator terminal",
    )
    value, file_sha = _content_addressed_receipt(
        path,
        label="pilot operator terminal",
        expected_kind="arc_agi3_contiguous_pilot_operator_terminal",
        expected_status="PASS",
        run_root=run_root,
    )
    expected_fields = {
        "schema",
        "kind",
        "status",
        "pilot_sequence",
        "game",
        "authoritative_target",
        "reached",
        "complete",
        "campaign_root",
        "image_digest",
        "control_contract_sha256",
        "scheduler_policy_sha256",
        "meta_protocol_sha256",
        "production_stack_attestation_sha256",
        "operator_configuration_sha256",
        "host_child_ledger_audit",
        "runner_state_audit",
        "scheduler_audit",
        "unified_audit",
        "terminal_retention_receipt",
        "active_primary_attempts",
        "active_auxiliary_assignments",
        "pending_scheduler_decision",
        "pending_auxiliary_decision",
        "pilot_only",
        "canonical_lineage_authority",
        "synthetic_evidence",
        "receipt_sha256",
    }
    if (
        file_sha != reference_sha
        or set(value) != expected_fields
        or value.get("pilot_sequence") != pilot_sequence
        or value.get("game") != game
        or value.get("authoritative_target") != target
        or value.get("reached") != target
        or value.get("complete") is not True
        or value.get("campaign_root") != str(run_root)
        or value.get("image_digest") != image_digest
        or value.get("control_contract_sha256")
        != control_contract_sha256
        or value.get("scheduler_policy_sha256")
        != Scheduler.SCHEDULER_POLICY_SHA256
        or value.get("meta_protocol_sha256")
        != Supervisor.POST_INCIDENT_META_PROTOCOL_SHA256
        or value.get("production_stack_attestation_sha256")
        != production_stack_attestation_sha256
        or SHA256_RE.fullmatch(
            str(value.get("operator_configuration_sha256"))
        )
        is None
        or value.get("active_primary_attempts") != []
        or value.get("active_auxiliary_assignments") != []
        or value.get("pending_scheduler_decision") is not False
        or value.get("pending_auxiliary_decision") is not False
        or value.get("pilot_only") is not True
        or value.get("canonical_lineage_authority") is not False
        or value.get("synthetic_evidence") is not False
    ):
        raise PilotContractError(
            "pilot operator terminal is not exact production authority"
        )
    for name in (
        "host_child_ledger_audit",
        "runner_state_audit",
        "scheduler_audit",
        "unified_audit",
        "terminal_retention_receipt",
    ):
        source_path, _source_sha, source = _evidence_reference(
            value[name],
            label=f"pilot terminal {name}",
        )
        if (
            run_root not in source_path.resolve().parents
            or source.get("status") != "PASS"
        ):
            raise PilotContractError(
                f"pilot terminal {name} is not bound PASS evidence"
            )
    return value, file_sha


def _verify_pilot_named_audit(
    reference: object,
    *,
    audit_name: str,
    run_root: Path,
    pilot_sequence: int,
    game: str,
    target: int,
    operator_terminal_file_sha256: str,
    production_stack_attestation_sha256: str,
) -> dict[str, Any]:
    path, reference_sha, _receipt = _evidence_reference(
        reference,
        label=f"pilot {audit_name} audit",
    )
    value, file_sha = _content_addressed_receipt(
        path,
        label=f"pilot {audit_name} audit",
        expected_kind=f"arc_agi3_contiguous_pilot_{audit_name}_audit",
        expected_status="PASS",
        run_root=run_root,
    )
    expected_fields = {
        "schema",
        "kind",
        "status",
        "audit_name",
        "pilot_sequence",
        "game",
        "authoritative_target",
        "verified_level_count",
        "campaign_root",
        "operator_terminal_file_sha256",
        "production_stack_attestation_sha256",
        "verifier_contract_sha256",
        "checks",
        "source_evidence",
        "source_evidence_sha256",
        "synthetic_evidence",
        "result_authority",
        "receipt_sha256",
    }
    checks = value.get("checks")
    sources = value.get("source_evidence")
    if (
        file_sha != reference_sha
        or set(value) != expected_fields
        or value.get("audit_name") != audit_name
        or value.get("pilot_sequence") != pilot_sequence
        or value.get("game") != game
        or value.get("authoritative_target") != target
        or value.get("verified_level_count") != target
        or value.get("campaign_root") != str(run_root)
        or value.get("operator_terminal_file_sha256")
        != operator_terminal_file_sha256
        or value.get("production_stack_attestation_sha256")
        != production_stack_attestation_sha256
        or value.get("verifier_contract_sha256")
        != hashlib.sha256(
            _canonical_json({
                "audit_name": audit_name,
                "required_checks":
                    list(PILOT_AUDIT_REQUIRED_CHECKS[audit_name]),
            })
        ).hexdigest()
        or not isinstance(checks, Mapping)
        or set(checks) != set(PILOT_AUDIT_REQUIRED_CHECKS[audit_name])
        or any(checks[name] is not True for name in checks)
        or not isinstance(sources, list)
        or not sources
        or value.get("source_evidence_sha256")
        != hashlib.sha256(_canonical_json(sources)).hexdigest()
        or value.get("synthetic_evidence") is not False
        or value.get("result_authority") != "pilot_gate_input_only"
    ):
        raise PilotContractError(
            f"pilot {audit_name} is not exact bound PASS evidence"
        )
    for index, source in enumerate(sources, start=1):
        if (
            not isinstance(source, Mapping)
            or set(source) != {"path", "sha256", "kind", "status"}
            or source.get("status") != "PASS"
            or not isinstance(source.get("kind"), str)
        ):
            raise PilotContractError(
                f"pilot {audit_name} source {index} is malformed"
            )
        source_path, _source_sha, source_value = _evidence_reference(
            {
                "path": source.get("path"),
                "sha256": source.get("sha256"),
                "status": source.get("status"),
            },
            label=f"pilot {audit_name} source {index}",
        )
        if (
            run_root not in source_path.resolve().parents
            or source_value.get("kind") != source.get("kind")
            or source_value.get("status") != "PASS"
        ):
            raise PilotContractError(
                f"pilot {audit_name} source {index} is not bound"
            )
    return value


def _validate_pilot_evidence(
    body: Mapping[str, object],
    *,
    key: bytes,
    run_root: Path,
    expected_sequence: int,
    expected_previous_sha256: str | None,
    expected_image_digest: str,
    expected_control_contract_sha256: str,
    expected_production_stack_attestation_path: Path,
    expected_production_stack_attestation_sha256: str,
) -> int:
    game, target = PILOT_MANIFEST[expected_sequence - 1]
    audits = body.get("audit_receipts")
    handoffs = body.get("meta_handoff_receipts")
    if (
        not isinstance(audits, Mapping)
        or set(audits) != set(PILOT_AUDITS)
        or not isinstance(handoffs, list)
        or len(handoffs) > 1
    ):
        raise PilotContractError(
            "pilot evidence inventories differ from the frozen contract"
        )
    genesis_path, _genesis_sha, genesis = _evidence_reference(
        body["empty_root_genesis_receipt"],
        label="pilot empty-root genesis",
    )
    if genesis_path.resolve() != run_root / "empty_root_genesis.json":
        raise PilotContractError(
            "pilot genesis is outside its canonical run root"
        )
    genesis = _authenticated_value(
        genesis, key=key, label="pilot empty-root genesis"
    )
    genesis_fields = {
        "schema",
        "kind",
        "status",
        "pilot_sequence",
        "game",
        "authoritative_target",
        "pilot_manifest_sha256",
        "previous_run_receipt_sha256",
        "roots",
        "image_digest",
        "control_contract_sha256",
        "production_stack_attestation_path",
        "production_stack_attestation_sha256",
        "canonical_lineage_authority",
    }
    root_names = (
        "artifact_root",
        "wip_root",
        "controller_state_root",
        "scheduler_root",
    )
    roots = genesis.get("roots")
    if (
        set(genesis) != genesis_fields
        or genesis.get("schema") != SCHEMA
        or genesis.get("kind")
        != "arc_agi3_contiguous_pilot_empty_root_genesis"
        or genesis.get("status") != "PASS"
        or genesis.get("pilot_sequence") != expected_sequence
        or genesis.get("game") != game
        or genesis.get("authoritative_target") != target
        or genesis.get("pilot_manifest_sha256")
        != PILOT_MANIFEST_SHA256
        or genesis.get("previous_run_receipt_sha256")
        != expected_previous_sha256
        or genesis.get("image_digest") != expected_image_digest
        or genesis.get("control_contract_sha256")
        != expected_control_contract_sha256
        or genesis.get("production_stack_attestation_path")
        != str(expected_production_stack_attestation_path)
        or genesis.get("production_stack_attestation_sha256")
        != expected_production_stack_attestation_sha256
        or genesis.get("canonical_lineage_authority") is not False
        or not isinstance(roots, Mapping)
        or set(roots) != set(root_names)
        or any(
            not isinstance(roots[name], Mapping)
            or set(roots[name])
            != {
                "path",
                "initial_entry_count",
                "initial_entries_sha256",
            }
            or not isinstance(roots[name].get("path"), str)
            or not Path(str(roots[name]["path"])).is_absolute()
            or roots[name].get("initial_entry_count") != 0
            or roots[name].get("initial_entries_sha256")
            != EMPTY_ENTRIES_SHA256
            for name in root_names
        )
        or len({
            str(roots[name]["path"]) for name in root_names
        }) != len(root_names)
    ):
        raise PilotContractError(
            "pilot genesis does not prove four distinct empty roots"
        )
    resolved_roots = [
        Path(str(roots[name]["path"])).resolve()
        for name in root_names
    ]
    if (
        any(
            left == right
            or left in right.parents
            or right in left.parents
            for index, left in enumerate(resolved_roots)
            for right in resolved_roots[index + 1:]
        )
        or any(
            Path(str(roots[name]["path"])).resolve()
            != run_root / name
            for name in root_names
        )
    ):
        raise PilotContractError(
            "pilot empty roots overlap or contain one another"
        )
    terminal, terminal_file_sha256 = _verify_pilot_operator_terminal(
        body["operator_terminal_receipt"],
        run_root=run_root,
        pilot_sequence=expected_sequence,
        game=game,
        target=target,
        image_digest=expected_image_digest,
        control_contract_sha256=expected_control_contract_sha256,
        production_stack_attestation_sha256=(
            expected_production_stack_attestation_sha256
        ),
    )
    for audit_name in PILOT_AUDITS:
        _verify_pilot_named_audit(
            audits[audit_name],
            audit_name=audit_name,
            run_root=run_root,
            pilot_sequence=expected_sequence,
            game=game,
            target=target,
            operator_terminal_file_sha256=terminal_file_sha256,
            production_stack_attestation_sha256=(
                expected_production_stack_attestation_sha256
            ),
        )
    for handoff in handoffs:
        handoff_path, _sha, receipt = _evidence_reference(
            handoff,
            label="pilot meta-proposer handoff",
            expected_statuses=frozenset({"DIAGNOSED"}),
        )
        if run_root not in handoff_path.resolve().parents:
            raise PilotContractError(
                "pilot meta handoff escaped its run root"
            )
        try:
            verified_handoff = (
                Supervisor.verify_post_incident_meta_terminal_receipt(
                    handoff_path,
                    expected_campaign_root=run_root,
                    expected_operator_configuration_sha256=str(
                        terminal["operator_configuration_sha256"]
                    ),
                )
            )
        except Exception as exc:
            raise PilotContractError(
                "pilot meta handoff is not the real quarantine-only path"
            ) from exc
        if (
            verified_handoff.get("status") != "DIAGNOSED"
            or verified_handoff.get("recommended_operator_action")
            != "REMATERIALIZE_AND_REPROBE_CONTROLLER_SUBSTRATE"
        ):
            raise PilotContractError(
                "pilot meta handoff did not exercise safe recovery"
            )
    return len(handoffs)


def verify_production_pilot_outcome(
    outcome_path: Path,
    *,
    authentication_key_path: Path,
    expected_sequence: int,
    expected_previous_sha256: str | None,
    expected_image_digest: str,
    expected_control_contract_sha256: str,
    expected_production_stack_attestation_path: Path,
    expected_production_stack_attestation_sha256: str,
) -> dict[str, Any]:
    """Verify untrusted executor output before the host authenticates a run."""

    if expected_sequence not in {1, 2}:
        raise PilotContractError("pilot sequence is outside the manifest")
    path = Path(outcome_path).resolve()
    game, target = PILOT_MANIFEST[expected_sequence - 1]
    run_root = path.parent
    if (
        path != run_root / PRODUCTION_OUTCOME_NAME
        or run_root.name != f"{expected_sequence:02d}-{game}"
    ):
        raise PilotContractError(
            "pilot production outcome is outside its canonical run root"
        )
    body, file_sha256 = _content_addressed_receipt(
        path,
        label="pilot production outcome",
        expected_kind="arc_agi3_contiguous_production_pilot_outcome",
        expected_status="PASS",
        run_root=run_root,
    )
    expected_fields = {
        "schema",
        "kind",
        "status",
        "pilot_sequence",
        "game",
        "authoritative_target",
        "reached",
        "pilot_manifest_sha256",
        "previous_run_receipt_sha256",
        "image_digest",
        "control_contract_sha256",
        "scheduler_policy_sha256",
        "meta_protocol_sha256",
        "production_stack_attestation_path",
        "production_stack_attestation_sha256",
        "empty_root_genesis_receipt",
        "operator_terminal_receipt",
        "audit_receipts",
        "clean_continuation_restart_count",
        "meta_handoff_receipts",
        "pilot_only",
        "canonical_lineage_authority",
        "synthetic_evidence",
        "result_authority",
        "receipt_sha256",
    }
    if (
        set(body) != expected_fields
        or body.get("pilot_sequence") != expected_sequence
        or body.get("game") != game
        or body.get("authoritative_target") != target
        or body.get("reached") != target
        or body.get("pilot_manifest_sha256")
        != PILOT_MANIFEST_SHA256
        or body.get("previous_run_receipt_sha256")
        != expected_previous_sha256
        or body.get("image_digest") != expected_image_digest
        or body.get("control_contract_sha256")
        != expected_control_contract_sha256
        or body.get("scheduler_policy_sha256")
        != Scheduler.SCHEDULER_POLICY_SHA256
        or body.get("meta_protocol_sha256")
        != Supervisor.POST_INCIDENT_META_PROTOCOL_SHA256
        or body.get("production_stack_attestation_path")
        != str(expected_production_stack_attestation_path)
        or body.get("production_stack_attestation_sha256")
        != expected_production_stack_attestation_sha256
        or isinstance(
            body.get("clean_continuation_restart_count"), bool
        )
        or not isinstance(
            body.get("clean_continuation_restart_count"), int
        )
        or int(body.get("clean_continuation_restart_count", 0)) < 1
        or body.get("pilot_only") is not True
        or body.get("canonical_lineage_authority") is not False
        or body.get("synthetic_evidence") is not False
        or body.get("result_authority")
        != "quarantine_pending_host_admission"
    ):
        raise PilotContractError(
            "pilot production outcome differs from the frozen contract"
        )
    verify_production_stack_attestation(
        expected_production_stack_attestation_path,
        expected_sha256=(
            expected_production_stack_attestation_sha256
        ),
        expected_image_digest=expected_image_digest,
        expected_control_contract_sha256=(
            expected_control_contract_sha256
        ),
    )
    meta_count = _validate_pilot_evidence(
        body,
        key=_read_key(authentication_key_path),
        run_root=run_root,
        expected_sequence=expected_sequence,
        expected_previous_sha256=expected_previous_sha256,
        expected_image_digest=expected_image_digest,
        expected_control_contract_sha256=(
            expected_control_contract_sha256
        ),
        expected_production_stack_attestation_path=(
            expected_production_stack_attestation_path
        ),
        expected_production_stack_attestation_sha256=(
            expected_production_stack_attestation_sha256
        ),
    )
    return {
        **body,
        "path": str(path),
        "file_sha256": file_sha256,
        "meta_handoff_count": meta_count,
    }


def verify_pilot_run_receipt(
    receipt_path: Path,
    *,
    authentication_key_path: Path,
    expected_sequence: int,
    expected_previous_sha256: str | None,
    expected_image_digest: str,
    expected_control_contract_sha256: str,
    expected_production_stack_attestation_path: Path,
    expected_production_stack_attestation_sha256: str,
) -> dict[str, Any]:
    """Reopen one exact full-game pilot and every named PASS artifact."""

    if expected_sequence not in {1, 2}:
        raise PilotContractError("pilot sequence is outside the manifest")
    key = _read_key(authentication_key_path)
    raw = _read_control(receipt_path, label="pilot run receipt")
    value = _strict_json(raw, label="pilot run receipt")
    body = _authenticated_value(
        value, key=key, label="pilot run receipt"
    )
    game, target = PILOT_MANIFEST[expected_sequence - 1]
    run_receipt_path = Path(receipt_path).resolve()
    run_root = run_receipt_path.parent
    if (
        run_receipt_path != run_root / PILOT_RUN_NAME
        or run_root.name != f"{expected_sequence:02d}-{game}"
    ):
        raise PilotContractError(
            "pilot run receipt is outside its canonical run root"
        )
    expected_fields = {
        "schema",
        "kind",
        "status",
        "pilot_sequence",
        "game",
        "authoritative_target",
        "reached",
        "pilot_manifest_sha256",
        "previous_run_receipt_sha256",
        "image_digest",
        "control_contract_sha256",
        "scheduler_policy_sha256",
        "meta_protocol_sha256",
        "production_stack_attestation_path",
        "production_stack_attestation_sha256",
        "production_outcome_receipt",
        "empty_root_genesis_receipt",
        "operator_terminal_receipt",
        "audit_receipts",
        "clean_continuation_restart_count",
        "meta_handoff_receipts",
        "pilot_only",
        "canonical_lineage_authority",
        "synthetic_evidence",
        "host_admission_authority",
    }
    audits = body.get("audit_receipts")
    handoffs = body.get("meta_handoff_receipts")
    if (
        set(body) != expected_fields
        or body.get("schema") != SCHEMA
        or body.get("kind")
        != "arc_agi3_contiguous_pilot_run"
        or body.get("status") != "PASS"
        or body.get("pilot_sequence") != expected_sequence
        or body.get("game") != game
        or body.get("authoritative_target") != target
        or body.get("reached") != target
        or body.get("pilot_manifest_sha256")
        != PILOT_MANIFEST_SHA256
        or body.get("previous_run_receipt_sha256")
        != expected_previous_sha256
        or body.get("image_digest") != expected_image_digest
        or body.get("control_contract_sha256")
        != expected_control_contract_sha256
        or body.get("scheduler_policy_sha256")
        != Scheduler.SCHEDULER_POLICY_SHA256
        or body.get("meta_protocol_sha256")
        != Supervisor.POST_INCIDENT_META_PROTOCOL_SHA256
        or body.get("production_stack_attestation_path")
        != str(expected_production_stack_attestation_path)
        or body.get("production_stack_attestation_sha256")
        != expected_production_stack_attestation_sha256
        or body.get("pilot_only") is not True
        or body.get("canonical_lineage_authority") is not False
        or body.get("synthetic_evidence") is not False
        or body.get("host_admission_authority")
        != "authenticated_pilot_controller"
        or isinstance(
            body.get("clean_continuation_restart_count"), bool
        )
        or not isinstance(
            body.get("clean_continuation_restart_count"), int
        )
        or int(body.get("clean_continuation_restart_count", 0)) < 1
        or not isinstance(audits, Mapping)
        or set(audits) != set(PILOT_AUDITS)
        or not isinstance(handoffs, list)
        or len(handoffs) > 1
    ):
        raise PilotContractError(
            "pilot run differs from the frozen full-game contract"
        )
    verify_production_stack_attestation(
        expected_production_stack_attestation_path,
        expected_sha256=(
            expected_production_stack_attestation_sha256
        ),
        expected_image_digest=expected_image_digest,
        expected_control_contract_sha256=(
            expected_control_contract_sha256
        ),
    )
    outcome_path, outcome_file_sha256, _outcome_receipt = (
        _evidence_reference(
            body["production_outcome_receipt"],
            label="pilot production outcome",
        )
    )
    if outcome_path.resolve() != run_root / PRODUCTION_OUTCOME_NAME:
        raise PilotContractError(
            "pilot run references a noncanonical production outcome"
        )
    outcome = verify_production_pilot_outcome(
        outcome_path,
        authentication_key_path=authentication_key_path,
        expected_sequence=expected_sequence,
        expected_previous_sha256=expected_previous_sha256,
        expected_image_digest=expected_image_digest,
        expected_control_contract_sha256=(
            expected_control_contract_sha256
        ),
        expected_production_stack_attestation_path=(
            expected_production_stack_attestation_path
        ),
        expected_production_stack_attestation_sha256=(
            expected_production_stack_attestation_sha256
        ),
    )
    if outcome["file_sha256"] != outcome_file_sha256:
        raise PilotContractError(
            "pilot production outcome file binding changed"
        )
    evidence_fields = {
        "empty_root_genesis_receipt",
        "operator_terminal_receipt",
        "audit_receipts",
        "clean_continuation_restart_count",
        "meta_handoff_receipts",
    }
    if any(body[name] != outcome[name] for name in evidence_fields):
        raise PilotContractError(
            "authenticated pilot run changed production outcome evidence"
        )
    meta_handoff_count = _validate_pilot_evidence(
        body,
        key=key,
        run_root=run_root,
        expected_sequence=expected_sequence,
        expected_previous_sha256=expected_previous_sha256,
        expected_image_digest=expected_image_digest,
        expected_control_contract_sha256=(
            expected_control_contract_sha256
        ),
        expected_production_stack_attestation_path=(
            expected_production_stack_attestation_path
        ),
        expected_production_stack_attestation_sha256=(
            expected_production_stack_attestation_sha256
        ),
    )
    return {
        **value,
        "path": str(Path(receipt_path).resolve()),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "meta_handoff_count": meta_handoff_count,
    }


def issue_pilot_gate_receipt(
    *,
    run_receipt_paths: Sequence[Path],
    authentication_key_path: Path,
    output_path: Path,
    image_digest: str,
    control_contract_sha256: str,
    production_stack_attestation_path: Path,
    production_stack_attestation_sha256: str,
) -> dict[str, object]:
    """Issue the only full-launch pilot gate from exact ordered PASS runs."""

    if (
        len(run_receipt_paths) != 2
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}", image_digest
        )
        is None
        or SHA256_RE.fullmatch(control_contract_sha256) is None
        or SHA256_RE.fullmatch(
            production_stack_attestation_sha256
        )
        is None
    ):
        raise PilotContractError(
            "pilot gate inputs are malformed"
        )
    key = _read_key(authentication_key_path)
    verified: list[dict[str, Any]] = []
    predecessor: str | None = None
    for sequence, path in enumerate(run_receipt_paths, start=1):
        run = verify_pilot_run_receipt(
            path,
            authentication_key_path=authentication_key_path,
            expected_sequence=sequence,
            expected_previous_sha256=predecessor,
            expected_image_digest=image_digest,
            expected_control_contract_sha256=(
                control_contract_sha256
            ),
            expected_production_stack_attestation_path=(
                production_stack_attestation_path
            ),
            expected_production_stack_attestation_sha256=(
                production_stack_attestation_sha256
            ),
        )
        verified.append(run)
        predecessor = str(run["receipt_sha256"])
    meta_handoff_count = sum(
        int(item["meta_handoff_count"]) for item in verified
    )
    if meta_handoff_count < 1:
        raise PilotContractError(
            "pilot gate requires one real production meta handoff"
        )
    body = {
        "schema": SCHEMA,
        "kind": "arc_agi3_contiguous_pilot_gate",
        "status": "PASS",
        "full_campaign_launch_gate": "UNLOCKED",
        "pilot_manifest_sha256": PILOT_MANIFEST_SHA256,
        "pilot_games": [game for game, _target in PILOT_MANIFEST],
        "pilot_targets": [target for _game, target in PILOT_MANIFEST],
        "run_receipts": [
            {
                "pilot_sequence": index,
                "game": PILOT_MANIFEST[index - 1][0],
                "path": item["path"],
                "file_sha256": item["file_sha256"],
                "receipt_sha256": item["receipt_sha256"],
            }
            for index, item in enumerate(verified, start=1)
        ],
        "meta_handoff_count": meta_handoff_count,
        "image_digest": image_digest,
        "control_contract_sha256": control_contract_sha256,
        "scheduler_policy_sha256":
            Scheduler.SCHEDULER_POLICY_SHA256,
        "meta_protocol_sha256":
            Supervisor.POST_INCIDENT_META_PROTOCOL_SHA256,
        "production_stack_attestation_path":
            str(production_stack_attestation_path),
        "production_stack_attestation_sha256":
            production_stack_attestation_sha256,
        "pilot_lineage_canonical": False,
    }
    value, _file_sha256 = _write_authenticated(
        output_path,
        body,
        key=key,
        label="pilot gate receipt",
    )
    return value


def verify_pilot_gate_receipt(
    receipt_path: Path,
    *,
    authentication_key_path: Path,
    expected_image_digest: str,
    expected_control_contract_sha256: str,
    expected_production_stack_attestation_sha256: str,
) -> dict[str, Any]:
    """Reopen the gate and both runs; no cached PASS is launch authority."""

    key = _read_key(authentication_key_path)
    raw = _read_control(receipt_path, label="pilot gate receipt")
    value = _strict_json(raw, label="pilot gate receipt")
    body = _authenticated_value(
        value, key=key, label="pilot gate receipt"
    )
    expected_fields = {
        "schema",
        "kind",
        "status",
        "full_campaign_launch_gate",
        "pilot_manifest_sha256",
        "pilot_games",
        "pilot_targets",
        "run_receipts",
        "meta_handoff_count",
        "image_digest",
        "control_contract_sha256",
        "scheduler_policy_sha256",
        "meta_protocol_sha256",
        "production_stack_attestation_path",
        "production_stack_attestation_sha256",
        "pilot_lineage_canonical",
    }
    runs = body.get("run_receipts")
    if (
        set(body) != expected_fields
        or body.get("schema") != SCHEMA
        or body.get("kind") != "arc_agi3_contiguous_pilot_gate"
        or body.get("status") != "PASS"
        or body.get("full_campaign_launch_gate") != "UNLOCKED"
        or body.get("pilot_manifest_sha256")
        != PILOT_MANIFEST_SHA256
        or body.get("pilot_games") != ["ft09", "lp85"]
        or body.get("pilot_targets") != [6, 8]
        or body.get("image_digest") != expected_image_digest
        or body.get("control_contract_sha256")
        != expected_control_contract_sha256
        or body.get("scheduler_policy_sha256")
        != Scheduler.SCHEDULER_POLICY_SHA256
        or body.get("meta_protocol_sha256")
        != Supervisor.POST_INCIDENT_META_PROTOCOL_SHA256
        or not isinstance(
            body.get("production_stack_attestation_path"), str
        )
        or not Path(
            str(body.get("production_stack_attestation_path"))
        ).is_absolute()
        or body.get("production_stack_attestation_sha256")
        != expected_production_stack_attestation_sha256
        or body.get("pilot_lineage_canonical") is not False
        or not isinstance(runs, list)
        or len(runs) != 2
    ):
        raise PilotContractError(
            "pilot gate differs from the frozen launch contract"
        )
    attestation_path = Path(
        str(body["production_stack_attestation_path"])
    )
    verify_production_stack_attestation(
        attestation_path,
        expected_sha256=(
            expected_production_stack_attestation_sha256
        ),
        expected_image_digest=expected_image_digest,
        expected_control_contract_sha256=(
            expected_control_contract_sha256
        ),
    )
    predecessor: str | None = None
    meta_handoff_count = 0
    for sequence, reference in enumerate(runs, start=1):
        game = PILOT_MANIFEST[sequence - 1][0]
        if (
            not isinstance(reference, Mapping)
            or set(reference)
            != {
                "pilot_sequence",
                "game",
                "path",
                "file_sha256",
                "receipt_sha256",
            }
            or reference.get("pilot_sequence") != sequence
            or reference.get("game") != game
            or not isinstance(reference.get("path"), str)
            or not Path(str(reference["path"])).is_absolute()
            or SHA256_RE.fullmatch(
                str(reference.get("file_sha256"))
            )
            is None
            or SHA256_RE.fullmatch(
                str(reference.get("receipt_sha256"))
            )
            is None
        ):
            raise PilotContractError(
                "pilot gate run reference is malformed"
            )
        run_path = Path(str(reference["path"]))
        run_raw = _read_control(
            run_path, label="pilot gate run receipt"
        )
        if hashlib.sha256(run_raw).hexdigest() != reference[
            "file_sha256"
        ]:
            raise PilotContractError(
                "pilot gate run file changed"
            )
        verified = verify_pilot_run_receipt(
            run_path,
            authentication_key_path=authentication_key_path,
            expected_sequence=sequence,
            expected_previous_sha256=predecessor,
            expected_image_digest=expected_image_digest,
            expected_control_contract_sha256=(
                expected_control_contract_sha256
            ),
            expected_production_stack_attestation_path=(
                attestation_path
            ),
            expected_production_stack_attestation_sha256=(
                expected_production_stack_attestation_sha256
            ),
        )
        if verified["receipt_sha256"] != reference["receipt_sha256"]:
            raise PilotContractError(
                "pilot gate run content address changed"
            )
        predecessor = str(verified["receipt_sha256"])
        meta_handoff_count += int(verified["meta_handoff_count"])
    if (
        meta_handoff_count < 1
        or body.get("meta_handoff_count") != meta_handoff_count
    ):
        raise PilotContractError(
            "pilot gate lacks its real meta handoff"
        )
    return {
        **value,
        "path": str(Path(receipt_path).resolve()),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
    }


def initialize_empty_pilot_execution(
    *,
    base_root: Path,
    pilot_sequence: int,
    authentication_key_path: Path,
    previous_run_receipt_sha256: str | None,
    image_digest: str,
    control_contract_sha256: str,
    production_stack_attestation_path: Path,
    production_stack_attestation_sha256: str,
) -> PilotExecution:
    """Create and seal the four empty roots before external pilot work."""

    if pilot_sequence not in {1, 2}:
        raise PilotContractError("pilot sequence is outside the manifest")
    key = _read_key(authentication_key_path)
    root = Path(base_root).resolve()
    Supervisor._operator_lease_private_directory(
        root, create=True, label="pilot base root"
    )
    game, target = PILOT_MANIFEST[pilot_sequence - 1]
    run_root = root / f"{pilot_sequence:02d}-{game}"
    Supervisor._operator_lease_private_directory(
        run_root, create=True, label="pilot run root"
    )
    roots = {
        name: run_root / name
        for name in (
            "artifact_root",
            "wip_root",
            "controller_state_root",
            "scheduler_root",
        )
    }
    for name, selected in roots.items():
        Supervisor._operator_lease_private_directory(
            selected, create=True, label=f"pilot {name}"
        )
    genesis_path = run_root / "empty_root_genesis.json"
    if genesis_path.exists() or genesis_path.is_symlink():
        raw = _read_control(
            genesis_path, label="pilot empty-root genesis"
        )
        retained = _strict_json(
            raw, label="pilot empty-root genesis"
        )
        retained_body = _authenticated_value(
            retained,
            key=key,
            label="pilot empty-root genesis",
        )
        expected_roots = {
            name: {
                "path": str(selected),
                "initial_entry_count": 0,
                "initial_entries_sha256": EMPTY_ENTRIES_SHA256,
            }
            for name, selected in roots.items()
        }
        expected_body = {
            "schema": SCHEMA,
            "kind":
                "arc_agi3_contiguous_pilot_empty_root_genesis",
            "status": "PASS",
            "pilot_sequence": pilot_sequence,
            "game": game,
            "authoritative_target": target,
            "pilot_manifest_sha256": PILOT_MANIFEST_SHA256,
            "previous_run_receipt_sha256":
                previous_run_receipt_sha256,
            "roots": expected_roots,
            "image_digest": image_digest,
            "control_contract_sha256": control_contract_sha256,
            "production_stack_attestation_path":
                str(production_stack_attestation_path),
            "production_stack_attestation_sha256":
                production_stack_attestation_sha256,
            "canonical_lineage_authority": False,
        }
        if retained_body != expected_body:
            raise PilotContractError(
                "recovered pilot genesis changed its frozen inputs"
            )
        genesis_file_sha256 = hashlib.sha256(raw).hexdigest()
    else:
        allowed = set(roots)
        if {
            path.name for path in run_root.iterdir()
        } != allowed:
            raise PilotContractError(
                "new pilot run root contains pre-genesis state"
            )
        for name, selected in roots.items():
            if list(selected.iterdir()):
                raise PilotContractError(
                    f"pilot {name} is not empty at genesis"
                )
        genesis_body = {
            "schema": SCHEMA,
            "kind":
                "arc_agi3_contiguous_pilot_empty_root_genesis",
            "status": "PASS",
            "pilot_sequence": pilot_sequence,
            "game": game,
            "authoritative_target": target,
            "pilot_manifest_sha256": PILOT_MANIFEST_SHA256,
            "previous_run_receipt_sha256":
                previous_run_receipt_sha256,
            "roots": {
                name: {
                    "path": str(selected),
                    "initial_entry_count": 0,
                    "initial_entries_sha256":
                        EMPTY_ENTRIES_SHA256,
                }
                for name, selected in roots.items()
            },
            "image_digest": image_digest,
            "control_contract_sha256": control_contract_sha256,
            "production_stack_attestation_path":
                str(production_stack_attestation_path),
            "production_stack_attestation_sha256":
                production_stack_attestation_sha256,
            "canonical_lineage_authority": False,
        }
        _genesis, genesis_file_sha256 = _write_authenticated(
            genesis_path,
            genesis_body,
            key=key,
            label="pilot empty-root genesis",
        )
    return PilotExecution(
        schema=SCHEMA,
        pilot_sequence=pilot_sequence,
        game=game,
        authoritative_target=target,
        run_root=run_root,
        artifact_root=roots["artifact_root"],
        wip_root=roots["wip_root"],
        controller_state_root=roots["controller_state_root"],
        scheduler_root=roots["scheduler_root"],
        empty_root_genesis_receipt_path=genesis_path,
        empty_root_genesis_receipt_sha256=genesis_file_sha256,
        previous_run_receipt_sha256=previous_run_receipt_sha256,
        pilot_manifest_sha256=PILOT_MANIFEST_SHA256,
        image_digest=image_digest,
        control_contract_sha256=control_contract_sha256,
        production_stack_attestation_path=(
            production_stack_attestation_path
        ),
        production_stack_attestation_sha256=(
            production_stack_attestation_sha256
        ),
    )


def execute_frozen_pilots(
    *,
    base_root: Path,
    executor: ProductionPilotExecutor,
    authentication_key_path: Path,
    gate_receipt_path: Path,
    image_digest: str,
    control_contract_sha256: str,
    production_stack_attestation_path: Path,
    production_stack_attestation_sha256: str,
) -> dict[str, object]:
    """Production-reachable top-level pilot sequence; no game is selectable."""

    attestation_path = getattr(
        executor, "production_stack_attestation_path", None
    )
    declared_attestation_path = Path(
        production_stack_attestation_path
    ).resolve()
    if (
        not isinstance(attestation_path, Path)
        or attestation_path.resolve() != declared_attestation_path
    ):
        raise PilotContractError(
            "pilot executor lacks its sealed production stack evidence"
        )
    attestation_path = declared_attestation_path
    verify_production_stack_attestation(
        attestation_path,
        expected_sha256=production_stack_attestation_sha256,
        expected_image_digest=image_digest,
        expected_control_contract_sha256=(
            control_contract_sha256
        ),
    )
    if Path(gate_receipt_path).exists() or Path(
        gate_receipt_path
    ).is_symlink():
        return verify_pilot_gate_receipt(
            gate_receipt_path,
            authentication_key_path=authentication_key_path,
            expected_image_digest=image_digest,
            expected_control_contract_sha256=(
                control_contract_sha256
            ),
            expected_production_stack_attestation_sha256=(
                production_stack_attestation_sha256
            ),
        )
    run_receipts: list[Path] = []
    predecessor: str | None = None
    for pilot_sequence in (1, 2):
        execution = initialize_empty_pilot_execution(
            base_root=base_root,
            pilot_sequence=pilot_sequence,
            authentication_key_path=authentication_key_path,
            previous_run_receipt_sha256=predecessor,
            image_digest=image_digest,
            control_contract_sha256=control_contract_sha256,
            production_stack_attestation_path=attestation_path,
            production_stack_attestation_sha256=(
                production_stack_attestation_sha256
            ),
        )
        expected_receipt_path = execution.run_root / PILOT_RUN_NAME
        expected_outcome_path = (
            execution.run_root / PRODUCTION_OUTCOME_NAME
        )
        if (
            expected_receipt_path.exists()
            or expected_receipt_path.is_symlink()
        ):
            receipt_path = expected_receipt_path
        else:
            if (
                expected_outcome_path.exists()
                or expected_outcome_path.is_symlink()
            ):
                outcome_path = expected_outcome_path
            else:
                outcome_path = Path(executor.execute_game(execution))
            if outcome_path.resolve() != expected_outcome_path:
                raise PilotContractError(
                    "pilot executor returned a noncanonical production outcome"
                )
            outcome = verify_production_pilot_outcome(
                outcome_path,
                authentication_key_path=authentication_key_path,
                expected_sequence=pilot_sequence,
                expected_previous_sha256=predecessor,
                expected_image_digest=image_digest,
                expected_control_contract_sha256=(
                    control_contract_sha256
                ),
                expected_production_stack_attestation_path=(
                    attestation_path
                ),
                expected_production_stack_attestation_sha256=(
                    production_stack_attestation_sha256
                ),
            )
            run_body = {
                "schema": SCHEMA,
                "kind": "arc_agi3_contiguous_pilot_run",
                "status": "PASS",
                "pilot_sequence": pilot_sequence,
                "game": execution.game,
                "authoritative_target":
                    execution.authoritative_target,
                "reached": execution.authoritative_target,
                "pilot_manifest_sha256": PILOT_MANIFEST_SHA256,
                "previous_run_receipt_sha256": predecessor,
                "image_digest": image_digest,
                "control_contract_sha256":
                    control_contract_sha256,
                "scheduler_policy_sha256":
                    Scheduler.SCHEDULER_POLICY_SHA256,
                "meta_protocol_sha256":
                    Supervisor.POST_INCIDENT_META_PROTOCOL_SHA256,
                "production_stack_attestation_path":
                    str(attestation_path),
                "production_stack_attestation_sha256":
                    production_stack_attestation_sha256,
                "production_outcome_receipt": {
                    "path": outcome["path"],
                    "sha256": outcome["file_sha256"],
                    "status": "PASS",
                },
                "empty_root_genesis_receipt":
                    outcome["empty_root_genesis_receipt"],
                "operator_terminal_receipt":
                    outcome["operator_terminal_receipt"],
                "audit_receipts": outcome["audit_receipts"],
                "clean_continuation_restart_count":
                    outcome["clean_continuation_restart_count"],
                "meta_handoff_receipts":
                    outcome["meta_handoff_receipts"],
                "pilot_only": True,
                "canonical_lineage_authority": False,
                "synthetic_evidence": False,
                "host_admission_authority":
                    "authenticated_pilot_controller",
            }
            _write_authenticated(
                expected_receipt_path,
                run_body,
                key=_read_key(authentication_key_path),
                label="pilot run receipt",
            )
            receipt_path = expected_receipt_path
        verified = verify_pilot_run_receipt(
            receipt_path,
            authentication_key_path=authentication_key_path,
            expected_sequence=pilot_sequence,
            expected_previous_sha256=predecessor,
            expected_image_digest=image_digest,
            expected_control_contract_sha256=(
                control_contract_sha256
            ),
            expected_production_stack_attestation_path=(
                attestation_path
            ),
            expected_production_stack_attestation_sha256=(
                production_stack_attestation_sha256
            ),
        )
        run_receipts.append(receipt_path)
        predecessor = str(verified["receipt_sha256"])
    issued = issue_pilot_gate_receipt(
        run_receipt_paths=run_receipts,
        authentication_key_path=authentication_key_path,
        output_path=gate_receipt_path,
        image_digest=image_digest,
        control_contract_sha256=control_contract_sha256,
        production_stack_attestation_path=attestation_path,
        production_stack_attestation_sha256=(
            production_stack_attestation_sha256
        ),
    )
    return verify_pilot_gate_receipt(
        gate_receipt_path,
        authentication_key_path=authentication_key_path,
        expected_image_digest=image_digest,
        expected_control_contract_sha256=(
            control_contract_sha256
        ),
        expected_production_stack_attestation_sha256=(
            production_stack_attestation_sha256
        ),
    )


__all__ = [
    "PILOT_AUDIT_REQUIRED_CHECKS",
    "PILOT_MANIFEST",
    "PILOT_MANIFEST_SHA256",
    "PRODUCTION_OUTCOME_NAME",
    "PilotContractError",
    "PilotExecution",
    "ProductionPilotExecutor",
    "execute_frozen_pilots",
    "initialize_empty_pilot_execution",
    "issue_pilot_gate_receipt",
    "verify_pilot_gate_receipt",
    "verify_production_pilot_outcome",
    "verify_production_stack_attestation",
    "verify_pilot_run_receipt",
]
