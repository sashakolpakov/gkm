from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import arc_agi3_contiguous_pilot as P
import arc_agi3_contiguous_scheduler as Scheduler
import arc_agi3_contiguous_supervisor as Supervisor
import arc_agi3_container_backend as Container
import gkm_legs


def _plain_receipt(path: Path, *, kind: str, status: str) -> dict:
    value = {"schema": 1, "kind": kind, "status": status}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(P._canonical_json(value) + b"\n")
    path.chmod(0o400)
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "status": status,
    }


def _self_receipt(path: Path, body: dict) -> dict:
    value = {
        **body,
        "receipt_sha256": hashlib.sha256(
            P._canonical_json(body)
        ).hexdigest(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(P._canonical_json(value) + b"\n")
    path.chmod(0o400)
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "status": str(body["status"]),
    }


def _control(path: Path, value: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(P._canonical_json(value) + b"\n")
    path.chmod(0o400)
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "status": "CONTROL",
    }


class _MetaDriver:
    def run_attached_stream(
        self,
        argv,
        *,
        timeout_seconds,
        stdout_path,
        stderr_path,
        stdout_limit_bytes,
        stderr_limit_bytes,
    ):
        del timeout_seconds, stdout_limit_bytes, stderr_limit_bytes
        request_path = Path(tuple(argv)[4])
        response_path = Path(tuple(argv)[6])
        request_raw = request_path.read_bytes()
        response = {
            "schema": Supervisor.POST_INCIDENT_META_SCHEMA,
            "kind":
                "arc_agi3_contiguous_post_incident_meta_response",
            "protocol_sha256":
                Supervisor.POST_INCIDENT_META_PROTOCOL_SHA256,
            "request_sha256":
                hashlib.sha256(request_raw).hexdigest(),
            "status": "DIAGNOSED",
            "diagnosis_code": "isolated_state_root_unwritable",
            "diagnosis_summary":
                "The isolated controller state root must be rematerialized.",
            "socratic_challenge":
                "Could a read-only inherited state root explain the failure?",
            "recommended_operator_action":
                "REMATERIALIZE_AND_REPROBE_CONTROLLER_SUBSTRATE",
            "scheduler_authority": False,
            "solver_authority": False,
            "wip_authority": False,
            "cost_authority": False,
            "retry_authority": False,
            "dispatch_authority": False,
            "promotion_authority": False,
        }
        response_path.write_bytes(
            Supervisor._operator_lease_canonical_json(response) + b"\n"
        )
        response_path.chmod(0o600)
        Path(stdout_path).write_bytes(b"")
        Path(stdout_path).chmod(0o600)
        Path(stderr_path).write_bytes(b"")
        Path(stderr_path).chmod(0o600)
        return SimpleNamespace(
            returncode=0,
            timed_out=False,
            output_overflow=False,
        )


def _meta_handoff(
    run_root: Path, *, operator_configuration_sha256: str
) -> dict:
    executable = run_root / "meta_driver"
    executable.write_bytes(b"#!/bin/sh\nexit 99\n")
    executable.chmod(0o700)
    configuration = run_root / "meta_configuration.json"
    configuration.write_bytes(b'{"schema":1}\n')
    configuration.chmod(0o400)
    diagnostic = Supervisor.PostIncidentMetaDiagnostic(
        run_root,
        operator_configuration_sha256=operator_configuration_sha256,
        driver_executable=executable,
        driver_executable_sha256=hashlib.sha256(
            executable.read_bytes()
        ).hexdigest(),
        driver_configuration=configuration,
        driver_configuration_sha256=hashlib.sha256(
            configuration.read_bytes()
        ).hexdigest(),
        driver_attestation_sha256="e" * 64,
        operation_timeout_seconds=60,
        command_runner=_MetaDriver(),
    )
    projection = {
        "schema": Supervisor.POST_INCIDENT_META_SCHEMA,
        "kind":
            "arc_agi3_contiguous_substrate_incident_projection",
        "operator_incident": {
            "attempt_id": "pilot-attempt",
            "operation": "substrate_health_reprobe",
            "fault_domain": "controller_substrate",
            "operation_consecutive": 2,
            "domain_consecutive": 2,
            "threshold": 2,
            "reason_code":
                "deterministic_substrate_configuration_repeated",
        },
        "substrate_incident": {
            "attempt_id": "pilot-attempt",
            "substrate_identity_sha256": "1" * 64,
            "failure_receipt_sha256": "2" * 64,
            "failure_class": "DETERMINISTIC_CONFIGURATION",
            "failure_code": "state_root_unwritable",
            "health_probe_count": 1,
            "attempted_remediation_epochs_sha256": "3" * 64,
            "last_health_probe_sha256": "4" * 64,
        },
        "incident_event_sequence": 12,
        "incident_event_digest": "5" * 64,
    }
    receipt = diagnostic.run_once(projection)
    assert receipt["status"] == "DIAGNOSED"
    return {
        "path": str(diagnostic.terminal_path),
        "sha256": hashlib.sha256(
            diagnostic.terminal_path.read_bytes()
        ).hexdigest(),
        "status": "DIAGNOSED",
    }


class _ProductionPilotExecutor:
    def __init__(
        self,
        production_stack_attestation_path: Path,
        *,
        meta_sequence: int | None = 2,
    ):
        self.production_stack_attestation_path = (
            production_stack_attestation_path
        )
        self.meta_sequence = meta_sequence
        self.executions: list[P.PilotExecution] = []

    def execute_game(self, execution: P.PilotExecution) -> Path:
        assert execution.game == P.PILOT_MANIFEST[
            execution.pilot_sequence - 1
        ][0]
        assert all(
            not list(root.iterdir())
            for root in (
                execution.artifact_root,
                execution.wip_root,
                execution.controller_state_root,
                execution.scheduler_root,
            )
        )
        self.executions.append(execution)
        evidence = execution.run_root / "evidence"
        evidence.mkdir(mode=0o700)
        operator_configuration_sha256 = "d" * 64
        terminal_sources = {
            name: _plain_receipt(
                evidence / f"terminal_{name}.json",
                kind=f"arc_agi3_contiguous_{name}",
                status="PASS",
            )
            for name in (
                "host_child_ledger_audit",
                "runner_state_audit",
                "scheduler_audit",
                "unified_audit",
                "terminal_retention_receipt",
            )
        }
        terminal = _self_receipt(
            evidence / "operator_terminal.json",
            {
                "schema": 1,
                "kind":
                    "arc_agi3_contiguous_pilot_operator_terminal",
                "status": "PASS",
                "pilot_sequence": execution.pilot_sequence,
                "game": execution.game,
                "authoritative_target":
                    execution.authoritative_target,
                "reached": execution.authoritative_target,
                "complete": True,
                "campaign_root": str(execution.run_root),
                "image_digest": execution.image_digest,
                "control_contract_sha256":
                    execution.control_contract_sha256,
                "scheduler_policy_sha256":
                    Scheduler.SCHEDULER_POLICY_SHA256,
                "meta_protocol_sha256":
                    Supervisor.POST_INCIDENT_META_PROTOCOL_SHA256,
                "production_stack_attestation_sha256":
                    execution.production_stack_attestation_sha256,
                "operator_configuration_sha256":
                    operator_configuration_sha256,
                **terminal_sources,
                "active_primary_attempts": [],
                "active_auxiliary_assignments": [],
                "pending_scheduler_decision": False,
                "pending_auxiliary_decision": False,
                "pilot_only": True,
                "canonical_lineage_authority": False,
                "synthetic_evidence": False,
            },
        )
        audits = {}
        for name in P.PILOT_AUDITS:
            source_kind = f"arc_agi3_contiguous_{name}_source"
            source = _plain_receipt(
                evidence / f"{name}_source.json",
                kind=source_kind,
                status="PASS",
            )
            source_evidence = [{
                **source,
                "kind": source_kind,
            }]
            audit_body = {
                "schema": 1,
                "kind": f"arc_agi3_contiguous_pilot_{name}_audit",
                "status": "PASS",
                "audit_name": name,
                "pilot_sequence": execution.pilot_sequence,
                "game": execution.game,
                "authoritative_target":
                    execution.authoritative_target,
                "verified_level_count":
                    execution.authoritative_target,
                "campaign_root": str(execution.run_root),
                "operator_terminal_file_sha256":
                    terminal["sha256"],
                "production_stack_attestation_sha256":
                    execution.production_stack_attestation_sha256,
                "verifier_contract_sha256": hashlib.sha256(
                    P._canonical_json({
                        "audit_name": name,
                        "required_checks": list(
                            P.PILOT_AUDIT_REQUIRED_CHECKS[name]
                        ),
                    })
                ).hexdigest(),
                "checks": {
                    check: True
                    for check in P.PILOT_AUDIT_REQUIRED_CHECKS[name]
                },
                "source_evidence": source_evidence,
                "source_evidence_sha256": hashlib.sha256(
                    P._canonical_json(source_evidence)
                ).hexdigest(),
                "synthetic_evidence": False,
                "result_authority": "pilot_gate_input_only",
            }
            audits[name] = _self_receipt(
                evidence / f"{name}.json", audit_body
            )
        handoffs = []
        if execution.pilot_sequence == self.meta_sequence:
            handoffs.append(_meta_handoff(
                execution.run_root,
                operator_configuration_sha256=(
                    operator_configuration_sha256
                ),
            ))
        body = {
            "schema": P.SCHEMA,
            "kind":
                "arc_agi3_contiguous_production_pilot_outcome",
            "status": "PASS",
            "pilot_sequence": execution.pilot_sequence,
            "game": execution.game,
            "authoritative_target":
                execution.authoritative_target,
            "reached": execution.authoritative_target,
            "pilot_manifest_sha256":
                execution.pilot_manifest_sha256,
            "previous_run_receipt_sha256":
                execution.previous_run_receipt_sha256,
            "image_digest": execution.image_digest,
            "control_contract_sha256":
                execution.control_contract_sha256,
            "scheduler_policy_sha256":
                Scheduler.SCHEDULER_POLICY_SHA256,
            "meta_protocol_sha256":
                Supervisor.POST_INCIDENT_META_PROTOCOL_SHA256,
            "production_stack_attestation_path": str(
                execution.production_stack_attestation_path
            ),
            "production_stack_attestation_sha256":
                execution.production_stack_attestation_sha256,
            "empty_root_genesis_receipt": {
                "path": str(
                    execution.empty_root_genesis_receipt_path
                ),
                "sha256":
                    execution.empty_root_genesis_receipt_sha256,
                "status": "PASS",
            },
            "operator_terminal_receipt": terminal,
            "audit_receipts": audits,
            "clean_continuation_restart_count": 1,
            "meta_handoff_receipts": handoffs,
            "pilot_only": True,
            "canonical_lineage_authority": False,
            "synthetic_evidence": False,
            "result_authority":
                "quarantine_pending_host_admission",
        }
        path = execution.run_root / P.PRODUCTION_OUTCOME_NAME
        _self_receipt(
            path,
            body,
        )
        return path


def _fixture(tmp_path: Path, monkeypatch):
    key = (tmp_path / "pilot.key").resolve()
    key.write_bytes(bytes(range(32)))
    key.chmod(0o400)
    image_digest = "sha256:" + "a" * 64
    control_contract_sha256 = "b" * 64
    runtime = _control(
        tmp_path / "runtime.json",
        {
            "schema": 1,
            "kind": "arc_agi3_python_runtime_manifest",
        },
    )
    source_path = tmp_path / "pilot_executor.py"
    source_path.write_bytes(b"# sealed production pilot executor\n")
    source_path.chmod(0o400)
    source = {
        "path": str(source_path.resolve()),
        "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "status": "CONTROL",
    }
    executor_control_path = (
        "arc/crack_lab/arc_agi3_container_backend.py"
    )
    conformance = _plain_receipt(
        tmp_path / "prelaunch_conformance.json",
        kind="arc_agi3_contiguous_conformance",
        status="PASS",
    )
    conformance_path = Path(conformance["path"])
    conformance_value = {
        "schema": 3,
        "kind": "arc_agi3_contiguous_conformance",
        "status": "PASS",
        "launch_authority": False,
        "container_image_digest": None,
        "control_contract_sha256": control_contract_sha256,
        "control_contract_files_sha256": {
            "arc/crack_lab/gkm_legs.py": "6" * 64,
            executor_control_path: source["sha256"],
        },
        "suite_runtime_manifest_path": runtime["path"],
        "suite_runtime_manifest_sha256": runtime["sha256"],
    }
    conformance_path.chmod(0o600)
    conformance_path.write_bytes(
        P._canonical_json(conformance_value) + b"\n"
    )
    conformance_path.chmod(0o400)
    conformance["sha256"] = hashlib.sha256(
        conformance_path.read_bytes()
    ).hexdigest()
    monkeypatch.setattr(
        P.Supervisor.Conformance,
        "validate_result",
        lambda value: value,
    )
    attempt = _control(
        tmp_path / "launch_attestation.json",
        {"schema": 1, "image": image_digest},
    )
    monkeypatch.setattr(
        Container,
        "_load_launch_attestation",
        lambda _path, *, expected_sha256: SimpleNamespace(
            image=SimpleNamespace(manifest_digest=image_digest),
            document_sha256=expected_sha256,
        ),
    )
    state_root = str((tmp_path / "isolated-state").resolve())
    controller = _control(
        tmp_path / "controller_launch.json",
        {
            "schema": 1,
            "kind": "arc_agi3_controller_launch",
            "credentials_in_argv_or_env": False,
            "bridge_or_arena_mounts": 0,
            "egress_live_probe_before_controller_create": True,
            "controller_image_digest": "sha256:" + "7" * 64,
            "state_root": state_root,
        },
    )
    guardian = _control(
        tmp_path / "guardian_start.json",
        {
            "schema": 1,
            "kind": "arc_agi3_controller_guardian_start",
            "state_root_write_probe": {
                "schema": 1,
                "kind": "controller_state_root_write_probe",
                "status": "PASS",
                "probe_absent_after_fsync": True,
            },
        },
    )
    substrate = _control(
        tmp_path / "substrate_preflight.json",
        {
            "schema": 1,
            "kind": "contiguous_substrate_preflight",
            "status": "PASS",
            "state_root": state_root,
            "state_root_write_probe_status": "PASS",
            "guardian_start_receipt_path": guardian["path"],
            "guardian_start_receipt_sha256": guardian["sha256"],
            "controller_launch_receipt_sha256":
                controller["sha256"],
            "proposer_container_started": False,
            "bridge_connected": False,
            "thread_started": False,
            "turn_started": False,
            "controller_inspect_absent": True,
            "controller_identity_query_empty": True,
            "controller_no_descendants": True,
            "egress_proxy_inspect_absent": True,
            "egress_proxy_identity_query_empty": True,
            "egress_proxy_no_descendants": True,
        },
    )
    backend = _control(
        tmp_path / "backend_launch.json",
        {
            "schema": 1,
            "kind": "contiguous_backend_launch",
            "launch": {
                "substrate_preflight_receipt_path":
                    substrate["path"],
                "substrate_preflight_receipt_sha256":
                    substrate["sha256"],
                "controller_launch_receipt_path":
                    controller["path"],
                "controller_launch_receipt_sha256":
                    controller["sha256"],
                "controller_guardian_start_receipt_path":
                    guardian["path"],
                "controller_guardian_start_receipt_sha256":
                    guardian["sha256"],
            },
        },
    )
    ledger = _control(
        tmp_path / "host_child_ledger.json",
        {
            "schema": 1,
            "kind": "arc_agi3_managed_host_child_ledger_audit",
            "status_counts": {
                "PENDING": 0,
                "ACTIVE": 0,
                "TERMINAL": 0,
                "CLEAN": 0,
            },
            "all_receipts_authenticated": True,
            "external_absence_proof_required_count": 0,
            "all_children_accounted_for": True,
        },
    )
    stack_path = (tmp_path / "production_stack_attestation.json").resolve()
    stack_value = {
        "schema": 1,
        "kind":
            "arc_agi3_contiguous_pilot_production_stack_attestation",
        "status": "PASS",
        "image_digest": image_digest,
        "control_contract_sha256": control_contract_sha256,
        "scheduler_policy_sha256":
            Scheduler.SCHEDULER_POLICY_SHA256,
        "meta_protocol_sha256":
            Supervisor.POST_INCIDENT_META_PROTOCOL_SHA256,
        "prelaunch_conformance": conformance,
        "runtime_manifest": runtime,
        "pilot_executor_source": source,
        "pilot_executor_control_path": executor_control_path,
        "attempt_launch_attestation": attempt,
        "backend_launch_attestation": backend,
        "controller_launch_attestation": controller,
        "guardian_start_receipt": guardian,
        "substrate_preflight_receipt": substrate,
        "host_child_ledger_audit": ledger,
        "backend_contract_sha256": "8" * 64,
        "input_bundle_contract_sha256": "9" * 64,
        "admission_contract_sha256": "a" * 64,
        "gkm_legs_source_sha256": "6" * 64,
        "perception_seed_sha256": hashlib.sha256(
            gkm_legs.PERCEPTION_SEED.encode("utf-8")
        ).hexdigest(),
        "production_stack_scope": "pilot_only_ft09_then_lp85",
        "full_campaign_launch_authority": False,
        "production_entry_reachable": True,
        "synthetic_evidence": False,
    }
    stack_path.write_bytes(P._canonical_json(stack_value) + b"\n")
    stack_path.chmod(0o400)
    return {
        "base_root": (tmp_path / "pilots").resolve(),
        "authentication_key_path": key,
        "gate_receipt_path": (tmp_path / "pilot_gate.json").resolve(),
        "image_digest": image_digest,
        "control_contract_sha256": control_contract_sha256,
        "production_stack_attestation_sha256": hashlib.sha256(
            stack_path.read_bytes()
        ).hexdigest(),
        "production_stack_attestation_path": stack_path,
    }


def test_frozen_pilot_executor_runs_exact_order_and_unlocks_gate(
    tmp_path, monkeypatch,
):
    options = _fixture(tmp_path, monkeypatch)
    executor = _ProductionPilotExecutor(
        options["production_stack_attestation_path"]
    )
    issued = P.execute_frozen_pilots(
        executor=executor, **options
    )
    assert [
        item.game for item in executor.executions
    ] == ["ft09", "lp85"]
    assert [
        item.authoritative_target for item in executor.executions
    ] == [6, 8]
    assert issued["status"] == "PASS"
    assert issued["full_campaign_launch_gate"] == "UNLOCKED"
    assert issued["meta_handoff_count"] == 1
    verified = P.verify_pilot_gate_receipt(
        options["gate_receipt_path"],
        authentication_key_path=options[
            "authentication_key_path"
        ],
        expected_image_digest=options["image_digest"],
        expected_control_contract_sha256=options[
            "control_contract_sha256"
        ],
        expected_production_stack_attestation_sha256=options[
            "production_stack_attestation_sha256"
        ],
    )
    assert verified["pilot_games"] == ["ft09", "lp85"]
    assert verified["pilot_targets"] == [6, 8]


def test_pilot_gate_rejects_no_real_meta_handoff(
    tmp_path, monkeypatch
):
    options = _fixture(tmp_path, monkeypatch)
    executor = _ProductionPilotExecutor(
        options["production_stack_attestation_path"],
        meta_sequence=None,
    )
    with pytest.raises(
        P.PilotContractError,
        match="one real production meta handoff",
    ):
        P.execute_frozen_pilots(
            executor=executor, **options
        )
    assert [
        item.game for item in executor.executions
    ] == ["ft09", "lp85"]
    assert not options["gate_receipt_path"].exists()


def test_pilot_gate_rejects_run_substitution_after_outcome(
    tmp_path, monkeypatch
):
    options = _fixture(tmp_path, monkeypatch)
    executor = _ProductionPilotExecutor(
        options["production_stack_attestation_path"]
    )
    P.execute_frozen_pilots(executor=executor, **options)
    gate_path = options["gate_receipt_path"]
    gate = json.loads(gate_path.read_text(encoding="ascii"))
    gate["run_receipts"].reverse()
    body = {
        key: value
        for key, value in gate.items()
        if key not in {
            "receipt_sha256",
            "host_authentication_sha256",
        }
    }
    gate = {
        **body,
        "receipt_sha256": hashlib.sha256(
            P._canonical_json(body)
        ).hexdigest(),
        "host_authentication_sha256": P._authentication(
            P._read_key(options["authentication_key_path"]),
            body,
        ),
    }
    gate_path.chmod(0o600)
    gate_path.write_bytes(P._canonical_json(gate) + b"\n")
    gate_path.chmod(0o400)
    with pytest.raises(
        P.PilotContractError,
        match="run reference",
    ):
        P.verify_pilot_gate_receipt(
            gate_path,
            authentication_key_path=options[
                "authentication_key_path"
            ],
            expected_image_digest=options["image_digest"],
            expected_control_contract_sha256=options[
                "control_contract_sha256"
            ],
            expected_production_stack_attestation_sha256=options[
                "production_stack_attestation_sha256"
            ],
        )


def test_pilot_gate_reopens_every_run_byte(tmp_path, monkeypatch):
    options = _fixture(tmp_path, monkeypatch)
    executor = _ProductionPilotExecutor(
        options["production_stack_attestation_path"]
    )
    P.execute_frozen_pilots(executor=executor, **options)
    run = executor.executions[0].run_root / "pilot_run.json"
    run.chmod(0o600)
    run.write_bytes(b'{"schema":1}\n')
    run.chmod(0o400)
    with pytest.raises(P.PilotContractError):
        P.verify_pilot_gate_receipt(
            options["gate_receipt_path"],
            authentication_key_path=options[
                "authentication_key_path"
            ],
            expected_image_digest=options["image_digest"],
            expected_control_contract_sha256=options[
                "control_contract_sha256"
            ],
            expected_production_stack_attestation_sha256=options[
                "production_stack_attestation_sha256"
            ],
        )


def test_pilot_controller_recovers_existing_runs_without_reexecution(
    tmp_path, monkeypatch,
):
    options = _fixture(tmp_path, monkeypatch)
    first = _ProductionPilotExecutor(
        options["production_stack_attestation_path"]
    )
    P.execute_frozen_pilots(executor=first, **options)
    options["gate_receipt_path"].unlink()

    class NoReexecution:
        production_stack_attestation_path = options[
            "production_stack_attestation_path"
        ]

        def execute_game(self, _execution):
            raise AssertionError(
                "authenticated completed pilot was executed twice"
            )

    recovered = P.execute_frozen_pilots(
        executor=NoReexecution(), **options
    )
    assert recovered["status"] == "PASS"
    assert recovered["pilot_games"] == ["ft09", "lp85"]


def test_pilot_rejects_full_launch_authority_as_circular_input(
    tmp_path, monkeypatch
):
    options = _fixture(tmp_path, monkeypatch)
    stack_path = options["production_stack_attestation_path"]
    stack = json.loads(stack_path.read_text(encoding="ascii"))
    conformance_path = Path(stack["prelaunch_conformance"]["path"])
    conformance = json.loads(
        conformance_path.read_text(encoding="ascii")
    )
    conformance["launch_authority"] = True
    conformance["container_image_digest"] = options["image_digest"]
    conformance_path.chmod(0o600)
    conformance_path.write_bytes(
        P._canonical_json(conformance) + b"\n"
    )
    conformance_path.chmod(0o400)
    stack["prelaunch_conformance"]["sha256"] = hashlib.sha256(
        conformance_path.read_bytes()
    ).hexdigest()
    stack_path.chmod(0o600)
    stack_path.write_bytes(P._canonical_json(stack) + b"\n")
    stack_path.chmod(0o400)
    with pytest.raises(
        P.PilotContractError,
        match="nonproduction evidence",
    ):
        P.verify_production_stack_attestation(
            stack_path,
            expected_sha256=hashlib.sha256(
                stack_path.read_bytes()
            ).hexdigest(),
            expected_image_digest=options["image_digest"],
            expected_control_contract_sha256=options[
                "control_contract_sha256"
            ],
        )
