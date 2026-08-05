from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

import arc_agi3_contiguous_orchestrator as O
import arc_agi3_contiguous_runner as Runner
import arc_agi3_contiguous_scheduler as Scheduler
import arc_agi3_container_backend as Container
import test_arc_agi3_contiguous_runner as RunnerTest


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii") + b"\n"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _attempt_spec(tmp_path: Path) -> Runner.AttemptSpec:
    runner, backend, _, _, _ = RunnerTest.make_runner(
        tmp_path, max_lanes=1
    )
    runner.cycle()
    assert len(backend.specs) == 1
    return next(iter(backend.specs.values()))


def test_tool_hashes_cover_exact_solver_image_build_controls():
    root = Path(O.__file__).resolve().parent
    tools = O._tool_hashes()
    expected_paths = {
        "arena_rpc_client": root / "arc_agi3_arena_rpc_client.py",
        "replay_worker": root / "arc_agi3_container_worker.py",
        "proposer_worker": root / "arc_agi3_proposer_worker.py",
        "source_schema": root / "arc_agi3_source_schema.py",
        "container_recipe": (
            root / "container" / "Containerfile.arc-agi3-contiguous"
        ),
        "solver_requirements": (
            root / "container" / "arc_agi3_solver_requirements.lock"
        ),
    }
    for name, path in expected_paths.items():
        assert tools[name] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert set(Container.trusted_worker_hashes().values()) <= set(
        tools.values()
    )
    engine_names = (
        "arena",
        "legs_runtime",
        "arena_rpc",
        "arena_rpc_client",
        "replay_worker",
        "proposer_worker",
        "source_schema",
        "container_recipe",
        "solver_requirements",
        "container_backend",
    )
    assert tools["engine"] == O._json_sha256({
        name: tools[name] for name in engine_names
    })


def test_collector_requires_clean_arena_close_even_without_candidate(
    tmp_path,
):
    result = O.TrustedCandidateCollector()(
        SimpleNamespace(),
        Runner.BackendPoll(
            status="exited",
            observation_sha256="a" * 64,
            exit_code=0,
        ),
        None,
        {"status": "completed"},
        tmp_path,
    )
    assert result.kind == "infrastructure"
    assert result.wip is None
    assert "clean trusted Arena session" in result.reason


class _ConfiguredAuxiliaryDriverRunner:
    """Protocol-complete driver double; scheduler fields arrive only in JSON."""

    def __init__(self, campaign_root: Path):
        self.campaign_root = campaign_root
        self.calls: list[tuple[tuple[str, ...], dict]] = []
        self.corrupt_next_binding = False
        self.corrupt_next_stream_binding = False

    @staticmethod
    def _write_json(path: Path, value: object) -> tuple[str, str]:
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = _json_bytes(value)
        path.write_bytes(raw)
        path.chmod(0o400)
        return str(path), _sha(raw)

    def _result(self, request: dict) -> dict:
        decision = Scheduler.auxiliary_decision_from_dict(
            request["decision"]
        )
        assignment = (
            self.campaign_root
            / "auxiliary"
            / decision.assignment_id
        )
        host = assignment / "host"
        operation = request["operation"]
        arguments = request["arguments"]
        if operation == "prepare":
            manifest = assignment / "input" / "manifest.json"
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest_raw = Scheduler.canonical_json(
                asdict(decision.input_manifest)
            )
            manifest.write_bytes(manifest_raw)
            manifest.chmod(0o400)
            bundle_path, bundle_sha = self._write_json(
                host / "input_bundle_receipt.json",
                {
                    "schema": 1,
                    "kind": "auxiliary_private_input_bundle",
                    "assignment_id": decision.assignment_id,
                    "frontier_sha256": decision.frontier_sha256,
                    "parent_checkpoint_sha256":
                        decision.parent_checkpoint_sha256,
                    "input_manifest_sha256":
                        decision.input_manifest_sha256,
                    "observation_ledger_sha256":
                        decision.observation_ledger_sha256,
                    "input_bundle_contract_sha256":
                        decision.input_bundle_contract_sha256,
                    "immutable_inputs": True,
                    "live_lineage_mounted": False,
                    "public_observations_only": True,
                },
            )
            return {
                "input_manifest_path": str(manifest),
                "input_manifest_sha256": _sha(manifest_raw),
                "input_bundle_receipt_path": bundle_path,
                "input_bundle_receipt_sha256": bundle_sha,
            }
        if operation == "launch":
            path, digest = self._write_json(
                host / "launch_receipt.json",
                {
                    "schema": 1,
                    "kind": "auxiliary_backend_launch",
                    "assignment_id": decision.assignment_id,
                    "backend_contract_sha256":
                        decision.backend_contract_sha256,
                    "expert_id": decision.expert_id,
                    "thread_id": decision.thread_id,
                    "model": decision.model,
                    "reasoning_effort": decision.reasoning_effort,
                    "fresh_context": True,
                    "live_lineage_write_authority": False,
                },
            )
            return {
                "launch_receipt_path": path,
                "launch_receipt_sha256": digest,
            }
        if operation == "poll":
            assert arguments["timeout_seconds"] == (
                Runner.POLL_TIMEOUT_SECONDS
            )
            return {
                "status": "exited",
                "observation_sha256": "d" * 64,
                "reason": "none",
            }
        if operation == "collect":
            observation = (
                decision.input_manifest
                .authenticated_public_observation_receipt_sha256s[0]
            )
            output = Scheduler.AuxiliaryOutputEvidence(
                schema=1,
                assignment_id=decision.assignment_id,
                expert_id=decision.expert_id,
                thread_id=decision.thread_id,
                specialization=decision.specialization,
                frontier_sha256=decision.frontier_sha256,
                parent_checkpoint_sha256=(
                    decision.parent_checkpoint_sha256
                ),
                input_manifest_sha256=(
                    decision.input_manifest_sha256
                ),
                output_manifest_sha256=hashlib.sha256(
                    f"output:{decision.assignment_id}".encode()
                ).hexdigest(),
                public_observation_receipt_sha256s=(observation,),
                challenge=Scheduler.SocraticChallengeEvidence(
                    schema=1,
                    hypothesis="The public evidence supports this model.",
                    counter_hypothesis="The pattern is incidental.",
                    falsification_attempt=(
                        "Replayed the distinguishing public prefix."
                    ),
                    observation_receipt_sha256s=(observation,),
                    rejected_conclusions=(
                        "The incidental account did not survive.",
                    ),
                    surviving_conclusions=(
                        "The bounded model remains consistent.",
                    ),
                ),
                quarantined_artifact_sha256s=(
                    hashlib.sha256(
                        f"artifact:{decision.assignment_id}".encode()
                    ).hexdigest(),
                ),
                result_authority="quarantine_only",
                mutates_live_lineage=False,
            )
            return {
                "output": asdict(output),
                "cost_used": 0.25,
                "abort_reason": None,
            }
        if operation == "teardown":
            output = arguments["collection"]["output"]
            path, digest = self._write_json(
                host / "teardown_receipt.json",
                {
                    "schema": 1,
                    "kind": "auxiliary_backend_teardown",
                    "assignment_id": decision.assignment_id,
                    "backend_contract_sha256":
                        decision.backend_contract_sha256,
                    "output_manifest_sha256":
                        output["output_manifest_sha256"],
                    "descendants_absent": True,
                    "live_lineage_mutated": False,
                },
            )
            return {
                "teardown_receipt_path": path,
                "teardown_receipt_sha256": digest,
            }
        if operation == "admit":
            output = Scheduler.auxiliary_output_from_dict(
                arguments["output"]
            )
            common = {
                "schema": 1,
                "assignment_id": decision.assignment_id,
                "frontier_sha256": decision.frontier_sha256,
                "parent_checkpoint_sha256":
                    decision.parent_checkpoint_sha256,
                "output_manifest_sha256":
                    output.output_manifest_sha256,
            }
            replay_path, replay_sha = self._write_json(
                host / "fresh_replay_receipt.json",
                {
                    **common,
                    "kind": "auxiliary_fresh_public_replay",
                    "status": "PASS",
                },
            )
            taint_path, taint_sha = self._write_json(
                host / "taint_receipt.json",
                {
                    **common,
                    "kind": "auxiliary_taint_scan",
                    "status": "CLEAN",
                },
            )
            provenance_path, provenance_sha = self._write_json(
                host / "provenance_receipt.json",
                {
                    **common,
                    "kind": "auxiliary_provenance_scan",
                    "status": "PASS",
                },
            )
            profile = Scheduler.ComplexityProfile(
                schema=1,
                profile_id="profile:" + decision.assignment_id,
                round_index=decision.round_index,
                frontier_sha256=decision.frontier_sha256,
                observation_receipt_sha256=(
                    output.public_observation_receipt_sha256s[0]
                ),
                taint_scan_receipt_sha256=taint_sha,
                priorities=("mechanism_induction", "exact_planning"),
            )
            admitted_sha = Scheduler.sha256_json(asdict(profile))
            admission_path, admission_sha = self._write_json(
                host / "admission_receipt.json",
                {
                    **common,
                    "kind": "auxiliary_profile_admission",
                    "authority": "host_only",
                    "admission_contract_sha256":
                        decision.admission_contract_sha256,
                    "fresh_replay_receipt_sha256": replay_sha,
                    "taint_receipt_sha256": taint_sha,
                    "provenance_receipt_sha256": provenance_sha,
                    "admitted_evidence_sha256": admitted_sha,
                    "verdict": "ADMITTED",
                },
            )
            return {
                "verdict": "ADMITTED",
                "profile": asdict(profile),
                "reason": None,
                "fresh_replay_receipt_path": replay_path,
                "fresh_replay_receipt_sha256": replay_sha,
                "taint_receipt_path": taint_path,
                "taint_receipt_sha256": taint_sha,
                "provenance_receipt_path": provenance_path,
                "provenance_receipt_sha256": provenance_sha,
                "admission_receipt_path": admission_path,
                "admission_receipt_sha256": admission_sha,
            }
        if operation == "abort":
            teardown = None
            if arguments["prior_phase"] == "RUNNING":
                path, digest = self._write_json(
                    host / "abort_teardown_receipt.json",
                    {
                        "schema": 1,
                        "kind": "auxiliary_backend_abort_teardown",
                        "assignment_id": decision.assignment_id,
                        "backend_contract_sha256":
                            decision.backend_contract_sha256,
                        "prior_phase": "RUNNING",
                        "descendants_absent": True,
                        "live_lineage_mutated": False,
                    },
                )
                teardown = {
                    "teardown_receipt_path": path,
                    "teardown_receipt_sha256": digest,
                }
            return {"cost_used": 0.0, "teardown": teardown}
        raise AssertionError(operation)

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
        command = tuple(argv)
        assert command[1::2] == (
            "--configuration",
            "--request",
            "--response",
        )
        request_path = Path(command[4])
        response_path = Path(command[6])
        request = json.loads(request_path.read_text(encoding="utf-8"))
        self.calls.append((command, request))
        result = self._result(request)
        envelope = {
            "schema": 1,
            "kind": "arc_agi3_contiguous_auxiliary_driver_response",
            "driver_protocol_sha256":
                O.AUXILIARY_DRIVER_PROTOCOL_SHA256,
            "operation": request["operation"],
            "assignment_id": request["assignment_id"],
            "decision_sha256": (
                "f" * 64
                if self.corrupt_next_binding
                else request["decision_sha256"]
            ),
            "request_sha256": Scheduler.sha256_json(request),
            "result": result,
        }
        self.corrupt_next_binding = False
        response_path.write_bytes(_json_bytes(envelope))
        response_path.chmod(0o400)
        stdout_path.write_bytes(b"")
        stdout_path.chmod(0o400)
        stderr = (
            b"Traceback (most recent call last):\n"
            b'  File "/private/harness/gkm_arena.py", line 12, in step\n'
            b"KeyboardInterrupt\n"
        )
        stderr_path.write_bytes(stderr)
        stderr_path.chmod(0o400)
        result = SimpleNamespace(
            returncode=0,
            timed_out=False,
            output_overflow=False,
            stdout_sha256=_sha(b""),
            stdout_bytes=0,
            stderr_sha256=(
                "0" * 64
                if self.corrupt_next_stream_binding
                else _sha(stderr)
            ),
            stderr_bytes=len(stderr),
        )
        self.corrupt_next_stream_binding = False
        return result


class _LongRunningAuxiliaryDriverRunner(
    _ConfiguredAuxiliaryDriverRunner
):
    def __init__(
        self,
        campaign_root: Path,
        *,
        terminal_after: int | None = None,
    ):
        super().__init__(campaign_root)
        self.poll_calls = 0
        self.terminal_after = terminal_after

    def _result(self, request: dict) -> dict:
        if request["operation"] != "poll":
            return super()._result(request)
        self.poll_calls += 1
        terminal = (
            self.terminal_after is not None
            and self.poll_calls >= self.terminal_after
        )
        return {
            "status": "exited" if terminal else "running",
            "observation_sha256": "d" * 64,
            "reason": "none",
        }


class _MalformedPollAuxiliaryDriverRunner(
    _ConfiguredAuxiliaryDriverRunner
):
    def _result(self, request: dict) -> dict:
        if request["operation"] == "poll":
            return {
                "status": "malformed",
                "observation_sha256": "d" * 64,
                "reason": "none",
            }
        return super()._result(request)


def _production_auxiliary_backend(
    tmp_path: Path,
) -> tuple[
    O.ProductionAuxiliaryBackend,
    _ConfiguredAuxiliaryDriverRunner,
]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    campaign = tmp_path / "campaign"
    executable = tmp_path / "auxiliary-driver"
    executable.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
    executable.chmod(0o700)
    executable_sha256 = _sha(executable.read_bytes())
    driver_configuration = tmp_path / "auxiliary-driver.json"
    driver_configuration.write_bytes(b'{"schema":1}\n')
    driver_configuration.chmod(0o400)
    driver_configuration_sha256 = _sha(
        driver_configuration.read_bytes()
    )
    launch = Scheduler.AuxiliaryLaunchConfiguration(
        schema=1,
        automatic_dispatch_enabled=True,
        backend_attested=True,
        input_bundle_attested=True,
        admission_gate_attested=True,
        model="gpt-5.6-sol",
        reasoning_effort="max",
        backend_contract_sha256="a" * 64,
        input_bundle_contract_sha256="b" * 64,
        admission_contract_sha256="c" * 64,
        supervisory_proposer=(
            Scheduler.SupervisoryProposerLaunchConfiguration(
                schema=1,
                role=Scheduler.SUPERVISORY_PROPOSER_ROLE,
                automatic_dispatch_enabled=False,
                model="gpt-5.6-sol",
                reasoning_effort="max",
                context_limit_tokens=200_000,
                max_concurrency=1,
            )
        ),
    )
    attestation = tmp_path / "auxiliary-attestation.json"
    attestation.write_bytes(_json_bytes({
        "schema": 1,
        "kind": "arc_agi3_contiguous_auxiliary_backend_attestation",
        "driver_protocol_sha256":
            O.AUXILIARY_DRIVER_PROTOCOL_SHA256,
        "driver_executable_sha256": executable_sha256,
        "driver_configuration_sha256":
            driver_configuration_sha256,
        "backend_contract_sha256":
            launch.backend_contract_sha256,
        "input_bundle_contract_sha256":
            launch.input_bundle_contract_sha256,
        "admission_contract_sha256":
            launch.admission_contract_sha256,
        "model": launch.model,
        "reasoning_effort": launch.reasoning_effort,
        "production_isolation_attested": True,
        "immutable_private_input_attested": True,
        "host_admission_attested": True,
        "descriptor_confined_receipts_attested": True,
        "post_incident_meta_protocol_sha256":
            O.Supervisor.POST_INCIDENT_META_PROTOCOL_SHA256,
        "post_incident_meta_diagnostic_attested": True,
        "post_incident_meta_result_authority":
            "quarantine_only",
    }))
    attestation.chmod(0o400)
    configuration = O.AuxiliaryBackendDriverConfiguration(
        schema=1,
        driver_executable=executable,
        driver_executable_sha256=executable_sha256,
        driver_configuration=driver_configuration,
        driver_configuration_sha256=driver_configuration_sha256,
        backend_attestation=attestation,
        backend_attestation_sha256=_sha(attestation.read_bytes()),
        operation_timeout_seconds=60,
    )
    command_runner = _ConfiguredAuxiliaryDriverRunner(campaign)
    backend = O.ProductionAuxiliaryBackend(
        campaign_root=campaign,
        command_runner=command_runner,
        configuration=configuration,
        launch_configuration=launch,
    )
    return backend, command_runner


def _running_auxiliary_fixture(
    tmp_path: Path,
) -> tuple[
    O.ProductionAuxiliaryBackend,
    _LongRunningAuxiliaryDriverRunner,
    Scheduler.AuxiliaryDecision,
    Runner.AuxiliaryPreparedInput,
    Runner.AuxiliaryLaunch,
]:
    backend, _ = _production_auxiliary_backend(tmp_path)
    runner = RunnerTest._SyntheticAuxiliaryRunner(
        tmp_path / "campaign", backend
    )
    runner.cycle()
    assignment = next(
        iter(runner.state()["auxiliary_assignments"].values())
    )
    assert assignment["state"].phase == "RUNNING"
    assert assignment["prepared"] is not None
    assert assignment["launched"] is not None
    driver = _LongRunningAuxiliaryDriverRunner(
        tmp_path / "campaign"
    )
    backend.command_runner = driver
    return (
        backend,
        driver,
        assignment["decision"],
        assignment["prepared"],
        assignment["launched"],
    )


def _restart_auxiliary_backend(
    backend: O.ProductionAuxiliaryBackend,
    driver: _LongRunningAuxiliaryDriverRunner,
) -> O.ProductionAuxiliaryBackend:
    return O.ProductionAuxiliaryBackend(
        campaign_root=backend.campaign_root,
        command_runner=driver,
        configuration=backend.driver_configuration,
        launch_configuration=backend.launch_configuration,
    )


def _poll_operation_root(
    backend: O.ProductionAuxiliaryBackend,
    decision: Scheduler.AuxiliaryDecision,
) -> Path:
    roots = tuple(
        (
            backend.campaign_root
            / "auxiliary"
            / decision.assignment_id
            / "host"
            / "driver"
        ).glob("poll-*")
    )
    assert len(roots) == 1
    return roots[0]


@pytest.mark.parametrize(
    ("cut", "recovery_additional_driver_calls"),
    (
        ("before_checkpoint_rename", 0),
        ("after_checkpoint_fsync", 1),
        ("mid_compaction_after_response", 1),
        ("mid_compaction_before_sample_rmdir", 1),
        ("after_transient_removal", 1),
    ),
)
def test_auxiliary_poll_checkpoint_crash_cuts_are_bounded(
    tmp_path,
    cut,
    recovery_additional_driver_calls,
):
    (
        backend,
        driver,
        decision,
        prepared,
        launched,
    ) = _running_auxiliary_fixture(tmp_path / cut)
    first = backend.poll(
        decision,
        prepared,
        launched,
        timeout_seconds=Runner.POLL_TIMEOUT_SECONDS,
    )
    assert first.status == "running"
    operation_root = _poll_operation_root(backend, decision)
    assert not tuple(operation_root.glob("sample-*"))
    assert len(tuple(operation_root.glob("poll_checkpoint_*.json"))) == 1

    backend._poll_crash_cut = cut
    with pytest.raises(Runner.SimulatedCrash):
        backend.poll(
            decision,
            prepared,
            launched,
            timeout_seconds=Runner.POLL_TIMEOUT_SECONDS,
        )
    calls_after_crash = driver.poll_calls
    recovered = _restart_auxiliary_backend(backend, driver)
    result = recovered.poll(
        decision,
        prepared,
        launched,
        timeout_seconds=Runner.POLL_TIMEOUT_SECONDS,
    )
    assert result.status == "running"
    assert driver.poll_calls == (
        calls_after_crash + recovery_additional_driver_calls
    )
    assert not tuple(operation_root.glob("sample-*"))
    assert not (operation_root / "response.json").exists()
    assert not (operation_root / "response_binding.json").exists()
    assert not (
        operation_root / "poll_checkpoint_pending.json"
    ).exists()
    checkpoints = tuple(
        operation_root.glob("poll_checkpoint_*.json")
    )
    assert len(checkpoints) == 1
    checkpoint = json.loads(
        checkpoints[0].read_text(encoding="utf-8")
    )
    assert checkpoint["sample_sequence"] == (
        2
        if cut == "before_checkpoint_rename"
        else 3
    )


def test_auxiliary_poll_recovery_rejects_unknown_operation_root_entry(
    tmp_path,
):
    (
        backend,
        driver,
        decision,
        prepared,
        launched,
    ) = _running_auxiliary_fixture(tmp_path)
    assert backend.poll(
        decision,
        prepared,
        launched,
        timeout_seconds=Runner.POLL_TIMEOUT_SECONDS,
    ).status == "running"
    operation_root = _poll_operation_root(backend, decision)
    injected = operation_root / "driver-junk.bin"
    injected.write_bytes(b"unbounded")
    injected.chmod(0o400)
    recovered = _restart_auxiliary_backend(backend, driver)
    with pytest.raises(
        Runner.AuxiliaryBackendFatalError,
        match="operation_root_has_unknown_evidence",
    ):
        recovered.poll(
            decision,
            prepared,
            launched,
            timeout_seconds=Runner.POLL_TIMEOUT_SECONDS,
        )


def test_auxiliary_malformed_poll_retains_one_bounded_forensic_sample(
    tmp_path,
):
    backend, _ = _production_auxiliary_backend(tmp_path)
    runner = RunnerTest._SyntheticAuxiliaryRunner(
        tmp_path / "campaign", backend
    )
    runner.cycle()
    assignment = next(
        iter(runner.state()["auxiliary_assignments"].values())
    )
    assert assignment["state"].phase == "RUNNING"
    malformed = _MalformedPollAuxiliaryDriverRunner(
        tmp_path / "campaign"
    )
    backend.command_runner = malformed
    runner.cycle()
    aborted = next(
        iter(runner.state()["auxiliary_assignments"].values())
    )
    assert aborted["state"].phase == "ABORTED"
    assert aborted["abort_reason"] == (
        "driver_poll_checkpoint_invalid"
    )
    operation_root = _poll_operation_root(
        backend, aborted["decision"]
    )
    samples = tuple(operation_root.glob("sample-*"))
    assert len(samples) == 1
    assert {
        item.name for item in operation_root.iterdir()
    } == {
        "request.json",
        "response.json",
        "response_binding.json",
        samples[0].name,
    }
    assert {item.name for item in samples[0].iterdir()} == {
        "stdout.bin",
        "stderr.bin",
        "stderr_visibility_receipt.json",
        "invocation_receipt.json",
    }
    restarted = _restart_auxiliary_backend(
        backend, malformed
    )
    assert restarted._poll_transient_invocations(
        operation_root
    ) == samples


def test_auxiliary_poll_360_minute_projection_retains_constant_evidence(
    tmp_path,
):
    (
        backend,
        driver,
        decision,
        prepared,
        launched,
    ) = _running_auxiliary_fixture(tmp_path)
    for _ in range(8):
        observed = backend.poll(
            decision,
            prepared,
            launched,
            timeout_seconds=Runner.POLL_TIMEOUT_SECONDS,
        )
        assert observed.status == "running"
        backend = _restart_auxiliary_backend(backend, driver)
    operation_root = _poll_operation_root(backend, decision)
    assert len(tuple(operation_root.iterdir())) == 2
    checkpoint_path = next(
        operation_root.glob("poll_checkpoint_*.json")
    )
    checkpoint = json.loads(
        checkpoint_path.read_text(encoding="utf-8")
    )
    projected_polls = int((360 * 60) / 0.05)
    assert projected_polls == 432_000
    checkpoint["sample_sequence"] = projected_polls - 1
    checkpoint["previous_checkpoint_sha256"] = "e" * 64
    checkpoint["sample_identity_sha256"] = (
        backend._poll_sample_identity(
            request_sha256=checkpoint["request_sha256"],
            sample_sequence=projected_polls - 1,
            previous_checkpoint_sha256="e" * 64,
        )
    )
    checkpoint_path.unlink()
    projected_checkpoint_path = (
        operation_root
        / f"poll_checkpoint_{(projected_polls - 1) % 2}.json"
    )
    projected_checkpoint_path.write_bytes(_json_bytes(checkpoint))
    projected_checkpoint_path.chmod(0o400)

    backend = _restart_auxiliary_backend(backend, driver)
    final = backend.poll(
        decision,
        prepared,
        launched,
        timeout_seconds=Runner.POLL_TIMEOUT_SECONDS,
    )
    assert final.status == "running"
    retained = tuple(operation_root.iterdir())
    assert len(retained) == 2
    assert {item.name for item in retained} >= {"request.json"}
    final_checkpoint = json.loads(
        next(
            operation_root.glob("poll_checkpoint_*.json")
        ).read_text(encoding="utf-8")
    )
    assert final_checkpoint["sample_sequence"] == projected_polls
    assert not tuple(operation_root.glob("sample-*"))


def _operator_path_matrix_fixture(tmp_path: Path):
    authority_roots = {}
    for name in (
        "docker_config_root",
        "runtime_control_snapshot_root",
        "canonical_root",
        "environments_root",
    ):
        selected = tmp_path / "authority-roots" / name
        selected.mkdir(parents=True, mode=0o700)
        authority_roots[name] = selected
    files = {}
    for name in (
        "docker_binary",
        "docker_socket",
        "python_executable",
        "python_runtime_manifest",
        "credential_source",
        "launch_attestation",
        "conformance_result",
        "pilot_gate_receipt",
        "pilot_authentication_key",
    ):
        selected = tmp_path / "authority-files" / name
        selected.parent.mkdir(parents=True, exist_ok=True)
        selected.write_text(f"{name}\n", encoding="ascii")
        selected.chmod(0o400)
        files[name] = selected
    config_path = tmp_path / "operator.json"
    config_path.write_text("{}\n", encoding="ascii")
    config_path.chmod(0o400)
    auxiliary_files = {}
    for name in (
        "driver_executable",
        "driver_configuration",
        "backend_attestation",
    ):
        selected = tmp_path / "auxiliary" / name
        selected.parent.mkdir(parents=True, exist_ok=True)
        selected.write_text(f"{name}\n", encoding="ascii")
        selected.chmod(0o400)
        auxiliary_files[name] = selected
    auxiliary = SimpleNamespace(**auxiliary_files)
    placement_parent = tmp_path / "canaries"
    placement_parent.mkdir(mode=0o700)
    placements = {
        category: (
            "ARC_AGI3_PATH_MATRIX_CANARY"
            if category == "environment"
            else str(placement_parent / f"{category}.txt")
        )
        for category in O.Taint.CONTROLLER_CANARY_CATEGORIES
    }
    mutable_parent = tmp_path / "mutable"
    mutable_parent.mkdir(mode=0o700)
    paths = {
        "campaign_root": mutable_parent / "campaign",
        "promotion_root": mutable_parent / "promotion",
        "replay_evidence_root": mutable_parent / "replay",
        **authority_roots,
        **files,
    }
    return config_path, paths, auxiliary, placements


def test_operator_path_matrix_is_exact_and_pre_mutation(tmp_path):
    config_path, paths, auxiliary, placements = (
        _operator_path_matrix_fixture(tmp_path)
    )
    mutable = tuple(
        paths[name]
        for name in (
            "campaign_root",
            "promotion_root",
            "replay_evidence_root",
        )
    )
    projection = O._operator_path_relationship_projection(
        config_path=config_path,
        paths=paths,
        auxiliary_configuration=auxiliary,
        canary_placements=placements,
    )
    assert projection["allowed_matrix"][
        "mutable_root:any_other"
    ] == "disjoint"
    assert all(not path.exists() for path in mutable)

    nested_paths = dict(paths)
    nested_paths["campaign_root"] = (
        paths["canonical_root"] / "campaign"
    )
    with pytest.raises(
        O.ContiguousOrchestratorError,
        match="relationship is forbidden",
    ):
        O._operator_path_relationship_projection(
            config_path=config_path,
            paths=nested_paths,
            auxiliary_configuration=auxiliary,
            canary_placements=placements,
        )
    assert not nested_paths["campaign_root"].exists()

    with pytest.raises(
        O.ContiguousOrchestratorError,
        match="canonical explicit absolute path",
    ):
        O._absolute_path(
            f"{tmp_path}/mutable/../aliased",
            label="dot-dot inverse",
        )
    with pytest.raises(
        O.ContiguousOrchestratorError,
        match="canonical explicit absolute path",
    ):
        O._absolute_path("/", label="filesystem-root inverse")
    for broad in (
        Path.home().resolve(),
        Path(O.__file__).resolve().parents[2],
        Path(tempfile.gettempdir()).resolve(),
    ):
        broad_paths = dict(paths)
        broad_paths["campaign_root"] = broad
        with pytest.raises(
            O.ContiguousOrchestratorError,
            match="ambient broad directory",
        ):
            O._operator_path_relationship_projection(
                config_path=config_path,
                paths=broad_paths,
                auxiliary_configuration=auxiliary,
                canary_placements=placements,
            )

    checkout_child_paths = dict(paths)
    checkout_child_paths["campaign_root"] = (
        Path(O.__file__).resolve().parents[2]
        / "arc"
        / "crack_lab"
        / "quarantined_attempts"
    )
    with pytest.raises(
        O.ContiguousOrchestratorError,
        match="campaign bytes in the checkout",
    ):
        O._operator_path_relationship_projection(
            config_path=config_path,
            paths=checkout_child_paths,
            auxiliary_configuration=auxiliary,
            canary_placements=placements,
        )
    checkout_canary_placements = dict(placements)
    checkout_canary_placements["repository"] = str(
        Path(O.__file__).resolve().parents[2]
        / "arc"
        / "crack_lab"
        / ".forbidden_canary"
    )
    with pytest.raises(
        O.ContiguousOrchestratorError,
        match="canary bytes in the checkout",
    ):
        O._operator_path_relationship_projection(
            config_path=config_path,
            paths=paths,
            auxiliary_configuration=auxiliary,
            canary_placements=checkout_canary_placements,
        )
    assert not checkout_child_paths["campaign_root"].exists()
    assert not Path(
        checkout_canary_placements["repository"]
    ).exists()

    symlink_target = tmp_path / "symlink-target"
    symlink_target.mkdir()
    (symlink_target / "credential").write_text(
        "credential\n", encoding="ascii"
    )
    symlink_parent = tmp_path / "credential-alias"
    symlink_parent.symlink_to(
        symlink_target, target_is_directory=True
    )
    symlink_paths = dict(paths)
    symlink_paths["credential_source"] = (
        symlink_parent / "credential"
    )
    with pytest.raises(
        O.ContiguousOrchestratorError,
        match="symlink alias",
    ):
        O._operator_path_relationship_projection(
            config_path=config_path,
            paths=symlink_paths,
            auxiliary_configuration=auxiliary,
            canary_placements=placements,
        )

    hardlink_placements = dict(placements)
    hardlink_path = Path(
        hardlink_placements["repository"]
    )
    os.link(paths["credential_source"], hardlink_path)
    with pytest.raises(
        O.ContiguousOrchestratorError,
        match="endpoint identity is aliased",
    ):
        O._operator_path_relationship_projection(
            config_path=config_path,
            paths=paths,
            auxiliary_configuration=auxiliary,
            canary_placements=hardlink_placements,
        )
    hardlink_path.unlink()

    config = SimpleNamespace(
        config_path=config_path,
        auxiliary_backend_configuration=auxiliary,
        canary_placements=placements,
        path_relationships=projection,
        path_relationships_sha256=O._json_sha256(projection),
        **paths,
    )
    replacement = paths["credential_source"].with_name(
        "credential_source_replacement"
    )
    replacement.write_text(
        paths["credential_source"].read_text(encoding="ascii"),
        encoding="ascii",
    )
    replacement.chmod(0o400)
    os.replace(replacement, paths["credential_source"])
    with pytest.raises(
        O.ContiguousOrchestratorError,
        match="identities changed before first mutation",
    ):
        O._revalidate_operator_path_relationships(config)
    assert all(not path.exists() for path in mutable)


def test_formal_auxiliary_backend_executes_only_policy_selected_quarantine_path(
    tmp_path, monkeypatch
):
    backend, command_runner = _production_auxiliary_backend(tmp_path)
    driver_configuration = backend.driver_configuration
    serialized_driver_configuration = {
        "schema": 1,
        "driver_executable": str(
            driver_configuration.driver_executable
        ),
        "driver_executable_sha256":
            driver_configuration.driver_executable_sha256,
        "driver_configuration": str(
            driver_configuration.driver_configuration
        ),
        "driver_configuration_sha256":
            driver_configuration.driver_configuration_sha256,
        "backend_attestation": str(
            driver_configuration.backend_attestation
        ),
        "backend_attestation_sha256":
            driver_configuration.backend_attestation_sha256,
        "operation_timeout_seconds":
            driver_configuration.operation_timeout_seconds,
    }
    assert O._parse_auxiliary_backend_configuration(
        serialized_driver_configuration,
        backend.launch_configuration,
    ) == driver_configuration
    with pytest.raises(
        O.ContiguousOrchestratorError,
        match="schema is not exact",
    ):
        O._parse_auxiliary_backend_configuration(
            {
                **serialized_driver_configuration,
                "game": "lf52",
            },
            backend.launch_configuration,
        )
    with pytest.raises(
        O.ContiguousOrchestratorError,
        match="requires automatic auxiliary dispatch",
    ):
        O._verify_auxiliary_backend_configuration(
            driver_configuration,
            Scheduler.disabled_auxiliary_launch_configuration(),
        )
    runner = RunnerTest._SyntheticAuxiliaryRunner(
        tmp_path / "campaign", backend
    )

    for _ in range(8):
        runner.cycle()
        assignments = runner.state()["auxiliary_assignments"].values()
        if any(
            item["state"].specialization == "complexity_diagnosis"
            and item["state"].phase == "ADMITTED"
            for item in assignments
        ):
            break
    diagnosis = next(
        item
        for item in runner.state()["auxiliary_assignments"].values()
        if item["state"].specialization == "complexity_diagnosis"
    )
    assert diagnosis["state"].phase == "ADMITTED"
    assert diagnosis["state"].trigger_no_progress == 5
    assert diagnosis["state"].reasoning_effort == "max"
    assert diagnosis["state"].output is not None
    assert diagnosis["state"].output.result_authority == "quarantine_only"
    assert diagnosis["state"].output.mutates_live_lineage is False
    assert diagnosis["state"].output.challenge.falsification_attempt

    # The fixed process interface has no caller-selected scheduling switch.
    assert command_runner.calls
    for argv, request in command_runner.calls:
        assert argv[1::2] == (
            "--configuration",
            "--request",
            "--response",
        )
        joined = "\n".join(argv)
        for forbidden in (
            request["decision"]["game"],
            request["decision"]["reasoning_effort"],
            request["decision"]["specialization"],
        ):
            assert forbidden not in joined
        assert request["decision"]["no_progress"] >= 5
        assert request["decision"]["policy_sha256"] == (
            Scheduler.SCHEDULER_POLICY_SHA256
        )

    # Python/harness stderr bytes remain exact host evidence while the only
    # visibility receipt contains a fixed clean one-line projection.
    stderr_files = list(
        (tmp_path / "campaign" / "auxiliary").rglob("stderr.bin")
    )
    assert stderr_files
    for stderr_path in stderr_files:
        assert b"gkm_arena.py" in stderr_path.read_bytes()
        visibility = json.loads(
            stderr_path.with_name(
                "stderr_visibility_receipt.json"
            ).read_text(encoding="utf-8")
        )
        assert visibility["raw_surface_classification"] == (
            "python_or_harness_traceback"
        )
        assert visibility["proposer_visible_taint_status"] == "CLEAN"
        assert visibility["proposer_visible_stderr"] == (
            O.Transport.PROBE_STDERR_SANITIZED_LINE
        )
        assert "gkm_arena" not in (
            visibility["proposer_visible_stderr"]
        )

    # A driver cannot substitute another scheduler decision behind the same
    # operation response.
    second_root = tmp_path / "corrupt"
    corrupt_backend, corrupt_runner = _production_auxiliary_backend(
        second_root
    )
    synthetic = RunnerTest._SyntheticAuxiliaryRunner(
        second_root / "campaign", corrupt_backend
    )
    corrupt_runner.corrupt_next_binding = True
    synthetic.cycle()
    assignment = next(
        iter(synthetic.state()["auxiliary_assignments"].values())
    )
    assert assignment["state"].phase == "ABORTED"
    assert assignment["abort_reason"] == (
        "driver_response_binding_invalid"
    )

    poll_root = tmp_path / "corrupt-poll"
    poll_backend, poll_driver = _production_auxiliary_backend(
        poll_root
    )
    poll_runner = RunnerTest._SyntheticAuxiliaryRunner(
        poll_root / "campaign", poll_backend
    )
    poll_runner.cycle()
    running = next(
        iter(poll_runner.state()["auxiliary_assignments"].values())
    )
    assert running["state"].phase == "RUNNING"
    poll_driver.corrupt_next_binding = True
    poll_runner.cycle()
    terminal = next(
        iter(poll_runner.state()["auxiliary_assignments"].values())
    )
    assert terminal["state"].phase == "ABORTED"
    assert terminal["abort_reason"] == (
        "driver_response_binding_invalid"
    )

    admission_root = tmp_path / "corrupt-admission"
    admission_backend, admission_driver = (
        _production_auxiliary_backend(admission_root)
    )
    admission_runner = RunnerTest._SyntheticAuxiliaryRunner(
        admission_root / "campaign", admission_backend
    )
    admission_runner.cycle()
    admission_runner.cycle()
    quarantined = next(
        iter(
            admission_runner.state()[
                "auxiliary_assignments"
            ].values()
        )
    )
    assert quarantined["state"].phase == "QUARANTINED"
    admission_driver.corrupt_next_binding = True
    admission_runner.cycle()
    rejected = next(
        iter(
            admission_runner.state()[
                "auxiliary_assignments"
            ].values()
        )
    )
    assert rejected["state"].phase == "REJECTED"

    # The stream classifier must consume the exact bytes reported by the
    # bounded command runner.  A reported digest for different stderr bytes
    # aborts before those bytes can receive a visibility classification.
    stream_root = tmp_path / "corrupt-stream"
    stream_backend, stream_driver = _production_auxiliary_backend(
        stream_root
    )
    stream_runner = RunnerTest._SyntheticAuxiliaryRunner(
        stream_root / "campaign", stream_backend
    )
    stream_driver.corrupt_next_stream_binding = True
    stream_runner.cycle()
    stream_assignment = next(
        iter(
            stream_runner.state()[
                "auxiliary_assignments"
            ].values()
        )
    )
    assert stream_assignment["state"].phase == "ABORTED"
    assert stream_assignment["abort_reason"] == (
        "driver_stream_binding_invalid"
    )
    prepare_streams = list(
        (
            stream_root
            / "campaign"
            / "auxiliary"
            / stream_assignment["decision"].assignment_id
            / "host"
            / "driver"
        ).glob("prepare-*/invocation-*/stderr.bin")
    )
    assert len(prepare_streams) == 1
    assert not prepare_streams[0].with_name(
        "stderr_visibility_receipt.json"
    ).exists()

    # Process-local path maps may be empty after an operator restart.  Before
    # each recovered phase invokes its driver, the runner descriptor-reopens
    # every journaled prerequisite and exact-byte rebinds it.  Start with the
    # INPUT_PREPARED -> launch boundary.
    prepared_restart_root = tmp_path / "prepared-restart-rebind"
    prepared_backend, prepared_driver = _production_auxiliary_backend(
        prepared_restart_root
    )
    prepared_runner = RunnerTest._SyntheticAuxiliaryRunner(
        prepared_restart_root / "campaign", prepared_backend
    )
    prepared_assignment_id = prepared_runner._reserve_auxiliary(
        prepared_runner.state()
    )
    assert prepared_assignment_id is not None
    prepared_runner._prepare_auxiliary(
        prepared_runner.state()["auxiliary_assignments"][
            prepared_assignment_id
        ]
    )
    assert prepared_runner.state()["auxiliary_assignments"][
        prepared_assignment_id
    ]["state"].phase == "INPUT_PREPARED"
    recovered_prepared_backend = O.ProductionAuxiliaryBackend(
        campaign_root=prepared_restart_root / "campaign",
        command_runner=prepared_driver,
        configuration=prepared_backend.driver_configuration,
        launch_configuration=prepared_backend.launch_configuration,
    )
    prepared_ordering: list[tuple[str, str]] = []
    original_prepared_read = (
        recovered_prepared_backend.read_confined_receipt
    )
    original_prepared_stream = prepared_driver.run_attached_stream

    def tracked_prepared_read(
        observed_decision,
        path_value,
        *,
        maximum,
    ):
        prepared_ordering.append(("read", Path(path_value).name))
        return original_prepared_read(
            observed_decision,
            path_value,
            maximum=maximum,
        )

    def tracked_prepared_stream(argv, **kwargs):
        request = json.loads(
            Path(argv[4]).read_text(encoding="utf-8")
        )
        prepared_ordering.append(("driver", request["operation"]))
        return original_prepared_stream(argv, **kwargs)

    monkeypatch.setattr(
        recovered_prepared_backend,
        "read_confined_receipt",
        tracked_prepared_read,
    )
    monkeypatch.setattr(
        prepared_driver,
        "run_attached_stream",
        tracked_prepared_stream,
    )
    prepared_runner.auxiliary_backend = recovered_prepared_backend
    prepared_runner.cycle()
    first_prepared_driver = next(
        index
        for index, event in enumerate(prepared_ordering)
        if event[0] == "driver"
    )
    assert prepared_ordering[first_prepared_driver] == (
        "driver", "launch"
    )
    assert {
        name
        for kind, name in prepared_ordering[:first_prepared_driver]
        if kind == "read"
    } == {"manifest.json", "input_bundle_receipt.json"}

    # The same authenticated-journal recovery rule applies to RUNNING and
    # QUARANTINED.  The ordering trace proves the reads precede each operation.
    restart_root = tmp_path / "restart-rebind"
    restart_backend, restart_driver = _production_auxiliary_backend(
        restart_root
    )
    restart_runner = RunnerTest._SyntheticAuxiliaryRunner(
        restart_root / "campaign", restart_backend
    )
    restart_runner.cycle()
    assert next(
        iter(
            restart_runner.state()[
                "auxiliary_assignments"
            ].values()
        )
    )["state"].phase == "RUNNING"
    restarted_backend = O.ProductionAuxiliaryBackend(
        campaign_root=restart_root / "campaign",
        command_runner=restart_driver,
        configuration=restart_backend.driver_configuration,
        launch_configuration=restart_backend.launch_configuration,
    )
    ordering: list[tuple[str, str]] = []
    original_restart_read = (
        restarted_backend.read_confined_receipt
    )
    original_restart_stream = restart_driver.run_attached_stream

    def tracked_restart_read(
        observed_decision,
        path_value,
        *,
        maximum,
    ):
        ordering.append(("read", Path(path_value).name))
        return original_restart_read(
            observed_decision,
            path_value,
            maximum=maximum,
        )

    def tracked_restart_stream(argv, **kwargs):
        request = json.loads(
            Path(argv[4]).read_text(encoding="utf-8")
        )
        ordering.append(("driver", request["operation"]))
        return original_restart_stream(argv, **kwargs)

    monkeypatch.setattr(
        restarted_backend,
        "read_confined_receipt",
        tracked_restart_read,
    )
    monkeypatch.setattr(
        restart_driver,
        "run_attached_stream",
        tracked_restart_stream,
    )
    restart_runner.auxiliary_backend = restarted_backend
    restart_runner.cycle()
    first_driver = next(
        index
        for index, event in enumerate(ordering)
        if event[0] == "driver"
    )
    assert ordering[first_driver] == ("driver", "poll")
    assert {
        name for kind, name in ordering[:first_driver]
        if kind == "read"
    } == {
        "manifest.json",
        "input_bundle_receipt.json",
        "launch_receipt.json",
    }
    quarantined_after_restart = next(
        iter(
            restart_runner.state()[
                "auxiliary_assignments"
            ].values()
        )
    )
    assert quarantined_after_restart["state"].phase == "QUARANTINED"

    restarted_admission_backend = O.ProductionAuxiliaryBackend(
        campaign_root=restart_root / "campaign",
        command_runner=restart_driver,
        configuration=restart_backend.driver_configuration,
        launch_configuration=restart_backend.launch_configuration,
    )
    ordering.clear()
    original_admission_read = (
        restarted_admission_backend.read_confined_receipt
    )

    def tracked_admission_read(
        observed_decision,
        path_value,
        *,
        maximum,
    ):
        ordering.append(("read", Path(path_value).name))
        return original_admission_read(
            observed_decision,
            path_value,
            maximum=maximum,
        )

    monkeypatch.setattr(
        restarted_admission_backend,
        "read_confined_receipt",
        tracked_admission_read,
    )
    restart_runner.auxiliary_backend = restarted_admission_backend
    restart_runner.cycle()
    first_driver = next(
        index
        for index, event in enumerate(ordering)
        if event[0] == "driver"
    )
    assert ordering[first_driver] == ("driver", "admit")
    assert {
        name for kind, name in ordering[:first_driver]
        if kind == "read"
    } == {
        "manifest.json",
        "input_bundle_receipt.json",
        "launch_receipt.json",
        "teardown_receipt.json",
    }
    assert next(
        iter(
            restart_runner.state()[
                "auxiliary_assignments"
            ].values()
        )
    )["state"].phase == "ADMITTED"

    # A changed prerequisite after restart is not silently rebound.  Its
    # authenticated digest fails before the poll driver receives control.
    restart_tamper_root = tmp_path / "restart-tamper"
    old_backend, old_driver = _production_auxiliary_backend(
        restart_tamper_root
    )
    restart_tamper_runner = RunnerTest._SyntheticAuxiliaryRunner(
        restart_tamper_root / "campaign", old_backend
    )
    restart_tamper_runner.cycle()
    old_assignment = next(
        iter(
            restart_tamper_runner.state()[
                "auxiliary_assignments"
            ].values()
        )
    )
    old_launch = old_assignment["launched"]
    assert old_launch is not None
    launch_path = Path(old_launch.launch_receipt_path)
    changed_launch = json.loads(
        launch_path.read_text(encoding="utf-8")
    )
    changed_launch["fresh_context"] = False
    replacement_launch = launch_path.with_name(
        "changed_launch_receipt.json"
    )
    replacement_launch.write_bytes(_json_bytes(changed_launch))
    replacement_launch.chmod(0o400)
    os.replace(replacement_launch, launch_path)
    restarted_tamper_backend = O.ProductionAuxiliaryBackend(
        campaign_root=restart_tamper_root / "campaign",
        command_runner=old_driver,
        configuration=old_backend.driver_configuration,
        launch_configuration=old_backend.launch_configuration,
    )
    restart_tamper_runner.auxiliary_backend = (
        restarted_tamper_backend
    )
    calls_before_tamper = len(old_driver.calls)
    # Exact authority drift is campaign-wide fatal for the current cycle.  It
    # must not be downgraded to a lane-local recoverable diagnostic because a
    # later lane could otherwise act after the host has observed corrupted
    # authenticated state.
    with pytest.raises(
        Runner.ExactAuthorityGateError,
        match="auxiliary backend launch digest changed",
    ):
        restart_tamper_runner.cycle()
    assert len(old_driver.calls) == calls_before_tamper

    # A successful response is canonical and has a durable raw-byte binding.
    # Replacing it before a new backend process recovers the operation cannot
    # be reparsed or acted upon.
    response_root = tmp_path / "response-recovery"
    response_backend, response_driver = (
        _production_auxiliary_backend(response_root)
    )
    response_runner = RunnerTest._SyntheticAuxiliaryRunner(
        response_root / "campaign", response_backend
    )
    response_runner.cycle()
    response_assignment = next(
        iter(
            response_runner.state()[
                "auxiliary_assignments"
            ].values()
        )
    )
    response_decision = response_assignment["decision"]
    prepare_response = next(
        (
            response_root
            / "campaign"
            / "auxiliary"
            / response_decision.assignment_id
            / "host"
            / "driver"
        ).glob("prepare-*/response.json")
    )
    response_binding_path = prepare_response.with_name(
        "response_binding.json"
    )
    assert response_binding_path.is_file()
    altered_envelope = json.loads(
        prepare_response.read_text(encoding="utf-8")
    )
    altered_envelope["decision_sha256"] = "f" * 64
    altered_response = prepare_response.with_name(
        "altered_response.json"
    )
    altered_response.write_bytes(_json_bytes(altered_envelope))
    altered_response.chmod(0o400)
    os.replace(altered_response, prepare_response)
    recovered_response_backend = O.ProductionAuxiliaryBackend(
        campaign_root=response_root / "campaign",
        command_runner=response_driver,
        configuration=response_backend.driver_configuration,
        launch_configuration=response_backend.launch_configuration,
    )
    response_calls_before = len(response_driver.calls)
    with pytest.raises(
        Runner.AuxiliaryBackendFatalError,
        match="driver_response_recovery_binding_invalid",
    ):
        recovered_response_backend.prepare(response_decision)
    assert len(response_driver.calls) == response_calls_before

    # Lexical ancestry is not confinement: ``..`` and a symlinked directory
    # component are rejected before any driver-selected file can be trusted.
    decision = diagnosis["decision"]
    prepared = diagnosis["prepared"]
    assert prepared is not None
    assignment_root = (
        tmp_path / "campaign" / "auxiliary" / decision.assignment_id
    )
    dotdot = str(
        assignment_root
        / "input"
        / ".."
        / "input"
        / "manifest.json"
    )
    with pytest.raises(
        Runner.AuxiliaryBackendFatalError,
        match="driver_result_path_escape",
    ):
        backend.read_confined_receipt(
            decision,
            dotdot,
            maximum=Runner.MAX_AUXILIARY_RECEIPT_BYTES,
        )
    outside = tmp_path / "outside-driver-parent"
    outside.mkdir()
    outside_manifest = outside / "manifest.json"
    outside_manifest.write_bytes(
        Path(prepared.input_manifest_path).read_bytes()
    )
    outside_manifest.chmod(0o400)
    linked_parent = assignment_root / "linked-input"
    linked_parent.symlink_to(outside, target_is_directory=True)
    with pytest.raises(
        Runner.AuxiliaryBackendFatalError,
        match="driver_result_path_alias",
    ):
        backend._confined_path(
            str(linked_parent / "manifest.json"),
            decision=decision,
            label="symlink-parent adversary",
        )

    # Once a driver-returned path is bound, replacing one of its real parent
    # directories is detected by component identity, even if the replacement
    # contains byte-identical material.
    rename_root = tmp_path / "rename-race"
    rename_backend, _ = _production_auxiliary_backend(rename_root)
    rename_runner = RunnerTest._SyntheticAuxiliaryRunner(
        rename_root / "campaign", rename_backend
    )
    rename_runner.cycle()
    rename_assignment = next(
        iter(
            rename_runner.state()[
                "auxiliary_assignments"
            ].values()
        )
    )
    rename_decision = rename_assignment["decision"]
    rename_prepared = rename_assignment["prepared"]
    assert rename_prepared is not None
    original_manifest = Path(rename_prepared.input_manifest_path)
    manifest_raw = original_manifest.read_bytes()
    original_parent = original_manifest.parent
    displaced_parent = original_parent.with_name("input-displaced")
    original_parent.rename(displaced_parent)
    original_parent.mkdir(mode=0o700)
    replacement_manifest = original_parent / original_manifest.name
    replacement_manifest.write_bytes(manifest_raw)
    replacement_manifest.chmod(0o400)
    with pytest.raises(
        Runner.AuxiliaryBackendFatalError,
        match="driver_result_path_replaced",
    ):
        rename_backend.read_confined_receipt(
            rename_decision,
            rename_prepared.input_manifest_path,
            maximum=Runner.MAX_AUXILIARY_RECEIPT_BYTES,
        )

    # One descriptor-stable byte string supplies both the digest and JSON
    # checks.  This deterministic A->B pathname swap would pass the former
    # two-open verifier (digest A, expected JSON B); it now rejects A as the
    # wrong exact receipt.  Duplicate and noncanonical JSON are also closed.
    swap_root = tmp_path / "single-read-swap"
    swap_backend, _ = _production_auxiliary_backend(swap_root)
    swap_runner = RunnerTest._SyntheticAuxiliaryRunner(
        swap_root / "campaign", swap_backend
    )
    swap_runner.cycle()
    swap_assignment = next(
        iter(
            swap_runner.state()[
                "auxiliary_assignments"
            ].values()
        )
    )
    swap_decision = swap_assignment["decision"]
    swap_launch = swap_assignment["launched"]
    assert swap_launch is not None
    swap_path = Path(swap_launch.launch_receipt_path)
    raw_a = swap_path.read_bytes()
    value_a = json.loads(raw_a)
    value_b = {**value_a, "fresh_context": False}
    replacement = swap_path.with_name("replacement.json")
    replacement.write_bytes(_json_bytes(value_b))
    replacement.chmod(0o400)
    read_count = 0

    def racing_reader(
        observed_decision,
        path_value,
        *,
        maximum,
    ):
        nonlocal read_count
        assert observed_decision == swap_decision
        assert path_value == str(swap_path)
        assert maximum == Runner.MAX_AUXILIARY_RECEIPT_BYTES
        read_count += 1
        os.replace(replacement, swap_path)
        return raw_a

    monkeypatch.setattr(
        swap_backend, "read_confined_receipt", racing_reader
    )
    with pytest.raises(
        Runner.ContiguousRunnerError,
        match="not the exact host-bound receipt",
    ):
        swap_runner._verify_auxiliary_receipt(
            swap_decision,
            str(swap_path),
            _sha(raw_a),
            expected=value_b,
            label="deterministic swap adversary",
        )
    assert read_count == 1

    duplicate = b'{"schema":1,"schema":1}\n'
    monkeypatch.setattr(
        swap_backend,
        "read_confined_receipt",
        lambda *_args, **_kwargs: duplicate,
    )
    with pytest.raises(
        Runner.ContiguousRunnerError,
        match="not strict JSON",
    ):
        swap_runner._verify_auxiliary_receipt(
            swap_decision,
            str(swap_path),
            _sha(duplicate),
            expected={"schema": 1},
            label="duplicate-key adversary",
        )
    noncanonical = b'{ "schema": 1 }\n'
    monkeypatch.setattr(
        swap_backend,
        "read_confined_receipt",
        lambda *_args, **_kwargs: noncanonical,
    )
    with pytest.raises(
        Runner.ContiguousRunnerError,
        match="encoding is not canonical",
    ):
        swap_runner._verify_auxiliary_receipt(
            swap_decision,
            str(swap_path),
            _sha(noncanonical),
            expected={"schema": 1},
            label="noncanonical adversary",
        )


def test_formal_operator_rejects_cli_and_preflight_before_mutation(
    tmp_path, monkeypatch
):
    direct_environment = dict(os.environ)
    direct_environment.pop("PYTHONPATH", None)
    direct = subprocess.run(
        [
            sys.executable,
            "-I",
            "-E",
            "-s",
            "-B",
            str(Path(O.__file__).resolve()),
            "--help",
        ],
        cwd=tmp_path,
        env=direct_environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert direct.returncode == 0, direct.stderr
    assert "--config" in direct.stdout

    parser = O._build_operator_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--config", "first.json", "--config", "second.json"]
        )
    with pytest.raises(SystemExit):
        parser.parse_args(["--config", "config.json", "--unknown"])

    campaign = tmp_path / "must-not-exist"
    config = SimpleNamespace(
        campaign_root=campaign,
        launch_attestation=tmp_path / "attestation.json",
        conformance_result=tmp_path / "conformance.json",
        canonical_root=tmp_path / "canonical",
        environments_root=tmp_path / "environments",
        python_executable=tmp_path / "python",
        python_executable_sha256="b" * 64,
        python_runtime_manifest=tmp_path / "runtime.json",
        python_runtime_manifest_sha256="c" * 64,
        runtime_control_snapshot_root=tmp_path / "snapshot",
        pilot_gate_receipt=tmp_path / "pilot-gate.json",
        pilot_authentication_key=tmp_path / "pilot.key",
        pilot_production_stack_attestation_sha256="9" * 64,
        backend_configuration=SimpleNamespace(
            image_digest="sha256:" + "a" * 64
        ),
        terminal_condition=O.CANONICAL_TERMINAL_CONDITION,
    )

    preflight_calls = 0

    def reject_preflight(*_args, **_kwargs):
        nonlocal preflight_calls
        preflight_calls += 1
        raise O.Supervisor.SupervisorContractError(
            "terminal authority absent"
        )

    monkeypatch.setattr(
        O.Supervisor, "launch_preflight", reject_preflight
    )
    steered = SimpleNamespace(
        **{
            **vars(config),
            "terminal_condition": "complete_or_quiescent_blocked",
        }
    )
    with pytest.raises(
        O.ContiguousOrchestratorError,
        match="fixed by campaign policy",
    ):
        O.run_operator(steered)
    assert preflight_calls == 0
    assert not campaign.exists()

    with pytest.raises(
        O.Supervisor.SupervisorContractError,
        match="terminal authority",
    ):
        O.run_operator(config)
    assert preflight_calls == 1
    assert not campaign.exists()


def test_formal_operator_holds_one_process_lease_across_mutable_path(
    tmp_path, monkeypatch
):
    campaign = (tmp_path / "campaign").resolve()
    config = SimpleNamespace(
        campaign_root=campaign,
        config_sha256="a" * 64,
        launch_attestation=tmp_path / "attestation.json",
        conformance_result=tmp_path / "conformance.json",
        canonical_root=tmp_path / "canonical",
        environments_root=tmp_path / "environments",
        python_executable=Path(sys.executable).resolve(),
        python_executable_sha256="b" * 64,
        python_runtime_manifest=tmp_path / "runtime.json",
        python_runtime_manifest_sha256="c" * 64,
        runtime_control_snapshot_root=tmp_path / "snapshot",
        pilot_gate_receipt=tmp_path / "pilot-gate.json",
        pilot_authentication_key=tmp_path / "pilot.key",
        pilot_production_stack_attestation_sha256="9" * 64,
        backend_configuration=SimpleNamespace(
            image_digest="sha256:" + "d" * 64
        ),
        auxiliary_backend_configuration=object(),
        auxiliary_launch_configuration=object(),
        terminal_condition=O.CANONICAL_TERMINAL_CONDITION,
    )
    monkeypatch.setattr(
        O.Supervisor,
        "launch_preflight",
        lambda *_args, **_kwargs: {
            "control_contract_sha256": "e" * 64,
            "conformance_registry_sha256": "f" * 64,
        },
    )
    monkeypatch.setattr(
        O,
        "_verify_auxiliary_backend_configuration",
        lambda *_args, **_kwargs: {
            "backend_contract_sha256": "1" * 64,
            "input_bundle_contract_sha256": "2" * 64,
            "admission_contract_sha256": "3" * 64,
        },
    )
    observed = {}

    def owned(
        selected_config,
        *,
        preflight,
        auxiliary_attestation,
        operator_lease,
    ):
        assert selected_config is config
        assert preflight["control_contract_sha256"] == "e" * 64
        assert auxiliary_attestation[
            "backend_contract_sha256"
        ] == "1" * 64
        operator_lease.assert_healthy()
        current = json.loads(
            operator_lease.current_path.read_text(encoding="ascii")
        )
        assert current["status"] == "ACTIVE"
        assert current["owner_instance_id"] == (
            operator_lease.owner_instance_id
        )
        with pytest.raises(
            O.Supervisor.SupervisorContractError,
            match="another live contiguous operator",
        ):
            O.Supervisor.OperatorLease(
                campaign,
                operator_configuration_sha256=config.config_sha256,
                acquire_timeout_seconds=0.05,
                heartbeat_interval_seconds=60,
            ).acquire()
        observed["owner"] = operator_lease.owner_instance_id
        return {"status": "BLOCKED", "receipt_sha256": "4" * 64}

    monkeypatch.setattr(O, "_run_operator_owned_impl", owned)
    assert O._run_operator_impl(config)["status"] == "BLOCKED"
    current = json.loads(
        (
            campaign
            / O.Supervisor.OPERATOR_LEASE_ROOT_NAME
            / "current.json"
        ).read_text(encoding="ascii")
    )
    assert current["owner_instance_id"] == observed["owner"]
    assert current["status"] == "RELEASED"


def test_post_genesis_fatal_is_durable_redacted_operator_incident(
    tmp_path, monkeypatch
):
    campaign = (tmp_path / "campaign").resolve()
    campaign.mkdir()
    (campaign / "operator_genesis.json").write_text(
        '{"schema":1}\n', encoding="utf-8"
    )
    config = SimpleNamespace(
        campaign_root=campaign,
        config_sha256="a" * 64,
    )

    def fatal(_config):
        raise RuntimeError(
            "DO_NOT_PERSIST_THIS_EXCEPTION_MESSAGE_OR_SECRET"
        )

    monkeypatch.setattr(O, "_run_operator_impl", fatal)
    result = O.run_operator(config)
    assert result["status"] == "OPERATOR_INCIDENT"
    assert result["reason_code"] == "uncaught_post_genesis_fatal"
    assert result["error_class"] == "RuntimeError"
    assert result["journal_status"] == "UNAVAILABLE"
    incident = campaign / "operator_incident.json"
    retained = incident.read_bytes()
    assert b"DO_NOT_PERSIST" not in retained
    assert json.loads(retained) == result
    assert result["receipt_sha256"] == O._json_sha256({
        key: value
        for key, value in result.items()
        if key != "receipt_sha256"
    })
    assert O.run_operator(config) == result


def _latched_substrate_meta_fixture(tmp_path):
    operator_incident = {
        "attempt_id": "attempt-1",
        "operation": "substrate_health_reprobe",
        "fault_domain": "controller_substrate",
        "operation_consecutive": 2,
        "domain_consecutive": 2,
        "threshold": 2,
        "reason_code":
            "deterministic_substrate_configuration_repeated",
    }
    state = {
        "operator_incident": operator_incident,
        "substrate_incident": {
            "attempt_id": "attempt-1",
            "substrate_identity_sha256": "a" * 64,
            "failure_receipt_sha256": "b" * 64,
            "failure_class": "DETERMINISTIC_CONFIGURATION",
            "failure_code": "runtime_manifest_drift",
            "health_probe_count": 1,
            "attempted_remediation_epochs": ["epoch-1"],
            "last_health_probe": {
                "status": "FAILED",
                "failure_code": "runtime_manifest_drift",
            },
        },
    }
    events = [{
        "sequence": 9,
        "digest": "c" * 64,
        "kind": "OPERATOR_INCIDENT",
        "payload": operator_incident,
    }]
    auxiliary = SimpleNamespace(
        driver_executable=tmp_path / "driver",
        driver_executable_sha256="d" * 64,
        driver_configuration=tmp_path / "driver.json",
        driver_configuration_sha256="e" * 64,
        backend_attestation_sha256="f" * 64,
        operation_timeout_seconds=60,
    )
    config = SimpleNamespace(
        campaign_root=tmp_path / "campaign",
        promotion_root=tmp_path / "promotion",
        config_sha256="1" * 64,
        auxiliary_backend_configuration=auxiliary,
        auxiliary_launch_configuration=object(),
    )
    runner = SimpleNamespace(
        state=lambda: state,
        journal=SimpleNamespace(read=lambda: events),
    )
    lease = SimpleNamespace(assert_healthy=lambda: None)
    return config, runner, lease, state, events


def test_latched_substrate_meta_path_is_reachable_and_no_authority(
    tmp_path, monkeypatch
):
    config, runner, lease, state, events = (
        _latched_substrate_meta_fixture(tmp_path)
    )
    monkeypatch.setattr(
        O,
        "_verify_auxiliary_backend_configuration",
        lambda *_args: {"status": "PASS"},
    )
    monkeypatch.setattr(
        O,
        "_post_incident_meta_protected_snapshot",
        lambda **_kwargs: "2" * 64,
    )
    captured = {}

    class Diagnostic:
        def __init__(self, *_args, **kwargs):
            captured["configuration"] = kwargs

        def run_once(self, projection):
            captured["projection"] = projection
            return {
                "status": "DIAGNOSED",
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

    monkeypatch.setattr(
        O.Supervisor, "PostIncidentMetaDiagnostic", Diagnostic
    )
    result = O._run_latched_substrate_meta_diagnostic(
        config,
        runner=runner,
        state=state,
        journal_events=events,
        command_runner=object(),
        operator_lease=lease,
    )
    assert result["status"] == "DIAGNOSED"
    assert result["recommended_operator_action"] == (
        "REMATERIALIZE_AND_REPROBE_CONTROLLER_SUBSTRATE"
    )
    assert captured["projection"]["operator_incident"][
        "attempt_id"
    ] == "attempt-1"
    assert "game" not in captured["projection"]["substrate_incident"]
    assert "source" not in json.dumps(captured["projection"])
    assert "wip" not in json.dumps(captured["projection"])
    assert all(
        result[field] is False
        for field in (
            "scheduler_authority",
            "solver_authority",
            "wip_authority",
            "cost_authority",
            "retry_authority",
            "dispatch_authority",
            "promotion_authority",
        )
    )


def test_latched_substrate_meta_path_rejects_authority_mutation(
    tmp_path, monkeypatch
):
    config, runner, lease, state, events = (
        _latched_substrate_meta_fixture(tmp_path)
    )
    monkeypatch.setattr(
        O,
        "_verify_auxiliary_backend_configuration",
        lambda *_args: {"status": "PASS"},
    )
    monkeypatch.setattr(
        O,
        "_post_incident_meta_protected_snapshot",
        lambda **_kwargs: "2" * 64,
    )

    class MutatingDiagnostic:
        def __init__(self, *_args, **_kwargs):
            pass

        def run_once(self, _projection):
            state["substrate_incident"]["health_probe_count"] = 2
            return {"status": "DIAGNOSED"}

    monkeypatch.setattr(
        O.Supervisor,
        "PostIncidentMetaDiagnostic",
        MutatingDiagnostic,
    )
    with pytest.raises(
        O.ContiguousOrchestratorError,
        match="mutated campaign authority",
    ):
        O._run_latched_substrate_meta_diagnostic(
            config,
            runner=runner,
            state={
                **state,
                "substrate_incident": {
                    **state["substrate_incident"],
                    "health_probe_count": 1,
                },
            },
            journal_events=events,
            command_runner=object(),
            operator_lease=lease,
        )


def test_meta_projection_is_stable_after_recovery_journal_events(
    tmp_path,
):
    _config, _runner, _lease, state, events = (
        _latched_substrate_meta_fixture(tmp_path)
    )
    first = O._post_incident_meta_projection(state, events)
    later = [
        *events,
        {
            "sequence": 10,
            "digest": "d" * 64,
            "kind": "META_SUBSTRATE_RECOVERY_AUTHORIZED",
            "payload": {"authorization_id": "later"},
        },
    ]
    assert O._post_incident_meta_projection(state, later) == first
    assert first["incident_event_sequence"] == 9
    assert first["incident_event_digest"] == "c" * 64


def _meta_recovery_state(*, recovered, mutated_lane=False):
    state = {
        "operator_incident": (
            None
            if recovered
            else {"operation": "substrate_health_reprobe"}
        ),
        "substrate_incident": (
            None
            if recovered
            else {
                "meta_recovery": {
                    "phase": "FAILED",
                },
            }
        ),
        "failure_operation_circuits": {
            "substrate_health_reprobe:controller_substrate": {
                "consecutive": 0 if recovered else 2,
            },
        },
        "failure_domain_circuits": {
            "controller_substrate": {
                "consecutive": 0 if recovered else 2,
            },
        },
        "lanes": {
            "ft09": {
                "reached": 1 if mutated_lane else 0,
                "no_progress": 13,
            },
        },
        "attempts": {"attempt-1": {"phase": "CLOSED"}},
        "settled_cost_units": 0,
        "solved_levels": 0,
        "total_levels": 183,
    }
    return state


def test_meta_recovery_requires_fresh_pass_journal_before_resume():
    latched = _meta_recovery_state(recovered=False)
    restored = _meta_recovery_state(recovered=True)
    calls = []

    class FakeRunner:
        def apply_meta_substrate_recovery(self, **kwargs):
            calls.append(kwargs)
            return restored

        def state(self):
            return restored

        journal = SimpleNamespace(
            read=lambda: [
                {"kind": "META_SUBSTRATE_RECOVERY_AUTHORIZED"},
                {"kind": "META_SUBSTRATE_HEALTH_RESTORED"},
                {"kind": "META_SUBSTRATE_RESUME_AUTHORIZED"},
            ],
        )

    resumed, observed = O._apply_latched_substrate_meta_recovery(
        runner=FakeRunner(),
        latched_state=latched,
        meta_diagnostic={
            "status": "DIAGNOSED",
            "recommended_operator_action":
                O.Runner.META_SUBSTRATE_RECOVERY_RECOMMENDATION,
            "request_sha256": "a" * 64,
            "response_sha256": "b" * 64,
            "receipt_sha256": "c" * 64,
        },
        operator_lease=SimpleNamespace(assert_healthy=lambda: None),
    )
    assert resumed is True
    assert observed == restored
    assert calls == [{
        "meta_request_sha256": "a" * 64,
        "meta_response_sha256": "b" * 64,
        "meta_terminal_sha256": "c" * 64,
        "recommendation":
            O.Runner.META_SUBSTRATE_RECOVERY_RECOMMENDATION,
    }]


def test_meta_recovery_rejects_solver_authority_mutation():
    latched = _meta_recovery_state(recovered=False)
    mutated = _meta_recovery_state(
        recovered=True, mutated_lane=True
    )

    class FakeRunner:
        def apply_meta_substrate_recovery(self, **_kwargs):
            return mutated

        def state(self):
            return mutated

        journal = SimpleNamespace(read=lambda: [])

    with pytest.raises(
        O.ContiguousOrchestratorError,
        match="mutated campaign authority",
    ):
        O._apply_latched_substrate_meta_recovery(
            runner=FakeRunner(),
            latched_state=latched,
            meta_diagnostic={
                "status": "DIAGNOSED",
                "recommended_operator_action":
                    O.Runner.META_SUBSTRATE_RECOVERY_RECOMMENDATION,
                "request_sha256": "a" * 64,
                "response_sha256": "b" * 64,
                "receipt_sha256": "c" * 64,
            },
            operator_lease=SimpleNamespace(
                assert_healthy=lambda: None
            ),
        )


def test_meta_no_safe_recommendation_never_calls_runner_recovery():
    latched = _meta_recovery_state(recovered=False)

    class FakeRunner:
        def apply_meta_substrate_recovery(self, **_kwargs):
            raise AssertionError("quarantined text gained authority")

    resumed, observed = O._apply_latched_substrate_meta_recovery(
        runner=FakeRunner(),
        latched_state=latched,
        meta_diagnostic={
            "status": "DIAGNOSED",
            "recommended_operator_action":
                "NO_SAFE_AUTOMATED_RECOVERY",
        },
        operator_lease=SimpleNamespace(assert_healthy=lambda: None),
    )
    assert resumed is False
    assert observed == latched


def test_production_host_child_ledger_is_campaign_bound(
    tmp_path, monkeypatch
):
    captured = {}

    class CommandRunner:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        Container, "SubprocessCommandRunner", CommandRunner
    )
    campaign = (tmp_path / "campaign").resolve()
    config = SimpleNamespace(
        campaign_root=campaign,
        docker_socket=(tmp_path / "docker.sock").resolve(),
        docker_config_root=(tmp_path / "docker-config").resolve(),
    )
    runner = O._production_command_runner(config)
    assert isinstance(runner, CommandRunner)
    assert captured == {
        "docker_socket": config.docker_socket,
        "docker_config": config.docker_config_root,
        "invocation_ledger_root":
            campaign / "host_child_invocations",
    }


def test_host_child_ledger_audit_requires_quiescent_accounting(
    tmp_path,
):
    campaign = (tmp_path / "campaign").resolve()
    audit = {
        "schema": 1,
        "kind": "arc_agi3_managed_host_child_ledger_audit",
        "ledger_root": str(
            campaign / "host_child_invocations"
        ),
        "authentication_key_sha256": "a" * 64,
        "invocation_count": 1,
        "status_counts": {
            "PENDING": 0,
            "ACTIVE": 0,
            "TERMINAL": 1,
            "CLEAN": 0,
        },
        "startup_recovered_count": 0,
        "startup_recovery": [],
        "records": [{"invocation_id": "sealed"}],
        "all_receipts_authenticated": True,
        "external_absence_proof_required_count": 0,
        "all_children_accounted_for": True,
        "authentication_sha256": "b" * 64,
    }
    assert O._validate_host_child_ledger_audit(
        audit,
        campaign_root=campaign,
        require_quiescent=True,
    ) == audit
    active = {
        **audit,
        "status_counts": {
            "PENDING": 0,
            "ACTIVE": 1,
            "TERMINAL": 0,
            "CLEAN": 0,
        },
    }
    with pytest.raises(
        O.ContiguousOrchestratorError,
        match="incomplete or nonquiescent",
    ):
        O._validate_host_child_ledger_audit(
            active,
            campaign_root=campaign,
            require_quiescent=True,
        )
    unaccounted = {
        **audit,
        "external_absence_proof_required_count": 1,
        "all_children_accounted_for": False,
    }
    with pytest.raises(O.ContiguousOrchestratorError):
        O._validate_host_child_ledger_audit(
            unaccounted,
            campaign_root=campaign,
            require_quiescent=True,
        )


def test_storage_incident_has_reachable_exact_terminal_projection(
    tmp_path,
):
    config = SimpleNamespace(
        campaign_root=(tmp_path / "campaign").resolve(),
        config_sha256="a" * 64,
    )
    incident = {
        "reason_code": "journal_or_storage_exhausted",
        "failed_event_id": "dispatch:ft09",
        "failed_event_kind": "ATTEMPT_RESERVED",
        "failure_stage": "event_commit",
        "error_code": "ENOSPC",
        "storage_snapshot": {},
        "solver_authority": False,
        "wip_authority": False,
        "cost_authority": False,
        "promotion_authority": False,
        "status": "OPERATOR_INCIDENT",
    }
    state = {
        "storage_incident": incident,
        "solved_levels": 41,
        "total_levels": 183,
    }
    events = [{
        "sequence": 99,
        "digest": "b" * 64,
        "kind": "JOURNAL_OR_STORAGE_EXHAUSTED",
        "payload": incident,
    }]
    terminal = O._storage_exhausted_terminal_value(
        config, state=state, journal_events=events
    )
    assert terminal["status"] == (
        "JOURNAL_OR_STORAGE_EXHAUSTED"
    )
    assert terminal["active_primary_attempts"] == []
    assert terminal["active_auxiliary_assignments"] == []
    assert terminal["receipt_sha256"] == O._json_sha256({
        key: value
        for key, value in terminal.items()
        if key != "receipt_sha256"
    })
    forged = {
        **incident,
        "promotion_authority": True,
    }
    with pytest.raises(O.ContiguousOrchestratorError):
        O._storage_exhausted_terminal_value(
            config,
            state={**state, "storage_incident": forged},
            journal_events=[
                {**events[0], "payload": forged},
            ],
        )


def test_main_preflight_failure_is_structured_stdout_without_prose(
    monkeypatch, capfd
):
    def reject(_path):
        raise O.ContiguousOrchestratorError(
            "DO_NOT_PRINT_THIS_PREFLIGHT_SECRET"
        )

    monkeypatch.setattr(
        O, "load_operator_configuration", reject
    )
    assert O.main(["--config", "/absolute/config.json"]) == 2
    captured = capfd.readouterr()
    assert captured.err == ""
    assert "DO_NOT_PRINT" not in captured.out
    value = json.loads(captured.out)
    assert value["status"] == "PREFLIGHT_FAILED"
    assert value["reason_code"] == "operator_preflight_failed"
    assert value["error_class"] == "ContiguousOrchestratorError"


def _operator_canary_fixture(tmp_path: Path):
    campaign = tmp_path / "campaign"
    credential = tmp_path / "credential.json"
    credential.write_text('{"token":"test-only"}\n', encoding="utf-8")
    environment_name = (
        f"ARC_TEST_CANARY_{os.getpid()}_"
        f"{hashlib.sha256(str(tmp_path).encode()).hexdigest()[:12]}"
    ).upper()
    placements = {
        category: (
            environment_name
            if category == "environment"
            else str(tmp_path / "placements" / f"{category}.txt")
        )
        for category in O.Taint.CONTROLLER_CANARY_CATEGORIES
    }
    config = SimpleNamespace(
        campaign_root=campaign,
        credential_source=credential,
        canary_placements=placements,
    )
    planting = O._load_or_create_canary_planting(config)
    terminal_path = campaign / "operator_terminal_blocked.json"
    terminal_value = {
        "schema": 1,
        "kind": "arc_agi3_contiguous_operator_terminal",
        "status": "BLOCKED",
        "terminal_cleanup_intent": str(
            O._operator_terminal_cleanup_intent_path(campaign)
        ),
        "canary_cleanup_required": True,
    }
    return config, planting, terminal_path, terminal_value


_TERMINAL_CLEANUP_FAULT_POINTS = (
    "before_terminal_receipt_durable",
    "after_terminal_receipt_durable",
    *tuple(
        point
        for index in range(
            len(O.Taint.CONTROLLER_CANARY_CATEGORIES) - 1
        )
        for point in (
            f"before_unlink:file:{index}",
            f"after_unlink:file:{index}",
        )
    ),
    "before_unset:environment",
    "after_unset:environment",
    "before_unlink:escrow",
    "after_unlink:escrow",
)


@pytest.mark.parametrize(
    "fault_point", _TERMINAL_CLEANUP_FAULT_POINTS
)
def test_terminal_canary_cleanup_is_exact_and_crash_resumable(
    tmp_path, fault_point
):
    config, planting, terminal_path, terminal_value = (
        _operator_canary_fixture(tmp_path)
    )
    fired = False

    def inject(point):
        nonlocal fired
        if point == fault_point and not fired:
            fired = True
            raise RuntimeError(f"injected:{point}")

    with pytest.raises(RuntimeError, match="injected"):
        O._finalize_operator_terminal(
            campaign_root=config.campaign_root,
            planting=planting,
            terminal_receipt_path=terminal_path,
            terminal_value=terminal_value,
            fault_hook=inject,
        )
    assert fired
    intent_path = O._operator_terminal_cleanup_intent_path(
        config.campaign_root
    )
    if intent_path.exists():
        recovered = O._resume_terminal_canary_cleanup(
            config.campaign_root
        )
    else:
        recovered = O._finalize_operator_terminal(
            campaign_root=config.campaign_root,
            planting=planting,
            terminal_receipt_path=terminal_path,
            terminal_value=terminal_value,
        )
    assert terminal_path.is_file()
    assert recovered["canary_live_values_cleaned"] is True
    assert all(not path.exists() for path in planting.file_paths)
    assert os.environ.get(planting.environment_name) is None
    assert not planting.escrow_path.exists()
    assert O._resume_terminal_canary_cleanup(
        config.campaign_root
    )["terminal_cleanup_receipt_sha256"] == recovered[
        "terminal_cleanup_receipt_sha256"
    ]


def test_terminal_canary_cleanup_rejects_same_bytes_new_inode(
    tmp_path,
):
    assert not hasattr(O, "_cleanup_canary_planting")
    config, planting, terminal_path, terminal_value = (
        _operator_canary_fixture(tmp_path)
    )

    def stop_before_first_unlink(point):
        if point == "before_unlink:file:0":
            raise RuntimeError("intent durable")

    with pytest.raises(RuntimeError, match="intent durable"):
        O._finalize_operator_terminal(
            campaign_root=config.campaign_root,
            planting=planting,
            terminal_receipt_path=terminal_path,
            terminal_value=terminal_value,
            fault_hook=stop_before_first_unlink,
        )
    target = planting.file_paths[0]
    raw = target.read_bytes()
    target.unlink()
    target.write_bytes(raw)
    target.chmod(0o400)
    try:
        with pytest.raises(
            O.ContiguousOrchestratorError,
            match="substituted canary",
        ):
            O._resume_terminal_canary_cleanup(config.campaign_root)
    finally:
        os.environ.pop(planting.environment_name, None)


def test_quiescent_blocked_terminal_ignores_completed_lanes_and_requires_no_work(
):
    blocked = O._quiescent_authenticated_blocked_projection(
        {
            "complete": False,
            "solved_levels": 8,
            "total_levels": 9,
            "pending_scheduler_decision": None,
            "pending_auxiliary_decision": None,
            "lanes": {
                "done": {
                    "reached": 8,
                    "target": 8,
                    "active": None,
                    "blocked": None,
                },
                "stopped": {
                    "reached": 0,
                    "target": 1,
                    "active": None,
                    "blocked": "arena_parent_terminal_before_target",
                },
            },
            "auxiliary_assignments": {},
        },
        journal_head_sequence=17,
        journal_head_digest="a" * 64,
    )
    assert blocked is not None
    assert blocked["status"] == "BLOCKED"
    assert blocked["unresolved_frontiers"] == [{
        "game": "stopped",
        "reached": 0,
        "target": 1,
        "blocker": "arena_parent_terminal_before_target",
    }]

    live = {
        "state": SimpleNamespace(phase="RUNNING")
    }
    assert O._quiescent_authenticated_blocked_projection(
        {
            "complete": False,
            "pending_scheduler_decision": None,
            "pending_auxiliary_decision": None,
            "lanes": {
                "stopped": {
                    "reached": 0,
                    "target": 1,
                    "active": None,
                    "blocked": "arena_parent_terminal_before_target",
                },
            },
            "auxiliary_assignments": {"aux": live},
        },
        journal_head_sequence=18,
        journal_head_digest="b" * 64,
    ) is None


def test_unified_audit_requires_runner_and_promotion_evidence(
    tmp_path, monkeypatch
):
    runner, _, gate, _, _ = RunnerTest.make_runner(
        tmp_path, max_lanes=1
    )
    scheduler_receipt = Scheduler.audit_campaign(runner.root)
    assert scheduler_receipt["verdict"] == "PASS"
    scheduler_path = tmp_path / "scheduler-audit.json"
    scheduler_path.write_bytes(
        Scheduler.canonical_json(scheduler_receipt) + b"\n"
    )
    runner_receipt = Runner.audit_runner_state_read_only(runner.root)

    unified = O.audit_contiguous_campaign_unified(
        campaign_root=runner.root,
        scheduler_audit_receipt_path=scheduler_path,
        runner_state_receipt=runner_receipt,
        promotion_root=gate.root,
    )
    assert unified["status"] == "PASS"
    assert unified["solved_levels"] == 0
    assert unified["verified_promotion_boundaries"] == 0
    assert O.verify_contiguous_campaign_unified_audit(
        unified,
        campaign_root=runner.root,
        scheduler_audit_receipt_path=scheduler_path,
        runner_state_receipt=runner_receipt,
        promotion_root=gate.root,
    ) == unified

    # Even if an upstream policy-verifier result is substituted with a forged
    # scheduler-only promotion claim, the unified layer derives solved status
    # from the full runner reducer and selected production promotion store.
    forged = json.loads(json.dumps(scheduler_receipt))
    forged["summary"]["policy_promoted_levels"] = 1
    forged["summary"]["promotions"] = 1
    forged_body = {
        key: value
        for key, value in forged.items()
        if key != "receipt_sha256"
    }
    forged["receipt_sha256"] = Scheduler.sha256_json(forged_body)
    monkeypatch.setattr(
        O.Scheduler,
        "verify_audit_receipt",
        lambda *_args, **_kwargs: forged,
    )
    rejected = O.audit_contiguous_campaign_unified(
        campaign_root=runner.root,
        scheduler_audit_receipt_path=scheduler_path,
        runner_state_receipt=runner_receipt,
        promotion_root=gate.root,
    )
    assert rejected["status"] == "FAIL"
    assert rejected["solved_levels"] == 0
    assert rejected["verified_promotion_boundaries"] == 0
    assert "disagree" in rejected["findings"][0]


def test_complete_unified_audit_requires_retention_bound_scheduler_pass(
    tmp_path, monkeypatch
):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    promotion_root = tmp_path / "promotions"
    promotion_root.mkdir()
    scheduler_body = {
        "schema": 1,
        "kind": "ARC_AGI3_CONTIGUOUS_SCHEDULER_AUDIT",
        "verdict": "PASS",
        "campaign_root": str(campaign),
        "policy_name": Scheduler.POLICY_NAME,
        "policy_sha256": Scheduler.SCHEDULER_POLICY_SHA256,
        "proposer_policy_sha256": Scheduler.PROPOSER_POLICY_SHA256,
        "journal_events": 23,
        "journal_head_sequence": 23,
        "journal_head_digest": "1" * 64,
        "control_files": {},
        "summary": {
            "journal_prefix": {"events": 23},
            "policy_promoted_levels": 183,
            "total_levels": 183,
            "live_reservation_units": 0,
        },
        "findings": [],
    }
    scheduler_receipt = {
        **scheduler_body,
        "receipt_sha256": Scheduler.sha256_json(scheduler_body),
    }
    scheduler_path = tmp_path / "scheduler.json"
    scheduler_path.write_bytes(
        Scheduler.canonical_json(scheduler_receipt) + b"\n"
    )
    runner_receipt = {
        "status": "PASS",
        "campaign_root": str(campaign),
        "campaign_id": "campaign:test",
        "inventory_sha256": "2" * 64,
        "scheduler_policy_sha256":
            Scheduler.SCHEDULER_POLICY_SHA256,
        "operator_configuration_sha256": "3" * 64,
        "journal_event_count": 23,
        "journal_head_sequence": 23,
        "journal_head_digest": "1" * 64,
        "journal_prefix": {"events": 23},
        "solved_levels": 183,
        "total_levels": 183,
        "complete": True,
        "lane_boundaries": [],
        "live_budget_reservations": [],
        "receipt_sha256": "4" * 64,
    }
    calls = {"ordinary": 0, "pre_retention": 0, "retention": 0}

    monkeypatch.setattr(
        Runner,
        "verify_runner_state_audit",
        lambda *_args, **_kwargs: runner_receipt,
    )

    def forbidden_ordinary(*_args, **_kwargs):
        calls["ordinary"] += 1
        raise AssertionError(
            "complete terminal audit reopened deleted WIP"
        )

    monkeypatch.setattr(
        Scheduler, "verify_audit_receipt", forbidden_ordinary
    )

    def verify_pre(_campaign, _path, *, expected_receipt_sha256):
        calls["pre_retention"] += 1
        assert expected_receipt_sha256 == (
            scheduler_receipt["receipt_sha256"]
        )
        return scheduler_receipt

    monkeypatch.setattr(
        Scheduler,
        "verify_pre_retention_audit_receipt",
        verify_pre,
    )

    def verify_retention(
        _campaign,
        _runner,
        *,
        pre_cleanup_audits,
        **_kwargs,
    ):
        calls["retention"] += 1
        assert pre_cleanup_audits == {
            "promotion_replay":
                promotion_replay_receipt["receipt_sha256"],
            "scheduler": scheduler_receipt["receipt_sha256"]
        }
        return {"status": "PASS", "receipt_sha256": "5" * 64}

    monkeypatch.setattr(
        Runner,
        "audit_terminal_attempt_retention",
        verify_retention,
    )
    monkeypatch.setattr(
        O,
        "_read_only_promotion_records",
        lambda *_args, **_kwargs: ([], 183),
    )
    promotion_replay_receipt = (
        O._terminal_promotion_replay_audit_value(
            campaign_root=campaign,
            promotion_root=promotion_root,
            runner_state_receipt=runner_receipt,
        )
    )
    promotion_replay_path = (
        scheduler_path.parent
        / O.TERMINAL_PROMOTION_REPLAY_AUDIT_NAME
    )
    promotion_replay_path.write_bytes(
        O._canonical_json(promotion_replay_receipt) + b"\n"
    )
    os.chmod(promotion_replay_path, 0o400)

    unified = O.audit_contiguous_campaign_unified(
        campaign_root=campaign,
        scheduler_audit_receipt_path=scheduler_path,
        runner_state_receipt=runner_receipt,
        promotion_root=promotion_root,
    )
    assert unified["status"] == "PASS", unified["findings"]
    assert unified["complete"] is True
    assert unified["terminal_retention_receipt_sha256"] == "5" * 64
    assert (
        unified[
            "pre_retention_promotion_replay_receipt_sha256"
        ]
        == promotion_replay_receipt["receipt_sha256"]
    )
    assert calls == {
        "ordinary": 0,
        "pre_retention": 1,
        "retention": 1,
    }


def test_terminal_scheduler_audit_resumes_partial_retention_without_wip(
    tmp_path, monkeypatch
):
    campaign = tmp_path / "campaign"
    audit_root = campaign / "terminal_audits"
    audit_root.mkdir(parents=True)
    intent = campaign / Runner.TERMINAL_RETENTION_INTENT_NAME
    intent.write_text("{}\n", encoding="utf-8")
    scheduler_receipt = {
        "verdict": "PASS",
        "receipt_sha256": "a" * 64,
    }
    scheduler_path = audit_root / "scheduler.json"
    scheduler_path.write_text(
        json.dumps(scheduler_receipt) + "\n", encoding="utf-8"
    )
    calls = {"pre": 0}

    def verify_pre(root, path, *, expected_receipt_sha256):
        calls["pre"] += 1
        assert root == campaign
        assert path == scheduler_path
        assert expected_receipt_sha256 == "a" * 64
        return scheduler_receipt

    monkeypatch.setattr(
        Scheduler,
        "verify_pre_retention_audit_receipt",
        verify_pre,
    )
    monkeypatch.setattr(
        Scheduler,
        "audit_campaign",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError(
                "partial retention reopened deleted transient WIP"
            )
        ),
    )
    assert O._load_or_create_terminal_scheduler_audit(
        campaign_root=campaign,
        audit_root=audit_root,
    ) == (scheduler_receipt, scheduler_path)
    assert calls == {"pre": 1}

    promotion_root = tmp_path / "promotions"
    promotion_root.mkdir()
    runner_receipt = {
        "complete": True,
        "campaign_id": "campaign:test",
        "receipt_sha256": "b" * 64,
        "journal_head_sequence": 23,
        "journal_head_digest": "c" * 64,
        "solved_levels": 183,
        "total_levels": 183,
        "lane_boundaries": [],
    }
    monkeypatch.setattr(
        O,
        "_read_only_promotion_records",
        lambda *_args, **_kwargs: ([], 183),
    )
    with pytest.raises(
        O.ContiguousOrchestratorError,
        match="lacks its promotion/replay audit",
    ):
        O._load_or_create_terminal_promotion_replay_audit(
            campaign_root=campaign,
            promotion_root=promotion_root,
            audit_root=audit_root,
            runner_state_receipt=runner_receipt,
        )
    promotion_receipt = O._terminal_promotion_replay_audit_value(
        campaign_root=campaign,
        promotion_root=promotion_root,
        runner_state_receipt=runner_receipt,
    )
    promotion_path = (
        audit_root / O.TERMINAL_PROMOTION_REPLAY_AUDIT_NAME
    )
    promotion_path.write_bytes(
        O._canonical_json(promotion_receipt) + b"\n"
    )
    os.chmod(promotion_path, 0o400)
    assert O._load_or_create_terminal_promotion_replay_audit(
        campaign_root=campaign,
        promotion_root=promotion_root,
        audit_root=audit_root,
        runner_state_receipt=runner_receipt,
    ) == (promotion_receipt, promotion_path)

    intent.unlink()
    (campaign / Runner.TERMINAL_RETENTION_EVIDENCE_NAME).mkdir()
    with pytest.raises(
        O.ContiguousOrchestratorError,
        match="without their intent",
    ):
        O._load_or_create_terminal_scheduler_audit(
            campaign_root=campaign,
            audit_root=audit_root,
        )


def _candidate(
    spec: Runner.AttemptSpec,
    *,
    include_data: bool = True,
    candidate_path: list[int] | None = None,
    players_suffix: bytes = b"",
) -> Runner.PromotionCandidate:
    output = Path(spec.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source = output / "source"
    source.mkdir()
    payloads = {
        entry.name: entry.read_bytes()
        for entry in Path(spec.parent_source_path).iterdir()
        if entry.is_file()
    }
    payloads["players.py"] += (
        b"\n# exact next player\n" + players_suffix
    )
    if include_data:
        payloads["policy_data.json"] = b'{"version":1}\n'
        payloads["solver_notes.txt"] = b"reusable public constant\n"
    for name, raw in payloads.items():
        (source / name).write_bytes(raw)
    worker = {
        "schema": 1,
        "kind": "arc_agi3_contiguous_proposer_worker",
        "attempt_id": spec.attempt_id,
        "authoritative": False,
    }
    worker_raw = _json_bytes(worker)
    (output / "worker_outcome.json").write_bytes(worker_raw)
    exported = {
        "worker_outcome.json": _sha(worker_raw),
        **{
            f"source/{name}": _sha(raw)
            for name, raw in sorted(payloads.items())
        },
    }
    manifest = {
        "schema": 1,
        "game": spec.game,
        "target_level": spec.target_level,
        "parent_checkpoint_sha256": spec.parent_checkpoint_sha256,
        "candidate_path": (
            [1] if candidate_path is None else candidate_path
        ),
        "exported_files_sha256": exported,
    }
    manifest_raw = _json_bytes(manifest)
    manifest_path = output / O.CANDIDATE_NAME
    manifest_path.write_bytes(manifest_raw)
    return Runner.PromotionCandidate(
        game=spec.game,
        from_level=spec.target_level - 1,
        to_level=spec.target_level,
        parent_checkpoint_sha256=spec.parent_checkpoint_sha256,
        candidate_manifest_path=str(manifest_path),
        candidate_manifest_sha256=_sha(manifest_raw),
        probe_isolation_mode=(
            RunnerTest.TEST_PROBE_ISOLATION_MODE
        ),
        probe_isolation_evidence_sha256=(
            RunnerTest.TEST_PROBE_ISOLATION_SHA256
        ),
        supervisory_handoff_sha256=None,
        supervisory_native_reproduction_receipt_sha256=None,
    )


class _Replay:
    def __init__(self, root: Path):
        self.root = root

    def replay_from_zero(
        self,
        *,
        spec: Runner.AttemptSpec,
        source_payloads,
    ) -> O.IsolatedReplayEvidence:
        self.root.mkdir(parents=True, exist_ok=True)
        arena = self.root / "arena.jsonl"
        outcome = self.root / "worker_outcome.json"
        stdout = self.root / "stdout.log"
        stderr = self.root / "stderr.log"
        arena.write_bytes(b'{"kind":"replay"}\n')
        outcome.write_bytes(b'{"status":"completed"}\n')
        stdout.write_bytes(b"replay complete\n")
        stderr.write_bytes(b"")
        return O.IsolatedReplayEvidence(
            schema=O.SCHEMA,
            replay_id="replay-1",
            game=spec.game,
            target_level=spec.target_level,
            observed_level=spec.target_level,
            observed_path=(1,),
            exact_path=(1,),
            source_tree_sha256=O._source_tree_sha256(source_payloads),
            replay_image_reference=(
                "gkm/replay@sha256:" + "a" * 64
            ),
            replay_image_digest="sha256:" + "a" * 64,
            container_id="b" * 64,
            launch_attestation_sha256="c" * 64,
            running_observation_sha256="d" * 64,
            arena_transcript_path=str(arena),
            arena_transcript_sha256=_sha(arena.read_bytes()),
            worker_outcome_path=str(outcome),
            worker_outcome_sha256=_sha(outcome.read_bytes()),
            stdout_path=str(stdout),
            stdout_sha256=_sha(stdout.read_bytes()),
            stderr_path=str(stderr),
            stderr_sha256=_sha(stderr.read_bytes()),
            teardown_proof_sha256="e" * 64,
        )


def _attempt_evidence(
    candidate: Runner.PromotionCandidate,
    *,
    attempt_id: str,
    campaign_id: str,
):
    genesis_body = {
        "schema": Runner.JOURNAL_SCHEMA,
        "sequence": 1,
        "event_id": "campaign:genesis",
        "kind": "GENESIS",
        "recorded_at": 1.0,
        "previous_digest": None,
        "payload": {"campaign_id": campaign_id},
    }
    genesis_digest = Runner.DurableAttemptJournal._event_digest(
        genesis_body
    )
    events = [{**genesis_body, "digest": genesis_digest}]
    prior = genesis_digest
    for sequence, kind in enumerate(
        ("ATTEMPT_COLLECTED", "ATTEMPT_TORN_DOWN", "ATTEMPT_RESULT"),
        start=2,
    ):
        payload = {"attempt_id": attempt_id}
        if kind == "ATTEMPT_RESULT":
            payload["candidate"] = asdict(candidate)
        body = {
            "schema": Runner.JOURNAL_SCHEMA,
            "sequence": sequence,
            "event_id": f"event-{sequence}",
            "kind": kind,
            "recorded_at": float(sequence),
            "previous_digest": prior,
            "payload": payload,
        }
        digest = Runner.DurableAttemptJournal._event_digest(body)
        events.append({**body, "digest": digest})
        prior = digest
    return SimpleNamespace(
        collection=SimpleNamespace(
            result=Runner.AttemptResult(
                kind="candidate", candidate=candidate
            )
        ),
        teardown=SimpleNamespace(),
        collected_sequence=2,
        collected_event_sha256=events[1]["digest"],
        teardown_sequence=3,
        teardown_event_sha256=events[2]["digest"],
        result_sequence=4,
        result_event_sha256=events[3]["digest"],
        journal_prefix=tuple(events),
        journal_prefix_sha256=O._json_sha256(events),
        journal_genesis_sha256=events[0]["digest"],
    )


def _install_gate_stubs(
    monkeypatch,
    candidate,
    *,
    attempt_id,
    campaign_id,
):
    evidence = _attempt_evidence(
        candidate,
        attempt_id=attempt_id,
        campaign_id=campaign_id,
    )
    monkeypatch.setattr(
        O, "_load_attempt_evidence", lambda _spec: evidence
    )
    monkeypatch.setattr(
        O, "_exact_path", lambda _game, _path, _level: [1]
    )
    monkeypatch.setattr(
        O, "_path_replay", lambda _game, _level, _path: None
    )
    monkeypatch.setattr(
        O.gkm_arena, "validate", lambda *_args, **_kwargs: True
    )
    monkeypatch.setattr(
        O, "_scan_primary_files", lambda *_args, **_kwargs: None
    )

    def copy_transcripts(
        self,
        *,
        spec,
        manifest_raw,
        replay,
        attempt_evidence,
        destination,
    ):
        destination.mkdir(parents=True, exist_ok=True)
        values = {
            "candidate_manifest.json": manifest_raw,
            "arena_source_replay.jsonl": Path(
                replay.arena_transcript_path
            ).read_bytes(),
            "certification.json": _json_bytes({
                "schema": O.SCHEMA,
                "kind":
                    "arc_agi3_contiguous_replay_certification",
                "game": spec.game,
                "target_level": spec.target_level,
                "attempt_id": spec.attempt_id,
                "candidate_manifest_sha256":
                    candidate.candidate_manifest_sha256,
                "isolated_replay":
                    O._public_replay_evidence(replay),
                "attempt_evidence": {
                    "collected_sequence":
                        attempt_evidence.collected_sequence,
                    "collected_event_sha256":
                        attempt_evidence
                        .collected_event_sha256,
                    "teardown_sequence":
                        attempt_evidence.teardown_sequence,
                    "teardown_event_sha256":
                        attempt_evidence
                        .teardown_event_sha256,
                    "result_sequence":
                        attempt_evidence.result_sequence,
                    "result_event_sha256":
                        attempt_evidence.result_event_sha256,
                    "journal_prefix":
                        list(attempt_evidence.journal_prefix),
                    "journal_prefix_sha256":
                        attempt_evidence.journal_prefix_sha256,
                    "journal_genesis_sha256":
                        attempt_evidence.journal_genesis_sha256,
                },
            }),
        }
        result = {}
        for name, raw in values.items():
            path = destination / name
            path.write_bytes(raw)
            result[f"transcripts/{name}"] = _sha(raw)
        return result

    monkeypatch.setattr(
        O.ProductionPromotionGate,
        "_copy_attempt_transcripts",
        copy_transcripts,
    )


def test_production_promotion_binds_transcript_kind_independent_of_source_name(
    tmp_path,
    monkeypatch,
):
    """Publication passes an explicit evidence kind across every copy."""

    def write(name, raw):
        path = tmp_path / name
        path.write_bytes(raw)
        return path, _sha(raw)

    # Deliberately misleading source names prove that classification does not
    # come from the acquisition filename.
    app_path, _ = write("legacy-generic.log", b'{"app":true}\n')
    arena_path, _ = write("proposer_last.log", b'{"arena":true}\n')
    replay_arena_path, replay_arena_sha = write(
        "replay-generic.log", b'{"replay":true}\n'
    )
    worker_path, worker_sha = write(
        "worker_outcome.json", b'{"ok":true}\n'
    )
    stdout_path, stdout_sha = write("stdout.bin", b"")
    stderr_path, stderr_sha = write("stderr.bin", b"")
    replay_worker_path, replay_worker_sha = write(
        "replay-worker.json", b'{"ok":true}\n'
    )
    replay_stdout_path, replay_stdout_sha = write(
        "replay-stdout.bin", b""
    )
    replay_stderr_path, replay_stderr_sha = write(
        "replay-stderr.bin", b""
    )

    collection = SimpleNamespace(
        host_transcript_path=str(arena_path),
        app_server_transcript_path=str(app_path),
        worker_outcome_sha256=worker_sha,
        container_stdout_path=str(stdout_path),
        container_stdout_sha256=stdout_sha,
        container_stderr_path=str(stderr_path),
        container_stderr_sha256=stderr_sha,
        output_tree_sha256="1" * 64,
        token_usage_receipt_sha256="2" * 64,
        final_thread_binding_sha256="3" * 64,
        final_transcript_chain_sha256="4" * 64,
        bridge_export_receipt_sha256="5" * 64,
        secret_scan_receipt_sha256="6" * 64,
        taint_scan_receipt_sha256="7" * 64,
        app_server_state_tree_sha256="8" * 64,
    )
    evidence = SimpleNamespace(
        collection=collection,
        teardown=SimpleNamespace(),
        collected_sequence=1,
        collected_event_sha256="a" * 64,
        teardown_sequence=2,
        teardown_event_sha256="b" * 64,
        result_sequence=3,
        result_event_sha256="c" * 64,
        journal_prefix=(),
        journal_prefix_sha256="d" * 64,
        journal_genesis_sha256="e" * 64,
    )
    replay = SimpleNamespace(
        arena_transcript_path=str(replay_arena_path),
        arena_transcript_sha256=replay_arena_sha,
        worker_outcome_path=str(replay_worker_path),
        worker_outcome_sha256=replay_worker_sha,
        stdout_path=str(replay_stdout_path),
        stdout_sha256=replay_stdout_sha,
        stderr_path=str(replay_stderr_path),
        stderr_sha256=replay_stderr_sha,
    )
    spec = SimpleNamespace(
        game="zz99",
        target_level=1,
        attempt_id="attempt-1",
        app_server_transcript_path=str(app_path),
        host_transcript_path=str(arena_path),
        output_dir=str(tmp_path),
    )

    monkeypatch.setattr(
        O.Runner, "_backend_collection_to_dict", lambda _value: {}
    )
    monkeypatch.setattr(O, "_app_scan_policy", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(O, "_public_replay_evidence", lambda _value: {})
    monkeypatch.setattr(O, "asdict", lambda _value: {})
    scans = []

    def scan_evidence(path, *, evidence_kind, app_server_policy=None):
        scans.append((Path(path).name, evidence_kind))
        return SimpleNamespace(hits=())

    monkeypatch.setattr(O.Taint, "scan_evidence", scan_evidence)
    gate = object.__new__(O.ProductionPromotionGate)
    gate.secret_sentinels = ()
    records = gate._copy_attempt_transcripts(
        spec=spec,
        manifest_raw=b"{}\n",
        replay=replay,
        attempt_evidence=evidence,
        destination=tmp_path / "retained",
    )

    assert scans[:4] == [
        ("candidate_manifest.json", "candidate_output"),
        ("app_server.jsonl", "app_server_jsonl"),
        ("arena_attempt.jsonl", "backend_jsonl"),
        ("arena_source_replay.jsonl", "backend_jsonl"),
    ]
    assert {
        "transcripts/app_server.jsonl",
        "transcripts/arena_attempt.jsonl",
        "transcripts/arena_source_replay.jsonl",
    } <= set(records)


def test_positive_per_file_growth_does_not_cross_cancel():
    before = {
        "legs.py": b"x" * 100,
        "players.py": b"y" * 10,
        "solve.py": b"z",
    }
    after = {
        "legs.py": b"x" * 10,
        "players.py": b"y" * 20,
        "solve.py": b"z",
    }
    marginal, before_map, after_map = (
        O._marginal_description_growth(before, after)
    )
    assert marginal == 10
    assert before_map["legs.py"] == 100
    assert after_map["legs.py"] == 10
    # Preserve valid pinned-runtime Python while changing every byte at the
    # same per-file length; source-schema validation is part of this helper's
    # contract.
    rewritten = {
        name: b"q" * len(raw)
        for name, raw in before.items()
    }
    assert O._marginal_description_growth(
        before, rewritten
    )[0] == 0


def test_retained_journal_requires_full_genesis_to_result_prefix(tmp_path):
    spec = _attempt_spec(tmp_path)
    candidate = _candidate(spec)
    evidence = _attempt_evidence(
        candidate,
        attempt_id=spec.attempt_id,
        campaign_id=spec.campaign_id,
    )
    event_hashes = {
        "collected": evidence.collected_event_sha256,
        "teardown": evidence.teardown_event_sha256,
        "result": evidence.result_event_sha256,
    }
    retained = {
        "collected_sequence": evidence.collected_sequence,
        "teardown_sequence": evidence.teardown_sequence,
        "result_sequence": evidence.result_sequence,
        "journal_prefix": list(evidence.journal_prefix),
        "journal_prefix_sha256": evidence.journal_prefix_sha256,
        "journal_genesis_sha256": evidence.journal_genesis_sha256,
    }
    O._validate_retained_journal_evidence(
        retained,
        event_hashes=event_hashes,
        attempt_id=spec.attempt_id,
        campaign_id=spec.campaign_id,
    )
    suffix_only = {
        **retained,
        "journal_prefix": retained["journal_prefix"][1:],
    }
    suffix_only["journal_prefix_sha256"] = O._json_sha256(
        suffix_only["journal_prefix"]
    )
    with pytest.raises(
        O.ContiguousOrchestratorError,
        match="hash chain",
    ):
        O._validate_retained_journal_evidence(
            suffix_only,
            event_hashes=event_hashes,
            attempt_id=spec.attempt_id,
            campaign_id=spec.campaign_id,
        )


def test_candidate_inventory_rejects_undeclared_suffix_file(tmp_path):
    spec = _attempt_spec(tmp_path)
    candidate = _candidate(spec)
    output = Path(spec.output_dir)
    (output / "source" / "undeclared.py").write_text(
        "raise AssertionError\n", encoding="utf-8"
    )
    with pytest.raises(
        O.ContiguousOrchestratorError,
        match="undeclared or missing",
    ):
        O._load_candidate(spec, candidate)


@pytest.mark.parametrize(
    "forbidden_import",
    (
        b"from .legs import choose_action\n",
        b"from arc.crack_lab import gkm_arena\n",
        b"import environment_files\n",
        b"import unknown_ambient_solver_package\n",
    ),
)
def test_candidate_import_closure_fails_before_replay_or_promotion(
    tmp_path,
    monkeypatch,
    forbidden_import,
):
    spec = _attempt_spec(tmp_path)
    candidate = _candidate(
        spec, players_suffix=forbidden_import
    )
    _install_gate_stubs(
        monkeypatch,
        candidate,
        attempt_id=spec.attempt_id,
        campaign_id=spec.campaign_id,
    )
    replay_root = tmp_path / "forbidden-source-must-not-replay"
    promotion_root = tmp_path / "promotion-store"
    gate = O.ProductionPromotionGate(
        promotion_root,
        replay_executor=_Replay(replay_root),
    )
    with pytest.raises(
        Runner.PromotionRejected,
        match="candidate source violates the shared source schema",
    ):
        gate.commit(spec=spec, candidate=candidate)
    assert not replay_root.exists()
    assert not (promotion_root / spec.game).exists()


def test_promotion_rejects_actions_after_first_exact_boundary(
    tmp_path,
    monkeypatch,
):
    spec = _attempt_spec(tmp_path)
    candidate = _candidate(spec, candidate_path=[1, 2])
    _install_gate_stubs(
        monkeypatch,
        candidate,
        attempt_id=spec.attempt_id,
        campaign_id=spec.campaign_id,
    )
    replay_root = tmp_path / "must-not-replay"
    gate = O.ProductionPromotionGate(
        tmp_path / "promotion-store",
        replay_executor=_Replay(replay_root),
    )
    with pytest.raises(
        Runner.PromotionRejected,
        match="continues past its first exact boundary",
    ):
        gate.commit(spec=spec, candidate=candidate)
    assert not replay_root.exists()


def test_promotion_never_uses_action7_after_reward_as_validation(
    tmp_path,
    monkeypatch,
):
    spec = _attempt_spec(tmp_path)
    candidate = _candidate(spec, candidate_path=[1, 7])
    _install_gate_stubs(
        monkeypatch,
        candidate,
        attempt_id=spec.attempt_id,
        campaign_id=spec.campaign_id,
    )
    replay_root = tmp_path / "action7-must-not-be-replayed"
    gate = O.ProductionPromotionGate(
        tmp_path / "promotion-store",
        replay_executor=_Replay(replay_root),
    )
    with pytest.raises(
        Runner.PromotionRejected,
        match="continues past its first exact boundary",
    ):
        gate.commit(spec=spec, candidate=candidate)
    assert not replay_root.exists()


def test_schema_v2_commit_hashes_all_declared_source_and_recovers_ack_loss(
    tmp_path,
    monkeypatch,
):
    spec = _attempt_spec(tmp_path)
    candidate = _candidate(spec)
    _install_gate_stubs(
        monkeypatch,
        candidate,
        attempt_id=spec.attempt_id,
        campaign_id=spec.campaign_id,
    )
    gate = O.ProductionPromotionGate(
        tmp_path / "promotion-store",
        replay_executor=_Replay(tmp_path / "replay"),
        fault_at="after_version",
    )
    with pytest.raises(OSError, match="durable version"):
        gate.commit(spec=spec, candidate=candidate)
    gate.fault_at = None
    commit = gate.recover(spec=spec, candidate=candidate)
    assert commit is not None
    version = (
        tmp_path
        / "promotion-store"
        / spec.game
        / O.VERSIONS_NAME
        / commit.source_version_id
    )
    subject = version / f"{spec.game}_legs"
    manifest = json.loads(
        (
            subject
            / "promotion_evidence"
            / "level_01"
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["winning_source_files"] == [
        "legs.py",
        "players.py",
        "policy_data.json",
        "solve.py",
        "solver_notes.txt",
    ]
    receipt = json.loads(
        (subject / O.HOST_RECEIPT_NAME).read_text(encoding="utf-8")
    )
    assert (
        receipt["release_source_tree_sha256"]
        == O.Release._json_sha256({
            name: manifest["promoted_files_sha256"][name]
            for name in manifest["winning_source_files"]
        })
    )
    assert gate.recover(spec=spec, candidate=candidate) == commit


def test_store_rejects_symlink_root_and_dangling_pointer(
    tmp_path,
):
    target = tmp_path / "real"
    target.mkdir(mode=0o700)
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    with pytest.raises(
        O.ContiguousOrchestratorError, match="root cannot be a symlink"
    ):
        O.ProductionPromotionGate(
            alias, replay_executor=_Replay(tmp_path / "replay")
        )
    gate = O.ProductionPromotionGate(
        tmp_path / "store",
        replay_executor=_Replay(tmp_path / "replay-2"),
    )
    game_root = gate._game_root("ar25")
    (game_root / O.POINTER_NAME).symlink_to("missing.json")
    with pytest.raises(
        O.ContiguousOrchestratorError, match="pointer is aliased"
    ):
        gate._current(game_root)


def _fake_replay_runtime(monkeypatch, *, observed_level: int):
    import arc_agi3_arena_rpc as ArenaRpc

    class Thread:
        def join(self, timeout=None):
            return None

        def is_alive(self):
            return False

    class Session:
        def __init__(self, game, *, binding, parent_path, token):
            self.game = game

        def host_result(self):
            return SimpleNamespace(
                levels_completed=observed_level, path=[1]
            )

    class Server:
        def __init__(self, session, socket_path, transcript_path):
            transcript_path.write_bytes(b'{"kind":"arena"}\n')

        def start_thread(self):
            return Thread()

        def wait(self, timeout=None):
            return None

        def shutdown(self):
            return None

    monkeypatch.setattr(
        ArenaRpc,
        "ArenaSessionBinding",
        lambda **values: SimpleNamespace(**values),
    )
    monkeypatch.setattr(ArenaRpc, "ArenaHostSession", Session)
    monkeypatch.setattr(ArenaRpc, "ArenaRpcServer", Server)
    backend = object.__new__(Container.DockerContainerBackend)
    teardown_causes = []

    def build(_self, low_spec):
        return SimpleNamespace(
            image=SimpleNamespace(
                manifest_digest="sha256:" + "a" * 64
            ),
            container_id="b" * 64,
            document_sha256="c" * 64,
        )

    def start(_self, attestation, low_spec):
        solve = (low_spec.parent_input / "solve.py").read_bytes()
        (low_spec.export_root / "worker_outcome.json").write_bytes(
            _json_bytes({
                "schema": "arc-agi3-container-worker/v1",
                "status": "completed",
                "solver_sha256": _sha(solve),
                "elapsed_ns": 1,
                "error": None,
                "authoritative": False,
            })
        )
        return SimpleNamespace(
            attestation=attestation,
            running_observation_sha256="f" * 64,
        )

    def observe(_self, attestation, low_spec, timeout_seconds):
        return SimpleNamespace(
            running=False,
            status="exited",
            exit_code=0,
            oom_killed=False,
            error="",
        )

    def logs(_self, attestation, low_spec):
        return SimpleNamespace(
            stdout="",
            stderr="",
            stdout_sha256=_sha(b""),
            stderr_sha256=_sha(b""),
        )

    def teardown(_self, running, *, cause, graceful_seconds):
        teardown_causes.append(cause)
        return SimpleNamespace(
            container_id="b" * 64,
            cause=cause.value,
            container_inspect_absent=True,
            container_top_absent=True,
            identity_label_query_empty=True,
            no_descendants=True,
            proof_sha256="9" * 64,
        )

    monkeypatch.setattr(
        Container.DockerContainerBackend,
        "build_launch_attestation",
        build,
    )
    monkeypatch.setattr(
        Container.DockerContainerBackend, "start_attested", start
    )
    monkeypatch.setattr(
        Container.DockerContainerBackend,
        "observe_container_state",
        observe,
    )
    monkeypatch.setattr(
        Container.DockerContainerBackend,
        "collect_terminal_logs",
        logs,
    )
    monkeypatch.setattr(
        Container.DockerContainerBackend, "teardown", teardown
    )
    monkeypatch.setattr(
        O, "_exact_path", lambda _game, _path, _level: [1]
    )
    return backend, teardown_causes


def test_docker_replay_preserves_observed_start_hash_and_rejects_overshoot(
    tmp_path,
    monkeypatch,
):
    spec = _attempt_spec(tmp_path)
    payloads = {
        entry.name: entry.read_bytes()
        for entry in Path(spec.parent_source_path).iterdir()
        if entry.is_file()
    }
    backend, causes = _fake_replay_runtime(
        monkeypatch, observed_level=spec.target_level
    )
    executor = O.DockerReplayExecutor(
        backend,
        replay_image_reference=(
            "gkm/replay@sha256:" + "a" * 64
        ),
        evidence_root=tmp_path / "real-replay",
        timeout_seconds=5,
    )
    evidence = executor.replay_from_zero(
        spec=spec, source_payloads=payloads
    )
    assert evidence.running_observation_sha256 == "f" * 64
    assert causes == [Container.TeardownCause.NORMAL_EXIT]

    backend, causes = _fake_replay_runtime(
        monkeypatch, observed_level=spec.target_level + 1
    )
    executor = O.DockerReplayExecutor(
        backend,
        replay_image_reference=(
            "gkm/replay@sha256:" + "a" * 64
        ),
        evidence_root=tmp_path / "overshoot-replay",
        timeout_seconds=5,
    )
    with pytest.raises(
        O.ContiguousOrchestratorError, match="exact boundary"
    ):
        executor.replay_from_zero(
            spec=spec, source_payloads=payloads
        )
    assert causes == [Container.TeardownCause.CONTAINMENT_FAULT]


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_sigkill_partial_staging_is_reconciled_to_quarantine(tmp_path):
    spec = _attempt_spec(tmp_path)
    candidate = _candidate(spec)
    store = tmp_path / "killed-store"
    pid = os.fork()
    if pid == 0:  # pragma: no cover - assertions run in the parent.
        gate = O.ProductionPromotionGate(
            store, replay_executor=_Replay(tmp_path / "child-replay")
        )
        game_root = gate._game_root(spec.game)
        descriptor = gate._lock(game_root)
        version_id = "f" * 32
        try:
            intent = gate._intent_document(
                version_id=version_id,
                spec=spec,
                candidate=candidate,
            )
            intent_path = (
                game_root / O.INTENTS_NAME / f"{version_id}.json"
            )
            O._write_new_json(intent_path, intent)
            os.chmod(intent_path, 0o400)
            stage = game_root / O.STAGING_NAME / version_id
            stage.mkdir()
            (stage / "partial").mkdir()
            (stage / "partial" / "bytes").write_bytes(b"partial")
            O._fsync_directory(stage / "partial")
            O._fsync_directory(stage)
            O._fsync_directory(game_root / O.STAGING_NAME)
            os.kill(os.getpid(), signal.SIGKILL)
        finally:
            gate._unlock(descriptor)
        os._exit(99)
    waited, status = os.waitpid(pid, 0)
    assert waited == pid
    assert os.WIFSIGNALED(status)
    assert os.WTERMSIG(status) == signal.SIGKILL

    gate = O.ProductionPromotionGate(
        store, replay_executor=_Replay(tmp_path / "parent-replay")
    )
    game_root = gate._game_root(spec.game)
    descriptor = gate._lock(game_root)
    try:
        gate._reconcile_locked(game_root)
    finally:
        gate._unlock(descriptor)
    assert not any((game_root / O.STAGING_NAME).iterdir())
    quarantined = game_root / O.QUARANTINE_NAME / (
        "f" * 32 + ".staging"
    )
    assert (quarantined / "partial" / "bytes").read_bytes() == b"partial"
