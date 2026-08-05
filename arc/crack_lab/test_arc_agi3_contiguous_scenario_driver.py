from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path

import pytest

import arc_agi3_contiguous_scenario_driver as D


class _FixtureObserver:
    """Unit-only typed observer; never installed in the production registry."""

    def observe(self, definition, context):
        root = context.output_root / "fixture-evidence" / definition.scenario_id
        root.mkdir(parents=True, mode=0o700)
        evidence = []
        for observation_id in definition.required_observations:
            body = {
                "schema": 1,
                "kind": "unit_fixture_machine_observation",
                "scenario_id": definition.scenario_id,
                "observation_id": observation_id,
                "observed": True,
            }
            raw = D.canonical_json(body)
            path = root / f"{observation_id}.json"
            path.write_bytes(raw)
            path.chmod(0o400)
            evidence.append(D.ScenarioEvidence(
                observation_id=observation_id,
                kind="unit_fixture_machine_observation",
                path=str(path),
                sha256=D.sha256(raw),
                bytes=len(raw),
            ))
        rows = [dataclasses.asdict(item) for item in evidence]
        return D.ProductionObservation(
            schema=1,
            kind=D.OBSERVATION_KIND,
            scenario_id=definition.scenario_id,
            owner=definition.owner,
            status="PASS",
            machine_observed=True,
            required_observations=definition.required_observations,
            evidence=tuple(evidence),
            evidence_sha256=D.sha256(D.canonical_json(rows)),
        )

    def verify(self, definition, observation, evidence):
        assert set(evidence) == set(definition.required_observations)
        for observation_id, raw in evidence.items():
            assert json.loads(raw) == {
                "schema": 1,
                "kind": "unit_fixture_machine_observation",
                "scenario_id": definition.scenario_id,
                "observation_id": observation_id,
                "observed": True,
            }


def _run(tmp_path):
    repository = Path(__file__).resolve().parents[2]
    runtime = tmp_path / "runtime.json"
    runtime.write_bytes(b'{"kind":"test-runtime"}\n')
    runtime.chmod(0o400)
    output = tmp_path / "scenario-run"
    result = D.run(
        repository=repository,
        runtime_manifest_path=runtime,
        runtime_manifest_sha256=hashlib.sha256(
            runtime.read_bytes()
        ).hexdigest(),
        output_root=output,
    )
    return repository, runtime, output, result


def test_missing_production_observers_emit_typed_blocked_receipts(tmp_path):
    repository, _runtime, output, result = _run(tmp_path)
    assert result["status"] == "BLOCKED"
    assert result["launch_authority"] is False
    assert [row["scenario_id"] for row in result[
        "scenario_receipts"
    ]] == [f"S{index:02d}" for index in range(1, 13)]
    assert {row["status"] for row in result[
        "scenario_receipts"
    ]} == {"BLOCKED"}
    verified = D.verify(
        output / "scenario_driver_receipt.json",
        repository=repository,
    )
    assert verified["status"] == "BLOCKED"
    assert verified["launch_authority"] is False


def test_present_observer_is_reverified_but_missing_owners_keep_gate_closed(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        D, "PRODUCTION_OBSERVERS", {"S01": _FixtureObserver()}
    )
    repository, _runtime, output, result = _run(tmp_path)
    assert result["status"] == "BLOCKED"
    assert result["launch_authority"] is False
    assert result["scenario_receipts"][0]["status"] == "PASS"
    assert {
        row["status"] for row in result["scenario_receipts"][1:]
    } == {"BLOCKED"}
    verified = D.verify(
        output / "scenario_driver_receipt.json",
        repository=repository,
    )
    assert verified["status"] == "BLOCKED"
    assert verified["scenario_statuses"][0] == "PASS"


def test_all_typed_observer_contracts_form_exact_pass_aggregate(
    tmp_path, monkeypatch
):
    # This exercises only the generic schema in the current interpreter.  The
    # sealed production verifier starts a fresh interpreter, where the in-tree
    # registry remains empty until genuine S01--S12 observers are implemented.
    monkeypatch.setattr(
        D,
        "PRODUCTION_OBSERVERS",
        {
            definition.scenario_id: _FixtureObserver()
            for definition in D.SCENARIOS
        },
    )
    repository, _runtime, output, result = _run(tmp_path)
    assert result["status"] == "PASS"
    assert result["launch_authority"] is True
    assert result["blockers"] == []
    verified = D.verify(
        output / "scenario_driver_receipt.json",
        repository=repository,
    )
    assert verified["status"] == "PASS"
    assert verified["launch_authority"] is True
    assert verified["scenario_statuses"] == ["PASS"] * 12


def test_observer_cannot_omit_required_machine_evidence(
    tmp_path, monkeypatch
):
    class Incomplete(_FixtureObserver):
        def observe(self, definition, context):
            complete = super().observe(definition, context)
            evidence = complete.evidence[:-1]
            return dataclasses.replace(
                complete,
                evidence=evidence,
                evidence_sha256=D.sha256(D.canonical_json([
                    dataclasses.asdict(item) for item in evidence
                ])),
            )

    monkeypatch.setattr(
        D, "PRODUCTION_OBSERVERS", {"S01": Incomplete()}
    )
    with pytest.raises(
        D.ScenarioDriverError, match="S01 production observation"
    ):
        _run(tmp_path)


def test_pass_evidence_mutation_is_rejected_on_reopen(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        D, "PRODUCTION_OBSERVERS", {"S01": _FixtureObserver()}
    )
    repository, _runtime, output, result = _run(tmp_path)
    first = result["scenario_receipts"][0]
    receipt = json.loads(Path(first["path"]).read_bytes())
    evidence = Path(receipt["observation"]["evidence"][0]["path"])
    evidence.chmod(0o600)
    evidence.write_bytes(evidence.read_bytes() + b" ")
    evidence.chmod(0o400)
    with pytest.raises(
        D.ScenarioDriverError, match="evidence bytes or ownership differ"
    ):
        D.verify(
            output / "scenario_driver_receipt.json",
            repository=repository,
        )


def test_scenario_specific_verifier_cannot_mutate_evidence(
    tmp_path, monkeypatch
):
    class MutatingVerifier(_FixtureObserver):
        def verify(self, definition, observation, evidence):
            super().verify(definition, observation, evidence)
            path = Path(observation.evidence[0].path)
            path.chmod(0o600)
            path.write_bytes(path.read_bytes() + b" ")
            path.chmod(0o400)

    monkeypatch.setattr(
        D, "PRODUCTION_OBSERVERS", {"S01": MutatingVerifier()}
    )
    with pytest.raises(
        D.ScenarioDriverError,
        match="evidence changed during scenario-specific verification",
    ):
        _run(tmp_path)


def test_scenario_receipt_root_rejects_extra_file(tmp_path):
    repository, _runtime, output, _result = _run(tmp_path)
    extra = output / "scenarios" / "extra.json"
    extra.write_bytes(b"{}\n")
    extra.chmod(0o400)
    with pytest.raises(
        D.ScenarioDriverError, match="root inventory differs"
    ):
        D.verify(
            output / "scenario_driver_receipt.json",
            repository=repository,
        )


def test_non_object_aggregate_fails_with_typed_error(tmp_path):
    repository, _runtime, output, _result = _run(tmp_path)
    aggregate = output / "scenario_driver_receipt.json"
    aggregate.chmod(0o600)
    aggregate.write_bytes(b"[]\n")
    aggregate.chmod(0o400)
    with pytest.raises(
        D.ScenarioDriverError,
        match="scenario driver receipt is malformed",
    ):
        D.verify(aggregate, repository=repository)


def test_regular_read_rejects_in_read_directory_entry_replacement(
    tmp_path, monkeypatch
):
    target = (tmp_path / "evidence.json").resolve()
    replacement = target.with_name("replacement.json")
    target.write_bytes(b'{"authority":false}\n')
    replacement.write_bytes(b'{"authority":true}\n')
    target.chmod(0o400)
    replacement.chmod(0o400)
    target_inode = target.stat().st_ino
    original_read = D.os.read
    swapped = False

    def swap_after_read(descriptor, size):
        nonlocal swapped
        raw = original_read(descriptor, size)
        if (
            not swapped
            and raw
            and os.fstat(descriptor).st_ino == target_inode
        ):
            os.replace(replacement, target)
            swapped = True
        return raw

    monkeypatch.setattr(D.os, "read", swap_after_read)
    with pytest.raises(
        D.ScenarioDriverError,
        match="pointer or metadata changed|changed during observation",
    ):
        D._read_regular(target, label="test production evidence")
    assert swapped is True


def test_aggregate_mutation_after_initial_read_is_rejected_at_close(
    tmp_path, monkeypatch
):
    repository, _runtime, output, _result = _run(tmp_path)
    aggregate = output / "scenario_driver_receipt.json"
    original_read = D._read_regular
    mutated = False

    def mutate_after_initial_read(path, **kwargs):
        nonlocal mutated
        raw = original_read(path, **kwargs)
        if Path(path) == aggregate and not mutated:
            aggregate.chmod(0o600)
            aggregate.write_bytes(raw + b" ")
            aggregate.chmod(0o400)
            mutated = True
        return raw

    monkeypatch.setattr(D, "_read_regular", mutate_after_initial_read)
    with pytest.raises(
        D.ScenarioDriverError,
        match="scenario receipt bytes changed during verification",
    ):
        D.verify(aggregate, repository=repository)
    assert mutated is True


def test_s09_schema_requires_absorbing_reward_and_fresh_replay():
    s09 = next(
        definition
        for definition in D.SCENARIOS
        if definition.scenario_id == "S09"
    )
    assert s09.owner == "arc_agi3_contiguous_s09_v2"
    assert "context_specific_action7_exact_or_reconstruct" in (
        s09.required_observations
    )
    assert "reward_boundary_absorbing_no_action7" in (
        s09.required_observations
    )
    assert "fresh_replay_from_sealed_reward" in (
        s09.required_observations
    )


def test_scenario_driver_rejects_forged_pass_and_mutation(tmp_path):
    repository, _runtime, output, _result = _run(tmp_path)
    receipt = output / "scenarios" / "S01.json"
    os.chmod(receipt, 0o600)
    body = json.loads(receipt.read_bytes())
    body["status"] = "PASS"
    body["launch_authority"] = True
    receipt.write_bytes(D.canonical_json(body))
    receipt.chmod(0o400)
    with pytest.raises(D.ScenarioDriverError, match="S01"):
        D.verify(
            output / "scenario_driver_receipt.json",
            repository=repository,
        )


def test_scenario_driver_rejects_hash_consistent_aggregate_pass(
    tmp_path,
):
    repository, _runtime, output, _result = _run(tmp_path)
    aggregate_path = output / "scenario_driver_receipt.json"
    aggregate_path.chmod(0o600)
    aggregate = json.loads(aggregate_path.read_bytes())
    aggregate["status"] = "PASS"
    aggregate["launch_authority"] = True
    aggregate["blockers"] = []
    aggregate_path.write_bytes(D.canonical_json(aggregate))
    aggregate_path.chmod(0o400)
    with pytest.raises(
        D.ScenarioDriverError,
        match="aggregate binding differs",
    ):
        D.verify(aggregate_path, repository=repository)


def test_scenario_driver_rejects_hash_consistent_child_substitution(
    tmp_path,
):
    repository, _runtime, output, _result = _run(tmp_path)
    aggregate_path = output / "scenario_driver_receipt.json"
    first = output / "scenarios" / "S01.json"
    second = output / "scenarios" / "S02.json"
    first.chmod(0o600)
    first.write_bytes(second.read_bytes())
    first.chmod(0o400)
    aggregate_path.chmod(0o600)
    aggregate = json.loads(aggregate_path.read_bytes())
    aggregate["scenario_receipts"][0]["sha256"] = D.sha256(
        first.read_bytes()
    )
    aggregate["scenario_receipts_sha256"] = D.sha256(
        D.canonical_json(aggregate["scenario_receipts"])
    )
    aggregate_path.write_bytes(D.canonical_json(aggregate))
    aggregate_path.chmod(0o400)
    with pytest.raises(D.ScenarioDriverError, match="S01"):
        D.verify(aggregate_path, repository=repository)


def test_scenario_driver_rejects_child_path_substitution(tmp_path):
    repository, _runtime, output, _result = _run(tmp_path)
    aggregate_path = output / "scenario_driver_receipt.json"
    aggregate_path.chmod(0o600)
    aggregate = json.loads(aggregate_path.read_bytes())
    aggregate["scenario_receipts"][0]["path"] = (
        aggregate["scenario_receipts"][1]["path"]
    )
    aggregate["scenario_receipts_sha256"] = D.sha256(
        D.canonical_json(aggregate["scenario_receipts"])
    )
    aggregate_path.write_bytes(D.canonical_json(aggregate))
    aggregate_path.chmod(0o400)
    with pytest.raises(D.ScenarioDriverError, match="S01"):
        D.verify(aggregate_path, repository=repository)


def test_scenario_driver_rejects_noncanonical_aggregate_location(
    tmp_path,
):
    repository, _runtime, output, _result = _run(tmp_path)
    original = output / "scenario_driver_receipt.json"
    substituted = output / "renamed.json"
    original.rename(substituted)
    with pytest.raises(
        D.ScenarioDriverError,
        match="wrong canonical path",
    ):
        D.verify(substituted, repository=repository)
