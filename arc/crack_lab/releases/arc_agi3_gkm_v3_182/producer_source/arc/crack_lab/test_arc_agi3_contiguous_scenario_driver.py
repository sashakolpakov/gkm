from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

import arc_agi3_contiguous_scenario_driver as D


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


def test_s01_s12_driver_emits_only_typed_blocked_receipts(tmp_path):
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
