from __future__ import annotations

import copy
import errno
import fcntl
import hashlib
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import pytest

import codex_campaign_reboot_recovery as Recovery
import codex_campaign_runner as R


def _item() -> dict[str, object]:
    binding = R.Status.validate_frontier_binding({
        "frontier_binding_schema": R.Status.FRONTIER_BINDING_SCHEMA,
        "game": "ar25",
        "reached": 0,
        "target_level": 1,
        "parent_action_count": 0,
        "parent_checkpoint_sha256": R.Status.ZERO_SHA256,
        "parent_source_tree_sha256": R.Status.ZERO_SHA256,
        "frontier_sha256": R.Status._sha256_json({
            "game": "ar25",
            "reached": 0,
            "parent_checkpoint_sha256": R.Status.ZERO_SHA256,
        }),
    })
    return {
        **binding,
        "game": "ar25",
        "target_level": 1,
        "effort": "medium",
        "minutes": 15,
        "retry_complexity_n": 0,
        "dispatch_mode": "fresh_frontier",
        "recommended_auxiliary_parallelism": 0,
        "cost_control_enabled": True,
        "max_campaign_runs": 60,
        "max_campaign_tokens": 32_000_000,
        "required_headroom_percent": 6,
        "warm_wip_available": False,
        "seed_mode": "zero_seed",
        "wip_mode": "exclude",
        "lineage_input_mode": "zero_seed+exclude",
        "argv": [
            "python3", "-u", "arc/crack_lab/gkm_legs.py",
            "--game=ar25", "--max-level=1",
            "--proposer=codex", "--model=gpt-5.6-sol",
            "--minutes=15", "--codex-effort=medium",
            "--codex-allocation-policy=drain",
            "--debrief-policy=never", "--transient-retries=0",
            "--codex-weekly-reserve=25", "--codex-weekly-headroom=6",
            "--codex-max-campaign-runs=60",
            "--codex-max-campaign-tokens=32000000",
            "--seed-mode=zero_seed", "--wip-mode=exclude",
            "--expected-parent-reached=0",
            "--expected-parent-action-count=0",
            f"--expected-parent-checkpoint-sha256={R.Status.ZERO_SHA256}",
            f"--expected-parent-source-tree-sha256={R.Status.ZERO_SHA256}",
            f"--expected-frontier-sha256={binding['frontier_sha256']}",
        ],
    }


def _quarantined_dispatch(
    tmp_path: Path,
    monkeypatch,
    *,
    arm: bool = True,
    lock_schema: str = "hashed_external_v1",
    with_wip: bool = False,
    with_wip_parent: bool = False,
    verified_parent: bool = True,
    durable_wip_capsule: bool = False,
    retry_complexity_n: int = 0,
):
    monkeypatch.setattr(R, "HERE", tmp_path)
    artifact = tmp_path / "agent_solutions" / "ar25_legs"
    artifact.mkdir(parents=True)
    wip = artifact / "wip_context" / "level_01"
    if with_wip_parent:
        wip.parent.mkdir(parents=True)
    if with_wip:
        wip.mkdir(parents=True)
        (wip / "latest.json").write_text(
            '{"attempt":"baseline"}\n', encoding="utf-8"
        )
    scratch = tmp_path / "scratch"
    protected_root = scratch / ".proposer_transcripts"
    lock_root = scratch / ".workspace_locks"
    protected_root.mkdir(parents=True)
    lock_root.mkdir()
    tag = f"arc_agi3_n{retry_complexity_n}_fresh_frontier"
    workspace_name = f"gkm_legs_ws_ar25_{tag}_deadbeef"
    workspace = scratch / workspace_name
    protected = protected_root / workspace_name
    workspace.mkdir()
    protected.mkdir()
    (workspace / "probe.py").write_text("import subprocess\n", encoding="utf-8")
    transcript_name = "codex_turn_20260805T000000000000Z_ar25_L1_propose.jsonl"
    diagnostics_name = (
        "codex_turn_20260805T000000000000Z_ar25_L1_propose.stderr.log"
    )
    transcript = (
        json.dumps({"type": "thread.started", "thread_id": "reboot-thread"})
        + "\n"
        + json.dumps({
            "type": "item.completed",
            "item": {
                "id": "process-query",
                "type": "command_execution",
                "command": "python probe.py",
                "aggregated_output": "",
            },
        })
        + "\n"
        + json.dumps({"type": "turn.completed", "usage": {}})
        + "\n"
    ).encode()
    diagnostics = b""
    (protected / transcript_name).write_bytes(transcript)
    (protected / diagnostics_name).write_bytes(diagnostics)
    ledger = tmp_path / "usage.jsonl"
    ledger.write_bytes(b"")
    item = copy.deepcopy(_item())
    item["retry_complexity_n"] = retry_complexity_n
    if verified_parent:
        item["seed_mode"] = "verified_parent"
        item["lineage_input_mode"] = "verified_parent+exclude"
        item["argv"] = [
            "--seed-mode=verified_parent"
            if argument == "--seed-mode=zero_seed"
            else argument
            for argument in item["argv"]
        ]
    if lock_schema == "in_workspace_v1":
        monkeypatch.setattr(
            R, "_lock_schema", lambda _item: "in_workspace_v1"
        )
        exact_lock = workspace / ".orchestrate.lock"
    elif lock_schema == "hashed_external_v1":
        exact_lock = R.Legs._workspace_lock_path(os.fspath(workspace))
    else:
        raise AssertionError(f"unsupported test lock schema: {lock_schema}")
    exact_lock.write_text("", encoding="utf-8")
    item["argv"].extend([f"--tag={tag}", f"--codex-ledger={ledger}"])
    record = {
        "event": "codex_exec",
        "started_at": "2026-08-05T00:00:00+00:00",
        "thread_id": "reboot-thread",
        "transcript": transcript_name,
        "diagnostics": diagnostics_name,
        "workspace": workspace_name,
        "game": "ar25",
        "target_level": 1,
        "run_label": "ar25:L1:propose",
        "model": "gpt-5.6-sol",
        "reasoning_effort": "medium",
        "minutes_limit": 15,
        "allocation_policy": "drain",
        "reached": 0,
        "parent_action_count": 0,
        **{
            field: item[field]
            for field in R.Status.FRONTIER_BINDING_FIELDS
        },
        "returncode": -9,
        "failure_class": "infrastructure",
        "protected_transcript_status": "sealed",
        "protected_transcript_sha256": hashlib.sha256(transcript).hexdigest(),
        "protected_diagnostics_status": "sealed",
        "protected_diagnostics_sha256": hashlib.sha256(diagnostics).hexdigest(),
        "observed_tokens": 123,
    }
    monkeypatch.setattr(R.Legs, "SCRATCH", os.fspath(scratch))
    monkeypatch.setattr(R, "_checkpoint_reached", lambda _game: 0)
    expected_binding = R.Status.validate_frontier_binding({
        field: item[field]
        for field in (
            *R.Status.FRONTIER_BINDING_FIELDS,
            "game",
            "reached",
            "target_level",
            "parent_action_count",
        )
    })
    monkeypatch.setattr(
        R.Status,
        "exact_frontier_binding",
        lambda *_args, **_kwargs: expected_binding,
    )
    monkeypatch.setattr(R, "_taint_gate", lambda: None)
    ledger_before = R._capture_ledger_prefix(ledger)
    marker = R._arm_dispatch_quarantine(
        item,
        ledger_before=ledger_before,
        wip_before=R._capture_wip_rollback(item),
        canonical_before=R._capture_canonical_rollback(item),
        durable_wip_capsule=durable_wip_capsule,
    )
    R.Guard.append_ledger(record, ledger)
    R._write_dispatch_quarantine_record(marker, {
        "event": "dispatch_unquiesced",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "exception_type": "UnquiescedChildError",
        "reason": "exact child process-tree quiescence was not proven",
        "child_pid": 12345,
        "child_returncode": -9,
        "workspace": workspace.name,
        "protected": protected.name,
        "transcript": transcript_name,
        "workspace_identity": [workspace.stat().st_dev, workspace.stat().st_ino],
        "protected_identity": [protected.stat().st_dev, protected.stat().st_ino],
    })
    dispatch_id = marker.dispatch_id
    marker_path = marker.path
    R._close_dispatch_quarantine(marker)
    armed_boot = Recovery.BootIdentity(
        "darwin_kern_bootsessionuuid",
        "11111111-1111-4111-8111-111111111111",
    )
    current_boot = Recovery.BootIdentity(
        "darwin_kern_bootsessionuuid",
        "22222222-2222-4222-8222-222222222222",
    )
    fixture = {
        "item": item,
        "artifact": artifact,
        "wip": wip,
        "workspace": workspace,
        "protected": protected,
        "exact_lock": exact_lock,
        "ledger": ledger,
        "marker": marker_path,
        "dispatch_id": dispatch_id,
        "armed_boot": armed_boot,
        "current_boot": current_boot,
    }
    if arm:
        outcome = R._arm_post_reboot_recovery(
            item,
            confirm_dispatch_id=dispatch_id,
            boot_identity_provider=lambda: armed_boot,
        )
        if not durable_wip_capsule:
            assert outcome["result"] == (
                "post_reboot_wip_confirmation_required"
            )
            outcome = R._arm_post_reboot_recovery(
                item,
                confirm_dispatch_id=dispatch_id,
                confirm_current_wip_state_sha256=outcome[
                    "current_wip_state_sha256"
                ],
                boot_identity_provider=lambda: armed_boot,
            )
        fixture["nonce"] = outcome["recovery_nonce"]
    return fixture


def _recover(fixture):
    return R._recover_post_reboot_quarantine(
        fixture["item"],
        confirm_dispatch_id=fixture["dispatch_id"],
        confirm_recovery_nonce=fixture["nonce"],
        boot_identity_provider=lambda: fixture["current_boot"],
    )


_AUTO_CONFIRM_WIP = object()


def _arm(fixture, confirm_current_wip_state_sha256=_AUTO_CONFIRM_WIP):
    outcome = R._arm_post_reboot_recovery(
        fixture["item"],
        confirm_dispatch_id=fixture["dispatch_id"],
        confirm_current_wip_state_sha256=(
            None
            if confirm_current_wip_state_sha256 is _AUTO_CONFIRM_WIP
            else confirm_current_wip_state_sha256
        ),
        boot_identity_provider=lambda: fixture["armed_boot"],
    )
    if (
        confirm_current_wip_state_sha256 is _AUTO_CONFIRM_WIP
        and outcome.get("result") == "post_reboot_wip_confirmation_required"
    ):
        return R._arm_post_reboot_recovery(
            fixture["item"],
            confirm_dispatch_id=fixture["dispatch_id"],
            confirm_current_wip_state_sha256=outcome[
                "current_wip_state_sha256"
            ],
            boot_identity_provider=lambda: fixture["armed_boot"],
        )
    return outcome


def _canonical_rows(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def _write_canonical_rows(path: Path, rows) -> None:
    path.write_bytes(b"".join(Recovery.canonical_json_line(row) for row in rows))


def _set_exec_record_fields(fixture, **fields) -> None:
    rows = R.Guard.read_ledger(fixture["ledger"])
    assert len(rows) == 1 and rows[0]["event"] == "codex_exec"
    rows[0].update(fields)
    _write_canonical_rows(fixture["ledger"], rows)


_LEGACY_WRAPPER_INNER_RETURNCODE_SPLIT = {
    "returncode": 0,
    "interrupted": True,
    "surviving_process_group": False,
    "timed_out": False,
    "allocation_expired": False,
    "launch_error": None,
    "postflight_error": None,
    "failure_class": None,
    "failure_detail_class": None,
    "protected_transcript_status": "sealed",
    "protected_transcript_error": None,
    "public_action_protocol_violation": False,
    "terminal_errors": [],
}


def _operator_receipt(rows):
    matches = [
        row for row in rows
        if row.get("event")
        == "codex_post_reboot_operator_recovery_completed"
    ]
    assert len(matches) == 1
    return matches[0]


def _arm_sidecar(fixture) -> Path:
    marker = fixture["marker"]
    return marker.parent / f".{marker.name}.post_reboot_arm"


def _tombstones(fixture) -> tuple[Path, Path]:
    return R._post_reboot_tombstones(
        fixture["dispatch_id"],
        fixture["workspace"],
        fixture["protected"],
    )


def test_post_reboot_recovery_happy_path_is_exact_and_noncounting(
    tmp_path, monkeypatch
):
    fixture = _quarantined_dispatch(tmp_path, monkeypatch)

    result = _recover(fixture)

    assert result["result"] == "tainted_noncounting"
    assert result["retry_complexity_n"] == 0
    assert result["operator_recovery"] == "post_reboot_authenticated"
    assert not fixture["marker"].exists()
    assert not fixture["workspace"].exists()
    assert not fixture["protected"].exists()
    assert not fixture["exact_lock"].exists()
    rows = R.Guard.read_ledger(fixture["ledger"])
    assert [row["event"] for row in rows] == [
        "codex_exec",
        "codex_exec_classification_correction",
        "codex_taint_cleanup_completed",
        "codex_post_reboot_operator_recovery_completed",
        "codex_dispatch_release_authorized",
    ]
    assert rows[1]["retry_increment"] == 0
    assert rows[2]["retry_increment"] == 0
    receipt = rows[3]
    assert receipt["dispatch_id"] == fixture["dispatch_id"]
    assert receipt["retry_increment"] == 0
    assert receipt["current_boot_identity"]["source"] == (
        "darwin_kern_bootsessionuuid"
    )

    sealed = fixture["ledger"].read_bytes()
    repeated = _recover(fixture)
    assert repeated["operator_recovery"] == "post_reboot_already_completed"
    assert fixture["ledger"].read_bytes() == sealed


def test_legacy_retry6_unquiesced_arm_and_recovery_are_unchanged(
    tmp_path, monkeypatch
):
    fixture = _quarantined_dispatch(
        tmp_path,
        monkeypatch,
        arm=False,
        durable_wip_capsule=False,
        retry_complexity_n=6,
    )
    armed = _arm(fixture)
    assert armed["result"] == "post_reboot_recovery_armed"
    fixture["nonce"] = armed["recovery_nonce"]
    parsed = Recovery.parse_dispatch_marker(
        fixture["marker"].read_bytes(), require_recovery_arm=True
    )
    assert parsed.armed["retry_complexity_n"] == 6
    assert parsed.unquiesced["event"] == "dispatch_unquiesced"
    monkeypatch.setattr(
        R.Status,
        "ranked_frontiers",
        lambda frontiers, _turns: [{
            **frontiers[0], "retry_complexity_n": 6
        }],
    )

    outcome = _recover(fixture)
    assert outcome["result"] == "tainted_noncounting"
    assert outcome["retry_complexity_n"] == 6
    assert outcome["operator_recovery"] == "post_reboot_authenticated"
    assert [
        row["event"] for row in R.Guard.read_ledger(fixture["ledger"])
    ] == [
        "codex_exec",
        "codex_exec_classification_correction",
        "codex_taint_cleanup_completed",
        "codex_post_reboot_operator_recovery_completed",
        "codex_dispatch_release_authorized",
    ]


def test_post_reboot_recovery_rejects_same_boot_without_mutation(
    tmp_path, monkeypatch
):
    fixture = _quarantined_dispatch(tmp_path, monkeypatch)
    fixture["current_boot"] = fixture["armed_boot"]

    with pytest.raises(R.CampaignPlanError, match="still running.*armed boot"):
        _recover(fixture)

    assert fixture["marker"].is_file()
    assert fixture["workspace"].is_dir()
    assert [
        row["event"] for row in R.Guard.read_ledger(fixture["ledger"])
    ] == ["codex_exec"]


@pytest.mark.parametrize("mutation", ("malformed", "hardlink"))
def test_post_reboot_recovery_rejects_malformed_or_aliased_marker(
    tmp_path, monkeypatch, mutation
):
    fixture = _quarantined_dispatch(tmp_path, monkeypatch)
    outside = tmp_path / "outside-marker"
    if mutation == "malformed":
        rows = [
            json.loads(line)
            for line in fixture["marker"].read_text(encoding="utf-8").splitlines()
        ]
        rows[0]["unexpected"] = True
        fixture["marker"].write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
    else:
        os.link(fixture["marker"], outside)

    pattern = (
        "not canonically encoded"
        if mutation == "malformed" else "inode custody"
    )
    with pytest.raises(R.CampaignPlanError, match=pattern):
        _recover(fixture)

    assert fixture["marker"].is_file()
    assert fixture["workspace"].is_dir()
    assert [
        row["event"] for row in R.Guard.read_ledger(fixture["ledger"])
    ] == ["codex_exec"]


def test_post_reboot_recovery_rejects_changed_canonical_baseline(
    tmp_path, monkeypatch
):
    fixture = _quarantined_dispatch(tmp_path, monkeypatch)
    (fixture["artifact"] / "changed.py").write_bytes(b"not the baseline\n")

    with pytest.raises(R.CampaignPlanError, match="canonical baseline changed"):
        _recover(fixture)

    assert fixture["marker"].is_file()
    assert fixture["workspace"].is_dir()
    assert [
        row["event"] for row in R.Guard.read_ledger(fixture["ledger"])
    ] == ["codex_exec"]


def test_post_reboot_recovery_cleanup_failure_preserves_marker(
    tmp_path, monkeypatch
):
    fixture = _quarantined_dispatch(tmp_path, monkeypatch)

    def fail_cleanup(*_args, **_kwargs):
        raise R.CampaignPlanError("synthetic exact cleanup failure")

    monkeypatch.setattr(
        R, "_resume_post_reboot_generation_cleanup", fail_cleanup
    )
    with pytest.raises(R.CampaignPlanError, match="synthetic exact cleanup"):
        _recover(fixture)

    assert fixture["marker"].is_file()
    assert fixture["workspace"].is_dir()
    assert fixture["protected"].is_dir()
    assert [
        row["event"] for row in R.Guard.read_ledger(fixture["ledger"])
    ] == ["codex_exec", "codex_exec_classification_correction"]


def test_post_reboot_recovery_is_blocked_by_dispatch_lock(
    tmp_path, monkeypatch
):
    fixture = _quarantined_dispatch(tmp_path, monkeypatch)
    lock = R._acquire_scheduler_dispatch_lock(fixture["item"])
    try:
        with pytest.raises(R.CampaignPlanError, match="another writer"):
            _recover(fixture)
    finally:
        R._release_scheduler_artifact_lock(lock)

    assert fixture["marker"].is_file()
    assert fixture["workspace"].is_dir()
    assert [
        row["event"] for row in R.Guard.read_ledger(fixture["ledger"])
    ] == ["codex_exec"]


def test_post_reboot_recovery_requires_exact_operator_dispatch_id(
    tmp_path, monkeypatch
):
    fixture = _quarantined_dispatch(tmp_path, monkeypatch)

    with pytest.raises(R.CampaignPlanError, match="confirmation does not match"):
        R._recover_post_reboot_quarantine(
            fixture["item"],
            confirm_dispatch_id="f" * 32,
            confirm_recovery_nonce=fixture["nonce"],
            boot_identity_provider=lambda: fixture["current_boot"],
        )

    assert fixture["marker"].is_file()
    assert [
        row["event"] for row in R.Guard.read_ledger(fixture["ledger"])
    ] == ["codex_exec"]


def test_post_reboot_recovery_cli_requires_explicit_nonce(
    tmp_path, monkeypatch
):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps({
            "initial_queue": [_item()],
            "reserve_percent": 25,
            "cost_control_enabled": True,
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(R, "_authoritative_targets", lambda: {"ar25": 8})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "codex_campaign_runner.py",
            "--plan",
            os.fspath(plan_path),
            "--recover-post-reboot-quarantine",
            "ar25",
            "--confirm-dispatch-id",
            "a" * 32,
        ],
    )

    with pytest.raises(R.CampaignPlanError, match="requires.*nonce"):
        R.main()


@pytest.mark.parametrize("action", ("arm", "recover"))
def test_interrupted_generation_cli_routes_explicit_operator_action(
    tmp_path, monkeypatch, capsys, action
):
    item = _item()
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps({"initial_queue": [item], "reserve_percent": 25}),
        encoding="utf-8",
    )
    calls = []
    monkeypatch.setattr(R, "_authoritative_targets", lambda: {"ar25": 8})
    monkeypatch.setattr(
        R, "_project_runner_receipt", lambda _plan, selected, **_kwargs: selected
    )
    monkeypatch.setattr(
        R, "validate_item", lambda selected, *_args, **_kwargs: selected["argv"]
    )
    monkeypatch.setattr(R, "validate_inventory_item", lambda *_args: None)
    monkeypatch.setattr(R, "_checkpoint_reached", lambda _game: 0)

    def arm(selected, *, confirm_dispatch_id):
        calls.append(("arm", selected["game"], confirm_dispatch_id, None))
        return {"result": "interrupted_generation_release_armed"}

    def recover(
        selected, *, confirm_dispatch_id, confirm_recovery_nonce
    ):
        calls.append((
            "recover",
            selected["game"],
            confirm_dispatch_id,
            confirm_recovery_nonce,
        ))
        return {"result": "sandbox_isolated_noncounting"}

    monkeypatch.setattr(R, "_arm_interrupted_generation_release", arm)
    monkeypatch.setattr(R, "_recover_interrupted_generation_release", recover)
    flag = (
        "--arm-interrupted-generation-release"
        if action == "arm"
        else "--recover-interrupted-generation-release"
    )
    argv = [
        "codex_campaign_runner.py",
        "--plan", os.fspath(plan_path),
        flag, "ar25",
        "--confirm-dispatch-id", "a" * 32,
    ]
    if action == "recover":
        argv.extend(["--confirm-recovery-nonce", "b" * 32])
    monkeypatch.setattr(sys, "argv", argv)

    assert R.main() == 0
    assert calls == [(
        action, "ar25", "a" * 32,
        None if action == "arm" else "b" * 32,
    )]
    assert json.loads(capsys.readouterr().out)["outcomes"][0]["result"]


@pytest.mark.parametrize(
    ("boundary", "expected_events"),
    (
        (
            "correction",
            ["codex_exec", "codex_exec_classification_correction"],
        ),
        (
            "cleanup",
            [
                "codex_exec",
                "codex_exec_classification_correction",
                "codex_taint_cleanup_completed",
            ],
        ),
        (
            "operator",
            [
                "codex_exec",
                "codex_exec_classification_correction",
                "codex_taint_cleanup_completed",
                "codex_post_reboot_operator_recovery_completed",
            ],
        ),
    ),
)
def test_post_reboot_recovery_resumes_each_durable_ledger_phase_once(
    tmp_path, monkeypatch, boundary, expected_events
):
    fixture = _quarantined_dispatch(tmp_path, monkeypatch)
    target_event = {
        "correction": "codex_exec_classification_correction",
        "cleanup": "codex_taint_cleanup_completed",
        "operator": "codex_post_reboot_operator_recovery_completed",
    }[boundary]
    original = R._append_recovery_phase_cas

    def append_then_crash(state, record):
        result = original(state, record)
        if record["event"] == target_event:
            raise R.CampaignPlanError(f"synthetic crash after {boundary}")
        return result

    monkeypatch.setattr(R, "_append_recovery_phase_cas", append_then_crash)
    with pytest.raises(R.CampaignPlanError, match=f"after {boundary}"):
        _recover(fixture)
    assert fixture["marker"].is_file()
    assert [
        row["event"] for row in R.Guard.read_ledger(fixture["ledger"])
    ] == expected_events

    monkeypatch.setattr(R, "_append_recovery_phase_cas", original)
    result = _recover(fixture)
    assert result["result"] == "tainted_noncounting"
    assert not fixture["marker"].exists()
    events = [row["event"] for row in R.Guard.read_ledger(fixture["ledger"])]
    assert events == [
        "codex_exec",
        "codex_exec_classification_correction",
        "codex_taint_cleanup_completed",
        "codex_post_reboot_operator_recovery_completed",
        "codex_dispatch_release_authorized",
    ]
    assert len(events) == len(set(events))


def test_post_reboot_recovery_resumes_cleanup_done_before_cleanup_row(
    tmp_path, monkeypatch
):
    fixture = _quarantined_dispatch(tmp_path, monkeypatch)
    original = R._append_recovery_phase_cas

    def crash_before_cleanup_row(state, record):
        if record["event"] == "codex_taint_cleanup_completed":
            raise R.CampaignPlanError("synthetic crash before cleanup row")
        return original(state, record)

    monkeypatch.setattr(
        R, "_append_recovery_phase_cas", crash_before_cleanup_row
    )
    with pytest.raises(R.CampaignPlanError, match="before cleanup row"):
        _recover(fixture)
    assert not fixture["workspace"].exists()
    assert not fixture["protected"].exists()
    assert [
        row["event"] for row in R.Guard.read_ledger(fixture["ledger"])
    ] == ["codex_exec", "codex_exec_classification_correction"]

    monkeypatch.setattr(R, "_append_recovery_phase_cas", original)
    assert _recover(fixture)["result"] == "tainted_noncounting"
    assert [
        row["event"] for row in R.Guard.read_ledger(fixture["ledger"])
    ] == [
        "codex_exec",
        "codex_exec_classification_correction",
        "codex_taint_cleanup_completed",
        "codex_post_reboot_operator_recovery_completed",
        "codex_dispatch_release_authorized",
    ]


def test_post_reboot_recovery_resolves_unlink_before_fsync_ambiguity(
    tmp_path, monkeypatch
):
    fixture = _quarantined_dispatch(tmp_path, monkeypatch)
    original = R._release_dispatch_quarantine

    def unlink_then_crash(marker, item, authority):
        R._validate_recovery_marker_seal(marker)
        record, intent_identity = R._install_dispatch_release_intent(
            marker, authority
        )
        R._ensure_dispatch_release_authority_row(
            item,
            marker.root_fd,
            record,
            intent_identity,
            allow_new_authority_append=True,
        )
        os.unlink(marker.name, dir_fd=marker.root_fd)
        raise R.CampaignPlanError("synthetic crash before marker-root fsync")

    monkeypatch.setattr(R, "_release_dispatch_quarantine", unlink_then_crash)
    with pytest.raises(R.CampaignPlanError, match="marker-root fsync"):
        _recover(fixture)
    assert not fixture["marker"].exists()
    sealed = fixture["ledger"].read_bytes()

    monkeypatch.setattr(R, "_release_dispatch_quarantine", original)
    result = _recover(fixture)
    assert result["operator_recovery"] == "post_reboot_already_completed"
    assert fixture["ledger"].read_bytes() == sealed


def test_operator_retry_retires_pre_authority_release_preparing(
    tmp_path, monkeypatch
):
    fixture = _quarantined_dispatch(tmp_path, monkeypatch)
    preparing = (
        fixture["marker"].parent
        / f".{fixture['marker'].name}.release_preparing"
    )
    intent = (
        fixture["marker"].parent
        / f".{fixture['marker'].name}.release_intent"
    )
    real_replace = R.os.replace
    injected = False

    def fail_release_install(source, target, *args, **kwargs):
        nonlocal injected
        if (
            os.fspath(source) == preparing.name
            and os.fspath(target) == intent.name
        ):
            injected = True
            raise OSError(errno.EIO, "synthetic pre-authority release crash")
        return real_replace(source, target, *args, **kwargs)

    monkeypatch.setattr(R.os, "replace", fail_release_install)
    with pytest.raises(R.CampaignPlanError):
        _recover(fixture)
    assert injected
    assert fixture["marker"].is_file()
    assert preparing.is_file()
    assert not intent.exists()
    assert not any(
        row.get("event") == "codex_dispatch_release_authorized"
        for row in R.Guard.read_ledger(fixture["ledger"])
    )

    monkeypatch.setattr(R.os, "replace", real_replace)
    outcome = _recover(fixture)
    assert outcome["result"] == "tainted_noncounting"
    assert not fixture["marker"].exists()
    assert not preparing.exists()
    assert not intent.exists()
    assert sum(
        row.get("event") == "codex_dispatch_release_authorized"
        for row in R.Guard.read_ledger(fixture["ledger"])
    ) == 1


def test_pre_reboot_arm_is_idempotent_after_reported_fsync_failure(
    tmp_path, monkeypatch
):
    fixture = _quarantined_dispatch(tmp_path, monkeypatch, arm=False)
    marker_root_identity = (
        fixture["marker"].parent.stat().st_dev,
        fixture["marker"].parent.stat().st_ino,
    )
    real_fsync = R.os.fsync
    failed = False
    preflight = R._arm_post_reboot_recovery(
        fixture["item"],
        confirm_dispatch_id=fixture["dispatch_id"],
        boot_identity_provider=lambda: fixture["armed_boot"],
    )
    confirmed_wip = preflight["current_wip_state_sha256"]

    def fsync_then_report_failure(descriptor):
        nonlocal failed
        result = real_fsync(descriptor)
        metadata = os.fstat(descriptor)
        if (
            not failed
            and (metadata.st_dev, metadata.st_ino) == marker_root_identity
        ):
            failed = True
            raise OSError("synthetic reported fsync failure")
        return result

    monkeypatch.setattr(R.os, "fsync", fsync_then_report_failure)
    with pytest.raises(R.CampaignPlanError, match="atomically install"):
        R._arm_post_reboot_recovery(
            fixture["item"],
            confirm_dispatch_id=fixture["dispatch_id"],
            confirm_current_wip_state_sha256=confirmed_wip,
            boot_identity_provider=lambda: fixture["armed_boot"],
        )
    monkeypatch.setattr(R.os, "fsync", real_fsync)

    parsed = Recovery.parse_dispatch_marker(
        fixture["marker"].read_bytes(), require_recovery_arm=True
    )
    outcome = R._arm_post_reboot_recovery(
        fixture["item"],
        confirm_dispatch_id=fixture["dispatch_id"],
        confirm_current_wip_state_sha256=confirmed_wip,
        boot_identity_provider=lambda: fixture["armed_boot"],
    )
    assert outcome["result"] == "post_reboot_recovery_already_armed"
    assert outcome["recovery_nonce"] == parsed.recovery_arm["recovery_nonce"]
    assert len(fixture["marker"].read_text().splitlines()) == 3


def test_pre_reboot_arm_retry_fsyncs_after_pre_persistence_failure(
    tmp_path, monkeypatch
):
    fixture = _quarantined_dispatch(tmp_path, monkeypatch, arm=False)
    marker_root_identity = (
        fixture["marker"].parent.stat().st_dev,
        fixture["marker"].parent.stat().st_ino,
    )
    real_fsync = R.os.fsync
    failed_before = False

    def fail_before_root_fsync(descriptor):
        nonlocal failed_before
        metadata = os.fstat(descriptor)
        if (
            not failed_before
            and (metadata.st_dev, metadata.st_ino) == marker_root_identity
        ):
            failed_before = True
            raise OSError("synthetic pre-persistence root fsync failure")
        return real_fsync(descriptor)

    monkeypatch.setattr(R.os, "fsync", fail_before_root_fsync)
    with pytest.raises(R.CampaignPlanError, match="atomically install"):
        _arm(fixture)
    assert failed_before
    Recovery.parse_dispatch_marker(
        fixture["marker"].read_bytes(), require_recovery_arm=True
    )

    retry_root_fsyncs = 0

    def count_retry_root_fsync(descriptor):
        nonlocal retry_root_fsyncs
        metadata = os.fstat(descriptor)
        if (metadata.st_dev, metadata.st_ino) == marker_root_identity:
            retry_root_fsyncs += 1
        return real_fsync(descriptor)

    monkeypatch.setattr(R.os, "fsync", count_retry_root_fsync)
    outcome = _arm(fixture)
    assert outcome["result"] == "post_reboot_recovery_already_armed"
    assert retry_root_fsyncs == 1


def test_pre_reboot_arm_retry_refsyncs_complete_sidecar(
    tmp_path, monkeypatch
):
    fixture = _quarantined_dispatch(tmp_path, monkeypatch, arm=False)
    real_fsync = R.os.fsync
    real_atomic_arm = R._atomic_recovery_arm_replace
    staged_identity = None
    active = False

    def track_atomic_arm(marker, arm_record):
        nonlocal active
        active = True
        try:
            return real_atomic_arm(marker, arm_record)
        finally:
            active = False

    def fail_before_staged_file_fsync(descriptor):
        nonlocal staged_identity
        metadata = os.fstat(descriptor)
        if (
            active
            and staged_identity is None
            and R.stat.S_ISREG(metadata.st_mode)
        ):
            staged_identity = (metadata.st_dev, metadata.st_ino)
            raise OSError("synthetic pre-persistence sidecar fsync failure")
        return real_fsync(descriptor)

    monkeypatch.setattr(R, "_atomic_recovery_arm_replace", track_atomic_arm)
    monkeypatch.setattr(R.os, "fsync", fail_before_staged_file_fsync)
    with pytest.raises(R.CampaignPlanError, match="durably prepare"):
        _arm(fixture)
    sidecar = _arm_sidecar(fixture)
    assert sidecar.is_file()
    assert staged_identity == (sidecar.stat().st_dev, sidecar.stat().st_ino)

    retry_staged_fsyncs = 0

    def count_retry_staged_fsync(descriptor):
        nonlocal retry_staged_fsyncs
        metadata = os.fstat(descriptor)
        if (metadata.st_dev, metadata.st_ino) == staged_identity:
            retry_staged_fsyncs += 1
        return real_fsync(descriptor)

    monkeypatch.setattr(R, "_atomic_recovery_arm_replace", real_atomic_arm)
    monkeypatch.setattr(R.os, "fsync", count_retry_staged_fsync)
    outcome = _arm(fixture)
    assert outcome["result"] == "post_reboot_recovery_armed"
    assert retry_staged_fsyncs == 1
    assert not sidecar.exists()


@pytest.mark.parametrize("mutation", ("replacement", "hardlink"))
def test_staging_reseal_rejects_inode_replacement_or_alias(
    tmp_path, mutation
):
    root = tmp_path / "staging-custody"
    root.mkdir(mode=0o700)
    staged = root / "stage"
    payload = b"exact staged bytes\n"
    staged.write_bytes(payload)
    os.chmod(staged, 0o600)
    expected_identity = (staged.stat().st_dev, staged.stat().st_ino)
    outside = tmp_path / "outside-staging"
    if mutation == "replacement":
        replacement = root / "replacement"
        replacement.write_bytes(payload)
        os.chmod(replacement, 0o600)
        os.replace(replacement, staged)
    else:
        os.link(staged, outside)
    root_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        with pytest.raises(R.CampaignPlanError, match="unsafe staging custody"):
            R._durably_reseal_staged_file_at(
                root_fd,
                staged.name,
                expected_payload=payload,
                expected_identity=expected_identity,
                label="synthetic staging file",
            )
    finally:
        os.close(root_fd)
    assert staged.read_bytes() == payload
    if mutation == "hardlink":
        assert outside.read_bytes() == payload


def test_pre_reboot_arm_rejects_duplicate_arm_row(tmp_path, monkeypatch):
    fixture = _quarantined_dispatch(tmp_path, monkeypatch)
    rows = fixture["marker"].read_text(encoding="utf-8").splitlines()
    fixture["marker"].write_text(
        "\n".join([*rows, rows[-1]]) + "\n", encoding="utf-8"
    )

    with pytest.raises(R.CampaignPlanError, match="row count"):
        _recover(fixture)


def test_boot_identity_change_is_independent_of_wall_clock():
    armed = Recovery.BootIdentity(
        "linux_proc_boot_id", "11111111-1111-4111-8111-111111111111"
    )
    current = Recovery.BootIdentity(
        "linux_proc_boot_id", "22222222-2222-4222-8222-222222222222"
    )

    assert Recovery.require_changed_boot_identity(
        armed.source, armed.value, current
    ) == current


@pytest.mark.parametrize("boundary", (1, 17, 257))
def test_pre_reboot_arm_recovers_partial_sidecar_writes(
    tmp_path, monkeypatch, boundary
):
    fixture = _quarantined_dispatch(tmp_path, monkeypatch, arm=False)
    prefix = fixture["marker"].read_bytes()
    real_write = R.os.write
    calls = 0

    def partial_then_enospc(descriptor, payload):
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_write(descriptor, payload[:boundary])
        raise OSError(errno.ENOSPC, "synthetic sidecar ENOSPC")

    monkeypatch.setattr(R.os, "write", partial_then_enospc)
    with pytest.raises(R.CampaignPlanError, match="durably prepare"):
        _arm(fixture)
    assert fixture["marker"].read_bytes() == prefix
    assert _arm_sidecar(fixture).is_file()

    monkeypatch.setattr(R.os, "write", real_write)
    outcome = _arm(fixture)
    assert outcome["result"] == "post_reboot_recovery_armed"
    assert not _arm_sidecar(fixture).exists()
    parsed = Recovery.parse_dispatch_marker(
        fixture["marker"].read_bytes(), require_recovery_arm=True
    )
    assert parsed.recovery_arm["armed_marker_identity"] == [
        fixture["marker"].stat().st_dev,
        fixture["marker"].stat().st_ino,
    ]
    assert parsed.recovery_arm["pre_arm_marker_identity"] != (
        parsed.recovery_arm["armed_marker_identity"]
    )


@pytest.mark.parametrize("alias", ("hardlink", "symlink"))
def test_pre_reboot_arm_rejects_aliased_sidecar_without_touching_target(
    tmp_path, monkeypatch, alias
):
    fixture = _quarantined_dispatch(tmp_path, monkeypatch, arm=False)
    prefix = fixture["marker"].read_bytes()
    outside = tmp_path / "outside-arm-sidecar"
    outside.write_bytes(b"outside must survive\n")
    sidecar = _arm_sidecar(fixture)
    if alias == "hardlink":
        os.link(outside, sidecar)
    else:
        os.symlink(outside, sidecar)

    with pytest.raises(R.CampaignPlanError, match="sidecar|inode custody"):
        _arm(fixture)
    assert outside.read_bytes() == b"outside must survive\n"
    assert fixture["marker"].read_bytes() == prefix


def test_pre_reboot_arm_recovers_replace_reported_failure(
    tmp_path, monkeypatch
):
    fixture = _quarantined_dispatch(tmp_path, monkeypatch, arm=False)
    real_replace = R.os.replace
    failed = False

    def replace_then_report_failure(*args, **kwargs):
        nonlocal failed
        result = real_replace(*args, **kwargs)
        if not failed:
            failed = True
            raise OSError("synthetic replace report failure")
        return result

    monkeypatch.setattr(R.os, "replace", replace_then_report_failure)
    with pytest.raises(R.CampaignPlanError, match="atomically install"):
        _arm(fixture)
    monkeypatch.setattr(R.os, "replace", real_replace)

    parsed = Recovery.parse_dispatch_marker(
        fixture["marker"].read_bytes(), require_recovery_arm=True
    )
    outcome = _arm(fixture)
    assert outcome["result"] == "post_reboot_recovery_already_armed"
    assert outcome["recovery_nonce"] == parsed.recovery_arm["recovery_nonce"]


def test_recovery_rejects_canonical_but_malformed_arm_row(
    tmp_path, monkeypatch
):
    fixture = _quarantined_dispatch(tmp_path, monkeypatch)
    rows = _canonical_rows(fixture["marker"])
    rows[2]["unexpected"] = True
    _write_canonical_rows(fixture["marker"], rows)

    with pytest.raises(R.CampaignPlanError, match="arm row.*invalid schema"):
        _recover(fixture)


@pytest.mark.parametrize("mutation", ("duplicate_key", "nan"))
def test_recovery_rejects_non_strict_marker_json(
    tmp_path, monkeypatch, mutation
):
    fixture = _quarantined_dispatch(tmp_path, monkeypatch)
    lines = fixture["marker"].read_bytes().splitlines(keepends=True)
    if mutation == "duplicate_key":
        lines[2] = lines[2][:-2] + b',"event":"post_reboot_recovery_armed"}\n'
        pattern = "duplicate JSON object key"
    else:
        lines[0] = lines[0].replace(b'"pid":', b'"pid":NaN,"ignored_pid":', 1)
        pattern = "non-standard JSON constant"
    fixture["marker"].write_bytes(b"".join(lines))

    with pytest.raises(R.CampaignPlanError, match=pattern):
        _recover(fixture)


def test_recovery_rejects_canonical_root_mode_drift(tmp_path, monkeypatch):
    fixture = _quarantined_dispatch(tmp_path, monkeypatch)
    original_mode = fixture["artifact"].stat().st_mode & 0o777
    os.chmod(fixture["artifact"], original_mode ^ 0o100)

    with pytest.raises(R.CampaignPlanError, match="canonical/WIP metadata"):
        _recover(fixture)


def test_recovery_rejects_wip_ctime_drift_with_restored_mode(
    tmp_path, monkeypatch
):
    fixture = _quarantined_dispatch(
        tmp_path, monkeypatch, with_wip=True
    )
    original_mode = fixture["wip"].stat().st_mode & 0o777
    os.chmod(fixture["wip"], original_mode ^ 0o100)
    os.chmod(fixture["wip"], original_mode)

    with pytest.raises(R.CampaignPlanError, match="canonical/WIP metadata"):
        _recover(fixture)


def test_recovery_arms_against_historical_wip_projection_after_interruption(
    tmp_path, monkeypatch
):
    fixture = _quarantined_dispatch(
        tmp_path,
        monkeypatch,
        arm=False,
        with_wip=True,
        verified_parent=True,
    )
    rows_before = _canonical_rows(fixture["marker"])
    historical_snapshot = rows_before[0]["target_wip_snapshot"]
    assert "target_wip_snapshot" not in fixture["item"]
    assert rows_before[0]["projected_item_sha256"] == hashlib.sha256(
        json.dumps(
            fixture["item"], sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()

    attempt = fixture["wip"] / "interrupted_attempt"
    attempt.mkdir()
    (attempt / "evidence.json").write_text(
        '{"phase":"interrupted"}\n', encoding="utf-8"
    )
    (fixture["wip"] / "latest.json").write_text(
        '{"attempt":"interrupted_attempt"}\n', encoding="utf-8"
    )
    incident_state = R._capture_wip_rollback(fixture["item"])
    assert list(incident_state.baseline_snapshot) != historical_snapshot

    before_preflight = fixture["marker"].read_bytes()
    preflight = _arm(fixture, None)
    assert preflight["result"] == "post_reboot_wip_confirmation_required"
    assert preflight["current_wip_state_sha256"] == (
        R._wip_recovery_state_sha256(incident_state)
    )
    assert fixture["marker"].read_bytes() == before_preflight
    with pytest.raises(R.CampaignPlanError, match="does not match"):
        _arm(fixture, "f" * 64)
    assert fixture["marker"].read_bytes() == before_preflight

    outcome = _arm(fixture, preflight["current_wip_state_sha256"])
    fixture["nonce"] = outcome["recovery_nonce"]
    armed_rows = _canonical_rows(fixture["marker"])
    assert armed_rows[0]["target_wip_snapshot"] == historical_snapshot
    assert armed_rows[2]["wip_state_sha256"] == (
        R._wip_recovery_state_sha256(incident_state)
    )
    assert armed_rows[2]["wip_recovery_authority"] == (
        "operator_confirmed_quarantined_wip_v1"
    )
    assert armed_rows[2]["wip_disposition"] == "discard_latest_pointer"

    assert _recover(fixture)["result"] == "tainted_noncounting"
    assert not (fixture["wip"] / "latest.json").exists()
    assert attempt.is_dir()
    receipt = _operator_receipt(R.Guard.read_ledger(fixture["ledger"]))
    assert receipt["wip_recovery_authority"] == (
        "operator_confirmed_quarantined_wip_v1"
    )
    assert receipt["wip_disposition"] == "discard_latest_pointer"
    assert _recover(fixture)["operator_recovery"] == (
        "post_reboot_already_completed"
    )


def test_legacy_marker_requires_full_wip_confirmation_even_when_snapshot_matches(
    tmp_path, monkeypatch
):
    fixture = _quarantined_dispatch(tmp_path, monkeypatch, arm=False)
    marker_before = fixture["marker"].read_bytes()
    rows = _canonical_rows(fixture["marker"])
    assert list(R._target_wip_snapshot(fixture["item"])) == rows[0][
        "target_wip_snapshot"
    ]

    preflight = _arm(fixture, None)

    assert preflight["result"] == "post_reboot_wip_confirmation_required"
    assert preflight["wip_disposition"] == "confirmed_latest_absent"
    assert fixture["marker"].read_bytes() == marker_before


@pytest.mark.parametrize("operation", ("arm", "recover"))
def test_active_legacy_three_row_recovery_arm_is_rejected(
    tmp_path, monkeypatch, operation
):
    fixture = _quarantined_dispatch(tmp_path, monkeypatch)
    rows = _canonical_rows(fixture["marker"])
    for field in Recovery.RECOVERY_ARM_V2_KEYS - Recovery.RECOVERY_ARM_V1_KEYS:
        rows[2].pop(field)
    _write_canonical_rows(fixture["marker"], rows)
    Recovery.parse_dispatch_marker(
        fixture["marker"].read_bytes(), require_recovery_arm=True
    )
    ledger_before = fixture["ledger"].read_bytes()

    with pytest.raises(
        R.CampaignPlanError,
        match="legacy recovery arm lacks explicit current-WIP authority",
    ):
        if operation == "arm":
            _arm(fixture)
        else:
            _recover(fixture)
    assert fixture["marker"].is_file()
    assert fixture["ledger"].read_bytes() == ledger_before


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("seed_mode", "zero_seed"),
        ("wip_mode", "restore_clean_same_frontier"),
        ("lineage_input_mode", "verified_parent+restore_clean_same_frontier"),
        ("expected_wip_attempt", "untrusted_attempt"),
    ),
)
def test_legacy_wip_confirmation_requires_excluded_unselected_input(
    field, value
):
    item = _item()
    item.update({
        "seed_mode": "verified_parent",
        "wip_mode": "exclude",
        "lineage_input_mode": "verified_parent+exclude",
        "warm_wip_available": True,
        "expected_wip_attempt": None,
    })
    item[field] = value

    with pytest.raises(R.CampaignPlanError, match="requires verified-parent"):
        R._validate_legacy_wip_exclusion_item(item)


def test_recovery_rejects_wip_mutation_after_operator_confirmation(
    tmp_path, monkeypatch
):
    fixture = _quarantined_dispatch(
        tmp_path, monkeypatch, arm=False, with_wip=True
    )
    (fixture["wip"] / "latest.json").write_text(
        '{"attempt":"interrupted"}\n', encoding="utf-8"
    )
    preflight = _arm(fixture, None)
    outcome = _arm(fixture, preflight["current_wip_state_sha256"])
    fixture["nonce"] = outcome["recovery_nonce"]
    (fixture["wip"] / "after_confirmation.txt").write_text(
        "changed\n", encoding="utf-8"
    )

    with pytest.raises(R.CampaignPlanError, match="canonical/WIP metadata"):
        _recover(fixture)


def test_legacy_wip_discard_replays_after_unlink_fsync_crash(
    tmp_path, monkeypatch
):
    fixture = _quarantined_dispatch(
        tmp_path, monkeypatch, arm=False, with_wip=True
    )
    (fixture["wip"] / "latest.json").write_text(
        '{"attempt":"interrupted"}\n', encoding="utf-8"
    )
    preflight = _arm(fixture, None)
    outcome = _arm(fixture, preflight["current_wip_state_sha256"])
    fixture["nonce"] = outcome["recovery_nonce"]
    original = R._discard_confirmed_wip_latest_pointer

    def unlink_then_crash(*args, **kwargs):
        original(*args, **kwargs)
        raise R.CampaignPlanError("synthetic crash after WIP discard fsync")

    monkeypatch.setattr(
        R, "_discard_confirmed_wip_latest_pointer", unlink_then_crash
    )
    with pytest.raises(R.CampaignPlanError, match="discard fsync"):
        _recover(fixture)
    assert not (fixture["wip"] / "latest.json").exists()
    assert fixture["marker"].is_file()

    monkeypatch.setattr(
        R, "_discard_confirmed_wip_latest_pointer", original
    )
    assert _recover(fixture)["result"] == "tainted_noncounting"


def test_legacy_wip_discard_replays_when_cleanup_intent_precedes_callback(
    tmp_path, monkeypatch
):
    fixture = _quarantined_dispatch(
        tmp_path, monkeypatch, arm=False, with_wip=True
    )
    latest = fixture["wip"] / "latest.json"
    latest.write_text('{"attempt":"interrupted"}\n', encoding="utf-8")
    preflight = _arm(fixture, None)
    outcome = _arm(fixture, preflight["current_wip_state_sha256"])
    fixture["nonce"] = outcome["recovery_nonce"]
    original = R._discard_confirmed_wip_latest_pointer

    def crash_before_discard(*_args, **_kwargs):
        raise R.CampaignPlanError("synthetic crash before WIP discard callback")

    monkeypatch.setattr(
        R, "_discard_confirmed_wip_latest_pointer", crash_before_discard
    )
    with pytest.raises(R.CampaignPlanError, match="before WIP discard"):
        _recover(fixture)
    assert latest.is_file()
    assert list(fixture["marker"].parent.glob(".codex_recovery_*.intent"))
    assert [
        row["event"] for row in R.Guard.read_ledger(fixture["ledger"])
    ] == ["codex_exec", "codex_exec_classification_correction"]

    monkeypatch.setattr(
        R, "_discard_confirmed_wip_latest_pointer", original
    )
    assert _recover(fixture)["result"] == "tainted_noncounting"
    assert not latest.exists()


def test_legacy_wip_discard_replays_unlink_before_parent_fsync(
    tmp_path, monkeypatch
):
    fixture = _quarantined_dispatch(
        tmp_path, monkeypatch, arm=False, with_wip=True
    )
    latest = fixture["wip"] / "latest.json"
    latest.write_text('{"attempt":"interrupted"}\n', encoding="utf-8")
    preflight = _arm(fixture, None)
    outcome = _arm(fixture, preflight["current_wip_state_sha256"])
    fixture["nonce"] = outcome["recovery_nonce"]
    original = R._discard_confirmed_wip_latest_pointer

    def unlink_before_fsync(_item, _arm_record, _state):
        descriptor = os.open(
            fixture["wip"],
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.unlink("latest.json", dir_fd=descriptor)
        finally:
            os.close(descriptor)
        raise R.CampaignPlanError("synthetic crash before WIP parent fsync")

    monkeypatch.setattr(
        R, "_discard_confirmed_wip_latest_pointer", unlink_before_fsync
    )
    with pytest.raises(R.CampaignPlanError, match="before WIP parent fsync"):
        _recover(fixture)
    assert not latest.exists()
    wip_stat = fixture["wip"].stat(follow_symlinks=False)
    wip_identity = (wip_stat.st_dev, wip_stat.st_ino)
    real_fsync = R.os.fsync
    parent_fsynced = False

    def record_parent_fsync(descriptor):
        nonlocal parent_fsynced
        metadata = os.fstat(descriptor)
        if (metadata.st_dev, metadata.st_ino) == wip_identity:
            parent_fsynced = True
        return real_fsync(descriptor)

    monkeypatch.setattr(
        R, "_discard_confirmed_wip_latest_pointer", original
    )
    monkeypatch.setattr(R.os, "fsync", record_parent_fsync)
    assert _recover(fixture)["result"] == "tainted_noncounting"
    assert parent_fsynced
    assert not latest.exists()


def test_v2_capsule_restores_replaced_latest_and_removes_new_attempt(
    tmp_path, monkeypatch
):
    fixture = _quarantined_dispatch(
        tmp_path,
        monkeypatch,
        arm=False,
        with_wip=True,
        durable_wip_capsule=True,
    )
    latest = fixture["wip"] / "latest.json"
    baseline_bytes = latest.read_bytes()
    baseline = latest.stat(follow_symlinks=False)
    baseline_state = R._capture_wip_rollback(fixture["item"])
    replacement = fixture["wip"] / "latest.replacement"
    replacement.write_text(
        '{"attempt":"interrupted_attempt"}\n', encoding="utf-8"
    )
    os.chmod(replacement, 0o640)
    os.replace(replacement, latest)
    attempt = fixture["wip"] / "interrupted_attempt"
    attempt.mkdir()
    (attempt / "evidence.json").write_text(
        '{"phase":"interrupted"}\n', encoding="utf-8"
    )

    outcome = _arm(fixture, None)
    fixture["nonce"] = outcome["recovery_nonce"]
    arm = _canonical_rows(fixture["marker"])[2]
    dispatch = _canonical_rows(fixture["marker"])[0]
    assert arm["wip_recovery_authority"] == (
        "dispatch_full_wip_rollback_capsule_v1"
    )
    assert arm["wip_disposition"] == "restore_historical_baseline"
    assert dispatch["wip_restore_logical_state_schema"] == (
        R.WIP_LOGICAL_RESTORE_SCHEMA
    )
    assert arm["restored_wip_logical_state_sha256"] == dispatch[
        "wip_restore_logical_state_sha256"
    ]

    assert _recover(fixture)["result"] == "tainted_noncounting"
    restored = latest.stat(follow_symlinks=False)
    assert latest.read_bytes() == baseline_bytes
    assert stat.S_IMODE(restored.st_mode) == stat.S_IMODE(baseline.st_mode)
    assert restored.st_mtime_ns == baseline.st_mtime_ns
    assert restored.st_uid == baseline.st_uid
    assert restored.st_gid == baseline.st_gid
    assert not attempt.exists()
    assert not fixture["marker"].exists()
    restored_state = R._capture_wip_rollback(fixture["item"])
    assert restored_state.baseline_snapshot[1] != baseline_state.baseline_snapshot[1]
    assert R._wip_recovery_state_sha256(restored_state) != (
        R._wip_recovery_state_sha256(baseline_state)
    )
    assert R._wip_logical_restore_state_sha256(restored_state) == (
        dispatch["wip_restore_logical_state_sha256"]
    )
    assert R._wip_logical_restore_state_sha256(baseline_state) == (
        dispatch["wip_restore_logical_state_sha256"]
    )
    with pytest.raises(
        R.CampaignPlanError, match="logical restore state changed"
    ):
        R._validate_capsule_restored_wip_state(
            restored_state,
            baseline_state,
            baseline_state.baseline_snapshot[1],
        )
    receipt = _operator_receipt(R.Guard.read_ledger(fixture["ledger"]))
    assert receipt["wip_restore_logical_state_schema"] == (
        R.WIP_LOGICAL_RESTORE_SCHEMA
    )
    assert receipt["restored_wip_logical_state_sha256"] == dispatch[
        "wip_restore_logical_state_sha256"
    ]


def test_v2_marker_rejects_historical_inode_digest_as_logical_target(
    tmp_path, monkeypatch
):
    fixture = _quarantined_dispatch(
        tmp_path,
        monkeypatch,
        arm=False,
        with_wip=True,
        durable_wip_capsule=True,
    )
    rows = _canonical_rows(fixture["marker"])
    historical_inode_digest = rows[0]["target_wip_snapshot"][1]
    assert historical_inode_digest != rows[0][
        "wip_restore_logical_state_sha256"
    ]
    rows[0]["wip_restore_logical_state_sha256"] = historical_inode_digest
    _write_canonical_rows(fixture["marker"], rows)

    with pytest.raises(
        R.CampaignPlanError, match="capsule binding is invalid"
    ):
        _arm(fixture, None)


def _legacy_capsule_v1_record(record, state):
    legacy = copy.deepcopy(record)
    legacy["schema"] = R.WIP_ROLLBACK_CAPSULE_SCHEMA_V1
    legacy["state"].pop("absence_custody")
    legacy["state_sha256"] = R._wip_capsule_state_sha256(
        state, R.WIP_ROLLBACK_CAPSULE_SCHEMA_V1
    )
    legacy["restore_logical_state_schema"] = (
        R.WIP_LOGICAL_RESTORE_SCHEMA_V1
    )
    legacy["restore_logical_state_sha256"] = (
        R._wip_logical_restore_state_sha256(
            state, schema=R.WIP_LOGICAL_RESTORE_SCHEMA_V1
        )
    )
    return legacy


def test_capsule_v1_existing_root_roundtrip_remains_byte_semantic_compatible(
    tmp_path, monkeypatch
):
    fixture = _quarantined_dispatch(
        tmp_path,
        monkeypatch,
        arm=False,
        with_wip=True,
        durable_wip_capsule=True,
    )
    state = R._capture_wip_rollback(fixture["item"])
    record = R._wip_rollback_capsule_record(
        fixture["item"], state, "0" * 32
    )
    legacy = _legacy_capsule_v1_record(record, state)

    restored = R._state_from_wip_rollback_capsule(
        legacy, fixture["item"]
    )
    assert restored.absence_custody is None
    assert R._wip_recovery_state_sha256(restored) == (
        R._wip_recovery_state_sha256(state)
    )
    assert R._wip_logical_restore_state_sha256(
        restored, schema=R.WIP_LOGICAL_RESTORE_SCHEMA_V1
    ) == legacy["restore_logical_state_sha256"]


def test_capsule_v1_absent_root_fails_closed_before_mutation(
    tmp_path, monkeypatch
):
    fixture = _quarantined_dispatch(
        tmp_path,
        monkeypatch,
        arm=False,
        with_wip_parent=True,
        durable_wip_capsule=True,
    )
    state = R._capture_wip_rollback(fixture["item"])
    record = R._wip_rollback_capsule_record(
        fixture["item"], state, "0" * 32
    )
    legacy = _legacy_capsule_v1_record(record, state)
    restored = R._state_from_wip_rollback_capsule(
        legacy, fixture["item"]
    )
    fixture["wip"].mkdir()
    sentinel = fixture["wip"] / "sentinel.json"
    sentinel.write_text("{}\n", encoding="utf-8")

    with pytest.raises(R.CampaignPlanError, match="lacks absence custody"):
        R._restore_wip_from_rollback_capsule(
            fixture["item"], restored, legacy
        )
    assert sentinel.read_text(encoding="utf-8") == "{}\n"


def test_capsule_v1_v2_schema_pairing_is_disjoint(tmp_path, monkeypatch):
    fixture = _quarantined_dispatch(
        tmp_path,
        monkeypatch,
        arm=False,
        with_wip=True,
        durable_wip_capsule=True,
    )
    state = R._capture_wip_rollback(fixture["item"])
    record = R._wip_rollback_capsule_record(
        fixture["item"], state, "0" * 32
    )
    record["restore_logical_state_schema"] = (
        R.WIP_LOGICAL_RESTORE_SCHEMA_V1
    )
    record["restore_logical_state_sha256"] = (
        R._wip_logical_restore_state_sha256(
            state, schema=R.WIP_LOGICAL_RESTORE_SCHEMA_V1
        )
    )

    with pytest.raises(R.CampaignPlanError, match="logical restore seal"):
        R._state_from_wip_rollback_capsule(record, fixture["item"])


@pytest.mark.parametrize("attack", ("symlink", "hardlink"))
def test_v2_capsule_rejects_aliased_latest_attack(
    tmp_path, monkeypatch, attack
):
    fixture = _quarantined_dispatch(
        tmp_path,
        monkeypatch,
        arm=False,
        with_wip=True,
        durable_wip_capsule=True,
    )
    latest = fixture["wip"] / "latest.json"
    latest.unlink()
    outside = tmp_path / "outside-latest.json"
    outside.write_text('{"attempt":"outside"}\n', encoding="utf-8")
    if attack == "symlink":
        latest.symlink_to(outside)
    else:
        os.link(outside, latest)

    with pytest.raises(
        R.CampaignPlanError, match="target WIP inventory contains an unsafe file"
    ):
        _arm(fixture, None)
    assert fixture["marker"].is_file()
    assert outside.read_text(encoding="utf-8") == '{"attempt":"outside"}\n'


def test_v2_capsule_restore_replays_after_restore_fsync_crash(
    tmp_path, monkeypatch
):
    fixture = _quarantined_dispatch(
        tmp_path,
        monkeypatch,
        arm=False,
        with_wip=True,
        durable_wip_capsule=True,
    )
    latest = fixture["wip"] / "latest.json"
    baseline_bytes = latest.read_bytes()
    latest.write_text('{"attempt":"interrupted"}\n', encoding="utf-8")
    attempt = fixture["wip"] / "interrupted_attempt"
    attempt.mkdir()
    (attempt / "evidence.json").write_text("{}\n", encoding="utf-8")
    outcome = _arm(fixture, None)
    fixture["nonce"] = outcome["recovery_nonce"]
    original = R._restore_wip_from_rollback_capsule

    def restore_then_crash(*args, **kwargs):
        original(*args, **kwargs)
        raise R.CampaignPlanError("synthetic crash after WIP restore fsync")

    monkeypatch.setattr(
        R, "_restore_wip_from_rollback_capsule", restore_then_crash
    )
    with pytest.raises(R.CampaignPlanError, match="restore fsync"):
        _recover(fixture)
    assert latest.read_bytes() == baseline_bytes
    assert not attempt.exists()
    assert fixture["marker"].is_file()

    monkeypatch.setattr(R, "_restore_wip_from_rollback_capsule", original)
    assert _recover(fixture)["result"] == "tainted_noncounting"
    assert latest.read_bytes() == baseline_bytes


@pytest.mark.parametrize(
    "boundary", ("nested_file_parent", "nested_rmdir_parent", "top_rmdir_parent")
)
def test_v2_nested_namespace_fsync_crash_replays_before_cleanup_commit(
    tmp_path, monkeypatch, boundary
):
    fixture = _quarantined_dispatch(
        tmp_path,
        monkeypatch,
        arm=False,
        with_wip=True,
        durable_wip_capsule=True,
    )
    attempt = fixture["wip"] / "interrupted_attempt"
    nested = attempt / "nested"
    nested.mkdir(parents=True)
    evidence = nested / "evidence.json"
    evidence.write_text("{}\n", encoding="utf-8")
    outcome = _arm(fixture, None)
    fixture["nonce"] = outcome["recovery_nonce"]
    target = {
        "nested_file_parent": nested,
        "nested_rmdir_parent": attempt,
        "top_rmdir_parent": fixture["wip"],
    }[boundary]
    target_stat = target.stat(follow_symlinks=False)
    target_identity = (target_stat.st_dev, target_stat.st_ino)
    real_fsync = R.os.fsync
    injected = False

    def fsync_then_crash(descriptor):
        nonlocal injected
        result = real_fsync(descriptor)
        metadata = os.fstat(descriptor)
        if not injected and (metadata.st_dev, metadata.st_ino) == target_identity:
            injected = True
            raise OSError(errno.EIO, f"synthetic {boundary} fsync crash")
        return result

    monkeypatch.setattr(R.os, "fsync", fsync_then_crash)
    with pytest.raises(R.CampaignPlanError, match="capsule restore failed"):
        _recover(fixture)
    assert injected
    assert not evidence.exists()
    if boundary == "nested_file_parent":
        assert nested.is_dir()
    elif boundary == "nested_rmdir_parent":
        assert attempt.is_dir()
        assert not nested.exists()
    else:
        assert not attempt.exists()
    assert fixture["marker"].is_file()
    assert [
        row["event"] for row in R.Guard.read_ledger(fixture["ledger"])
    ] == ["codex_exec", "codex_exec_classification_correction"]

    monkeypatch.setattr(R.os, "fsync", real_fsync)
    assert _recover(fixture)["result"] == "tainted_noncounting"
    assert not attempt.exists()
    assert [
        row["event"] for row in R.Guard.read_ledger(fixture["ledger"])
    ] == [
        "codex_exec",
        "codex_exec_classification_correction",
        "codex_taint_cleanup_completed",
        "codex_post_reboot_operator_recovery_completed",
        "codex_dispatch_release_authorized",
    ]
    assert not fixture["marker"].exists()


def test_v2_absent_baseline_rmdir_crash_refsyncs_exact_parent(
    tmp_path, monkeypatch
):
    fixture = _quarantined_dispatch(
        tmp_path,
        monkeypatch,
        arm=False,
        with_wip_parent=True,
        durable_wip_capsule=True,
    )
    parent_before = fixture["wip"].parent.stat(follow_symlinks=False)
    fixture["wip"].mkdir(parents=True)
    (fixture["wip"] / "latest.json").write_text(
        '{"attempt":"interrupted"}\n', encoding="utf-8"
    )
    os.utime(
        fixture["wip"].parent,
        ns=(parent_before.st_atime_ns, parent_before.st_mtime_ns),
    )
    outcome = _arm(fixture, None)
    fixture["nonce"] = outcome["recovery_nonce"]
    expected_logical = _canonical_rows(fixture["marker"])[0][
        "wip_restore_logical_state_sha256"
    ]
    parent = fixture["wip"].parent
    parent_stat = parent.stat(follow_symlinks=False)
    parent_identity = (parent_stat.st_dev, parent_stat.st_ino)
    real_fsync = R.os.fsync
    injected = False

    def crash_before_parent_fsync(descriptor):
        nonlocal injected
        metadata = os.fstat(descriptor)
        if not injected and (metadata.st_dev, metadata.st_ino) == parent_identity:
            injected = True
            raise OSError(errno.EIO, "synthetic rmdir-before-parent-fsync crash")
        return real_fsync(descriptor)

    monkeypatch.setattr(R.os, "fsync", crash_before_parent_fsync)
    with pytest.raises(R.CampaignPlanError, match="capsule restore failed"):
        _recover(fixture)
    assert injected
    assert not fixture["wip"].exists()
    assert list(fixture["marker"].parent.glob(".codex_recovery_*.intent"))

    parent_fsynced = False

    def record_parent_fsync(descriptor):
        nonlocal parent_fsynced
        metadata = os.fstat(descriptor)
        if (metadata.st_dev, metadata.st_ino) == parent_identity:
            parent_fsynced = True
        return real_fsync(descriptor)

    monkeypatch.setattr(R.os, "fsync", record_parent_fsync)
    assert _recover(fixture)["result"] == "tainted_noncounting"
    assert parent_fsynced
    assert not fixture["wip"].exists()
    parent_after = parent.stat(follow_symlinks=False)
    assert (parent_after.st_dev, parent_after.st_ino) == parent_identity
    assert parent_after.st_mode == parent_before.st_mode
    assert parent_after.st_uid == parent_before.st_uid
    assert parent_after.st_gid == parent_before.st_gid
    assert parent_after.st_mtime_ns == parent_before.st_mtime_ns
    assert R._wip_logical_restore_state_sha256(
        R._capture_wip_rollback(fixture["item"])
    ) == expected_logical


def test_v2_already_absent_baseline_fsyncs_exact_existing_parent(
    tmp_path, monkeypatch
):
    fixture = _quarantined_dispatch(
        tmp_path,
        monkeypatch,
        arm=False,
        with_wip_parent=True,
        durable_wip_capsule=True,
    )
    parent = fixture["wip"].parent
    outcome = _arm(fixture, None)
    fixture["nonce"] = outcome["recovery_nonce"]
    parent_stat = parent.stat(follow_symlinks=False)
    parent_identity = (parent_stat.st_dev, parent_stat.st_ino)
    real_fsync = R.os.fsync
    parent_fsynced = False

    def record_parent_fsync(descriptor):
        nonlocal parent_fsynced
        metadata = os.fstat(descriptor)
        if (metadata.st_dev, metadata.st_ino) == parent_identity:
            parent_fsynced = True
        return real_fsync(descriptor)

    monkeypatch.setattr(R.os, "fsync", record_parent_fsync)
    assert _recover(fixture)["result"] == "tainted_noncounting"
    assert parent_fsynced
    assert not fixture["wip"].exists()


def test_v2_natural_absent_root_parent_mtime_transition_arms_and_restores(
    tmp_path, monkeypatch
):
    fixture = _quarantined_dispatch(
        tmp_path,
        monkeypatch,
        arm=False,
        with_wip_parent=True,
        durable_wip_capsule=True,
    )
    parent = fixture["wip"].parent
    baseline_parent = parent.stat(follow_symlinks=False)
    fixture["wip"].mkdir()
    (fixture["wip"] / "latest.json").write_text(
        '{"attempt":"interrupted"}\n', encoding="utf-8"
    )
    assert parent.stat(follow_symlinks=False).st_mtime_ns != (
        baseline_parent.st_mtime_ns
    )

    outcome = _arm(fixture, None)
    fixture["nonce"] = outcome["recovery_nonce"]
    expected_logical = _canonical_rows(fixture["marker"])[0][
        "wip_restore_logical_state_sha256"
    ]
    assert _recover(fixture)["result"] == "tainted_noncounting"

    restored_parent = parent.stat(follow_symlinks=False)
    assert restored_parent.st_mtime_ns == baseline_parent.st_mtime_ns
    assert not fixture["wip"].exists()
    assert R._wip_logical_restore_state_sha256(
        R._capture_wip_rollback(fixture["item"])
    ) == expected_logical


def test_v2_absent_wrapper_is_rejected_before_dispatch_admission(
    tmp_path, monkeypatch
):
    with pytest.raises(
        R.CampaignPlanError, match="preexisting WIP context parent"
    ):
        _quarantined_dispatch(
            tmp_path,
            monkeypatch,
            arm=False,
            durable_wip_capsule=True,
        )
    assert not list(tmp_path.rglob(".campaign_quarantine/*.jsonl"))


def test_v2_release_replays_after_capsule_fsync_before_marker_unlink(
    tmp_path, monkeypatch
):
    fixture = _quarantined_dispatch(
        tmp_path,
        monkeypatch,
        arm=False,
        with_wip=True,
        durable_wip_capsule=True,
    )
    latest = fixture["wip"] / "latest.json"
    baseline_bytes = latest.read_bytes()
    replacement = fixture["wip"] / "latest.replacement"
    replacement.write_text('{"attempt":"interrupted"}\n', encoding="utf-8")
    os.replace(replacement, latest)
    attempt = fixture["wip"] / "interrupted_attempt"
    attempt.mkdir()
    (attempt / "evidence.json").write_text("{}\n", encoding="utf-8")
    outcome = _arm(fixture, None)
    fixture["nonce"] = outcome["recovery_nonce"]
    armed = _canonical_rows(fixture["marker"])[0]
    capsule = fixture["marker"].parent / armed[
        "wip_rollback_capsule_name"
    ]
    assert capsule.is_file()
    original = R._release_dispatch_quarantine

    def retire_capsule_then_crash(marker, item, authority):
        record, intent_identity = R._install_dispatch_release_intent(
            marker, authority
        )
        R._ensure_dispatch_release_authority_row(
            item,
            marker.root_fd,
            record,
            intent_identity,
            allow_new_authority_append=True,
        )
        metadata = os.stat(
            marker.capsule_name,
            dir_fd=marker.root_fd,
            follow_symlinks=False,
        )
        assert (metadata.st_dev, metadata.st_ino) == marker.capsule_identity
        os.unlink(marker.capsule_name, dir_fd=marker.root_fd)
        os.fsync(marker.root_fd)
        raise R.CampaignPlanError(
            "synthetic crash after capsule fsync before marker unlink"
        )

    monkeypatch.setattr(
        R, "_release_dispatch_quarantine", retire_capsule_then_crash
    )
    with pytest.raises(R.CampaignPlanError, match="capsule fsync"):
        _recover(fixture)
    assert fixture["marker"].is_file()
    assert not capsule.exists()
    assert latest.read_bytes() == baseline_bytes
    assert not attempt.exists()
    sealed = fixture["ledger"].read_bytes()

    monkeypatch.setattr(R, "_release_dispatch_quarantine", original)
    outcome = _recover(fixture)
    assert outcome["operator_recovery"] == "post_reboot_already_completed"
    assert not fixture["marker"].exists()
    assert fixture["ledger"].read_bytes() == sealed


def test_v2_missing_capsule_before_operator_receipt_fails_closed(
    tmp_path, monkeypatch
):
    fixture = _quarantined_dispatch(
        tmp_path,
        monkeypatch,
        with_wip=True,
        durable_wip_capsule=True,
    )
    armed = _canonical_rows(fixture["marker"])[0]
    capsule = fixture["marker"].parent / armed[
        "wip_rollback_capsule_name"
    ]
    capsule.unlink()
    R._fsync_directory(capsule.parent)
    ledger_before = fixture["ledger"].read_bytes()

    with pytest.raises(
        R.CampaignPlanError, match="disappeared before operator completion"
    ):
        _recover(fixture)
    assert fixture["marker"].is_file()
    assert fixture["ledger"].read_bytes() == ledger_before


@pytest.mark.parametrize(
    "mutation", ("marker_snapshot", "marker_hash", "immutable_item")
)
def test_historical_recovery_projection_rejects_mutation(
    tmp_path, monkeypatch, mutation
):
    fixture = _quarantined_dispatch(
        tmp_path, monkeypatch, arm=False, with_wip=True
    )
    attempt = fixture["wip"] / "interrupted_attempt"
    attempt.mkdir()
    (attempt / "evidence.json").write_text("{}\n", encoding="utf-8")
    if mutation == "marker_snapshot":
        rows = _canonical_rows(fixture["marker"])
        rows[0]["target_wip_snapshot"][0] = os.fspath(
            tmp_path / "wrong_wip_level"
        )
        _write_canonical_rows(fixture["marker"], rows)
        pattern = "does not bind the projected plan item"
    elif mutation == "marker_hash":
        rows = _canonical_rows(fixture["marker"])
        rows[0]["projected_item_sha256"] = "f" * 64
        _write_canonical_rows(fixture["marker"], rows)
        pattern = "historical recovery projection"
    else:
        fixture["item"]["retry_complexity_n"] = 1
        pattern = "historical recovery projection"

    with pytest.raises(R.CampaignPlanError, match=pattern):
        _arm(fixture)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("level_mode", 0o700),
        ("level_uid", 123456),
        ("level_gid", 123456),
        ("level_xattrs", (("user.synthetic", b"value"),)),
        ("level_ctime_ns", 1),
    ),
)
def test_wip_recovery_hash_seals_authoritative_metadata(
    tmp_path, monkeypatch, field, value
):
    fixture = _quarantined_dispatch(
        tmp_path, monkeypatch, arm=False, with_wip=True
    )
    state = R._capture_wip_rollback(fixture["item"])
    changed = replace(state, **{field: value})
    assert R._wip_recovery_state_sha256(changed) != (
        R._wip_recovery_state_sha256(state)
    )


def test_wip_recovery_hash_excludes_read_updated_file_atime(
    tmp_path, monkeypatch
):
    fixture = _quarantined_dispatch(
        tmp_path, monkeypatch, arm=False, with_wip=True
    )
    state = R._capture_wip_rollback(fixture["item"])
    entries = dict(state.entries)
    latest = entries["latest.json"]
    entries["latest.json"] = (*latest[:-1], latest[-1] + 1)
    changed = replace(state, entries=entries)
    assert R._wip_recovery_state_sha256(changed) == (
        R._wip_recovery_state_sha256(state)
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("child_returncode", -15),
        ("exception_type", "SyntheticWrongException"),
    ),
)
def test_pre_reboot_arm_rejects_exec_marker_binding_mutation(
    tmp_path, monkeypatch, field, value
):
    fixture = _quarantined_dispatch(tmp_path, monkeypatch, arm=False)
    rows = _canonical_rows(fixture["marker"])
    rows[1][field] = value
    _write_canonical_rows(fixture["marker"], rows)

    with pytest.raises(R.CampaignPlanError, match="exact ledger generation"):
        _arm(fixture)


def test_legacy_recovery_binds_wrapper_kill_to_interrupted_inner_clean_exit(
    tmp_path, monkeypatch
):
    fixture = _quarantined_dispatch(tmp_path, monkeypatch, arm=False)
    _set_exec_record_fields(
        fixture,
        **_LEGACY_WRAPPER_INNER_RETURNCODE_SPLIT,
    )

    armed = _arm(fixture)
    assert armed["result"] == "post_reboot_recovery_armed"
    fixture["nonce"] = armed["recovery_nonce"]
    outcome = _recover(fixture)

    assert outcome["result"] == "tainted_noncounting"
    assert outcome["operator_recovery"] == "post_reboot_authenticated"
    assert [
        row["event"] for row in R.Guard.read_ledger(fixture["ledger"])
    ] == [
        "codex_exec",
        "codex_exec_classification_correction",
        "codex_taint_cleanup_completed",
        "codex_post_reboot_operator_recovery_completed",
        "codex_dispatch_release_authorized",
    ]


@pytest.mark.parametrize(
    ("record_fields", "wrapper_returncode"),
    (
        (
            {
                **_LEGACY_WRAPPER_INNER_RETURNCODE_SPLIT,
                "returncode": 1,
            },
            -9,
        ),
        (
            {
                **_LEGACY_WRAPPER_INNER_RETURNCODE_SPLIT,
                "interrupted": False,
            },
            -9,
        ),
        (
            {
                **_LEGACY_WRAPPER_INNER_RETURNCODE_SPLIT,
                "surviving_process_group": True,
            },
            -9,
        ),
        (
            _LEGACY_WRAPPER_INNER_RETURNCODE_SPLIT,
            -15,
        ),
    ),
    ids=(
        "nonzero-inner-exit",
        "not-interrupted",
        "surviving-inner-group",
        "wrapper-not-watchdog-killed",
    ),
)
def test_legacy_returncode_split_rejects_incomplete_binding_shape(
    tmp_path, monkeypatch, record_fields, wrapper_returncode
):
    fixture = _quarantined_dispatch(tmp_path, monkeypatch, arm=False)
    _set_exec_record_fields(fixture, **record_fields)
    if wrapper_returncode != -9:
        rows = _canonical_rows(fixture["marker"])
        rows[1]["child_returncode"] = wrapper_returncode
        _write_canonical_rows(fixture["marker"], rows)

    with pytest.raises(R.CampaignPlanError, match="exact ledger generation"):
        _arm(fixture)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("timed_out", True),
        ("allocation_expired", True),
        ("launch_error", "OSError"),
        ("postflight_error", "rate-limit postflight failed"),
        ("failure_class", "infrastructure"),
        ("failure_detail_class", "known_transient"),
        ("protected_transcript_status", "unavailable"),
        ("protected_transcript_error", "OSError"),
        ("public_action_protocol_violation", True),
        ("terminal_errors", ["synthetic terminal error"]),
    ),
)
def test_legacy_returncode_split_rejects_nonclean_inner_terminal_shape(
    tmp_path, monkeypatch, field, value
):
    fixture = _quarantined_dispatch(tmp_path, monkeypatch, arm=False)
    _set_exec_record_fields(
        fixture,
        **{
            **_LEGACY_WRAPPER_INNER_RETURNCODE_SPLIT,
            field: value,
        },
    )

    with pytest.raises(R.CampaignPlanError, match="exact ledger generation"):
        _arm(fixture)


def test_v2_marker_rejects_legacy_wrapper_inner_returncode_split(
    tmp_path, monkeypatch
):
    fixture = _quarantined_dispatch(
        tmp_path,
        monkeypatch,
        arm=False,
        with_wip=True,
        durable_wip_capsule=True,
    )
    _set_exec_record_fields(
        fixture,
        **_LEGACY_WRAPPER_INNER_RETURNCODE_SPLIT,
    )

    with pytest.raises(R.CampaignPlanError, match="exact ledger generation"):
        _arm(fixture)


@pytest.mark.parametrize("valid_operator_discard", (True, False))
def test_ordinary_taint_release_from_armed_legacy_marker_requires_exact_operator_discard(
    tmp_path, monkeypatch, valid_operator_discard
):
    fixture = _quarantined_dispatch(
        tmp_path, monkeypatch, arm=True, with_wip=True
    )
    marker, parsed = R._read_existing_dispatch_quarantine(
        fixture["item"], require_recovery_arm=True
    )
    try:
        state = R._rebind_post_reboot_ledger(
            fixture["item"],
            parsed,
            marker=marker,
            current_boot=fixture["armed_boot"],
        )
        correction = R._build_post_reboot_correction(
            fixture["item"],
            state.record,
            reason="operator discarded unpublished generation",
            transcript_sha=state.record["protected_transcript_sha256"],
            diagnostics_sha=state.record["protected_diagnostics_sha256"],
        )
        R._append_recovery_phase_cas(state, correction)
        state = R._rebind_post_reboot_ledger(
            fixture["item"],
            parsed,
            marker=marker,
            current_boot=fixture["armed_boot"],
        )
        cleanup = R._build_post_reboot_cleanup(
            fixture["item"], state.record, parsed
        )
        R._append_recovery_phase_cas(state, cleanup)
        state = R._rebind_post_reboot_ledger(
            fixture["item"],
            parsed,
            marker=marker,
            current_boot=fixture["armed_boot"],
        )
        result = {
            "game": fixture["item"]["game"],
            "target_level": fixture["item"]["target_level"],
            "reached": fixture["item"]["reached"],
            "result": "tainted_noncounting",
            "retry_complexity_n": fixture["item"]["retry_complexity_n"],
            "operator_recovery": (
                "discarded_unpublished_same_boot_generation"
                if valid_operator_discard else "forged_operator_discard"
            ),
            "detached_processes_proven_absent": False,
            "published_frontier_unchanged": True,
        }
        if valid_operator_discard:
            authority = R._build_dispatch_release_authority(
                fixture["item"],
                marker,
                state.ledger,
                result,
                kind="ordinary_safe_terminal_v1",
            )
            assert authority["terminal_event"] == (
                "codex_taint_cleanup_completed"
            )
        else:
            with pytest.raises(
                R.CampaignPlanError, match="later failure phase"
            ):
                R._build_dispatch_release_authority(
                    fixture["item"],
                    marker,
                    state.ledger,
                    result,
                    kind="ordinary_safe_terminal_v1",
                )
    finally:
        R._close_dispatch_quarantine(marker)


@pytest.mark.parametrize(
    "target_event",
    (
        "codex_exec_classification_correction",
        "codex_taint_cleanup_completed",
        "codex_post_reboot_operator_recovery_completed",
    ),
)
@pytest.mark.parametrize("boundary", (1, 23))
def test_phase_intent_repairs_partial_ledger_append(
    tmp_path, monkeypatch, target_event, boundary
):
    fixture = _quarantined_dispatch(tmp_path, monkeypatch)
    ledger_identity = (
        fixture["ledger"].stat().st_dev,
        fixture["ledger"].stat().st_ino,
    )
    real_append = R._append_recovery_phase_cas
    real_write = R.os.write
    active = False
    partial_written = False

    def append_with_target(state, record):
        nonlocal active
        if record["event"] != target_event:
            return real_append(state, record)
        active = True
        try:
            return real_append(state, record)
        finally:
            active = False

    def partial_ledger_write(descriptor, payload):
        nonlocal partial_written
        identity = os.fstat(descriptor)
        if active and (identity.st_dev, identity.st_ino) == ledger_identity:
            if not partial_written:
                partial_written = True
                return real_write(descriptor, payload[:boundary])
            raise OSError(errno.ENOSPC, "synthetic ledger ENOSPC")
        return real_write(descriptor, payload)

    monkeypatch.setattr(R, "_append_recovery_phase_cas", append_with_target)
    monkeypatch.setattr(R.os, "write", partial_ledger_write)
    with pytest.raises(R.CampaignPlanError, match="phase append failed"):
        _recover(fixture)
    assert partial_written
    assert list(fixture["marker"].parent.glob(".codex_recovery_*.intent"))

    monkeypatch.setattr(R, "_append_recovery_phase_cas", real_append)
    monkeypatch.setattr(R.os, "write", real_write)
    assert _recover(fixture)["result"] == "tainted_noncounting"
    events = [row["event"] for row in R.Guard.read_ledger(fixture["ledger"])]
    assert events == [
        "codex_exec",
        "codex_exec_classification_correction",
        "codex_taint_cleanup_completed",
        "codex_post_reboot_operator_recovery_completed",
        "codex_dispatch_release_authorized",
    ]
    assert not list(fixture["marker"].parent.glob(".codex_recovery_*"))


@pytest.mark.parametrize(
    "target_event",
    (
        "codex_exec_classification_correction",
        "codex_post_reboot_operator_recovery_completed",
    ),
)
def test_installed_intent_retry_refsyncs_parent_before_ledger_reconcile(
    tmp_path, monkeypatch, target_event
):
    fixture = _quarantined_dispatch(tmp_path, monkeypatch)
    root_stat = fixture["marker"].parent.stat(follow_symlinks=False)
    root_identity = (root_stat.st_dev, root_stat.st_ino)
    ledger_stat = fixture["ledger"].stat(follow_symlinks=False)
    ledger_identity = (ledger_stat.st_dev, ledger_stat.st_ino)
    real_append = R._append_recovery_phase_cas
    real_replace = R.os.replace
    real_fsync = R.os.fsync
    active = False
    installed = False
    failed = False

    def target_append(state, record):
        nonlocal active
        if record["event"] != target_event:
            return real_append(state, record)
        active = True
        try:
            return real_append(state, record)
        finally:
            active = False

    def track_install(source, destination, *args, **kwargs):
        nonlocal installed
        result = real_replace(source, destination, *args, **kwargs)
        if active and str(destination).endswith(".intent"):
            installed = True
        return result

    def fail_before_installed_parent_fsync(descriptor):
        nonlocal failed
        metadata = os.fstat(descriptor)
        if (
            active
            and installed
            and not failed
            and (metadata.st_dev, metadata.st_ino) == root_identity
        ):
            failed = True
            raise OSError(errno.EIO, "synthetic installed-intent fsync crash")
        return real_fsync(descriptor)

    monkeypatch.setattr(R, "_append_recovery_phase_cas", target_append)
    monkeypatch.setattr(R.os, "replace", track_install)
    monkeypatch.setattr(R.os, "fsync", fail_before_installed_parent_fsync)
    with pytest.raises(R.CampaignPlanError, match="install.*phase intent"):
        _recover(fixture)
    assert installed and failed
    assert list(fixture["marker"].parent.glob(".codex_recovery_*.intent"))

    real_confirm = R._fsync_and_revalidate_installed_intent
    confirming = False
    parent_fsynced = False
    intent_revalidated = False
    real_write = R.os.write

    def record_confirm(*args, **kwargs):
        nonlocal confirming, intent_revalidated
        confirming = True
        try:
            result = real_confirm(*args, **kwargs)
        finally:
            confirming = False
        assert parent_fsynced
        intent_revalidated = True
        return result

    def record_fsync(descriptor):
        nonlocal parent_fsynced
        metadata = os.fstat(descriptor)
        if confirming and (metadata.st_dev, metadata.st_ino) == root_identity:
            parent_fsynced = True
        return real_fsync(descriptor)

    def require_revalidated_intent_before_ledger_write(descriptor, payload):
        metadata = os.fstat(descriptor)
        if (metadata.st_dev, metadata.st_ino) == ledger_identity:
            assert intent_revalidated
        return real_write(descriptor, payload)

    monkeypatch.setattr(R, "_append_recovery_phase_cas", real_append)
    monkeypatch.setattr(R.os, "replace", real_replace)
    monkeypatch.setattr(R.os, "fsync", record_fsync)
    monkeypatch.setattr(
        R, "_fsync_and_revalidate_installed_intent", record_confirm
    )
    monkeypatch.setattr(
        R.os, "write", require_revalidated_intent_before_ledger_write
    )
    assert _recover(fixture)["result"] == "tainted_noncounting"
    assert parent_fsynced and intent_revalidated


def test_installed_cleanup_intent_refsyncs_parent_before_wip_discard(
    tmp_path, monkeypatch
):
    fixture = _quarantined_dispatch(
        tmp_path, monkeypatch, arm=False, with_wip=True
    )
    latest = fixture["wip"] / "latest.json"
    latest.write_text('{"attempt":"interrupted"}\n', encoding="utf-8")
    preflight = _arm(fixture, None)
    outcome = _arm(fixture, preflight["current_wip_state_sha256"])
    fixture["nonce"] = outcome["recovery_nonce"]
    root_stat = fixture["marker"].parent.stat(follow_symlinks=False)
    root_identity = (root_stat.st_dev, root_stat.st_ino)
    real_append = R._append_recovery_phase_cas
    real_replace = R.os.replace
    real_fsync = R.os.fsync
    active = False
    installed = False
    failed = False

    def target_cleanup(state, record, **kwargs):
        nonlocal active
        if record["event"] != "codex_taint_cleanup_completed":
            return real_append(state, record, **kwargs)
        active = True
        try:
            return real_append(state, record, **kwargs)
        finally:
            active = False

    def track_install(source, destination, *args, **kwargs):
        nonlocal installed
        result = real_replace(source, destination, *args, **kwargs)
        if active and str(destination).endswith(".intent"):
            installed = True
        return result

    def fail_before_parent_fsync(descriptor):
        nonlocal failed
        metadata = os.fstat(descriptor)
        if (
            active
            and installed
            and not failed
            and (metadata.st_dev, metadata.st_ino) == root_identity
        ):
            failed = True
            raise OSError(errno.EIO, "synthetic cleanup-intent fsync crash")
        return real_fsync(descriptor)

    monkeypatch.setattr(R, "_append_recovery_phase_cas", target_cleanup)
    monkeypatch.setattr(R.os, "replace", track_install)
    monkeypatch.setattr(R.os, "fsync", fail_before_parent_fsync)
    with pytest.raises(R.CampaignPlanError, match="install.*phase intent"):
        _recover(fixture)
    assert installed and failed
    assert latest.is_file()

    real_confirm = R._fsync_and_revalidate_installed_intent
    real_discard = R._discard_confirmed_wip_latest_pointer
    confirming = False
    parent_fsynced = False
    intent_revalidated = False

    def record_confirm(*args, **kwargs):
        nonlocal confirming, intent_revalidated
        confirming = True
        try:
            result = real_confirm(*args, **kwargs)
        finally:
            confirming = False
        assert parent_fsynced
        intent_revalidated = True
        return result

    def record_fsync(descriptor):
        nonlocal parent_fsynced
        metadata = os.fstat(descriptor)
        if confirming and (metadata.st_dev, metadata.st_ino) == root_identity:
            parent_fsynced = True
        return real_fsync(descriptor)

    def require_revalidated_intent_before_discard(*args, **kwargs):
        assert intent_revalidated
        return real_discard(*args, **kwargs)

    monkeypatch.setattr(R, "_append_recovery_phase_cas", real_append)
    monkeypatch.setattr(R.os, "replace", real_replace)
    monkeypatch.setattr(R.os, "fsync", record_fsync)
    monkeypatch.setattr(
        R, "_fsync_and_revalidate_installed_intent", record_confirm
    )
    monkeypatch.setattr(
        R,
        "_discard_confirmed_wip_latest_pointer",
        require_revalidated_intent_before_discard,
    )
    assert _recover(fixture)["result"] == "tainted_noncounting"
    assert parent_fsynced and intent_revalidated
    assert not latest.exists()


def test_phase_intent_recovers_partial_preparing_write(tmp_path, monkeypatch):
    fixture = _quarantined_dispatch(tmp_path, monkeypatch)
    ledger_identity = (
        fixture["ledger"].stat().st_dev,
        fixture["ledger"].stat().st_ino,
    )
    real_append = R._append_recovery_phase_cas
    real_write = R.os.write
    active = False
    partial_written = False

    def target_correction(state, record):
        nonlocal active
        if record["event"] != "codex_exec_classification_correction":
            return real_append(state, record)
        active = True
        try:
            return real_append(state, record)
        finally:
            active = False

    def partial_intent_write(descriptor, payload):
        nonlocal partial_written
        identity = os.fstat(descriptor)
        if active and (identity.st_dev, identity.st_ino) != ledger_identity:
            if not partial_written:
                partial_written = True
                return real_write(descriptor, payload[:19])
            raise OSError(errno.ENOSPC, "synthetic intent ENOSPC")
        return real_write(descriptor, payload)

    monkeypatch.setattr(R, "_append_recovery_phase_cas", target_correction)
    monkeypatch.setattr(R.os, "write", partial_intent_write)
    with pytest.raises(R.CampaignPlanError, match="prepare.*phase intent"):
        _recover(fixture)
    assert list(
        fixture["marker"].parent.glob(".codex_recovery_*.intent.preparing")
    )
    assert [
        row["event"] for row in R.Guard.read_ledger(fixture["ledger"])
    ] == ["codex_exec"]

    monkeypatch.setattr(R, "_append_recovery_phase_cas", real_append)
    monkeypatch.setattr(R.os, "write", real_write)
    assert _recover(fixture)["result"] == "tainted_noncounting"
    assert not list(fixture["marker"].parent.glob(".codex_recovery_*"))


def test_phase_intent_retry_refsyncs_complete_preparing_inode(
    tmp_path, monkeypatch
):
    fixture = _quarantined_dispatch(tmp_path, monkeypatch)
    ledger_identity = (
        fixture["ledger"].stat().st_dev,
        fixture["ledger"].stat().st_ino,
    )
    real_fsync = R.os.fsync
    real_prepare = R._prepare_recovery_phase_intent_locked
    staged_identity = None
    active = False

    def track_prepare(state, record, expected_raw):
        nonlocal active
        active = True
        try:
            return real_prepare(state, record, expected_raw)
        finally:
            active = False

    def fail_before_preparing_fsync(descriptor):
        nonlocal staged_identity
        metadata = os.fstat(descriptor)
        identity = (metadata.st_dev, metadata.st_ino)
        if (
            active
            and staged_identity is None
            and R.stat.S_ISREG(metadata.st_mode)
            and identity != ledger_identity
        ):
            staged_identity = identity
            raise OSError("synthetic pre-persistence intent fsync failure")
        return real_fsync(descriptor)

    monkeypatch.setattr(R, "_prepare_recovery_phase_intent_locked", track_prepare)
    monkeypatch.setattr(R.os, "fsync", fail_before_preparing_fsync)
    with pytest.raises(R.CampaignPlanError, match="durably prepare"):
        _recover(fixture)
    preparing = list(
        fixture["marker"].parent.glob(".codex_recovery_*.intent.preparing")
    )
    assert len(preparing) == 1
    assert staged_identity == (
        preparing[0].stat().st_dev,
        preparing[0].stat().st_ino,
    )

    retry_staged_fsyncs = 0

    def count_retry_preparing_fsync(descriptor):
        nonlocal retry_staged_fsyncs
        metadata = os.fstat(descriptor)
        if (metadata.st_dev, metadata.st_ino) == staged_identity:
            retry_staged_fsyncs += 1
        return real_fsync(descriptor)

    monkeypatch.setattr(R, "_prepare_recovery_phase_intent_locked", real_prepare)
    monkeypatch.setattr(R.os, "fsync", count_retry_preparing_fsync)
    assert _recover(fixture)["result"] == "tainted_noncounting"
    assert retry_staged_fsyncs == 1
    assert not list(fixture["marker"].parent.glob(".codex_recovery_*"))


def test_operator_preparing_intent_resumes_on_a_later_boot(
    tmp_path, monkeypatch
):
    fixture = _quarantined_dispatch(tmp_path, monkeypatch)
    original = R._durably_reseal_staged_file_at
    crashed = False

    def crash_after_operator_preparing_fsync(
        root_fd,
        name,
        *,
        expected_payload,
        expected_identity,
        label,
    ):
        nonlocal crashed
        result = original(
            root_fd,
            name,
            expected_payload=expected_payload,
            expected_identity=expected_identity,
            label=label,
        )
        if not crashed and name.endswith("_operator.intent.preparing"):
            crashed = True
            raise R.CampaignPlanError(
                "synthetic crash after operator preparing fsync"
            )
        return result

    monkeypatch.setattr(
        R,
        "_durably_reseal_staged_file_at",
        crash_after_operator_preparing_fsync,
    )
    with pytest.raises(R.CampaignPlanError, match="operator preparing fsync"):
        _recover(fixture)
    assert crashed
    assert len(list(
        fixture["marker"].parent.glob("*_operator.intent.preparing")
    )) == 1
    assert [
        row["event"] for row in R.Guard.read_ledger(fixture["ledger"])
    ] == [
        "codex_exec",
        "codex_exec_classification_correction",
        "codex_taint_cleanup_completed",
    ]

    monkeypatch.setattr(R, "_durably_reseal_staged_file_at", original)
    later_boot = Recovery.BootIdentity(
        fixture["current_boot"].source,
        "33333333-3333-4333-8333-333333333333",
    )
    outcome = R._recover_post_reboot_quarantine(
        fixture["item"],
        confirm_dispatch_id=fixture["dispatch_id"],
        confirm_recovery_nonce=fixture["nonce"],
        boot_identity_provider=lambda: later_boot,
    )
    assert outcome["result"] == "tainted_noncounting"
    receipt = _operator_receipt(R.Guard.read_ledger(fixture["ledger"]))
    assert receipt["current_boot_identity"] == (
        Recovery.boot_identity_receipt(fixture["current_boot"])
    )


@pytest.mark.parametrize(
    "target_event",
    (
        "codex_exec_classification_correction",
        "codex_taint_cleanup_completed",
        "codex_post_reboot_operator_recovery_completed",
    ),
)
@pytest.mark.parametrize("mutation", ("unrelated_row", "inode_replace"))
def test_phase_cas_rejects_interposed_ledger_mutation(
    tmp_path, monkeypatch, target_event, mutation
):
    fixture = _quarantined_dispatch(tmp_path, monkeypatch)
    original = R._append_recovery_phase_cas
    injected = False

    def interpose(state, record):
        nonlocal injected
        if record["event"] == target_event and not injected:
            injected = True
            if mutation == "unrelated_row":
                with fixture["ledger"].open("ab") as stream:
                    stream.write(Recovery.canonical_json_line({
                        "event": "synthetic_unrelated_interposition",
                    }))
                    stream.flush()
                    os.fsync(stream.fileno())
            else:
                replacement = fixture["ledger"].with_suffix(".replacement")
                replacement.write_bytes(fixture["ledger"].read_bytes())
                os.replace(replacement, fixture["ledger"])
        return original(state, record)

    monkeypatch.setattr(R, "_append_recovery_phase_cas", interpose)
    pattern = "phase changed" if mutation == "unrelated_row" else "inode changed"
    with pytest.raises(R.CampaignPlanError, match=pattern):
        _recover(fixture)
    assert injected
    rows = Recovery.parse_canonical_jsonl(
        fixture["ledger"].read_bytes(), label="interposed test ledger"
    )
    assert not any(row.get("event") == target_event for row in rows)


@pytest.mark.parametrize("presence", tuple(product((False, True), repeat=4)))
def test_tombstone_inventory_accepts_exactly_reachable_phases(presence):
    labels = ("W", "P", "Wt", "Pt")
    inventory = frozenset(
        label for label, exists in zip(labels, presence) if exists
    )
    if inventory in R.VALID_RECOVERY_TOMBSTONE_INVENTORIES:
        assert R._validate_recovery_tombstone_inventory(*presence) == inventory
    else:
        with pytest.raises(R.CampaignPlanError, match="reachable tombstone"):
            R._validate_recovery_tombstone_inventory(*presence)


@pytest.mark.parametrize("renamed", ("workspace", "protected"))
def test_tombstone_cleanup_resumes_after_each_durable_rename(
    tmp_path, monkeypatch, renamed
):
    fixture = _quarantined_dispatch(tmp_path, monkeypatch)
    original = R._rename_recovery_tombstone
    crashed = False

    def rename_then_crash(source, target, expected_identity):
        nonlocal crashed
        result = original(source, target, expected_identity)
        if not crashed and source == fixture[renamed]:
            crashed = True
            raise R.CampaignPlanError(
                f"synthetic crash after {renamed} tombstone fsync"
            )
        return result

    monkeypatch.setattr(R, "_rename_recovery_tombstone", rename_then_crash)
    with pytest.raises(R.CampaignPlanError, match=f"after {renamed}"):
        _recover(fixture)
    assert crashed
    workspace_tombstone, protected_tombstone = _tombstones(fixture)
    assert workspace_tombstone.exists()
    assert protected_tombstone.exists() is (renamed == "protected")

    monkeypatch.setattr(R, "_rename_recovery_tombstone", original)
    assert _recover(fixture)["result"] == "tainted_noncounting"
    assert not workspace_tombstone.exists()
    assert not protected_tombstone.exists()


@pytest.mark.parametrize("partial", ("workspace", "protected"))
def test_tombstone_cleanup_resumes_mid_rmtree(
    tmp_path, monkeypatch, partial
):
    fixture = _quarantined_dispatch(tmp_path, monkeypatch)
    original = R.shutil.rmtree
    crashed = False

    def partial_rmtree(path, *args, **kwargs):
        nonlocal crashed
        selected_parent = (
            fixture["workspace"].parent
            if partial == "workspace" else fixture["protected"].parent
        )
        path = Path(path)
        if not crashed and path.parent == selected_parent:
            crashed = True
            if partial == "workspace":
                (path / "probe.py").unlink()
            else:
                next(path.glob("*.jsonl")).unlink()
            raise OSError(f"synthetic partial {partial} rmtree")
        return original(path, *args, **kwargs)

    partial_rmtree.avoids_symlink_attacks = original.avoids_symlink_attacks
    monkeypatch.setattr(R.shutil, "rmtree", partial_rmtree)
    with pytest.raises(R.CampaignPlanError, match="tombstone cleanup failed"):
        _recover(fixture)
    assert crashed

    monkeypatch.setattr(R.shutil, "rmtree", original)
    assert _recover(fixture)["result"] == "tainted_noncounting"


def test_in_workspace_lock_can_disappear_only_inside_partial_tombstone(
    tmp_path, monkeypatch
):
    fixture = _quarantined_dispatch(
        tmp_path, monkeypatch, lock_schema="in_workspace_v1"
    )
    original = R.shutil.rmtree
    crashed = False

    def unlink_inner_lock_then_crash(path, *args, **kwargs):
        nonlocal crashed
        path = Path(path)
        if not crashed and path.parent == fixture["workspace"].parent:
            crashed = True
            (path / ".orchestrate.lock").unlink()
            raise OSError("synthetic crash after inner lock deletion")
        return original(path, *args, **kwargs)

    unlink_inner_lock_then_crash.avoids_symlink_attacks = (
        original.avoids_symlink_attacks
    )
    monkeypatch.setattr(R.shutil, "rmtree", unlink_inner_lock_then_crash)
    with pytest.raises(R.CampaignPlanError, match="tombstone cleanup failed"):
        _recover(fixture)
    assert crashed

    monkeypatch.setattr(R.shutil, "rmtree", original)
    assert _recover(fixture)["result"] == "tainted_noncounting"


@pytest.mark.parametrize("replacement", ("symlink", "hardlink", "directory"))
def test_recovery_rejects_replaced_workspace_tombstone(
    tmp_path, monkeypatch, replacement
):
    fixture = _quarantined_dispatch(tmp_path, monkeypatch)
    original = R._rename_recovery_tombstone

    def crash_after_workspace(source, target, expected_identity):
        result = original(source, target, expected_identity)
        if source == fixture["workspace"]:
            raise R.CampaignPlanError("synthetic workspace rename boundary")
        return result

    monkeypatch.setattr(R, "_rename_recovery_tombstone", crash_after_workspace)
    with pytest.raises(R.CampaignPlanError, match="rename boundary"):
        _recover(fixture)
    monkeypatch.setattr(R, "_rename_recovery_tombstone", original)

    tombstone, _protected_tombstone = _tombstones(fixture)
    preserved = tombstone.with_name(f"{tombstone.name}.preserved")
    os.rename(tombstone, preserved)
    outside = tmp_path / f"outside-{replacement}"
    if replacement == "symlink":
        outside.mkdir()
        (outside / "sentinel").write_text("preserve\n", encoding="utf-8")
        os.symlink(outside, tombstone)
    elif replacement == "hardlink":
        outside.write_text("preserve\n", encoding="utf-8")
        os.link(outside, tombstone)
    else:
        tombstone.mkdir()
        (tombstone / "sentinel").write_text("preserve\n", encoding="utf-8")

    with pytest.raises(R.CampaignPlanError, match="tombstone|directory"):
        _recover(fixture)
    if outside.exists() and outside.is_file():
        assert outside.read_text(encoding="utf-8") == "preserve\n"
    elif outside.exists():
        assert (outside / "sentinel").read_text(encoding="utf-8") == "preserve\n"


@pytest.mark.parametrize("inventory", ("protected_only", "workspace_tomb_only"))
def test_cleanup_rejects_unreachable_real_inventory(
    tmp_path, monkeypatch, inventory
):
    fixture = _quarantined_dispatch(tmp_path, monkeypatch)
    original = R._append_recovery_phase_cas

    def crash_after_correction(state, record):
        result = original(state, record)
        if record["event"] == "codex_exec_classification_correction":
            raise R.CampaignPlanError("synthetic correction boundary")
        return result

    monkeypatch.setattr(R, "_append_recovery_phase_cas", crash_after_correction)
    with pytest.raises(R.CampaignPlanError, match="correction boundary"):
        _recover(fixture)
    monkeypatch.setattr(R, "_append_recovery_phase_cas", original)

    workspace_tombstone, _protected_tombstone = _tombstones(fixture)
    if inventory == "protected_only":
        R.shutil.rmtree(fixture["workspace"])
    else:
        os.rename(fixture["workspace"], workspace_tombstone)
        R.shutil.rmtree(fixture["protected"])
    with pytest.raises(R.CampaignPlanError, match="reachable tombstone"):
        _recover(fixture)


def test_cleanup_retries_stale_external_lock_after_directories_are_gone(
    tmp_path, monkeypatch
):
    fixture = _quarantined_dispatch(tmp_path, monkeypatch)
    original = R.os.unlink

    def fail_external_lock_unlink(path, *args, **kwargs):
        if os.fspath(path) == os.fspath(fixture["exact_lock"]):
            raise OSError("synthetic external lock unlink failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(R.os, "unlink", fail_external_lock_unlink)
    with pytest.raises(R.CampaignPlanError, match="tombstone cleanup failed"):
        _recover(fixture)
    assert not fixture["workspace"].exists()
    assert not fixture["protected"].exists()
    assert fixture["exact_lock"].exists()

    monkeypatch.setattr(R.os, "unlink", original)
    assert _recover(fixture)["result"] == "tainted_noncounting"
    assert not fixture["exact_lock"].exists()


def test_cleanup_resolves_external_lock_unlink_fsync_ambiguity(
    tmp_path, monkeypatch
):
    fixture = _quarantined_dispatch(tmp_path, monkeypatch)
    original = R._fsync_directory
    failed = False

    def fsync_then_crash(path):
        nonlocal failed
        result = original(path)
        if not failed and Path(path) == fixture["exact_lock"].parent:
            failed = True
            raise OSError("synthetic lock-parent fsync report failure")
        return result

    monkeypatch.setattr(R, "_fsync_directory", fsync_then_crash)
    with pytest.raises(R.CampaignPlanError, match="tombstone cleanup failed"):
        _recover(fixture)
    assert failed
    assert not fixture["workspace"].exists()
    assert not fixture["protected"].exists()
    assert not fixture["exact_lock"].exists()

    monkeypatch.setattr(R, "_fsync_directory", original)
    assert _recover(fixture)["result"] == "tainted_noncounting"


def test_cleanup_rejects_external_lock_disappearance_before_rename(
    tmp_path, monkeypatch
):
    fixture = _quarantined_dispatch(tmp_path, monkeypatch)
    original = R._append_recovery_phase_cas

    def crash_after_correction(state, record):
        result = original(state, record)
        if record["event"] == "codex_exec_classification_correction":
            raise R.CampaignPlanError("synthetic correction boundary")
        return result

    monkeypatch.setattr(R, "_append_recovery_phase_cas", crash_after_correction)
    with pytest.raises(R.CampaignPlanError, match="correction boundary"):
        _recover(fixture)
    monkeypatch.setattr(R, "_append_recovery_phase_cas", original)
    fixture["exact_lock"].unlink()

    with pytest.raises(R.CampaignPlanError, match="lock disappeared"):
        _recover(fixture)
    assert fixture["workspace"].is_dir()
    assert fixture["protected"].is_dir()


def test_recovery_rejects_armed_marker_inode_replacement(
    tmp_path, monkeypatch
):
    fixture = _quarantined_dispatch(tmp_path, monkeypatch)
    replacement = fixture["marker"].with_name("replacement-marker")
    replacement.write_bytes(fixture["marker"].read_bytes())
    os.chmod(replacement, 0o600)
    os.replace(replacement, fixture["marker"])

    with pytest.raises(R.CampaignPlanError, match="arm does not bind"):
        _recover(fixture)
    assert [
        row["event"] for row in R.Guard.read_ledger(fixture["ledger"])
    ] == ["codex_exec"]


@pytest.mark.parametrize("target", ("canonical", "wip"))
def test_recovery_rejects_actual_root_xattr_drift(
    tmp_path, monkeypatch, target
):
    if not hasattr(os, "setxattr"):
        pytest.skip("platform has no xattr API")
    fixture = _quarantined_dispatch(
        tmp_path, monkeypatch, with_wip=(target == "wip")
    )
    path = fixture["artifact"] if target == "canonical" else fixture["wip"]
    last_error = None
    for name in ("user.gkm_recovery_test", "com.gkm.recovery_test"):
        try:
            os.setxattr(path, name, b"changed")
        except OSError as exc:
            last_error = exc
            continue
        break
    else:
        pytest.skip(f"test filesystem has no writable xattrs: {last_error}")

    with pytest.raises(R.CampaignPlanError, match="canonical/WIP metadata"):
        _recover(fixture)


def test_recovery_rejects_noncanonical_ledger_phase(
    tmp_path, monkeypatch
):
    fixture = _quarantined_dispatch(tmp_path, monkeypatch)
    original = R._append_recovery_phase_cas

    def crash_after_correction(state, record):
        result = original(state, record)
        if record["event"] == "codex_exec_classification_correction":
            raise R.CampaignPlanError("synthetic correction boundary")
        return result

    monkeypatch.setattr(R, "_append_recovery_phase_cas", crash_after_correction)
    with pytest.raises(R.CampaignPlanError, match="correction boundary"):
        _recover(fixture)
    monkeypatch.setattr(R, "_append_recovery_phase_cas", original)

    lines = fixture["ledger"].read_bytes().splitlines(keepends=True)
    correction = json.loads(lines[1])
    lines[1] = json.dumps(correction, sort_keys=True).encode("utf-8") + b"\n"
    fixture["ledger"].write_bytes(b"".join(lines))
    with pytest.raises(R.CampaignPlanError, match="not canonically encoded"):
        _recover(fixture)


def test_recovery_uses_phase_order_when_cross_boot_clock_runs_backwards(
    tmp_path, monkeypatch
):
    fixture = _quarantined_dispatch(tmp_path, monkeypatch)
    timestamps = iter((
        datetime(2003, 1, 1, tzinfo=timezone.utc),
        datetime(2002, 1, 1, tzinfo=timezone.utc),
        datetime(2001, 1, 1, tzinfo=timezone.utc),
        datetime(2000, 1, 1, tzinfo=timezone.utc),
    ))

    class ReverseDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            value = next(timestamps)
            return value if tz is None else value.astimezone(tz)

    monkeypatch.setattr(R, "datetime", ReverseDateTime)
    assert _recover(fixture)["result"] == "tainted_noncounting"
    rows = R.Guard.read_ledger(fixture["ledger"])
    phase_times = [
        datetime.fromisoformat(row["recorded_at"])
        for row in rows[1:]
    ]
    assert phase_times == sorted(phase_times, reverse=True)


def test_completed_replay_allows_a_later_kernel_boot(tmp_path, monkeypatch):
    fixture = _quarantined_dispatch(tmp_path, monkeypatch)
    assert _recover(fixture)["result"] == "tainted_noncounting"
    later_boot = Recovery.BootIdentity(
        fixture["current_boot"].source,
        "33333333-3333-4333-8333-333333333333",
    )

    outcome = R._recover_post_reboot_quarantine(
        fixture["item"],
        confirm_dispatch_id=fixture["dispatch_id"],
        confirm_recovery_nonce=fixture["nonce"],
        boot_identity_provider=lambda: later_boot,
    )
    assert outcome["operator_recovery"] == "post_reboot_already_completed"


def test_markerless_completed_v1_receipt_uses_exact_legacy_keyset(
    tmp_path, monkeypatch
):
    fixture = _quarantined_dispatch(tmp_path, monkeypatch)
    assert _recover(fixture)["result"] == "tainted_noncounting"
    rows = _canonical_rows(fixture["ledger"])
    receipt = _operator_receipt(rows)
    receipt["schema"] = Recovery.LEGACY_OPERATOR_RECOVERY_SCHEMA
    for field in (
        "wip_recovery_authority",
        "confirmed_current_wip_state_sha256",
        "wip_disposition",
        "discard_survivor_sha256",
        "restored_wip_logical_state_sha256",
        "wip_restore_logical_state_schema",
        "wip_rollback_capsule_name",
        "wip_rollback_capsule_identity",
        "wip_rollback_capsule_bytes",
        "wip_rollback_capsule_sha256",
        "wip_rollback_capsule_state_sha256",
    ):
        receipt.pop(field)
    rows = [
        row for row in rows
        if row.get("event") != "codex_dispatch_release_authorized"
    ]
    _write_canonical_rows(fixture["ledger"], rows)

    outcome = _recover(fixture)
    assert outcome["operator_recovery"] == "post_reboot_already_completed"


def test_markerless_completed_v2_receipt_requires_release_authority_row(
    tmp_path, monkeypatch
):
    fixture = _quarantined_dispatch(tmp_path, monkeypatch)
    assert _recover(fixture)["result"] == "tainted_noncounting"
    rows = [
        row for row in _canonical_rows(fixture["ledger"])
        if row.get("event") != "codex_dispatch_release_authorized"
    ]
    _write_canonical_rows(fixture["ledger"], rows)

    with pytest.raises(
        R.CampaignPlanError,
        match="current completed recovery lacks durable release authority",
    ):
        _recover(fixture)


@pytest.mark.parametrize("schema_case", ("v1_extra", "v2_missing"))
def test_markerless_completed_receipt_schemas_are_disjoint(
    tmp_path, monkeypatch, schema_case
):
    fixture = _quarantined_dispatch(tmp_path, monkeypatch)
    assert _recover(fixture)["result"] == "tainted_noncounting"
    rows = _canonical_rows(fixture["ledger"])
    receipt = _operator_receipt(rows)
    if schema_case == "v1_extra":
        receipt["schema"] = Recovery.LEGACY_OPERATOR_RECOVERY_SCHEMA
    else:
        receipt.pop("wip_disposition")
    _write_canonical_rows(fixture["ledger"], rows)

    with pytest.raises(R.CampaignPlanError, match="invalid exact schema"):
        _recover(fixture)


def test_marker_resume_allows_later_boot_after_operator_receipt(
    tmp_path, monkeypatch
):
    fixture = _quarantined_dispatch(tmp_path, monkeypatch)
    original = R._append_recovery_phase_cas

    def crash_after_operator(state, record):
        result = original(state, record)
        if record["event"] == "codex_post_reboot_operator_recovery_completed":
            raise R.CampaignPlanError("synthetic operator boundary")
        return result

    monkeypatch.setattr(R, "_append_recovery_phase_cas", crash_after_operator)
    with pytest.raises(R.CampaignPlanError, match="operator boundary"):
        _recover(fixture)
    monkeypatch.setattr(R, "_append_recovery_phase_cas", original)
    later_boot = Recovery.BootIdentity(
        fixture["current_boot"].source,
        "33333333-3333-4333-8333-333333333333",
    )

    outcome = R._recover_post_reboot_quarantine(
        fixture["item"],
        confirm_dispatch_id=fixture["dispatch_id"],
        confirm_recovery_nonce=fixture["nonce"],
        boot_identity_provider=lambda: later_boot,
    )
    assert outcome["result"] == "tainted_noncounting"
    assert not fixture["marker"].exists()


def test_completed_replay_allows_unrelated_canonical_ledger_tail(
    tmp_path, monkeypatch
):
    fixture = _quarantined_dispatch(tmp_path, monkeypatch)
    assert _recover(fixture)["result"] == "tainted_noncounting"
    with fixture["ledger"].open("ab") as stream:
        stream.write(Recovery.canonical_json_line({
            "event": "synthetic_later_unrelated_row",
            "game": "unrelated-game",
        }))
        stream.flush()
        os.fsync(stream.fileno())

    outcome = _recover(fixture)
    assert outcome["operator_recovery"] == "post_reboot_already_completed"


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("dispatch_id", None),
        ("thread_id", "reboot-thread"),
        (
            "transcript",
            "codex_turn_20260805T000000000000Z_ar25_L1_propose.jsonl",
        ),
    ),
)
def test_completed_replay_rejects_conflicting_later_tail(
    tmp_path, monkeypatch, field, value
):
    fixture = _quarantined_dispatch(tmp_path, monkeypatch)
    assert _recover(fixture)["result"] == "tainted_noncounting"
    if field == "dispatch_id":
        value = fixture["dispatch_id"]
    with fixture["ledger"].open("ab") as stream:
        stream.write(Recovery.canonical_json_line({
            "event": "synthetic_later_conflicting_row",
            field: value,
        }))
        stream.flush()
        os.fsync(stream.fileno())

    with pytest.raises(R.CampaignPlanError, match="conflicting later"):
        _recover(fixture)


@pytest.mark.parametrize(
    "recorded_at",
    (None, "not-a-timestamp", "2026-08-05T12:00:00"),
)
def test_completed_replay_rejects_invalid_operator_timestamp(
    tmp_path, monkeypatch, recorded_at
):
    fixture = _quarantined_dispatch(tmp_path, monkeypatch)
    assert _recover(fixture)["result"] == "tainted_noncounting"
    rows = _canonical_rows(fixture["ledger"])
    rows[3]["recorded_at"] = recorded_at
    _write_canonical_rows(fixture["ledger"], rows)

    with pytest.raises(
        R.CampaignPlanError, match="operator receipt recorded_at"
    ):
        _recover(fixture)


@pytest.mark.parametrize("field", ("dispatch_id", "recovery_nonce"))
@pytest.mark.parametrize("value", (None, "0" * 31, "g" * 32))
def test_completed_replay_rejects_malformed_confirmation(
    tmp_path, monkeypatch, field, value
):
    fixture = _quarantined_dispatch(tmp_path, monkeypatch)
    assert _recover(fixture)["result"] == "tainted_noncounting"
    dispatch_id = fixture["dispatch_id"]
    nonce = fixture["nonce"]
    if field == "dispatch_id":
        dispatch_id = value
        pattern = "dispatch confirmation is malformed"
    else:
        nonce = value
        pattern = "nonce confirmation is malformed"

    with pytest.raises(R.CampaignPlanError, match=pattern):
        R._recover_post_reboot_quarantine(
            fixture["item"],
            confirm_dispatch_id=dispatch_id,
            confirm_recovery_nonce=nonce,
            boot_identity_provider=lambda: fixture["current_boot"],
        )


@pytest.mark.parametrize("field", ("dispatch_id", "recovery_nonce"))
@pytest.mark.parametrize("value", (None, "0" * 31, "g" * 32))
def test_completed_replay_rejects_malformed_receipt_identifier(
    tmp_path, monkeypatch, field, value
):
    fixture = _quarantined_dispatch(tmp_path, monkeypatch)
    assert _recover(fixture)["result"] == "tainted_noncounting"
    rows = _canonical_rows(fixture["ledger"])
    rows[3][field] = value
    _write_canonical_rows(fixture["ledger"], rows)
    pattern = "receipt dispatch ID" if field == "dispatch_id" else "receipt nonce"

    with pytest.raises(R.CampaignPlanError, match=pattern):
        _recover(fixture)


LEGACY_HISTORICAL_RUNNER_SOURCE_SHA256 = (
    "bb3474290d3411f980d53ffcee75be8234e634d478b1136677b9c6a93fe9ec64"
)
BOUNDARY_V2_HISTORICAL_RUNNER_SOURCE_SHA256 = (
    "7455d304c96f5b070ecb4e62a45bcca21e4d5faf52027b8c3434dc094f7e7b0b"
)
BOUNDARY_V2_HISTORICAL_RUNNER_HEAD = (
    "246405c1cd903e1dcde9d3a4c6eed1ec93cf2c1f"
)
BOUNDARY_V3_HISTORICAL_RUNNER_SOURCE_SHA256 = (
    "18b5a3f1da18d10e9f7dba2c73b5d097abe691bd1b2cdfad3f3dcdf99d6a9fc0"
)
BOUNDARY_V3_HISTORICAL_RUNNER_HEAD = (
    "aa666cc3ff4c2167e12ce32b317bc3fe6c45a867"
)
BOUNDARY_V4_HISTORICAL_RUNNER_SOURCE_SHA256 = (
    "3bbd7ca93c9d74eef0b532ca8159283ce6d7fa81b6be316f0792a72ccd054398"
)
BOUNDARY_V4_HISTORICAL_RUNNER_HEAD = (
    "b37d0a0bece4c18da5cdc37f88f829e3a491fee9"
)
FAILURE_REVISION_HISTORICAL_RUNNER_SOURCE_SHA256 = (
    "eefa34dcef63adb4d99deb07de9fa1920d8ef792080427557d5460201ff32f94"
)
FAILURE_REVISION_HISTORICAL_RUNNER_HEAD = (
    "0eeb29d6cce57a71dc0a20bffc471d21849d03de"
)


def test_historical_runner_registries_are_exact_and_coherent():
    expected_heads = {
        LEGACY_HISTORICAL_RUNNER_SOURCE_SHA256:
            "c1f8168f230732f2d745c234555b3e3dfcb8aefa",
        BOUNDARY_V2_HISTORICAL_RUNNER_SOURCE_SHA256:
            BOUNDARY_V2_HISTORICAL_RUNNER_HEAD,
        BOUNDARY_V3_HISTORICAL_RUNNER_SOURCE_SHA256:
            BOUNDARY_V3_HISTORICAL_RUNNER_HEAD,
        BOUNDARY_V4_HISTORICAL_RUNNER_SOURCE_SHA256:
            BOUNDARY_V4_HISTORICAL_RUNNER_HEAD,
        FAILURE_REVISION_HISTORICAL_RUNNER_SOURCE_SHA256:
            FAILURE_REVISION_HISTORICAL_RUNNER_HEAD,
    }
    assert set(R.PINNED_HISTORICAL_RUNNERS) == set(expected_heads)
    assert set(R.SANDBOX_CONTRACTS) == set(expected_heads)
    assert Recovery.APPROVED_SANDBOXED_GENERATION_SOURCES == frozenset(
        expected_heads
    )
    assert {
        source: metadata["head_commit"]
        for source, metadata in R.PINNED_HISTORICAL_RUNNERS.items()
    } == expected_heads
    assert Recovery.QUIESCED_INCOMPLETE_RUNNERS == {
        source: R.PINNED_HISTORICAL_RUNNERS[source]
        for source in expected_heads
        if source != FAILURE_REVISION_HISTORICAL_RUNNER_SOURCE_SHA256
    }
    assert len(set(R.SANDBOX_CONTRACTS.values())) == 1
    for source, metadata in R.PINNED_HISTORICAL_RUNNERS.items():
        assert R.SHA256_RE.fullmatch(source)
        assert R.GIT_COMMIT_RE.fullmatch(metadata["head_commit"])
        assert set(metadata) == {
            "head_commit", "evidence_schema", "lock_schema"
        }
        assert metadata["evidence_schema"] in R.EVIDENCE_SCHEMAS
        assert metadata["lock_schema"] in R.LOCK_SCHEMAS
        assert R.SHA256_RE.fullmatch(R.SANDBOX_CONTRACTS[source])


@pytest.mark.parametrize(
    ("source_sha256", "head_commit"),
    (
        (
            LEGACY_HISTORICAL_RUNNER_SOURCE_SHA256,
            "c1f8168f230732f2d745c234555b3e3dfcb8aefa",
        ),
        (
            BOUNDARY_V2_HISTORICAL_RUNNER_SOURCE_SHA256,
            BOUNDARY_V2_HISTORICAL_RUNNER_HEAD,
        ),
        (
            BOUNDARY_V3_HISTORICAL_RUNNER_SOURCE_SHA256,
            BOUNDARY_V3_HISTORICAL_RUNNER_HEAD,
        ),
        (
            BOUNDARY_V4_HISTORICAL_RUNNER_SOURCE_SHA256,
            BOUNDARY_V4_HISTORICAL_RUNNER_HEAD,
        ),
        (
            FAILURE_REVISION_HISTORICAL_RUNNER_SOURCE_SHA256,
            FAILURE_REVISION_HISTORICAL_RUNNER_HEAD,
        ),
    ),
)
def test_registered_historical_runner_source_matches_pinned_head(
    source_sha256, head_commit
):
    repo = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [
            "git", "-C", os.fspath(repo), "show",
            f"{head_commit}:arc/crack_lab/gkm_legs.py",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        pytest.skip("pinned historical runner commit is unavailable")
    assert hashlib.sha256(proc.stdout).hexdigest() == source_sha256


def _sandboxed_generation_fixture(
    tmp_path, monkeypatch, *,
    source_sha256=LEGACY_HISTORICAL_RUNNER_SOURCE_SHA256,
    interrupted_exec=False,
    boundary_finding_counts=None,
    scratch_relative=Path("abandoned_scratch"),
):
    monkeypatch.setattr(R, "HERE", tmp_path)
    artifact = tmp_path / "agent_solutions" / "ar25_legs"
    artifact.mkdir(parents=True)
    (artifact / "solver.py").write_bytes(b"sealed canonical baseline\n")
    wip = artifact / "wip_context" / "level_01"
    wip.mkdir(parents=True)
    (wip / "latest.json").write_bytes(b'{"attempt":"baseline"}\n')

    scratch = tmp_path / scratch_relative
    protected_root = scratch / ".proposer_transcripts"
    lock_root = scratch / ".workspace_locks"
    protected_root.mkdir(parents=True)
    lock_root.mkdir()
    item = copy.deepcopy(_item())
    tag = "arc_agi3_n0_fresh_frontier"
    workspace_name = f"gkm_legs_ws_ar25_{tag}_deadbeef"
    workspace = scratch / workspace_name
    protected = protected_root / workspace_name
    workspace.mkdir()
    protected.mkdir()
    (workspace / "attempt.txt").write_bytes(b"isolated unpublished attempt\n")
    if interrupted_exec:
        (workspace / "dynamic_probe.py").write_bytes(b'exec("pass")\n')
    transcript_name = "codex_turn_20260805T000000000000Z_ar25_L1_propose.jsonl"
    (protected / transcript_name).write_bytes(b'{"type":"thread.started"}\n')
    ledger = tmp_path / "usage.jsonl"
    ledger.write_bytes(b"")
    historical = (
        {
            "schema": 1,
            "worktree": os.fspath(Path(__file__).resolve().parents[2]),
            "cwd": os.fspath(tmp_path),
            "interpreter": os.fspath(Path(sys.executable).absolute()),
            "head_commit": BOUNDARY_V2_HISTORICAL_RUNNER_HEAD,
            "source_sha256": BOUNDARY_V2_HISTORICAL_RUNNER_SOURCE_SHA256,
            "artifacts_root": os.fspath(tmp_path / "agent_solutions"),
            "scratch_root": os.fspath(scratch),
            "ledger": os.fspath(ledger),
            "lock_schema": "in_workspace_v1",
            "evidence_schema": "sealed_transcript_only_v1",
        }
        if interrupted_exec
        else {
            "source_sha256": source_sha256,
            "scratch_root": os.fspath(scratch),
            "lock_schema": "hashed_external_v1",
            "evidence_schema": "sealed_transcript_only_v1",
            "cwd": os.fspath(tmp_path),
        }
    )
    item["historical_runner"] = historical
    item["argv"].extend([f"--tag={tag}", f"--codex-ledger={ledger}"])
    exact_lock = (
        workspace / ".orchestrate.lock"
        if interrupted_exec
        else R.Legs._workspace_lock_path(os.fspath(workspace))
    )
    exact_lock.write_bytes(b"")
    os.chmod(exact_lock, 0o600)

    monkeypatch.setattr(R.Legs, "SCRATCH", os.fspath(scratch))
    monkeypatch.setattr(R, "_checkpoint_reached", lambda _game: 0)
    expected_binding = R.Status.validate_frontier_binding({
        field: item[field]
        for field in (
            *R.Status.FRONTIER_BINDING_FIELDS,
            "game",
            "reached",
            "target_level",
            "parent_action_count",
        )
    })
    monkeypatch.setattr(
        R.Status,
        "exact_frontier_binding",
        lambda *_args, **_kwargs: expected_binding,
    )
    monkeypatch.setattr(R, "_taint_gate", lambda: None)
    monkeypatch.setattr(
        R,
        "_reconstruct_historical_recovery_item",
        lambda selected, _authority, **_kwargs: selected,
    )
    monkeypatch.setattr(
        R, "_revalidate_historical_control", lambda _item, **_kwargs: None
    )
    if interrupted_exec:
        monkeypatch.setattr(
            R, "_historical_tester_scaffolds", lambda *_args, **_kwargs: {}
        )

    marker = R._arm_dispatch_quarantine(
        item,
        ledger_before=R._capture_ledger_prefix(ledger),
        wip_before=R._capture_wip_rollback(item),
        canonical_before=R._capture_canonical_rollback(item),
        durable_wip_capsule=True,
    )
    R._write_dispatch_quarantine_record(marker, {
        "event": (
            "dispatch_unquiesced" if interrupted_exec else "dispatch_failed"
        ),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "exception_type": "UnquiescedChildError",
        "reason": "synthetic Darwin descendant terminal was unavailable",
        **({} if interrupted_exec else {"child_pid": 424242}),
        "child_returncode": -signal.SIGINT if interrupted_exec else None,
        "workspace": workspace.name,
        "protected": protected.name,
        "transcript": transcript_name,
        "workspace_identity": [workspace.stat().st_dev, workspace.stat().st_ino],
        "protected_identity": [protected.stat().st_dev, protected.stat().st_ino],
        **(
            {"boundary_finding_counts": boundary_finding_counts}
            if boundary_finding_counts is not None else {}
        ),
    })
    marker_path = marker.path
    dispatch_id = marker.dispatch_id
    R._close_dispatch_quarantine(marker)
    rows = _canonical_rows(marker_path)
    capsule = marker_path.parent / rows[0]["wip_rollback_capsule_name"]
    return {
        "item": item,
        "artifact": artifact,
        "wip": wip,
        "scratch": scratch,
        "workspace": workspace,
        "protected": protected,
        "workspace_name": workspace_name,
        "transcript_name": transcript_name,
        "exact_lock": exact_lock,
        "ledger": ledger,
        "marker": marker_path,
        "capsule": capsule,
        "dispatch_id": dispatch_id,
        "boot": Recovery.BootIdentity(
            "darwin_kern_bootsessionuuid",
            "11111111-1111-4111-8111-111111111111",
        ),
    }


def _quiesced_incomplete_evidence_fixture(
    tmp_path, monkeypatch, *,
    source_sha256=BOUNDARY_V2_HISTORICAL_RUNNER_SOURCE_SHA256,
):
    monkeypatch.setattr(R, "HERE", tmp_path)
    artifact = tmp_path / "agent_solutions" / "ar25_legs"
    artifact.mkdir(parents=True)
    canonical = artifact / "solver.py"
    canonical.write_bytes(b"sealed canonical baseline\n")
    wip = artifact / "wip_context" / "level_01"
    wip.mkdir(parents=True)
    baseline_latest = b'{"attempt":"baseline"}\n'
    (wip / "latest.json").write_bytes(baseline_latest)

    scratch = tmp_path / "historical_scratch"
    protected_root = scratch / ".proposer_transcripts"
    protected_root.mkdir(parents=True)
    item = copy.deepcopy(_item())
    tag = "arc_agi3_n0_fresh_frontier"
    workspace_name = f"gkm_legs_ws_ar25_{tag}_incomplete"
    workspace = scratch / workspace_name
    protected = protected_root / workspace_name
    workspace.mkdir()
    (workspace / "host_seed.txt").write_bytes(b"opaque host seed\n")
    ledger = tmp_path / "usage.jsonl"
    ledger.write_bytes(b"")
    metadata = R.PINNED_HISTORICAL_RUNNERS[source_sha256]
    historical = {
        "schema": 1,
        "worktree": os.fspath(tmp_path),
        "cwd": os.fspath(tmp_path),
        "interpreter": os.fspath(Path(sys.executable).absolute()),
        "head_commit": metadata["head_commit"],
        "source_sha256": source_sha256,
        "artifacts_root": os.fspath(tmp_path / "agent_solutions"),
        "scratch_root": os.fspath(scratch),
        "ledger": os.fspath(ledger),
        "lock_schema": metadata["lock_schema"],
        "evidence_schema": metadata["evidence_schema"],
    }
    item["historical_runner"] = historical
    item["argv"].extend([f"--tag={tag}", f"--codex-ledger={ledger}"])
    exact_lock = (
        workspace / ".orchestrate.lock"
        if metadata["lock_schema"] == "in_workspace_v1"
        else R.Legs._workspace_lock_path(os.fspath(workspace))
    )
    exact_lock.parent.mkdir(parents=True, exist_ok=True)
    exact_lock.write_bytes(b"")
    os.chmod(exact_lock, 0o600)

    monkeypatch.setattr(R.Legs, "SCRATCH", os.fspath(scratch))
    monkeypatch.setattr(R, "_checkpoint_reached", lambda _game: 0)
    expected_binding = R.Status.validate_frontier_binding({
        field: item[field]
        for field in (
            *R.Status.FRONTIER_BINDING_FIELDS,
            "game",
            "reached",
            "target_level",
            "parent_action_count",
        )
    })
    monkeypatch.setattr(
        R.Status,
        "exact_frontier_binding",
        lambda *_args, **_kwargs: expected_binding,
    )
    monkeypatch.setattr(R, "_taint_gate", lambda: None)
    monkeypatch.setattr(
        R,
        "_reconstruct_historical_recovery_item",
        lambda selected, _authority, **_kwargs: selected,
    )
    monkeypatch.setattr(
        R, "_revalidate_historical_control", lambda _item, **_kwargs: None
    )

    marker = R._arm_dispatch_quarantine(
        item,
        ledger_before=R._capture_ledger_prefix(ledger),
        wip_before=R._capture_wip_rollback(item),
        canonical_before=R._capture_canonical_rollback(item),
        durable_wip_capsule=True,
    )
    # Simulate disposable child-side WIP written before the environment
    # failure.  The durable capsule remains the only restore authority.
    attempt = wip / "incomplete_attempt"
    attempt.mkdir()
    (attempt / "note.txt").write_bytes(b"unpublished\n")
    (wip / "latest.json").write_bytes(
        b'{"attempt":"incomplete_attempt"}\n'
    )
    observed = R.GuardedChildResult(
        returncode=1,
        workspace=workspace_name,
        transcript=None,
        workspace_identity=(workspace.stat().st_dev, workspace.stat().st_ino),
        protected_identity=None,
        process_tree_quiesced=True,
        descendant_quiescence_unproven=False,
        detached_processes_proven_absent=False,
        normal_exit_left_captured_descendants=False,
    )
    with pytest.raises(R.CampaignPlanError) as producer_failure:
        R._seal_zero_ledger_observation(item, observed)
    assert str(producer_failure.value) == (
        Recovery.QUIESCED_INCOMPLETE_EVIDENCE_REASON
    )
    R._write_dispatch_quarantine_record(marker, {
        "event": "dispatch_failed",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "exception_type": "CampaignPlanError",
        "reason": str(producer_failure.value),
        "child_returncode": 1,
        "workspace": workspace_name,
        "protected": workspace_name,
        "transcript": None,
        "workspace_identity": [
            workspace.stat().st_dev,
            workspace.stat().st_ino,
        ],
        "protected_identity": None,
        "process_tree_quiesced": True,
        "descendant_quiescence_unproven": False,
        "detached_processes_proven_absent": False,
        "normal_exit_left_captured_descendants": False,
    })
    marker_path = marker.path
    dispatch_id = marker.dispatch_id
    scheduler_pid = os.getpid()
    R._close_dispatch_quarantine(marker)
    rows = _canonical_rows(marker_path)
    capsule = marker_path.parent / rows[0]["wip_rollback_capsule_name"]

    def absent(parsed):
        observed = datetime.now(timezone.utc).isoformat()
        return {
            "scheduler_pid": parsed.armed["pid"],
            "scheduler_pid_absence_sample_count": R.SANDBOX_ABSENCE_SAMPLES,
            "scheduler_pid_absence_window_ns": 1,
            "scheduler_pid_absence_first_at": observed,
            "scheduler_pid_absence_last_at": observed,
        }

    monkeypatch.setattr(
        R, "_observe_quiesced_incomplete_evidence_absence", absent
    )
    return {
        "item": item,
        "artifact": artifact,
        "canonical": canonical,
        "wip": wip,
        "baseline_latest": baseline_latest,
        "attempt": attempt,
        "scratch": scratch,
        "workspace": workspace,
        "protected": protected,
        "exact_lock": exact_lock,
        "ledger": ledger,
        "marker": marker_path,
        "capsule": capsule,
        "dispatch_id": dispatch_id,
        "scheduler_pid": scheduler_pid,
    }


def _quiesced_pre_workspace_fixture(tmp_path, monkeypatch):
    fixture = _quiesced_incomplete_evidence_fixture(tmp_path, monkeypatch)
    rows = _canonical_rows(fixture["marker"])
    for field in (
        "workspace",
        "protected",
        "transcript",
        "workspace_identity",
        "protected_identity",
    ):
        rows[1][field] = None
    fixture["marker"].write_bytes(b"".join(
        Recovery.canonical_json_line(row) for row in rows
    ))
    (fixture["workspace"] / "host_seed.txt").unlink()
    fixture["exact_lock"].unlink()
    fixture["workspace"].rmdir()
    return fixture


@pytest.mark.parametrize("source_sha256", (
    BOUNDARY_V2_HISTORICAL_RUNNER_SOURCE_SHA256,
))
def test_quiesced_incomplete_evidence_parser_is_exact(
    tmp_path, monkeypatch, source_sha256
):
    fixture = _quiesced_incomplete_evidence_fixture(
        tmp_path, monkeypatch, source_sha256=source_sha256
    )
    raw = fixture["marker"].read_bytes()
    parsed = Recovery.parse_quiesced_incomplete_evidence_marker(
        raw, require_recovery_arm=False
    )
    assert parsed.dispatch_id == fixture["dispatch_id"]
    assert parsed.unquiesced["transcript"] is None
    assert parsed.unquiesced["protected_identity"] is None
    assert parsed.recovery_arm is None
    R._validate_quiesced_incomplete_evidence_binding(
        fixture["item"], parsed
    )

    rows = _canonical_rows(fixture["marker"])
    mutations = (
        (1, "child_returncode", 0),
        (1, "transcript", "invented.jsonl"),
        (1, "protected_identity", [1, 2]),
        (1, "process_tree_quiesced", False),
        (1, "detached_processes_proven_absent", True),
        (0, "armed_schema", "scheduler_dispatch_armed_v1"),
    )
    for row_index, field, value in mutations:
        changed = copy.deepcopy(rows)
        changed[row_index][field] = value
        with pytest.raises(Recovery.RecoveryEvidenceError):
            Recovery.parse_quiesced_incomplete_evidence_marker(
                b"".join(
                    Recovery.canonical_json_line(row) for row in changed
                ),
                require_recovery_arm=False,
            )
    for mutate_runner in (
        lambda runner: runner.update({"unexpected": True}),
        lambda runner: runner.update({"head_commit": "0" * 40}),
        lambda runner: runner.pop("ledger"),
    ):
        changed = copy.deepcopy(rows)
        mutate_runner(changed[0]["historical_runner"])
        with pytest.raises(
            Recovery.RecoveryEvidenceError, match="runner receipt"
        ):
            Recovery.parse_quiesced_incomplete_evidence_marker(
                b"".join(
                    Recovery.canonical_json_line(row) for row in changed
                ),
                require_recovery_arm=False,
            )
    with pytest.raises(Recovery.RecoveryEvidenceError, match="exactly two"):
        Recovery.parse_quiesced_incomplete_evidence_marker(
            raw + Recovery.canonical_json_line({"event": "extra"})
        )


def test_failure_revision_runner_is_not_registered_for_quiesced_replay(
    tmp_path, monkeypatch
):
    fixture = _quiesced_incomplete_evidence_fixture(
        tmp_path,
        monkeypatch,
        source_sha256=FAILURE_REVISION_HISTORICAL_RUNNER_SOURCE_SHA256,
    )
    with pytest.raises(
        Recovery.RecoveryEvidenceError, match="runner receipt"
    ):
        Recovery.parse_quiesced_incomplete_evidence_marker(
            fixture["marker"].read_bytes(), require_recovery_arm=False
        )


def test_quiesced_pre_workspace_parser_requires_exact_all_none_profile(
    tmp_path, monkeypatch
):
    fixture = _quiesced_pre_workspace_fixture(tmp_path, monkeypatch)
    raw = fixture["marker"].read_bytes()
    parsed = Recovery.parse_quiesced_incomplete_evidence_marker(
        raw, require_recovery_arm=False
    )
    assert all(
        parsed.unquiesced[field] is None
        for field in (
            "workspace",
            "protected",
            "transcript",
            "workspace_identity",
            "protected_identity",
        )
    )

    rows = _canonical_rows(fixture["marker"])
    for field, value in (
        ("workspace", "partial"),
        ("protected", "partial"),
        ("workspace_identity", [1, 2]),
    ):
        changed = copy.deepcopy(rows)
        changed[1][field] = value
        with pytest.raises(Recovery.RecoveryEvidenceError):
            Recovery.parse_quiesced_incomplete_evidence_marker(
                b"".join(
                    Recovery.canonical_json_line(row) for row in changed
                ),
                require_recovery_arm=False,
            )


def test_quiesced_pre_workspace_recovery_is_noncounting_and_idempotent(
    tmp_path, monkeypatch
):
    fixture = _quiesced_pre_workspace_fixture(tmp_path, monkeypatch)
    first = R._recover_quiesced_incomplete_evidence(
        fixture["item"], confirm_dispatch_id=fixture["dispatch_id"]
    )
    assert first["result"] == "infrastructure_noncounting"
    assert first["quiesced_incomplete_evidence_replayed"] is False
    assert not fixture["marker"].exists()
    assert not fixture["capsule"].exists()
    assert not fixture["attempt"].exists()
    assert (
        fixture["wip"] / "latest.json"
    ).read_bytes() == fixture["baseline_latest"]
    rows = R.Guard.read_ledger(fixture["ledger"])
    assert rows[0]["schema"] == R.QUIESCED_PRE_WORKSPACE_EVENT_SCHEMA
    assert rows[0]["workspace"] is None
    assert rows[0]["workspace_identity"] is None
    assert rows[0]["workspace_generation_absent"] is True
    assert rows[0]["protected_generation_absent"] is True
    assert R.Status.infrastructure_noncounting_events(rows) == [rows[0]]

    second = R._recover_quiesced_incomplete_evidence(
        fixture["item"], confirm_dispatch_id=fixture["dispatch_id"]
    )
    assert second["result"] == (
        "infrastructure_noncounting_already_completed"
    )


def test_quiesced_pre_workspace_recovery_rejects_generation_appearance(
    tmp_path, monkeypatch
):
    fixture = _quiesced_pre_workspace_fixture(tmp_path, monkeypatch)
    appeared = fixture["scratch"] / (
        R._dispatch_workspace_prefix(fixture["item"]) + "appeared"
    )
    appeared.mkdir()
    with pytest.raises(R.CampaignPlanError, match="generation appeared"):
        R._recover_quiesced_incomplete_evidence(
            fixture["item"], confirm_dispatch_id=fixture["dispatch_id"]
        )
    assert fixture["marker"].exists()
    assert fixture["capsule"].exists()


def test_quiesced_incomplete_evidence_recovery_is_noncounting_and_minimal(
    tmp_path, monkeypatch
):
    fixture = _quiesced_incomplete_evidence_fixture(tmp_path, monkeypatch)
    result = R._recover_quiesced_incomplete_evidence(
        fixture["item"], confirm_dispatch_id=fixture["dispatch_id"]
    )
    assert result["result"] == "infrastructure_noncounting"
    assert result["quiesced_incomplete_evidence_replayed"] is False
    assert not fixture["workspace"].exists()
    assert not fixture["protected"].exists()
    assert not fixture["marker"].exists()
    assert not fixture["capsule"].exists()
    assert not fixture["attempt"].exists()
    assert (
        fixture["wip"] / "latest.json"
    ).read_bytes() == fixture["baseline_latest"]
    assert fixture["canonical"].read_bytes() == b"sealed canonical baseline\n"
    rows = R.Guard.read_ledger(fixture["ledger"])
    assert [row["event"] for row in rows] == [
        R.ZERO_LEDGER_EVENT,
        "codex_dispatch_release_authorized",
    ]
    assert rows[0]["schema"] == (
        R.QUIESCED_INCOMPLETE_EVIDENCE_EVENT_SCHEMA
    )
    assert "transcript" not in rows[0]
    assert "protected_identity" not in rows[0]
    assert R.Status.infrastructure_noncounting_events(rows) == [rows[0]]


@pytest.mark.parametrize(
    ("mutation", "pattern"),
    (
        ("ledger", "ledger suffix is ambiguous"),
        ("protected", "protected namespace appeared"),
        ("workspace_identity", "workspace identity changed"),
        ("canonical", "sealed baseline"),
        ("capsule", "capsule"),
    ),
)
def test_quiesced_incomplete_evidence_recovery_rejects_mutated_authority(
    tmp_path, monkeypatch, mutation, pattern
):
    fixture = _quiesced_incomplete_evidence_fixture(tmp_path, monkeypatch)
    if mutation == "ledger":
        R.Guard.append_ledger({"event": "unrelated"}, fixture["ledger"])
    elif mutation == "protected":
        fixture["protected"].mkdir()
    elif mutation == "workspace_identity":
        original = fixture["workspace"].with_name("original-workspace")
        fixture["workspace"].rename(original)
        fixture["workspace"].mkdir()
        lock = fixture["workspace"] / ".orchestrate.lock"
        lock.write_bytes(b"")
        os.chmod(lock, 0o600)
    elif mutation == "canonical":
        fixture["canonical"].write_bytes(b"mutated canonical\n")
    else:
        with fixture["capsule"].open("ab") as handle:
            handle.write(b"x")
    with pytest.raises(R.CampaignPlanError, match=pattern):
        R._recover_quiesced_incomplete_evidence(
            fixture["item"], confirm_dispatch_id=fixture["dispatch_id"]
        )
    assert fixture["marker"].exists()
    assert all(
        row.get("schema")
        != R.QUIESCED_INCOMPLETE_EVIDENCE_EVENT_SCHEMA
        for row in R.Guard.read_ledger(fixture["ledger"])
    )


def test_quiesced_incomplete_evidence_recovery_rejects_active_lock(
    tmp_path, monkeypatch
):
    fixture = _quiesced_incomplete_evidence_fixture(tmp_path, monkeypatch)
    held = fixture["exact_lock"].open("rb")
    fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(R.CampaignPlanError, match="lock remains active"):
            R._recover_quiesced_incomplete_evidence(
                fixture["item"],
                confirm_dispatch_id=fixture["dispatch_id"],
            )
    finally:
        fcntl.flock(held.fileno(), fcntl.LOCK_UN)
        held.close()


def test_quiesced_incomplete_evidence_recovery_rejects_live_scheduler_pid(
    tmp_path, monkeypatch
):
    real_observe = R._observe_quiesced_incomplete_evidence_absence
    fixture = _quiesced_incomplete_evidence_fixture(tmp_path, monkeypatch)
    rows = _canonical_rows(fixture["marker"])
    failure_at = datetime.fromisoformat(rows[1]["recorded_at"])
    monkeypatch.setattr(
        R, "_current_boot_started_at", lambda: failure_at.replace(
            year=failure_at.year - 1
        )
    )
    monkeypatch.setattr(
        R.Contiguous,
        "_process_identity",
        lambda _pid: (1, 1, 1, "R"),
    )
    monkeypatch.setattr(
        R, "_observe_quiesced_incomplete_evidence_absence", real_observe
    )
    with pytest.raises(R.CampaignPlanError, match="scheduler PID remains live"):
        R._recover_quiesced_incomplete_evidence(
            fixture["item"], confirm_dispatch_id=fixture["dispatch_id"]
        )


def test_quiesced_incomplete_evidence_recovery_resumes_after_event_append(
    tmp_path, monkeypatch
):
    fixture = _quiesced_incomplete_evidence_fixture(tmp_path, monkeypatch)
    real_restore = R._restore_wip_from_rollback_capsule

    def crash_before_restore(*_args, **_kwargs):
        raise RuntimeError("synthetic crash after infrastructure append")

    monkeypatch.setattr(
        R, "_restore_wip_from_rollback_capsule", crash_before_restore
    )
    with pytest.raises(RuntimeError, match="after infrastructure append"):
        R._recover_quiesced_incomplete_evidence(
            fixture["item"], confirm_dispatch_id=fixture["dispatch_id"]
        )
    rows = R.Guard.read_ledger(fixture["ledger"])
    assert [row["schema"] for row in rows] == [
        R.QUIESCED_INCOMPLETE_EVIDENCE_EVENT_SCHEMA
    ]
    assert fixture["workspace"].exists()
    monkeypatch.setattr(
        R, "_restore_wip_from_rollback_capsule", real_restore
    )
    result = R._recover_quiesced_incomplete_evidence(
        fixture["item"], confirm_dispatch_id=fixture["dispatch_id"]
    )
    assert result["quiesced_incomplete_evidence_replayed"] is True
    assert not fixture["workspace"].exists()


def test_quiesced_incomplete_evidence_recovery_resumes_workspace_tombstone(
    tmp_path, monkeypatch
):
    fixture = _quiesced_incomplete_evidence_fixture(tmp_path, monkeypatch)
    real_rmtree = R.shutil.rmtree
    crashed = False

    def crash_after_rename(path, *args, **kwargs):
        nonlocal crashed
        if not crashed and Path(path).name.startswith(
            ".post_reboot_cleanup_"
        ):
            crashed = True
            raise OSError("synthetic tombstone crash")
        return real_rmtree(path, *args, **kwargs)

    crash_after_rename.avoids_symlink_attacks = getattr(
        real_rmtree, "avoids_symlink_attacks", False
    )
    monkeypatch.setattr(R.shutil, "rmtree", crash_after_rename)
    with pytest.raises(R.CampaignPlanError, match="workspace cleanup failed"):
        R._recover_quiesced_incomplete_evidence(
            fixture["item"], confirm_dispatch_id=fixture["dispatch_id"]
        )
    tombstones = list(fixture["scratch"].glob(".post_reboot_cleanup_*"))
    assert len(tombstones) == 1
    monkeypatch.setattr(R.shutil, "rmtree", real_rmtree)
    result = R._recover_quiesced_incomplete_evidence(
        fixture["item"], confirm_dispatch_id=fixture["dispatch_id"]
    )
    assert result["quiesced_incomplete_evidence_replayed"] is True
    assert not tombstones[0].exists()


@pytest.mark.parametrize(
    "wal_crash", ("preparing", "pre_authority", "partial_authority")
)
def test_quiesced_incomplete_evidence_retires_unauthorized_release_wal(
    tmp_path, monkeypatch, wal_crash
):
    fixture = _quiesced_incomplete_evidence_fixture(tmp_path, monkeypatch)
    real_replace = R.os.replace
    real_ensure = R._ensure_dispatch_release_authority_row

    if wal_crash == "preparing":
        def crash_before_intent_install(source, target, *args, **kwargs):
            if str(source).endswith(".release_preparing") and str(
                target
            ).endswith(".release_intent"):
                raise OSError(errno.EIO, "synthetic preparing WAL crash")
            return real_replace(source, target, *args, **kwargs)

        monkeypatch.setattr(R.os, "replace", crash_before_intent_install)
    elif wal_crash == "pre_authority":
        def crash_before_authority(
            item,
            root_fd,
            record,
            intent_identity,
            *,
            allow_new_authority_append=False,
            **kwargs,
        ):
            if allow_new_authority_append:
                raise RuntimeError("synthetic pre-authority WAL crash")
            return real_ensure(
                item,
                root_fd,
                record,
                intent_identity,
                allow_new_authority_append=False,
                **kwargs,
            )

        monkeypatch.setattr(
            R, "_ensure_dispatch_release_authority_row", crash_before_authority
        )
    else:
        def crash_during_authority(
            item,
            root_fd,
            record,
            intent_identity,
            *,
            allow_new_authority_append=False,
            **kwargs,
        ):
            if not allow_new_authority_append:
                return real_ensure(
                    item,
                    root_fd,
                    record,
                    intent_identity,
                    allow_new_authority_append=False,
                    **kwargs,
                )
            authority = record["release_authority"]
            line = Recovery.canonical_json_line(
                authority["authority_record"]
            )
            descriptor = os.open(
                authority["ledger"], os.O_WRONLY | os.O_APPEND
            )
            try:
                os.write(descriptor, line[:len(line) // 2])
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            raise RuntimeError("synthetic partial-authority WAL crash")

        monkeypatch.setattr(
            R, "_ensure_dispatch_release_authority_row", crash_during_authority
        )

    with pytest.raises((RuntimeError, R.CampaignPlanError)):
        R._recover_quiesced_incomplete_evidence(
            fixture["item"], confirm_dispatch_id=fixture["dispatch_id"]
        )
    assert fixture["marker"].exists()
    assert fixture["capsule"].exists()
    assert not fixture["workspace"].exists()

    monkeypatch.setattr(R.os, "replace", real_replace)
    monkeypatch.setattr(
        R, "_ensure_dispatch_release_authority_row", real_ensure
    )
    result = R._recover_quiesced_incomplete_evidence(
        fixture["item"], confirm_dispatch_id=fixture["dispatch_id"]
    )
    assert result["result"] == "infrastructure_noncounting"
    assert not fixture["marker"].exists()
    assert not fixture["capsule"].exists()
    assert [row["event"] for row in R.Guard.read_ledger(
        fixture["ledger"]
    )] == [R.ZERO_LEDGER_EVENT, "codex_dispatch_release_authorized"]


def test_quiesced_incomplete_evidence_reconciles_authorized_release_wal(
    tmp_path, monkeypatch
):
    fixture = _quiesced_incomplete_evidence_fixture(tmp_path, monkeypatch)
    real_finish = R._finish_dispatch_release_intent

    def crash_after_authority(*_args, **_kwargs):
        raise RuntimeError("synthetic crash after release authority")

    monkeypatch.setattr(
        R, "_finish_dispatch_release_intent", crash_after_authority
    )
    with pytest.raises(RuntimeError, match="after release authority"):
        R._recover_quiesced_incomplete_evidence(
            fixture["item"], confirm_dispatch_id=fixture["dispatch_id"]
        )
    rows = R.Guard.read_ledger(fixture["ledger"])
    assert [row["event"] for row in rows] == [
        R.ZERO_LEDGER_EVENT,
        "codex_dispatch_release_authorized",
    ]
    assert fixture["marker"].exists()
    monkeypatch.setattr(R, "_finish_dispatch_release_intent", real_finish)
    result = R._recover_quiesced_incomplete_evidence(
        fixture["item"], confirm_dispatch_id=fixture["dispatch_id"]
    )
    assert result["result"] == "infrastructure_noncounting_already_completed"
    assert not fixture["marker"].exists()
    assert not fixture["capsule"].exists()


def test_quiesced_incomplete_authorized_replay_checks_state_before_deletion(
    tmp_path, monkeypatch
):
    fixture = _quiesced_incomplete_evidence_fixture(tmp_path, monkeypatch)
    real_finish = R._finish_dispatch_release_intent

    def crash_after_authority(*_args, **_kwargs):
        raise RuntimeError("synthetic authorized WAL crash")

    monkeypatch.setattr(
        R, "_finish_dispatch_release_intent", crash_after_authority
    )
    with pytest.raises(RuntimeError, match="authorized WAL crash"):
        R._recover_quiesced_incomplete_evidence(
            fixture["item"], confirm_dispatch_id=fixture["dispatch_id"]
        )
    fixture["canonical"].write_bytes(b"mutation after authority\n")
    monkeypatch.setattr(R, "_finish_dispatch_release_intent", real_finish)
    with pytest.raises(
        R.CampaignPlanError, match="restored baseline changed"
    ):
        R._recover_quiesced_incomplete_evidence(
            fixture["item"], confirm_dispatch_id=fixture["dispatch_id"]
        )
    assert fixture["marker"].exists()
    assert fixture["capsule"].exists()


def test_quiesced_incomplete_evidence_markerless_completion_is_idempotent(
    tmp_path, monkeypatch
):
    fixture = _quiesced_incomplete_evidence_fixture(tmp_path, monkeypatch)
    first = R._recover_quiesced_incomplete_evidence(
        fixture["item"], confirm_dispatch_id=fixture["dispatch_id"]
    )
    assert first["result"] == "infrastructure_noncounting"
    sealed = fixture["ledger"].read_bytes()
    second = R._recover_quiesced_incomplete_evidence(
        fixture["item"], confirm_dispatch_id=fixture["dispatch_id"]
    )
    assert second["result"] == "infrastructure_noncounting_already_completed"
    assert fixture["ledger"].read_bytes() == sealed


def test_quiesced_incomplete_evidence_final_cas_rejects_mutated_baseline(
    tmp_path, monkeypatch
):
    fixture = _quiesced_incomplete_evidence_fixture(tmp_path, monkeypatch)
    real_ensure = R._ensure_dispatch_release_authority_row
    mutated = False

    def mutate_before_final_cas(
        item,
        root_fd,
        record,
        intent_identity,
        *,
        allow_new_authority_append=False,
        **kwargs,
    ):
        nonlocal mutated
        if allow_new_authority_append and not mutated:
            mutated = True
            fixture["canonical"].write_bytes(b"late canonical mutation\n")
        return real_ensure(
            item,
            root_fd,
            record,
            intent_identity,
            allow_new_authority_append=allow_new_authority_append,
            **kwargs,
        )

    monkeypatch.setattr(
        R, "_ensure_dispatch_release_authority_row", mutate_before_final_cas
    )
    with pytest.raises(R.CampaignPlanError, match="sealed baseline"):
        R._recover_quiesced_incomplete_evidence(
            fixture["item"], confirm_dispatch_id=fixture["dispatch_id"]
        )
    assert mutated is True
    assert all(
        row.get("event") != "codex_dispatch_release_authorized"
        for row in R.Guard.read_ledger(fixture["ledger"])
    )
    assert fixture["marker"].exists()


def test_quiesced_incomplete_post_authority_gate_preserves_evidence_on_mutation(
    tmp_path, monkeypatch
):
    fixture = _quiesced_incomplete_evidence_fixture(tmp_path, monkeypatch)
    real_finish = R._finish_dispatch_release_intent
    mutated = False

    def mutate_after_authority(
        item, root_fd, record, intent_identity, **kwargs
    ):
        nonlocal mutated
        if not mutated:
            mutated = True
            fixture["canonical"].write_bytes(
                b"mutation after authority append\n"
            )
        return real_finish(
            item, root_fd, record, intent_identity, **kwargs
        )

    monkeypatch.setattr(
        R, "_finish_dispatch_release_intent", mutate_after_authority
    )
    with pytest.raises(R.CampaignPlanError, match="sealed baseline"):
        R._recover_quiesced_incomplete_evidence(
            fixture["item"], confirm_dispatch_id=fixture["dispatch_id"]
        )
    assert mutated is True
    assert fixture["marker"].exists()
    assert fixture["capsule"].exists()
    assert [row["event"] for row in R.Guard.read_ledger(
        fixture["ledger"]
    )] == [R.ZERO_LEDGER_EVENT, "codex_dispatch_release_authorized"]


def test_quiesced_incomplete_evidence_wrong_dispatch_keeps_quarantine(
    tmp_path, monkeypatch
):
    fixture = _quiesced_incomplete_evidence_fixture(tmp_path, monkeypatch)
    with pytest.raises(R.CampaignPlanError, match="operator confirmation"):
        R._recover_quiesced_incomplete_evidence(
            fixture["item"], confirm_dispatch_id="0" * 32
        )
    assert fixture["marker"].exists()
    assert fixture["workspace"].exists()
    assert R.Guard.read_ledger(fixture["ledger"]) == []


def _append_sandbox_exec(fixture, *, interrupted_exec=False):
    item = fixture["item"]
    record = {
        "event": "codex_exec",
        "started_at": "2026-08-07T23:00:00+00:00",
        "thread_id": "sandbox-exec-thread",
        "transcript": fixture["transcript_name"],
        "workspace": fixture["workspace_name"],
        "game": item["game"],
        "target_level": item["target_level"],
        "run_label": f"{item['game']}:L{item['target_level']}:propose",
        "model": "gpt-5.6-sol",
        "reasoning_effort": item["effort"],
        "minutes_limit": item["minutes"],
        "allocation_policy": "drain",
        "reached": item["reached"],
        "parent_action_count": item["parent_action_count"],
        **{
            field: item[field]
            for field in R.Status.FRONTIER_BINDING_FIELDS
        },
        "returncode": 0,
        "failure_class": None,
        "timed_out": False,
        "interrupted": interrupted_exec,
    }
    if interrupted_exec:
        protected = fixture["protected"] / fixture["transcript_name"]
        record.update({
            "allocation_expired": False,
            "surviving_process_group": False,
            "public_action_protocol_violation": False,
            "protected_transcript_status": "sealed",
            "protected_transcript_error": None,
            "protected_transcript_sha256": hashlib.sha256(
                protected.read_bytes()
            ).hexdigest(),
            "launch_error": None,
            "postflight_error": None,
            "failure_detail_class": None,
            "terminal_errors": [],
        })
    R.Guard.append_ledger(record, fixture["ledger"])
    return record


def _sandbox_generation_content_path(fixture, tree_kind):
    if tree_kind == "workspace":
        return fixture["workspace"] / "attempt.txt"
    assert tree_kind == "protected"
    return fixture["protected"] / fixture["transcript_name"]


@pytest.mark.parametrize("mutation", ("add", "remove", "content"))
def test_generation_tree_observation_rejects_nested_concurrent_mutation(
    tmp_path, monkeypatch, mutation
):
    root = tmp_path / "opaque_generation"
    nested = root / "nested"
    nested.mkdir(parents=True)
    trigger = nested / "a_trigger.txt"
    target = nested / "z_target.txt"
    trigger.write_bytes(b"opaque trigger\n")
    target.write_bytes(b"opaque baseline\n")
    real_read = R.Legs._read_single_link_regular
    injected = False

    def read_then_mutate(selected):
        nonlocal injected
        payload = real_read(selected)
        if Path(selected) == trigger and not injected:
            injected = True
            if mutation == "add":
                (nested / "m_added.txt").write_bytes(b"concurrent add\n")
            elif mutation == "remove":
                target.unlink()
            else:
                target.write_bytes(b"concurrent content change\n")
        return payload

    monkeypatch.setattr(
        R.Legs, "_read_single_link_regular", read_then_mutate
    )
    with pytest.raises(
        R.CampaignPlanError,
        match="opaque generation.*(changed|unstable)",
    ):
        R._generation_tree_observation_sha256(
            root, label="opaque generation"
        )
    assert injected is True


def _arm_sandboxed_generation(fixture, monkeypatch):
    def absent(**_kwargs):
        observed = datetime.now(timezone.utc).isoformat()
        return {
            "absence_sample_count": R.SANDBOX_ABSENCE_SAMPLES,
            "absence_window_ns": 1,
            "absence_first_at": observed,
            "absence_last_at": observed,
        }

    monkeypatch.setattr(R, "_observe_named_root_group_absence", absent)
    return R._arm_sandboxed_generation_release(
        fixture["item"],
        confirm_dispatch_id=fixture["dispatch_id"],
        boot_identity_provider=lambda: fixture["boot"],
    )


def _interrupted_generation_fixture(
    tmp_path, monkeypatch, *, boundary_finding_counts=None
):
    fixture = _sandboxed_generation_fixture(
        tmp_path,
        monkeypatch,
        interrupted_exec=True,
        boundary_finding_counts=boundary_finding_counts,
    )
    fixture["execution"] = _append_sandbox_exec(
        fixture, interrupted_exec=True
    )
    return fixture


def _install_fixture_canonical_transition(fixture, monkeypatch):
    """Pin one synthetic metadata-only drift to exercise the migration lane."""

    armed = _canonical_rows(fixture["marker"])[0]
    target = fixture["artifact"] / "solver.py"
    metadata = target.stat(follow_symlinks=False)
    os.utime(
        target,
        ns=(metadata.st_atime_ns, metadata.st_mtime_ns + 1),
    )
    observed = R._capture_canonical_rollback(fixture["item"]).digest
    historical = armed["historical_runner"]
    frontier = armed["frontier_binding"]
    pin = Recovery.AuditedCanonicalDigestTransition(
        transition_id="synthetic-metadata-refresh-v1",
        dispatch_id=fixture["dispatch_id"],
        game=armed["game"],
        target_level=armed["target_level"],
        reached=frontier["reached"],
        parent_action_count=frontier["parent_action_count"],
        retry_complexity_n=armed["retry_complexity_n"],
        projected_item_sha256=armed["projected_item_sha256"],
        runner_source_sha256=historical["source_sha256"],
        runner_head_commit=historical["head_commit"],
        canonical_root_identity=tuple(armed["canonical_root_identity"]),
        canonical_digest=armed["canonical_digest"],
        observed_canonical_digest=observed,
        frontier_binding_schema=frontier["frontier_binding_schema"],
        parent_checkpoint_sha256=frontier["parent_checkpoint_sha256"],
        parent_source_tree_sha256=frontier["parent_source_tree_sha256"],
        frontier_sha256=frontier["frontier_sha256"],
    )
    monkeypatch.setattr(
        Recovery,
        "AUDITED_CANONICAL_DIGEST_TRANSITIONS",
        {fixture["dispatch_id"]: pin},
    )
    return pin


def _arm_interrupted_generation(fixture, monkeypatch):
    def absent(_parsed):
        observed = datetime.now(timezone.utc).isoformat()
        return {
            "absence_sample_count": R.SANDBOX_ABSENCE_SAMPLES,
            "absence_window_ns": 1,
            "absence_first_at": observed,
            "absence_last_at": observed,
        }

    monkeypatch.setattr(R, "_observe_interrupted_scheduler_absence", absent)
    return R._arm_interrupted_generation_release(
        fixture["item"],
        confirm_dispatch_id=fixture["dispatch_id"],
        boot_identity_provider=lambda: fixture["boot"],
    )


def _recover_interrupted_generation(fixture, armed):
    return R._recover_interrupted_generation_release(
        fixture["item"],
        confirm_dispatch_id=fixture["dispatch_id"],
        confirm_recovery_nonce=armed["recovery_nonce"],
        boot_identity_provider=lambda: fixture["boot"],
    )


def _armed_one_exec_crash_fixture(
    tmp_path, monkeypatch, interrupted_exec, *, canonical_transition=False
):
    if interrupted_exec:
        fixture = _interrupted_generation_fixture(tmp_path, monkeypatch)
        if canonical_transition:
            _install_fixture_canonical_transition(fixture, monkeypatch)
        armed = _arm_interrupted_generation(fixture, monkeypatch)
        recover = lambda: _recover_interrupted_generation(fixture, armed)
    else:
        fixture = _sandboxed_generation_fixture(tmp_path, monkeypatch)
        _append_sandbox_exec(fixture)
        armed = _arm_sandboxed_generation(fixture, monkeypatch)
        recover = lambda: R._recover_sandboxed_generation_release(
            fixture["item"],
            confirm_dispatch_id=fixture["dispatch_id"],
            confirm_recovery_nonce=armed["recovery_nonce"],
            boot_identity_provider=lambda: fixture["boot"],
        )
    return fixture, armed, recover


@pytest.mark.parametrize(
    "counts", (None, {"dynamic_execution": 1})
)
def test_interrupted_generation_parser_accepts_exact_legacy_and_counted_rows(
    tmp_path, monkeypatch, counts
):
    fixture = _interrupted_generation_fixture(
        tmp_path, monkeypatch, boundary_finding_counts=counts
    )
    parsed = Recovery.parse_interrupted_generation_marker(
        fixture["marker"].read_bytes(), require_recovery_arm=False
    )
    assert parsed.unquiesced.get("boundary_finding_counts") == counts
    with pytest.raises(Recovery.RecoveryEvidenceError):
        Recovery.parse_sandboxed_generation_marker(
            fixture["marker"].read_bytes(), require_recovery_arm=False
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("event", "dispatch_failed"),
        ("child_returncode", -signal.SIGKILL),
        ("child_pid", 424242),
        ("boundary_finding_counts", {}),
        ("boundary_finding_counts", {"dynamic_execution": 2}),
        ("boundary_finding_counts", {"detached_process_escape": 1}),
        ("boundary_finding_counts", {"shell_or_subprocess_escape": 1}),
        ("boundary_finding_counts", {
            "dynamic_execution": 1,
            "detached_process_escape": 1,
        }),
    ),
)
def test_interrupted_generation_parser_rejects_nearby_marker_profiles(
    tmp_path, monkeypatch, field, value
):
    fixture = _interrupted_generation_fixture(tmp_path, monkeypatch)
    rows = _canonical_rows(fixture["marker"])
    rows[1][field] = value
    raw = b"".join(Recovery.canonical_json_line(row) for row in rows)
    with pytest.raises(Recovery.RecoveryEvidenceError):
        Recovery.parse_interrupted_generation_marker(
            raw, require_recovery_arm=False
        )


def test_interrupted_generation_parser_does_not_retrofit_v3_pin(
    tmp_path, monkeypatch
):
    fixture = _interrupted_generation_fixture(tmp_path, monkeypatch)
    rows = _canonical_rows(fixture["marker"])
    historical = rows[0]["historical_runner"]
    historical["source_sha256"] = BOUNDARY_V3_HISTORICAL_RUNNER_SOURCE_SHA256
    historical["head_commit"] = BOUNDARY_V3_HISTORICAL_RUNNER_HEAD
    raw = b"".join(Recovery.canonical_json_line(row) for row in rows)
    with pytest.raises(Recovery.RecoveryEvidenceError, match="exact v2 runner"):
        Recovery.parse_interrupted_generation_marker(
            raw, require_recovery_arm=False
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("returncode", 1),
        ("interrupted", False),
        ("timed_out", True),
        ("allocation_expired", True),
        ("surviving_process_group", True),
        ("public_action_protocol_violation", True),
        ("protected_transcript_status", "missing"),
        ("terminal_errors", ["synthetic"]),
    ),
)
def test_interrupted_generation_arm_rejects_nearby_exec_profiles(
    tmp_path, monkeypatch, field, value
):
    fixture = _interrupted_generation_fixture(tmp_path, monkeypatch)
    rows = R.Guard.read_ledger(fixture["ledger"])
    rows[0][field] = value
    _write_canonical_rows(fixture["ledger"], rows)
    marker_before = fixture["marker"].read_bytes()
    with pytest.raises(R.CampaignPlanError, match="exact sealed exec-zero"):
        _arm_interrupted_generation(fixture, monkeypatch)
    assert fixture["marker"].read_bytes() == marker_before
    assert len(_canonical_rows(fixture["marker"])) == 2


def test_interrupted_generation_rejects_arm_contract_before_wal(
    tmp_path, monkeypatch
):
    fixture = _interrupted_generation_fixture(tmp_path, monkeypatch)
    armed = _arm_interrupted_generation(fixture, monkeypatch)
    rows = _canonical_rows(fixture["marker"])
    rows[2]["sandbox_contract_sha256"] = "f" * 64
    _write_canonical_rows(fixture["marker"], rows)
    ledger_before = fixture["ledger"].read_bytes()
    with pytest.raises(R.CampaignPlanError, match="arm schema"):
        _recover_interrupted_generation(fixture, armed)
    assert fixture["ledger"].read_bytes() == ledger_before
    assert fixture["marker"].exists()
    assert fixture["capsule"].exists()


@pytest.mark.parametrize("drift", ("transcript", "clean", "different"))
def test_interrupted_generation_arm_rejects_evidence_drift(
    tmp_path, monkeypatch, drift
):
    fixture = _interrupted_generation_fixture(tmp_path, monkeypatch)
    if drift == "transcript":
        (fixture["protected"] / fixture["transcript_name"]).write_bytes(
            b'{"type":"tampered"}\n'
        )
    elif drift == "clean":
        (fixture["workspace"] / "dynamic_probe.py").unlink()
    else:
        (fixture["workspace"] / "second_dynamic.py").write_bytes(
            b'exec("pass")\n'
        )
    with pytest.raises(R.CampaignPlanError):
        _arm_interrupted_generation(fixture, monkeypatch)
    assert len(_canonical_rows(fixture["marker"])) == 2
    assert len(R.Guard.read_ledger(fixture["ledger"])) == 1


def test_interrupted_generation_observer_is_darwin_only(
    tmp_path, monkeypatch
):
    fixture = _interrupted_generation_fixture(tmp_path, monkeypatch)
    parsed = Recovery.parse_interrupted_generation_marker(
        fixture["marker"].read_bytes(), require_recovery_arm=False
    )
    monkeypatch.setattr(R.sys, "platform", "linux")
    with pytest.raises(R.CampaignPlanError, match="Darwin incident"):
        R._observe_interrupted_scheduler_absence(parsed)


def test_interrupted_generation_arm_rejects_live_scheduler_and_active_lock(
    tmp_path, monkeypatch
):
    fixture = _interrupted_generation_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        R,
        "_observe_interrupted_scheduler_absence",
        lambda _parsed: (_ for _ in ()).throw(
            R.CampaignPlanError(
                "interrupted-generation scheduler PID remains live"
            )
        ),
    )
    with pytest.raises(R.CampaignPlanError, match="scheduler PID remains live"):
        R._arm_interrupted_generation_release(
            fixture["item"],
            confirm_dispatch_id=fixture["dispatch_id"],
            boot_identity_provider=lambda: fixture["boot"],
        )

    monkeypatch.setattr(
        R,
        "_observe_interrupted_scheduler_absence",
        lambda _parsed: {
            "absence_sample_count": R.SANDBOX_ABSENCE_SAMPLES,
            "absence_window_ns": 1,
            "absence_first_at": datetime.now(timezone.utc).isoformat(),
            "absence_last_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    lock = fixture["exact_lock"].open("r+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(R.CampaignPlanError, match="workspace lock remains active"):
            R._arm_interrupted_generation_release(
                fixture["item"],
                confirm_dispatch_id=fixture["dispatch_id"],
                boot_identity_provider=lambda: fixture["boot"],
            )
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def test_interrupted_generation_arm_requires_full_taint_pass(
    tmp_path, monkeypatch
):
    fixture = _interrupted_generation_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        R,
        "_taint_gate",
        lambda: (_ for _ in ()).throw(
            R.CampaignPlanError("synthetic full taint failure")
        ),
    )
    with pytest.raises(R.CampaignPlanError, match="full taint failure"):
        _arm_interrupted_generation(fixture, monkeypatch)
    assert len(_canonical_rows(fixture["marker"])) == 2


@pytest.mark.parametrize(
    "drift", ("payload", "inventory", "mode", "mtime", "xattr")
)
def test_interrupted_canonical_transition_rejects_every_post_pin_drift(
    tmp_path, monkeypatch, drift
):
    fixture = _interrupted_generation_fixture(tmp_path, monkeypatch)
    _install_fixture_canonical_transition(fixture, monkeypatch)
    target = fixture["artifact"] / "solver.py"
    if drift == "payload":
        target.write_bytes(b"changed after audited observation\n")
    elif drift == "inventory":
        (fixture["artifact"] / "unexpected.txt").write_bytes(b"drift\n")
    elif drift == "mode":
        os.chmod(target, stat.S_IMODE(target.stat().st_mode) ^ stat.S_IXUSR)
    elif drift == "mtime":
        metadata = target.stat(follow_symlinks=False)
        os.utime(
            target,
            ns=(metadata.st_atime_ns, metadata.st_mtime_ns + 1),
        )
    else:
        attribute = (
            "com.gkm.scheduler-test"
            if sys.platform == "darwin"
            else "user.gkm_scheduler_test"
        )
        try:
            os.setxattr(
                target,
                attribute,
                b"changed after audited observation",
                follow_symlinks=False,
            )
        except (AttributeError, OSError):
            pytest.skip("extended attributes are unavailable")

    marker_before = fixture["marker"].read_bytes()
    with pytest.raises(
        R.CampaignPlanError,
        match="canonical/frontier baseline changed",
    ):
        _arm_interrupted_generation(fixture, monkeypatch)
    assert fixture["marker"].read_bytes() == marker_before
    assert len(R.Guard.read_ledger(fixture["ledger"])) == 1


def test_interrupted_generation_has_no_generic_canonical_transition(
    tmp_path, monkeypatch
):
    fixture = _interrupted_generation_fixture(tmp_path, monkeypatch)
    target = fixture["artifact"] / "solver.py"
    metadata = target.stat(follow_symlinks=False)
    os.utime(
        target,
        ns=(metadata.st_atime_ns, metadata.st_mtime_ns + 1),
    )
    with pytest.raises(
        R.CampaignPlanError,
        match="canonical/frontier baseline changed",
    ):
        _arm_interrupted_generation(fixture, monkeypatch)
    assert len(_canonical_rows(fixture["marker"])) == 2


def test_interrupted_canonical_transition_parser_rejects_arm_neighbours(
    tmp_path, monkeypatch
):
    fixture = _interrupted_generation_fixture(tmp_path, monkeypatch)
    pin = _install_fixture_canonical_transition(fixture, monkeypatch)
    _arm_interrupted_generation(fixture, monkeypatch)
    rows = _canonical_rows(fixture["marker"])
    mutations = (
        ("observed_canonical_digest", "f" * 64),
        ("canonical_digest", "e" * 64),
        ("canonical_digest_transition_schema", "nearby-transition-v1"),
        ("canonical_digest_transition_id", "nearby-transition-v1"),
        ("projected_item_sha256", "d" * 64),
    )
    for field, value in mutations:
        changed = copy.deepcopy(rows)
        changed[2][field] = value
        raw = b"".join(
            Recovery.canonical_json_line(row) for row in changed
        )
        with pytest.raises(Recovery.RecoveryEvidenceError):
            Recovery.parse_interrupted_generation_marker(
                raw, require_recovery_arm=True
            )
    assert rows[2]["canonical_digest"] == pin.canonical_digest


def test_interrupted_canonical_transition_pin_binds_every_incident_coordinate(
    tmp_path, monkeypatch
):
    fixture = _interrupted_generation_fixture(tmp_path, monkeypatch)
    pin = _install_fixture_canonical_transition(fixture, monkeypatch)
    armed = _canonical_rows(fixture["marker"])[0]
    assert Recovery.audited_canonical_digest_transition(
        armed, pin.observed_canonical_digest
    ) == pin

    mutations = (
        (("dispatch_id",), "0" * 32),
        (("game",), "nearby"),
        (("target_level",), armed["target_level"] + 1),
        (("retry_complexity_n",), armed["retry_complexity_n"] + 1),
        (("projected_item_sha256",), "1" * 64),
        (("canonical_root_identity",), [1, 2]),
        (("canonical_digest",), "2" * 64),
        (("historical_runner", "source_sha256"), "3" * 64),
        (("historical_runner", "head_commit"), "4" * 40),
        (("frontier_binding", "game"), "nearby"),
        (("frontier_binding", "target_level"), 2),
        (("frontier_binding", "reached"), 1),
        (("frontier_binding", "parent_action_count"), 1),
        (("frontier_binding", "frontier_binding_schema"), 2),
        (("frontier_binding", "parent_checkpoint_sha256"), "5" * 64),
        (("frontier_binding", "parent_source_tree_sha256"), "6" * 64),
        (("frontier_binding", "frontier_sha256"), "7" * 64),
    )
    for path, value in mutations:
        changed = copy.deepcopy(armed)
        target = changed
        for component in path[:-1]:
            target = target[component]
        target[path[-1]] = value
        assert Recovery.audited_canonical_digest_transition(
            changed, pin.observed_canonical_digest
        ) is None
    assert Recovery.audited_canonical_digest_transition(
        armed, "8" * 64
    ) is None


@pytest.mark.parametrize("canonical_transition", (False, True))
def test_interrupted_generation_recovery_is_noncounting_and_idempotent(
    tmp_path, monkeypatch, canonical_transition
):
    fixture = _interrupted_generation_fixture(
        tmp_path,
        monkeypatch,
        boundary_finding_counts={"dynamic_execution": 1},
    )
    pin = (
        _install_fixture_canonical_transition(fixture, monkeypatch)
        if canonical_transition else None
    )
    baseline_latest = (fixture["wip"] / "latest.json").read_bytes()
    (fixture["wip"] / "latest.json").write_bytes(
        b'{"attempt":"interrupted"}\n'
    )
    disposable = fixture["wip"] / "interrupted.txt"
    disposable.write_bytes(b"restore from capsule\n")
    old_scratch = _sandbox_tree_snapshot(fixture["scratch"])
    armed = _arm_interrupted_generation(fixture, monkeypatch)
    nonce = armed["recovery_nonce"]
    installed_arm = _canonical_rows(fixture["marker"])[2]
    if pin is not None:
        assert installed_arm["recovery_arm_schema"] == (
            Recovery.INTERRUPTED_GENERATION_TRANSITION_ARM_SCHEMA
        )
        assert installed_arm["canonical_digest"] == pin.canonical_digest
        assert installed_arm["observed_canonical_digest"] == (
            pin.observed_canonical_digest
        )
    else:
        assert installed_arm["recovery_arm_schema"] == (
            Recovery.INTERRUPTED_GENERATION_ARM_SCHEMA
        )

    result = R._recover_interrupted_generation_release(
        fixture["item"],
        confirm_dispatch_id=fixture["dispatch_id"],
        confirm_recovery_nonce=nonce,
        boot_identity_provider=lambda: fixture["boot"],
    )

    assert result["result"] == "sandbox_isolated_noncounting"
    assert (fixture["wip"] / "latest.json").read_bytes() == baseline_latest
    assert not disposable.exists()
    assert _sandbox_tree_snapshot(fixture["scratch"]) == old_scratch
    assert not fixture["marker"].exists()
    assert not fixture["capsule"].exists()
    rows = R.Guard.read_ledger(fixture["ledger"])
    assert [row["event"] for row in rows] == [
        "codex_exec",
        "codex_exec_classification_correction",
        R.SANDBOX_ABANDON_EVENT,
        "codex_dispatch_release_authorized",
    ]
    correction, event = rows[1:3]
    assert correction["boundary_finding_counts"] == {"dynamic_execution": 1}
    assert event["boundary_finding_counts"] == {"dynamic_execution": 1}
    assert event["schema"] == (
        R.INTERRUPTED_EXEC_TRANSITION_ABANDON_EVENT_SCHEMA
        if canonical_transition
        else R.INTERRUPTED_EXEC_ABANDON_EVENT_SCHEMA
    )
    if pin is not None:
        assert event["canonical_digest"] == pin.canonical_digest
        assert (
            event["observed_canonical_digest"]
            == pin.observed_canonical_digest
        )
        assert event["canonical_digest_transition_id"] == pin.transition_id
        assert event["canonical_digest_transition_schema"] == (
            Recovery.CANONICAL_DIGEST_TRANSITION_SCHEMA
        )
    assert event["retry_increment"] == 0
    assert event["codex_exec_appended"] is True
    assert event["taint_verdict"] == "quarantined"
    assert rows[3]["terminal_kind"] == R.INTERRUPTED_RELEASE_AUTHORITY_KIND
    R._validate_sandbox_abandon_event(fixture["item"], event)
    changed_counts = dict(event)
    changed_counts["boundary_finding_counts"] = {
        "detached_process_escape": 1
    }
    with pytest.raises(R.CampaignPlanError, match="boundary findings"):
        R._validate_sandbox_abandon_event(fixture["item"], changed_counts)
    changed_tree = dict(correction)
    changed_tree["protected_tree_sha256"] = "f" * 64
    with pytest.raises(R.CampaignPlanError, match="evidence seal"):
        R._validate_sandbox_exec_classification(
            fixture["item"], rows[0], changed_tree, terminal=event
        )
    for field, value in (
        ("absence_window_ns", 60_000_000_001),
        ("absence_last_at", "2099-01-01T00:00:00+00:00"),
    ):
        changed_absence = dict(event)
        changed_absence[field] = value
        with pytest.raises(R.CampaignPlanError):
            R._validate_sandbox_abandon_event(
                fixture["item"], changed_absence
            )
    assert R.Status.infrastructure_noncounting_events(rows) == [event]
    if pin is not None:
        sealed_rows = copy.deepcopy(rows)
        sealed_ledger = fixture["ledger"].read_bytes()
        for field, value in (
            ("canonical_digest", "9" * 64),
            ("observed_canonical_digest", "f" * 64),
            ("canonical_root_identity", [1, 2]),
            ("projected_item_sha256", "e" * 64),
            ("historical_runner_source_sha256", "d" * 64),
            ("historical_runner_head_commit", "c" * 40),
            ("canonical_digest_transition_schema", "nearby-v1"),
            ("canonical_digest_transition_id", "nearby-v1"),
        ):
            changed_rows = copy.deepcopy(sealed_rows)
            changed_rows[2][field] = value
            _write_canonical_rows(fixture["ledger"], changed_rows)
            assert R.Status.infrastructure_noncounting_events(
                changed_rows
            ) == []
            with pytest.raises(R.CampaignPlanError):
                R._completed_sandbox_isolation_result(
                    fixture["item"],
                    confirm_dispatch_id=fixture["dispatch_id"],
                    confirm_recovery_nonce=nonce,
                    interrupted_exec=True,
                )
            fixture["ledger"].write_bytes(sealed_ledger)
    with pytest.raises(
        R.CampaignPlanError, match="abandoned sandbox namespace"
    ):
        R._reject_abandoned_scratch_root(
            fixture["scratch"], fixture["ledger"]
        )

    if pin is not None:
        monkeypatch.setattr(
            Recovery, "AUDITED_CANONICAL_DIGEST_TRANSITIONS", {}
        )
    sealed = fixture["ledger"].read_bytes()
    repeated = R._recover_interrupted_generation_release(
        fixture["item"],
        confirm_dispatch_id=fixture["dispatch_id"],
        confirm_recovery_nonce=nonce,
        boot_identity_provider=lambda: fixture["boot"],
    )
    assert repeated["result"] == (
        "interrupted_sandbox_isolated_noncounting_already_completed"
    )
    assert fixture["ledger"].read_bytes() == sealed


@pytest.mark.parametrize("crash_after", ("capsule", "marker"))
@pytest.mark.parametrize("canonical_transition", (False, True))
def test_interrupted_release_replays_half_retired_namespace(
    tmp_path, monkeypatch, crash_after, canonical_transition
):
    fixture = _interrupted_generation_fixture(tmp_path, monkeypatch)
    if canonical_transition:
        _install_fixture_canonical_transition(fixture, monkeypatch)
    armed = _arm_interrupted_generation(fixture, monkeypatch)
    real_release = R._release_dispatch_quarantine
    real_ensure = R._ensure_dispatch_release_authority_row

    def crash_during_retirement(
        marker,
        item,
        authority,
        *,
        before_authority_append=None,
        before_retirement=None,
    ):
        record, intent_identity = R._install_dispatch_release_intent(
            marker, authority
        )
        real_ensure(
            item,
            marker.root_fd,
            record,
            intent_identity,
            allow_new_authority_append=True,
            before_authority_append=before_authority_append,
        )
        if before_retirement is not None:
            before_retirement(record, intent_identity)
        real_ensure(item, marker.root_fd, record, intent_identity)
        os.unlink(marker.capsule_name, dir_fd=marker.root_fd)
        os.fsync(marker.root_fd)
        if crash_after == "capsule":
            raise RuntimeError("synthetic crash after capsule unlink")
        os.unlink(marker.name, dir_fd=marker.root_fd)
        os.fsync(marker.root_fd)
        raise RuntimeError("synthetic crash after marker unlink")

    monkeypatch.setattr(
        R, "_release_dispatch_quarantine", crash_during_retirement
    )
    with pytest.raises(RuntimeError, match=f"after {crash_after} unlink"):
        _recover_interrupted_generation(fixture, armed)
    assert not fixture["capsule"].exists()
    assert fixture["marker"].exists() is (crash_after == "capsule")
    assert list(fixture["marker"].parent.glob("*.release_intent"))

    monkeypatch.setattr(R, "_release_dispatch_quarantine", real_release)
    result = _recover_interrupted_generation(fixture, armed)
    assert result["result"] == (
        "interrupted_sandbox_isolated_noncounting_already_completed"
    )
    assert not fixture["marker"].exists()
    assert not fixture["capsule"].exists()
    assert not list(fixture["marker"].parent.glob("*.release_intent"))


@pytest.mark.parametrize(
    "source_sha256",
    (
        LEGACY_HISTORICAL_RUNNER_SOURCE_SHA256,
        BOUNDARY_V2_HISTORICAL_RUNNER_SOURCE_SHA256,
    ),
)
def test_sandboxed_generation_parser_accepts_only_exact_two_and_three_row_shapes(
    tmp_path, monkeypatch, source_sha256
):
    fixture = _sandboxed_generation_fixture(
        tmp_path, monkeypatch, source_sha256=source_sha256
    )
    parsed = Recovery.parse_sandboxed_generation_marker(
        fixture["marker"].read_bytes(), require_recovery_arm=False
    )
    assert parsed.unquiesced["event"] == "dispatch_failed"
    assert parsed.unquiesced["child_returncode"] is None
    assert parsed.recovery_arm is None

    outcome = _arm_sandboxed_generation(fixture, monkeypatch)
    assert outcome["result"] == "sandboxed_generation_release_armed"
    armed = Recovery.parse_sandboxed_generation_marker(
        fixture["marker"].read_bytes(), require_recovery_arm=True
    )
    assert armed.recovery_arm["process_tree_quiesced"] is False
    assert armed.recovery_arm["detached_processes_proven_absent"] is False
    assert armed.recovery_arm["scratch_root"] == os.fspath(fixture["scratch"])


@pytest.mark.parametrize(
    ("field", "value", "pattern"),
    (
        ("child_returncode", -1, "exact incident shape"),
        ("child_pid", None, "child pid"),
        ("workspace_identity", None, "workspace_identity"),
    ),
)
def test_sandboxed_generation_parser_rejects_malformed_or_null_failure_fields(
    tmp_path, monkeypatch, field, value, pattern
):
    fixture = _sandboxed_generation_fixture(tmp_path, monkeypatch)
    rows = _canonical_rows(fixture["marker"])
    rows[1][field] = value
    with pytest.raises(Recovery.RecoveryEvidenceError, match=pattern):
        Recovery.parse_sandboxed_generation_marker(
            b"".join(Recovery.canonical_json_line(row) for row in rows),
            require_recovery_arm=False,
        )


@pytest.mark.parametrize(
    ("field", "value", "pattern"),
    (
        ("process_tree_quiesced", None, "arm schema"),
        ("detached_processes_proven_absent", None, "arm schema"),
        ("scratch_root", None, "scratch_root"),
        ("workspace_identity", [1, 2], "binding changed"),
    ),
)
def test_sandboxed_generation_parser_rejects_arm_field_binding_mutation(
    tmp_path, monkeypatch, field, value, pattern
):
    fixture = _sandboxed_generation_fixture(tmp_path, monkeypatch)
    _arm_sandboxed_generation(fixture, monkeypatch)
    rows = _canonical_rows(fixture["marker"])
    rows[2][field] = value
    with pytest.raises(Recovery.RecoveryEvidenceError, match=pattern):
        Recovery.parse_sandboxed_generation_marker(
            b"".join(Recovery.canonical_json_line(row) for row in rows),
            require_recovery_arm=True,
        )


def test_sandboxed_generation_arm_refuses_live_scheduler_process(
    tmp_path, monkeypatch
):
    fixture = _sandboxed_generation_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(R.sys, "platform", "darwin")
    monkeypatch.setattr(
        R.Contiguous,
        "_process_identity",
        lambda pid: (pid, 0, 0, "R") if pid == os.getpid() else None,
    )
    monkeypatch.setattr(
        R.Contiguous, "_process_group_has_live_members", lambda _pgid: False
    )
    with pytest.raises(R.CampaignPlanError, match="scheduler PID remains live"):
        R._arm_sandboxed_generation_release(
            fixture["item"],
            confirm_dispatch_id=fixture["dispatch_id"],
            boot_identity_provider=lambda: fixture["boot"],
        )


@pytest.mark.parametrize("drift", ("ledger", "canonical"))
def test_sandboxed_generation_arm_refuses_ledger_or_canonical_drift(
    tmp_path, monkeypatch, drift
):
    fixture = _sandboxed_generation_fixture(tmp_path, monkeypatch)
    if drift == "ledger":
        R.Guard.append_ledger({"event": "synthetic_unbound_row"}, fixture["ledger"])
        pattern = "zero or one exact bound exec suffix"
    else:
        (fixture["artifact"] / "late.py").write_bytes(b"canonical drift\n")
        pattern = "canonical/frontier baseline changed"
    with pytest.raises(R.CampaignPlanError, match=pattern):
        _arm_sandboxed_generation(fixture, monkeypatch)


def test_sandboxed_generation_arm_checks_canonical_before_ledger_reconciliation(
    tmp_path, monkeypatch
):
    fixture = _sandboxed_generation_fixture(tmp_path, monkeypatch)
    (fixture["artifact"] / "late.py").write_bytes(b"canonical drift\n")
    ledger_before = fixture["ledger"].read_bytes()
    marker_before = fixture["marker"].read_bytes()
    ledger_read = False

    def forbidden_ledger_read(*_args, **_kwargs):
        nonlocal ledger_read
        ledger_read = True
        raise AssertionError("ledger reconciliation ran before canonical gate")

    monkeypatch.setattr(
        R, "_read_post_reboot_ledger_surface", forbidden_ledger_read
    )
    with pytest.raises(
        R.CampaignPlanError,
        match="canonical/frontier baseline changed",
    ):
        _arm_sandboxed_generation(fixture, monkeypatch)

    assert ledger_read is False
    assert fixture["ledger"].read_bytes() == ledger_before
    assert fixture["marker"].read_bytes() == marker_before


@pytest.mark.parametrize("tree_kind", ("workspace", "protected"))
def test_sandboxed_generation_arm_rejects_same_inode_tree_mutation_before_install(
    tmp_path, monkeypatch, tree_kind
):
    fixture = _sandboxed_generation_fixture(tmp_path, monkeypatch)
    target = _sandbox_generation_content_path(fixture, tree_kind)
    identity = (target.stat().st_dev, target.stat().st_ino)
    observations = 0

    def mutate_during_rebound_absence(**_kwargs):
        nonlocal observations
        observations += 1
        if observations == 2:
            target.write_bytes(b"same inode changed before arm install\n")
        observed = datetime.now(timezone.utc).isoformat()
        return {
            "absence_sample_count": R.SANDBOX_ABSENCE_SAMPLES,
            "absence_window_ns": 1,
            "absence_first_at": observed,
            "absence_last_at": observed,
        }

    monkeypatch.setattr(
        R, "_observe_named_root_group_absence", mutate_during_rebound_absence
    )
    with pytest.raises(
        R.CampaignPlanError,
        match="tree content changed before arm installation",
    ):
        R._arm_sandboxed_generation_release(
            fixture["item"],
            confirm_dispatch_id=fixture["dispatch_id"],
            boot_identity_provider=lambda: fixture["boot"],
        )

    assert observations == 2
    assert (target.stat().st_dev, target.stat().st_ino) == identity
    parsed = Recovery.parse_sandboxed_generation_marker(
        fixture["marker"].read_bytes(), require_recovery_arm=False
    )
    assert parsed.recovery_arm is None


def test_sandboxed_generation_arm_refuses_active_workspace_lock(
    tmp_path, monkeypatch
):
    fixture = _sandboxed_generation_fixture(tmp_path, monkeypatch)
    lock = fixture["exact_lock"].open("r+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(
            R.CampaignPlanError, match="workspace lock remains active"
        ):
            _arm_sandboxed_generation(fixture, monkeypatch)
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def _sandbox_tree_snapshot(root: Path):
    snapshot = {}
    for path in sorted((root, *root.rglob("*"))):
        relative = "." if path == root else path.relative_to(root).as_posix()
        metadata = path.stat(follow_symlinks=False)
        if path.is_dir():
            snapshot[relative] = (
                "directory",
                metadata.st_dev,
                metadata.st_ino,
                stat.S_IMODE(metadata.st_mode),
            )
        else:
            payload = path.read_bytes()
            snapshot[relative] = (
                "file",
                metadata.st_dev,
                metadata.st_ino,
                stat.S_IMODE(metadata.st_mode),
                hashlib.sha256(payload).hexdigest(),
            )
    return snapshot


def test_sandboxed_generation_recovery_restores_only_artifacts_and_abandons_scratch(
    tmp_path, monkeypatch
):
    fixture = _sandboxed_generation_fixture(tmp_path, monkeypatch)
    baseline_latest = (fixture["wip"] / "latest.json").read_bytes()
    (fixture["wip"] / "latest.json").write_bytes(
        b'{"attempt":"isolated-unpublished"}\n'
    )
    disposable = fixture["wip"] / "isolated-attempt.txt"
    disposable.write_bytes(b"must be rolled back\n")
    armed = _arm_sandboxed_generation(fixture, monkeypatch)
    nonce = armed["recovery_nonce"]
    old_scratch = _sandbox_tree_snapshot(fixture["scratch"])

    result = R._recover_sandboxed_generation_release(
        fixture["item"],
        confirm_dispatch_id=fixture["dispatch_id"],
        confirm_recovery_nonce=nonce,
        boot_identity_provider=lambda: fixture["boot"],
    )

    assert result["result"] == "sandbox_isolated_noncounting"
    assert result["process_tree_quiesced"] is False
    assert result["detached_processes_proven_absent"] is False
    assert (fixture["wip"] / "latest.json").read_bytes() == baseline_latest
    assert not disposable.exists()
    assert _sandbox_tree_snapshot(fixture["scratch"]) == old_scratch
    assert fixture["workspace"].is_dir()
    assert fixture["protected"].is_dir()
    assert fixture["exact_lock"].is_file()
    assert not fixture["marker"].exists()
    assert not fixture["capsule"].exists()
    rows = R.Guard.read_ledger(fixture["ledger"])
    assert [row["event"] for row in rows] == [
        R.SANDBOX_ABANDON_EVENT,
        "codex_dispatch_release_authorized",
    ]
    event = rows[0]
    assert event["recovery_nonce"] == nonce
    assert event["retry_increment"] == 0
    assert event["codex_exec_appended"] is False
    assert event["process_tree_quiesced"] is False
    assert event["detached_processes_proven_absent"] is False
    with pytest.raises(
        R.CampaignPlanError, match="abandoned sandbox namespace"
    ):
        R._reject_abandoned_scratch_root(
            fixture["scratch"], fixture["ledger"]
        )
    fresh_scratch = tmp_path / "fresh_scratch"
    fresh_scratch.mkdir()
    R._reject_abandoned_scratch_root(fresh_scratch, fixture["ledger"])

    sealed = fixture["ledger"].read_bytes()
    repeated = R._recover_sandboxed_generation_release(
        fixture["item"],
        confirm_dispatch_id=fixture["dispatch_id"],
        confirm_recovery_nonce=nonce,
        boot_identity_provider=lambda: fixture["boot"],
    )
    assert repeated["result"] == (
        "sandbox_isolated_noncounting_already_completed"
    )
    assert fixture["ledger"].read_bytes() == sealed


def test_sandboxed_one_exec_recovery_classifies_noncounting_and_restores_capsule(
    tmp_path, monkeypatch
):
    fixture = _sandboxed_generation_fixture(tmp_path, monkeypatch)
    baseline_latest = (fixture["wip"] / "latest.json").read_bytes()
    (fixture["wip"] / "latest.json").write_bytes(
        b'{"attempt":"isolated-one-exec"}\n'
    )
    disposable = fixture["wip"] / "isolated-one-exec.txt"
    disposable.write_bytes(b"restore from capsule\n")
    execution = _append_sandbox_exec(fixture)
    old_scratch = _sandbox_tree_snapshot(fixture["scratch"])

    armed = _arm_sandboxed_generation(fixture, monkeypatch)
    parsed = Recovery.parse_sandboxed_generation_marker(
        fixture["marker"].read_bytes(), require_recovery_arm=True
    )
    assert parsed.recovery_arm is not None
    assert parsed.recovery_arm["recovery_arm_schema"] == (
        Recovery.SANDBOXED_GENERATION_EXEC_ARM_SCHEMA
    )
    assert parsed.recovery_arm["exec_record_sha256"] == (
        R._recovery_record_sha256(execution)
    )

    result = R._recover_sandboxed_generation_release(
        fixture["item"],
        confirm_dispatch_id=fixture["dispatch_id"],
        confirm_recovery_nonce=armed["recovery_nonce"],
        boot_identity_provider=lambda: fixture["boot"],
    )

    assert result["result"] == "sandbox_isolated_noncounting"
    assert (fixture["wip"] / "latest.json").read_bytes() == baseline_latest
    assert not disposable.exists()
    assert _sandbox_tree_snapshot(fixture["scratch"]) == old_scratch
    assert fixture["workspace"].is_dir()
    assert fixture["protected"].is_dir()
    assert fixture["exact_lock"].is_file()
    assert not fixture["marker"].exists()
    assert not fixture["capsule"].exists()
    rows = R.Guard.read_ledger(fixture["ledger"])
    assert [row["event"] for row in rows] == [
        "codex_exec",
        "codex_exec_classification_correction",
        R.SANDBOX_ABANDON_EVENT,
        "codex_dispatch_release_authorized",
    ]
    correction, event = rows[1:3]
    assert correction["schema"] == R.SANDBOX_EXEC_CLASSIFICATION_SCHEMA
    assert correction["failure_class"] == "infrastructure"
    assert correction["retry_increment"] == 0
    assert correction["exec_record_sha256"] == R._recovery_record_sha256(
        rows[0]
    )
    assert event["schema"] == R.SANDBOX_EXEC_ABANDON_EVENT_SCHEMA
    assert event["codex_exec_appended"] is True
    assert event["retry_increment"] == 0
    assert event["exec_record_sha256"] == R._recovery_record_sha256(rows[0])
    assert event["classification_record_sha256"] == (
        R._recovery_record_sha256(correction)
    )

    sealed = fixture["ledger"].read_bytes()
    repeated = R._recover_sandboxed_generation_release(
        fixture["item"],
        confirm_dispatch_id=fixture["dispatch_id"],
        confirm_recovery_nonce=armed["recovery_nonce"],
        boot_identity_provider=lambda: fixture["boot"],
    )
    assert repeated["result"] == (
        "sandbox_isolated_noncounting_already_completed"
    )
    assert fixture["ledger"].read_bytes() == sealed


def test_sandboxed_one_exec_recovery_rejects_exec_bytes_changed_after_arm(
    tmp_path, monkeypatch
):
    fixture = _sandboxed_generation_fixture(tmp_path, monkeypatch)
    _append_sandbox_exec(fixture)
    armed = _arm_sandboxed_generation(fixture, monkeypatch)
    rows = R.Guard.read_ledger(fixture["ledger"])
    rows[0]["returncode"] = 17
    _write_canonical_rows(fixture["ledger"], rows)

    with pytest.raises(
        R.CampaignPlanError, match="exec suffix changed after arming"
    ):
        R._recover_sandboxed_generation_release(
            fixture["item"],
            confirm_dispatch_id=fixture["dispatch_id"],
            confirm_recovery_nonce=armed["recovery_nonce"],
            boot_identity_provider=lambda: fixture["boot"],
        )
    assert fixture["marker"].is_file()
    assert fixture["capsule"].is_file()


def test_sandbox_lane_preserves_foreign_contained_residual_correction_intent(
    tmp_path, monkeypatch
):
    fixture = _sandboxed_generation_fixture(tmp_path, monkeypatch)
    execution = _append_sandbox_exec(fixture)
    transcript = fixture["protected"] / fixture["transcript_name"]
    execution["protected_transcript_sha256"] = hashlib.sha256(
        transcript.read_bytes()
    ).hexdigest()
    _write_canonical_rows(fixture["ledger"], [execution])
    armed = _arm_sandboxed_generation(fixture, monkeypatch)

    marker, parsed = R._read_existing_dispatch_quarantine(
        fixture["item"],
        require_recovery_arm=True,
        marker_parser=Recovery.parse_sandboxed_generation_marker,
    )
    try:
        ledger, baseline, suffix = R._read_post_reboot_ledger_surface(
            fixture["item"], parsed.armed
        )
        assert suffix == [execution]
        foreign = R._build_contained_normal_exit_residual_correction(
            fixture["item"],
            execution,
            dispatch_id=fixture["dispatch_id"],
        )
        state = R.PostRebootLedgerState(
            dispatch_id=fixture["dispatch_id"],
            intent_root=marker.root,
            intent_root_identity=marker.root_identity,
            ledger=ledger,
            baseline=baseline,
            record=execution,
            correction=None,
            cleanup=None,
            operator=None,
        )
        expected_raw = baseline.raw_prefix + Recovery.canonical_json_line(
            execution
        )
        intent_name, selected = R._prepare_recovery_phase_intent_locked(
            state, foreign, expected_raw
        )
        assert selected["record"] == foreign
        intent_path = marker.root / intent_name
        before_intent = intent_path.read_bytes()
        intent_identity = (
            intent_path.stat().st_dev,
            intent_path.stat().st_ino,
        )
        before_ledger = fixture["ledger"].read_bytes()
    finally:
        R._close_dispatch_quarantine(marker)

    with pytest.raises(
        R.CampaignPlanError,
        match="sandbox exec classification has an invalid exact schema",
    ):
        R._recover_sandboxed_generation_release(
            fixture["item"],
            confirm_dispatch_id=fixture["dispatch_id"],
            confirm_recovery_nonce=armed["recovery_nonce"],
            boot_identity_provider=lambda: fixture["boot"],
        )

    assert fixture["ledger"].read_bytes() == before_ledger
    assert intent_path.read_bytes() == before_intent
    assert (intent_path.stat().st_dev, intent_path.stat().st_ino) == (
        intent_identity
    )
    assert stat.S_IMODE(intent_path.stat().st_mode) == 0o600
    assert fixture["marker"].is_file()
    assert fixture["capsule"].is_file()


@pytest.mark.parametrize("tree_kind", ("workspace", "protected"))
@pytest.mark.parametrize(
    ("one_exec", "phase", "expected_events"),
    (
        (False, "terminal", []),
        (True, "classification", ["codex_exec"]),
    ),
)
def test_sandboxed_recovery_rejects_same_inode_tree_mutation_after_arm(
    tmp_path, monkeypatch, tree_kind, one_exec, phase, expected_events
):
    fixture = _sandboxed_generation_fixture(tmp_path, monkeypatch)
    if one_exec:
        _append_sandbox_exec(fixture)
    armed = _arm_sandboxed_generation(fixture, monkeypatch)
    target = _sandbox_generation_content_path(fixture, tree_kind)
    identity = (target.stat().st_dev, target.stat().st_ino)
    target.write_bytes(b"same inode changed after arm\n")

    with pytest.raises(
        R.CampaignPlanError,
        match=f"tree content changed before {phase} append",
    ):
        R._recover_sandboxed_generation_release(
            fixture["item"],
            confirm_dispatch_id=fixture["dispatch_id"],
            confirm_recovery_nonce=armed["recovery_nonce"],
            boot_identity_provider=lambda: fixture["boot"],
        )

    assert (target.stat().st_dev, target.stat().st_ino) == identity
    assert [
        row["event"] for row in R.Guard.read_ledger(fixture["ledger"])
    ] == expected_events
    assert fixture["marker"].is_file()
    assert fixture["capsule"].is_file()


@pytest.mark.parametrize("tree_kind", ("workspace", "protected"))
@pytest.mark.parametrize(
    "mutation", ("same_bytes_new_inode", "mode", "xattr")
)
def test_sandboxed_recovery_rejects_generation_metadata_drift_after_arm(
    tmp_path, monkeypatch, tree_kind, mutation
):
    fixture = _sandboxed_generation_fixture(tmp_path, monkeypatch)
    armed = _arm_sandboxed_generation(fixture, monkeypatch)
    target = _sandbox_generation_content_path(fixture, tree_kind)
    before = target.stat(follow_symlinks=False)
    if mutation == "same_bytes_new_inode":
        replacement = target.with_name(f"{target.name}.replacement")
        replacement.write_bytes(target.read_bytes())
        os.chmod(replacement, stat.S_IMODE(before.st_mode))
        os.utime(
            replacement,
            ns=(before.st_atime_ns, before.st_mtime_ns),
        )
        os.replace(replacement, target)
        assert target.stat(follow_symlinks=False).st_ino != before.st_ino
    elif mutation == "mode":
        os.chmod(target, stat.S_IMODE(before.st_mode) ^ stat.S_IXUSR)
    else:
        attribute = (
            "com.gkm.scheduler-test"
            if sys.platform == "darwin"
            else "user.gkm_scheduler_test"
        )
        try:
            if hasattr(os, "setxattr"):
                os.setxattr(
                    target,
                    attribute,
                    b"changed after arm",
                    follow_symlinks=False,
                )
            elif sys.platform == "darwin":
                subprocess.run(
                    [
                        "/usr/bin/xattr",
                        "-w",
                        attribute,
                        "changed after arm",
                        os.fspath(target),
                    ],
                    check=True,
                    capture_output=True,
                )
            else:
                pytest.skip("extended attributes are unavailable")
        except (AttributeError, OSError, subprocess.CalledProcessError):
            pytest.skip("extended attributes are unavailable")

    with pytest.raises(
        R.CampaignPlanError,
        match="tree content changed before terminal append",
    ):
        R._recover_sandboxed_generation_release(
            fixture["item"],
            confirm_dispatch_id=fixture["dispatch_id"],
            confirm_recovery_nonce=armed["recovery_nonce"],
            boot_identity_provider=lambda: fixture["boot"],
        )

    assert fixture["ledger"].read_bytes() == b""
    assert fixture["marker"].is_file()
    assert fixture["capsule"].is_file()


@pytest.mark.parametrize("tree_kind", ("workspace", "protected"))
def test_sandboxed_recovery_rechecks_same_inode_tree_immediately_before_release(
    tmp_path, monkeypatch, tree_kind
):
    fixture = _sandboxed_generation_fixture(tmp_path, monkeypatch)
    _append_sandbox_exec(fixture)
    armed = _arm_sandboxed_generation(fixture, monkeypatch)
    target = _sandbox_generation_content_path(fixture, tree_kind)
    identity = (target.stat().st_dev, target.stat().st_ino)
    real_build = R._build_dispatch_release_authority

    def build_then_mutate(*args, **kwargs):
        authority = real_build(*args, **kwargs)
        target.write_bytes(b"same inode changed before release\n")
        return authority

    monkeypatch.setattr(
        R, "_build_dispatch_release_authority", build_then_mutate
    )
    with pytest.raises(
        R.CampaignPlanError,
        match="tree content changed before release",
    ):
        R._recover_sandboxed_generation_release(
            fixture["item"],
            confirm_dispatch_id=fixture["dispatch_id"],
            confirm_recovery_nonce=armed["recovery_nonce"],
            boot_identity_provider=lambda: fixture["boot"],
        )

    assert (target.stat().st_dev, target.stat().st_ino) == identity
    assert [row["event"] for row in R.Guard.read_ledger(fixture["ledger"])] == [
        "codex_exec",
        "codex_exec_classification_correction",
        R.SANDBOX_ABANDON_EVENT,
    ]
    assert fixture["marker"].is_file()
    assert fixture["capsule"].is_file()


def test_sandboxed_recovery_rechecks_canonical_after_final_tree_hash(
    tmp_path, monkeypatch
):
    fixture = _sandboxed_generation_fixture(tmp_path, monkeypatch)
    _append_sandbox_exec(fixture)
    armed = _arm_sandboxed_generation(fixture, monkeypatch)
    real_revalidate = R._revalidate_sandboxed_generation_tree_hashes

    def revalidate_then_mutate(*args, **kwargs):
        result = real_revalidate(*args, **kwargs)
        if kwargs.get("phase") == "before release":
            (fixture["artifact"] / "post-tree-hash-drift.py").write_bytes(
                b"unauthorized canonical drift\n"
            )
        return result

    monkeypatch.setattr(
        R,
        "_revalidate_sandboxed_generation_tree_hashes",
        revalidate_then_mutate,
    )
    with pytest.raises(
        R.CampaignPlanError,
        match="canonical/frontier baseline changed",
    ):
        R._recover_sandboxed_generation_release(
            fixture["item"],
            confirm_dispatch_id=fixture["dispatch_id"],
            confirm_recovery_nonce=armed["recovery_nonce"],
            boot_identity_provider=lambda: fixture["boot"],
        )

    assert fixture["marker"].is_file()
    assert fixture["capsule"].is_file()
    assert [row["event"] for row in R.Guard.read_ledger(fixture["ledger"])] == [
        "codex_exec",
        "codex_exec_classification_correction",
        R.SANDBOX_ABANDON_EVENT,
    ]


@pytest.mark.parametrize(
    "profile", ("sandboxed", "interrupted", "interrupted_transition")
)
def test_sandboxed_one_exec_recovery_resumes_after_durable_classification(
    tmp_path, monkeypatch, profile
):
    interrupted_exec = profile != "sandboxed"
    fixture, _armed, recover = _armed_one_exec_crash_fixture(
        tmp_path,
        monkeypatch,
        interrupted_exec,
        canonical_transition=profile == "interrupted_transition",
    )
    real_append = R._append_recovery_phase_cas
    crashed = False

    def append_then_crash(state, record, **kwargs):
        nonlocal crashed
        committed = real_append(state, record, **kwargs)
        if (
            not crashed
            and record.get("event")
            == "codex_exec_classification_correction"
        ):
            crashed = True
            raise RuntimeError("synthetic crash after sandbox classification")
        return committed

    monkeypatch.setattr(R, "_append_recovery_phase_cas", append_then_crash)
    with pytest.raises(RuntimeError, match="sandbox classification"):
        recover()
    assert [row["event"] for row in R.Guard.read_ledger(fixture["ledger"])] == [
        "codex_exec", "codex_exec_classification_correction",
    ]
    assert fixture["marker"].is_file()
    assert fixture["capsule"].is_file()

    monkeypatch.setattr(R, "_append_recovery_phase_cas", real_append)
    result = recover()
    assert result["result"] == "sandbox_isolated_noncounting"
    rows = R.Guard.read_ledger(fixture["ledger"])
    assert [row["event"] for row in rows] == [
        "codex_exec",
        "codex_exec_classification_correction",
        R.SANDBOX_ABANDON_EVENT,
        "codex_dispatch_release_authorized",
    ]
    if profile == "interrupted_transition":
        assert rows[2]["schema"] == (
            R.INTERRUPTED_EXEC_TRANSITION_ABANDON_EVENT_SCHEMA
        )


@pytest.mark.parametrize(
    "profile", ("sandboxed", "interrupted", "interrupted_transition")
)
def test_sandboxed_one_exec_recovery_repairs_partial_terminal_intent(
    tmp_path, monkeypatch, profile
):
    interrupted_exec = profile != "sandboxed"
    fixture, _armed, recover = _armed_one_exec_crash_fixture(
        tmp_path,
        monkeypatch,
        interrupted_exec,
        canonical_transition=profile == "interrupted_transition",
    )
    ledger_identity = (
        fixture["ledger"].stat().st_dev,
        fixture["ledger"].stat().st_ino,
    )
    real_append = R._append_recovery_phase_cas
    real_write = R.os.write
    active = False
    partial_written = False

    def target_terminal(state, record, **kwargs):
        nonlocal active
        if record.get("event") != R.SANDBOX_ABANDON_EVENT:
            return real_append(state, record, **kwargs)
        active = True
        try:
            return real_append(state, record, **kwargs)
        finally:
            active = False

    def partial_terminal_write(descriptor, payload):
        nonlocal partial_written
        metadata = os.fstat(descriptor)
        if active and (metadata.st_dev, metadata.st_ino) == ledger_identity:
            if not partial_written:
                partial_written = True
                return real_write(descriptor, payload[:23])
            raise OSError(errno.ENOSPC, "synthetic sandbox terminal ENOSPC")
        return real_write(descriptor, payload)

    monkeypatch.setattr(R, "_append_recovery_phase_cas", target_terminal)
    monkeypatch.setattr(R.os, "write", partial_terminal_write)
    with pytest.raises(R.CampaignPlanError, match="phase append failed"):
        recover()
    assert partial_written
    intents = list(
        fixture["marker"].parent.glob(".codex_recovery_*.intent")
    )
    assert len(intents) == 1
    assert stat.S_IMODE(intents[0].stat().st_mode) == 0o600
    pending = R._parse_recovery_phase_intent(
        intents[0].read_bytes(), label="sandbox terminal recovery phase intent"
    )
    expected_terminal = dict(pending["record"])
    assert expected_terminal["event"] == R.SANDBOX_ABANDON_EVENT
    if profile == "interrupted_transition":
        assert expected_terminal["schema"] == (
            R.INTERRUPTED_EXEC_TRANSITION_ABANDON_EVENT_SCHEMA
        )
    assert fixture["marker"].is_file()
    assert fixture["capsule"].is_file()

    monkeypatch.setattr(R, "_append_recovery_phase_cas", real_append)
    monkeypatch.setattr(R.os, "write", real_write)
    result = recover()

    assert result["result"] == "sandbox_isolated_noncounting"
    rows = R.Guard.read_ledger(fixture["ledger"])
    assert [row["event"] for row in rows] == [
        "codex_exec",
        "codex_exec_classification_correction",
        R.SANDBOX_ABANDON_EVENT,
        "codex_dispatch_release_authorized",
    ]
    assert rows[2] == expected_terminal
    assert not intents[0].exists()
    assert not fixture["marker"].exists()
    assert not fixture["capsule"].exists()


def test_abandoned_scratch_rejects_an_ancestor_retry_root(tmp_path, monkeypatch):
    fixture = _sandboxed_generation_fixture(tmp_path, monkeypatch)
    armed = _arm_sandboxed_generation(fixture, monkeypatch)
    R._recover_sandboxed_generation_release(
        fixture["item"],
        confirm_dispatch_id=fixture["dispatch_id"],
        confirm_recovery_nonce=armed["recovery_nonce"],
        boot_identity_provider=lambda: fixture["boot"],
    )

    with pytest.raises(
        R.CampaignPlanError, match="abandoned sandbox namespace"
    ):
        R._reject_abandoned_scratch_root(
            fixture["scratch"].parent, fixture["ledger"]
        )


def _complete_sandbox_abandonment(fixture, monkeypatch):
    armed = _arm_sandboxed_generation(fixture, monkeypatch)
    R._recover_sandboxed_generation_release(
        fixture["item"],
        confirm_dispatch_id=fixture["dispatch_id"],
        confirm_recovery_nonce=armed["recovery_nonce"],
        boot_identity_provider=lambda: fixture["boot"],
    )


def test_removed_abandoned_scratch_allows_only_unrelated_fresh_root(
    tmp_path, monkeypatch
):
    fixture = _sandboxed_generation_fixture(tmp_path, monkeypatch)
    _complete_sandbox_abandonment(fixture, monkeypatch)
    shutil.rmtree(fixture["scratch"])
    fresh = tmp_path / "fresh_retry_root"
    fresh.mkdir()

    R._reject_abandoned_scratch_root(fresh, fixture["ledger"])
    with pytest.raises(
        R.CampaignPlanError, match="abandoned sandbox namespace"
    ):
        R._reject_abandoned_scratch_root(tmp_path, fixture["ledger"])

    fixture["scratch"].mkdir()
    with pytest.raises(
        R.CampaignPlanError, match="abandoned sandbox namespace"
    ):
        R._reject_abandoned_scratch_root(
            fixture["scratch"], fixture["ledger"]
        )
    descendant = fixture["scratch"] / "descendant_retry"
    descendant.mkdir()
    with pytest.raises(
        R.CampaignPlanError, match="abandoned sandbox namespace"
    ):
        R._reject_abandoned_scratch_root(descendant, fixture["ledger"])


def test_removed_abandoned_scratch_inode_moved_below_fresh_root_is_rejected(
    tmp_path, monkeypatch
):
    fixture = _sandboxed_generation_fixture(tmp_path, monkeypatch)
    _complete_sandbox_abandonment(fixture, monkeypatch)
    fresh = tmp_path / "fresh_retry_root"
    fresh.mkdir()
    fixture["scratch"].rename(fresh / "moved_abandoned_root")

    with pytest.raises(
        R.CampaignPlanError, match="contains an abandoned sandbox inode"
    ):
        R._reject_abandoned_scratch_root(fresh, fixture["ledger"])


def test_removed_abandoned_scratch_inode_moved_above_fresh_root_is_rejected(
    tmp_path, monkeypatch
):
    fixture = _sandboxed_generation_fixture(tmp_path, monkeypatch)
    _complete_sandbox_abandonment(fixture, monkeypatch)
    moved_parent = tmp_path / "moved_parent"
    fixture["scratch"].rename(moved_parent)
    fresh = moved_parent / "fresh_retry_root"
    fresh.mkdir()

    with pytest.raises(
        R.CampaignPlanError, match="ancestry contains an abandoned sandbox inode"
    ):
        R._reject_abandoned_scratch_root(fresh, fixture["ledger"])


def test_removed_abandoned_scratch_rejects_symlinked_fresh_tree(
    tmp_path, monkeypatch
):
    fixture = _sandboxed_generation_fixture(tmp_path, monkeypatch)
    _complete_sandbox_abandonment(fixture, monkeypatch)
    shutil.rmtree(fixture["scratch"])
    fresh = tmp_path / "fresh_retry_root"
    fresh.mkdir()
    target = tmp_path / "link_target"
    target.mkdir()
    (fresh / "alias").symlink_to(target, target_is_directory=True)

    with pytest.raises(R.CampaignPlanError, match="contains a symlink"):
        R._reject_abandoned_scratch_root(fresh, fixture["ledger"])


def test_removed_abandoned_scratch_rejects_unreadable_fresh_tree(
    tmp_path, monkeypatch
):
    fixture = _sandboxed_generation_fixture(tmp_path, monkeypatch)
    _complete_sandbox_abandonment(fixture, monkeypatch)
    shutil.rmtree(fixture["scratch"])
    fresh = tmp_path / "fresh_retry_root"
    fresh.mkdir()
    unreadable = fresh / "unreadable"
    unreadable.mkdir()
    os.chmod(unreadable, 0)
    try:
        with pytest.raises(R.CampaignPlanError, match="completely traversed"):
            R._reject_abandoned_scratch_root(fresh, fixture["ledger"])
    finally:
        os.chmod(unreadable, 0o700)


def test_abandoned_scratch_authentication_rejects_ledger_append_race(
    tmp_path, monkeypatch
):
    fixture = _sandboxed_generation_fixture(tmp_path, monkeypatch)
    _complete_sandbox_abandonment(fixture, monkeypatch)
    shutil.rmtree(fixture["scratch"])
    fresh = tmp_path / "fresh_retry_root"
    fresh.mkdir()
    real_traversal = R._mutable_root_traversal_receipt
    appended = False

    def racing_traversal(*args, **kwargs):
        nonlocal appended
        result = real_traversal(*args, **kwargs)
        if not appended:
            appended = True
            R.Guard.append_ledger(
                {"event": "synthetic_concurrent_control_row"},
                fixture["ledger"],
            )
        return result

    monkeypatch.setattr(
        R, "_mutable_root_traversal_receipt", racing_traversal
    )
    with pytest.raises(R.CampaignPlanError, match="ledger changed"):
        R._reject_abandoned_scratch_root(fresh, fixture["ledger"])
    assert appended is True


def test_abandoned_scratch_authentication_rejects_candidate_replacement_race(
    tmp_path, monkeypatch
):
    fixture = _sandboxed_generation_fixture(tmp_path, monkeypatch)
    _complete_sandbox_abandonment(fixture, monkeypatch)
    shutil.rmtree(fixture["scratch"])
    fresh = tmp_path / "fresh_retry_root"
    fresh.mkdir()
    real_traversal = R._mutable_root_traversal_receipt
    replaced = False

    def racing_traversal(*args, **kwargs):
        nonlocal replaced
        result = real_traversal(*args, **kwargs)
        if not replaced:
            replaced = True
            fresh.rename(tmp_path / "replaced_retry_root")
            fresh.mkdir()
        return result

    monkeypatch.setattr(
        R, "_mutable_root_traversal_receipt", racing_traversal
    )
    with pytest.raises(R.CampaignPlanError, match="ancestry changed"):
        R._reject_abandoned_scratch_root(fresh, fixture["ledger"])
    assert replaced is True


def test_abandoned_scratch_authentication_rejects_deep_tree_drift(
    tmp_path, monkeypatch
):
    fixture = _sandboxed_generation_fixture(tmp_path, monkeypatch)
    _complete_sandbox_abandonment(fixture, monkeypatch)
    shutil.rmtree(fixture["scratch"])
    fresh = tmp_path / "fresh_retry_root"
    nested = fresh / "nested"
    nested.mkdir(parents=True)
    real_traversal = R._mutable_root_traversal_receipt
    traversals = 0

    def racing_traversal(*args, **kwargs):
        nonlocal traversals
        result = real_traversal(*args, **kwargs)
        traversals += 1
        if traversals == 1:
            (nested / "late_control_file").write_bytes(b"synthetic\n")
        return result

    monkeypatch.setattr(
        R, "_mutable_root_traversal_receipt", racing_traversal
    )
    with pytest.raises(R.CampaignPlanError, match="scratch tree changed"):
        R._reject_abandoned_scratch_root(fresh, fixture["ledger"])
    assert traversals == 2


def test_cross_game_scratch_admission_blocks_abandonment_append(
    tmp_path, monkeypatch
):
    fixture = _sandboxed_generation_fixture(tmp_path, monkeypatch)
    armed = _arm_sandboxed_generation(fixture, monkeypatch)
    other_game = copy.deepcopy(fixture["item"])
    other_game["game"] = "bp35"
    sealed_ledger = fixture["ledger"].read_bytes()

    monkeypatch.setattr(
        R,
        "_project_runner_receipt",
        lambda _plan, selected, **_kwargs: selected,
    )
    monkeypatch.setattr(R, "validate_item", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(R, "validate_inventory_item", lambda *_args: None)
    monkeypatch.setattr(R, "validate_live_policy_item", lambda _item: None)
    monkeypatch.setattr(R, "active_workspace_lock", lambda _game: None)
    monkeypatch.setattr(
        R, "item_is_admissible", lambda *_args, **_kwargs: (True, "")
    )

    class SharedScratchObserved(RuntimeError):
        pass

    def observe_shared_scratch_custody(_item):
        with pytest.raises(
            R.CampaignPlanError,
            match="owns the exact artifact scratch admission",
        ):
            R._recover_sandboxed_generation_release(
                fixture["item"],
                confirm_dispatch_id=fixture["dispatch_id"],
                confirm_recovery_nonce=armed["recovery_nonce"],
                boot_identity_provider=lambda: fixture["boot"],
            )
        raise SharedScratchObserved

    monkeypatch.setattr(
        R, "_acquire_scheduler_lineage_lock", observe_shared_scratch_custody
    )
    with pytest.raises(SharedScratchObserved):
        R._run_item({}, other_game, allowance=object())

    assert fixture["ledger"].read_bytes() == sealed_ledger
    assert fixture["marker"].is_file()


@pytest.mark.parametrize(
    "recovery_relation",
    ("parent_of_admission", "child_of_admission"),
)
def test_related_cross_game_scratch_admission_blocks_abandonment_append(
    tmp_path, monkeypatch, recovery_relation
):
    fixture = _sandboxed_generation_fixture(
        tmp_path,
        monkeypatch,
        scratch_relative=(
            Path("scratch_parent")
            if recovery_relation == "parent_of_admission"
            else Path("scratch_parent") / "recovery_child"
        ),
    )
    armed = _arm_sandboxed_generation(fixture, monkeypatch)
    if recovery_relation == "parent_of_admission":
        admission_root = fixture["scratch"] / "admission_child"
        admission_root.mkdir()
    else:
        admission_root = fixture["scratch"].parent
    other_game = copy.deepcopy(fixture["item"])
    other_game["game"] = "bp35"
    other_game["historical_runner"] = dict(other_game["historical_runner"])
    other_game["historical_runner"]["scratch_root"] = os.fspath(
        admission_root
    )
    sealed_ledger = fixture["ledger"].read_bytes()

    monkeypatch.setattr(
        R,
        "_project_runner_receipt",
        lambda _plan, selected, **_kwargs: selected,
    )
    monkeypatch.setattr(R, "validate_item", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(R, "validate_inventory_item", lambda *_args: None)
    monkeypatch.setattr(R, "validate_live_policy_item", lambda _item: None)
    monkeypatch.setattr(R, "active_workspace_lock", lambda _game: None)
    monkeypatch.setattr(
        R, "item_is_admissible", lambda *_args, **_kwargs: (True, "")
    )

    class RelatedScratchObserved(RuntimeError):
        pass

    def observe_related_scratch_custody(_item):
        with pytest.raises(
            R.CampaignPlanError,
            match="owns the exact artifact scratch admission",
        ):
            R._recover_sandboxed_generation_release(
                fixture["item"],
                confirm_dispatch_id=fixture["dispatch_id"],
                confirm_recovery_nonce=armed["recovery_nonce"],
                boot_identity_provider=lambda: fixture["boot"],
            )
        raise RelatedScratchObserved

    monkeypatch.setattr(
        R, "_acquire_scheduler_lineage_lock", observe_related_scratch_custody
    )
    with pytest.raises(RelatedScratchObserved):
        R._run_item({}, other_game, allowance=object())

    assert fixture["ledger"].read_bytes() == sealed_ledger
    assert fixture["marker"].is_file()


def test_sandboxed_arm_sidecar_replay_tolerates_dynamic_absence_observations(
    tmp_path, monkeypatch
):
    fixture = _sandboxed_generation_fixture(tmp_path, monkeypatch)
    observations = []

    def dynamic_absence(**_kwargs):
        sequence = len(observations) + 1
        observed = datetime.now(timezone.utc).isoformat()
        receipt = {
            "absence_sample_count": R.SANDBOX_ABSENCE_SAMPLES,
            "absence_window_ns": sequence,
            "absence_first_at": observed,
            "absence_last_at": observed,
        }
        observations.append(receipt)
        return receipt

    monkeypatch.setattr(R, "_observe_named_root_group_absence", dynamic_absence)
    real_atomic = R._atomic_recovery_arm_replace
    real_fsync = R.os.fsync
    active = False
    injected = False

    def track_atomic(*args, **kwargs):
        nonlocal active
        active = True
        try:
            return real_atomic(*args, **kwargs)
        finally:
            active = False

    def fsync_sidecar_then_fail(descriptor):
        nonlocal injected
        result = real_fsync(descriptor)
        metadata = os.fstat(descriptor)
        if active and not injected and stat.S_ISREG(metadata.st_mode):
            injected = True
            raise OSError("synthetic sandbox arm sidecar fsync report failure")
        return result

    monkeypatch.setattr(R, "_atomic_recovery_arm_replace", track_atomic)
    monkeypatch.setattr(R.os, "fsync", fsync_sidecar_then_fail)
    with pytest.raises(R.CampaignPlanError, match="durably prepare"):
        R._arm_sandboxed_generation_release(
            fixture["item"],
            confirm_dispatch_id=fixture["dispatch_id"],
            boot_identity_provider=lambda: fixture["boot"],
        )
    assert injected
    sidecar = fixture["marker"].parent / (
        f".{fixture['marker'].name}.sandboxed_generation_arm"
    )
    staged = Recovery.parse_sandboxed_generation_marker(
        sidecar.read_bytes(), require_recovery_arm=True
    ).recovery_arm
    assert staged is not None

    monkeypatch.setattr(R, "_atomic_recovery_arm_replace", real_atomic)
    monkeypatch.setattr(R.os, "fsync", real_fsync)
    outcome = R._arm_sandboxed_generation_release(
        fixture["item"],
        confirm_dispatch_id=fixture["dispatch_id"],
        boot_identity_provider=lambda: fixture["boot"],
    )

    assert outcome["result"] == "sandboxed_generation_release_armed"
    assert outcome["recovery_nonce"] == staged["recovery_nonce"]
    installed = Recovery.parse_sandboxed_generation_marker(
        fixture["marker"].read_bytes(), require_recovery_arm=True
    ).recovery_arm
    assert installed == staged
    assert observations[0]["absence_window_ns"] != observations[2][
        "absence_window_ns"
    ]
    assert not sidecar.exists()


def test_sandboxed_recovery_lock_acquisition_preserves_lock_metadata(
    tmp_path, monkeypatch
):
    fixture = _sandboxed_generation_fixture(tmp_path, monkeypatch)
    armed = _arm_sandboxed_generation(fixture, monkeypatch)
    before = fixture["exact_lock"].stat(follow_symlinks=False)
    expected = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
        before.st_uid,
        before.st_gid,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )

    R._recover_sandboxed_generation_release(
        fixture["item"],
        confirm_dispatch_id=fixture["dispatch_id"],
        confirm_recovery_nonce=armed["recovery_nonce"],
        boot_identity_provider=lambda: fixture["boot"],
    )

    after = fixture["exact_lock"].stat(follow_symlinks=False)
    assert (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
        after.st_uid,
        after.st_gid,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ) == expected


@pytest.mark.parametrize("drift", ("canonical", "workspace_lock"))
def test_sandboxed_one_exec_rechecks_authority_after_wip_restore(
    tmp_path, monkeypatch, drift
):
    fixture = _sandboxed_generation_fixture(tmp_path, monkeypatch)
    (fixture["wip"] / "latest.json").write_bytes(
        b'{"attempt":"force-capsule-restore"}\n'
    )
    _append_sandbox_exec(fixture)
    armed = _arm_sandboxed_generation(fixture, monkeypatch)
    real_restore = R._restore_wip_from_rollback_capsule

    def restore_then_drift(*args, **kwargs):
        restored = real_restore(*args, **kwargs)
        if drift == "canonical":
            (fixture["artifact"] / "post-restore-drift.py").write_bytes(
                b"unauthorized canonical drift\n"
            )
        else:
            os.chmod(fixture["exact_lock"], 0o640)
        return restored

    monkeypatch.setattr(
        R, "_restore_wip_from_rollback_capsule", restore_then_drift
    )
    pattern = (
        "canonical artifact changed"
        if drift == "canonical" else "authority changed before release"
    )
    with pytest.raises(R.CampaignPlanError, match=pattern):
        R._recover_sandboxed_generation_release(
            fixture["item"],
            confirm_dispatch_id=fixture["dispatch_id"],
            confirm_recovery_nonce=armed["recovery_nonce"],
            boot_identity_provider=lambda: fixture["boot"],
        )
    assert fixture["marker"].is_file()
    assert fixture["capsule"].is_file()


@pytest.mark.parametrize(
    "profile", ("sandboxed", "interrupted", "interrupted_transition")
)
def test_partial_sandbox_wip_restore_replays_only_behind_durable_event(
    tmp_path, monkeypatch, profile
):
    interrupted_exec = profile != "sandboxed"
    fixture = (
        _interrupted_generation_fixture(tmp_path, monkeypatch)
        if interrupted_exec
        else _sandboxed_generation_fixture(tmp_path, monkeypatch)
    )
    if profile == "interrupted_transition":
        _install_fixture_canonical_transition(fixture, monkeypatch)
    baseline_latest = (fixture["wip"] / "latest.json").read_bytes()
    (fixture["wip"] / "latest.json").write_bytes(
        b'{"attempt":"isolated-unpublished"}\n'
    )
    disposable = fixture["wip"] / "isolated-attempt.txt"
    disposable.write_bytes(b"must survive until authorized rollback\n")
    armed = (
        _arm_interrupted_generation(fixture, monkeypatch)
        if interrupted_exec
        else _arm_sandboxed_generation(fixture, monkeypatch)
    )
    if interrupted_exec:
        recover = lambda: _recover_interrupted_generation(fixture, armed)
    else:
        recover = lambda: R._recover_sandboxed_generation_release(
            fixture["item"],
            confirm_dispatch_id=fixture["dispatch_id"],
            confirm_recovery_nonce=armed["recovery_nonce"],
            boot_identity_provider=lambda: fixture["boot"],
        )
    real_restore = R._restore_wip_from_rollback_capsule
    observed_durable_event = False

    def partial_restore_then_crash(*_args, **_kwargs):
        nonlocal observed_durable_event
        observed_durable_event = any(
            row.get("event") == R.SANDBOX_ABANDON_EVENT
            for row in R.Guard.read_ledger(fixture["ledger"])
        )
        (fixture["wip"] / "latest.json").write_bytes(baseline_latest)
        raise RuntimeError("synthetic crash during authorized WIP restore")

    monkeypatch.setattr(
        R, "_restore_wip_from_rollback_capsule", partial_restore_then_crash
    )
    with pytest.raises(RuntimeError, match="authorized WIP restore"):
        recover()
    assert observed_durable_event is True
    assert fixture["marker"].is_file()
    assert fixture["capsule"].is_file()
    assert disposable.is_file()
    expected_prefix = (
        ["codex_exec", "codex_exec_classification_correction"]
        if interrupted_exec else []
    )
    assert [row["event"] for row in R.Guard.read_ledger(fixture["ledger"])] == [
        *expected_prefix, R.SANDBOX_ABANDON_EVENT
    ]
    if profile == "interrupted_transition":
        assert R.Guard.read_ledger(fixture["ledger"])[-1]["schema"] == (
            R.INTERRUPTED_EXEC_TRANSITION_ABANDON_EVENT_SCHEMA
        )

    monkeypatch.setattr(R, "_restore_wip_from_rollback_capsule", real_restore)
    result = recover()
    assert result["result"] == "sandbox_isolated_noncounting"
    assert not disposable.exists()
    assert (fixture["wip"] / "latest.json").read_bytes() == baseline_latest
    assert [row["event"] for row in R.Guard.read_ledger(fixture["ledger"])] == [
        *expected_prefix,
        R.SANDBOX_ABANDON_EVENT,
        "codex_dispatch_release_authorized",
    ]


def test_sandboxed_recovery_checks_canonical_before_any_wal_reconciliation(
    tmp_path, monkeypatch
):
    fixture = _sandboxed_generation_fixture(tmp_path, monkeypatch)
    armed = _arm_sandboxed_generation(fixture, monkeypatch)
    (fixture["artifact"] / "late.py").write_bytes(b"canonical drift\n")

    def state(path):
        metadata = path.stat(follow_symlinks=False)
        return (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
            metadata.st_size,
            path.read_bytes(),
        )

    before = {
        name: state(fixture[name])
        for name in ("marker", "capsule", "ledger")
    }
    called = []

    def forbidden_retire(*_args, **_kwargs):
        called.append("release_wal")
        raise AssertionError("release WAL ran before canonical gate")

    def forbidden_ledger_read(*_args, **_kwargs):
        called.append("phase_wal")
        raise AssertionError("phase WAL ran before canonical gate")

    monkeypatch.setattr(
        R, "_retire_incomplete_release_for_operator", forbidden_retire
    )
    monkeypatch.setattr(
        R, "_read_post_reboot_ledger_surface", forbidden_ledger_read
    )
    with pytest.raises(
        R.CampaignPlanError,
        match="canonical/frontier baseline changed",
    ):
        R._recover_sandboxed_generation_release(
            fixture["item"],
            confirm_dispatch_id=fixture["dispatch_id"],
            confirm_recovery_nonce=armed["recovery_nonce"],
            boot_identity_provider=lambda: fixture["boot"],
        )

    assert called == []
    assert {
        name: state(fixture[name])
        for name in ("marker", "capsule", "ledger")
    } == before


@pytest.mark.parametrize("wal_crash", ("partial_authority", "authorized"))
@pytest.mark.parametrize(
    "profile", ("zero", "one", "interrupted", "interrupted_transition")
)
def test_sandboxed_release_wal_reconciles_before_marker_replay(
    tmp_path, monkeypatch, wal_crash, profile
):
    interrupted_exec = profile in {"interrupted", "interrupted_transition"}
    fixture = (
        _interrupted_generation_fixture(tmp_path, monkeypatch)
        if interrupted_exec
        else _sandboxed_generation_fixture(tmp_path, monkeypatch)
    )
    if profile == "one":
        _append_sandbox_exec(fixture)
    if profile == "interrupted_transition":
        _install_fixture_canonical_transition(fixture, monkeypatch)
    armed = (
        _arm_interrupted_generation(fixture, monkeypatch)
        if interrupted_exec
        else _arm_sandboxed_generation(fixture, monkeypatch)
    )
    if interrupted_exec:
        recover = lambda: _recover_interrupted_generation(fixture, armed)
    else:
        recover = lambda: R._recover_sandboxed_generation_release(
            fixture["item"],
            confirm_dispatch_id=fixture["dispatch_id"],
            confirm_recovery_nonce=armed["recovery_nonce"],
            boot_identity_provider=lambda: fixture["boot"],
        )
    real_ensure = R._ensure_dispatch_release_authority_row
    real_finish = R._finish_dispatch_release_intent

    if wal_crash == "authorized":
        def crash_after_authority(
            item, root_fd, record, intent_identity, **_kwargs
        ):
            real_ensure(
                item,
                root_fd,
                record,
                intent_identity,
                allow_new_authority_append=True,
            )
            raise RuntimeError("synthetic crash after release authorization")

        monkeypatch.setattr(
            R, "_finish_dispatch_release_intent", crash_after_authority
        )
    else:
        def crash_during_authority(
            item,
            root_fd,
            record,
            intent_identity,
            *,
            allow_new_authority_append=False,
            **kwargs,
        ):
            if not allow_new_authority_append:
                return real_ensure(
                    item,
                    root_fd,
                    record,
                    intent_identity,
                    allow_new_authority_append=False,
                    **kwargs,
                )
            callback = kwargs.get("before_authority_append")
            if callback is not None:
                callback(record, intent_identity)
            authority = record["release_authority"]
            line = Recovery.canonical_json_line(
                authority["authority_record"]
            )
            descriptor = os.open(
                authority["ledger"], os.O_WRONLY | os.O_APPEND
            )
            try:
                os.write(descriptor, line[:len(line) // 2])
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            raise RuntimeError("synthetic crash during release authorization")

        monkeypatch.setattr(
            R, "_ensure_dispatch_release_authority_row", crash_during_authority
        )

    with pytest.raises(RuntimeError, match="release authorization"):
        recover()
    assert fixture["marker"].is_file()
    assert fixture["capsule"].is_file()

    monkeypatch.setattr(R, "_ensure_dispatch_release_authority_row", real_ensure)
    monkeypatch.setattr(R, "_finish_dispatch_release_intent", real_finish)
    result = recover()

    expected = (
        (
            "interrupted_sandbox_isolated_noncounting_already_completed"
            if interrupted_exec
            else "sandbox_isolated_noncounting_already_completed"
        )
        if wal_crash == "authorized"
        else "sandbox_isolated_noncounting"
    )
    assert result["result"] == expected
    assert not fixture["marker"].exists()
    assert not fixture["capsule"].exists()
    expected_events = (
        ["codex_exec", "codex_exec_classification_correction"]
        if profile != "zero" else []
    ) + [R.SANDBOX_ABANDON_EVENT, "codex_dispatch_release_authorized"]
    final_rows = R.Guard.read_ledger(fixture["ledger"])
    assert [row["event"] for row in final_rows] == expected_events
    if profile == "interrupted_transition":
        assert final_rows[-2]["schema"] == (
            R.INTERRUPTED_EXEC_TRANSITION_ABANDON_EVENT_SCHEMA
        )


@pytest.mark.parametrize("authorization", ("missing", "wrong_terminal_hash"))
def test_markerless_sandbox_completion_requires_matching_release_authority(
    tmp_path, monkeypatch, authorization
):
    fixture = _sandboxed_generation_fixture(tmp_path, monkeypatch)
    armed = _arm_sandboxed_generation(fixture, monkeypatch)
    parsed = Recovery.parse_sandboxed_generation_marker(
        fixture["marker"].read_bytes(), require_recovery_arm=True
    )
    event = R._build_sandbox_abandon_event(fixture["item"], parsed)
    R.Guard.append_ledger(event, fixture["ledger"])
    if authorization == "wrong_terminal_hash":
        R.Guard.append_ledger({
            "event": "codex_dispatch_release_authorized",
            "schema": "scheduler_dispatch_release_authorized_v1",
            "dispatch_id": fixture["dispatch_id"],
            "terminal_kind": R.SANDBOX_RELEASE_AUTHORITY_KIND,
            "terminal_event": R.SANDBOX_ABANDON_EVENT,
            "terminal_record_sha256": "f" * 64,
        }, fixture["ledger"])
    os.unlink(fixture["capsule"])
    os.unlink(fixture["marker"])
    R._fsync_directory(fixture["marker"].parent)

    with pytest.raises(
        (R.CampaignPlanError, R.NoDispatchQuarantine),
        match="authorization|quarantine",
    ):
        R._recover_sandboxed_generation_release(
            fixture["item"],
            confirm_dispatch_id=fixture["dispatch_id"],
            confirm_recovery_nonce=armed["recovery_nonce"],
            boot_identity_provider=lambda: fixture["boot"],
        )


def _detached_test_process(
    pid: int,
    ppid: int,
    pgid: int,
    sid: int,
    *,
    holder: bool,
    parent_in_closure: bool,
) -> dict[str, object]:
    started = f"darwin:{pid}:1"
    started_sha = hashlib.sha256(json.dumps(
        {"os_process_start": started, "pid": pid},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")).hexdigest()
    return {
        "pid": pid,
        "ppid": ppid,
        "pgid": pgid,
        "sid": sid,
        "state": "L",
        "start_identity": started,
        "start_identity_sha256": started_sha,
        "holder": holder,
        "parent_in_closure": parent_in_closure,
    }


def _detached_test_inventory(
    processes: list[dict[str, object]],
) -> dict[str, object]:
    groups: dict[str, list[int]] = {}
    for record in processes:
        groups.setdefault(str(record["pgid"]), []).append(int(record["pid"]))
    return {
        "holders": sorted(
            int(record["pid"]) for record in processes if record["holder"]
        ),
        "processes": processes,
        "groups": {
            key: sorted(value) for key, value in sorted(groups.items())
        },
    }


def test_detached_signal_path_is_exact_pid_and_descendants_first(monkeypatch):
    identities = {
        100: (1, 100, 100, "L", "darwin:100:1"),
        101: (100, 100, 100, "L", "darwin:101:1"),
        102: (101, 100, 100, "L", "darwin:102:1"),
    }
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(
        R.os, "uname", lambda: type("U", (), {"sysname": "Darwin"})()
    )
    monkeypatch.setattr(
        R.Contiguous, "_scoped_group_pids", lambda pgid: set(identities)
    )
    monkeypatch.setattr(
        R.Contiguous, "_process_identity", lambda pid: identities.get(pid)
    )
    monkeypatch.setattr(
        R.Contiguous,
        "_signal_owned_process_group",
        lambda *_args: pytest.fail("killpg must not be used"),
    )
    monkeypatch.setattr(R.os, "kill", lambda pid, sig: sent.append((pid, sig)))

    R._signal_authenticated_detached_groups(
        identities, {100}, signal.SIGKILL
    )

    assert sent == [
        (102, signal.SIGKILL),
        (101, signal.SIGKILL),
        (100, signal.SIGKILL),
    ]


def test_detached_signal_rejects_unauthorised_group_member(monkeypatch):
    identities = {
        100: (1, 100, 100, "L", "darwin:100:1"),
    }
    live = {
        **identities,
        101: (100, 100, 100, "L", "darwin:101:1"),
    }
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(
        R.os, "uname", lambda: type("U", (), {"sysname": "Darwin"})()
    )
    monkeypatch.setattr(
        R.Contiguous, "_scoped_group_pids", lambda pgid: set(live)
    )
    monkeypatch.setattr(
        R.Contiguous, "_process_identity", lambda pid: live.get(pid)
    )
    monkeypatch.setattr(R.os, "kill", lambda pid, sig: sent.append((pid, sig)))

    with pytest.raises(R.CampaignPlanError, match="unauthorised"):
        R._signal_authenticated_detached_groups(
            identities, {100}, signal.SIGSTOP
        )
    assert sent == []


def test_detached_signal_rejects_pid_birth_drift(monkeypatch):
    expected = {
        100: (1, 100, 100, "L", "darwin:100:1"),
    }
    rebound = {
        100: (1, 100, 100, "L", "darwin:100:2"),
    }
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(
        R.os, "uname", lambda: type("U", (), {"sysname": "Darwin"})()
    )
    monkeypatch.setattr(
        R.Contiguous, "_scoped_group_pids", lambda _pgid: {100}
    )
    monkeypatch.setattr(
        R.Contiguous, "_process_identity", lambda pid: rebound.get(pid)
    )
    monkeypatch.setattr(R.os, "kill", lambda pid, sig: sent.append((pid, sig)))

    with pytest.raises(R.CampaignPlanError, match="unauthorised"):
        R._signal_authenticated_detached_groups(
            expected, {100}, signal.SIGSTOP
        )
    assert sent == []


def test_detached_adoption_rejects_disconnected_ppid_cycle():
    anchor = _detached_test_process(
        100, 1, 100, 100, holder=True, parent_in_closure=False
    )
    armed_inventory = _detached_test_inventory([anchor])
    arm = {"anchor_pids": [100], "holder_inventory": armed_inventory}
    cycle_left = _detached_test_process(
        200, 201, 200, 200, holder=True, parent_in_closure=True
    )
    cycle_right = _detached_test_process(
        201, 200, 201, 201, holder=False, parent_in_closure=True
    )
    malicious = _detached_test_inventory(
        [anchor, cycle_left, cycle_right]
    )

    with pytest.raises(R.CampaignPlanError, match="directed anchor"):
        R._validate_delegated_detached_inventory(arm, malicious)


def test_recursive_holder_scan_parses_darwin_rc1_pid_output(
    tmp_path, monkeypatch
):
    root = R._bound_mutable_root(
        tmp_path,
        R._host_directory_identity(tmp_path, "test root"),
        label="test root",
    )
    scanner = Path("/bin/ps")
    metadata = scanner.stat(follow_symlinks=False)
    monkeypatch.setattr(
        R,
        "_authenticated_lsof_executable",
        lambda: (scanner, (metadata.st_dev, metadata.st_ino)),
    )
    monkeypatch.setattr(R, "_mutable_root_traversal_receipt", lambda _root: "x")
    invoked: list[list[str]] = []

    def fake_run(argv, **_kwargs):
        invoked.append(argv)
        return subprocess.CompletedProcess(
            argv, 1, stdout=b"123\n", stderr=b""
        )

    monkeypatch.setattr(R.subprocess, "run", fake_run)

    assert R._open_file_holder_pids(root) == frozenset({123})
    assert invoked[0][3:6] == ["-t", "+w", "+D"]
    assert "-w" not in invoked[0]


def test_recursive_holder_scan_accepts_complete_rc1_no_match(
    tmp_path, monkeypatch
):
    root = R._bound_mutable_root(
        tmp_path,
        R._host_directory_identity(tmp_path, "test root"),
        label="test root",
    )
    scanner = Path("/bin/ps")
    metadata = scanner.stat(follow_symlinks=False)
    monkeypatch.setattr(
        R,
        "_authenticated_lsof_executable",
        lambda: (scanner, (metadata.st_dev, metadata.st_ino)),
    )
    monkeypatch.setattr(R, "_mutable_root_traversal_receipt", lambda _root: "x")
    monkeypatch.setattr(
        R.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 1, stdout=b"", stderr=b""
        ),
    )

    assert R._open_file_holder_pids(root) == frozenset()


def test_recursive_holder_scan_accepts_rc1_self_only_snapshot(
    tmp_path, monkeypatch
):
    root = R._bound_mutable_root(
        tmp_path,
        R._host_directory_identity(tmp_path, "test root"),
        label="test root",
    )
    scanner = Path("/bin/ps")
    metadata = scanner.stat(follow_symlinks=False)
    observer = (1, 2, 3, "L", "darwin:1:1")
    monkeypatch.setattr(
        R,
        "_authenticated_lsof_executable",
        lambda: (scanner, (metadata.st_dev, metadata.st_ino)),
    )
    monkeypatch.setattr(R, "_mutable_root_traversal_receipt", lambda _root: "x")
    monkeypatch.setattr(
        R.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 1, stdout=f"{os.getpid()}\n".encode("ascii"), stderr=b""
        ),
    )
    monkeypatch.setattr(
        R.Contiguous,
        "_process_identity",
        lambda pid: observer if pid == os.getpid() else None,
    )
    monkeypatch.setattr(
        R,
        "_authenticated_holder_process_closure",
        lambda _pids: pytest.fail("self-only scan must not authenticate foreign PIDs"),
    )

    R._authenticated_open_file_holder_snapshot(
        (root,), phase="synthetic self-held quarantine"
    )


def test_recursive_holder_scan_rejects_self_birth_drift(
    tmp_path, monkeypatch
):
    root = R._bound_mutable_root(
        tmp_path,
        R._host_directory_identity(tmp_path, "test root"),
        label="test root",
    )
    scanner = Path("/bin/ps")
    metadata = scanner.stat(follow_symlinks=False)
    observer_identities = iter((
        (1, 2, 3, "L", "darwin:1:1"),
        (1, 2, 3, "L", "darwin:1:2"),
    ))
    monkeypatch.setattr(
        R,
        "_authenticated_lsof_executable",
        lambda: (scanner, (metadata.st_dev, metadata.st_ino)),
    )
    monkeypatch.setattr(R, "_mutable_root_traversal_receipt", lambda _root: "x")
    monkeypatch.setattr(
        R.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 1, stdout=f"{os.getpid()}\n".encode("ascii"), stderr=b""
        ),
    )
    monkeypatch.setattr(
        R.Contiguous,
        "_process_identity",
        lambda pid: next(observer_identities) if pid == os.getpid() else None,
    )

    with pytest.raises(R.CampaignPlanError, match="observer identity changed"):
        R._authenticated_open_file_holder_snapshot(
            (root,), phase="synthetic self-reuse quarantine"
        )


def test_recursive_holder_scan_rc1_foreign_pid_still_blocks(
    tmp_path, monkeypatch
):
    root = R._bound_mutable_root(
        tmp_path,
        R._host_directory_identity(tmp_path, "test root"),
        label="test root",
    )
    scanner = Path("/bin/ps")
    metadata = scanner.stat(follow_symlinks=False)
    foreign_pid = 424242
    observer = (1, 2, 3, "L", "darwin:1:1")
    foreign = (1, foreign_pid, foreign_pid, "L", "darwin:424242:1")
    monkeypatch.setattr(
        R,
        "_authenticated_lsof_executable",
        lambda: (scanner, (metadata.st_dev, metadata.st_ino)),
    )
    monkeypatch.setattr(R, "_mutable_root_traversal_receipt", lambda _root: "x")
    monkeypatch.setattr(
        R.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 1, stdout=f"{foreign_pid}\n".encode("ascii"), stderr=b""
        ),
    )
    monkeypatch.setattr(
        R.Contiguous,
        "_process_identity",
        lambda pid: observer if pid == os.getpid() else foreign,
    )
    monkeypatch.setattr(
        R,
        "_authenticated_holder_process_closure",
        lambda pids: {pid: foreign for pid in pids},
    )

    with pytest.raises(R.CampaignPlanError, match="open-file holder"):
        R._authenticated_open_file_holder_snapshot(
            (root,), phase="synthetic foreign-held quarantine"
        )


@pytest.mark.parametrize(
    ("returncode", "stdout", "stderr"),
    (
        (0, b"", b""),
        (1, b"", b"incomplete scan"),
        (2, b"", b""),
        (1, b"not-a-pid\n", b""),
    ),
)
def test_recursive_holder_scan_rejects_incomplete_or_malformed_result(
    tmp_path, monkeypatch, returncode, stdout, stderr
):
    root = R._bound_mutable_root(
        tmp_path,
        R._host_directory_identity(tmp_path, "test root"),
        label="test root",
    )
    scanner = Path("/bin/ps")
    metadata = scanner.stat(follow_symlinks=False)
    monkeypatch.setattr(
        R,
        "_authenticated_lsof_executable",
        lambda: (scanner, (metadata.st_dev, metadata.st_ino)),
    )
    monkeypatch.setattr(R, "_mutable_root_traversal_receipt", lambda _root: "x")
    monkeypatch.setattr(
        R.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], returncode, stdout=stdout, stderr=stderr
        ),
    )

    with pytest.raises(
        R.CampaignPlanError, match="complete inventory|malformed|ambiguous"
    ):
        R._open_file_holder_pids(root)


def test_recursive_holder_scan_rejects_changed_nofollow_traversal(
    tmp_path, monkeypatch
):
    root = R._bound_mutable_root(
        tmp_path,
        R._host_directory_identity(tmp_path, "test root"),
        label="test root",
    )
    scanner = Path("/bin/ps")
    metadata = scanner.stat(follow_symlinks=False)
    receipts = iter(("before", "after"))
    monkeypatch.setattr(
        R,
        "_authenticated_lsof_executable",
        lambda: (scanner, (metadata.st_dev, metadata.st_ino)),
    )
    monkeypatch.setattr(
        R, "_mutable_root_traversal_receipt", lambda _root: next(receipts)
    )
    monkeypatch.setattr(
        R.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 1, stdout=b"", stderr=b""
        ),
    )

    with pytest.raises(R.CampaignPlanError, match="authority changed"):
        R._open_file_holder_pids(root)


def test_recursive_holder_scan_rejects_unreadable_nofollow_tree(tmp_path):
    sealed = tmp_path / "sealed"
    sealed.mkdir()
    os.chmod(sealed, 0)
    root = R._bound_mutable_root(
        tmp_path,
        R._host_directory_identity(tmp_path, "test root"),
        label="test root",
    )
    try:
        with pytest.raises(R.CampaignPlanError, match="completely traversed"):
            R._open_file_holder_pids(root)
    finally:
        os.chmod(sealed, 0o700)


def test_recursive_holder_scan_rejects_cross_device_directory(
    tmp_path, monkeypatch
):
    mounted = tmp_path / "mounted"
    mounted.mkdir()
    root = R._bound_mutable_root(
        tmp_path,
        R._host_directory_identity(tmp_path, "test root"),
        label="test root",
    )
    real_stat = os.stat

    class CrossDeviceMetadata:
        def __init__(self, metadata):
            self._metadata = metadata

        @property
        def st_dev(self):
            return root.identity[0] + 1

        def __getattr__(self, name):
            return getattr(self._metadata, name)

    def cross_device_stat(path, *args, **kwargs):
        metadata = real_stat(path, *args, **kwargs)
        if path == mounted.name and kwargs.get("dir_fd") is not None:
            return CrossDeviceMetadata(metadata)
        return metadata

    monkeypatch.setattr(R.os, "stat", cross_device_stat)

    with pytest.raises(R.CampaignPlanError, match="crosses a filesystem"):
        R._mutable_root_traversal_receipt(root)


def test_abandoned_inode_proof_rejects_cross_device_nondirectory(
    tmp_path, monkeypatch
):
    foreign = tmp_path / "foreign"
    foreign.write_bytes(b"synthetic\n")
    root = R._bound_mutable_root(
        tmp_path,
        R._host_directory_identity(tmp_path, "test root"),
        label="test root",
    )
    real_stat = os.stat

    class CrossDeviceMetadata:
        def __init__(self, metadata):
            self._metadata = metadata

        @property
        def st_dev(self):
            return root.identity[0] + 1

        def __getattr__(self, name):
            return getattr(self._metadata, name)

    def cross_device_stat(path, *args, **kwargs):
        metadata = real_stat(path, *args, **kwargs)
        if path == foreign.name and kwargs.get("dir_fd") is not None:
            return CrossDeviceMetadata(metadata)
        return metadata

    monkeypatch.setattr(R.os, "stat", cross_device_stat)

    with pytest.raises(R.CampaignPlanError, match="crosses a filesystem"):
        R._mutable_root_traversal_receipt(
            root, require_same_device=True
        )


def test_recursive_holder_scan_rejects_directory_to_symlink_race(
    tmp_path, monkeypatch
):
    child = tmp_path / "child"
    child.mkdir()
    root = R._bound_mutable_root(
        tmp_path,
        R._host_directory_identity(tmp_path, "test root"),
        label="test root",
    )
    real_stat = os.stat
    swapped = False

    def swapping_stat(path, *args, **kwargs):
        nonlocal swapped
        metadata = real_stat(path, *args, **kwargs)
        descriptor = kwargs.get("dir_fd")
        if path == child.name and descriptor is not None and not swapped:
            os.rename(
                child.name,
                "child-original",
                src_dir_fd=descriptor,
                dst_dir_fd=descriptor,
            )
            os.symlink("..", child.name, dir_fd=descriptor)
            swapped = True
        return metadata

    monkeypatch.setattr(R.os, "stat", swapping_stat)

    with pytest.raises(R.CampaignPlanError, match="completely traversed"):
        R._mutable_root_traversal_receipt(root)
    assert swapped


def test_recursive_holder_scan_bounds_scandir_while_consuming(
    tmp_path, monkeypatch
):
    root = R._bound_mutable_root(
        tmp_path,
        R._host_directory_identity(tmp_path, "test root"),
        label="test root",
    )
    consumed = 0

    class Entry:
        def __init__(self, name):
            self.name = name

    class Entries:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def __iter__(self):
            nonlocal consumed
            for index in range(5):
                consumed += 1
                yield Entry(f"entry-{index}")

    monkeypatch.setattr(R, "MAX_OPEN_FILE_HOLDER_PIDS", 1)
    monkeypatch.setattr(R.os, "scandir", lambda _descriptor: Entries())

    with pytest.raises(R.CampaignPlanError, match="exceeded its entry bound"):
        R._mutable_root_traversal_receipt(root)
    assert consumed == 5


def test_mutation_guard_holder_scan_is_foreign_reject_only(monkeypatch):
    observer = (1, 2, 3, "L", "darwin:1:1")
    foreign_pid = 424242
    foreign = (1, foreign_pid, foreign_pid, "L", "darwin:424242:1")
    monkeypatch.setattr(
        R, "_open_file_holder_pids", lambda _root: frozenset({foreign_pid})
    )
    monkeypatch.setattr(
        R.Contiguous,
        "_process_identity",
        lambda pid: observer if pid == os.getpid() else foreign,
    )
    monkeypatch.setattr(
        R,
        "_authenticated_holder_process_closure",
        lambda pids: {pid: foreign for pid in pids},
    )
    monkeypatch.setattr(
        R,
        "_signal_authenticated_detached_groups",
        lambda *_args, **_kwargs: pytest.fail(
            "mutation-guard holders must never authorize a signal"
        ),
    )

    with pytest.raises(R.CampaignPlanError, match="open-file holder"):
        R._reject_discovered_mutation_guard_holders(
            (object(),), phase="synthetic mutation guard"
        )


def test_detached_adoption_namespace_has_constructive_completion_bound(
    tmp_path,
):
    os.chmod(tmp_path, 0o700)
    root_fd = os.open(tmp_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    dispatch_id = "1" * 32
    recovery_nonce = "2" * 32
    marker = R.DispatchQuarantine(
        root=tmp_path,
        root_fd=root_fd,
        root_identity=(tmp_path.stat().st_dev, tmp_path.stat().st_ino),
        name="ar25.jsonl",
        path=tmp_path / "ar25.jsonl",
        marker_fd=-1,
        marker_identity=(3, 4),
        dispatch_id=dispatch_id,
    )
    prefix = (
        f".{marker.name}.{dispatch_id}.{recovery_nonce}."
        "detached_teardown_adoption_"
    )
    try:
        for index in range(R.MAX_DETACHED_TEARDOWN_ADOPTIONS + 1):
            path = tmp_path / f"{prefix}{index:032x}"
            path.write_bytes(b"")
            os.chmod(path, 0o600)
        with pytest.raises(R.CampaignPlanError, match="exceeded its bound"):
            R._read_detached_teardown_adoptions(
                marker,
                {"recovery_nonce": recovery_nonce},
                (5, 6),
            )
    finally:
        os.close(root_fd)


def test_detached_teardown_cli_actions_are_exposed(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["runner", "--help"])
    with pytest.raises(SystemExit) as stopped:
        R.main()
    assert stopped.value.code == 0
    help_text = capsys.readouterr().out
    assert "--arm-detached-generation-teardown" in help_text
    assert "--execute-detached-generation-teardown" in help_text
    assert "--confirm-stale-recovery-nonce" in help_text


@pytest.mark.parametrize(
    ("arguments", "message"),
    (
        (
            [
                "--arm-detached-generation-teardown=ar25",
                "--confirm-dispatch-id=" + "1" * 32,
            ],
            "requires only --confirm-stale-recovery-nonce",
        ),
        (
            [
                "--execute-detached-generation-teardown=ar25",
                "--confirm-dispatch-id=" + "1" * 32,
                "--confirm-recovery-nonce=" + "2" * 32,
                "--confirm-stale-recovery-nonce=" + "3" * 32,
            ],
            "accepted only by detached teardown arm",
        ),
        (
            [
                "--arm-detached-generation-teardown=ar25",
                "--confirm-dispatch-id=" + "1" * 32,
                "--confirm-stale-recovery-nonce=" + "3" * 32,
                "--confirm-current-wip-state-sha256=" + "4" * 64,
            ],
            "does not accept current-WIP confirmation",
        ),
    ),
)
def test_detached_teardown_cli_rejects_nonce_flag_mix(
    tmp_path, monkeypatch, arguments, message
):
    plan = tmp_path / "plan.json"
    plan.write_text('{"initial_queue": []}', encoding="utf-8")
    monkeypatch.setattr(R, "_authoritative_targets", lambda: {})
    monkeypatch.setattr(
        sys,
        "argv",
        ["runner", f"--plan={plan}", *arguments],
    )

    with pytest.raises(R.CampaignPlanError, match=message):
        R.main()


def _arm_detached_test_fixture(fixture, monkeypatch):
    stale = (
        _arm_interrupted_generation(fixture, monkeypatch)
        if "execution" in fixture
        else _arm_sandboxed_generation(fixture, monkeypatch)
    )
    inventory = _detached_test_inventory([
        _detached_test_process(
            900001,
            1,
            900001,
            900001,
            holder=True,
            parent_in_closure=False,
        )
    ])
    scanner = Path("/bin/ps")
    scanner_stat = scanner.stat(follow_symlinks=False)
    monkeypatch.setattr(
        R,
        "_authenticated_lsof_executable",
        lambda: (
            scanner,
            (scanner_stat.st_dev, scanner_stat.st_ino),
        ),
    )
    monkeypatch.setattr(
        R,
        "_fixed_point_detached_holder_inventory",
        lambda *_args, **_kwargs: copy.deepcopy(inventory),
    )
    monkeypatch.setattr(
        R, "_reject_discovered_mutation_guard_holders", lambda *_a, **_k: None
    )
    detached = R._arm_detached_generation_teardown(
        fixture["item"],
        confirm_dispatch_id=fixture["dispatch_id"],
        confirm_stale_recovery_nonce=stale["recovery_nonce"],
        boot_identity_provider=lambda: fixture["boot"],
    )
    return stale, detached


def _install_detached_test_completion(fixture):
    marker, _parsed = R._read_existing_dispatch_quarantine(
        fixture["item"],
        require_recovery_arm=True,
        marker_parser=R._parse_detached_teardown_marker,
    )
    arm_name, completion_name, _correction_name, preparing_name = (
        R._detached_teardown_names(marker.name, marker.dispatch_id)
    )
    arm, arm_identity, _payload = R._read_durable_recovery_record_at(
        marker.root_fd,
        arm_name,
        root_path=marker.root,
        root_identity=marker.root_identity,
        label="detached teardown arm",
    )
    arm = R._validate_detached_teardown_arm_record(
        arm, marker_name=marker.name
    )
    observed = datetime.now(timezone.utc).isoformat()
    completion = {
        "schema": R.DETACHED_TEARDOWN_COMPLETION_SCHEMA,
        "event": R.DETACHED_TEARDOWN_COMPLETION_EVENT,
        "recorded_at": observed,
        "dispatch_id": marker.dispatch_id,
        "recovery_nonce": arm["recovery_nonce"],
        "arm_record_sha256": R._recovery_record_sha256(arm),
        "arm_identity": list(arm_identity),
        "holder_inventory_sha256": arm["holder_inventory_sha256"],
        "adoption_record_names": [],
        "adoption_record_sha256s": [],
        **R._sandboxed_generation_tree_hashes(
            fixture["workspace"], fixture["protected"]
        ),
        "boundary_finding_counts": (
            {"dynamic_execution": 1} if "execution" in fixture else {}
        ),
        "terminal_taint_scan_passed": True,
        "captured_process_identities_absent": True,
        "captured_groups_absent": True,
        "open_file_holders_absent": True,
        "stale_recovery_arm_retirement_authorized": True,
        "absence_sample_count": R.OPEN_FILE_HOLDER_ABSENCE_SAMPLES,
        "absence_window_ns": 1,
        "absence_first_at": observed,
        "absence_last_at": observed,
    }
    completion = R._validate_detached_teardown_completion_record(
        completion,
        marker=marker,
        arm=arm,
        arm_identity=arm_identity,
    )
    R._install_durable_recovery_record_at(
        marker.root_fd,
        completion_name,
        completion,
        root_path=marker.root,
        root_identity=marker.root_identity,
        label="detached teardown completion",
    )
    return marker, arm, completion, preparing_name


@pytest.mark.parametrize("staging", ("partial", "complete"))
def test_detached_marker_retirement_recovers_staging_cut(
    tmp_path, monkeypatch, staging
):
    fixture = _interrupted_generation_fixture(
        tmp_path,
        monkeypatch,
        boundary_finding_counts={"dynamic_execution": 1},
    )
    _arm_detached_test_fixture(fixture, monkeypatch)
    marker, arm, completion, preparing_name = (
        _install_detached_test_completion(fixture)
    )
    try:
        rows = _canonical_rows(fixture["marker"])
        prefix = b"".join(
            Recovery.canonical_json_line(row) for row in rows[:2]
        )
        staged = marker.root / preparing_name
        staged.write_bytes(prefix if staging == "complete" else prefix[:7])
        os.chmod(staged, 0o600)

        correction = R._retire_detached_teardown_stale_arm(
            marker, arm, completion
        )
    finally:
        R._close_dispatch_quarantine(marker)

    assert correction["stale_recovery_arm_invalidated"] is True
    assert len(_canonical_rows(fixture["marker"])) == 2
    R._assert_no_active_detached_teardown(fixture["item"])


def test_detached_correction_allows_fresh_bound_sandbox_arm(
    tmp_path, monkeypatch
):
    fixture = _interrupted_generation_fixture(
        tmp_path,
        monkeypatch,
        boundary_finding_counts={"dynamic_execution": 1},
    )
    stale, _detached = _arm_detached_test_fixture(fixture, monkeypatch)
    marker, arm, completion, _preparing_name = (
        _install_detached_test_completion(fixture)
    )
    try:
        correction = R._retire_detached_teardown_stale_arm(
            marker, arm, completion
        )
    finally:
        R._close_dispatch_quarantine(marker)

    R._assert_no_active_detached_teardown(fixture["item"])
    fresh = R._arm_interrupted_generation_release(
        fixture["item"],
        confirm_dispatch_id=fixture["dispatch_id"],
        boot_identity_provider=lambda: fixture["boot"],
    )
    assert fresh["recovery_nonce"] != stale["recovery_nonce"]
    assert correction["new_marker_identity"] == _canonical_rows(
        fixture["marker"]
    )[2]["pre_arm_marker_identity"]
    R._assert_no_active_detached_teardown(fixture["item"])


def test_detached_retirement_recovers_post_rename_pre_correction(
    tmp_path, monkeypatch
):
    fixture = _interrupted_generation_fixture(
        tmp_path,
        monkeypatch,
        boundary_finding_counts={"dynamic_execution": 1},
    )
    _arm_detached_test_fixture(fixture, monkeypatch)
    marker, arm, completion, _preparing_name = (
        _install_detached_test_completion(fixture)
    )
    rows = _canonical_rows(fixture["marker"])
    prefix = b"".join(
        Recovery.canonical_json_line(row) for row in rows[:2]
    )
    R._close_dispatch_quarantine(marker)
    replacement = fixture["marker"].parent / ".synthetic_prefix_replace"
    replacement.write_bytes(prefix)
    os.chmod(replacement, 0o600)
    os.replace(replacement, fixture["marker"])
    R._fsync_directory(fixture["marker"].parent)

    rebound, _parsed = R._read_existing_dispatch_quarantine(
        fixture["item"],
        require_recovery_arm=False,
        marker_parser=R._parse_detached_teardown_marker,
    )
    try:
        correction = R._retire_detached_teardown_stale_arm(
            rebound, arm, completion
        )
    finally:
        R._close_dispatch_quarantine(rebound)

    assert correction["stale_recovery_arm_invalidated"] is True
    assert correction["old_marker_identity"] != correction["new_marker_identity"]
    R._assert_no_active_detached_teardown(fixture["item"])


def test_detached_partial_anchor_loss_uses_adopted_replay(monkeypatch):
    first = _detached_test_process(
        100, 1, 100, 100, holder=True, parent_in_closure=False
    )
    second = _detached_test_process(
        200, 1, 200, 200, holder=True, parent_in_closure=False
    )
    arm = {
        "anchor_pids": [100, 200],
        "holder_inventory": _detached_test_inventory([first, second]),
    }
    identities = {
        100: None,
        200: (1, 200, 200, "L", "darwin:200:1"),
    }
    monkeypatch.setattr(
        R.Contiguous, "_process_identity", lambda pid: identities[pid]
    )

    assert R._detached_anchor_births_live(arm) is False

    root = R.BoundMutableRoot(
        label="abandoned workspace",
        path=Path("/synthetic/workspace"),
        identity=(1, 2),
    )
    monkeypatch.setattr(
        R, "_discovery_lsof_holder_pids", lambda *_a, **_k: frozenset({200})
    )
    monkeypatch.setattr(R.time, "sleep", lambda _seconds: None)
    cumulative = {
        100: (1, 100, 100, "L", "darwin:100:1"),
        200: (1, 200, 200, "L", "darwin:200:1"),
    }
    R._prove_current_workspace_holders_are_adopted((root,), cumulative)

    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(
        R.os, "uname", lambda: type("U", (), {"sysname": "Darwin"})()
    )
    monkeypatch.setattr(
        R.Contiguous,
        "_scoped_group_pids",
        lambda pgid: {200} if pgid == 200 else set(),
    )
    monkeypatch.setattr(R.os, "kill", lambda pid, sig: sent.append((pid, sig)))
    R._signal_authenticated_detached_groups(
        cumulative, {100, 200}, signal.SIGKILL
    )
    assert sent == [(200, signal.SIGKILL)]


def test_detached_completion_replay_reproves_terminal_state(
    tmp_path, monkeypatch
):
    fixture = _interrupted_generation_fixture(
        tmp_path,
        monkeypatch,
        boundary_finding_counts={"dynamic_execution": 1},
    )
    _stale, detached = _arm_detached_test_fixture(fixture, monkeypatch)
    marker, _arm, _completion, _preparing = (
        _install_detached_test_completion(fixture)
    )
    R._close_dispatch_quarantine(marker)
    calls = {"process": 0, "holders": 0, "envelope": 0}

    def process_absence(*_args, **_kwargs):
        calls["process"] += 1
        observed = datetime.now(timezone.utc).isoformat()
        return {
            "absence_sample_count": R.OPEN_FILE_HOLDER_ABSENCE_SAMPLES,
            "absence_window_ns": 1,
            "absence_first_at": observed,
            "absence_last_at": observed,
        }

    def holder_absence(*_args, **_kwargs):
        calls["holders"] += 1

    def envelope(selected, *_args, **_kwargs):
        calls["envelope"] += 1
        return (
            selected,
            None,
            None,
            None,
            fixture["execution"],
            (),
            (),
            fixture["workspace"],
            fixture["protected"],
        )

    monkeypatch.setattr(
        R, "_prove_detached_process_and_group_absence", process_absence
    )
    monkeypatch.setattr(
        R, "_prove_bound_mutable_root_holder_absence", holder_absence
    )
    monkeypatch.setattr(
        R, "_validate_detached_teardown_execution_envelope", envelope
    )
    monkeypatch.setattr(
        R,
        "_detached_terminal_boundary_counts",
        lambda *_args, **_kwargs: {"dynamic_execution": 1},
    )

    result = R._execute_detached_generation_teardown(
        fixture["item"],
        confirm_dispatch_id=fixture["dispatch_id"],
        confirm_recovery_nonce=detached["recovery_nonce"],
        boot_identity_provider=lambda: fixture["boot"],
    )

    assert result["result"] == "detached_generation_teardown_already_completed"
    assert calls == {"process": 2, "holders": 2, "envelope": 2}
    assert len(_canonical_rows(fixture["marker"])) == 2
