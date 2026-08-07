from __future__ import annotations

import copy
import errno
import hashlib
import json
import os
import stat
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
