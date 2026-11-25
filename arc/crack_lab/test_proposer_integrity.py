"""Regression cases for caught violations that still invalidate a whole turn."""
import json
import sys
import time

import pytest

import gkm_legs as G
from proposer_integrity import codex_reported_violations, external_temp_write


def transcript(message):
    return json.dumps({"type": "item.completed", "item": {
        "type": "agent_message", "text": message,
    }}) + "\n"


@pytest.mark.parametrize("message,kind", [
    ("A probe supplied malformed `(7,)`; local validation rejected it. "
     "Your rule invalidates this turn, so I’m stopping.",
     "public_action_protocol_violation"),
    ("Turn invalidated: a probe attempted action 6 with x=65, "
     "outside the allowed 0..63 frame. I stopped immediately.",
     "public_action_protocol_violation"),
    ("Stopped: a probe generated an out-of-frame coordinate, rejected locally "
     "before `env.step`. Under your rule, this invalidates and terminates this "
     "proposer turn. Level 9 remains unmet. `checkpoint.json` is unchanged.",
     "public_action_protocol_violation"),
    ("I violated your workspace-only boundary by compiling a probe in /tmp, "
     "invalidating this attempt.", "workspace_boundary_violation"),
    ("I found a boundary violation in my work: I compiled a worker outside "
     "the workspace. Your rule invalidates this attempt.",
     "workspace_boundary_violation"),
    ("Stopped: a shell heredoc attempted to create a temporary file outside "
     "the workspace and was blocked. Under your rule, this invalidates "
     "the discovery lineage. This attempt must be discarded and restarted "
     "from the admitted parent.", "workspace_boundary_violation"),
    ("A compiler attempted to write outside the workspace and was blocked. "
     "The discovery lineage was invalidated.", "workspace_boundary_violation"),
    ("A shell here-document attempted a temporary-file write at an uncontrolled "
     "location and was blocked. I’m treating this attempt as invalid under your "
     "clean-room boundary and stopping here.", "workspace_boundary_violation"),
    ("I violated the workspace boundary by writing outside the workspace. "
     "This invalidates the discovery lineage.", "workspace_boundary_violation"),
])
def test_explicit_violation_blocks_runtime_and_release_audits(tmp_path, message, kind):
    import audit_submission_taint as audit
    raw = transcript(message)
    assert kind in codex_reported_violations(raw)
    path = tmp_path / "codex_turn.jsonl"
    path.write_text(raw)
    assert G._file_taint_reason(str(path), path.name) is not None
    assert kind in audit.scan_file(path)
    assert kind in {i["kind"] for i in audit.audit_transcript_containment(tmp_path)["incidents"]}


@pytest.mark.parametrize("message", [
    "A future turn would be invalidated if a probe attempted an out-of-frame "
    "coordinate outside 0..63.",
    "If a probe supplied malformed `(7,)`, the rule invalidates this turn.",
    "I have not violated the workspace boundary. I will keep outputs local.",
    "If a shell heredoc attempted to create a temporary file outside the "
    "workspace, the rule invalidates the discovery lineage.",
    "A shell heredoc would be blocked if it attempted to write outside the "
    "workspace. That invalidates the discovery lineage.",
    "A shell heredoc attempted to create a file inside the workspace. Writing "
    "outside the workspace invalidates the discovery lineage.",
    "If a shell here-document attempted a temporary-file write at an uncontrolled "
    "location, I would treat this attempt as invalid under the clean-room boundary.",
])
def test_policy_discussion_is_not_a_violation(message):
    assert codex_reported_violations(transcript(message)) == set()


@pytest.mark.parametrize("message", [
    "A probe generated an out-of-frame coordinate. This invalidates and terminates this proposer turn.",
    "Stopped: a shell heredoc attempted to create a temporary file outside "
    "the workspace and was blocked. Under your rule, this invalidates the discovery lineage.",
    "A shell here-document attempted a temporary-file write at an uncontrolled "
    "location and was blocked. I’m treating this attempt as invalid under your "
    "clean-room boundary and stopping here.",
    G.A.PUBLIC_ACTION_PROTOCOL_VIOLATION_MARKER,
])
def test_incremental_scan_detects_a_split_record(tmp_path, message):
    raw = transcript(message).encode()
    path = tmp_path / "capture.jsonl"
    split = len(raw) // 2
    path.write_bytes(raw[:split])
    found, offset, carry = G._scan_transcript_protocol_marker(str(path), 0, b"")
    assert not found
    with path.open("ab") as handle:
        handle.write(raw[split:])
    found, offset, carry = G._scan_transcript_protocol_marker(str(path), offset, carry)
    assert found
    assert offset == len(raw)
    assert carry == b""


@pytest.mark.parametrize("command", [
    "clang++ probe.cpp -O3 -o /tmp/probe",
    "subprocess.run(['clang++', 'probe.cpp', '-o', '/private/tmp/probe'])",
    "cat > /tmp/probe.py <<'PY'",
    "Path('/tmp/result.json').write_text(payload)",
    "open('/tmp/result.json', 'w').write(payload)",
])
def test_explicit_external_temp_outputs_are_detected(tmp_path, command):
    import audit_submission_taint as audit
    assert external_temp_write(command)
    path = tmp_path / "codex_turn.jsonl"
    path.write_text(json.dumps({"type": "item.completed", "item": {
        "type": "command_execution", "command": command, "aggregated_output": "",
    }}) + "\n")
    assert G._file_taint_reason(str(path), path.name) is not None
    assert "workspace_boundary_violation" in audit.scan_file(path)
    assert "workspace_boundary_violation" in {i["kind"] for i in audit.audit_transcript_containment(tmp_path)["incidents"]}


@pytest.mark.parametrize("command", [
    "clang++ probe.cpp -O3 -o .tmp/probe",
    "subprocess.run(['clang++', 'probe.cpp', '-o', 'probe'])",
    "cat > notes.md <<'MD'",
    "open('/tmp/public.log', 'r').read()",
    "python probe.py > /dev/null",
])
def test_workspace_outputs_and_read_only_paths_are_allowed(command):
    assert not external_temp_write(command)


@pytest.mark.parametrize("message", [
    "A probe supplied malformed `(7,)`; local validation rejected it. Your rule invalidates this turn.",
    "I violated your workspace-only boundary by compiling a probe in /tmp, invalidating this attempt.",
    "Stopped: a shell heredoc attempted to create a temporary file outside "
    "the workspace and was blocked. Under your rule, this invalidates the discovery lineage.",
])
def test_invalid_protected_turn_cannot_create_resumable_wip(tmp_path, monkeypatch, message):
    ws = tmp_path / "scratch" / "workspace"
    ws.mkdir(parents=True)
    (ws / "probe.py").write_text("print(1)\n")
    protected = ws.parent / ".proposer_transcripts" / ws.name
    protected.mkdir(parents=True)
    (protected / "codex_turn_invalid.jsonl").write_text(transcript(message))
    artifact = tmp_path / "artifact"
    monkeypatch.setattr(G, "artifact_dir", lambda game, tag="": str(artifact))
    with pytest.raises(G.WorkspaceTainted):
        G.snapshot_wip_context("test", str(ws), 2, "after_propose", reached=1, verbose=False)
    assert not (artifact / "wip_context").exists()


@pytest.mark.parametrize("exit_immediately", [False, True])
def test_boundary_self_invalidation_is_typed_before_return(
    tmp_path, monkeypatch, exit_immediately,
):
    ws = tmp_path / "ws"
    ws.mkdir()
    ledger = tmp_path / "usage.jsonl"
    snapshot = {"rateLimitsByLimitId": {"codex": {
        "planType": "plus",
        "secondary": {
            "usedPercent": 0, "resetsAt": 1_800_000_000,
            "windowDurationMins": 10_080,
        },
    }}}
    monkeypatch.setattr(G.CUG, "query_rate_limits", lambda: snapshot)
    message = (
        "Stopped: a shell heredoc attempted to create a temporary file outside "
        "the workspace and was blocked. Under your rule, this invalidates "
        "the discovery lineage."
    )
    events = [
        {"type": "thread.started", "thread_id": "boundary-report"},
        {"type": "item.completed", "item": {
            "type": "agent_message", "text": message,
        }},
    ]
    if exit_immediately:
        events.append({"type": "turn.completed", "usage": {
            "input_tokens": 11, "output_tokens": 7,
        }})
    child_code = (
        "import json,time\n"
        f"events={events!r}\n"
        "for event in events: print(json.dumps(event), flush=True)\n"
        + ("" if exit_immediately else "time.sleep(60)\n")
    )
    monkeypatch.setattr(
        G, "_codex_command",
        lambda ws, task, model, effort: [sys.executable, "-u", "-c", child_code],
    )
    started = time.monotonic()
    with pytest.raises(G.ProposerProtocolViolation, match="workspace boundary"):
        G._codex_agent(
            str(ws), "offline boundary regression", None, 1,
            ledger_path=str(ledger), run_label="boundary:L1:propose",
            game="boundary", target_level=1,
        )
    assert time.monotonic() - started < 10
    row = json.loads(ledger.read_text())
    assert row["failure_class"] == "taint"
    assert row["failure_detail_class"] == "workspace_boundary_violation"
    assert row["public_action_protocol_violation"] is False
    assert row["protected_transcript_status"] == "sealed"
    assert G._workspace_or_protected_taint_reason(str(ws)) is not None
