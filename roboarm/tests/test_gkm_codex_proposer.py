from __future__ import annotations

import json
import os
import stat
import uuid
from pathlib import Path

from roboarm_game.gkm.runner import (
    CODEX_PERMISSION_PROFILE,
    CampaignConfig,
    run_codex_proposer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _case_root(label: str) -> Path:
    root = (
        PROJECT_ROOT
        / "artifacts"
        / "gkm-tests"
        / f"codex-{label}-{uuid.uuid4().hex}"
    )
    root.mkdir(parents=True)
    return root


def _fake_codex(root: Path, body: str) -> Path:
    executable = root / "bin" / "codex"
    executable.parent.mkdir()
    executable.write_text(
        "#!/bin/sh\nset -eu\n" + body,
        encoding="utf-8",
    )
    executable.chmod(
        executable.stat().st_mode
        | stat.S_IXUSR
        | stat.S_IXGRP
        | stat.S_IXOTH
    )
    return executable


def _run(root: Path, monkeypatch, body: str, *, timeout: int = 5):
    executable = _fake_codex(root, body)
    monkeypatch.setenv(
        "PATH",
        f"{executable.parent}{os.pathsep}/usr/bin{os.pathsep}/bin",
    )
    workspace = root / "workspace"
    evidence = root / "evidence"
    workspace.mkdir()
    evidence.mkdir()
    transcript = evidence / "proposer.jsonl"
    stderr = evidence / "proposer.stderr"
    code = run_codex_proposer(
        workspace,
        "public proposal-only prompt",
        transcript,
        stderr,
        CampaignConfig(
            model="gpt-5.6-sol",
            provider="codex",
            reasoning_effort="high",
            proposer_timeout_seconds=timeout,
        ),
    )
    return code, transcript, stderr


def test_headless_codex_seals_transcript_without_actuation_channel(
    monkeypatch,
):
    root = _case_root("complete")
    event = json.dumps(
        {
            "type": "item.completed",
            "item": {
                "type": "agent_message",
                "text": "valid scenario proposal emitted",
            },
        },
        separators=(",", ":"),
    )
    code, transcript, stderr = _run(
        root,
        monkeypatch,
        "if IFS= read -r unexpected; then exit 23; fi\n"
        "printf '%s\\n' \"$@\" > codex-args.txt\n"
        f"printf '%s\\n' '{event}'\n",
    )

    assert code == 0
    assert transcript.read_text().strip() == event
    assert stderr.read_text() == ""
    assert (
        transcript.with_suffix(".last.md").read_text()
        == "valid scenario proposal emitted\n"
    )
    containment = json.loads(
        transcript.with_suffix(".containment.json").read_text()
    )
    assert containment["process_group_quiesced"] is True
    assert containment["boundary_marker"] is False
    assert containment["timed_out"] is False
    assert containment["permission_profile"] == CODEX_PERMISSION_PROFILE
    assert containment["network_proxy_enabled"] is True
    assert containment["sandbox_network_enabled"] is False
    assert containment["allowlisted_unix_sockets"] == []
    assert containment["actuation_channel_present"] is False
    assert containment["web_search_disabled"] is True
    arguments = (root / "workspace" / "codex-args.txt").read_text()
    assert "features.network_proxy.enabled=true" in arguments
    assert (
        f'permissions.{CODEX_PERMISSION_PROFILE}.extends=":workspace"'
        in arguments
    )
    assert (
        f"permissions.{CODEX_PERMISSION_PROFILE}.network.enabled=false"
        in arguments
    )
    assert (
        f"permissions.{CODEX_PERMISSION_PROFILE}.network.unix_sockets={{}}"
        in arguments
    )
    assert f'default_permissions="{CODEX_PERMISSION_PROFILE}"' in arguments
    assert "project_doc_max_bytes=0" in arguments
    assert "web_search=\"disabled\"" in arguments
    assert ".arena.json" not in arguments
    assert str(root / "evidence") not in arguments
    assert str(root / "workspace" / ".tmp" / "codex") in arguments


def test_headless_codex_quarantines_actuation_boundary_marker(monkeypatch):
    root = _case_root("boundary")
    code, transcript, stderr = _run(
        root,
        monkeypatch,
        "printf '%s\\n' "
        "'PROPOSER_ACTUATION_BOUNDARY_VIOLATION: attempted live step'\n"
        "sleep 60\n",
    )

    assert code == 65
    assert "PROPOSER_ACTUATION_BOUNDARY_VIOLATION" in stderr.read_text()
    containment = json.loads(
        transcript.with_suffix(".containment.json").read_text()
    )
    assert containment["boundary_marker"] is True
    assert containment["process_group_quiesced"] is True


def test_headless_codex_timeout_stops_spawned_process_group(monkeypatch):
    root = _case_root("timeout")
    code, transcript, stderr = _run(
        root,
        monkeypatch,
        "sleep 60 &\nwait\n",
        timeout=1,
    )

    assert code == 124
    assert "PROPOSER_TIMEOUT" in stderr.read_text()
    containment = json.loads(
        transcript.with_suffix(".containment.json").read_text()
    )
    assert containment["timed_out"] is True
    assert containment["process_group_quiesced"] is True
