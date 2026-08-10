from __future__ import annotations

import json
import uuid
from pathlib import Path

from roboarm_game.gkm.taint import inspect_generation, protected_manifest
from roboarm_game.gkm.arena import RoboArmConnector
from roboarm_game.gkm.workspace import materialize_workspace


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _workspace(label: str) -> tuple[Path, Path, dict[str, str]]:
    root = (
        PROJECT_ROOT
        / "artifacts"
        / "gkm-tests"
        / f"taint-{label}-{uuid.uuid4().hex}"
    )
    root.mkdir(parents=True)
    connector = RoboArmConnector()
    public_evidence = {
        "schema_version": 1,
        "kind": "roboarm_host_sealed_public_evidence",
        "game_id": "rb01-v1",
        "round_id": "rb01-round-1",
        "seed": 0,
        "generation": 1,
        "initial_observation": connector.initial_observation(),
        "attempts": [],
        "host_feedback": [],
        "authority_boundary": {
            "connector_visible_to_proposer": False,
        },
        "receipt_sha256": "test-fixture",
    }
    workspace = materialize_workspace(
        root / "workspace",
        write_root=root,
        public_evidence=public_evidence,
        generation=1,
    ).root
    transcript = root / "transcript.jsonl"
    return workspace, transcript, protected_manifest(workspace)


def _command(transcript: Path, command: str) -> None:
    transcript.write_text(
        json.dumps(
            {
                "type": "item.completed",
                "item": {
                    "type": "command_execution",
                    "command": command,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_generation_taints_explicit_slash_tmp_write() -> None:
    workspace, transcript, baseline = _workspace("slash-tmp")
    _command(
        transcript,
        "python3 -c 'open(\"/tmp/not-allowed\", \"w\").write(\"x\")'",
    )

    report = inspect_generation(workspace, transcript, baseline)

    assert not report.clean
    assert any("host path access" in reason for reason in report.reasons)


def test_generation_rejects_inline_player_steps() -> None:
    workspace, transcript, baseline = _workspace("inline-step")
    _command(transcript, "python3 gkm_propose.py")
    (workspace / "players.py").write_text(
        """\
from legs import *


def propose_level_1(evidence):
    evidence.step(1)
""",
        encoding="utf-8",
    )

    report = inspect_generation(workspace, transcript, baseline)

    assert not report.clean
    assert any("calls step inline" in reason for reason in report.reasons)


def test_generation_allows_python_nc_coordinate_variable() -> None:
    workspace, transcript, baseline = _workspace("python-nc-variable")
    _command(
        transcript,
        "python3 - <<'PY'\nnr,nc=r+dr,c+dc\nnc = nc + 1\nPY",
    )

    report = inspect_generation(workspace, transcript, baseline)

    assert report.clean


def test_generation_taints_network_client_commands() -> None:
    for executable in ("curl", "nc"):
        workspace, transcript, baseline = _workspace(f"network-{executable}")
        _command(transcript, f"{executable} example.invalid")

        report = inspect_generation(workspace, transcript, baseline)

        assert not report.clean
        assert any("external network" in reason for reason in report.reasons)
