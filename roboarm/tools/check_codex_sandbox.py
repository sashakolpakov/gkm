"""No-model proof of the proposal-only headless Codex sandbox boundary."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import uuid

from roboarm_game.gkm.arena import RoboArmConnector
from roboarm_game.gkm.replay import write_json
from roboarm_game.gkm.runner import (
    PROJECT_ROOT,
    _codex_environment,
    _codex_permission_configs,
)
from roboarm_game.gkm.workspace import materialize_workspace


CLIENT_SOURCE = '''\
from __future__ import annotations

import base64
import json
from pathlib import Path
import socket
import sys


def blocked(operation):
    try:
        operation()
    except (ImportError, OSError, PermissionError):
        return True
    return False


evidence = json.loads(
    Path("evidence.json").read_text(encoding="utf-8")
)
raw_frame = base64.b64decode(
    evidence["initial_observation"]["frame_b64"],
    validate=True,
)
private_path = Path(sys.argv[1])
tcp_port = int(sys.argv[2])
unix_path = sys.argv[3]
outside_write = Path(sys.argv[4])


def tcp_connect():
    connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connection.settimeout(2.0)
    try:
        connection.connect(("127.0.0.1", tcp_port))
    finally:
        connection.close()


def unix_connect():
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.settimeout(2.0)
    try:
        connection.connect(unix_path)
    finally:
        connection.close()


result = {
    "public_frame_bytes": len(raw_frame),
    "sensor_contract_id":
        evidence["initial_observation"]["sensor_contract_id"],
    "public_telemetry_present": isinstance(
        evidence["initial_observation"].get("telemetry"),
        dict,
    ),
    "private_import_blocked": blocked(
        lambda: __import__("roboarm_game.gkm.arena")
    ),
    "private_read_blocked": blocked(private_path.read_bytes),
    "tcp_blocked": blocked(tcp_connect),
    "unix_socket_blocked": blocked(unix_connect),
    "outside_write_blocked": blocked(
        lambda: outside_write.write_text("not allowed", encoding="utf-8")
    ),
    "arena_file_present": Path("arena.py").exists(),
    "arena_config_present": Path(".arena.json").exists(),
}
print("SANDBOX_PROBE", json.dumps(result, sort_keys=True))
'''


def _safe_create(path: Path, text: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(text)


def run_probe(label: str | None = None) -> dict[str, object]:
    probe_id = label or f"probe-{uuid.uuid4().hex[:12]}"
    if not probe_id or any(
        character
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for character in probe_id
    ):
        raise ValueError("probe label contains unsupported characters")

    root = (
        PROJECT_ROOT
        / "artifacts"
        / "codex-sandbox-probes"
        / probe_id
    )
    root.mkdir(parents=True, exist_ok=False)
    workspace = root / "workspace"
    campaign_tmp = workspace / ".tmp" / "codex"
    outside_write = PROJECT_ROOT / "src" / "roboarm_game" / (
        f"sandbox-denial-{uuid.uuid4().hex}.txt"
    )
    private_source = (
        PROJECT_ROOT / "src" / "roboarm_game" / "dynamics.py"
    )
    unix_path = root / "forbidden.sock"

    tcp_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_server.bind(("127.0.0.1", 0))
    tcp_server.listen(1)
    tcp_port = int(tcp_server.getsockname()[1])
    unix_server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    unix_server.bind(str(unix_path))
    unix_server.listen(1)
    connector = RoboArmConnector()
    public_evidence = {
        "schema_version": 2,
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
            "unix_socket_visible_to_proposer": False,
        },
        "receipt_sha256": "sandbox-probe",
    }
    try:
        materialize_workspace(
            workspace,
            write_root=root,
            public_evidence=public_evidence,
            generation=1,
        )
        client = workspace / "sandbox_probe_client.py"
        _safe_create(client, CLIENT_SOURCE)
        environment = _codex_environment(campaign_tmp)
        configs = _codex_permission_configs(
            workspace,
            campaign_tmp,
        )
        command = [
            "codex",
            "sandbox",
            *[
                item
                for setting in configs
                for item in ("--config", setting)
            ],
            "--",
            "/opt/homebrew/bin/python3",
            str(client),
            str(private_source),
            str(tcp_port),
            str(unix_path),
            str(outside_write),
        ]
        process = subprocess.run(
            command,
            cwd=workspace,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
    finally:
        tcp_server.close()
        unix_server.close()
        if unix_path.exists() and not unix_path.is_dir():
            unix_path.unlink()

    result_line = next(
        (
            line[len("SANDBOX_PROBE ") :]
            for line in process.stdout.splitlines()
            if line.startswith("SANDBOX_PROBE ")
        ),
        None,
    )
    client_result = (
        json.loads(result_line)
        if result_line is not None
        else None
    )
    report: dict[str, object] = {
        "schema_version": 2,
        "probe_id": probe_id,
        "codex_returncode": process.returncode,
        "client_result": client_result,
        "connector_visible_to_proposer": False,
        "connector_committed_actions": connector.committed_actions,
        "connector_preflight_actions": connector.preflight_actions,
        "outside_write_created": outside_write.exists(),
        "stdout": process.stdout,
        "stderr": process.stderr,
    }
    passed = bool(
        process.returncode == 0
        and isinstance(client_result, dict)
        and client_result.get("public_frame_bytes") == 72 * 128 * 3
        and client_result.get("sensor_contract_id")
        == "rb01-roarm-c920-v3"
        and client_result.get("public_telemetry_present") is True
        and client_result.get("private_import_blocked") is True
        and client_result.get("private_read_blocked") is True
        and client_result.get("tcp_blocked") is True
        and client_result.get("unix_socket_blocked") is True
        and client_result.get("outside_write_blocked") is True
        and client_result.get("arena_file_present") is False
        and client_result.get("arena_config_present") is False
        and connector.committed_actions == 0
        and connector.preflight_actions == 0
        and not outside_write.exists()
    )
    report["passed"] = passed
    write_json(root / "report.json", report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prove Codex receives sealed frames but no connector or socket, "
            "while private reads, network, and outside writes fail"
        )
    )
    parser.add_argument("--label")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report = run_probe(arguments.label)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
