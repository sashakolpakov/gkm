"""Fresh proposal-source execution and exact public-action replay gates."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ..environment import make_env
from ..observation import (
    FRAME_ENCODING,
    FRAME_SHAPE,
    OBSERVATION_SCHEMA_VERSION,
    SENSOR_CONTRACT_ID,
    camera_model,
    frame_record,
    validated_operational_telemetry,
)
from .accounting import canonical_json_sha256


@dataclass(frozen=True, slots=True)
class ProposalRun:
    returncode: int
    stdout: str
    stderr: str
    result: dict[str, object] | None


def run_proposal_source(
    workspace: Path,
    *,
    timeout_seconds: int = 120,
) -> ProposalRun:
    """Execute proposal source offline with no connector or actuation channel."""

    tmp = workspace / ".tmp"
    cache = workspace / ".cache"
    tmp.mkdir(exist_ok=True)
    cache.mkdir(exist_ok=True)
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "TMPDIR": str(tmp),
        "XDG_CACHE_HOME": str(cache),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }
    process = subprocess.run(
        [sys.executable, "gkm_propose.py"],
        cwd=workspace,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_seconds,
        check=False,
    )
    result: dict[str, object] | None = None
    for line in process.stdout.splitlines():
        if not line.startswith("SCENARIO_PROPOSALS "):
            continue
        try:
            value = json.loads(line[len("SCENARIO_PROPOSALS ") :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            result = value
    return ProposalRun(
        returncode=process.returncode,
        stdout=process.stdout,
        stderr=process.stderr,
        result=result,
    )


def run_solver(workspace: Path, *, timeout_seconds: int = 120) -> ProposalRun:
    """Compatibility alias for the now proposal-only source harness."""

    return run_proposal_source(
        workspace,
        timeout_seconds=timeout_seconds,
    )


def load_candidate_path(path: Path) -> list[int]:
    if not path.is_file() or path.is_symlink():
        raise ValueError("candidate path artifact is missing")
    data = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(data, dict)
        or set(data) != {"schema_version", "target_level", "actions"}
        or data["schema_version"] != 1
        or data["target_level"] != 1
        or not isinstance(data["actions"], list)
    ):
        raise ValueError("candidate path artifact has an invalid schema")
    actions: list[int] = []
    for value in data["actions"]:
        if isinstance(value, bool) or not isinstance(value, int) or value not in range(1, 7):
            raise ValueError("candidate path contains an invalid action")
        actions.append(int(value))
    if not actions or len(actions) > 160:
        raise ValueError("candidate path length is outside round limits")
    return actions


def exact_path_replay(
    actions: Sequence[int],
    *,
    seed: int = 0,
) -> dict[str, object]:
    """Replay from zero and return the first exact level-acquisition boundary."""

    env = make_env("rb01-v1", seed=seed, scenario="round-1")
    initial = env.reset()
    initial_sha, initial_b64 = frame_record(initial)
    initial_telemetry = validated_operational_telemetry(env.telemetry())
    initial_telemetry_sha = canonical_json_sha256(initial_telemetry)
    snapshot_method = getattr(env, "snapshot", None)
    initial_visual_state = (
        snapshot_method() if snapshot_method is not None else None
    )
    steps: list[dict[str, object]] = []
    boundary: list[int] | None = None
    for action in actions:
        before = env.frame()
        before_sha, _ = frame_record(before)
        before_telemetry = validated_operational_telemetry(env.telemetry())
        before_telemetry_sha = canonical_json_sha256(before_telemetry)
        after = env.step(action)
        after_sha, after_b64 = frame_record(after)
        telemetry = validated_operational_telemetry(env.telemetry())
        telemetry_sha = canonical_json_sha256(telemetry)
        visual_state = snapshot_method() if snapshot_method is not None else None
        steps.append(
            {
                "turn": len(steps) + 1,
                "role": "replay",
                "action": int(action),
                "before_frame_sha256": before_sha,
                "before_telemetry_sha256": before_telemetry_sha,
                "frame_sha256": after_sha,
                "frame_b64": after_b64,
                "telemetry_sha256": telemetry_sha,
                "telemetry": telemetry,
                "levels_completed": int(env.levels_completed),
                "terminal": bool(env.terminal()),
                "visual_state": visual_state,
            }
        )
        if env.levels_completed >= 1:
            boundary = [int(value) for value in actions[: len(steps)]]
            break
        if env.terminal():
            break
    replay = {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "game_id": "rb01-v1",
        "scenario": "round-1",
        "seed": seed,
        "sensor_contract_id": SENSOR_CONTRACT_ID,
        "frame_encoding": FRAME_ENCODING,
        "frame_shape": list(FRAME_SHAPE),
        "camera_model": camera_model(),
        "initial_frame_sha256": initial_sha,
        "initial_frame_b64": initial_b64,
        "initial_telemetry_sha256": initial_telemetry_sha,
        "initial_telemetry": initial_telemetry,
        "initial_visual_state": initial_visual_state,
        "actions_supplied": len(actions),
        "exact_actions": boundary,
        "levels_completed": int(env.levels_completed),
        "terminal": bool(env.terminal()),
        "steps": steps,
    }
    replay["receipt_sha256"] = canonical_json_sha256(replay)
    return replay


def write_json(path: Path, value: object) -> None:
    if path.is_symlink():
        raise ValueError(f"refusing symlinked evidence path: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(payload)


__all__ = [
    "ProposalRun",
    "exact_path_replay",
    "load_candidate_path",
    "run_proposal_source",
    "run_solver",
    "write_json",
]
