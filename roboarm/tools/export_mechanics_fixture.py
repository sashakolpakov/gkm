"""Export clearly segregated Python-authoritative browser mechanics fixtures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Final

from roboarm_game import make_env
from roboarm_game.canonical import (
    CANONICAL_PICK_PLACE_ACTIONS,
    LOW_CLEARANCE_COLLISION_ACTIONS,
)
from roboarm_game.gkm.accounting import canonical_json_sha256
from roboarm_game.gkm.replay import write_json
from roboarm_game.observation import (
    FRAME_ENCODING,
    FRAME_SHAPE,
    OBSERVATION_SCHEMA_VERSION,
    SENSOR_CONTRACT_ID,
    camera_model,
    frame_record,
    validated_operational_telemetry,
)


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DEFAULT_DESTINATION: Final[Path] = (
    PROJECT_ROOT / "web" / "public" / "mechanics-test"
)


def _attempt(
    *,
    fixture_id: str,
    disposition: str,
    actions: tuple[int, ...],
) -> dict[str, object]:
    env = make_env("rb01-v1", seed=0, scenario="pick-place")
    initial = env.reset()
    initial_sha, initial_b64 = frame_record(initial)
    initial_telemetry = validated_operational_telemetry(env.telemetry())
    initial_telemetry_sha = canonical_json_sha256(initial_telemetry)
    initial_visual_state = env.snapshot()
    steps: list[dict[str, object]] = []
    for turn, action in enumerate(actions, 1):
        before_sha, _ = frame_record(env.frame())
        before_telemetry = validated_operational_telemetry(env.telemetry())
        before_telemetry_sha = canonical_json_sha256(before_telemetry)
        frame = env.step(action)
        frame_sha, frame_b64 = frame_record(frame)
        telemetry = validated_operational_telemetry(env.telemetry())
        telemetry_sha = canonical_json_sha256(telemetry)
        steps.append(
            {
                "turn": turn,
                "role": "developer_mechanics_test",
                "action": action,
                "before_frame_sha256": before_sha,
                "before_telemetry_sha256": before_telemetry_sha,
                "frame_sha256": frame_sha,
                "frame_b64": frame_b64,
                "telemetry_sha256": telemetry_sha,
                "telemetry": telemetry,
                "levels_completed": int(env.levels_completed),
                "terminal": bool(env.terminal()),
                "visual_state": env.snapshot(),
            }
        )

    snapshot = env.snapshot()
    if disposition == "expected-rejection":
        if (
            snapshot["robot"]["rejected"] is not True
            or snapshot["robot"]["rejectionReason"]
            != "gripper_barrier_collision"
            or env.levels_completed != 0
        ):
            raise AssertionError("collision fixture did not end at its rejection")
    elif disposition == "completed":
        if env.levels_completed != 1 or not env.terminal():
            raise AssertionError("completion fixture did not finish")
    else:
        raise ValueError("unsupported mechanics-test disposition")

    attempt: dict[str, object] = {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "attempt_kind": "mechanics-test",
        "disposition": disposition,
        "trace_role": "developer_mechanics_test",
        "campaign_id": "canonical-mechanics-fixture",
        "fixture_id": fixture_id,
        "game_id": "rb01-v1",
        "scenario": "pick-place",
        "seed": 0,
        "sensor_contract_id": SENSOR_CONTRACT_ID,
        "frame_encoding": FRAME_ENCODING,
        "frame_shape": list(FRAME_SHAPE),
        "camera_model": camera_model(),
        "initial_frame_sha256": initial_sha,
        "initial_frame_b64": initial_b64,
        "initial_telemetry_sha256": initial_telemetry_sha,
        "initial_telemetry": initial_telemetry,
        "initial_visual_state": initial_visual_state,
        "actions": list(actions),
        "steps": steps,
    }
    attempt["replay_receipt_sha256"] = canonical_json_sha256(attempt)
    return attempt


def export_mechanics_fixture(
    destination: Path = DEFAULT_DESTINATION,
) -> dict[str, object]:
    target = destination.resolve(strict=False)
    if not target.is_relative_to(PROJECT_ROOT.resolve(strict=True)):
        raise ValueError("mechanics fixture destination must stay below roboarm")
    target.mkdir(parents=True, exist_ok=True)

    attempts = {
        "collision_attempt.json": _attempt(
            fixture_id="low-clearance-full-gripper-barrier-collision",
            disposition="expected-rejection",
            actions=LOW_CLEARANCE_COLLISION_ACTIONS,
        ),
        "successful_replay.json": _attempt(
            fixture_id="canonical-pick-carry-place",
            disposition="completed",
            actions=CANONICAL_PICK_PLACE_ACTIONS,
        ),
    }
    for filename, attempt in attempts.items():
        write_json(target / filename, attempt)

    manifest: dict[str, object] = {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "export_kind": "developer-mechanics-test",
        "campaign_id": "canonical-mechanics-fixture",
        "attempts": list(attempts),
        "sensor_contract_id": SENSOR_CONTRACT_ID,
        "frame_encoding": FRAME_ENCODING,
        "frame_shape": list(FRAME_SHAPE),
        "camera_model": camera_model(),
        "notice": (
            "Developer mechanics regression only; not proposer activity, "
            "Godel-Kolmogorov machine learning, discovery, or promotion "
            "evidence."
        ),
        "attempt_receipts": {
            filename: attempt["replay_receipt_sha256"]
            for filename, attempt in attempts.items()
        },
    }
    manifest["export_receipt_sha256"] = canonical_json_sha256(manifest)
    write_json(target / "manifest.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export Python-authoritative developer mechanics replays"
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_DESTINATION,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    manifest = export_mechanics_fixture(arguments.destination)
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
