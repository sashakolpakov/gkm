"""Regenerate the authoritative Python/browser operational parity fixture."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from roboarm_game import make_env
from roboarm_game.canonical import CANONICAL_PICK_PLACE_ACTIONS
from roboarm_game.observation import FRAME_SHAPE, SENSOR_CONTRACT_ID


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DESTINATION = PROJECT_ROOT / "references" / "operational_parity.json"
KEY_TURNS = frozenset((0, 18, 32, 46, 62, 63))


def keyframe(env: object) -> dict[str, object]:
    snapshot = env.snapshot()
    return {
        "turn": snapshot["turn"],
        "joints": snapshot["robot"]["joints"],
        "tcp": snapshot["robot"]["anchors"]["tcp"],
        "object": snapshot["object"]["position"],
        "aperture": snapshot["robot"]["gripperAperture"],
        "attached": snapshot["object"]["attached"],
        "success": snapshot["success"],
        "events": snapshot["events"],
        "frameSha256": hashlib.sha256(env.frame().tobytes()).hexdigest(),
    }


def main() -> None:
    env = make_env("rb01-v1", seed=0, scenario="pick-place")
    frames: list[dict[str, object]] = []
    for turn in range(len(CANONICAL_PICK_PLACE_ACTIONS) + 1):
        if turn in KEY_TURNS:
            frames.append(keyframe(env))
        if turn < len(CANONICAL_PICK_PLACE_ACTIONS):
            env.step(CANONICAL_PICK_PLACE_ACTIONS[turn])

    snapshot = env.snapshot()
    fixture = {
        "schemaVersion": 2,
        "sensorContractId": SENSOR_CONTRACT_ID,
        "frameShape": list(FRAME_SHAPE),
        "sceneId": snapshot["sceneId"],
        "actionCount": len(CANONICAL_PICK_PLACE_ACTIONS),
        "producer": "roboarm_game OperationalWorld, seed 0",
        "purpose": "Python/browser turn-boundary parity fixture",
        "keyframes": frames,
    }
    DESTINATION.write_text(
        json.dumps(fixture, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
