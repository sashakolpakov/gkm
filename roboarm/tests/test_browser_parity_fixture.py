from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from roboarm_game import make_env
from roboarm_game.canonical import CANONICAL_PICK_PLACE_ACTIONS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARITY_FIXTURE = PROJECT_ROOT / "references" / "operational_parity.json"


def _assert_close(actual: object, expected: object) -> None:
    assert np.allclose(actual, expected, atol=2e-11, rtol=0.0)


def test_browser_parity_fixture_is_fresh_from_python_authoritative_world() -> None:
    fixture = json.loads(PARITY_FIXTURE.read_text(encoding="utf-8"))
    assert fixture["schemaVersion"] == 2
    assert fixture["sensorContractId"] == "rb01-roarm-c920-v3"
    assert fixture["frameShape"] == [72, 128, 3]
    assert fixture["sceneId"] == "pick-place-v2"
    assert fixture["actionCount"] == len(CANONICAL_PICK_PLACE_ACTIONS) == 63
    expected_by_turn = {
        keyframe["turn"]: keyframe for keyframe in fixture["keyframes"]
    }

    env = make_env("rb01-v1", seed=0, scenario="pick-place")
    for turn in range(len(CANONICAL_PICK_PLACE_ACTIONS) + 1):
        expected = expected_by_turn.get(turn)
        if expected is not None:
            snapshot = env.snapshot()
            _assert_close(snapshot["robot"]["joints"], expected["joints"])
            _assert_close(snapshot["robot"]["anchors"]["tcp"], expected["tcp"])
            _assert_close(snapshot["object"]["position"], expected["object"])
            assert snapshot["robot"]["gripperAperture"] == expected["aperture"]
            assert snapshot["object"]["attached"] is expected["attached"]
            assert snapshot["success"] is expected["success"]
            assert snapshot["events"] == expected["events"]
            assert hashlib.sha256(env.frame().tobytes()).hexdigest() == (
                expected["frameSha256"]
            )
        if turn < len(CANONICAL_PICK_PLACE_ACTIONS):
            env.step(CANONICAL_PICK_PLACE_ACTIONS[turn])
