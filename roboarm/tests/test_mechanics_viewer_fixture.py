from __future__ import annotations

import json
from pathlib import Path
import uuid

from roboarm_game.gkm.accounting import canonical_json_sha256

from tools.export_mechanics_fixture import export_mechanics_fixture


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_mechanics_fixture_is_authoritative_and_explicitly_not_gkm() -> None:
    destination = (
        PROJECT_ROOT
        / "artifacts"
        / "mechanics-fixture-tests"
        / uuid.uuid4().hex
    )
    manifest = export_mechanics_fixture(destination)
    assert manifest["export_kind"] == "developer-mechanics-test"
    assert "not proposer activity" in str(manifest["notice"]).lower()

    collision = json.loads(
        (destination / "collision_attempt.json").read_text(encoding="utf-8")
    )
    completed = json.loads(
        (destination / "successful_replay.json").read_text(encoding="utf-8")
    )
    assert collision["attempt_kind"] == "mechanics-test"
    assert collision["disposition"] == "expected-rejection"
    assert collision["steps"][-1]["visual_state"]["robot"]["rejected"] is True
    assert collision["steps"][-1]["visual_state"]["robot"]["rejectionReason"] == (
        "gripper_barrier_collision"
    )
    assert completed["attempt_kind"] == "mechanics-test"
    assert completed["disposition"] == "completed"
    assert completed["steps"][-1]["levels_completed"] == 1
    assert completed["steps"][-1]["terminal"] is True

    for attempt in (collision, completed):
        receipt = attempt.pop("replay_receipt_sha256")
        assert receipt == canonical_json_sha256(attempt)

    public_fixture = PROJECT_ROOT / "web" / "public" / "mechanics-test"
    for filename in (
        "manifest.json",
        "collision_attempt.json",
        "successful_replay.json",
    ):
        generated = json.loads(
            (destination / filename).read_text(encoding="utf-8")
        )
        checked_in = json.loads(
            (public_fixture / filename).read_text(encoding="utf-8")
        )
        assert checked_in == generated
