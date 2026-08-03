import hashlib
import json
import sys
import pytest
from types import SimpleNamespace

import replay_scorecard as R
from replay_scorecard import decode_action


def _write_content_addressed_receipt(tmp_path, body):
    body = {
        **body,
        "release_identity": {"source_revision": "a" * 40},
        "control_contract": {
            "files_sha256": {
                "arc/crack_lab/arc_agi3_release_gate.py": "b" * 64
            }
        },
        "verifier": {
            "files_sha256": {
                "arc/crack_lab/arc_agi3_release_gate.py": "b" * 64
            }
        },
        "inventory_metadata_sha256": {"wa30/meta.json": "c" * 64},
    }
    raw = (
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode()
    digest = hashlib.sha256(raw).hexdigest()
    path = tmp_path / f"{digest}.json"
    path.write_bytes(raw)
    return path


def test_decode_action_supports_keys_and_coordinate_tokens():
    assert decode_action(4) == (4, None)
    assert decode_action([6, 38, 17]) == (6, {"x": 38, "y": 17})
    assert decode_action((6, 0, 63)) == (6, {"x": 0, "y": 63})


@pytest.mark.parametrize(
    "action",
    (
        [5, 1, 2],
        [6, 1],
        [6, -1, 0],
        [6, 64, 0],
        [6, 0, 64],
        [6, True, 2],
        True,
        1.0,
        "1",
        0,
        6,
        8,
    ),
)
def test_decode_action_rejects_invalid_tokens(action):
    with pytest.raises(ValueError):
        decode_action(action)


def test_replay_rebuilds_from_server_reported_level_after_reset(monkeypatch):
    class RollbackEnv:
        def __init__(self):
            self.levels = 0
            self.action2_calls = 0

        def frame(self):
            state = "WIN" if self.levels == 2 else "NOT_FINISHED"
            return SimpleNamespace(
                levels_completed=self.levels,
                state=SimpleNamespace(name=state),
            )

        def reset(self):
            self.levels = 0
            return self.frame()

        def step(self, action, data=None):
            if action == 1:
                self.levels = 1
                return self.frame()
            self.action2_calls += 1
            if self.action2_calls == 1:
                return None
            self.levels = 2
            return self.frame()

    monkeypatch.setattr(R.time, "sleep", lambda _: None)
    env = RollbackEnv()
    reached = R.replay(
        env,
        [[1], [2]],
        {"ACTION1": 1, "ACTION2": 2},
        "rollback",
        verbose=False,
    )
    assert reached == 2
    assert env.action2_calls == 2


def test_parse_games_all_is_sorted_and_rejects_duplicates(tmp_path):
    for game in ("wa30", "ls20"):
        root = tmp_path / f"{game}_legs"
        root.mkdir()
        (root / "checkpoint.json").write_text("{}")
    assert R.parse_games("all", tmp_path) == ["ls20", "wa30"]
    with pytest.raises(ValueError, match="duplicate"):
        R.parse_games("wa30,wa30", tmp_path)


def test_release_binding_requires_exact_checkpoint_bytes(tmp_path):
    artifact_root = tmp_path / "release"
    checkpoint_hashes = {}
    checkpoints = {}
    evidence = {}
    inventory = {"ls20": 1, "wa30": 1}
    for game in inventory:
        game_root = artifact_root / f"{game}_legs"
        game_root.mkdir(parents=True)
        value = {
            "game": game,
            "reached": 1,
            "total_marginal_C": 1,
            "records": [{"level": 1, "marginal_C": 1}],
            "final_path": [1],
            "validated": True,
        }
        path = game_root / "checkpoint.json"
        path.write_text(json.dumps(value))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        checkpoint_hashes[game] = digest
        checkpoints[game] = value
        evidence[game] = [{"checkpoint_sha256": digest}]
    receipt = _write_content_addressed_receipt(
        tmp_path,
        {
            "inventory": inventory,
            "claimed_inventory": inventory,
            "claimed_level_count": 2,
            "authoritative_level_count": 2,
            "canonical_tree_sha256": "a" * 64,
            "evidence": evidence,
        },
    )

    binding = R.release_binding(
        receipt,
        ["ls20", "wa30"],
        checkpoints,
        checkpoint_hashes,
    )
    assert binding["claimed_level_count"] == 2
    assert binding["canonical_tree_sha256"] == "a" * 64
    assert binding["receipt_sha256"] == receipt.stem

    stale_hashes = dict(checkpoint_hashes)
    stale_hashes["wa30"] = "0" * 64
    with pytest.raises(ValueError, match="bytes differ"):
        R.release_binding(
            receipt,
            ["ls20", "wa30"],
            checkpoints,
            stale_hashes,
        )

    wrong_name = tmp_path / f"{'0' * 64}.json"
    wrong_name.write_bytes(receipt.read_bytes())
    with pytest.raises(R.FrozenReleaseError, match="content hash"):
        R.release_binding(
            wrong_name,
            ["ls20", "wa30"],
            checkpoints,
            checkpoint_hashes,
        )


def test_load_checkpoint_hashes_the_same_bytes_it_parses(tmp_path):
    game_root = tmp_path / "wa30_legs"
    game_root.mkdir()
    raw = b'{"game":"wa30","reached":1,"final_path":[1]}\n'
    (game_root / "checkpoint.json").write_bytes(raw)
    value, digest = R.load_checkpoint("wa30", tmp_path)
    assert value["final_path"] == [1]
    assert digest == hashlib.sha256(raw).hexdigest()


def test_main_rejects_receipt_swap_between_gate_and_binding(
    monkeypatch, tmp_path
):
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}\n")
    checkpoint = {"game": "wa30", "reached": 1, "final_path": [1]}
    monkeypatch.setattr(
        R,
        "verify_frozen_release",
        lambda **kwargs: {"status": "PASS", "receipt_sha256": "a" * 64},
    )
    monkeypatch.setattr(
        R,
        "release_binding",
        lambda *args: {
            "receipt_sha256": "b" * 64,
            "claimed_level_count": 1,
            "authoritative_level_count": 183,
        },
    )
    monkeypatch.setattr(R, "load_checkpoint", lambda *args: (checkpoint, "c" * 64))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "replay_scorecard.py",
            "--mode",
            "online",
            "--games",
            "wa30",
            "--artifact-root",
            str(artifact_root),
            "--release-receipt",
            str(receipt),
            "--preflight-only",
        ],
    )
    assert R.main() == 2


def test_write_new_json_never_overwrites(tmp_path):
    target = tmp_path / "receipt.json"
    R.write_new_json(target, {"status": "PASS"})
    assert json.loads(target.read_text()) == {"status": "PASS"}
    with pytest.raises(FileExistsError):
        R.write_new_json(target, {"status": "FAIL"})


def test_public_docs_do_not_publish_mode_only_replay_commands():
    repo = R.GKM
    paths = (
        repo / "README.md",
        repo / "REPRODUCE_ARC.md",
        repo / "arc/manuscript/README.md",
        repo / "arc/manuscript/arc_agi3.tex",
        repo / "arc/crack_lab/replay_scorecard.py",
    )
    forbidden = (
        "python arc/crack_lab/replay_scorecard.py --mode online\n",
        "python arc/crack_lab/replay_scorecard.py --mode competition\n",
        "python3 arc/crack_lab/replay_scorecard.py --mode online\n",
        "python3 arc/crack_lab/replay_scorecard.py --mode competition\n",
    )
    for path in paths:
        text = path.read_text()
        assert not any(command in text for command in forbidden), path
