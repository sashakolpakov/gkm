import hashlib
import json
import sys
import pytest
from types import SimpleNamespace

import replay_scorecard as R
from replay_scorecard import decode_action

REAL_VERIFY_PUBLIC_REVISION = R.verify_public_revision
REAL_VERIFY_RUNTIME_REVISION = R.verify_runtime_revision


@pytest.fixture(autouse=True)
def _public_revision_is_reachable(monkeypatch):
    monkeypatch.setattr(
        R,
        "verify_public_revision",
        lambda revision: {
            "sha": revision,
            "html_url": f"https://github.com/sashakolpakov/gkm/commit/{revision}",
        },
    )
    monkeypatch.setattr(
        R,
        "verify_runtime_revision",
        lambda revision, games: {
            "source_revision": revision,
            "files_sha256": {},
            "manifest_sha256": "f" * 64,
        },
    )


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


def test_replay_reuses_remote_wrapper_initial_frame_without_extra_reset():
    class Env:
        def __init__(self):
            self.observation_space = SimpleNamespace(
                levels_completed=0,
                state=SimpleNamespace(name="NOT_FINISHED"),
            )
            self.reset_calls = 0

        def reset(self):
            self.reset_calls += 1
            raise AssertionError("initial wrapper RESET must be reused")

        def step(self, action, data=None):
            return SimpleNamespace(
                levels_completed=1,
                state=SimpleNamespace(name="WIN"),
            )

    env = Env()
    assert R.replay(
        env,
        [[1]],
        {"ACTION1": 1},
        "wa30",
        verbose=False,
    ) == 1
    assert env.reset_calls == 0


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
        evidence[game] = [
            {"checkpoint_sha256": digest, "action_count": 1}
        ]
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
    assert binding["release_identity_source_revision"] == "a" * 40

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


def test_main_rejects_complete_source_revision_mismatch(monkeypatch, tmp_path):
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}\n")
    checkpoint = {"game": "wa30", "reached": 183, "final_path": [1]}
    monkeypatch.setattr(
        R,
        "verify_frozen_release",
        lambda **kwargs: {"status": "PASS", "receipt_sha256": "c" * 64},
    )
    monkeypatch.setattr(
        R,
        "release_binding",
        lambda *args: {
            "receipt_sha256": "c" * 64,
            "claimed_level_count": 183,
            "authoritative_level_count": 183,
            "release_identity_source_revision": "a" * 40,
        },
    )
    monkeypatch.setattr(
        R,
        "load_checkpoint",
        lambda *args: (checkpoint, "d" * 64),
    )
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
            "--source-revision",
            "b" * 40,
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


@pytest.mark.parametrize(
    "raw",
    (
        b'{"game":"wa30","game":"ls20"}\n',
        b'{"game":"wa30","value":NaN}\n',
        b'{"game":"wa30","value":Infinity}\n',
    ),
)
def test_checkpoint_loader_rejects_duplicate_and_nonfinite_json(tmp_path, raw):
    path = tmp_path / "wa30_legs" / "checkpoint.json"
    path.parent.mkdir()
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="duplicate|non-finite"):
        R.load_checkpoint("wa30", tmp_path)


def test_checkpoint_secure_read_handles_short_reads(monkeypatch, tmp_path):
    path = tmp_path / "wa30_legs" / "checkpoint.json"
    path.parent.mkdir()
    path.write_text('{"game":"wa30","reached":1}\n')
    real_read = R.os.read
    calls = 0

    def short_read(descriptor, size):
        nonlocal calls
        calls += 1
        return real_read(descriptor, min(size, 3))

    monkeypatch.setattr(R.os, "read", short_read)
    value, digest = R.load_checkpoint("wa30", tmp_path)
    assert value["reached"] == 1
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert calls > 1


def test_checkpoint_inode_swap_between_stat_and_open_fails(monkeypatch, tmp_path):
    path = tmp_path / "wa30_legs" / "checkpoint.json"
    path.parent.mkdir()
    path.write_text('{"game":"wa30","reached":1}\n')
    original = tmp_path / "original.json"
    real_open = R.os.open
    swapped = False

    def swapping_open(target, flags, *args):
        nonlocal swapped
        if R.Path(target) == path and not swapped:
            swapped = True
            path.rename(original)
            path.write_text('{"game":"wa30","reached":2}\n')
        return real_open(target, flags, *args)

    monkeypatch.setattr(R.os, "open", swapping_open)
    with pytest.raises(ValueError, match="changed during secure open"):
        R.load_checkpoint("wa30", tmp_path)


def test_checkpoint_in_place_mutation_during_read_fails(monkeypatch, tmp_path):
    path = tmp_path / "wa30_legs" / "checkpoint.json"
    path.parent.mkdir()
    original = b'{"game":"wa30","reached":1}\n'
    replacement = b'{"game":"wa30","reached":2}\n'
    path.write_bytes(original)
    real_read = R.os.read
    mutated = False

    def mutating_read(descriptor, size):
        nonlocal mutated
        chunk = real_read(descriptor, size)
        if not mutated:
            mutated = True
            with path.open("r+b") as handle:
                handle.write(replacement)
                handle.flush()
                R.os.fsync(handle.fileno())
        return chunk

    monkeypatch.setattr(R.os, "read", mutating_read)
    with pytest.raises(ValueError, match="changed during secure read"):
        R.load_checkpoint("wa30", tmp_path)


def test_immutable_source_url_requires_exact_public_revision():
    revision = "a" * 40
    assert R.immutable_source_url(
        f"https://github.com/sashakolpakov/gkm/tree/{revision}",
        revision,
    )
    assert not R.immutable_source_url(
        f"https://github.com/sashakolpakov/gkm/tree/{revision}/arc",
        revision,
    )
    assert not R.immutable_source_url(
        "https://github.com/sashakolpakov/gkm",
        revision,
    )
    assert not R.immutable_source_url(
        f"https://github.com/sashakolpakov/gkm/tree/{'b' * 40}",
        revision,
    )


@pytest.mark.parametrize("outcome", ("reachable", "missing", "mismatch"))
def test_public_revision_reachability_is_exact(monkeypatch, outcome):
    revision = "a" * 40

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self, limit):
            value = {
                "sha": revision if outcome != "mismatch" else "b" * 40,
                "html_url": (
                    "https://github.com/sashakolpakov/gkm/commit/"
                    + (revision if outcome != "mismatch" else "b" * 40)
                ),
            }
            return json.dumps(value).encode()

    def open_commit(request, timeout):
        if outcome == "missing":
            raise R.urllib.error.URLError("404")
        return Response()

    monkeypatch.setattr(R.urllib.request, "urlopen", open_commit)
    if outcome == "reachable":
        assert REAL_VERIFY_PUBLIC_REVISION(revision)["sha"] == revision
    else:
        with pytest.raises(ValueError, match="reachable|different"):
            REAL_VERIFY_PUBLIC_REVISION(revision)


def _provider_environment(game, target):
    environment_id = f"{game}-deadbeef"
    run = {
        "id": environment_id,
        "guid": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        "score": 100.0,
        "levels_completed": target,
        "actions": target,
        "resets": 1,
        "state": "WIN",
        "completed": True,
        "level_scores": [100.0] * target,
        "level_actions": [1] * target,
        "level_baseline_actions": [1] * target,
        "number_of_levels": target,
        "number_of_environments": 1,
        "message": None,
    }
    return {
        "id": environment_id,
        "runs": [run],
        "score": 100.0,
        "actions": target,
        "levels_completed": target,
        "completed": True,
        "level_count": target,
        "resets": 1,
    }


def _provider_tag_scores():
    values = []
    for index, score_id in enumerate(
        ("click", "keyboard", "keyboard_click"), start=1
    ):
        values.append(
            {
                "id": score_id,
                "guid": f"{index}" * 8
                + "-1111-4111-8111-"
                + f"{index}" * 12,
                "score": 100.0,
                "levels_completed": 1,
                "actions": 1,
                "resets": 0,
                "state": "NOT_FINISHED",
                "completed": False,
                "level_scores": None,
                "level_actions": None,
                "level_baseline_actions": None,
                "number_of_levels": 1,
                "number_of_environments": 1,
                "message": None,
            }
        )
    return values


def _provider_aggregate(*, card_id, source_url, mode, plan, tags, opaque):
    environments = [
        _provider_environment(game, value["reached"])
        for game, value in plan.items()
    ]
    total_levels = sum(value["reached"] for value in plan.values())
    return {
        "source_url": source_url,
        "tags": tags,
        "opaque": opaque,
        "card_id": card_id,
        "score": 100.0,
        "environments": environments,
        "tags_scores": _provider_tag_scores(),
        "competition_mode": mode == "competition",
        "total_environments_completed": len(plan),
        "total_environments": len(plan),
        "total_levels_completed": total_levels,
        "total_levels": total_levels,
        "total_actions": total_levels,
    }


def test_closed_scorecard_must_match_frozen_plan_exactly():
    card_id = "12345678-1234-1234-1234-123456789abc"
    source_url = (
        "https://github.com/sashakolpakov/gkm/tree/" + "a" * 40
    )
    plan = {
        "ls20": {"reached": 1},
        "wa30": {"reached": 2},
    }
    tags = ["gkm-v3"]
    opaque = {"schema": 1}
    aggregate = _provider_aggregate(
        card_id=card_id,
        source_url=source_url,
        mode="competition",
        plan=plan,
        tags=tags,
        opaque=opaque,
    )
    assert R.validate_closed_scorecard(
        aggregate,
        mode="competition",
        card_id=card_id,
        source_url=source_url,
        games=["ls20", "wa30"],
        plan=plan,
        scorecard_tags=tags,
        scorecard_opaque=opaque,
    ) == aggregate

    stale = dict(aggregate)
    stale["environments"] = list(aggregate["environments"])
    stale["environments"][1] = _provider_environment("wa30", 3)
    with pytest.raises(
        ValueError, match="nested run accounting is invalid|endpoint differs"
    ):
        R.validate_closed_scorecard(
            stale,
            mode="competition",
            card_id=card_id,
            source_url=source_url,
            games=["ls20", "wa30"],
            plan=plan,
            scorecard_tags=tags,
            scorecard_opaque=opaque,
        )


def test_closed_scorecard_rejects_absent_or_wrong_mode_aggregate():
    arguments = {
        "mode": "competition",
        "card_id": "12345678-1234-1234-1234-123456789abc",
        "source_url": "https://example.test/source",
        "games": ["wa30"],
        "plan": {"wa30": {"reached": 1}},
        "scorecard_tags": ["tag"],
        "scorecard_opaque": None,
    }
    with pytest.raises(ValueError, match="aggregate is absent"):
        R.validate_closed_scorecard(None, **arguments)
    with pytest.raises(ValueError, match="provenance differs|operation mode"):
        wrong_mode = _provider_aggregate(
            card_id=arguments["card_id"],
            source_url=arguments["source_url"],
            mode="online",
            plan=arguments["plan"],
            tags=arguments["scorecard_tags"],
            opaque=arguments["scorecard_opaque"],
        )
        R.validate_closed_scorecard(wrong_mode, **arguments)


def test_schema2_command_identity_does_not_publish_host_paths(tmp_path):
    receipt = tmp_path / "receipt.json"
    args = SimpleNamespace(
        mode="competition",
        release_receipt=receipt,
        expected_claimed_levels=183,
        preflight_only=False,
        source_url="https://github.com/sashakolpakov/gkm/tree/" + "a" * 40,
        source_revision="a" * 40,
        tags="gkm,final",
    )
    identity = R.command_identity(
        args=args,
        games=["wa30"],
        artifact_root=tmp_path / "artifacts",
    )
    encoded = json.dumps(identity)
    assert str(tmp_path) not in encoded
    assert identity["entrypoint"] == "arc/crack_lab/replay_scorecard.py"
    assert identity["artifact_root"] == "<external>/artifacts"


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


def _complete_remote_args(tmp_path, output=None, *, mode="online"):
    return SimpleNamespace(
        mode=mode,
        release_receipt=tmp_path / "release.json",
        expected_claimed_levels=183,
        preflight_only=False,
        source_url=(
            "https://github.com/sashakolpakov/gkm/tree/" + "a" * 40
        ),
        source_revision="a" * 40,
        tags="ignored,for-complete-runs",
        output_json=R.canonical_run_receipt_path("b" * 64, mode),
    )


def _complete_binding():
    return {
        "binding_scope": "endpoint_checkpoint_bytes_only_after_full_gate",
        "receipt_sha256": "b" * 64,
        "canonical_tree_sha256": "c" * 64,
        "release_identity_source_revision": "a" * 40,
        "claimed_inventory": {"wa30": 1},
        "claimed_level_count": 183,
        "authoritative_level_count": 183,
    }


def _intent_payload(*, mode="online"):
    operation_id = "1" * 64
    receipt_sha256 = "b" * 64
    revision = "a" * 40
    opaque = R.complete_scorecard_opaque(
        operation_id=operation_id,
        mode=mode,
        receipt_sha256=receipt_sha256,
        canonical_tree_sha256="c" * 64,
        revision=revision,
    )
    return {
        "mode": mode,
        "source_url": (
            f"https://github.com/sashakolpakov/gkm/tree/{revision}"
        ),
        "source_revision": revision,
        "arc_agi_toolkit_version": "0.9.9",
        "release_receipt_sha256": receipt_sha256,
        "canonical_tree_sha256": "c" * 64,
        "checkpoint_sha256_digest": "d" * 64,
        "command_sha256": "e" * 64,
        "output_receipt": "runs/online.json",
        "tags": R.complete_scorecard_tags(
            mode=mode,
            receipt_sha256=receipt_sha256,
            revision=revision,
        ),
        "opaque": opaque,
    }


def test_release_verification_projection_cannot_leak_paths_or_secrets(tmp_path):
    raw = {
        "status": "PASS",
        "games": 25,
        "levels": 183,
        "inventory_sha256": "1" * 64,
        "canonical_tree_sha256": "2" * 64,
        "evidence_sha256": "3" * 64,
        "verifier_sha256": "4" * 64,
        "control_contract_sha256": "5" * 64,
        "receipt_sha256": "6" * 64,
        "verification_context_source_revision": "7" * 40,
        "receipt": str(tmp_path / "private" / "receipt.json"),
        "nested": {
            "repo": str(R.GKM),
            "home": "/Users/private-user",
            "api_key": "ARC_API_KEY=top-secret",
        },
    }
    projected = R.project_release_verification(raw)
    serialized = json.dumps({"schema": 2, "release_verification": projected})
    assert set(projected) == {
        "status",
        "games",
        "levels",
        "inventory_sha256",
        "canonical_tree_sha256",
        "evidence_sha256",
        "verifier_sha256",
        "control_contract_sha256",
        "receipt_sha256",
        "verification_context_source_revision",
    }
    for forbidden in (
        str(tmp_path),
        str(R.GKM),
        "/Users/",
        "/tmp/",
        "ARC_API_KEY",
        "top-secret",
    ):
        assert forbidden not in serialized


def test_run_journal_rejects_truncation_and_wrong_identity(tmp_path):
    journal_path = tmp_path / "journal.jsonl"
    journal_path.write_bytes(b"{")
    with pytest.raises(R.RunJournalError, match="truncated"):
        with R.CompleteRunJournal(
            journal_path, journal_id="a" * 64
        ):
            pass

    journal_path.unlink()
    with R.CompleteRunJournal(
        journal_path, journal_id="a" * 64
    ) as journal:
        journal.append(
            kind="INTENT",
            operation_id="1" * 64,
            payload=_intent_payload(),
        )
    with pytest.raises(R.RunJournalError, match="identity mismatch"):
        with R.CompleteRunJournal(
            journal_path, journal_id="f" * 64
        ):
            pass


def test_unresolved_intent_blocks_competition_start(tmp_path):
    journal_path = tmp_path / "journal.jsonl"
    with R.CompleteRunJournal(
        journal_path, journal_id="a" * 64
    ) as journal:
        journal.append(
            kind="INTENT",
            operation_id="1" * 64,
            payload=_intent_payload(),
        )
    with R.CompleteRunJournal(
        journal_path, journal_id="a" * 64
    ) as journal:
        with pytest.raises(
            R.RunJournalError, match="OPEN_OUTCOME_AMBIGUOUS"
        ):
            journal.assert_can_start("competition")


def _terminal_online_without_publishing_receipt(monkeypatch, tmp_path):
    monkeypatch.setattr(R, "RUN_JOURNAL_ROOT", tmp_path / "journals")
    receipt_sha256 = "b" * 64
    operation_id = "1" * 64
    card_id = "12345678-1234-4234-8234-123456789abc"
    journal_path = R.canonical_run_journal_path(receipt_sha256)
    opaque = R.complete_scorecard_opaque(
        operation_id=operation_id,
        mode="online",
        receipt_sha256=receipt_sha256,
        canonical_tree_sha256="c" * 64,
        revision="a" * 40,
    )
    checkpoints = {"wa30": "d" * 64}
    command = {"mode": "online", "entrypoint": "replay_scorecard.py"}
    payload = _intent_payload()
    payload["checkpoint_sha256_digest"] = R.json_sha256(checkpoints)
    payload["command_sha256"] = R.json_sha256(command)
    payload["opaque"] = opaque
    core = {
        "schema": 2,
        "mode": "online",
        "status": "PASS",
        "scorecard_id": card_id,
        "scorecard_url": f"https://arcprize.org/scorecards/{card_id}",
        "scorecard_open": {"status": "confirmed", "error_type": None},
        "scorecard_close": {"status": "confirmed", "error_type": None},
        "scorecard_tags": payload["tags"],
        "scorecard_opaque": opaque,
        "source_url": payload["source_url"],
        "source_revision": "a" * 40,
        "arc_agi_toolkit_version": "0.9.9",
        "started_at_utc": "2026-08-03T00:00:00Z",
        "closed_at_utc": "2026-08-03T00:02:00Z",
        "scorecard_close_started_at_utc": "2026-08-03T00:01:00Z",
        "scorecard_close_finished_at_utc": "2026-08-03T00:02:00Z",
        "command": command,
        "artifact_root": "arc/crack_lab/agent_solutions",
        "release_receipt": "arc/crack_lab/release.json",
        "release_binding": {"receipt_sha256": receipt_sha256},
        "release_verification": {"receipt_sha256": receipt_sha256},
        "checkpoint_sha256": checkpoints,
        "claimed_levels": 183,
        "authoritative_levels": 183,
        "stored_actions": 1,
        "results": {"wa30": {"remote": 1, "claimed": 1}},
        "aggregate": {},
    }
    with R.CompleteRunJournal(
        journal_path,
        journal_id=R.journal_id_for_release(receipt_sha256),
    ) as journal:
        intent = journal.append(
            kind="INTENT", operation_id=operation_id, payload=payload
        )
        opened = journal.append(
            kind="OPENED",
            operation_id=operation_id,
            payload={
                "card_id": card_id,
                "scorecard_url": f"https://arcprize.org/scorecards/{card_id}",
            },
        )
        receipt = R.finalize_journal_receipt(
            journal=journal,
            receipt_core=core,
            operation_id=operation_id,
            opaque=opaque,
            intent_record=intent,
            opened_record=opened,
            terminal_outcome="CLOSED_CONFIRMED_PASS",
        )
    return journal_path, receipt


def test_competition_blocks_terminal_snapshot_without_online_receipt(
    monkeypatch, tmp_path
):
    journal_path, _ = _terminal_online_without_publishing_receipt(
        monkeypatch, tmp_path
    )
    assert not R.canonical_run_receipt_path("b" * 64, "online").exists()
    with R.CompleteRunJournal(
        journal_path, journal_id=R.journal_id_for_release("b" * 64)
    ) as journal:
        with pytest.raises(R.RunJournalError, match="absent or invalid"):
            journal.assert_can_start("competition")


def test_competition_requires_cryptographically_bound_online_publication(
    monkeypatch, tmp_path
):
    journal_path, receipt = _terminal_online_without_publishing_receipt(
        monkeypatch, tmp_path
    )
    receipt_path = R.canonical_run_receipt_path("b" * 64, "online")
    R.write_new_json(receipt_path, receipt)
    with R.CompleteRunJournal(
        journal_path, journal_id=R.journal_id_for_release("b" * 64)
    ) as journal:
        journal.assert_can_start("competition")

    receipt_path.write_text('{"schema":2,"schema":1}\n')
    with R.CompleteRunJournal(
        journal_path, journal_id=R.journal_id_for_release("b" * 64)
    ) as journal:
        with pytest.raises(R.RunJournalError, match="duplicate key"):
            journal.assert_can_start("competition")


def test_run_journal_lock_and_path_replacement_fail_closed(tmp_path):
    journal_path = tmp_path / "journal.jsonl"
    with R.CompleteRunJournal(
        journal_path, journal_id="a" * 64
    ) as first:
        with pytest.raises(R.RunJournalError, match="holds the journal lock"):
            with R.CompleteRunJournal(
                journal_path, journal_id="a" * 64
            ):
                pass
        moved = tmp_path / "moved.jsonl"
        journal_path.rename(moved)
        journal_path.write_bytes(b"")
        with pytest.raises(R.RunJournalError, match="replaced"):
            first.append(
                kind="INTENT",
                operation_id="1" * 64,
                payload=_intent_payload(),
            )


def test_journal_samples_size_only_after_acquiring_lock(monkeypatch, tmp_path):
    journal_path = tmp_path / "journal.jsonl"
    journal_path.touch()
    event = {
        "schema": 1,
        "journal_id": "a" * 64,
        "sequence": 1,
        "previous_event_sha256": None,
        "event": "INTENT",
        "operation_id": "1" * 64,
        "timestamp_utc": "2026-08-03T00:00:00Z",
        "payload": _intent_payload(),
    }
    line = R.canonical_json(event) + b"\n"
    real_flock = R.fcntl.flock
    injected = False

    def finishing_prior_holder(descriptor, operation):
        nonlocal injected
        if operation & R.fcntl.LOCK_EX and not injected:
            injected = True
            with journal_path.open("ab") as handle:
                handle.write(line)
                handle.flush()
                R.os.fsync(handle.fileno())
        return real_flock(descriptor, operation)

    monkeypatch.setattr(R.fcntl, "flock", finishing_prior_holder)
    with R.CompleteRunJournal(
        journal_path, journal_id="a" * 64
    ) as journal:
        assert len(journal.records) == 1


def test_journal_locked_read_handles_short_pread(monkeypatch, tmp_path):
    journal_path = tmp_path / "journal.jsonl"
    with R.CompleteRunJournal(
        journal_path, journal_id="a" * 64
    ) as journal:
        journal.append(
            kind="INTENT",
            operation_id="1" * 64,
            payload=_intent_payload(),
        )
    real_pread = R.os.pread
    calls = 0

    def short_pread(descriptor, size, offset):
        nonlocal calls
        calls += 1
        return real_pread(descriptor, min(size, 7), offset)

    monkeypatch.setattr(R.os, "pread", short_pread)
    with R.CompleteRunJournal(
        journal_path, journal_id="a" * 64
    ) as journal:
        assert len(journal.records) == 1
    assert calls > 1


@pytest.mark.parametrize("mutation", ("append", "truncate", "overwrite"))
def test_journal_detects_advisory_lock_ignoring_mutation(tmp_path, mutation):
    journal_path = tmp_path / "journal.jsonl"
    with R.CompleteRunJournal(
        journal_path, journal_id="a" * 64
    ) as journal:
        journal.append(
            kind="INTENT",
            operation_id="1" * 64,
            payload=_intent_payload(),
        )
        with journal_path.open("r+b") as handle:
            if mutation == "append":
                handle.seek(0, R.os.SEEK_END)
                handle.write(b"x")
            elif mutation == "truncate":
                handle.truncate(len(journal.raw) - 1)
            else:
                handle.seek(0)
                handle.write(b"[")
            handle.flush()
            R.os.fsync(handle.fileno())
        with pytest.raises(
            R.RunJournalError, match="changed outside|replaced"
        ):
            journal.assert_can_start("online")


def test_existing_canonical_output_blocks_remote_open(monkeypatch, tmp_path):
    monkeypatch.setattr(R, "RUN_JOURNAL_ROOT", tmp_path / "journals")
    output = R.canonical_run_receipt_path("b" * 64, "online")
    output.parent.mkdir(parents=True)
    output.write_text("reserved\n")

    class Arcade:
        open_calls = 0

        def open_scorecard(self, **kwargs):
            self.open_calls += 1
            raise AssertionError("remote open must not be called")

    arcade = Arcade()
    result = R.execute_remote_run(
        args=_complete_remote_args(tmp_path),
        arcade=arcade,
        engine_action_cls={},
        games=["wa30"],
        plan={"wa30": {"reached": 1, "final_path": []}},
        segs={"wa30": []},
        artifact_root=tmp_path,
        checkpoint_hashes={"wa30": "d" * 64},
        binding=_complete_binding(),
        release_verification={"status": "PASS"},
        arc_agi_version="0.9.9",
        started_at_utc="2026-08-03T00:00:00Z",
        claimed_levels=183,
        complete_release=True,
    )
    assert result == 2
    assert arcade.open_calls == 0


def test_output_creation_race_is_lost_before_intent_or_remote_open(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(R, "RUN_JOURNAL_ROOT", tmp_path / "journals")
    output = R.canonical_run_receipt_path("b" * 64, "online")

    class LosingReservationRace:
        def __init__(self, path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("external winner\n")
            raise FileExistsError(path)

    monkeypatch.setattr(R, "ReservedReceipt", LosingReservationRace)

    class Arcade:
        open_calls = 0

        def open_scorecard(self, **kwargs):
            self.open_calls += 1
            raise AssertionError("remote open must not be called")

    arcade = Arcade()
    result = R.execute_remote_run(
        args=_complete_remote_args(tmp_path),
        arcade=arcade,
        engine_action_cls={},
        games=["wa30"],
        plan={"wa30": {"reached": 1, "final_path": []}},
        segs={"wa30": []},
        artifact_root=tmp_path,
        checkpoint_hashes={"wa30": "d" * 64},
        binding=_complete_binding(),
        release_verification={"status": "PASS"},
        arc_agi_version="0.9.9",
        started_at_utc="2026-08-03T00:00:00Z",
        claimed_levels=183,
        complete_release=True,
    )
    assert result == 2
    assert arcade.open_calls == 0
    assert output.read_text() == "external winner\n"
    journal = R.canonical_run_journal_path("b" * 64)
    assert journal.read_bytes() == b""


def test_server_success_client_crash_stays_ambiguous_and_never_retries(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(R, "RUN_JOURNAL_ROOT", tmp_path / "journals")
    lifecycle = []
    real_fsync = R.os.fsync

    def tracked_fsync(descriptor):
        lifecycle.append("fsync")
        return real_fsync(descriptor)

    monkeypatch.setattr(R.os, "fsync", tracked_fsync)

    class ServerCreatedButNoResponse:
        def __init__(self):
            self.open_calls = 0
            self.last_opaque = None

        def open_scorecard(self, *, source_url, tags, opaque):
            self.open_calls += 1
            self.last_opaque = opaque
            lifecycle.append("open")
            raise TimeoutError("response lost after server creation")

    arcade = ServerCreatedButNoResponse()
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    common = {
        "arcade": arcade,
        "engine_action_cls": {},
        "games": ["wa30"],
        "plan": {"wa30": {"reached": 1, "final_path": []}},
        "segs": {"wa30": []},
        "artifact_root": artifact_root,
        "checkpoint_hashes": {"wa30": "d" * 64},
        "binding": _complete_binding(),
        "release_verification": {"status": "PASS"},
        "arc_agi_version": "0.9.9",
        "started_at_utc": "2026-08-03T00:00:00Z",
        "claimed_levels": 183,
        "complete_release": True,
    }
    first_output = R.canonical_run_receipt_path("b" * 64, "online")
    result = R.execute_remote_run(
        args=_complete_remote_args(tmp_path, first_output), **common
    )
    assert result == 1
    assert arcade.open_calls == 1
    assert lifecycle.index("open") >= 2
    assert arcade.last_opaque["gkm_operation_id"]
    receipt = json.loads(first_output.read_text())
    assert receipt["run_journal"]["terminal_outcome"] == (
        "OPEN_OUTCOME_AMBIGUOUS"
    )

    before_retry = first_output.read_bytes()
    result = R.execute_remote_run(
        args=_complete_remote_args(tmp_path), **common
    )
    assert result == 2
    assert arcade.open_calls == 1
    assert first_output.read_bytes() == before_retry


def test_opened_append_failure_precedes_any_environment_mutation(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(R, "RUN_JOURNAL_ROOT", tmp_path / "journals")
    original_append = R.CompleteRunJournal.append

    def fail_opened(self, *, kind, operation_id, payload):
        if kind == "OPENED":
            raise OSError("simulated opened fsync failure")
        return original_append(
            self, kind=kind, operation_id=operation_id, payload=payload
        )

    monkeypatch.setattr(R.CompleteRunJournal, "append", fail_opened)

    class Arcade:
        make_calls = 0

        def open_scorecard(self, **kwargs):
            return "12345678-1234-4234-8234-123456789abc"

        def make(self, *args, **kwargs):
            self.make_calls += 1

        def close_scorecard(self, card_id):
            return None

    arcade = Arcade()
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    output = R.canonical_run_receipt_path("b" * 64, "online")
    result = R.execute_remote_run(
        args=_complete_remote_args(tmp_path, output),
        arcade=arcade,
        engine_action_cls={},
        games=["wa30"],
        plan={"wa30": {"reached": 1, "final_path": []}},
        segs={"wa30": []},
        artifact_root=artifact_root,
        checkpoint_hashes={"wa30": "d" * 64},
        binding=_complete_binding(),
        release_verification={"status": "PASS"},
        arc_agi_version="0.9.9",
        started_at_utc="2026-08-03T00:00:00Z",
        claimed_levels=183,
        complete_release=True,
    )
    assert result == 1
    assert arcade.make_calls == 0
    assert output.exists()
    assert output.read_bytes() == b""


def test_terminal_append_failure_prevents_receipt_publication(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(R, "RUN_JOURNAL_ROOT", tmp_path / "journals")
    original_append = R.CompleteRunJournal.append

    def fail_terminal(self, *, kind, operation_id, payload):
        if kind == "TERMINAL":
            raise OSError("simulated terminal fsync failure")
        return original_append(
            self, kind=kind, operation_id=operation_id, payload=payload
        )

    monkeypatch.setattr(R.CompleteRunJournal, "append", fail_terminal)
    card_id = "12345678-1234-4234-8234-123456789abc"
    source_url = "https://github.com/sashakolpakov/gkm/tree/" + "a" * 40

    class Card:
        def model_dump(self, **kwargs):
            return {
                "card_id": card_id,
                "source_url": source_url,
                "competition_mode": False,
                "score": 100.0,
                "environments": [],
                "total_levels_completed": 0,
                "total_environments": 0,
            }

    class Arcade:
        def open_scorecard(self, **kwargs):
            return card_id

        def close_scorecard(self, value):
            return Card()

    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    output = R.canonical_run_receipt_path("b" * 64, "online")
    result = R.execute_remote_run(
        args=_complete_remote_args(tmp_path, output),
        arcade=Arcade(),
        engine_action_cls={},
        games=[],
        plan={},
        segs={},
        artifact_root=artifact_root,
        checkpoint_hashes={},
        binding=_complete_binding(),
        release_verification={"status": "PASS"},
        arc_agi_version="0.9.9",
        started_at_utc="2026-08-03T00:00:00Z",
        claimed_levels=183,
        complete_release=True,
    )
    assert result == 1
    assert output.exists()
    assert output.read_bytes() == b""


def test_closed_scorecard_accepts_full_provider_retry_history():
    card_id = "12345678-1234-4234-8234-123456789abc"
    source_url = "https://github.com/sashakolpakov/gkm/tree/" + "a" * 40
    plan = {"wa30": {"reached": 2}}
    tags = ["gkm-v3"]
    opaque = {"schema": 1}
    aggregate = _provider_aggregate(
        card_id=card_id,
        source_url=source_url,
        mode="competition",
        plan=plan,
        tags=tags,
        opaque=opaque,
    )
    environment = aggregate["environments"][0]
    failed = json.loads(json.dumps(environment["runs"][0]))
    failed.update(
        {
            "guid": "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
            "score": 0.0,
            "levels_completed": 0,
            "actions": 1,
            "resets": 1,
            "state": "GAME_OVER",
            "completed": False,
            "level_scores": [0.0, 0.0],
            "level_actions": [1, 0],
        }
    )
    environment["runs"].insert(0, failed)
    environment["actions"] += 1
    environment["resets"] += 1
    aggregate["total_actions"] += 1
    assert R.validate_closed_scorecard(
        aggregate,
        mode="competition",
        card_id=card_id,
        source_url=source_url,
        games=["wa30"],
        plan=plan,
        scorecard_tags=tags,
        scorecard_opaque=opaque,
    ) == aggregate


def test_closed_scorecard_rejects_noncanonical_provider_environment_id():
    card_id = "12345678-1234-4234-8234-123456789abc"
    source_url = "https://github.com/sashakolpakov/gkm/tree/" + "a" * 40
    plan = {"wa30": {"reached": 1}}
    aggregate = _provider_aggregate(
        card_id=card_id,
        source_url=source_url,
        mode="online",
        plan=plan,
        tags=[],
        opaque=None,
    )
    environment = aggregate["environments"][0]
    environment["id"] = "wa30-/Users/private"
    environment["runs"][0]["id"] = environment["id"]
    with pytest.raises(ValueError, match="ambiguous|ID mismatch"):
        R.validate_closed_scorecard(
            aggregate,
            mode="online",
            card_id=card_id,
            source_url=source_url,
            games=["wa30"],
            plan=plan,
            scorecard_tags=[],
            scorecard_opaque=None,
        )


@pytest.mark.parametrize("redirect", ("receipts", "release", "snapshots"))
def test_canonical_output_parent_symlink_blocks_before_remote_open(
    monkeypatch, tmp_path, redirect
):
    journal_root = tmp_path / "journals"
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.setattr(R, "RUN_JOURNAL_ROOT", journal_root)
    if redirect == "receipts":
        journal_root.mkdir()
        (journal_root / "receipts").symlink_to(
            outside, target_is_directory=True
        )
    elif redirect == "release":
        (journal_root / "receipts").mkdir(parents=True)
        (journal_root / "receipts" / ("b" * 64)).symlink_to(
            outside, target_is_directory=True
        )
    else:
        journal_root.mkdir()
        (journal_root / "snapshots").symlink_to(
            outside, target_is_directory=True
        )

    class Arcade:
        open_calls = 0

        def open_scorecard(self, **kwargs):
            self.open_calls += 1
            raise AssertionError("remote open must not occur")

    arcade = Arcade()
    result = R.execute_remote_run(
        args=_complete_remote_args(tmp_path),
        arcade=arcade,
        engine_action_cls={},
        games=["wa30"],
        plan={"wa30": {"reached": 1, "final_path": []}},
        segs={"wa30": []},
        artifact_root=tmp_path,
        checkpoint_hashes={"wa30": "d" * 64},
        binding=_complete_binding(),
        release_verification={"status": "PASS"},
        arc_agi_version="0.9.9",
        started_at_utc="2026-08-03T00:00:00Z",
        claimed_levels=183,
        complete_release=True,
    )
    assert result == 2
    assert arcade.open_calls == 0
    assert list(outside.iterdir()) == []


def test_reserved_receipt_detects_same_length_in_place_overwrite(
    monkeypatch, tmp_path
):
    target = tmp_path / "receipts" / "online.json"
    with R.ReservedReceipt(target) as reservation:
        real_write = R.os.write
        corrupted = False

        def overwrite_after_write(descriptor, payload):
            nonlocal corrupted
            written = real_write(descriptor, payload)
            if descriptor == reservation.fd and not corrupted:
                corrupted = True
                R.os.pwrite(descriptor, b"X" * written, 0)
                R.os.fsync(descriptor)
            return written

        monkeypatch.setattr(R.os, "write", overwrite_after_write)
        with pytest.raises(R.RunJournalError, match="changed during publication"):
            reservation.publish_json({"status": "PASS"})
        assert reservation.published is False


def test_runtime_revision_binds_head_tracked_and_dynamic_sources(monkeypatch):
    revision = "a" * 40
    game = "wa30"
    environment_path, environment_digest = (
        R.RUNTIME_ENVIRONMENT_SOURCE_SHA256[game]
    )
    tracked = {
        path: f"tracked:{path}".encode() for path in R.RUNTIME_TRACKED_PATHS
    }
    working = dict(tracked)
    environment_bytes = b"environment-source"
    monkeypatch.setitem(
        R.RUNTIME_ENVIRONMENT_SOURCE_SHA256,
        game,
        (environment_path, hashlib.sha256(environment_bytes).hexdigest()),
    )

    def git_output(arguments, *, max_bytes):
        if arguments[0] == "rev-parse":
            return (revision + "\n").encode()
        object_name = arguments[-1]
        relative = object_name.split(":", 1)[1]
        if arguments[0] == "cat-file":
            return f"{len(tracked[relative])}\n".encode()
        return tracked[relative]

    def read_source(path, *, label):
        relative = R._lexical_absolute(path).relative_to(
            R._lexical_absolute(R.GKM)
        ).as_posix()
        if relative == environment_path:
            return environment_bytes
        return working[relative]

    monkeypatch.setattr(R, "_git_output", git_output)
    monkeypatch.setattr(R, "read_bounded_regular", read_source)
    result = REAL_VERIFY_RUNTIME_REVISION(revision, [game])
    assert result["source_revision"] == revision
    assert result["files_sha256"][environment_path] != environment_digest

    working[R.RUNTIME_TRACKED_PATHS[0]] = b"dirty-working-copy"
    with pytest.raises(ValueError, match="working runtime source differs"):
        REAL_VERIFY_RUNTIME_REVISION(revision, [game])


def test_runtime_revision_requires_exact_head(monkeypatch):
    monkeypatch.setattr(
        R,
        "_git_output",
        lambda arguments, max_bytes: ("b" * 40 + "\n").encode(),
    )
    with pytest.raises(ValueError, match="HEAD differs"):
        REAL_VERIFY_RUNTIME_REVISION("a" * 40, ["wa30"])


def test_runtime_mismatch_blocks_before_journal_reservation_or_open(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(R, "RUN_JOURNAL_ROOT", tmp_path / "journals")
    monkeypatch.setattr(
        R,
        "verify_runtime_revision",
        lambda revision, games: (_ for _ in ()).throw(
            ValueError("dirty runtime")
        ),
    )

    class Arcade:
        open_calls = 0

        def open_scorecard(self, **kwargs):
            self.open_calls += 1

    arcade = Arcade()
    output = R.canonical_run_receipt_path("b" * 64, "online")
    result = R.execute_remote_run(
        args=_complete_remote_args(tmp_path, output),
        arcade=arcade,
        engine_action_cls={},
        games=["wa30"],
        plan={"wa30": {"reached": 1, "final_path": []}},
        segs={"wa30": []},
        artifact_root=tmp_path,
        checkpoint_hashes={"wa30": "d" * 64},
        binding=_complete_binding(),
        release_verification={"status": "PASS"},
        arc_agi_version="0.9.9",
        started_at_utc="2026-08-03T00:00:00Z",
        claimed_levels=183,
        complete_release=True,
    )
    assert result == 2
    assert arcade.open_calls == 0
    assert not output.exists()
    assert not R.canonical_run_journal_path("b" * 64).exists()


def test_env_loader_reads_only_arc_api_key_after_explicit_call(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(R, "GKM", tmp_path)
    monkeypatch.delenv("ARC_API_KEY", raising=False)
    monkeypatch.delenv("UNRELATED_SECRET", raising=False)
    (tmp_path / ".env").write_text(
        "UNRELATED_SECRET=must-not-enter-environ\nARC_API_KEY=arc-only\n"
    )
    assert R.load_arc_api_key() == "arc-only"
    assert "UNRELATED_SECRET" not in R.os.environ


def test_complete_main_checks_runtime_before_dynamic_segmentation(
    monkeypatch, tmp_path
):
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    receipt = tmp_path / "release.json"
    receipt.write_text("{}\n")
    order = []
    revision = "a" * 40
    monkeypatch.setattr(R, "toolkit_version", lambda: "0.9.9")
    def verify_release(**kwargs):
        order.append("release")
        return {}

    monkeypatch.setattr(R, "verify_frozen_release", verify_release)
    monkeypatch.setattr(
        R,
        "project_release_verification",
        lambda value: {
            "receipt_sha256": "b" * 64,
            "verification_context_source_revision": revision,
        },
    )
    monkeypatch.setattr(
        R,
        "load_checkpoint",
        lambda *args: (
            {"game": "wa30", "reached": 183, "final_path": []},
            "d" * 64,
        ),
    )
    monkeypatch.setattr(
        R,
        "release_binding",
        lambda *args: {
            "receipt_sha256": "b" * 64,
            "claimed_level_count": 183,
            "authoritative_level_count": 183,
            "release_identity_source_revision": revision,
        },
    )

    def runtime_check(source_revision, games):
        order.append("runtime")
        return {}

    def segment(game, actions):
        order.append("segment")
        return [[] for _ in range(183)]

    monkeypatch.setattr(R, "verify_runtime_revision", runtime_check)
    monkeypatch.setattr(R, "level_segments", segment)
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
            "--expected-claimed-levels",
            "183",
            "--source-revision",
            revision,
            "--preflight-only",
        ],
    )
    assert R.main() == 0
    assert order == ["runtime", "release", "runtime", "segment"]


def test_execute_never_prints_raw_closed_card_repr(capsys, tmp_path):
    card_id = "12345678-1234-4234-8234-123456789abc"
    source_url = "https://example.test/source"
    secret = "ARC_API_KEY=must-not-appear"
    aggregate = _provider_aggregate(
        card_id=card_id,
        source_url=source_url,
        mode="online",
        plan={},
        tags=["test"],
        opaque=None,
    )

    class Card:
        def __repr__(self):
            return f"Card({secret})"

        def model_dump(self, **kwargs):
            return aggregate

    class Arcade:
        def open_scorecard(self, **kwargs):
            return card_id

        def close_scorecard(self, value):
            return Card()

    args = SimpleNamespace(
        mode="online",
        release_receipt=None,
        expected_claimed_levels=None,
        preflight_only=False,
        source_url=source_url,
        source_revision=None,
        tags="test",
        output_json=None,
    )
    assert R.execute_remote_run(
        args=args,
        arcade=Arcade(),
        engine_action_cls={},
        games=[],
        plan={},
        segs={},
        artifact_root=tmp_path,
        checkpoint_hashes={},
        binding=None,
        release_verification=None,
        arc_agi_version="0.9.9",
        started_at_utc="2026-08-03T00:00:00Z",
        claimed_levels=0,
        complete_release=False,
    ) == 0
    assert secret not in capsys.readouterr().out


def test_invalid_provider_card_is_not_copied_into_fail_receipt(
    capsys, tmp_path
):
    card_id = "12345678-1234-4234-8234-123456789abc"
    source_url = "https://example.test/source"
    secret = "ARC_API_KEY=must-not-persist"
    aggregate = _provider_aggregate(
        card_id=card_id,
        source_url=source_url,
        mode="online",
        plan={},
        tags=["test"],
        opaque=None,
    )
    aggregate["private_path"] = f"/Users/private/{secret}"

    class Card:
        def model_dump(self, **kwargs):
            return aggregate

    class Arcade:
        def open_scorecard(self, **kwargs):
            return card_id

        def close_scorecard(self, value):
            return Card()

    output = tmp_path / "run.json"
    args = SimpleNamespace(
        mode="online",
        release_receipt=None,
        expected_claimed_levels=None,
        preflight_only=False,
        source_url=source_url,
        source_revision=None,
        tags="test",
        output_json=output,
    )
    assert R.execute_remote_run(
        args=args,
        arcade=Arcade(),
        engine_action_cls={},
        games=[],
        plan={},
        segs={},
        artifact_root=tmp_path,
        checkpoint_hashes={},
        binding=None,
        release_verification=None,
        arc_agi_version="0.9.9",
        started_at_utc="2026-08-03T00:00:00Z",
        claimed_levels=0,
        complete_release=False,
    ) == 1
    receipt = json.loads(output.read_text())
    assert receipt["aggregate"] is None
    assert secret not in output.read_text()
    assert secret not in capsys.readouterr().out


@pytest.mark.parametrize("field", ("schema", "sequence", "opaque_schema"))
def test_run_journal_rejects_boolean_integer_fields(field):
    journal_id = "a" * 64
    event = {
        "schema": 1,
        "journal_id": journal_id,
        "sequence": 1,
        "previous_event_sha256": None,
        "event": "INTENT",
        "operation_id": "1" * 64,
        "timestamp_utc": "2026-08-03T00:00:00Z",
        "payload": _intent_payload(),
    }
    if field == "opaque_schema":
        event["payload"]["opaque"]["schema"] = True
    else:
        event[field] = True
    with pytest.raises(R.RunJournalError, match="sequence|opaque schema"):
        R.parse_run_journal(
            R.canonical_json(event) + b"\n",
            expected_journal_id=journal_id,
        )


@pytest.mark.parametrize(
    "raw",
    (
        b'{"schema":1,"schema":1}\n',
        b'{"schema":NaN}\n',
    ),
)
def test_run_journal_uses_strict_json(raw):
    with pytest.raises(R.RunJournalError, match="duplicate|non-finite"):
        R.parse_run_journal(raw)
