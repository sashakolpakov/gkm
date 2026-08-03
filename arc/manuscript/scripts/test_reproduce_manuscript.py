from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

import reproduce_manuscript


V2_REVISION = "c1f8168f230732f2d745c234555b3e3dfcb8aefa"
COMPLETE_REVISION = "d" * 40


def _write_receipt(
    directory: Path,
    *,
    revision: str,
    inventory: dict[str, int] | None = None,
) -> tuple[Path, dict[str, object]]:
    body: dict[str, object] = {
        "release_identity": {"source_revision": revision}
    }
    if inventory is not None:
        body["inventory"] = inventory
    raw = (
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        + b"\n"
    )
    path = directory / f"{hashlib.sha256(raw).hexdigest()}.json"
    path.write_bytes(raw)
    return path, body


def _write_raw_receipt(directory: Path, raw: bytes) -> Path:
    path = directory / f"{hashlib.sha256(raw).hexdigest()}.json"
    path.write_bytes(raw)
    return path


def _verification_identity(path: Path, revision: str) -> dict[str, str]:
    return {
        "receipt_sha256": path.stem,
        "verification_context_source_revision": revision,
    }


def _git_history_repo(directory: Path) -> tuple[Path, str, str]:
    repo = directory / "repo"
    source = repo / reproduce_manuscript.FROZEN_HISTORY_TREE / "wa30_legs"
    source.mkdir(parents=True)
    (source / "players.py").write_text("published = True\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Reproduction Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "freeze",
        ],
        check=True,
    )
    revision = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    tree = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "rev-parse",
            f"{revision}:{reproduce_manuscript.FROZEN_HISTORY_TREE}",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    return repo, revision, tree


def _payload() -> dict:
    systems = {}
    for index, system in enumerate(reproduce_manuscript.SYSTEMS, start=1):
        systems[system] = {
            field: index for field in reproduce_manuscript.STAT_FIELDS
        }
        systems[system]["sharp_drops_with_literal_reuse"] = [{}] * index
    return {"summary": {"systems": systems}}


def test_summary_counts_coupled_witnesses() -> None:
    summary = reproduce_manuscript._summary(_payload())
    assert summary["GKM"]["sharp_drops_with_literal_reuse"] == 1
    assert summary["Retrodict"]["sharp_drops_with_literal_reuse"] == 4


def test_generated_stats_are_machine_readable(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[3]
    payload = json.loads(
        (repo / "arc/audit_results/marginal-literal-reuse.json").read_text()
    )
    summary = reproduce_manuscript._summary(payload)
    tex_path, md_path = reproduce_manuscript._write_generated_stats(
        summary, payload, tmp_path,
    )
    assert (
        rf"\newcommand{{\GKMExactWins}}{{"
        f"{summary['GKM']['exact_winning_checkpoints']}"
        "}"
    ) in tex_path.read_text()
    assert "| Retrodict | 170 memory (145 transitions) | 0 |" in md_path.read_text()


def test_release_verification_uses_revision_bound_wrapper(
    monkeypatch, tmp_path: Path
) -> None:
    calls = []
    receipt, expected_body = _write_receipt(
        tmp_path, revision=V2_REVISION
    )

    def fake_run_json(command: list[str], *, cwd: Path) -> dict:
        calls.append((command, cwd))
        return {
            "status": "PASS",
            "kind": "partial_campaign_freeze",
            "games": 25,
            "claimed_levels": 181,
            "authoritative_levels": 183,
            "unclaimed_boundaries": [
                {"game": "lf52", "level": 9},
                {"game": "lf52", "level": 10},
            ],
            **_verification_identity(receipt, V2_REVISION),
        }

    monkeypatch.setattr(reproduce_manuscript, "_run_json", fake_run_json)
    repo = tmp_path / "repo"
    verifier = tmp_path / "verifier"
    result, receipt_body = reproduce_manuscript._verify_frozen_release(
        repo=repo,
        release_root=tmp_path / "artifacts",
        release_receipt=receipt,
        verifier_root=verifier,
    )
    assert result["status"] == "PASS"
    assert receipt_body == expected_body
    command, cwd = calls[0]
    assert cwd == repo
    assert command[:2] == [
        sys.executable,
        "arc/crack_lab/verify_frozen_release.py",
    ]
    snapshot = Path(command[command.index("--receipt") + 1])
    assert snapshot != receipt
    assert snapshot.name == receipt.name
    assert command[-2:] == ["--verifier-root", str(verifier.resolve())]


def test_complete_release_verification_is_accepted_and_counted(
    monkeypatch, tmp_path: Path
) -> None:
    receipt, _ = _write_receipt(tmp_path, revision=COMPLETE_REVISION)
    verification = {
        "status": "PASS",
        "games": 25,
        "levels": 183,
        **_verification_identity(receipt, COMPLETE_REVISION),
    }
    monkeypatch.setattr(
        reproduce_manuscript,
        "_run_json",
        lambda _command, *, cwd: dict(verification),
    )
    result, _receipt_body = reproduce_manuscript._verify_frozen_release(
        repo=tmp_path / "repo",
        release_root=tmp_path / "artifacts",
        release_receipt=receipt,
        verifier_root=None,
    )
    assert result["levels"] == 183

    release, taint, boundaries, protocol = (
        reproduce_manuscript._receipt_bound_audit_reports(
            result, receipt.stem
        )
    )
    assert release == {
        "schema": 2,
        "verdict": "PASS",
        "authority": "schema-v2 complete-release receipt verification",
        "receipt_sha256": receipt.stem,
        "claimed_boundaries": 183,
        "unclaimed_boundaries": [],
    }
    assert taint["canonical"]["files"] == 183
    assert boundaries["checkpoints"] == boundaries["exact"] == 183
    assert protocol["boundaries"] == 183


def test_receipt_body_cannot_be_swapped_after_verification(
    monkeypatch, tmp_path: Path
) -> None:
    inventory = {"ft09": 2, "tr87": 6, "wa30": 1}
    receipt, expected_body = _write_receipt(
        tmp_path,
        revision=COMPLETE_REVISION,
        inventory=inventory,
    )
    original_raw = receipt.read_bytes()

    def fake_run_json(command: list[str], *, cwd: Path) -> dict[str, object]:
        snapshot = Path(command[command.index("--receipt") + 1])
        assert snapshot != receipt
        assert snapshot.read_bytes() == original_raw
        receipt.write_text("swapped after verification\n")
        return {
            "status": "PASS",
            "games": 25,
            "levels": 183,
            **_verification_identity(receipt, COMPLETE_REVISION),
        }

    monkeypatch.setattr(reproduce_manuscript, "_run_json", fake_run_json)
    _verification, retained_body = reproduce_manuscript._verify_frozen_release(
        repo=tmp_path / "repo",
        release_root=tmp_path / "artifacts",
        release_receipt=receipt,
        verifier_root=None,
    )
    assert retained_body == expected_body
    payload = {
        "rows": [
            {
                "system": "GKM",
                "game": game,
                "completed_level": level,
                "source_checkpoint_exact": True,
            }
            for game, level in (("ft09", 1), ("wa30", 1))
        ]
    }
    assert reproduce_manuscript._source_audit_scope(
        retained_body, payload
    )["replay_verified_endpoint_wins"] == 9


@pytest.mark.parametrize(
    "raw, message",
    (
        (b"not-json\n", "invalid JSON"),
        (
            (
                '{"release_identity":{"source_revision":"'
                + COMPLETE_REVISION
                + '"},"release_identity":{"source_revision":"'
                + COMPLETE_REVISION
                + '"}}\n'
            ).encode("ascii"),
            "duplicate key",
        ),
        (b"{}\n", "valid source revision"),
    ),
)
def test_malformed_or_duplicate_receipt_is_rejected_before_verification(
    monkeypatch, tmp_path: Path, raw: bytes, message: str
) -> None:
    receipt = _write_raw_receipt(tmp_path, raw)
    monkeypatch.setattr(
        reproduce_manuscript,
        "_run_json",
        lambda *_args, **_kwargs: pytest.fail("malformed receipt was executed"),
    )
    with pytest.raises(RuntimeError, match=message):
        reproduce_manuscript._verify_frozen_release(
            repo=tmp_path / "repo",
            release_root=tmp_path / "artifacts",
            release_receipt=receipt,
            verifier_root=None,
        )


@pytest.mark.parametrize(
    "identity_override",
    (
        {"verification_context_source_revision": "e" * 40},
        {"receipt_sha256": "f" * 64},
    ),
)
def test_v2_verification_identity_must_match_exact_receipt(
    monkeypatch, tmp_path: Path, identity_override: dict[str, str]
) -> None:
    receipt, _ = _write_receipt(tmp_path, revision=V2_REVISION)
    identity = _verification_identity(receipt, V2_REVISION)
    identity.update(identity_override)
    monkeypatch.setattr(
        reproduce_manuscript,
        "_run_json",
        lambda _command, *, cwd: {
            "status": "PASS",
            "kind": "partial_campaign_freeze",
            "games": 25,
            "claimed_levels": 181,
            "authoritative_levels": 183,
            "unclaimed_boundaries": [
                {"game": "lf52", "level": 9},
                {"game": "lf52", "level": 10},
            ],
            **identity,
        },
    )
    with pytest.raises(RuntimeError, match="identity did not match"):
        reproduce_manuscript._verify_frozen_release(
            repo=tmp_path / "repo",
            release_root=tmp_path / "artifacts",
            release_receipt=receipt,
            verifier_root=None,
        )


def test_partial_release_reports_remain_v2_compatible() -> None:
    verification = {
        "status": "PASS",
        "kind": "partial_campaign_freeze",
        "games": 25,
        "claimed_levels": 181,
        "authoritative_levels": 183,
        "unclaimed_boundaries": [
            {"game": "lf52", "level": 9},
            {"game": "lf52", "level": 10},
        ],
    }
    release, taint, boundaries, protocol = (
        reproduce_manuscript._receipt_bound_audit_reports(
            verification, "b" * 64
        )
    )
    assert release["authority"] == (
        "schema-v2 partial-release receipt verification"
    )
    assert release["claimed_boundaries"] == 181
    assert release["unclaimed_boundaries"] == verification[
        "unclaimed_boundaries"
    ]
    assert taint["canonical"]["files"] == 181
    assert boundaries["checkpoints"] == boundaries["exact"] == 181
    assert protocol["boundaries"] == 181


@pytest.mark.parametrize(
    "verification",
    (
        {"status": "FAIL", "games": 25, "levels": 183},
        {"status": "PASS", "games": 24, "levels": 183},
        {"status": "PASS", "games": "25", "levels": 183},
        {"status": "PASS", "games": 25, "levels": 182},
        {"status": "PASS", "games": 25, "levels": "183"},
        {
            "status": "PASS",
            "kind": "partial_campaign_freeze",
            "games": 25,
            "claimed_levels": "181",
            "authoritative_levels": 183,
            "unclaimed_boundaries": [
                {"game": "lf52", "level": 9},
                {"game": "lf52", "level": 10},
            ],
        },
        {
            "status": "PASS",
            "kind": "partial_campaign_freeze",
            "games": 25,
            "claimed_levels": 180,
            "authoritative_levels": 183,
            "unclaimed_boundaries": [
                {"game": "lf52", "level": 8},
                {"game": "lf52", "level": 9},
                {"game": "lf52", "level": 10},
            ],
        },
        {
            "status": "PASS",
            "kind": "partial_campaign_freeze",
            "games": 25,
            "claimed_levels": 181,
            "authoritative_levels": 183,
            "unclaimed_boundaries": [
                {"game": "lf52", "level": 8},
                {"game": "lf52", "level": 10},
            ],
        },
        {
            "status": "PASS",
            "kind": "unknown_release_kind",
            "games": 25,
            "levels": 183,
        },
    ),
)
def test_release_verification_shapes_fail_closed(
    verification: dict[str, object]
) -> None:
    with pytest.raises(RuntimeError, match="did not verify"):
        reproduce_manuscript._release_verification_summary(
            verification, "c" * 64
        )


def test_complete_release_history_is_bound_to_receipt_revision(
    tmp_path: Path,
) -> None:
    repo, revision, tree = _git_history_repo(tmp_path)
    assert reproduce_manuscript._history_tree_identity(
        repo=repo,
        complete_release=True,
        explicit_history_root=None,
        history_revision=revision,
        verification_context_source_revision=revision,
    ) == tree
    assert reproduce_manuscript._history_tree_identity(
        repo=repo,
        complete_release=True,
        explicit_history_root=tmp_path / "history",
        history_revision=revision,
        verification_context_source_revision=revision,
    ) == tree

    with pytest.raises(RuntimeError, match="must equal"):
        reproduce_manuscript._history_tree_identity(
            repo=repo,
            complete_release=True,
            explicit_history_root=None,
            history_revision="e" * 40,
            verification_context_source_revision=revision,
        )


def test_caller_cannot_assert_an_independent_complete_history_hash(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["reproduce_manuscript.py", "--history-tree-sha1", "f" * 40],
    )
    with pytest.raises(SystemExit):
        reproduce_manuscript._args()


def test_partial_release_keeps_v2_history_defaults(tmp_path: Path) -> None:
    assert reproduce_manuscript._history_tree_identity(
        repo=tmp_path / "unused",
        complete_release=False,
        explicit_history_root=None,
        history_revision=reproduce_manuscript.FROZEN_SOURCE_HISTORY_REVISION,
        verification_context_source_revision=V2_REVISION,
    ) == ""
    assert reproduce_manuscript._history_tree_identity(
        repo=tmp_path / "unused",
        complete_release=False,
        explicit_history_root=tmp_path / "history",
        history_revision=reproduce_manuscript.FROZEN_SOURCE_HISTORY_REVISION,
        verification_context_source_revision=V2_REVISION,
    ) == reproduce_manuscript.FROZEN_SOURCE_HISTORY_TREE_SHA1


def test_history_is_extracted_from_pinned_git_revision(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    source = repo / reproduce_manuscript.FROZEN_HISTORY_TREE / "wa30_legs"
    source.mkdir(parents=True)
    (source / "players.py").write_text("published = True\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Reproduction Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "freeze",
        ],
        check=True,
    )
    revision = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    (source / "players.py").write_text("mutable = True\n")

    extracted = reproduce_manuscript._materialize_history(
        repo=repo,
        revision=revision,
        output=tmp_path / "extracted",
    )
    assert (extracted / "wa30_legs/players.py").read_text() == "published = True\n"


def test_archive_history_must_match_complete_git_tree(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    source = repo / "history"
    (source / "wa30_legs").mkdir(parents=True)
    (source / "wa30_legs/players.py").write_text("published = True\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    root_tree = subprocess.run(
        ["git", "-C", str(repo), "write-tree"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    tree_sha1 = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", f"{root_tree}:history"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()

    copied = reproduce_manuscript._copy_authenticated_history(
        source=source,
        output=tmp_path / "copied",
        expected_tree_sha1=tree_sha1,
    )
    assert (copied / "wa30_legs/players.py").read_text() == "published = True\n"

    (source / "unexpected.py").write_text("extra = True\n")
    with pytest.raises(RuntimeError, match="does not match"):
        reproduce_manuscript._copy_authenticated_history(
            source=source,
            output=tmp_path / "rejected",
            expected_tree_sha1=tree_sha1,
        )


def test_source_audit_scope_separates_endpoint_wins_from_source_rows() -> None:
    receipt = {"claimed_inventory": {"ft09": 2, "tr87": 6, "wa30": 1}}
    payload = {
        "rows": [
            {
                "system": "GKM",
                "game": game,
                "completed_level": level,
                "source_checkpoint_exact": True,
            }
            for game, level in (("ft09", 1), ("wa30", 1))
        ]
    }
    scope = reproduce_manuscript._source_audit_scope(receipt, payload)
    assert scope["replay_verified_endpoint_wins"] == 9
    assert scope["admissible_exact_winning_source_checkpoints"] == 2
    assert scope["excluded_from_source_marginals"] == [
        {"game": game, "level": level}
        for game, level in reproduce_manuscript.FROZEN_SOURCE_AUDIT_EXCLUSIONS
    ]

    payload["rows"].append(
        {
            "system": "GKM",
            "game": "tr87",
            "completed_level": 1,
            "source_checkpoint_exact": True,
        }
    )
    with pytest.raises(RuntimeError, match="exclusions changed"):
        reproduce_manuscript._source_audit_scope(receipt, payload)


def test_source_audit_scope_accepts_complete_receipt_inventory() -> None:
    receipt = {"inventory": {"ft09": 2, "tr87": 6, "wa30": 1}}
    payload = {
        "rows": [
            {
                "system": "GKM",
                "game": game,
                "completed_level": level,
                "source_checkpoint_exact": True,
            }
            for game, level in (("ft09", 1), ("wa30", 1))
        ]
    }
    scope = reproduce_manuscript._source_audit_scope(receipt, payload)
    assert scope["replay_verified_endpoint_wins"] == 9


def test_complete_source_scope_requires_new_levels_and_exact_old_exclusions() -> None:
    inventory = {"ft09": 2, "lf52": 10, "tr87": 6, "wa30": 1}
    excluded = set(reproduce_manuscript.FROZEN_SOURCE_AUDIT_EXCLUSIONS)
    source_ids = {
        (game, level)
        for game, reached in inventory.items()
        for level in range(1, reached + 1)
        if (game, level) not in excluded
    }
    payload = {
        "rows": [
            {
                "system": "GKM",
                "game": game,
                "completed_level": level,
                "source_checkpoint_exact": True,
            }
            for game, level in sorted(source_ids)
        ]
    }
    scope = reproduce_manuscript._source_audit_scope(
        {"inventory": inventory}, payload
    )
    assert scope["replay_verified_endpoint_wins"] == 19
    assert scope["admissible_exact_winning_source_checkpoints"] == 12
    assert scope["excluded_from_source_marginals"] == [
        {"game": game, "level": level}
        for game, level in reproduce_manuscript.FROZEN_SOURCE_AUDIT_EXCLUSIONS
    ]

    payload["rows"] = [
        row
        for row in payload["rows"]
        if not (row["game"] == "lf52" and row["completed_level"] == 10)
    ]
    with pytest.raises(RuntimeError, match="exclusions changed"):
        reproduce_manuscript._source_audit_scope(
            {"inventory": inventory}, payload
        )
