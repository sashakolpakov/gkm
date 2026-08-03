from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import reproduce_manuscript


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

    def fake_run_json(command: list[str], *, cwd: Path) -> dict:
        calls.append((command, cwd))
        return {
            "status": "PASS",
            "claimed_levels": 181,
            "authoritative_levels": 183,
        }

    monkeypatch.setattr(reproduce_manuscript, "_run_json", fake_run_json)
    repo = tmp_path / "repo"
    verifier = tmp_path / "verifier"
    result = reproduce_manuscript._verify_frozen_release(
        repo=repo,
        release_root=tmp_path / "artifacts",
        release_receipt=tmp_path / "receipt.json",
        verifier_root=verifier,
    )
    assert result["status"] == "PASS"
    command, cwd = calls[0]
    assert cwd == repo
    assert command[:2] == [
        sys.executable,
        "arc/crack_lab/verify_frozen_release.py",
    ]
    assert command[-2:] == ["--verifier-root", str(verifier.resolve())]


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
