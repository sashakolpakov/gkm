from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from arc.crack_lab import verify_frozen_release as V


RECEIPT_NAME = (
    "140e37ca7014d5aa6a48a3808fd94e90209c56499dbcd7df9f0fe733a29a7681.json"
)
SOURCE_REVISION = "c1f8168f230732f2d745c234555b3e3dfcb8aefa"


def test_checked_in_v2_receipt_verifies_in_its_bound_revision() -> None:
    repo = Path(__file__).resolve().parents[2]
    available = subprocess.run(
        ["git", "-C", str(repo), "cat-file", "-e", f"{SOURCE_REVISION}^{{commit}}"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if available.returncode != 0:
        pytest.skip("receipt-bound Git revision is unavailable in this checkout")
    release = repo / "arc/crack_lab/releases/arc_agi3_gkm_v2_181"
    result = V.verify_frozen_release(
        receipt_path=release / "receipts" / RECEIPT_NAME,
        canonical_root=release / "artifacts",
        repo_root=repo,
    )
    assert result["status"] == "PASS"
    assert result["claimed_levels"] == 181
    assert result["authoritative_levels"] == 183
    assert result["receipt_sha256"] == RECEIPT_NAME.removesuffix(".json")
    assert result["verification_context_source_revision"] == SOURCE_REVISION
    assert result["unclaimed_boundaries"] == [
        {"game": "lf52", "level": 9},
        {"game": "lf52", "level": 10},
    ]


def test_receipt_filename_must_be_its_content_hash(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}\n")
    with pytest.raises(V.FrozenReleaseError, match="content hash"):
        V.load_receipt(receipt)


def test_checked_in_receipt_needs_no_private_game_source() -> None:
    repo = Path(__file__).resolve().parents[2]
    receipt = (
        repo
        / "arc/crack_lab/releases/arc_agi3_gkm_v2_181/receipts"
        / RECEIPT_NAME
    )
    _, expected = V.load_receipt(receipt)
    environment_paths = sorted(
        path for path in expected if path.startswith("environment_files/")
    )
    assert len(environment_paths) == 25
    assert all(path.endswith("/metadata.json") for path in environment_paths)


@pytest.mark.parametrize("value", ("../secret", "/absolute", "a/./b", "a//b"))
def test_receipt_paths_are_strictly_relative(value: str) -> None:
    with pytest.raises(V.FrozenReleaseError, match="unsafe path"):
        V._safe_relative_path(value, label="test")


def test_supplied_verifier_is_copied_as_an_exact_allowlist(
    tmp_path: Path,
) -> None:
    supplied = tmp_path / "supplied"
    bound = supplied / "arc/crack_lab/gate.py"
    bound.parent.mkdir(parents=True)
    raw = b"print('bound')\n"
    bound.write_bytes(raw)
    (supplied / "sitecustomize.py").write_text(
        "raise RuntimeError('must never enter the execution tree')\n"
    )

    copied = V._copy_bound_root(
        supplied,
        {"arc/crack_lab/gate.py": V._sha256_bytes(raw)},
        tmp_path / "private",
    )

    assert (copied / "arc/crack_lab/gate.py").read_bytes() == raw
    assert not (copied / "sitecustomize.py").exists()
    assert sorted(
        path.relative_to(copied).as_posix()
        for path in copied.rglob("*")
        if path.is_file()
    ) == ["arc/crack_lab/gate.py"]


def test_supplied_verifier_rejects_symlinked_parent_components(
    tmp_path: Path,
) -> None:
    supplied = tmp_path / "supplied"
    supplied.mkdir()
    outside = tmp_path / "outside"
    bound = outside / "crack_lab/gate.py"
    bound.parent.mkdir(parents=True)
    raw = b"print('outside')\n"
    bound.write_bytes(raw)
    (supplied / "arc").symlink_to(outside, target_is_directory=True)

    with pytest.raises(V.FrozenReleaseError, match="path component"):
        V._copy_bound_root(
            supplied,
            {"arc/crack_lab/gate.py": V._sha256_bytes(raw)},
            tmp_path / "private",
        )
