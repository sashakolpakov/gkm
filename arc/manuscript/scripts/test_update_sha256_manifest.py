from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import update_sha256_manifest as M


def test_manifest_renderer_is_ordered_and_exact(tmp_path: Path) -> None:
    base = tmp_path / "arc/manuscript"
    base.mkdir(parents=True)
    (base / "a.txt").write_bytes(b"alpha\n")
    (base / "b.txt").write_bytes(b"beta\n")
    rendered = M.render_manifest(base, ("b.txt", "a.txt")).decode()
    assert rendered == (
        f"{hashlib.sha256(b'beta\n').hexdigest()}  b.txt\n"
        f"{hashlib.sha256(b'alpha\n').hexdigest()}  a.txt\n"
    )


def test_manifest_renderer_rejects_symlinked_inputs(tmp_path: Path) -> None:
    base = tmp_path / "arc/manuscript"
    base.mkdir(parents=True)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside\n")
    (base / "linked.txt").symlink_to(outside)
    with pytest.raises(M.ManifestError):
        M.render_manifest(base, ("linked.txt",))
