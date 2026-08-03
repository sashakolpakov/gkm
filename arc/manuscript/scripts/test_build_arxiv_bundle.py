from __future__ import annotations

import zipfile
from pathlib import Path

from arc.manuscript.scripts.build_arxiv_bundle import SOURCE_FILES, build_bundle


def test_bundle_is_minimal_and_reproducible(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"

    first_hash = build_bundle(root, first)
    second_hash = build_bundle(root, second)

    assert first_hash == second_hash
    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        assert archive.namelist() == [path.as_posix() for path in SOURCE_FILES]
        assert "arc_agi3.tex" in archive.namelist()
        assert all(not name.endswith((".aux", ".log", ".out", ".pdf")) for name in archive.namelist())
