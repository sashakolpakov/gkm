from __future__ import annotations

from pathlib import Path

import pytest

from roboarm_game.source_guard import (
    PUBLIC_SOURCE_PATHS,
    SourceBoundaryError,
    list_public_sources,
    materialize_public_sources,
    read_public_source,
)


PRIVATE_NAMES = {
    "__init__.py",
    "environment.py",
    "render.py",
    "source_guard.py",
    "state.py",
}


def test_public_projection_contains_exact_allowlist(tmp_path: Path) -> None:
    outputs = materialize_public_sources(tmp_path / "solver-view", write_root=tmp_path)
    relative_outputs = {path.name for path in outputs}

    assert list_public_sources() == PUBLIC_SOURCE_PATHS
    assert relative_outputs == set(PUBLIC_SOURCE_PATHS)
    assert {path.name for path in (tmp_path / "solver-view").iterdir()} == set(
        PUBLIC_SOURCE_PATHS
    )
    assert not (relative_outputs & PRIVATE_NAMES)
    assert "ARC API" in read_public_source("README.md")
    assert "class Environment" in read_public_source("protocol.py")
    assert "class Action" in read_public_source("interface.py")


@pytest.mark.parametrize(
    "attack",
    [
        "../environment.py",
        "../../arc/README.md",
        "/etc/passwd",
        "environment.py",
        "README.md/../environment.py",
        "",
    ],
)
def test_read_guard_rejects_traversal_and_private_files(attack: str) -> None:
    with pytest.raises(SourceBoundaryError, match="not public"):
        read_public_source(attack)


def test_projection_rejects_destination_outside_write_root(tmp_path: Path) -> None:
    declared_root = tmp_path / "declared"
    declared_root.mkdir()
    with pytest.raises(SourceBoundaryError, match="escaped"):
        materialize_public_sources(
            tmp_path / "outside" / "solver-view",
            write_root=declared_root,
        )
    assert not (tmp_path / "outside").exists()


def test_projection_rejects_symlink_destination(tmp_path: Path) -> None:
    real_destination = tmp_path / "real-destination"
    real_destination.mkdir()
    linked_destination = tmp_path / "linked-destination"
    linked_destination.symlink_to(real_destination, target_is_directory=True)

    with pytest.raises(SourceBoundaryError, match="symlink"):
        materialize_public_sources(linked_destination, write_root=tmp_path)
    assert not tuple(real_destination.iterdir())


def test_projection_refuses_symlink_output(tmp_path: Path) -> None:
    destination = tmp_path / "solver-view"
    destination.mkdir()
    protected = tmp_path / "protected"
    protected.write_text("unchanged", encoding="utf-8")
    (destination / "protocol.py").symlink_to(protected)

    with pytest.raises(SourceBoundaryError, match="symlink"):
        materialize_public_sources(destination, write_root=tmp_path)
    assert protected.read_text(encoding="utf-8") == "unchanged"
