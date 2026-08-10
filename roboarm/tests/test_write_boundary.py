from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_project_tree_contains_no_symlink_escape() -> None:
    escapes: list[str] = []
    for path in PROJECT_ROOT.rglob("*"):
        if path.is_relative_to(PROJECT_ROOT / ".venv"):
            # A standard venv intentionally links its interpreter to the
            # system installation; it does not write through that link.
            continue
        if path.is_symlink() and not path.resolve(strict=False).is_relative_to(PROJECT_ROOT):
            escapes.append(str(path.relative_to(PROJECT_ROOT)))
    assert escapes == []


def test_all_configured_output_paths_are_project_local() -> None:
    configured = (
        PROJECT_ROOT / "artifacts",
        PROJECT_ROOT / "artifacts" / "pytest-cache",
        PROJECT_ROOT / "artifacts" / "pytest-tmp",
        PROJECT_ROOT / "artifacts" / "pycache",
        PROJECT_ROOT / "artifacts" / "tmp",
        PROJECT_ROOT / "artifacts" / "xdg-cache",
        PROJECT_ROOT / "artifacts" / "public-source-projection",
        PROJECT_ROOT / "artifacts" / "write-audit.json",
    )
    assert all(path.resolve(strict=False).is_relative_to(PROJECT_ROOT) for path in configured)
