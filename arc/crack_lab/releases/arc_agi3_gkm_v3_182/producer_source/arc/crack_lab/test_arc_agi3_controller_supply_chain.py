from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path

import pytest

import arc_agi3_controller_supply_chain as S


def _fixture_paths(root: Path) -> tuple[tuple[Path, bool], ...]:
    rows: list[tuple[Path, bool]] = []
    for index, (_path, executable) in enumerate(S.OBSERVED_PATHS):
        path = root / f"file-{index}"
        path.write_bytes(f"controller-file-{index}\n".encode("ascii"))
        path.chmod(0o555 if executable else 0o444)
        rows.append((path, executable))
    return tuple(rows)


def test_controller_supply_chain_manifest_and_recipe_are_exact_and_exclusive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    paths = _fixture_paths(tmp_path)
    monkeypatch.setattr(S, "OBSERVED_PATHS", paths)
    value = S.build_manifest(codex_cli_version="codex-cli 0.145.0")
    assert value == {
        "schema": 1,
        "kind": "arc_agi3_controller_supply_chain",
        "codex_cli_version": "codex-cli 0.145.0",
        "files": [
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
                "executable": executable,
            }
            for path, executable in paths
        ],
    }
    output = tmp_path / "supply-chain.json"
    digest = S.write_new_manifest(output, value)
    expected = S._canonical_json(value) + b"\n"
    assert output.read_bytes() == expected
    assert digest == hashlib.sha256(expected).hexdigest()
    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    with pytest.raises(FileExistsError):
        S.write_new_manifest(output, value)
    recipe = (
        Path(__file__).parent
        / "container"
        / "Containerfile.arc-agi3-controller"
    ).read_text(encoding="utf-8")
    generator = (
        Path(__file__).parent
        / "arc_agi3_controller_supply_chain.py"
    )
    assert "python3 - <<'PY'" not in recipe
    assert "arc_agi3_controller_supply_chain.py" in recipe
    assert hashlib.sha256(generator.read_bytes()).hexdigest() in recipe
    assert "sha256sum --check --strict" in recipe
    assert "--codex-cli-version" in recipe


def test_controller_supply_chain_rejects_alias_and_mode_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    paths = list(_fixture_paths(tmp_path))
    target, executable = paths[0]
    alias = tmp_path / "alias"
    alias.symlink_to(target)
    paths[0] = (alias, executable)
    monkeypatch.setattr(S, "OBSERVED_PATHS", tuple(paths))
    with pytest.raises(S.SupplyChainError, match="unsafe"):
        S.build_manifest(codex_cli_version="codex-cli 0.145.0")

    paths[0] = (target, executable)
    os.chmod(target, 0o444)
    monkeypatch.setattr(S, "OBSERVED_PATHS", tuple(paths))
    with pytest.raises(S.SupplyChainError, match="unsafe"):
        S.build_manifest(codex_cli_version="codex-cli 0.145.0")
    with pytest.raises(S.SupplyChainError, match="version"):
        S.build_manifest(codex_cli_version="0.145.0\nforged")
