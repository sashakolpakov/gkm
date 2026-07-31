from __future__ import annotations

import json
from pathlib import Path

import pytest

import generate_empirical_tables as G


def _write_game(root: Path, game: str, values: list[int], *, total: int) -> None:
    game_dir = root / f"{game}_legs"
    game_dir.mkdir(parents=True)
    (game_dir / "checkpoint.json").write_text(
        json.dumps(
            {
                "game": game,
                "reached": len(values),
                "total_marginal_C": sum(values),
                "records": [
                    {"level": level, "marginal_C": value, "reached": True}
                    for level, value in enumerate(values, start=1)
                ],
                "final_path": [1] * total,
                "validated": True,
            }
        )
    )
    (game_dir / "legs.py").write_text("def leg(env):\n    return env\n")
    (game_dir / "players.py").write_text("def play_level_1(env):\n    leg(env)\n")


def test_sign_reversal_metric_selects_full_alternation() -> None:
    assert G._sign_reversals([36, 95, 77, 377, 54, 77, 2, 292, 203]) == 7
    assert G._sign_reversals([40, 54, 86, 114, 138, 170, 158]) == 1


def test_load_game_checks_and_measures_checkpoint(tmp_path: Path) -> None:
    _write_game(tmp_path, "ar25", [9, 4], total=3)
    row = G.load_game(tmp_path, "ar25")
    assert row.marginals == (9, 4)
    assert row.actions == 3
    assert row.total_levels == 8
    assert row.direction_reversals == 0


def test_load_game_rejects_nonconsecutive_records(tmp_path: Path) -> None:
    _write_game(tmp_path, "ar25", [9, 4], total=3)
    path = tmp_path / "ar25_legs" / "checkpoint.json"
    payload = json.loads(path.read_text())
    payload["records"][1]["level"] = 3
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="nonconsecutive"):
        G.load_game(tmp_path, "ar25")


def test_renderers_mark_pending_and_nonexistent_levels(tmp_path: Path) -> None:
    _write_game(tmp_path, "lf52", [116, 119], total=4)
    row = G.load_game(tmp_path, "lf52")
    markdown = G.render_markdown((row,))
    tex = G.render_tex((row,))
    rst = G.render_rst((row,))
    assert "pending" in markdown
    assert "--" in markdown
    assert r"\mathrm{n/a}" in tex
    assert "n/a" in rst


def test_authoritative_inventory_is_exact() -> None:
    assert len(G.AUTHORITATIVE_LEVELS) == 25
    assert sum(G.AUTHORITATIVE_LEVELS.values()) == 183
