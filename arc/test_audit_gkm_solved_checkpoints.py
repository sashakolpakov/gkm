import json

import audit_gkm_solved_checkpoints as Audit


def test_orphan_recovery_snapshot_is_exact_winning_source(tmp_path):
    game = tmp_path / "demo_legs"
    attempt = (
        game
        / "wip_context"
        / "level_01"
        / "recovered_existing_workspace_solver_deadbeef"
    )
    files = attempt / "files"
    files.mkdir(parents=True)
    for name, source in {
        "legs.py": "def leg(env):\n    env.step(1)\n",
        "players.py": "def play_level_1(env):\n    leg(env)\n",
        "solve.py": "def solve(env):\n    return None\n",
    }.items():
        (files / name).write_text(source)
    (attempt / "metadata.json").write_text(
        json.dumps(
            {
                "phase": "recovered_existing_workspace_solver",
                "reached": 1,
                "attempt": attempt.name,
                "created_at": "2026-07-28T00:00:00+00:00",
            }
        )
    )
    (game / "checkpoint.json").write_text(
        json.dumps(
            {
                "reached": 1,
                "records": [
                    {"level": 1, "marginal_C": 7, "reached": True}
                ],
            }
        )
    )

    rows, gaps = Audit.analyse_game(game)

    assert gaps == []
    assert len(rows) == 1
    assert rows[0].phase == "recovered_existing_workspace_solver"


def test_skipped_debrief_auto_solve_is_exactly_reconstructed(tmp_path):
    game = tmp_path / "demo_legs"
    exact_parent = (
        game / "wip_context" / "level_01" / "reached_before_debrief_parent"
    )
    exact_parent_files = exact_parent / "files"
    exact_parent_files.mkdir(parents=True)
    for name, source in {
        "legs.py": "def reused_leg(env):\n    env.step(1)\n",
        "players.py": "def play_level_1(env):\n    reused_leg(env)\n",
        "solve.py": "def solve(env):\n    return None\n",
    }.items():
        (exact_parent_files / name).write_text(source)
    (exact_parent / "metadata.json").write_text(
        json.dumps(
            {
                "phase": "reached_before_debrief",
                "reached": 1,
                "attempt": exact_parent.name,
                "created_at": "2026-07-27T23:59:00+00:00",
            }
        )
    )
    parent = (
        game / "wip_context" / "level_01" / "debrief_skipped_parent"
    )
    parent_files = parent / "files"
    parent_files.mkdir(parents=True)
    for name, source in {
        "legs.py": "def reused_leg(env):\n    env.step(1)\n",
        "players.py": "def play_level_1(env):\n    reused_leg(env)\n",
        "solve.py": "def solve(env):\n    return None\n",
    }.items():
        (parent_files / name).write_text(source)
    (parent / "metadata.json").write_text(
        json.dumps(
            {
                "phase": "debrief_skipped_policy",
                "reached": 1,
                "attempt": parent.name,
                "created_at": "2026-07-28T00:00:00+00:00",
            }
        )
    )

    win = (
        game / "wip_context" / "level_02"
        / "auto_solve_debrief_skipped_win"
    )
    win_files = win / "files"
    win_files.mkdir(parents=True)
    for name, source in {
        "legs.py": "def reused_leg(env):\n    env.step(1)\n",
        "players.py": (
            "def play_level_1(env):\n    reused_leg(env)\n\n"
            "def play_level_2(env):\n    reused_leg(env)\n"
        ),
        "solve.py": "def solve(env):\n    return None\n",
    }.items():
        (win_files / name).write_text(source)
    (win / "metadata.json").write_text(
        json.dumps(
            {
                "phase": "auto_solve_debrief_skipped",
                "reached": 2,
                "attempt": win.name,
                "created_at": "2026-07-28T00:01:00+00:00",
            }
        )
    )
    (game / "checkpoint.json").write_text(
        json.dumps(
            {
                "reached": 2,
                "records": [
                    {"level": 1, "marginal_C": 7, "reached": True},
                    {"level": 2, "marginal_C": 3, "reached": True},
                ],
            }
        )
    )

    rows, gaps = Audit.analyse_game(game)

    assert gaps == []
    assert [row.phase for row in rows] == [
        "reached_before_debrief",
        "reconstructed_auto_solve_boundary",
    ]
    assert rows[1].levels_spanned == 1
