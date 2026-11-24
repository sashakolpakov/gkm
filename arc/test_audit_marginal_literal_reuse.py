from __future__ import annotations

import sys
import json
from pathlib import Path


ARC_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ARC_ROOT))

from audit_marginal_literal_reuse import (  # noqa: E402
    classify_baseline_command,
    conditional_ast_marginal,
    transition_fields,
    unchanged_called_definitions,
    command_reuse,
)
import audit_marginal_literal_reuse as audit  # noqa: E402


def test_conditional_ast_marginal_reuses_literal_nodes_and_charges_rewrite() -> None:
    previous = {
        "legs.py": b"def old_leg(x):\n    return x + 1\n",
        "players.py": b"from legs import old_leg\n",
    }
    unchanged_plus_binding = {
        **previous,
        "players.py": (
            previous["players.py"]
            + b"\ndef play_level_2(env):\n    return old_leg(env)\n"
        ),
    }
    marginal, reused, novel = conditional_ast_marginal(
        previous, unchanged_plus_binding
    )
    assert marginal > 0
    assert reused == 2
    assert novel == 1

    same_size_rewrite = {
        **previous,
        "legs.py": b"def old_leg(x):\n    return x - 1\n",
    }
    marginal, reused, novel = conditional_ast_marginal(
        previous, same_size_rewrite
    )
    assert marginal > 0
    assert reused == 1
    assert novel == 1


def test_literal_witness_requires_a_direct_winning_call() -> None:
    previous = {
        "legs.py": (
            b"def helper(x):\n    return x\n\n"
            b"def solve(x):\n    return helper(x)\n"
        ),
        "players.py": b"",
    }
    current = {
        **previous,
        "players.py": b"def play_level_2(env):\n    return solve(env)\n",
    }
    reused = unchanged_called_definitions(
        previous,
        current,
        "play_level_2",
        allowed_files={"legs.py"},
    )
    assert reused == ["legs.py:solve"]
    assert "legs.py:helper" not in reused


def test_sharp_drop_rule_is_half_or_more() -> None:
    assert transition_fields(50, 100) == (50, 0.5, True)
    assert transition_fields(51, 100) == (49, 0.51, False)
    assert transition_fields(50, None) == (None, None, False)


def test_baseline_python_command_uses_the_general_accounting_engine():
    previous = {"world_model_engine.py": b"def helper(x):\n    return x\n"}
    command = "python3 - <<'PY'\nfrom world_model_engine import helper as chosen\ndef glue(fn):\n    return fn(1)\nglue(chosen)\nPY"
    report = command_reuse(previous, previous, command)
    assert report["paths"]["world_model_engine.py:helper"] == ["__winning_command_0.py:<module>", "__winning_command_0.py:glue", "world_model_engine.py:helper"]


def test_importing_without_calling_a_helper_is_not_a_reachable_path():
    files = {"world_model_engine.py": b"def helper(x):\n    return x\n"}
    report = command_reuse(files, files, "python3 -c 'from world_model_engine import helper'")
    assert report["paths"] == {}


def test_general_accounting_reports_callbacks_in_opine_planners(tmp_path):
    engine = tmp_path / "engine.py"
    engine.write_text("def helper(state):\n    return []\n\ndef planner(state):\n    fn = helper\n    return fn(state)\n")
    audit_json = tmp_path / "audit.json"
    audit_json.write_text(json.dumps({"rows": [{"game": "demo", "completed_levels": n, "engine_path": "engine.py", "synth_planner_witness": True} for n in [1, 2]]}))
    row = audit.opine_rows(tmp_path, audit_json)[1]
    assert row.source_reuse["paths"]["game_engine.py:helper"]
    assert row.source_reuse["retained_entrypoint_targets"] == ["game_engine.py:planner"]


def test_missing_winning_source_is_not_bridged_by_marginal_or_reuse(tmp_path, monkeypatch):
    game = tmp_path / "demo_legs"
    game.mkdir()
    (game / "checkpoint.json").write_text(json.dumps({"reached": 5}))
    leg = b"def retained(env):\n    env.step(1)\n"

    def snapshot(_game, level):
        if level == 2:
            return None  # e.g. an unauthenticated paid-debrief reconstruction
        return {"phase": "reached_before_debrief"}, {
            "legs.py": leg,
            "players.py": f"def play_level_{level}(env):\n    retained(env)\n".encode(),
        }

    monkeypatch.setattr(audit, "exact_snapshot", snapshot)
    rows = {row.completed_level: row for row in audit.gkm_rows(tmp_path)}
    assert set(rows) == {1, 3, 4, 5}
    assert rows[3].exact_adjacent_transition is False
    assert rows[3].marginal_ast_zlib_bytes is None
    assert rows[3].hard_literal_reuse_witness is False
    assert rows[4].exact_adjacent_transition is True
    assert rows[4].marginal_ast_zlib_bytes is not None
    assert rows[4].previous_level_marginal_ast_zlib_bytes is None
    assert rows[4].sharp_marginal_drop is False
    assert rows[5].previous_level_marginal_ast_zlib_bytes is not None


def test_baseline_command_classification_keeps_literal_plans_separate() -> None:
    assert (
        classify_baseline_command(
            "python3 plan_executor.py ACTION1 ACTION2 ACTION3"
        )
        == "literal_action_program_via_executor"
    )
    assert (
        classify_baseline_command(
            "python3 - <<'PY'\nplan = ['ACTION1']; "
            "subprocess.run(['python3','client/client.py','move',*plan])\nPY"
        )
        == "inline_literal_action_program"
    )
    assert (
        classify_baseline_command(
            "python3 - <<'PY'\nfrom world_model_main_planner import planner\n"
            "print(planner(state))\nPY"
        )
        == "inline_world_model_program"
    )
