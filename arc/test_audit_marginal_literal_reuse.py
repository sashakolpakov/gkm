from __future__ import annotations

import sys
from pathlib import Path


ARC_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ARC_ROOT))

from audit_marginal_literal_reuse import (  # noqa: E402
    classify_baseline_command,
    conditional_ast_marginal,
    transition_fields,
    unchanged_called_definitions,
)


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
