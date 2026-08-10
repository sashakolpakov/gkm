from __future__ import annotations

from roboarm_game.gkm.lineage import (
    conditional_ast_marginal,
    historical_description_complexity,
    historical_marginal_complexity,
    unchanged_called_legs,
)


def test_historical_marginal_is_positive_net_growth_per_source_file():
    previous = {
        "legs.py": "VALUES = [1, 2]\n",
        "players.py": "def propose_level_1(evidence):\n    return []\n",
    }
    current = {
        "legs.py": "VALUES = [1, 2, 3]\nEXTRA = 1\n",
        "players.py": "def propose_level_1(evidence): return []\n",
    }

    expected = max(
        0,
        historical_description_complexity(current["legs.py"])
        - historical_description_complexity(previous["legs.py"]),
    )
    assert historical_marginal_complexity(previous, current) == expected


def test_conditional_ast_marginal_charges_a_same_size_rewrite():
    previous = {"legs.py": "def leg():\n    return 1\n"}
    current = {"legs.py": "def leg():\n    return 2\n"}

    marginal, reused, novel = conditional_ast_marginal(previous, current)

    assert marginal > 0
    assert reused == 0
    assert novel == 1


def test_unchanged_leg_reuse_distinguishes_direct_from_transitive_calls():
    previous = {
        "legs.py": "def retained_leg(evidence):\n    return [evidence]\n",
        "players.py": "def propose_level_1(evidence):\n    return []\n",
    }
    current = {
        "legs.py": (
            "def retained_leg(evidence):\n"
            "    return [evidence]\n\n"
            "def new_composition(evidence):\n"
            "    return retained_leg(evidence)\n"
        ),
        "players.py": (
            "def propose_level_1(evidence):\n"
            "    return new_composition(evidence)\n"
        ),
    }

    direct, transitive, hashes = unchanged_called_legs(previous, current)

    assert direct == []
    assert transitive == ["legs.py:retained_leg"]
    assert set(hashes) == {"legs.py:retained_leg"}
