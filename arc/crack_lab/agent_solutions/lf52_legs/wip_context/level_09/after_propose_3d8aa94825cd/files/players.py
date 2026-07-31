# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import (
    solve_grid_wrapped_bridge_carrier_peg_solitaire,
    solve_multi_bridge_wrapped_carrier_peg_solitaire,
    solve_parallel_wrapped_bridge_carrier_peg_solitaire,
    solve_bridge_carrier_peg_solitaire,
    solve_peg_solitaire,
    solve_peg_solitaire_with_carrier,
    solve_wrapped_bridge_carrier_peg_solitaire,
)


def play_level_1(env):
    solve_peg_solitaire(env)


def play_level_2(env):
    solve_peg_solitaire_with_carrier(env)


def play_level_3(env):
    solve_peg_solitaire_with_carrier(env)


def play_level_4(env):
    solve_bridge_carrier_peg_solitaire(env)


def play_level_5(env):
    solve_bridge_carrier_peg_solitaire(env, max_align_states=650)


def play_level_6(env):
    solve_wrapped_bridge_carrier_peg_solitaire(env)


def play_level_7(env):
    solve_parallel_wrapped_bridge_carrier_peg_solitaire(env)


def play_level_8(env):
    solve_grid_wrapped_bridge_carrier_peg_solitaire(env)


def play_level_9(env):
    solve_multi_bridge_wrapped_carrier_peg_solitaire(env)
