# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa


def play_level_1(env):
    solve_peg_solitaire(env)


def play_level_2(env):
    solve_peg_solitaire_with_carrier(env)


def play_level_3(env):
    solve_peg_solitaire_with_carrier(env)


def play_level_4(env):
    solve_bridge_carrier_peg_solitaire(env)


def play_level_5(env):
    solve_bridge_carrier_peg_solitaire(env, max_search_states=300)
