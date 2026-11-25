# Per-level players only compose reusable legs.
from legs import solve_peg_jumps, solve_peg_ferry


def play_level_1(env):
    solve_peg_jumps(env)


def play_level_2(env):
    solve_peg_ferry(env)
