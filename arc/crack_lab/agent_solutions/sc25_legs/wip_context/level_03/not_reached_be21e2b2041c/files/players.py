# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa


def play_level_1(env):
    probe_known_solution(env, movement_action=3, max_steps=16, prime=True)
    probe_known_click_variants(env, movement_action=3, max_steps=16, prime=True)
    prime_board(env)
    select_grid_cells_of_color(
        env,
        xs=(25, 30, 35),
        ys=(50, 55, 60),
        color=0,
    )
    move_until_level_progress(env, action=3, max_steps=16)


def play_level_2(env):
    probe_known_solution(env, movement_action=1, max_steps=8)
    probe_known_click_variants(env, movement_action=1, max_steps=8)
    select_grid_cells_of_color(
        env,
        xs=(25, 30, 35),
        ys=(50, 55, 60),
        color=0,
    )
    move_until_level_progress(env, action=1, max_steps=8)


def play_level_3(env):
    probe_level_3_observations(env)
