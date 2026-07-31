# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa


def play_level_1(env):
    prime_board(env)
    select_grid_cells_of_color(
        env,
        xs=(25, 30, 35),
        ys=(50, 55, 60),
        color=0,
    )
    move_until_level_progress(env, action=3, max_steps=16)


def play_level_2(env):
    select_grid_cells_of_color(
        env,
        xs=(25, 30, 35),
        ys=(50, 55, 60),
        color=0,
    )
    move_until_level_progress(env, action=1, max_steps=8)


def play_level_3(env):
    move_until_level_progress(env, action=4, max_steps=3)
    select_grid_cells_of_color(
        env,
        xs=(25, 30, 35),
        ys=(50, 55, 60),
        color=0,
    )
    move_until_level_progress(env, action=3, max_steps=4)
    move_until_level_progress(env, action=2, max_steps=4)
    move_until_level_progress(env, action=3, max_steps=2)
