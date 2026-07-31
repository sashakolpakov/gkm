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


def play_level_4(env):
    move_until_level_progress(env, action=2, max_steps=2)
    move_until_level_progress(env, action=3, max_steps=2)
    select_grid_cells_of_color(
        env,
        xs=(30,),
        ys=(50, 60),
        color=2,
    )
    select_grid_cells_of_color(
        env,
        xs=(25, 35),
        ys=(55,),
        color=2,
    )
    move_until_level_progress(env, action=2, max_steps=1)
    move_until_level_progress(env, action=3, max_steps=1)
    select_grid_cells_of_color(
        env,
        xs=(30,),
        ys=(50, 55, 60),
        color=2,
    )
    move_until_level_progress(env, action=4, max_steps=9)
    select_grid_cells_of_color(
        env,
        xs=(30,),
        ys=(50, 60),
        color=2,
    )
    select_grid_cells_of_color(
        env,
        xs=(25, 35),
        ys=(55,),
        color=2,
    )
    move_until_level_progress(env, action=2, max_steps=2)
    move_until_level_progress(env, action=4, max_steps=3)


def play_level_5(env):
    # Activate the small central body, then hand off to the lower region.
    select_grid_cells_of_color(
        env,
        xs=(30,),
        ys=(50, 60),
        color=2,
    )
    select_grid_cells_of_color(
        env,
        xs=(25, 35),
        ys=(55,),
        color=2,
    )
    select_grid_cells_of_color(
        env,
        xs=(25, 30),
        ys=(50,),
        color=2,
    )
    select_grid_cells_of_color(
        env,
        xs=(30,),
        ys=(55,),
        color=2,
    )
    move_until_level_progress(env, action=3, max_steps=9)

    # Reach the lower-left switch and remove the first corridor barrier.
    move_until_level_progress(env, action=2, max_steps=2)
    move_until_level_progress(env, action=3, max_steps=1)
    select_grid_cells_of_color(
        env,
        xs=(30,),
        ys=(50, 55, 60),
        color=2,
    )

    # Reach the upper switch and remove the second corridor barrier.
    move_until_level_progress(env, action=4, max_steps=2)
    move_until_level_progress(env, action=1, max_steps=1)
    select_grid_cells_of_color(
        env,
        xs=(30,),
        ys=(50, 55, 60),
        color=2,
    )

    # Hand off to the now-unblocked right runner and enter the goal.
    select_grid_cells_of_color(
        env,
        xs=(30,),
        ys=(50, 60),
        color=2,
    )
    select_grid_cells_of_color(
        env,
        xs=(25, 35),
        ys=(55,),
        color=2,
    )
    select_grid_cells_of_color(
        env,
        xs=(25, 30),
        ys=(50,),
        color=2,
    )
    select_grid_cells_of_color(
        env,
        xs=(30,),
        ys=(55,),
        color=2,
    )
    move_until_level_progress(env, action=1, max_steps=6)


def play_level_6(env):
    # Select the plus runner, then hand off to the corner runner.
    select_grid_cells_of_color(
        env,
        xs=(30,),
        ys=(50, 60),
        color=2,
    )
    select_grid_cells_of_color(
        env,
        xs=(25, 35),
        ys=(55,),
        color=2,
    )
    select_grid_cells_of_color(
        env,
        xs=(25, 30),
        ys=(50,),
        color=2,
    )
    select_grid_cells_of_color(
        env,
        xs=(30,),
        ys=(55,),
        color=2,
    )
    move_until_level_progress(env, action=4, max_steps=2)
    move_until_level_progress(env, action=1, max_steps=2)

    # Switch through the plus and vertical runners to stage the left region.
    select_grid_cells_of_color(
        env,
        xs=(30,),
        ys=(50, 60),
        color=2,
    )
    select_grid_cells_of_color(
        env,
        xs=(25, 35),
        ys=(55,),
        color=2,
    )
    select_grid_cells_of_color(
        env,
        xs=(30,),
        ys=(50, 55, 60),
        color=2,
    )
    move_until_level_progress(env, action=3, max_steps=1)

    # Complete the handoffs that open the final upward route.
    select_grid_cells_of_color(
        env,
        xs=(25, 30),
        ys=(50,),
        color=2,
    )
    select_grid_cells_of_color(
        env,
        xs=(30,),
        ys=(55,),
        color=2,
    )
    select_grid_cells_of_color(
        env,
        xs=(30,),
        ys=(50, 55, 60),
        color=2,
    )
    select_grid_cells_of_color(
        env,
        xs=(25, 30),
        ys=(50,),
        color=2,
    )
    select_grid_cells_of_color(
        env,
        xs=(30,),
        ys=(55,),
        color=2,
    )
    move_until_level_progress(env, action=1, max_steps=1)
    move_until_level_progress(env, action=4, max_steps=1)
    move_until_level_progress(env, action=1, max_steps=5)
