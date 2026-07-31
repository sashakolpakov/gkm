# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa


def play_level_1(env):
    # Push the initially selected ring through the central barrier.
    move_steps(env, RIGHT, 3)

    # Select the transferred ring and center it on the right target.
    select_at(env, 43, 31)
    move_steps(env, UP, 1)
    move_steps(env, RIGHT, 1)

    # Reselect the left ring and center it on the left target.
    select_at(env, 25, 31)
    move_steps(env, LEFT, 4)
    move_steps(env, DOWN, 1)


def play_level_2(env):
    # Stage the vertical piece against the lower horizontal gap.
    select_at(env, 34, 44)
    move_steps(env, LEFT, 1)

    # Use the square as a pusher to transfer the vertical piece left.
    select_at(env, 44, 47)
    move_steps(env, LEFT, 4)

    # Align the vertical piece below the upper-left room.
    select_at(env, 16, 44)
    move_steps(env, LEFT, 2)
    move_steps(env, UP, 4)

    # Transfer the square left with the horizontal piece.
    select_at(env, 41, 34)
    move_steps(env, DOWN, 4)
    move_steps(env, LEFT, 1)

    # The square can now push the vertical piece into the upper room.
    select_at(env, 18, 47)
    move_steps(env, LEFT, 2)
    move_steps(env, UP, 4)

    # Seat the vertical and square pieces in their left-side targets.
    select_at(env, 10, 20)
    move_steps(env, UP, 2)

    select_at(env, 10, 38)
    move_steps(env, DOWN, 2)
    move_steps(env, LEFT, 1)

    # Seat the two pieces whose targets remain in the lower-right room.
    select_at(env, 41, 46)
    move_steps(env, UP, 2)
    move_steps(env, RIGHT, 5)

    select_at(env, 37, 55)
    move_steps(env, UP, 1)
    move_steps(env, RIGHT, 5)


def play_level_3(env):
    # Drive the small ring around the right large ring and send it down a lane.
    move_steps(env, DOWN, 1)
    move_steps(env, RIGHT, 4)
    move_steps(env, DOWN, 1)
    move_steps(env, LEFT, 1)
    move_steps(env, DOWN, 2)

    # Transfer the two large rings leftward in turn.
    move_steps(env, LEFT, 6)
    move_steps(env, DOWN, 2)
    move_steps(env, LEFT, 7)

    # Finish the lower transfer, then seat the small ring.
    move_steps(env, DOWN, 1)
    move_steps(env, RIGHT, 1)
    move_steps(env, UP, 2)
    move_steps(env, RIGHT, 5)


def play_level_4(env):
    # Send the lower small ring through the barrier.
    move_steps(env, RIGHT, 3)
    move_steps(env, DOWN, 1)

    # Interlock that ring with the large ring and lift it into the upper room.
    select_at(env, 30, 48)
    move_steps(env, DOWN, 3)
    move_steps(env, LEFT, 1)
    move_steps(env, UP, 1)

    # Use the upper ring to align and seat the transferred large ring.
    select_at(env, 30, 30)
    move_steps(env, UP, 1)
    move_steps(env, LEFT, 3)
    move_steps(env, DOWN, 1)
    move_steps(env, RIGHT, 1)

    move_steps(env, DOWN, 2)
    move_steps(env, RIGHT, 5)
    move_steps(env, UP, 7)

    # Seat the upper small ring, then finish with the lower one.
    move_steps(env, DOWN, 3)
    move_steps(env, RIGHT, 5)

    select_at(env, 27, 57)
    move_steps(env, UP, 1)
    move_steps(env, RIGHT, 4)


def play_level_5(env):
    # Climb until the cycling corridor agents hand the ring through the barrier.
    move_steps(env, UP, 12)

    # Traverse above the vertical wall and seat the ring in the right target.
    move_steps(env, RIGHT, 7)
    move_steps(env, DOWN, 1)


def play_level_6(env):
    # Use the three cycling corridor agents to hand both rings across the
    # sealed barriers, then center them on their differently sized targets.
    move_steps(env, UP, 1)
    move_steps(env, LEFT, 2)
    move_steps(env, UP, 2)
    move_steps(env, LEFT, 7)
    move_steps(env, UP, 2)
    move_steps(env, LEFT, 3)
    move_steps(env, UP, 1)
    move_steps(env, LEFT, 2)
    move_steps(env, DOWN, 1)
    move_steps(env, LEFT, 2)
    move_steps(env, UP, 1)
    move_steps(env, RIGHT, 7)
    move_steps(env, DOWN, 2)
    move_steps(env, LEFT, 2)
    move_steps(env, DOWN, 1)
    move_steps(env, RIGHT, 1)
    move_steps(env, UP, 3)
    move_steps(env, RIGHT, 7)
    move_steps(env, UP, 3)
    move_steps(env, RIGHT, 1)
