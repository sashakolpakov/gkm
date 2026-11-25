# Per-level players only compose reusable legs.
from legs import move, select_object


def play_level_1(env):
    move(env, 4, 3)
    select_object(env, 43, 31)
    move(env, 4)
    move(env, 1)
    select_object(env, 25, 31)
    move(env, 3, 4)
    move(env, 2)


def play_level_2(env):
    # Launch the square and tall shape into the left room.
    move(env, 4, 4)
    move(env, 1, 3)
    move(env, 3)
    select_object(env, 13, 44)
    move(env, 3)
    move(env, 1, 4)

    # Stage the square below the tall shape and launch it upward.
    select_object(env, 17, 47)
    move(env, 3, 3)
    move(env, 1, 4)
    select_object(env, 10, 18)
    move(env, 1, 2)

    # Fill the remaining matching outlines.
    select_object(env, 8, 38)
    move(env, 2, 2)
    select_object(env, 49, 46)
    move(env, 2, 2)
    move(env, 4)
    select_object(env, 41, 34)
    move(env, 4, 5)
    move(env, 2, 2)


def play_level_3(env):
    # Approach the upper cross through its corner, then launch it left twice.
    move(env, 4, 4)
    move(env, 2, 2)
    move(env, 3, 7)

    # Approach the lower cross from above-left and launch it right.
    move(env, 2, 3)
    move(env, 3, 7)
    move(env, 2, 2)
    move(env, 4)

    # Both crosses now match their outlines; fill the avatar's outline.
    move(env, 1, 2)
    move(env, 4, 5)
