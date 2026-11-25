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
