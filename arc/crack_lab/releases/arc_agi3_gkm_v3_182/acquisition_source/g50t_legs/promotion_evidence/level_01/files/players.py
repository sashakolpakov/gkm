# Each player only composes reusable legs.
from legs import walk_straight, release_recorded_companion


def play_level_1(env):
    walk_straight(env, 4, 4)
    release_recorded_companion(env)
    walk_straight(env, 2, 7)
    walk_straight(env, 4, 5)
