# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa


def play_level_1(env):
    reunite_mirrored_pair(env, MIRRORED_PAIR_ASCENT)


def play_level_2(env):
    reunite_mirrored_pair(env, MIRRORED_PAIR_MAZE_REUNION)


def play_level_3(env):
    clear_selectable_corridor_blockers(env)
    reunite_mirrored_pair(env, SELECTABLE_PAIR_REUNION)
    assemble_smaller_agents(env)
