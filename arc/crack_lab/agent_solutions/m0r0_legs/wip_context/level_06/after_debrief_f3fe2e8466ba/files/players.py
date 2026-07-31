# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa


def play_level_1(env):
    reunite_mirrored_pair(env, MIRRORED_PAIR_ASCENT)


def play_level_2(env):
    reunite_mirrored_pair(env, MIRRORED_PAIR_MAZE_REUNION)


def play_level_3(env):
    relocate_selectable_blockers(env, SELECTABLE_CORRIDOR_BLOCKER_CLEARANCE)
    reunite_mirrored_pair(env, SELECTABLE_PAIR_REUNION)
    assemble_smaller_agents(env)


def play_level_4(env):
    relocate_selectable_blockers(env, CENTRAL_BLOCKER_CLEARANCE)
    reunite_mirrored_pair(env, PARKED_BLOCKER_PAIR_REUNION)


def play_level_5(env):
    reunite_mirrored_pair(env, SWITCH_GATED_PAIR_REUNION)


def play_level_6(env):
    reunite_helper_pinned_pair(env, HELPER_PINNED_PAIR_REUNION)
