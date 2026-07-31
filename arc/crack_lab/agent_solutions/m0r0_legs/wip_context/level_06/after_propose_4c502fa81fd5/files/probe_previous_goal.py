import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

import legs
import perception as P
import players


def compact(env):
    return [
        (blob.color, blob.bbox, blob.area)
        for blob in P.connected_components(env.frame(), colors=(1, 9, 10, 11), min_area=2)
    ]


def run(env):
    players.play_level_1(env)
    players.play_level_2(env)
    legs.relocate_selectable_blockers(env, legs.SELECTABLE_CORRIDOR_BLOCKER_CLEARANCE)
    legs.reunite_mirrored_pair(env, legs.SELECTABLE_PAIR_REUNION)
    route = legs._SMALLER_AGENT_ASSEMBLY
    for index, action in enumerate(route):
        if isinstance(action, tuple):
            env.step(*action)
        else:
            env.step(action)
        if index >= len(route) - 10 or env.levels_completed > 2:
            print(index, action, env.levels_completed, compact(env))
        if env.levels_completed > 2:
            break


A.run_program("m0r0", run)
