import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

import perception as P
import players


def objects(env):
    return [
        (blob.color, blob.bbox, blob.area)
        for blob in P.connected_components(env.frame(), colors=(9, 10, 12, 14), min_area=2)
    ]


def step(env, action):
    if isinstance(action, tuple):
        env.step(*action)
    else:
        env.step(action)


def run(env):
    for level in range(1, 6):
        getattr(players, f"play_level_{level}")(env)
    route = (
        1, 1, (6, 31, 43), 4, 4, 4,
        1, 1, 1, 1, 1, 1,
        (6, 19, 15),
        2, 2, 2, 2, 2, 2, 2, 2,
        3,
    )
    for index, action in enumerate(route):
        step(env, action)
        if index >= len(route) - 4:
            print(index, action, env.levels_completed, objects(env))


A.run_program("m0r0", run)
