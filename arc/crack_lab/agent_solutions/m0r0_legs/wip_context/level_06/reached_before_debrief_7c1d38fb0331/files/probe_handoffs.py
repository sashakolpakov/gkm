import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

import perception as P
import players


def compact(env):
    wanted = (1, 9, 10, 11, 12, 14)
    objects = [
        (blob.color, blob.bbox, blob.area)
        for blob in P.connected_components(env.frame(), colors=wanted, min_area=2)
    ]
    frame = P.arr(env.frame())
    center = {
        "wall8": int((frame[40:48, 28:36] == 8).sum()),
        "empty5": int((frame[40:48, 28:36] == 5).sum()),
    }
    return objects, center


def act(env, action):
    if isinstance(action, tuple):
        env.step(*action)
    else:
        env.step(action)


def run(env):
    for level in range(1, 6):
        getattr(players, f"play_level_{level}")(env)

    route = [
        ("top", 1), ("top", 1),
        ("tiny", (6, 31, 43)),
        ("tiny-left", 3), ("tiny-left", 3), ("tiny-up", 1),
        ("tiny-left", 3),
        ("tiny-climb", 1), ("tiny-climb", 1), ("tiny-climb", 1),
        ("tiny-climb", 1), ("tiny-climb", 1),
        ("pair", (6, 19, 15)),
        ("right-descends", 2), ("right-descends", 2),
        ("right-descends", 2), ("right-descends", 2),
        ("right-descends", 2), ("right-descends", 2),
        ("right-descends", 2),
        ("cross", 4), ("cross", 4), ("cross", 4), ("cross", 4),
        ("cross", 4), ("cross", 4), ("cross", 4), ("align", 3),
        ("tiny-again", (6, 19, 19)),
        ("tiny-down", 2), ("tiny-down", 2), ("tiny-down", 2),
        ("tiny-down", 2), ("tiny-down", 2),
        ("tiny-around", 4), ("tiny-around", 2), ("tiny-around", 2),
        ("tiny-below", 3),
        ("pair-again", (6, 23, 15)),
        ("left-descends", 2), ("left-descends", 2),
        ("left-descends", 2), ("left-descends", 2),
        ("left-descends", 2), ("left-descends", 2),
        ("left-descends", 2),
        ("tiny-final", (6, 19, 47)),
        ("tiny-right", 4), ("tiny-right", 4), ("tiny-up", 1),
        ("pair-final", (6, 23, 43)),
        ("stage-reunion", 4), ("reunite", 3),
    ]
    print("START", compact(env))
    for label, action in route:
        act(env, action)
        print(label, action, env.levels_completed, compact(env))
        if env.terminal():
            break


A.run_program("m0r0", run)
