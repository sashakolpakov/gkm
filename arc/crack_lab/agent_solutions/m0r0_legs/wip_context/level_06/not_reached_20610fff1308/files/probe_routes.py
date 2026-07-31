import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

import perception as P
import players


def objects(frame):
    return [
        (o["color"], o["bbox"], o["area"])
        for o in P.object_candidates(frame, min_area=4)
        if o["color"] not in (5, 6, 7, 8)
    ]


def run(env):
    for level in range(1, 6):
        getattr(players, f"play_level_{level}")(env)
    routes = (
        (1,),
        (1, 1),
        (1, 5),
        (1, 5, 1),
        (2,),
        (2, 2),
        (2, 5),
        (2, 5, 2),
        (3, 2, 2),
        (4, 2, 2),
    )
    base = P.arr(env.frame()).copy()
    for route in routes:
        node = P.replay(env, route)
        d = P.frame_delta(base, node.frame())
        print("ROUTE", route, "LEVELS", node.levels_completed, "DELTA", (d["count"], d["bbox"]), "OBJECTS", objects(node.frame()))
    main_route = (1, 1) + (2,) * 7 + (4,) * 4
    staged_routes = (
        main_route,
        ((6, 31, 43), 4, (6, 19, 23)) + main_route,
        ((6, 31, 43), 4, 4, (6, 19, 23)) + main_route,
        ((6, 31, 43), 4, 4, 4, (6, 19, 23)) + main_route,
    )
    for route in staged_routes:
        node = env.clone()
        for action in route:
            if node.terminal():
                break
            if isinstance(action, tuple):
                node.step(*action)
            else:
                node.step(action)
        print("FULL_ROUTE", route[:4], "LEN", len(route), "LEVELS", node.levels_completed, "OBJECTS", objects(node.frame()))


A.run_program("m0r0", run)
