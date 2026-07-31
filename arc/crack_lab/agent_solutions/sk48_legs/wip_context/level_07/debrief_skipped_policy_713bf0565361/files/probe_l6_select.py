import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import players
from perception import connected_components, frame_delta


def reach_level_6(env):
    for level in range(1, 6):
        getattr(players, f"play_level_{level}")(env)
        assert env.levels_completed == level


def positions(env, colors=(6, 8, 9, 15)):
    return tuple(
        (blob.color, round(blob.centroid[0]), round(blob.centroid[1]))
        for blob in connected_components(
            env.frame(), colors=colors, min_area=12
        )
        if blob.centroid[0] < 53
    )


def probe(env):
    reach_level_6(env)
    base = env.frame()
    clicks = {
        "left": (8, 28),
        "top": (32, 5),
        "eight": (32, 34),
        "nine": (38, 28),
        "empty": (20, 20),
        "left_tether": (14, 28),
        "top_tether": (32, 10),
    }
    for name, (x, y) in clicks.items():
        for action in (1, 2, 3, 4):
            node = env.clone()
            node.step(6, x, y)
            node.step(action)
            delta = frame_delta(base, node.frame())
            print(
                name, action,
                "D", (delta["count"], delta["bbox"]),
                "P", positions(node),
            )


A.run_program("sk48", probe)
