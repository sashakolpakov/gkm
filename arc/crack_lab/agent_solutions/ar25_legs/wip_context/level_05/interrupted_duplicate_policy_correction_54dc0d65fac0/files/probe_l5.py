import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import players
from perception import action_deltas, color_counts, connected_components


def compact_blobs(frame):
    return [
        (b.color, b.bbox, b.area)
        for b in connected_components(frame, min_area=4)
    ]


def probe(env):
    for level in range(1, 5):
        getattr(players, f"play_level_{level}")(env)
    print("level", env.levels_completed + 1, "actions", env.actions)
    print("counts", color_counts(env.frame()))
    print("blobs", compact_blobs(env.frame()))
    print(
        "deltas",
        {
            action: (delta["count"], delta["bbox"])
            for action, delta in action_deltas(env, env.actions).items()
        },
    )


arena.run_program("ar25", probe)
