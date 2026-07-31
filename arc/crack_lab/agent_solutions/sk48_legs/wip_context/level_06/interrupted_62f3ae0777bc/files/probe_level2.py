"""Compact clean-room observations for the pristine level-2 entry."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import assemble_telescoping_chain
from perception import action_deltas, color_counts, object_candidates


def summarize(env):
    assemble_telescoping_chain(env)
    print("level", int(env.levels_completed), "terminal", bool(env.terminal()))
    print("actions", tuple(env.actions))
    print("colors", color_counts(env.frame()))
    objects = object_candidates(env.frame(), min_area=4)
    print(
        "objects",
        [
            (o["color"], o["bbox"], o["area"], o["cell_sig"])
            for o in objects
        ],
    )
    print(
        "deltas",
        {
            action: {
                "count": delta["count"],
                "bbox": delta["bbox"],
                "samples": delta["samples"][:12],
            }
            for action, delta in action_deltas(env, env.actions).items()
        },
    )


arena.run_program("sk48", summarize)
