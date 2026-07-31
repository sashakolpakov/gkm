"""Verify action-6 selection and movement for each level-2 figure."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
from players import play_level_1


POINTS = ((12, 12), (45, 15), (18, 39), (51, 51))


def objects(frame):
    return [
        (b.color, b.bbox, b.area)
        for b in perception.connected_components(
            frame, colors=(0, 9, 11, 14), min_area=4
        )
    ]


def probe(env):
    play_level_1(env)
    for point in POINTS:
        selected = env.clone()
        selected.step(6, *point)
        print(
            "select", point,
            "counts", perception.color_counts(selected.frame()),
            "objects", objects(selected.frame()),
        )
        print(
            "moves",
            {
                action: (
                    delta["count"], delta["bbox"],
                )
                for action, delta in perception.action_deltas(
                    selected, actions=(1, 2, 3, 4, 5)
                ).items()
            },
        )


arena.run_program("cn04", probe)
