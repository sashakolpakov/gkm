"""Test vertical turns from every lane immediately after the first deep flip."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_route_deletions import enter_level_9, replay, route
from probe_l9_twelve_fast_frontier import SKIPS


def blobs(env, colors):
    return tuple(
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(env.frame(), colors=colors, min_area=2)
        if blob.bbox[0] < 63
    )


def avatar_x(env):
    avatars = connected_components(env.frame(), colors=(9,), min_area=3)
    return round(avatars[0].centroid[1])


def probe(env):
    enter_level_9(env)
    root = replay(env, route()[:111], skips=SKIPS)
    for lefts in range(10):
        lane = root.clone()
        for _ in range(lefts):
            lane.step(3)
        x = avatar_x(lane)
        print(
            "LANE",
            9 - lefts,
            "lefts",
            lefts,
            "state",
            compact(lane),
            "controls",
            controls(lane),
            "objects",
            blobs(lane, (7, 8, 9, 11, 12, 14)),
        )
        for name, action in (
            ("ABOVE", (6, x, 21)),
            ("SAME", (6, x, 27)),
            ("BELOW", (6, x, 33)),
        ):
            child = lane.clone()
            child.step(*action)
            print(
                "TURN",
                9 - lefts,
                name,
                action,
                "terminal",
                bool(child.terminal()),
                "state",
                compact(child),
                "controls",
                controls(child),
                "objects",
                blobs(child, (7, 8, 9, 11, 12, 14)),
            )


if __name__ == "__main__":
    arena.run_program("bp35", probe)
