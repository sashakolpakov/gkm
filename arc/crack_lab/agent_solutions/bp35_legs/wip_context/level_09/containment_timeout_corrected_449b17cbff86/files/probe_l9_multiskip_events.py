"""Log only observational transitions in the multi-skip route."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_route_deletions import enter_level_9, route, step
from probe_l9_twelve_fast_frontier import SKIPS


def boxes(env, colors):
    return tuple(
        blob.bbox
        for blob in connected_components(env.frame(), colors=colors, min_area=2)
        if blob.bbox[0] < 63
    )


def observation(env):
    return (
        bool(env.terminal()),
        int(env.levels_completed) + 1,
        boxes(env, (9, 11)),
        boxes(env, (7,)),
        tuple(controls(env)),
        compact(env)["grid9"],
    )


def trace(root, count):
    child = root.clone()
    omitted = SKIPS | set(range(11, 11 + count))
    previous = observation(child)
    applied = 0
    print("START", count, previous, flush=True)
    for index, (section, action) in enumerate(route()):
        if index in omitted:
            continue
        step(child, action)
        applied += 1
        current = observation(child)
        if current != previous:
            print(
                "EVENT",
                count,
                "index",
                index,
                "applied",
                applied,
                section,
                action,
                current,
                flush=True,
            )
            previous = current
        if child.terminal() or int(child.levels_completed) >= 9:
            break


def probe(env):
    enter_level_9(env)
    trace(env, 6)
    trace(env, 10)


if __name__ == "__main__":
    arena.run_program("bp35", probe)
