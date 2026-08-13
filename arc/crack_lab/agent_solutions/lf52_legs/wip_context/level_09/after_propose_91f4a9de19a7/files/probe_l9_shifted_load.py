"""Load L9's lone local survivor into a carrier shifted left of entry."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import connected_components, safe_step
from probe_l9_onepeg_openings import move, onepeg_paths, parse


TARGET_BRIDGES = frozenset({
    (18, 30), (24, 24), (36, 24), (42, 24),
})


def compact(frame):
    return tuple(sorted(
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(
            frame, colors=(7, 9, 11, 12, 14, 15)
        )
        if blob.color not in (9, 12) or blob.area >= 12
    ))


def probe(env):
    with open("checkpoint.json") as stream:
        campaign = json.load(stream)["final_path"]
    for action in campaign:
        safe_step(env, tuple(action) if isinstance(action, list) else action)

    goals = onepeg_paths(env.frame())
    target = next(
        item for item in goals
        if item[1] == (36, 18) and frozenset(item[2]) == TARGET_BRIDGES
    )
    depth, _, _, path = target
    for source, destination in path:
        move(env, source, destination)
    print("shifted_before", depth, parse(env.frame()), compact(env.frame()),
          flush=True)
    safe_step(env, 3)
    safe_step(env, 3)
    print("shifted_carrier", parse(env.frame()), compact(env.frame()),
          flush=True)
    move(env, (36, 18), (36, 30))
    print("shifted_loaded", int(env.levels_completed), parse(env.frame()),
          compact(env.frame()), flush=True)
    for offset in range(10):
        print("shifted_scan", offset, compact(env.frame()), flush=True)
        safe_step(env, 4)


if __name__ == "__main__":
    arena.run_program("lf52", probe)
