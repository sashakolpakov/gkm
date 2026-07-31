"""Try crossing the three-skip catch gate without consuming a switch."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_multi_skip_bottom import root_for
from probe_l9_route_deletions import enter_level_9


def blobs(env, color):
    return tuple(
        blob
        for blob in connected_components(env.frame(), colors=(color,), min_area=3)
        if blob.bbox[0] < 63
    )


def avatar(env):
    found = blobs(env, 9)
    return found[0].bbox if found else ()


def catches(env):
    return tuple(
        (6, round(blob.centroid[1]), round(blob.centroid[0]))
        for blob in blobs(env, 15)
        if blob.area == 21
    )


def brief(env):
    return (
        bool(env.terminal()),
        int(env.levels_completed),
        avatar(env),
        len(controls(env)),
        compact(env)["grid9"],
    )


def run(root, name, actions):
    child = root_for(root, 3)
    print(name, 0, brief(child), flush=True)
    for index, action in enumerate(actions, 1):
        child.step(*action) if isinstance(action, tuple) else child.step(action)
        print(name, index, action, brief(child), flush=True)
        if child.terminal() or int(child.levels_completed) >= 9:
            break


def probe(env):
    enter_level_9(env)
    root = root_for(env, 3)
    print("CATCHES", catches(root), flush=True)
    for catch in catches(root):
        child = root.clone()
        child.step(*catch)
        child.step(4)
        child.step(4)
        print(("ONE", catch), brief(child), flush=True)
    variants = {
        "ROW": ((6, 21, 39), 4, 4),
        "ROW_SUPPORT": ((6, 21, 39), (6, 21, 45), 4, 4),
        "BELOW": ((6, 21, 45), 4, 4),
        "ROW_THEN_C4": ((6, 21, 39), 4, (6, 27, 39), 4),
        "ROW_THEN_C4_BELOW": ((6, 21, 39), 4, (6, 27, 45), 4),
        "CLEAR_PAIR": ((6, 21, 39), (6, 27, 45), 4, 4),
        "PRESERVE_DROP_BELOW": (
            (6, 21, 39),
            4,
            (6, 27, 39),
            4,
        )
        + ((6, 27, 45),) * 7,
        "PRESERVE_DROP_ABOVE": (
            (6, 21, 39),
            4,
            (6, 27, 39),
            4,
        )
        + ((6, 27, 33),) * 7,
    }
    for name, actions in variants.items():
        run(env, name, actions)


if __name__ == "__main__":
    arena.run_program("bp35", probe)
