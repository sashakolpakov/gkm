"""Map the pristine prize-wall entrance as a horizontal placement puzzle."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_route_deletions import enter_level_9


def boxes(env, color):
    return tuple(
        blob.bbox
        for blob in connected_components(env.frame(), colors=(color,), min_area=2)
        if blob.bbox[0] < 63
    )


def report(label, env):
    print(
        label,
        "terminal",
        bool(env.terminal()),
        "levels",
        int(env.levels_completed),
        "controls",
        controls(env),
        "goals",
        boxes(env, 7),
        "avatar",
        boxes(env, 9),
        "grid",
        compact(env)["grid9"],
        flush=True,
    )


def run(root, name, prefix, suffix):
    child = root.clone()
    for action in prefix:
        child.step(*action) if isinstance(action, tuple) else child.step(action)
    report((name, "PRE"), child)
    visible = controls(child)
    if not visible:
        return
    child.step(*visible[0])
    report((name, "FLIP"), child)
    for index, action in enumerate(suffix, 1):
        child.step(*action) if isinstance(action, tuple) else child.step(action)
        report((name, index, action), child)
        if child.terminal() or int(child.levels_completed) >= 9:
            break


def probe(env):
    enter_level_9(env)
    report("ENTRY", env)
    clear = ((6, 3, 39),)
    variants = {
        "RAW": (),
        "CLEAR": clear,
        "C_R1": clear + (4,),
        "C_R2": clear + (4, 4),
        "C_R3": clear + (4, 4, 4),
        "C_R4": clear + (4, 4, 4, 4),
        "C_L1": clear + (3,),
    }
    for name, prefix in variants.items():
        run(env, name, prefix, (3, 3, 4, 4, (6, 27, 33), (6, 27, 21)))


if __name__ == "__main__":
    arena.run_program("bp35", probe)
