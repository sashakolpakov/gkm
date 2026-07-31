"""Map immediate actions from the safe c4 landing beneath the yellow stopper."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_boosted_supported_landing import report
from probe_l9_control_row import controls
from probe_l9_handoff_clickmap import avatar
from probe_l9_presecond_top_c4_climb import goals, top_c4
from probe_l9_right_trap_stage import full_catches
from probe_l9_route_deletions import enter_level_9


def upper_landing(root):
    child = top_c4(root)
    child.step(6, 27, 33)
    return child


def visible_objects(env):
    return tuple(
        (blob.color, blob.bbox, blob.area, (round(blob.centroid[1]), round(blob.centroid[0])))
        for blob in connected_components(
            env.frame(), colors=(7, 8, 12, 14, 15), min_area=2
        )
        if blob.bbox[0] < 63
    )


def probe(env):
    enter_level_9(env)
    root = upper_landing(env)
    report("ROOT", root)
    print("OBJECTS", visible_objects(root), flush=True)
    actions = [(3,), (4,), (7,)]
    actions.extend(
        action
        for _, _, action in full_catches(root)
        if action[2] <= 51
    )
    for action in actions:
        child = root.clone()
        child.step(*action)
        print(
            "ACTION",
            action,
            "terminal",
            bool(child.terminal()),
            "avatar",
            avatar(child),
            "controls",
            controls(child),
            "goals",
            goals(child),
            "objects",
            visible_objects(child),
            flush=True,
        )
        if not child.terminal():
            report(("AFTER", action), child)


if __name__ == "__main__":
    arena.run_program("bp35", probe)
