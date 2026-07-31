"""Use the yellow gate to cross staged column three in exactly nineteen turns."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_route_deletions import enter_level_9
from probe_l9_second_lower_entry import goals, staged_entry


def avatars(env):
    return tuple(
        (blob.bbox, blob.area)
        for blob in connected_components(env.frame(), colors=(9,), min_area=3)
        if blob.bbox[0] < 63
    )


def report(label, env):
    print(
        label,
        "terminal",
        bool(env.terminal()),
        "level",
        int(env.levels_completed) + 1,
        "controls",
        controls(env),
        "goals",
        goals(env),
        "avatar",
        avatars(env),
        "state",
        compact(env),
    )


def probe(env):
    enter_level_9(env)
    child = staged_entry(env, 3)
    report("ENTRY", child)
    actions = [
        *([(6, 21, 33)] * 5),
        (6, 27, 27),
        4,
        4,
        4,
    ]
    for index, action in enumerate(actions, 1):
        child.step(*action) if isinstance(action, tuple) else child.step(action)
        report(("STEP", index, action), child)
        if child.terminal() or int(child.levels_completed) >= 9:
            return
    for action in (
        3,
        4,
        7,
        (6, 39, 33),
        (6, 39, 27),
        (6, 39, 39),
        (6, 45, 33),
        (6, 51, 33),
        (6, 51, 27),
    ):
        branch = child.clone()
        branch.step(*action) if isinstance(action, tuple) else branch.step(action)
        report(("ACTION", action), branch)


if __name__ == "__main__":
    arena.run_program("bp35", probe)
