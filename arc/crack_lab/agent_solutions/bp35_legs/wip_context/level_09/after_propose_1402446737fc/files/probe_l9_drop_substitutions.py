"""Stage final shelf cells during the five mandatory column-three falls."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact
from probe_l9_route_deletions import enter_level_9
from probe_l9_second_lower_entry import goals
from probe_l9_staged_entry_trace import replay


DROP = (6, 21, 33)


def avatar(env):
    return tuple(
        blob.bbox
        for blob in connected_components(env.frame(), colors=(9,), min_area=3)
        if blob.bbox[0] < 63
    )


def brief(env):
    return (
        bool(env.terminal()),
        int(env.levels_completed) + 1,
        avatar(env),
        goals(env),
        compact(env)["grid9"],
    )


def run(root, name, falls):
    child = replay(root, skips=(0,))
    for action in falls:
        child.step(*action)
        if child.terminal():
            break
    if not child.terminal():
        for action in ((6, 27, 27), 4, 4, 4):
            child.step(*action) if isinstance(action, tuple) else child.step(action)
            if child.terminal():
                break
    print(name, falls, brief(child), flush=True)


def probe(env):
    enter_level_9(env)
    run(env, "BASE", (DROP,) * 5)
    for step in range(2, 6):
        below_y = 63 - 6 * (step - 1)
        row_y = 57 - 6 * (step - 1)
        for label, action in (
            ("SUPPORT7", (6, 45, below_y)),
            ("CLEAR8", (6, 51, row_y)),
            ("CLEAR9", (6, 57, row_y)),
            ("SUPPORT9", (6, 57, below_y)),
        ):
            falls = [DROP] * 5
            falls[step - 1] = action
            run(env, (label, step), tuple(falls))


if __name__ == "__main__":
    arena.run_program("bp35", probe)
