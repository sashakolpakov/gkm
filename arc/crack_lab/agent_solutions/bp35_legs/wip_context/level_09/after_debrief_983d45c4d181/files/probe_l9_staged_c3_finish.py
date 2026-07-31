"""Continue the staged column-three cross over its final support gap."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_route_deletions import enter_level_9
from probe_l9_second_lower_entry import goals, staged_entry


def frontier(root):
    child = staged_entry(root, 3)
    for action in (
        *((6, 21, 33),) * 5,
        (6, 27, 27),
        4,
        4,
        4,
    ):
        child.step(*action) if isinstance(action, tuple) else child.step(action)
    return child


def avatar(env):
    return tuple(
        blob.bbox
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
        "avatar",
        avatar(env),
        "controls",
        controls(env),
        "goals",
        goals(env),
        "grid",
        compact(env)["grid9"],
        flush=True,
    )


def run(root, name, actions):
    child = frontier(root)
    report((name, 0), child)
    for index, action in enumerate(actions, 1):
        child.step(*action) if isinstance(action, tuple) else child.step(action)
        report((name, index, action), child)
        if child.terminal() or int(child.levels_completed) >= 9:
            break


def probe(env):
    enter_level_9(env)
    variants = {
        "C7_RRR": ((6, 45, 33), 4, 4, 4),
        "C7_R_CLEAR8_RR": (
            (6, 45, 33),
            4,
            (6, 51, 27),
            4,
            4,
        ),
        "CLEAR8_C7_RRR": (
            (6, 51, 27),
            (6, 45, 33),
            4,
            4,
            4,
        ),
        "CLEAR8_R_CLEAR7_RR": (
            (6, 51, 27),
            4,
            (6, 45, 27),
            4,
            4,
        ),
        "C7_RR_C9_R": ((6, 45, 33), 4, 4, (6, 57, 33), 4),
        "C7_R_C8_R": ((6, 45, 33), 4, (6, 51, 33), 4, 4),
        "C7_C9_RRR": ((6, 45, 33), (6, 57, 33), 4, 4, 4),
        "C7_RR_DOWN": ((6, 45, 33), 4, 4, (6, 51, 33), (6, 51, 33)),
    }
    for name, actions in variants.items():
        run(env, name, actions)


if __name__ == "__main__":
    arena.run_program("bp35", probe)
