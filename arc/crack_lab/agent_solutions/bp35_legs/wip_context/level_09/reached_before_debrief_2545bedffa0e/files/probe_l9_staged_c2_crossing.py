"""Clear moving horizontal gates during the staged column-two descent."""

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


def step(env, action):
    env.step(*action) if isinstance(action, tuple) else env.step(action)


def depth5(root):
    child = staged_entry(root, 2)
    for _ in range(5):
        child.step(6, 15, 33)
    return child


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


def run(root, name, actions):
    child = depth5(root)
    report((name, "START"), child)
    for index, action in enumerate(actions, 1):
        step(child, action)
        report((name, index, action), child)
        if child.terminal() or int(child.levels_completed) >= 9:
            return


def probe(env):
    enter_level_9(env)
    run(env, "CLEAR_C3", ((6, 21, 27), 4, 4, 4, 4, 4, (6, 51, 27), 4, 4))
    run(env, "CLEAR_C8", ((6, 51, 27), (6, 21, 27), 4, 4, 4, 4, 4, 4, 4))
    run(env, "YELLOW", ((6, 27, 27), (6, 21, 27), 4, 4, 4, 4, 4, (6, 51, 27), 4, 4))


if __name__ == "__main__":
    arena.run_program("bp35", probe)
