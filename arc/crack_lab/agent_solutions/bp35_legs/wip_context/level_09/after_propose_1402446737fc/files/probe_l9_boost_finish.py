"""Find the deadline-tight corridor from the boosted handoff to the goal wall."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_boost_direct import direct
from probe_l9_control_row import compact, controls
from probe_l9_route_deletions import enter_level_9


def goals(env):
    return tuple(
        blob.bbox
        for blob in connected_components(env.frame(), colors=(7,), min_area=2)
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
        goals(env),
        "grid",
        compact(env)["grid9"],
        flush=True,
    )


def turned(root):
    child = direct(root)
    child.step(*controls(child)[0])
    return child


def run(root, name, actions):
    child = turned(root)
    report((name, 0), child)
    for index, token in enumerate(actions, 1):
        action = token
        if token == "control":
            visible = controls(child)
            if not visible:
                report((name, index, "NO_CONTROL"), child)
                break
            action = visible[-1]
        child.step(*action) if isinstance(action, tuple) else child.step(action)
        report((name, index, action), child)
        if child.terminal() or int(child.levels_completed) >= 9:
            break


def probe(env):
    enter_level_9(env)
    variants = {
        "RIGHT": (4,) * 9,
        "CLEAR_ROW": ((6, 21, 27), 4, 4, 4, "control", 3, 3, 3),
        "LOWER_CLEAR": (
            (6, 15, 33),
            (6, 21, 27),
            4,
            4,
            "control",
            3,
            3,
        ),
        "CLEAR_BELOW": ((6, 21, 33), 4, 4, 4, "control", 3, 3),
    }
    for name, actions in variants.items():
        run(env, name, actions)


if __name__ == "__main__":
    arena.run_program("bp35", probe)
