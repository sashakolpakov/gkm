"""Cross the catch lane after the useful three-skip top switch."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_multi_skip_bottom import root_for
from probe_l9_route_deletions import enter_level_9


def pieces(env):
    return tuple(
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(
            env.frame(), colors=(7, 8, 9, 11, 12, 14, 15), min_area=2
        )
        if blob.bbox[0] < 63
        and (blob.color in (7, 8, 9, 11, 12, 14) or blob.area == 21)
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
        "grid",
        compact(env)["grid9"],
        "pieces",
        pieces(env),
        flush=True,
    )


def stage(root, switch_index=0):
    child = root_for(root, 3)
    child.step(*controls(child)[switch_index])
    child.step(6, 21, 27)
    child.step(4)
    child.step(6, 27, 27)
    child.step(4)
    return child


def run(root, name, actions, switch_index=0):
    child = stage(root, switch_index)
    report((name, 0), child)
    for index, token in enumerate(actions, 1):
        action = token
        if token == "top":
            action = controls(child)[0]
        elif token == "bottom":
            action = controls(child)[-1]
        child.step(*action) if isinstance(action, tuple) else child.step(action)
        report((name, index, action), child)
        if child.terminal() or int(child.levels_completed) >= 9:
            break


def probe(env):
    enter_level_9(env)
    for switch_index in range(3):
        run(env, ("ROOT_SWITCH", switch_index), (), switch_index)
    variants = {
        "MOVE": (3, 4, 4),
        "TOP_LEFT": ("top", 3, 3, 4),
        "BOTTOM_LEFT": ("bottom", 3, 3, 4),
        "DROP_C4": ((6, 27, 33),) * 5,
        "CLIMB_C4": ((6, 27, 21),) * 5,
    }
    for name, actions in variants.items():
        run(env, name, actions)


if __name__ == "__main__":
    arena.run_program("bp35", probe)
