"""Carry the extra boosted switch through both handoffs to the prize wall."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_preserve_boost import boosted
from probe_l9_route_deletions import enter_level_9


DROP = (6, 27, 33)
LOWER = (6, 15, 33)


def goals(env):
    return tuple(
        (blob.bbox, blob.area)
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


def select(items, index):
    return items[index if index >= 0 else len(items) + index]


def approach(root, first_index, second_index):
    child = boosted(root)
    child.step(*controls(child)[0])
    child.step(*DROP)
    report((first_index, second_index, "FIRST"), child)
    child.step(*select(controls(child), first_index))
    child.step(3)
    child.step(3)
    report((first_index, second_index, "SECOND"), child)
    visible = controls(child)
    if not visible:
        return child
    child.step(*select(visible, second_index))
    child.step(*LOWER)
    child.step(4)
    child.step(4)
    child.step(4)
    report((first_index, second_index, "WALL"), child)
    return child


def probe(env):
    enter_level_9(env)
    for first_index in (0, 1, -1):
        for second_index in (0, -1):
            child = approach(env, first_index, second_index)
            visible = controls(child)
            if visible and not child.terminal():
                child.step(*visible[-1])
                report((first_index, second_index, "FINAL_FLIP"), child)
                for action in (3, 3, 3, 4, (6, 15, 33)):
                    child.step(*action) if isinstance(action, tuple) else child.step(action)
                    report((first_index, second_index, action), child)
                    if child.terminal() or int(child.levels_completed) >= 9:
                        break


if __name__ == "__main__":
    arena.run_program("bp35", probe)
