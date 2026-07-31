"""Map the short lower-barrier continuation after the two-flip handoff."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact
from probe_l9_multiskip_elevator import elevator
from probe_l9_route_deletions import enter_level_9


LOWER = (6, 15, 33)


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
        "levels",
        int(env.levels_completed),
        "grid",
        compact(env)["grid9"],
        "pieces",
        pieces(env),
        flush=True,
    )


def lower(root):
    child = elevator(root)
    child.step(*LOWER)
    return child


def run(root, name, actions):
    child = lower(root)
    report((name, 0), child)
    for index, action in enumerate(actions, 1):
        child.step(*action) if isinstance(action, tuple) else child.step(action)
        report((name, index, action), child)
        if child.terminal() or int(child.levels_completed) >= 9:
            break


def probe(env):
    enter_level_9(env)
    variants = {
        "RIGHT": (4,) * 12,
        "CLEAR_WALK": (
            (6, 21, 27),
            4,
            (6, 27, 27),
            4,
            (6, 33, 27),
            4,
            (6, 39, 27),
            4,
        ),
        "SUPPORT_WALK": (
            (6, 21, 33),
            4,
            (6, 27, 33),
            4,
            (6, 33, 33),
            4,
            (6, 39, 33),
            4,
        ),
        "DROP_GAP": ((6, 33, 33),) * 8,
    }
    for name, actions in variants.items():
        run(env, name, actions)


if __name__ == "__main__":
    arena.run_program("bp35", probe)
