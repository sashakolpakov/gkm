"""Test the pristine two-switch handoff at one chosen horizontal position."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_route_deletions import enter_level_9


SPEC = sys.argv[1] if len(sys.argv) > 1 else "L1"


def boxes(env, color):
    return tuple(
        blob.bbox
        for blob in connected_components(env.frame(), colors=(color,), min_area=2)
        if blob.bbox[0] < 63
    )


def report(label, env):
    print(
        SPEC,
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


def probe(env):
    enter_level_9(env)
    env.step(*controls(env)[0])
    move = 3 if SPEC[0] == "L" else 4
    for _ in range(int(SPEC[1:])):
        env.step(move)
    report("PRE_SECOND", env)
    env.step(*controls(env)[0])
    report("SECOND", env)
    for index, action in enumerate(
        (3, 4, (6, 21, 33), (6, 21, 45), 3, 4), 1
    ):
        env.step(*action) if isinstance(action, tuple) else env.step(action)
        report((index, action), env)
        if env.terminal() or int(env.levels_completed) >= 9:
            break


if __name__ == "__main__":
    arena.run_program("bp35", probe)
