"""Cross the two yellow cells after arresting the lower wall gap."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_flipped_c5_supports import avatar, root_state
from probe_l9_route_deletions import enter_level_9


def goals(env):
    return tuple(
        blob.bbox
        for blob in connected_components(env.frame(), colors=(7,), min_area=2)
        if blob.bbox[0] < 63
    )


def yellows(env):
    return tuple(
        (6, round(blob.centroid[1]), round(blob.centroid[0]))
        for blob in connected_components(env.frame(), colors=(14,), min_area=3)
        if blob.bbox[0] < 63
    )


def report(label, env):
    print(
        label,
        "terminal",
        bool(env.terminal()),
        "levels",
        int(env.levels_completed),
        "avatar",
        avatar(env),
        "controls",
        controls(env),
        "goals",
        goals(env),
        "yellows",
        yellows(env),
        "grid",
        compact(env)["grid9"],
        flush=True,
    )


def step(env, action):
    env.step(*action) if isinstance(action, tuple) else env.step(action)


def probe(env):
    enter_level_9(env)
    child = root_state(env)
    actions = [(6, 21, 27), 3, 3]
    for index, action in enumerate(actions, 1):
        step(child, action)
        report((index, action), child)
    for label in ("RIGHT_YELLOW", "LEFT_YELLOW"):
        visible = sorted(yellows(child), key=lambda action: action[1], reverse=True)
        if not visible:
            break
        action = visible[0]
        step(child, action)
        report((label, action), child)
        step(child, 3)
        report((label, 3), child)
        if child.terminal() or int(child.levels_completed) >= 9:
            return
    for index in range(1, 9):
        step(child, 3)
        report(("LEFT", index), child)
        if child.terminal() or int(child.levels_completed) >= 9:
            return


if __name__ == "__main__":
    arena.run_program("bp35", probe)
