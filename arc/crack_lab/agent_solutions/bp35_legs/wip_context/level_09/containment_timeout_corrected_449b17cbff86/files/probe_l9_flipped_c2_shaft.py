"""Enter and climb the column-two shaft exposed by the yellow trapdoors."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_flipped_c5_supports import avatar, root_state
from probe_l9_route_deletions import enter_level_9


def blobs(env, color):
    return tuple(
        blob.bbox
        for blob in connected_components(env.frame(), colors=(color,), min_area=2)
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
        blobs(env, 7),
        "grid",
        compact(env)["grid9"],
        flush=True,
    )


def step(env, action):
    env.step(*action) if isinstance(action, tuple) else env.step(action)


def enter_c2(root):
    child = root_state(root)
    for action in ((6, 21, 27), 3):
        step(child, action)
    for _ in range(2):
        yellow = sorted(
            (
                (6, round(blob.centroid[1]), round(blob.centroid[0]))
                for blob in connected_components(
                    child.frame(), colors=(14,), min_area=3
                )
                if blob.bbox[0] < 63
            ),
            key=lambda action: action[1],
            reverse=True,
        )[0]
        step(child, yellow)
        step(child, 3)
    child.step(6, 15, 39)
    child.step(3)
    return child


def probe(env):
    enter_level_9(env)
    child = enter_c2(env)
    report("C2", child)
    for height in range(1, 15):
        child.step(6, 15, 33)
        report(("CLIMB", height), child)
        if child.terminal() or int(child.levels_completed) >= 9:
            break


if __name__ == "__main__":
    arena.run_program("bp35", probe)
