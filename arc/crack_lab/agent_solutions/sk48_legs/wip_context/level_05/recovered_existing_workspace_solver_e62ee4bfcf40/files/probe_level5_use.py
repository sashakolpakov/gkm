"""Compare action 6 with direct actions at the threaded contact frontier."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import players
from search_level5_threaded import (
    CONTROLLED_PAIR,
    LOWER_FINAL_EIGHT,
    PREFIX,
    SUFFIX,
    SURPLUS_TRANSFER,
    apply,
    observe,
)


TO_SINGLE_EIGHT = (
    (3,) * 7
    + (4,) * 3
    + (2,) * 2
    + (4,) * 4
    + (3,) * 4
    + (1,) * 2
)


def summary(env):
    observation = observe(env)
    return (
        env.levels_completed,
        observation[2],
        observation[3],
        observation[4],
        observation[1],
    )


def replay(root, path):
    branch = root.clone()
    try:
        apply(branch, path)
    except IndexError:
        return "INVALID"
    return summary(branch)


def compare(label, root):
    print(label, summary(root))
    print("USE_DELTA", replay(root, (6,)) == summary(root))
    for action in (1, 2, 3, 4):
        direct = replay(root, (action,))
        used = replay(root, (6, action))
        print("COMPARE", action, direct == used, direct, used)
    for path in (
        (6, 4, 4),
        (6, 6, 4),
        (6, 3, 4),
        (6, 1, 4),
        (6, 2, 4),
        (6, 4, 3),
        (6, 4, 6, 4),
    ):
        print("SEQUENCE", path, replay(root, path))


def run(env):
    for level in range(1, 5):
        getattr(players, f"play_level_{level}")(env)
    apply(
        env,
        PREFIX
        + CONTROLLED_PAIR
        + LOWER_FINAL_EIGHT
        + SUFFIX
        + SURPLUS_TRANSFER
        + TO_SINGLE_EIGHT,
    )
    compare("CONTACT", env)
    pair = env.clone()
    for label, path in (
        ("ALIGN", (4,)),
        ("LIFT", (1,)),
        ("EXTEND_TO_WALL", (4,) * 3),
        ("DETACH_FROM_WALL", (3,) * 7),
        ("PARK_RIGHT", (4,) * 3 + (3,) * 3),
        ("APPROACH_ABOVE", (1,) + (4,) * 4),
        ("PUSH_DOWN", (2,) + (3,) * 4 + (2,)),
        ("THREAD_PAIR", (4,) * 7),
    ):
        apply(pair, path)
        print("RETHREAD", label, summary(pair))
    wall = env.clone()
    try:
        apply(wall, (4,) * 3)
        compare("WALL", wall)
    except IndexError:
        print("WALL_INVALID")


levels, path, error = arena.run_program("sk48", run)
print("END", levels, len(path), error)
