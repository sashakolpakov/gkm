"""Compact traces of already-solved collection mechanics."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception as p
import players


def state(env):
    objects = p.object_candidates(env.frame())
    avatar = next(
        o["bbox"][:2]
        for o in objects
        if o["color"] == 6 and o["area"] == 18 and o["bbox"][0] < 53
    )
    pieces = tuple(
        (o["color"], *o["bbox"][:2])
        for o in objects
        if o["color"] in (8, 9, 14) and o["bbox"][0] < 53
    )
    target = tuple(
        (o["color"], *o["bbox"][:2])
        for o in objects
        if o["color"] in (8, 9, 14) and o["bbox"][0] >= 53
    )
    return avatar, pieces, target


def steps(env, action, count):
    for _ in range(count):
        env.step(action)


def run(env):
    print("L1", state(env))
    for label, action, count in (
        ("U3", 1, 3),
        ("E4", 4, 4),
        ("R4", 3, 4),
        ("D2", 2, 2),
        ("E4", 4, 4),
        ("R3", 3, 3),
        ("U2", 1, 2),
        ("D1", 2, 1),
        ("E3", 4, 3),
    ):
        steps(env, action, count)
        print(label, state(env), env.levels_completed)

    print("L2", state(env))
    steps(env, 1, 4)
    print("U4", state(env))
    for index, (span, reach, compact) in enumerate(
        ((5, 6, 6), (4, 6, 5), (2, 5, 4)), start=1
    ):
        for label, action, count in (
            ("E", 4, span),
            ("D", 2, 1),
            ("U", 1, 1),
            ("R", 3, span),
            ("D", 2, 1),
            ("W", 4, reach),
            ("C", 3, compact),
        ):
            steps(env, action, count)
            print(f"S{index}{label}", state(env))
    steps(env, 2, 1)
    steps(env, 4, 4)
    print("L2END", state(env), env.levels_completed)


levels, path, error = arena.run_program("sk48", run)
print("END", levels, len(path), error)
