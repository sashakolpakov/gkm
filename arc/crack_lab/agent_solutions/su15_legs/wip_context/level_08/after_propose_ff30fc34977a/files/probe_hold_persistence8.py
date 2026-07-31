import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from probe_clean8 import PREFIX, body_groups


TARGETS = ((19, 56), (55, 7), (55, 56))


def center(group):
    return (
        round(sum(row for row, _ in group) / len(group)),
        round(sum(col for _, col in group) / len(group)),
    )


def centers(env):
    return tuple(center(group) for group in body_groups(env.frame()))


def hold_nearest(env, target):
    row, col = min(
        centers(env),
        key=lambda point: max(
            abs(point[0] - target[0]), abs(point[1] - target[1])
        ),
    )
    env.step(6, col, row)
    return 6, col, row


def inspect(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(*action)
    for action in PREFIX:
        env.step(*action)
    print("ROOT", centers(env))
    for target in TARGETS:
        action = hold_nearest(env, target)
        print("HOLD", target, action, centers(env))
    for target in reversed(TARGETS):
        action = hold_nearest(env, target)
        print("HOLD", target, action, centers(env))


A.run_program("su15", inspect)
