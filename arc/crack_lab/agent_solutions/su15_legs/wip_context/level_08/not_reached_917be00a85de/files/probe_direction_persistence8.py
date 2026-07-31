import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from probe_clean8 import body_groups


def center(group):
    return (
        round(sum(row for row, _ in group) / len(group)),
        round(sum(col for _, col in group) / len(group)),
    )


def centers(env):
    return tuple(center(group) for group in body_groups(env.frame()))


def inspect(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(*action)
    for name, first in (
        ("N", (6, 45, 31)),
        ("W", (6, 43, 33)),
        ("E", (6, 47, 33)),
        ("S", (6, 45, 34)),
        ("NW", (6, 44, 32)),
        ("NE", (6, 46, 32)),
        ("SW", (6, 44, 34)),
        ("SE", (6, 46, 34)),
        ("HOLD", (6, 45, 33)),
    ):
        node = env.clone()
        before = centers(node)
        node.step(*first)
        after_first = centers(node)
        other = min(
            after_first,
            key=lambda point: max(abs(point[0] - 50), abs(point[1] - 49)),
        )
        node.step(6, other[1], other[0])
        print(name, before, after_first, centers(node))


A.run_program("su15", inspect)
