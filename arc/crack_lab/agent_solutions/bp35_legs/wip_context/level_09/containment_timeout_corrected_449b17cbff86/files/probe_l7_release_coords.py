"""Test whether coordinate action 7 releases a named held support or LIFO top."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import _cell_shape, click_action
from perception import frame_delta


A = (2, 2)
B = (4, 4)
CA = click_action(*A)
CB = click_action(*B)
RIGHT = (4,)


def shapes(frame):
    return _cell_shape(frame, *A), _cell_shape(frame, *B)


def probe(env):
    with open("checkpoint.json") as stream:
        for action in json.load(stream)["final_path"]:
            env.step(action)
    root = env.clone()
    tests = {
        "two_plain": [CA, CB, (7,)],
        "two_release_a": [CA, CB, (7, CA[1], CA[2])],
        "two_release_b": [CA, CB, (7, CB[1], CB[2])],
        "two_release_empty": [CA, CB, (7, 9, 3)],
        "move_plain": [CA, RIGHT, (7,)],
        "move_release_a": [CA, RIGHT, (7, CA[1], CA[2])],
        "move_release_empty": [CA, RIGHT, (7, 9, 3)],
        "matched_a": [CA, (7, CA[1], CA[2])],
    }
    for name, actions in tests.items():
        node = root.clone()
        states = [shapes(node.frame())]
        deltas = []
        for action in actions:
            before = node.frame()
            node.step(*action)
            delta = frame_delta(before, node.frame())
            deltas.append((action, delta["count"], delta["bbox"]))
            states.append(shapes(node.frame()))
        print(
            "RELEASE_COORD", name, states, deltas,
            bool(node.terminal()), flush=True,
        )


arena.run_program("bp35", probe)
