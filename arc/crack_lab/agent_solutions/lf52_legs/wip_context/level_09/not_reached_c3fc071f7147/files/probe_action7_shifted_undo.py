"""Test undoing a carrier load after moving the carrier/view."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import _bridge_carrier_state
from perception import arr, safe_step


OPENING = (
    ((42, 18), (42, 30)),
    ((48, 24), (36, 24)),
    ((42, 24), (30, 24)),
    ((36, 24), (24, 24)),
    ((30, 24), (18, 24)),
    ((18, 24), (18, 36)),
    ((18, 36), (30, 36)),
    ((24, 36), (36, 36)),
    ((30, 36), (42, 36)),
    ((36, 36), (48, 36)),
    ((48, 42), (48, 30)),
    ((48, 30), (36, 30)),
    ((48, 36), (36, 36)),
    ((36, 30), (36, 42)),
)


def normalized(frame):
    image = arr(frame).copy()
    image[0] = 0
    return image.tobytes()


def summarize(node):
    state = _bridge_carrier_state(node.frame())
    return (
        tuple(sorted(state[1])),
        tuple(sorted(state[2])),
        tuple(sorted(state[4])),
        state[5],
        int(node.levels_completed),
    )


def move(node, source, destination):
    safe_step(node, (6, source[1] + 1, source[0] + 1))
    safe_step(node, (6, destination[1] + 1, destination[0] + 1))


def probe(env):
    with open("checkpoint.json") as stream:
        campaign = json.load(stream)["final_path"]
    prior = int(env.levels_completed)
    entry = None
    for action in campaign:
        safe_step(env, tuple(action) if isinstance(action, list) else action)
        current = int(env.levels_completed)
        if prior < 8 <= current:
            entry = env.clone()
            break
        prior = current

    pre = entry.clone()
    for source, destination in OPENING[:-1]:
        move(pre, source, destination)

    for count in (0, 1, 3, 9):
        baseline = pre.clone()
        for _ in range(count):
            safe_step(baseline, 4)

        shifted_undo = pre.clone()
        move(shifted_undo, *OPENING[-1])
        for _ in range(count):
            safe_step(shifted_undo, 4)
        before_undo = summarize(shifted_undo)
        safe_step(shifted_undo, 7)

        print(
            "shifted_load_undo",
            count,
            "baseline",
            summarize(baseline),
            "loaded",
            before_undo,
            "undo",
            summarize(shifted_undo),
            "equals_baseline",
            normalized(shifted_undo.frame()) == normalized(baseline.frame()),
            flush=True,
        )


arena.run_program("lf52", probe)
