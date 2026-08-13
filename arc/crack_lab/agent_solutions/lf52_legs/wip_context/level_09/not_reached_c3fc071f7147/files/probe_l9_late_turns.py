"""Compare late loaded-carrier turns after returning fully on-screen."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import arr, connected_components, safe_step


FIRST_RELAY = (
    ((42, 18), (42, 30)), ((48, 24), (36, 24)),
    ((42, 24), (30, 24)), ((36, 24), (24, 24)),
    ((30, 24), (18, 24)), ((18, 24), (18, 36)),
    ((18, 36), (30, 36)), ((24, 36), (36, 36)),
    ((30, 36), (42, 36)), ((36, 36), (48, 36)),
    ((48, 42), (48, 30)), ((48, 30), (36, 30)),
    ((48, 36), (36, 36)), ((36, 30), (36, 42)),
)


def move(env, source, destination):
    safe_step(env, (6, source[1] + 1, source[0] + 1))
    safe_step(env, (6, destination[1] + 1, destination[0] + 1))


def key(env):
    return arr(env.frame())[1:, :].tobytes()


def signature(env):
    return tuple(
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(
            env.frame(), colors=(9, 11, 12, 14, 15)
        )
        if blob.color != 9 or blob.area >= 12
    )


def settle_left(env):
    changes = 0
    for _ in range(20):
        before = key(env)
        safe_step(env, 3)
        changes += key(env) != before
    return changes


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    for source, destination in FIRST_RELAY:
        move(env, source, destination)
    if os.environ.get("OPT_UNLOAD") == "1":
        safe_step(env, (6, 23, 37))
        safe_step(env, (6, 11, 37))

    rows = []
    for offset in range(15):
        base = env.clone()
        for _ in range(offset):
            safe_step(base, 4)
        base_changes = settle_left(base)
        base_key = key(base)

        turns = []
        for action in (1, 2):
            child = env.clone()
            for _ in range(offset):
                safe_step(child, 4)
            safe_step(child, action)
            return_changes = settle_left(child)
            turns.append((
                action,
                return_changes,
                key(child) == base_key,
                signature(child) if key(child) != base_key else (),
            ))
        rows.append((offset, base_changes, tuple(turns)))
    print("late_turns", rows, flush=True)


arena.run_program("lf52", probe)
