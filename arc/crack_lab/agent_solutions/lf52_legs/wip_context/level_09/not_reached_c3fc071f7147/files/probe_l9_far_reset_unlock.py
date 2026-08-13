"""Test whether far bridge rearrangements unlock the blocked frontier."""

import json
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


def pieces(frame):
    return tuple(
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(frame, colors=(9, 11, 12, 14, 15))
        if blob.color != 9 or blob.area >= 12
    )


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    root = env.clone()
    for source, destination in FIRST_RELAY:
        move(root, source, destination)
    for _ in range(14):
        safe_step(root, 4)

    variants = (
        ("right", (((18, 22), (18, 34)),)),
        ("left", (((18, 28), (18, 16)),)),
        ("left2", (((18, 28), (18, 16)),
                    ((18, 22), (18, 10)))),
        ("left3", (((18, 28), (18, 16)),
                    ((18, 22), (18, 10)),
                    ((18, 16), (18, 4)))),
    )
    for name, moves in variants:
        node = root.clone()
        for source, destination in moves:
            move(node, source, destination)
        enabled = []
        for action in (1, 2, 3, 4):
            child = node.clone()
            before = key(child)
            safe_step(child, action)
            if key(child) != before:
                enabled.append((action, pieces(child.frame())))
        forward = node.clone()
        changed = []
        for count in range(1, 21):
            before = key(forward)
            safe_step(forward, 4)
            if key(forward) != before:
                changed.append(count)
            if forward.levels_completed > node.levels_completed:
                break
        print("variant", name, "enabled", tuple(enabled),
              "right_changes", tuple(changed),
              "level", int(forward.levels_completed),
              "pieces", pieces(forward.frame()), flush=True)


arena.run_program("lf52", probe)
