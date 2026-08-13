"""Test key actions after selecting visible level-9 pieces."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import arr, connected_components, frame_delta, safe_step


FIRST_RELAY = (
    ((42, 18), (42, 30)), ((48, 24), (36, 24)),
    ((42, 24), (30, 24)), ((36, 24), (24, 24)),
    ((30, 24), (18, 24)), ((18, 24), (18, 36)),
    ((18, 36), (30, 36)), ((24, 36), (36, 36)),
    ((30, 36), (42, 36)), ((36, 36), (48, 36)),
    ((48, 42), (48, 30)), ((48, 30), (36, 30)),
    ((48, 36), (36, 36)), ((36, 30), (36, 42)),
)


def play_move(env, move):
    source, destination = move
    safe_step(env, (6, source[1] + 1, source[0] + 1))
    safe_step(env, (6, destination[1] + 1, destination[0] + 1))


def compact(frame):
    return tuple(
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(frame, colors=(3, 9, 11, 12, 14, 15))
        if blob.color != 9 or blob.area >= 12
    )


def puzzle_delta(before, after):
    left = arr(before).copy()
    right = arr(after).copy()
    right[0] = left[0]
    delta = frame_delta(left, right)
    return delta["count"], delta["bbox"]


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    for move in FIRST_RELAY:
        play_move(env, move)

    patterns = (
        (1, 4, 1), (2, 4, 2), (3, 4, 3),
        (1, 1, 4, 1, 1), (2, 2, 4, 2, 2),
        (1, 4, 2), (2, 4, 1), (1, 2, 4, 2, 1),
    )
    base_frame = env.frame()
    for pattern in patterns:
        node = env.clone()
        for action in pattern:
            safe_step(node, action)
        print("key_pattern", pattern,
              puzzle_delta(base_frame, node.frame()),
              int(node.levels_completed), compact(node.frame()))

    sources = {
        "carried_peg": (6, 23, 37),
        "bridge": (6, 17, 37),
        "new_peg": (6, 53, 13),
        "fixed_bridge": (6, 59, 25),
    }
    for name, source in sources.items():
        selected = env.clone()
        before = selected.frame()
        safe_step(selected, source)
        print("selected", name, puzzle_delta(before, selected.frame()),
              compact(selected.frame()))
        for action in (1, 2, 3, 4, 7):
            child = selected.clone()
            selected_frame = child.frame()
            safe_step(child, action)
            print("selected_key", name, action,
                  puzzle_delta(selected_frame, child.frame()),
                  int(child.levels_completed), compact(child.frame()))


arena.run_program("lf52", probe)
