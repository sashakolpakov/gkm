"""Label the preserved level-9 route by visible lattice piece transitions."""

import json
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import _bridge_carrier_state


def play(env, action):
    if isinstance(action, list):
        env.step(*action)
    else:
        env.step(action)


def patch_kind(frame, position):
    row, col = position
    patch = np.asarray(frame)[max(0, row - 1):min(64, row + 5),
                              max(0, col - 1):min(64, col + 5)]
    counts = {
        color: int(np.count_nonzero(patch == color))
        for color in (1, 3, 9, 11, 12, 14, 15)
    }
    return max(counts, key=counts.get), tuple(
        (color, count) for color, count in counts.items() if count
    )


def compact_state(frame):
    slots, pegs, carriers, bridges, _, selected = _bridge_carrier_state(frame)
    return {
        "p": tuple(sorted(pegs)),
        "c": tuple(sorted(carriers)),
        "b": tuple(sorted(bridges)),
        "s": selected,
        "n": len(slots),
    }


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        prefix = json.load(checkpoint_file)["final_path"]
    with open("level9_candidate_102.json") as candidate_file:
        candidate = json.load(candidate_file)
    for action in prefix:
        play(env, action)
    node = env.clone()

    index = 0
    while index < len(candidate):
        action = candidate[index]
        if isinstance(action, int):
            before = compact_state(node.frame())
            play(node, action)
            after = compact_state(node.frame())
            print("KEY", index + 1, action, {"before": before, "after": after})
            index += 1
            continue

        if index + 1 >= len(candidate) or isinstance(candidate[index + 1], int):
            raise RuntimeError(f"unpaired coordinate action at {index + 1}")
        destination_action = candidate[index + 1]
        source = (action[2] - 1, action[1] - 1)
        destination = (
            destination_action[2] - 1,
            destination_action[1] - 1,
        )
        midpoint = (
            (source[0] + destination[0]) // 2,
            (source[1] + destination[1]) // 2,
        )
        before_frame = node.frame()
        before = {
            "src": patch_kind(before_frame, source),
            "mid": patch_kind(before_frame, midpoint),
            "dst": patch_kind(before_frame, destination),
            "state": compact_state(before_frame),
        }
        play(node, action)
        play(node, destination_action)
        after_frame = node.frame()
        after = {
            "src": patch_kind(after_frame, source),
            "mid": patch_kind(after_frame, midpoint),
            "dst": patch_kind(after_frame, destination),
            "state": compact_state(after_frame),
        }
        print(
            "MOVE",
            (index + 1, index + 2),
            (source, midpoint, destination),
            {"before": before, "after": after, "level": node.levels_completed},
        )
        index += 2


levels, path, error = arena.run_program("lf52", probe)
print("HARNESS", {"levels": levels, "moves": len(path), "error": str(error) if error else None})
