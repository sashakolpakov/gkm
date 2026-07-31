"""Search short catch staging sequences that arrest the second flip on row four."""

import sys
from collections import deque

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_boosted_gate_alignment import gate
from probe_l9_boosted_supported_landing import report
from probe_l9_control_row import controls
from probe_l9_route_deletions import enter_level_9


def key(env):
    frame = np.asarray(env.frame()).copy()
    frame[63, :] = 0
    return frame.tobytes()


def actions(env):
    return tuple(
        dict.fromkeys(
            (6, round(blob.centroid[1]), round(blob.centroid[0]))
            for blob in connected_components(
                env.frame(), colors=(12, 14, 15), min_area=3
            )
            if blob.bbox[0] < 63 and blob.area == 21
        )
    )


def replay(root, path):
    child = root.clone()
    for action in path:
        child.step(*action)
    return child


def avatar_top(env):
    avatars = connected_components(env.frame(), colors=(9,), min_area=3)
    return avatars[0].bbox[0] if avatars else 99


def probe(env):
    enter_level_9(env)
    root = gate(env, 1)
    queue = deque([()])
    seen = {key(root)}
    while queue and len(seen) <= 2500:
        path = queue.popleft()
        node = replay(root, path)
        visible = controls(node)
        if visible:
            flipped = node.clone()
            flipped.step(*visible[0])
            if (
                not flipped.terminal()
                and avatar_top(flipped) <= 30
                and len(controls(flipped)) >= 2
            ):
                print("FOUND", path, "states", len(seen), flush=True)
                report("FLIPPED", flipped)
                return
        if len(path) >= 3 or node.terminal():
            continue
        for action in actions(node):
            child = node.clone()
            child.step(*action)
            if child.terminal():
                continue
            state = key(child)
            if state in seen:
                continue
            seen.add(state)
            queue.append(path + (action,))
    print("NO_FOUND", "states", len(seen), "queue", len(queue), flush=True)


if __name__ == "__main__":
    arena.run_program("bp35", probe)
