import json
import time

import gkm_try as H

from perception import connected_components
from probe_clean8 import PREFIX, body_groups, body_pixels
from probe_frontier23 import SUFFIX_23


def inspect(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(*action)
    start_level = int(env.levels_completed)
    root = env.clone()
    fixed = PREFIX + SUFFIX_23[:5]
    for action in fixed:
        root.step(*action)

    def key(node):
        square = tuple(
            blob.bbox
            for blob in connected_components(
                node.frame(), colors=(8,), min_area=9
            )
            if blob.bbox[0] >= 10
        )
        return square, body_groups(node.frame())

    frontier = [(root, [])]
    seen = {key(root)}
    clone_steps = 0
    started = time.monotonic()
    for depth in range(1, 4):
        next_frontier = []
        for node, path in frontier:
            actions = [
                (6, col, row)
                for row, col in sorted(body_pixels(node.frame()))
            ]
            actions.append((6, 32, 32))
            for action in actions:
                child = node.clone()
                child.step(*action)
                clone_steps += 1
                wait = clone_steps / 300 - (time.monotonic() - started)
                if wait > 0:
                    time.sleep(wait)
                child_path = path + [action]
                if int(child.levels_completed) > start_level:
                    print("FOUND", fixed + child_path)
                    return
                if child.terminal():
                    continue
                child_key = key(child)
                if child_key in seen:
                    continue
                seen.add(child_key)
                next_frontier.append((child, child_path))
        frontier = next_frontier
        print(
            "DEPTH",
            depth,
            "frontier",
            len(frontier),
            "seen",
            len(seen),
            flush=True,
        )
    print("NO_PATH", len(seen))


H.A.run_program("su15", inspect)
