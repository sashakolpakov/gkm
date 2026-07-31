"""Bounded macro search for the first verified movement of a level-5 eight."""
import heapq
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena
import numpy as np

import perception as p
import players


def dynamic_frame(env):
    data = p.arr(env.frame())
    return np.where(np.isin(data, (0, 6, 8, 9)), data, 0).tobytes()


def eights(env):
    return tuple(
        blob.bbox[:2]
        for blob in p.connected_components(env.frame(), colors=(8,), min_area=4)
        if blob.bbox[0] < 53
    )


def pieces(env):
    return tuple(
        (blob.color, *blob.bbox[:2])
        for blob in p.connected_components(
            env.frame(), colors=(8, 9), min_area=4
        )
        if blob.bbox[0] < 53
    )


def search(root, max_states=5000, max_depth=14):
    initial_eights = eights(root)
    counter = 0
    queue = [(0, counter, (), None, ())]
    seen = {(dynamic_frame(root), ())}
    expanded = 0
    while queue and len(seen) < max_states:
        depth, _, path, last_action, context = heapq.heappop(queue)
        node = p.replay(root, path)
        expanded += 1
        if depth >= max_depth:
            continue
        for action in (1, 2, 3, 4):
            if action == last_action:
                continue
            branch = node.clone()
            branch_context = context
            before = dynamic_frame(branch)
            for count in range(1, 7):
                branch.step(action)
                child_path = path + (action,) * count
                if eights(branch) != initial_eights:
                    return child_path, branch, expanded, len(seen)
                after = dynamic_frame(branch)
                if action in (1, 2) and after != before:
                    branch_context = ()
                elif action in (3, 4):
                    branch_context = (branch_context + (action,))[-6:]
                child_key = (after, branch_context)
                before = after
                if child_key in seen:
                    continue
                seen.add(child_key)
                counter += 1
                heapq.heappush(
                    queue,
                    (
                        depth + 1,
                        counter,
                        child_path,
                        action,
                        branch_context,
                    ),
                )
        if expanded % 250 == 0:
            print("PROGRESS", expanded, len(seen), depth)
    return None, None, expanded, len(seen)


def run(env):
    for level in range(1, 5):
        getattr(players, f"play_level_{level}")(env)
    path, node, expanded, seen = search(env)
    print("SEARCH", list(path) if path else None, expanded, seen)
    if node is not None:
        print("PIECES", pieces(node))


levels, path, error = arena.run_program("sk48", run)
print("END", levels, len(path), error)
