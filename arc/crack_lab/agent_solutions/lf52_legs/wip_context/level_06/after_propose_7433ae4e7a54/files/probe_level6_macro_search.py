import sys
import time
from heapq import heappop, heappush

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

import perception as P
from probe_level6_lower import brief, click, pieces
from probe_level6_lower_route import stage_remote


def apply(node, action):
    if isinstance(action, int):
        node.step(action)
    else:
        _, source, destination = action
        click(node, source)
        click(node, destination)


def search(root, max_states=1200, max_depth=55, max_seconds=150):
    base_level = root.levels_completed
    started = time.monotonic()

    def reconstruct(path):
        node = root.clone()
        for action in path:
            apply(node, action)
        return node

    def key(node):
        return P.arr(node.frame())[1:].tobytes()

    queue = [(0, 0, 0, (), 0)]
    serial = 0
    seen = {key(root)}
    best_captures = 0
    while queue and len(seen) < max_states:
        if time.monotonic() - started > max_seconds:
            break
        _, depth, _, path, captures = heappop(queue)
        if depth >= max_depth:
            continue
        node = reconstruct(path)
        _, _, movable, pegs, static = pieces(node.frame())
        observed = brief(node)[3]
        actions = tuple(("macro", source, destination) for source, _, destination in observed)
        actions += (1, 2, 3, 4)
        for action in actions:
            child_path = path + (action,)
            child = reconstruct(child_path)
            if child.levels_completed > base_level:
                return child_path, len(seen), captures
            child_key = key(child)
            if child_key in seen:
                continue
            seen.add(child_key)
            child_captures = captures
            if not isinstance(action, int):
                _, source, destination = action
                midpoint = (
                    (source[0] + destination[0]) // 2,
                    (source[1] + destination[1]) // 2,
                )
                if source in pegs and midpoint in pegs:
                    child_captures += 1
                    if child_captures > best_captures:
                        best_captures = child_captures
                        print(
                            "capture progress", best_captures,
                            "depth", depth + 1, "path", child_path,
                            flush=True,
                        )
            next_macros = brief(child)[3]
            next_pegs = pieces(child.frame())[3]
            capture_ready = any(
                source in next_pegs
                and (
                    (source[0] + destination[0]) // 2,
                    (source[1] + destination[1]) // 2,
                ) in next_pegs
                for source, _, destination in next_macros
            )
            serial += 1
            priority = (
                -100 * child_captures
                - 20 * int(capture_ready)
                - 3 * len(next_macros)
                + depth + 1
            )
            heappush(
                queue,
                (priority, depth + 1, serial, child_path, child_captures),
            )
    return None, len(seen), best_captures


def probe(env):
    stage_remote(env)
    path, states, captures = search(env)
    print("macro search", path, "states", states, "captures", captures)
    if path is not None:
        base_level = env.levels_completed
        for action in path:
            apply(env, action)
        print("reward", env.levels_completed - base_level)


A.run_program("lf52", probe)
