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


def search(root, max_states=1200, max_depth=100, max_seconds=600):
    base_level = root.levels_completed
    started = time.monotonic()

    def key(node):
        return P.arr(node.frame())[1:].tobytes()

    serial = 0
    queue = [(0, 0, serial, root.clone(), (), 0)]
    seen = {key(root)}
    best_captures = 0
    best_pair_gap = 10 ** 9
    while queue and len(seen) < max_states:
        if time.monotonic() - started > max_seconds:
            break
        _, depth, _, node, path, captures = heappop(queue)
        if depth >= max_depth:
            continue
        _, _, _, pegs, _ = pieces(node.frame())
        observed = brief(node)[3]
        actions = tuple(("macro", source, destination) for source, _, destination in observed)
        actions += (1, 2, 3, 4)
        for action in actions:
            child = node.clone()
            apply(child, action)
            child_path = path + (action,)
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
            ordered_pegs = sorted(next_pegs)
            pair_gap = min(
                (
                    abs(first[0] - second[0])
                    + abs(first[1] - second[1])
                    + (
                        0 if (
                            first[0] == second[0]
                            or first[1] == second[1]
                        ) else 40
                    )
                )
                for index, first in enumerate(ordered_pegs)
                for second in ordered_pegs[index + 1:]
            ) if len(ordered_pegs) >= 2 else 10 ** 6
            if pair_gap < best_pair_gap:
                best_pair_gap = pair_gap
                print(
                    "pair progress", best_pair_gap,
                    "depth", depth + 1, "path", child_path,
                    flush=True,
                )
            serial += 1
            priority = (
                -100 * child_captures
                - 30 * int(capture_ready)
                - 8 * len(next_pegs)
                - 3 * len(next_macros)
                + pair_gap
                + depth + 1
            )
            heappush(
                queue,
                (
                    priority, depth + 1, serial, child, child_path,
                    child_captures,
                ),
            )
    return None, len(seen), best_captures


def probe(env):
    stage_remote(env)
    for action in (1, 1, 3, 3, 1, 1):
        env.step(action)
    click(env, (30, 28))
    click(env, (18, 28))
    click(env, (18, 28))
    click(env, (18, 40))
    print("immediate capture", brief(env))
    path, states, captures = search(env)
    print("after upper search", path, "states", states, "captures", captures)
    if path is not None:
        base_level = env.levels_completed
        for action in path:
            apply(env, action)
        print("reward", env.levels_completed - base_level)


A.run_program("lf52", probe)
