import json
import sys
from heapq import heappop, heappush

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

import perception as P


LOCAL_MACROS = (
    ((18, 18), (30, 18)),
    ((24, 18), (36, 18)),
    ((30, 18), (42, 18)),
    ((36, 18), (48, 18)),
    ((48, 12), (48, 24)),
    ((54, 24), (42, 24)),
    ((42, 18), (42, 30)),
    ((42, 24), (42, 36)),
    ((42, 30), (42, 42)),
    ((42, 36), (42, 48)),
)


def click(env, cell):
    env.step(6, cell[1] + 1, cell[0] + 1)


def pieces(frame):
    blobs = P.connected_components(frame, colors=(1, 8, 12, 14))
    holes = {
        blob.top_left
        for blob in blobs
        if blob.color == 1 and blob.size == (4, 4) and blob.area == 16
    }
    carriers = {
        blob.top_left
        for blob in blobs
        if blob.color == 12 and blob.size == (4, 4)
    }
    bridges = {
        blob.top_left
        for blob in blobs
        if blob.color == 8 and blob.size == (4, 4)
    }
    pegs = {
        blob.top_left
        for blob in blobs
        if blob.color == 14 and blob.size == (4, 4)
    }
    return holes, carriers, bridges, pegs


def macros(frame):
    holes, carriers, bridges, pegs = pieces(frame)
    occupied = bridges | pegs
    out = []
    for source in sorted(occupied):
        for destination in sorted(holes | carriers):
            dr = destination[0] - source[0]
            dc = destination[1] - source[1]
            if (abs(dr), abs(dc)) not in ((12, 0), (0, 12)):
                continue
            midpoint = (source[0] + dr // 2, source[1] + dc // 2)
            if midpoint not in occupied:
                continue
            if source in bridges and midpoint not in pegs:
                continue
            out.append((source, destination))
    return tuple(out)


def stage(env):
    with open("checkpoint.json") as handle:
        prefix = json.load(handle)["final_path"]
    for action in prefix:
        env.step(*(action if isinstance(action, list) else [action]))
    for source, destination in LOCAL_MACROS:
        click(env, source)
        click(env, destination)
    _, carriers, bridges, _ = pieces(env.frame())
    click(env, next(iter(bridges)))
    click(env, next(iter(carriers)))
    for _ in range(9):
        env.step(4)


def frame_key(env, hidden_suffix):
    frame = P.arr(env.frame())
    visible_bridge = bool(pieces(frame)[2])
    suffix = () if visible_bridge else hidden_suffix[-8:]
    return frame[1:].tobytes(), suffix


def search(root, max_states=900, max_depth=55):
    base_level = root.levels_completed
    serial = 0
    queue = [(0, 0, serial, root, (), (), 0)]
    seen = {frame_key(root, ())}
    best_captures = 0
    while queue and len(seen) < max_states:
        _, depth, _, node, path, suffix, captures = heappop(queue)
        available = macros(node.frame())
        actions = tuple(("macro", macro) for macro in available)
        actions += tuple(("key", action) for action in (1, 2, 3, 4))
        for action_kind, action in actions:
            child = node.clone()
            before_pegs = len(pieces(child.frame())[3])
            if action_kind == "macro":
                source, destination = action
                click(child, source)
                click(child, destination)
                child_path = path + (
                    (6, source[1] + 1, source[0] + 1),
                    (6, destination[1] + 1, destination[0] + 1),
                )
                child_suffix = ()
            else:
                child.step(action)
                child_path = path + (action,)
                child_suffix = (suffix + (action,))[-8:]
            if child.levels_completed > base_level:
                return child_path, len(seen), captures
            child_depth = depth + 1
            if child_depth >= max_depth:
                continue
            after_pegs = len(pieces(child.frame())[3])
            child_captures = captures
            if action_kind == "macro" and after_pegs < before_pegs:
                child_captures += before_pegs - after_pegs
                if child_captures > best_captures:
                    best_captures = child_captures
                    print(
                        "progress", best_captures,
                        "depth", child_depth,
                        "path", child_path,
                    )
            key = frame_key(child, child_suffix)
            if key in seen:
                continue
            seen.add(key)
            serial += 1
            has_macro = bool(macros(child.frame()))
            priority = -20 * child_captures - 3 * has_macro
            heappush(
                queue,
                (
                    priority, child_depth, serial, child, child_path,
                    child_suffix, child_captures,
                ),
            )
    return None, len(seen), best_captures


def probe(env):
    stage(env)
    path, states, captures = search(env.clone())
    print("search result", path, "states", states, "captures", captures)
    if path:
        clone = env.clone()
        for action in path:
            clone.step(*action) if isinstance(action, tuple) else clone.step(action)
        print("reward", clone.levels_completed - env.levels_completed)


A.run_program("lf52", probe)
