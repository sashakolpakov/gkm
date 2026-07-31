import heapq
import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import arr, frame_delta


BUTTONS = {
    "A<": (6, 4, 51), "A>": (6, 10, 51), "B": (6, 17, 51),
    "C<": (6, 26, 51), "C>": (6, 32, 51), "D": (6, 39, 51),
    "F<": (6, 4, 58), "F>": (6, 10, 58), "G": (6, 17, 58),
    "H<": (6, 26, 58), "H>": (6, 32, 58),
}
ROTATIONS = {"B", "D", "G"}
MACROS = tuple(BUTTONS)
PREFIX = (
    "F<", "A>", "B", "B", "A>", "F>", "A>", "C>", "A>", "C>",
    "A>", "D",
    "H>", "H>", "H>", "H>", "H>",
    "H<", "H<", "H<", "H<", "H<",
    "C<", "C<", "C<",
    "F>", "F>", "F>", "F>",
    "D", "D",
    "C>", "C>", "C>", "C>",
)
RINGS = {
    (6, 22), (7, 21), (7, 23), (8, 22),
    (15, 25), (16, 24), (16, 26), (17, 25),
}


def key(env):
    return arr(env.frame())[:42].tobytes()


def lower_marker(frame):
    grid = arr(frame)[:42]
    points = tuple(
        (int(r), int(c)) for r, c in zip(*((grid == 13).nonzero()))
        if (int(r), int(c)) not in RINGS and (int(r), int(c)) != (7, 22)
    )
    return points[0] if len(points) == 1 else (99, 99)


def score(frame):
    marker = lower_marker(frame)
    if 99 in marker:
        return 999
    return abs(marker[0] - 16) + abs(marker[1] - 25)


def apply_macro(env, name):
    presses = 0
    limit = 1 if name in ROTATIONS else 16
    for _ in range(limit):
        before = env.frame()
        env.step(*BUTTONS[name])
        changed = frame_delta(before, env.frame())["count"]
        if changed <= 1:
            break
        presses += 1
        if env.levels_completed > 6:
            break
    return presses


def search(root, max_states=12000, max_depth=28):
    serial = 0
    queue = [(score(root.frame()) // 8, 0, serial, root.clone(), ())]
    seen = {key(root)}
    best = score(root.frame())
    while queue and len(seen) < max_states:
        _, depth, _, node, path = heapq.heappop(queue)
        if depth >= max_depth:
            continue
        for name in MACROS:
            child = node.clone()
            presses = apply_macro(child, name)
            child_key = key(child)
            if presses == 0 or child_key in seen:
                continue
            seen.add(child_key)
            child_path = path + ((name, presses),)
            if child.levels_completed > 6:
                return child_path, len(seen)
            child_score = score(child.frame())
            if child_score < best:
                best = child_score
                print("best", best, lower_marker(child.frame()), child_path, flush=True)
            serial += 1
            heapq.heappush(
                queue,
                (depth + 1 + child_score // 8, depth + 1, serial,
                 child, child_path),
            )
    return None, len(seen)


def run(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    for name in PREFIX:
        env.step(*BUTTONS[name])
    print("root", lower_marker(env.frame()), flush=True)
    path, states = search(env)
    print("states", states)
    print("path", path)


arena.run_program("s5i5", run)
