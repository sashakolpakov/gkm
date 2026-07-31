import heapq
import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import arr


CONTROLS = (
    ("A<", (6, 4, 51)), ("A>", (6, 10, 51)), ("B", (6, 17, 51)),
    ("C<", (6, 26, 51)), ("C>", (6, 32, 51)), ("D", (6, 39, 51)),
    ("E<", (6, 54, 51)), ("E>", (6, 60, 51)),
    ("F<", (6, 4, 58)), ("F>", (6, 10, 58)), ("G", (6, 17, 58)),
    ("H<", (6, 26, 58)), ("H>", (6, 32, 58)), ("I", (6, 60, 58)),
)
LOOKUP = dict(CONTROLS)
PREFIX = (
    "F<", "A>", "B", "B", "A>", "F>", "A>", "C>", "A>", "C>",
    "A>", "G", "C>", "C>",
    "H>", "H>", "H>", "H>", "H>", "H>", "H>",
)
RINGS = {
    (6, 22), (7, 21), (7, 23), (8, 22),
    (15, 25), (16, 24), (16, 26), (17, 25),
}


def key(env):
    return arr(env.frame())[:48].tobytes()


def moving_marker(frame):
    grid = arr(frame)
    points = tuple(
        (int(r), int(c)) for r, c in zip(*((grid[:48] == 13).nonzero()))
        if (int(r), int(c)) not in RINGS and int(r) != 7
    )
    return points[0] if len(points) == 1 else (99, 99)


def distance(frame):
    r, c = moving_marker(frame)
    grid = arr(frame)
    top = tuple(
        int(c) for r, c in zip(*((grid[:10] == 13).nonzero()))
        if (int(r), int(c)) not in RINGS
    )
    top_penalty = abs(top[0] - 22) if len(top) == 1 else 20
    return abs(r - 16) + abs(c - 25) + top_penalty


def search(root, max_states=10000, max_depth=35):
    queue = [(distance(root.frame()), 0, 0, ())]
    seen = {key(root)}
    serial = 0
    best = distance(root.frame())
    while queue and len(seen) < max_states:
        _, depth, _, path = heapq.heappop(queue)
        node = root.clone()
        for index in path:
            node.step(*CONTROLS[index][1])
        if depth >= max_depth:
            continue
        for index, (name, action) in enumerate(CONTROLS):
            child = node.clone()
            before = key(child)
            child.step(*action)
            child_key = key(child)
            if child_key == before or child_key in seen:
                continue
            seen.add(child_key)
            child_path = path + (index,)
            child_distance = distance(child.frame())
            if child.levels_completed > 6:
                return child_path, len(seen)
            if child_distance < best:
                best = child_distance
                print(
                    "best", best, moving_marker(child.frame()),
                    [CONTROLS[i][0] for i in child_path],
                    flush=True,
                )
            serial += 1
            heapq.heappush(
                queue,
                (child_distance + (len(child_path) // 3),
                 len(child_path), serial, child_path),
            )
        if len(seen) % 1000 < len(CONTROLS):
            print("seen", len(seen), "queue", len(queue), "best", best,
                  flush=True)
    return None, len(seen)


def run(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    for name in PREFIX:
        env.step(*LOOKUP[name])
    path, states = search(env)
    print("states", states)
    print("path", None if path is None else [CONTROLS[i][0] for i in path])


arena.run_program("s5i5", run)
