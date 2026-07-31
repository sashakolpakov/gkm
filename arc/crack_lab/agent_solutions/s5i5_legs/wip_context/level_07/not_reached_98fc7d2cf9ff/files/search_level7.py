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
RINGS = {
    (6, 22), (7, 21), (7, 23), (8, 22),
    (15, 25), (16, 24), (16, 26), (17, 25),
}
GOALS = ((7, 22), (16, 25))


def key(env):
    return arr(env.frame())[:48].tobytes()


def markers(frame):
    grid = arr(frame)
    return tuple(
        (int(r), int(c)) for r, c in zip(*((grid[:48] == 13).nonzero()))
        if (int(r), int(c)) not in RINGS
    )


def score(frame):
    points = markers(frame)
    if len(points) != 2:
        return 999
    direct = sum(abs(a - c) + abs(b - d)
                 for (a, b), (c, d) in zip(points, GOALS))
    crossed = sum(abs(a - c) + abs(b - d)
                  for (a, b), (c, d) in zip(points, reversed(GOALS)))
    return min(direct, crossed)


def search(root, max_states=30000, max_depth=80):
    serial = 0
    start_score = score(root.frame())
    queue = [((start_score + 2) // 3, start_score, serial, ())]
    seen = {key(root)}
    best = start_score
    while queue and len(seen) < max_states:
        _, _, _, path = heapq.heappop(queue)
        node = root.clone()
        for index in path:
            node.step(*CONTROLS[index][1])
        depth = len(path)
        if depth >= max_depth:
            continue
        for index, (_, action) in enumerate(CONTROLS):
            child = node.clone()
            child.step(*action)
            child_key = key(child)
            if child_key in seen:
                continue
            seen.add(child_key)
            child_path = path + (index,)
            if child.levels_completed > 6:
                return child_path, len(seen), best
            child_score = score(child.frame())
            if child_score < best:
                best = child_score
                print("best", best, markers(child.frame()),
                      [CONTROLS[i][0] for i in child_path])
            serial += 1
            heapq.heappush(
                queue,
                (len(child_path) + (child_score + 2) // 3,
                 child_score, serial, child_path),
            )
            if len(seen) % 1000 == 0:
                print("seen", len(seen), "frontier", len(queue),
                      "depth", len(child_path), "best", best)
    return None, len(seen), best


def run(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    path, states, best = search(env)
    print("states", states, "best", best)
    print("path", None if path is None else [CONTROLS[i][0] for i in path])


arena.run_program("s5i5", run)
