import json
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from legs import _bridge_carrier_state
from perception import arr


PREFIX = (
    (6, 13, 25), (6, 25, 25),
    3, 3, 2, 2, 2,
    (6, 25, 25), (6, 37, 25),
    4, 4, 4, 2, 2, 2,
    (6, 43, 25), (6, 31, 25),
)


def act(env, action):
    env.step(*action) if isinstance(action, tuple) else env.step(action)


def key(env):
    return arr(env.frame())[1:, :].tobytes()


def pegs(env):
    pixels = {tuple(point) for point in zip(*((arr(env.frame()) == 14).nonzero()))}
    out = []
    while pixels:
        seed = pixels.pop()
        stack = [seed]
        component = {seed}
        while stack:
            row, col = stack.pop()
            for point in (
                (row - 1, col), (row + 1, col),
                (row, col - 1), (row, col + 1),
            ):
                if point in pixels:
                    pixels.remove(point)
                    component.add(point)
                    stack.append(point)
        rows = [point[0] for point in component]
        cols = [point[1] for point in component]
        if max(rows) - min(rows) == 3 and max(cols) - min(cols) == 3:
            out.append((min(rows), min(cols)))
    return tuple(sorted(out))


def clicks(env):
    for row, col in pegs(env):
        for dr, dc in ((-12, 0), (12, 0), (0, -12), (0, 12)):
            dest = row + dr, col + dc
            if 0 <= dest[0] < 64 and 0 <= dest[1] < 64:
                yield (
                    (6, col + 1, row + 1),
                    (6, dest[1] + 1, dest[0] + 1),
                )


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    assert checkpoint["game"] == "lf52" and checkpoint["validated"]
    for action in checkpoint["final_path"]:
        act(env, tuple(action) if isinstance(action, list) else action)
    for action in PREFIX:
        act(env, action)

    base_level = env.levels_completed
    queue = deque([(env.clone(), ())])
    seen = {key(env)}
    transfers = []
    solution = None
    while queue and len(seen) < 240:
        node, path = queue.popleft()
        if len(path) >= 14:
            continue
        candidates = list(clicks(node)) + [(a,) for a in (1, 2, 3, 4)]
        for macro in candidates:
            child = node.clone()
            before_pegs = pegs(child)
            for action in macro:
                act(child, action)
            child_path = path + macro
            if child.levels_completed > base_level:
                solution = child_path
                queue.clear()
                break
            after = key(child)
            after_pegs = pegs(child)
            if len(macro) == 2 and after_pegs != before_pegs:
                transfers.append((path, macro, before_pegs, after_pegs))
            elif len(macro) == 2:
                continue
            if after not in seen:
                seen.add(after)
                queue.append((child, child_path))
        if solution is not None:
            break
    print("SEARCH", len(seen), "QUEUE", len(queue), "TRANSFERS", len(transfers))
    for item in transfers[:20]:
        print("TRANSFER", item)
    print("SOLUTION", solution)


levels, path, err = A.run_program("lf52", probe)
print("END", levels, len(path), err)
