import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

import legs
import perception as P
import players


ROWS = tuple(3 + 6 * i for i in range(10))
COLS = tuple(15 + 6 * j for j in range(8))
TO_FINAL = (
    ((4,),) * 4
    + ((6, 51, 39), (6, 45, 33), (4,), (6, 51, 33), (3,), (6, 51, 57))
    + ((3,), (3,), (6, 27, 33), (3,), (6, 21, 33), (3,), (3,))
    + ((4,),) * 4 + ((6, 39, 33),)
)


def reach_level_five(env):
    for level in range(1, 5):
        getattr(players, f"play_level_{level}")(env)
        if env.levels_completed != level:
            return False
    return True


def run(root, path):
    node = root.clone()
    legs.run_actions(node, path)
    return node


def avatar(node):
    frame = node.frame()
    for i, y in enumerate(ROWS):
        for j, x in enumerate(COLS):
            if int(frame[y][x]) in legs.AVATAR_COLORS:
                return i, j
    return None


def compact(node):
    return (node.levels_completed, node.terminal(), legs.moves_used(node.frame()), avatar(node))


def choices(node):
    frame = node.frame()
    out = [(3,), (4,)]
    for y in ROWS:
        for x in COLS:
            if int(frame[y][x]) not in (3, 5, 9, 10, 11):
                out.append((6, x, y))
    return out


def probe(env):
    if not reach_level_five(env):
        print("advance_failed", env.levels_completed)
        return
    root = run(env, TO_FINAL)
    print("root", compact(root))
    before = root.frame()
    for action in choices(root):
        child = run(root, (action,))
        delta = P.frame_delta(before, child.frame())
        print("one", action, compact(child), "shift", legs.band_shift(before, child.frame()),
              "delta", (delta["count"], delta["bbox"]))

    queue = deque([()])
    seen = {bytes(int(v) for row in before for v in row)}
    expanded = 0
    while queue and expanded < 2500:
        path = queue.popleft()
        node = run(root, path)
        expanded += 1
        if len(path) >= 20 or node.terminal():
            continue
        for action in choices(node):
            child = node.clone()
            child.step(*action)
            new_path = path + (action,)
            if child.levels_completed > root.levels_completed:
                print("SOLVED", new_path, compact(child), "expanded", expanded)
                return
            if child.terminal() or avatar(child) is None:
                continue
            key = bytes(int(v) for row in child.frame() for v in row)
            if key in seen:
                continue
            seen.add(key)
            queue.append(new_path)
    print("NO_PATH", expanded, len(seen))


if __name__ == "__main__":
    A.run_program("bp35", probe)
