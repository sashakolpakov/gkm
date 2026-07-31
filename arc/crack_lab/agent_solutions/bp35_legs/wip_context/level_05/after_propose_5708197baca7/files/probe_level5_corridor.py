import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

import legs
import players


ROWS = tuple(3 + 6 * i for i in range(10))
COLS = tuple(15 + 6 * j for j in range(8))
TO_FINAL = (
    ((4,),) * 4
    + ((6, 51, 39), (6, 45, 33), (4,), (6, 51, 33), (3,), (6, 51, 57))
    + ((3,), (3,), (6, 27, 33), (3,), (6, 21, 33), (3,), (3,))
    + ((4,),) * 4 + ((6, 39, 33),)
)


def avatar(node):
    frame = node.frame()
    for i, y in enumerate(ROWS):
        for j, x in enumerate(COLS):
            if int(frame[y][x]) in legs.AVATAR_COLORS:
                return i, j
    return None


def compact(node):
    frame = node.frame()
    specials = [
        (i, j, int(frame[y][x]))
        for i, y in enumerate(ROWS)
        for j, x in enumerate(COLS)
        if int(frame[y][x]) not in (3, 5, 9, 10, 11)
    ]
    return node.levels_completed, node.terminal(), avatar(node), specials


def run(root, path):
    node = root.clone()
    legs.run_actions(node, path)
    return node


def probe(env):
    for level in range(1, 5):
        getattr(players, f"play_level_{level}")(env)
    root = run(env, TO_FINAL)
    node = root
    print("root", compact(node))
    path = []
    for action in ((4,), (4,), (4,)):
        before = node.frame()
        node = run(node, (action,))
        path.append(action)
        print("right", len(path), "shift", legs.band_shift(before, node.frame()), compact(node))

    for label, action in (
        ("support", (6, 57, 33)),
        ("hazard", (6, 57, 27)),
        ("support", (6, 57, 33)),
    ):
        before = node.frame()
        node = run(node, (action,))
        path.append(action)
        print(label, "shift", legs.band_shift(before, node.frame()), compact(node))

    for _ in range(6):
        if node.levels_completed > 4 or node.terminal():
            break
        before = node.frame()
        node = run(node, ((3,),))
        path.append((3,))
        print("left", "shift", legs.band_shift(before, node.frame()), compact(node))
    print("path", path)


if __name__ == "__main__":
    A.run_program("bp35", probe)
