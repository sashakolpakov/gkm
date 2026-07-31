import itertools
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import perception as P
import players


PLUS = [(6, 30, 50), (6, 25, 55), (6, 35, 55), (6, 30, 60)]
VERTICAL = [(6, 30, 50), (6, 30, 55), (6, 30, 60)]
COORDS = [(6, x, y) for y in (50, 55, 60) for x in (25, 30, 35)]


def apply(node, actions):
    for action in actions:
        node.step(*action) if isinstance(action, tuple) else node.step(action)
    return node


def remote(node):
    return tuple(
        (b.color, b.bbox, b.area)
        for b in P.connected_components(
            node.frame(), colors=(9, 10), min_area=1
        )
        if b.bbox[1] >= 47
    )


def probe(env):
    for k in (1, 2, 3):
        getattr(players, f"play_level_{k}")(env)
    grown = apply(
        env.clone(),
        [2, 2, 3, 3] + PLUS + [2] + [3] * 8
        + VERTICAL + [3, 3],
    )
    keyed_grown = apply(
        env.clone(),
        [2, 2, 4, 4] + PLUS + [2] + [3] * 16
        + VERTICAL + [3, 3] + PLUS,
    )
    print("GROWN", grown.levels_completed, grown.terminal(), remote(grown))
    frame = P.arr(grown.frame())
    for row in frame[32:42, 45:59]:
        print("CROP", "".join({2: "#", 5: ".", 9: "a", 10: "A"}.get(int(v), "?") for v in row))
    for action in env.actions:
        node = grown.clone()
        before = node.frame()
        node.step(action)
        print(
            "ACTION", action, node.levels_completed, node.terminal(),
            P.frame_delta(before, node.frame())["bbox"], remote(node)
        )
    print("KEYED_GROWN", keyed_grown.levels_completed, keyed_grown.terminal(), remote(keyed_grown))
    for action in env.actions:
        node = keyed_grown.clone()
        before = node.frame()
        node.step(action)
        print(
            "KEYED_ACTION", action, node.levels_completed, node.terminal(),
            P.frame_delta(before, node.frame())["bbox"], remote(node)
        )
    base_remote = remote(grown)
    changes = []
    wins = []
    for length in (1, 2, 3, 4, 5, 6, 7, 8, 9):
        for combo in itertools.combinations(COORDS, length):
            node = grown.clone()
            for action in combo:
                if node.terminal():
                    break
                node.step(*action)
            if remote(node) != base_remote:
                changes.append((combo, remote(node)))
            if node.levels_completed > env.levels_completed:
                wins.append(combo)
    print("REMOTE_CHANGES", changes)
    print("WINS", wins)


levels, path, err = A.run_program("sc25", probe)
print("DONE", levels, len(path), err)
