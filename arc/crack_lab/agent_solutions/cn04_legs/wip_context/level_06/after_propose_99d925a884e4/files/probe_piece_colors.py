import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena
import numpy as np

import players

LEVEL_5_PREWIN = (
    [5] * 4 + [1] * 3 + [4] * 6 + [(6, 54, 6)] + [5] * 3
    + [(6, 5, 38)] + [1] * 10 + [4] * 10
    + [(6, 47, 47)] + [5] * 3 + [1] * 10 + [4]
)


def summary(frame):
    a = np.asarray(frame)
    bg = int(np.bincount(a.ravel()).argmax())
    colored = []
    for r in range(3, 64, 3):
        for c in range(0, 64, 3):
            value = int(a[r, c])
            if value not in (bg, 4):
                colored.append((r // 3, c // 3, value))
    if not colored:
        return []
    r0 = min(r for r, _, _ in colored)
    c0 = min(c for _, c, _ in colored)
    return [(r - r0, c - c0, value) for r, c, value in colored]


def absolute_summary(frame):
    a = np.asarray(frame)
    bg = int(np.bincount(a.ravel()).argmax())
    return [
        (r // 3, c // 3, int(a[r, c]))
        for r in range(3, 64, 3)
        for c in range(0, 64, 3)
        if int(a[r, c]) not in (bg, 4)
    ]


def selection_roots(env):
    roots = {np.asarray(env.frame()).tobytes(): (None, env.clone())}
    frame = np.asarray(env.frame())
    bg = int(np.bincount(frame.ravel()).argmax())
    for row in range(1, 64, 3):
        for col in range(1, 64, 3):
            if int(frame[row, col]) in (bg, 0):
                continue
            node = env.clone()
            node.step(6, col, row)
            roots.setdefault(np.asarray(node.frame()).tobytes(), ((col, row), node))
    return roots


def probe(env):
    while env.levels_completed < 4:
        getattr(players, f"play_level_{env.levels_completed + 1}")(env)
    print("L5_INITIAL_SELECTIONS")
    for click, node in selection_roots(env).values():
        print("PIECE", click, summary(node.frame()))
    prior = env.clone()
    for action in LEVEL_5_PREWIN:
        prior.step(*action) if isinstance(action, tuple) else prior.step(action)
    print("L5_PREWIN_SELECTIONS")
    for click, node in selection_roots(prior).values():
        print("PIECE", click, absolute_summary(node.frame()))
    players.play_level_5(env)
    print("L6_INITIAL_SELECTIONS")
    for click, node in selection_roots(env).values():
        print("PIECE", click, summary(node.frame()))
        turned = node.clone()
        turned.step(5)
        print(" TURN1", summary(turned.frame()))


arena.run_program("cn04", probe)
