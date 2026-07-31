import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena
import numpy as np

import players


CLICKS = ((49, 7), (28, 22), (7, 40), (37, 46))


def selected_signature(frame):
    grid = np.asarray(frame)
    points = set(zip(*np.where(~np.isin(grid, (4, 9)))))
    if not points:
        return (), {}
    r0 = min(row for row, _ in points)
    c0 = min(col for _, col in points)
    normalized = tuple(sorted((row - r0, col - c0) for row, col in points))
    values, counts = np.unique(grid, return_counts=True)
    colors = {
        int(value): int(count)
        for value, count in zip(values, counts)
        if int(value) not in (4, 9)
    }
    return normalized, colors


def probe(env):
    while env.levels_completed < 5:
        getattr(players, f"play_level_{env.levels_completed + 1}")(env)
    roots = [(None, env.clone())]
    for click in CLICKS:
        node = env.clone()
        node.step(6, *click)
        roots.append((click, node))
    for click, root in roots:
        seen = {}
        print("PIECE", click)
        for turns in range(9):
            signature, colors = selected_signature(root.frame())
            repeat = seen.get(signature)
            print(
                turns,
                "pixels",
                len(signature),
                "bbox",
                (
                    max((r for r, _ in signature), default=-1) + 1,
                    max((c for _, c in signature), default=-1) + 1,
                ),
                "colors",
                colors,
                "repeat",
                repeat,
            )
            seen.setdefault(signature, turns)
            root.step(5)


arena.run_program("cn04", probe)
