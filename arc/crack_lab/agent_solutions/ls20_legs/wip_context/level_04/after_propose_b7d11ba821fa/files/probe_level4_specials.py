import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena
import numpy as np

import perception
import players


ROUTES = {
    "left": (3, 3, 3, 2, 2, 2, 4, 1, 2, 3, 3, 1, 1, 1),
    "right": (3, 3, 3, 2, 2, 2, 3, 2, 2, 3, 3, 1, 2),
}


def rows(frame, row_slice, col_slice):
    return tuple(
        "".join(f"{int(value):X}" for value in row)
        for row in np.asarray(frame)[row_slice, col_slice]
    )


def tile(frame, row, col):
    return rows(frame, slice(5 * row, 5 * row + 5), slice(5 * col - 1, 5 * col + 4))


def glyph(frame):
    sampled = np.asarray(frame)[55:61:2, 3:9:2]
    return tuple("".join(f"{int(value):X}" for value in row) for row in sampled)


def replay(root, path):
    clone = root.clone()
    for action in path:
        clone.step(action)
    return clone


def probe(env):
    players.play_level_1(env)
    players.play_level_2(env)
    players.play_level_3(env)
    root = env.clone()
    print("initial", "glyph", glyph(root.frame()))
    for name, route in ROUTES.items():
        node = replay(root, route)
        print(
            name,
            "path",
            route,
            "glyph",
            glyph(node.frame()),
            "left_tile",
            tile(node.frame(), 6, 5),
            "right_tile",
            tile(node.frame(), 6, 7),
        )
        before = np.asarray(node.frame()).copy()
        for action in node.actions:
            child = node.clone()
            child.step(int(action))
            delta = perception.frame_delta(before, child.frame())
            print(
                name,
                "action",
                int(action),
                "level",
                int(child.levels_completed),
                "delta",
                delta["count"],
                delta["bbox"],
                "glyph",
                glyph(child.frame()),
                "left_tile",
                tile(child.frame(), 6, 5),
                "right_tile",
                tile(child.frame(), 6, 7),
            )
        cycle = node.clone()
        leave, enter = ((2, 1) if name == "left" else (1, 2))
        for contact in range(2, 9):
            cycle.step(leave)
            cycle.step(enter)
            print(
                name,
                "contact",
                contact,
                "level",
                int(cycle.levels_completed),
                "glyph",
                glyph(cycle.frame()),
            )
            if int(cycle.levels_completed) > 3:
                break


arena.run_program("ls20", probe)
