import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

import legs
import perception as P
import players


ROWS = tuple(3 + 6 * i for i in range(10))
COLS = tuple(15 + 6 * j for j in range(8))
SYMBOL = {
    0: "0",
    3: "#",
    5: "#",
    7: "P",
    8: "G",
    9: "A",
    10: ".",
    11: "a",
    12: "X",
    13: "x",
    14: "Y",
    15: "H",
}
TO_STAGE_THREE = (
    ((4,),) * 4
    + ((6, 51, 39), (6, 45, 33), (4,), (6, 51, 33), (3,), (6, 51, 57))
    + ((3,), (3,), (6, 27, 33), (3,), (6, 21, 33), (3,), (3,))
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


def avatar_cell(node):
    frame = node.frame()
    for i, y in enumerate(ROWS):
        for j, x in enumerate(COLS):
            if int(frame[y][x]) in legs.AVATAR_COLORS:
                return i, j
    return None


def summary(node):
    frame = node.frame()
    return {
        "terminal": node.terminal(),
        "level": node.levels_completed,
        "moves": legs.moves_used(frame),
        "avatar": avatar_cell(node),
        "grid": "/".join(
            "".join(SYMBOL.get(int(frame[y][x]), "?") for x in COLS)
            for y in ROWS
        ),
    }


def probe(env):
    if not reach_level_five(env):
        print("advance_failed", env.levels_completed)
        return
    stage_three = run(env, TO_STAGE_THREE)
    candidate = ((4,),) * 4 + ((6, 39, 33),)
    frontier = run(stage_three, candidate)
    print(
        "aligned_support",
        "shift",
        legs.band_shift(stage_three.frame(), frontier.frame()),
        "before",
        summary(stage_three),
        "after",
        summary(frontier),
    )
    before = frontier.frame()
    avatar = avatar_cell(frontier)
    for action in ((3,), (4,)):
        child = run(frontier, (action,))
        print("move", action, "shift", legs.band_shift(before, child.frame()), summary(child))
    if avatar is None:
        return
    ai, aj = avatar
    for i, y in enumerate(ROWS):
        for j, x in enumerate(COLS):
            color = int(before[y][x])
            if color in legs.OPEN_COLORS + legs.WALL_COLORS + legs.AVATAR_COLORS:
                continue
            if abs(i - ai) > 1 or abs(j - aj) > 1:
                continue
            child = run(frontier, ((6, x, y),))
            delta = P.frame_delta(before, child.frame())
            print(
                "local",
                (i, j, color),
                "shift",
                legs.band_shift(before, child.frame()),
                "delta",
                (delta["count"], delta["bbox"]),
                summary(child),
            )


if __name__ == "__main__":
    A.run_program("bp35", probe)
