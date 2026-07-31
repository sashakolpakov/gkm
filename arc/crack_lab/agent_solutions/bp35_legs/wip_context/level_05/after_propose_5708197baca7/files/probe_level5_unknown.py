import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

import legs
import perception as P
import players


ROWS = tuple(3 + 6 * i for i in range(10))
COLS = tuple(15 + 6 * j for j in range(8))
PHASE_FOUR = (
    ((4,),) * 4
    + ((6, 51, 39), (6, 45, 33), (4,), (6, 51, 33), (3,), (6, 51, 57))
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


def result(node):
    return {
        "terminal": node.terminal(),
        "level": node.levels_completed,
        "moves": legs.moves_used(node.frame()),
        "avatar": avatar_cell(node),
    }


def probe(env):
    if not reach_level_five(env):
        print("advance_failed", env.levels_completed)
        return
    root = run(env, PHASE_FOUR)
    known = {0, 3, 5, 7, 8, 9, 10, 11, 14, 15}
    unknown = [
        (i, j, int(root.frame()[y][x]))
        for i, y in enumerate(ROWS)
        for j, x in enumerate(COLS)
        if int(root.frame()[y][x]) not in known
    ]
    print("unknown_cells", unknown, "root", result(root))

    for lefts in (0, 1, 2):
        positioned = run(root, ((3,),) * lefts)
        for target in ((5, 1), (5, 2)):
            action = (6, COLS[target[1]], ROWS[target[0]])
            for clicks in (1, 2, 3):
                child = run(positioned, (action,) * clicks)
                crossed = run(child, ((3,),))
                delta = P.frame_delta(positioned.frame(), child.frame())
                print(
                    "test",
                    {"lefts": lefts, "target": target, "clicks": clicks},
                    "click",
                    result(child),
                    "delta",
                    (delta["count"], delta["bbox"]),
                    "then_left",
                    result(crossed),
                )


if __name__ == "__main__":
    A.run_program("bp35", probe)
