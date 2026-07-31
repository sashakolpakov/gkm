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
    14: "Y",
    15: "H",
}


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


def grid(frame):
    return "/".join(
        "".join(SYMBOL.get(int(frame[y][x]), "?") for x in COLS)
        for y in ROWS
    )


def summary(node):
    frame = node.frame()
    return {
        "terminal": node.terminal(),
        "level": node.levels_completed,
        "moves": legs.moves_used(frame),
        "avatar": legs.avatar_column(frame),
        "grid": grid(frame),
    }


def probe(env):
    if not reach_level_five(env):
        print("advance_failed", env.levels_completed)
        return
    root = env.clone()
    paths = [
        (),
        ((4,),) * 4 + ((6, 51, 39),),
        ((4,),) * 4 + ((6, 51, 39), (6, 45, 33)),
        ((4,),) * 4 + ((6, 51, 39), (6, 45, 33), (4,), (6, 51, 33)),
        ((4,),) * 4
        + ((6, 51, 39), (6, 45, 33), (4,), (6, 51, 33), (3,), (6, 51, 57)),
    ]
    states = []
    for index, path in enumerate(paths):
        node = run(root, path)
        states.append(node)
        shift = 0 if index == 0 else legs.band_shift(states[index - 1].frame(), node.frame())
        print("phase", index, "shift", shift, summary(node))

    phase_four = states[-1]
    for count in range(1, 7):
        child = run(phase_four, ((3,),) * count)
        print("left", count, summary(child))

    for prefix in ((), ((3,),), ((3,),) * 2, ((3,),) * 3):
        positioned = run(phase_four, prefix)
        before = positioned.frame()
        for i, y in enumerate(ROWS):
            for j, x in enumerate(COLS):
                color = int(before[y][x])
                if color not in (7, 8, 14, 15):
                    continue
                child = run(positioned, ((6, x, y),))
                delta = P.frame_delta(before, child.frame())
                if child.terminal() or legs.band_shift(before, child.frame()) or delta["count"] > 25:
                    print(
                        "frontier_click",
                        len(prefix),
                        (i, j, color),
                        "shift",
                        legs.band_shift(before, child.frame()),
                        "delta",
                        (delta["count"], delta["bbox"]),
                        summary(child),
                    )


if __name__ == "__main__":
    A.run_program("bp35", probe)
