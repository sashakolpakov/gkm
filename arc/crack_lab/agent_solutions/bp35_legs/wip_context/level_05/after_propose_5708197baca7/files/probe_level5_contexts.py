import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

import legs
import perception as P
import players


ROWS = tuple(3 + 6 * i for i in range(10))
COLS = tuple(15 + 6 * j for j in range(8))
SYMBOL = {3: "#", 5: "#", 7: "B", 8: "G", 9: "A", 10: ".", 11: "A", 14: "Y", 15: "H"}


def reach_level_five(env):
    for level in range(1, 4):
        getattr(players, f"play_level_{level}")(env)
        if env.levels_completed != level:
            return False
    path = legs.gravity_room_search(env)
    print("level4_path", path)
    legs.run_actions(env, path)
    if env.levels_completed != 4:
        return False
    return True


def grid(frame):
    return tuple(
        "".join(SYMBOL.get(int(frame[y][x]), "?") for x in COLS)
        for y in ROWS
    )


def compact(env):
    frame = env.frame()
    counts = P.color_counts(frame)
    objects = P.connected_components(frame, colors=(7, 8, 9, 11, 14, 15), min_area=4)
    return {
        "level": env.levels_completed,
        "terminal": env.terminal(),
        "special_counts": {c: counts.get(c, 0) for c in (7, 8, 9, 11, 14, 15)},
        "objects": [(b.color, b.bbox, b.area) for b in objects],
        "grid": grid(frame),
    }


def run(root, path):
    node = root.clone()
    for action in path:
        node.step(*action)
        if node.terminal():
            break
    return node


def affordances(node, label):
    before = P.color_counts(node.frame())
    print(label, compact(node))
    for i, y in enumerate(ROWS):
        for j, x in enumerate(COLS):
            color = int(node.frame()[y][x])
            if color not in (7, 8, 14, 15):
                continue
            child = run(node, [(6, x, y)])
            after = P.color_counts(child.frame())
            changes = {
                c: (before.get(c, 0), after.get(c, 0))
                for c in (7, 8, 9, 11, 14, 15)
                if before.get(c, 0) != after.get(c, 0)
            }
            print(
                label,
                "click",
                (i, j, color),
                "terminal",
                child.terminal(),
                "changes",
                changes,
            )


def probe(env):
    if not reach_level_five(env):
        print("advance_failed", env.levels_completed)
        return
    root = env.clone()
    toggle = (6, 51, 39)
    post = run(root, [(4,)] * 4 + [toggle])
    phase_two = run(post, [(6, 45, 33)])
    phase_three = run(phase_two, [(4,), (6, 51, 33)])
    phase_four = run(phase_three, [(3,), (6, 51, 57)])
    affordances(phase_four, "phase_four")
    affordances(run(phase_four, [(3,)]), "phase_four_left")
    affordances(run(phase_four, [(4,)]), "phase_four_right")


if __name__ == "__main__":
    A.run_program("bp35", probe)
