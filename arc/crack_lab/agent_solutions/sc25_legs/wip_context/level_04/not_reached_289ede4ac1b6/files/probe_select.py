import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import perception as P
import players


PLUS = [(6, 30, 50), (6, 25, 55), (6, 35, 55), (6, 30, 60)]
VERTICAL = [(6, 30, 50), (6, 30, 55), (6, 30, 60)]


def apply(node, actions):
    for action in actions:
        node.step(*action) if isinstance(action, tuple) else node.step(action)


def parts(node):
    return tuple(
        (b.color, b.bbox, b.area)
        for b in P.connected_components(
            node.frame(), colors=(6, 9, 10, 13, 14), min_area=1
        )
        if b.color != 14 or b.bbox[1] < 60
    )


def probe(env):
    for k in (1, 2, 3):
        getattr(players, f"play_level_{k}")(env)
    clicks = {
        "green": (6, 8, 29),
        "green_ref": (6, 5, 15),
        "blue": (6, 48, 37),
        "blue_inset": (6, 9, 29),
        "marker": (6, 44, 28),
        "avatar": (6, 36, 20),
        "target": (6, 53, 37),
    }
    for name, click in clicks.items():
        node = env.clone()
        before = node.frame()
        node.step(*click)
        print(
            "CLICK", name, P.frame_delta(before, node.frame()),
            parts(node)
        )
        for command_name, command in {"V": VERTICAL, "X": PLUS}.items():
            test = node.clone()
            apply(test, command)
            print(
                "COMMAND", name, command_name,
                test.levels_completed, parts(test)
            )


levels, path, err = A.run_program("sc25", probe)
print("DONE", levels, len(path), err)
