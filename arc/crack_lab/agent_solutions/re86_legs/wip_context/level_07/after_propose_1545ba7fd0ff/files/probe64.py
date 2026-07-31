import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import DOWN, RIGHT, UP, USE, arr


PATH = json.load(open("checkpoint.json"))["final_path"]
TARGETS = ((30, 45), (48, 39), (48, 51))


def points(node):
    frame = arr(node.frame())
    return {
        (int(row), int(col))
        for row, col in zip(*((frame == 12).nonzero()))
    }


def score(node):
    shape = points(node)
    distances = tuple(
        min(abs(row - tr) + abs(col - tc) for row, col in shape)
        for tr, tc in TARGETS
    )
    bbox = (
        min(row for row, _ in shape),
        min(col for _, col in shape),
        max(row for row, _ in shape),
        max(col for _, col in shape),
    )
    return sum(distances), distances, bbox, len(shape)


def run(env):
    for action in PATH:
        env.step(action)
    root = env.clone()
    root.step(USE)
    best = (999, ())
    for ups in range(3, 10):
        for first_rights in range(3, 11):
            for downs in range(0, 11):
                for last_rights in range(0, 11):
                    route = (
                        (UP,) * ups
                        + (RIGHT,) * first_rights
                        + (DOWN,) * downs
                        + (RIGHT,) * last_rights
                    )
                    node = root.clone()
                    for action in route:
                        node.step(action)
                    value = score(node)
                    if value[0] < best[0]:
                        best = value[0], route
                        print(
                            "BEST",
                            value,
                            (ups, first_rights, downs, last_rights),
                            flush=True,
                        )
                    if value[0] == 0:
                        print(
                            "SOLVED",
                            (ups, first_rights, downs, last_rights),
                            value,
                            flush=True,
                        )
                        return
    print("DONE", best[0], len(best[1]), flush=True)


A.run_program("re86", run)
