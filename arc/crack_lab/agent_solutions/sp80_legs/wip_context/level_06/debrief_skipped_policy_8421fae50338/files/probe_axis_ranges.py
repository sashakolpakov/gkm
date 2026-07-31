"""One-piece acceptance intervals around known lower-level wins."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from legs import drive_objects
from players import play_level_1, play_level_2, play_level_3


PLANS = {
    2: [
        ((14, 18), [4] * 3),
        ((34, 26), [4] * 2),
        ((30, 38), [4] * 2),
    ],
    3: [
        ((15, 21), [4]),
        ((49, 29), [3] * 10),
        ((19, 33), [4] * 5),
        ((47, 41), [4]),
    ],
    4: [
        ((24, 18), [4] * 3),
        ((45, 18), [3] * 8),
        ((18, 30), [4] * 8),
        ((49, 33), [3] * 10),
        ((43, 42), [3] * 7),
    ],
}

POINTS = {
    2: {"A": (24, 18), "B": (40, 26), "C": (34, 38)},
    3: {"A": (20, 21), "B": (10, 29), "C": (35, 33), "N": (48, 41)},
    4: {
        "A": (35, 18), "B": (20, 18), "C": (36, 30),
        "D": (20, 33), "E": (22, 42),
    },
}


def accepted(arranged, base_level, point, direction):
    wins = []
    for count in range(17):
        node = arranged.clone()
        node.step(6, *point)
        for _ in range(count):
            node.step(direction)
        node.step(5)
        if node.levels_completed > base_level:
            wins.append(count)
    return tuple(wins)


def probe(env):
    play_level_1(env)
    for level in range(2, 5):
        base = env.levels_completed
        arranged = env.clone()
        drive_objects(arranged, PLANS[level], commit=[])
        print("L", level, {
            (name, direction): accepted(arranged, base, point, direction)
            for name, point in POINTS[level].items()
            for direction in (1, 2, 3, 4)
        })
        if level == 2:
            play_level_2(env)
        elif level == 3:
            play_level_3(env)


if __name__ == "__main__":
    levels, path, err = A.run_program("sp80", probe)
    print("PROBE_RESULT", levels, len(path), err)
