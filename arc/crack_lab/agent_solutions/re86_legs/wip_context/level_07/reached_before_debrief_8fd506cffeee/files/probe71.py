import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import DOWN, LEFT, RIGHT, UP, USE, arr, color_counts


PATH = json.load(open("checkpoint.json"))["final_path"]
CHAR = {"U": UP, "D": DOWN, "L": LEFT, "R": RIGHT}

SMALL = (
    (RIGHT,) * 10
    + (UP,) * 7
    + (RIGHT,)
    + (DOWN,) * 4
    + (UP,) * 9
    + (LEFT,) * 6
    + (DOWN,) * 2
    + (RIGHT,) * 6
    + (DOWN,) * 7
)
OUTLINE = (
    (UP,) * 3
    + (RIGHT,) * 4
    + (UP,) * 2
    + (RIGHT,) * 5
    + (UP,) * 6
    + (LEFT,) * 13
    + (UP,) * 5
    + (DOWN,) * 5
    + (RIGHT,) * 13
)
LARGE_BASE = tuple(
    CHAR[ch]
    for ch in "ULLUUUUUUUULLLUUURRRDDDDUUUURUUUDDDRRDRLULLLUDLL"
)
LARGE_SHIFT = tuple(CHAR[ch] for ch in "RRRRRDRRLLULLLL")
LARGE = (
    LARGE_BASE
    + LARGE_SHIFT
    + LARGE_SHIFT
    + (UP,) * 4
    + (DOWN,) * 4
)


def apply(env, route):
    for action in route:
        env.step(action)


def brief(env, tag):
    frame = arr(env.frame())
    print(
        tag,
        "level",
        env.levels_completed,
        "moves",
        len(frame),
        "colors",
        {
            color: count
            for color, count in color_counts(frame).items()
            if color in (4, 8, 9, 11)
        },
        flush=True,
    )


def run(env):
    for action in PATH:
        env.step(action)
    root = env.clone()
    apply(root, SMALL)
    brief(root, "SMALL")
    root.step(USE)
    apply(root, OUTLINE)
    brief(root, "OUTLINE")
    root.step(USE)
    apply(root, LARGE)
    brief(root, "LARGE")
    print("TOTAL", len(SMALL) + len(OUTLINE) + len(LARGE) + 2, flush=True)


print("RUN", A.run_program("re86", run))
