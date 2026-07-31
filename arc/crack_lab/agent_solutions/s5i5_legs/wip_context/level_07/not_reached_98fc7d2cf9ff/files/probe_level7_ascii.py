import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import arr


CONTROLS = {
    "A<": (6, 4, 51), "A>": (6, 10, 51), "B": (6, 17, 51),
    "C<": (6, 26, 51), "C>": (6, 32, 51), "D": (6, 39, 51),
    "E<": (6, 54, 51), "E>": (6, 60, 51),
    "F<": (6, 4, 58), "F>": (6, 10, 58), "G": (6, 17, 58),
    "H<": (6, 26, 58), "H>": (6, 32, 58), "I": (6, 60, 58),
}
PREFIX = (
    "F<", "A>", "B", "B", "A>", "F>", "A>", "C>", "A>", "C>",
    "A>", "D", "H>", "H>", "H>", "H>", "H>",
)
CHARS = {3: ".", 8: "8", 9: "9", 10: "0", 11: "1", 12: "2",
         13: "*", 14: "4", 15: "#"}


def picture(frame):
    grid = arr(frame)
    lines = []
    for r in range(42):
        text = "".join(CHARS.get(int(grid[r, c]), " ") for c in range(64))
        if text.strip():
            lines.append(f"{r:02d} {text.rstrip()}")
    return "\n".join(lines)


def run(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    print("START")
    print(picture(env.frame()))
    for name in PREFIX:
        env.step(*CONTROLS[name])
    print("FRONTIER")
    print(picture(env.frame()))


arena.run_program("s5i5", run)
