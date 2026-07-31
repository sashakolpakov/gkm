import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from probe_finish8 import PREFIX


def show(frame, row, col):
    symbols = {5: ".", 7: "A", 8: "S", 9: "#"}
    return "/".join(
        "".join(symbols.get(int(frame[r][c]), "?")
                for c in range(col - 5, col + 6))
        for r in range(row - 5, row + 6)
    )


def inspect(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(*action)
    initial = env.frame()
    print("initial")
    for target in ((19, 7), (19, 56), (55, 7), (55, 56)):
        print(target, show(initial, *target))
    for action in PREFIX:
        env.step(*action)
    print("near")
    for target in ((19, 7), (19, 56), (55, 7), (55, 56)):
        print(target, show(env.frame(), *target))


A.run_program("su15", inspect)
