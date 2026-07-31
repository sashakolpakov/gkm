"""Print cn04 level-2 frames on their 3-pixel logical lattice."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
from players import play_level_1


SYMBOL = {0: "A", 4: "-", 8: "x", 9: "D", 11: "C", 12: ".", 14: "B"}


def lattice(frame):
    a = perception.arr(frame)
    rows = []
    for r in range(3, 60, 3):
        rows.append("".join(SYMBOL.get(int(a[r, c]), "?") for c in range(3, 60, 3)))
    return rows


def probe(env):
    print("level1_initial")
    for i, row in enumerate(lattice(env.frame())):
        print(f"{i:02d} {row}")
    prefinish = perception.replay(env, [2] * 7 + [4] * 4)
    print("level1_prefinish")
    for i, row in enumerate(lattice(prefinish.frame())):
        print(f"{i:02d} {row}")
    play_level_1(env)
    print("level2_initial")
    for i, row in enumerate(lattice(env.frame())):
        print(f"{i:02d} {row}")
    print("    " + "".join(str(i // 10 or " ") for i in range(19)))
    print("    " + "".join(str(i % 10) for i in range(19)))


arena.run_program("cn04", probe)
