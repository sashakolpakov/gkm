"""Measure the remaining horizon and exhaustively check it when small."""
import itertools
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from search_l3 import PREFIX, ACTIONS


def run(env):
    for action in PREFIX:
        env.step(*action)
    root = env.clone()
    probe = root.clone()
    horizon = None
    for depth in range(1, 41):
        probe.step(*ACTIONS[0])
        if probe.levels_completed > 2:
            print("repeat-win", depth, flush=True)
            return
        if probe.terminal():
            horizon = depth
            break
    print("terminal-horizon", horizon, flush=True)
    if horizon is None or horizon > 15:
        return
    checked = 0
    for depth in range(horizon + 1):
        for bits in itertools.product((0, 1), repeat=depth):
            clone = root.clone()
            for bit in bits:
                clone.step(*ACTIONS[bit])
                if clone.levels_completed > 2 or clone.terminal():
                    break
            checked += 1
            if clone.levels_completed > 2:
                path = "".join("LR"[bit] for bit in bits)
                print("FOUND", path, "depth", depth,
                      "checked", checked, flush=True)
                return
    print("no-win", "checked", checked, flush=True)


if __name__ == "__main__":
    A.run_program("lp85", run)
