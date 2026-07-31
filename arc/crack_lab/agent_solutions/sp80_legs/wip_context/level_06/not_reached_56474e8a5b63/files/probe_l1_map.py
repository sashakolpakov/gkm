"""Exhaustive observational win map for the single-piece first level."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A


def axis_moves(start, target, negative, positive):
    return [negative if target < start else positive] * (
        abs(target - start) // 4
    )


def probe(env):
    base = env.levels_completed
    wins = {}
    for top in range(0, 57, 4):
        lefts = []
        for left in range(0, 45, 4):
            node = env.clone()
            for action in (
                axis_moves(16, top, 1, 2)
                + axis_moves(12, left, 3, 4)
            ):
                node.step(action)
            node.step(5)
            if node.levels_completed > base:
                lefts.append(left)
        if lefts:
            wins[top] = tuple(lefts)
    print("L1_WIN_MAP", wins)


if __name__ == "__main__":
    levels, path, err = A.run_program("sp80", probe)
    print("PROBE_RESULT", levels, len(path), err)
