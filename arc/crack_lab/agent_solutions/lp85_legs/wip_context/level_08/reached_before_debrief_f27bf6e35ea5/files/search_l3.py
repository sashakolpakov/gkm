"""Bounded level-3 search with mirrored-joint agreement as dense progress."""
import sys
import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A


PREFIX = [
    (6, 5, 32), (6, 5, 32), (6, 5, 32), (6, 5, 32), (6, 5, 32),
    (6, 39, 17), (6, 48, 35), (6, 39, 17), (6, 39, 17),
    (6, 39, 17), (6, 48, 35), (6, 48, 35), (6, 48, 35),
]
ACTIONS = ((6, 23, 41), (6, 35, 41))


def agreement(env):
    f = np.asarray(env.frame())
    rows = (19, 22, 25, 28, 31, 34, 37)
    pairs = ((15, 45), (18, 42), (21, 39), (24, 36), (27, 33))
    score = total = 0
    for r in rows:
        for left, right in pairs:
            a, b = int(f[r, left]), int(f[r, right])
            if a not in (3, 4) or b not in (3, 4):
                total += 1
                score += a == b
    return score, total


def run(env):
    for action in PREFIX:
        env.step(*action)
    base_level = env.levels_completed
    root = env.clone()

    def replay(path):
        node = root.clone()
        for symbol in path:
            node.step(*ACTIONS[symbol == "R"])
        return node

    beam = [""]
    seen = {np.asarray(root.frame()).tobytes()}
    best = (-1, None)
    for depth in range(35):
        ranked = []
        for path in beam:
            node = replay(path)
            metric = agreement(node)
            if metric[0] > best[0]:
                best = (metric[0], path)
                print("best", metric, path, flush=True)
            if node.levels_completed > base_level:
                print("FOUND", path, "depth", len(path),
                      "states", len(seen), flush=True)
                return
            if depth == 34 or node.terminal():
                continue
            for symbol, action in zip("LR", ACTIONS):
                child = node.clone()
                child.step(*action)
                key = np.asarray(child.frame()).tobytes()
                if key not in seen:
                    seen.add(key)
                    ranked.append((agreement(child)[0], path + symbol))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        beam = [path for _, path in ranked[:48]]
        if not beam:
            break
    print("not-found", "states", len(seen), "best", best, flush=True)


if __name__ == "__main__":
    A.run_program("lp85", run)
