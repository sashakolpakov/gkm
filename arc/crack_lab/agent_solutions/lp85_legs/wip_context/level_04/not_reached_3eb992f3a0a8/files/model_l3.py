"""Infer and validate the two level-3 board permutations from observations."""
import itertools
import sys
from collections import deque

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from search_l3 import PREFIX, ACTIONS


POSITIONS = tuple(
    (r, c)
    for r, cols in (
        (19, (21, 24, 27, 33, 36, 39)),
        (22, (18, 30, 42)),
        (25, (15, 27, 33, 45)),
        (28, (15, 27, 33, 45)),
        (31, (15, 27, 33, 45)),
        (34, (18, 30, 42)),
        (37, (21, 24, 27, 33, 36, 39)),
    )
    for c in cols
)


def state(env):
    f = np.asarray(env.frame())
    return tuple(int(f[r, c]) for r, c in POSITIONS)


def run(env):
    for action in PREFIX:
        env.step(*action)
    root = env.clone()
    samples = [[], ["L"], ["R"]]
    for n in range(2, 10):
        samples.extend(list(bits) for bits in itertools.product("LR", repeat=n)
                       if sum(x == "R" for x in bits) in (1, n // 2, n - 1)) 
        if len(samples) >= 45:
            break
    samples = samples[:45]
    transitions = [[], []]
    for path in samples:
        node = root.clone()
        for symbol in path:
            node.step(*ACTIONS[symbol == "R"])
        before = state(node)
        for index, action in enumerate(ACTIONS):
            child = node.clone()
            child.step(*action)
            transitions[index].append((before, state(child)))

    permutations = []
    for index, examples in enumerate(transitions):
        candidates = []
        for destination in range(len(POSITIONS)):
            sources = [
                source for source in range(len(POSITIONS))
                if all(before[source] == after[destination]
                       for before, after in examples)
            ]
            candidates.append(sources)
        ambiguous = sum(len(x) != 1 for x in candidates)
        print("action", "LR"[index], "ambiguous", ambiguous,
              "candidates", candidates, flush=True)
        permutations.append(tuple(x[0] if len(x) == 1 else -1
                                  for x in candidates))
    if any(-1 in p for p in permutations):
        return

    # Validate inferred transitions on contexts not used above.
    for path in ("LRRLLRLRRL", "RRLLLRRLLR", "LRLRRLLRLR"):
        node = root.clone()
        for symbol in path:
            node.step(*ACTIONS[symbol == "R"])
        before = state(node)
        for index, action in enumerate(ACTIONS):
            child = node.clone()
            child.step(*action)
            predicted = tuple(before[source] for source in permutations[index])
            print("validate", path, "LR"[index],
                  predicted == state(child), flush=True)

    mirror_pairs = tuple(
        (POSITIONS.index((r, c)), POSITIONS.index((r, 60 - c)))
        for r in (19, 25, 28, 31, 37)
        for c in (15, 18, 21, 24, 27)
        if (r, c) in POSITIONS and (r, 60 - c) in POSITIONS
    ) + (
        (POSITIONS.index((22, 18)), POSITIONS.index((22, 42))),
        (POSITIONS.index((34, 18)), POSITIONS.index((34, 42))),
    )

    def score(values):
        return sum(values[a] == values[b] for a, b in mirror_pairs)

    start = state(root)
    q = deque([(start, "")])
    seen = {start}
    best = (-1, "")
    found = None
    while q:
        values, path = q.popleft()
        current = score(values)
        if current > best[0]:
            best = (current, path)
            print("symbolic-best", current, len(mirror_pairs), path, flush=True)
        if current == len(mirror_pairs):
            found = path
            break
        if len(path) >= 18:
            continue
        for symbol, permutation in zip("LR", permutations):
            child = tuple(values[source] for source in permutation)
            if child not in seen:
                seen.add(child)
                q.append((child, path + symbol))
    print("symbolic-result", found, "states", len(seen), flush=True)
    if found is not None:
        clone = root.clone()
        for symbol in found:
            clone.step(*ACTIONS[symbol == "R"])
        print("reward-check", clone.levels_completed,
              "terminal", clone.terminal(), "score", score(state(clone)),
              flush=True)


if __name__ == "__main__":
    A.run_program("lp85", run)
