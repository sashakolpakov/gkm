"""Infer level-4 permutations and search a 180-degree-symmetry dense goal."""
from collections import deque
import math
import itertools
import random
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from solve import solve


CENTERS = (15, 45)
OFFSETS = ((-6, 0), (-3, 0), (0, -6), (0, -3), (0, 0),
           (0, 3), (0, 6), (3, 0), (6, 0))
CLICKS = (
    (6, 15, 6), (6, 15, 25), (6, 6, 15), (6, 25, 15),
)
PAIR_INDICES = ((0, 8), (1, 7), (2, 6), (3, 5))


def state(frame):
    f = np.asarray(frame)
    return tuple(
        int(f[r + dr, c + dc])
        for r in CENTERS for c in CENTERS for dr, dc in OFFSETS
    )


def symmetry_score(s):
    return sum(
        s[base + a] == s[base + b]
        for base in (0, 9, 18, 27) for a, b in PAIR_INDICES
    )


def infer_permutations(env):
    rng = random.Random(8504)
    node = env.clone()
    samples = [[] for _ in CLICKS]
    for _ in range(120):
        op = rng.randrange(4)
        before = state(node.frame())
        node.step(*CLICKS[op])
        after = state(node.frame())
        samples[op].append((before, after))

    permutations = []
    for transitions in samples:
        perm = []
        for dest in range(36):
            candidates = [
                source for source in range(36)
                if all(after[dest] == before[source]
                       for before, after in transitions)
            ]
            if len(candidates) != 1:
                raise RuntimeError(("ambiguous permutation", dest, candidates))
            perm.append(candidates[0])
        permutations.append(tuple(perm))
    return tuple(permutations)


def apply_perm(s, perm):
    return tuple(s[source] for source in perm)


def search(start, permutations, max_states=500000, max_depth=24):
    queue = deque([(start, ())])
    seen = {start}
    best = (symmetry_score(start), ())
    while queue and len(seen) <= max_states:
        current, path = queue.popleft()
        score = symmetry_score(current)
        if score > best[0]:
            best = (score, path)
            print("best", score, "depth", len(path), "path", path)
        if score == 16:
            return path
        if len(path) >= max_depth:
            continue
        for op, perm in enumerate(permutations):
            child = apply_perm(current, perm)
            if child not in seen:
                seen.add(child)
                queue.append((child, path + (op,)))
    print("exhausted", len(seen), "best", best)
    return None


def beam_search(start, permutations, width=60000, max_depth=60):
    frontier = {start: ()}
    seen = {start}
    best_score = symmetry_score(start)
    for depth in range(1, max_depth + 1):
        candidates = {}
        for current, path in frontier.items():
            for op, perm in enumerate(permutations):
                if path and op == (path[-1] ^ 1):
                    continue
                child = apply_perm(current, perm)
                if child in seen:
                    continue
                child_path = path + (op,)
                score = symmetry_score(child)
                if score == 16:
                    return child_path
                candidates.setdefault(child, child_path)
        if not candidates:
            break
        ranked = sorted(
            candidates.items(),
            key=lambda item: (symmetry_score(item[0]), hash(item[0])),
            reverse=True,
        )[:width]
        frontier = dict(ranked)
        seen.update(frontier)
        level_best = symmetry_score(ranked[0][0])
        if level_best > best_score:
            best_score = level_best
            print("beam_best", best_score, "depth", depth,
                  "path", ranked[0][1])
    print("beam_exhausted", len(seen), "best", best_score)
    return None


def anneal_search(start, permutations, restarts=500, steps=4000):
    rng = random.Random(850405)
    global_best = symmetry_score(start)
    for restart in range(restarts):
        current = start
        score = symmetry_score(current)
        path = []
        for step in range(steps):
            temperature = 1.8 * (1.0 - step / steps) + 0.18
            options = []
            for op, perm in enumerate(permutations):
                if path and op == (path[-1] ^ 1):
                    continue
                child = apply_perm(current, perm)
                options.append((op, child, symmetry_score(child)))
            op, child, child_score = rng.choice(options)
            delta = child_score - score
            if delta >= 0 or rng.random() < math.exp(delta / temperature):
                current, score = child, child_score
                path.append(op)
                if score > global_best:
                    global_best = score
                    print("anneal_best", score, "restart", restart,
                          "step", step, "length", len(path))
                if score == 16:
                    return tuple(path)
        if restart % 100 == 99:
            print("anneal_progress", restart + 1, "best", global_best)
    return None


def macro_search(start, permutations):
    powers = []
    for perm in (permutations[0], permutations[2]):
        row = [start]
        for _ in range(19):
            row.append(apply_perm(row[-1], perm))
        powers.append(row)

    # Recompute powers from arbitrary states while enumerating four alternating
    # rotations, matching the compact solution family used by earlier levels.
    def rotate(s, perm, amount):
        for _ in range(amount):
            s = apply_perm(s, perm)
        return s

    for order in ((0, 2, 0, 2), (2, 0, 2, 0)):
        for amounts in itertools.product(range(20), repeat=4):
            current = start
            for op, amount in zip(order, amounts):
                current = rotate(current, permutations[op], amount)
            if symmetry_score(current) == 16:
                path = []
                for op, amount in zip(order, amounts):
                    path.extend([op] * amount)
                print("macro_solution", order, amounts, "length", len(path))
                return tuple(path)
    return None


def run(env):
    solve(env)
    start = state(env.frame())
    permutations = infer_permutations(env)
    print("permutations", permutations)
    print("start_score", symmetry_score(start))
    path = macro_search(start, permutations)
    if path is None:
        path = search(start, permutations, max_states=100000)
    if path is None:
        path = anneal_search(start, permutations)
    print("candidate", path)
    if path is not None:
        clone = env.clone()
        for op in path:
            clone.step(*CLICKS[op])
        print("verify", clone.levels_completed, clone.terminal(),
              symmetry_score(state(clone.frame())))


if __name__ == "__main__":
    A.run_program("lp85", run)
