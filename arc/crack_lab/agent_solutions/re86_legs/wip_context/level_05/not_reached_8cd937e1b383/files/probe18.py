"""Offline geometric feasibility: can 3 fixed shapes cover all 10 markers?"""
from itertools import product

R, C = 63, 64          # usable rows 0..62 (row 63 is the meter bar), cols 0..63
M9 = [(6, 21), (6, 39), (45, 33), (51, 24), (51, 45), (60, 33)]
M8 = [(27, 51), (33, 57), (36, 42), (42, 54)]
MARK = {p: 9 for p in M9}
MARK.update({p: 8 for p in M8})

X = {(0, 0)} | {(s1 * d, s2 * d) for d in range(1, 12) for s1 in (1, -1) for s2 in (1, -1)}
DIA = {(dr, dc) for dr in range(-9, 10) for dc in range(-9, 10) if abs(dr) + abs(dc) == 9}
PLUS = {(0, 0)} | {(0, s * d) for d in range(1, 15) for s in (1, -1)} \
                | {(s * d, 0) for d in range(1, 15) for s in (1, -1)}
SHAPES = {"X": X, "DIA": DIA, "PLUS": PLUS}

CENTERS = [(r, c) for r in range(0, R, 3) for c in range(0, C, 3)]


def covered(cells, ctr):
    return frozenset(m for m in MARK if (m[0] - ctr[0], m[1] - ctr[1]) in cells)


# best single-shape coverage per (shape, color) under strict (no wrong-color) rule
opts = {}
for name, cells in SHAPES.items():
    for color in (9, 8):
        best = []
        for ctr in CENTERS:
            cov = covered(cells, ctr)
            right = frozenset(m for m in cov if MARK[m] == color)
            wrong = cov - right
            best.append((len(right), len(wrong), ctr, right))
        best.sort(key=lambda t: (-t[0], t[1]))
        opts[(name, color)] = best
        top = best[0]
        strict = max((t for t in best if t[1] == 0), key=lambda t: t[0])
        print(f"{name:4s} as {color}: max_right={top[0]} (wrong={top[1]}) @{top[2]}  "
              f"| strict-best={strict[0]} @{strict[2]} {sorted(strict[3])}")

print()
# exhaustive: assign each of the 3 shapes a color, then search placements
names = list(SHAPES)
found = 0
for colors in product((9, 8), repeat=3):
    # per-shape candidate placements: keep maximal right-sets
    cand = []
    for name, color in zip(names, colors):
        seen = {}
        for _, wrong, ctr, right in opts[(name, color)]:
            if right and (right not in seen or wrong < seen[right][0]):
                seen[right] = (wrong, ctr)
        cand.append([(r, w, c) for r, (w, c) in seen.items()])
    for a, b, c in product(*cand):
        if a[0] | b[0] | c[0] == frozenset(MARK):
            print("SOLUTION", colors, a, b, c)
            found += 1
            if found > 3:
                break
    if found > 3:
        break
if not found:
    print("NO 3-shape static cover exists (permissive or strict).")
    # how much can we cover at most?
    best_total = 0
    for colors in product((9, 8), repeat=3):
        cand = []
        for name, color in zip(names, colors):
            seen = set()
            for _, wrong, ctr, right in opts[(name, color)]:
                seen.add(right)
            cand.append(list(seen))
        for a, b, c in product(*cand):
            t = len(a | b | c)
            if t > best_total:
                best_total = t
                arg = (colors, sorted(a), sorted(b), sorted(c))
    print("max markers coverable =", best_total, "of", len(MARK))
    print("  ", arg)
