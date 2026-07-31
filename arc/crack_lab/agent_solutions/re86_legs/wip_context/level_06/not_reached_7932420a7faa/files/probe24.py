from itertools import product

M8 = [(27, 51), (33, 57), (36, 42), (42, 54)]
M9 = [(6, 21), (6, 39), (45, 33), (51, 24), (51, 45), (60, 33)]
MARK = {p: 8 for p in M8}
MARK.update({p: 9 for p in M9})

SHAPES = {
    'A': ('x', 11, (42, 24)),
    'B': ('diamond', 9, (18, 30)),
    'C': ('plus', 14, (33, 54)),
}


def covers(kind, R, r, c, p):
    dr, dc = p[0] - r, p[1] - c
    if kind == 'x':
        return abs(dr) == abs(dc) and 0 < abs(dr) <= R
    if kind == 'diamond':
        return abs(dr) + abs(dc) == R
    return (dr == 0) != (dc == 0) and max(abs(dr), abs(dc)) <= R


opts = {}
for name, (kind, R, ctr) in SHAPES.items():
    out = []
    for r in range(0, 63, 3):
        for c in range(0, 64, 3):
            if (r - ctr[0]) % 3 or (c - ctr[1]) % 3:
                continue
            cov = frozenset(p for p in MARK if covers(kind, R, r, c, p))
            if not cov:
                continue
            cols = {MARK[p] for p in cov}
            if len(cols) != 1:
                continue
            out.append((r, c, cov, cols.pop()))
    opts[name] = out
    print(name, 'placements', len(out), 'max_cov', max(len(o[2]) for o in out))

target = frozenset(MARK)
sols = []
for a, b, cc in product(opts['A'], opts['B'], opts['C']):
    if a[2] | b[2] | cc[2] == target:
        cost = (abs(a[0] - SHAPES['A'][2][0]) + abs(a[1] - SHAPES['A'][2][1])
                + abs(b[0] - SHAPES['B'][2][0]) + abs(b[1] - SHAPES['B'][2][1])
                + abs(cc[0] - SHAPES['C'][2][0]) + abs(cc[1] - SHAPES['C'][2][1]))
        sols.append((cost, a[:2], a[3], b[:2], b[3], cc[:2], cc[3]))
sols.sort()
print('solutions', len(sols))
for s in sols[:8]:
    print(s)
