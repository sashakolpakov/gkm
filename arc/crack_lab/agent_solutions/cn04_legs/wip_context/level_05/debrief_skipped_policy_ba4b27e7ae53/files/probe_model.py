from collections import Counter


def rotate(points, pivot, turns):
    out = set(points)
    for _ in range(turns):
        out = {(pivot[0] + (c - pivot[1]),
                pivot[1] - (r - pivot[0])) for r, c in out}
    return out


def shift(points, dy, dx):
    return {(r + dy, c + dx) for r, c in points}


fixed = {
    (2, 18), (3, 16), (3, 18), (4, 16), (4, 18),
    (5, 16), (5, 18), (6, 16), (6, 17), (6, 18),
}
central_cells = {
    (2, 17), (2, 18), (3, 16), (3, 17),
    (4, 17), (4, 18), (4, 19), (5, 17),
}
central_ends = {(2, 18), (3, 16), (4, 19)}

left_cells0 = {
    (13, 2), (13, 3), (13, 4), (14, 2),
    (15, 2), (15, 3), (15, 4), (16, 2),
    (17, 2), (17, 3), (17, 4),
}
left_ends0 = {(13, 4), (15, 4), (17, 4)}
right_cells0 = {
    (16, 16), (17, 16), (17, 18),
    (18, 16), (18, 17), (18, 18),
}
right_ends0 = {(16, 16), (17, 18)}


def covered(ends, own_cells, others):
    return sum(p in others for p in ends)


hits = []
for lr in range(4):
    lc0 = rotate(left_cells0, (15, 4), lr)
    le0 = rotate(left_ends0, (15, 4), lr)
    for ldy in range(-15, 6):
        for ldx in range(-8, 19):
            lc = shift(lc0, ldy, ldx)
            le = shift(le0, ldy, ldx)
            if not all(0 <= r <= 20 and 0 <= c <= 21 for r, c in lc):
                continue
            for rr in range(4):
                rc0 = rotate(right_cells0, (17, 17), rr)
                re0 = rotate(right_ends0, (17, 17), rr)
                for rdy in range(-16, 5):
                    for rdx in range(-16, 7):
                        rc = shift(rc0, rdy, rdx)
                        re = shift(re0, rdy, rdx)
                        if not all(0 <= r <= 20 and 0 <= c <= 21 for r, c in rc):
                            continue
                        ccov = covered(central_ends, central_cells, le | re)
                        lcov = covered(le, lc, central_ends | re)
                        rcov = covered(re, rc, central_ends | le)
                        score = ccov + lcov + rcov
                        if score == 8:
                            union = len(fixed | central_cells | lc | rc)
                            hits.append((union, lr, ldy, ldx,
                                         rr, rdy, rdx,
                                         tuple(sorted(le)),
                                         tuple(sorted(re))))

print("COUNT", len(hits))
for hit in sorted(hits)[:80]:
    print(hit)
