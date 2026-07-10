# Shared predicate library. p_<name>(panel) -> float | bool

import math
import numpy as np
from scipy import ndimage


def _pts(panel):
    ys, xs = np.nonzero(np.asarray(panel) > 0)
    return xs.astype(float), ys.astype(float)


def _neighbor_counts(panel):
    a = (np.asarray(panel) > 0).astype(np.int32)
    k = np.ones((3, 3), dtype=np.int32)
    k[1, 1] = 0
    nb = ndimage.convolve(a, k, mode="constant", cval=0)
    return nb[a > 0]


def _circle_fit(xs, ys):
    n = len(xs)
    if n < 3:
        return 0.0, 0.0, 0.0, 1e9
    A = np.column_stack([xs, ys, np.ones(n)])
    b = xs * xs + ys * ys
    try:
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0, 0.0, 0.0, 1e9
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    r2 = sol[2] + cx * cx + cy * cy
    if r2 <= 0:
        return cx, cy, 0.0, 1e9
    r = math.sqrt(r2)
    d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    return cx, cy, r, float(np.sqrt(np.mean((d - r) ** 2)))


def _trace_path(panel):
    a = np.asarray(panel) > 0
    ys, xs = np.nonzero(a)
    if len(xs) == 0:
        return []
    coords = set(zip(xs.tolist(), ys.tolist()))

    def nnb(p):
        return sum(1 for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                   if (dx or dy) and (p[0] + dx, p[1] + dy) in coords)

    eps = sorted(p for p in coords if nnb(p) == 1)
    start = eps[0] if eps else min(coords, key=lambda p: (p[1], p[0]))
    visited, path, cur, pdir = set(), [], start, None
    while cur is not None:
        visited.add(cur)
        path.append(cur)
        x, y = cur
        cands = [((x + dx, y + dy), (dx, dy)) for dx in (-1, 0, 1)
                 for dy in (-1, 0, 1) if (dx or dy)
                 and (x + dx, y + dy) in coords
                 and (x + dx, y + dy) not in visited]
        nxt = None
        if cands:
            if pdir is not None:
                cands.sort(key=lambda it: (-(it[1][0] * pdir[0] + it[1][1] * pdir[1]),
                                           it[0][1], it[0][0]))
            else:
                cands.sort(key=lambda it: (it[0][1], it[0][0]))
            nxt, pdir = cands[0]
        else:
            best = None
            for q in coords - visited:
                d2 = (q[0] - x) ** 2 + (q[1] - y) ** 2
                if d2 <= 8 and (best is None or d2 < best[0]):
                    best = (d2, q)
            if best is not None:
                nxt, pdir = best[1], None
        cur = nxt
    return path


def _resampled_turns(panel, step=5.0):
    path = _trace_path(panel)
    if len(path) < 3:
        return np.zeros(0)
    pts = np.asarray(path, dtype=float)
    seg = np.sqrt((np.diff(pts, axis=0) ** 2).sum(1))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 2 * step:
        return np.zeros(0)
    t = np.arange(0.0, s[-1] + 1e-9, step)
    rx = np.interp(t, s, pts[:, 0])
    ry = np.interp(t, s, pts[:, 1])
    ang = np.arctan2(np.diff(ry), np.diff(rx))
    d = np.diff(ang)
    return (d + math.pi) % (2 * math.pi) - math.pi


def p_ink_count(panel):
    return float(np.count_nonzero(np.asarray(panel) > 0))


def p_component_count(panel):
    a = (np.asarray(panel) > 0).astype(np.int32)
    _, n = ndimage.label(a, structure=np.ones((3, 3), dtype=np.int32))
    return float(n)


def p_endpoint_pixels(panel):
    nb = _neighbor_counts(panel)
    return float(np.sum(nb == 1)) if nb.size else 0.0


def p_junction_pixels(panel):
    nb = _neighbor_counts(panel)
    return float(np.sum(nb >= 3)) if nb.size else 0.0


def p_is_single_open_curve(panel):
    return (p_component_count(panel) == 1.0
            and p_endpoint_pixels(panel) == 2.0
            and p_junction_pixels(panel) == 0.0)


def p_circle_fit_residual(panel):
    xs, ys = _pts(panel)
    if len(xs) < 3:
        return 1.0
    _, _, _, rms = _circle_fit(xs, ys)
    return float(min(rms / 128.0, 1.0))


def p_circle_fit_max_residual(panel):
    """Max |dist - radius| over ink for best-fit circle, / 128."""
    xs, ys = _pts(panel)
    if len(xs) < 3:
        return 1.0
    cx, cy, r, _ = _circle_fit(xs, ys)
    if r <= 0:
        return 1.0
    d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    return float(min(np.max(np.abs(d - r)) / 128.0, 1.0))


def p_circle_fit_rel_residual(panel):
    xs, ys = _pts(panel)
    if len(xs) < 3:
        return 1.0
    _, _, r, rms = _circle_fit(xs, ys)
    return float(min(rms / r, 1.0)) if r > 1e-6 else 1.0


def p_circle_fit_radius(panel):
    xs, ys = _pts(panel)
    if len(xs) < 3:
        return 10.0
    _, _, r, _ = _circle_fit(xs, ys)
    return float(min(r / 128.0, 10.0))


def p_arc_extent_deg(panel):
    xs, ys = _pts(panel)
    if len(xs) < 3:
        return 0.0
    cx, cy, r, _ = _circle_fit(xs, ys)
    if r <= 0:
        return 0.0
    ang = np.sort(np.arctan2(ys - cy, xs - cx))
    if len(ang) < 2:
        return 0.0
    max_gap = max(float(np.max(np.diff(ang))),
                  float(2 * math.pi - (ang[-1] - ang[0])))
    return float(math.degrees(2 * math.pi - max_gap))


def p_sagitta_ratio(panel):
    """Max perpendicular deviation from chord between farthest ink pixels,
    divided by chord length. ~0 straight, 0.5 semicircle."""
    xs, ys = _pts(panel)
    n = len(xs)
    if n < 3:
        return 0.0
    pts = np.column_stack([xs, ys])
    sub = pts[np.linspace(0, n - 1, min(n, 400)).astype(int)]
    d2 = ((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1)
    i, j = np.unravel_index(np.argmax(d2), d2.shape)
    a, b = sub[i], sub[j]
    chord = math.hypot(b[0] - a[0], b[1] - a[1])
    if chord < 1e-6:
        return 0.0
    dx, dy = b[0] - a[0], b[1] - a[1]
    dev = np.abs(dx * (a[1] - pts[:, 1]) - dy * (a[0] - pts[:, 0])) / chord
    return float(np.max(dev) / chord)


def p_total_turning_deg(panel):
    """Sum of |turn angles| along traced resampled path, degrees."""
    d = _resampled_turns(panel)
    return float(math.degrees(np.sum(np.abs(d)))) if d.size else 0.0


def p_net_turning_deg(panel):
    """|sum of signed turns| along traced path, degrees."""
    d = _resampled_turns(panel)
    return float(math.degrees(abs(np.sum(d)))) if d.size else 0.0


def p_turning_consistency(panel):
    """Fraction of turns with the majority sign (1 = always same bend)."""
    d = _resampled_turns(panel)
    d = d[np.abs(d) > 1e-3]
    if d.size == 0:
        return 0.0
    pos = float(np.sum(d > 0))
    return float(max(pos, d.size - pos) / d.size)


def p_max_turn_deg(panel):
    """Largest single |turn| along path (kink detector), degrees."""
    d = _resampled_turns(panel)
    return float(math.degrees(np.max(np.abs(d)))) if d.size else 0.0


def p_turn_cv(panel):
    """Coefficient of variation of signed turn angles along the path.
    Near 0 for a constant-curvature circular arc; large for straight
    lines, kinks, or varying curvature."""
    d = _resampled_turns(panel)
    if d.size < 3:
        return 10.0
    m = float(np.mean(d))
    if abs(m) < 1e-6:
        return 10.0
    return float(min(np.std(d) / abs(m), 10.0))


def p_elongation(panel):
    xs, ys = _pts(panel)
    if len(xs) < 3:
        return 0.0
    ev = np.linalg.eigvalsh(np.cov(np.column_stack([xs, ys]).T))
    if ev[1] <= 0:
        return 0.0
    return float(math.sqrt(max(ev[0], 0.0) / ev[1]))


def p_span_norm(panel):
    a = np.asarray(panel) > 0
    ys, xs = np.nonzero(a)
    if len(xs) == 0:
        return 0.0
    return float(math.hypot(xs.max() - xs.min(), ys.max() - ys.min()) / 128.0)
