# Shared predicate library. p_<name>(panel) -> float | bool
import numpy as np
from collections import deque


def _points(panel):
    ys, xs = np.where(panel > 0)
    return np.stack([xs, ys], axis=1).astype(float)


def _fit_circle(pts):
    """Algebraic (Kasa) circle fit. Returns (cx, cy, r)."""
    x, y = pts[:, 0], pts[:, 1]
    A = np.stack([x, y, np.ones_like(x)], axis=1)
    b = x ** 2 + y ** 2
    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    r = float(np.sqrt(max(sol[2] + cx ** 2 + cy ** 2, 1e-9)))
    return cx, cy, r


def _geodesic_arclen(panel):
    """Longest shortest-path (8-connectivity) distance from the point
    farthest from the centroid -- an estimate of curve length in pixels,
    robust to stroke thickness (unlike raw pixel count)."""
    pts = _points(panel)
    n = len(pts)
    idx_of = {tuple(p): i for i, p in enumerate(pts.tolist())}
    c = pts.mean(axis=0)
    start = int(np.argmax(np.sum((pts - c) ** 2, axis=1)))
    dist = -np.ones(n, dtype=int)
    dist[start] = 0
    q = deque([start])
    while q:
        i = q.popleft()
        x, y = pts[i]
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nb = (x + dx, y + dy)
                j = idx_of.get(nb)
                if j is not None and dist[j] == -1:
                    dist[j] = dist[i] + 1
                    q.append(j)
    return float(dist.max())


def _circle_fit_residual_ratio(panel):
    """Mean absolute deviation of ink pixels from their best-fit circle,
    normalized by the fitted radius. Near 0 for a clean circular arc;
    large for corners, multi-arc shapes, or blobby/overlapping strokes."""
    pts = _points(panel)
    if len(pts) < 5:
        return 0.0
    cx, cy, r = _fit_circle(pts)
    dist = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
    resid = np.abs(dist - r)
    return float(resid.mean() / r)


def _arc_span_deg(panel):
    """Angular span (degrees) of the curve if treated as an arc of its
    best-fit circle: geodesic arc length / fitted radius, in degrees."""
    pts = _points(panel)
    if len(pts) < 5:
        return 0.0
    _, _, r = _fit_circle(pts)
    L = _geodesic_arclen(panel)
    return float(np.degrees(L / r))


def p_is_regular_circular_arc(panel):
    """A single stroke that traces a clean arc of one fixed circle (low
    fit residual) spanning a plausible, non-degenerate angular range --
    excludes corners/kinks, multi-arc shapes, near-straight lines, and
    near-full loops, all of which break either the circle fit or the
    span band."""
    resid = _circle_fit_residual_ratio(panel)
    span = _arc_span_deg(panel)
    return bool(resid < 0.012 and 100.0 < span < 115.0)


def p_is_wide_hand_drawn_arc(panel):
    """A single stroke that traces a wide (~170-210 degree), imperfect
    arc of roughly one circle: circle-fit residual in a middle band
    (loose enough for a hand-drawn wobble, tight enough to reject
    self-intersecting/multi-loop scribbles, but not so tight it passes
    a near-perfect circle) and span in a middle band (excludes both
    shallow near-straight arcs and near-closed loops)."""
    resid = _circle_fit_residual_ratio(panel)
    span = _arc_span_deg(panel)
    return bool(0.015 < resid < 0.05 and 170.0 < span < 210.0)


def p_num_components(panel):
    """Count of 8-connected foreground components (number of drawn strokes/objects)."""
    from scipy import ndimage
    _, n = ndimage.label(panel, structure=np.ones((3, 3)))
    return float(n)
