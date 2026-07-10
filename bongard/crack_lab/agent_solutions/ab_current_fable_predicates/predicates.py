# Shared predicate library. p_<name>(panel) -> float | bool

import numpy as np
import math
from scipy import ndimage


def _ink_xy(panel):
    """Return ink pixel coordinates as float arrays (xs, ys)."""
    ys, xs = np.nonzero(panel)
    return xs.astype(float), ys.astype(float)


def _fit_circle(xs, ys):
    """Least-squares (Kasa) circle fit. Returns (cx, cy, r)."""
    A = np.column_stack([xs, ys, np.ones_like(xs)])
    b = xs * xs + ys * ys
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    r2 = sol[2] + cx * cx + cy * cy
    r = math.sqrt(max(r2, 1e-9))
    return cx, cy, r


def _neighbor_counts(panel):
    """For each pixel, number of 8-neighbors that are ink."""
    p = (panel > 0).astype(int)
    k = np.ones((3, 3), dtype=int)
    return ndimage.convolve(p, k, mode='constant', cval=0) - p


def p_ink_count(panel) -> float:
    """Total number of ink pixels."""
    return float((panel > 0).sum())


def p_num_components(panel) -> float:
    """Number of 8-connected components of ink."""
    _, n = ndimage.label(panel > 0, structure=np.ones((3, 3), dtype=int))
    return float(n)


def p_num_endpoints(panel) -> float:
    """Number of ink pixels having exactly one ink neighbor (curve endpoints)."""
    p = (panel > 0).astype(int)
    nb = _neighbor_counts(panel)
    return float(((p == 1) & (nb == 1)).sum())


def p_arc_extent_deg(panel) -> float:
    """Angular extent (degrees) of ink around its best-fit circle center.

    Closed shapes -> ~360; a semicircular arc -> ~180; shallow arc -> small.
    """
    xs, ys = _ink_xy(panel)
    if xs.size < 3:
        return 0.0
    cx, cy, _ = _fit_circle(xs, ys)
    ang = np.degrees(np.arctan2(ys - cy, xs - cx))
    ang = np.sort(ang)
    gaps = np.diff(ang)
    wrap = ang[0] + 360.0 - ang[-1]
    max_gap = max(float(gaps.max()) if gaps.size else 0.0, wrap)
    return 360.0 - max_gap


def p_arc_dev_from_half(panel) -> float:
    """Absolute deviation of arc extent from 180 degrees (half circle).

    Small for semicircular arcs; large both for closed/near-closed shapes
    and for shallow arcs.
    """
    return abs(p_arc_extent_deg(panel) - 180.0)


def p_circle_fit_rms(panel) -> float:
    """RMS radial residual of best-fit circle, normalized by radius."""
    xs, ys = _ink_xy(panel)
    if xs.size < 3:
        return 0.0
    cx, cy, r = _fit_circle(xs, ys)
    d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    return float(np.sqrt(np.mean((d - r) ** 2)) / max(r, 1e-9))


def p_chord_over_arclen(panel) -> float:
    """Distance between the two curve endpoints divided by ink pixel count.

    ~0.64 for a semicircular arc; 0 for closed curves (no endpoints);
    approaches 1 for a straight/shallow segment.
    """
    p = (panel > 0).astype(int)
    nb = _neighbor_counts(panel)
    eys, exs = np.nonzero((p == 1) & (nb == 1))
    if eys.size != 2:
        return 0.0
    d = math.hypot(float(exs[0] - exs[1]), float(eys[0] - eys[1]))
    n = float(p.sum())
    return d / max(n, 1.0)


def p_num_holes(panel) -> float:
    """Number of enclosed background regions (holes) in the drawing.

    Background is 4-connected (complementary to 8-connected ink); a hole is
    a background component that does not touch the image border.
    """
    bg = (panel == 0)
    lbl, n = ndimage.label(bg)  # 4-connectivity by default
    if n == 0:
        return 0.0
    border = np.unique(np.concatenate([
        lbl[0, :], lbl[-1, :], lbl[:, 0], lbl[:, -1]
    ]))
    border = set(int(v) for v in border if v != 0)
    holes = sum(1 for v in range(1, n + 1) if v not in border)
    return float(holes)
