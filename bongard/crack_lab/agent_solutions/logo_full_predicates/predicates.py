# Shared predicate library. p_<name>(panel) -> float | bool
import numpy as np


def _xy(panel):
    ys, xs = np.nonzero(panel)
    return xs.astype(float), ys.astype(float)


def _fit_circle(xs, ys):
    A = np.c_[2 * xs, 2 * ys, np.ones(len(xs))]
    b = xs ** 2 + ys ** 2
    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c = sol
    r = np.sqrt(max(c + cx ** 2 + cy ** 2, 1e-9))
    return cx, cy, r


def p_circle_fit_residual(panel):
    """RMS distance of ink pixels from the best-fit circle, normalized by bbox diagonal.
    Low for a clean single circular arc; high for S-curves, straight lines, or multi-loop shapes."""
    xs, ys = _xy(panel)
    if len(xs) < 5:
        return 0.0
    cx, cy, r = _fit_circle(xs, ys)
    d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    resid = d - r
    diag = np.hypot(xs.max() - xs.min(), ys.max() - ys.min())
    if diag < 1e-6:
        return 0.0
    return float(np.sqrt(np.mean(resid ** 2)) / diag)


def p_arc_angular_span_deg(panel):
    """Angular span (degrees) subtended at the fitted circle's center by the ink pixels."""
    xs, ys = _xy(panel)
    if len(xs) < 5:
        return 0.0
    cx, cy, r = _fit_circle(xs, ys)
    ang = np.unwrap(np.sort(np.arctan2(ys - cy, xs - cx)))
    span = ang.max() - ang.min()
    return float(np.degrees(span))


def p_arc_span_deviation_from_120(panel):
    """Absolute difference (degrees) between the arc's angular span and 120 degrees
    (a 1/3-circle arc). Near zero for such arcs, large for other spans (near-straight,
    near-full-circle, or S-curves whose span is measured around the wrong center)."""
    return float(abs(p_arc_angular_span_deg(panel) - 120.0))


def p_arc_defect_score(panel):
    """Composite 'how far from a clean single 120-degree circular arc' score: the
    worse of (a) how badly the ink deviates from any circle, and (b) how far the
    arc's angular span is from 120 degrees, each normalized to a comparable scale.
    Near zero for a clean 1/3-circle arc; large for S-curves, straight segments,
    multi-loop shapes, or arcs of a different span."""
    resid = p_circle_fit_residual(panel) / 0.01
    dev = p_arc_span_deviation_from_120(panel) / 6.0
    return float(max(resid, dev))
