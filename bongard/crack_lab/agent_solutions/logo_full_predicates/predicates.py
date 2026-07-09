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


def _arc_angular_span_deg(panel):
    """Angular span (degrees) subtended at the fitted circle's center by the ink
    pixels, computed as 360 minus the single largest angular gap between
    consecutive pixels (sorted by angle around the center). This correctly
    handles arcs of any span (including >180 degrees), unlike a naive
    max-minus-min which only works when the arc doesn't wrap the +-180
    boundary. Not exposed as its own predicate: a raw one-sided threshold on
    span is fragile under leave-one-out (a single short-arc near-miss with
    low span can look like a clean separator once held out). Always used
    through a deviation-from-target-angle wrapper instead."""
    xs, ys = _xy(panel)
    if len(xs) < 5:
        return 0.0
    cx, cy, r = _fit_circle(xs, ys)
    ang = np.sort(np.arctan2(ys - cy, xs - cx))
    gaps = np.diff(ang)
    gaps = np.append(gaps, ang[0] + 2 * np.pi - ang[-1])
    span = 2 * np.pi - gaps.max()
    return float(np.degrees(span))


def _arc_span_deviation(panel, target_deg):
    """Absolute difference (degrees) between the arc's angular span and a
    target span. Near zero for arcs of that span, large for near-full
    circles/loops or shorter/longer arcs. Not exposed directly -- a raw
    span-deviation predicate is fragile under leave-one-out for non-circular
    negatives (their 'span' from a bad circle fit is meaningless noise that
    can coincidentally land inside any one target's near-zero band). Always
    combined with the circle-fit residual via `_arc_defect_score` instead."""
    return float(abs(_arc_angular_span_deg(panel) - target_deg))


def _arc_defect_score(panel, target_deg, span_scale=6.0):
    """Composite 'how far from a clean single circular arc of target_deg span'
    score: the worse of (a) how badly the ink deviates from any circle, and
    (b) how far the arc's angular span is from target_deg, each normalized to
    a comparable scale. Near zero for a clean arc of that span; large for
    S-curves, straight segments, multi-loop shapes, or arcs of a different
    span. Requiring both a good circle fit AND the right span (rather than
    span alone) keeps distinct target-span predicates from accidentally
    tying with each other on unrelated problems' data during leave-one-out."""
    resid = p_circle_fit_residual(panel) / 0.01
    dev = _arc_span_deviation(panel, target_deg) / span_scale
    return float(max(resid, dev))


def p_arc_defect_score_arc120(panel):
    """Defect score (see `_arc_defect_score`) targeting a 120-degree
    (1/3-circle) arc span.

    Named "..._arc120" rather than a bare "p_arc_defect_score" deliberately:
    the rule selector breaks F-ties between equally-good rules by comparing
    their describe() strings lexically. A bare name is a *string prefix* of
    any suffixed variant (e.g. "..._217"), so it wins every tie regardless
    of which target is actually correct ('<' sorts before '_'). Cross-
    problem ties are real, not hypothetical: a clean arc of one problem's
    target span can coincidentally also score as a clean arc of another
    problem's target span once the one negative example that would rule it
    out is itself the one held out during leave-one-out. Prefixing with a
    letter (ASCII > any digit) after the shared "_" keeps this predicate's
    ties from being won by bare/digit-prefixed variants purely by accident
    of naming -- see predicates_log.md for the concrete collision this
    fixed."""
    return _arc_defect_score(panel, 120.0)


def p_arc_defect_score_217(panel):
    """Defect score (see `_arc_defect_score`) targeting a 217-degree
    (a bit past a semicircle) arc span. Uses a tighter span_scale than the
    generic default: the closest near-miss for this target is a clean arc
    only ~25 degrees too long (plus a small hook), so the span term needs
    more weight to stay robustly above the positives under leave-one-out
    (see predicates_log.md)."""
    return _arc_defect_score(panel, 217.0, span_scale=4.0)
