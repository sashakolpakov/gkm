# Shared predicate library. p_<name>(panel) -> float | bool
import numpy as np
from scipy.spatial import cKDTree


def _xy(panel):
    ys, xs = np.nonzero(panel)
    return xs.astype(float), ys.astype(float)


def _order_curve(panel):
    """Trace the ink pixels of a single-stroke curve into a spatially ordered
    polyline via greedy nearest-neighbor walk, starting from a pixel with few
    close neighbors (a likely endpoint/tip). Reusable whenever a predicate
    needs to walk along a drawn curve (e.g. to split it into sub-arcs) rather
    than treat it as an unordered point cloud. Assumes a single connected
    stroke without heavy branching; not meaningful for scattered/multi-blob
    panels."""
    xs, ys = _xy(panel)
    pts = np.stack([xs, ys], axis=1)
    n = len(pts)
    if n < 3:
        return pts
    tree = cKDTree(pts)
    counts = np.array([len(c) for c in tree.query_ball_point(pts, r=3.0)])
    start = int(np.argmin(counts))
    used = np.zeros(n, dtype=bool)
    order = [start]
    used[start] = True
    cur = start
    for _ in range(n - 1):
        d = np.linalg.norm(pts - pts[cur], axis=1)
        d[used] = np.inf
        nxt = int(np.argmin(d))
        if not np.isfinite(d[nxt]):
            break
        order.append(nxt)
        used[nxt] = True
        cur = nxt
    return pts[order]


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


def _arc_span_from_points(xs, ys, cx, cy):
    ang = np.sort(np.arctan2(ys - cy, xs - cx))
    gaps = np.diff(ang)
    gaps = np.append(gaps, ang[0] + 2 * np.pi - ang[-1])
    return float(np.degrees(2 * np.pi - gaps.max()))


def _best_two_arc_split(panel):
    """Walk the curve (see `_order_curve`) and find the single split point
    that, when the curve is cut into a first and second piece and each piece
    is independently circle-fit, minimizes the summed circle-fit RMS
    residual. Returns (split_index, n_points, (cx1,cy1,r1), (cx2,cy2,r2)).
    This locates the true corner/inflection of a two-circular-arc curve
    (e.g. a wave/S-shape) without relying on noisy per-pixel curvature,
    which is fragile at 1px scale. Not meaningful for curves that aren't
    reasonably close to two arcs (chaotic shapes still return *a* split, just
    a meaningless one -- pair with a residual-quality check if that
    distinction matters)."""
    pts = _order_curve(panel)
    n = len(pts)
    if n < 25:
        return None
    best = None
    for split in range(10, n - 10):
        xs1, ys1 = pts[:split, 0], pts[:split, 1]
        xs2, ys2 = pts[split:, 0], pts[split:, 1]
        cx1, cy1, r1 = _fit_circle(xs1, ys1)
        cx2, cy2, r2 = _fit_circle(xs2, ys2)
        resid1 = np.sqrt(np.mean((np.hypot(xs1 - cx1, ys1 - cy1) - r1) ** 2))
        resid2 = np.sqrt(np.mean((np.hypot(xs2 - cx2, ys2 - cy2) - r2) ** 2))
        total = resid1 + resid2
        if best is None or total < best[0]:
            best = (total, split, (cx1, cy1, r1), (cx2, cy2, r2))
    _, split, c1, c2 = best
    return split, n, c1, c2


def p_two_arc_span_sum_deviation(panel):
    """|sum of the two arcs' angular spans (degrees), from the best
    two-circular-arc split of the curve (see `_best_two_arc_split`) - 297|.
    A clean two-arc "wave" curve (two roughly semicircle-ish bumps joined at
    a corner) has a fairly fixed total turning of ~297 degrees regardless of
    scale or which way the arcs bend; this is near zero for that shape and
    large for a single clean arc (much less total span), a closed loop/lens
    (also much less, since each "half" is a short tip-to-tip arc), or a
    messy multi-corner shape (usually much more). Target of 297 and the
    two-sided deviation (rather than a one-sided span threshold) were tuned
    against problem_02's near-miss curves -- see predicates_log.md."""
    res = _best_two_arc_split(panel)
    if res is None:
        return 297.0
    split, n, c1, c2 = res
    pts = _order_curve(panel)
    xs1, ys1 = pts[:split, 0], pts[:split, 1]
    xs2, ys2 = pts[split:, 0], pts[split:, 1]
    s1 = _arc_span_from_points(xs1, ys1, c1[0], c1[1])
    s2 = _arc_span_from_points(xs2, ys2, c2[0], c2[1])
    return float(abs((s1 + s2) - 297.0))


def p_arc_defect_score_217(panel):
    """Defect score (see `_arc_defect_score`) targeting a 217-degree
    (a bit past a semicircle) arc span. Uses a tighter span_scale than the
    generic default: the closest near-miss for this target is a clean arc
    only ~25 degrees too long (plus a small hook), so the span term needs
    more weight to stay robustly above the positives under leave-one-out
    (see predicates_log.md)."""
    return _arc_defect_score(panel, 217.0, span_scale=4.0)
