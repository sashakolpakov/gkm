# Shared predicate library. p_<name>(panel) -> float | bool
import numpy as np
from scipy.spatial import cKDTree, ConvexHull
from scipy.ndimage import binary_fill_holes, binary_dilation


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


def _simplify_dp(pts, eps=3.5):
    """Douglas-Peucker simplification of an ordered polyline (as returned by
    `_order_curve`) down to its dominant corner vertices, dropping points
    within `eps` pixels of the straight chord between two kept points.
    Reusable whenever a predicate needs the *vertices* of a drawn polygon
    (to measure corner angles, convexity, perimeter, ...) rather than the
    raw, densely-sampled outline pixels."""
    def rdp(p):
        if len(p) < 3:
            return p
        start, end = p[0], p[-1]
        line = end - start
        norm = np.linalg.norm(line)
        if norm < 1e-9:
            d = np.linalg.norm(p - start, axis=1)
        else:
            cr = line[0] * (p[:, 1] - start[1]) - line[1] * (p[:, 0] - start[0])
            d = np.abs(cr) / norm
        idx = int(np.argmax(d))
        if d[idx] > eps:
            left = rdp(p[:idx + 1])
            right = rdp(p[idx:])
            return np.vstack([left[:-1], right])
        else:
            return np.vstack([start, end])
    return rdp(pts)


def _polygon_turns(panel, eps=3.5):
    """Trace a panel's ink into an ordered, corner-simplified closed
    polygon (via `_order_curve` + `_simplify_dp`, always closing the loop
    back to the start) and return the per-vertex signed turning angle
    (normalized cross product of consecutive edge directions, in
    [-1, 1] ~ sin(turn angle)). Reusable for any predicate about polygon
    corners: convexity, reflex-vertex counting, sharpest-spike angle, etc.
    Assumes a single-stroke outline without heavy branching (see
    `_order_curve`); meaningless for scattered point clouds."""
    pts = _order_curve(panel)
    simp = _simplify_dp(pts, eps=eps)
    simp = np.vstack([simp, simp[0]])
    n = len(simp) - 1
    turns = np.zeros(n)
    for i in range(n):
        p0, p1, p2 = simp[i - 1], simp[i], simp[(i + 1) % n]
        v1, v2 = p1 - p0, p2 - p1
        l1, l2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if l1 < 1e-6 or l2 < 1e-6:
            continue
        turns[i] = (v1[0] * v2[1] - v1[1] * v2[0]) / (l1 * l2)
    return turns, simp


def p_reflex_vertex_count(panel, min_turn_ratio=0.12):
    """Count of reflex (concave) corners in a panel's traced outline
    polygon: vertices whose turn direction is opposite the shape's
    majority turn direction, ignoring near-straight vertices (|turn| below
    `min_turn_ratio`) so pixelation jitter along smooth/near-straight
    stretches doesn't get miscounted as concavity. A convex polygon scores
    0; a shape with one or more notches/spikes cut into it (or, for a
    self-crossing curve, extra corners created by the crossing) scores
    higher."""
    turns, _ = _polygon_turns(panel)
    if len(turns) == 0:
        return 0.0
    majority_sign = 1 if (turns > 0).sum() >= (turns < 0).sum() else -1
    sig = np.abs(turns) > min_turn_ratio
    return float((((turns * majority_sign) < 0) & sig).sum())


def p_perim_convexhull_ratio(panel):
    """Ratio of a panel's traced-polygon perimeter to its point cloud's
    convex hull perimeter. Near 1.0 for a (near-)convex shape, since the
    hull then hugs the outline closely; much greater than 1.0 for a shape
    whose boundary winds in and out a lot relative to its hull (deep
    notches, thin spiky protrusions, multi-lobe/self-crossing curves)."""
    pts = _order_curve(panel)
    simp = _simplify_dp(pts)
    simp_c = np.vstack([simp, simp[0]])
    perim = np.sum(np.linalg.norm(np.diff(simp_c, axis=0), axis=1))
    xs, ys = _xy(panel)
    hull = ConvexHull(np.stack([xs, ys], axis=1))
    hull_perim = hull.area  # 2D ConvexHull: .area is the perimeter
    return float(perim / hull_perim) if hull_perim > 0 else 0.0


def _polygon_turn_degrees(panel, eps=3.5):
    """Like `_polygon_turns` but returns the signed exterior turn angle in
    degrees (via atan2 of consecutive edge directions) instead of a
    normalized cross product. Distinguishes a merely-bent corner from one
    that folds back sharply on itself (turn magnitude approaching 180 deg)
    in a way a sin-like normalized cross product cannot (sin saturates and
    turns back down past 90 deg)."""
    pts = _order_curve(panel)
    simp = _simplify_dp(pts, eps=eps)
    simp = np.vstack([simp, simp[0]])
    n = len(simp) - 1
    degs = np.zeros(n)
    for i in range(n):
        p0, p1, p2 = simp[i - 1], simp[i], simp[(i + 1) % n]
        v1, v2 = p1 - p0, p2 - p1
        if np.linalg.norm(v1) < 1e-6 or np.linalg.norm(v2) < 1e-6:
            continue
        a1 = np.degrees(np.arctan2(v1[1], v1[0]))
        a2 = np.degrees(np.arctan2(v2[1], v2[0]))
        degs[i] = (a2 - a1 + 180) % 360 - 180
    return degs


def p_max_reflex_angle_deg(panel, min_deg=8.0):
    """Sharpest concave (reflex) corner in a panel's traced outline, in
    degrees of exterior turn (0 = straight through, 180 = folds
    completely back on itself). Only considers corners turning against the
    shape's majority turn direction (the concave ones), ignoring
    near-straight vertices below `min_deg`. A shape with no meaningful
    concave corner (fully convex, or an open curve with only mild bends)
    scores 0; a shape with a deep, narrow concave notch or a self-crossing
    pinch scores close to 180."""
    degs = _polygon_turn_degrees(panel)
    if len(degs) == 0:
        return 0.0
    sig = np.abs(degs) > min_deg
    pos_c = ((degs > 0) & sig).sum()
    neg_c = ((degs < 0) & sig).sum()
    majority = 1 if pos_c >= neg_c else -1
    reflex_mask = sig & (np.sign(degs) != majority)
    if not reflex_mask.any():
        return 0.0
    return float(np.max(np.abs(degs[reflex_mask])))


def p_notch_severity(panel):
    """Combined concavity-severity score: `p_reflex_vertex_count` (how many
    concave corners the outline has) times `p_perim_convexhull_ratio` (how
    much the outline's perimeter overshoots its convex hull's). Either
    signal alone has a confusable near-miss: a purely convex but very
    thin/spiky shape (sharp acute corners, no reflex vertices) inflates
    the perimeter ratio without any reflex corners, while a shape with
    reflex corners but only shallow ones has a middling ratio. Requiring
    both -- reflex corners AND a large perimeter overshoot -- multiplies
    to a low score whenever either factor is nearly zero, cleanly
    separating shapes with genuine deep/plentiful concave notches from
    both convex-spiky shapes and mildly-concave ones."""
    return float(p_reflex_vertex_count(panel) * p_perim_convexhull_ratio(panel))


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


def _filled_mask(panel):
    """Fill the interior enclosed by a (possibly self-crossing) closed
    curve's ink pixels: dilate by 1px to seal 1-pixel gaps at self-crossings
    then flood-fill. Reusable whenever a predicate needs the enclosed area
    of a curve rather than just its outline pixels."""
    return binary_fill_holes(binary_dilation(panel.astype(bool), iterations=1))


def p_180_rotational_self_iou(panel):
    """IoU of a shape's filled interior with itself rotated 180 degrees
    about its own centroid. Shapes with 2-fold rotational (point) symmetry
    -- e.g. an S/pinwheel curve made of two similar bumps on opposite sides
    of a center -- score high (~0.6+) since the rotated copy nearly
    reproduces the original. Shapes lacking that symmetry -- a single open
    arc, or several lens/petal lobes all meeting at one shared hub point
    rather than arranged around a center -- score much lower (~0.3 or
    below), since the rotated copy lands on mostly empty space instead.
    Named with a leading digit (`p_180_...`) so its describe() string sorts
    lexically before other predicates' names (`p_arc_...`, `p_circle_...`):
    several existing predicates happen to also reach 0 training error on a
    given problem's 12 panels by coincidence, and the harness's rule
    selector breaks ties between equally-good (error, cost) rules by
    picking the lexically smallest description, independent of which one
    is actually robust under leave-one-out. See predicates_log.md
    problem_03 for the concrete case this mattered."""
    f = _filled_mask(panel)
    if f.sum() == 0:
        return 0.0
    f180 = np.rot90(f, 2)
    ys, xs = np.nonzero(f)
    cy, cx = ys.mean(), xs.mean()
    ys2, xs2 = np.nonzero(f180)
    cy2, cx2 = ys2.mean(), xs2.mean()
    dy, dx = int(round(cy - cy2)), int(round(cx - cx2))
    f180s = np.roll(np.roll(f180, dy, axis=0), dx, axis=1)
    inter = np.logical_and(f, f180s).sum()
    union = np.logical_or(f, f180s).sum()
    return float(inter / union) if union > 0 else 0.0
