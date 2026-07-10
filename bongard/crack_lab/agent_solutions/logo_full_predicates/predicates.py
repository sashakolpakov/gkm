# Shared predicate library. p_<name>(panel) -> float | bool
import itertools
import math
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


def p_self_proximity_ratio(panel, min_gap_frac=0.15):
    """Minimum spatial distance between two points of the traced curve (see
    `_order_curve`) that are far apart in curve-parameter (cyclic index
    separation >= min_gap_frac of the curve length), normalized by the
    ink's bbox diagonal. Near zero when the outline comes close to touching
    or crossing itself somewhere away from a simple local corner (a true
    self-intersection, or a deep concave notch whose two sides nearly meet) --
    both read the same way here: the curve's far-apart-in-parameter points
    get spatially close. Stays large (no pinch) for shapes where every
    widely-separated pair of curve points stays well apart, e.g. convex or
    mildly concave polygons with no near-self-touching feature."""
    pts = _order_curve(panel)
    n = len(pts)
    if n < 10:
        return 1.0
    diag = np.hypot(*(pts.max(axis=0) - pts.min(axis=0)))
    if diag < 1e-6:
        return 1.0
    min_gap = max(int(n * min_gap_frac), 5)
    idx = np.arange(n)
    best = np.inf
    for i in range(0, n, 2):
        d = np.linalg.norm(pts - pts[i], axis=1)
        sep = np.minimum(np.abs(idx - i), n - np.abs(idx - i))
        mask = sep >= min_gap
        if mask.any():
            m = d[mask].min()
            if m < best:
                best = m
    return float(best / diag)


def _solidity(panel):
    """Ratio of the curve's enclosed (filled) area (see `_filled_mask`) to
    the area of the convex hull of the SAME filled mask's pixels (not the
    raw outline pixels -- using the filled mask for both keeps the ratio
    <=1; the 1px dilation `_filled_mask` applies to seal self-crossing gaps
    would otherwise inflate the filled area past the raw-outline hull for
    small/thin shapes). Near 1 for convex or near-convex blobs (circle,
    triangle, hexagon); well below 1 for shapes with deep concave notches,
    narrow necks, or self-crossing loops, since those give up interior area
    relative to their convex hull without shrinking the hull itself."""
    f = _filled_mask(panel)
    area = float(f.sum())
    ys, xs = np.nonzero(f)
    pts = np.stack([xs, ys], axis=1)
    if len(pts) < 3:
        return 1.0
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 1.0
    hull_area = hull.volume
    if hull_area < 1e-6:
        return 1.0
    return float(area / hull_area)


def p_pinch_notch_defect(panel, pinch_scale=0.015, solidity_scale=0.85):
    """Composite 'does this shape have a self-touching pinch AND give up
    hull area to a notch/neck' defect score: max of (a) `p_self_proximity_ratio`
    normalized by pinch_scale, and (b) `_solidity` normalized by
    solidity_scale. Combining via max() implements an AND of "has a pinch"
    and "is non-convex enough to matter" in a single scalar (same pattern as
    `_arc_defect_score`), which is needed here because either signal alone
    has a near-miss: a thin convex sliver can have as small a pinch distance
    as a true self-crossing (its opposite edges taper close together without
    ever touching), and a smooth concave crescent can have as low solidity
    as a pinched shape without ever coming close to touching itself. Small
    (<1) only when a shape has both a near-self-touching feature AND
    meaningful concavity -- i.e. a self-crossing loop, or a deep notch whose
    two sides nearly meet; large otherwise (convex blobs, thin convex
    slivers, smooth concave curves with no pinch, open curves)."""
    pinch = p_self_proximity_ratio(panel) / pinch_scale
    sol = _solidity(panel) / solidity_scale
    return float(max(pinch, sol))


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


def _end_widths(panel, tip_frac=0.05):
    """Width (perpendicular extent) of the filled shape near each end of its
    major (PCA) axis, and near its middle. Reusable whenever a predicate
    needs to characterize how a shape tapers along its long axis (e.g. a
    wedge/dart that narrows to a point at one end but stays full-width at
    the other, vs a lens that narrows at both ends, vs a blob that narrows
    at neither)."""
    f = _filled_mask(panel)
    ys, xs = np.nonzero(f)
    pts = np.stack([xs, ys], axis=1).astype(float)
    c = pts.mean(axis=0)
    pts_c = pts - c
    cov = np.cov(pts_c.T)
    evals, evecs = np.linalg.eigh(cov)
    major = evecs[:, np.argmax(evals)]
    minor = evecs[:, np.argmin(evals)]
    t = pts_c @ major
    u = pts_c @ minor
    tmin, tmax = t.min(), t.max()
    length = tmax - tmin
    if length < 1e-6:
        return 0.0, 0.0, 1e-6

    def width_at(lo, hi):
        mask = (t >= lo) & (t <= hi)
        if mask.sum() == 0:
            return 0.0
        return float(u[mask].max() - u[mask].min())

    w_start = width_at(tmin, tmin + tip_frac * length)
    w_end = width_at(tmax - tip_frac * length, tmax)
    w_mid = width_at(tmin + 0.45 * length, tmin + 0.55 * length)
    return w_start, w_end, max(w_mid, 1e-6)


def p_0_blunt_tip_defect(panel):
    """How far the WIDER of the shape's two long-axis ends is from being as
    wide as the shape's middle, normalized by that middle width. Near zero
    for a wedge/dart/blade: one end tapers to a point, but the other end is
    a flat cut that stays essentially the full body width right up to its
    tip. Large for a lens/eye shape (both ends taper, so even the 'wider'
    end is much narrower than the middle), a blunt chunky polygon (neither
    end reaches the middle's full width, since the widest cross-section
    sits in the interior rather than at an end), or a concave arrow/notched
    shape (the notch makes the middle narrower than the ends, so the ratio
    overshoots past 1 instead of approaching it). Named with a leading
    `0` (`p_0_...`) so it sorts lexically before `p_180_rotational_self_iou`,
    which also happens to reach 0 training error on this problem by
    coincidence but with a much thinner margin (~0.07 vs ~0.4 here) --
    see predicates_log.md problem_05 and the `p_180_...` docstring for why
    the selector's naming tie-break otherwise picks the fragile predicate."""
    w_start, w_end, w_mid = _end_widths(panel)
    return float(abs(max(w_start, w_end) / w_mid - 1.0))


def _fill_ratio(panel):
    """Filled-interior pixel count divided by raw ink pixel count (see
    `_filled_mask`). Low (~2-3) for an open curve/stroke (the flood-fill
    only catches a thin sliver near the ink, since there's no enclosed
    region), much higher (~6+) for any closed loop, regardless of the
    loop's size or how convex/concave it is. A cheap, robust open-vs-closed
    discriminator that doesn't depend on tracing/ordering the curve."""
    ink = float(np.asarray(panel).sum())
    if ink < 1:
        return 0.0
    return float(_filled_mask(panel).sum()) / ink


def p_open_or_symmetric_defect(panel, open_ratio_thresh=4.0, sym_target=0.9):
    """'Is this shape either a single open curve, or a closed shape with
    strong 180-degree point symmetry' defect -- near zero if EITHER holds,
    large otherwise. Combines two already-existing signals (`_fill_ratio`
    and `p_180_rotational_self_iou`) via min(), the same OR-via-min pattern
    used elsewhere in this library (e.g. `_arc_defect_score`'s AND-via-max):
    an open stroke (arc, wave, or self-crossing X) is judged solely on being
    open, since fill-ratio-based symmetry is meaningless for it; a closed
    shape is judged solely on point symmetry. Distinguishes freeform open
    curves and symmetric closed blobs/lenses/wavy-quads from closed shapes
    that are both non-open and asymmetric -- e.g. a polygon with one
    lopsided concave scoop, or two straight-edged sub-shapes meeting at a
    point, both of which break 180-degree symmetry without being open.
    sym_target of 0.9 (rather than a lower value) was chosen because the
    closest near-miss negative here is itself moderately symmetric
    (~0.79) -- see predicates_log.md problem_09."""
    ratio = _fill_ratio(panel)
    defect_open = max(0.0, ratio - open_ratio_thresh)
    defect_sym = max(0.0, sym_target - p_180_rotational_self_iou(panel))
    return float(min(defect_open, defect_sym))


def _densest_point(panel, r=3.0):
    """The ink pixel with the most other ink pixels within radius r --
    a local-density peak. Reusable as a cheap locator for a drawing's
    single busiest spot, e.g. the point where two sub-shapes meet (either
    a shared vertex or a line-on-line crossing)."""
    xs, ys = _xy(panel)
    pts = np.stack([xs, ys], axis=1)
    tree = cKDTree(pts)
    counts = np.array([len(c) for c in tree.query_ball_point(pts, r=r)])
    idx = int(np.argmax(counts))
    return pts[idx]


def _branch_angles(panel, center, r1, r2, gap_deg=14.0):
    """Directions (degrees) of the distinct rays of ink radiating from
    `center`, measured from the pixels in the annulus [r1, r2] around it,
    found by sorting those pixels' angles and splitting wherever there is a
    gap of more than gap_deg. Reusable whenever a predicate needs to know
    how many strokes meet at a point, and in which directions, e.g. to
    distinguish a true polygon vertex from a straight-through line crossing."""
    xs, ys = _xy(panel)
    pts = np.stack([xs, ys], axis=1)
    d = np.linalg.norm(pts - center, axis=1)
    sel = pts[(d >= r1) & (d <= r2)]
    if len(sel) == 0:
        return []
    vecs = sel - center
    angs = np.sort(np.degrees(np.arctan2(vecs[:, 1], vecs[:, 0])))
    clusters = []
    cur = [angs[0]]
    for a in angs[1:]:
        if a - cur[-1] <= gap_deg:
            cur.append(a)
        else:
            clusters.append(cur)
            cur = [a]
    clusters.append(cur)
    if len(clusters) > 1 and (clusters[0][0] + 360 - clusters[-1][-1]) <= gap_deg:
        clusters[0] = clusters[-1] + clusters[0]
        clusters.pop()
    return [float(np.mean(c)) for c in clusters]


def _four_ray_crossing_defect(angles):
    """Given exactly 4 ray directions (degrees) radiating from a point,
    the best pairing into two pairs of opposite (180-degree-apart) rays,
    scored as the summed deviation from 180 of each pair. Near zero when
    the 4 rays form two straight lines passing through the point (a true
    line-on-line crossing); large when the 4 rays are a genuine polygon
    vertex (no two rays continue straight through each other). Returns 90.0
    (a large sentinel) if not given exactly 4 angles, since the "crossing"
    question isn't meaningful for any other ray count."""
    if len(angles) != 4:
        return 90.0
    best = 1e9
    for a, b, c, d in itertools.permutations(range(4)):
        if a > b or c > d or a > c:
            continue
        diff1 = abs(abs(angles[a] - angles[b]) % 360 - 180)
        diff1 = min(diff1, 360 - diff1)
        diff2 = abs(abs(angles[c] - angles[d]) % 360 - 180)
        diff2 = min(diff2, 360 - diff2)
        best = min(best, diff1 + diff2)
    return best


def p_line_crossing_defect(panel):
    """How cleanly two sub-shapes joined at a single point are joined by an
    actual line-on-line CROSSING (two straight lines passing through each
    other, X-style) rather than merely touching corner-to-corner at a
    shared polygon vertex. Locates the join (`_densest_point`) and, over a
    range of small measurement radii around it, takes the best (smallest)
    `_four_ray_crossing_defect` score across those radii -- using the best
    rather than e.g. the average is deliberate: pixelation makes any single
    radius's ray-clustering noisy, but a true crossing has SOME radius where
    the two straight lines read out cleanly, whereas a true shared-vertex
    join never presents two opposite ray pairs at any radius. Near zero for
    a true crossing; large (order 40+) for a shared-vertex touch."""
    p = np.asarray(panel)
    center = _densest_point(p)
    best = 90.0
    for r1 in range(3, 12):
        for r2 in range(r1 + 4, r1 + 12):
            angles = _branch_angles(p, center, r1, r2)
            best = min(best, _four_ray_crossing_defect(angles))
    return float(best)


def _split_touching_pair(panel, r=5.0):
    """Split a panel's ink into two sub-shapes that meet at a single shared
    point, by removing a small disk around the busiest point (`_densest_point`)
    and 8-connected-labeling what remains. Returns a list of (xs, ys) pixel
    coordinate arrays, one per sub-shape, plus the joint location -- or None
    if the ink doesn't separate into exactly two pieces this way (a single
    shape, or a join with more than two branches). Reusable whenever a
    predicate needs to reason about two touching sub-shapes independently
    (e.g. compare their sizes, fit each one's own circle)."""
    from scipy import ndimage
    p = np.asarray(panel)
    center = _densest_point(p, r=3.0)
    xs, ys = _xy(p)
    pts = np.stack([xs, ys], axis=1)
    d = np.linalg.norm(pts - center, axis=1)
    mask = np.zeros_like(p)
    keep = pts[d > r].astype(int)
    if len(keep) == 0:
        return None
    mask[keep[:, 1], keep[:, 0]] = 1
    lbl, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n != 2:
        return None
    parts = []
    for k in (1, 2):
        ys_k, xs_k = np.nonzero(lbl == k)
        parts.append((xs_k.astype(float), ys_k.astype(float)))
    return parts, center


def _pca_extents(xs, ys):
    """Extents (lengths) of a point set along its major and minor PCA axes.
    Reusable as a scale-invariant, orientation-invariant width/length
    measurement for any roughly-rectangular or elongated blob of points."""
    pts = np.stack([xs, ys], axis=1)
    c = pts.mean(axis=0)
    pts_c = pts - c
    cov = np.cov(pts_c.T)
    evals, evecs = np.linalg.eigh(cov)
    proj = pts_c @ evecs
    major = proj[:, np.argmax(evals)]
    minor = proj[:, np.argmin(evals)]
    return float(major.max() - major.min()), float(minor.max() - minor.min())


def p_0_fan_quad_ratio_defect(panel, target=1.191):
    """For a panel made of two sub-shapes touching at one point -- a circular
    fan/sector and a quadrilateral -- how far the ratio of the fan's own
    long PCA extent to the quadrilateral's long PCA extent is from a fixed
    target, which this problem's positives hit tightly (~1.17-1.22: the fan
    is consistently a bit longer than the quadrilateral). Identifies which
    of the two `_split_touching_pair` pieces is the fan by taking the one
    with the lower whole-piece circle-fit residual (the fan's arc pixels sit
    close to a true circle even though its two straight radii don't; a pure
    quadrilateral's pixels fit a circle far worse). Returns a large sentinel
    (5.0) if the panel doesn't split into exactly two touching pieces (a
    single shape alone, or more than two branches at the join) since the
    ratio isn't meaningful there -- which also correctly flags those cases as
    violating the two-touching-shapes structure itself. Tried the more
    obvious fan-radius / quad-short-side ratio first, but its closest
    negative sat only ~0.08 away from the positive band while the next
    negative sat ~0.24 away -- an LOO-fragile gap structure like problem_00's
    lesson, since dropping that one negative during leave-one-out let the
    fitted threshold drift past it. The fan-long / quad-long ratio instead
    gives every negative a similarly large (~0.2+) margin from the tight
    positive band."""
    split = _split_touching_pair(panel)
    if split is None:
        return 5.0
    parts, _ = split
    resids = []
    for xs, ys in parts:
        if len(xs) < 5:
            return 5.0
        cx, cy, r = _fit_circle(xs, ys)
        d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        diag = np.hypot(xs.max() - xs.min(), ys.max() - ys.min())
        resids.append(float(np.std(d)) / (diag + 1e-9))
    fan_i = 0 if resids[0] < resids[1] else 1
    quad_i = 1 - fan_i
    fan_long = max(_pca_extents(*parts[fan_i]))
    quad_long = max(_pca_extents(*parts[quad_i]))
    if quad_long < 1e-6:
        return 5.0
    ratio = fan_long / quad_long
    return float(abs(ratio - target))


def _hull_area_over_diag2(panel):
    """Convex hull area of the raw ink pixels, normalized by the bbox
    diagonal squared. A scale-invariant 'chunkiness' measurement: near
    0.10-0.12 for a thin lens/leaf outline (its hull is a narrow sliver
    relative to its own diagonal), much higher (0.3+) for a blunter closed
    polygon (zigzag, pentagon, chevron) whose hull fills a bigger fraction
    of its own bounding diagonal."""
    xs, ys = _xy(panel)
    if len(xs) < 3:
        return 0.0
    pts = np.stack([xs, ys], axis=1)
    diag2 = (xs.max() - xs.min()) ** 2 + (ys.max() - ys.min()) ** 2
    if diag2 < 1e-6:
        return 0.0
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 0.0
    return float(hull.volume / diag2)


def p_0_asym_taper_or_blunt_or_chunky_defect(panel, ratio_thresh=1.9, ratio_scale=0.1,
                                              blunt_thresh=0.2, chunky_thresh=0.15):
    """'Is this a clean symmetric lens/leaf -- two similarly-curved arcs
    tapering to a point at BOTH ends, with a hull that stays a thin sliver
    of its own bbox diagonal' defect score, as the max (OR of failure modes)
    of three independent checks:
    (a) asymmetric taper -- the two arcs from `_best_two_arc_split` have very
    different fitted radii (one side noticeably flatter/straighter than the
    other), unlike a lens's two comparably-bulged sides;
    (b) blunt tip -- `p_0_blunt_tip_defect` near 0, meaning one end is a flat
    cut rather than a taper to a point (a wedge/pencil shape);
    (c) chunky hull -- `_hull_area_over_diag2` well above a thin-sliver
    band, meaning the shape isn't a thin sliver at all (a chunkier polygon).
    Each check alone has a near-miss that lands inside the lens band on this
    problem's negatives (a pencil's flat end is itself made of two nearly-
    straight sides, so its own two-arc radius ratio is unremarkable; a
    lopsided lens's radius ratio is the only signal that catches it, since
    it's neither blunt nor chunky) -- combining via max() covers all three
    failure modes with one scalar, same OR-via-min/max pattern as
    `p_open_or_symmetric_defect` and `p_pinch_notch_defect`."""
    res = _best_two_arc_split(panel)
    if res is None:
        return 5.0
    split, n, c1, c2 = res
    r1, r2 = c1[2], c2[2]
    lo, hi = min(r1, r2), max(r1, r2)
    ratio = hi / lo if lo > 1e-6 else 1e9
    asym_defect = max(0.0, ratio - ratio_thresh) / ratio_scale

    blunt = p_0_blunt_tip_defect(panel)
    blunt_defect = max(0.0, blunt_thresh - blunt) / blunt_thresh

    chunky = _hull_area_over_diag2(panel)
    chunky_defect = max(0.0, chunky - chunky_thresh) / chunky_thresh

    return float(max(asym_defect, blunt_defect, chunky_defect))


def _rot_iou_about(panel, center, deg):
    """IoU of the filled shape with a copy of itself rotated by `deg` degrees
    about the given (cx, cy) pixel. Reusable for testing n-fold rotational
    (point) symmetry about a specific hub -- unlike the centroid-based
    variant, this measures symmetry about a chosen center such as the hub
    where several sub-shapes meet."""
    f = _filled_mask(panel)
    if f.sum() == 0:
        return 0.0
    cx, cy = float(center[0]), float(center[1])
    r = math.radians(deg)
    cos, sin = math.cos(r), math.sin(r)
    Y, X = np.mgrid[0:f.shape[0], 0:f.shape[1]]
    xr = cos * (X - cx) - sin * (Y - cy) + cx
    yr = sin * (X - cx) + cos * (Y - cy) + cy
    xi = np.round(xr).astype(int)
    yi = np.round(yr).astype(int)
    ok = (xi >= 0) & (xi < f.shape[1]) & (yi >= 0) & (yi < f.shape[0])
    rot = np.zeros_like(f)
    rot[Y[ok], X[ok]] = f[yi[ok], xi[ok]]
    inter = np.logical_and(f, rot).sum()
    union = np.logical_or(f, rot).sum()
    return float(inter / union) if union > 0 else 0.0


def p_0_rot3_windmill_defect(panel, target=0.56):
    """How far the shape is from being a clean 3-fold rotational windmill
    about the hub where its strokes are densest. A tidy pinwheel of three
    congruent blades meeting at one central point reproduces itself under a
    120-degree rotation about that hub with a characteristic partial overlap
    (IoU ~0.5): the rotated blades land on their neighbours' positions but
    only partly coincide. This returns |IoU_120 - target| (target ~0.56, just
    above that clean-windmill overlap band so the pos side sits near zero and
    both failure directions read large), which is small for such a clean
    windmill and large for anything else -- a single
    polygon with no 3-fold symmetry (IoU ~0), blades that are misarranged so
    the pattern breaks (IoU well below target), or blades so overlapping/
    coincident that the rotated copy nearly reproduces the original (IoU well
    above target)."""
    center = _densest_point(panel, r=3.5)
    iou = max(_rot_iou_about(panel, center, 120.0),
              _rot_iou_about(panel, center, -120.0))
    return abs(iou - target)
