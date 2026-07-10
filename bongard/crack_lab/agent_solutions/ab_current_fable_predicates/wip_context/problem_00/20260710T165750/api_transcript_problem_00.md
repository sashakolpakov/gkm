Looking at the panels: positives are all single smooth circular arcs (roughly quarter-circle). Negatives are near-misses: touching circles (closed loops/holes), an S-curve (inflection, bad circle fit), a line with a branch (extra endpoints), a shallow arc (small angular extent), an arc with a barb (branch/endpoints), two closed leaf shapes (holes).

I'll build generic measurements: connectivity, holes, endpoints/branchpoints, circle-fit quality, and arc angular extent.

```python
# Shared predicate library. p_<name>(panel) -> float | bool
import math

import numpy as np
from scipy import ndimage


def _ink_coords(panel):
    ys, xs = np.nonzero(panel)
    return xs.astype(float), ys.astype(float)


def p_ink_count(panel):
    """Total number of ink pixels."""
    return float(np.count_nonzero(panel))


def p_n_components(panel):
    """Number of 8-connected ink components."""
    lab, n = ndimage.label(panel > 0, structure=np.ones((3, 3)))
    return float(n)


def p_n_holes(panel):
    """Number of enclosed background regions (holes)."""
    bg = (np.pad(panel, 1) == 0)
    lab, n = ndimage.label(bg)  # 4-connectivity
    return float(n - 1)


def _neighbor_count(panel):
    ink = (panel > 0).astype(int)
    k = np.ones((3, 3))
    k[1, 1] = 0
    nb = ndimage.convolve(ink, k, mode="constant", cval=0)
    return nb, ink


def p_n_endpoints(panel):
    """Ink pixels with exactly one ink neighbor (stroke endpoints)."""
    nb, ink = _neighbor_count(panel)
    return float(np.sum((ink == 1) & (nb == 1)))


def p_n_branchpoints(panel):
    """Ink pixels with 3+ ink neighbors (junctions; noisy on thick strokes)."""
    nb, ink = _neighbor_count(panel)
    return float(np.sum((ink == 1) & (nb >= 3)))


def _circle_fit(panel):
    """Least-squares (Kasa) circle fit to all ink pixels.
    Returns (cx, cy, r, normalized_residual)."""
    xs, ys = _ink_coords(panel)
    if len(xs) < 3:
        return 64.0, 64.0, 1.0, 1.0
    A = np.column_stack([xs, ys, np.ones_like(xs)])
    b = xs ** 2 + ys ** 2
    try:
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return 64.0, 64.0, 1.0, 1.0
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    r = math.sqrt(max(sol[2] + cx * cx + cy * cy, 1e-9))
    d = np.hypot(xs - cx, ys - cy)
    resid = float(np.std(d) / max(r, 1e-9))
    return float(cx), float(cy), float(r), resid


def p_circle_fit_residual(panel):
    """Std of radial distance / radius for best-fit circle (0 = perfect arc)."""
    return _circle_fit(panel)[3]


def p_circle_fit_radius(panel):
    """Radius of least-squares circle fit to the ink."""
    return _circle_fit(panel)[2]


def p_arc_extent_deg(panel):
    """Angular extent (degrees) of ink around the best-fit circle center."""
    cx, cy, r, resid = _circle_fit(panel)
    xs, ys = _ink_coords(panel)
    if len(xs) == 0:
        return 0.0
    ang = np.sort(np.arctan2(ys - cy, xs - cx))
    if len(ang) == 1:
        return 0.0
    gaps = np.diff(ang)
    wrap = ang[0] + 2.0 * math.pi - ang[-1]
    maxgap = max(float(gaps.max()), float(wrap))
    return float(math.degrees(2.0 * math.pi - maxgap))


def p_bbox_diag(panel):
    """Diagonal length of the ink bounding box."""
    xs, ys = _ink_coords(panel)
    if len(xs) == 0:
        return 0.0
    return float(math.hypot(xs.max() - xs.min(), ys.max() - ys.min()))


def p_bbox_aspect(panel):
    """Bounding-box aspect ratio (short side / long side, in [0,1])."""
    xs, ys = _ink_coords(panel)
    if len(xs) == 0:
        return 0.0
    w = xs.max() - xs.min() + 1.0
    h = ys.max() - ys.min() + 1.0
    return float(min(w, h) / max(w, h))
```

Please run the verifier and report the RESULT line. My expectation: positives should show low `circle_fit_residual`, `arc_extent_deg` near 90, `n_holes == 0`, `n_endpoints == 2`, while each negative violates at least one of these; the harness should find a small separating combination.

RESULT solved=False heldout=0.750 train=0.917 rule="p_circle_fit_radius>=39.15" rule_cost=1.5 predicate_errors=0

The radius rule is a near-miss artifact. The real rule looks like "a single ~quarter-circle arc" — negatives have closed loops, inflections (S-curves), branches, or too-shallow/straight curves. I'll add path-tracing measurements: total turning angle along the stroke (net vs absolute), which should be ≈90° with consistent sign for positives.

```python
# Shared predicate library. p_<name>(panel) -> float | bool
import math
from collections import deque

import numpy as np
from scipy import ndimage


def _ink_coords(panel):
    ys, xs = np.nonzero(panel)
    return xs.astype(float), ys.astype(float)


def p_ink_count(panel):
    """Total number of ink pixels."""
    return float(np.count_nonzero(panel))


def p_n_components(panel):
    """Number of 8-connected ink components."""
    lab, n = ndimage.label(panel > 0, structure=np.ones((3, 3)))
    return float(n)


def p_n_holes(panel):
    """Number of enclosed background regions (holes)."""
    bg = (np.pad(panel, 1) == 0)
    lab, n = ndimage.label(bg)  # 4-connectivity
    return float(n - 1)


def _neighbor_count(panel):
    ink = (panel > 0).astype(int)
    k = np.ones((3, 3))
    k[1, 1] = 0
    nb = ndimage.convolve(ink, k, mode="constant", cval=0)
    return nb, ink


def p_n_endpoints(panel):
    """Ink pixels with exactly one ink neighbor (stroke endpoints)."""
    nb, ink = _neighbor_count(panel)
    return float(np.sum((ink == 1) & (nb == 1)))


def p_n_branchpoints(panel):
    """Ink pixels with 3+ ink neighbors (junctions; noisy on thick strokes)."""
    nb, ink = _neighbor_count(panel)
    return float(np.sum((ink == 1) & (nb >= 3)))


def _circle_fit(panel):
    """Least-squares (Kasa) circle fit to all ink pixels.
    Returns (cx, cy, r, normalized_residual)."""
    xs, ys = _ink_coords(panel)
    if len(xs) < 3:
        return 64.0, 64.0, 1.0, 1.0
    A = np.column_stack([xs, ys, np.ones_like(xs)])
    b = xs ** 2 + ys ** 2
    try:
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return 64.0, 64.0, 1.0, 1.0
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    r = math.sqrt(max(sol[2] + cx * cx + cy * cy, 1e-9))
    d = np.hypot(xs - cx, ys - cy)
    resid = float(np.std(d) / max(r, 1e-9))
    return float(cx), float(cy), float(r), resid


def p_circle_fit_residual(panel):
    """Std of radial distance / radius for best-fit circle (0 = perfect arc)."""
    return _circle_fit(panel)[3]


def p_circle_fit_radius(panel):
    """Radius of least-squares circle fit to the ink."""
    return _circle_fit(panel)[2]


def p_arc_extent_deg(panel):
    """Angular extent (degrees) of ink around the best-fit circle center."""
    cx, cy, r, resid = _circle_fit(panel)
    xs, ys = _ink_coords(panel)
    if len(xs) == 0:
        return 0.0
    ang = np.sort(np.arctan2(ys - cy, xs - cx))
    if len(ang) == 1:
        return 0.0
    gaps = np.diff(ang)
    wrap = ang[0] + 2.0 * math.pi - ang[-1]
    maxgap = max(float(gaps.max()), float(wrap))
    return float(math.degrees(2.0 * math.pi - maxgap))


def p_bbox_diag(panel):
    """Diagonal length of the ink bounding box."""
    xs, ys = _ink_coords(panel)
    if len(xs) == 0:
        return 0.0
    return float(math.hypot(xs.max() - xs.min(), ys.max() - ys.min()))


def p_bbox_aspect(panel):
    """Bounding-box aspect ratio (short side / long side, in [0,1])."""
    xs, ys = _ink_coords(panel)
    if len(xs) == 0:
        return 0.0
    w = xs.max() - xs.min() + 1.0
    h = ys.max() - ys.min() + 1.0
    return float(min(w, h) / max(w, h))


# ---------- stroke path tracing ----------

def _bfs_dist(pts, start):
    """Geodesic (8-connected) pixel distance from start over set of ink pixels."""
    dist = {start: 0}
    q = deque([start])
    while q:
        y, x = q.popleft()
        d = dist[(y, x)]
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                p = (y + dy, x + dx)
                if p in pts and p not in dist:
                    dist[p] = d + 1
                    q.append(p)
    return dist


def _path_samples(panel, step=6):
    """Order ink pixels by geodesic distance from a stroke endpoint (or the
    geodesically farthest pixel if no endpoint exists), bin by distance and
    average -> smoothed ordered sample points along the stroke."""
    ink = np.argwhere(panel > 0)
    if len(ink) < 3:
        return []
    pts = set(map(tuple, ink))
    nb, inkmask = _neighbor_count(panel)
    ends = np.argwhere((inkmask == 1) & (nb == 1))
    if len(ends) > 0:
        start = tuple(ends[0])
    else:
        d0 = _bfs_dist(pts, tuple(ink[0]))
        start = max(d0.items(), key=lambda kv: (kv[1], kv[0]))[0]
    dist = _bfs_dist(pts, start)
    maxd = max(dist.values())
    nbins = maxd // step + 1
    sums = np.zeros((nbins, 2))
    cnts = np.zeros(nbins)
    for (y, x), d in dist.items():
        b = d // step
        sums[b, 0] += x
        sums[b, 1] += y
        cnts[b] += 1
    good = cnts > 0
    return list(sums[good] / cnts[good][:, None])


def _turning(panel, step=6):
    """(net_turning_deg, abs_turning_deg) of tangent direction along stroke."""
    pts = _path_samples(panel, step=step)
    if len(pts) < 3:
        return 0.0, 0.0
    angs = []
    for a, b in zip(pts[:-1], pts[1:]):
        angs.append(math.atan2(b[1] - a[1], b[0] - a[0]))
    net = 0.0
    tot = 0.0
    for a1, a2 in zip(angs[:-1], angs[1:]):
        d = (a2 - a1 + math.pi) % (2.0 * math.pi) - math.pi
        net += d
        tot += abs(d)
    return abs(math.degrees(net)), math.degrees(tot)


def p_net_turning_deg(panel):
    """|integrated signed turning| along the stroke (deg). ~90 for quarter arc,
    ~0 for a line or an S-curve, ~360 for a closed loop."""
    return _turning(panel)[0]


def p_abs_turning_deg(panel):
    """Integrated |turning| along the stroke (deg); large for wiggly curves."""
    return _turning(panel)[1]


def p_turning_consistency(panel):
    """net/abs turning in [0,1]: 1 = curvature never changes sign."""
    net, tot = _turning(panel)
    if tot < 1e-6:
        return 1.0
    return float(net / tot)


def p_endpoint_geodesic_span(panel):
    """Geodesic stroke length (max BFS distance from traced start), pixels."""
    ink = np.argwhere(panel > 0)
    if len(ink) < 2:
        return 0.0
    pts = set(map(tuple, ink))
    nb, inkmask = _neighbor_count(panel)
    ends = np.argwhere((inkmask == 1) & (nb == 1))
    if len(ends) > 0:
        start = tuple(ends[0])
    else:
        d0 = _bfs_dist(pts, tuple(ink[0]))
        start = max(d0.items(), key=lambda kv: (kv[1], kv[0]))[0]
    dist = _bfs_dist(pts, start)
    return float(max(dist.values()))


def p_chord_over_pathlen(panel):
    """Straight-line span of traced path / geodesic length (1 = straight)."""
    pts = _path_samples(panel, step=3)
    if len(pts) < 2:
        return 1.0
    chord = math.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1])
    plen = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        plen += math.hypot(b[0] - a[0], b[1] - a[1])
    if plen < 1e-6:
        return 1.0
    return float(chord / plen)
```

Positives should show `net_turning_deg` ≈ 90 with `turning_consistency` ≈ 1, while negatives are ~0 (line, S-curve), ~360 (loops), small (shallow arc), or inconsistent (branched/barbed strokes). Please run the verifier and report the RESULT line.

RESULT solved=False heldout=0.625 train=0.917 rule="p_circle_fit_radius>=39.15" rule_cost=1.5 predicate_errors=0

The tracing was likely fragile (bin-averaging, multi-component panels). I've rewritten the path tracer to extract an explicit shortest-path pixel chain on the largest component, resample it, and measure turning on that — much more robust. Also added curvature-sign-change counting.

```python
# Shared predicate library. p_<name>(panel) -> float | bool
import math
from collections import deque

import numpy as np
from scipy import ndimage


def _ink_coords(panel):
    ys, xs = np.nonzero(panel)
    return xs.astype(float), ys.astype(float)


def p_ink_count(panel):
    """Total number of ink pixels."""
    return float(np.count_nonzero(panel))


def p_n_components(panel):
    """Number of 8-connected ink components."""
    lab, n = ndimage.label(panel > 0, structure=np.ones((3, 3)))
    return float(n)


def p_n_holes(panel):
    """Number of enclosed background regions (holes)."""
    bg = (np.pad(panel, 1) == 0)
    lab, n = ndimage.label(bg)  # 4-connectivity
    return float(n - 1)


def _neighbor_count(panel):
    ink = (panel > 0).astype(int)
    k = np.ones((3, 3))
    k[1, 1] = 0
    nb = ndimage.convolve(ink, k, mode="constant", cval=0)
    return nb, ink


def p_n_endpoints(panel):
    """Ink pixels with exactly one ink neighbor (stroke endpoints)."""
    nb, ink = _neighbor_count(panel)
    return float(np.sum((ink == 1) & (nb == 1)))


def p_n_branchpoints(panel):
    """Ink pixels with 3+ ink neighbors (junctions; noisy on thick strokes)."""
    nb, ink = _neighbor_count(panel)
    return float(np.sum((ink == 1) & (nb >= 3)))


def _circle_fit(panel):
    """Least-squares (Kasa) circle fit to all ink pixels.
    Returns (cx, cy, r, normalized_residual)."""
    xs, ys = _ink_coords(panel)
    if len(xs) < 3:
        return 64.0, 64.0, 1.0, 1.0
    A = np.column_stack([xs, ys, np.ones_like(xs)])
    b = xs ** 2 + ys ** 2
    try:
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return 64.0, 64.0, 1.0, 1.0
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    r = math.sqrt(max(sol[2] + cx * cx + cy * cy, 1e-9))
    d = np.hypot(xs - cx, ys - cy)
    resid = float(np.std(d) / max(r, 1e-9))
    return float(cx), float(cy), float(r), resid


def p_circle_fit_residual(panel):
    """Std of radial distance / radius for best-fit circle (0 = perfect arc)."""
    return _circle_fit(panel)[3]


def p_circle_fit_radius(panel):
    """Radius of least-squares circle fit to the ink."""
    return _circle_fit(panel)[2]


def p_arc_extent_deg(panel):
    """Angular extent (degrees) of ink around the best-fit circle center."""
    cx, cy, r, resid = _circle_fit(panel)
    xs, ys = _ink_coords(panel)
    if len(xs) == 0:
        return 0.0
    ang = np.sort(np.arctan2(ys - cy, xs - cx))
    if len(ang) == 1:
        return 0.0
    gaps = np.diff(ang)
    wrap = ang[0] + 2.0 * math.pi - ang[-1]
    maxgap = max(float(gaps.max()), float(wrap))
    return float(math.degrees(2.0 * math.pi - maxgap))


def p_bbox_diag(panel):
    """Diagonal length of the ink bounding box."""
    xs, ys = _ink_coords(panel)
    if len(xs) == 0:
        return 0.0
    return float(math.hypot(xs.max() - xs.min(), ys.max() - ys.min()))


def p_bbox_aspect(panel):
    """Bounding-box aspect ratio (short side / long side, in [0,1])."""
    xs, ys = _ink_coords(panel)
    if len(xs) == 0:
        return 0.0
    w = xs.max() - xs.min() + 1.0
    h = ys.max() - ys.min() + 1.0
    return float(min(w, h) / max(w, h))


# ---------- stroke path tracing ----------

_N8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _largest_component_pts(panel):
    lab, n = ndimage.label(panel > 0, structure=np.ones((3, 3)))
    if n == 0:
        return set()
    sizes = ndimage.sum(panel > 0, lab, range(1, n + 1))
    best = int(np.argmax(sizes)) + 1
    return set(map(tuple, np.argwhere(lab == best)))


def _bfs(pts, start):
    """BFS over pixel set. Returns (dist dict, parent dict)."""
    dist = {start: 0}
    par = {start: None}
    q = deque([start])
    while q:
        p = q.popleft()
        d = dist[p]
        for dy, dx in _N8:
            np_ = (p[0] + dy, p[1] + dx)
            if np_ in pts and np_ not in dist:
                dist[np_] = d + 1
                par[np_] = p
                q.append(np_)
    return dist, par


def _trace_chain(panel):
    """Longest geodesic pixel chain through the largest ink component:
    from an endpoint (if any) to the geodesically farthest pixel."""
    pts = _largest_component_pts(panel)
    if len(pts) < 3:
        return []
    # endpoints within this component
    ends = [p for p in pts
            if sum(((p[0] + dy, p[1] + dx) in pts) for dy, dx in _N8) == 1]
    seed = ends[0] if ends else next(iter(sorted(pts)))
    d0, _ = _bfs(pts, seed)
    start = max(d0.items(), key=lambda kv: (kv[1], kv[0]))[0]
    d1, par = _bfs(pts, start)
    end = max(d1.items(), key=lambda kv: (kv[1], kv[0]))[0]
    chain = []
    p = end
    while p is not None:
        chain.append(p)
        p = par[p]
    chain.reverse()
    return chain  # list of (y, x)


def _polyline(panel, step=8):
    """Resample traced chain every `step` pixels -> [(x, y), ...]."""
    chain = _trace_chain(panel)
    if len(chain) < 3:
        return []
    idx = list(range(0, len(chain), step))
    if idx[-1] != len(chain) - 1:
        idx.append(len(chain) - 1)
    return [(chain[i][1], chain[i][0]) for i in idx]


def _turn_angles(panel, step=8):
    pts = _polyline(panel, step=step)
    if len(pts) < 3:
        return []
    angs = [math.atan2(b[1] - a[1], b[0] - a[0])
            for a, b in zip(pts[:-1], pts[1:])]
    turns = []
    for a1, a2 in zip(angs[:-1], angs[1:]):
        turns.append((a2 - a1 + math.pi) % (2.0 * math.pi) - math.pi)
    return turns


def p_net_turning_deg(panel):
    """|integrated signed turning| along the main stroke chain (deg).
    ~90 for a quarter arc, ~0 for a line or S-curve, ~180 for half loop."""
    t = _turn_angles(panel)
    return abs(math.degrees(sum(t))) if t else 0.0


def p_abs_turning_deg(panel):
    """Integrated |turning| along the main stroke chain (deg)."""
    t = _turn_angles(panel)
    return math.degrees(sum(abs(x) for x in t)) if t else 0.0


def p_turning_consistency(panel):
    """net/abs turning in [0,1]: 1 = curvature never changes sign."""
    t = _turn_angles(panel)
    tot = sum(abs(x) for x in t)
    if tot < 1e-6:
        return 1.0
    return float(abs(sum(t)) / tot)


def p_curvature_sign_changes(panel):
    """Number of significant curvature sign flips along the stroke."""
    t = _turn_angles(panel)
    thr = math.radians(8.0)
    sig

```python
# Shared predicate library. p_<name>(panel) -> float | bool
import math
from collections import deque

import numpy as np
from scipy import ndimage


def _ink_coords(panel):
    ys, xs = np.nonzero(panel)
    return xs.astype(float), ys.astype(float)


def p_ink_count(panel):
    """Total number of ink pixels."""
    return float(np.count_nonzero(panel))


def p_n_components(panel):
    """Number of 8-connected ink components."""
    lab, n = ndimage.label(panel > 0, structure=np.ones((3, 3)))
    return float(n)


def p_n_holes(panel):
    """Number of enclosed background regions (holes)."""
    bg = (np.pad(panel, 1) == 0)
    lab, n = ndimage.label(bg)  # 4-connectivity
    return float(n - 1)


def _neighbor_count(panel):
    ink = (panel > 0).astype(int)
    k = np.ones((3, 3))
    k[1, 1] = 0
    nb = ndimage.convolve(ink, k, mode="constant", cval=0)
    return nb, ink


def p_n_endpoints(panel):
    """Ink pixels with exactly one ink neighbor (stroke endpoints)."""
    nb, ink = _neighbor_count(panel)
    return float(np.sum((ink == 1) & (nb == 1)))


def p_n_branchpoints(panel):
    """Ink pixels with 3+ ink neighbors (junctions; noisy on thick strokes)."""
    nb, ink = _neighbor_count(panel)
    return float(np.sum((ink == 1) & (nb >= 3)))


def _circle_fit(panel):
    """Least-squares (Kasa) circle fit to all ink pixels.
    Returns (cx, cy, r, normalized_residual)."""
    xs, ys = _ink_coords(panel)
    if len(xs) < 3:
        return 64.0, 64.0, 1.0, 1.0
    A = np.column_stack([xs, ys, np.ones_like(xs)])
    b = xs ** 2 + ys ** 2
    try:
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return 64.0, 64.0, 1.0, 1.0
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    r = math.sqrt(max(sol[2] + cx * cx + cy * cy, 1e-9))
    d = np.hypot(xs - cx, ys - cy)
    resid = float(np.std(d) / max(r, 1e-9))
    return float(cx), float(cy), float(r), resid


def p_circle_fit_residual(panel):
    """Std of radial distance / radius for best-fit circle (0 = perfect arc)."""
    return _circle_fit(panel)[3]


def p_circle_fit_radius(panel):
    """Radius of least-squares circle fit to the ink."""
    return _circle_fit(panel)[2]


def p_arc_extent_deg(panel):
    """Angular extent (degrees) of ink around the best-fit circle center."""
    cx, cy, r, resid = _circle_fit(panel)
    xs, ys = _ink_coords(panel)
    if len(xs) == 0:
        return 0.0
    ang = np.sort(np.arctan2(ys - cy, xs - cx))
    if len(ang) == 1:
        return 0.0
    gaps = np.diff(ang)
    wrap = ang[0] + 2.0 * math.pi - ang[-1]
    maxgap = max(float(gaps.max()), float(wrap))
    return float(math.degrees(2.0 * math.pi - maxgap))


def p_bbox_diag(panel):
    """Diagonal length of the ink bounding box."""
    xs, ys = _ink_coords(panel)
    if len(xs) == 0:
        return 0.0
    return float(math.hypot(xs.max() - xs.min(), ys.max() - ys.min()))


def p_bbox_aspect(panel):
    """Bounding-box aspect ratio (short side / long side, in [0,1])."""
    xs, ys = _ink_coords(panel)
    if len(xs) == 0:
        return 0.0
    w = xs.max() - xs.min() + 1.0
    h = ys.max() - ys.min() + 1.0
    return float(min(w, h) / max(w, h))


# ---------- stroke path tracing ----------

_N8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _largest_component_pts(panel):
    lab, n = ndimage.label(panel > 0, structure=np.ones((3, 3)))
    if n == 0:
        return set()
    sizes = ndimage.sum(panel > 0, lab, range(1, n + 1))
    best = int(np.argmax(sizes)) + 1
    return set(map(tuple, np.argwhere(lab == best)))


def _bfs(pts, start):
    """BFS over pixel set. Returns (dist dict, parent dict)."""
    dist = {start: 0}
    par = {start: None}
    q = deque([start])
    while q:
        p = q.popleft()
        d = dist[p]
        for dy, dx in _N8:
            q2 = (p[0] + dy, p[1] + dx)
            if q2 in pts and q2 not in dist:
                dist[q2] = d + 1
                par[q2] = p
                q.append(q2)
    return dist, par


def _trace_chain(panel):
    """Longest geodesic pixel chain through the largest ink component
    (double BFS: farthest point from a seed, then farthest from that)."""
    pts = _largest_component_pts(panel)
    if len(pts) < 3:
        return []
    seed = next(iter(sorted(pts)))
    d0, _ = _bfs(pts, seed)
    start = max(d0.items(), key=lambda kv: (kv[1], kv[0]))[0]
    d1, par = _bfs(pts, start)
    end = max(d1.items(), key=lambda kv: (kv[1], kv[0]))[0]
    chain = []
    p = end
    while p is not None:
        chain.append(p)
        p = par[p]
    chain.reverse()
    return chain  # list of (y, x)


def _polyline(panel, step=8):
    """Resample traced chain every `step` pixels -> [(x, y), ...]."""
    chain = _trace_chain(panel)
    if len(chain) < 3:
        return []
    idx = list(range(0, len(chain), step))
    if idx[-1] != len(chain) - 1:
        idx.append(len(chain) - 1)
    return [(float(chain[i][1]), float(chain[i][0])) for i in idx]


def _turn_angles(panel, step=8):
    pts = _polyline(panel, step=step)
    if len(pts) < 3:
        return []
    angs = [math.atan2(b[1] - a[1], b[0] - a[0])
            for a, b in zip(pts[:-1], pts[1:])]
    turns = []
    for a1, a2 in zip(angs[:-1], angs[1:]):
        turns.append((a2 - a1 + math.pi) % (2.0 * math.pi) - math.pi)
    return turns


def p_net_turning_deg(panel):
    """|integrated signed turning| along the main stroke chain (deg).
    ~90 for a quarter arc, ~0 for a line or S-curve."""
    t = _turn_angles(panel)
    return abs(math.degrees(sum(t))) if t else 0.0


def p_abs_turning_deg(panel):
    """Integrated |turning| along the main stroke chain (deg)."""
    t = _turn_angles(panel)
    return math.degrees(sum(abs(x) for x in t)) if t else 0.0


def p_turning_consistency(panel):
    """net/abs turning in [0,1]: 1 = curvature never changes sign."""
    t = _turn_angles(panel)
    tot = sum(abs(x) for x in t)
    if tot < 1e-6:
        return 1.0
    return float(abs(sum(t)) / tot)


def p_curvature_sign_changes(panel):
    """Number of significant curvature sign flips along the stroke."""
    t = _turn_angles(panel)
    thr = math.radians(8.0)
    sig = [x for x in t if abs(x) > thr]
    flips = 0
    for a, b in zip(sig[:-1], sig[1:]):
        if a * b < 0:
            flips += 1
    return float(flips)


def p_path_length(panel):
    """Length in pixels of the longest geodesic chain of the main component."""
    return float(len(_trace_chain(panel)))


def p_chord_over_pathlen(panel):
    """Endpoint chord / chain length of main stroke (1 = straight line)."""
    chain = _trace_chain(panel)
    if len(chain) < 2:
        return 1.0
    chord = math.hypot(chain[-1][0] - chain[0][0], chain[-1][1] - chain[0][1])
    return float(chord / max(len(chain) - 1, 1))
```

Please run the verifier and report the RESULT line.

RESULT solved=False heldout=0.736 train=0.917 rule="p_circle_fit_radius>=39.15" rule_cost=1.5 predicate_errors=0

The harness keeps falling back to the radius rule, so my turning measures must not be separating cleanly — likely noise from coarse resampling, and no predicate expressing "close to a quarter turn". I've added chain smoothing, a `|net turn − 90°|` deviation measure, a sharpest-corner detector, and turning-rate uniformity.

```python
# Shared predicate library. p_<name>(panel) -> float | bool
import math
from collections import deque

import numpy as np
from scipy import ndimage


def _ink_coords(panel):
    ys, xs = np.nonzero(panel)
    return xs.astype(float), ys.astype(float)


def p_ink_count(panel):
    """Total number of ink pixels."""
    return float(np.count_nonzero(panel))


def p_n_components(panel):
    """Number of 8-connected ink components."""
    lab, n = ndimage.label(panel > 0, structure=np.ones((3, 3)))
    return float(n)


def p_n_holes(panel):
    """Number of enclosed background regions (holes)."""
    bg = (np.pad(panel, 1) == 0)
    lab, n = ndimage.label(bg)  # 4-connectivity
    return float(n - 1)


def _neighbor_count(panel):
    ink = (panel > 0).astype(int)
    k = np.ones((3, 3))
    k[1, 1] = 0
    nb = ndimage.convolve(ink, k, mode="constant", cval=0)
    return nb, ink


def p_n_endpoints(panel):
    """Ink pixels with exactly one ink neighbor (stroke endpoints)."""
    nb, ink = _neighbor_count(panel)
    return float(np.sum((ink == 1) & (nb == 1)))


def p_n_branchpoints(panel):
    """Ink pixels with 3+ ink neighbors (junctions; noisy on thick strokes)."""
    nb, ink = _neighbor_count(panel)
    return float(np.sum((ink == 1) & (nb >= 3)))


def _circle_fit(panel):
    """Least-squares (Kasa) circle fit to all ink pixels.
    Returns (cx, cy, r, normalized_residual)."""
    xs, ys = _ink_coords(panel)
    if len(xs) < 3:
        return 64.0, 64.0, 1.0, 1.0
    A = np.column_stack([xs, ys, np.ones_like(xs)])
    b = xs ** 2 + ys ** 2
    try:
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return 64.0, 64.0, 1.0, 1.0
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    r = math.sqrt(max(sol[2] + cx * cx + cy * cy, 1e-9))
    d = np.hypot(xs - cx, ys - cy)
    resid = float(np.std(d) / max(r, 1e-9))
    return float(cx), float(cy), float(r), resid


def p_circle_fit_residual(panel):
    """Std of radial distance / radius for best-fit circle (0 = perfect arc)."""
    return _circle_fit(panel)[3]


def p_circle_fit_radius(panel):
    """Radius of least-squares circle fit to the ink."""
    return _circle_fit(panel)[2]


def p_arc_extent_deg(panel):
    """Angular extent (degrees) of ink around the best-fit circle center."""
    cx, cy, r, resid = _circle_fit(panel)
    xs, ys = _ink_coords(panel)
    if len(xs) == 0:
        return 0.0
    ang = np.sort(np.arctan2(ys - cy, xs - cx))
    if len(ang) == 1:
        return 0.0
    gaps = np.diff(ang)
    wrap = ang[0] + 2.0 * math.pi - ang[-1]
    maxgap = max(float(gaps.max()), float(wrap))
    return float(math.degrees(2.0 * math.pi - maxgap))


def p_bbox_diag(panel):
    """Diagonal length of the ink bounding box."""
    xs, ys = _ink_coords(panel)
    if len(xs) == 0:
        return 0.0
    return float(math.hypot(xs.max() - xs.min(), ys.max() - ys.min()))


def p_bbox_aspect(panel):
    """Bounding-box aspect ratio (short side / long side, in [0,1])."""
    xs, ys = _ink_coords(panel)
    if len(xs) == 0:
        return 0.0
    w = xs.max() - xs.min() + 1.0
    h = ys.max() - ys.min() + 1.0
    return float(min(w, h) / max(w, h))


# ---------- stroke path tracing ----------

_N8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _largest_component_pts(panel):
    lab, n = ndimage.label(panel > 0, structure=np.ones((3, 3)))
    if n == 0:
        return set()
    sizes = ndimage.sum(panel > 0, lab, range(1, n + 1))
    best = int(np.argmax(sizes)) + 1
    return set(map(tuple, np.argwhere(lab == best)))


def _bfs(pts, start):
    """BFS over pixel set. Returns (dist dict, parent dict)."""
    dist = {start: 0}
    par = {start: None}
    q = deque([start])
    while q:
        p = q.popleft()
        d = dist[p]
        for dy, dx in _N8:
            q2 = (p[0] + dy, p[1] + dx)
            if q2 in pts and q2 not in dist:
                dist[q2] = d + 1
                par[q2] = p
                q.append(q2)
    return dist, par


def _trace_chain(panel):
    """Longest geodesic pixel chain through the largest ink component
    (double BFS: farthest point from a seed, then farthest from that)."""
    pts = _largest_component_pts(panel)
    if len(pts) < 3:
        return []
    seed = next(iter(sorted(pts)))
    d0, _ = _bfs(pts, seed)
    start = max(d0.items(), key=lambda kv: (kv[1], kv[0]))[0]
    d1, par = _bfs(pts, start)
    end = max(d1.items(), key=lambda kv: (kv[1], kv[0]))[0]
    chain = []
    p = end
    while p is not None:
        chain.append(p)
        p = par[p]
    chain.reverse()
    return chain  # list of (y, x)


def _polyline(panel, step=8):
    """Smooth traced chain (moving average) and resample every `step` px
    -> [(x, y), ...]."""
    chain = _trace_chain(panel)
    if len(chain) < 3:
        return []
    arr = np.array([(c[1], c[0]) for c in chain], dtype=float)
    w = 5
    if len(arr) > w:
        kern = np.ones(w) / w
        sm = np.empty_like(arr)
        sm[:, 0] = np.convolve(arr[:, 0], kern, mode="same")
        sm[:, 1] = np.convolve(arr[:, 1], kern, mode="same")
        h = w // 2
        sm[:h] = arr[:h]
        sm[-h:] = arr[-h:]
        arr = sm
    idx = list(range(0, len(arr), step))
    if idx[-1] != len(arr) - 1:
        idx.append(len(arr) - 1)
    return [tuple(arr[i]) for i in idx]


def _turn_angles(panel, step=8):
    pts = _polyline(panel, step=step)
    if len(pts) < 3:
        return []
    angs = [math.atan2(b[1] - a[1], b[0] - a[0])
            for a, b in zip(pts[:-1], pts[1:])]
    turns = []
    for a1, a2 in zip(angs[:-1], angs[1:]):
        turns.append((a2 - a1 + math.pi) % (2.0 * math.pi) - math.pi)
    return turns


def p_net_turning_deg(panel):
    """|integrated signed turning| along the main stroke chain (deg).
    ~90 for a quarter arc, ~0 for a line or S-curve."""
    t = _turn_angles(panel)
    return abs(math.degrees(sum(t))) if t else 0.0


def p_abs_turning_deg(panel):
    """Integrated |turning| along the main stroke chain (deg)."""
    t = _turn_angles(panel)
    return math.degrees(sum(abs(x) for x in t)) if t else 0.0


def p_turning_consistency(panel):
    """net/abs turning in [0,1]: 1 = curvature never changes sign."""
    t = _turn_angles(panel)
    tot = sum(abs(x) for x in t)
    if tot < 1e-6:
        return 1.0
    return float(abs(sum(t)) / tot)


def p_curvature_sign_changes(panel):
    """Number of significant curvature sign flips along the stroke."""
    t = _turn_angles(panel)
    thr = math.radians(8.0)
    sig = [x for x in t if abs(x) > thr]
    flips = 0
    for a, b in zip(sig[:-1], sig[1:]):
        if a * b < 0:
            flips += 1
    return float(flips)


def p_path_length(panel):
    """Length in pixels of the longest geodesic chain of the main component."""
    return float(len(_trace_chain(panel)))


def p_chord_over_pathlen(panel):
    """Endpoint chord / chain length of main stroke (1 = straight line)."""
    chain = _trace_chain(panel)
    if len(chain) < 2:
        return 1.0
    chord = math.hypot(chain[-1][0] - chain[0][0], chain[-1][1] - chain[0][1])
    return float(chord / max(len(chain) - 1, 1))


def p_turn_dev_from_quarter(panel):
    """|net turning - 90 deg|: 0 means the stroke makes a quarter turn."""
    return abs(p_net_turning_deg(panel) - 90.0)


def p_max_turn_deg(panel):
    """Sharpest single turning angle along the stroke (corner detector)."""
    t = _turn_angles(panel)
    return math.degrees(max(abs(x) for x in t)) if t else 0.0


def p_turning_rate_std(panel):
    """Std of signed turning per step (deg); low = uniform-curvature arc."""
    t = _turn_angles(panel)
    if len(t) < 2:
        return 0.0
    return float(np.std(np.degrees(t)))
```

Please run the verifier and report the RESULT line.

RESULT solved=False heldout=0.736 train=0.917 rule="p_circle_fit_radius>=39.15" rule_cost=1.5 predicate_errors=0

Let me actually inspect the values instead of guessing.

I need to run diagnostics. Let me compute the predicate values for all panels first — writing a quick debug script and examining output before finalizing.

Actually, I should use my tools here. Let me do that:

Since I can only reply with the file, let me reason more carefully about what distinguishes the sides, given the harness keeps picking the radius rule (which fails on ~1 training panel and generalizes badly).

Re-examining the panels: all positives are single arcs of a *large* circle — roughly quarter circles with radius ≈ 40–60px. Negatives: two touching small circles (small radius), a shallow S-curve (huge fit radius — this likely breaks the radius rule), a nearly-straight line (huge radius), a shallow arc (large radius), a hook with a sharp barb, two closed leaves. So radius alone can't work; the natural rule is "the ink is a single arc that fits a circle well AND spans a substantial angle". `p_circle_fit_residual` low AND `p_arc_extent_deg` near 90 should do it — but negatives like the shallow arc (neg_3) also fit a circle well, just with small extent... its extent around the fit center would be small only if the radius is huge. Key risk: `p_arc_extent_deg` uses angular gap which is fine for thin arcs.

A likely bug: for closed shapes (neg_0, neg_5) extent ≈ 360, S-curve fits badly, straight line has huge radius so tiny extent, shallow arc has moderate extent (< 60°?), barbed arc — the barb inflates residual. So a compound rule `residual < t1 AND extent in [70,120]` should separate. The harness only found the radius rule, suggesting my extent/residual values aren't behaving as expected — possibly stroke thickness noise. Let me add a robust combined predicate: fraction of ink well-explained by the fit circle, and extent computed via histogram occupancy (robust to outlier pixels like the barb): fit circle on the *chain* (not all ink), then measure coverage.

```python
# Shared predicate library. p_<name>(panel) -> float | bool
import math
from collections import deque

import numpy as np
from scipy import ndimage


def _ink_coords(panel):
    ys, xs = np.nonzero(panel)
    return xs.astype(float), ys.astype(float)


def p_ink_count(panel):
    """Total number of ink pixels."""
    return float(np.count_nonzero(panel))


def p_n_components(panel):
    """Number of 8-connected ink components."""
    lab, n = ndimage.label(panel > 0, structure=np.ones((3, 3)))
    return float(n)


def p_n_holes(panel):
    """Number of enclosed background regions (holes)."""
    bg = (np.pad(panel, 1) == 0)
    lab, n = ndimage.label(bg)  # 4-connectivity
    return float(n - 1)


def _neighbor_count(panel):
    ink = (panel > 0).astype(int)
    k = np.ones((3, 3))
    k[1, 1] = 0
    nb = ndimage.convolve(ink, k, mode="constant", cval=0)
    return nb, ink


def p_n_endpoints(panel):
    """Ink pixels with exactly one ink neighbor (stroke endpoints)."""
    nb, ink = _neighbor_count(panel)
    return float(np.sum((ink == 1) & (nb == 1)))


def p_n_branchpoints(panel):
    """Ink pixels with 3+ ink neighbors (junctions; noisy on thick strokes)."""
    nb, ink = _neighbor_count(panel)
    return float(np.sum((ink == 1) & (nb >= 3)))


def _fit_circle_pts(xs, ys):
    """Kasa circle fit to point arrays. Returns (cx, cy, r, norm_resid)."""
    if len(xs) < 3:
        return 64.0, 64.0, 1.0, 1.0
    A = np.column_stack([xs, ys, np.ones_like(xs)])
    b = xs ** 2 + ys ** 2
    try:
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return 64.0, 64.0, 1.0, 1.0
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    r = math.sqrt(max(sol[2] + cx * cx + cy * cy, 1e-9))
    d = np.hypot(xs - cx, ys - cy)
    resid = float(np.std(d) / max(r, 1e-9))
    return float(cx), float(cy), float(r), resid


def _circle_fit(panel):
    xs, ys = _ink_coords(panel)
    return _fit_circle_pts(xs, ys)


def p_circle_fit_residual(panel):
    """Std of radial distance / radius for best-fit circle (0 = perfect arc)."""
    return _circle_fit(panel)[3]


def p_circle_fit_radius(panel):
    """Radius of least-squares circle fit to the ink."""
    return _circle_fit(panel)[2]


def p_arc_extent_deg(panel):
    """Angular extent (degrees) of ink around the best-fit circle center."""
    cx, cy, r, resid = _circle_fit(panel)
    xs, ys = _ink_coords(panel)
    if len(xs) == 0:
        return 0.0
    ang = np.sort(np.arctan2(ys - cy, xs - cx))
    if len(ang) == 1:
        return 0.0
    gaps = np.diff(ang)
    wrap = ang[0] + 2.0 * math.pi - ang[-1]
    maxgap = max(float(gaps.max()), float(wrap))
    return float(math.degrees(2.0 * math.pi - maxgap))


def p_bbox_diag(panel):
    """Diagonal length of the ink bounding box."""
    xs, ys = _ink_coords(panel)
    if len(xs) == 0:
        return 0.0
    return float(math.hypot(xs.max() - xs.min(), ys.max() - ys.min()))


def p_bbox_aspect(panel):
    """Bounding-box aspect ratio (short side / long side, in [0,1])."""
    xs, ys = _ink_coords(panel)
    if len(xs) == 0:
        return 0.0
    w = xs.max() - xs.min() + 1.0
    h = ys.max() - ys.min() + 1.0
    return float(min(w, h) / max(w, h))


# ---------- stroke path tracing ----------

_N8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _largest_component_pts(panel):
    lab, n = ndimage.label(panel > 0, structure=np.ones((3, 3)))
    if n == 0:
        return set()
    sizes = ndimage.sum(panel > 0, lab, range(1, n + 1))
    best = int(np.argmax(sizes)) + 1
    return set(map(tuple, np.argwhere(lab == best)))


def _bfs(pts, start):
    """BFS over pixel set. Returns (dist dict, parent dict)."""
    dist = {start: 0}
    par = {start: None}
    q = deque([start])
    while q:
        p = q.popleft()
        d = dist[p]
        for dy, dx in _N8:
            q2 = (p[0] + dy, p[1] + dx)
            if q2 in pts and q2 not in dist:
                dist[q2] = d + 1
                par[q2] = p
                q.append(q2)
    return dist, par


def _trace_chain(panel):
    """Longest geodesic pixel chain through the largest ink component
    (double BFS: farthest point from a seed, then farthest from that)."""
    pts = _largest_component_pts(panel)
    if len(pts) < 3:
        return []
    seed = next(iter(sorted(pts)))
    d0, _ = _bfs(pts, seed)
    start = max(d0.items(), key=lambda kv: (kv[1], kv[0]))[0]
    d1, par = _bfs(pts, start)
    end = max(d1.items(), key=lambda kv: (kv[1], kv[0]))[0]
    chain = []
    p = end
    while p is not None:
        chain.append(p)
        p = par[p]
    chain.reverse()
    return chain  # list of (y, x)


def _polyline(panel, step=8):
    """Smooth traced chain (moving average) and resample every `step` px
    -> [(x, y), ...]."""
    chain = _trace_chain(panel)
    if len(chain) < 3:
        return []
    arr = np.array([(c[1], c[0]) for c in chain], dtype=float)
    w = 5
    if len(arr) > w:
        kern = np.ones(w) / w
        sm = np.empty_like(arr)
        sm[:, 0] = np.convolve(arr[:, 0], kern, mode="same")
        sm[:, 1] = np.convolve(arr[:, 1], kern, mode="same")
        h = w // 2
        sm[:h] = arr[:h]
        sm[-h:] = arr[-h:]
        arr = sm
    idx = list(range(0, len(arr), step))
    if idx[-1] != len(arr) - 1:
        idx.append(len(arr) - 1)
    return [tuple(arr[i]) for i in idx]


def _turn_angles(panel, step=8):
    pts = _polyline(panel, step=step)
    if len(pts) < 3:
        return []
    angs = [math.atan2(b[1] - a[1], b[0] - a[0])
            for a, b in zip(pts[:-1], pts[1:])]
    turns = []
    for a1, a2 in zip(angs[:-1], angs[1:]):
        turns.append((a2 - a1 + math.pi) % (2.0 * math.pi) - math.pi)
    return turns


def p_net_turning_deg(panel):
    """|integrated signed turning| along the main stroke chain (deg).
    ~90 for a quarter arc, ~0 for a line or S-curve."""
    t = _turn_angles(panel)
    return abs(math.degrees(sum(t))) if t else 0.0


def p_abs_turning_deg(panel):
    """Integrated |turning| along the main stroke chain (deg)."""
    t = _turn_angles(panel)
    return math.degrees(sum(abs(x) for x in t)) if t else 0.0


def p_turning_consistency(panel):
    """net/abs turning in [0,1]: 1 = curvature never changes sign."""
    t = _turn_angles(panel)
    tot = sum(abs(x) for x in t)
    if tot < 1e-6:
        return 1.0
    return float(abs(sum(t)) / tot)


def p_curvature_sign_changes(panel):
    """Number of significant curvature sign flips along the stroke."""
    t = _turn_angles(panel)
    thr = math.radians(8.0)
    sig = [x for x in t if abs(x) > thr]
    flips = 0
    for a, b in zip(sig[:-1], sig[1:]):
        if a * b < 0:
            flips += 1
    return float(flips)


def p_path_length(panel):
    """Length in pixels of the longest geodesic chain of the main component."""
    return float(len(_trace_chain(panel)))


def p_chord_over_pathlen(panel):
    """Endpoint chord / chain length of main stroke (1 = straight line)."""
    chain = _trace_chain(panel)
    if len(chain) < 2:
        return 1.0
    chord = math.hypot(chain[-1][0] - chain[0][0], chain[-1][1] - chain[0][1])
    return float(chord / max(len(chain) - 1, 1))


def p_turn_dev_from_quarter(panel):
    """|net turning - 90 deg|: 0 means the stroke makes a quarter turn."""
    return abs(p_net_turning_deg(panel) - 90.0)


def p_max_turn_deg(panel):
    """Sharpest single turning angle along the stroke (corner detector)."""
    t = _turn_angles(panel)
    return math.degrees(max(abs(x) for x in t)) if t else 0.0


def p_turning_rate_std(panel):
    """Std of signed turning per step (deg); low = uniform-curvature arc."""
    t = _turn_angles(panel)
    if len(t) < 2:
        return 0.0
    return float(np.std(np.degrees(t)))


# ---------- chain-based circle-arc measurements ----------

def _chain_circle(panel):
    """Circle fit to the traced main chain (thin, ordered) rather than all ink.
    Returns (cx, cy, r, norm_resid, chain)."""
    chain = _trace_chain(panel)
    if len(chain) < 5:
        return 64.0, 64.0, 1.0, 1.0, chain
    xs = np.array([c[1] for c in chain], dtype=float)
    ys = np.array([c[0] for c in chain], dtype=float)
    cx, cy, r, resid = _fit_circle_pts(xs, ys)
    return cx, cy, r, resid, chain


def p_chain_circle_residual(panel):
    """Circle-fit residual of the traced main stroke chain (0 = clean arc)."""
    return _chain_circle(panel)[3]


def p_chain_arc_extent_deg(panel):
    """Angular extent of the traced chain around its own best-fit circle,
    measured monotonically start-to-end (deg). ~90 quarter arc, ~0 line/S."""
    cx, cy, r, resid, chain = _chain_circle(panel)
    if len(chain) < 5 or resid > 0.5:
        return 0.0
    ang = np.unwrap(np.arctan2(
        np.array([c[0] for c in chain], dtype=float) - cy,
        np.array([c[1] for c in chain], dtype=float) - cx))
    return float(abs(math.degrees(ang[-1] - ang[0])))


def p_chain_arc_dev_from_quarter(panel):
    """|chain arc extent - 90 deg|; 0 = exact quarter-circle arc."""
    return abs(p_chain_arc_extent_deg(panel) - 90.0)


def p_ink_near_chain_circle_frac(panel):
    """Fraction of ALL ink pixels lying within 3px of the chain-fit circle.
    Low if there are extra strokes/barbs off the main arc."""
    cx, cy, r, resid, chain = _chain_circle(panel)
    xs, ys = _ink_coords(panel)
    if len(xs) == 0:
        return 0.0
    d = np.abs(np.hypot(xs - cx, ys - cy) - r)
    return float(np.mean(d <= 3.0))
```

Please run the verifier and report the RESULT line.

RESULT solved=False heldout=0.653 train=0.917 rule="p_chain_circle_residual<=0.008628" rule_cost=1.5 predicate_errors=0

=====

