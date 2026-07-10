Looking at the panels: positives are all single open smooth arcs of roughly half-circle extent; negatives are a closed circle, closed leaf shapes, a nearly-closed arc with a hook, and a shallow arc. I'll build predicates for holes/closedness and arc angular extent via circle fitting.

```python
# Shared predicate library. p_<name>(panel) -> float | bool

import math
import numpy as np
from scipy import ndimage


def _ink_points(panel):
    """Return Nx2 float array of (x, y) ink pixel coordinates."""
    ys, xs = np.nonzero(np.asarray(panel) > 0)
    return np.column_stack([xs, ys]).astype(float)


def _circle_fit(pts):
    """Least-squares (Kasa) circle fit. Returns (cx, cy, r) or None."""
    if pts.shape[0] < 3:
        return None
    x = pts[:, 0]
    y = pts[:, 1]
    A = np.column_stack([2.0 * x, 2.0 * y, np.ones_like(x)])
    b = x * x + y * y
    try:
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy, c = sol
    r2 = c + cx * cx + cy * cy
    if r2 <= 0:
        return None
    return cx, cy, math.sqrt(r2)


def p_ink_area(panel):
    """Number of ink pixels."""
    return float(np.count_nonzero(np.asarray(panel) > 0))


def p_num_components(panel):
    """Number of 8-connected ink components."""
    arr = (np.asarray(panel) > 0).astype(np.uint8)
    structure = np.ones((3, 3), dtype=int)
    _, n = ndimage.label(arr, structure=structure)
    return float(n)


def p_num_holes(panel):
    """Number of enclosed background regions (holes)."""
    arr = (np.asarray(panel) > 0).astype(np.uint8)
    padded = np.pad(arr, 1, mode="constant", constant_values=0)
    bg = (padded == 0).astype(np.uint8)
    _, n = ndimage.label(bg)  # 4-connectivity for background
    return float(max(0, n - 1))


def p_arc_extent_deg(panel):
    """Angular extent (degrees) of ink around its best-fit circle center."""
    pts = _ink_points(panel)
    fit = _circle_fit(pts)
    if fit is None:
        return 0.0
    cx, cy, _ = fit
    ang = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    ang = np.sort(ang)
    if ang.size < 2:
        return 0.0
    gaps = np.diff(ang)
    wrap = (ang[0] + 2.0 * math.pi) - ang[-1]
    max_gap = max(float(gaps.max()) if gaps.size else 0.0, float(wrap))
    return float(360.0 - math.degrees(max_gap))


def p_arc_extent_dev_from_half(panel):
    """Absolute deviation of angular extent from a half circle (200 deg)."""
    return abs(p_arc_extent_deg(panel) - 200.0)


def p_circle_fit_rms(panel):
    """RMS radial residual of best-fit circle, normalized by radius."""
    pts = _ink_points(panel)
    fit = _circle_fit(pts)
    if fit is None:
        return 1.0
    cx, cy, r = fit
    if r <= 0:
        return 1.0
    d = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
    return float(np.sqrt(np.mean((d - r) ** 2)) / r)


def p_diameter_over_ink(panel):
    """Max pairwise ink distance divided by ink pixel count (chord/length proxy)."""
    pts = _ink_points(panel)
    n = pts.shape[0]
    if n < 2:
        return 0.0
    if n > 600:
        idx = np.linspace(0, n - 1, 600).astype(int)
        pts = pts[idx]
    d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2)
    return float(math.sqrt(d2.max()) / n)


def p_endpoint_count(panel):
    """Count ink pixels with exactly one 8-neighbor (stroke endpoints)."""
    arr = (np.asarray(panel) > 0).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=int)
    kernel[1, 1] = 0
    neigh = ndimage.convolve(arr.astype(int), kernel, mode="constant", cval=0)
    return float(np.count_nonzero((arr == 1) & (neigh == 1)))
```

Please run the verifier and report the RESULT line.

RESULT solved=False heldout=0.917 train=1.000 rule="p_arc_extent_dev_from_half<=29.29" rule_cost=1.5 predicate_errors=0

The extent-by-max-gap measure has thin margins (a hook near the opening or smearing shifts it). I'll add an angular *coverage* measure (fraction of angle bins around the fitted circle center that contain ink), which saturates at 360 for closed shapes (leaves, circle) and near-closed hooks, stays ~200 for the positive half-arcs, and is low for the shallow arc — larger margins on both sides.

```python
# Shared predicate library. p_<name>(panel) -> float | bool

import math
import numpy as np
from scipy import ndimage


def _ink_points(panel):
    """Return Nx2 float array of (x, y) ink pixel coordinates."""
    ys, xs = np.nonzero(np.asarray(panel) > 0)
    return np.column_stack([xs, ys]).astype(float)


def _circle_fit(pts):
    """Least-squares (Kasa) circle fit. Returns (cx, cy, r) or None."""
    if pts.shape[0] < 3:
        return None
    x = pts[:, 0]
    y = pts[:, 1]
    A = np.column_stack([2.0 * x, 2.0 * y, np.ones_like(x)])
    b = x * x + y * y
    try:
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy, c = sol
    r2 = c + cx * cx + cy * cy
    if r2 <= 0:
        return None
    return cx, cy, math.sqrt(r2)


def p_ink_area(panel):
    """Number of ink pixels."""
    return float(np.count_nonzero(np.asarray(panel) > 0))


def p_num_components(panel):
    """Number of 8-connected ink components."""
    arr = (np.asarray(panel) > 0).astype(np.uint8)
    structure = np.ones((3, 3), dtype=int)
    _, n = ndimage.label(arr, structure=structure)
    return float(n)


def p_num_holes(panel):
    """Number of enclosed background regions (holes)."""
    arr = (np.asarray(panel) > 0).astype(np.uint8)
    padded = np.pad(arr, 1, mode="constant", constant_values=0)
    bg = (padded == 0).astype(np.uint8)
    _, n = ndimage.label(bg)  # 4-connectivity for background
    return float(max(0, n - 1))


def p_arc_extent_deg(panel):
    """Angular extent (degrees) of ink around its best-fit circle center,
    measured as 360 minus the largest empty angular gap."""
    pts = _ink_points(panel)
    fit = _circle_fit(pts)
    if fit is None:
        return 0.0
    cx, cy, _ = fit
    ang = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    ang = np.sort(ang)
    if ang.size < 2:
        return 0.0
    gaps = np.diff(ang)
    wrap = (ang[0] + 2.0 * math.pi) - ang[-1]
    max_gap = max(float(gaps.max()) if gaps.size else 0.0, float(wrap))
    return float(360.0 - math.degrees(max_gap))


def p_arc_extent_dev_from_half

```python
# Shared predicate library. p_<name>(panel) -> float | bool

import math
import numpy as np
from scipy import ndimage


def _ink_points(panel):
    """Return Nx2 float array of (x, y) ink pixel coordinates."""
    ys, xs = np.nonzero(np.asarray(panel) > 0)
    return np.column_stack([xs, ys]).astype(float)


def _circle_fit(pts):
    """Least-squares (Kasa) circle fit. Returns (cx, cy, r) or None."""
    if pts.shape[0] < 3:
        return None
    x = pts[:, 0]
    y = pts[:, 1]
    A = np.column_stack([2.0 * x, 2.0 * y, np.ones_like(x)])
    b = x * x + y * y
    try:
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy, c = sol
    r2 = c + cx * cx + cy * cy
    if r2 <= 0:
        return None
    return cx, cy, math.sqrt(r2)


def p_ink_area(panel):
    """Number of ink pixels."""
    return float(np.count_nonzero(np.asarray(panel) > 0))


def p_num_components(panel):
    """Number of 8-connected ink components."""
    arr = (np.asarray(panel) > 0).astype(np.uint8)
    structure = np.ones((3, 3), dtype=int)
    _, n = ndimage.label(arr, structure=structure)
    return float(n)


def p_num_holes(panel):
    """Number of enclosed background regions (holes)."""
    arr = (np.asarray(panel) > 0).astype(np.uint8)
    padded = np.pad(arr, 1, mode="constant", constant_values=0)
    bg = (padded == 0).astype(np.uint8)
    _, n = ndimage.label(bg)  # 4-connectivity for background
    return float(max(0, n - 1))


def p_arc_extent_deg(panel):
    """Angular extent (degrees) of ink around its best-fit circle center,
    measured as 360 minus the largest empty angular gap."""
    pts = _ink_points(panel)
    fit = _circle_fit(pts)
    if fit is None:
        return 0.0
    cx, cy, _ = fit
    ang = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    ang = np.sort(ang)
    if ang.size < 2:
        return 0.0
    gaps = np.diff(ang)
    wrap = (ang[0] + 2.0 * math.pi) - ang[-1]
    max_gap = max(float(gaps.max()) if gaps.size else 0.0, float(wrap))
    return float(360.0 - math.degrees(max_gap))


def p_arc_extent_dev_from_half(panel):
    """Absolute deviation of angular extent from a half circle (200 deg)."""
    return abs(p_arc_extent_deg(panel) - 200.0)


def p_arc_coverage_deg(panel):
    """Degrees of angle bins (out of 360) around the best-fit circle
    center that contain at least one ink pixel. Robust coverage measure:
    closed shapes give ~360, half arcs ~180-220, shallow arcs low."""
    pts = _ink_points(panel)
    fit = _circle_fit(pts)
    if fit is None:
        return 0.0
    cx, cy, _ = fit
    ang = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    bins = np.floor((ang + math.pi) / (2.0 * math.pi) * 72.0).astype(int)
    bins = np.clip(bins, 0, 71)
    return float(np.unique(bins).size * 5.0)


def p_arc_coverage_dev_from_half(panel):
    """Absolute deviation of angular coverage from a half circle (200 deg)."""
    return abs(p_arc_coverage_deg(panel) - 200.0)


def p_circle_fit_rms(panel):
    """RMS radial residual of best-fit circle, normalized by radius."""
    pts = _ink_points(panel)
    fit = _circle_fit(pts)
    if fit is None:
        return 1.0
    cx, cy, r = fit
    if r <= 0:
        return 1.0
    d = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
    return float(np.sqrt(np.mean((d - r) ** 2)) / r)


def p_diameter_over_ink(panel):
    """Max pairwise ink distance divided by ink pixel count (chord/length proxy)."""
    pts = _ink_points(panel)
    n = pts.shape[0]
    if n < 2:
        return 0.0
    if n > 600:
        idx = np.linspace(0, n - 1, 600).astype(int)
        pts = pts[idx]
    d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2)
    return float(math.sqrt(d2.max()) / n)


def p_endpoint_count(panel):
    """Count ink pixels with exactly one 8-neighbor (stroke endpoints)."""
    arr = (np.asarray(panel) > 0).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=int)
    kernel[1, 1] = 0
    neigh = ndimage.convolve(arr.astype(int), kernel, mode="constant", cval=0)
    return float(np.count_nonzero((arr == 1) & (neigh == 1)))
```

RESULT solved=False heldout=0.917 train=1.000 rule="p_arc_coverage_dev_from_half<=35" rule_cost=1.5 predicate_errors=0

The extent measures work on train but with thin margins. I'll add a more robust arc-angle measure based on endpoint chord + sagitta (no circle fit needed), which cleanly saturates at 360 for closed/near-closed shapes and gives a low angle for shallow arcs, with wide margins around the positives' ~180-220 degrees.

```python
# Shared predicate library. p_<name>(panel) -> float | bool

import math
import numpy as np
from scipy import ndimage


def _ink_points(panel):
    """Return Nx2 float array of (x, y) ink pixel coordinates."""
    ys, xs = np.nonzero(np.asarray(panel) > 0)
    return np.column_stack([xs, ys]).astype(float)


def _circle_fit(pts):
    """Least-squares (Kasa) circle fit. Returns (cx, cy, r) or None."""
    if pts.shape[0] < 3:
        return None
    x = pts[:, 0]
    y = pts[:, 1]
    A = np.column_stack([2.0 * x, 2.0 * y, np.ones_like(x)])
    b = x * x + y * y
    try:
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy, c = sol
    r2 = c + cx * cx + cy * cy
    if r2 <= 0:
        return None
    return cx, cy, math.sqrt(r2)


def _endpoints(panel):
    """Return Nx2 (x, y) array of stroke endpoints (ink pixels with
    exactly one 8-neighbor)."""
    arr = (np.asarray(panel) > 0).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=int)
    kernel[1, 1] = 0
    neigh = ndimage.convolve(arr.astype(int), kernel, mode="constant", cval=0)
    ys, xs = np.nonzero((arr == 1) & (neigh == 1))
    return np.column_stack([xs, ys]).astype(float)


def p_ink_area(panel):
    """Number of ink pixels."""
    return float(np.count_nonzero(np.asarray(panel) > 0))


def p_num_components(panel):
    """Number of 8-connected ink components."""
    arr = (np.asarray(panel) > 0).astype(np.uint8)
    structure = np.ones((3, 3), dtype=int)
    _, n = ndimage.label(arr, structure=structure)
    return float(n)


def p_num_holes(panel):
    """Number of enclosed background regions (holes)."""
    arr = (np.asarray(panel) > 0).astype(np.uint8)
    padded = np.pad(arr, 1, mode="constant", constant_values=0)
    bg = (padded == 0).astype(np.uint8)
    _, n = ndimage.label(bg)  # 4-connectivity for background
    return float(max(0, n - 1))


def p_arc_extent_deg(panel):
    """Angular extent (degrees) of ink around its best-fit circle center,
    measured as 360 minus the largest empty angular gap."""
    pts = _ink_points(panel)
    fit = _circle_fit(pts)
    if fit is None:
        return 0.0
    cx, cy, _ = fit
    ang = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    ang = np.sort(ang)
    if ang.size < 2:
        return 0.0
    gaps = np.diff(ang)
    wrap = (ang[0] + 2.0 * math.pi) - ang[-1]
    max_gap = max(float(gaps.max()) if gaps.size else 0.0, float(wrap))
    return float(360.0 - math.degrees(max_gap))


def p_arc_extent_dev_from_half(panel):
    """Absolute deviation of angular extent from a half circle (200 deg)."""
    return abs(p_arc_extent_deg(panel) - 200.0)


def p_arc_coverage_deg(panel):
    """Degrees of angle bins (out of 360) around the best-fit circle
    center that contain at least one ink pixel."""
    pts = _ink_points(panel)
    fit = _circle_fit(pts)
    if fit is None:
        return 0.0
    cx, cy, _ = fit
    ang = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    bins = np.floor((ang + math.pi) / (2.0 * math.pi) * 72.0).astype(int)
    bins = np.clip(bins, 0, 71)
    return float(np.unique(bins).size * 5.0)


def p_arc_coverage_dev_from_half(panel):
    """Absolute deviation of angular coverage from a half circle (200 deg)."""
    return abs(p_arc_coverage_deg(panel) - 200.0)


def p_arc_angle_sagitta_deg(panel):
    """Arc angle (degrees) estimated from endpoint chord and sagitta.

    Uses the two farthest stroke endpoints as the chord; sagitta is the
    max perpendicular distance of ink from that chord. For a circular arc
    of angle t: sagitta/chord = tan(t/4)/2, so t = 4*atan(2*sag/chord).
    Closed curves (fewer than 2 endpoints) return 360."""
    eps = _endpoints(panel)
    if eps.shape[0] < 2:
        return 360.0
    # pick the pair of endpoints with maximum separation
    d2 = ((eps[:, None, :] - eps[None, :, :]) ** 2).sum(axis=2)
    i, j = np.unravel_index(np.argmax(d2), d2.shape)
    a = eps[i]
    b = eps[j]
    chord = math.sqrt(float(d2[i, j]))
    if chord < 2.0:
        return 360.0
    pts = _ink_points(panel)
    # perpendicular distance of all ink points from line a-b
    ux = (b[0] - a[0]) / chord
    uy = (b[1] - a[1]) / chord
    perp = np.abs((pts[

```python
# Shared predicate library. p_<name>(panel) -> float | bool

import math
import numpy as np
from scipy import ndimage


def _ink_points(panel):
    """Return Nx2 float array of (x, y) ink pixel coordinates."""
    ys, xs = np.nonzero(np.asarray(panel) > 0)
    return np.column_stack([xs, ys]).astype(float)


def _circle_fit(pts):
    """Least-squares (Kasa) circle fit. Returns (cx, cy, r) or None."""
    if pts.shape[0] < 3:
        return None
    x = pts[:, 0]
    y = pts[:, 1]
    A = np.column_stack([2.0 * x, 2.0 * y, np.ones_like(x)])
    b = x * x + y * y
    try:
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy, c = sol
    r2 = c + cx * cx + cy * cy
    if r2 <= 0:
        return None
    return cx, cy, math.sqrt(r2)


def _endpoints(panel):
    """Return Nx2 (x, y) array of stroke endpoints (ink pixels with
    exactly one 8-neighbor)."""
    arr = (np.asarray(panel) > 0).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=int)
    kernel[1, 1] = 0
    neigh = ndimage.convolve(arr.astype(int), kernel, mode="constant", cval=0)
    ys, xs = np.nonzero((arr == 1) & (neigh == 1))
    return np.column_stack([xs, ys]).astype(float)


def p_ink_area(panel):
    """Number of ink pixels."""
    return float(np.count_nonzero(np.asarray(panel) > 0))


def p_num_components(panel):
    """Number of 8-connected ink components."""
    arr = (np.asarray(panel) > 0).astype(np.uint8)
    structure = np.ones((3, 3), dtype=int)
    _, n = ndimage.label(arr, structure=structure)
    return float(n)


def p_num_holes(panel):
    """Number of enclosed background regions (holes)."""
    arr = (np.asarray(panel) > 0).astype(np.uint8)
    padded = np.pad(arr, 1, mode="constant", constant_values=0)
    bg = (padded == 0).astype(np.uint8)
    _, n = ndimage.label(bg)  # 4-connectivity for background
    return float(max(0, n - 1))


def p_arc_extent_deg(panel):
    """Angular extent (degrees) of ink around its best-fit circle center,
    measured as 360 minus the largest empty angular gap."""
    pts = _ink_points(panel)
    fit = _circle_fit(pts)
    if fit is None:
        return 0.0
    cx, cy, _ = fit
    ang = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    ang = np.sort(ang)
    if ang.size < 2:
        return 0.0
    gaps = np.diff(ang)
    wrap = (ang[0] + 2.0 * math.pi) - ang[-1]
    max_gap = max(float(gaps.max()) if gaps.size else 0.0, float(wrap))
    return float(360.0 - math.degrees(max_gap))


def p_arc_extent_dev_from_half(panel):
    """Absolute deviation of angular extent from a half circle (200 deg)."""
    return abs(p_arc_extent_deg(panel) - 200.0)


def p_arc_coverage_deg(panel):
    """Degrees of angle bins (out of 360) around the best-fit circle
    center that contain at least one ink pixel."""
    pts = _ink_points(panel)
    fit = _circle_fit(pts)
    if fit is None:
        return 0.0
    cx, cy, _ = fit
    ang = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    bins = np.floor((ang + math.pi) / (2.0 * math.pi) * 72.0).astype(int)
    bins = np.clip(bins, 0, 71)
    return float(np.unique(bins).size * 5.0)


def p_arc_coverage_dev_from_half(panel):
    """Absolute deviation of angular coverage from a half circle (200 deg)."""
    return abs(p_arc_coverage_deg(panel) - 200.0)


def p_arc_angle_sagitta_deg(panel):
    """Arc angle (degrees) estimated from endpoint chord and sagitta.

    Uses the two farthest stroke endpoints as the chord; sagitta is the
    max perpendicular distance of ink from that chord. For a circular arc
    of angle t: sagitta/chord = tan(t/4)/2, so t = 4*atan(2*sag/chord).
    Closed curves (fewer than 2 endpoints) return 360."""
    eps = _endpoints(panel)
    if eps.shape[0] < 2:
        return 360.0
    d2 = ((eps[:, None, :] - eps[None, :, :]) ** 2).sum(axis=2)
    i, j = np.unravel_index(np.argmax(d2), d2.shape)
    a = eps[i]
    b = eps[j]
    chord = math.sqrt(float(d2[i, j]))
    if chord < 2.0:
        return 360.0
    pts = _ink_points(panel)
    ux = (b[0] - a[0]) / chord
    uy = (b[1] - a[1]) / chord
    perp = np.abs((pts[:, 0] - a[0]) * uy - (pts[:, 1] - a[1]) * ux)
    sag = float(perp.max())
    ang = 4.0 * math.degrees(math.atan2(2.0 * sag, chord))
    return float(min(ang, 360.0))


def p_arc_angle_dev_from_half(panel):
    """Absolute deviation of sagitta-based arc angle from 200 degrees."""
    return abs(p_arc_angle_sagitta_deg(panel) - 200.0)


def p_circle_fit_rms(panel):
    """RMS radial residual of best-fit circle, normalized by radius."""
    pts = _ink_points(panel)
    fit = _circle_fit(pts)
    if fit is None:
        return 1.0
    cx, cy, r = fit
    if r <= 0:
        return 1.0
    d = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
    return float(np.sqrt(np.mean((d - r) ** 2)) / r)


def p_diameter_over_ink(panel):
    """Max pairwise ink distance divided by ink pixel count (chord/length proxy)."""
    pts = _ink_points(panel)
    n = pts.shape[0]
    if n < 2:
        return 0.0
    if n > 600:
        idx = np.linspace(0, n - 1, 600).astype(int)
        pts = pts[idx]
    d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2)
    return float(math.sqrt(d2.max()) / n)


def p_endpoint_count(panel):
    """Count ink pixels with exactly one 8-neighbor (stroke endpoints)."""
    arr = (np.asarray(panel) > 0).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=int)
    kernel[1, 1] = 0
    neigh = ndimage.convolve(arr.astype(int), kernel, mode="constant", cval=0)
    return float(np.count_nonzero((arr == 1) & (neigh == 1)))
```

RESULT solved=False heldout=0.917 train=1.000 rule="p_arc_coverage_dev_from_half<=35" rule_cost=1.5 predicate_errors=0

The band on coverage is tight because closed negatives (circle, leaves) get coverage values that the circle-fit can make erratic. I'll add a predicate that hard-rejects panels with enclosed regions (holes) before measuring coverage deviation — then only the two open negatives (near-closed hook arc, shallow arc) compete on the threshold, and they're far from the positives' ~200 degrees, giving a much wider LOO-safe margin.

```python
# Shared predicate library. p_<name>(panel) -> float | bool

import math
import numpy as np
from scipy import ndimage


def _ink_points(panel):
    """Return Nx2 float array of (x, y) ink pixel coordinates."""
    ys, xs = np.nonzero(np.asarray(panel) > 0)
    return np.column_stack([xs, ys]).astype(float)


def _circle_fit(pts):
    """Least-squares (Kasa) circle fit. Returns (cx, cy, r) or None."""
    if pts.shape[0] < 3:
        return None
    x = pts[:, 0]
    y = pts[:, 1]
    A = np.column_stack([2.0 * x, 2.0 * y, np.ones_like(x)])
    b = x * x + y * y
    try:
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy, c = sol
    r2 = c + cx * cx + cy * cy
    if r2 <= 0:
        return None
    return cx, cy, math.sqrt(r2)


def _endpoints(panel):
    """Return Nx2 (x, y) array of stroke endpoints (ink pixels with
    exactly one 8-neighbor)."""
    arr = (np.asarray(panel) > 0).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=int)
    kernel[1, 1] = 0
    neigh = ndimage.convolve(arr.astype(int), kernel, mode="constant", cval=0)
    ys, xs = np.nonzero((arr == 1) & (neigh == 1))
    return np.column_stack([xs, ys]).astype(float)


def p_ink_area(panel):
    """Number of ink pixels."""
    return float(np.count_nonzero(np.asarray(panel) > 0))


def p_num_components(panel):
    """Number of 8-connected ink components."""
    arr = (np.asarray(panel) > 0).astype(np.uint8)
    structure = np.ones((3, 3), dtype=int)
    _, n = ndimage.label(arr, structure=structure)
    return float(n)


def p_num_holes(panel):
    """Number of enclosed background regions (holes)."""
    arr = (np.asarray(panel) > 0).astype(np.uint8)
    padded = np.pad(arr, 1, mode="constant", constant_values=0)
    bg = (padded == 0).astype(np.uint8)
    _, n = ndimage.label(bg)  # 4-connectivity for background
    return float(max(0, n - 1))


def p_arc_extent_deg(panel):
    """Angular extent (degrees) of ink around its best-fit circle center,
    measured as 360 minus the largest empty angular gap."""
    pts = _ink_points(panel)
    fit = _circle_fit(pts)
    if fit is None:
        return 0.0
    cx, cy, _ = fit
    ang = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    ang = np.sort(ang)
    if ang.size < 2:
        return 0.0
    gaps = np.diff(ang)
    wrap = (ang[0] + 2.0 * math.pi) - ang[-1]
    max_gap = max(float(gaps.max()) if gaps.size else 0.0, float(wrap))
    return float(360.0 - math.degrees(max_gap))


def p_arc_extent_dev_from_half(panel):
    """Absolute deviation of angular extent from a half circle (200 deg)."""
    return abs(p_arc_extent_deg(panel) - 200.0)


def p_arc_coverage_deg(panel):
    """Degrees of angle bins (out of 360) around the best-fit circle
    center that contain at least one ink pixel."""
    pts = _ink_points(panel)
    fit = _circle_fit(pts)
    if fit is None:
        return 0.0
    cx, cy, _ = fit
    ang = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    bins = np.floor((ang + math.pi) / (2.0 * math.pi) * 72.0).astype(int)
    bins = np.clip(bins, 0, 71)
    return float(np.unique(bins).size * 5.0)


def p_arc_coverage_dev_from_half(panel):
    """Absolute deviation of angular coverage from a half circle (200 deg)."""
    return abs(p_arc_coverage_deg(panel) - 200.0)


def p_open_arc_coverage_dev(panel):
    """Deviation of angular coverage from 200 deg, but shapes that enclose
    any region (holes) are pushed to the max value 360. Separates open
    roughly-half-circle arcs from closed shapes, near-closed arcs, and
    shallow arcs with a wide margin."""
    if p_num_holes(panel) > 0:
        return 360.0
    return abs(p_arc_coverage_deg(panel) - 200.0)


def p_is_open_half_arc(panel):
    """True if the panel is a single open curve with no enclosed region
    whose angular coverage is roughly a half circle (145..255 deg)."""
    if p_num_holes(panel) > 0:
        return False
    return abs(p_arc_coverage_deg(panel) - 200.0) <= 55.0


def p_arc_angle_sagitta_deg(panel):
    """Arc angle (degrees) estimated from endpoint chord and sagitta.

    Uses the two farthest stroke endpoints as the chord; sagitta is the
    max perpendicular distance of ink from that chord. For a circular arc
    of angle t: sagitta/chord = tan(t/4)/2, so t = 4*atan(2*sag/chord).
    Closed curves (fewer than 2 endpoints) return 360."""
    eps = _endpoints(panel)
    if eps.shape[0] < 2:
        return 360.0
    d2 = ((eps[:, None, :] - eps[None, :, :]) ** 2).sum(axis=2)
    i, j = np.unravel_index(np.argmax(d2), d2.shape)
    a = eps[i]
    b = eps[j]
    chord

=====

