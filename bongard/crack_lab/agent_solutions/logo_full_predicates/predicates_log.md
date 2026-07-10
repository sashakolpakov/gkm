# Predicate library notes

## problem_00: 120-degree circular arcs vs. other curves
Rule: positives are all clean single circular arcs spanning ~120 degrees
(1/3 of a circle); negatives are near-misses that fail one of two ways —
either the ink isn't well-fit by any single circle (S-curves with an
inflection, straight/near-straight segments, closed multi-lobe shapes), or
it is a clean arc but of the wrong angular span (too short, too long, or a
near-full loop).

Added generic reusable predicates:
- `_fit_circle` (helper): least-squares circle fit (Kasa method) to the ink
  pixels. Reusable any time "is this a circular arc" comes up.
- `p_circle_fit_residual`: RMS distance of pixels from the fitted circle,
  normalized by bbox diagonal. Low = clean arc; high = anything else
  (S-curve, straight line, multi-loop).
- `p_arc_angular_span_deg`: angular span (degrees) subtended at the fitted
  circle's center. Useful whenever a problem cares about how "wide" an arc
  is (quarter circle vs. sliver vs. near-full loop), independent of scale.
- `p_arc_span_deviation_from_120`: |span - 120|. Specific to this problem's
  120-degree target, but the pattern (deviation-from-target-angle) is
  reusable by copying with a different constant.
- `p_arc_defect_score`: max of the two normalized signals above. This is the
  one actually used by the solved rule.

### Lesson on leave-one-out robustness
The harness's rule search picks the single cheapest atom (by cost, i.e.
fewest predicates) that gets 0 training error — it does NOT prefer the
"true" decision boundary. With a single scalar per panel, if one negative
example sits much closer to the positive cluster than the rest of the
negatives, removing it during leave-one-out lets the learned threshold
drift up past that example's true value, causing a heldout miss even
though train accuracy is 100%.
Fix: when combining two normalized sub-scores with `max()`, tune the
relative scaling (the normalization denominators) so that the closest
negative's own value stays clearly above the midpoint that would be
selected if it were excluded from training — i.e. make sure no single
negative is a scaling-fragile outlier relative to the "next closest"
negative. Check this numerically (recompute per-fold thresholds) rather
than just eyeballing overall separation, since global separation looking
"fine" (e.g. 3x margin) can still fail LOO if the gap structure is uneven.

## problem_02: two-circular-arc "wave" curves vs. single arcs / closed loops / other multi-arc curves
Rule: positives are single-stroke curves made of exactly two circular arcs
joined at one corner (a "wave": either same-direction double bump like an M,
or opposite-direction like an S), regardless of orientation or which way the
bumps point. Negatives fail differently: a single clean arc (no corner), a
closed loop (lens/leaf shape with real enclosed area), or — the genuinely
hard near-miss — another two-arc corner curve that looks topologically
identical to the eye (same S-shape gestalt, one corner, two open arcs) but
has much less total turning across its two arcs.

Per-pixel curvature (`np.diff` of tangent angle along the traced path) was
too noisy at this resolution to count corners/arcs reliably — pixelation
jaggies create spurious sign changes and spurious "corner" spikes both at
the true corner and elsewhere, so naive corner-counting or max-turn-angle
thresholds did not separate the hard near-miss from the positives.

What worked: fit the curve as *two* circular arcs by trying every split
point along the traced path and picking the split that minimizes total
circle-fit RMS residual (this robustly locates the true corner even under
pixel noise, since it's a whole-curve optimization rather than a local
derivative). Then sum the two arcs' angular spans. Clean two-arc wave
curves in this problem all land within ~5 degrees of 297 total degrees of
turning, regardless of scale/orientation/bump direction. The hard near-miss
negative had ~64 degrees less total turning — one of its two "arcs" was
much flatter than a true bump, which a single-circle-residual check on the
whole curve doesn't detect but the two-arc split's span sum does. Other
negative types (single arc, closed loop, messy multi-corner shape) deviate
from 297 by >100 degrees, or in one case (a near-closed crescent) by ~15-18
degrees — still comfortably outside the ~5-degree positive band.

Added generic reusable predicates:
- `_order_curve` (helper): greedy nearest-neighbor trace of a single-stroke
  curve's ink pixels into a spatially ordered polyline, starting from a
  low-neighbor-count pixel (likely tip). Reusable any time a predicate needs
  to walk along a curve rather than treat it as an unordered point cloud.
  Assumes one connected stroke without heavy branching.
- `_best_two_arc_split` (helper): tries every split of the ordered curve
  into two pieces, circle-fits each piece, and returns the split minimizing
  total residual. Locates the corner of a two-arc curve without relying on
  noisy per-pixel curvature. Reusable for any "is this made of (about) two
  arcs" question — e.g. measuring each arc's own radius/span separately.
- `p_two_arc_span_sum_deviation`: |sum of the two split arcs' angular spans
  - 297|. This is the predicate the solved rule uses
  (`p_two_arc_span_sum_deviation<=9.825`). The 297 target is specific to
  this problem's curve family; the reusable pattern — split into two arcs,
  sum their spans, compare to a target — mirrors `_arc_defect_score`'s
  target_deg parameter for single arcs and can be copied with a different
  constant.

### Lesson: near-identical topology can still separate on a continuous measurement
When two shapes share the same qualitative structure (same stroke topology,
same number of corners, same gestalt), don't assume the differentiating
rule must be a different topological feature. Try summing a continuous,
scale-invariant quantity (here: total angular turning) across the shape's
parts — near-miss negatives are often quantitatively, not structurally,
different from the positives.
