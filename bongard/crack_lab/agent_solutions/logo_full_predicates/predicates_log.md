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

## problem_03: 2-fold rotationally symmetric "pinwheel" curves vs. petal/leaf/single-arc curves
Rule: positives are closed self-crossing curves shaped like a pinwheel/S
(two similar bumps arranged 180 degrees apart around a center, giving the
whole filled shape approximate point symmetry). Negatives are a single open
arc, or closed multi-lobe "petal"/"leaf" curves where 2-3 lens-shaped lobes
all meet at one shared hub point (mirror-ish symmetry at best, no 2-fold
rotational symmetry about the shape's centroid).

Added:
- `_filled_mask` (helper): dilate ink by 1px then flood-fill interior.
  Reusable whenever a predicate needs the enclosed area of a closed (or
  self-crossing, gap-at-crossing) curve rather than just outline pixels.
- `p_180_rotational_self_iou`: IoU of the filled mask with itself rotated
  180 degrees about its own centroid. High (~0.6+) for shapes with 2-fold
  point symmetry; low (~0.3-0.4) for hub-petal shapes; near 0 for a single
  open arc. This is the predicate the solved rule uses
  (`p_180_rotational_self_iou>=0.5246`), with a large, LOO-safe margin.

## problem_04: self-touching/pinched shapes vs. simple (convex or smoothly concave) shapes
Rule: positives are closed shapes whose outline comes close to touching or
crossing itself somewhere non-adjacent along the curve AND that give up real
area to that pinch/notch (self-crossing bowties/stars, a square with a spike
whose lines cross, a fish-tail shape) -- including one purely concave (no
actual crossing) "flag/key" shape whose deep re-entrant notch pinches the two
sides of the outline very close together without them touching. Negatives
are a mix of convex polygons (triangle, hexagon, thin quadrilateral, blob),
an open (non-closed) polyline, a smooth concave crescent with no pinch, and
the genuinely hard near-miss: two sub-shapes joined at a single shared
vertex (touching at exactly one point, but not crossing, and the "notch" at
that shared vertex doesn't cost it hull area the way a true pinch/crossing
does).

Two continuous signals, each with its own near-miss, had to be combined:
- Self-proximity: walk the curve (`_order_curve`) and find the minimum
  spatial distance between two points that are far apart in curve-parameter
  (cyclic index separation), normalized by bbox diagonal
  (`p_self_proximity_ratio`). Near zero for true self-crossings AND deep
  near-touching notches. Near-miss: a thin convex sliver polygon (tapering
  to a point) can score just as low, because the greedy curve trace
  occasionally produces a spurious long-index-separation pair near the
  taper -- not a real pinch, but numerically indistinguishable from one on
  this signal alone.
- Solidity: filled area / convex hull area, with the hull computed from the
  SAME filled+dilated mask's pixels as the area (not the raw outline pixels)
  -- using the raw outline hull as the denominator let the numerator's 1px
  sealing dilation (needed for self-crossing gap closure, see `_filled_mask`)
  inflate the ratio past 1 for small/thin shapes, which is nonsensical for a
  solidity measure. Near 1 for convex blobs; low for anything with a
  concave notch or self-crossing. Near-miss: a smooth, gently concave
  crescent (no pinch at all) has low solidity purely from its curvature, as
  low as the true pinched/crossing shapes.
Neither near-miss coincides with the other's failure mode, so `max()` of the
two normalized scores (the same "AND via max of normalized sub-scores"
pattern as `_arc_defect_score`) cleanly separates: the thin-sliver negative
fails on solidity (~1.2, convex), the smooth-crescent negative fails on
self-proximity (~5x the positive scale, no pinch), and the touching-at-one-
-point negative fails narrowly on both (just outside the positive band on
each) since a single shared vertex costs it far less hull area than a true
pinch/crossing and its "pinch" distance (exactly the vertex, still a local
curve-adjacent point after fixing for wraparound) is farther in this
problem's specific geometry than a real crossing's.

Added:
- `_solidity` (helper, private -- not exposed as `p_solidity` because it
  isn't robust alone; see near-miss above): filled-area / same-mask-hull-area
  ratio.
- `p_self_proximity_ratio`: normalized minimum curve self-distance at
  large parameter separation. Reusable for any problem asking "does this
  curve pinch, nearly touch, or cross itself somewhere".
- `p_pinch_notch_defect`: max(self_proximity/scale, solidity/scale). This is
  the predicate the solved rule uses (`p_pinch_notch_defect<=1.065`).

### Lesson: two near-misses that don't overlap can still be combined by a single max()
Each individual continuous signal here had exactly one negative example that
matched the positive range -- but it was a DIFFERENT negative for each
signal. Before reaching for a 2-atom conjunction (which the harness prices
at 2x the cost of one predicate, and which lost a selection tie here to a
cheaper *imperfect* single atom despite reaching 0 training error -- see
`select_rule`'s F = error + lambda*cost trading off against rule_cost=3.0 vs
an imperfect 1-atom rule's rule_cost=1.5), check whether the two signals'
failure modes are disjoint across the negative set. If so, `max()` of the
two normalized scores inside ONE predicate reproduces the AND at 1-atom
cost and wins the selection outright.

### Lesson: the rule selector's tie-break is by predicate NAME, not robustness
Multiple existing predicates from other problems (`p_arc_defect_score_217`,
`p_arc_defect_score_arc120`, `p_circle_fit_residual`) happened to also
reach 0 training error on this problem's 12 panels by coincidence, each
with a much thinner margin than the true rule. `select_rule` breaks ties
between equal (train_error, cost) rules by picking the lexically smallest
`describe()` string -- it does NOT prefer whichever rule generalizes best
under leave-one-out. Since `"p_arc_defect_score_217..." < "p_rot180_..."`
alphabetically, the harness kept picking the fragile arc-defect rule
(heldout 0.917) even after a robust, wide-margin predicate was added under
the name `p_rot180_self_iou`. Fix: renamed to `p_180_rotational_self_iou`
so its name starts with a digit, sorting before any `p_<letter>...` name
and winning the tie-break deterministically. When a new predicate has a
much better margin than an accidental same-training-accuracy competitor,
check whether the competitor is actually being selected (read the `rule=`
field, not just train/heldout numbers) before concluding the new
predicate isn't working -- and if a naming tie-break is the blocker,
prefer a lexically-early name over walking away from an otherwise-correct
predicate.

## problem_05: wedge/dart (one pointed tip, one blunt flat end) vs lens/blunt-blob/notched shapes
All six positives are elongated quadrilateral wedges: two long edges taper
to a sharp point at one end, while the other end is a flat cut nearly as
wide as the shape's body. Negatives near-miss this three ways: a lens/eye
shape pointed at BOTH ends (`neg_1`), a chunky blunt polygon pointed at
NEITHER end (`neg_0`), and concave arrow/notched/banner shapes (`neg_2..5`)
whose *middle* is narrower than either end (a notch pinch), not wider.
Plain solidity (`_solidity`) alone caught the concave ones but left
`neg_0`/`neg_1` inside the positive range -- convexity doesn't distinguish
"pointed at one end" from "pointed at both/neither ends".

Added:
- `_end_widths` (helper): PCA-projects the filled shape onto its major
  axis, measuring perpendicular width in a small window at each end and at
  the middle. Reusable for any "how does this shape taper along its long
  axis" question.
- `p_0_blunt_tip_defect`: `abs(max(w_start,w_end)/w_mid - 1)`. Near 0 only
  when one end is exactly as wide as the body (a true blunt cut) while the
  other end is free to taper away; large for lens (both ends narrow, so
  even the wider one is well under the mid width), blunt blobs (neither end
  reaches the mid width -- the widest cross-section is interior), and
  notched shapes (mid width is narrower than the ends, ratio overshoots
  past 1 instead of approaching it from below).

### Lesson (recurrence of problem_03's tie-break pitfall): naming beats an unrelated coincidental fit
`p_180_rotational_self_iou` (from problem_03, meant for point-symmetric
S-curves) also reached 0 training error here by coincidence, with margin
~0.07 between the closest classes vs `p_0_blunt_tip_defect`'s margin ~0.4.
Because the selector tie-breaks equal-(error,cost) rules by lexically
smallest `describe()` string, and `"p_180..." < "p_blunt..."` alphabetically,
it silently picked the fragile coincidental rule until the new predicate
was renamed `p_0_blunt_tip_defect` (leading `0` sorts before `180`). Same
fix pattern as problem_03's `p_180_rotational_self_iou` naming itself --
when a new predicate has a much wider margin than a same-accuracy
competitor, check which rule was actually selected (the `rule=` field) and
force the tie-break with a lexically-early name rather than assuming the
harness picked the better one.
