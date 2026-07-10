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

## problem_06: two polygons joined at one point, crossing vs. merely touching
Visually, every panel is two closed polygons (a bigger one plus a small
triangle) meeting at a single shared point -- but positives' two lines
actually CROSS there (a true X: two straight edges pass through each
other), while negatives' shapes only TOUCH corner-to-corner at a shared
polygon vertex (a Y/T-ish meeting where no two of the edges continue
straight through each other).

Solved for free by an existing predicate, no new code needed:
`p_0_blunt_tip_defect` (from problem_05's wedge/dart rule) reaches 0
training error with a wide, LOO-safe margin (positives <=0.19, negatives
>=0.49) on this problem's panels too, purely by coincidence of what it
measures on this shape family -- confirmed by rerunning the tester with a
purpose-built new predicate absent and seeing the same rule/heldout=1.0
selected. Per the reuse-first instructions, left the library unchanged.

Built (then removed after confirming it wasn't needed) a purpose-built
crossing-vs-touching predicate, in case a future problem needs the actual
concept: locate the busiest point (`_densest_point`, a KD-tree local-density
peak -- generic, reusable for any "where do sub-shapes join" question),
read out the ray directions radiating from it at several small annulus
radii (`_branch_angles`, generic ray-clustering-by-angle-gap helper), and
score how well the rays pair up into two opposite (180-degree) straight
lines (`_four_ray_crossing_defect`). Taking the BEST (smallest) defect
across a range of radii, rather than one fixed radius, mattered: at any
single radius, pixelation noise or an off-by-a-few-px density-peak pick
misestimates one ray's angle enough to spoil the pairing, but a true
crossing has *some* radius where the two lines read out cleanly, while a
true vertex-touch never presents a good opposite-pair at any radius. These
three helpers were deleted from predicates.py (not committed to the shared
library) once the existing predicate was confirmed sufficient -- documented
here only so the technique doesn't have to be rediscovered if a future
problem's near-miss genuinely hinges on crossing-vs-touching and no
existing predicate happens to correlate with it.

### Lesson: verify a suspiciously easy solve before adding anything
When `bongard_try.py` reports solved=True using an existing predicate whose
name/semantics don't obviously match the visual rule you identified by eye,
don't assume the harness is wrong or that your visual read is wrong --
check whether the existing predicate is *coincidentally* correlated with
the real distinguishing feature on this specific 12-panel sample (rerun
with any new candidate predicate removed/absent to confirm it still
solves). If it does, the reuse-minimizing objective says to leave the
library alone rather than adding a semantically-cleaner predicate that
would just add marginal cost for no accuracy gain.

## problem_07: circular sector + quadrilateral touching at a point, in matched proportion
Every panel is a circular sector ("fan") and a quadrilateral ("rectangle")
that meet at a single shared point (a sector corner touching a rectangle
corner) -- but positives' two sub-shapes are always drawn at a matched
relative scale (the fan is consistently a bit longer, end to end, than the
rectangle), while negatives fail either structurally (only one sub-shape
present at all -- a fan/wedge alone with no rectangle -- or two sub-shapes
of the wrong kind, e.g. a zigzag+fan or two triangles/bowtie with no fan)
or, in the one genuine near-miss (`neg_1`), by having the *same* two shape
types touching the *same* way but at a visibly mismatched scale (a
noticeably bigger rectangle relative to its fan).

Added:
- `_split_touching_pair` (helper): splits a panel's ink into two sub-shapes
  by removing a small disk around `_densest_point` (the join) and labeling
  what remains; returns None if that doesn't yield exactly two pieces.
  Reusable any time a predicate needs to reason about two touching
  sub-shapes independently rather than as one blob -- also directly useful
  as a structural gate (single shape, or a >2-branch join, fails to split).
- `_pca_extents` (helper): (major, minor) axis lengths of a point set via
  PCA -- a generic orientation-invariant length/width measurement, reused
  from the eigh-based pattern already used inline in `_end_widths`.
- `p_0_fan_quad_ratio_defect`: splits the panel, identifies which piece is
  the fan (lower whole-piece circle-fit residual -- same "lower residual
  wins" logic as elsewhere), and scores `|fan_long/quad_long - 1.191|`
  where `_long` is each piece's own PCA major-axis extent. Sentinel 5.0
  when the panel doesn't split into two pieces.

### Lesson: try several ratio denominators before trusting the first one that separates
The first candidate ratio tried, fan-radius / quad-*short*-side, separated
all 12 training panels with a seemingly comfortable margin (positives
0.70-0.77, closest negative 0.622) -- but `neg_1`, the one genuine
near-miss, sat only ~0.08 below the positive band while the *next*
negative sat ~0.24 below it. Per problem_00's lesson, this lopsided gap
structure means the LOO-fitted threshold (roughly the midpoint between the
positive band and the closest *remaining* negative) drifts to ~0.156 once
`neg_1` itself is the held-out test case, misclassifying it. Recomputing
the same underlying shapes with a different ratio -- fan's own long extent
over quad's long extent, rather than fan radius over quad short extent --
gave every negative a similarly large (~0.2+) margin from a *tighter*
positive band (spread 0.046 vs 0.068), because it's less sensitive to
exactly where the disk-removal cut happens to bite into each piece.
Lesson: when a first ratio's margin structure is uneven across negatives,
don't just tune the threshold -- try swapping in a different
(still-simple, still-scale-invariant) numerator/denominator pairing from
the same measured quantities before adding a second predicate.

### Recurrence: naming tie-break (see problem_03, problem_05)
Once the ratio itself was fixed, `p_180_rotational_self_iou` still won the
fold where `neg_4` was held out (it coincidentally also reaches 0 training
error there, by pure chance on this problem's shapes), because it sorts
lexically before `p_fan_quad_ratio_defect`. Renamed to
`p_0_fan_quad_ratio_defect` (leading `0` sorts before `1` in `p_180...`),
the same fix pattern as `p_0_blunt_tip_defect`. Check the `rule=` field
whenever heldout is just short of solved with a wide-margin new predicate
in play -- it may already be correct and merely losing a naming tie.

## problem_09: open curves / symmetric closed blobs vs lopsided-concave or two-part shapes
Visually the hardest-to-eyeball problem so far -- the six positives look
unrelated at a glance (an open two-arc wave, a closed lens, a single open
arc, a closed wavy quadrilateral, a closed rounded blob, and a crossing
X-stroke). What they share: each is EITHER a single open stroke (no
enclosed interior) OR a closed shape with strong 180-degree point
symmetry. Negatives are closed shapes that are neither: three are a single
polygon-ish outline with one or two straight edges replaced by a lopsided
concave arc scoop (breaks symmetry without being open), one is a straight-
edged zigzag/notched polygon (also asymmetric), and two are actually two
separate straight-edged sub-shapes (a quadrilateral + a triangle/chevron)
touching at one point.

Tried and abandoned: RDP polygon segmentation + per-segment line-vs-circle
classification to test a "no mixed straight+curved edges" hypothesis --
too noisy at this pixel resolution (spurious short segments), and it also
doesn't explain the all-straight-edged negatives (zigzag, two-touching-
shapes). Tried raw pixel-adjacency branch-point counting (degree>=3 in the
ink mask) to detect self-crossings/junctions directly -- also too noisy,
since a jagged 1px-wide curve's normal bends already create many spurious
"degree >= 3" pixels; not usable as a signal here.

What worked: `_fill_ratio` (new helper) -- filled-interior-pixels /
raw-ink-pixels via the existing `_filled_mask` -- cleanly separates open
strokes (~2.4-2.8) from any closed loop (~5.7-14), independent of size or
convexity, without needing to trace/order the curve at all. Combined with
the existing `p_180_rotational_self_iou` (from problem_03) via `min()`
(the same OR-via-min pattern `_arc_defect_score` uses for AND-via-max):
`p_open_or_symmetric_defect` is near zero if EITHER the shape is open OR
it's closed-and-symmetric, and large only for closed-and-asymmetric shapes
-- which covers both negative failure modes (lopsided concave scoop,
two-touching-sub-shapes) with one scalar, no separate junction-detection
predicate needed.

### Lesson: OR-via-min lets one scalar cover qualitatively different
### positive subgroups that a single monotonic measurement can't
Positives here split into two subgroups (open vs closed) that no single
existing scale (solidity, circle-fit residual, raw symmetry) treated
alike -- closed-shape solidity for the wavy-quad positive (0.884) was
actually *lower* than the zigzag negative's solidity (0.958), so plain
convexity was not the answer despite looking plausible from the pictures.
Whenever positives visually split into "obviously different" cases, check
whether each case is cleanly gated by a DIFFERENT existing signal, then
combine those signals with min() (defect scores, OR semantics) rather
than searching for one measurement that treats both cases the same way.

### Threshold tuning for LOO margin (recurrence of problem_00's lesson)
The first `sym_target` tried (0.85) gave a positive/negative gap of only
0.0558 between the positive band (0) and the closest negative (a lopsided-
concave kite whose 180-degree self-IoU, ~0.79, is coincidentally not that
low). Leave-one-out on that negative would have put the fitted threshold
above its own value once it was excluded from training. Raising
`sym_target` to 0.9 (tightening how symmetric a closed shape must be to
count as "symmetric enough") pushed that same negative's defect up to
~0.106 while leaving every positive's defect at exactly 0 (they all clear
either branch by a wide margin), restoring a safe LOO gap -- worth
rechecking the *target/threshold constant* inside a composite predicate,
not just which raw signals to combine, when the initial margin is thin.

## problem_10

Positives: thin symmetric "leaf"/lens outlines -- two smoothly bulging
arcs of comparable curvature meeting at a sharp point at BOTH ends.
Negatives were three qualitatively different near-misses of that same
lens silhouette: (a) a lopsided lens whose two sides bulge by very
different amounts (one side almost straight), (b) a pencil/wedge shape
that tapers to a point at one end but is cut flat (blunt) at the other,
and (c) chunkier straight-edged polygons (zigzag, pentagon, chevron)
that aren't a thin sliver at all.

No single existing scalar covered all three failure modes (each is a
near-miss of "clean symmetric lens" along a different axis), so
`p_0_asym_taper_or_blunt_or_chunky_defect` composes three defects via
max() (OR of failure modes, same pattern as `p_open_or_symmetric_defect`
and `p_pinch_notch_defect`):
- reused `_best_two_arc_split` (already existed for wave-curve problems)
  to get each side's fitted-circle radius, then flagged a big radius
  ratio between the two sides (asymmetric taper) -- new helper is just
  the ratio, no new curve-tracing code;
- reused `p_0_blunt_tip_defect` directly (near-0 flags a blunt/flat end);
- added one new helper `_hull_area_over_diag2` (convex hull area of the
  raw ink pixels / bbox-diagonal^2) as a scale-invariant chunkiness
  measurement to flag the polygon negatives.

### Recurrence of the LOO-thin-margin lesson, this time for a *composite*
The first attempt (`ratio_scale=0.3`) gave the lopsided-lens negative an
asym-defect of only 0.45, close enough to the positive band (0) that
once that single negative was excluded from a leave-one-out round, the
next-closest negative sat at 0.97 and the fitted threshold landed at
their midpoint (~0.49) -- just above the excluded negative's own true
value (0.45), so it was misclassified once tested alone. Shrinking
`ratio_scale` to 0.1 (steepening how fast the defect grows past
`ratio_thresh`) pushed that same negative's defect to ~1.35, comfortably
above where any LOO-refit threshold could land, while leaving every
positive and every other negative's defect unchanged. Lesson generalizes
problem_08's "recheck the threshold constant" note: when a composite
defect combines several branches via max(), check each branch's OWN
margin against the *next-closest* example on its own side once its
nearest neighbor is hypothetically removed -- not just the margin in the
full dataset -- since that's exactly what leave-one-out exposes.

## problem_47: closed loop with a tail ("fish") vs. open arcs, plain circle, multi-lobe leaf
Rule: positives are a single thin closed loop (a lens/leaf outline) with an
extra short tail/fin stroke crossing out through one of its cusps (a
"fish" shape). Negatives fail differently: no loop at all (a single open
arc), a loop but no tail and much fatter (a plain circle, whose enclosed
area is huge relative to its own stroke length), or a closed shape with
more than one lobe (two leaves joined, i.e. 2+ enclosed regions).

Added generic reusable predicates:
- `_enclosed_holes` (helper): labels the background and returns the pixel
  counts of components that don't touch the panel border -- i.e. the areas
  a stroke encloses. 0 for any open curve, 1 for a single closed loop
  (regardless of extra tails sticking outside it, since those stay
  background-connected to the outer region), 2+ for multi-lobe shapes.
  Reusable any time a rule cares about loop-closure or lobe count.
- `p_num_enclosed_holes`, `p_hole_to_ink_ratio`: raw scalar wrappers around
  `_enclosed_holes` (count, and largest-hole-area / stroke-ink-count -- the
  latter distinguishes a thin lens's modest hole from a fat circle's much
  bigger one at the same stroke length).
- `p_0_a_loop_tail_defect`: the actual composite used by the solved rule,
  OR-ing three failure modes (no loop / multi-lobe / fat-ring) via max(),
  same pattern as `p_0_asym_taper_or_blunt_or_chunky_defect`.

### Lesson: tie-break collisions with EXISTING predicates, not just target-parameter siblings
The `_arc_defect_score_arc120` naming note (problem_00) warned about ties
between two variants of the *same new* predicate. This problem showed the
same failure mode against a totally unrelated *pre-existing* predicate:
`p_0_blunt_tip_defect` isn't semantically related to loops at all, but on
several leave-one-out folds it coincidentally also reached 0 training
error on the remaining 10 panels, tied on cost, and won the describe()-
string tie-break purely because "blunt" < "loop_tail" alphabetically --
causing it to be used (and to misclassify) on the held-out pair instead of
the actually-correct predicate. Fix was the same trick: rename with an
early-sorting infix (`p_0_a_...`) so ties resolve in the correct
predicate's favor. Takeaway: when leave-one-out heldout accuracy is poor
despite train=1.0 and a big apparent margin, check for *which* predicate
each failing fold actually selected (see the per-fold `select_rule` replay
snippet) before assuming the predicate itself is wrong -- it may just be
losing a naming tie-break against something in the existing library.

## problem_12: clean 3-fold triangle pinwheel vs near-misses
Positives: three congruent triangles meeting apex-to-apex at one hub, spread
as a tidy windmill. Negatives: single block/zigzag arrows (no hub symmetry)
plus two decoy triangle-clusters -- one with blades so overlapping/coincident
the figure is *over*-symmetric, one with a blade misarranged so the 3-fold
pattern is broken. All three hole-count / ray-count / endpoint heuristics
failed because the two decoys have 3 triangular holes and 6 rays just like
the positives.

### New predicate `p_0_rot3_windmill_defect` (+ helper `_rot_iou_about`)
Key measurement: IoU of the filled shape with a copy rotated 120 deg about
the DENSEST point (the hub, via existing `_densest_point`), not the centroid.
A clean 3-blade windmill lands at IoU ~0.5 (neighbours overlap only partly);
arrows ~0; over-coincident blades ~0.9; broken pattern ~0.33. So the signal
is a BAND, not a threshold. Encoded as |IoU_120 - target| so a one-sided
`<=` atom works. `_rot_iou_about` generalizes `p_180_rotational_self_iou`
from centroid+180 to arbitrary-center+arbitrary-angle.

### Lesson: two-sided band -> deviation-from-target turns it one-sided
When positives occupy a middle band and negatives fall on BOTH sides
(here IoU low for arrows/broken, high for coincident), a raw threshold
can't separate. Return |value - target| and the harness's `<=` atom
captures the band. Reused the same "|x - target|" shape as the arc-span
and fan-quad-ratio predicates.

### Lesson: pick `target` from the LOO-robust interval, not the pos mean
The pos IoU band was [0.498, 0.541] but target=0.51 (its center) only gave
heldout 0.917: the nearest negative (broken decoy, IoU 0.33) sat close
enough that leave-one-out threshold jitter misclassified it. Sweeping target
through the harness's own `verify()` showed a solid solved=True plateau for
target in [0.53, 0.60]; picking 0.56 (plateau center) maximizes LOO margin
on both sides. Lesson: when a big-margin predicate still fails LOO, sweep
its constant against `verify()` and choose the middle of the solved
interval rather than the data's raw mean. Same early-sort naming trick
(`p_0_...`) was again needed so it wins describe() ties over `p_arc_*`.

## problem_13: circle-with-one-flat-chord blob vs open arcs / lens+tail / fish
Positives: a near-circular closed shape where one small arc has been
replaced by a straight chord (looks like a circle with a single flat
side, or a "D"). Negatives: plain open arcs (no closure at all), a
lens/leaf with an extra tail or triangle sticking out, and a fish curve
(multiple crossings). No new predicate needed -- `p_180_rotational_self_iou`
(from problem_02) already separates perfectly: a blobby near-circle is
close to its own 180-degree rotation even with one flat chord, while the
open arcs and tailed/crossing shapes are not. Good reminder to run the
full existing library against new panels before reaching for a new
measurement -- `_solidity` alone would also have worked (pos ~1.013-1.015
vs neg ~<=0.939) but was redundant once the search found a zero-new-cost
solve.

## problem_14: elongated shapes OR non-pinching shapes vs. compact-and-pinched shapes
Visually the two sides are wildly heterogeneous by topology -- both sides mix
single closed polygons, single shapes with a concave notch, open (unclosed)
paths, and pairs of sub-shapes touching at a shared vertex. Extensive manual
inspection of topology (open/closed, convex/concave, number of touching
sub-shapes, self-crossing-vs-touching, arc-vs-line content) found NO clean
separator: every topological category present on the positive side had a
near-identical counterpart on the negative side (e.g. pos_4 and neg_0 are
both a single closed "pac-man with a deep notch" shape, differing only in
degree; pos_2 and pos_5 are both pure straight-line polygons, matching
several negatives' pure-polygon construction). This is the clearest instance
yet of the recurring lesson (see problem_02's log entry): near-identical
gestalt/topology across the two sides means the rule is a continuous
measurement, not a structural/topological one.

The actual separator turned out to be a disjunction of two continuous
measurements, each covering the panels the other misses:
- `p_elongation` (new: major/minor PCA-extent ratio of the raw ink) is high
  (>=1.87) for pos_0, pos_2, pos_4, pos_5 -- shapes that are visually
  stretched/wedge-like regardless of their notch/touch/convexity details --
  and stays <=1.68 for every negative.
- `p_self_proximity_ratio` (existing, from problem_04) is high (>=0.084) for
  the two positives elongation misses (pos_1, pos_3 -- both compact/round-ish
  shapes whose outline never comes close to touching itself), and low
  (<=0.013) for every negative.
No single positive needs both signals, and no negative satisfies either one:
elongation and non-pinching cover two disjoint subsets of the positive side,
so `min()` of the two features' shortfalls-below-fixed-threshold (0 if either
condition holds, positive only if a shape is both compact AND pinched) is a
clean single-scalar OR, same pattern as `p_open_or_symmetric_defect` and
`p_pinch_notch_defect`.

Added:
- `p_elongation`: major/minor PCA-extent ratio (`_pca_extents`) of a panel's
  raw ink pixels. Simple, generic, reusable anytime "how stretched/thin is
  this shape, independent of rotation" comes up -- surprisingly wasn't
  already in the library despite `_pca_extents` existing since problem_00.
- `p_elongated_or_unpinched_defect`: the OR-via-min combinator described
  above. The specific thresholds (elong>=1.7756, selfprox>=0.0485) are
  fixed constants picked from this problem's own gap structure (neg-max to
  pos-min midpoint on each feature) -- baking them into the predicate
  function (rather than leaving them for the harness to fit) is what makes
  the OR expressible as a single atom at all, and keeps the rule LOO-robust
  since removing any one held-out panel doesn't change these constants.

Explored and discarded (added no value once elongation+self-proximity was
found, so removed from the library to keep marginal cost down): an RDP
(Ramer-Douglas-Peucker) polyline simplifier to split a traced curve into
corner-to-corner edges, a per-edge straight-vs-arc classifier from that
split, and a sliding-window max-corner-turn-angle measurement. All three
were built while testing (and ultimately rejecting) the hypothesis "positive
shapes contain a curved/arc segment" -- disproved by pos_2/pos_5 (pure
straight-line polygons, positive) and neg_0 (majority-arc pac-man, negative).
The RDP splitter is a generically useful technique (cleaner than
`_order_curve` + fixed-window turning angle for corner-finding, since it
looks at whole-run deviation rather than a single window straddling a
corner) and may be worth re-adding if a future problem's near-miss actually
hinges on per-edge straight-vs-arc classification.

### Lesson: when every topological category on one side has a near-twin on the other, stop looking for structure and start looking for a disjunction of two continuous cuts
Plotting several candidate continuous features side by side (not just one at
a time) revealed that the positive side split cleanly into two non-
overlapping groups under two different existing/simple measurements, each
group failing the OTHER measurement. Single-feature threshold search (what
`bongard_try.py`'s own atom search does) can't find this because no single
feature separates all 12 -- it required manually noticing the complementary
gap structure across two feature columns at once, then hand-writing the
OR-combinator predicate so the harness's single-atom-threshold search could
use it as one atom.

## problem_15: quadrilateral + small triangle joined at a point, big/small area ratio
Positives: a square (or similar quadrilateral) with a small triangle
attached at one shared vertex, sticking out to the side -- consistently the
square's enclosed area is ~19-20x the triangle's. Negatives are near-misses
that share the "two closed loops joined at a point" gestalt but fail on the
size ratio: two comparable-sized shapes touching at a point (ratio ~1),
several triangles only (no quadrilateral), or a big shape + a triangle whose
area ratio is only ~10-14x (still visibly "small triangle on a big shape" to
the eye, but the ratio sits well below the ~19-20x the positives all share).

Added:
- `_enclosed_hole_areas`: labels the background regions enclosed by a
  (1px-dilated, to seal corner gaps) line drawing, largest first, ignoring
  regions touching the image border. Reusable whenever a shape is made of
  multiple closed loops joined at a point/edge and their individual sizes
  (not just total filled area, cf. `_filled_mask`) matter.
- `p_00_hole_pair_area_ratio`: ratio of the two largest such hole areas.
  Robust, wide-margin separator here (positives ~19-20, closest negative
  13.8).

### Lesson: tie-breaking-by-name can hand the rule to a fragile coincidental separator
Even though `p_00_hole_pair_area_ratio` alone gets perfect leave-one-out
accuracy, adding it under a "natural" name (`p_two_largest_hole_area_ratio`)
left the problem unsolved (heldout=0.778): two *unrelated* existing
predicates (`p_circle_fit_residual`, `p_0_blunt_tip_defect`) each happened to
also reach zero training error on this problem's data via a razor-thin,
non-robust margin (gaps of ~0.002-0.008 between the closest pos/neg values).
`select_rule`'s tie-break for equal training-error/cost atoms is lexical
name order, with no notion of margin size -- so whichever coincidental
near-separator's name sorted first kept winning individual leave-one-out
folds, including the folds where its thin margin actually gets the held-out
panel wrong. Renaming the new predicate to `p_00_...` (sorts before every
other name in the file, including all `p_0_*` and `p_circle_*` ones) made it
win every tie instead, and since it's genuinely robust across all folds,
heldout went to 1.000. General pattern: when a new predicate has a wide,
robust separating margin but the harness still reports heldout<1 despite
train=1, suspect a same-training-error rival predicate winning individual
LOO folds by alphabetical accident (diagnose by rerunning `select_rule` on
each held-out pair with the current values matrix and checking rule.describe());
if so, prefix the new predicate's name so it sorts first (`p_00_...` is
currently the earliest-sorting prefix in this file) rather than trying to
out-engineer the existing rival's margin.

## problem_16: single-lobe twisted quadrilateral, one hole + compact
Positives: a single self-crossing outline forming one asymmetric twisted
lobe (like a sheared/kinked ribbon) -- exactly one enclosed background
region, and compact (PCA major/minor extent ratio near 1.0-1.03).
Negatives split into two near-miss groups: (a) a pinwheel of 3+ separately
overlapping triangles -- same "single connected ink component" gestalt but
3 distinct enclosed holes instead of 1 (`_enclosed_hole_areas`); and (b) a
single-hole shape that is NOT compact -- a symmetric hourglass/bowtie or a
zigzag arrow, both with elongation ~1.23-1.26 vs positives' ~1.0-1.03.
Neither "hole count" nor "elongation" alone separates all 12; each only
catches one of the two negative groups.

Added `p_00_single_hole_compact_defect`: AND-via-max of `len(_enclosed_hole_areas)-1`
and `p_elongation(panel) - 1.15` shortfalls -- the AND counterpart to the
existing OR-via-min pattern (`p_elongated_or_unpinched_defect`). Zero only
when a shape has exactly one hole AND is compact; positive if either fails.
Reused `_enclosed_hole_areas` and `p_elongation` as-is, no other new code.
Prefixed `p_00_` per the tie-break lesson above (multiple existing predicates
could otherwise coincidentally tie on zero training error and win LOO folds
by alphabetical luck).

### Pattern: AND-via-max mirrors OR-via-min
When two existing scalar features each cleanly separate one of two
complementary negative subgroups from all positives, and positives require
BOTH conditions to hold, combine their fixed-threshold shortfalls with
`max()` (defect nonzero if either fails) rather than writing two separate
atoms -- the harness only searches single-atom threshold rules. This is the
mirror image of the earlier OR-via-min combinator: use `min()` when
positives need EITHER condition, `max()` when positives need BOTH.
