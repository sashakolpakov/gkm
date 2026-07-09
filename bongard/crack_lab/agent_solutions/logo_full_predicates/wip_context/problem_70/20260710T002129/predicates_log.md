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
