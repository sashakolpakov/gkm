# Predicate log

## problem_00 -- solved
Rule: `p_is_regular_circular_arc >= 0.5` (single atom, train=1.0, heldout=1.0)

Panels are single open strokes. Positives are all clean arcs of one circle
spanning ~107-111 degrees. Negatives are near-misses along two independent
axes: bad circle fit (corners/kinks: neg_4; multi-arc/blob shapes: neg_0,
neg_5; wobbly double-curvature: neg_2) OR good circle fit but wrong span
(neg_1 ~93 deg, neg_3 ~80 deg -- neg_3 in particular is geometrically a
*perfect* arc, just the wrong angle, so no single raw measurement
(circle-fit residual alone, or radius alone) separates it from the
positives -- both directions of error are needed).

Recurring pattern worth remembering: the MDL rule search prices every
*atom* at a fixed CALL_COST+BINDING_COST regardless of the predicate's
internal complexity, and prefers fewer atoms even at some training error
(lambda=0.1 rule cost vs error tradeoff). So whenever separating the two
sides genuinely needs a conjunction/bracket on the same underlying
measurement (e.g. lo < x < hi), fold that logic into one composite
`p_*` boolean predicate rather than exposing the raw measurement(s) as
separate `p_*` atoms -- otherwise leave-one-out folds that drop the
"deceptive" near-miss panel let the harness fall back to the cheaper,
simpler (but globally wrong) raw atom via a lexical tie-break, and
heldout accuracy drops below 1.0 even though train looked solved.
Keep such raw sub-measurements as private helpers (no `p_` prefix) so
they aren't independently selectable, and only expose the composite.

New reusable building blocks added to `predicates.py`:
- `_fit_circle` / `_circle_fit_residual_ratio`: Kasa algebraic circle fit
  and normalized residual -- general "is this ink a circle/arc" measure.
- `_geodesic_arclen` / `_arc_span_deg`: BFS geodesic path length over
  foreground pixels (robust to stroke thickness) turned into an angular
  span via the fitted radius -- general arc-shape descriptor.
- `p_num_components`: connected-component count, generically useful for
  "how many objects/strokes" questions in future problems.
- `p_is_regular_circular_arc`: composite "well-formed single circular
  arc of plausible span" predicate; reuse directly if a future problem's
  near-miss structure matches (clean arc vs corner/blob/wrong-span).

## problem_01 -- solved
Rule: `p_is_wide_hand_drawn_arc >= 0.5` (single atom, train=1.0, heldout=1.0)

Same family as problem_00 (single open stroke, circle-fit + span
composite) but with a *different* residual and span band, so it needed
a new composite rather than reusing `p_is_regular_circular_arc` as-is:
positives here are wide (~180-205 deg) hand-drawn arcs with a
noticeably worse circle-fit residual (~0.032-0.035) than problem_00's
tight arcs. Negatives fail for four distinct reasons, requiring both
bounds on both measurements: neg_0 is a near-perfect circle (residual
far *below* the positive band, ~0.010 -- positives are visibly more
irregular/hand-drawn than a true circle); neg_3 is a near-full loop
(span 224, above the band); neg_4 is a shallow arc (span 110, below
the band); neg_1/neg_2/neg_5 are self-intersecting multi-loop "leaf"
scribbles (residual 0.24-0.31, far above the band, since two crossing
strokes fit no single circle well).

Added `p_is_wide_hand_drawn_arc` to `predicates.py` as a sibling
composite to `p_is_regular_circular_arc`, reusing the same private
helpers (`_circle_fit_residual_ratio`, `_arc_span_deg`). Recurring
pattern: this "single stroke, circle-fit residual band + angular span
band" template is likely to recur across problems with open curved
strokes -- if a future problem's positive/negative split matches one of
these two bands, reuse directly; if it's a third distinct band, add
another sibling composite rather than parameterizing, since predicates
must stay parameterless.
