# Semantic Failure Case Notes

These notes summarize the first typed semantic-track repair for the reported
proxy-collapse cases. They are not claims that all problems are solved; they
document how failures are now classified.

## Before

The semantic runner could accept or score candidates where a rich English
claim was compiled into an unrelated scalar:

- `bird3`: wavy / bird-like -> `symmetry_residual`
- `three_mismatch_triangles2`: four blades / pinwheel -> object count, bbox
  aspect, fill, crude symmetry
- `mismatch_triangle_square6`: quadrilateral + triangle -> bbox measurement
  of largest component
- `two_intersect_circles`: two circular lobes intersecting -> `closure_ratio`
- `symmetric_fish`: accepted through area disparity rather than fish-like
  structure

## After

The semantic compiler inspects the semantic requirements and executable graph.
If a rich term is named, the graph must contain the required primitive witness
path and the final score must depend on it.

Expected diagnostics:

- `bird3`: `MISSING_LEG` unless the cone contains part graph, attachment,
  curve/arc, and symmetry evidence.
- `three_mismatch_triangles2`: `MISSING_LEG` unless the cone contains a
  part graph and radial-arrangement witness with four parts.
- `mismatch_triangle_square6`: `MISSING_LEG` unless the cone contains
  triangle witness, quadrilateral witness, and contact/attachment witness.
- `two_intersect_circles`: `MISSING_LEG` unless the cone contains circle-pair
  and circle-intersection witnesses.
- `symmetric_fish`: `MISSING_LEG` unless the cone contains fish-like
  primitive structure such as body/tail parts, contact, and symmetry.

## What Still Needs Work

The initial geometry legs are deliberately simple. They provide typed witness
paths and useful diagnostics, not production-quality recognition. The next
work is improving the primitive legs while preserving the invariant that rich
terms cannot be discharged by scalar proxies.
