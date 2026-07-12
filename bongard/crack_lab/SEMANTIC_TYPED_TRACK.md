# Typed Semantic Track

This directory now has two separate Bongard paths.

- `UNRESTRICTED`: the existing `bongard_arena.py` + `bongard_legs.py`
  predicate-library path. Arbitrary `p_*(panel)` measurements are legal and
  selected by the existing MDL verifier.
- `SEMANTIC-PURE`: typed semantic cones. A candidate must compile through the
  witnesses required by its semantic claim before verifier or MDL selection.
- `HYBRID`: reserved for semantic cones plus explicitly priced residual guards.
  Hybrid results must not be reported as semantic-pure.

## Adaptation Notes

The repository already had the unrestricted control path:

- `bongard_arena.py`: rendering, predicate loading, MDL conjunction search,
  rotated pair holdout, free-energy helper.
- `bongard_legs.py`: proposer loop, promoted/WIP artifact discipline, taint
  checks, marginal complexity admission.
- `bongard_api_agent.py`: API proposer rung with paired semantic text and
  executable predicate code.

The typed semantic implementation is additive. It does not replace the
unrestricted path or promoted artifacts.

## Main Invariant

No semantic concept may be weakened during compilation.

A candidate that says `triangle attached to square` must produce typed
evidence such as:

```text
ContourWitness -> PolygonWitness -> TriangleWitness
ContourWitness -> PolygonWitness -> QuadrilateralWitness
PartGraphWitness -> ContactWitness
```

It may not compile only through:

```text
bbox_aspect
bbox_fill
raw area
closure_ratio
symmetry_residual
```

Those scalar measurements remain legal in the unrestricted track and remain
legal in semantic-pure only when the semantic claim actually names the scalar
property, such as elongated, sparse, open, large, or symmetric.

## Generality Rule

Nothing problem-specific is admissible in the harness. There is no concept
lexicon, no per-concept requirement table, and no hard-coded composite
gluing. The harness knows leg contracts, witness types and general checks;
all semantic novelty (terms, witness demands, gluings) comes from the
proposer and is verified mechanically.

## Witness Honesty

A witness-producing leg must verify the structure it claims. `detect_contact`
returns a ContactWitness only when parts actually meet at a stroke junction;
`detect_intersection` requires a crossing (4+ incident branches); when the
relation is absent the leg raises instead of fabricating evidence. Absence
claims are expressed through counting measurements (`contact_count`,
`intersection_count`, `part_count`, `object_count`, `contour_closedness`)
which return 0 honestly.

## Files

- `visual_witnesses.py`: serializable witness dataclasses.
- `semantic_legs.py`: typed leg contracts and deterministic implementations
  (junction-based part decomposition, path-ordered contours, RDP polygon
  fitting, Kåsa circle fitting).
- `semantic_requirements.py`: general term-coverage audit. No lexicon: a
  declared term must be covered by witness types/legs in the score's
  dependency cone, by a used leg's own `proxy_for` contract, or by a
  declared gluing; otherwise MISSING_LEG with registry-derived suggestions.
- `semantic_compiler.py`: type checking, dependency checking, gluing
  validation, and `MISSING_LEG` enforcement.
- `semantic_verifier.py`: support and image-level LOO verification, executed
  cone-invariance (naturality) checks for declared morphisms, and per-panel
  gluing verification.
- `semantic_selection.py`: risk vectors, conditional complexity breakdowns,
  track labels, and Pareto frontier support.
- `cofibrations.py`: gluing-morphism contracts (see below) — machinery only,
  no concept-specific specs.
- `semantic_artifacts.py`: promoted/WIP artifact discipline and taint scan.
- `run_semantic_cone.py`: verifier-in-the-loop runner (multi-round feedback,
  checkpoints, promotion/WIP snapshots).

## Adding a Witness

1. Add a dataclass to `visual_witnesses.py`.
2. Include geometry, confidence/residual, source IDs, child witnesses, and
   provenance.
3. Keep it trace-serializable via `to_trace`.

## Adding a Leg

1. Implement a deterministic function in `semantic_legs.py`.
2. Register it with a `LegContract` containing domain, codomain, invariances,
   failure modes, complexity cost, and version.
3. Add tests that the leg returns the declared type or fails explicitly.

## Term Coverage (no lexicon)

There is no requirement table to edit. When a hypothesis declares a term,
the compiler audits it mechanically:

1. tokens of the term are matched against witness types and leg names inside
   the score's dependency cone (registry-driven stem match);
2. tokens explicitly named by a used leg's `proxy_for` contract are covered
   (e.g. `closure_ratio` declares `open`/`closed`);
3. tokens carried by a declared gluing (its name, types, attachment leg) are
   covered;
4. number words are threshold content handled by the fitted rule;
5. anything else raises MISSING_LEG with suggestions computed from the
   registry, never from a concept table.

To make a new term expressible, add a general leg/witness whose contract
carries that vocabulary — not a mapping entry.

## Cofibrations are Gluings, and Proposer-Generated

A cofibration here is the canonical morphism `A -> A ⊔_I P`: a source
witness glued to a patch along an interface. It is verified up to the glue
map — a consistent renaming of witness IDs plus numeric tolerance — never by
field-for-field inclusion, and projections recover the source up to that
identification, never by `==` on the nose.

Specs are NOT hard-coded per concept in the library. The proposer emits
them inside its cone IR (`cofibrations` field), binding `source_node` and
`target_node` to diagram nodes. The compiler checks the declared nodes,
types and attachment leg (a missing attachment leg is a MISSING_LEG demand),
and the verifier runs `verify_cofibration` on the actual witness values of
every panel. Hard-coded specs are allowed only as unit-test fixtures.

## Cone Invariance is Executed

Declared `preservation_morphisms` are not decoration. Morphisms with an
exact pixel action (translate, rotate, reflect) are applied to every panel
and the cone's decision must be invariant; violations are counted as
`naturality_errors` and reject the cone. Morphisms without an exact action
on 1-px strokes (e.g. uniform_scale) are reported in `unchecked_morphisms`
instead of being silently passed.

## Example MISSING_LEG

For a hypothesis saying `two intersecting circles` but using only
`closure_ratio`, the compiler's coverage audit reports a structured failure
with registry-derived suggestions:

```text
MISSING_LEG
semantic term: two circles
required: CircleIntersectionWitness + CirclePairWitness + CircleWitness
available paths terminate at:
- Measurement
- Object
- Panel
- Scene
missing:
- circle_pair_intersection
- circle_residual
- fit_circle
- fit_multiple_circles
```

This is a useful result. It means the system identified representation
poverty instead of accepting a short proxy.

## Current Expressible Structure

Faithful typed paths now exist for:

- polygon side counts via path-ordered contour -> RDP polygon witness
  (triangle/quadrilateral classification refuses wrong side counts)
- circle and arc via Kåsa fit (an open arc honestly refuses `fit_circle`)
- open/closed via contour topology (`contour_closedness`)
- part decomposition at real stroke junctions, with honest
  contact/attachment (3-branch junctions) and intersection (4+ branches)
- radial arrangement with measured angular/radial uniformity
- skeleton endpoint/branch/cycle counts

Composite names (any concept with no matching registry structure) must be
assembled from primitive witnesses plus proposer-declared gluings. There is
no default black-box concept leg and no concept lexicon anywhere in the
harness.

## Runner Discipline

`run_semantic_cone.py` gives the proposer up to `--rounds`
verifier-in-the-loop turns per problem (structured tool output, so malformed
JSON cannot kill a run). Each round's compile errors, MISSING_LEG
structures, per-panel score tables, misclassified panels and invariance
violations are fed back mechanically. Mirroring the unrestricted track:

- solved problems are promoted into
  `agent_solutions/<tag>_semantic/` (`checkpoint.json`,
  `promoted_cones.json`, harness-only `results.json`, `README.md`),
  gated on a taint scan of the run workspace;
- failed attempts are snapshotted append-only under
  `wip_context/<opaque_id>/<timestamp>/` and never admitted;
- ground-truth concept names never enter the run workspace;
- replay is the verifier: re-running `verify_hypothesis` on promoted cone IR
  must reproduce every verdict.

## Kolmogorov Selection

Semantic admissibility is only a gate. Among admissible implementations,
selection still uses risk plus conditional complexity:

```text
F = R + lambda C(M | L)
```

`semantic_selection.py` records the full risk vector and complexity breakdown
instead of hiding naturality, counterfactual, parser, archive, or complexity
costs in a single opaque score.
