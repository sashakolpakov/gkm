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

## Files

- `visual_witnesses.py`: serializable witness dataclasses.
- `semantic_legs.py`: typed leg contracts and deterministic implementations.
- `semantic_requirements.py`: auditable lexicon mapping semantic terms to
  required witness paths.
- `semantic_compiler.py`: type checking, dependency checking, and
  `MISSING_LEG` enforcement.
- `semantic_verifier.py`: support and image-level LOO verification for typed
  cones.
- `semantic_selection.py`: risk vectors, conditional complexity breakdowns,
  track labels, and Pareto frontier support.
- `cofibrations.py`: practical witness-preserving extension contracts.

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

## Adding a Semantic Requirement

1. Add an entry in `SEMANTIC_REQUIREMENTS`.
2. Include aliases, accepted witness types, primitive required types, and
   missing-leg hints.
3. Do not map rich named concepts to scalar proxy legs.

## Adding a Cofibration

1. Add a `CofibrationSpec` in `cofibrations.py`.
2. Declare source/target types, preserved fields, interface fields, added
   fields, and attachment/projection legs.
3. Add a preservation test using `verify_cofibration`.

## Example MISSING_LEG

For a hypothesis saying `two intersecting circles` but using only
`closure_ratio`, the semantic compiler reports a structured failure:

```text
MISSING_LEG
semantic term: circle
required: ContourWitness + CircleWitness
available paths terminate at:
- Measurement
- Object
- Panel
- Scene
missing:
- extract_contours
- fit_circle
```

This is a useful result. It means the system identified representation
poverty instead of accepting a short proxy.

## Current Expressible Concepts

Faithful typed paths now exist for:

- triangle via contour -> polygon -> triangle witness
- quadrilateral via contour -> polygon -> quadrilateral witness
- circle via contour -> circle witness
- circle pair/intersection via circle-pair and intersection witnesses
- radial arrangement via part graph -> radial arrangement witness
- contact/attachment via part graph -> contact witness
- skeleton endpoint/branch/cycle counts

Composite names such as bird-like, fish-like, lamp-like, and pinwheel-like
must be built from lower-level primitive witnesses. There is no default
black-box `bird_like(image)` leg.

## Kolmogorov Selection

Semantic admissibility is only a gate. Among admissible implementations,
selection still uses risk plus conditional complexity:

```text
F = R + lambda C(M | L)
```

`semantic_selection.py` records the full risk vector and complexity breakdown
instead of hiding naturality, counterfactual, parser, archive, or complexity
costs in a single opaque score.
