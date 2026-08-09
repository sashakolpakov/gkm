# Panel-primary soft predicates

## What the exact-unused drill established

The 2026-08-09 three-task TRAIN drill reached no predicate or query decision.
Its outcomes were two proposer gaps and one deterministic pre-model capacity
failure.  All three came from making hard object segmentation a prerequisite
for vision:

- In the `bd` task, one visibly nonempty support panel froze with zero object
  proposals.  No object-local citation could cover all six panels.
- In the `hd` task, one visibly nonempty support panel also froze with zero
  object proposals.  The proposer nevertheless described the useful visual
  contrast as one broad smooth sweep versus several strong direction changes.
- In the `ff` task, one panel fragmented into sixteen proposals.  The fixed
  per-object atlas made a support sheet larger than the four-megabyte transport
  limit, so no model call was made.

The hard frontend is therefore useful optional evidence, but it is not a sound
gate for whether the vision model may describe a complete panel.

## Proposed replacement pipeline

The intended primary lane is:

```text
exact panel pixels
-> frozen affirmative panel atoms proposed from all support panels
-> complete panel-by-atom observation table under one frozen protocol
-> verified calibration receipt for that exact observer contract
-> scientific four-disposition projection
-> deterministic positive-conjunction version space
-> frozen Python predicate
-> query observation under the identical protocol
-> model-free, tamper-detecting replay
```

This document is a design and the new Python module is only a closed semantic
scaffold.  It does **not** yet implement the vision proposer/observer, a sealed
calibration receipt, freeze-before-query chronology, pixel custody, replay, or
benchmark authority.  Those are blockers to calling this lane a scientific
benchmark.  Supplying an arbitrary digest is not a calibration receipt and
cannot enable `present`, `certified_absent`, or support survivors.

Typed geometry, object crops, and part atlases may accompany a panel as
additional evidence.  They may refine or explain an atom, but an empty,
indeterminate, expensive, or oversized geometry result cannot erase the raw
panel or make a panel atom structurally impossible.

## Atom contract

An atom is positive visible prose, such as `bird-like silhouette`, `several
oblique corners`, or `one broad smooth sweep`.  Its identity binds:

- exact prose bytes and grammatical scope;
- optional affirmative witness clauses;
- proposer artifact and orientation of origin;
- the observer model, prompt, output schema, image presentation, atom order,
  batch composition, and repeat policy.

The scaffold can bind declared digests for these fields, but it does not prove
that an observer call used them.  That requires a sealed call receipt and
custody verifier outside the current module.

Prose is data, never executable source.  Python supplies the only executable
operators.  The initial language contains atoms and positive conjunctions;
there is no `Not`, polarity flip, arbitrary code, threshold search, or
post-query formula change.

## What is and is not rigorous

Python is sufficient to rigorously check formula construction, support
consistency, four-valued evaluation and, once their receipt layers exist,
provenance, chronology, freezing, and replay.  The current scaffold implements
the closed formula/observation semantics only; it does not implement those
receipt layers.  Lean may remain as an optional checker/export target, but it
is not part of predicate identity or benchmark semantics and must be removable
without changing a decision.

Neither Python nor Lean proves from pixels alone that a phrase such as
`bird-like` is visually true.  That claim comes from the frozen vision
measurement protocol.  Its error properties require empirical calibration.

The eventual scientific lane retains four dispositions:

- `present`: calibrated evidence supports the atom;
- `certified_absent`: calibrated evidence supports its absence;
- `indeterminate`: evidence is weak, conflicting, uncalibrated, or outside a
  declared measurement/resource domain;
- `error`: custody, protocol, transport, or verification failed.

Two repeated answers from the same model and prompt measure repeatability, not
independence.  The present scaffold has no verified calibration authority.
Consequently both model-agreed `present` and model-agreed `mismatch` project to
`indeterminate`; observer/transport failures project to `error`.  The strings
`repeated_present`, `repeated_mismatch`, `repeated_indeterminate`, and
`disagreement` are operational diagnostics only.  They are not dispositions
and cannot create scientific survivors.

Adding scientific states later requires a typed calibration artifact and a
verifier that binds its population, labels, observer contract, acceptance
criteria, error bounds, and validity period to the exact run.  A free-form
manifest digest or Boolean switch is deliberately insufficient.

## Support/query identity

Support and query panels must use the same deployed instrument:

- the same selected atom vector, in the same order;
- the same prompt, model, schema, image naming, crop/panel presentation, and
  batch context;
- the same existential or panel-global semantics;
- the same repeat and disposition projection.

Observing the full proposed vocabulary on support but only the selected atoms
on query is a protocol change and is not allowed.  Proposer citations are
provenance, not a different support truth semantics.

## EOD engineering drill versus scientific benchmark

The EOD headless-Codex drill may exercise panel presentation, atom quality,
schema compliance, repeatability, disagreement, capacity, cost, and latency.
It must be labelled `engineering_diagnostic`.  It may report operational
consensus rates, task accuracy/coverage under the frozen operational rule, and
explicitly named `engineering_only` support survivors.  Those numbers measure
the deployed Codex instrument; they are not scientific visual-truth accuracy,
scientifically verified predicates, or scientific support survivors.

### Separate operational decision path

The scaffold now defines a second version space solely to make the EOD drill
executable while calibration is missing.  This path consumes
`PanelSoftOperationalConsensus` directly and never changes or aliases the
scientific `Disposition` projection.  Its positive `all_of` interpreter is:

- `match` only when every selected atom is `repeated_present`;
- `error` when any selected atom is `error`;
- `indeterminate` when any selected atom is disagreement or
  `repeated_indeterminate` (after the error check);
- otherwise `nonmatch` when at least one selected atom is
  `repeated_mismatch`.

An engineering formula survives only when it is `match` on all six supports
from its native orientation and `nonmatch` on all six contrast supports.  The
language remains affirmative atoms and positive conjunctions only.  A reversed
candidate is not rescued by `Not`, a polarity flip, or treating uncertainty as
nonmatch.

If both orientations have survivors, the selected engineering predicate pair
contains exactly one native survivor per orientation.  Selection is fixed as
fewest atoms followed by formula digest, and the content-addressed pair binds
the complete engineering version-space digest.  This is selection for an
engineering drill, not model ranking and not scientific synthesis.

For one query panel, both selected formulas must be evaluated under the exact
same vocabulary and observer contract.  A side is emitted only from the full
two-sided witness:

```text
side0 = side0 formula match AND side1 formula nonmatch
side1 = side1 formula match AND side0 formula nonmatch
```

A nonmatch by itself never predicts the opposite side.  Two matches, two
nonmatches, disagreement, or indeterminacy abstain; any error produces an error.
Every operational artifact and enum is labelled `engineering_only`,
`uncalibrated`, not scientific evidence, and not benchmark-authoritative.
Freeze-before-query chronology and sealed call receipts remain external
blockers and are recorded as unverified on these artifacts.

Python remains the canonical executable semantics for both paths.  Lean is an
optional cross-check/export target and can be removed without changing formula
identity, survivor selection, or query decisions.

A `scientific_benchmark` label remains blocked until all of the following are
implemented and verified:

- sealed proposer and observer call receipts tied to exact input pixels;
- a non-forgeable calibration authority for both presence and absence under
  the exact observer contract;
- formula freeze before any query pixels are created or released;
- identical support/query observation context, enforced from receipts rather
  than asserted by caller-provided digests;
- model-free, tamper-detecting replay with an explicit benchmark authority.

## Synthesis and reporting

Python enumerates the preregistered positive atom/conjunction inventory.  A
scientific survivor requires every target support to be `present` and every
contrast support to be `certified_absent`.  The current scaffold therefore
always has zero scientific survivors for non-error vision votes.  Separately,
the operational path may produce engineering-only survivors using the rule
above.  If no scientific survivor exists, observer errors and repeat
disagreement are reported before the calibration-authority gap.  Nothing in
either path is rescued by negation.

Every run reports atom count, conjunction count, survivor count, observer
disagreement, indeterminate/error counts, any candidate cap, and ranker
baselines (first, shortest, random, and model-ranked) once scientific survivors
exist.  An EOD engineering drill instead reports its diagnostic measurements
and the missing-authority gap explicitly.

## Immediate implementation order

1. Add the direct whole-panel atom proposer and observer lane using raw panel
   custody and transport primitives already present in the panel-rubric code.
   Do not reuse its paired A-versus-B rubric or decision rule.
2. Keep the closed backend-neutral atom/conjunction IR and Python interpreter;
   its current vote projection is intentionally indeterminate/error only.
3. Build sealed observer receipts and complete support observation tables, then
   enforce the identical selected-vector protocol on query from those receipts.
4. Define and verify a typed calibration artifact for both presence and
   absence.  Do not add a digest-only enable switch.
5. Implement freeze-before-query custody and model-free replay authority.
6. Retain object/part geometry as optional typed evidence, with bounded
   preprocessing and paginated atlases.
7. Run explicitly labelled engineering diagnostics on exact-unused TRAIN while
   the authority blockers remain.  Promote to scientific benchmarking only
   after the gates above pass, then remove superseded paired-rubric
   and mandatory-anchor paths only after the replacement passes replay and
   equivalence gates.
