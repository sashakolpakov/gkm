Related Work and Claim Boundary
===============================

The project does not claim that predicate invention, visual analogy, program
induction, or description-length selection is new. Relevant traditions
include:

* Bongard problems and modern visual concept-learning datasets, including
  Bongard-LOGO, Bongard-HOI, and Bongard-OpenWorld;
* inductive logic programming, predicate invention, and meta-interpretive
  learning;
* program induction and library learning, including compression-based systems
  such as DreamCoder;
* minimum description length, Bayesian model selection, and free-energy
  structure/function tradeoffs;
* curriculum generation and PowerPlay-style searches for the next unsolved
  task;
* hierarchical reinforcement learning, skills, options, and reusable program
  fragments;
* calibrated machine perception, selective prediction, conformal/interval
  reasoning, and out-of-distribution evaluation;
* proof assistants and proof-producing program synthesis.

The narrow question here is whether a visual concept learner can accumulate a
typed, reusable observation library under explicit novelty cost and
archive-preserving admission, while maintaining a strict information boundary
between labeled support and unseen queries.

That question has two irreducible layers. Vision produces empirical witnesses
such as shape, topology, angle, contact, or a calibrated soft description.
The closed predicate layer composes those witnesses and can be checked
mechanically. Its reference evaluator and cold replay are pure Python, and its
serialized typed contract does not depend on a proof assistant. The Lean
backend is an optional, removable cross-check. Either implementation is
conditional on the perceptual inputs; neither proves their correspondence to
pixels.

The current PURE support-prototype baseline failed development calibration.
Coordinate-wise interval boxes erased correlations between preprocessing
scenarios; one centroid per side could not represent multimodal classes; the
one-group proposal restriction excluded cross-group concepts; and the neutral
raster observables lacked semantic and relational vision. Thus the current
failure is evidence of an inadequate representation, not evidence that
negating the synthesized predicate is a valid solution.

The controlled abstraction-emergence experiment supplied primitive atoms and
tested only the reuse incentive. The action-program adapter bypassed pixels.
Those and the other pre-rewrite pilots do not satisfy the complete-corpus
exposure and freeze/query/reveal protocol and must not be used as evidence for
official visual benchmark performance. Their original code, reports, and
longer bibliography notes are preserved at the annotated Git tag
``pre-bongard-complete-rewrite-20260805``.
