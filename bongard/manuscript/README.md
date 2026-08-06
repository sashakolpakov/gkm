# Bongard manuscript

`free_energy_abstraction.tex` is the single current research manuscript. Its
title is *Growing Verified Visual Predicates for Bongard-LOGO: A
Complete-Corpus Protocol*.

The paper specifies the complete official corpus boundary, visual and soft
measurement contracts, positive-only typed predicate IR, growing-leg admission,
sealed headless-proposer evaluation, and the limits of what replay and formal
verification establish. It intentionally reports no complete-corpus benchmark
accuracy yet. Exploratory legacy runs are not presented as protocol results.

Pure Python defines the canonical predicate, evaluation, and replay semantics.
Lean is an optional, removable cross-check; neither benchmark execution nor the
meaning of an artifact depends on it. The current PURE support-prototype
baseline failed development calibration: coordinate-wise interval boxes lost
correlations between preprocessing scenarios, one centroid per side mismatched
multimodal classes, the one-group restriction excluded cross-group concepts,
and the neutral raster features lacked semantic and relational vision. This is
a representation diagnosis, not a complete-corpus benchmark result.

From the repository root:

```bash
make -C bongard/manuscript
```

This builds `free_energy_abstraction.pdf`. `semantic_cones.tex` is only a
compatibility wrapper around the canonical source, not an independent
manuscript. To verify that entry point:

```bash
make -C bongard/manuscript compatibility
```

`references.bib` is the shared bibliography. Generated LaTeX files are build
products, not independent manuscript sources. Pre-rewrite manuscript material
and exploratory reports are preserved at the annotated Git tag
`pre-bongard-complete-rewrite-20260805`.
