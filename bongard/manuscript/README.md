# Bongard manuscript

`free_energy_abstraction.tex` is the single current research manuscript. Its
title is *Growing Verified Visual Predicates for Bongard-LOGO: A
Complete-Corpus Protocol*.

The paper specifies the complete official corpus boundary, visual and soft
measurement contracts, positive-only typed predicate IR, growing-leg admission,
sealed headless-proposer evaluation, and the limits of what replay and formal
verification establish. It intentionally reports no complete-corpus benchmark
accuracy yet. Exploratory legacy runs are not presented as protocol results.

From the repository root:

```bash
make -C bongard/manuscript
```

This builds `free_energy_abstraction.pdf`. `semantic_cones.tex` is retained only
as a compatibility wrapper around the canonical source. To verify that old
entry point:

```bash
make -C bongard/manuscript compatibility
```

`references.bib` remains the shared bibliography. Generated LaTeX files are
build products, not independent manuscript sources.
