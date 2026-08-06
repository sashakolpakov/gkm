# Bongard manuscript

[`free_energy_abstraction.tex`](free_energy_abstraction.tex) is the sole
current manuscript: *Operational Visual Predicates for Bongard-LOGO: A
Python-Authoritative Protocol*.

The paper describes the implemented visual-semantic boundary, not a successful
benchmark result. Its central claims are deliberately narrow:

- Python alone defines predicate execution, calibration, replay, benchmark
  decisions, and scientific artifact IDs. Lean or another proof checker may
  only consume a frozen artifact as a detached optional sidecar; deleting it
  cannot change a result, decision, or ID.
- The direct catalog contains exactly ten positive count families, with target
  counts 1 through 8. Angle bands, complete point-contact signatures, and
  persistent object/part identity are not implemented direct predicates.
- Soft prose such as “bird-like” is a frozen operational observation. It is
  not a proof about the pixels and becomes calibrated only after a valid
  development fit.
- A1 failed before scoring. Its receipt is
  `sha256:9aa247d953204bb12c06a09af6c081c47ae884be8e9c642a9a2bb6d587ba40cb`
  and its terminal failure is
  `sha256:a130d9e608c38581d34043d4d9c071f93483026592ec9c27a406dbad46d65b83`.
  All 48 proposer calls succeeded: 37 emitted accepted soft claims, 10 were
  direct-only, and 1 was rejected by the typed parser. All 37 scorer calls were
  transport errors. There were zero scores, labels remained withheld, and no
  calibration, semantic accuracy, or negation evidence was produced.
- A2 was a distinct repaired-protocol experiment, not an A1 retry. It used
  protocol
  `sha256:2d9261c763d3f9242ffc7cf42d773f54aa1a51f29b610e10b75c9ae59dea81ca`,
  fresh seed
  `eb031fe199b7d7553444d29cd213663c8afaf99d9b9cccec896f862f445a40b1`,
  and durable successor
  `sha256:9b7cb7ee7d759e899f5194d115a8bd20ebf8e078397a64de8f4b32e6805b1ce8`.
  A concurrent agent edit changed `typed_visual_proposal.py` after freeze, so
  A2 was invalidated by live source mutation. It wrote no Stage-A terminal
  artifact. Its incident file digest is
  `sha256:4ace426bafbc051f2ad620dd8cdb3742a365b43503c673a9acc462665d47ccd4`.
  Process output showed 48 proposer and 34 scorer launches only; outputs were
  lost, labels were not revealed, and no semantic inference is valid. The same
  cohort may not be rerun.
- Stage B is unauthorized by both A1 and A2. The completed semantic-capacity
  audit leaves exactly 24 BD + 0 constituent-disjoint HD = 24 DRILL units, so
  the old 48-task design is impossible. Against the complete live ledger, DEV
  has 16 BD + 0 HD units. The default 24-task Stage B fails before pixels, and
  a 16-task BD-only pilot cannot meet its 24-cluster minimum.
- The earlier 28-unit upper bound failed to project complete-A2 exposure into
  the HD constituent-token exclusion set; every remaining DRILL HD pair
  intersects that set. This is not corpus exhaustion: 10,069 of 10,200
  train/validation task IDs remain exact-unused after A2. The 24-unit ceiling
  comes from applying a
  strict constituent-independence rule to calibration. A future calibration
  frame must exploit the larger training population with explicit dependence
  accounting while retaining a stricter evaluation holdout.
- Before its seed or model output exists, A3 fixes eight clusters per score bin.
  Its 90% two-bin Hoeffding radius is 0.480161, the smallest bin size that can
  possibly decide against the unchanged 0.5 boundary. It remains descriptive.
- New receipts bind the complete executable Bongard Python source boundary;
  post-exposure source drift
  now persists a typed failed receipt with labels withheld. Exact-identity
  caches cut the synthetic Stage-A path from 161.15 s to 11.50 s and Stage B
  from 218.88 s to 51.10 s.
- Visual-semantic SEALED/test execution is hard-disabled.

The pinned extracted-corpus manifest is
`sha256:6fa51548520190a412812ba8f872dc3c7a7a2b2c47c0e42a4d9f6df351dce138`.
There are no result placeholders: A2 is an invalidated incident, not a pending
measurement.

Build the canonical paper from the repository root:

```bash
make -C bongard/manuscript
```

This produces `free_energy_abstraction.pdf`. The compatibility entry point
`semantic_cones.tex` inputs the same canonical source and does not define a
second paper:

```bash
make -C bongard/manuscript compatibility
```

`references.bib` is shared and unchanged. Generated LaTeX files are build
products. Pre-rewrite manuscript history is preserved at the annotated tag
`pre-bongard-complete-rewrite-20260805`.
