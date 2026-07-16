# Socratic revision record for `arc_agi3.tex`

This file preserves the questions, code evidence, and manuscript decisions used in
the July 2026 revision. The governing rule is that executable code, canonical
checkpoints, artifact-history manifests, and replay results take precedence over
manuscript prose.

## Pass 1: empirical claims and falsifiability

### Questions

1. What has actually been promoted, and what has only been retained as WIP?
2. Which values come from canonical root checkpoints, which come from the two
   complete-history sidecars, and which come from the closed ARC scorecard?
3. Does "same architecture" mean identical solver source, or an unchanged harness
   and promotion protocol that produces game-specific solver artifacts?
4. Can every numerical statement be defeated by a concrete replay, manifest, or
   source-diff countertest?
5. Does the prose ever interpret positive net file growth as proof that no shorter
   composition existed?

### Evidence consulted

- `arc/crack_lab/gkm_legs.py`: `description_complexity`,
  `marginal_complexity`, checkpoint deduplication, workspace locking, taint checks,
  and promotion order.
- Canonical `checkpoint.json` files for `wa30`, `ls20`, `ft09`, `g50t`, `r11l`,
  `sp80`, `tr87`, and `tu93`.
- `arc/manuscript/history_manifest.py` and both generated complete-history
  manifests.
- `arc/crack_lab/replay_scorecard.py` and the closed Competition-Mode scorecard
  `9e166671-0953-42f3-89de-a0fd57d7b147`.
- Repository tests that scan all eight promoted roots, require unique ledger rows,
  recompute checkpoint totals, and reject tainted promotion inputs.

### Findings and decisions

- There are eight replay-valid promoted roots: two complete games, five games
  stopped after L4, and `tu93` through L1. `sc25` remains WIP and is excluded.
- The official score is 17.136507936507936% over 25 public games, with 37/183
  levels and 1456 API actions including eight resets. Artifact path lengths exclude
  those resets.
- `wa30`'s root checkpoint is an operational resume checkpoint with a 1243-unit
  post-base record. The canonical manuscript sidecar preserves the nine-entry
  1458-unit historical ledger. The manuscript must name this distinction rather
  than imply that the root checkpoint itself contains nine rows.
- "One architecture" is retained only as a statement about the unchanged
  game-agnostic harness, interface, proposer protocol, taint boundary, and promotion
  rule. The generated `legs.py` and `players.py` files are necessarily game-specific
  learned artifacts.
- A large marginal charge establishes retained file growth under the implemented
  proxy. It does not prove that the incumbent library was incapable of a shorter
  solution. Semantic reuse claims require the player call graph, adjacent source,
  and fresh replay.
- The abstract and endpoint table are expanded to report the official score and all
  promoted endpoints, while complete per-level historical analysis remains confined
  to `wa30` and `ls20`.

## Pass 2: mathematical objects and inferential scope

### Questions

1. Is the manuscript's empirical frontier actually Kolmogorov's classical structure
   function, or an inverse complexity-at-fit analogue motivated by it?
2. Are retained description length and the pathwise positive-growth ledger denoted by
   different symbols and used in the correct free-energy expressions?
3. Does the categorical object accommodate history-dependent and sequential Python
   programs, rather than pretending that every leg is a frame-only policy?
4. Which categorical statements are theorems, and which empirical claims require
   replay evidence beyond the elementary union theorem?
5. Does the ideal two-part cone code accurately describe the implementation, or is it
   being presented as an equality with the coarse source counter?

### Findings and decisions

- The classical finite-set structure function is stated explicitly as
  `h_x(alpha) = min log |S|` subject to `x in S` and `K(S) <= alpha`. The solver
  frontier is named an inverse empirical complexity-at-fit function, not the
  classical object itself.
- `D(s)` denotes retained executable description. `C_{<=k}` denotes historical
  positive net growth. The implementation's diagnostic free energy uses the latter;
  the Legendre--Fenchel frontier uses the former. They coincide only under additional
  monotone-growth assumptions, which are now stated.
- Policies are partial functions on finite action-observation histories augmented by
  finite solver control state. This supports search state, program counters, and
  sequential leg dispatch without granting the solver hidden environment state.
- The compatible-policy colimit remains an elementary union in a poset category. Its
  evidential relevance comes only from named source factorization and replayed overlap
  compatibility; the theorem is not offered as an empirical result.
- The cone code is an ideal two-part description schema. The implemented metric is a
  language-dependent surrogate for changes in that schema and can net same-file
  replacement to zero.

## Pass 3: scholarly presentation and argumentative economy

### Questions

1. Does the paper read as a research article with a question, formal object, method,
   results, comparison, and delimited conclusion, rather than as a lab report?
2. Do tables and figures appear after they are introduced, fit the text measure, and
   expose rather than obscure the empirical record?
3. Are all eight promoted endpoints visible without overloading the two complete
   history case studies?
4. Is the prose precise and impersonal, especially where it discusses proposer
   protocol violations and the OPINE comparator?
5. Does the abstract remain below the 1800-character constraint after adding the
   official score?

### Findings and decisions

- The empirical section is renamed accordingly and divided into complete-history
  interpretation and the bounded cross-game campaign.
- Float order is constrained so the endpoint table cannot precede the section and
  paragraph that introduce it. The long `wa30` ledger is displayed rather than forced
  into an overfull prose line.
- A new source-driven small-multiples figure reads all five L4 profiles from canonical
  root checkpoints, validates their rows and totals, and uses one shared ordinate.
  Exact values remain in the endpoint table.
- Figure captions distinguish measured net growth from semantic reuse attribution.
  The taint incident is described as a recorded interface violation and operational
  cheating classification, not as evidence about model character.
- The OPINE conclusion remains firm but criterion-relative: the published artifact
  does not satisfy the paper's compressed generative structure-function criterion.
  This is supported by the measured archive rather than by rhetoric.
- The abstract is checked mechanically after all edits and must remain below 1800
  characters.
