# Cracking Real Bongard: Rule Deduction From Raw Panels Under Free Energy

**Status: Architect draft, awaiting Engineer reconciliation pass.**

This document specifies the Bongard crack: applying the Kolmogorov-Schmidhuber
harness that cracked ARC-AGI-3 (`arc/crack_lab/`, promoted artifacts `wa30` 9/9
and `ls20` 7/7) to *real* Bongard rule deduction from raw pixel panels. It is
written by the Architect role; the reconciliation log at the end records
Engineer corrections before and during implementation.

## 0. What exists today, and why it is not this

An audit of `bongard/` (July 2026) against the target "rule deduction from
scratch":

```text
artifact                              what it actually is                      gap
------------------------------------  ---------------------------------------  -----------------------------
run_bongard_symbolic_baseline.py      Bongard-STYLE induction over opaque-      not visual at all
run_bongard_sparse_classifier.py      object SEQUENCES (palindrome, ...)
run_abstraction_emergence.py          predicate-macro selection scaffold        atoms are hand-defined
run_bongard_logo_adapter.py           external Bongard-LOGO, but consumes       never touches pixels;
                                      ACTION PROGRAMS, not images; metadata     metadata mode is privileged;
                                      mode explicitly privileged                macro mode solves 6/26 Abstract
```

`bongard_first_plan.md` Stage 4 (the visual path, `image -> parser -> symbolic
scene -> sparse solver`) was deferred until the symbolic path had honest failure
modes. It now has them: Abstract action-only is a clean, diagnosed
representation failure (`bongard_logo_report.md`). By the plan's own gate, the
visual path is unblocked — and the ARC crack harness gives it a stronger form
than Stage 4 envisioned: **the parser is not fixed infrastructure; it is
proposed, verified, and priced inside the loop.**

## 1. The correspondence (the load-bearing claim)

The ARC crack loop maps onto Bongard almost line-for-line:

```text
ARC-AGI-3 crack (gkm_arena / gkm_legs)     Bongard crack
-----------------------------------------  --------------------------------------------
step(action) -> frame                      panels() -> 12 raw bitmaps (static, no actions)
reward = levels_completed                  R = held-out panel error (rotate 1+1 out, all splits)
simulator as ground-truth verifier         the panel split itself: exact, deterministic, replayable
level K                                    one Bongard problem
legs.py (reusable skills)                  predicates.py (perception routines from raw pixels)
play_level_K(env) composes legs            rule_k(panel) composes library predicates
debrief refactors repeats into legs        debrief refactors repeated vision code into predicates
C_marginal: new legs only, reuse free      C_marginal: new predicates only, reuse free
ls20 sawtooth novelty trace                sawtooth over the BP corpus  <- the headline plot
replay-validated promotion                 deterministic re-run on panels; promotion to agent_solutions/
```

Why Bongard is the *purer* Kolmogorov-Schmidhuber substrate: a Bongard
problem's intended rule is **by construction the minimal description separating
the two sides in a human perceptual vocabulary** — that is what makes the
problems fair for humans. On ARC, MDL was a selector we imposed; on Bongard,
MDL *is the task definition*. The two-sided near-miss structure is exactly the
counterexample-rich panel design this repository already established matters.
The Schmidhuber pieces slot in directly: PowerPlay ordering (always attempt the
cheapest-marginal-C unsolved problem next — a self-paced curriculum over the
corpus), compression progress (prefer the problem that most compresses the
library), and the per-problem debrief as the empirically-discharged
self-rewrite.

**The deep claim.** Minimize total `F = R + lambda * C_marginal` across the
whole corpus and the shared predicate library that emerges should approximate
the human perceptual vocabulary the set was designed around — counting,
convexity, holes, symmetry, elongation, containment, curve-vs-polygon. That is
the colimit-cone thesis (library = diagram, rule = mediating morphism, new
problem = Kan extension, predicate invention = pushout that pays;
`../COLIMIT_CONE_APPROACH.md` Section 0) on the most iconic substrate in the
concept-induction literature, as its fourth independent instance. Published
Bongard attempts either hand over the visual vocabulary (Phaeaco, Depweg et
al.'s visual language) or use a pretrained black box (VLMs, which remain weak
at rule articulation). Nobody prices the library and lets it emerge.

## 2. The one real structural difference, faced honestly

ARC had an information-rich verifier (simulator + `clone()` lookahead,
thousands of steps). Bongard verification is ~12 bits per problem: many
separating rules exist in a rich predicate space. Three defenses, all existing
house idioms:

1. **`lambda * C` carries the weight.** Among separating rules, admit the
   cheapest given the library. This is not a regularizer bolted on; it is the
   task (Section 1).
2. **Rotated leave-one-out + tie-set reporting.** Hold out 1+1 panels, rotate
   over all 36 splits; a rule is *exact* only if every rotation classifies its
   held-out pair correctly. When multiple minimal separators survive, report
   the tie-set. Under-determination is the already-understood goal-induction
   phenomenon (`home` vs `home AND safe`): free energy commits to the simpler
   hypothesis, and that is a property of the panel set, not a bug.
3. **Two structural controls, free of charge:**
   - **shuffled-sides**: reassign the 12 panels to sides at random; admission
     must fail (no cheap separator) or held-out accuracy must sit at chance.
   - **no-share**: charge each predicate's full definition per problem; the
     library must die, exactly as in the abstraction-emergence no-share
     control. If it survives, the discount is not the causal factor.

## 3. Division of labor (and the proposer-economics question)

The proposer writes **only perception legs**: predicates from raw pixels
(segmentation, counting, convexity, holes, symmetry, ...). This is where the
no-hand-coding rule lives — the human contributions remain exactly the legal
three: (1) the thin raw harness, (2) a neutral human-preconception prompt,
(3) the verify-by-panels admission loop.

Rule **composition** over the library needs no LLM at all: the existing
sparse-conjunction MDL selector (LOGO adapter / sparse classifier) searches
exhaustively over library atoms, so "this is the MDL rule given the library"
is an exact statement, not a sampled one. Consequences:

- LLM spend concentrates on genuinely novel perception; the marginal-C
  sawtooth *is* the cost curve.
- Bongard problems are tiny (12 static images, instant verification, no
  596-action replays), making this the ideal cheap substrate for the standing
  question from the ARC crack: *how weak a proposer can the same harness lift
  to competence?*

**Proposer ladder (weak-first, escalate on evidence).** The default proposer is
**headless Claude Code with Sonnet** (`claude -p --model sonnet` — already
supported by `gkm_arena.propose_text`, a one-flag change, ~4-5x cheaper than
Opus). Escalate to Opus only when a problem remains unsolved after N Sonnet
rounds, and log every escalation: *which problems need a strong proposer* is a
second novelty signal alongside marginal C. The ARC negative (a prompt-only
local model mis-reasoned multi-step reachability) does not transfer directly —
writing a single perception predicate over a static panel is a far lower bar
than planning under barriers, and the deterministic MDL selector, not the
proposer, does the rule composition. The lean Messages-API proposer
(`gkm_api_agent.py` pattern) is the next rung down the ladder after Sonnet.

**Priors (neutral, wa30-style).** Static-vision world priors only: the panels
contain objects; boundaries, counts, sizes, shapes, positions, and relations
matter; the rule is simple; the two sides are near-misses of each other. No
predicate names, no recipes ("check convexity" is forbidden), mirroring the
neutralized wa30 priors.

**The "from scratch" line, declared upfront.** The proposer gets
numpy/scipy/PIL-grade primitives — the same legality as the ARC proposer's
Python — and **no pretrained vision models**. Classical image ops are tools;
a CNN/VLM feature extractor would smuggle in the vocabulary the experiment is
supposed to grow.

## 4. Targets and the leakage protocol

Two tiers, with different honest claims:

```text
tier                     source                                claim it supports
-----------------------  ------------------------------------  --------------------------------
primary / quantitative   Bongard-LOGO rendered to IMAGES,      induction from scratch
                         fresh sampler seeds (leakage-proof,   (cannot be memorized; ground-
                         unlimited, ground-truth concept       truth names enable articulation
                         names, published baselines)           match at scale)
flagship / qualitative   classic set: Bongard's 100 +          articulate-AND-verify
                         Foundalis's index (~280+), raw GIFs   (leakage-caveated)
```

Leakage protocol for the classic set (the BPs are certainly in training data):

1. **Memorization probe:** ask the proposer for the rule from the BP index
   number alone, no images — measures what recall alone delivers.
2. Never expose problem numbers or filenames to the proposer; permute panel
   order; re-render where possible.
3. Verification is exact regardless of memorization — a recalled rule must
   still be expressed as a program over library predicates and separate the
   panels under all LOO rotations. So classic-set results honestly claim
   "articulate-and-verify"; only fresh-generated results claim induction from
   scratch.

**Known ceiling, enumerated upfront:** world-knowledge BPs (numerals, letters,
meta/self-referential problems) are skipped and flagged, like the LOGO
adapter's undersupplied attribute pairs. Report per-category.

**Literature bar:** Phaeaco solved ~10-15 classic BPs open-endedly; Depweg et
al. (2018) ~35/232 with a hand-designed visual language; VLMs remain weak at
articulation. Cracking means: **100+ classic BPs with articulated, verified,
MDL-minimal rules, plus the emergent-library result** — which is the part no
prior system can even state.

## 5. Accounting

Direct reuse of the `gkm_legs` marginal accounting with the v3 priced-binding
discipline (`../COLIMIT_CONE_APPROACH.md` Section 11):

```text
def(predicate)      charged ONCE, when first admitted to predicates.py
                    (LOC proxy + literal-cost, as in gkm_legs; hard-coded
                    lookup tables of panel answers carry full MDL cost)
rule_k              per-problem: conjunction atoms cost call_cost + binding_cost
                    (binding = which measurement/threshold/object-set fills the
                    predicate's slot — priced, not free)
C_marginal(k)       new predicate definitions admitted while solving problem k
F(k)                R(k) + lambda * C_marginal(k),
                    R(k) = rotated-LOO held-out error on problem k
```

Admission: a candidate (new predicates + rule) is admitted only if it
verifiably lowers `F` on the real panels; debrief refactors repeated perception
code across problems into shared predicates; promotion re-runs every promoted
rule deterministically on its panels (the replay analogue).

## 6. Preregistered predictions and falsifiers

Predictions, stated before any run:

1. **Sawtooth collapse.** Marginal C collapses over the corpus: early problems
   pay for segmentation/counting; later problems compose for near-zero.
2. **Novelty alignment.** Marginal-C spikes align with the corpus's known
   taxonomy boundaries (texture, curvature classes, topology, size relations)
   — marginal free energy as novelty detector, the ls20 result re-instantiated.
3. **Controls behave.** no-share kills the library; shuffled-sides fails
   admission or scores at chance.
4. **Articulation match.** Selected rules name-match the catalogued solutions
   (Foundalis for classic; concept names for LOGO) on most solved problems;
   mismatches concentrate in provably under-determined panel sets (tie-sets).

Falsifiers:

1. Flat marginal C (the library never amortizes) — consequence 3 of the
   general principle dies on this substrate.
2. No growth in per-problem solve rate as the library grows — the
   cone-connectivity claim (better-connected search space) dies.
3. Only metadata-grade privileged hints ever crack Abstract-class concepts —
   the representation-poverty diagnosis was terminal, not treatable.
4. Shuffled-sides admits rules at real-problem rates — the free-energy
   admission is not doing the selecting.

## 7. Phases

```text
phase 0  audit (done, Section 0): existing bongard/ is symbolic; nothing
         consumes raw panels; the crack harness is the right machine
phase 1  bongard/crack_lab: thin arena (panel loader, LOO-rotation verifier,
         marginal-C accounting — gkm_legs pattern reused near-verbatim) on
         ~30-50 fresh-seed RENDERED Bongard-LOGO problems; SONNET headless
         proposer by default (Opus escalation on stuck problems, escalations
         logged) with neutral static-vision priors; enforced predicates.py
         library; first sawtooth + both controls
phase 2  scale; PowerPlay ordering over the corpus; descend the proposer
         ladder further (lean Messages-API loop) — Bongard as the cheap
         substrate for the how-weak-a-proposer question
phase 3  classic Foundalis set from raw GIFs as flagship: articulation
         name-match vs catalogued solutions, per-category report, leakage
         protocol of Section 4
```

Dataset policy (unchanged house rule): nothing vendored; Bongard-LOGO cloned
under `downloads/`, Foundalis GIFs downloaded outside version control; only
small derived metadata cached.

## 8. Engineer plan (reconciled, July 2026)

The build lives in `bongard/crack_lab/` as a **sibling** of `arc/crack_lab/`
(house convention: siblings, not modifications). Two modules plus tests:

```text
bongard/crack_lab/
  bongard_arena.py    the raw substrate: fresh-seed LOGO sampler bridge,
                      deterministic pure-numpy rasterizer (action strings ->
                      panels), Problem = 12 bitmaps, the MDL conjunction
                      selector over proposer predicates, rotated-LOO verify,
                      free energy
  bongard_legs.py     enforced predicate-library orchestration: workspace
                      (predicates.py is the ONLY place logic accumulates),
                      tester, Sonnet-first headless Claude proposer with Opus
                      escalation (escalations logged), marginal-C accounting,
                      checkpoint.json, promotion to agent_solutions/,
                      taint check, WIP context
  test_bongard_arena.py / test_bongard_legs.py
                      offline tests: injectable propose_fn; witness predicates
                      live ONLY in tests (representability floors, never
                      shipped to the proposer)
```

Protocol detail fixed by the Engineer: the proposer sees all 12 panels (as a
human does) and writes only `predicates.py`. The harness selector then runs the
rotated leave-one-out: for each of the 36 (pos_i, neg_j) holdouts it selects
the min-F conjunction over library atoms **using only the other 10 panels**
and classifies the held-out pair. `R = held-out error over all 72 predictions`;
solved requires all 72 correct plus a full-panel separating rule. This keeps
the articulated rule well-defined (the full-panel selection) while the rotation
is the overfit guard.

## Reconciliation Log

- **R1 (Engineer -> Architect, resolved):** `gkm_legs` does not separate
  cleanly — its Report/levels/paths/replay types are ARC-shaped throughout.
  Decision: sibling module reusing the *idioms* (LOC+literal-cost complexity
  proxy, validated-checkpoint promotion gating, WIP snapshots, workspace taint
  markers) rather than the code. Bongard's verify is a pure function of
  (predicates source, panels); no step budget, no clone, simpler WIP story.
- **R2 (Engineer -> Architect, resolved):** rendering goes through our own
  deterministic pure-numpy rasterizer of LOGO action strings (turn/arc
  denormalization conventions copied from `run_bongard_logo_adapter.py`),
  with per-panel seeded placement (rotation/scale/translation). No
  turtle/Tk dependency; bit-exact replays follow from determinism. The
  official painter renders differently — our panels are *a* faithful visual
  realization of the action programs, not pixel-identical to the published
  dataset; stated in reports.
- **R3 (Engineer -> Architect, resolved):** the adapter's selector is bound to
  `LogoSceneObject` scenes; the crack selector is a fresh minimal MDL
  conjunction search whose atoms are thresholded outputs of proposer-authored
  predicate callables (`p_*(panel) -> float|bool`), thresholds taken from
  train-panel value midpoints, atom cost = call + binding (threshold/op).
  Candidate-atom ranking by train separation is kept as a search-budget cap.
