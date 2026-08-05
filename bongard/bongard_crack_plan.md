# Cracking Real Bongard: Rule Deduction From Raw Panels Under Free Energy

**Status: reconciled implementation plan; historical proposals are marked.**

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
C_marginal: new legs only, reuse free      C_conditional: newly reached definition identities
                                           plus per-use call/binding structure
ls20 sawtooth novelty trace                sawtooth over the BP corpus  <- the headline plot
replay-validated promotion                 corpus/panel/source-bound checkpoint and artifact evidence
```

Why Bongard is the *purer* Kolmogorov-Schmidhuber substrate: a Bongard
problem's intended rule is **by construction the minimal description separating
the two sides in a human perceptual vocabulary** — that is what makes the
problems fair for humans. On ARC, MDL was a selector we imposed; on Bongard,
MDL *is the task definition*. The two-sided near-miss structure is exactly the
counterexample-rich panel design this repository already established matters.
The Schmidhuber pieces slot in directly: PowerPlay ordering (always attempt the
cheapest-conditional-C unsolved problem next — a self-paced curriculum over the
corpus), compression progress (prefer the problem that most compresses the
library), and the per-problem debrief as the empirically-discharged
self-rewrite.

**The deep claim.** Minimize risk and then exact conditional description cost
across the whole corpus and the shared predicate library that emerges should approximate
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

1. **Conditional complexity carries the tie-break.** The implemented priced
   selector minimizes empirical error first and exact conditional cost second.
   This avoids mixing raw source LOC and a risk fraction on an arbitrary scalar
   scale while still choosing the cheapest separator given the already-paid
   library (Section 5).
2. **Rotated leave-one-out + deterministic tie-breaking.** Hold out 1+1 panels, rotate
   over all 36 splits; a rule is *exact* only if every rotation classifies its
   held-out pair correctly. When multiple equal-risk/equal-cost separators
   survive, the implemented selector chooses one by a stable lexical key; it
   does not claim to enumerate a tie-set. Under-determination is the
   already-understood goal-induction
   phenomenon (`home` vs `home AND safe`): free energy commits to the simpler
   hypothesis, and that is a property of the panel set, not a bug.
3. **Two structural controls, free of charge:**
   - **shuffled-sides**: reassign the 12 panels to sides at random; admission
     must fail (no cheap separator) or held-out accuracy must sit at chance.
   - **no-share**: hold the primary accepted sources, rules, risks, and outcomes
     fixed, then repay every rule's full reachable definition closure per
     problem. This is an accounting counterfactual for amortization, not a
     fresh proposer run or a causal solve-rate arm.

## 3. Division of labor (and the proposer-economics question)

The proposer writes **only perception legs**: predicates from raw pixels
(segmentation, counting, convexity, holes, symmetry, ...). This is where the
no-hand-coding rule lives — the human contributions remain exactly the legal
three: (1) the thin raw harness, (2) a neutral human-preconception prompt,
(3) the verify-by-panels admission loop.

Rule **composition** over the library needs no LLM at all. The current selector
exhaustively compares constants and conjunctions of up to two atoms within its
declared 24-candidate search bound, so “this is the selected rule under the
bounded policy and exact priced source” is an exact statement, not a sampled
one. Consequences:

- LLM spend concentrates on genuinely novel perception; the conditional
  definition-charge trace is the reuse/novelty cost curve.
- Bongard problems are tiny (12 static images, instant verification, no
  596-action replays), making this the ideal cheap substrate for the standing
  question from the ARC crack: *how weak a proposer can the same harness lift
  to competence?*

**Proposer ladder (weak-first, escalate on evidence).** The implemented
unrestricted default is three bounded attempts through non-interactive
`codex exec`, explicitly requesting `gpt-5.6-sol` and medium reasoning. Each
ephemeral turn's prompt and private working directory supply twelve copied PNGs
plus the current source/log, but no harness-workspace path, and
shell/unified exec, network search, apps, plugins, browser/computer use, hooks,
skills, and sub-agents are disabled. Only a schema-valid complete source/log
response is applied by the outer harness. A consuming production turn requires
positive token use and a receipt binding the exact task, current/proposed
source and log, raw/semantic panel identities, structured output, unique turn,
output schema, CLI version, and resolved launcher file. This is local causal
provenance, not provider-signed attestation, and it does not digest every
transitive launcher dependency. If JSONL omits its optional model field, the
receipt says so instead of fabricating `actual=requested`. Codex CLI 0.146.0
cannot pre-disable `view_image`; any emitted image/tool event invalidates the
turn after execution. Every escalation is therefore evidenced: *which problems
need another proposer turn* is a second novelty signal alongside conditional
definition charge. The ARC negative (a prompt-only
local model mis-reasoned multi-step reachability) does not transfer directly —
writing a single perception predicate over a static panel is a far lower bar
than planning under barriers, and the deterministic MDL selector, not the
proposer, does the rule composition. The semantic Messages proposer is a
separate path: it resolves aliases to
concrete provider IDs and rejects every response whose reported model is
missing or different.

**Priors (neutral, wa30-style).** Static-vision world priors only: the panels
contain objects; boundaries, counts, sizes, shapes, positions, and relations
matter; the rule is simple; the two sides are near-misses of each other. No
predicate names, no recipes ("check convexity" is forbidden), mirroring the
neutralized wa30 priors.

**The "from scratch" line, declared upfront.** Predicate source runs inside the
versioned positive language `bongard-predicate-purity/v2`: only the exact
imports, restricted builtins, calls/values, methods/attributes, keyword forms,
owned-scratch mutation, and resource forms in
`predicate_capability_manifest()` are legal. A listed numpy/scipy/skimage root
does not authorize its other APIs, PIL is not a predicate capability, and no
pretrained vision model is available. A CNN/VLM feature extractor would
smuggle in the vocabulary the experiment is supposed to grow.

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

The unrestricted implementation now uses AST-backed conditional pricing:

```text
closure(rule_k)     exact transitive AST dependency union of selected p_*
                    predicates, helpers, constants, and imports; shared nodes
                    within one rule are counted once
definition_cost     exact non-comment LOC + literal/call payload + executable
                    AST-structure charge, keyed by source-content identity
                    rather than only by Python name
definition_charge   costs in closure(rule_k) whose identities were not reached
                    by an earlier accepted rule; unused library code is not paid
structure_charge    per-use call and threshold/operator binding cost for every
                    rule atom; it is never discounted by sharing
selection           minimize empirical error, then
                    definition_charge + structure_charge
```

The same immutable pricing context is written for the proposer-side tester and
used by the authoritative verifier. Only definitions reached by accepted rules
enter the paid ledger; rejected or unused code does not make later code free.
Changed helper source receives a new content identity even if the `p_*` body is
unchanged. Promotion records the accepted source snapshot, rule atoms, used,
charged, and reused nodes, and definition/structure/total receipts.

The no-share report is derived from the corresponding primary source trace. It
copies accepted rules, risks, and outcomes and substitutes a full-definition
charge on every accepted use. Therefore it supports a direct charge comparison
but no independent solve-rate conclusion.

## 6. Preregistered predictions and falsifiers

Historical scientific predictions, translated to the current receipts before
any Phase D run:

1. **Sawtooth collapse.** Conditional definition charge collapses over the
   corpus: early problems pay for segmentation/counting; later problems mostly
   pay per-use structure.
2. **Novelty alignment.** Definition-charge spikes align with the corpus's known
   taxonomy boundaries (texture, curvature classes, topology, size relations)
   — marginal free energy as novelty detector, the ls20 result re-instantiated.
3. **Controls behave.** Shared definition charge is lower than held-fixed
   no-share charge when accepted definitions are reused; full adaptive
   shuffled-side arms fail admission or score at chance. No-share solve rate is
   identical by construction and is not a prediction.
4. **Articulation match.** Selected rules name-match the catalogued solutions
   (Foundalis for classic; concept names for LOGO) on most solved problems;
   mismatches motivate a separately declared ambiguity analysis; the current
   selector does not publish tie-sets.

Falsifiers:

1. Flat conditional definition charge (the library never amortizes) — consequence 3 of the
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
         exact conditional definition/structure accounting) on
         ~30-50 fresh-seed RENDERED Bongard-LOGO problems; bounded headless
         Codex proposer attempts (`gpt-5.6-sol`, medium reasoning) with every
         consuming turn causally receipted; neutral static-vision priors; enforced predicates.py
         library; first sawtooth + both controls
phase 2  scale; PowerPlay ordering over the corpus; descend the proposer
         ladder further with separately preregistered models — Bongard as the cheap
         substrate for the how-weak-a-proposer question
phase 3  classic Foundalis set from raw GIFs as flagship: articulation
         name-match vs catalogued solutions, per-category report, leakage
         protocol of Section 4
```

**August 2026 status.** The offline Phase D machinery can prepare a write-once
maximum corpus and embedded panel bundle, independent source RNG streams,
balanced shuffled controls, gated nested 1/5/25 growth, the exact priced
selector, immutable paid ledger, held-fixed no-share derivation, track reports,
and deterministic artifact-certified campaign collection. For each primary or
shuffled proposer family, n1 is the only legal fresh start, n5 requires a
complete replay-valid n1 checkpoint, and n25 requires n5; jumps, shrinkage, or
incomplete predecessors fail before proposer construction or writes. The
default design has 27 arms:
for each of three scales, two tracks each receive one primary plus three
shuffled arms, and
unrestricted receives one accounting-only no-share arm. Semantic no-share is
excluded until there is a learned/base registry split. A paid
unrestricted-only n=1 exploratory pilot completed on 5 August 2026 (primary
0/1 ordinary miss, shuffled 0/1 canonical verifier failure, no-share 0/1),
with all three artifacts cold-replay certified. It is not the default study or
confirmatory evidence; the first Sonnet smoke remains only historical
engineering evidence.
The local write-once preregistration is a reproducibility manifest, not an
external timestamp; a confirmatory claim requires its digest to be published
or externally committed before the first proposer call.

Dataset policy (unchanged house rule): nothing vendored; Bongard-LOGO cloned
under `downloads/`, Foundalis GIFs downloaded outside version control; only
small derived metadata cached.

## 8. Engineer plan (reconciled, July 2026)

The build lives in `bongard/crack_lab/` as a **sibling** of `arc/crack_lab/`
(house convention: siblings, not modifications). Current load-bearing modules
include:

```text
bongard/crack_lab/
  bongard_arena.py    the raw substrate: fresh-seed LOGO sampler bridge,
                      deterministic pure-numpy rasterizer (action strings ->
                      panels), Problem = 12 bitmaps, the MDL conjunction
                      bounded exhaustive selector over proposer predicates,
                      rotated-LOO verify, exact priced selection
  predicate_pricing.py
                      positive predicate capability manifest, restricted
                      builtins, AST definition graph, transitive closure,
                      source-content identities, definition-cost receipts
  bongard_legs.py     enforced predicate-library orchestration: workspace
                      (predicates.py is the ONLY place logic accumulates),
                      tester, bounded headless Codex proposer attempts,
                      structured whole-source/log handoff, model/usage/event/
                      launcher receipts, immutable paid ledger, exact pricing
                      receipts, checkpoint/resume, held-fixed no-share,
                      promotion to agent_solutions/, taint check, WIP context
  codex_proposer.py   hardened ephemeral Codex transport: private copied
                      panels, schema-constrained output, strict event audit,
                      bounded process/output handling, receipt construction
  phase_d_protocol.py / prepare_phase_d.py
                      frozen corpus and panel bundle, balanced controls,
                      exact capability/pricing policy binding, preregistration,
                      immediate-prefix growth and per-arm/report validation
  collect_phase_d.py  execution-artifact certifier, exact-arm closure,
                      cold replay, canonical digest and write-once campaign
                      publication
  test_bongard_arena.py / test_bongard_legs.py
                      offline tests: injectable propose_fn; witness predicates
                      live ONLY in tests (representability floors, never
                      shipped to the proposer)
```

Protocol detail fixed by the Engineer: the proposer sees all 12 panels (as a
human does), may edit executable logic only in `predicates.py`, and may also
edit the non-executable `predicates_log.md`. The harness selector then runs the
rotated leave-one-out: for each of the 36 (pos_i, neg_j) holdouts it selects
the min-F conjunction over library atoms **using only the other 10 panels**
and classifies the held-out pair. `R = held-out error over all 72 predictions`;
solved requires all 72 correct plus a full-panel separating rule. This keeps
the articulated rule well-defined (the full-panel selection) while the rotation
is the overfit guard.

That rotation is a rule-selection diagnostic, not an untouched
representation-level holdout: the proposer has already seen every panel. An
identity-keyed lookup is instructed against and its literal/call payload is
charged and visible under `bongard-predicate-pricing/v3`, but rotated LOO cannot
categorically distinguish it from a general measurement. Generalization claims
therefore require separately held unseen instances.

## 9. Stage 1.5: describe-first A/B on a future cracked-25 corpus (planned)

If Stage 1 reaches 25 solved problems, those problems can become a
**solvability-controlled A/B corpus**: every included problem will already be
known crackable by the pinned headless Codex proposer, so arm differences can
measure the intervention rather than initial problem difficulty. No such
cracked-25 corpus exists yet.

**The question** (raised by the user, July 2026): does an explicit
DESCRIBE stage — write a human-like description of each panel and a
candidate one-sentence rule *before* implementing predicates — change what
gets discovered? Language as an inductive-bias channel: verbal priors may
pull predicates toward the generator's vocabulary (fewer proxy solves like
`p_pinch_notch_defect` cracking `has_eight_straight_lines`).

**Design:**

```text
proposer       a separately preregistered lean API loop; its provider/model is
               an experimental factor and is not implied by the current
               headless Codex production path
arm A          current prompt (implicit description, straight to predicates)
arm B          describe-first prompt (mandatory panel descriptions + candidate
               one-sentence rule logged to notes, then predicates)
corpus         the 25 stage-1-cracked problems, same seeds/rendering
libraries      each arm starts from an EMPTY predicates.py under its own tag
               (no contamination from the stage-1 library or across arms)
scoring        (1) solve rate under identical budgets;
               (2) articulation name-match vs the generator concept in
                   normalized language (results.json ground truth);
               (3) definition-charge trace shape (does describe-first change reuse?)
boundary       descriptions are hypothesis generation and articulation ONLY;
               the verified object remains deterministic p_*(panel) code.
               A VLM call inside a predicate is forbidden (non-deterministic,
               unpriceable, smuggles pretrained vision into verification).
```

**Prediction (stated before running):** arm B's solves name-match the
generator concept more often, at equal-or-slightly-lower solve rate (the
describe stage spends budget); arm B predicates are more reused (closer to
the human vocabulary => more shareable axes).

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
- **R4 (proposer -> Engineer, from the first smoke run, resolved):** the Sonnet
  proposer reported that it deliberately folded an AND of two raw
  measurements into ONE composite predicate because the rule search prices
  atoms flatly and LOO rotations could tie-break toward a cheaper-but-wrong
  single-threshold rule when the measurements were exposed separately. Total
  accounting held only at the historical whole-file level, while `rule_cost`
  understated the selected implementation. The current verifier prices the
  exact transitive definition closure of selected atoms, separately charges
  each atom's call/binding structure, and chooses by risk then conditional
  cost. Hiding the AND behind a name no longer erases its definition. See
  `bongard_crack_smoke_report.md` for the historical observation.
- **R3 (Engineer -> Architect, resolved):** the adapter's selector is bound to
  `LogoSceneObject` scenes; the crack selector is a fresh minimal MDL
  conjunction search whose atoms are thresholded outputs of proposer-authored
  predicate callables (`p_*(panel) -> float|bool`), thresholds taken from
  train-panel value midpoints. Every atom pays call + binding (threshold/op),
  and the rule also pays the conditional exact definition closure described in
  Section 5. Candidate-atom ranking by train separation is kept as a declared
  search-budget cap.
