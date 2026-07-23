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

## Pass 4: reproduction contract and observable coordinates

### Questions

1. Does ``replay validated'' name one operation, or does it conflate solver execution,
   local action-path replay, and remote scorecard replay?
2. Can a reader run every command named by the manuscript from this checkout?
3. Does the results section report the retained-description coordinate $D(s)$ required
   by its own empirical structure function, or only the historical ledger $C_{\leq k}$?
4. Which parts of the experiment can be reproduced without regenerating stochastic
   proposer calls?
5. Is the exposed environment interface stated completely and exactly?

### Evidence consulted

- `arc/crack_lab/gkm_arena.py`: fresh program execution and deterministic path
  validation.
- `arc/crack_lab/gkm_legs.py`: candidate execution, post-debrief re-execution,
  promotion gate, retained-description metric, and promoted core files.
- `arc/crack_lab/replay_scorecard.py`: remote online/Competition-Mode path replay.
- `arc/manuscript/build_artifact_history.py`, both history manifests, and all eight
  canonical endpoint source snapshots.
- Repository root `README.md` and the offline harness tests.

### Findings and decisions

- The method now distinguishes three validation layers: a fresh local solver run that
  produces a path, deterministic replay of that path on another local environment, and
  remote replay of stored paths for an ARC scorecard. The remote layer neither runs the
  solver nor invokes a proposer.
- The manuscript's reference to the nonexistent `REPRODUCE_ARC.md` is removed. A
  reproduction subsection now gives commands that exist in the checkout for sidecar
  validation, offline invariants, and online replay.
- The endpoint table now reports $D(s)=d(\ell)+d(p)$ for every promoted root:
  `wa30` 1334, `ls20` 397, `ft09` 426, `g50t` 349, `r11l` 924, `sp80` 309,
  `tr87` 650, and `tu93` 255. These are retained-size coordinates, not ledger sums or
  cross-game frontier values.
- The exact local interface now includes `reset()`, `actions`, and `terminal()` as well
  as the raw frame, reward, action, and clone operations.
- Artifact replay and accounting are reproducible; model identity, token use, clone
  interactions, and stochastic discovery are not uniformly logged and therefore are
  not claimed to be exactly regenerable.

## Pass 5: identification, counterclaims, and validity limits

### Questions

1. Does stopping five runs at level 4 identify what would be required after level 4?
2. Does a clean deny-list scan prove that no forbidden information reached a solver?
3. Does deterministic replay show policy robustness outside the recorded trajectory?
4. Does source growth in the OPINE archive prove the semantic absence of abstraction
   or reuse?
5. Which conclusions survive non-random game selection, uncontrolled proposer budgets,
   and incomparable source units?

### Findings and decisions

- The claim that progress beyond level 4 primarily requires more proposer compute is
  removed. The bounded campaign establishes portability only through observed depths;
  by construction it supplies no evidence about later levels.
- A promoted root is now called clean only in the operational sense that it passed the
  implemented lexical and path-based checks. The manuscript explicitly disclaims a
  complete information-flow guarantee.
- Replay is delimited to deterministic endpoint evidence. It does not establish the
  full policy domain or robustness under perturbed states.
- The OPINE comparison is reframed as a two-part description audit. Cached entry states
  are valid conditioning data but must be charged; the lack of a near-zero token-growth
  step or a large retained-size reduction does not prove that the program contains no
  reusable mechanism.
- The comparator verdict now describes the published auditable object---cumulative
  transition code plus cached entry data---and does not redefine the ordinary term
  ``world model'' or deny OPINE's planning utility.
- The validity section now separates construct, trajectory, leakage, sampling, compute,
  and cross-metric limits. The scorecard remains an endpoint result, not a success-rate
  estimate or compute-matched ranking.

## Pass 6: reader model and argumentative economy

### Questions

1. Does the title state the paper's actual contribution rather than imply a broad
   benchmark-solving or classical-Kolmogorov result?
2. Is the answer to the research question stated explicitly and early?
3. Can a reader tell what the structure function, categorical construction, source
   ledger, and replay gate each contribute?
4. Do figure captions report observables before interpretations?
5. Are future predictions conditioned on known failure modes of the metric?

### Findings and decisions

- The title is changed to *Auditing Solver Growth in ARC-AGI-3: Description-Length
  Ledgers and Empirical Policy Colimits*, and the date is fixed for reproducible builds.
- The introduction now gives the qualified answer: marginal complexity alone is not a
  semantic novelty detector; paired source provenance, named calls, and replay make it
  a falsifiable audit index.
- The respective roles are made explicit. Retained $D(s)$ locates an artifact at a
  verified depth, the historical ledger records when positive growth was paid, policy
  colimits describe compatible composition, and replay remains the behavioral test.
- The `ls20` and OPINE captions now lead with measured co-occurrence and token growth,
  avoiding causal or semantic conclusions unsupported by the plotted scalars.
- The proposed mechanic-change prediction is restricted to prospectively logged runs
  and conditioned on the absence of offsetting deletions that can hide growth.
- The fixed $\lambda=0.02$ value is identified as a diagnostic default with no effect on
  promotion or the reported conclusions.

## Pass 7: categorical adequacy and the role of cofibrations

### Questions

1. What information does a compatible-policy colimit discard even when the union
   theorem is correct?
2. Can cofibrations be defined on the empirical object without pretending that
   arbitrary Python programs form a model or Waldhausen category?
3. Which attachment property is actually required for reusable solver composition?
4. Does a cofibration label add anything beyond drawing a hook on every policy
   inclusion?
5. How should behavior-extending promotions be distinguished from behavior-preserving
   refactors?

### Evidence consulted

- The categorical definitions and two-part cone code already present in the manuscript.
- `arc/crack_lab/gkm_legs.py`: replay-preserving promotion, source diffs, debrief
  validation, and the fact that stable source interfaces are inspected rather than
  universally enforced.
- The complete `ls20` and `wa30` histories, particularly the thin L2/L4 search players
  and the literal-to-`ferry_each` L9 debrief.
- Mac Lane for colimits and Waldhausen for the inclusion-and-attachment role of
  cofibrations.

### Findings and decisions

- A colimit is extensional: it preserves the least compatible union but forgets whether
  that endpoint arose monolithically, through reusable attachments, or through a
  behavior-preserving refactor.
- A finite validated trace policy $T$ now determines the restriction category
  $\mathsf{Pol}_T(E)$. Its arrows are declared trace cofibrations. They preserve old
  actions, expose the relative attachment $U_q\setminus U_p$, compose, and remain
  cofibrations under pushout.
- Pushout stability is proved rather than asserted. It formalizes reuse under compatible
  change of context: the old behavior survives cobase change, while a dispatch conflict
  prevents the policies from entering the same trace-restricted category.
- Because every arrow of $\mathsf{Pol}_T(E)$ is a cofibration, the manuscript explicitly
  identifies the nontrivial policy-level data as the chosen filtration and pushout
  squares. Empirical selectivity enters at the program level through an attachment
  certificate containing unchanged legs, new glue, bindings, replay, and charge.
- A provenance category $\mathsf{Art}_T(E)$ and empirical behavior functor $B$ separate
  policy-cofibrational promotions from replay equivalences $s\simeq_Ts'$. The latter
  compare descriptions at fixed observed behavior and are the correct locus for
  refactoring/compression claims.
- The scope is delimited: no global pushouts for incompatible policies, exhaustive
  denotation of Python, lifting/factorization axioms, or model/Waldhausen structure are
  claimed.
- Three commutative diagrams now show the cofibrational pushout, the filtered promotion
  chain, and a replay-equivalent debrief.

## Pass 8: high-level framing, compression, and categorical exposition

### Questions

1. Does the title foreground all three mathematical objects without hiding the ARC-AGI-3
   empirical subject?
2. Can the abstract state the formal result, empirical result, and limitations in
   substantially less space?
3. Do the diagrams occur where their universal properties and empirical meanings are
   explained?
4. Are cofibrations connected to actual source evidence rather than left as ornamental
   formalism?
5. Does the conclusion distinguish endpoint colimits, acquisition filtrations, and
   fixed-behavior compression?

### Findings and decisions

- The Pass-6 title is superseded by *Kolmogorov Structure Functions, Categorical
  Colimits, and Cofibrations: Auditable Solver Growth in ARC-AGI-3*.
- The abstract is reduced from 1,593 raw characters (215 words) to 1,154 raw characters
  (146 words). It retains the scorecard endpoint, the structure-function coordinate,
  cofibrational attachment, replay-equivalent refactoring, and the compute limitation.
- The contribution list now contains an explicit formal contribution for partial-policy
  colimits, trace cofibrations, and replay equivalence.
- The discussion maps `ls20` L2/L4 to attachment-certified cofibrations: unchanged
  search legs, two or three units of glue, and replay. The `wa30` L9 debrief is instead
  treated as replay-equivalent re-description at fixed observed behavior.
- Compression is conditional on a shorter chosen description; the historical ledger
  never retroactively deletes the acquisition cost.
- The conclusion now states the division of labor: Kolmogorov structure functions
  compare description at fit, colimits give compatible extensional endpoints,
  cofibration filtrations preserve ordered attachments, and replay equivalence exposes
  refactors at fixed behavior.
- The added theory and three diagrams bring the built manuscript to 19 pages, within the
  permitted 25-page envelope.

## Pass 9: title and terminological consistency

### Questions

1. Does the title state the requested mathematical and agentic framing directly?
2. Is ``self-improving agent'' defined narrowly enough to match the implemented system?
3. Do the abstract, introduction, keywords, and conclusion use the same academic
   vocabulary without expanding the empirical claim?

### Findings and decisions

- The title is now *Kolmogorov Structure Functions, Colimits, and Cofibrations: Solving
  ARC-AGI-3 with Self-Improving Agents*.
- ``Self-improving'' is defined operationally as cumulative revision of retained solver
  code under a fixed harness and replay gate; unrestricted self-modification and
  proof-theoretic optimality are explicitly excluded.
- The abstract now opens with the cross-task reuse question for self-improving coding
  agents, while retaining the exact 37/183 endpoint and compute limitation.
- Keywords and conclusion are aligned with the title. The plural title names the class
  of systems; the manuscript continues to report one artifact-producing implementation
  and eight promoted game endpoints.

## Pass 10: contribution identity, observed properties, and resource truncation

### Questions

1. Is auditability the proposed method, or is it a property of the evidence produced by
   a new solver-growth approach?
2. Which structural properties are actually observed rather than merely made
   inspectable?
3. Does ``complexity drop'' mean a fall in retained description at fixed behavior, or a
   fall in the marginal acquisition cost of a later promotion?
4. What does the preserved record support when incomplete games end at imposed compute,
   interruption, or credit boundaries?
5. Does any claimed compression contradict the implemented source proxy?

### Evidence consulted

- The complete `ls20` and `wa30` manifests and their adjacent source snapshots.
- Root checkpoints and WIP `latest.json` records for the bounded campaign.
- Interrupted level-5 continuations for `g50t` and `tr87`, the explicit `tu93`
  level-2 `credit_out` record, and the `ls20` recovery after credit exhaustion.
- The recovered `wa30` level-9 suffix, its time-budget provenance, and the
  pre-/post-debrief `legs.py` and `players.py` sources.
- The implementation of `description_complexity` in `gkm_legs.py`.

### Findings and decisions

- The primary contribution is now stated as a verifier-gated, self-improving
  solver-growth approach. Auditability is its evidence contract and an important
  artifact property, not the name of the method.
- The empirical claim is positive on the observed histories. New mechanics and literal
  plans produce charged additions; recurring mechanics factor through unchanged named
  legs; and later reuse produces sharp marginal-complexity drops, including `ls20`
  43-to-2 and 45-to-3 and `g50t` 100-to-2.
- ``Complexity drop'' is used for marginal acquisition cost unless a fixed-behavior
  retained-description decrease is measured explicitly. The two notions are no longer
  conflated.
- The `wa30` level-9 debrief is replay-equivalent refactoring, not measured
  compression: the implemented retained description rises from 1250 to 1334 when the
  general ferry routines are charged. The manuscript now reports this directly.
- The partial campaign is described as resource-truncated rather than failure-truncated.
  Five runs were capped at level 4; preserved continuations end in interruption or
  exhausted credits rather than an architectural impossibility certificate. Complete
  histories also show that earlier time/credit boundaries were later crossed.
- The resulting inference is deliberately strong but finite: additional discovery
  compute is the immediate missing resource in the artifact record. No controlled
  scaling law was measured, so the manuscript does not claim that compute alone
  guarantees every remaining game.
- The abstract, research question, contribution list, approach section, empirical
  interpretation, validity discussion, and conclusion are aligned to this framing.

## Pass 11: solved checkpoints, executable contraction, and comparator credit

Status: checkpoint-timing counts below incorporate the stricter winning-action
fidelity correction made in Pass 12.

### Questions

1. Is the object called a checkpoint the solver that cleared a level, or merely an
   interim synthesis/commit/memory state?
2. Are cumulative executable description, marginal acquisition cost, operational
   reuse, and notebook memory being conflated?
3. Does OPINE-World receive credit for reusable retained model structure even where
   its winning policy came from the analyzer?
4. Does any comparator exhibit a strict level-to-level executable contraction after
   runtime dependencies and normalized AST are checked?
5. Which architecture most directly instantiates the Kolmogorov--Schmidhuber selection
   thesis, as distinct from which one has the most contractions or solved games?

### Evidence consulted

- GKM successful WIP snapshots and checkpoint ledgers, including
  `reached_before_debrief` sources and the deterministic auto-solve stub path.
- The complete OPINE run archive: `run_log.txt`, pre-reward synthesis ordering,
  `game_engine.py`, `l*.pkl` runtime data, and winning-plan provenance.
- The baseline1 GPT-5.5 xHigh per-game Git repositories, scorecards, authored source,
  normalized AST, execution logs, and scaffold.
- Retrodict transcripts, OpenTelemetry write/edit traces, final playbooks, scratch
  Python, and per-run solved-level metrics.

### Findings and decisions

- A single boundary rule now governs every system: retain the program or memory state
  that actually cleared the level. Failed/interim revisions are excluded. The previous
  OPINE all-revision profile is retained only as historical diagnostic code and is no
  longer manuscript evidence for a solved-level sawtooth.
- GKM has 33 exact winning sources across 37 clears; exact per-level source is absent
  for `ls20` L2 and `wa30` L1--L3 and is reported as missing rather than imputed.
  Twenty-four adjacent winning-source transitions all expand in compressed source.
  Ordinary wins use the captured pre-debrief source; four older auto-solve sources are
  reconstructed from the preceding retained source and the deterministic one-call
  player stub: `ft09` L2, `g50t` L3--L4, and `r11l` L3. The GKM sawtooth is therefore
  conditional novelty, not total source size.
- baseline1 has 160 post-solve retained snapshots representing 174 clears, but its
  commit occurs only at the start of the following iteration. Log alignment finds 50
  snapshots with no Python edit after the winning command and only 18 adjacent
  transitions whose two endpoints are exact. Four authored-source contractions survive
  that test and normalized-AST compression: `ar25` L4--L5, `cd82` L5--L6, `lp85`
  L6--L7, and `sb26` L6--L7. They retain 18--23 unchanged core definitions, and their
  winning turns execute retained model/planner machinery. The other seven contractions
  are post-solve-retained diagnostics, not exact winning-source evidence.
- OPINE has 153 positive-reward events, 146 with a pre-solve engine, and 121 adjacent
  measured transitions. Its retained code has two source/AST contractions, but only
  one remains after `l*.pkl` runtime dependencies are bundled; that win is
  analyzer-generated. Four wins are emitted by the synthesized planner, and none has a
  runtime-bundle contraction. Nevertheless `sb26`'s recursive rule, 47 unchanged
  planner transitions, and the identical `tr87` L5--L6 bundle are real retained-model
  reuse and are now credited explicitly.
- Retrodict has 170 solved memory checkpoints and 76 playbook-memory contractions.
  Twenty-three games contain no substantive scratch Python at any solve boundary; the
  other two show three scratch expansions and no contraction. Its released evidence is
  curated-memory reuse across context resets, not an executable-solver complexity
  trajectory.
- The manuscript now gives each comparator its strongest supported result. GKM remains
  the closest fit to the paper's Kolmogorov--Schmidhuber thesis because incumbent reuse
  is attempted before invention and marginal description cost is integrated into
  admission. This is a claim about selection structure, not leaderboard dominance or a
  denial that OPINE and baseline1 reuse or compress.

## Pass 12: winning-action fidelity, not merely post-solve retention

### Questions

1. Does each measured file bundle equal the source present at the action that cleared
   the level?
2. If an agent edits Python after the win but before a commit or debrief snapshot, has
   the audit mislabeled a retained state as the winning solver?
3. Can older auto-solve boundaries be reconstructed mechanically, without inferring a
   program from prose?
4. After enforcing exact endpoints on both sides, which cumulative contractions and
   reuse claims remain?

### Checks

- The baseline1 loop was followed from `iteration inspection` through Git snapshot,
  Codex turn, real winning command, post-win file changes, and `turn.completed`.
- Every one of its 174 clears was located from real client output or the guarded
  executor; simulator messages and status reads were excluded.
- GKM ordinary wins were moved from post-debrief snapshots to
  `reached_before_debrief`. For auto-solves, the analyzer replays the harness's
  deterministic transformation: take the preceding retained `players.py` and append
  the successful one-call `play_level_K` stub. Later debrief rewrites are excluded.
- The four surviving baseline1 contraction turns were inspected for shell writes as
  well as structured file changes; their post-win operations only read, verify, compile,
  or edit Markdown.

### Decisions

- The manuscript reports baseline1's 11 post-solve-retained contractions only as a
  diagnostic denominator. The exact result is four adjacent authored-source and AST
  contractions among 18 exact adjacent transitions.
- The manuscript reports 24, not 25, adjacent GKM transitions. The remaining measured
  jump crosses the missing `ls20` L2 source and is not a level-to-level datum.
- OPINE's pre-reward synthesis ordering already obeys the winning-action rule. Its
  engine-plus-runtime-state contraction remains on an analyzer-solved level, so the
  missing transient policy prevents a complete-solver claim.
- Retrodict's solve-boundary object remains curated memory. Its contractions are not
  relabeled as executable contraction merely because the memory helped later play.

## Pass 13: conditional novelty must meet a literal winning call

This pass supersedes Pass 10's use of the harness-native `43-to-2`,
`45-to-3`, and `100-to-2` charges as exact checkpoint-to-checkpoint evidence,
and Pass 11's attribution of baseline1's four retained-source contractions to
winning model reuse.

### Questions

1. At exact winning checkpoints, does the newly introduced executable AST
   become sharply smaller from one level to the next?
2. Does the later winning entry point directly call a named definition whose
   normalized AST is literally unchanged from the preceding winning
   checkpoint?
3. Do low marginals without such a call, or unchanged code not invoked by the
   winner, count as reuse?
4. Does the claim that OPINE, Retrodict, and baseline1 solve every level as a
   new task survive this joint test?

### Test

- For winning programs \(P_{L-1}\) and \(P_L\), normalize each top-level AST
  statement, remove the multiset of literal matches already present in
  \(P_{L-1}\), serialize the remainder, and compress it with zlib-9.
- Call the marginal drop sharp when the current value is at most half the
  preceding level's marginal.
- Attribute reuse only when the winning entry point directly calls an
  unchanged named definition. Static retention elsewhere in a workspace and a
  short action trace do not qualify.

### Findings and decisions

- GKM has 24 exact adjacent winning-source transitions. Twenty-two admit a
  level-to-level marginal comparison; 12 decrease and four are sharp. Nine
  winning players directly call unchanged leg literals. Two transitions
  satisfy both tests: `g50t` L4 drops from 2238 to 168 bytes and calls unchanged
  `solve_unlock_macro`; `ls20` L7 drops from 682 to 222 and calls unchanged
  `execute_path`.
- The other two sharp GKM drops, `ft09` L2 and `wa30` L9, have no unchanged
  direct leg call and are not labeled reuse. This falsifies any rule that
  equates a marginal trough with transfer automatically.
- OPINE has 121 adjacent pre-reward engine transitions. Of 115 comparable
  marginals, 49 decrease and 14 are sharp. Four synthesized-planner wins
  directly call unchanged engine definitions. Two are sharp coupled witnesses:
  `lp85` L4 drops from 5818 to 2550; its winning planner is literally identical
  to the L3 planner and directly calls unchanged `_cross_components`,
  `_cursor_pairs`, and `_square_blocks`. `tu93` L3 drops from 7091 to 2608 and
  directly calls unchanged `_find_player`, `_goal_topleft`, and
  `transition_function`.
- The categorical claim that OPINE solves every level anew is therefore false.
  Its hard reuse evidence is sparse but real. Twelve other sharp OPINE drops
  occur on analyzer-solved levels and are not assigned a literal executable
  reuse witness.
- baseline1 has 18 exact adjacent winning transitions and eight comparable
  marginal pairs. Five decrease, none by half. Every exact adjacent winning
  command is a fresh action program: four direct actions, six inline plans, and
  eight plans passed to `plan_executor.py`. None invokes a retained world-model
  definition. Its four exact retained authored-source/AST contractions remain
  artifact contractions, not demonstrated winning-solver reuse.
- Retrodict releases 170 solved memory checkpoints but no executable winning
  entry point. Its executable marginal and literal code-reuse count are both
  absent; its positive result remains curated-memory transfer.
- The manuscript, Markdown comparison notes, Sphinx documentation, and
  one-page summaries now use this joint test. The machine-readable evidence is
  `arc/audit_results/marginal-literal-reuse.json`, generated by
  `arc/audit_marginal_literal_reuse.py`.

## Pass 14: can the comparison table itself sustain the claim?

### Questions

1. Does the table distinguish a falling conditional marginal from an
   operational reuse witness?
2. Are all four sharp-drop/reuse intersections visible without reconstructing
   the argument from separate system discussions?
3. Does the presentation give OPINE its positive result while withholding the
   same label from baseline1's retained-source contractions and Retrodict's
   memory trajectory?
4. Is GKM's comparative claim limited to the measured exact set?

### Checks and decisions

- The JSON was asserted directly against the displayed counts: GKM has 12/22
  decreasing marginals, four sharp drops, and nine direct-call witnesses;
  OPINE has 49/115, 14, and four; baseline1 has 5/8, none, and zero across 18
  exact adjacent winning commands; Retrodict has no exact executable winning
  checkpoint.
- The four joint witnesses were checked by game, level, and byte pair:
  `g50t` L4 \(2238\to168\), `ls20` L7 \(682\to222\), `lp85` L4
  \(5818\to2550\), and `tu93` L3 \(7091\to2608\).
- The manuscript now states the asymmetric conclusion before the table and
  names the four coupled witnesses immediately after it. The table is fixed at
  the point of discussion rather than allowed to float away.
- OPINE is credited with hard level-to-level executable reuse. baseline1's
  four retained authored-source/AST contractions remain reported but are not
  called reuse because none of the 18 winning entry points calls retained
  world-model code. Retrodict remains a memory-transfer result because no
  executable winning entry point is released.
- “Strongest” is restricted to literal-leg evidence in the measured exact set;
  it is not a leaderboard, semantic-optimality, or compute-efficiency claim.
- No abstract material was added. The detail belongs in the
  solved-checkpoint comparison, where its definitions and artifact
  qualifications are available.

## Pass 15: does the architecture have a name before it has an acronym?

### Questions

1. What does a reader encounter before the first semantic use of `GKM`?
2. Is “Gödel–Kolmogorov Machine” repeated enough to become the architecture's
   name rather than a parenthetical expansion?
3. Does the name overclaim either Gödel-machine proof search or
   machine-independent Kolmogorov complexity?
4. Do the manuscript, one-page summary, repository landing pages, and
   comparator notes follow the same naming order?

### Checks and decisions

- In the manuscript, “Gödel–Kolmogorov Machine” now appears in the abstract,
  keywords, architecture definition, research-question answer, and contribution
  statement before the sentence that introduces `GKM`. The first later
  semantic use of the acronym is therefore licensed rather than assumed.
- The introduction explains the compound name. “Gödel” denotes inherited
  self-revision under a verification gate, while explicitly distinguishing
  empirical replay from Gödel-machine proof search. “Kolmogorov” denotes the
  description-length pressure and structure-function analysis, while explicitly
  disclaiming exact machine-independent complexity measurement.
- The acronym is retained after the full introduction because it is useful in
  tables and dense artifact comparisons. Repository paths and analyzer
  filenames containing `gkm` are identifiers and remain unchanged.
- The one-page summary uses the full name in its title, claim, and architectural
  comparison before switching to `GKM`. The root README, ARC README, Sphinx
  landing pages, audit README, OPINE comparison, `wa30` comparison, and outreach
  draft now introduce the full name before the abbreviation.
- Historical Socratic passes retain their original wording as a record of prior
  revisions; they are not treated as current manuscript exposition.
- The abstract grows only from 133 to 137 words and remains deliberately
  compact.

## Pass 16: does the forward revision preserve the name before abbreviation?

### Questions

1. Did the forward revision reintroduce `GKM` before the reader is told what it
   means?
2. Is the full name merely expanded once parenthetically, or does the prose
   establish it as the architecture's proper name before abbreviating it?
3. Does the later acronym reduce repetition without obscuring the paper's
   Gödel-machine and Kolmogorov-complexity commitments?
4. Did repository integration leave any surrounding documentation inconsistent
   with the paper's expanded endpoint table or corrected numerical scopes?

### Checks and decisions

- The canonical paper now says “Gödel–Kolmogorov Machine” in the abstract,
  keywords, architectural opening, and twice in the lineage discussion before
  the explicit abbreviation sentence following the contribution list. No prose
  use of `GKM` precedes that sentence.
- The transition is deliberate: the full-name passages explain replay-gated
  self-revision and Kolmogorov structure-function selection before the shorter
  label is used in section headings, tables, and dense comparisons. Lowercase
  `gkm` in stable repository filenames remains an identifier rather than an
  unexplained manuscript acronym.
- A mechanical first-occurrence scan and a PDF rebuild confirm that the source
  and rendered paper follow the same order. The compact abstract retains only
  the full name and does not spend words introducing an abbreviation it never
  reuses.
- The integrated paper distinguishes the official score, raw cleared-level
  coverage, stored-path actions, API actions, complete publication ledgers, and
  narrower operational checkpoint totals. Repository prose now also describes
  the five L4 artifacts as entries in the eight-endpoint summary while reserving
  “complete per-level history” for `wa30` and `ls20`.
- The forward manuscript remains consistent with the exact winning-checkpoint
  comparison: direct executable reuse is evidenced for the Gödel–Kolmogorov
  Machine and OPINE, not for baseline1 or Retrodict under the released-artifact
  test. This is an evidence claim, not an assertion of general superiority.

## Pass 17: is the acronym earned, and does the named architecture remain visible?

### Questions

1. Does the manuscript ever present `GKM` as though it were the architecture's
   primary name, rather than an abbreviation introduced after the reader knows the
   Gödel–Kolmogorov Machine?
2. Before the abbreviation is licensed, has the full name appeared often enough to
   connect the architecture to both Gödel-style gated self-revision and Kolmogorov
   description-length selection?
3. Once the paper turns to `GKM` for compact exposition, does the full name return at
   the main empirical claim and conclusion, where a reader may re-enter the argument?
4. Is the transition sentence ordinary academic prose, or does it sound like an
   editorial note about terminology management?

### Checks and decisions

- A source-order scan finds no uppercase standalone `GKM` before the explicit
  abbreviation sentence. Before that sentence, “Gödel–Kolmogorov Machine” appears in
  the abstract, keywords, architectural definition, and lineage discussion; the name
  is therefore established semantically rather than supplied as a parenthetical gloss.
- The former phrase “Having established the architecture's full name and scope” was
  too editorial. It is replaced by the direct academic formulation: “For concision in
  the remainder of the paper, we abbreviate the Gödel–Kolmogorov Machine as GKM.”
- After this transition, the acronym is used in headings, tables, equations, and dense
  comparisons where repetition would obstruct the prose. Stable lowercase repository
  identifiers such as `gkm_legs.py` remain code names, not manuscript abbreviations.
- The full name is restored at the opening of the decisive
  Kolmogorov–Schmidhuber evidence paragraph and at the conclusion. Thus the architecture
  remains named at major argumentative re-entry points without repeatedly re-expanding
  the acronym.
- The title and abstract contain no bare `GKM`. No scientific claim, numerical result,
  or comparator judgment was changed in this pass.

## Pass 18: did the expanded campaign change the evidence, or merely the scoreboard?

### Questions

1. Are the seven new clears incorporated as solved-level checkpoints, rather than as
   interim Codex turns or post-win debrief states?
2. Does the paper distinguish the historical `marginal_C` sawtooth from the stricter
   conditional normalized-AST comparator?
3. Does a new low ledger charge automatically become a sharp reuse result?
4. Are locally replayed clears attributed to the earlier public scorecard?
5. Has the prose retained an academic distinction between evidence, failure, and
   resource truncation?

### Checks and decisions

- The exact checkpoint export now contains 40 winning sources across 44 clears and 31
  exact adjacent transitions. Failed `sp80` L5 and `g50t` turns remain WIP and never
  enter the denominator. The cross-system analyzer was rerun from those checkpoint
  objects; no count was adjusted by hand.
- Of 29 comparable conditional AST marginals, 16 decrease and four fall by at least
  half. Ten winning entry points directly call unchanged definitions. The two sharp
  coupled witnesses remain `g50t` L4 and `ls20` L7; the OPINE coupled witnesses remain
  `lp85` L4 and `tu93` L3. The expansion therefore strengthens literal reuse without
  manufacturing a new sharp result.
- `ft09` L6 is the new direct witness. Its player calls the literally unchanged
  `solve_coupled_key_board`, while the conditional AST marginal decreases from 5008 to
  3730. The historical net source-growth charge falls from 177 to 2. The manuscript
  reports both values and says explicitly that the latter is not a substitute for the
  former.
- The public Competition-Mode card remains reported at 17.1365% and 37/183 raw clears.
  The current artifacts are separately reported at 44/183 and 1692 stored actions.
  Seven local replay certificates do not retroactively alter a closed public card.
- The expanded run is described as compute-bounded evidence: it began its sustained
  phase at 77% weekly allowance, stopped at 32%, and left the reset credit unused.
  Clean failed turns are reported as failed turns, not taint and not proof of
  impossibility. The resulting prose states what the artifacts establish without
  turning allowance consumption into a scientific virtue.
- The final taint rerun finds zero hits in 48 canonical promoted files and zero taint
  or integrity hits in every manifest-backed GPT-5.6 promotion chain. Ordinary mode
  passes. Strict complete-lineage mode fails only because older, pre-manifest artifacts
  have no promotion manifests; this is recorded as provenance incompleteness rather
  than mislabeled as source taint.
- “Gödel–Kolmogorov Machine” remains established before `GKM`; the title, abstract,
  category-theoretic sections, and conclusion remain consistent with the named
  architecture. The abstract was not lengthened.
