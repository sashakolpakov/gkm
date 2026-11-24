# Repository Ground-Truth Matrix

This matrix records how implementation facts constrain the manuscript. "Ground truth" here means the public repository's code and generated documentation, not an editorial preference.

| Topic | Repository source or code surface | Ground-truth observation | Manuscript consequence |
|---|---|---|---|
| Frozen release | Schema-1 partial-release receipt and generated ARC artifact status | 181 of 183 authoritative levels are admitted across all 25 games; only `lf52` L9--L10 are unclaimed | Report the receipt-bound release, not the mutable campaign workspace |
| Universal producer | Proposer contract, Arena, blank scaffold, complexity ledger, replay gate | The producer and admission system are shared across all games; learned executable outputs are game- and level-dependent | Do not confuse output specialization with a non-general producer |
| Official score | Root `README.md` and linked Competition-Mode scorecard | Official all-game score is `98.11664037825032%` across 25 public games | Report separately from raw level coverage |
| Raw coverage | Release receipt and generated endpoint table | 181 of 183 levels are cleared | Report separately as `98.907103825137%` |
| Replay action accounting | Certified checkpoint paths and scorecard | Stored paths total 7001; scorecard API calls total 7069 including resets | Distinguish stored actions from API actions |
| Uniform histories | Manuscript sidecars/history builder | `wa30` is 9/9 with 597 replay actions and ledger 318; `ls20` is 7/7 with 365 replay actions and ledger 760 | Use these histories for levelwise scalar profiles and exact source interpretation |
| Source metric | Frozen producer and source audit | D measures retained source; recorded C compares against the working source saved at the start of the winning iteration; G independently compares adjacent retained states. Source units count nonblank noncomment lines plus AST literal elements | Do not identify the workspace baseline with the preceding promotion or infer a harness error from C differing from G |
| Meaning of low marginal charge | Frozen source audit | A low scalar alone does not establish reuse; a static call site and replay are separate observations | Report static call-site candidates; runtime execution is unmeasured |
| Replay semantics | `replay_scorecard.py` | Load checkpoint `final_path`, segment by local replay, replay remotely without LLM discovery, compare reached depth, fail on desynchronization | Treat scorecard evidence as endpoint replay, not discovery efficiency |
| Local versus official interface | Root exact-claim text; ARC wrapper | Local harness exposes clone-based lookahead; official wrapper exposes reset/step rather than arbitrary state forking | Separate proposal/search cost from official replay action cost |
| Leg architecture | `gkm_legs.py`, promoted `legs.py`, `players.py`, dispatcher | Reusable legs are separated from thin level players and dispatch | Model source growth as cells plus bindings rather than a monolithic program |
| Frozen proposal order | `gkm_legs.py`, frozen campaign policy and queue | Incumbent legs are tried before invention; the proposer is asked for minimal glue; the first exact replay and provenance valid candidate is promoted | Describe a complexity sensitive construction path, not a tournament among replay valid winners |
| Frozen retention | Promotion and seeding code | The complete promoted solver tree becomes the parent for the next level | Explain how one successful pushout changes the diagram available to later pushouts |
| Frozen game allocation | Campaign status and policy code | Clean retry history controls the main order and effort ladder; retained source size contributes a heuristic priority; `F_diag` is not consumed | Keep compute allocation separate from program admission and from the later contiguous scheduler |
| LLM binding role | `llm_binder.py`, generated player code, source diffs | The proposer supplies new calls, argument bindings, sequencing, and residual glue | Interpret the LLM as supplying attaching morphisms and, when needed, new objects/interfaces |
| Cofibrant role | `cofibrant.py`, source history discipline | Additive extension preserves incumbent definitions and interfaces when reuse is claimed; the generic harness can validate refactors, but the frozen campaign disabled its debrief turn | Use cofibrations for literal source persistence; do not attribute adaptive refactoring to the frozen campaign |
| Gödel-style gate | `gkm_godel.py` and repository findings | The LLM writes a candidate modification; restricted compilation/execution evaluates it; adoption requires measured game improvement | Present the Gödel–Kolmogorov Machine as empirical Gödel-style self-revision under an external gate |
| PowerPlay | `powerplay.py`, findings, clean promotion archive | Search operates on task/solver modifications and retains successful competence; simpler reusable structure is preferred | Make PowerPlay the architectural curriculum and retention principle |
| Free energy | `FREE_ENERGY_EXPLANATION.md`, `gkm_legs.py`, frozen campaign policy | The theory supports local free energy selection and lambda sweeps. In the frozen campaign, `C_k` and `F_diag` were stored diagnostics and did not admit or rank programs | Keep a concise background account and distinguish it from the historical selection mechanism; do not impose one scalar weighting across unlike games |
| Colimit-cone program | `COLIMIT_CONE_APPROACH.md`, cone/ARC architecture | Shared source can be retained while later players add bindings; the implementation does not compute Kan extensions | Use the pushout as a source specification and keep the full theory separate |
| Artifact histories | `build_artifact_history.py`, immutable sidecars, Git promotions/WIP | Clean promotions, sidecars, and WIP snapshots jointly reconstruct chronological acquisition and debrief | Treat the provenance zigzag as a scientific object |
| Taint boundary | ARC harness checks and preserved dirty WIP | Known private runtime/source access is blocked; tainted workspaces are excluded; this is not a formal information-flow theorem | State the implemented security boundary precisely without weakening the solver-growth claim |
| Comparator audit | `audit_*_artifacts.py`, `audit_marginal_literal_reuse.py` | Exact solve boundaries, normalized-AST marginals, and winning-entry-point calls are kept separate | Preserve the stronger source-level reuse analysis and avoid equating retention with use |
| Reproducibility commands | `REPRODUCE_ARC.md`, root README | History check, offline tests, and zero-LLM online/competition replay address different layers | Publish all three reproduction layers in the protocol section |

The 2026-11-23 revision narrows exact-source coverage to 171 checkpoints, 144 adjacent
transitions, and 141 marginal comparisons. The earlier seven path-reconstruction
exclusions are joined by `g50t` L3/L4 and `r11l` L3 because their paid-debrief snapshots
do not authenticate pre-debrief source. There are 54 static unchanged-leg call-site
cases, including 13 with sharp novelty drops; actual calls are unmeasured. The
`dc22` recorded total stays 650. On authenticated L2–L6 pairs, C sums to 601 and
G to 801 while D grows from 50 to 851; no primary G1 is inferred from a template.
The working baseline explains the different C/G charges, while the incomplete audit
reconstruction was corrected without modifying the solver or counters.

The separate promoted-state audit has 158 manifest-bound states and 134 pairs in
24 games. Of 20 games with enough consecutive paired values, 18 reverse in both
series; aggregate C/G reversals are 45/43. The only unequal primary value is dc22
L5. Wa30 reverses twice in both on L2–L9; ls20 has one final reversal in both on
L2–L7. Initial values, missing pairs, and unauthenticated phase candidates do not
enter primary reversal counts. See `generated/source_charge_comparison.json` and
`REVISION_EVIDENCE.md`.

The central empirical analysis is the conditional marginal proxy M, not the C/G
source-growth comparison. Its 25-game figure and per-level JSON preserve missing
exact sources, predecessor gaps and unclaimed levels. dc22 has marginal dips at L2
and L5 despite increasing G. Both dc22 dips and both wa30 troughs have source paths
through new routines to unchanged helpers, as documented by the five-case audit.
cn04 L3 instead has a 23-action player calling only env.step. The direct-call
statistic excludes the intermediate wrappers and retains its fixed scope.
