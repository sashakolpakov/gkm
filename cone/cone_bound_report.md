# Colimit-Cone v3: Priced Bindings and Within-Task Factoring

This is the v3 experiment (`COLIMIT_CONE_APPROACH.md` Section 11). It removes
the free rebinding that made within-task factoring untestable in v1/v2, by
putting the perceived channel into the rule key for inline solvers and
controllers (no `SET_FOCUS` register) while keeping legs channel-blind over a
caller-bound slot, and by pricing the binding a CALL supplies. The question:
does the true `or_factor` analogue now fire — within a *single* task, does a
slot-based leg that is genuinely *reused* beat an inline solver that must
duplicate its body per channel? And does the cone correctly *not* pay when a
task uses distinct behaviors that are not reused?

## Reproduction

```bash
python3 cone/run_cone_bound.py --evolved --binding-free-ablation --show-rules
python3 -m unittest tests.test_cone_foraging_bound
```

Outputs to `output/cone_bound/summary.json`. Substrate: `cone_foraging_bound.py`
(a sibling of `cone_foraging.py`; legs reuse `cone_foraging.ConeGenome`
unchanged, so naturality is preserved).

## Protocol

Two measurement modes, mirroring the abstraction-emergence experiment which
establishes accounting by candidate construction and uses search only for
behavior:

- **accounting** — fix the witness legs; construct the inline solver and the
  cone for each task; compare free energy directly. Isolates accounting from
  search noise. The headline claim lives here.
- **evolved** — evolve inline and gluing solutions at matched budgets, select
  by validation elbow; reported honestly with the R9 loss-confound.

Conditions: `inline` (no calls), `shared` (each used leg paid once, priced
calls), `no_share` (leg definition recharged per call). Binding cost default
0.5; the `binding_free` ablation sets it to 0.

Three scenarios, each a single task, chosen to span the prediction space:

```text
scenario           task              structure                role
single_bound       forage            one phase, one behavior   control: nothing to factor
two_phase_bound    forage_then_home  SAME seek leg twice       the or_factor headline
forage_flee_bound  forage_flee       seek once + flee once     control: distinct, single-use
```

## Result (seed 29, accounting mode, priced bindings)

```text
scenario,condition,train_loss,solved,complexity,free_energy,legs
single_bound,inline,0.0134,True,18.00,0.0674,none
single_bound,shared,0.0134,True,21.00,0.0764,witness_seek
single_bound,no_share,0.0134,True,21.00,0.0764,witness_seek
two_phase_bound,inline,0.0216,True,36.00,0.1296,none
two_phase_bound,shared,0.0205,True,23.00,0.0895,witness_seek
two_phase_bound,no_share,0.0205,True,42.00,0.1465,witness_seek
forage_flee_bound,inline,0.0176,True,36.00,0.1256,none
forage_flee_bound,shared,0.0174,True,42.00,0.1434,witness_seek+witness_flee
forage_flee_bound,no_share,0.0174,True,42.00,0.1434,witness_seek+witness_flee
```

The inline witness duplicates its motion body once per channel (seek toward
food/home, flee away from hazard); the cone calls channel-blind legs:

```text
inline (duplicated):   s0:FOOD:{9 rules} ; s1:HOME:{9 rules}   (forage_then_home)
cone (factored):       s0:FOOD:ANY -> CALL_0_FOOD / s1
                       s1:HOME:ANY -> CALL_0_HOME / s2
                       leg seek: s0:{az}->move/s0 ; s0:HERE->RETURN/s0   (paid ONCE)
```

## Observations

1. **Within-task factoring now pays — the v1/v2 reversal.** In
   `two_phase_bound/shared` the cone costs 23.0 versus 36.0 inline: a single
   task with two phases factors its *reused* seek behavior through one leg
   called twice. In v1/v2 this scenario selected no leg, because free
   rebinding let the inline solver share its seek body across phases for free.
   Removing free rebinding makes the duplication real and the pushout fires.
   Falsification criterion 1 passes.
2. **No-share kills it.** Recharging the leg definition per call makes the
   `two_phase_bound` cone 42.0, more expensive than inline 36.0. The effect is
   shared-description-length, not call syntax. Falsification criterion 3
   passes.
3. **Single-phase control does not factor.** `single_bound` has one behavior
   and nothing to duplicate; shared (21.0) does not beat inline (18.0).
   Falsification criterion 2 passes.
4. **Distinct single-use behaviors do not factor either — the second
   control.** `forage_flee_bound` uses two *different* legs (seek for food,
   flee for hazard), each called exactly once. No leg is reused, so the cone
   (42.0) loses to inline (36.0), and crucially `shared == no_share == 42.0`:
   when every leg is single-use, charging the definition once or per-call
   coincide, and neither beats inline. This is the within-task analogue of the
   `or_control` result — merely *using a library* never pays; only *reuse*
   pays. It is also the single-use property (Section 4 property 1) appearing
   inside a multi-leg task.
5. **Not a binding-pricing artifact.** With binding_cost = 0 the
   `two_phase_bound` cone is 22.0, with the priced binding 23.0 — factoring
   pays in both cases; pricing only shifts the cone by the two bound calls
   (1.0 total). And `forage_flee_bound` still loses at binding_cost = 0
   (shared 41.0 > inline 36.0). Falsification criterion 4 passes.
6. **Search confound, reported honestly (R9).** In evolved mode the inline
   solver fails to solve `two_phase_bound` at this budget (train loss 0.30,
   not solved): the keyed perception roughly triples the rule-key space and
   the duplicated two-phase seek is hard to evolve cold, while the gluing
   search solves it (the leg is fixed, only the 2-rule dispatcher must be
   found). The evolved comparison is therefore confounded — the cone wins
   partly on loss — which is exactly why the headline claim is made in
   accounting mode. Separately, this is more evidence for the search half of
   the thesis: the cone search space is far better-connected than keyed
   solver space. (In evolved `forage_flee_bound`, search found hybrid
   one-call-plus-inline solutions; evolved rows are search evidence only.)

## A Bug the Renderer Caught

The first version of the v3 witnesses used the *seek* leg for the hazard
phase — bound to the HAZARD channel, which points *toward* danger. The ASCII
renderer (`cone_render.py`) made this visible immediately: the agent walked
onto the `X` and halted unsafe. Fixed by giving the bound witness a separate
`flee_index` (mirroring the v1/v2 witness) and a flee variant of the inline
motion body (move away from the hazard azimuth). A regression test
(`test_flee_witness_moves_away_from_hazard`) now pins it. This is a small
vindication of building the visualization: a behavioral error invisible in
aggregate loss was obvious in one rendered trace.

## Interpretation

The v3 thesis holds, and the control structure is now sharper than v1/v2's.
With binding made an explicit priced morphism and free rebinding removed, the
cone accounting rewards within-task factoring exactly when there is genuine
*reuse* of one behavior (two_phase: seek twice), and *both* the no-share
control and the distinct-single-use control (forage_flee: seek once + flee
once) confirm the effect is shared description length under reuse, not library
use in general, not call syntax, and not a one-channel artifact. The
`forage_then_home` scenario, an `or_control` analogue in v1/v2, becomes the
`or_factor` analogue in v3 — matching what the abstraction-emergence
experiment found for boolean predicates, now reproduced for behaviors under a
substrate that does not hand the agent free abstraction over its targets.

The naturality story survives: legs are still channel-blind, the witness legs
seek/flee correctly under any binding, and the seek leg is identical to the
v1/v2 one. Only the *callers* lost their free focus register — the asymmetry
the design intended: the abstract object stays natural; the contexts that use
it must name and pay for their bindings.

## ARC-AGI-3 Correspondence

This substrate is the intended step toward ARC-AGI-3 reality. In an ARC game
the agent perceives many objects at once and "what I do to red cells" versus
"what I do to blue cells" are different rules keyed on different colours; they
do not share a symbol unless the agent constructs and pays for a parameterized
abstraction over the colour slot. The v1/v2 ambient focus hid exactly this by
letting one channel-blind rule-set serve every target for free. The v3 keyed
perception is the honest model, and the priced-binding leg is the
parameterized abstraction. The connector `arc_agi3_adapter.py` maps frame
colour-slots onto this channel model directly.

## Caveats

- Single seed; the accounting claim is deterministic (constructed solvers, no
  search), so seed-robustness matters only for the evolved confound, already
  flagged as confounded.
- Witness legs are hand-written (representability floors). The accounting
  claim is about *relative* description length of inline versus cone given the
  same legs, so it does not depend on evolved legs; an evolved-leg version is
  a natural hardening step.
- Evolved inline failure on `two_phase_bound` at this budget should be reduced
  (larger budget, or curriculum) before any evolved-mode accounting claim is
  made; the evolved rows are search evidence only.

## Next Steps

- Lifted/evolved seek and flee legs in place of the witnesses, to make the
  accounting claim fully search-grounded.
- Larger inline budget to lift the R9 confound and give an honest evolved-mode
  accounting comparison.
- The goal-induction loop (the agent does not know the task; it infers the
  goal from interaction and selects cones by free energy) — the next milestone
  for both this substrate and the ARC connector.
