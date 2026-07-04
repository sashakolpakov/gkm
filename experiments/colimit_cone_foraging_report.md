# Colimit-Cone Foraging Experiment

Experimental tests of the colimit-cone approach (design, definitions, and
pitfalls in `COLIMIT_CONE_APPROACH.md`): can free-energy selection with
sum-over-legs / shared-leg-discounting cone accounting build a behavior
library only when reuse pays; does the cone buy search as well as description
length; do leg interfaces (`RETURN` boundaries) emerge under compositional
pressure; what gates leg quality; and does a library grow as a sequential
colimit?

All numbers below are from substrate v2 (the HAZARD channel and flee task
family are present; v1 scenarios carry no hazards but share the enlarged
action space). The v1 numbers from the original two-channel substrate are
superseded; the qualitative pattern reproduced.

## Reproduction

```bash
python3 experiments/run_colimit_cone_foraging.py --population 120 --generations 60 --joint --show-rules
python3 experiments/run_colimit_cone_foraging.py --scenario single --growth --population 120 --generations 60 --show-rules
python3 experiments/run_cone_leg_robustness.py
python3 -m unittest tests.test_cone_foraging
```

Outputs are written to `output/colimit_cone/`.

## Protocol

- Substrate: `cone_foraging.py` — multi-channel grid foraging (FOOD, HOME,
  HAZARD compasses). Machines are sparse FSAs keyed by
  `(state, azimuth-or-ANY)`; library legs read a channel bound at call time
  and terminate only by an explicit costed `RETURN`; controllers glue legs
  into tasks via `CALL(leg, channel)` actions.
- Tasks: `forage`, `homing`, `forage_then_home`, plus the hazard family
  `flee`, `forage_flee`, `flee_then_home` (the HAZARD channel reads HERE when
  no hazard is within `SAFE_RADIUS = 3`, so flee legs share the seek
  interface: act until the channel reads HERE).
- Cone accounting (`COLIMIT_CONE_APPROACH.md`, Section 4): `shared` pays each
  used leg's definition once plus `call_cost = 0.5` per call; `no_share` pays
  the full definition at every call; `inline` forbids calls; `witness` rows
  are hand-written floors (never part of any discovery claim). Lambda sweep
  `{0.001, 0.003, 0.01}`, validation-elbow selection, hidden levels evaluated
  once after selection. Disjoint train/validation/hidden level sets.
- Leg sources, in increasing order of autonomy: *lifted* (mechanically
  encapsulated from an evolved inline champion — discovered behavior,
  hand-designed lift operator), *cold joint* (`--joint`: leg bodies and
  gluings co-evolved from random initialization under the true joint
  accounting).
- Diagnostics: `call_champion_seen` (pitfall P5 — were calls ever competitive)
  and `leg_return` (does the selected leg carry a `RETURN` boundary).

## Headline Accounting Result (seed 29)

```text
scenario,condition,lambda,train_loss,val_loss,hidden_loss,train_solved,val_solved,hidden_solved,complexity,free_energy,leg,static_calls,leg_return
single,inline,0.0010,0.0191,0.0182,0.0233,1/1,1/1,1/1,11.00,0.0301,none,0,False
single,shared,0.0010,0.0191,0.0182,0.0233,1/1,1/1,1/1,11.00,0.0301,none,0,False
single,no_share,0.0010,0.0191,0.0182,0.0233,1/1,1/1,1/1,11.00,0.0301,none,0,False
multi,inline,0.0010,0.0233,0.0233,0.0299,2/2,2/2,2/2,21.00,0.0443,none,0,False
multi,shared,0.0030,0.0265,0.0222,0.1067,2/2,2/2,1/2,18.00,0.0805,lift_homing,2,True
multi,no_share,0.0010,0.0233,0.0233,0.0299,2/2,2/2,2/2,21.00,0.0443,none,0,False
two_phase,inline,0.0030,0.0189,0.0173,0.0186,1/1,1/1,1/1,16.00,0.0669,none,0,False
two_phase,shared,0.0030,0.0189,0.0173,0.0186,1/1,1/1,1/1,16.00,0.0669,none,0,False
two_phase,no_share,0.0030,0.0189,0.0173,0.0186,1/1,1/1,1/1,16.00,0.0669,none,0,False
multi_transfer,inline,0.0010,0.0259,0.1496,0.3221,1/1,0/1,0/1,26.00,0.0519,none,0,False
multi_transfer,shared,0.0100,0.0244,0.3223,0.0371,1/1,0/1,1/1,2.00,0.0444,lift_homing,2,True
```

The v1 pattern reproduces: no cone under single-task pressure, a cone under
cross-task reuse (complexity 18 versus 21), no cone under the no-share
ablation or in the free-rebinding `two_phase` control, and a transfer gluing
of marginal complexity 2.00 versus 26.00 inline.

## Compositional Pressure: RETURN Emerges (resolves R10)

The v1 cold-evolved leg had dropped its `RETURN` boundary, because nothing in
the `multi` support set composed behavior after a call. Scenario `multi_seq`
puts the sequencing task inside the cold joint support set
(design document Section 9.1). Result, seed 29:

```text
multi_seq,inline,0.0010,0.0449,0.6427,0.7393,3/3,2/3,2/3,35.00,0.0799,none,0,False
multi_seq,shared,0.0030,0.0568,0.0537,0.1505,3/3,3/3,1/3,21.00,0.1198,lift_homing,4,True
multi_seq,shared_joint,0.0100,0.0867,0.0705,0.0763,3/3,3/3,3/3,20.00,0.2867,joint_evolved,4,True
multi_seq,witness,0.0030,0.0371,0.0318,0.0358,3/3,3/3,3/3,25.00,0.1121,witness_seek,4,True
```

The selected cold cone:

```text
[forage]            s0:ANY -> CALL_0_FOOD / s2
[homing]            s0:ANY -> CALL_0_HOME / s1
[forage_then_home]  s0:ANY -> CALL_0_FOOD / s2
                    s2:HERE -> CALL_0_HOME / s0
[leg:joint_evolved] s0:E -> RIGHT / s0
                    s0:SE -> DOWN / s0
                    s0:ANY -> UP,LEFT / s1
                    s1:HERE -> RETURN / s1
                    s1:S -> DOWN / s1
                    s1:ANY -> STAY / s0
```

The boundary rule `s1:HERE -> RETURN` was *evolved*, not installed: with a
consumer that composes a second call after the first, the interface pays for
itself. The cone solves all three tasks on all panels (3/3 train, validation,
hidden) at complexity 20.0, against 35.0 for inline solvers that fail
validation. Across the six seeds, `multi_seq/shared` selects a leg in 5/6,
and every selected leg — lifted or cold — carries `RETURN`
(`leg_return = True`), while in plain `multi` cold legs may drop it. The
prediction of design Section 9.1 is confirmed: interfaces are selected
exactly when the ecology contains composition.

## Replicate Sweep (six seeds: 29, 101, 202, 303, 404, 505)

```text
prediction                          seeds confirming
single: no leg, shared & no_share    6/6
two_phase: no leg, shared & no_share 6/6
multi: leg selected in shared        4/6   (two seeds: inline already compact)
multi: no leg in no_share            6/6
multi_seq: leg selected in shared    5/6
multi_seq: leg has RETURN            5/5 of selections
multi_seq: no leg in no_share        4/6   (see loss-confound note below)
```

Budget-matched discovery on the transfer task (gluing search with the frozen
selected leg versus inline search; run only where a library was selected):

```text
seed     29    202   404   505    total
gluing   0/5   0/5   5/5   4/5    9/20
inline   0/5   0/5   0/5   0/5    0/20
```

Loss-confound note (reconciliation log R9): in `multi_seq`, inline evolution
frequently fails validation (a three-task budget problem), and in 2/6 seeds
`no_share` then selects a calling solution *on loss*, not on accounting —
calls win even at full per-call price when inline search cannot match their
behavior. Wherever inline solves the support set, the no-share control is
clean (12/12 across `single`, `two_phase`, `multi`). This is the
basin-connectivity claim showing up inside a control: when tasks get hard,
cones win on reachability before accounting even matters.

## Leg Robustness: Naturality Predicts Everything

`experiments/run_cone_leg_robustness.py`, six seeds. Three leg sources per
seed — lifted from the inline forage champion (vetted by 1 task), cold joint
on `{forage, homing}` (2 tasks), cold joint on the `multi_seq` set (3 tasks,
including a sequencing consumer; this source has seen the transfer task
family during evolution and is flagged in-distribution) — each evaluated on a
naturality probe (the same leg called under FOOD and HOME bindings over
hidden levels) and on budget-matched transfer discovery.

```text
source   probe passes   discovery (total)   has RETURN
lifted   2/6            8/30                6/6 (installed by the lift)
joint    0/6            0/30                4/6 (evolved)
seq      4/6            14/30               6/6 (evolved)
inline   —              0/30                —
```

Findings:

1. **The naturality probe predicts discovery perfectly.** Every
   (source, seed) cell whose leg passes the probe achieves at least one
   discovery success; every cell that fails the probe achieves zero. Probe
   losses are bimodal (about 0.01–0.07 for natural legs, 0.12–0.70 for junk).
   Naturality — behaving correctly under *both* bindings — is the measurable
   quality that makes a diagram object reusable.
2. **Vetting breadth helps, with a wrinkle.** Sequencing-vetted legs are the
   best source (4/6 natural) and lifted legs intermediate (2/6), but two-task
   cold joint legs failed in all six v2 seeds — worse than v1, where the same
   budget found the zigzag leg. The enlarged v2 action space makes
   cold co-discovery (controller must find CALL while the leg is still junk)
   harder, and two tasks' pressure no longer suffices at this budget; three
   consumers, including one that *requires* a working boundary, pull the leg
   into shape. Reuse pressure is not only what pays for the cone — it is what
   steers the search toward natural legs.
3. **Inline search never discovers the transfer task at this budget** (0/30),
   reproducing the v1 contrast wherever a natural leg exists.

## Library Growth as a Sequential Colimit

Staged protocol (design Section 9.3): phase 1 cold-evolves a seek leg on the
sequencing support set; phase 2 freezes it (definition already paid,
`free_legs` marginal accounting) and cold-evolves one new leg plus gluings on
the hazard family `{flee, forage_flee, flee_then_home}`, against no-legacy,
no-share, inline, and witness conditions.

The first growth run produced an honest negative with a sharp lesson. At the
growth seed, every phase-1 attempt produced a *non-natural* legacy leg (one
even encoded `HERE -> STAY` — a leg that never returns), and phase 2 then
walked into a trap the robustness study predicts: the already-paid junk leg
is *free*, so `shared_growth` selected it on price (marginal complexity 10.5)
and failed generalization (1/3 hidden), while `shared_no_legacy`'s fresh leg
solved everything (3/3 everywhere at 15.5). An inherited library is only an
asset if its legs are natural; a discounted junk leg is worse than no library.

The protocol now includes **library admission**: phase-1 candidate legs
(lambda sweep, two restarts each) must pass the naturality probe on
validation levels before being frozen into the library; admission failure is
reported rather than silently shipped. With admission in place:

```text
scenario,condition,lambda,train_loss,val_loss,hidden_loss,train_solved,val_solved,hidden_solved,complexity,leg,leg_return
growth_phase1,shared_joint,0.0030,0.0384,0.0330,0.0520,3/3,3/3,2/3,30.50,legacy_seek,True
growth_phase2,shared_growth,0.0100,0.0199,0.0247,0.3402,3/3,3/3,2/3,5.50,legacy_seek,True
growth_phase2,shared_no_legacy,0.0100,0.0420,0.0511,0.0464,3/3,3/3,3/3,15.50,new_leg,False
growth_phase2,no_share_growth,0.0010,0.0199,0.0247,0.0218,3/3,3/3,3/3,54.50,legacy_seek,True
growth_phase2,inline,0.0010,0.0273,0.6145,0.8622,3/3,2/3,0/3,26.00,none,False
growth_phase2,witness,0.0030,0.0235,0.0259,0.0241,3/3,3/3,3/3,45.50,witness_seek+witness_flee,True
```

The selected `shared_growth` gluings:

```text
[flee]            s0:ANY -> RIGHT / s0
[flee_then_home]  s0:ANY -> CALL_0_HOME / s0
[forage_flee]     s0:ANY -> CALL_0_FOOD,CALL_0_HOME / s0
```

Three observations:

1. **Cross-generation reuse is real and cheap.** The admitted legacy seek leg
   (evolved in phase 1 on food/home tasks) is called by two of the three
   hazard-family gluings under new bindings, at marginal complexity 5.50
   versus 15.50 for the no-legacy fresh leg and 26.00 for inline — the
   sequential-colimit prediction: the marginal cost of new competence falls
   as the library grows.
2. **Evolution found a creative — and slightly leaky — gluing instead of a
   new leg.** No flee leg was encapsulated. `forage_flee` is solved as
   `CALL(seek, FOOD), CALL(seek, HOME)`: walking to the home cell is used as
   a proxy for "away from the hazard", since homes are usually far from
   hazards. This passes train and validation but costs a hidden panel (2/3):
   the proxy is not the concept. With marginal accounting, a free legacy leg
   under an inventive binding outcompeted paying for a genuinely new
   behavior — the cheap-reuse trap in a subtler form than the junk-leg trap,
   and a concrete instance of shortcut pressure in cone space.
3. **The no-legacy ablation is the stronger generalizer here** (3/3 hidden):
   forced to pay for a fresh leg, it bought the real flee behavior. The
   library helped on price, not on truth — which says the next pressure to
   add is exactly the one the internal Bongard work used: counterexample-rich
   panels (here: levels where home is near the hazard), so the proxy gluing
   fails training rather than hidden evaluation.

## Caveats

- Six seeds, five discovery replicates per cell; per-cell confidence
  intervals would need more.
- `multi/shared` at seed 29 buys its complexity reduction at a hidden-panel
  cost (1/2): the lifted leg overfits slightly. The validation elbow cannot
  see hidden panels by design; this is the expected price of a one-candidate
  library, and the robustness study quantifies it.
- The `multi_seq` no-share control is loss-confounded in 2/6 seeds (see
  above); accounting conclusions there are conditional on inline search
  succeeding, exactly as reconciliation log R9 requires.
- `flee_then_home` is geometrically solvable by seeking home directly (homes
  generate outside the hazard radius), so composition evidence in the growth
  phase must come from `forage_flee`.
- Hazard placement guarantees escape room for a minimal antipodal flee leg
  (design Section 9.3); evolved legs may handle harder geometry, witnesses
  need not.

## Interpretation

The accounting claims of v1 survive replication on the extended substrate:
cones form only under genuine cross-task reuse, the no-share ablation removes
them wherever the control is clean, transfer through a frozen library is
nearly free, and gluing search discovers tasks that solver search cannot
(9/20 versus 0/20 at matched budgets, gated as predicted by leg quality).

The new results sharpen the picture in three ways:

> **Interfaces are ecological.** The `RETURN` boundary is selected exactly
> when the support set contains a consumer that composes behavior after the
> call (`multi_seq`), and is dropped when it does not (`multi`).

> **Naturality is the quality gate.** A leg's behavior under *both* bindings
> — the probe version of functoriality — perfectly predicts whether the cone
> built on it transfers. Reuse pressure during leg evolution is what produces
> naturality; single-task champions mostly do not lift into natural legs.

> **A library is not automatically an asset.** Marginal accounting makes
> inherited legs free, and a free unnatural leg actively poisons growth.
> Library admission must be vetted — which is the operational content of
> requiring diagram objects to be well-behaved before extending along them.

## Next Steps

- Growth with vetted admission across multiple seeds; report cross-generation
  reuse rates (legacy calls in phase-2 gluings).
- Scale the joint budget study: how much reuse pressure does cold leg
  discovery need as the action space grows?
- Counterexample-rich growth levels (homes generated *near* hazards) so the
  `CALL(seek, HOME)` flee-proxy fails in training, forcing a genuine flee leg
  — the level-design analogue of the hard negatives that fixed the
  underconstrained Bongard panels.
- A task where composition is geometrically forced for the flee leg
  (`forage_flee` analogues with multiple hazards), making the flee boundary
  as load-bearing as the seek boundary.
- The ARC-AGI-3 adapter discussion: the substrate now has the full
  discipline — cold leg discovery, naturality admission, marginal growth
  accounting — that the adapter would inherit.
