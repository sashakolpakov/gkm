# Abstraction Emergence Experiment

This is the first internal predicate-library experiment. It is deliberately not a Bongard-LOGO result yet. It is a controlled scaffold for the narrower question: when a family of deterministic tasks shares a hidden substructure, does free-energy accounting select a reusable predicate macro instead of duplicating the same primitive rule body in every task?

## Reproduction

```bash
python3 experiments/run_abstraction_emergence.py
python3 experiments/run_abstraction_emergence.py --scenario multi --show-rules
python3 -m unittest tests.test_abstraction_emergence
```

## Setup

Each example is an opaque object represented by primitive observations only. The latent reusable substructure is not supplied as metadata:

```text
solid_loop = low_closure_error AND high_hull_fill AND turn_balanced
```

Task labels are deterministic conjunctions that reuse this substructure, for example:

```text
solid_loop_curve     = solid_loop AND has_curve
solid_loop_thin      = solid_loop AND thin
solid_loop_symmetric = solid_loop AND symmetric_hint
solid_loop_many      = solid_loop AND many_segments
```

The solver conditions are:

```text
inline   each task solves independently with primitive atoms
shared   a predicate macro may be defined once and called cheaply by task rules
no_share macro syntax is allowed, but the macro definition is paid per task call
oracle   privileged task predicate is supplied directly; upper bound only
```

The objective is:

```text
F = total task loss + lambda * (library complexity + task rule complexity)
```

Default lambda sweep:

```text
0.005, 0.01, 0.02, 0.04
```

Selection is by training free energy at each lambda, then validation loss and complexity across lambdas. Transfer rows reuse the library selected on support tasks and report marginal task cost for the shared condition.

## Result

Run on May 23, 2026:

```text
scenario,condition,lambda,train_loss,val_loss,hidden_loss,mean_hidden_acc,exact_hidden,complexity,free_energy,macro
single,inline,0.0050,0.0000,0.0000,0.0000,1.000,1/1,5.00,0.0250,none
single,shared,0.0050,0.0000,0.0000,0.0000,1.000,1/1,5.00,0.0250,none
single,no_share,0.0050,0.0000,0.0000,0.0000,1.000,1/1,5.00,0.0250,none
single,oracle,0.0050,0.0000,0.0000,0.0000,1.000,1/1,2.00,0.0100,none
single_transfer,inline,0.0050,0.0000,0.0000,0.0000,1.000,1/1,5.00,0.0250,none
single_transfer,shared,0.0050,0.0000,0.0000,0.0000,1.000,1/1,5.00,0.0250,none
single_transfer,no_share,0.0050,0.0000,0.0000,0.0000,1.000,1/1,5.00,0.0250,none
single_transfer,oracle,0.0050,0.0000,0.0000,0.0000,1.000,1/1,2.00,0.0100,none
multi,inline,0.0050,0.0000,0.0000,0.0000,1.000,3/3,15.00,0.0750,none
multi,shared,0.0050,0.0000,0.0000,0.0000,1.000,3/3,11.05,0.0553,low_closure_error AND high_hull_fill AND turn_balanced
multi,no_share,0.0050,0.0000,0.0000,0.0000,1.000,3/3,15.00,0.0750,none
multi,oracle,0.0050,0.0000,0.0000,0.0000,1.000,3/3,6.00,0.0300,none
multi_transfer,inline,0.0050,0.0000,0.0000,0.0000,1.000,1/1,5.00,0.0250,none
multi_transfer,shared,0.0050,0.0000,0.0000,0.0000,1.000,1/1,2.35,0.0118,low_closure_error AND high_hull_fill AND turn_balanced
multi_transfer,no_share,0.0050,0.0000,0.0000,0.0000,1.000,1/1,6.35,0.0318,low_closure_error AND high_hull_fill AND turn_balanced
multi_transfer,oracle,0.0050,0.0000,0.0000,0.0000,1.000,1/1,2.00,0.0100,none
with_direct,inline,0.0050,0.0000,0.0000,0.0000,1.000,4/4,19.00,0.0950,none
with_direct,shared,0.0050,0.0000,0.0000,0.0000,1.000,4/4,12.40,0.0620,low_closure_error AND high_hull_fill AND turn_balanced
with_direct,no_share,0.0050,0.0000,0.0000,0.0000,1.000,4/4,19.00,0.0950,none
with_direct,oracle,0.0050,0.0000,0.0000,0.0000,1.000,4/4,8.00,0.0400,none
with_direct_transfer,inline,0.0050,0.0000,0.0000,0.0000,1.000,1/1,5.00,0.0250,none
with_direct_transfer,shared,0.0050,0.0000,0.0000,0.0000,1.000,1/1,2.35,0.0118,low_closure_error AND high_hull_fill AND turn_balanced
with_direct_transfer,no_share,0.0050,0.0000,0.0000,0.0000,1.000,1/1,6.35,0.0318,low_closure_error AND high_hull_fill AND turn_balanced
with_direct_transfer,oracle,0.0050,0.0000,0.0000,0.0000,1.000,1/1,2.00,0.0100,none
```

## Observations

1. Single-task pressure does not invent a macro. `shared` selects `none` and matches `inline` exactly: complexity 5.00, free energy 0.0250.
2. Multi-task pressure selects the reusable predicate. With three support tasks, `shared` discovers `low_closure_error AND high_hull_fill AND turn_balanced`, reducing complexity from 15.00 to 11.05 while keeping train, validation, and hidden loss at zero.
3. The no-share ablation kills the effect. When the predicate definition is paid per task call, the selected solution reverts to `none` and matches inline complexity. This is the key control: the result is driven by shared description length, not by macro syntax alone.
4. Transfer behaves as expected. After the multi-task support set learns the macro, `solid_loop_many` is solved with marginal shared complexity 2.35 instead of inline complexity 5.00. The no-share transfer cost is 6.35 because the macro definition is paid again.
5. Oracle remains an upper bound, not a discovery condition. It solves everything with lower complexity because the task predicate is supplied directly.

## Interpretation

This is a positive internal control for abstraction emergence under the free-energy story. It does not show that Bongard-LOGO predicates have been evolved from pixels or even from raw LOGO programs. It shows a narrower mechanism working cleanly: when multiple deterministic tasks reuse the same latent substructure, the complexity term can make a reusable predicate cheaper than duplicated inline solvers, while single-task and no-share controls do not create the predicate.

The next stronger experiment should replace the enumerated predicate candidates with evolved finite-state predicate automata over action-program observations, then run the same support, no-share, and transfer controls on Bongard-LOGO symbolic programs.
