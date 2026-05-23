# Abstraction Emergence Experiment

This is the first internal predicate-library experiment. It is deliberately not a Bongard-LOGO result yet. It is a controlled scaffold for the narrower question: when a deterministic solver repeats the same hidden substructure across tasks or across branches of one disjunctive task, does free-energy accounting select a reusable predicate macro instead of duplicating the primitive rule body?

## Reproduction

```bash
python3 experiments/run_abstraction_emergence.py
python3 experiments/run_abstraction_emergence.py --scenario multi --show-rules
python3 experiments/run_abstraction_emergence.py --scenario or_factor --show-rules
python3 -m unittest tests.test_abstraction_emergence
```

## Setup

Each example is an opaque object represented by primitive observations only. The latent reusable substructure is not supplied as metadata:

```text
solid_loop = low_closure_error AND high_hull_fill AND turn_balanced
```

Task labels are deterministic conjunctions or disjunctions that reuse this substructure, for example:

```text
solid_loop_curve         = solid_loop AND has_curve
solid_loop_thin          = solid_loop AND thin
solid_loop_symmetric     = solid_loop AND symmetric_hint
solid_loop_many          = solid_loop AND many_segments
curve_or_thin            = has_curve OR thin
solid_loop_curve_or_thin = (solid_loop AND has_curve) OR (solid_loop AND thin)
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

Selection is by training free energy at each lambda, then validation loss and complexity across lambdas. Transfer rows reuse the library selected on support tasks and report marginal task cost for the shared condition. For disjunctive tasks, candidate macro mutations are generated from repeated atoms in the best inline DNF branch structure; this models an encapsulation mutation after an inline solution exists.

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
or_control,inline,0.0050,0.0000,0.0000,0.0000,1.000,1/1,3.00,0.0150,none
or_control,shared,0.0050,0.0000,0.0000,0.0000,1.000,1/1,3.00,0.0150,none
or_control,no_share,0.0050,0.0000,0.0000,0.0000,1.000,1/1,3.00,0.0150,none
or_control,oracle,0.0050,0.0000,0.0000,0.0000,1.000,1/1,2.00,0.0100,none
or_control_transfer,inline,0.0050,0.0000,0.0000,0.0000,1.000,1/1,5.00,0.0250,none
or_control_transfer,shared,0.0050,0.0000,0.0000,0.0000,1.000,1/1,5.00,0.0250,none
or_control_transfer,no_share,0.0050,0.0000,0.0000,0.0000,1.000,1/1,5.00,0.0250,none
or_control_transfer,oracle,0.0050,0.0000,0.0000,0.0000,1.000,1/1,2.00,0.0100,none
or_factor,inline,0.0050,0.0000,0.0000,0.0000,1.000,1/1,9.00,0.0450,none
or_factor,shared,0.0050,0.0000,0.0000,0.0000,1.000,1/1,7.70,0.0385,low_closure_error AND high_hull_fill AND turn_balanced
or_factor,no_share,0.0050,0.0000,0.0000,0.0000,1.000,1/1,9.00,0.0450,none
or_factor,oracle,0.0050,0.0000,0.0000,0.0000,1.000,1/1,2.00,0.0100,none
or_factor_transfer,inline,0.0050,0.0000,0.0000,0.0000,1.000,1/1,5.00,0.0250,none
or_factor_transfer,shared,0.0050,0.0000,0.0000,0.0000,1.000,1/1,2.35,0.0118,low_closure_error AND high_hull_fill AND turn_balanced
or_factor_transfer,no_share,0.0050,0.0000,0.0000,0.0000,1.000,1/1,6.35,0.0318,low_closure_error AND high_hull_fill AND turn_balanced
or_factor_transfer,oracle,0.0050,0.0000,0.0000,0.0000,1.000,1/1,2.00,0.0100,none
```

## Observations

1. Single-task pressure does not invent a macro. `shared` selects `none` and matches `inline` exactly: complexity 5.00, free energy 0.0250.
2. Multi-task pressure selects the reusable predicate. With three support tasks, `shared` discovers `low_closure_error AND high_hull_fill AND turn_balanced`, reducing complexity from 15.00 to 11.05 while keeping train, validation, and hidden loss at zero.
3. The OR control does not invent a macro. `curve_or_thin` solves as `(has_curve) OR (thin)` with complexity 3.00 in both inline and shared conditions.
4. The OR factoring task does invent the macro. `(solid_loop AND has_curve) OR (solid_loop AND thin)` solves inline with repeated core structure at complexity 9.00, while `shared` factors out `solid_loop` and lowers complexity to 7.70.
5. The no-share ablation kills both effects. When the predicate definition is paid per task call, selected solutions revert to inline structure and macro syntax provides no benefit.
6. Transfer behaves as expected. After either multi-task support or OR factoring learns the macro, a new `solid_loop_*` task is solved with marginal shared complexity 2.35 instead of inline complexity 5.00. The no-share transfer cost is 6.35 because the macro definition is paid again.
7. Oracle remains an upper bound, not a discovery condition. It solves everything with lower complexity because the task predicate is supplied directly.

## Interpretation

This is a positive internal control for abstraction emergence under the free-energy story. It does not show that Bongard-LOGO predicates have been evolved from pixels or even from raw LOGO programs. It shows a narrower mechanism working cleanly: when deterministic solvers repeat the same latent substructure across tasks or across OR branches, the complexity term can make a reusable predicate cheaper than duplicated inline code. Single-task, unrelated-OR, and no-share controls do not create the predicate.

The next stronger experiment should replace the enumerated predicate candidates with evolved finite-state predicate automata over action-program observations, then run the same support, no-share, and transfer controls on Bongard-LOGO symbolic programs.
