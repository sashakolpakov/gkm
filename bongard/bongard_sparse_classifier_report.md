# Sparse Bongard Classifier Report

This experiment is Stage 2 of the Bongard-first plan. It evolves sparse deterministic classifiers for Bongard-style positive/negative concept problems over opaque-object sequences.

Run from the repository root:

```bash
python3 experiments/run_bongard_sparse_classifier.py
```

Larger panels can be requested without editing code:

```bash
python3 experiments/run_bongard_sparse_classifier.py --train-count 12 --validation-count 8 --hidden-count 24 --replicates 3
```

## Protocol

- Examples are positive/negative opaque-object sequences.
- Train, validation, and hidden-test splits use disjoint object pools.
- Train, validation, and hidden-test splits are counterexample-rich for concepts where random examples permit shortcuts.
- Initial populations are clean-slate random sparse rule tables. No precomputed programs, solver templates, seeded boundary rules, or target-specific initialization are inserted.
- Evolution minimizes training free energy:

```text
F_lambda(classifier) = classification_loss_train + lambda C_code
```

- `C_code` is expanded sparse program size: each encoded rule pays for its action sequence plus a rule overhead.
- Archive training can add currently misclassified counterexamples to the training loss during evolution; these are not solver templates or seeded rules.
- A lambda warmup can make early search loss-dominated, then bring in complexity pressure while evolution is still operating.
- Selection preserves a small loss frontier alongside free-energy elites so useful higher-dimensional intermediates are not eliminated before the archive exposes their advantage.
- Lambda selection uses the validation loss/complexity elbow.
- Hidden test is reported once after validation selection.
- Every configured concept is also evaluated on an exhaustive foreign-alphabet probe over all sequences of lengths 2 through 6 on three held-out objects.
- Balanced accuracy is reported for diagnostics because some advanced concepts are label-imbalanced on exhaustive probes.
- An exact discovery means train accuracy, validation accuracy, hidden accuracy, and exhaustive probe accuracy are all `1.00` after lambda selection.

## Expressivity Check

The `bidirectional_compare` automaton can express "scan until the end." The observation function emits `EOS` when the cursor reaches the end of the input, and the classifier action interpreter can execute `MOVE_RIGHT` until that happens, then execute `MOVE_LEFT` from an `EOS` rule. A unit test now constructs the exact `first_equals_last` classifier and verifies it on the exhaustive probe.

So `first_equals_last` is not an impossible task for the substrate. Failure to discover it is an optimization/search result.

## Separate Clean-Slate Experiments

The runner now treats each concept as a separate experiment with its own search parameters. Use `--concept` to run a single category:

```bash
python3 experiments/run_bongard_sparse_classifier.py --concept length_even
python3 experiments/run_bongard_sparse_classifier.py --concept has_adjacent_duplicate
python3 experiments/run_bongard_sparse_classifier.py --concept first_equals_last
```

The default `first_equals_last` configuration is intentionally larger than the easy categories: more states/search budget, more replicates, larger panels, and single-action rules because the known representable exact solution only needs single-action transitions. This reduces irrelevant macro-action search without inserting any target rule.

### `length_even`

Command:

```bash
python3 experiments/run_bongard_sparse_classifier.py --concept length_even
```

Result: `8/8` exact discoveries.

```text
concept,primitive,runs,exact_discoveries,mean_train_acc,mean_val_acc,mean_hidden_acc,mean_probe_acc,mean_complexity
length_even,stream,8,8,1.000,1.000,1.000,1.000,6.2
```

### `has_adjacent_duplicate`

Command:

```bash
python3 experiments/run_bongard_sparse_classifier.py --concept has_adjacent_duplicate
```

Result: `8/8` exact discoveries.

```text
concept,primitive,runs,exact_discoveries,mean_train_acc,mean_val_acc,mean_hidden_acc,mean_probe_acc,mean_complexity
has_adjacent_duplicate,compare,8,8,1.000,1.000,1.000,1.000,5.5
```

### `first_equals_last`

Negative control: larger clean genetic search without archive pressure still found sampled near misses rather than the exact rule.

```bash
python3 experiments/run_bongard_sparse_classifier.py --concept first_equals_last --replicates 4 --population 700 --generations 450 --states 3 --initial-rules 6 --max-rules 12 --max-rule-length 1 --lambda-points 1 --train-count 16 --validation-count 12 --hidden-count 32 --mutation-rate 0.12 --no-archive-training --stop-after-discovery
```

```text
concept,primitive,runs,exact_discoveries,mean_train_acc,mean_val_acc,mean_hidden_acc,mean_probe_acc,mean_complexity
first_equals_last,bidirectional_compare,4,0,1.000,0.979,0.953,0.621,20.0
```

Counterexample-archive search with late lambda warmup and loss-frontier preservation:

```bash
python3 experiments/run_bongard_sparse_classifier.py --concept first_equals_last --replicates 1 --population 700 --generations 450 --states 3 --initial-rules 6 --max-rules 12 --max-rule-length 1 --lambda-min 0.0001 --lambda-max 0.0001 --lambda-points 1 --train-count 16 --validation-count 12 --hidden-count 32 --mutation-rate 0.12 --lambda-warmup-fraction 0.9 --archive-training --archive-interval 40 --archive-add-per-interval 32 --stop-after-discovery
```

Result: `1/1` exact discovery. The selected classifier is evaluated in its naturally evolved form; there is no separate simplification step after evolution.

```text
concept,primitive,runs,exact_discoveries,mean_train_acc,mean_val_acc,mean_hidden_acc,mean_probe_acc,mean_complexity
first_equals_last,bidirectional_compare,1,1,1.000,1.000,1.000,1.000,12.0
```

Selected evolved rules:

```text
s0:TOKEN -> STORE_R0 / s1
s1:EOS -> PREDICT_TRUE / s2
s1:TOKEN -> MOVE_RIGHT / s2
s1:MATCH_R0 -> MOVE_RIGHT / s1
s2:TOKEN -> MOVE_RIGHT / s2
s2:MATCH_R0 -> STORE_R0 / s1
```

The automaton stores the first token, scans right, and reaches the accepting `EOS` state only when the most recent boundary-relevant token matches the stored first token. Middle repeats are handled because nonterminal matches are followed by continued scanning rather than immediate acceptance.

A small lambda-scale check gave the same exact six-rule solution for `lambda = 0.000010, 0.000020, 0.000050, 0.000100`. This is evidence for a developmental-overcapacity basin: the optimizer escapes trivial local minima by carrying a little more machinery than the known hand-constructed minimum. The five-rule construction is therefore a representability witness, not the expected evolutionary endpoint from cold search.

The stronger thesis is that the globally minimal basin may be practically unreachable by direct stochastic evolution under the same representation. A more natural trajectory is Raptor-like engineering simplification: first find a working but overbuilt machine, then let later selection, changed constraints, or new mutation operators discover simpler descendants. Complexity pressure matters, but it does not magically make the global minimum easy to enter.

### Developmental-Overcapacity Ablation

Same task, same archive/lambda settings, same search budget; only the maximum encoded rule capacity changes.

```text
condition,exact,train,val,hidden,probe,probe_bal,complexity,rules
minimal_cap_5_rules,False,0.50,0.50,0.50,0.67,0.50,0.0,0
selected_size_cap_6_rules,False,0.50,0.50,0.50,0.33,0.50,2.0,1
overcapacity_12_rules,True,1.00,1.00,1.00,1.00,1.00,12.0,6
```

The important result is that the six-rule final solution is not found when the genome is capped at six encoded rules from the start. Discovery requires a larger developmental search space, even though the ultimately selected exact classifier uses only six rules. This supports the overcapacity claim directly: temporary excess degrees of freedom are not just waste; they can be the path into a viable basin.

### Broader Bongard Rule-Discovery Matrix

A broader one-replicate streaming ablation was run over several Bongard-style rules. The point is not that every task must show the overcapacity pattern. The honest question is how often limited capacity works, how often overcapacity helps, and where both fail under the current budget.

```bash
python3 -u experiments/run_bongard_overcapacity_ablation.py --replicates 1
```

Completed fast-matrix rows:

```text
task,condition,replicate,exact,train,val,hidden,probe,probe_bal,complexity,rules
length_even,cap_3_rules,0,False,0.79,0.81,0.79,0.32,0.55,7.0,3
length_even,overcapacity_10_rules,0,False,0.79,0.81,0.79,0.32,0.55,15.0,6
length_multiple_of_three,cap_4_rules,0,False,0.50,0.50,0.50,0.31,0.50,0.0,0
length_multiple_of_three,overcapacity_12_rules,0,True,1.00,1.00,1.00,1.00,1.00,7.0,3
first_equals_second,cap_4_rules,0,False,0.50,0.50,0.50,0.33,0.50,2.0,1
first_equals_second,overcapacity_14_rules,0,False,0.50,0.50,0.50,0.67,0.50,0.0,0
last_two_equal,cap_5_rules,0,False,0.50,0.50,0.50,0.67,0.50,0.0,0
last_two_equal,overcapacity_14_rules,0,False,0.50,0.50,0.50,0.67,0.50,0.0,0
has_adjacent_duplicate,cap_4_rules,0,True,1.00,1.00,1.00,1.00,1.00,11.0,4
has_adjacent_duplicate,overcapacity_12_rules,0,True,1.00,1.00,1.00,1.00,1.00,20.0,8
first_equals_last,cap_5_rules,0,False,0.50,0.50,0.50,0.67,0.50,0.0,0
first_equals_last,cap_6_rules,0,False,0.50,0.50,0.50,0.33,0.50,2.0,1
first_equals_last,overcapacity_12_rules,0,True,1.00,1.00,1.00,1.00,1.00,12.0,6
```

Pilot taxonomy from these rows:

- **Overcapacity-required under this budget:** `length_multiple_of_three`, `first_equals_last`.
- **Limited capacity already sufficient:** `has_adjacent_duplicate`.
- **Ambiguous or underconstrained quick panels:** `first_equals_second`, `last_two_equal`. These should not be treated as failed rule-discovery cases until the training panel contains enough hard negatives.
- **Both conditions miss under quick budget:** `length_even` in this quick run.

Negative-heavy panels resolve two of the ambiguous quick rows:

```text
task,condition,exact,train,val,hidden,probe,probe_bal,complexity,rules
first_equals_second,cap_3_rules,False,0.77,0.67,0.67,0.67,0.50,0.0,0
first_equals_second,overcapacity_14_rules,True,1.00,1.00,1.00,1.00,1.00,6.0,3
last_two_equal,cap_7_rules,False,0.77,0.67,0.67,0.67,0.50,0.0,0
last_two_equal,overcapacity_14_rules,True,1.00,1.00,1.00,1.00,1.00,14.0,7
```

This supports two points at once. First, some Bongard patterns really are ambiguous under small balanced panels; humans often see the intended rule because they are primed by the problem family. Second, after adding hard negatives, these tasks also show the developmental-overcapacity pattern: the final selected solution uses 3 or 7 rules, but a cold search capped at that final rule count fails.

A separate `second_equals_last` pilot supports the same overcapacity pattern, but it is slower and should be run explicitly rather than included in the fast matrix:

```text
task,condition,exact,train,val,hidden,probe,probe_bal,complexity,rules
second_equals_last,cap_6_rules,False,0.58,0.54,0.61,0.34,0.50,12.0,6
second_equals_last,overcapacity_16_rules,True,1.00,1.00,1.00,1.00,1.00,12.0,6
```

The important point is not that overcapacity always wins. It does not. The evidence is a distribution of cases: some concepts are easy enough for limited caps, some are not solved by either setting at the quick budget, and some become discoverable only when the genome can carry more structure during development than the final selected classifier uses.

## Research Observation

The distinction between sampled success and rule discovery matters. Plain sampled train/validation/hidden success was not enough: earlier first/last near misses passed sampled panels but failed the exhaustive foreign-alphabet probe. Archive pressure changed the training landscape by feeding misclassified counterexamples back into the free-energy loss, and late lambda warmup avoided premature collapse to a trivial low-complexity classifier.

Current conclusion:

```text
clean sparse evolution discovers stream, local-comparison, and first/last boundary rules
when the optimizer has enough temporary degrees of freedom and complexity pressure is annealed inside evolution.
```

The useful research observation is not that a hand-compressed program exists. It is that extra dimensions help escape local minima, while the complexity term prevents those dimensions from being free. The minimal basin can still be hard to find; simplification should be studied as a later evolutionary trajectory, not assumed to appear just because `lambda C` is present. Hidden and exhaustive probes remain outside training and define discovery.
