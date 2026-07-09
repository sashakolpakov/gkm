# Bongard-LOGO Symbolic Adapter Report

This is the first external Bongard-LOGO test. The repository is downloaded outside version control under `downloads/Bongard-LOGO`; the full Google Drive image archive is not vendored or required for this symbolic run.

## Reproduction

From the repository root:

```bash
git clone https://github.com/NVlabs/Bongard-LOGO.git downloads/Bongard-LOGO
.venv/bin/python -m pip install pillow pandas
.venv/bin/python bongard/run_bongard_logo_adapter.py --dataset-dir downloads/Bongard-LOGO --source both --feature-set both --limit 40 --support-count 10 --validation-count 3 --hidden-count 3 --summary-only
```

The adapter calls the Bongard-LOGO Basic and Abstract samplers directly and extracts action programs without rendering images. Each generated problem is split into support/train, validation, and hidden examples. Candidate deterministic rules are sparse conjunctions of symbolic scene features, selected by:

```text
F_lambda(rule) = train_classification_loss(rule) + lambda C(rule)
```

Validation then chooses the loss/complexity elbow, and hidden examples are reported once after selection.

## Feature Modes

- `action`: uses only object counts and LOGO action-program skeletons, with stroke style removed.
- `metadata`: adds shape names, shape super-classes, and human-designed attributes from Bongard-LOGO metadata.

`metadata` is intentionally privileged. It is an upper-bound symbolic mode, not a visual solver and not a claim that the model inferred those concepts from pixels.

## First Results

With `--limit 40 --support-count 10 --validation-count 3 --hidden-count 3`:

```text
category,feature_set,problems,mean_train_acc,mean_val_acc,mean_hidden_acc,exact_hidden,mean_complexity
basic,action,40,1.000,1.000,0.996,39,4.3
abstract,action,26,0.598,0.558,0.583,3,1.1
abstract,metadata,26,1.000,1.000,1.000,26,2.5
```

A quick smaller-panel run (`--support-count 4 --validation-count 3 --hidden-count 3`) showed the expected shortcut pressure:

```text
category,feature_set,problems,mean_train_acc,mean_val_acc,mean_hidden_acc,exact_hidden,mean_complexity
basic,action,40,1.000,0.967,0.967,35,4.0
abstract,action,26,0.606,0.583,0.564,3,0.7
abstract,metadata,26,1.000,0.994,0.981,24,2.4
```

## Follow-Up Sweep

Basic was rerun over a broader one-shape sample:

```text
command: --source basic --feature-set action --limit 120 --support-count 10 --validation-count 3 --hidden-count 3 --max-rule-atoms 2
category,feature_set,problems,mean_train_acc,mean_val_acc,mean_hidden_acc,exact_hidden,mean_complexity
basic,action,120,1.000,0.997,0.999,119,6.0
```

For one-attribute Abstract concepts, more action-rule capacity did not help. With 10/3/3 panels:

```text
max_rule_atoms,mean_train_acc,mean_val_acc,mean_hidden_acc,exact_hidden,mean_complexity
1,0.594,0.538,0.583,3,1.0
2,0.598,0.558,0.583,3,1.1
3,0.598,0.558,0.583,3,1.1
```

With larger 20/5/5 panels, action-only Abstract became a cleaner failure rather than improving:

```text
max_rule_atoms,mean_train_acc,mean_val_acc,mean_hidden_acc,exact_hidden,mean_complexity
1,0.578,0.550,0.554,2,1.0
2,0.578,0.550,0.554,2,1.0
4,0.578,0.550,0.554,2,1.0
```

Two-attribute Abstract concepts make the same point more strongly. The adapter skips undersupplied attribute pairs before calling the sampler, because some pairs do not have enough positive/negative shapes for large support panels. On the first 80 viable concepts:

```text
condition,problems,mean_train_acc,mean_val_acc,mean_hidden_acc,exact_hidden,mean_complexity
action, atoms=3, support=20/5/5,80,0.620,0.620,0.599,3,1.3
metadata, atoms=2, support=10/3/3,80,0.999,0.998,0.996,78,3.3
metadata, atoms=3, support=10/3/3,80,1.000,0.998,0.996,78,3.3
metadata, atoms=2, support=20/5/5,80,1.000,1.000,1.000,80,3.4
metadata, atoms=3, support=20/5/5,80,1.000,1.000,1.000,80,3.4
```

Conclusion: more examples help when the right predicates are available; more conjunction capacity alone does not fix missing predicates. Abstract action-only failure is therefore not mainly a capacity issue in the sparse selector. It is a substrate issue: the action skeleton needs derived geometric predicates, or the learner must evolve/learn those predicates before the Bongard selector can use them.

## Predicate Macro Prototype

The adapter now has a first predicate-macro mode. It derives simple geometric measurements from LOGO action programs, exposes reusable macro predicates, and charges those macro predicates in rule complexity. These predicates are not Bongard metadata; they are built from the action trace. Examples include:

```text
macro:line_count>=3
macro:arc_count<=1
macro:closure_error<=0.08
macro:aspect_ratio>=2.5
macro:hull_fill>=0.9
macro:convex_fill_candidate
macro:thin_candidate
```

The selector can then use a macro as an atom inside a higher-level Bongard rule. This is the first small version of predicate invention as reusable rule macros. Candidate macro atoms are ranked by training-panel separation before conjunction search; this is a search-budget cap, not a post-hoc simplification. Use `--max-candidate-atoms 0` to disable the cap.

One-attribute Abstract rerun:

```text
command: --source abstract --feature-set all --limit 26 --support-count 10 --validation-count 3 --hidden-count 3 --max-rule-atoms 2 --max-candidate-atoms 20 --summary-only
category,feature_set,problems,mean_train_acc,mean_val_acc,mean_hidden_acc,exact_hidden,mean_complexity
abstract,action,26,0.598,0.558,0.583,3,1.1
abstract,macro,26,0.852,0.737,0.731,6,4.8
abstract,metadata,26,1.000,1.000,1.000,26,2.5
```

With larger 20/5/5 panels, macro mode remains better than action-only but does not close the gap:

```text
command: --source abstract --feature-set macro --limit 26 --support-count 20 --validation-count 5 --hidden-count 5 --max-rule-atoms 2 --max-candidate-atoms 20 --summary-only
category,feature_set,problems,mean_train_acc,mean_val_acc,mean_hidden_acc,exact_hidden,mean_complexity
abstract,macro,26,0.770,0.735,0.696,4,3.9
```

Example selected macro rules from a six-concept smoke run:

```text
convex: macro:abs_turn_total<=360 AND macro:convex_fill_candidate
has_curve: has_arc
has_straight_line: has_line
self_transposed: macro:heading_error>=0.05 AND macro:line_count<=6
```

This is the right research direction but not a solved predicate system. The invented macros recover part of the missing abstraction layer from action geometry, while failures such as symmetry still indicate missing or weak predicates.

## Observations

1. **Basic Shape is mostly accessible from action programs.** Exact action skeletons recover almost all one-shape Basic concepts once the support panel has enough hard negatives from the same superclass.

2. **Abstract Shape exposes the representation bottleneck.** Action skeletons alone solve only concepts directly visible in primitive strokes, such as `has_curve` or `has_straight_line`. Concepts like `convex`, `symmetric`, `thin_shape`, and `necked` need higher-level object predicates or a learned parser.

3. **The free-energy term behaves as expected under underspecified panels.** With too few negatives, lower-complexity shortcuts such as `action_count=2` can beat exact shape signatures on training free energy and survive validation by chance. Increasing support examples reduces this, which matches the earlier internal Bongard result: panel design matters.

4. **Metadata mode is a sanity check, not the target.** It shows that the few-shot selection loop can recover the intended Abstract attributes when those attributes are exposed. The real research task is to evolve or learn the intermediate predicates rather than hand them to the selector.

## Next Steps

- Generate counterexample-rich Basic and Abstract panels instead of accepting the sampler order passively.
- Feed LOGO scenes into the sparse FSA/transducer machinery rather than the temporary feature-rule selector.
- Add two-shape and two-attribute concepts after the one-object path is stable.
- Keep Freeform and visual rendering separate until the symbolic path has honest failure modes and reproducible baselines.
