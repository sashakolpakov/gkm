# Bongard-First Benchmark Plan

Bongard problems are a good next external target because they ask for few-shot
concept induction rather than direct answer memorization. A problem supplies
positive and negative examples; the solver must infer a compact rule and classify
held-out queries.

## External Targets

1. **Bongard-LOGO**
   - Repository: https://github.com/NVlabs/Bongard-LOGO
   - Paper: https://arxiv.org/abs/2010.00763
   - Why it fits: synthetic, program-guided, object/shape concepts, many
     procedurally generated instances.
   - Recent diagnostic support: https://arxiv.org/abs/2604.21346 argues that
     symbolic inputs expose a representation bottleneck on Bongard-LOGO, which
     is exactly the split we need between perception and sparse rule induction.
   - First use: adapter target, not vendored data. We should load generated
     problems or symbolic metadata outside this repo, then convert each problem
     to our internal concept-induction protocol.

2. **Bongard-OpenWorld**
   - Paper: https://arxiv.org/abs/2310.10207
   - Why it is harder: real images and open-vocabulary concepts.
   - Use later as a robustness check after the symbolic/LOGO pipeline works.

## Research Question

Can free-energy selection discover or select compact deterministic concept
solvers that generalize to hidden objects/examples?

The Bongard version of the objective is:

```text
F_lambda(h) = classification_loss_train(h) + lambda C(h)
```

where `h` is a deterministic rule/hypothesis. Selection happens in two stages:

```text
train examples      -> local free-energy minimization
validation examples -> choose loss/complexity elbow
hidden examples     -> report once after validation selection
```

## Why Start With A Local Procedural Bongard Scaffold

A local generator gives us controlled splits before dealing with image parsing:

```text
train objects      disjoint from validation objects
validation objects disjoint from hidden-test objects
hidden test        generated after rule selection
```

It also lets us vary exactly what primitive tier is needed:

```text
stream          length/position-only concepts
compare         equality concepts
bidirectional   concepts using both boundaries / scan directions
register/stack  memory-dependent concepts
```

This is not a replacement for Bongard-LOGO. It is the harness that lets us test
our free-energy protocol before writing a dataset adapter.

## Implementation Stages

### Stage 1: Symbolic Bongard Harness

File: `experiments/run_bongard_symbolic_baseline.py`

- Generate positive and negative opaque-object sequences from concept rules.
- Use disjoint train/validation/hidden-test object pools.
- Evaluate a small deterministic hypothesis library.
- Sweep `lambda`.
- Select by validation loss/complexity elbow.
- Report hidden-test accuracy.

This validates the evaluation protocol and exposes ambiguity when multiple
hypotheses fit a small training panel.

### Stage 2: Evolved Bongard Classifiers

File: `experiments/run_bongard_sparse_classifier.py`
Report: `experiments/bongard_sparse_classifier_report.md`

Replace the hand-coded hypothesis library with sparse deterministic classifiers:

```text
(state, TOKEN | EOS | BOS | MATCH_REGISTER_MASK) -> (actions, next_state, label?)
```

Current clean-slate results with concept-specific budgets and exhaustive discovery probes:

- `length_even` is discovered in 8/8 clean replicates by a stream classifier.
- `has_adjacent_duplicate` is discovered in 8/8 clean replicates by compare/register classifiers.
- `first_equals_last` initially produced sampled near misses under fixed search, but is discovered by counterexample-archive evolution with late lambda warmup and loss-frontier preservation. The successful selected classifier passes train, validation, hidden, and exhaustive foreign-alphabet probes at `1.00` with 6 evolved rules and complexity `12.0`.

The key lesson is that sampled validation can still be misleading. Counterexample-
rich train/validation/hidden splits reduce shortcut pressure, but discovery is
now credited only when the selected classifier also passes an exhaustive foreign-
alphabet probe. The optimizer must keep enough temporary dimensions to escape
local minima, while the complexity term makes those dimensions costly. The
minimal hand-written basin should not be treated as the expected cold-search
endpoint; like engineering simplification, leaner automata may require a later
trajectory after a working overbuilt machine exists.

A first ablation supports this: `first_equals_last` fails under five-rule and
six-rule caps, but succeeds when the genome can carry twelve rules during
development and then selects a six-rule exact classifier. A broader pilot matrix
shows the effect is not universal: `has_adjacent_duplicate` solves even under a
limited cap, while `length_multiple_of_three` and `first_equals_last` show clear
overcapacity benefit. Two initially missed rules, `first_equals_second` and
`last_two_equal`, are better understood as underconstrained panels: adding hard
negative examples makes both discoverable, and both then show the same pattern
where overcapacity succeeds but a cap equal to the final rule count fails. This
is the right evidence shape: task-by-task discovery rates plus panel-design
checks, not a claim that every Bongard rule needs overcapacity.

### Stage 3: Bongard-LOGO Adapter

This should be the next real target before Bongard-OpenWorld. Bongard-LOGO is
still visual, but its synthetic generation and shape/object concepts are closer
to the sparse symbolic protocol. Bongard-OpenWorld should wait until the solver
adapter works, because otherwise failures will mix perception, open-vocabulary
recognition, and rule induction.

Add an adapter with this shape:

```text
Bongard-LOGO problem
  -> positive examples
  -> negative examples
  -> action-program / attribute metadata when available
  -> internal opaque-object or relational scene encoding
  -> train/validation/hidden split
  -> same free-energy solver-selection harness
```

The first adapter should use symbolic metadata and action programs where the
external dataset exposes them, then treat raw images as a later perception
problem. This keeps the experiment about rule induction rather than object
recognition.

We should avoid committing the full dataset. The adapter should accept an
external dataset path.

### Stage 4: Image Path

Only after the symbolic path works:

```text
image -> parser/object extractor -> symbolic scene -> sparse solver
```

At that point the thesis can distinguish failures of perception from failures of
rule induction.

## Initial Success Criteria

1. The local symbolic Bongard harness recovers the intended rule on hidden
   disjoint-object examples.
2. Deliberately underpowered primitive tiers fail in interpretable ways.
3. Validation selection rejects over-complex or overfit hypotheses when a simpler
   rule has the same validation loss.
4. The Bongard-LOGO adapter can run without changing the free-energy selection
   code.
