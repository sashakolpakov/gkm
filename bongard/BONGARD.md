# Bongard: concept induction and the emergence of abstraction

This domain applies the GKM free-energy view to **Bongard-style concept induction**:
given positive and negative example sets over opaque-object sequences, evolve sparse
deterministic classifiers, and ask when free-energy accounting drives the *emergence
of reusable abstraction* (encapsulated predicate macros) rather than duplicated rule
bodies.

- **Common math:** [`../FREE_ENERGY_EXPLANATION.md`](../FREE_ENERGY_EXPLANATION.md).
- **Manuscript:** [`manuscript/free_energy_abstraction.tex`](manuscript/free_energy_abstraction.tex)
  — free-energy selection for predicate encapsulation in sparse deterministic
  solvers (build with `make -C manuscript`).
- **The crack lab (real Bongard from raw panels):**
  [`bongard_crack_plan.md`](bongard_crack_plan.md) — the Architect/Engineer
  plan for rule deduction from raw pixels with the ARC-style crack harness;
  [`crack_lab/`](crack_lab/) — the raw substrate + enforced predicate-library
  loop; first live result in
  [`bongard_crack_smoke_report.md`](bongard_crack_smoke_report.md)
  (Sonnet proposer, 2/2 fresh-seed rendered LOGO problems, reuse collapse
  visible at n=2).
- **Reports:** [`bongard_sparse_classifier_report.md`](bongard_sparse_classifier_report.md),
  [`abstraction_emergence_report.md`](abstraction_emergence_report.md),
  [`bongard_logo_report.md`](bongard_logo_report.md),
  [`abstraction_related_work.md`](abstraction_related_work.md),
  [`bongard_first_plan.md`](bongard_first_plan.md).

## The task

Examples are positive/negative sequences of opaque objects; the classifier sees only
relational observations, never token identity (the same tiered primitive vocabulary
as the transduction domain, reused here). Train, validation, and hidden-test splits
use **disjoint object pools** and are **counterexample-rich** for concepts where
random examples would otherwise admit shortcuts. Concepts range over structural
predicates such as `palindrome`, `first_equals_last`, `has_adjacent_duplicate`,
`length_multiple_of_three`, `all_unique`.

## Free energy

Initial populations are clean-slate random sparse rule tables — no seeded boundary
rules, solver templates, or target-specific initialisation. Evolution minimises
training free energy

```text
F_lambda(solver) = training_loss(solver) + lambda * C(solver)
```

and selection reports discovery on the disjoint hidden pool, following the
loss-complexity lens of [arXiv:2507.13543](https://arxiv.org/abs/2507.13543).

## Emergence of abstraction

The central experiment (`run_abstraction_emergence.py`) is a controlled scaffold for
one question: when a deterministic solver repeats the same hidden substructure across
tasks — or across branches of one disjunctive task — does free-energy accounting
select a reusable **predicate macro** instead of duplicating the primitive rule body?
The primitive observation atoms are given; what is *discovered* is the encapsulation
of a repeated conjunction (e.g. `solid_loop = low_closure_error AND high_hull_fill
AND turn_balanced`) as a shared macro. Because `C` prices the encoded structure,
sharing a macro is cheaper than duplicating it — parsimony pays for abstraction.

## Key modules

- [`run_bongard_symbolic_baseline.py`](run_bongard_symbolic_baseline.py) — symbolic
  Bongard-style baseline (concepts, problem construction, labeled iteration).
- [`run_bongard_sparse_classifier.py`](run_bongard_sparse_classifier.py) — evolved
  sparse deterministic classifier (imports the baseline + the transduction
  `pattern_fsa` primitives).
- [`run_bongard_overcapacity_ablation.py`](run_bongard_overcapacity_ablation.py) —
  paired overcapacity ablations over a fast rule matrix.
- [`run_bongard_logo_adapter.py`](run_bongard_logo_adapter.py) — local Bongard-LOGO
  symbolic adapter (no vendored data).
- [`run_abstraction_emergence.py`](run_abstraction_emergence.py) — the internal
  predicate-library abstraction scaffold.
- [`test_bongard_sparse_classifier.py`](test_bongard_sparse_classifier.py) /
  [`test_abstraction_emergence.py`](test_abstraction_emergence.py) — the domain tests.

## Run

```bash
python3 bongard/run_bongard_symbolic_baseline.py
python3 bongard/run_bongard_sparse_classifier.py --concept palindrome
python3 bongard/run_bongard_sparse_classifier.py --concept first_equals_last
python3 -u bongard/run_bongard_overcapacity_ablation.py --replicates 1
python3 bongard/run_abstraction_emergence.py --scenario or_factor --show-rules

# local Bongard-LOGO adapter (external data, not vendored):
git clone https://github.com/NVlabs/Bongard-LOGO.git downloads/Bongard-LOGO
python3 bongard/run_bongard_logo_adapter.py --dataset-dir downloads/Bongard-LOGO \
    --source both --feature-set both --limit 40 --support-count 10 \
    --validation-count 3 --hidden-count 3 --summary-only
```

## Tests

```bash
python -m pytest bongard/test_bongard_sparse_classifier.py bongard/test_abstraction_emergence.py -q
```

## Scope

The abstraction-emergence result is a controlled internal scaffold, **not** a
Bongard-LOGO benchmark result: the primitive atoms are hand-defined; what is measured
is whether free energy selects encapsulation over duplication. The LOGO adapter is
the bridge toward real Bongard-LOGO problems.
