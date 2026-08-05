# Bongard: concept induction and the emergence of abstraction

This domain applies the Gödel–Kolmogorov Machine free-energy view to **Bongard-style concept induction**:
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
  loop and typed semantic-cone track; first live engineering smoke in
  [`bongard_crack_smoke_report.md`](bongard_crack_smoke_report.md)
  (historical flat-pricing Sonnet run, 2/2 fresh-seed rendered LOGO problems;
  not current Phase D evidence).
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

## Current Raw-Panel Evidence Boundary

The offline Phase D protocol is implemented. One paid, unrestricted-only n=1
exploratory pilot completed on 5 August 2026: primary 0/1 ordinary unsolved,
one shuffled replicate 0/1 with a canonical verifier failure, and held-fixed
no-share 0/1 inherited from primary. Its write-once campaign digest is
`sha256:8be70918d2b57811a66787cdff845dbcb445eaf8e073f61443cea698845dfcf2`;
all three artifacts independently cold-replay. The pilot is not the default
27-arm study and is not confirmatory because its local preregistration was not
externally timestamped before execution, it has no semantic-pure arm, and n=1
supports no rate claim. `crack_lab/prepare_phase_d.py` freezes one maximum corpus and its exact
panel bytes, using independent basic/abstract RNG streams and nested 1/5/25
prefixes. `crack_lab/phase_d_protocol.py` preregisters 27 default arms: primary
and three balanced shuffled-side replicates for both `UNRESTRICTED` and
`SEMANTIC-PURE` at each scale, plus one unrestricted no-share arm per scale.
For every primary or shuffled proposer family, n1 is the only legal fresh
start; n5 requires a complete replay-valid n1 checkpoint in the same family
artifact, and n25 requires n5. Fresh higher-scale starts, shrinkage, skipped or
incomplete predecessors fail before proposer construction and before any write;
no-share scales are independent derived artifacts.
`crack_lab/collect_phase_d.py` accepts those reports only from their generated
execution-tag artifacts as one complete, cross-validated 27-arm set. It
cold-replays the source checkpoint or semantic RunSpecs and publishes a
deterministic write-once campaign with per-arm artifact certifications.
The locally written preregistration is a reproducibility manifest, not proof of
temporal priority; a confirmatory claim requires its digest to be published or
externally committed before the first proposer call.

Unrestricted predicates execute under `bongard-predicate-purity/v2`, a positive
capability manifest covering exact imports, restricted builtins, callable/value
and keyword forms, owned-scratch mutation, and resource bounds; a permitted
module root grants no other API. Selection then prices the exact transitive AST
closure of selected predicates, helpers, constants, and imports under
`bongard-predicate-pricing/v3`, adds per-use rule structure, and chooses
empirical risk first and conditional cost second within a bounded exhaustive
rule search. The paid ledger discounts only definitions reached by earlier
accepted rules. Identity-keyed lookup is instructed against and charged/visible,
but rotated LOO over the 12 already-seen panels cannot by itself distinguish it;
generalization requires unseen-instance evidence. No-share is a held-fixed
accounting replay of primary sources/rules/risks/outcomes, not a fresh proposer
run or a causal solve-rate control. Semantic no-share is intentionally absent
until learned legs can be separated from the fixed base registry.

Both tracks bind resume and artifacts to corpus, embedded panel bundle,
condition/control, and ordered prefix. Semantic promotion cold-replays the
complete candidate set and winner in a fresh process. Unrestricted promotion
sequentially reruns every exact attempted source and priced selection on the
embedded panels, checking rule, fold, risk, error, and receipt evidence under
verifier/runtime provenance. Canonical zero-admission verifier failures must
reproduce exactly and are reported separately from ordinary misses. Production
unrestricted turns use a pinned,
non-interactive Codex CLI invocation. Each receipt causally binds the exact
task, current and proposed source/log, raw PNG view, semantic panel set,
structured output, unique thread/event stream, requested model and reasoning
effort, positive usage, and CLI/launcher identity. A request flag is not
relabeled as JSONL-reported model evidence when that optional field is
absent. Independently, every semantic Anthropic response must report the exact
concrete model requested after alias resolution.

See [`CONTINUATION_PLAN.md`](CONTINUATION_PLAN.md) and
[`crack_lab/SEMANTIC_TYPED_TRACK.md`](crack_lab/SEMANTIC_TYPED_TRACK.md) for the
sequential execution protocol and evidence requirements.

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
