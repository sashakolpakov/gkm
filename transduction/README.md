# Register-Transducer Synthesis

This subject synthesizes compact deterministic register transducers over opaque token
streams. Disjoint token pools prevent identity memorization, while tiered primitive
sets expose which capabilities are required by each task family.

The original register-transducer search minimizes training loss plus encoded solver
size. Its runner performs an explicit lambda sweep, chooses from the validation
loss-complexity Pareto frontier, and evaluates hidden transitions only after
selection. The separate machine-growth experiments below instead retain the first
admissible attachment and record size as a diagnostic. Results remain conditional
on the supplied primitive vocabulary and finite search budget.

## Entry Points

- [Complete evidence and reproduction package](REPRODUCIBILITY.md): original
  register results regenerated with full records and united with the later
  experiments; offline replay needs no model access.
- [`TRANSDUCTION.md`](TRANSDUCTION.md): full subject guide.
- [`pattern_fsa.py`](pattern_fsa.py): transducer representation and search.
- [`run_register_transducer_benchmark.py`](run_register_transducer_benchmark.py):
  benchmark matrix.
- [`register_transducer_benchmark.md`](register_transducer_benchmark.md): report.
- [`manuscript/transduction.tex`](manuscript/transduction.tex): subject manuscript,
  including the machine-growth experiments; [build and evidence map](manuscript/README.md).
- [`COFIBRATION_EXPERIMENT_20261003.md`](COFIBRATION_EXPERIMENT_20261003.md):
  new machine-growth experiment with exact graph pushouts, mechanical glue,
  and `gpt-5.6-sol` proposals through existing Codex access. This separate
  variable-length curriculum does not replace the original benchmark results.
- [`GLUE_AB_PROTOCOL.md`](GLUE_AB_PROTOCOL.md): paired mechanical/Codex test
  with identical permitted glue. Both achieved 576/576 hidden examples;
  mechanical search was faster and needed no LLM calls, within this grammar.
- [`GROWTH_BENCHMARK.md`](GROWTH_BENCHMARK.md): larger cumulative A/B with
  returning calls, loops, branches and pipelines; five acquired pieces grow
  through twelve tasks, with long-input tests and an unshared execution control.
- [`COMPOSITION_BENCHMARK.md`](COMPOSITION_BENCHMARK.md): fixed-basis A/B that
  replaces fixed pipeline/branch templates with bounded nested expressions,
  measuring mechanical search effort against structured Codex proposals.
- [`RECURSIVE_BENCHMARK.md`](RECURSIVE_BENCHMARK.md): recursive prefix procedures
  with restored caller context and parameterized reuse, compared through
  counterexample-guided mechanical synthesis and Codex proposals.
- [`INTERFACE_BENCHMARK.md`](INTERFACE_BENCHMARK.md): discovery of helper
  arguments and return values under a shared cumulative library contract,
  followed by transfer through unchanged helpers. Both methods use the same
  reuse probe and verifier; no desired interfaces are supplied.
- [`CONTROL_FRONTIER.md`](CONTROL_FRONTIER.md): harder control structures with
  additional actions around recursive calls, comparing LLM proposals with a
  broader mechanical search that can express every tested task.
- [`FRONTIER_RETRIES.md`](FRONTIER_RETRIES.md): fixed repeated trials with a
  longer feedback loop and larger budgets for both proposers.
- [`TRANSFER_VALUE.md`](TRANSFER_VALUE.md): prospective library-removal controls
  and fresh-library comparisons measuring whether interface extensions repay
  their description cost on subsequent tasks.
- [`COFIBRATION_AB.md`](COFIBRATION_AB.md): direct cumulative comparison in
  which the LLM authors every attachment in its arm and mechanical search
  independently authors every attachment in the other.
- [`HARD_COFIBRATION_AB.md`](HARD_COFIBRATION_AB.md): harder recursive structure
  with returned information, using the same proposers and 20-minute task budgets.

Run from the repository root:

```bash
python3 transduction/run_register_transducer_benchmark.py
python -m pytest transduction/test_pattern_fsa.py -q
```
