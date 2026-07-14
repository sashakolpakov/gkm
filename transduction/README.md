# Register-Transducer Synthesis

This subject synthesizes compact deterministic register transducers over opaque token
streams. Disjoint token pools prevent identity memorization, while tiered primitive
sets expose which capabilities are required by each task family.

Selection minimizes training loss plus encoded solver size. The runner performs an
explicit lambda sweep, chooses from the validation loss-complexity Pareto frontier,
and evaluates hidden transitions only after selection. Results remain conditional on
the supplied primitive vocabulary and finite search budget.

## Entry Points

- [`TRANSDUCTION.md`](TRANSDUCTION.md): full subject guide.
- [`pattern_fsa.py`](pattern_fsa.py): transducer representation and search.
- [`run_register_transducer_benchmark.py`](run_register_transducer_benchmark.py):
  benchmark matrix.
- [`register_transducer_benchmark.md`](register_transducer_benchmark.md): report.
- [`manuscript/transduction.tex`](manuscript/transduction.tex): subject manuscript.

Run from the repository root:

```bash
python3 transduction/run_register_transducer_benchmark.py
python -m pytest transduction/test_pattern_fsa.py -q
```
