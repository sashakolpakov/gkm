# Sparse Register Transducer Benchmark

This experiment tests whether sparse deterministic transducers can discover
compact object-level rules over foreign alphabets. Train, validation, and hidden
test splits use disjoint object pools, so hidden-test success cannot come from
memorizing literal token IDs.

## Protocol

- Substrate: `pattern_fsa.py`
- Selection: training free energy during evolution, then validation
  loss-complexity elbow across a lambda sweep
- Hidden test: evaluated only after validation selection
- Object pools: disjoint train, validation, and hidden-test tokens
- Default examples in the reported matrix: 10 train, 6 validation, 6 hidden test
- Lambda sweep: `0.0` to `0.01`, 4 points unless noted in the repro script

Primitive tiers:

```text
stream   = MOVE_RIGHT, WRITE_CURRENT, HALT
register      = stream + STORE_Ri, WRITE_Ri
compare       = register + current-token/register equality observations
bidirectional = stream + MOVE_LEFT + BOS observation
```

## Results

| Task | Primitive tier | Hidden exact | Hidden loss | Complexity | Rules | Observation |
|---|---:|---:|---:|---:|---:|---|
| `copy`, length 3 | `stream` | 1.00 | 0.0000 | 3.0 | 1 | Trivial stream transduction works. |
| `swap`, length 2 | `stream` | 0.00 | 0.5000 | 3.0 | 1 | Fails: no memory for first token. |
| `swap`, length 2 | `register` | 1.00 | 0.0000 | 7.0 | 3 | Solves foreign-object swap. |
| `duplicate_first`, length 2 | `stream` | 1.00 | 0.0000 | 5.0 | 2 | Solves because macro-rules can write current twice. |
| `rotate_left`, length 3 | `stream` | 0.00 | 0.3333 | 3.0 | 1 | Fails: cannot delay first token. |
| `rotate_left`, length 3 | `register` | 1.00 | 0.0000 | 8.0 | 3 | Solves with one register after larger search. |
| `reverse`, length 3 | `register`, 2 regs | 1.00 | 0.0000 | 10.0 | 3 | Solves fixed-length reverse with internal memory. |
| `reverse`, length 5 | `bidirectional` | 1.00 | 0.0000 | 7.0 | 3 | Solves reverse by scanning to EOS and reading left. |
| `dedupe_pair` | `register` | 0.50 | 0.2500 | 2.0 | 1 | Half-solves; cannot branch on equality. |
| `dedupe_pair` | `compare` | 1.00 | 0.0000 | 4.0 | 1 | Solves using current-token/register equality. |

## Key Solver Motifs

`swap + register` learned the abstract operation:

```text
store first object
move to second object
write second object
write stored first object
```

`rotate_left + register` learned:

```text
store first object
move to second object
write second object
move to third object
write third object
write stored first object
```

`reverse + bidirectional` learned:

```text
TOKEN -> MOVE_RIGHT
EOS   -> MOVE_LEFT, switch to emit-left state
TOKEN -> WRITE_CURRENT, MOVE_LEFT
BOS   -> no rule, so halt
```

`dedupe_pair + compare` found a sparse conditional halt motif:

```text
TOKEN    -> STORE_R0, WRITE_R0, MOVE_RIGHT
MATCH_R0 -> no rule, so halt
TOKEN    -> same rule, so write the nonmatching second token
```

## Interpretation

The primitive tiers separate cleanly:

- `stream` handles copying and local duplication, but fails when a task requires
  delayed object recall.
- `register` enables positional rearrangement over foreign alphabets.
- `compare` enables equality-conditioned behavior.
- `bidirectional` uses the input tape as external read-only memory, so reversal
  does not require storing the whole prefix internally.

The `register` reverse result is fixed-length reverse. The `bidirectional`
reverse result is the cleaner length-general motif: go to `EOS`, then emit while
moving left until `BOS`. A stack is still the natural next memory tier for
one-way stream settings where the input cannot be revisited. Stack depth should
be counted in the free-energy complexity term.

## Reproduction

Run from the repository root:

```bash
python3 transduction/run_register_transducer_benchmark.py
```

The script prints a CSV summary and selected sparse rules. It uses deterministic
seeds, but future substrate changes may alter exact solver forms.
