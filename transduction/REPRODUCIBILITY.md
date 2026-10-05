# Transduction: complete evidence and reproduction package

The package joins the original register benchmark, the attachment and interface
studies, the two independent mechanical/LLM comparisons, and the retention
experiment. Failed and blocked conditions remain included. Deliberately
discarded invalid trials are not part of the evidence.

Repository release: [reproduction archive](../output/transduction_repro/transduction-repro-20261004.tar.gz)
and [SHA-256 checksum](../output/transduction_repro/transduction-repro-20261004.tar.gz.sha256).
Its extracted directory is `20261004-release`.

## Offline replay

Extract the archive into a new directory. No account, API key, model service or
access to the original checkout is needed for replay. Use Python 3.11 or later
on Linux or macOS;
the included manifest records the exact Python and dependency versions used.
Install the pinned replay and test dependencies:

```sh
python -m pip install -r transduction/requirements-repro.txt
python transduction/repro_package.py --package . --checksums
python transduction/repro_package.py --package . --verify --results ../replay-results
```

The results directory must be new. The full replay includes the stored finite
shape and continuation audits. It verifies selections and program execution,
not deterministic reproduction of remote model sampling or historical timings.
Several mechanical replay checks do repeat their deterministic search. Use
`--only register`, `--only hard`, or another ID from `manifest.json` to replay a
subset. The verifier refuses network calls, model subprocesses and writes to
the evidence, and blocks fallback reads from the original checkout. Replay
workers stop if resident memory exceeds 768 MiB. Logs and the new verification
receipt go outside the package.

## Contents and source versions

- `manifest.json` inventories every included study and its source version.
- `SHA256SUMS` covers every payload file, including all raw artifacts and code.
- `output/` preserves the experiment artifacts byte for byte: protocols, data,
  proposals and feedback, receipts, libraries, certificates, histories and
  evaluations. Paths here have the same relative layout as the repository.
- `frozen/<study>/` contains the corresponding executable sources. Files bound
  by a historical source hash are recovered from matching repository files or
  saved archives. `source-provenance.json` distinguishes those files from
  supplemental dependencies needed to run them.
- `transduction/` contains the current source, tests, reports and manuscript.
- `verification/` contains the complete offline replay receipt and logs,
  including regenerated auxiliary audits where their old code was unavailable.

The earliest tail-attachment pilot did not record source hashes. Its complete
program and execution artifacts are included, and the replay implementation is
bound by this package's checksum. Later studies retain their original source
hash checks. The composition replay redirects one historical absolute path to
the identical packaged growth library; its source and library hash checks stay
active. No historical record is rewritten to make replay pass.

Two early frontier shape audits lacked their exact auxiliary source version.
Those audits are regenerated with the bound current auditor and the original
discovery/execution sources. Every previously reported result field must match;
the new auditor hash and additional diagnostics are recorded separately. The
original audit files remain intact.

## Regenerated register benchmark

`output/transduction_register/20261004-reproduction-v1` is a new run on
4 October 2026, not a recovered original execution. The original ten-condition
runner and substrate were not changed. Scoped recording wrappers capture their
inputs and return values without changing search or selection.

It contains the frozen configuration and sources, all 40 lambda winners and
their generation histories, all ten selected executable programs, the complete
train/validation/hidden examples and selected-program executions, and a CSV
matrix. All ten rows reproduce the historical hidden accuracy, loss and size.
The original metric is deliberately retained: exact output at the instruction
cap does not require halting. Halting flags are saved so this remains visible.

To reproduce the entire evolutionary search, choose a new directory:

```sh
python transduction/register_artifacts.py --output ../register-new --prepare
python transduction/register_artifacts.py --output ../register-new --conditions 0 1 2 3 4 5 6 7 8 9
python transduction/register_artifacts.py --output ../register-new --finalize
python transduction/register_artifacts.py --output ../register-new --replay
```

Completed lambda searches are immutable checkpoints. Repeating `--conditions`
resumes unfinished work and checks existing checkpoints; use a genuinely new
directory for an independent full rerun. `--replay` never launches search.

## New discovery runs and manuscript build

Each study report gives its discovery command. Historical implementations are
under `frozen/`; use the corresponding version when repeating that protocol.
Unlike offline replay, fresh LLM discovery needs suitable Codex access and can
produce different results. No new model calls were made to assemble this package.
Finite search timeouts and model sampling are not promised to reproduce exactly
on different hardware or services.

Run tests and build the paper with:

```sh
PYTHONDONTWRITEBYTECODE=1 python -m pytest transduction -q -p no:cacheprovider
make -C transduction/manuscript
```

The repository includes the source package and data. A formal archival deposit
and DOI remain separate steps.
