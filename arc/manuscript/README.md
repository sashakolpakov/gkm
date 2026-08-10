# Gödel–Kolmogorov Machine ARC-AGI-3 manuscript

This directory contains the canonical manuscript and reproduction sources for the
Gödel–Kolmogorov Machine ARC-AGI-3 study. The Gödel–Kolmogorov Machine combines
verifier-gated program revision, inverse-colimit attachment, PowerPlay-style retained
competence, artificial curiosity, and description-length selection. After this full
introduction, **GKM** is used as the abbreviation in filenames and dense comparisons.

## Canonical deliverables

- `arc_agi3.tex` and `references.bib`: the integrated paper source.
- `arc_agi3.pdf`: the generated paper; build it locally rather than treating a bundled
  binary as the source of truth.
- `scripts/generate_figures.py`: exact source for the three empirical ledger figures.
- `scripts/generate_empirical_tables.py`: the release-bound 25-game, per-level
  marginal-complexity table generator for TeX, Markdown, JSON, and Sphinx.
- `figure_sources/inverse_colimit_attachment_standalone.tex`: standalone source for
  the inverse-colimit attachment diagram, which also appears inline in the paper.
- `requirements-figures.txt`: the Matplotlib version used for the delivered figure
  geometry.
- `SOCRATIC_PASSES.md` and `repo_ground_truth_matrix.md`: the current review and
  code-to-claim records. Repository history preserves superseded editorial drafts.
- `BUILD_VERIFICATION.md`: results and limits of the repository integration checks.
- `scripts/build_arxiv_bundle.py`: deterministic minimal arXiv source-package builder
  and isolated compile check.
- `SHA256SUMS.txt`: fixed-scope integrity manifest for the integrated source
  deliverables and generated empirical evidence. `make verify-manifest` checks it and
  `make manifest` regenerates it. It deliberately excludes the ignored local paper PDFs,
  whose TeX metadata records the build time; their page/diagnostic checks are recorded
  in `BUILD_VERIFICATION.md`.

## Build

From the repository root:

```bash
python -m pip install -r arc/manuscript/requirements-figures.txt
make -C arc/manuscript
```

The default target regenerates the empirical PNG/PDF figures, compiles the standalone
TikZ diagram, and builds the paper through BibTeX. To build only the paper or the
one-page companion:

```bash
make -C arc/manuscript paper
make -C arc/manuscript one-page
```

## arXiv source package

Do not upload the repository root or the whole `arc/manuscript/` directory. arXiv
compiles from the upload root, while the manuscript intentionally refers to
`generated/` and `figures/` by relative path. Build the minimal root-level package and
compile it in a fresh temporary directory with:

```bash
make -C arc/manuscript arxiv-check
```

The resulting ignored local file is
`arc/manuscript/build/arxiv/arc_agi3_arxiv.zip`. It contains exactly the main TeX
source, the BibTeX database, the two included generated TeX tables, and the two included
PNG figures. It excludes companion documents, repository evidence, scripts, caches,
and all LaTeX output/intermediate files.

The figure generator validates the expected PNG dimensions under Matplotlib 3.10.8:

- `figures/ls20_sawtooth.png`: 1728 × 912 pixels;
- `figures/bounded_campaign_profiles.png`: 2034 × 1072 pixels;
- `figures/marginal_complexity_profiles.png`: 2448 × 912 pixels.

Matplotlib 3.10.8 reproduces the supplied bundle byte-for-byte. Compatible later
versions can preserve the numerical data and validated geometry while changing raster
bytes through rendering details; the integration verification records the version used.

## Repository-level reproduction

The manuscript build reproduces the document, not the stochastic discovery campaign.
The repository supplies separate gates for the retained evidence:

The frozen artifacts and receipt were published at commit
`9235ed26627140460efa1f6ca5e4041470cddc14`. The schema-v2 receipt
`140e37ca7014d5aa6a48a3808fd94e90209c56499dbcd7df9f0fe733a29a7681`
separately binds control/verifier source revision
`c1f8168f230732f2d745c234555b3e3dfcb8aefa`. From the current reproduction
harness in a full-history Git checkout, run:

```bash
ENDPOINT_ROOT=/path/to/arc_agi3_gkm_v2_181/artifacts \
ACQUISITION_ROOT=/path/to/arc_agi3_gkm_v2_181/acquisition_source \
RELEASE_RECEIPT=/path/to/arc_agi3_gkm_v2_181/receipts/140e37ca7014d5aa6a48a3808fd94e90209c56499dbcd7df9f0fe733a29a7681.json \
make -C arc/manuscript reproduce
python arc/crack_lab/replay_scorecard.py \
  --mode online --games all \
  --artifact-root arc/crack_lab/releases/arc_agi3_gkm_v2_181/artifacts \
  --release-receipt arc/crack_lab/releases/arc_agi3_gkm_v2_181/receipts/140e37ca7014d5aa6a48a3808fd94e90209c56499dbcd7df9f0fe733a29a7681.json \
  --expected-claimed-levels 181
```

The two roots are deliberately distinct. The normalized endpoint capsules are the
authority for receipt, replay, taint, action, hash, and manifest checks. The frozen
acquisition programs are the authority for retained description length (D(s)), the
marginal-complexity table, and figures. Source-coupled reuse is recomputed from the
manuscript's frozen source-history snapshot at commit
`4d0e42f34d7b1db8305f03d725528dfdefe22511`, which the target extracts from local
Git rather than reading the mutable live campaign. That authority is distinct from the
artifact-publication commit `9235ed...` and receipt-bound verifier revision `c1f8168...`.
The release result is 181/183 replay-verified wins. The narrower winning-source
audit admits 174 exact checkpoints and excludes the seven deterministic path reconstructions
(`ft09` L2 and `tr87` L1--L6) from source-complexity denominators.
The final command performs a zero-LLM endpoint replay against public remote environments
and therefore requires the ARC API environment and credentials. See
[`../../REPRODUCE_ARC.md`](../../REPRODUCE_ARC.md) for the complete protocol and
security boundary.

`verify_frozen_release.py` automatically extracts the receipt-bound historical
verifier/control context from local Git; it never rebuilds the old receipt with current
controls. Full source/reuse reproduction therefore expects a full-history Git checkout.
Without `.git`, set both
`RELEASE_VERIFIER_ROOT=/path/to/extracted/c1f8168-source` and
`HISTORY_ROOT=/path/to/extracted/4d0e42f-agent-solutions` before `make reproduce`.
The latter is copied privately and admitted only if its complete Git-tree digest is
`85543629cfcafc70eb7230493f394059c8e0ac45`.
For endpoint-only scorecard preflight, pass the former tree with
`--release-verifier-root`; no source-history tree is needed.

The active/contiguous acquisition policy, including the complexity-triggered
independent side-expert and supervisory-proposer roles, is separately
machine-readable:

```bash
python arc/crack_lab/arc_agi3_contiguous_scheduler.py policy
PYTHONPATH=arc/crack_lab pytest -q \
  arc/crack_lab/test_arc_agi3_contiguous_scheduler.py
```

The side expert is not an `ultra` effort alias. In the exploratory campaign it
was an in-session audit subagent reassigned to an immutable private copy; its
model/effort was not exposed, it shared the host boundary, and its output had no
promotion authority. The contiguous specification preserves the independent,
orthogonal, public-observation analysis while adding a pinned launch manifest,
container isolation, journaled assignment/usage, and mandatory
taint/provenance/fresh-replay admission. The full manuscript reproduction pass
must verify these scheduler tests and must not claim the prospective controls
for the exploratory lineage. In the contiguous design, the scheduler may
allocate side-expert capacity but may not author the semantic brief: each
assignment is bound to an authenticated same-frontier native-proposer request
or an admitted supervisory handoff, with stale, cross-frontier, manual, and
substituted requests rejected.

The supervisory proposer is a separate, receipt-bound LLM role for an already
selected hard frontier. It may synthesize only authenticated native and
side-expert evidence into a quarantined `SUPERVISORY_HANDOFF`; it has no
scheduling, mutation, or promotion authority. The native proposer receives the
handoff only as an unverified hypothesis and must reproduce its cited
observations through the public interface before any derived solver evidence
can advance. A typed sidecar request inside that handoff may supply a
same-frontier side-expert brief, but it does not allocate the slot or gain
evidence authority.

Current prospective campaign code additionally supports bounded failure-revision
rounds in fresh ephemeral proposer threads. The aggregate is admissible only when its
protocol digest, frontier binding, ordered round records, sealed diagnostics, and
terminal or promotion evidence authenticate together; malformed, cross-frontier,
tainted, or incomplete aggregates fail closed. This mechanism postdates the frozen
181-boundary result and is not part of the historical release claim. The release
continues to be interpreted solely under the verifier/control revision bound by its
schema-v2 receipt.
