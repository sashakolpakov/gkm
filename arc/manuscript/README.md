# Gödel–Kolmogorov Machine ARC-AGI-3 manuscript

This directory contains the canonical manuscript and reproduction sources for the
Gödel–Kolmogorov Machine ARC-AGI-3 study. The Gödel–Kolmogorov Machine combines
verifier-gated program revision, inverse-colimit attachment, PowerPlay-style retained
competence, artificial curiosity, and description-length selection. After this full
introduction, **GKM** is used as the abbreviation in filenames and dense comparisons.

## Canonical deliverables

- `arc_agi3.tex` and `references.bib`: the integrated 25-page paper source.
- `arc_agi3.pdf`: the generated paper; build it locally rather than treating a bundled
  binary as the source of truth.
- `scripts/generate_figures.py`: exact source for the two empirical ledger figures.
- `figure_sources/inverse_colimit_attachment_standalone.tex`: standalone source for
  the inverse-colimit attachment diagram, which also appears inline in the paper.
- `requirements-figures.txt`: the Matplotlib version used for the delivered figure
  geometry.
- `development_memo.md`, `socratic_development_audit.md`, and
  `repo_ground_truth_matrix.md`: the editorial, mathematical, and code-to-claim record.
- `arc_agi3_original_to_forward_revised.diff`: provenance diff supplied with the
  forward revision. The canonical repository filename remains `arc_agi3.tex`.
- `BUILD_VERIFICATION.md`: results and limits of the repository integration checks.
- `SHA256SUMS.txt`: integrity manifest for the integrated source deliverables.

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

The figure generator validates the expected PNG dimensions under Matplotlib 3.10.8:

- `figures/ls20_sawtooth.png`: 1728 × 912 pixels;
- `figures/bounded_campaign_profiles.png`: 2034 × 1072 pixels.

Matplotlib 3.10.8 reproduces the supplied bundle byte-for-byte. Compatible later
versions can preserve the numerical data and validated geometry while changing raster
bytes through rendering details; the integration verification records the version used.

## Repository-level reproduction

The manuscript build reproduces the document, not the stochastic discovery campaign.
The repository supplies separate gates for the retained evidence:

```bash
python arc/manuscript/build_artifact_history.py --check
PYTHONPATH=arc/crack_lab pytest -q arc/crack_lab/test_gkm_legs.py \
  arc/crack_lab/test_replay_scorecard.py
python arc/crack_lab/replay_scorecard.py --mode online \
  --games wa30,ls20,ft09,g50t,r11l,sp80,tr87,tu93
```

The first command checks immutable history sidecars. The second runs offline metric,
checkpoint, action-encoding, locking, and taint tests. The final command performs a
zero-LLM endpoint replay against public remote environments and therefore requires
the ARC API environment and credentials. See [`../../REPRODUCE_ARC.md`](../../REPRODUCE_ARC.md)
for the complete protocol and security boundary.
