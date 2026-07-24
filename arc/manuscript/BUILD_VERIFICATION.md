# Build verification

Verified in the integrated repository on 2026-07-24.

## Documents

- `arc_agi3.pdf`: 24 pages, as reported by the final LaTeX log.
- `gkm_one_page_summary.pdf`: one page.
- `figure_sources/inverse_colimit_attachment_standalone.pdf`: one page.
- Abstract: 113 words after LaTeX stripping with `detex`; no displayed mathematics
  or benchmark arithmetic.
- The final logs contain no undefined citation/reference, overfull/underfull box, or
  LaTeX/package warning.

The canonical paper introduces “Gödel–Kolmogorov Machine” five times before defining
`GKM` after the contribution list. The abstract uses only the full name.

## Figures

The integrated build used Matplotlib 3.10.9 and regenerated:

- `figures/ls20_sawtooth.png`: 1728 × 912 pixels;
- `figures/bounded_campaign_profiles.png`: 2034 × 1072 pixels.

The generator's numerical inputs and geometry checks pass. The supplied bundle pinned
Matplotlib 3.10.8; its raster files are byte-identical only under that pinned rendering
environment. The 3.10.9 integration outputs are therefore treated as data-and-geometry
reproductions, not falsely reported as pixel-identical copies.

## Repository checks

- `python arc/manuscript/build_artifact_history.py --check`: passed.
- Focused replay, checkpoint, campaign-control, taint, and literal-reuse suite:
  87 tests passed.
- Canonical taint scan: 138 promoted files, zero hits. All 23 promotion-chain
  summaries have zero taint and integrity hits; older pre-manifest artifacts are
  reported with zero manifests rather than assigned invented ancestry.
- `python3 -m sphinx -W -b html docs docs/_build/html`: passed.
- `python -m py_compile arc/manuscript/scripts/generate_figures.py`: passed.

The integration did not rerun the remote ARC scorecard or stochastic discovery
campaign. Those operations require the ARC service and credentials; the manuscript
build and offline artifact gates do not. See `../../REPRODUCE_ARC.md` for that boundary.
