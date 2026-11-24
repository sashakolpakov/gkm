# Current submission verification

The canonical empirical manuscript, response, companion, and source ZIP report the
selected **GPT-5.6 campaign** throughout. They are overwritten in place; no alternate
manuscript versions or submission backups are retained. The separate theoretical
manuscript and its cover-letter files are outside this revision.

## Submission files

- `arc_agi3.pdf`: **20 pages**, 1,150,243 bytes. The original manuscript
  title and Alexander Kolpakov author metadata are preserved.
- `response_to_reviews.pdf`: **one page**, 163,208 bytes, with matching Markdown
  and TeX sources.
- `gkm_one_page_summary.pdf`: **one page**, 211,769 bytes, with matching Markdown
  and TeX sources.
- `build/arxiv/arc_agi3_arxiv.zip`: **eight source files**; isolated PDFLaTeX/BibTeX
  compilation passed. SHA-256:
  `456fd0fa34671321788c8d1388bfaa37761b2257e16d05fa8b2b7078fd7f14af`.

The three PDFs and ZIP are canonical versioned submission outputs. The source
checksum manifest excludes PDFs because of build-time metadata; Git records their
output blobs. Compiler intermediates are removed after log inspection.

All three PDF builds passed without LaTeX warnings, undefined references/citations,
or overfull/underfull boxes. Layout checks cover the front page, accounting formulas,
`dc22`, paired C/G table and figure, source-audit table, response, and companion.

## Frozen dataset and source interpretation

- Publication: `9235ed26627140460efa1f6ca5e4041470cddc14`.
- Schema-1 partial-release receipt:
  `140e37ca7014d5aa6a48a3808fd94e90209c56499dbcd7df9f0fe733a29a7681`.
- Receipt-bound verifier: `c1f8168f230732f2d745c234555b3e3dfcb8aefa`.
- Source history: `4d0e42f34d7b1db8305f03d725528dfdefe22511`, tree
  `85543629cfcafc70eb7230493f394059c8e0ac45`.
- Endpoint evidence: **25 games, 181/183 levels**, 7001 stored actions, 7069 official
  API calls, and **98.1166%** Competition-Mode score. Raw coverage is separately
  **98.9071%**. These are replay results, not discovery cost.
- Exact winning-source audit: **171 checkpoints, 144 adjacent transitions, 141
  marginal comparisons, 64 decreases, 21 sharp drops, 54 static call-site cases,
  13 coupled static cases, and eight uncoupled sharp drops**. Calls are not
  instrumented and missing source is not bridged.
- `D` is retained source size; `C_recorded` uses the working baseline saved at the
  start of the winning iteration. `G` uses authenticated adjacent promoted states.
  `F_diag` uses recorded C after candidate fixation, without ranking candidates.
- `dc22` retains its recorded full charge of **650**. On authenticated L2–L6 pairs,
  **C sums to 601 and G to 801**, taking **D from 50 to 851**. The different
  baselines explain C=2 versus G=202 at L5. The manuscript interpretation and an
  incomplete audit reconstruction were corrected; **the solver, counters, and
  replay were valid, and no solver repair was required**. Corrected conditional-AST
  novelty at L5/L6 is **828/1379 bytes**.

## Paired recorded charges and promoted growth

The primary audit has **158 manifest-bound states and 134 paired values in 24
games**. Of 20 games with at least three consecutive paired values, **18 reverse in
both series**. Aggregate C/G reversal counts are **45/43**, with only one unequal
primary value: `dc22` L5. Initial G1 is excluded, flat differences are ignored within
known contiguous segments, and gaps are not bridged.

- `dc22`, L2–L6: **2/0** C/G reversals; its recorded trough is absent from G.
- `wa30`, L2–L9: **2/2** reversals; the rise–fall–rise pattern persists in both.
- `ls20`, L2–L7: **1/1** reversals; a shared final downturn rather than repeated
  alternation.

Six `tr87` reconstruction capsules are excluded from historical retained growth.
An extended analysis adds 15 retained-phase candidates without independent promotion
authentication (147 pairs, 52/50 reversals); it is not pooled with the primary sample.
The initial standard-scaffold extension is explicitly conditional and not primary.
Promoted-state evidence here is separate from pre-debrief winning-source evidence for M.

All 25 table rows, the paired comparison artifacts, and all four PNG/PDF figure pairs
were regenerated from frozen evidence. Geometry checks passed: 1728×912, 2034×1072,
and 2448×912 for each of the two three-panel figures. Sources, hashes, provenance
tiers, paired ranges, and exclusions are recorded in `REVISION_EVIDENCE.md` and
`generated/source_charge_comparison.json`.

## Checks

**59 targeted tests passed**, including the paired-source comparison, figure,
artifact-document, source-scope, source-package, and checksum tests. The 10 paired
comparison/package tests also passed after the final placement-only change. The main PDF and eight-file ZIP
were rebuilt after increasing the paired-figure legend font for print legibility;
the data, plot geometry, response, and companion are unchanged. Five figure/package
tests passed after that legend-only refresh.

```bash
PYTHONPATH=arc python3 -m pytest -q \
  arc/test_audit_gkm_solved_checkpoints.py \
  arc/test_audit_marginal_literal_reuse.py \
  arc/test_build_artifact_docs.py \
  arc/manuscript/scripts/test_reproduce_manuscript.py \
  arc/manuscript/scripts/test_generate_empirical_tables.py \
  arc/manuscript/scripts/test_generate_figures.py \
  arc/manuscript/scripts/test_compare_source_charges.py \
  arc/manuscript/scripts/test_build_arxiv_bundle.py \
  arc/manuscript/scripts/test_update_sha256_manifest.py
```

Each canonical TeX document passed
`latexmk -pdf -interaction=nonstopmode -halt-on-error`; the eight-file source ZIP
passed `python3 scripts/build_arxiv_bundle.py --compile-check`.

Strict Sphinx 9.1.0 HTML (`python3 -m sphinx -W -b html docs <temporary>/html`)
passed all 10 sources, including the paired comparison and image. Temporary build
output was removed. The source checksum manifest and `git diff --check` passed.

All new source analysis is static: no solver/game implementation, discovery campaign,
or active campaign was executed or changed. Frozen endpoint/taint/action/receipt
evidence was retained rather than rerun. Actual calls, causal search savings, full
discovery cost, and held-out generalization remain unmeasured.
