# Build verification

Verified in the integrated repository on 2026-08-03 against frozen release commit
`9235ed26627140460efa1f6ca5e4041470cddc14` and schema-v2 receipt
`140e37ca7014d5aa6a48a3808fd94e90209c56499dbcd7df9f0fe733a29a7681`.

## Documents

- `arc_agi3.pdf`: 27 pages, 1,065,738 bytes.
- `gkm_one_page_summary.pdf`: one page, 180,307 bytes.
- The final logs contain no warning, undefined citation/reference,
  overfull/underfull box, or multiply-defined-label match.
- The manuscript byline and PDF metadata name Alexander Kolpakov solely.
- The ignored local PDFs are build products, not checksum-manifest inputs, because
  pdfTeX embeds build-time metadata. `SHA256SUMS.txt` covers the canonical sources,
  generated empirical tables/figures, reports, and audit outputs instead.
- The deterministic six-file arXiv source archive builds from its own upload root
  through PDFLaTeX and BibTeX. It contains no companion documents, repository evidence,
  caches, compiled PDFs, or LaTeX intermediate files.

## Frozen empirical endpoint

- 25 games, 181/183 admitted boundaries, and 7001 stored replay actions.
- Official Competition-Mode score: 98.11664037825032% in 7069 API calls.
- Distinct raw coverage: 181/183 = 98.907103825137%.
- Release verification: 181 exact boundaries; zero taint, action-protocol,
  replay, hash, manifest, or promotion-chain failures; `lf52` L9--L10 are the
  only explicitly unclaimed boundaries.
- Exact historical acquisition source is retained for 180 admitted boundaries;
  `ft09` L2 is excluded from source-coupled marginal comparisons rather than
  reconstructed post hoc.

## Tables and figures

- `generated/marginal_complexity_by_level.{md,tex,json}` and
  `docs/generated/marginal_complexity_by_level.rst` contain all 25 games and one
  explicit cell for every authoritative level.
- `figures/marginal_complexity_profiles.png`: 2448 x 912 pixels. It distinguishes
  the strongest raw oscillation (`su15`), strongest complete uniform-history
  sawtooth (`wa30`), and strict source-coupled reuse example (`ls20`).
- `figures/ls20_sawtooth.png`: 1728 x 912 pixels.
- `figures/bounded_campaign_profiles.png`: 2034 x 1072 pixels.

## Repository checks

- Unified release-bound manuscript reproduction: passed.
- Manuscript test target: 193 tests passed.
- Sphinx 9.1.0 strict HTML build (`-W`): passed.
- Main paper and one-page PDF builds: passed.
- Current source audit: 180 exact wins, 154 exact adjacent transitions, 153
  comparable marginals, 70 decreases, 23 sharp drops, 57 hard direct-reuse
  witnesses, and 14 coupled sharp-drop-plus-reuse witnesses.

## Scope

The release receipt certifies endpoint reproduction, evidence integrity, and the
declared audit boundary. It does not reproduce stochastic discovery transcripts,
measure clone-enabled search interaction, or turn the harness-native scalar ledger
into a semantic reuse certificate.
