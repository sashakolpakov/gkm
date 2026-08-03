# Build verification

Verified in the integrated repository on 2026-08-03 against frozen release commit
`9235ed26627140460efa1f6ca5e4041470cddc14` and schema-v2 receipt
`140e37ca7014d5aa6a48a3808fd94e90209c56499dbcd7df9f0fe733a29a7681`.

## Documents

- `arc_agi3.pdf`: 27 pages, 1,066,615 bytes.
- `gkm_one_page_summary.pdf`: one page, 180,729 bytes.
- The final logs contain no warning, undefined citation/reference,
  overfull/underfull box, or multiply-defined-label match.
- The manuscript byline and PDF metadata name Alexander Kolpakov solely.
- The ignored local PDFs are build products, not checksum-manifest inputs, because
  pdfTeX embeds build-time metadata. `SHA256SUMS.txt` covers the canonical sources,
  generated empirical tables/figures, reports, and audit outputs instead.
- The deterministic six-file arXiv source archive builds from its own upload root
  through PDFLaTeX and BibTeX. It contains no companion documents, repository evidence,
  caches, compiled PDFs, or LaTeX intermediate files.
- The rebuilt source archive SHA-256 is
  `3a1d879680d08cf078dd84c1d47ea38c29853ae0e42fb525c4e062f3f28b70a5`.

## Frozen empirical endpoint

- 25 games, 181/183 admitted boundaries, and 7001 stored replay actions.
- Official Competition-Mode score: 98.11664037825032% in 7069 API calls.
- Distinct raw coverage: 181/183 = 98.907103825137%.
- Revision-bound release verification invokes the gate and controls from source
  revision `c1f8168f230732f2d745c234555b3e3dfcb8aefa` and revalidates the complete
  sealed audit/evidence chain: 181 endpoint/action-boundary certificates; zero taint, action-protocol,
  replay-record, hash, manifest, or promotion-chain failures; `lf52` L9--L10 are
  the only explicitly unclaimed boundaries.
- Exact winning-source checkpoints are admitted for 174 of the 181 replay-verified
  wins. `ft09` L2 and `tr87` L1--L6 are replay-valid deterministic path reconstructions
  and are excluded from historical source-complexity denominators. The source/reuse
  audit is extracted from immutable
  source-history revision `4d0e42f34d7b1db8305f03d725528dfdefe22511`, distinct
  from both the endpoint publication commit and receipt-bound verifier revision.

## Tables and figures

- `generated/marginal_complexity_by_level.{md,tex,json}` and
  `docs/generated/marginal_complexity_by_level.rst` contain all 25 games and one
  explicit cell for every authoritative level. Their (D(s)) values and marginal
  ledgers come from the frozen acquisition-source tree, not the smaller normalized
  endpoint capsules; this corrects the former mixed-source `tr87` value to 1026.
- `figures/marginal_complexity_profiles.png`: 2448 x 912 pixels. It distinguishes
  the strongest raw oscillation (`su15`), strongest complete uniform-history
  sawtooth (`wa30`), and strict source-coupled reuse example (`ls20`).
- `figures/ls20_sawtooth.png`: 1728 x 912 pixels.
- `figures/bounded_campaign_profiles.png`: 2034 x 1072 pixels.

## Repository checks

- Unified release-bound manuscript reproduction: passed.
- Manuscript test target: 228 tests passed.
- Fixed-scope SHA-256 integrity manifest: passed.
- Sphinx 9.1.0 strict HTML build (`-W`): passed.
- Main paper and one-page PDF builds: passed.
- Receipt-bound all-game scorecard preflight: 25 games, 181/183 claimed levels,
  7001 stored actions, and 181 locally segmented boundaries passed before any
  remote replay.
- Current source audit: 174 admissible exact winning-source checkpoints, 149 exact
  adjacent transitions, 148 comparable marginals, 68 decreases, 23 sharp drops,
  57 hard direct-reuse
  witnesses, and 14 coupled sharp-drop-plus-reuse witnesses.

## Scope

The release receipt certifies endpoint reproduction, evidence integrity, and the
declared audit boundary. It does not reproduce stochastic discovery transcripts,
measure clone-enabled search interaction, or turn the harness-native scalar ledger
into a semantic reuse certificate.
