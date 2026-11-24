# Current submission verification

The canonical manuscript and submission files use the selected **GPT-5.6 campaign**.
The manuscript title is unchanged. Conditional marginal complexity is the central
question; working-cycle and admitted-state source-growth counters are supporting
accounting. Canonical outputs are overwritten in place, with no alternate drafts.
The separate theoretical manuscript and active discovery campaign are unchanged.

## Submission files

- `arc_agi3.pdf`: **24 pages**, 1,804,100 bytes.
- `response_to_reviews.pdf`: **one page**, 92,830 bytes.
- `gkm_one_page_summary.pdf`: **one page**, 213,023 bytes.
- `build/arxiv/arc_agi3_arxiv.zip`: **ten source files**, 1,214,737 bytes.
  Its main source matches the canonical TeX; isolated PDFLaTeX/BibTeX compilation
  passed. SHA-256: `8b62ce2deb25c760e7d32c897486397972bbce8d212b2226ba42e4962f851337`.

The three PDFs and ZIP are canonical versioned outputs. Source checksums exclude
compiled PDFs; Git records their exact blobs. PDFLaTeX logs contain no warnings,
undefined references/citations, or overfull/underfull boxes. The main marginal
figure and full-suite figure were inspected both directly and on rendered PDF pages.

## Marginal-complexity evidence

The fixed compressed-AST novelty proxy M is defined separately from ideal
K(P_k | P_{k-1}), source-growth counters C/G and discovery effort. M1 uses an
empty AST dictionary. Missing exact source, missing predecessors and unclaimed
levels remain unknown; plots never bridge those gaps or fill them with zero.

The main figure plots dc22, wa30 and ls20; the full-suite appendix plots all 25 games.
Orange rings mark decreases; green diamonds mark sharp drops with direct unchanged-leg
call sites. Both PNG and vector PDF outputs were generated. The ls20 standalone
figure also now plots M rather than its source-growth surrogate. The five figure
pairs have checked PNG geometry: 1728×912, 2034×1072, two at 2448×912, and 2688×2880.
The data and input hashes are in `generated/conditional_marginal_profiles.{json,md}`.

The source-only audit was regenerated from immutable Git blobs and matched both
`marginal-literal-reuse.json` and `revision_source_audit.json` byte for byte. It admits
**171 exact winning sources, 144 adjacent transitions and 141 comparable marginals**,
with **64 decreases, 21 half-or-more drops, 54 static call-site cases and 13 sharp
coupled static cases**. Runtime call execution remains unmeasured.

For dc22, M is 1058, 604, 716, 885, 828, 1379 bytes: it has dips at L2 and L5 even
though G increases strictly over authenticated L2–L6. Both dips call unchanged
walk_segments and toggle_control through newly added routines. Both wa30 troughs
similarly reach unchanged walk and interact through new routines. These concrete
indirect reuse paths are excluded by the fixed narrow direct-call statistic.

A new case audit binds previous/current file hashes, unchanged-definition hashes
and call paths for those four dips and the contrasting cn04 L3 player. The latter
has 23 literal actions with env.step as its only call, while M falls from 337 to
245 bytes. The claim concerns that new player's bypass of retained solver legs.
All five source pairs and helper/entry-point AST hashes were independently checked
against the frozen Git history. The case table was also inspected on the PDF page.
The canonical case outputs are `generated/reuse_case_audit.{json,md,tex}`; the
original suite-wide direct-call counts and admission rule are unchanged.

## Supporting accounting and frozen scope

The endpoint release remains **181/183** public levels, **98.1166%** Competition-Mode
score, 7001 stored actions and 7069 official API calls. These do not measure discovery
cost. The publication is `9235ed26627140460efa1f6ca5e4041470cddc14`; the schema-1
partial receipt is `140e37ca7014d5aa6a48a3808fd94e90209c56499dbcd7df9f0fe733a29a7681`.
The receipt-bound verifier is `c1f8168f230732f2d745c234555b3e3dfcb8aefa`.
Source history is `4d0e42f34d7b1db8305f03d725528dfdefe22511`, tree
`85543629cfcafc70eb7230493f394059c8e0ac45`.

The separate paired counter audit has 158 manifest-bound retained states and 134 C/G
pairs in 24 games, with initial-scaffold and reconstruction exclusions. It remains
supporting evidence. Recorded C refers to working source before the winning cycle;
G compares authenticated adjacent promotions. Neither reconstructs all failed edits.
The dc22 full recorded total remains 650; paired L2–L6 sums are C=601 and G=801,
with retained size rising from 50 to 851. The solver, counter and replay were valid.

## Checks and scope

**69 targeted tests passed**, covering source reconstruction and reuse analysis,
profile gap handling, conditional-versus-growth data, indirect wrapper paths,
duplicate-player resolution, shadowed names, excluded uncalled nested bodies,
figure series, artifact docs,
package closure, reproduction helpers and the source checksum manifest.

Strict Sphinx HTML passed all 10 documentation sources, including both new marginal
figures. The three canonical PDFs compiled; the ten-file source ZIP compiled in
isolation. Source analysis is static: no discovery run, game implementation, runtime
harness or frozen solver was modified. Earlier endpoint/receipt/taint/action evidence
was retained; no new online replay or endpoint gate is claimed here. Runtime calls,
causal search savings, full discovery cost and held-out performance remain unmeasured.
