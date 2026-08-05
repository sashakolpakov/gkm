# Grounded Bongard status — 2026-08-05

## Bottom line

The previous prose-membership result is falsified as a benchmark.  The
corrected grounded path passes the failure-derived point-contact regression
and honestly refuses a fresh bird-like problem that lies outside its frozen
observable catalog.  This is a repaired semantic boundary, not yet a broad
Bongard solver.

## Why the old result failed

The old object was proposition-conditioned perception:

```text
VLM(panel, all candidate rubrics) -> 0 / 0.5 / 1
```

Proposal and scoring could therefore select different parts, contacts, and
reference frames.  On the saved v4 run the scorer asserted a point contact in
negative panels where the typed image graph found none, and its three-level
memberships had no calibration semantics.  The alleged rotated LOO refit only
a threshold after proposal had seen every label: its error counts were exactly
six times support errors (`3 -> 18`, `5 -> 30`, `4 -> 24`).  That is threshold
sensitivity, not cross-validation.

## Corrected object

```text
support pixels
  -> headless Codex selects IDs from a closed catalog
  -> one candidate-independent pixel witness per panel
  -> typed observations with intervals/provenance
  -> support-only positive Boolean synthesis
  -> frozen formula
  -> hidden nuisance rerender, with no model call or refit
```

`SemanticAbsent` is a certified negative fact.  A failed fit is
`Indeterminate` and blocks admission.  The proposer cannot author code,
thresholds, memberships, or negation.

## Live repaired regression

Artifact: `semantic_grounded_runs/codex_eod_20260805_v1/campaign.json`

- Problem: `bd_mismatch_sector_rec2_0391`.
- Model: `gpt-5.6-sol`, medium reasoning, one isolated support-image turn.
- Codex selected small exterior gap (`low`), large exterior gap (`high`), and
  exterior-gap ratio (`high`).
- Harness selected the one-atom formula
  `point_contact_exterior_gap_ratio > 2.877074861773756`.
- Support: `12/12`, zero errors, zero indeterminates.
- Hidden rerender seed `20260806`: `12/12`, zero errors, zero indeterminates.
- Taint: `PURE` (conditional on the registered deterministic extractor).
- Cold replay: valid, `1/1` solved.
- Campaign digest:
  `sha256:9eb105f516fdc39b6fd359bf11bb96dba25acd20cbcb11a9de284b31c4ba070f`.

Additional deterministic nuisance stress (not additional model turns) passed
all 12 panels at seeds `20260807` and `20260905` as well.  Across the three
hidden seeds, the smallest positive conservative ratio lower bound was
`3.3681`; the largest present-negative upper bound was `1.9934`, leaving the
frozen `2.8771` threshold strictly between them.  Five of six negatives per
render carried a certified absence of the complete point-contact signature.

This regression is post-hoc with respect to representation design: the
point-contact signature was added after examining this failure.  It establishes
that the repaired witness and evaluation protocol survive nuisance rerendering;
it is not an unbiased estimate of novel-concept performance.

## Fresh frozen-catalog probe

Artifact: `semantic_grounded_runs/codex_blind_bird6_20260905_v1/campaign.json`

- Problem: `bd_bird6_0342`, selected only after the catalog/code were frozen.
- Codex described a balanced smooth two-lobe gestalt, but could select only the
  catalog's point-contact gap observables.
- Synthesis result: `no proposed intent produced a determinate positive atom`.
- Status: `UNSOLVED_SEMANTIC_GROUNDED`; no formula, no scored query claim.
- Cold replay: valid, `0/1` solved.
- Campaign digest:
  `sha256:012e3cf4c951a68a8e8d48d9c6df1f0fb140ee4c000ae16958cf2a6067fd68ee`.

One preceding live turn failed in transport before returning a proposal and is
classified as infrastructure failure, not as an empirical result.  The clean
retry above is the scientific artifact.

## Live open-vocabulary HYBRID probe

Artifact: `semantic_hybrid_runs/codex_bird6_latent_20260905_v1/campaign.json`

- Problem: `bd_bird6_0342` (`bird6`).  The support/query action strings are
  content-distinct, but Bongard-LOGO uses the same basic bird template.  This
  is a latent-program style/pose holdout, not novel bird-instance evidence.
- One headless `gpt-5.6-sol` support turn (medium reasoning) froze the literal
  affirmative claim `A single open S-shaped curve.`
- The executable leaf is
  `OperationalResemblance(frozen_bundle, target)`, not a theorem that the
  prose claim is true of the pixels.  It is explicitly `HYBRID-EXPLORATORY`
  and has no probability calibration.
- Three content-selected anchor/foil pairs were shown with one neutral target
  in two fresh turns.  The second turn reversed pair order and swapped every
  left/right placement.  A pair counted only when both turns agreed after the
  hidden role normalization.
- Query result: `8/12` correct, `9/12` determinate coverage, `8/9`
  determinate-only accuracy, three abstentions, and zero errors.  Positives
  were `6/6`; negatives were two correct, three abstentions, and one false
  positive.  Status: `UNSOLVED_HYBRID_EXPLORATORY`.
- There were 25 distinct authenticated Codex threads: one proposal and 24
  scorer turns.  There were no threshold fits, polarity flips, support
  self-scoring passes, semantic retries, or query labels in scorer inputs.
- Cold evidence replay is valid with zero model calls and explicitly records
  `perception_reexecuted: false`.
- Campaign digest:
  `sha256:5f4caff166d38a020b1a7043bd8146bc72f369828a69f99cabd2db56a870291a`.

The failure is informative.  The false-positive query is a single open
U-shaped curve.  The claim-relevant distinction is the reversal of signed
curvature in an S, but all three frozen foils happened to be closed compound
outlines.  The model therefore over-weighted the shared `single open curve`
topology and called the U an S.  A U-shaped support negative existed in the
six-foil pool but the claim-seeded content-rank selector omitted it.  Three
other negatives changed role choice under the fully swapped presentation and
correctly became indeterminate instead of silently becoming negative facts.

This exposes two separate next requirements: a candidate-independent contour
witness for open/unbranched structure plus a robust signed-curvature reversal,
and a preregistered reference policy with hard-negative/diversity coverage
(or all six pairs).  Either change is post-hoc for this artifact and must be
tested on a new frozen query; rerunning this query cannot be reported as a new
benchmark result.

## What “rigorous soft predicate” can mean

Prose cannot become an extensional predicate merely by translating it into
Lean.  For `bird-like`, rigor must attach to a frozen operational oracle:

1. freeze the text claim, transformation policy, positive prototypes, and hard
   negative foils before query evaluation;
2. evaluate each panel independently by contrastive comparisons against those
   frozen anchors, never in a batch of competing candidate rubrics;
3. aggregate repeated/paired comparisons with an externally calibrated
   abstention region;
4. preserve oracle provenance and uncertainty as a typed observation;
5. test the frozen observable on hidden panels and label it `HYBRID`.

A proof assistant can verify the Boolean IR, units, interval semantics,
content bindings, and replay theorem conditional on the oracle outputs.  It
cannot prove the visual ontology of “bird-like” from unformalized pixels.

## Next benchmark gate

Do not report a broad score until the observable catalog (including any
contrastive HYBRID oracles) is frozen first and evaluated on a preregistered,
unseen problem set.  Report separate coverage, abstention, and conditional
accuracy:

- catalog coverage: fraction receiving any determinate grounded formula;
- conditional support/query accuracy among covered problems;
- indeterminate/error rate;
- PURE and HYBRID results separately.
