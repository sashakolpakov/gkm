# GKM v3: GPT-6.1 zero-seed campaign, 182/183 public levels

This release contains a new GPT-6.1-sol acquisition campaign across all 25 public
ARC-AGI-3 games. It admits 182 of 183 levels (99.4536% raw coverage), completes 24
games, and leaves only lf52 level 10 unclaimed. The retained paths contain 6,685
actions. Raw coverage and stored action counts are distinct from the official
Competition-Mode score and API action accounting.

The campaign began without the previous release's solvers, solutions, action paths
or working notes. Discovery used the same raw frame/action/reward interface and
clone scaffold for every game. All game-specific solver code was model-proposed.
The host supervisor orchestrated runs, replayed candidates, audited provenance and
taint, and admitted passing results. It supplied no game-specific solver advice.
The producer model was gpt-6.1-sol throughout, with the original medium/high/xhigh/max
retry ladder. The 600-action full-game rule was fixed throughout.

A candidate was admitted after exact progress, independent replay, taint and
provenance checks. Complexity and free energy were recorded as diagnostics; they
were not candidate-ranking or admission rules. The retained verified-component
architecture draws on Jürgen Schmidhuber's Gödel-machine and PowerPlay programme.

## Reproducible source and evidence

- `producer_source/` contains the actual acquisition-time runner, policy, prompt,
  harness, taint scanner and their code dependencies, with per-file hashes.
  It contains no game implementations or previous solver library.
- `acquisition_source/` preserves the exact winning files and all 182 original
  promotion boundaries, with their original transcripts and hashes. It contains
  no failed WIP or later search directories.
- `artifacts/` is the separately certified schema-v2 tree for deterministic replay.
  Any reconstruction is explicitly labelled in its provenance and the certification
  report; it is not presented as original learned source.
- `release_identity.json` binds the public release-verification context. The original
  discovery producer is separately pinned below. The later verification context
  extends the receipt's default control manifest to include the GPT-6.1 scheduler,
  policy and shared integrity checks; it changes no solver or discovery code.
- `audits/` and the content-addressed release receipt document the fresh release
  checks. The source protocol records containment repairs and discarded descendants;
  the retained promotion chains pass the strengthened taint audit.
- `accounting/` includes the original token ledger snapshot, capture hashes and
  terminal counters, a deterministic cost report, and per-game totals.

The core campaign-control snapshot is [1492eeb7](https://github.com/sashakolpakov/gkm/tree/1492eeb7817950e2e943b05880907f5edcc6f445).
The release-verification context is [194b3a16](https://github.com/sashakolpakov/gkm/tree/194b3a16af6ac0e1dcbd436aace15dc4783d99ea).

The core snapshot preserves campaign controls separately from later framework
development; `producer_source/` additionally freezes the actual imported audit
helpers and subprocess entrypoints used by the acquisition runner. The API replay uses the frozen admitted action paths with
no model inference or clone calls.

Fresh source execution and independent path replay passed at all 182 boundaries.
No deterministic action-only reconstruction was required. The certification report
preserves its 150 historical-source and 32 retrospective-certification labels.

## Reporting cutoff and plateau

The reported acquisition window ends retrospectively at the current-frontier
retry n125: 2026-11-24 09:56:21 UTC, thread
`01a20cc3-31b4-76d3-88d4-389b77a8bef5`. This is execution 454 of the campaign,
not the 125th proposer turn or API request. Retry labels on earlier discarded frontiers are distinct.
All retained promotions predate this cutoff; the latest was on November 21.

The 641-action diagnostic completion was already available before that cutoff.
It needs 598 actions for levels 1--9 and 43 for level 10, exceeding the fixed limit
by 41 actions. It is excluded from the claimed result and score. Subsequent retries
plateaued at the same best route. Their usage remains in the `after_cutoff` and
`all_recorded` accounting; the retrospective cutoff does not erase that work.

[Token accounting and API-equivalent costs](accounting/README.md) price reported
usage with explicit standard and long-context scenarios. Three interrupted historical
turns have no terminal usage counter and remain unpriced. These are estimates from
recorded tokens, not an API invoice or a complete measurement of local compute.

## Relation to the paper and previous submission

The canonical manuscript currently analyses the selected older GPT-5.6 dataset.
This new acquisition is published separately; its results have not been silently
substituted into those figures. Version 2 and its historical scorecard remain
preserved. The new Community Leaderboard entry appends version 3 and links its own
Competition scorecard. Official numeric scores must be read from that scorecard,
not inferred from 182/183 raw coverage.

The architectural manuscript is [Kolmogorov Structure Functions, Inverse-Colimits,
and Cofibrations: Solving ARC-AGI-3 with Compute-Bounded Self-Improving Agents](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7115218).
The separate theoretical companion is [Pushout Attachments and Conditional
Complexity of Executable Models](https://arxiv.org/abs/2609.25024).

## Frozen acquired endpoints

| Game | Verified levels | Stored path actions |
| --- | ---: | ---: |
| ar25 | 8 | 255 |
| bp35 | 9 | 356 |
| cd82 | 6 | 73 |
| cn04 | 6 | 191 |
| dc22 | 6 | 473 |
| ft09 | 6 | 75 |
| g50t | 7 | 293 |
| ka59 | 7 | 350 |
| lf52 | 9 | 598 |
| lp85 | 8 | 79 |
| ls20 | 7 | 353 |
| m0r0 | 6 | 202 |
| r11l | 6 | 95 |
| re86 | 8 | 596 |
| s5i5 | 8 | 269 |
| sb26 | 8 | 124 |
| sc25 | 6 | 149 |
| sk48 | 8 | 580 |
| sp80 | 6 | 166 |
| su15 | 9 | 141 |
| tn36 | 7 | 168 |
| tr87 | 6 | 153 |
| tu93 | 9 | 185 |
| vc33 | 7 | 167 |
| wa30 | 9 | 594 |
