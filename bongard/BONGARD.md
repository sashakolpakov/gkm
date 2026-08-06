# Bongard documentation pointer

The canonical operational description is [README.md](README.md). The current
execution order and stopping rules are in
[CONTINUATION_PLAN.md](CONTINUATION_PLAN.md). Falsified baselines and the
documentation cleanup are recorded in [HISTORY.md](HISTORY.md).

The short version:

- Python is the sole runtime semantic authority. It defines predicates, the
  closed IR, evidence dispositions and projections, calibration, synthesis,
  selection, evaluation, persistence, cold replay, and decisions. Lean is
  neither imported nor required. The intended checker boundary is a detached,
  process-isolated, non-authoritative sidecar over persisted bytes. Attempt
  three ran Python-only and invoked no checker, but strict un-Lean migration
  remains incomplete: its
  provenance binds an earlier whole-tree commit containing checker source, its
  configuration retains the legacy `python-canonical/lean-optional/v1` literal,
  and the optional checker API remains in-process rather than process-isolated.
- Direct predicates come from a finite ten-entry exact-count catalog.
- Soft prose such as “bird-like” is a frozen operational measurement protocol,
  not a proof about pixels. An uncalibrated observer nonmatch is a distinct
  operational record and projects to semantic indeterminacy, never certified
  absence.
- A1 failed before scoring: 48 proposer calls succeeded, yielding 37 accepted
  soft claims, 10 direct-only records, and 1 parser rejection, but all 37
  scorer calls were transport errors. Labels remained withheld and no score,
  calibration, semantic accuracy, or negation evidence was produced.
- A2 was a distinct repaired-protocol DRILL experiment, not an A1 retry. It was
  **INVALIDATED BY LIVE SOURCE MUTATION** after a concurrent agent edit changed
  `typed_visual_proposal.py` following protocol/cohort freeze.
- A3 completed with exit 2 as a canonical scientific failure. All 22 proposer
  calls and all 15 attempted scorer calls succeeded. The funnel was 15 accepted
  soft claims, 6 direct-only records, and 1 parser rejection; scores were
  `0:8`, `0.5:1`, and `1:6`. The fixed lower bin had 9 clusters/1 affirmative,
  but the upper bin had only 6 clusters/5 affirmatives against the minimum of
  8, so no calibration was fitted.
- A3's intended-bin orientation was 13/15 versus 2/15 for the exact complement;
  at `score >= 0.5` it was 12/15 versus 3/15. Negation did not win.
- Stage B is unauthorized by A1, A2, and A3.
- Corrected capacity is semantic, not raw-task count: immediately before A3,
  DRILL had 24 `bd` + 0 constituent-disjoint `hd` = 24; exact v3 replay against
  the A3 successor ledger now gives strict DRILL capacity zero. DEV remains 16
  `bd` + 0 ledger-disjoint `hd`. The earlier 28-unit upper bound failed to
  project complete-A2 exposures into owner-independent HD constituent tokens.
  The old 24-task DEV claim likewise reused HD attributes.
- The post-A3 zero-capacity certificate is
  `sha256:48fba29c8a33a5fd773baed373694ac32d91a6f456b17ede563113eeeecd18b1`.
  The archive is not exhausted: 10,047 of 10,200 train/validation task IDs are
  exact-unused after A3 (FF 2,998; BD 3,434; HD 3,615). The 24-unit limit comes
  from a strict semantic
  independence policy. A future calibration frame should use the larger
  training population with explicit dependence accounting while keeping
  evaluation holdouts strict.
- A3 exposed 22 tasks. Complete-release authentication hashed official-test
  bytes, but no official-test task or panel was selected, exposed to the
  proposer or scorer, evaluated, or scored. A3's exact failure reason was
  `calibration score bins are underpopulated: 1`.
- A3's launcher receipt bound the JavaScript wrapper and reported CLI version,
  not the dynamically spawned native client bytes. This does not reverse the
  observed failure. New runs now stage, hash, execute, and recheck the exact
  native client digest
  `sha256:ae1d3ffe6d48aec6a4dc3f50e7eb8e0d11962485a6a9406c5a7012139383da02`.
- A3's twelve-panel descriptions were audit-only. The atomic smoke freezes 12
  isolated descriptions and proposes 1--12 pairwise-distinct exact affirmative
  observer questions from labelled descriptions only. Each question is at most
  192 UTF-8 bytes, matches `[A-Za-z0-9]+(?:[ -][A-Za-z0-9]+)*\?`, has one final
  ASCII `?`, and receives no normalization or repair. Python records the full
  atom-by-panel matrix and selects a positive conjunction of at most four atoms
  before query release. A successful path has 29 causal model receipts.
- The atomic archive has no `Not` or polarity flip. Its operational scope
  hard-codes calibration, semantic-truth, and benchmark authority false;
  calibrated semantic selection is disabled until a typed calibration
  artifact and interval rule can be cold-validated.
- The original atomic frame contained ten repeated-generator, exact-unseen
  training tasks. Attempt one consumed one. Attempt two historically selected
  from the remaining nine-ID universe, digest
  `sha256:094e195fd8892cf09bcb8287e68bd747fdbb47a87075a60d0d23c291b17466ed`,
  and consumed one more. Attempt three then selected once from the exact
  eight-ID universe, digest
  `sha256:3b1a0ce4f9df6e1f9881fb932ec680a988e76afde860c687154401d005c52ee9`.
  Its successor is
  `sha256:66678615dd766dcababfd57cb0435dfc2e18a366bd2a806127afb00a5b1ecfe6`;
  seven tasks remain, digest
  `sha256:5dc36a2336abdb46d8096f0951739c3d825fb6e88833cb9b19735b426d1df357`.
  These tasks test transport and synthesis, not independent generalization.
- The first live N=1 attempt, from commit
  `62ea577f5d86d109577f4f5e49b8b4866eb76c92` and tag
  `bongard-atomic-pre-smoke-20260806`, is an operational failure. Cache, config,
  and exact-task exposure were persisted, so the selected task is consumed and
  will not be rerolled; no prediction or terminal was persisted. The runner
  returned a typed `AtomicSmokeRun`, after which fallback terminal construction
  rejected its frozen `MappingProxy` precommit. Normal construction contains
  the same defect, although the outer error cannot identify the exception that
  first selected the fallback path. The exact error was `failed run precommit
  is not canonical JSON`, reason digest
  `2825061e41346b498f7ceb0e338b0382fa807b2c968d534703927d6ce5f8376d`.
  Run status, phase, output, and successful call count are irrecoverable; the
  call count is only known to lie in `0..29`. Labels were not materialized or
  revealed, and there is no score, calibration, semantic, benchmark, or
  official-test claim. The sanitized record is
  [`data/atomic_smoke_n1_operational_failure_v1.json`](data/atomic_smoke_n1_operational_failure_v1.json).
- Atomic command stores require mode `0700`. A prior setup launch stopped on a
  `0755` cache store before exposure and consumed nothing.
- Atomic attempt two ran once from commit
  `d0864525146a05795c030674fa0159feb43913c1` and tag
  `bongard-atomic-successor-pre-smoke-20260806`. It durably appended successor
  `sha256:bfd47a3797b4ac840630a4d0207e1fc04be386dba059db0e45e58e249501da8d`
  and closed 13 intents/results: 12 support descriptions and one valid
  text-only proposal receipt. All ten proposed questions ended in the `?`
  demanded by the prompt, but the shared soft-cue parser rejected U+003F. The
  exact error was `invalid positive_description: soft cue positive_description
  contains a forbidden prose character U+003F`, reason digest
  `34b41a10ae89287ed97c875c6833047ff5896a7081debd144f484833292fe42f`.
- Attempt two made no support-scoring or query call, froze no formula, persisted
  no prediction, revealed no label, and produced no score. Run and terminal
  artifacts persisted and cold replay passed. This is an implementation
  contract failure, not vision, predicate, negation, or benchmark evidence. Its
  sanitized record is
  [`data/atomic_smoke_attempt2_proposal_contract_failure_v1.json`](data/atomic_smoke_attempt2_proposal_contract_failure_v1.json),
  file SHA-256
  `242ebc5914020a683a6f34a0b50688bf3190f4c4cbd6d345d15ebb5e775eb6b3`.
- Atomic attempt three ran exactly once from commit
  `89b7124da62a5dd62a55abe7025bbaec8f90794b`, tag
  `bongard-atomic-attempt3-pre-smoke-20260806`. Its preflight and one-shot claim
  passed, and its journal closed 25 validated calls: 12 descriptions, one
  proposal, and 12 support scorings. The sole question was `Is a small triangle
  attached to a tilted quadrilateral?`. All six positives were `present`; the
  negatives were three `operational_nonmatch`, two `present`, and one
  `indeterminate`.
- It terminated at `support-scoring` with `NoExactSeparatorError`. The only atom
  was non-total because of the indeterminate observation, and the two negative
  `present` observations mean resolving it would still not separate support.
  No formula, query call, prediction, query-label materialization or reveal, or
  score occurred. The exact 25-call closed prefix cold-replayed. The post-hoc
  9/12 versus complement 2/12 support diagnostic is not an accuracy or score;
  negation did not win. The sanitized record is
  [`data/atomic_smoke_attempt3_no_exact_separator_v1.json`](data/atomic_smoke_attempt3_no_exact_separator_v1.json),
  file SHA-256
  `533d57f8e0757ecd819c4e5bd95eeb8f2d2478193a366ba67178cf291ee953fc`.
- The failure identifies a representation hole, not a Lean hole. The phrase was
  syntactically one atom but semantically bundled shape, relative size,
  orientation, and directed attachment. The scorer lost object roles and size
  direction. The next leg is candidate-independent typed scene graphs and
  factorized relational micro-predicates evaluated by Python.
- Visual-semantic official-test model execution is disabled. Complete-release
  authentication hashed official-test bytes, but no official-test task or panel
  was selected, exposed to a proposer or scorer, evaluated, or scored.

A1 command receipt:
`sha256:9aa247d953204bb12c06a09af6c081c47ae884be8e9c642a9a2bb6d587ba40cb`.
A1 terminal failure:
`sha256:a130d9e608c38581d34043d4d9c071f93483026592ec9c27a406dbad46d65b83`.

A2 consumed protocol / no-reroll seed / durable successor:

- `sha256:2d9261c763d3f9242ffc7cf42d773f54aa1a51f29b610e10b75c9ae59dea81ca`
- `eb031fe199b7d7553444d29cd213663c8afaf99d9b9cccec896f862f445a40b1`
- `sha256:9b7cb7ee7d759e899f5194d115a8bd20ebf8e078397a64de8f4b32e6805b1ce8`

A2 wrote no Stage-A terminal artifact. Its incident file digest is
`sha256:4ace426bafbc051f2ad620dd8cdb3742a365b43503c673a9acc462665d47ccd4`.
Process output showed 48 proposer and 34 scorer launches only; outputs were
lost, labels were not revealed, no semantic inference is valid, and the same
cohort may not be rerun.

A3 command receipt:
`sha256:2a01933321a0578af51a8db7f2a3c1cf5508908ee4521eb43d7a63f8f7985681`.
A3 terminal failure:
`sha256:cc1b86d7097a1986a7eeb2ddb3a82e30e302ff93a41cf64078be1c5be8df31eb`.
The sole A3 parser rejection came from matching `def` at the start of the
ordinary word `defines`; the parser now requires complete forbidden-keyword
matches. That post-A3 fix does not change the recorded result.
