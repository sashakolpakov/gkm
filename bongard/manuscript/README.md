# Bongard manuscript

[`free_energy_abstraction.tex`](free_energy_abstraction.tex) is the sole
current manuscript: *Operational Visual Predicates for Bongard-LOGO: A
Python-Authoritative Protocol*.

The paper describes the implemented visual-semantic boundary, not a successful
benchmark result. Its central claims are deliberately narrow:

- Python alone defines predicate execution, calibration, replay, benchmark
  decisions, and scientific artifact IDs. Lean or another proof checker may
  only consume a frozen artifact as a detached optional sidecar; deleting it
  cannot change a result, decision, or ID.
- The direct catalog contains exactly ten positive count families, with target
  counts 1 through 8. Angle bands, complete point-contact signatures, and
  persistent object/part identity are not implemented direct predicates.
- Soft prose such as “bird-like” is a frozen operational observation. It is
  not a proof about the pixels. An uncalibrated observer nonmatch has a
  distinct atomic record and projects to semantic indeterminacy, not certified
  absence.
- A1 failed before scoring. Its receipt is
  `sha256:9aa247d953204bb12c06a09af6c081c47ae884be8e9c642a9a2bb6d587ba40cb`
  and its terminal failure is
  `sha256:a130d9e608c38581d34043d4d9c071f93483026592ec9c27a406dbad46d65b83`.
  All 48 proposer calls succeeded: 37 emitted accepted soft claims, 10 were
  direct-only, and 1 was rejected by the typed parser. All 37 scorer calls were
  transport errors. There were zero scores, labels remained withheld, and no
  calibration, semantic accuracy, or negation evidence was produced.
- A2 was a distinct repaired-protocol experiment, not an A1 retry. It used
  protocol
  `sha256:2d9261c763d3f9242ffc7cf42d773f54aa1a51f29b610e10b75c9ae59dea81ca`,
  fresh seed
  `eb031fe199b7d7553444d29cd213663c8afaf99d9b9cccec896f862f445a40b1`,
  and durable successor
  `sha256:9b7cb7ee7d759e899f5194d115a8bd20ebf8e078397a64de8f4b32e6805b1ce8`.
  A concurrent agent edit changed `typed_visual_proposal.py` after freeze, so
  A2 was invalidated by live source mutation. It wrote no Stage-A terminal
  artifact. Its incident file digest is
  `sha256:4ace426bafbc051f2ad620dd8cdb3742a365b43503c673a9acc462665d47ccd4`.
  Process output showed 48 proposer and 34 scorer launches only; outputs were
  lost, labels were not revealed, and no semantic inference is valid. The same
  cohort may not be rerun.
- A3 completed the headless proposer/scorer path and exited 2 as a canonical
  scientific failure with reason `calibration score bins are underpopulated:
  1`. Its command receipt is
  `sha256:2a01933321a0578af51a8db7f2a3c1cf5508908ee4521eb43d7a63f8f7985681`
  and its terminal failure is
  `sha256:cc1b86d7097a1986a7eeb2ddb3a82e30e302ff93a41cf64078be1c5be8df31eb`.
  All 22 proposer calls succeeded (15 accepted soft, 6 direct-only, 1 parser
  rejection), as did all 15 scorer calls. Scores were `0:8`, `0.5:1`, `1:6`.
  The lower bin had 9 clusters/1 affirmative and the upper had 6/5, below the
  fixed 8-cluster minimum, so no fit was produced.
- A3's intended-bin orientation was 13/15 versus 2/15 for the complement; at
  `score >= 0.5`, it was 12/15 versus 3/15. Negation did not win. The parser
  rejection was the ordinary word `defines` accidentally matching forbidden
  `def`; the complete-keyword parser fix was made after A3.
- Stage B is unauthorized by A1, A2, and A3. The completed semantic-capacity
  audit found exactly 24 BD + 0 constituent-disjoint HD = 24 DRILL units before
  A3, so
  the old 48-task design is impossible. Against the complete live ledger, DEV
  has 16 BD + 0 HD units. The default 24-task Stage B fails before pixels, and
  a 16-task BD-only pilot cannot meet its 24-cluster minimum.
- The earlier 28-unit upper bound failed to project complete-A2 exposure into
  the HD constituent-token exclusion set; every remaining DRILL HD pair
  intersects that set. This is not corpus exhaustion: after A3, 10,047 of
  10,200 train/validation task IDs remain exact-unused (FF 2,998; BD 3,434; HD
  3,615). The 24-unit ceiling
  comes from applying a
  strict constituent-independence rule to calibration. A future calibration
  frame must exploit the larger training population with explicit dependence
  accounting while retaining a stricter evaluation holdout.
- A post-A3 audit found that its launcher digest authenticated the JavaScript
  wrapper plus reported CLI version, not the native client selected by that
  wrapper. There is no evidence of actual drift. New runs prospectively repair
  the gap by staging, hashing, executing, and rechecking the exact native
  client digest
  `sha256:ae1d3ffe6d48aec6a4dc3f50e7eb8e0d11962485a6a9406c5a7012139383da02`.
- Exact v3 replay against the A3 successor ledger now certifies strict DRILL
  capacity zero (zero eligible tasks/groups, `0 BD + 0 HD`), while DEV remains
  `16 BD + 0 HD`. The zero-capacity certificate is
  `sha256:48fba29c8a33a5fd773baed373694ac32d91a6f456b17ede563113eeeecd18b1`.
- A3's theoretical hole was concrete: descriptions were audit-only, proposal
  was one irreversible bundled guess, and no formula was evaluated. The atomic
  successor now freezes neutral descriptions, proposes 1--12 single-phrase
  atoms from labelled descriptions only, records a complete atom-by-panel
  matrix, and selects a deterministic positive conjunction before query
  release.
- The exact atomic success path has 29 causal model receipts. It has no `Not`
  or polarity flip. Its operational archive hard-codes calibration,
  semantic-truth, benchmark, and official-test authority false; calibrated
  semantic selection is disabled pending a cold-validatable typed calibration
  artifact.
- The historical first N=1 frame contained ten repeated-generator exact-unseen
  training tasks. It tested transport/synthesis, not independent
  generalization. One task was exposed and consumed, so the active successor
  frame contains exactly nine IDs, digest
  `sha256:094e195fd8892cf09bcb8287e68bd747fdbb47a87075a60d0d23c291b17466ed`.
  Richer typed grounding and powered label-blind evaluation remain future
  work. Lean remains an optional removable sidecar.
- The first live N=1 attempt, from commit
  `62ea577f5d86d109577f4f5e49b8b4866eb76c92` and tag
  `bongard-atomic-pre-smoke-20260806`, is an operational wrapper failure, not a
  Bongard result. Cache, config, and exact-task exposure persisted; prediction
  and terminal did not. The task is consumed without reroll. After the runner
  returned a typed `AtomicSmokeRun`, fallback terminal construction rejected
  its frozen `MappingProxy` precommit. Normal terminal construction contains
  the same defect, although the surviving error does not identify the first
  caught exception. The exact error was `failed run precommit is not canonical
  JSON`, reason digest
  `2825061e41346b498f7ceb0e338b0382fa807b2c968d534703927d6ce5f8376d`.
  Status, phase, output, and successful-call count are irrecoverable; calls are
  unknown in `0..29`. No prediction means no label materialization or reveal,
  and no score, calibration, semantic, benchmark, or official-test claim. The
  sanitized record is
  [`../data/atomic_smoke_n1_operational_failure_v1.json`](../data/atomic_smoke_n1_operational_failure_v1.json).
- Atomic stores require mode `0700`. A pre-exposure setup invocation rejected
  a `0755` cache store and consumed nothing.
- Attempt two is a distinct pre-live successor, not a reroll. It binds the
  incident, the historical A3 ledger
  `sha256:7c85922f238eb121a30d441ccf3528c665037a34240e07a06feef01cc30cd7c4`,
  and active predecessor
  `sha256:b0533c1a8e94a190f5f382be5031e4318acb6ded2b635ac32172ee238c97de0a`.
  The native launcher stages before secret generation and exact-task
  exposure. A fresh empty mode-`0700` journal persists a bound header, an
  exact intent before each of 29 transports, each validated result before the
  next intent, and its terminal before runner return. Existing or partial
  journals cannot be resumed or retried. No attempt-two outcome is claimed.
- New receipts bind the complete executable Bongard Python source boundary;
  post-exposure source drift
  now persists a typed failed receipt with labels withheld. Exact-identity
  caches cut the synthetic Stage-A path from 161.15 s to 11.50 s and Stage B
  from 218.88 s to 51.10 s.
- Visual-semantic SEALED/test execution is hard-disabled.

The pinned extracted-corpus manifest is
`sha256:6fa51548520190a412812ba8f872dc3c7a7a2b2c47c0e42a4d9f6df351dce138`.
There are no result placeholders: A2 is an invalidated incident and A3 is a
terminal scientific failure, not a pending measurement. The journaled atomic
attempt two is pre-live and has no outcome in this manuscript state.

Build the canonical paper from the repository root:

```bash
make -C bongard/manuscript
```

This produces `free_energy_abstraction.pdf`. The compatibility entry point
`semantic_cones.tex` inputs the same canonical source and does not define a
second paper:

```bash
make -C bongard/manuscript compatibility
```

`references.bib` is shared and unchanged. Generated LaTeX files are build
products. Pre-rewrite manuscript history is preserved at the annotated tag
`pre-bongard-complete-rewrite-20260805`.
