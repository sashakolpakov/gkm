# ARC-AGI-3 GPT-6.1 zero-seed campaign

The source baseline is frozen PR #37 commit
`9235ed26627140460efa1f6ca5e4041470cddc14`. The campaign copy excludes that
release's prior solutions and release artifacts, then applies only the
GPT-6.1 retry-policy binding, strict supervisor-noninterference taint gates,
workspace-local temporary-state containment, and empty-campaign audit support.
The local ARC game bundle is copied unchanged from the repository's canonical
`environment_files/` tree because those runtime files are not stored in the
frozen Git commit. They are available only to the game adapter; proposer
policy denies access to that tree and the complete-lineage taint gate rejects
any attempted source access.

This campaign starts at 0/183. It imports no solver, checkpoint, action path,
WIP capsule, transcript, retry history, game-specific hypothesis, or supervisor
hint from any earlier run. PR #37 and its frozen release remain preserved in
the repository, but they are not campaign inputs.

On 2026-11-18, after proposer turn 235 (lf52 L10 retry n56), generic WIP
bookkeeping was repaired before dispatching n57. Failed automatic reuse now
restores the exact pre-attempt `players.py`, and the Codex source index is
refreshed immediately before each proposal. This preserves the proposer's own
clean working code already allowed by the retry policy. The model, prompt,
600-action limit, and admission rules are unchanged. The `gkm_legs.py` SHA-256
changed from `bcc8c16af0a847ba5c72d0cafe7fb1b8aafa60cd6e685cff8f54940ee91148e4`
to `6b4755c4a044b83b54d53918b91e312a2b8fe77fa2165a4115bd7862b0137ab7`.

On 2026-11-21, shared runtime and release checks were repaired to reject explicit
self-invalidation reports about malformed actions and workspace escapes, and
explicit temporary-output paths outside the workspace. The Codex filesystem
sandbox now excludes global temporary directories from its writable roots.
A rescan of all 307 protected proposer captures found temporary-output violations
in the admitted re86 L4, r11l L2, and lf52 L6 histories. Their descendants were
removed, retaining the clean checkpoints re86 L3, r11l L1, and lf52 L5: 168/183
levels in total. These are containment incidents, not game-source-read findings.
Invalid descendant WIP and scratch code were deleted; protected transcripts and
token records remain, with explicit lineage classification corrections. All 25
retained promotion chains passed the strengthened audit, and the three restored
prefixes passed fresh replay under the corrected filesystem sandbox. The model,
retry ladder, prompt, and 600-action budget are unchanged.
The repaired policy files have SHA-256 digests:
`gkm_legs.py`: `c3bac08f92f7cffe69b476cbc54dc43be93026b8064ef757fed46869ead5ca45`;
`proposer_integrity.py`: `4514ce7f040e666b8f0020078971f7fcf01531a18b30e1b69e66ce49e19621b2`;
`audit_submission_taint.py`: `4d30f76e1f489623347752da23c14d30da56f3d43c8d13690f089ad63dac58aa`.

A later lf52 L9 n2 turn explicitly invalidated itself after generating an
out-of-frame coordinate rejected before execution. The shared fallback was
extended to recognize that wording during live monitoring and sealed accounting.
That turn's WIP was discarded, retaining the prior clean n1 WIP and all 181
accepted levels. All 322 protected captures were rescanned; the 25 accepted
promotion chains passed. The current SHA-256 digests are
`gkm_legs.py`: `e43cafee4c8078948e092324d67ce98fdacff9ef27d0bdc2b60b7ace3d3680f0`;
`proposer_integrity.py`: `a40209ae881611021626f90f5d5ff55f7d4bfe051c1b727bb33f9559dbbd42d2`.

Two subsequent lf52 L10 turns invalidated themselves after blocked shell
here-document temporary writes. The shared recognizer handles both reports in
live monitoring and sealed token accounting. Their workspaces and WIP were
removed; the prior clean WIP and all 182 accepted levels were retained. The
campaign now sets zsh's TMPPREFIX inside each workspace. An OS-sandbox test
confirmed that large here-documents work there while outside writes stay
blocked. All 352 protected captures were rescanned, and all 25 accepted
promotion chains passed. The current SHA-256 digests are
`gkm_legs.py`: `158de1bb64b634389391400ee8e0e6cb85bfad8d6b99a8696d624880e8813b37`;
`proposer_integrity.py`: `0156ddad873e0148a7ca26deb3c9c828026c8b01e1979df3e0e9a96ff2c77de1`.

The supervisor is limited to orchestration, replay, auditing, and admission.
It may select a frontier, launch the fixed harness, verify exact next-level
progress, independently replay a candidate, enforce provenance and taint
rules, and promote a passing candidate. It may not provide a route, action
suffix, mechanic hypothesis, scaffold, reviewed handoff, negative-evidence
brief, or solver code. The taint gate rejects any such artifact.

Every proposer turn uses `gpt-6.1-sol`. The retry ladder is unchanged:

- n0: medium, 15 minutes, clean start
- n1: high, 20 minutes, restore clean same-frontier WIP
- n2: xhigh, 25 minutes, restore
- n3: xhigh, 40 minutes, restore
- n4: max, 60 minutes, restore
- n5: max, 90 minutes, clean reset
- n6: max, 120 minutes, restore
- n7: max, 180 minutes, clean reset
- n8: max, 180 minutes, restore
- n9 and later: max, 300 minutes, alternating clean reset and restore

There is no final-six model or effort overlay. Promotion admits the first
proposer-authored candidate that reaches the exact next boundary and passes
independent replay, complete-lineage taint, provenance, and containment gates.
Complexity and free energy remain post-admission diagnostics only.
