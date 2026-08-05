# Headless Codex Phase D exploratory pilot — 5 August 2026

This directory contains a complete, write-once, three-arm Phase D protocol
pilot. It used one frozen Basic Bongard-LOGO problem, `PYTHONHASHSEED=0`, the
unrestricted track only, one shuffled-side replicate, and the held-fixed
no-share derivation. Adaptive proposer turns ran through non-interactive
`codex exec`, requesting `gpt-5.6-sol` with medium reasoning effort.

This is exploratory engineering evidence, not a confirmatory benchmark. The
preregistration was written locally but not published or externally committed
before the first paid call; n=1 gives no solve-rate estimate; there is no
semantic-pure arm; and the arm table is not the default 27-arm 1/5/25 design.

## Frozen identities

- Preregistration schema: `bongard.phase-d-preregistration/v6`
- Preregistration digest:
  `sha256:5c487ab217832a94bb674111d9c54acdae045d86f148f5d84ada6134f7d861ee`
- Corpus digest:
  `sha256:138cbabfc0214fe95fb926d0e7a38b7c1860ec95158eb961886265a43f075205`
- Campaign schema: `bongard.phase-d-campaign/v6`
- Campaign digest:
  `sha256:8be70918d2b57811a66787cdff845dbcb445eaf8e073f61443cea698845dfcf2`

The machine-readable result is [`campaign.json`](campaign.json); the exact
plan is [`phase_d_preregistration.json`](phase_d_preregistration.json).

## Results

| Arm | Terminal status | Solved | Held-out | Train | Consuming turns | Admitted charge |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| primary | `UNSOLVED_UNRESTRICTED` | 0/1 | 0.500 | 0.583333 | 3 | 0 |
| shuffled-sides r0 | `VERIFIER_FAILURE_UNRESTRICTED` | 0/1 | 0.000 | 0.000 | 3 | 0 |
| no-share | `UNSOLVED_UNRESTRICTED` | 0/1 | 0.500 | 0.583333 | derived | 0 |

The primary/no-share terminal rule was
`p_endpoint_turn_degrees<=6.607`. The shuffled terminal attempted source was
rejected by predicate admission (`append` on a receiver not certified as
direct locally owned storage), producing the exact canonical zero-admission
sentinel on cold replay. No candidate was accepted; all promoted libraries
therefore remain the 60-byte initial source with digest
`sha256:357e3a78fe66e45bd8ae862fb142d5932e7d4b67c3132a77d46fe947a87c0c39`.

The two adaptive arms contain six unique consuming Codex receipts: 64,641
input tokens, 28,489 output tokens, 19,351 reasoning-output tokens, and zero
cached-input tokens. No-share reuses the primary receipts by design. Before
the first shuffled consuming turn, one Codex call reached the fixed 15-minute
timeout; it was restored and classified as infrastructure, so it is not a
scientific turn and has no accepted usage receipt.

All three artifact certifications regenerate exactly, including checkpoint,
results, promoted-source, receipt, and track-report digests. The final local
test gate was 813 tests plus 12 subtests passing. The earlier
`codex_eod_20260805_v2` directory is intentionally excluded: its shuffled arm
hit a resource-cliff replay mismatch and never published a complete campaign.
