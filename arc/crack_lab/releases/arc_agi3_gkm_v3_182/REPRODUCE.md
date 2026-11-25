# Reproduce the frozen release

Use the public repository commit linked by the version-3 Community Leaderboard
entry. Install the repository requirements and ARC toolkit 0.9.9 / arcengine 0.9.3.
The game bundle must match the hash-bound versions used by the scorecard replayer.
No model calls are needed to check the release or replay its paths.

## Verify the receipt offline

From the repository root (Git history is needed for the historical controls):

```sh
python arc/crack_lab/verify_frozen_release.py \
  --canonical-root arc/crack_lab/releases/arc_agi3_gkm_v3_182/artifacts \
  --receipt arc/crack_lab/releases/arc_agi3_gkm_v3_182/receipts/*.json
```

The receipt binds all 25 games, all 182 admitted boundaries, and the explicitly
unclaimed lf52 level 10. It reopens the source/path/action/taint/hash evidence using
the exact historical verification context. The separately pinned original discovery
producer and the publication-only control-manifest extension are described in README.

## Recompute reported token costs

`accounting/usage.jsonl` retains the complete recorded ledger, including later
plateau work and classification corrections. `capture_usage.json` contains hashes
and terminal usage fields extracted from the sealed captures. Recompute without an
LLM using `arc.campaign_costs.build_report`; the release CI checks exact equality
with `token_costs.json`. Original winning transcripts remain in the acquisition and
certified evidence. Hashes and terminal counters are published for the other turns;
failed solver workspaces are not part of this release.

## Scorecard replay

Provide the ARC API key through the existing .env mechanism. The command below
opens a new scorecard; use `--preflight-only` for a local check without opening one.
Run ONLINE first, then Competition, changing `--output-json` each time:

```sh
python arc/crack_lab/replay_scorecard.py --mode online --games all \
  --artifact-root arc/crack_lab/releases/arc_agi3_gkm_v3_182/artifacts \
  --release-receipt arc/crack_lab/releases/arc_agi3_gkm_v3_182/receipts/*.json \
  --expected-claimed-levels 182 --source-revision "$(git rev-parse HEAD)" \
  --output-json /tmp/gkm-v3-online.json
```

An external replay is distinct from rediscovery: it consumes the admitted paths
and performs no inference or clone lookahead. Rerunning discovery requires the
`producer_source/` snapshot (including its `PROTOCOL.md`), an authenticated
Codex installation, the documented
blank-start isolation and retry ladder. Model generation is stochastic; replaying
its frozen output is the deterministic reproduction target.

The source closure is a separate acquisition-time tree. Run discovery from a
blank checkout of that tree with the external game bundle and declared packages;
do not seed it with `acquisition_source/` or `artifacts/`. Its runner and policy
provide `--help`. `source_manifest.json` records every included file hash.
