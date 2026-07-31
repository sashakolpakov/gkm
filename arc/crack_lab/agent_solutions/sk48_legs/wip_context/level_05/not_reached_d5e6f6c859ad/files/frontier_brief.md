# Unverified frontier brief: sk48 level 5

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Prior proposer progress

- The 3,000-state macro probe is still within its fixed cap and has not found a controlled `8` prefix yet. Its search reconstructs every candidate from the level-5 root, so runtime is dominated by safe replay rather than unbounded clone growth.
- The capped search’s best transition independently reproduces the useful peel: `up, retract, down` changes the attached sequence from `999888` to `99888`. The important untested geometry is approaching the `888` row from below with the final controlled `9`; I’m isolating that collision now, one alignment at a time.
- Approaching from below confirms that vertical contact is a hard collision: it neither pushes nor swaps an `8`. A single `9` also cannot pull the wall-backed `888` chain by simple contact/retract. I’m switching to a bounded reward search over full observational states to locate a valid transition sequence, while retaining the symbolic `898` score for diagnostics.
- The direct-clone BFS is progressing more slowly than the nominal step rate because nested arena clones grow expensive before its first 2,000-state report. I’ll give this fixed 20,000-state run one more short interval; if it remains silent, I’ll stop it and use shallower macro reconstruction instead.
- A key search issue is now clear: retracting can change which tail tokens are attached without changing the current pixels. Frame-only deduplication therefore collapses distinct states. I’m correcting the bounded search key to include the observable outcomes of one-step up/down probes, which exposes that latent attachment state while staying entirely on the clone API.
- The attachment-aware search is heavier because each candidate key includes two clone lookaheads, but it is still bounded at 5,000 states. It has not reached the first 250 expanded macros yet; I’ll keep it only through the next report or result, then tighten the candidate macros if needed.
- The lookahead fingerprint proved too costly on nested clones, so I stopped that run before its first report. I’m retaining the same partial-observability correction more cheaply by carrying the bounded horizontal-action suffix in the key; vertical moves that visibly resolve attachment reset that suffix.
- The cheaper key avoids the extra lookahead clones, but broad uniform macro expansion is still too slow to be the main route. I’m allowing this run to reach one progress sample so I can measure it, while narrowing the manual subproblem to how an attached row is transferred to a wall-backed row.
- The direct contact trace rules out “extend once, then retract” at the initial wall-backed row. To avoid guessing at the transfer mechanic, I’m reproducing the already-solved level-1/2 collection transitions symbolically—especially the prior case where a token is collected against the right wall—and then mapping that verified maneuver onto level 5.
- The prior-level trace exposes the missing mechanic: with no attached token, extend can move an invisible tether, and retract then pulls a contacted token. Level 5’s earlier probe looked only for pixel changes during extension and began from a non-reset tether. I’m testing the required reset–extend–retract sequence from the newly empty row.
- The empty train is genuinely dead after releasing the final anchor; reset–extend–retract cannot recover an `8`. The remaining contextual affordance not yet reproduced is action 6 at a vertical collision with a specific `8`, so I’m testing its post-action consequences at each of the three aligned contacts.
- Action 6 remains inert at all three vertical `9`–`8` contacts. The next physical possibility is different: extend the empty tether on an adjacent lane, cross it through one chosen `8`, then retract. That context was not covered by the prior same-row empty tests, so I’m probing it directly.

## Preserved local probes

- `probe_level5.py` (10202 bytes)
- `probe_level5_empty_collect.py` (2899 bytes)
- `probe_level5_vertical.py` (2251 bytes)
- `probe_prior_mechanics.py` (1887 bytes)
- `search_level5.py` (4089 bytes)
- `search_level5_bfs.py` (1408 bytes)
- `search_level5_eight.py` (2902 bytes)

Run or inspect the smallest relevant probe before writing another one.
