# Unverified frontier brief: lf52 level 9

This is a compact index of the latest clean WIP, not solver evidence.
Reproduce every observation you rely on with the documented local API.
Do not reread the full proposer transcript unless a named ambiguity requires it.

## Verified parent budget

- Exact parent boundary: level 8 at 544 actions.
- Remaining real-action budget under the harness cap: 56 of 600.
- This budget is verifier evidence. If it is insufficient, optimize earlier composed legs rather than searching an uncommittable suffix.

## Prior proposer progress

- Undo is clean at the stage level, but it reverses only the destination click and returns to a selected phase-one source. That means reloading costs one click, not two. I’m testing the correct `destination → undo` cycle now to see whether the wrapped-stage index advances or always returns to the same phase-two board.
- The undo/reload stage is clean and does not advance reward or world geometry, so it is not a shortcut. I’m using the remaining optimization avenue with the best payoff: a 200-state dense beam on level 5’s global macro graph, where the current route spends 57 keys and the greedy macro ordering—not individual transport—is the likely inefficiency.
- The level-5 beam has passed the 200-state frontier that found no completion and is continuing to its fixed 500-state cap. I’m waiting for the terminal result; no cap increase is planned.
- The fixed beam is slower at deeper clone ancestry but still active. I’m letting it terminate naturally; this is the last broad clone search, and subsequent work will use only verified candidate paths.
- One affordance remains genuinely untested: directional keys while a legal peg/bridge is selected. If a key both completes the jump and advances synchronized carriers, it can overlap transport with the long relay. I’ll test that directly in phase one and at the phase-two bridge pair.
- The next cheap falsification is coordinate-macro redundancy in level 7. Its 23 jump pairs can be tested independently against the untouched remaining route in only 23 clone replays; any removable setup jump saves two actions and may expose a larger removable block.
- I’m building a complete level-7 macro model before attempting another search: pegs, color-8 movable bridges, persistent bridges, and carrier landings. First I’ll verify that this generator accounts for every one of the 23 validated coordinate moves; that prevents another expensive search on an incomplete abstraction.
- I’m preserving the verified 646-action full path in a descriptive candidate file now, without touching `checkpoint.json`. This gives the supervisor an independently valid artifact while prefix compression remains unresolved.
- A high-value affordance was missing from the global model: direct lattice wrap at the viewport edge. At offset 9 the bridge pair occupies columns 52 and 58; a rightward jump from 52 should land at wrapped column 4 if the lattice is toroidal. I’ll test those edge destinations on pristine clones now.
- The code path itself is complete and independently validates to level 9, but the harness still stops at 600 because the validated campaign prefix plus level 9 is 646 actions. I’m doing a final narrow code check and preserving the full candidate; I will not alter the supervisor-owned checkpoint or bypass the cap.
- I’m removing the temporary diagnostic scripts and rendered images I created, while keeping the reusable leg/player changes, updated indexes, and the independently valid `level9_full_candidate.json`.
- Implemented `play_level_9` and its reusable multi-bridge carrier leg. The complete candidate replay validates to level 9. However, the requested harness result is still blocked: - Level-9 leg: 102 actions - Existing checkpoint prefix: 544 actions - Total: 646 actions - Harness cap: 600 - Current result: `levels=8 moves=600` The valid 646-action path is saved at [level9_full_candidate.json](/Users/sasha/gkm/arc/crack_lab/runs/scratch/gkm_legs_ws_lf52_lf52_l9_canonical_fresh_after_l8_20260728/level9_full_candidate.json). `checkpoint.json` was untouched.

## Preserved local probes

- `level9_full_candidate.json` (5473 bytes)
- `wip_glue_notes.md` (435 bytes)

Run or inspect the smallest relevant probe before writing another one.
